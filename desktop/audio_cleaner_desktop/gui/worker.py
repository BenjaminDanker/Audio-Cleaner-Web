"""A queue that cleans several files at once, each in its own thread.

Per-file signals: itemStarted(id), itemProgress(id, fraction), itemFinished(id, JobResult),
itemFailed(id, message), itemCancelled(id); idle() when nothing is left to do.
"""
from __future__ import annotations

import errno
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot

from acw_common.ffmpeg import MediaError

from .. import engine, sysinfo

log = logging.getLogger(__name__)

Runner = Callable[..., engine.JobResult]


@dataclass
class Job:
    id: int
    path: Path
    options: engine.JobOptions
    out_dir: Path | None = None


def failure_message(error: BaseException) -> str:
    """Text for the user; unexpected errors are logged by the caller."""
    if isinstance(error, MediaError):
        return str(error)
    if isinstance(error, OSError) and error.errno == errno.ENOSPC:
        return "The disk is full."
    if isinstance(error, MemoryError):
        return "Not enough memory. Try a lower number of files at once."
    return "Something went wrong; see the log."


class _JobThread(QThread):
    progress = Signal(int, float)
    succeeded = Signal(int, object)
    failed = Signal(int, str)
    stopped = Signal(int)

    def __init__(self, job: Job, runner: Runner, threads: int, parent: QObject):
        super().__init__(parent)
        self.job = job
        self.cancel = threading.Event()
        self._runner = runner
        self._threads = threads

    def run(self) -> None:
        job = self.job
        try:
            result = self._runner(
                job.path, job.options, out_dir=job.out_dir, threads=self._threads, cancel=self.cancel,
                progress=lambda f: self.progress.emit(job.id, f),
            )
        except engine.Cancelled:
            self.stopped.emit(job.id)
        except Exception as e:  # noqa: BLE001 - one bad file must not take the app down
            if not isinstance(e, (MediaError, MemoryError)) and not (isinstance(e, OSError) and e.errno == errno.ENOSPC):
                log.exception("cleaning %s failed", job.path)
            self.failed.emit(job.id, failure_message(e))
        else:
            self.succeeded.emit(job.id, result)


class JobQueue(QObject):
    itemStarted = Signal(int)
    itemProgress = Signal(int, float)
    itemFinished = Signal(int, object)
    itemFailed = Signal(int, str)
    itemCancelled = Signal(int)
    idle = Signal()

    def __init__(self, parent: QObject | None = None, runner: Runner = engine.process):
        super().__init__(parent)
        self._runner = runner
        self._pending: list[Job] = []
        self._running: dict[int, _JobThread] = {}
        self._active = False
        self._parallel = 0  # 0 = Auto
        self._cores: int | None = None  # physical cores / RAM override, for tests
        self._ram_gb: float | None = None

    # -- configuration ------------------------------------------------------

    def set_parallel(self, files_at_once: int) -> None:
        """1-4, or 0 for Auto."""
        self._parallel = files_at_once
        self._pump()

    def set_machine(self, cores: int | None, ram_gb: float | None) -> None:
        self._cores, self._ram_gb = cores, ram_gb

    def limit(self, subtitles: bool, rebuild_voice: bool = False, echo_track: bool = False) -> int:
        """Files to run at once given what the jobs do (each of these makes a job need more memory)."""
        return sysinfo.max_parallel(
            self._parallel, subtitles, cores=self._cores, ram_gb=self._ram_gb, rebuild_voice=rebuild_voice,
            echo_track=echo_track,
        )

    def limit_for(self, jobs: list[Job]) -> int:
        return self.limit(
            any(j.options.subtitles != "off" for j in jobs),
            any(j.options.rebuild_voice for j in jobs),
            any(j.options.echo_track for j in jobs),
        )

    # -- queue --------------------------------------------------------------

    def add(self, job: Job) -> None:
        self._pending.append(job)
        self._pump()

    def pending_ids(self) -> list[int]:
        return [j.id for j in self._pending]

    def running_ids(self) -> list[int]:
        return list(self._running)

    @property
    def active(self) -> bool:
        """True from start() until the queue runs dry or is cancelled."""
        return self._active

    @property
    def busy(self) -> bool:
        return bool(self._pending or self._running)

    def start(self) -> None:
        self._active = True
        self._pump()

    def remove(self, job_id: int) -> None:
        """Drop a waiting file, or cancel a running one."""
        self._pending = [j for j in self._pending if j.id != job_id]
        self.cancel(job_id)

    def move(self, job_id: int, delta: int) -> None:
        ids = self.pending_ids()
        if job_id not in ids:
            return
        i = ids.index(job_id)
        j = min(max(i + delta, 0), len(ids) - 1)
        self._pending.insert(j, self._pending.pop(i))

    def cancel(self, job_id: int) -> None:
        thread = self._running.get(job_id)
        if thread is not None:
            thread.cancel.set()

    def cancel_all(self) -> None:
        pending, self._pending = self._pending, []
        self._active = False
        for thread in self._running.values():
            thread.cancel.set()
        for job in pending:
            self.itemCancelled.emit(job.id)
        self._check_idle()

    def stopping(self, job_id: int) -> bool:
        """True once a running file was asked to stop (it still finishes its current step)."""
        thread = self._running.get(job_id)
        return thread is not None and thread.cancel.is_set()

    def shutdown(self, timeout_ms: int | None = None) -> bool:
        """Cancel everything and wait for the threads (at most `timeout_ms` each when given). Returns whether
        they all ended."""
        self.cancel_all()
        ended = True
        for thread in list(self._running.values()):
            ended = (thread.wait(timeout_ms) if timeout_ms is not None else thread.wait()) and ended
        if ended:
            self._running.clear()
        return ended

    # -- internals ----------------------------------------------------------

    def _pump(self) -> None:
        while self._active and self._pending:
            running = [t.job for t in self._running.values()]
            limit = self.limit_for([*self._pending, *running])
            if len(self._running) >= limit:
                return
            job = self._pending.pop(0)
            thread = _JobThread(job, self._runner, sysinfo.threads_per_job(limit), self)
            thread.progress.connect(self.itemProgress)
            thread.succeeded.connect(self._on_succeeded)
            thread.failed.connect(self._on_failed)
            thread.stopped.connect(self._on_stopped)
            self._running[job.id] = thread
            self.itemStarted.emit(job.id)
            thread.start()

    @Slot(int, object)
    def _on_succeeded(self, job_id: int, result: engine.JobResult) -> None:
        self._release(job_id)
        self.itemFinished.emit(job_id, result)
        self._after()

    @Slot(int, str)
    def _on_failed(self, job_id: int, message: str) -> None:
        self._release(job_id)
        self.itemFailed.emit(job_id, message)
        self._after()

    @Slot(int)
    def _on_stopped(self, job_id: int) -> None:
        self._release(job_id)
        self.itemCancelled.emit(job_id)
        self._after()

    def _release(self, job_id: int) -> None:
        thread = self._running.pop(job_id, None)
        if thread is not None:
            thread.wait()
            thread.deleteLater()

    def _after(self) -> None:
        self._pump()
        self._check_idle()

    def _check_idle(self) -> None:
        if not self._pending and not self._running:
            self._active = False
            self.idle.emit()
