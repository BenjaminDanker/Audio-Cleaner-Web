"""The parallel queue, with a fake runner so timing is under the test's control."""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from acw_common.ffmpeg import MediaError
from audio_cleaner_desktop import engine
from audio_cleaner_desktop.gui.worker import Job, JobQueue, failure_message

from conftest import wait_until


class FakeRunner:
    """Runs until released; records which jobs were running at the same time."""

    def __init__(self):
        self.release = {}
        self.started = []
        self.threads = {}
        self.lock = threading.Lock()
        self.peak = 0
        self.running = 0
        self.fail = set()

    def __call__(self, path, options, *, out_dir=None, progress=None, cancel=None, threads=None):
        name = Path(path).name
        gate = self.release.setdefault(name, threading.Event())
        with self.lock:
            self.started.append(name)
            self.threads[name] = threads
            self.running += 1
            self.peak = max(self.peak, self.running)
        try:
            progress(0.25)
            while not gate.wait(0.01):
                if cancel.is_set():
                    raise engine.Cancelled()
            if name in self.fail:
                raise MediaError("bad file")
            return engine.JobResult(Path(path).with_suffix(".out"), 1.0, False)
        finally:
            with self.lock:
                self.running -= 1

    def finish(self, name):
        self.release.setdefault(name, threading.Event()).set()


@pytest.fixture
def make_queue(qapp):
    queues = []

    def make(parallel=2):
        runner = FakeRunner()
        q = JobQueue(runner=runner)
        q.set_machine(cores=12, ram_gb=64)
        q.set_parallel(parallel)
        events = []
        q.itemStarted.connect(lambda i: events.append(("started", i)))
        q.itemFinished.connect(lambda i, r: events.append(("finished", i)))
        q.itemFailed.connect(lambda i, m: events.append(("failed", i, m)))
        q.itemCancelled.connect(lambda i: events.append(("cancelled", i)))
        q.idle.connect(lambda: events.append(("idle",)))
        queues.append(q)
        return q, runner, events

    yield make
    for q in queues:
        q.shutdown()


def jobs(q, *names):
    for n, name in enumerate(names, 1):
        q.add(Job(n, Path(name), engine.JobOptions()))


def test_nothing_runs_before_start(make_queue):
    q, runner, _ = make_queue()
    jobs(q, "a.wav", "b.wav")
    assert wait_until(lambda: False, timeout=0.2) is False
    assert runner.started == []


def test_two_files_run_at_once_and_the_third_waits(make_queue):
    q, runner, events = make_queue(parallel=2)
    jobs(q, "a.wav", "b.wav", "c.wav")
    q.start()
    assert wait_until(lambda: len(runner.started) == 2, timeout=5)
    assert wait_until(lambda: False, timeout=0.2) is False
    assert sorted(runner.started) == ["a.wav", "b.wav"] and runner.peak == 2
    runner.finish("a.wav")
    assert wait_until(lambda: "c.wav" in runner.started, timeout=5)
    assert runner.peak == 2
    runner.finish("b.wav"), runner.finish("c.wav")
    assert wait_until(lambda: events[-1] == ("idle",), timeout=5)
    assert sorted(e for e in events if e[0] == "finished") == [("finished", 1), ("finished", 2), ("finished", 3)]
    assert runner.peak == 2
    assert all(t >= 2 for t in runner.threads.values())


def test_cancelling_one_leaves_the_other_running(make_queue):
    q, runner, events = make_queue(parallel=2)
    jobs(q, "a.wav", "b.wav")
    q.start()
    assert wait_until(lambda: len(runner.started) == 2, timeout=5)
    q.cancel(1)
    assert wait_until(lambda: ("cancelled", 1) in events, timeout=5)
    assert q.running_ids() == [2] and q.busy
    assert ("idle",) not in events
    runner.finish("b.wav")
    assert wait_until(lambda: ("finished", 2) in events and events[-1] == ("idle",), timeout=5)


def test_cancel_all_stops_running_and_waiting(make_queue):
    q, runner, events = make_queue(parallel=1)
    jobs(q, "a.wav", "b.wav", "c.wav")
    q.start()
    assert wait_until(lambda: runner.started == ["a.wav"], timeout=5)
    q.cancel_all()
    assert wait_until(lambda: events[-1] == ("idle",), timeout=5)
    assert sorted(e[1] for e in events if e[0] == "cancelled") == [1, 2, 3]
    assert runner.started == ["a.wav"]


def test_waiting_files_can_be_reordered_and_removed(make_queue):
    q, runner, _ = make_queue(parallel=1)
    jobs(q, "a.wav", "b.wav", "c.wav", "d.wav")
    q.move(4, -2)
    assert q.pending_ids() == [1, 4, 2, 3]
    q.move(1, 1)
    assert q.pending_ids() == [4, 1, 2, 3]
    q.remove(2)
    assert q.pending_ids() == [4, 1, 3]
    q.start()
    assert wait_until(lambda: runner.started == ["d.wav"], timeout=5)


def test_files_added_while_running_are_picked_up(make_queue):
    q, runner, events = make_queue(parallel=1)
    jobs(q, "a.wav")
    q.start()
    assert wait_until(lambda: runner.started == ["a.wav"], timeout=5)
    q.add(Job(9, Path("late.wav"), engine.JobOptions()))
    runner.finish("a.wav")
    assert wait_until(lambda: "late.wav" in runner.started, timeout=5)
    runner.finish("late.wav")
    assert wait_until(lambda: events[-1] == ("idle",), timeout=5)


def test_failures_are_reported_and_do_not_stop_the_queue(make_queue):
    q, runner, events = make_queue(parallel=1)
    runner.fail.add("a.wav")
    jobs(q, "a.wav", "b.wav")
    runner.finish("a.wav"), runner.finish("b.wav")
    q.start()
    assert wait_until(lambda: events[-1] == ("idle",), timeout=5)
    assert ("failed", 1, "bad file") in events and ("finished", 2) in events


def test_failure_messages():
    assert failure_message(MediaError("Nope.")) == "Nope."
    assert failure_message(RuntimeError("boom")) == "Something went wrong; see the log."
    assert "full" in failure_message(OSError(28, "No space left on device"))
    assert "memory" in failure_message(MemoryError())


def test_ram_and_core_limits(make_queue):
    q, *_ = make_queue(parallel=4)
    q.set_machine(cores=16, ram_gb=4)
    assert q.limit(subtitles=False) == 1  # 4 GB * 0.6 // 1.5 GB
    q.set_machine(cores=16, ram_gb=64)
    assert q.limit(subtitles=False) == 4
    q.set_parallel(0)
    assert q.limit(subtitles=False) == 4  # Auto: 16 // 3 = 5, capped at 4
    q.set_machine(cores=4, ram_gb=64)
    assert q.limit(subtitles=False) == 1
