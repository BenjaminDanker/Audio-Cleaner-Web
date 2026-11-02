"""The main window: one window, one screen. The editor (gui/studio.py) is always there: before a file is chosen it shows where to drop one, choosing a
file fills it in, Start runs it and the same screen shows the result. The jobs (the files waiting and running, and every finished job) are a column
beside it, or under it when the window is narrow; opening one shows it in the same editor."""
from __future__ import annotations

import logging
import os
import shutil
import sys
import threading
import time
from collections.abc import Iterable
from pathlib import Path

from PySide6.QtCore import QRunnable, QSettings, QStandardPaths, Qt, QThreadPool, QTimer
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QBoxLayout, QFileDialog, QFrame, QLineEdit, QMainWindow, QMessageBox, QScrollArea, QSystemTrayIcon, QVBoxLayout,
    QWidget,
)

from acw_common.ffmpeg import MediaError

from .. import __version__, config, copies, engine, models, redo, resources, sysinfo
from ..updater import UpdateController
from . import media, shell, tips
from .about import AboutDialog
from .doc import Doc, DocEnv
from .history import DONE, FAILED, Entry, HistoryStore
from .info_dialog import InfoDialog
from .items import CANCELLED, DONE as ITEM_DONE, FAILED as ITEM_FAILED, RUNNING, WAITING, FileItem, format_size
from .jobs_column import JobsColumn, format_spent
from .preferences import Preferences
from .settings_dialog import SettingsDialog
from .speech_model import SpeechModelDownloader, size_text
from .studio import Studio
from .theme import set_property
from .widgets import Banner
from .worker import Job, JobQueue
from .workers import Worker
from ..labels import STRENGTH_LABELS, settings_dict

log = logging.getLogger(__name__)

JOBS_BESIDE = 1180  # below this width the jobs go under the editor
NARROW = 900  # below this much room the settings go under the editor too
JOBS_WIDTH = 300
SCRATCH_SAFETY = engine.SCRATCH_SAFETY
MODEL_UPDATE_DELAY_MS = 6000  # a model whose catalog entry changed is fetched this long after the window opens
SWEEP_DELAY_MS = 3000  # leftovers of an earlier run are swept this long after the window opens
QUIT_WAIT_MS = 2500  # how long quitting waits for running files to stop before giving up on them


def data_dir() -> Path:
    path = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation))
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_dir() -> Path:
    path = data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path



class _SweepTask(QRunnable):
    """Removes the scratch files an earlier run left behind (it crashed or was killed), off the window's thread."""

    def __init__(self, folders: list[Path]):
        super().__init__()
        self._folders = folders

    def run(self) -> None:
        try:
            removed = engine.sweep_leftovers(self._folders)
            if removed:
                log.info("removed %d leftover scratch items", removed)
        except Exception:  # noqa: BLE001 - housekeeping must never bother the user
            log.exception("sweeping leftovers failed")




def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


class MainWindow(QMainWindow):
    def __init__(
        self,
        files: Iterable[str] = (),
        *,
        settings: QSettings | None = None,
        history_path: Path | None = None,
        runner=engine.process,
        updater: UpdateController | None = None,
        doc_env: DocEnv | None = None,
    ):
        super().__init__()
        self.setWindowTitle(config.APP_NAME)
        self.setMinimumSize(640, 520)
        self.setAcceptDrops(True)
        self._qs = settings if settings is not None else QSettings()
        self.prefs = Preferences(self._qs)
        self.updater = updater or UpdateController(self.prefs, self)
        self._restart_when_idle = False  # "Restart now" was pressed while files were being cleaned
        self.history = HistoryStore(history_path or data_dir() / "history.sqlite3")
        self.queue = JobQueue(self, runner)
        self.queue.set_parallel(self.prefs.parallel)
        self._items: dict[int, FileItem] = {}
        self._queued: set[int] = set()
        self._next_id = 1
        self._batch_started = 0.0
        self._batch_done = 0
        self._batch_failed = 0
        self._tray: QSystemTrayIcon | None = None
        self._disk_text = ""
        self._missing = resources.check()
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(2)
        self._awake = sysinfo.SleepInhibitor()
        self._closing = False
        self.doc: Doc | None = None  # the file in the editor
        self._saving: Worker | None = None  # a change to a finished job is being saved
        self._settings_dialog: SettingsDialog | None = None
        self._to_probe: list[Doc] = []  # files added by the dozen are read a few at a time (not a thread each, all at once)
        self._probing: set[Doc] = set()
        self._redo = redo.run  # (what makes a change: tests replace it)
        self.doc_env = doc_env or DocEnv(
            prefs=self.prefs, scratch=lambda: self.history.jobs_root, voice_ready=lambda: resources.voice_model_dir() is not None,
            speech_ready=lambda: resources.speech_ready(), noise_ready=lambda: resources.noise_ready())
        self._jobs_timer = QTimer(self, singleShot=True, interval=0, timeout=self._draw_jobs)

        self._build_menus()
        self._build_ui()
        self._wire()
        geometry = self._qs.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        else:
            self.resize(1440, 900)
        self.add_paths(files)
        self._draw_jobs()
        QTimer.singleShot(SWEEP_DELAY_MS, self._sweep_leftovers)

    # -- construction -------------------------------------------------------

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        self._act_add = QAction("&Add files...", self, shortcut=QKeySequence.StandardKey.Open)
        self._act_add.setStatusTip("Choose video or audio files")
        self._act_folder = QAction("Add &folder...", self, shortcut=QKeySequence("Ctrl+Shift+O"))
        self._act_settings = QAction("&Settings...", self, shortcut=QKeySequence("Ctrl+,"), menuRole=QAction.MenuRole.PreferencesRole)
        quit_action = QAction("&Quit", self, shortcut=QKeySequence.StandardKey.Quit, menuRole=QAction.MenuRole.QuitRole)
        quit_action.triggered.connect(self.close)
        file_menu.addActions([self._act_add, self._act_folder])
        file_menu.addSeparator()
        file_menu.addAction(self._act_settings)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        help_menu = self.menuBar().addMenu("&Help")
        options = QAction("All the &options", self)
        options.triggered.connect(self.show_options)
        about = QAction(f"&About {config.APP_NAME}", self, menuRole=QAction.MenuRole.AboutRole)
        about.triggered.connect(lambda: AboutDialog(self, updates_on=self.prefs.check_updates).exec())
        logs = QAction("Open &log folder", self)
        logs.triggered.connect(lambda: shell.open_path(log_dir()))
        help_menu.addActions([options, about, logs])

    def _build_ui(self) -> None:
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.page = QWidget()
        self.page.setObjectName("Page")
        outer = QVBoxLayout(self.page)
        outer.setContentsMargins(16, 16, 16, 10)
        outer.setSpacing(8)

        self.banner_missing = Banner()
        self.banner_disk = Banner()
        self.banner_models = Banner()
        self.banner_models.action.connect(self._models_action)
        self.banner_update = Banner()
        self.banner_update.action.connect(self._update_action)
        self.banner_files = Banner()
        self._model_queue: list[SpeechModelDownloader] = []
        self._auto_models = False  # the queue is an update the app started by itself (quiet when it fails)
        self._noise_only = False  # the queue is just the (required) noise removal models, see fetch_noise_models
        for banner in (self.banner_missing, self.banner_disk, self.banner_models, self.banner_update, self.banner_files):
            outer.addWidget(banner)
        if self._missing:
            self.banner_missing.show_message(
                "Audio Cleaner is missing some of its own files (" + ", ".join(self._missing) + "). "
                "Please install it again.", "error",
            )

        self.noise_download = SpeechModelDownloader(self, resources.NOISE_MODEL)
        self.speech_download = SpeechModelDownloader(self)
        self.voice_download = SpeechModelDownloader(self, resources.VOICE_MODEL)
        for dl in self._downloaders():
            dl.finished.connect(lambda ok, message, dl=dl: self._model_download_finished(dl, ok, message))
            dl.progress.connect(lambda done, total, dl=dl: self._model_queue and self._model_queue[0] is dl
                                and self._show_model_progress(done, total))
        self._fetch_missing_noise_models()

        self.body = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.body.setSpacing(14)
        self.studio = Studio(self.prefs, self.speech_download, self.voice_download)
        self.studio.problems = self._problems
        self.studio.out_dir = self.prefs.output_dir
        self.jobs = JobsColumn(self.history, lambda: list(self._items.values()), lambda: self._queued)
        self.jobs.setFixedWidth(JOBS_WIDTH)
        self.body.addWidget(self.studio, 1, Qt.AlignmentFlag.AlignTop)
        self.body.addWidget(self.jobs, 0, Qt.AlignmentFlag.AlignTop)
        outer.addLayout(self.body)
        outer.addStretch(1)
        from PySide6.QtWidgets import QLabel

        version = QLabel(f"Version {__version__}")
        version.setObjectName("Small")
        version.setAlignment(Qt.AlignmentFlag.AlignRight)
        outer.addWidget(version)
        self.scroll.setWidget(self.page)
        self.setCentralWidget(self.scroll)

    def _wire(self) -> None:
        self._act_add.triggered.connect(self.choose_files)
        self._act_folder.triggered.connect(self.choose_folder)
        self._act_settings.triggered.connect(self.show_settings)
        self.updater.readyChanged.connect(self._update_ready)
        s = self.studio
        s.chooseFiles.connect(self.choose_files)
        s.chooseFolder.connect(self.choose_folder)
        s.differentFile.connect(self.different_file)
        s.primaryClicked.connect(self.primary)
        s.cancelClicked.connect(self.cancel_current)
        s.deleteRequested.connect(self.delete_current)
        s.changeFolder.connect(self._change_folder)
        s.nextToOriginal.connect(self._next_to_original)
        s.infoRequested.connect(self.show_options)
        s.outputChanged.connect(self._check_disk)
        s.findOriginal.connect(self.find_original)
        s.subtitlesOnly.connect(self.save_subtitles_only)
        j = self.jobs
        j.openItem.connect(self.open_item)
        j.openEntry.connect(self.open_entry)
        j.removeItem.connect(self.remove_item)
        j.stopItem.connect(self._stop_item)
        j.moveItem.connect(self._move)
        j.deleteEntry.connect(self.delete_entry)
        j.deleteAll.connect(self.delete_all)
        j.startAll.connect(self.start_all)
        j.stopAll.connect(self.request_cancel_all)

        q = self.queue
        q.itemStarted.connect(self._on_started)
        q.itemProgress.connect(self._on_progress)
        q.itemFinished.connect(self._on_finished)
        q.itemFailed.connect(self._on_failed)
        q.itemCancelled.connect(self._on_cancelled)
        q.idle.connect(self._on_idle)

        for keys in ("Ctrl+Return", "Ctrl+Enter"):
            QShortcut(QKeySequence(keys), self, activated=self.primary)
        QShortcut(QKeySequence("Esc"), self, activated=self._escape)

    def _escape(self) -> None:
        """Esc stops the cleaning, unless it is meant for something else: closing a help text, or clearing the search."""
        if tips.is_showing():
            return
        focus = QApplication.focusWidget()
        if isinstance(focus, QLineEdit):  # Esc in a name box or the search only leaves the box
            focus.clearFocus()
            return
        self.request_cancel_all()

    # -- dialogs ----------------------------------------------------------------

    def show_options(self) -> None:
        InfoDialog(self).exec()

    def show_settings(self) -> None:
        if self._settings_dialog is None:
            self._settings_dialog = SettingsDialog(self.prefs, self.speech_download, self.voice_download, self.updater, self)
            self._settings_dialog.changed.connect(self._settings_changed)
        self._settings_dialog.show()
        self._settings_dialog.raise_()
        self._settings_dialog.activateWindow()

    def _change_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose where to save the cleaned files", self.prefs.output_folder or self.prefs.last_input_folder)
        if folder:
            self.prefs.output_folder = folder
            self.prefs.output_mode = "folder"
            self._settings_changed()

    def _next_to_original(self) -> None:
        self.prefs.output_mode = "next"
        self._settings_changed()

    # -- adding files ---------------------------------------------------------

    def choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Choose video or audio files", self.prefs.last_input_folder, media.dialog_filter()
        )
        if paths:
            self.prefs.last_input_folder = str(Path(paths[0]).parent)
            self.add_paths(paths)

    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose a folder with videos or recordings", self.prefs.last_input_folder)
        if folder:
            self.prefs.last_input_folder = folder
            if not self.add_paths([folder]):
                QMessageBox.information(self, config.APP_NAME, "There are no video or audio files in that folder.")

    def add_paths(self, paths: Iterable[str | Path]) -> int:
        """Take files in (folders are searched): the first one is in the editor at once, in place, and the rest wait in the jobs. Returns how many
        files were taken in (one that is already in the list is shown instead, and counts as taken)."""
        known = {i.path: i for i in self._items.values() if i.state in (WAITING, RUNNING, CANCELLED)}
        added, first = 0, None
        for path in media.expand(paths):
            item = known.get(path)
            if item is not None:  # already in the list: it is shown instead of adding a twin
                if item.state == CANCELLED:
                    item.state = WAITING
                first = first or item
                added += 1
                continue
            item = FileItem(self._next_id, path, has_video=path.suffix.lower() in media.VIDEO_EXTENSIONS)
            item.source = engine.Source(item.has_video, path.suffix.lower())  # a guess, until the file is read
            item.out_name = engine.default_output_name(path, item.source, self.prefs.job_options())
            self._next_id += 1
            self._items[item.id] = item
            doc = Doc(self.doc_env, path, item=item, parent=self)
            doc.out_name = item.out_name
            doc.unreadable.connect(lambda message, d=doc: self._unreadable(d, message))
            doc.factsReady.connect(self._draw_jobs)
            item.doc = doc
            if self._missing:
                item.probed = True
            else:
                self._to_probe.append(doc)
            first = first or item
            added += 1
        self._probe_more()
        if first is not None:
            self.open_item(first.id)
        self._draw_jobs()
        self._check_disk()
        return added

    PROBES_AT_ONCE = 4

    def _probe_more(self) -> None:
        """Start reading the next waiting files (the one in the editor first), a few at a time."""
        while self._to_probe and len(self._probing) < self.PROBES_AT_ONCE:
            doc = self._to_probe.pop(0)
            if doc.item is None or doc.item.id not in self._items:
                continue
            self._probing.add(doc)
            doc.factsReady.connect(lambda d=doc: self._probed_one(d))
            doc.unreadable.connect(lambda _m, d=doc: self._probed_one(d))
            doc.open_file()

    def _probed_one(self, doc: Doc) -> None:
        self._probing.discard(doc)
        if not self._closing:
            self._probe_more()

    def _unreadable(self, doc: Doc, message: str) -> None:
        """A file that can't be read: it is not taken in, and it is said so (in the editor when it is the one that was to be shown)."""
        item = doc.item
        if item is None or self._closing:
            return
        text = f"{doc.name}: {message}"
        was_current = self.doc is doc
        self.remove_item(item.id)
        if was_current:
            self.studio.show_problem(text)
        else:
            self.banner_files.show_message(text, "warn")

    # -- the editor shows a file ------------------------------------------------

    def _key_of(self, doc: Doc | None) -> tuple[str, int] | None:
        if doc is None:
            return None
        if doc.entry is not None:
            return ("entry", doc.entry.id)
        return ("item", doc.item.id) if doc.item is not None else None

    def _show(self, doc: Doc | None) -> None:
        """Put this file in the editor, in place (no new window, no page change). None: the empty screen."""
        old = self.doc
        if old is doc:
            self.studio.set_doc(doc)
            self.jobs.set_current(self._key_of(doc))
            return
        if old is not None:
            old.release()
            if old.item is None:  # a job opened from the list: nothing else holds it
                old.dispose()
        self.doc = doc
        if doc is not None:
            if doc in self._to_probe:  # the one in the editor is read first
                self._to_probe.remove(doc)
                self._to_probe.insert(0, doc)
                self._probe_more()
            doc.activate()
            if doc.is_job:
                doc.open_session()
        self.studio.show_problem("")
        self.studio.set_doc(doc)
        self.jobs.set_current(self._key_of(doc))

    def open_item(self, item_id: int) -> None:
        item = self._items.get(item_id)
        if item is not None and item.doc is not None:
            self._show(item.doc)

    def open_entry(self, entry_id: int) -> None:
        """A job of the list in the editor: the result, its settings and the parts, in the same window; the sound is made ready in the background."""
        for item in self._items.values():
            if item.entry_id == entry_id and item.doc is not None and item.doc.entry is not None:
                self._show(item.doc)
                return
        entry = self.history.get(entry_id)
        if entry is None:
            return
        job = None
        if entry.job_dir:
            folder = copies.job_dir(self.history.jobs_root, entry.job_dir)
            loaded = copies.JobCopies.load(folder) if folder is not None else None
            job = loaded if loaded is not None and loaded else None
        check = job.check_original() if job is not None else copies.OriginalCheck("missing")
        doc = Doc.for_job(self.doc_env, entry, job, check, parent=self)
        doc.stage = "done"
        if entry.status != DONE:
            doc.note, doc.note_kind = entry.message or "This job failed.", "error"
        self._show(doc)

    def different_file(self) -> None:
        """"Different file": the file leaves the editor (one that was not started is taken off the list; a job stays there)."""
        doc = self.doc
        if doc is None:
            return
        if doc.item is not None and doc.item.state in (WAITING, CANCELLED, ITEM_FAILED) and doc.entry is None and doc.item.id not in self._queued:
            self.remove_item(doc.item.id)
        else:
            self._show(None)

    def find_original(self) -> None:
        """The original was moved or renamed: point at it (it is only accepted if it is the same file)."""
        doc = self.doc
        if doc is None or doc.job is None or doc.entry is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Find the original file", str(doc.job.original_path.parent))
        if not path:
            return
        check = doc.job.relocate(Path(path))
        if not check.ok:
            doc.note, doc.note_kind = check.text, "warn"
            doc.changed.emit()
            return
        self.history.set_source(doc.entry.id, str(Path(path)))
        doc.path = Path(path)
        doc.original = doc.job.check_original()
        doc.note, doc.note_kind = "Found it. This job can be changed and saved again.", "info"
        doc.session_state = "none"
        doc.open_session()
        doc.changed.emit()

    def remove_item(self, item_id: int) -> None:
        item = self._items.get(item_id)
        if item is None or item.state == RUNNING:
            return
        self.queue.remove(item_id)
        self._queued.discard(item_id)
        self._items.pop(item_id, None)
        doc, item.doc = item.doc, None
        if doc is not None:
            self._probing.discard(doc)
            if doc in self._to_probe:
                self._to_probe.remove(doc)
            if self.doc is doc:
                self.doc = None
                self.studio.set_doc(None)
                self.jobs.set_current(None)
            doc.dispose()
        self._probe_more()
        self._draw_jobs()
        self._check_disk()

    def clear_finished(self) -> None:
        """Take the cancelled files off the list (the finished ones are jobs: they stay until they are deleted)."""
        for item in [i for i in self._items.values() if i.state in (CANCELLED,)]:
            self.remove_item(item.id)

    # -- what stops a start ---------------------------------------------------------

    def _problems(self, doc: Doc) -> list[str]:
        """What stops this file from being started (or its changes from being saved), in words: shown over the editor, and the button waits."""
        out: list[str] = []
        if self._missing:
            out.append("Audio Cleaner is missing some of its own files, so it can't clean anything until it is installed again.")
        if doc.model is None:
            return out
        if doc.is_job and doc.job is None:
            out.append("This job has no saved copies. It was cleaned before Audio Cleaner kept them, or they were removed. You can still listen to it and "
                       "delete it, but it can't be changed and saved again; clean the original again to get that.")
            return out
        needs_run = doc.stage == "ready" or doc.changed_sound or doc.changed_output
        if not needs_run:
            return out
        if doc.stage == "ready" and not self.noise_download.ready:
            out.append("Noise removal needs its models, which aren't downloaded yet (23 MB, once). They download by themselves: start when it is done.")
        if doc.subtitles != "off" and not self.speech_download.ready and (doc.stage == "ready" or doc.changed_output or doc.changed_sound):
            out.append("Subtitles need the speech model (a one-time download of 1.6 GB). Download it under Subtitles, or turn them off.")
        tl = doc.model.draft
        if tl.layers["rebuild"] and not self.voice_download.ready and (doc.stage == "ready" or doc.changed_sound):
            out.append("Voice rebuild needs its model (a one-time download of 0.5 GB). Download it under Voice, or choose Noise removal.")
        source = doc.facts.source() if doc.facts is not None else None
        if source is not None and doc.out_name:
            problem = engine.check_output_name(doc.out_name, source)[1]
            if problem:
                out.append(problem)
        elif not doc.out_name and doc.facts is not None:
            out.append("Type a name for the result.")
        if doc.is_job and doc.job is not None:
            plan = redo.plan(doc.job, self._wanted(doc), doc.original, source=source, voice_model=self.voice_download.ready,
                             speech_model=self.speech_download.ready, noise_models=self.noise_download.ready)
            for p in plan.problems:
                if p not in out:
                    out.append(p)
        return out

    # -- starting and stopping -----------------------------------------------

    def primary(self) -> None:
        """The button of the start row: Start a file that has not been started, Save changes to a result, or open the result."""
        doc = self.doc
        if doc is None or doc.model is None:
            return
        if doc.stage == "ready" and doc.item is not None:
            self.start_item(doc.item)
        elif doc.stage == "done":
            if doc.changed_sound or doc.changed_output:
                self.save_changes(doc)
            else:
                self.open_result(doc)

    def start(self) -> None:
        """Start the file in the editor (the same as the Start button)."""
        self.primary()

    def start_all(self) -> None:
        """Start every file that is waiting, each with its own choices."""
        for item in [i for i in self._items.values() if i.state == WAITING and i.id not in self._queued]:
            self.start_item(item)

    def start_item(self, item: FileItem) -> bool:
        doc = item.doc
        if self._missing or doc is None or doc.model is None or item.state == RUNNING or item.id in self._queued:
            return False
        if self._problems(doc):
            return False
        if item.state in (CANCELLED, ITEM_FAILED):
            item.state, item.progress, item.message = WAITING, 0.0, ""
        if not self.queue.busy:
            self._batch_started = time.monotonic()
            self._batch_done = self._batch_failed = 0
        item.out_name = doc.out_name
        item.options, item.timeline = doc.run_plan()
        self._queued.add(item.id)
        item.keep_dir = copies.plan_job_dir(self.history.jobs_root)  # where this run keeps the copies that let its result be changed later
        doc.stage, doc.run_fraction, doc.saving = "processing", 0.0, False
        doc.run_text = "In line: it starts when the files before it are done" if self.queue.busy else "Getting started…"
        doc.note = ""
        self.queue.add(Job(item.id, item.path, item.options, self.prefs.output_dir(), output_name=item.out_name, keep_dir=item.keep_dir,
                           timeline=item.timeline))
        if not self.queue.active:
            self.queue.start()
        doc.changed.emit()
        self._draw_jobs()
        return True

    def request_cancel_all(self) -> None:
        if not self.queue.busy:
            return
        if self._confirm("Stop cleaning?", "Files that are still being cleaned will be discarded. Finished files are kept."):
            self.queue.cancel_all()
            for item_id in self.queue.running_ids():
                self._mark_stopping(item_id)
            self._draw_jobs()

    def cancel_current(self) -> None:
        """"Cancel job" in the start row: stop the file that runs (or is in line), or the change that is being saved."""
        doc = self.doc
        if doc is None:
            return
        if self._saving is not None and doc.saving:
            self._saving.cancel.set()
            doc.run_text = "Stopping…"
            doc.changed.emit()
            return
        item = doc.item
        if item is None:
            return
        if item.state == RUNNING:
            self._stop_item(item.id)
        elif item.id in self._queued:
            self.queue.remove(item.id)
            self._queued.discard(item.id)
            if item.keep_dir is not None:
                copies.delete_job_dir(item.keep_dir.parent, item.keep_dir.name)
                item.keep_dir = None
            doc.stage, doc.run_fraction = "ready", None
            doc.changed.emit()
            self._draw_jobs()

    def _stop_item(self, item_id: int) -> None:
        self.queue.cancel(item_id)
        self._mark_stopping(item_id)
        self._draw_jobs()

    def _mark_stopping(self, item_id: int) -> None:
        """Say so at once: a file finishes its current step before it stops."""
        item = self._items.get(item_id)
        if item is not None and item.state == RUNNING:
            item.stopping = True
            if item.doc is not None:
                item.doc.run_text = "Stopping… a file finishes the step it is on first"
                item.doc.changed.emit()
            self._draw_jobs()

    def _move(self, item_id: int, delta: int) -> None:
        """A file that waits goes earlier or later in line (among the ones that wait; the queue follows for the ones already in it)."""
        ids = [i.id for i in self._items.values() if i.state == WAITING]
        if item_id not in ids:
            return
        k = ids.index(item_id)
        j = k + delta
        if not 0 <= j < len(ids):
            return
        order = list(self._items)
        slots = [order.index(i) for i in ids]
        ids[k], ids[j] = ids[j], ids[k]
        for slot, i in zip(slots, ids):
            order[slot] = i
        self._items = {i: self._items[i] for i in order}
        if item_id in self.queue.pending_ids():
            self.queue.move(item_id, delta)
        self._draw_jobs()

    def _confirm(self, title: str, text: str) -> bool:
        box = QMessageBox(QMessageBox.Icon.Question, title, text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, self)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        return box.exec() == QMessageBox.StandardButton.Yes

    # -- models ------------------------------------------------------------------

    def _downloaders(self) -> tuple[SpeechModelDownloader, ...]:
        return self.noise_download, self.speech_download, self.voice_download

    def _fetch_missing_noise_models(self) -> None:
        """The noise removal models are required and setup normally installed them. When they are not here (its
        download failed, a silent update that showed no download page, or this is not an installed copy), they are
        downloaded right away (23 MB): noise removal is what the app is for. A failure shows the banner's "Try again"."""
        if self._missing or self.noise_download.ready:
            return
        self.fetch_noise_models()

    def _models_action(self) -> None:
        """The banner's button: the noise models only while they are what is offered, else everything missing."""
        self.fetch_noise_models() if self._noise_only else self.fetch_models()

    def fetch_noise_models(self) -> None:
        """Download just the required noise removal models (the rest can wait: Settings and the next start offer them)."""
        if self._missing:
            return
        self._auto_models = False
        self._noise_only = True
        self._model_queue = [self.noise_download]
        self._next_model()

    def fetch_models(self) -> None:
        """Download every model that is not here yet, one after the other (setup normally installed them; this
        finishes a download that failed there)."""
        if self._missing:
            return
        self._auto_models = False
        self._noise_only = False
        self._model_queue = [d for d in self._downloaders() if not d.ready]
        self._next_model()

    def schedule_model_update(self, delay_ms: int = MODEL_UPDATE_DELAY_MS) -> None:
        """Once, shortly after the window opened: see update_models()."""
        QTimer.singleShot(delay_ms, self.update_models)

    def update_models(self) -> None:
        """A model the user had, whose catalog entry changed since (so the copy on disk no longer counts), is downloaded
        again in the background: the features that need it keep working without a visit to Settings. Regardless of the
        updates setting (the models are part of the app working), once per start: a failure (offline, metered, disk
        full) is not retried by itself and shows in Settings as usual. A model the user never had is not fetched here.
        When nothing is to be updated, the copies of older versions are cleaned up."""
        if self._missing or self._closing or any(d.running for d in self._downloaders()):
            return
        queue = [d for d in self._downloaders() if models.needs_update(d.model)]
        if not queue:
            self._clean_old_models()
            return
        self._auto_models = True
        self._noise_only = False
        self._model_queue = queue
        self._next_model()

    def _clean_old_models(self) -> None:
        """Delete what older versions of the models left on disk, off the GUI thread and after any download that is
        still finishing (models.remove_stale only deletes what the installed models have replaced)."""
        def work() -> None:
            for dl in self._downloaders():
                dl.wait()
            if self._closing:
                return
            try:
                removed = models.remove_stale()
            except Exception:  # noqa: BLE001 - tidying up must never bother anybody
                logging.getLogger(__name__).exception("could not delete outdated models")
                return
            if removed:
                logging.getLogger(__name__).info("deleted outdated model data: %s", [str(p) for p in removed])

        threading.Thread(target=work, name="model-cleanup", daemon=True).start()

    def _next_model(self) -> None:
        queue = getattr(self, "_model_queue", [])
        while queue and queue[0].ready:
            queue.pop(0)
        if not queue:
            self.banner_models.hide()
            return
        dl = queue[0]
        if not dl.running:
            dl.start()
        self._show_model_progress(0, dl.model.size)

    def _show_model_progress(self, done: int, total: int) -> None:
        left = models.download_bytes([d.model for d in getattr(self, "_model_queue", [])]) or total
        pct = int(100 * done / max(1, total))
        if self._noise_only:
            text = f"Downloading the noise removal models ({size_text(left)}, once; {pct}%). You can start cleaning when it is done."
        elif self._auto_models:
            text = (f"Updating the models for subtitles, phone-quality recordings and rebuilding voices "
                    f"({size_text(left)} in all; {pct}% of the current one). Cleaning works meanwhile.")
        else:
            names = "noise removal, subtitles, phone-quality recordings and rebuilding voices"
            if self.noise_download.ready:
                names, ready = names.split(", ", 1)[1], " Noise removal works already."
            else:
                ready = ""
            text = (f"Getting everything ready: downloading the models for {names} ({size_text(left)} in all, once; "
                    f"{pct}% of the current one).{ready}")
        self.banner_models.show_message(text, "info")

    def _model_download_finished(self, downloader: SpeechModelDownloader, ok: bool, message: str) -> None:
        queue = getattr(self, "_model_queue", [])
        if not ok and self._auto_models:
            # An update the app started itself gets one try per start: no banner, no retry. The reason is in Settings
            # (the model's line there), where "Try again" starts it.
            self._auto_models = False
            queue.clear()
            self.banner_models.hide()
            return
        if not ok:
            if message:  # not a cancel on quit
                self.banner_models.show_message(
                    f"A model could not be downloaded: {message}", "warn", action_text="Try again",
                )
            return
        if queue and queue[0] is downloader:
            queue.pop(0)
        if queue:
            self._next_model()
        else:
            self.banner_models.hide()
            if self._noise_only:
                self._notify(config.APP_NAME, "Noise removal is ready: you can start cleaning.")
            elif not self._auto_models:
                self._notify(config.APP_NAME, "Everything is ready: noise removal, subtitles and rebuilding voices work now.")
            self._auto_models = False
            self._noise_only = False
            self._clean_old_models()
        self._models_changed()


    # -- queue events ---------------------------------------------------------

    def _progress_text(self, item: FileItem) -> str:
        if item.stopping:
            return "Stopping… a file finishes the step it is on first"
        text = f"Cleaning {int(item.progress * 100)}%"
        elapsed = time.monotonic() - item.started_at
        if item.progress > 0.05 and elapsed > 5:
            text += f" · about {format_spent(elapsed * (1 - item.progress) / item.progress)} left"
        return text

    def _on_started(self, item_id: int) -> None:
        item = self._items.get(item_id)
        if item is not None:
            item.state, item.progress, item.started_at, item.stopping = RUNNING, 0.0, time.monotonic(), False
            item.job_name = item.out_name
            self._awake.acquire()
            if item.doc is not None:
                item.doc.stage, item.doc.run_fraction, item.doc.run_text = "processing", 0.0, "Getting started…"
                item.doc.changed.emit()
            self._draw_jobs()

    def _on_progress(self, item_id: int, fraction: float) -> None:
        item = self._items.get(item_id)
        if item is not None:
            item.progress = fraction
            if item.doc is not None:
                item.doc.run_fraction, item.doc.run_text = fraction, self._progress_text(item)
                item.doc.progressed.emit()
            self.jobs.update_progress(item)

    def _on_finished(self, item_id: int, result: engine.JobResult) -> None:
        item = self._items.get(item_id)
        self._queued.discard(item_id)
        if item is None:
            return
        paths = result.paths
        item.state, item.progress, item.outputs, item.stopping = ITEM_DONE, 1.0, paths, False
        item.duration, item.has_video = result.duration, result.has_video
        notes = []
        if item.options and item.options.subtitles != "off" and result.subtitles is None and not result.embedded:
            notes.append("no speech found for subtitles")
        item.picked_strength, item.picked_rebuild = result.strength, result.rebuilt
        if result.strength:
            notes.append(f"{dict(STRENGTH_LABELS)[result.strength]} noise reduction picked for this file: it keeps your voice exactly as recorded")
        if result.notice:
            notes.append(result.notice)
        item.message = "; ".join(notes)
        self._batch_done += 1
        entry_id = self._record(item, DONE, "", [str(p) for p in paths], kept_dir=result.kept_dir, made=result.made)
        item.entry_id = entry_id
        doc = item.doc
        if doc is not None:
            entry = self.history.get(entry_id)
            job = self._load_job(entry)
            check = job.check_original() if job is not None else copies.OriginalCheck("missing")
            doc.finished(entry, job, check)
            doc.stage, doc.run_fraction, doc.saving, doc.run_text = "done", None, False, ""
            doc.note = "Saved as " + " and ".join(p.name for p in paths) + (f" ({item.message})" if item.message else "") + "."
            doc.note_kind = "info"
            if self.doc is doc:
                doc.open_session()
            doc.changed.emit()
        self._draw_jobs()

    def _load_job(self, entry: Entry | None) -> copies.JobCopies | None:
        if entry is None or not entry.job_dir:
            return None
        folder = copies.job_dir(self.history.jobs_root, entry.job_dir)
        job = copies.JobCopies.load(folder) if folder is not None else None
        return job if job is not None and job else None

    def _on_failed(self, item_id: int, message: str) -> None:
        item = self._items.get(item_id)
        self._queued.discard(item_id)
        if item is None:
            return
        item.state, item.message, item.stopping = ITEM_FAILED, message, False
        self._batch_failed += 1
        self._record(item, FAILED, message, [], kept_dir=self._copies_left(item))
        if item.doc is not None:
            item.doc.stage, item.doc.run_fraction = "ready", None
            item.doc.note, item.doc.note_kind = message, "error"
            item.doc.changed.emit()
        self._draw_jobs()

    def _copies_left(self, item: FileItem) -> Path | None:
        """A job that failed keeps the copies it had made before it failed (the result can still be saved from them); a folder with
        none is removed. Returns the folder that stays."""
        folder, item.keep_dir = item.keep_dir, None
        if folder is None:
            return None
        job = copies.JobCopies.load(folder) if folder.is_dir() else None
        if job is not None and job:
            return folder
        copies.delete_job_dir(folder.parent, folder.name)
        return None

    def _on_cancelled(self, item_id: int) -> None:
        item = self._items.get(item_id)
        self._queued.discard(item_id)
        if item is not None:
            if item.keep_dir is not None:  # stopped: nothing is kept
                copies.delete_job_dir(item.keep_dir.parent, item.keep_dir.name)
                item.keep_dir = None
            item.state, item.progress, item.stopping = CANCELLED, 0.0, False
            if item.doc is not None:
                item.doc.stage, item.doc.run_fraction = "ready", None
                item.doc.note, item.doc.note_kind = "Stopped. Nothing was saved.", "info"
                item.doc.changed.emit()
            self._draw_jobs()

    def _on_idle(self) -> None:
        self._awake.release()
        engine.release_memory()  # an idle app does not keep the models in memory
        if self._restart_when_idle:
            QTimer.singleShot(0, self._restart_for_update)
        done, failed = self._batch_done, self._batch_failed
        if (done or failed) and not self.isActiveWindow():
            text = _plural(done, "file", "files") + " cleaned" + (f", {failed} failed" if failed else "")
            self._notify(config.APP_NAME, text)
        self._draw_jobs()


    def _notify(self, title: str, text: str) -> None:
        QApplication.alert(self, 0)
        if not (QSystemTrayIcon.isSystemTrayAvailable() and QSystemTrayIcon.supportsMessages()):
            return
        if self._tray is None:
            self._tray = QSystemTrayIcon(self.windowIcon(), self)
        self._tray.show()
        self._tray.showMessage(title, text, self.windowIcon(), 8000)
        QTimer.singleShot(12000, self._tray.hide)


    def _record(self, item: FileItem, status: str, message: str, outputs: list[str], kept_dir: Path | None = None, made: dict | None = None) -> int:
        options = item.options or self.prefs.job_options()
        elapsed = time.monotonic() - item.started_at if item.started_at else 0.0
        item.keep_dir = None
        settings = {**settings_dict(options)}
        if item.timeline is not None and made is not None:  # a run of the timeline as it stood: told with what it did
            settings.update({"leveling": made.get("leveling", True), "loudness": made.get("loudness", "podcast"),
                             "rebuild_voice": bool(made.get("rebuilt")), "strength_db": made.get("atten_lim_db"),
                             "noise": made.get("noise") if made.get("noise") in ("light", "heavy") else "auto", "echo_track": bool(made.get("echo"))})
            if made.get("parts"):
                settings["parts"] = True
        if status == DONE and item.picked_strength:
            settings["picked_strength"] = item.picked_strength
        if status == DONE and item.picked_rebuild:
            settings["picked_rebuild"] = item.picked_rebuild
        return self.history.add(Entry(
            name=item.path.name, source=str(item.path), duration=item.duration, has_video=item.has_video,
            settings=settings, seconds_taken=elapsed, outputs=outputs, status=status, message=message,
            job_dir=kept_dir.name if kept_dir is not None else "", copies_bytes=copies.scan_bytes(kept_dir) if kept_dir is not None else 0,
        ))

    # -- saving a change to a finished job ----------------------------------------------------------------------

    def _wanted(self, doc: Doc) -> redo.Wanted:
        return redo.Wanted(subtitles=doc.subtitles, language=doc.language if doc.language in engine.LANGUAGES else "auto", video_size=doc.video_size,
                           output_name=doc.out_name.strip(), timeline=doc.model.effective().to_json() if doc.model is not None else None)

    def ask_replace(self, name: str) -> str:
        """A file with the chosen name is next to the old result: "replace" (the old one goes to the bin), "new" (another name) or "cancel"."""
        box = QMessageBox(QMessageBox.Icon.Question, "A file with that name exists",
                          f"{name} already exists. Replace it with the new file?\n\nThe old one goes to the {shell.TRASH_NAME}, so you can bring it back.",
                          QMessageBox.StandardButton.NoButton, self)
        replace = box.addButton("Replace it", QMessageBox.ButtonRole.DestructiveRole)
        new = box.addButton("Keep both", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        return "replace" if clicked is replace else "new" if clicked is new else "cancel"

    def save_changes(self, doc: Doc, *, run=None) -> None:
        """"Save changes": make the result for the draft and the choices as they are, from the job's saved copies, as a new file, in the same editor."""
        job, entry = doc.job, doc.entry
        run = run or self._redo
        if job is None or entry is None or self._saving is not None or self._problems(doc):
            return
        wanted = self._wanted(doc)
        target = redo.result_dir(job) / wanted.output_name
        replace = None
        if target.exists():
            answer = self.ask_replace(wanted.output_name)
            if answer == "cancel":
                return
            replace = target if answer == "replace" else None
        if replace is not None:
            self.studio.release_player()  # the file that goes to the bin may be the one the player has open
        worker = Worker(lambda cancel, progress: run(job, wanted, progress=progress, cancel=cancel, replace=replace, trash=shell.move_to_trash), self)
        worker.progress.connect(lambda f: self._save_progress(doc, f))
        worker.done.connect(lambda result: self._saved(doc, result))
        worker.failed.connect(lambda m: self._save_over(doc, m, "error"))
        worker.stopped.connect(lambda: self._save_over(doc, "Stopped. Nothing was saved.", "info"))
        self._saving = worker
        doc.stage, doc.saving, doc.run_fraction, doc.run_text, doc.note = "processing", True, 0.0, "Making the parts that changed…", ""
        doc.changed.emit()
        worker.start()

    def _save_progress(self, doc: Doc, fraction: float) -> None:
        doc.run_fraction, doc.run_text = fraction, f"Saving {int(fraction * 100)}%"
        doc.progressed.emit()

    def _finish_save(self, doc: Doc) -> None:
        worker, self._saving = self._saving, None
        if worker is not None:
            worker.wait()
            worker.deleteLater()
        doc.stage, doc.saving, doc.run_fraction, doc.run_text = "done", False, None, ""

    def _save_over(self, doc: Doc, text: str, kind: str) -> None:
        self._finish_save(doc)
        doc.note, doc.note_kind = text, kind
        doc.changed.emit()

    def _saved(self, doc: Doc, result: engine.JobResult) -> None:
        self._finish_save(doc)
        entry, job = doc.entry, doc.job
        fresh = copies.JobCopies.load(job.dir)  # the job file was changed by the thread: read it again
        job = fresh if fresh is not None else job
        made = result.made or {}
        settings = {**entry.settings, "leveling": made.get("leveling", True), "loudness": made.get("loudness", "podcast"),
                    "subtitles": made.get("subtitles", "off"), "rebuild_voice": bool(made.get("rebuilt")), "strength_db": made.get("atten_lim_db")}
        settings.pop("picked_strength", None)
        settings.pop("picked_rebuild", None)
        if made.get("parts"):
            settings["parts"] = True
        else:
            settings.pop("parts", None)
        self.history.set_job_result(entry.id, [str(p) for p in result.paths], settings, job.dir.name, job.bytes())
        doc.original = job.check_original()
        doc.saved(self.history.get(entry.id), job)
        doc.note, doc.note_kind = f"Saved {result.output.name}" + (f" ({result.notice})" if result.notice else "") + ".", "info"
        doc.changed.emit()
        self._draw_jobs()

    def save_subtitles_only(self, doc: Doc | None = None) -> None:
        """The subtitle file of a job from the transcript it saved: needs neither the original nor a model."""
        doc = doc or self.doc
        job, entry = (doc.job, doc.entry) if doc is not None else (None, None)
        if job is None or entry is None or doc.subtitles not in redo.TEXT_KINDS or self._saving is not None:
            return
        last = redo.last_result(job)
        folder = redo.result_dir(job)
        stem = engine.split_name(last.name)[0] if last is not None else job.original_path.stem + "-cleaned"
        try:
            path = redo.save_subtitles(job, doc.subtitles, folder, stem)
        except (MediaError, OSError) as e:
            doc.note, doc.note_kind = str(e), "error"
            doc.changed.emit()
            return
        job.add_save([path], {**job.made, "subtitles": doc.subtitles})
        self.history.set_job_result(entry.id, [*entry.outputs, str(path)], entry.settings, job.dir.name, job.bytes())
        doc.entry = self.history.get(entry.id)
        doc.note, doc.note_kind = f"Saved {path.name}.", "info"
        doc.changed.emit()
        self._draw_jobs()

    def open_result(self, doc: Doc) -> None:
        """"Download again" of this app: the file, in the player the computer uses for it."""
        files = doc.result_paths()
        main = next((p for p in files if p.suffix.lower() not in (".srt", ".vtt", ".txt") and p.exists()), None) or next((p for p in files if p.exists()), None)
        if main is None or not shell.open_path(main):
            doc.note, doc.note_kind = "This file has been moved or deleted.", "warn"
            doc.changed.emit()

    # -- deleting -------------------------------------------------------------------------------------------------

    def _stayed(self, ids: list[int]) -> None:
        if ids:
            QMessageBox.warning(self, "Delete", "Some saved copies could not be removed (a file may be in use). The job stays in the list; try again later.")

    def delete_current(self) -> None:
        if self.doc is not None and self.doc.entry is not None:
            self.delete_entry(self.doc.entry.id)

    def delete_entry(self, entry_id: int) -> None:
        """Delete one job after asking: its saved copies and its place in the list; never the original or the cleaned files."""
        entry = self.history.get(entry_id)
        if entry is None:
            return
        if self._saving is not None and self.doc is not None and self.doc.entry is not None and self.doc.entry.id == entry_id:
            return  # a change is being saved from its copies
        if entry.job_dir:
            text = (f"Delete \"{entry.name}\"?\n\nIts saved copies ({format_size(entry.copies_bytes)}) and its place in this list are deleted, so it can't "
                    "be changed and saved again. Your original and the cleaned files are not touched.")
        else:
            text = f"Remove \"{entry.name}\" from the list?\n\nYour original and the cleaned files are not touched."
        if not self._confirm("Delete job", text):
            return
        for item in self._items.values():
            if item.entry_id == entry_id:
                item.entry_id = 0
        if self.doc is not None and self.doc.entry is not None and self.doc.entry.id == entry_id:
            gone = self.doc
            self._show(None)
            if gone.item is not None:
                self._items.pop(gone.item.id, None)
                gone.dispose()
        stayed = self.history.delete([entry_id])
        self._draw_jobs()
        self._stayed(stayed)

    def delete_all(self) -> None:
        c = self.history.copies_total()
        n = self.history.count()
        what = f" and the {format_size(c.bytes)} of saved copies" if c.jobs else ""
        if not self._confirm("Delete all jobs", f"Delete all {n} job{'s' if n != 1 else ''}{what}?\n\nYour originals and cleaned files are not touched, "
                                                "but the jobs can't be changed and saved again."):
            return
        if self._saving is not None:
            return
        if self.doc is not None and self.doc.entry is not None:
            gone = self.doc
            self._show(None)
            if gone.item is not None:
                self._items.pop(gone.item.id, None)
                gone.dispose()
        for item in [i for i in self._items.values() if i.state in (ITEM_DONE, ITEM_FAILED)]:
            self._items.pop(item.id, None)
            if item.doc is not None:
                item.doc.dispose()
        stayed = self.history.clear()
        self._draw_jobs()
        self._stayed(stayed)

    # -- state ------------------------------------------------------------------

    def _settings_changed(self) -> None:
        self.queue.set_parallel(self.prefs.parallel)
        for item in self._items.values():
            if item.state == WAITING and not item.out_custom and item.source is not None and item.doc is not None:
                item.out_name = engine.default_output_name(item.path, item.source, self.prefs.job_options())  # the usual name follows the type of audio files
                item.doc.out_name = item.out_name
                item.doc.changed.emit()
        if self.doc is not None:
            self.studio._schedule()
        self._check_disk()

    def _models_changed(self) -> None:
        """A model arrived (or did not): what needs it can start (or says so)."""
        if self.doc is not None:
            self.studio._schedule()
            self.studio.inspector._refresh_models()

    def _draw_jobs(self) -> None:
        if self._closing:
            return
        self.jobs.current = self._key_of(self.doc)
        self.jobs.refresh()
        self.studio._schedule() if self.doc is not None else None
        self._check_disk()

    # -- updates --------------------------------------------------------------------------------------------

    def _update_ready(self) -> None:
        """A newer version has been downloaded (updater.py checked its signature and checksum)."""
        version = self.updater.ready_version
        if not version:
            self.banner_update.hide()
        elif sys.platform == "darwin":
            self.banner_update.show_message(f"Audio Cleaner {version} is downloaded and ready to install.", "info", action_text="Install")
        else:
            self.banner_update.show_message(
                f"Audio Cleaner {version} is ready. It installs when you close the app.", "info", action_text="Restart now")

    def _update_action(self) -> None:
        version = self.updater.ready_version
        if not version:
            return
        if sys.platform == "darwin":
            if self.updater.open_installer():
                self.banner_update.show_message(
                    f"The installer for {version} is open. Quit Audio Cleaner while it installs.", "info")
            else:
                self.banner_update.show_message("The update could not be opened. It will be downloaded again the next time.", "warn")
            return
        if self.queue.busy:  # never interrupt a batch
            self._restart_when_idle = True
            self.banner_update.show_message(
                f"Audio Cleaner will restart to install {version} when the files are finished.", "info")
            return
        self._restart_for_update()

    def _restart_for_update(self) -> None:
        """Close the app; setup then installs the new version and starts it (updater.install_on_close)."""
        self._restart_when_idle = False
        if self.queue.busy or self._closing:
            return
        self.updater.relaunch = True
        self.close()
        if not self._closing:  # closing was refused: stay as we were
            self.updater.relaunch = False

    def _check_disk(self) -> None:
        text = self._disk_warning()
        if text != self._disk_text:
            self._disk_text = text
            if text:
                self.banner_disk.show_message(text, "warn")
            else:
                self.banner_disk.hide()

    def _disk_warning(self) -> str:
        """A message when a folder looks too full for the temporary files the waiting jobs need."""
        out_dir = self.prefs.output_dir()
        subtitles = self.prefs.subtitles != "off"
        by_folder: dict[Path, list[float]] = {}
        for item in self._items.values():
            if item.state == WAITING and item.duration:
                folder = out_dir or item.path.parent
                by_folder.setdefault(folder, []).append(engine.estimated_scratch_gb(item.duration, item.channels))
        at_once = self.queue.limit(subtitles, self.prefs.job_options().may_rebuild, self.prefs.echo_track)
        scratch = engine.scratch_root()  # where the work files go on Windows and macOS, while it has room
        for folder, needs in by_folder.items():
            need = sum(sorted(needs, reverse=True)[:at_once]) * SCRATCH_SAFETY
            free = self._free_gb(folder)
            if free is None:
                continue
            if scratch is not None:
                free_scratch = self._free_gb(scratch)
                if free_scratch is not None and free_scratch >= need:
                    continue
            if free < need:
                return (
                    f"There may not be enough free space for the temporary files in {folder}: about {need:.1f} GB is needed "
                    f"while cleaning and {free:.1f} GB is free. Free up some space or choose another folder to save to."
                )
        return ""

    @staticmethod
    def _free_gb(folder: Path) -> float | None:
        probe = folder
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            return shutil.disk_usage(probe).free / 1e9
        except OSError:
            return None


    # -- window events ------------------------------------------------------------

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._reflow()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._reflow()

    def _reflow(self) -> None:
        """Wide: the jobs are a column beside the editor. Narrow: they are under it."""
        width = self.scroll.viewport().width() if hasattr(self, "scroll") else self.width()
        beside = width >= JOBS_BESIDE
        room = width - 32 - ((JOBS_WIDTH + 14) if beside else 0)  # what the editor and the settings have between them
        self.studio.set_narrow(room < NARROW)
        want = QBoxLayout.Direction.LeftToRight if beside else QBoxLayout.Direction.TopToBottom
        if self.body.direction() != want:
            self.body.setDirection(want)
            self.jobs.setMinimumWidth(JOBS_WIDTH if beside else 0)
            self.jobs.setMaximumWidth(JOBS_WIDTH if beside else 16777215)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            set_property(self.studio.drop, "over", True)

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        set_property(self.studio.drop, "over", False)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        set_property(self.studio.drop, "over", False)
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            event.acceptProposedAction()
            self.add_paths(paths)

    def closeEvent(self, event) -> None:  # noqa: N802
        if (self.queue.busy or self._saving is not None) and not self._confirm(
            "Quit Audio Cleaner?", "Files are still being cleaned. If you quit now, they will be discarded."
        ):
            event.ignore()
            return
        self._closing = True
        self._pool.clear()
        self.updater.cancel()  # a download resumes from where it stopped next time
        for dl in self._downloaders():
            dl.cancel()  # resumes from where it stopped next time
        for signal in (self.queue.itemFinished, self.queue.itemFailed, self.queue.itemCancelled, self.queue.idle):
            try:
                signal.disconnect()  # nothing may reach the history once it is closed
            except (RuntimeError, TypeError):
                pass
        stopped = self.queue.shutdown(QUIT_WAIT_MS)
        for dl in self._downloaders():
            dl.wait(0.5)
        self._pool.waitForDone(QUIT_WAIT_MS)
        self._awake.release()
        if self._saving is not None:  # a save stops; its scratch files go
            self._saving.cancel.set()
            stopped = self._saving.wait(QUIT_WAIT_MS) and stopped
        # (what runs for each file stops, and the live sound deletes its scratch files)
        docs = [i.doc for i in self._items.values() if i.doc is not None]
        if self.doc is not None and self.doc not in docs:
            docs.append(self.doc)
        for doc in docs:
            doc.dispose()
        self.studio.shutdown()
        self._qs.setValue("geometry", self.saveGeometry())
        self._qs.sync()
        self.history.close()
        self.updater.install_on_close()  # Windows: setup for a downloaded update starts now and outlives the app
        super().closeEvent(event)
        if not stopped:
            # A file is in the middle of a step that does not stop quickly. Leave without waiting for it: what it
            # left in the scratch folder is swept away the next time the app starts.
            logging.shutdown()
            os._exit(0)

    def _sweep_leftovers(self) -> None:
        if self._closing:
            return
        folders = [*self.history.recent_folders(), self.history.jobs_root]  # (the live sound's scratch files are in the jobs folder)
        protect = {i.keep_dir.name for i in self._items.values() if i.keep_dir is not None}
        try:
            self.history.reconcile(protect=protect)  # the sizes of the saved copies, and folders nothing in the list points to
        except Exception:  # noqa: BLE001 - housekeeping must never bother the user
            log.exception("checking the saved copies failed")
        out = self.prefs.output_dir()
        if out is not None:
            folders.append(out)
        self._pool.start(_SweepTask(folders))
