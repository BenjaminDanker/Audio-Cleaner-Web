"""The main window: Clean tab (queue, settings, preview) and History tab."""
from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from collections.abc import Iterable
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QSettings, QStandardPaths, Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSystemTrayIcon, QTabWidget, QVBoxLayout, QWidget,
)

from acw_common.ffmpeg import MediaError

from .. import __version__, config, engine, resources, sysinfo
from . import media, queue_view, shell, tips
from .about import AboutDialog
from .history import DONE, FAILED, Entry, HistoryStore
from .history_view import HistoryPage, format_spent, settings_dict
from .player import PreviewPane
from .preferences import Preferences
from .queue_view import CANCELLED, DONE as ITEM_DONE, FAILED as ITEM_FAILED, RUNNING, WAITING, FileItem, FileRow, QueueView
from .settings_panel import SettingsPanel
from .speech_model import SpeechModelDownloader, size_text
from .theme import set_property
from .widgets import Banner
from .worker import Job, JobQueue

log = logging.getLogger(__name__)

SETTINGS_SHARE = 0.34  # of the window width
SETTINGS_MIN_WIDTH = 340
SETTINGS_MAX_WIDTH = 520
SCRATCH_SAFETY = engine.SCRATCH_SAFETY
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


class _ProbeSignals(QObject):
    done = Signal(int, object, str)  # id, ProbeResult | None, error message


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


class _ProbeTask(QRunnable):
    """Reads a file's length and kind in the background so the list can show them."""

    def __init__(self, item_id: int, path: Path):
        super().__init__()
        self.signals = _ProbeSignals()
        self._id, self._path = item_id, path

    def run(self) -> None:
        try:
            result, error = engine.probe(self._path), ""
        except MediaError as e:
            result, error = None, str(e)
        except Exception:  # noqa: BLE001 - never let a bad file kill the pool thread
            log.exception("probing %s failed", self._path)
            result, error = None, "This file could not be read."
        self.signals.done.emit(self._id, result, error)


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
    ):
        super().__init__()
        self.setWindowTitle(config.APP_NAME)
        self.setMinimumSize(900, 520)
        self.setAcceptDrops(True)
        self._qs = settings if settings is not None else QSettings()
        self.prefs = Preferences(self._qs)
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
        self._history_stale = False
        self._closing = False

        self._build_menus()
        self._build_ui()
        self._wire()
        geometry = self._qs.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        else:
            self.resize(1080, 720)
        self._fit_settings()
        self.add_paths(files)
        self._refresh_state()
        QTimer.singleShot(SWEEP_DELAY_MS, self._sweep_leftovers)

    # -- construction -------------------------------------------------------

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        self._act_add = QAction("&Add files...", self, shortcut=QKeySequence.StandardKey.Open)
        self._act_add.setStatusTip("Choose video or audio files")
        self._act_folder = QAction("Add &folder...", self, shortcut=QKeySequence("Ctrl+Shift+O"))
        quit_action = QAction("&Quit", self, shortcut=QKeySequence.StandardKey.Quit, menuRole=QAction.MenuRole.QuitRole)
        quit_action.triggered.connect(self.close)
        file_menu.addActions([self._act_add, self._act_folder])
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        help_menu = self.menuBar().addMenu("&Help")
        about = QAction(f"&About {config.APP_NAME}", self, menuRole=QAction.MenuRole.AboutRole)
        about.triggered.connect(lambda: AboutDialog(self).exec())
        logs = QAction("Open &log folder", self)
        logs.triggered.connect(lambda: shell.open_path(log_dir()))
        help_menu.addActions([about, logs])

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(24, 12, 24, 10)
        outer.setSpacing(8)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(False)
        outer.addWidget(self.tabs, 1)

        self.tabs.addTab(self._build_clean_tab(), "Clean")
        self.history_page = HistoryPage(self.history)
        self.tabs.addTab(self.history_page, "History")

        footer = QHBoxLayout()
        version = QLabel(f"Version {__version__}")
        version.setObjectName("Small")
        footer.addStretch(1)
        footer.addWidget(version)
        outer.addLayout(footer)
        self.setCentralWidget(root)

    def _build_clean_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 16, 0, 0)
        layout.setSpacing(12)

        self.banner_missing = Banner()
        self.banner_disk = Banner()
        self.banner_models = Banner()
        self.banner_models.action.connect(self.fetch_models)
        self._model_queue: list[SpeechModelDownloader] = []
        layout.addWidget(self.banner_missing)
        layout.addWidget(self.banner_disk)
        layout.addWidget(self.banner_models)
        if self._missing:
            self.banner_missing.show_message(
                "Audio Cleaner is missing some of its own files (" + ", ".join(self._missing) + "). "
                "Please install it again.", "error",
            )

        body = QHBoxLayout()
        body.setSpacing(16)
        layout.addLayout(body, 1)

        left = QVBoxLayout()
        left.setSpacing(12)
        self.queue_card = QFrame()
        self.queue_card.setObjectName("Card")
        card = QVBoxLayout(self.queue_card)
        card.setContentsMargins(10, 10, 10, 6)
        card.setSpacing(4)
        self.drop_zone = queue_view.DropZone()
        self.view = QueueView()
        card.addWidget(self.drop_zone, 1)
        card.addWidget(self.view, 1)
        left.addWidget(self.queue_card, 1)
        self.preview: PreviewPane | None = None  # created on first use, so QtMultimedia only loads when needed
        self._left = left
        body.addLayout(left, 1)

        self.speech_download = SpeechModelDownloader(self)
        self.highs_download = SpeechModelDownloader(self, resources.HIGHS_MODEL)
        self.voice_download = SpeechModelDownloader(self, resources.VOICE_MODEL)
        for dl in self._downloaders():
            dl.finished.connect(lambda ok, message, dl=dl: self._model_download_finished(dl, ok, message))
            dl.progress.connect(lambda done, total, dl=dl: self._model_queue and self._model_queue[0] is dl
                                and self._show_model_progress(done, total))
        self.settings_panel = SettingsPanel(
            self.prefs, self.speech_download, highs_downloader=self.highs_download, voice_downloader=self.voice_download,
        )
        self._settings_scroll = scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self.settings_panel)
        scroll.viewport().setAutoFillBackground(False)
        body.addWidget(scroll)

        bottom = QHBoxLayout()
        bottom.setSpacing(12)
        status = QVBoxLayout()
        status.setSpacing(6)
        self.status = QLabel()
        self.status.setObjectName("Muted")
        self.overall = QProgressBar()
        self.overall.setRange(0, 1000)
        self.overall.setTextVisible(False)
        self.overall.setToolTip("Progress of all the files together")
        status.addWidget(self.status)
        status.addWidget(self.overall)
        bottom.addLayout(status, 1)
        self.clear_button = QPushButton("Clear finished")
        self.clear_button.setToolTip("Take the finished, failed and cancelled files off the list. The files themselves stay.")
        bottom.addWidget(self.clear_button)
        self.primary = QPushButton()
        self.primary.setObjectName("Primary")
        self.primary.setMinimumWidth(170)
        bottom.addWidget(self.primary)
        layout.addLayout(bottom)

        self.setTabOrder(self.drop_zone, self.view)
        self.setTabOrder(self.view, scroll)
        return page

    def _wire(self) -> None:
        self._act_add.triggered.connect(self.choose_files)
        self._act_folder.triggered.connect(self.choose_folder)
        self.drop_zone.addFiles.connect(self.choose_files)
        self.drop_zone.addFolder.connect(self.choose_folder)
        self.primary.clicked.connect(self._primary_clicked)
        self.clear_button.clicked.connect(self.clear_finished)
        self.settings_panel.changed.connect(self._settings_changed)
        self.tabs.currentChanged.connect(self._tab_changed)

        q = self.queue
        q.itemStarted.connect(self._on_started)
        q.itemProgress.connect(self._on_progress)
        q.itemFinished.connect(self._on_finished)
        q.itemFailed.connect(self._on_failed)
        q.itemCancelled.connect(self._on_cancelled)
        q.idle.connect(self._on_idle)

        for keys in ("Ctrl+Return", "Ctrl+Enter"):
            QShortcut(QKeySequence(keys), self, activated=self.start)
        QShortcut(QKeySequence("Esc"), self, activated=self._escape)

    def _escape(self) -> None:
        """Esc stops the cleaning, unless it is meant for something else: closing a help text, or clearing the
        History search."""
        if tips.is_showing() or self.tabs.currentIndex() != 0:
            return
        self.request_cancel_all()

    def _tab_changed(self, index: int) -> None:
        if index == 1 and self._history_stale:
            self._history_stale = False
            self.history_page.refresh()

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
        """Queue files (folders are searched). Returns how many were added."""
        known = {i.path for i in self._items.values() if i.state in (WAITING, RUNNING)}
        finished = {i.path: i for i in self._items.values() if i.state in (ITEM_DONE, ITEM_FAILED, CANCELLED)}
        added = 0
        for path in media.expand(paths):
            if path in known:
                continue
            if path in finished:  # already in the list: clean it again in its own row instead of adding a twin
                self._requeue(finished.pop(path))
                added += 1
                continue
            item = FileItem(self._next_id, path, has_video=path.suffix.lower() in media.VIDEO_EXTENSIONS)
            self._next_id += 1
            self._items[item.id] = item
            row = FileRow(item)
            row.removeRequested.connect(self.remove_item)
            row.cancelRequested.connect(self._stop_item)
            row.previewRequested.connect(self.show_preview)
            row.revealRequested.connect(self._reveal)
            row.againRequested.connect(self.clean_again)
            row.moveRequested.connect(self._move)
            self.view.add(row)
            if self._missing:
                item.probed = True
            else:
                task = _ProbeTask(item.id, path)
                task.signals.done.connect(self._on_probed)
                self._pool.start(task)
            if self.queue.active:
                self._enqueue(item)
            added += 1
        self._refresh_state()
        return added

    def _requeue(self, item: FileItem) -> None:
        """Put a finished, failed or cancelled row back in line (with the settings at the time it starts)."""
        item.replaces = {p: st for p, st in item.made.items() if p in item.outputs}
        item.state, item.progress, item.message, item.outputs = WAITING, 0.0, "", []
        self._refresh_row(item)
        if self.queue.active:
            self._enqueue(item)

    def clean_again(self, item_id: int) -> None:
        """Clean one finished row again with the current settings, right away."""
        item = self._items.get(item_id)
        if item is None or item.state not in (ITEM_DONE, ITEM_FAILED, CANCELLED) or self._missing:
            return
        self._requeue(item)
        if not self.queue.active:
            self._batch_started = time.monotonic()
            self._batch_done = self._batch_failed = 0
            self._enqueue(item)
            self.queue.start()
        self._refresh_state()

    def _take_place_of_previous(self, item: FileItem, paths: list[Path]) -> tuple[list[Path], bool]:
        """After "Clean again": give each new file the name of the earlier result of the same type, once that
        earlier file is safely in the Recycle Bin / Trash. Files that were changed since, or that can't be
        moved to the bin, are left alone and the new file keeps its own name. When the results now go to another
        folder, the earlier ones stay where they are. An earlier subtitle file the new run did not write again
        goes to the bin with the earlier result it belonged to."""
        replaced = False
        out = []
        for new in paths:
            target = next((p for p in item.replaces if p.suffix.lower() == new.suffix.lower()), None)
            if target is not None and target.parent == new.parent:
                item.replaces.pop(target)
                if self._unchanged(item, target) and target != new and shell.move_to_trash(target):
                    replaced = True  # the earlier version is safe in the bin, whatever happens to the name
                    try:
                        os.replace(new, target)
                        new = target
                    except OSError:
                        log.warning("could not rename %s to %s", new, target, exc_info=True)
            out.append(new)
        if replaced:
            for stale in [p for p in item.replaces if p.suffix.lower() in (".srt", ".vtt", ".txt")]:
                if self._unchanged(item, stale):
                    shell.move_to_trash(stale)
        item.replaces = {}
        return out, replaced

    @staticmethod
    def _unchanged(item: FileItem, path: Path) -> bool:
        try:
            st = path.stat()
            return (st.st_size, st.st_mtime_ns) == item.made.get(path)
        except OSError:
            return False

    def _enqueue(self, item: FileItem) -> None:
        item.options = self.prefs.job_options()
        self._queued.add(item.id)
        self.queue.add(Job(item.id, item.path, item.options, self.prefs.output_dir()))

    @Slot(int, object, str)
    def _on_probed(self, item_id: int, result, error: str) -> None:
        item = self._items.get(item_id)
        if item is None:
            return
        item.probed = True
        if result is not None:
            item.duration, item.has_video, item.channels = result.duration, result.has_video, result.audio_channels
        elif item.state == WAITING:
            self._queued.discard(item_id)
            self.queue.remove(item_id)
            item.state, item.message = ITEM_FAILED, error
            self._record(item, FAILED, error, [])
        self._refresh_row(item)
        self._refresh_state()

    # -- starting and stopping -----------------------------------------------

    def start(self) -> None:
        if self._missing or self.queue.busy:
            return
        if self.prefs.subtitles != "off" and not self.speech_download.ready and not self._ask_for_model(self.speech_download):
            return
        if self.prefs.rebuild_voice and not self.voice_download.ready and not self._ask_for_model(self.voice_download):
            return
        self._batch_started = time.monotonic()
        self._batch_done = self._batch_failed = 0
        for row in self.view.rows():
            if row.item.state == WAITING:
                self._enqueue(row.item)
        self.queue.start()
        self._refresh_state()

    def _ask_for_model(self, downloader: SpeechModelDownloader) -> bool:
        """A model that a setting needs is not here: setup downloads them, so this only happens when that
        download failed (offline during setup). Offers to download it now; returns False (nothing is started
        now)."""
        voice = downloader is self.voice_download
        what = "The model that rebuilds the voice" if voice else "The speech model for subtitles"
        off = "turn Rebuild the voice off" if voice else "turn subtitles off"
        if downloader.running:
            QMessageBox.information(
                self, config.APP_NAME,
                f"{what} is still downloading (the progress is shown at the top). Start again when it is done, "
                f"or {off} to clean the files now.",
            )
            return False
        box = QMessageBox(
            QMessageBox.Icon.Question, config.APP_NAME,
            f"{what} was not installed, probably because the download during setup did not work. Download it now "
            f"({size_text(downloader.model.size)})?\n\n"
            f"You can start cleaning when it is done, or {off} to clean the files right away.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, self,
        )
        box.button(QMessageBox.StandardButton.Yes).setText("Download")
        box.button(QMessageBox.StandardButton.No).setText("Not now")
        if box.exec() == QMessageBox.StandardButton.Yes:
            self.fetch_models()
        return False

    def _downloaders(self) -> tuple[SpeechModelDownloader, ...]:
        return self.speech_download, self.highs_download, self.voice_download

    def fetch_models(self) -> None:
        """Download every model that is not here yet, one after the other (setup normally installed them; this
        finishes a download that failed there). Noise removal works meanwhile."""
        if self._missing:
            return
        self._model_queue = [d for d in self._downloaders() if not d.ready]
        self._next_model()

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
        left = sum(d.model.size for d in getattr(self, "_model_queue", [])) or total
        pct = int(100 * done / max(1, total))
        self.banner_models.show_message(
            f"Getting everything ready: downloading the models for subtitles, phone-quality recordings and "
            f"rebuilding voices ({size_text(left)} in all, once; {pct}% of the current one). Noise removal works already.",
            "info",
        )

    def _model_download_finished(self, downloader: SpeechModelDownloader, ok: bool, message: str) -> None:
        queue = getattr(self, "_model_queue", [])
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
            self._notify(config.APP_NAME, "Everything is ready: subtitles, restoring highs and rebuilding voices work now.")
        self._refresh_state()

    def _primary_clicked(self) -> None:
        if self.queue.busy:
            self.request_cancel_all()
        else:
            self.start()

    def request_cancel_all(self) -> None:
        if not self.queue.busy:
            return
        if self._confirm("Stop cleaning?", "Files that are still being cleaned will be discarded. Finished files are kept."):
            self.queue.cancel_all()
            for item_id in self.queue.running_ids():
                self._mark_stopping(item_id)
            self._refresh_state()

    def _stop_item(self, item_id: int) -> None:
        self.queue.cancel(item_id)
        self._mark_stopping(item_id)
        self._refresh_state()

    def _mark_stopping(self, item_id: int) -> None:
        """Say so at once: a file finishes its current step before it stops."""
        item = self._items.get(item_id)
        if item is not None and item.state == RUNNING:
            item.stopping = True
            self._refresh_row(item)

    def _confirm(self, title: str, text: str) -> bool:
        box = QMessageBox(QMessageBox.Icon.Question, title, text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, self)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        return box.exec() == QMessageBox.StandardButton.Yes

    def remove_item(self, item_id: int) -> None:
        item = self._items.get(item_id)
        if item is None or item.state == RUNNING:
            return
        self.queue.remove(item_id)
        self._queued.discard(item_id)
        self._items.pop(item_id, None)
        self.view.remove(item_id)
        self._refresh_state()

    def clear_finished(self) -> None:
        for item in [i for i in self._items.values() if i.state in (ITEM_DONE, ITEM_FAILED, CANCELLED)]:
            self.remove_item(item.id)

    def _move(self, item_id: int, delta: int) -> None:
        if self.view.move_among_waiting(item_id, delta) and item_id in self.queue.pending_ids():
            self.queue.move(item_id, delta)

    # -- queue events ---------------------------------------------------------

    def _on_started(self, item_id: int) -> None:
        item = self._items.get(item_id)
        if item is not None:
            item.state, item.progress, item.started_at, item.stopping = RUNNING, 0.0, time.monotonic(), False
            self._awake.acquire()
            self._refresh_row(item)
            self._refresh_state()

    def _on_progress(self, item_id: int, fraction: float) -> None:
        item = self._items.get(item_id)
        if item is not None:
            item.progress = fraction
            self._refresh_row(item)
            self._update_overall()

    def _on_finished(self, item_id: int, result: engine.JobResult) -> None:
        item = self._items.get(item_id)
        self._queued.discard(item_id)
        if item is None:
            return
        previewing = self.preview is not None and self.preview.isVisible() and self.preview.key == item_id
        if previewing and item.replaces:
            self.preview.release()
        paths, replaced = self._take_place_of_previous(item, result.paths) if item.replaces else (result.paths, False)
        item.state, item.progress, item.outputs, item.stopping = ITEM_DONE, 1.0, paths, False
        item.duration, item.has_video = result.duration, result.has_video
        item.made = {}
        for p in paths:
            try:
                st = p.stat()
                item.made[p] = (st.st_size, st.st_mtime_ns)
            except OSError:
                pass
        notes = []
        if replaced:
            notes.append(f"the earlier version is in the {shell.TRASH_NAME}")
        if item.options and item.options.subtitles != "off" and result.subtitles is None and not result.embedded:
            notes.append("no speech found for subtitles")
        if result.notice:
            notes.append(result.notice)
        item.message = "; ".join(notes)
        self._batch_done += 1
        self._record(item, DONE, "", [str(p) for p in paths])
        self._refresh_row(item)
        if previewing:
            self.preview.show_files(item.id, item.path, paths[0], item.has_video, item.path.name)
        self._refresh_state()

    def _on_failed(self, item_id: int, message: str) -> None:
        item = self._items.get(item_id)
        self._queued.discard(item_id)
        if item is None:
            return
        item.state, item.message, item.stopping = ITEM_FAILED, message, False
        self._batch_failed += 1
        self._record(item, FAILED, message, [])
        self._refresh_row(item)
        self._refresh_state()

    def _on_cancelled(self, item_id: int) -> None:
        item = self._items.get(item_id)
        self._queued.discard(item_id)
        if item is not None:
            item.state, item.progress, item.stopping = CANCELLED, 0.0, False
            self._refresh_row(item)
            self._refresh_state()

    def _on_idle(self) -> None:
        self._awake.release()
        engine.release_memory()  # an idle app does not keep the models in memory
        self._refresh_state()
        done, failed = self._batch_done, self._batch_failed
        if (done or failed) and not self.isActiveWindow():
            text = _plural(done, "file", "files") + " cleaned" + (f", {failed} failed" if failed else "")
            self._notify(config.APP_NAME, text)

    def _notify(self, title: str, text: str) -> None:
        QApplication.alert(self, 0)
        if not (QSystemTrayIcon.isSystemTrayAvailable() and QSystemTrayIcon.supportsMessages()):
            return
        if self._tray is None:
            self._tray = QSystemTrayIcon(self.windowIcon(), self)
        self._tray.show()
        self._tray.showMessage(title, text, self.windowIcon(), 8000)
        QTimer.singleShot(12000, self._tray.hide)

    def _record(self, item: FileItem, status: str, message: str, outputs: list[str]) -> None:
        options = item.options or self.prefs.job_options()
        elapsed = time.monotonic() - item.started_at if item.started_at else 0.0
        self.history.add(Entry(
            name=item.path.name, source=str(item.path), duration=item.duration, has_video=item.has_video,
            settings=settings_dict(options), seconds_taken=elapsed, outputs=outputs, status=status, message=message,
        ))
        if self.tabs.currentIndex() == 1:
            self.history_page.refresh()
        else:
            self._history_stale = True  # drawn when the tab is opened

    # -- preview / reveal -----------------------------------------------------

    def show_preview(self, item_id: int) -> None:
        item = self._items.get(item_id)
        if item is None:
            return
        cleaned = item.outputs[0] if item.state == ITEM_DONE and item.outputs and item.outputs[0].exists() else None
        if self.preview is None:
            self.preview = PreviewPane()
            self._left.addWidget(self.preview)
        self.preview.show_files(item.id, item.path, cleaned, item.has_video, item.path.name)
        self.preview.show()

    def _reveal(self, item_id: int) -> None:
        item = self._items.get(item_id)
        if item is not None and item.outputs:
            shell.reveal_in_folder(item.outputs[0])

    # -- state ------------------------------------------------------------------

    def _settings_changed(self) -> None:
        self.queue.set_parallel(self.prefs.parallel)
        self._refresh_state()

    def _refresh_row(self, item: FileItem) -> None:
        row = self.view.row(item.id)
        if row is not None:
            row.refresh()

    def _refresh_state(self) -> None:
        busy = self.queue.busy
        has_files = len(self.view) > 0
        self.drop_zone.set_compact(has_files)
        self.view.setVisible(has_files)
        waiting = len([i for i in self._items.values() if i.state == WAITING])
        finished = any(i.state in (ITEM_DONE, ITEM_FAILED, CANCELLED) for i in self._items.values())
        self.clear_button.setEnabled(finished)

        if busy:
            stopping = not self.queue.pending_ids() and all(
                self._items[i].stopping for i in self.queue.running_ids() if i in self._items
            )
            self.primary.setText("Stopping..." if stopping else "Cancel all")
            set_property(self.primary, "mode", "cancel")
            self.primary.setToolTip("Stop cleaning. Files that are finished are kept (Esc)")
            self.primary.setEnabled(not stopping)
        else:
            self.primary.setText("Clean " + _plural(waiting, "file", "files") if waiting else "Clean")
            set_property(self.primary, "mode", "")
            self.primary.setToolTip("Start cleaning the files in the list (Ctrl+Enter)")
            self.primary.setEnabled(waiting > 0 and not self._missing)
        self._update_overall()
        self._check_disk()

    def _update_overall(self) -> None:
        items = [i for i in self._items.values() if i.state in (WAITING, RUNNING, ITEM_DONE)]
        running = [i for i in items if i.state == RUNNING]
        total = sum(max(i.duration, 1.0) for i in items)
        progress = (
            sum(max(i.duration, 1.0) * (1.0 if i.state == ITEM_DONE else i.progress if i.state == RUNNING else 0.0) for i in items)
            / total if total else 0.0
        )
        busy = self.queue.busy
        self.overall.setVisible(busy)
        self.overall.setValue(int(progress * 1000))
        done = len([i for i in items if i.state == ITEM_DONE])
        if busy:
            text = f"{done} of {len(items)} done"
            if running:
                text += f"  ·  {len(running)} cleaning"
            elapsed = time.monotonic() - self._batch_started
            if progress > 0.05 and elapsed > 5:
                text += f"  ·  about {format_spent(elapsed * (1 - progress) / progress)} left"
            self.status.setText(text)
        elif self._batch_done or self._batch_failed:
            text = _plural(self._batch_done, "file", "files") + " cleaned"
            self.status.setText(text + (f", {self._batch_failed} failed" if self._batch_failed else ""))
        else:
            waiting = len([i for i in items if i.state == WAITING])
            self.status.setText(f"{_plural(waiting, 'file', 'files')} ready" if waiting else "Add some files to get started")

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
        at_once = self.queue.limit(subtitles, self.prefs.rebuild_voice, self.prefs.echo_track)
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

    def _fit_settings(self) -> None:
        """Settings take about a third of the window, and always as much as their content needs."""
        needed = self.settings_panel.minimumSizeHint().width() + self._settings_scroll.verticalScrollBar().sizeHint().width() + 4
        width = min(max(SETTINGS_MIN_WIDTH, int(self.width() * SETTINGS_SHARE)), SETTINGS_MAX_WIDTH)
        self._settings_scroll.setFixedWidth(max(width, needed))

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._fit_settings()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not getattr(self, "_grown", False):
            self._grown = True
            QTimer.singleShot(0, self._grow_to_fit_settings)

    def _grow_to_fit_settings(self) -> None:
        """Make the window just tall enough that the settings need no scroll bar (fonts and display scaling
        differ between computers, so this is measured, not fixed), as far as the screen allows."""
        missing = self.settings_panel.full_height() - self._settings_scroll.viewport().height()
        if missing <= 0 or self.isMaximized() or self.isFullScreen():
            return
        screen = self.screen().availableGeometry()
        frame = self.frameGeometry().height() - self.height()
        height = min(self.height() + missing + 2, screen.height() - frame)
        if height <= self.height():
            return
        self.resize(self.width(), height)
        top = self.frameGeometry().top()
        if top + height + frame > screen.bottom():
            self.move(self.x(), max(screen.top(), screen.bottom() - height - frame))

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            set_property(self.queue_card, "drop", True)

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        set_property(self.queue_card, "drop", False)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        set_property(self.queue_card, "drop", False)
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            event.acceptProposedAction()
            self.add_paths(paths)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.queue.busy and not self._confirm(
            "Quit Audio Cleaner?", "Files are still being cleaned. If you quit now, they will be discarded."
        ):
            event.ignore()
            return
        self._closing = True
        self._pool.clear()  # probes still waiting in line are not needed any more
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
        if self.preview is not None:
            self.preview.stop()
        self._qs.setValue("geometry", self.saveGeometry())
        self._qs.sync()
        self.history.close()
        super().closeEvent(event)
        if not stopped:
            # A file is in the middle of a step that does not stop quickly. Leave without waiting for it: what it
            # left in the scratch folder is swept away the next time the app starts.
            logging.shutdown()
            os._exit(0)

    def _sweep_leftovers(self) -> None:
        if self._closing:
            return
        folders = [*self.history.recent_folders()]
        out = self.prefs.output_dir()
        if out is not None:
            folders.append(out)
        self._pool.start(_SweepTask(folders))
