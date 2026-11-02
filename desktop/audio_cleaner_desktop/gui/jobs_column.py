"""The jobs, in a column beside the editor (or under it when the window is narrow): the files waiting and running, with their progress, and every
finished or failed job, newest first. Opening one shows it in the editor, in the same window."""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QLineEdit, QMenu, QMessageBox, QProgressBar, QPushButton, QSizePolicy, QToolButton, QVBoxLayout, QWidget,
)

from ..labels import describe_settings
from . import shell
from .history import DONE, Entry, HistoryStore
from .items import CANCELLED, RUNNING, WAITING, FileItem, format_size
from .studio_widgets import small_label
from .theme import set_property
from .widgets import ElidedLabel, IconButton, format_time

PAGE_SIZE = 25
SORT_LABELS = {"newest": "Newest first", "oldest": "Oldest first", "name": "Name (A to Z)", "disk": "Most disk used", "length": "Longest first"}
WIDTH = 300


def format_spent(seconds: float) -> str:
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} min" if minutes else ("0 min" if seconds == 0 else "< 1 min")
    return f"{minutes // 60} h {minutes % 60:02d} min"


def format_hours(seconds: float) -> str:
    if seconds < 3600:
        return f"{round(seconds / 60)} min" if seconds >= 60 else ("0 min" if seconds == 0 else "< 1 min")
    return f"{seconds / 3600:.1f} h"


class JobCard(QFrame):
    """One job: the kind (video or audio), its name, how it stands, and buttons for what can be done with it. A click opens it in the editor."""

    clicked = Signal()

    def __init__(self, name: str, video: bool, status: str, state: str, meta: str, tip: str = ""):
        super().__init__()
        self.setObjectName("JobCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tip)
        self.setAccessibleName(f"{name}: {status}")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 8, 8)
        outer.setSpacing(4)
        top = QHBoxLayout()
        top.setSpacing(8)
        self.badge = QLabel("VIDEO" if video else "AUDIO")
        self.badge.setObjectName("Badge")
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge.setFixedSize(46, 20)
        set_property(self.badge, "kind", "video" if video else "audio")
        self.name = ElidedLabel(name, Qt.TextElideMode.ElideMiddle)
        self.name.setObjectName("JobName")
        self.status = QLabel(status)
        set_property(self.status, "state", state)
        top.addWidget(self.badge)
        top.addWidget(self.name, 1)
        top.addWidget(self.status)
        outer.addLayout(top)
        self.second = QHBoxLayout()
        self.second.setSpacing(4)
        self.meta = ElidedLabel(meta)
        self.meta.setObjectName("Small")
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.hide()
        self.second.addWidget(self.meta, 1)
        self.second.addWidget(self.bar, 1)
        outer.addLayout(self.second)
        self.buttons = QHBoxLayout()
        self.buttons.setSpacing(2)
        self.buttons.addStretch(1)
        self.second.addLayout(self.buttons)

    def add_button(self, icon: str, tip: str, handler: Callable[[], None]) -> None:
        b = IconButton(icon, tip, size=16)
        b.setFixedSize(26, 26)
        b.clicked.connect(lambda: handler())
        self.buttons.addWidget(b)

    def set_progress(self, fraction: float | None) -> None:
        self.bar.setVisible(fraction is not None)
        self.meta.setVisible(fraction is None)
        if fraction is not None:
            self.bar.setValue(int(fraction * 1000))

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class JobsColumn(QFrame):
    openItem = Signal(int)
    openEntry = Signal(int)
    removeItem = Signal(int)
    stopItem = Signal(int)
    moveItem = Signal(int, int)  # a file that waits: earlier (-1) or later (1) in line
    deleteEntry = Signal(int)
    deleteAll = Signal()
    startAll = Signal()
    stopAll = Signal()

    def __init__(self, store: HistoryStore, items: Callable[[], list[FileItem]], queued: Callable[[], set[int]], parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("JobsCol")
        self.store = store
        self._items, self._queued = items, queued
        self.current: tuple[str, int] | None = None  # ("item", id) or ("entry", id): the one in the editor
        self.sort = "newest"
        self._shown = PAGE_SIZE
        self._cards: dict[tuple[str, int], JobCard] = {}
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)
        head = QHBoxLayout()
        title = QLabel("Your jobs")
        title.setObjectName("ColHead")
        self.menu_button = QToolButton()
        self.menu_button.setText("⋯")
        self.menu_button.setObjectName("MenuDots")
        self.menu_button.setToolTip("Start or stop everything, sort the list, delete jobs")
        self.menu_button.setAccessibleName("More about the jobs")
        self.menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.menu = QMenu(self.menu_button)
        self.act_start_all = self.menu.addAction("Start all the files that are waiting")
        self.act_stop_all = self.menu.addAction("Stop everything that is running")
        self.menu.addSeparator()
        sort_menu = self.menu.addMenu("Sort by")
        group = QActionGroup(self)
        for key, label in SORT_LABELS.items():
            act = QAction(label, self, checkable=True)
            act.setChecked(key == self.sort)
            act.triggered.connect(lambda _=False, k=key: self._set_sort(k))
            group.addAction(act)
            sort_menu.addAction(act)
        self.menu.addSeparator()
        self.act_delete_all = self.menu.addAction("Delete all jobs…")
        self.menu_button.setMenu(self.menu)
        self.act_start_all.triggered.connect(self.startAll)
        self.act_stop_all.triggered.connect(self.stopAll)
        self.act_delete_all.triggered.connect(self.deleteAll)
        head.addWidget(title, 1)
        head.addWidget(self.menu_button)
        outer.addLayout(head)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search by name, folder or settings")
        self.search.setClearButtonEnabled(True)
        self.search.setToolTip("Type part of a file name, a folder, or what it was made with (for example Gentle or leveled)")
        outer.addWidget(self.search)
        self.note = small_label("Every file stays here, with its saved copies, until you delete it. Open one to listen, change it and save again, or delete it.")
        outer.addWidget(self.note)
        self.list = QVBoxLayout()
        self.list.setSpacing(6)
        outer.addLayout(self.list)
        self.empty = small_label("")
        self.empty.setObjectName("Plain")
        self.empty.hide()
        outer.addWidget(self.empty)
        self.more = QPushButton("Show more")
        self.more.setToolTip("Show older jobs")
        self.more.clicked.connect(self._show_more)
        outer.addWidget(self.more, 0, Qt.AlignmentFlag.AlignHCenter)
        self.totals = small_label("")
        outer.addWidget(self.totals)
        outer.addStretch(1)
        self._debounce = QTimer(self, singleShot=True, interval=150, timeout=self._search_changed)
        self.search.textChanged.connect(lambda _: self._debounce.start())

    # ------------------------------------------------------------------------------------------------------ the list
    def _set_sort(self, key: str) -> None:
        self.sort = key
        self._shown = PAGE_SIZE
        self.refresh()

    def _search_changed(self) -> None:
        self._shown = PAGE_SIZE
        self.refresh()

    def _show_more(self) -> None:
        self._shown += PAGE_SIZE
        self.refresh()

    def set_current(self, key: tuple[str, int] | None) -> None:
        self.current = key
        for k, card in self._cards.items():
            set_property(card, "current", k == key)

    def refresh(self) -> None:
        """Draw the list again: what is waiting or running, then the finished jobs (a page at a time)."""
        while self.list.count():
            item = self.list.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()
                item.widget().deleteLater()
        self._cards.clear()
        text = self.search.text()
        items = [i for i in self._items() if i.state in (WAITING, RUNNING, CANCELLED) and self._matches(i, text)]
        for it in items:
            self._add(("item", it.id), self._item_card(it))
        total = self.store.count(text)
        entries = self.store.search(text, limit=self._shown, offset=0, sort=self.sort)
        for e in entries:
            self._add(("entry", e.id), self._entry_card(e))
        self.more.setVisible(total > len(entries))
        shown = len(items) + len(entries)
        self.empty.setVisible(shown == 0)
        self.empty.setText("Nothing matches your search." if text.strip() else "Nothing here yet. Your files show up here once you choose them.")
        self.note.setVisible(shown > 0 or not text.strip())
        t = self.store.totals()
        c = self.store.copies_total()
        self.totals.setText(f"{t.files} file{'s' if t.files != 1 else ''} cleaned · {format_hours(t.audio_seconds + t.video_seconds)} of sound · "
                            f"saved copies {format_size(c.bytes)}" if t.files or c.jobs else "")
        self.totals.setToolTip("Files cleaned, the length of what they held, and the disk the saved copies take (they let a result be changed and saved again)")
        waiting = sum(1 for i in self._items() if i.state == WAITING)
        running = any(i.state == RUNNING for i in self._items())
        self.act_start_all.setEnabled(waiting > 0)
        self.act_start_all.setText(f"Start all {waiting} files that are waiting" if waiting > 1 else "Start all the files that are waiting")
        self.act_stop_all.setEnabled(running or any(i.id in self._queued() for i in self._items()))
        self.act_delete_all.setEnabled(self.store.count() > 0)
        self.set_current(self.current)

    @staticmethod
    def _matches(item: FileItem, text: str) -> bool:
        hay = f"{item.path.name} {item.path.parent}".lower()
        return all(w.lower() in hay for w in text.split())

    def _add(self, key: tuple[str, int], card: JobCard) -> None:
        self._cards[key] = card
        self.list.addWidget(card)
        card.clicked.connect(lambda k=key: (self.openItem if k[0] == "item" else self.openEntry).emit(k[1]))

    def _item_card(self, it: FileItem) -> JobCard:
        queued = it.id in self._queued()
        status, state = {
            WAITING: ("In line" if queued else "Not started", "warn" if queued else ""),
            RUNNING: ((("Restarting..." if it.restart else "Stopping..."), "warn") if it.stopping else (f"Cleaning {int(it.progress * 100)}%", "cleaning")),
            CANCELLED: ("Cancelled", "warn"),
        }[it.state]
        bits = [format_time(it.duration)] if it.duration else (["Reading file..."] if not it.probed else [])
        try:
            bits.append(format_size(it.path.stat().st_size))
        except OSError:
            pass
        card = JobCard(it.path.name, it.has_video, status, state, "  ·  ".join(bits), str(it.path))
        if it.state == RUNNING:
            card.set_progress(it.progress)
            card.add_button("stop", "Stop cleaning this file", lambda i=it.id: self.stopItem.emit(i)) if not it.stopping else None
        else:
            if it.state == WAITING and sum(1 for x in self._items() if x.state == WAITING) > 1:
                card.add_button("up", "Move this file earlier in the queue", lambda i=it.id: self.moveItem.emit(i, -1))
                card.add_button("down", "Move this file later in the queue", lambda i=it.id: self.moveItem.emit(i, 1))
            card.add_button("close", "Take this file off the list (the file itself stays)", lambda i=it.id: self.removeItem.emit(i))
        return card

    def _entry_card(self, e: Entry) -> JobCard:
        when = datetime.fromtimestamp(e.finished_at).strftime("%d %b, %H:%M")
        if e.status == DONE:
            meta = f"{when}  ·  {format_time(e.duration)}  ·  {describe_settings(e.settings, e.has_video)}"
        else:
            meta = f"{when}  ·  {e.message}"
        card = JobCard(e.name, e.has_video, "Done" if e.status == DONE else "Failed", "done" if e.status == DONE else "failed", meta, "\n".join([e.source, *e.outputs]) + "\n\n" + meta)
        outputs = [Path(p) for p in e.outputs]
        main = next((p for p in outputs if p.suffix.lower() not in (".srt", ".vtt", ".txt")), None)
        if e.status == DONE and main is not None:
            card.add_button("play", "Open the cleaned file with your default player", lambda p=main: self._open(p))
            card.add_button("folder", "Show the cleaned file in its folder", lambda p=main: shell.reveal_in_folder(p))
        card.add_button("trash", "Delete this job: its saved copies and its place in this list. Your original and the cleaned files are not touched.",
                        lambda i=e.id: self.deleteEntry.emit(i))
        return card

    def _open(self, path: Path) -> None:
        if not path.exists() or not shell.open_path(path):
            QMessageBox.information(self, "Open", "This file has been moved or deleted.")

    def update_progress(self, item: FileItem) -> None:
        """A running file moved on: only its card is touched."""
        card = self._cards.get(("item", item.id))
        if card is None or item.state != RUNNING:
            return
        card.status.setText(f"Cleaning {int(item.progress * 100)}%")
        card.set_progress(item.progress)
