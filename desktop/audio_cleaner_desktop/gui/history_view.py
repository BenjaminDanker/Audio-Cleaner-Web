"""The Jobs tab: totals, search, sorting and one row per cleaned (or failed) file, newest first, a page at a time. A row opens the job
(job_view.py): listen, change the strength and save again from the copies the app kept, see what they take, delete the job."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea, QStackedWidget, QVBoxLayout, QWidget,
)

from ..labels import describe_settings, settings_dict  # noqa: F401 - they live in labels.py (no Qt); importable from here as before
from . import shell
from .history import DONE, SORTS, CopiesTotal, Entry, HistoryStore, Totals
from .job_view import JobEnv, JobView
from .queue_view import format_size
from .theme import set_property
from .widgets import ElidedLabel, IconButton, format_time


def format_hours(seconds: float) -> str:
    if seconds < 3600:
        return f"{round(seconds / 60)} min" if seconds >= 60 else ("0 min" if seconds == 0 else "< 1 min")
    return f"{seconds / 3600:.1f} h"


def format_spent(seconds: float) -> str:
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} min" if minutes else ("0 min" if seconds == 0 else "< 1 min")
    return f"{minutes // 60} h {minutes % 60:02d} min"


PAGE_SIZE = 50  # rows drawn at first, and added by each "Show more"


class _Stat(QFrame):
    def __init__(self, caption: str, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.value = QLabel("0")
        self.value.setObjectName("Heading")
        self.caption = QLabel(caption)
        self.caption.setObjectName("Small")
        layout.addWidget(self.value)
        layout.addWidget(self.caption)


SORT_LABELS = {
    "newest": "Newest first", "oldest": "Oldest first", "name": "Name (A to Z)", "disk": "Most disk used", "length": "Longest first",
}


class _HistoryRow(QFrame):
    """One job. A click opens it; the buttons open the cleaned file, show it in its folder or delete the job."""

    def __init__(self, entry: Entry, page: "HistoryPage"):
        super().__init__()
        self.entry = entry
        self._page = page
        self.setObjectName("Row")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        outputs = [Path(p) for p in entry.outputs]
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 10, 10)
        layout.setSpacing(12)

        badge = QLabel("VIDEO" if entry.has_video else "AUDIO")
        badge.setObjectName("Badge")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedSize(48, 22)
        set_property(badge, "kind", "video" if entry.has_video else "audio")
        layout.addWidget(badge, 0, Qt.AlignmentFlag.AlignVCenter)

        middle = QVBoxLayout()
        middle.setSpacing(3)
        name = ElidedLabel(entry.name)
        name.setToolTip("\n".join([entry.source, *entry.outputs]))
        when = datetime.fromtimestamp(entry.finished_at).strftime("%d %b %Y, %H:%M")
        if entry.status == DONE:
            meta = f"{when}  \u00b7  {format_time(entry.duration)}  \u00b7  took {format_spent(entry.seconds_taken)}  \u00b7  " \
                   + describe_settings(entry.settings, entry.has_video)
        else:
            meta = f"{when}  \u00b7  {entry.message}"
        detail = ElidedLabel(meta)
        detail.setObjectName("Small")
        detail.setToolTip(meta)
        middle.addWidget(name)
        middle.addWidget(detail)
        layout.addLayout(middle, 1)

        self.disk = QLabel(format_size(entry.copies_bytes) if entry.job_dir else "")
        self.disk.setObjectName("Small")
        self.disk.setToolTip("Disk used by the saved copies of this job, which let it be changed and saved again" if entry.job_dir
                             else "")
        layout.addWidget(self.disk)

        status = QLabel("Done" if entry.status == DONE else "Failed")
        set_property(status, "state", "done" if entry.status == DONE else "failed")
        layout.addWidget(status)

        open_job = IconButton("waveform", "Open this job: listen, change the strength and save again, see its saved copies")
        open_job.clicked.connect(lambda: page.open_job(entry.id))
        buttons = [open_job]
        if entry.status == DONE and outputs:
            open_btn = IconButton("play", "Open the cleaned file with your default player")
            open_btn.clicked.connect(lambda: self._open(outputs[0]))
            reveal = IconButton("folder", "Show the cleaned file in its folder")
            reveal.clicked.connect(lambda: shell.reveal_in_folder(outputs[0]))
            buttons += [open_btn, reveal]
        remove = IconButton("trash", "Delete this job: its saved copies and its place in this list. Your original and the cleaned "
                                     "files are not touched.")
        remove.clicked.connect(lambda: page.delete_entry(entry.id))
        buttons.append(remove)
        for b in buttons:
            b.setFixedSize(32, 32)
            layout.addWidget(b)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self._page.open_job(self.entry.id)
        super().mouseReleaseEvent(event)

    def _open(self, path: Path) -> None:
        if not path.exists() or not shell.open_path(path):
            QMessageBox.information(self, "Open", "This file has been moved or deleted.")


class HistoryPage(QWidget):
    """The Jobs tab: the list, and (one click on a row) the job view in its place."""

    def __init__(self, store: HistoryStore, parent: QWidget | None = None, *, env: JobEnv | None = None):
        super().__init__(parent)
        self._store = store
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._stack = QStackedWidget()
        outer.addWidget(self._stack)
        page = QWidget()
        self._stack.addWidget(page)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 16, 0, 0)
        layout.setSpacing(14)

        totals = QFrame()
        totals.setObjectName("Card")
        totals_row = QHBoxLayout(totals)
        totals_row.setContentsMargins(24, 16, 24, 16)
        self._stats = {k: _Stat(c) for k, c in (
            ("files", "Files cleaned"), ("audio", "Audio cleaned"), ("video", "Video cleaned"), ("spent", "Time spent cleaning"),
            ("disk", "Saved copies"),
        )}
        for stat in self._stats.values():
            totals_row.addWidget(stat, 1)
        self._stats["disk"].setToolTip("The copies kept so a result can be changed and saved again without cleaning it again. They stay "
                                       "until you delete the job.")
        layout.addWidget(totals)

        bar = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search by name, folder or settings")
        self._search.setClearButtonEnabled(True)
        self._search.setToolTip("Type part of a file name, a folder, or what it was made with (for example Gentle or leveled)")
        self._sort = QComboBox()
        for key, label in SORT_LABELS.items():
            self._sort.addItem(label, key)
        self._sort.setToolTip("The order of the list")
        self._clear = QPushButton("Delete all")
        self._clear.setToolTip("Delete every job below with its saved copies. Your original and cleaned files are not touched.")
        self._clear.clicked.connect(self._clear_all)
        bar.addWidget(self._search, 1)
        bar.addWidget(self._sort)
        bar.addWidget(self._clear)
        layout.addLayout(bar)

        card = QFrame()
        card.setObjectName("Card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        self._empty = QLabel()
        self._empty.setObjectName("Muted")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        card_layout.addWidget(self._empty, 1)
        card_layout.addWidget(self._scroll, 1)
        layout.addWidget(card, 1)
        self._count = QLabel()
        self._count.setObjectName("Small")
        layout.addWidget(self._count)

        self.view = JobView(store, env)
        self.view.closed.connect(self.close_job)
        self.view.deleteRequested.connect(self.delete_entry)
        self.view.changed.connect(self.refresh)
        self._stack.addWidget(self.view)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(150)
        self._debounce.timeout.connect(self._search_changed)
        self._search.textChanged.connect(lambda _: self._debounce.start())
        self._sort.currentIndexChanged.connect(lambda _: self._search_changed())
        self._shown = PAGE_SIZE
        self._rows_layout: QVBoxLayout | None = None
        self._more: QPushButton | None = None
        self._loaded = 0
        self.refresh()

    # -- the list -----------------------------------------------------------------------------------------------

    @property
    def sort(self) -> str:
        return self._sort.currentData() or "newest"

    def _search_changed(self) -> None:
        self._shown = PAGE_SIZE
        self.refresh()

    def _show_more(self) -> None:
        """The next page, added under the rows that are there (they are not drawn again)."""
        if self._rows_layout is None:
            return
        text = self._search.text()
        entries = self._store.search(text, limit=PAGE_SIZE, offset=self._loaded, sort=self.sort)
        for entry in entries:
            self._rows_layout.insertWidget(self._rows_layout.count() - 2, _HistoryRow(entry, self))
        self._loaded += len(entries)
        self._shown = self._loaded
        self._update_more(self._store.count(text))

    def _update_more(self, total: int) -> None:
        if self._more is not None:
            self._more.setVisible(total > self._loaded)
        self._count.setText(f"Showing {self._loaded} of {total}" if total > self._loaded else
                            (f"{total} job{'s' if total != 1 else ''}" if total else ""))

    def refresh(self) -> None:
        """Rebuild the list: the newest PAGE_SIZE jobs (more on request, a page at a time), not all of them at once."""
        totals = self._store.totals()
        self._show_totals(totals, self._store.copies_total())
        text = self._search.text()
        total = self._store.count(text)
        entries = self._store.search(text, limit=self._shown, offset=0, sort=self.sort)
        holder = QWidget()
        holder.setObjectName("RowHolder")
        rows = QVBoxLayout(holder)
        rows.setContentsMargins(8, 8, 8, 8)
        rows.setSpacing(2)
        for entry in entries:
            rows.addWidget(_HistoryRow(entry, self))
        self._more = QPushButton("Show more")
        self._more.setToolTip("Show older jobs")
        self._more.clicked.connect(self._show_more)
        rows.addWidget(self._more, 0, Qt.AlignmentFlag.AlignCenter)
        rows.addStretch(1)
        self._rows_layout = rows
        self._loaded = len(entries)
        self._update_more(total)
        old = self._scroll.takeWidget()
        if old is not None:
            old.deleteLater()
        self._scroll.setWidget(holder)
        searching = bool(text.strip())
        self._scroll.setVisible(bool(entries))
        self._empty.setVisible(not entries)
        self._empty.setText(
            "Nothing matches your search." if searching else "Files you clean will be listed here.\nNothing leaves your computer."
        )
        self._clear.setEnabled(totals.files > 0 or bool(entries) or self._store.count() > 0)

    def _show_totals(self, t: Totals, c: CopiesTotal) -> None:
        self._stats["files"].value.setText(str(t.files))
        self._stats["audio"].value.setText(format_hours(t.audio_seconds))
        self._stats["video"].value.setText(format_hours(t.video_seconds))
        self._stats["spent"].value.setText(format_spent(t.seconds_spent))
        self._stats["disk"].value.setText(format_size(c.bytes))
        self._stats["disk"].caption.setText(f"Saved copies ({c.jobs} job{'s' if c.jobs != 1 else ''})")

    # -- one job ------------------------------------------------------------------------------------------------

    def open_job(self, entry_id: int) -> None:
        if self.view.show_entry(entry_id):
            self._stack.setCurrentWidget(self.view)

    def close_job(self) -> None:
        """Back to the list."""
        self.view.unload()
        self._stack.setCurrentIndex(0)
        self.refresh()

    @property
    def viewing(self) -> int | None:
        """The id of the job that is open, if any."""
        return self.view.entry.id if self._stack.currentWidget() is self.view and self.view.entry is not None else None

    def shutdown(self, timeout_ms: int | None = None) -> bool:
        """The window is closing: stop what the job view is doing and delete its scratch files. False if a save did not stop in time."""
        return self.view.shutdown(timeout_ms)

    # -- deleting -------------------------------------------------------------------------------------------------

    def confirm(self, title: str, text: str) -> bool:
        box = QMessageBox(QMessageBox.Icon.Question, title, text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, self)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        return box.exec() == QMessageBox.StandardButton.Yes

    def _stayed(self, ids: list[int]) -> None:
        if ids:
            QMessageBox.warning(self, "Delete", "Some saved copies could not be removed (a file may be in use). The job stays in the "
                                                "list; try again later.")

    def delete_entry(self, entry_id: int) -> None:
        """Delete one job after asking: its saved copies and its place in the list; never the original or the cleaned files."""
        entry = self._store.get(entry_id)
        if entry is None:
            return
        if self.view.busy and self.view.entry is not None and self.view.entry.id == entry_id:
            return  # a change is being saved from its copies
        if entry.job_dir:
            text = (f"Delete \"{entry.name}\"?\n\nIts saved copies ({format_size(entry.copies_bytes)}) and its place in this list are "
                    "deleted, so it can't be changed and saved again. Your original and the cleaned files are not touched.")
        else:
            text = f"Remove \"{entry.name}\" from the list?\n\nYour original and the cleaned files are not touched."
        if not self.confirm("Delete job", text):
            return
        if self.viewing == entry_id:
            self.view.unload()
        stayed = self._store.delete([entry_id])
        if self.viewing == entry_id:
            self._stack.setCurrentIndex(0)
        self.refresh()
        self._stayed(stayed)

    def _clear_all(self) -> None:
        c = self._store.copies_total()
        n = self._store.count()
        what = f" and the {format_size(c.bytes)} of saved copies" if c.jobs else ""
        if not self.confirm("Delete all jobs", f"Delete all {n} job{'s' if n != 1 else ''}{what}?\n\nYour originals and cleaned files are "
                                                "not touched, but the jobs can't be changed and saved again."):
            return
        if self.view.busy:
            return
        self.view.unload()
        self._stack.setCurrentIndex(0)
        stayed = self._store.clear()
        self.refresh()
        self._stayed(stayed)
