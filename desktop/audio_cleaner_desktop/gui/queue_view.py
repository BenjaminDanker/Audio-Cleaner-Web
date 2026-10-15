"""The list of files: the drop zone (empty state) and one row per file."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget,
)

from .. import engine
from . import icons
from .theme import set_property
from .widgets import ElidedLabel, IconButton, format_time

WAITING, RUNNING, DONE, FAILED, CANCELLED = "waiting", "running", "done", "failed", "cancelled"


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1000 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000
    return ""


@dataclass
class FileItem:
    id: int
    path: Path
    state: str = WAITING
    progress: float = 0.0
    duration: float = 0.0
    has_video: bool = False
    channels: int = 2
    message: str = ""
    outputs: list[Path] = field(default_factory=list)
    started_at: float = 0.0
    probed: bool = False
    options: engine.JobOptions | None = None


class DropZone(QFrame):
    """Big invitation when the list is empty, a slim strip once files are in."""

    addFiles = Signal()
    addFolder = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("DropZone")
        stack = QVBoxLayout(self)
        stack.setContentsMargins(0, 0, 0, 0)
        self._pages = QStackedWidget()
        stack.addWidget(self._pages)
        self._pages.addWidget(self._big_page())
        self._pages.addWidget(self._compact_page())

    def set_compact(self, compact: bool) -> None:
        self._pages.setCurrentIndex(1 if compact else 0)
        self.setMaximumHeight(60 if compact else 16777215)

    def _buttons(self) -> tuple[QPushButton, QPushButton]:
        files = QPushButton("Add files...")
        files.setToolTip("Choose video or audio files from your computer (Ctrl+O)")
        files.clicked.connect(self.addFiles)
        folder = QPushButton("Add folder...")
        folder.setToolTip("Add every video and audio file inside a folder, including its sub-folders")
        folder.clicked.connect(self.addFolder)
        return files, folder

    def _big_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 28, 24, 28)
        layout.setSpacing(10)
        layout.addStretch(1)
        self._art = QLabel()
        self._art.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._art)
        title = QLabel("Drop a video or recording here")
        title.setObjectName("Heading")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        text = QLabel(
            "Audio Cleaner removes background noise from the voice in your videos and recordings.\n"
            "Everything happens on your computer; nothing is uploaded."
        )
        text.setObjectName("Muted")
        text.setWordWrap(True)
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(text)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addStretch(1)
        for button in self._buttons():
            row.addWidget(button)
        row.addStretch(1)
        layout.addSpacing(6)
        layout.addLayout(row)
        layout.addStretch(1)
        self._refresh_art()
        return page

    def _compact_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(16, 10, 12, 10)
        layout.setSpacing(8)
        label = ElidedLabel("Drop more files or folders here")
        label.setObjectName("Muted")
        layout.addWidget(label, 1)
        for button in self._buttons():
            layout.addWidget(button)
        return page

    def _refresh_art(self) -> None:
        pm = icons.pixmap("upload", self.palette().placeholderText().color(), 44, self.devicePixelRatioF())
        self._art.setPixmap(pm)

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange and hasattr(self, "_art"):
            self._refresh_art()


class FileRow(QFrame):
    removeRequested = Signal(int)
    cancelRequested = Signal(int)
    previewRequested = Signal(int)
    revealRequested = Signal(int)
    moveRequested = Signal(int, int)

    def __init__(self, item: FileItem, parent: QWidget | None = None):
        super().__init__(parent)
        self.item = item
        self.setObjectName("Row")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 10, 10)
        layout.setSpacing(12)

        self._badge = QLabel()
        self._badge.setObjectName("Badge")
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._badge.setFixedSize(48, 22)
        layout.addWidget(self._badge, 0, Qt.AlignmentFlag.AlignVCenter)

        middle = QVBoxLayout()
        middle.setSpacing(3)
        self._name = ElidedLabel(item.path.name, Qt.TextElideMode.ElideMiddle)
        self._name.setToolTip(str(item.path))
        middle.addWidget(self._name)
        self._second = QStackedWidget()
        self._second.setFixedHeight(18)
        self._meta = ElidedLabel()
        self._meta.setObjectName("Small")
        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setTextVisible(False)
        bar_page = QWidget()
        bar_layout = QVBoxLayout(bar_page)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.addWidget(self._bar, 0, Qt.AlignmentFlag.AlignVCenter)
        self._second.addWidget(self._meta)
        self._second.addWidget(bar_page)
        middle.addWidget(self._second)
        layout.addLayout(middle, 1)

        self._status = ElidedLabel()
        self._status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._status.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        bold = self.font()
        bold.setBold(True)
        self._status.setFixedWidth(QFontMetrics(bold).horizontalAdvance("Cleaning 100%") + 8)
        layout.addWidget(self._status)

        self._up = IconButton("up", "Move this file earlier in the queue")
        self._down = IconButton("down", "Move this file later in the queue")
        self._preview = IconButton("play", "Listen to the original and the cleaned version")
        self._reveal = IconButton("folder", "Show the cleaned file in its folder")
        self._stop = IconButton("close", "Stop cleaning this file")
        self._remove = IconButton("trash", "Remove this file from the list")
        self._buttons = (self._up, self._down, self._preview, self._reveal, self._stop, self._remove)
        actions = QWidget()
        actions_layout = QHBoxLayout(actions)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(4)
        for b in self._buttons:
            b.setFixedSize(32, 32)
            actions_layout.addWidget(b)
        layout.addWidget(actions)
        self._up.clicked.connect(lambda: self.moveRequested.emit(item.id, -1))
        self._down.clicked.connect(lambda: self.moveRequested.emit(item.id, 1))
        self._preview.clicked.connect(lambda: self.previewRequested.emit(item.id))
        self._reveal.clicked.connect(lambda: self.revealRequested.emit(item.id))
        self._stop.clicked.connect(lambda: self.cancelRequested.emit(item.id))
        self._remove.clicked.connect(lambda: self.removeRequested.emit(item.id))
        self.refresh()

    def refresh(self) -> None:
        item = self.item
        state = item.state
        self._badge.setText("VIDEO" if item.has_video else "AUDIO")
        set_property(self._badge, "kind", "video" if item.has_video else "audio")

        text, style = {
            WAITING: ("Waiting", ""),
            RUNNING: (f"Cleaning {int(item.progress * 100)}%", "cleaning"),
            DONE: ("Done", "done"),
            FAILED: ("Failed", "failed"),
            CANCELLED: ("Cancelled", "warn"),
        }[state]
        self._status.setText(text)
        set_property(self._status, "state", style)

        if state == RUNNING:
            self._bar.setValue(int(item.progress * 1000))
            self._second.setCurrentIndex(1)
        else:
            self._second.setCurrentIndex(0)
            self._meta.setText(self._meta_text())
            set_property(self._meta, "state", "failed" if state == FAILED else "")

        self._up.setVisible(state == WAITING)
        self._down.setVisible(state == WAITING)
        self._stop.setVisible(state == RUNNING)
        self._preview.setVisible(state in (WAITING, DONE))
        self._reveal.setVisible(state == DONE)
        self._remove.setVisible(state != RUNNING)
        self.setAccessibleName(f"{item.path.name}: {text}")

    def _meta_text(self) -> str:
        item = self.item
        if item.state == FAILED:
            return item.message
        if item.state == DONE and item.outputs:
            return "Saved as " + " and ".join(p.name for p in item.outputs) + (f" ({item.message})" if item.message else "")
        if not item.probed:
            return "Reading file..."
        parts = [format_time(item.duration)] if item.duration else []
        try:
            parts.append(format_size(item.path.stat().st_size))
        except OSError:
            pass
        return "  ·  ".join(parts)


class QueueView(QScrollArea):
    """A scrolling column of FileRows."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        holder = QWidget()
        holder.setObjectName("RowHolder")
        self._layout = QVBoxLayout(holder)
        self._layout.setContentsMargins(8, 4, 8, 8)
        self._layout.setSpacing(2)
        self._layout.addStretch(1)
        self.setWidget(holder)
        self._rows: dict[int, FileRow] = {}

    def __len__(self) -> int:
        return len(self._rows)

    def add(self, row: FileRow) -> None:
        self._layout.insertWidget(self._layout.count() - 1, row)
        self._rows[row.item.id] = row

    def row(self, item_id: int) -> FileRow | None:
        return self._rows.get(item_id)

    def rows(self) -> list[FileRow]:
        """In display order."""
        return [self._layout.itemAt(i).widget() for i in range(self._layout.count() - 1)]

    def remove(self, item_id: int) -> None:
        row = self._rows.pop(item_id, None)
        if row is not None:
            self._layout.removeWidget(row)
            row.deleteLater()

    def move_among_waiting(self, item_id: int, delta: int) -> bool:
        """Swap a waiting row with its nearest waiting neighbour. Returns whether it moved."""
        rows = self.rows()
        waiting = [r for r in rows if r.item.state == WAITING]
        row = self._rows[item_id]
        j = waiting.index(row) + delta
        if not 0 <= j < len(waiting):
            return False
        self._layout.removeWidget(row)
        self._layout.insertWidget(self._layout.indexOf(waiting[j]) + (1 if delta > 0 else 0), row)
        return True
