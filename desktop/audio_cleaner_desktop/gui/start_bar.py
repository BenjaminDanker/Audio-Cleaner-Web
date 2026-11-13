"""The start row under the editor and the settings (the web's Bar.jsx): the progress while a run goes, then Details (what will be done and about
how long) or Your files beside the one button, which says about how long a run takes (where the website says what it costs). It is in the page, not pinned to the window; a computer's own work has no balance, so nothing is billed here."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget

from . import shell
from .items import format_size
from .studio_widgets import Pane, link_button, small_label
from .widgets import ElidedLabel, IconButton


class Popover(QFrame):
    """A small window that opens above its button and closes at the next click anywhere else."""

    def __init__(self, parent: QWidget):
        super().__init__(parent, Qt.WindowType.Popup)
        self.setObjectName("Popover")
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(16, 14, 16, 12)
        self.lay.setSpacing(6)
        self.setMinimumWidth(380)
        self.setMaximumWidth(460)

    def clear(self) -> None:
        while self.lay.count():
            item = self.lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.deleteLater()
            elif item.layout() is not None:
                _drop(item.layout())

    def row(self, left: str, right: str = "", *, bold: bool = False) -> None:
        line = QHBoxLayout()
        a, b = QLabel(left), QLabel(right)
        a.setWordWrap(True)
        if bold:
            for w in (a, b):
                f = w.font()
                f.setBold(True)
                w.setFont(f)
        line.addWidget(a, 1)
        line.addWidget(b, 0, Qt.AlignmentFlag.AlignTop)
        self.lay.addLayout(line)

    def note(self, text: str) -> None:
        self.lay.addWidget(small_label(text))

    def show_above(self, anchor: QWidget) -> None:
        """Above the link, its right side at the link's right side (the link sits at the right of the window, beside the button)."""
        self.adjustSize()
        top_left = anchor.mapToGlobal(QPoint(0, 0))
        x = max(anchor.window().mapToGlobal(QPoint(0, 0)).x(), top_left.x() + anchor.width() - self.width())  # (its width: at least the minimum)
        self.move(x, top_left.y() - self.sizeHint().height() - 10)
        self.show()


def _drop(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget() is not None:
            item.widget().hide()
            item.widget().deleteLater()
        elif item.layout() is not None:
            _drop(item.layout())


SHARE_LONG, SHARE_SHORT = "Use these settings for other files…", "Settings for others…"  # (the short one in a narrow window)
NARROW = 820


class StartBar(Pane):
    startClicked = Signal()
    startAllClicked = Signal()
    shareClicked = Signal()
    cancelClicked = Signal()
    deleteRequested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent, name="Pane")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(14)
        self.details = link_button("Details", "What will be done, and about how long it takes")
        self.files = link_button("Your files", "The files this made, with their sizes")
        self.details.clicked.connect(lambda: self._open("details"))
        self.files.clicked.connect(lambda: self._open("files"))
        self.progress_box = QWidget()
        pl = QHBoxLayout(self.progress_box)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(12)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setMinimumWidth(120)
        self.bar.setMaximumWidth(420)
        self.bar.setAccessibleName("Progress")
        self.line = QLabel()
        self.line.setObjectName("Small")
        self.cancel = link_button("Cancel job")
        self.cancel.clicked.connect(self.cancelClicked)
        pl.addWidget(self.bar, 1)
        pl.addWidget(self.line)
        pl.addWidget(self.cancel)
        self.button = QPushButton("Start")
        self.button.setObjectName("Primary")
        self.button.setMinimumWidth(200)
        self.button.setMinimumHeight(40)
        self.button.clicked.connect(self.startClicked)
        # Several files: this file's settings for the others, and all of them in line at once (they run one after the other).
        self.share = link_button(SHARE_LONG, "Give the files that are waiting the settings of this file")
        self.share.clicked.connect(self.shareClicked)
        self.share.hide()
        self.all_button = QPushButton("Start all")
        self.all_button.setMinimumHeight(40)
        self.all_button.clicked.connect(self.startAllClicked)
        self.all_button.hide()
        lay.addWidget(self.progress_box, 1)
        lay.addStretch(1)
        lay.addWidget(self.share)
        lay.addWidget(self.details)  # (beside the button, like the website's balance)
        lay.addWidget(self.files)
        lay.addWidget(self.all_button)
        lay.addWidget(self.button)
        self.popover = Popover(self)
        self._fill: dict[str, Callable[[Popover], None]] = {}
        self.set_mode("details")
        self.button.setEnabled(False)

    def set_fillers(self, details: Callable[[Popover], None], files: Callable[[Popover], None]) -> None:
        self._fill = {"details": details, "files": files}

    def set_mode(self, mode: str, *, text: str = "", fraction: float | None = None, cancel: str = "") -> None:
        """"details", "files", "progress" (with its line, how far, and the words of the cancel link), or "empty" (no file: only the button)."""
        self.mode = mode
        self.details.setVisible(mode == "details")
        self.files.setVisible(mode == "files")
        self.progress_box.setVisible(mode == "progress")
        if mode == "progress":
            self.line.setText(text)
            self.bar.setRange(0, 0 if fraction is None else 1000)
            if fraction is not None:
                self.bar.setValue(int(fraction * 1000))
            self.cancel.setText(cancel)
            self.cancel.setVisible(bool(cancel))

    def set_many(self, share: bool, waiting: int) -> None:
        """`share`: offer this file's settings to the files that wait; `waiting`: how many files are not started yet (Start all shows from 2)."""
        self.share.setVisible(share)
        self.all_button.setVisible(waiting >= 2)
        self.all_button.setText(f"Start all {waiting}")
        self.all_button.setToolTip(f"Put all {waiting} files that are waiting in line, each with its own settings. They are cleaned one after "
                                   "the other, so you can leave them to it.")

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.share.setText(SHARE_SHORT if self.width() < NARROW else SHARE_LONG)

    def set_button(self, label: str, enabled: bool, tip: str = "", state: str = "") -> None:
        self.button.setText(label)
        self.button.setEnabled(enabled)
        self.button.setToolTip(tip)
        self.button.setProperty("state", state)

    def _open(self, which: str) -> None:
        fill = self._fill.get(which)
        if fill is None:
            return
        self.popover.clear()
        fill(self.popover)
        self.popover.show_above(self.details if which == "details" else self.files)

    def close_popover(self) -> None:
        self.popover.hide()


def file_row(pop: Popover, path: Path, label: str = "") -> None:
    """A made file in "Your files": its name, its size, and buttons to open it or to show it in its folder."""
    line = QHBoxLayout()
    name = ElidedLabel(path.name if path.exists() else f"{path.name}  (moved or deleted)")
    name.setToolTip(str(path))
    if label:
        name.setText(f"{name.text()}  {label}")
    size = QLabel(format_size(path.stat().st_size) if path.exists() else "")
    size.setObjectName("Small")
    line.addWidget(name, 1)
    line.addWidget(size)
    if path.exists():
        open_btn = IconButton("play", "Open it with your default player")
        open_btn.clicked.connect(lambda _=False, p=path: shell.open_path(p))
        show = IconButton("folder", "Show it in its folder")
        show.clicked.connect(lambda _=False, p=path: shell.reveal_in_folder(p))
        for b in (open_btn, show):
            b.setFixedSize(28, 28)
            line.addWidget(b)
    pop.lay.addLayout(line)
