"""Use these settings for other files: the files that wait, each with a tick (all of them ticked to begin with), and what they will get. Each file
keeps its own name and can still be opened and changed on its own afterwards; Start all then runs them one after the other."""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from .studio_widgets import link_button, small_label


@dataclass(frozen=True)
class Choice:
    id: int
    name: str
    meta: str  # its length and size, as the job list says them


class ShareSettingsDialog(QDialog):
    def __init__(self, source: str, summary: str, files: list[Choice], *, had_parts: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Use these settings for other files")
        self.setObjectName("ShareDialog")
        self.resize(560, 520)
        outer = QVBoxLayout(self)
        outer.setSpacing(10)
        title = QLabel("Use these settings for other files")
        title.setObjectName("Heading")
        outer.addWidget(title)
        what = QLabel(f"The files you tick get what <b>{_escape(source)}</b> is set to: {_escape(summary)}.")
        what.setTextFormat(Qt.TextFormat.RichText)
        what.setWordWrap(True)
        outer.addWidget(what)
        if had_parts:
            outer.addWidget(small_label("The parts you changed in this file stay with it: the other files get what most of this file has."))
        outer.addWidget(small_label("Each file keeps its own name. You can still open any of them and change it on its own afterwards."))

        head = QHBoxLayout()
        head.addWidget(QLabel("Files that are waiting"), 1)
        every = link_button("All", "Tick every file")
        none = link_button("None", "Untick every file")
        head.addWidget(every)
        head.addWidget(none)
        outer.addLayout(head)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        holder = QWidget()
        rows = QVBoxLayout(holder)
        rows.setContentsMargins(0, 0, 8, 0)
        rows.setSpacing(4)
        self.boxes: dict[int, QCheckBox] = {}
        for f in files:
            line = QHBoxLayout()
            box = QCheckBox(f.name)
            box.setChecked(True)
            box.setToolTip(f.name)
            box.toggled.connect(self._count)
            self.boxes[f.id] = box
            line.addWidget(box, 1)
            if f.meta:
                line.addWidget(small_label(f.meta))
            rows.addLayout(line)
        rows.addStretch(1)
        scroll.setWidget(holder)
        outer.addWidget(scroll, 1)
        every.clicked.connect(lambda: self._tick_all(True))
        none.clicked.connect(lambda: self._tick_all(False))

        buttons = QDialogButtonBox()
        self.ok = QPushButton()
        self.ok.setObjectName("Primary")
        self.ok.setDefault(True)
        buttons.addButton(self.ok, QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self._count()
        self.ok.setFocus()

    def _tick_all(self, on: bool) -> None:
        for box in self.boxes.values():
            box.setChecked(on)

    def _count(self) -> None:
        n = len(self.chosen())
        self.ok.setText(f"Use for {n} file{'s' if n != 1 else ''}" if n else "Use for the ticked files")
        self.ok.setEnabled(n > 0)

    def chosen(self) -> list[int]:
        return [i for i, box in self.boxes.items() if box.isChecked()]


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
