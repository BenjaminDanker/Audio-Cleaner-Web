"""The (i) next to the settings' heading: "All the options", every option side by side, and Good to know (the web's InfoModal)."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from . import copy


def _heading(text: str, level: int = 3) -> QLabel:
    label = QLabel(text)
    label.setObjectName("DlgH3" if level == 3 else "DlgH4")
    label.setWordWrap(True)
    return label


def _paragraph(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    return label


def _bullets(items: list[str]) -> QLabel:
    label = QLabel("<ul style='margin-left:14px; margin-top:2px; margin-bottom:2px'>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>")
    label.setTextFormat(Qt.TextFormat.RichText)
    label.setWordWrap(True)
    return label


class InfoDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("All the options")
        self.setObjectName("InfoDialog")
        self.resize(760, 640)
        outer = QVBoxLayout(self)
        head = QHBoxLayout()
        title = QLabel("All the options")
        title.setObjectName("Heading")
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        head.addWidget(title, 1)
        head.addWidget(close)
        outer.addLayout(head)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        holder = QWidget()
        body = QVBoxLayout(holder)
        body.setSpacing(8)
        body.setContentsMargins(0, 0, 12, 0)
        scroll.setWidget(holder)
        outer.addWidget(scroll, 1)

        body.addWidget(_heading("Voice: Noise removal or Voice rebuild?"))
        two = QHBoxLayout()
        for title, points in (("Noise removal", copy.VOICE_POINTS), ("Voice rebuild", copy.REBUILD_POINTS)):
            col = QVBoxLayout()
            col.addWidget(_heading(title, 4))
            col.addWidget(_bullets(points))
            col.addStretch(1)
            two.addLayout(col, 1)
        body.addLayout(two)
        body.addWidget(_heading("Noise removal"))
        body.addWidget(_paragraph(f"<b>Strength</b> {copy.STRENGTH_TEXT}"))
        body.addWidget(_paragraph(f"<b>Remover</b> {copy.REMOVER_TEXT}"))
        body.addWidget(_heading("Echo removal"))
        body.addWidget(_paragraph(copy.ECHO_TEXT))
        body.addWidget(_heading("Leveling"))
        body.addWidget(_paragraph(copy.LEVELING_TEXT))
        grid = QGridLayout()
        for i, (name, text) in enumerate(copy.LEVELING_ROWS):
            a = QLabel(f"<b>{name}</b>")
            grid.addWidget(a, i, 0)
            grid.addWidget(_paragraph(text), i, 1)
        grid.setColumnStretch(1, 1)
        body.addLayout(grid)
        body.addWidget(_heading("Subtitles"))
        body.addWidget(_paragraph(copy.SUBTITLES_TEXT))
        good = _heading("Good to know")
        good.setObjectName("DlgH3")
        good.setAccessibleName("Good to know")
        body.addWidget(good)
        body.addWidget(_bullets(copy.GOOD_TO_KNOW))
        body.addWidget(_heading("Good to know"))
        body.addWidget(_bullets(copy.GOOD_TO_KNOW_ALTERNATIVES))
        body.addStretch(1)
