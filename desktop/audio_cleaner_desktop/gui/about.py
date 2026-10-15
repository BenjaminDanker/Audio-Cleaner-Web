"""The About dialog: version, licence notice and the third-party notices that ship with the app."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QPlainTextEdit, QVBoxLayout, QWidget,
)

from .. import __version__, config, resources

ICON = Path(__file__).resolve().parent.parent / "assets" / "icon.png"
QT_NOTICE = (
    "Audio Cleaner uses Qt and PySide6 under the GNU LGPL v3, dynamically linked. You may replace those libraries in "
    "the application folder with your own builds. FFmpeg is included as separate programs built under the LGPL v2.1; "
    "see ffmpeg-buildinfo.txt for sources and build steps."
)


def notice_files() -> list[Path]:
    folder = resources.licenses_dir()
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.iterdir() if p.is_file()), key=lambda p: (p.stem.lower() != "readme", p.name.lower()))


class AboutDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"About {config.APP_NAME}")
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 20)
        layout.setSpacing(12)

        head = QHBoxLayout()
        head.setSpacing(16)
        icon = QLabel()
        icon.setPixmap(QPixmap(str(ICON)).scaled(72, 72, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        head.addWidget(icon, 0, Qt.AlignmentFlag.AlignTop)
        text = QVBoxLayout()
        text.setSpacing(2)
        name = QLabel(config.APP_NAME)
        name.setObjectName("Heading")
        version = QLabel(f"Version {__version__}")
        version.setObjectName("Muted")
        rights = QLabel("Free to use \u00b7 \u00a9 2026 Benjamin Danker")
        for w in (name, version, rights):
            text.addWidget(w)
        head.addLayout(text, 1)
        layout.addLayout(head)

        qt = QLabel(QT_NOTICE)
        qt.setWordWrap(True)
        layout.addWidget(qt)

        title = QLabel("Third-party notices")
        title.setObjectName("CardTitle")
        layout.addWidget(title)
        files = notice_files()
        if files:
            self._picker = QComboBox()
            self._picker.setToolTip("Licence texts of the software that is part of Audio Cleaner")
            for p in files:
                self._picker.addItem(p.name, p)
            self._viewer = QPlainTextEdit()
            self._viewer.setReadOnly(True)
            self._viewer.setMinimumHeight(220)
            self._picker.currentIndexChanged.connect(self._show_selected)
            layout.addWidget(self._picker)
            layout.addWidget(self._viewer, 1)
            self._show_selected()
        else:
            empty = QLabel("No third-party notices were found. They are part of the installed app.")
            empty.setObjectName("Muted")
            empty.setWordWrap(True)
            layout.addWidget(empty)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        self._close = buttons.button(QDialogButtonBox.StandardButton.Close)
        self._close.setDefault(True)
        layout.addWidget(buttons)
        self.resize(self.sizeHint())

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._close.setFocus()

    def _show_selected(self) -> None:
        path: Path = self._picker.currentData()
        try:
            self._viewer.setPlainText(path.read_text(encoding="utf-8", errors="replace"))
        except OSError as e:
            self._viewer.setPlainText(f"Could not read {path.name}: {e.strerror or e}")
