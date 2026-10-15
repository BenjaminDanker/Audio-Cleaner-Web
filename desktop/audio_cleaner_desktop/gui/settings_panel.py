"""The settings card: every option has a short tooltip and a "?" button with a longer explanation."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QToolTip, QVBoxLayout, QWidget,
)

from .. import engine
from .preferences import Preferences
from .widgets import ElidedLabel, IconButton, Segmented

STRENGTH_LABELS = [("gentle", "Gentle"), ("balanced", "Balanced"), ("strong", "Strong"), ("max", "Maximum")]
LOUDNESS_LABELS = {
    "podcast": "Podcast (-16 LUFS)", "youtube": "YouTube (-14 LUFS)", "broadcast": "Broadcast (-23 LUFS)",
    "keep": "Keep the original level",
}
SUBTITLE_LABELS = {
    "off": "No subtitles", "srt": "SRT file", "vtt": "VTT file", "txt": "Text transcript", "embed": "Inside the video",
}
FORMAT_LABELS = {"same": "Same as the original", "wav": "WAV", "flac": "FLAC", "mp3": "MP3", "m4a": "M4A", "opus": "Opus"}
PARALLEL_LABELS = {0: "Automatic", 1: "1", 2: "2", 3: "3", 4: "4"}


class InfoButton(IconButton):
    """A small "?" that explains a setting; hover shows the text, a click pins it near the cursor."""

    def __init__(self, text: str, parent: QWidget | None = None):
        super().__init__("info", text, size=16, parent=parent)
        self._text = text
        self.setFixedSize(24, 24)
        self.setAccessibleName("More about this setting")
        self.setAccessibleDescription(text)
        self.clicked.connect(lambda: QToolTip.showText(QCursor.pos(), self._text, self))


class _Row(QWidget):
    def __init__(self, title: str, info: str, control: QWidget | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.layout_.setSpacing(6)
        head = QHBoxLayout()
        head.setSpacing(2)
        self.title = QLabel(title)
        head.addWidget(self.title)
        head.addWidget(InfoButton(info))
        head.addStretch(1)
        self.layout_.addLayout(head)
        if control is not None:
            self.layout_.addWidget(control)


def _combo(options: dict, tooltip: str) -> QComboBox:
    box = QComboBox()
    for key, label in options.items():
        box.addItem(label, key)
    box.setToolTip(tooltip)
    box.setMaximumWidth(1000)
    return box


def _select(box: QComboBox, key: object) -> None:
    box.setCurrentIndex(max(0, box.findData(key)))


class SettingsPanel(QFrame):
    changed = Signal()

    def __init__(self, prefs: Preferences, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Card")
        self._prefs = prefs
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 20)
        layout.setSpacing(16)
        title = QLabel("Settings")
        title.setObjectName("CardTitle")
        layout.addWidget(title)

        self._strength = Segmented(STRENGTH_LABELS)
        self._strength.setToolTip("Lower keeps more room tone and music")
        layout.addWidget(_Row(
            "Noise reduction", "How much of the background gets removed. Gentle leaves more of the room and any music and "
            "sounds the most natural. Maximum removes as much as it can, which suits noisy places. Not sure? Maximum "
            "works well for most voice recordings.", self._strength,
        ))

        polish_row = _Row(
            "Voice polish", "After the noise is gone, the voice is evened out so it sounds clear and steady: the level is "
            "matched across the recording, a gentle compressor smooths loud and quiet words, the whole file is set to the "
            "loudness below, and a limiter stops any peak from clipping. Turn it off to keep only the noise removal.",
        )
        self._polish = QCheckBox("Polish the voice")
        self._polish.setToolTip("Level match, gentle compression, loudness target and a limiter")
        polish_row.layout_.addWidget(self._polish)
        self._loudness = _combo(LOUDNESS_LABELS, "How loud the finished file should be. Podcast suits most things.")
        polish_row.layout_.addWidget(self._loudness)
        layout.addWidget(polish_row)

        self._subtitles = _combo(SUBTITLE_LABELS, "Write down what is said, as a subtitle file or inside the video")
        self._language = _combo(
            {"auto": "Detect the language", **{c: n for c, (n, _) in engine.LANGUAGES.items()}},
            "The language that is spoken. Detecting it works well for clear speech.",
        )
        sub_row = _Row(
            "Subtitles", "Listens to the original recording and writes down what is said. Pick a subtitle file (SRT or "
            "VTT) or a plain text transcript to get a file next to the cleaned one, or put the subtitles inside the "
            "video so any player can switch them on. Audio files get an SRT file instead. This runs on your computer, so "
            "nothing is uploaded, but it adds time to each file. Choose the language yourself if the speech is noisy or "
            "mixes accents.", self._subtitles,
        )
        sub_row.layout_.addWidget(self._language)
        layout.addWidget(sub_row)

        self._where = Segmented([("next", "Next to original"), ("folder", "Choose folder")])
        self._where.setToolTip("Where the cleaned files are saved")
        where_row = _Row(
            "Save to", "Cleaned files get \"-cleaned\" added to their name and never replace anything. They are saved next to "
            "the original, or all together in a folder you pick.", self._where,
        )
        folder_line = QHBoxLayout()
        self._folder = ElidedLabel()
        self._folder.setObjectName("Small")
        self._browse = QPushButton("Change...")
        self._browse.setToolTip("Pick the folder for the cleaned files")
        self._browse.clicked.connect(self._change_folder)
        folder_line.addWidget(self._folder, 1)
        folder_line.addWidget(self._browse)
        self._folder_box = QWidget()
        self._folder_box.setLayout(folder_line)
        folder_line.setContentsMargins(0, 0, 0, 0)
        where_row.layout_.addWidget(self._folder_box)
        layout.addWidget(where_row)

        self._format = _combo(FORMAT_LABELS, "The file type for cleaned audio files. Videos always keep their own type.")
        layout.addWidget(_Row(
            "Audio file type", "Used when you clean audio files. \"Same as the original\" keeps MP3 as MP3, WAV as WAV and so "
            "on. FLAC and WAV keep full quality; MP3, M4A and Opus are smaller. Videos always keep their own format.",
            self._format,
        ))

        self._parallel = _combo(PARALLEL_LABELS, "How many files are cleaned at the same time")
        layout.addWidget(_Row(
            "Files at once", "Cleaning several files at the same time finishes a batch sooner, but each file takes a bit "
            "longer and uses more memory. Automatic picks a number that suits your computer, and never more than its "
            "memory allows.", self._parallel,
        ))
        layout.addStretch(1)

        self._load()
        self._strength.changed.connect(lambda v: self._store("strength", v))
        self._polish.toggled.connect(lambda v: self._store("polish", v))
        self._loudness.currentIndexChanged.connect(lambda _: self._store("loudness", self._loudness.currentData()))
        self._subtitles.currentIndexChanged.connect(lambda _: self._store("subtitles", self._subtitles.currentData()))
        self._language.currentIndexChanged.connect(lambda _: self._store("language", self._language.currentData()))
        self._format.currentIndexChanged.connect(lambda _: self._store("audio_format", self._format.currentData()))
        self._parallel.currentIndexChanged.connect(lambda _: self._store("parallel", self._parallel.currentData()))
        self._where.changed.connect(self._where_changed)

    def _load(self) -> None:
        p = self._prefs
        self._strength.set_value(p.strength)
        self._polish.setChecked(p.polish)
        _select(self._loudness, p.loudness)
        _select(self._subtitles, p.subtitles)
        _select(self._language, p.language)
        _select(self._format, p.audio_format)
        _select(self._parallel, p.parallel)
        mode = p.output_mode if p.output_folder else "next"
        self._where.set_value(mode)
        self._refresh_enabled()

    def _store(self, key: str, value: object) -> None:
        setattr(self._prefs, key, value)
        self._refresh_enabled()
        self.changed.emit()

    def _where_changed(self, mode: str) -> None:
        if mode == "folder" and not self._prefs.output_folder and not self._choose_folder():
            self._where.set_value("next")
            return
        self._store("output_mode", mode)

    def _choose_folder(self) -> bool:
        start = self._prefs.output_folder or self._prefs.last_input_folder
        folder = QFileDialog.getExistingDirectory(self, "Choose where to save the cleaned files", start)
        if not folder:
            return False
        self._prefs.output_folder = folder
        return True

    def _change_folder(self) -> None:
        if self._choose_folder():
            self._store("output_mode", "folder")

    def _refresh_enabled(self) -> None:
        self._loudness.setEnabled(self._polish.isChecked())
        self._language.setVisible(self._subtitles.currentData() != "off")
        folder_mode = self._where.value() == "folder"
        self._folder_box.setVisible(folder_mode)
        self._folder.setText(self._prefs.output_folder)
