"""Settings: what is about the app and not about a file, which the editor screen has no place for: where the results are saved, the usual type of
audio files, how many files are cleaned at once, updates, and the models that are downloaded once (the speech model, the one that rebuilds the voice).
Everything about a file itself (strength, Leveling, subtitles, the name and type of the result) is on the screen."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from .. import sysinfo
from ..labels import FORMAT_LABELS
from ..updater import UpdateController
from . import tips
from .preferences import Preferences
from .speech_model import SpeechModelDownloader, SpeechModelPanel
from .widgets import ElidedLabel, IconButton, Segmented, WrapCheck

PARALLEL_LABELS = {0: "Automatic", 1: "1", 2: "2", 3: "3", 4: "4"}


class HelpButton(IconButton):
    """A small "?" that explains a setting; a click keeps the text up (gui/tips.py)."""

    def __init__(self, text: str, parent: QWidget | None = None):
        super().__init__("info", text, size=16, parent=parent)
        self._text = text
        self.setFixedSize(24, 24)
        self.setAccessibleName("More about this setting")
        self.setAccessibleDescription(text)
        self.clicked.connect(lambda: tips.pin(self._text, self))


class _Row(QWidget):
    def __init__(self, title: str, info: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.layout_.setSpacing(6)
        head = QHBoxLayout()
        head.setSpacing(2)
        label = QLabel(title)
        label.setObjectName("CardTitle")
        head.addWidget(label)
        head.addWidget(HelpButton(info))
        head.addStretch(1)
        self.layout_.addLayout(head)


class SettingsDialog(QDialog):
    changed = Signal()

    def __init__(self, prefs: Preferences, speech: SpeechModelDownloader, voice: SpeechModelDownloader, updater: UpdateController, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(520, 640)
        self._prefs, self.updater = prefs, updater
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        holder = QWidget()
        lay = QVBoxLayout(holder)
        lay.setSpacing(16)
        scroll.setWidget(holder)
        outer.addWidget(scroll, 1)

        self._where = Segmented([("next", "Next to original"), ("folder", "Choose folder")])
        self._where.setToolTip("Where the cleaned files are saved")
        where = _Row("Save to", "Cleaned files get \"-cleaned\" added to their name, next to the original or all together in a folder you pick.\n\n"
                                "They never replace your original. When you save a changed result under the name of the earlier one, the earlier file goes to "
                                "the Recycle Bin (Trash on a Mac), so you can still get it back.")
        where.layout_.addWidget(self._where)
        line = QHBoxLayout()
        self._folder = ElidedLabel()
        self._folder.setObjectName("Small")
        self._browse = QPushButton("Change...")
        self._browse.setToolTip("Pick the folder for the cleaned files")
        line.addWidget(self._folder, 1)
        line.addWidget(self._browse)
        self._folder_box = QWidget()
        self._folder_box.setLayout(line)
        line.setContentsMargins(0, 0, 0, 0)
        where.layout_.addWidget(self._folder_box)
        lay.addWidget(where)

        self._format = QComboBox()
        for key, label in FORMAT_LABELS.items():
            self._format.addItem(label, key)
        fmt = _Row("Audio file type", "The file type audio files are saved as, unless you pick another one for a file in the Output pane. Videos keep their own type "
                                      "unless you pick another for them there.\n\n• Same as the original: MP3 stays MP3, WAV stays WAV, and so on\n"
                                      "• FLAC and WAV: full quality, bigger files\n• MP3, M4A and Opus: smaller files")
        fmt.layout_.addWidget(self._format)
        lay.addWidget(fmt)

        self._parallel = QComboBox()
        for key, label in PARALLEL_LABELS.items():
            self._parallel.addItem(label, key)
        par = _Row("Files at once", "Cleaning several files at the same time finishes a batch sooner, but each file takes a bit longer and uses more memory: "
                                    f"about {sysinfo.job_memory_gb():.1f} GB for each file at once, {sysinfo.job_memory_gb(subtitles=True):.1f} GB with subtitles "
                                    f"and {sysinfo.job_memory_gb(rebuild_voice=True):.1f} GB with Voice rebuild.\n\nAutomatic picks a number that suits your "
                                    "computer, and never more than its memory allows.")
        par.layout_.addWidget(self._parallel)
        lay.addWidget(par)

        models = _Row("Models", "Everything runs on this computer, so the models are downloaded once. Noise removal's are downloaded by setup (or by the app "
                                "when they are missing); the speech model for subtitles and the one that rebuilds the voice are downloaded when you want them.")
        self.speech_panel = SpeechModelPanel(speech)
        self.voice_panel = SpeechModelPanel(voice, need="Voice rebuild needs an extra model", what="the model that rebuilds the voice")
        speech.changed.connect(self._refresh)
        voice.changed.connect(self._refresh)
        self._speech, self._voice = speech, voice
        models.layout_.addWidget(self.speech_panel)
        models.layout_.addWidget(self.voice_panel)
        lay.addWidget(models)

        self._updates = WrapCheck("Check for updates when Audio Cleaner starts")
        self._updates.setToolTip("Looks for a newer version a few seconds after the app starts, at most once a day")
        up = _Row("Updates", "When this is on, Audio Cleaner looks for a newer version a few seconds after it starts (at most once a day) and downloads it in the "
                             "background. It sends nothing but a normal web request: your IP address and the version of this app.\n\n"
                             "• Windows: the new version installs by itself when you close the app, over the old one. Your settings, history and models stay.\n"
                             "• Mac: a message offers to open the installer; it asks for your password as any installer does.\n\n"
                             "Every update is checked against a digital signature and a checksum before it is used. Turn this off to update by hand (download "
                             "the new installer from the releases page and run it).")
        up.layout_.addWidget(self._updates)
        row = QHBoxLayout()
        self._check_now = QPushButton("Check now")
        self._check_now.setToolTip("Look for a newer version right now")
        self._status = QLabel()
        self._status.setObjectName("Small")
        self._status.setWordWrap(True)
        row.addWidget(self._check_now)
        row.addWidget(self._status, 1)
        up.layout_.addLayout(row)
        lay.addWidget(up)
        lay.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        bottom.addWidget(close)
        outer.addLayout(bottom)

        self._load()
        self._where.changed.connect(self._where_changed)
        self._browse.clicked.connect(self._change_folder)
        self._format.currentIndexChanged.connect(lambda _: self._store("audio_format", self._format.currentData()))
        self._parallel.currentIndexChanged.connect(lambda _: self._store("parallel", self._parallel.currentData()))
        self._updates.toggled.connect(lambda v: self._store("check_updates", v))
        self._check_now.clicked.connect(updater.check_now)
        updater.statusChanged.connect(self._show_update_status)
        self._show_update_status(updater.status)

    def _load(self) -> None:
        p = self._prefs
        self._format.setCurrentIndex(max(0, self._format.findData(p.audio_format)))
        self._parallel.setCurrentIndex(max(0, self._parallel.findData(p.parallel)))
        self._updates.setChecked(p.check_updates)
        self._where.set_value(p.output_mode if p.output_folder else "next")
        self._refresh()

    def _refresh(self) -> None:
        self._folder_box.setVisible(self._where.value() == "folder")
        self._folder.setText(self._prefs.output_folder)
        for panel, dl in ((self.speech_panel, self._speech), (self.voice_panel, self._voice)):
            panel.refresh()
            panel.setVisible(True)

    def _store(self, key: str, value: object) -> None:
        setattr(self._prefs, key, value)
        self._refresh()
        self.changed.emit()

    def _show_update_status(self, text: str) -> None:
        self._status.setText(text)
        self._status.setVisible(bool(text))
        self._check_now.setEnabled(not self.updater.busy)

    def _where_changed(self, mode: str) -> None:
        if mode == "folder" and not self._prefs.output_folder and not self.choose_folder():
            self._where.set_value("next")
            return
        self._store("output_mode", mode)

    def choose_folder(self) -> bool:
        start = self._prefs.output_folder or self._prefs.last_input_folder
        folder = QFileDialog.getExistingDirectory(self, "Choose where to save the cleaned files", start)
        if not folder:
            return False
        self._prefs.output_folder = folder
        return True

    def _change_folder(self) -> None:
        if self.choose_folder():
            self._where.set_value("folder")
            self._store("output_mode", "folder")
