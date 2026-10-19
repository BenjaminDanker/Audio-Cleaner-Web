"""The settings card: every option has a short tooltip and a "?" button with a longer explanation."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from .. import engine, resources, sysinfo
from ..updater import UpdateController
from . import tips
from .preferences import Preferences
from .speech_model import SpeechModelDownloader, SpeechModelPanel
from .widgets import ElidedLabel, IconButton, Segmented, WrapCheck

STRENGTH_LABELS = [("gentle", "Gentle"), ("balanced", "Balanced"), ("strong", "Strong"), ("max", "Maximum")]
# The same wording as the website's Denoising form.
LOUDNESS_LABELS = {
    "podcast": "Podcast (\u221216 LUFS)", "youtube": "YouTube (\u221214 LUFS)", "broadcast": "Broadcast (\u221223 LUFS)",
    "keep": "Keep original loudness",
}
LOUDNESS_SHORT = {"podcast": "podcast", "youtube": "YouTube", "broadcast": "broadcast", "keep": "original loudness"}
NOISE_LABELS = {"auto": "Automatic", "light": "Light noise", "heavy": "Heavy noise"}
SUBTITLE_LABELS = {
    "off": "No subtitles", "srt": "SRT file", "vtt": "VTT file", "txt": "Text transcript", "embed": "Inside the video",
}
FORMAT_LABELS = {"same": "Same as the original", "wav": "WAV", "flac": "FLAC", "mp3": "MP3", "m4a": "M4A", "opus": "Opus"}
VIDEO_SIZE_LABELS = {
    "same": "Same as the original", "2160": "2160p (4K)", "1440": "1440p", "1080": "1080p (Full HD)", "720": "720p (HD)",
    "480": "480p",
}
PARALLEL_LABELS = {0: "Automatic", 1: "1", 2: "2", 3: "3", 4: "4"}


class InfoButton(IconButton):
    """A small "?" that explains a setting; hover shows the text, a click keeps it up (gui/tips.py)."""

    def __init__(self, text: str, parent: QWidget | None = None):
        super().__init__("info", text, size=16, parent=parent)
        self._text = text
        self.setFixedSize(24, 24)
        self.setAccessibleName("More about this setting")
        self.setAccessibleDescription(text)
        self.clicked.connect(lambda: tips.pin(self._text, self))


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

    def __init__(self, prefs: Preferences, downloader: SpeechModelDownloader | None = None, parent: QWidget | None = None,
                 highs_downloader: SpeechModelDownloader | None = None,
                 voice_downloader: SpeechModelDownloader | None = None, updater: UpdateController | None = None):
        super().__init__(parent)
        self.setObjectName("Card")
        self._prefs = prefs
        self.updater = updater or UpdateController(prefs, self)
        self.downloader = downloader or SpeechModelDownloader(self)
        self.highs_downloader = highs_downloader or SpeechModelDownloader(self, resources.HIGHS_MODEL)
        self.voice_downloader = voice_downloader or SpeechModelDownloader(self, resources.VOICE_MODEL)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 20)
        layout.setSpacing(16)
        title = QLabel("Settings")
        title.setObjectName("CardTitle")
        layout.addWidget(title)

        self._auto = WrapCheck("Pick the cleanup settings for each file")
        self._auto.setToolTip("Each file gets the noise reduction and extra cleanup that fit it")
        layout.addWidget(_Row(
            "Automatic best settings",
            "Each file is listened to before it is cleaned and gets what fits it:\n"
            "\u2022 Noise reduction: Gentle for a quiet recording, up to Maximum for a noisy one\n"
            "\u2022 Background noise: the way of removing noise that suits it\n"
            "\u2022 Missing highs: brought back when they are missing (phone calls, small files)\n"
            "\u2022 Rebuild the voice: when it sounds echoey, muffled or unclear (needs its model, downloaded once)\n\n"
            "The noise is measured by the noise remover itself, so wind, keyboards and chatter count as much as a "
            "steady hiss. The list shows what each file got. Untick it to choose these yourself.",
            self._auto,
        ))

        self._strength = Segmented(STRENGTH_LABELS)
        self._strength.setToolTip("How much background noise to remove. Gentle sounds the most natural.")
        layout.addWidget(_Row(
            "Noise reduction",
            "How much of the background gets removed.\n\n"
            "\u2022 Gentle: takes the edge off and keeps some room sound and music. Sounds the most natural.\n"
            "\u2022 Balanced: most of the noise goes, the room still sounds like a room.\n"
            "\u2022 Strong: nearly all of it goes.\n"
            "\u2022 Maximum: removes as much as it can. Best for noisy places.\n\n"
            "Not sure? Maximum works well for most voice recordings. Try Gentle if the voice starts to sound thin.",
            self._strength,
        ))

        self._noise = _combo(NOISE_LABELS, "What kind of background noise the recording has. Automatic measures it.")
        noise_row = _Row(
            "Background noise",
            "Audio Cleaner has two ways of removing noise, each best at its own kind of recording.\n\n"
            "\u2022 Automatic: measures each file and picks for you\n"
            "\u2022 Light noise: hiss, hum, a quiet room or a fan. Keeps the voice the most natural.\n"
            "\u2022 Heavy noise: a caf\u00e9, traffic, a crowd or wind, with noise almost as loud as the voice",
            self._noise,
        )
        self._auto_note = QLabel("Picked for each file while \u201cAutomatic best settings\u201d is on.")
        self._auto_extra_note = QLabel("Restoring the highs and rebuilding the voice are picked for each file while "
                                       "\u201cAutomatic best settings\u201d is on.")
        self._auto_extra_note.setObjectName("Small")
        self._auto_extra_note.setWordWrap(True)
        self._auto_extra_note.hide()
        self._auto_note.setObjectName("Small")
        self._auto_note.setWordWrap(True)
        self._auto_note.hide()
        noise_row.layout_.addWidget(self._auto_note)
        # Rebuilding the voice replaces the noise removal, so the two settings above don't apply while it is on.
        self._rebuild_note = QLabel("Noise reduction and Background noise aren\u2019t used while \u201cRebuild the voice\u201d "
                                    "is on: it removes the noise its own way. Untick it below to use them.")
        self._rebuild_note.setObjectName("Small")
        self._rebuild_note.setWordWrap(True)
        self._rebuild_note.hide()
        noise_row.layout_.addWidget(self._rebuild_note)
        layout.addWidget(noise_row)

        leveling_row = _Row(
            "Leveling",
            "After the noise is gone, the voice is evened out so it sounds clear and steady:\n"
            "\u2022 quiet and loud parts of the recording are matched\n"
            "\u2022 a gentle compressor smooths loud and quiet words\n"
            "\u2022 the whole file is set to the loudness you pick\n"
            "\u2022 a limiter stops any peak from clipping\n\n"
            "Loudness is measured in LUFS (loudness units). The closer to 0, the louder:\n"
            "\u2022 -14: YouTube, Spotify and most streaming sites\n"
            "\u2022 -16: podcasts, and a good choice for most things\n"
            "\u2022 -23: TV and radio broadcast\n"
            "Sites turn louder files down anyway, so pick where it will be played.\n\n"
            "Turn leveling off to keep only the noise removal.",
        )
        self._leveling = QCheckBox("Leveling")
        self._leveling.setToolTip("Even out the voice and set the loudness below")
        leveling_row.layout_.addWidget(self._leveling)
        self._loudness = _combo(
            LOUDNESS_LABELS, "How loud the finished file is, in LUFS (closer to 0 is louder). Podcast suits most things."
        )
        leveling_row.layout_.addWidget(self._loudness)
        layout.addWidget(leveling_row)

        self._subtitles = _combo(SUBTITLE_LABELS, "Write down what is said, as a subtitle file or inside the video")
        self._language = _combo(
            {"auto": "Detect the language", **{c: n for c, (n, _) in engine.LANGUAGES.items()}},
            "The language that is spoken. Detecting it works well for clear speech.",
        )
        sub_row = _Row(
            "Subtitles",
            "Listens to the original recording and writes down what is said.\n\n"
            "\u2022 SRT or VTT file: subtitles saved next to the cleaned file\n"
            "\u2022 Text transcript: just the words, as plain text\n"
            "\u2022 Inside the video: any player can switch them on (audio files get an SRT file instead)\n\n"
            "It runs on your computer, so nothing is uploaded, but it adds time to each file. Choose the language "
            "yourself if the speech is noisy or mixes accents.",
            self._subtitles,
        )
        sub_row.layout_.addWidget(self._language)
        self.speech_panel = SpeechModelPanel(self.downloader)
        sub_row.layout_.addWidget(self.speech_panel)
        self.downloader.changed.connect(self._refresh_enabled)
        layout.addWidget(sub_row)

        self._highs = QCheckBox("Restore missing highs (phone-quality recordings)")
        self._highs.setToolTip("For phone calls and other recordings that sound dull or muffled")
        self._echo = WrapCheck("Remove speaker echo using the second audio track (your desktop sound)")
        self._echo.setToolTip("For recordings with your desktop sound on a second audio track")
        self._rebuild = WrapCheck("Rebuild the voice")
        self._rebuild.setToolTip("Also removes room echo and brings back dull sound. It takes several times longer. "
                                 "Noise reduction and Background noise do not apply then.")
        extra_row = _Row(
            "Extra cleanup",
            "Restore missing highs: phone calls, voice messages, old or low-quality recordings have lost their high "
            "frequencies, which makes them sound dull and muffled. Audio Cleaner brings them back, only for "
            "recordings that really are missing them; normal recordings are never changed. It needs an extra model, "
            "downloaded once.\n\n"
            "Remove speaker echo: if your speakers' sound got into your microphone (a game, a call, music), Audio "
            "Cleaner can remove it, but only when the recording also has that speaker sound on its own audio track. "
            "OBS can record that: in Settings > Output (Output Mode: Advanced) > Recording, tick audio tracks 1 "
            "and 2; then in Edit > Advanced Audio Properties put your mic on track 1 only and Desktop Audio on "
            "track 2 only. Files with one audio track are cleaned as usual.\n\n"
            "Rebuild the voice: instead of filtering the noise out, Audio Cleaner rebuilds the voice, which also "
            "removes room echo and brings back dull sound. It takes several times longer, and on very echoey "
            "recordings it can change a word, so listen before you publish. The Noise reduction and Background "
            "noise settings do not apply to it.",
        )
        extra_row.layout_.addWidget(self._auto_extra_note)
        extra_row.layout_.addWidget(self._highs)
        self.highs_panel = SpeechModelPanel(
            self.highs_downloader, need="Restoring the highs needs an extra model", what="the model that restores highs",
        )
        extra_row.layout_.addWidget(self.highs_panel)
        self.highs_downloader.changed.connect(self._refresh_enabled)
        extra_row.layout_.addWidget(self._echo)
        extra_row.layout_.addWidget(self._rebuild)
        self.voice_panel = SpeechModelPanel(
            self.voice_downloader, need="Rebuilding the voice needs an extra model", what="the model that rebuilds the voice",
        )
        extra_row.layout_.addWidget(self.voice_panel)
        self.voice_downloader.changed.connect(self._refresh_enabled)
        layout.addWidget(extra_row)

        self._where = Segmented([("next", "Next to original"), ("folder", "Choose folder")])
        self._where.setToolTip("Where the cleaned files are saved")
        where_row = _Row(
            "Save to",
            "Cleaned files get \"-cleaned\" added to their name, next to the original or all together in a folder "
            "you pick.\n\n"
            "They never replace your original. Cleaning a file again with \"Clean again\" replaces that file's "
            "earlier cleaned version, which goes to the Recycle Bin (Trash on a Mac), so you can still get it back.",
            self._where,
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

        self._format = _combo(
            FORMAT_LABELS,
            "The usual file type for cleaned audio files. You can pick another for any file in the list, under its name.",
        )
        layout.addWidget(_Row(
            "Audio file type",
            "The file type audio files are saved as, unless you pick another one for a file in the list (the type "
            "next to its name). Videos keep their own type unless you pick another for them there.\n\n"
            "\u2022 Same as the original: MP3 stays MP3, WAV stays WAV, and so on\n"
            "\u2022 FLAC and WAV: full quality, bigger files\n"
            "\u2022 MP3, M4A and Opus: smaller files",
            self._format,
        ))

        self._video_size = _combo(
            VIDEO_SIZE_LABELS,
            "The size of the picture in videos you save. Bigger than the original only stretches the picture.",
        )
        layout.addWidget(_Row(
            "Video size",
            "How big the picture is in the videos you save (a video made from a sound has a plain black picture).\n\n"
            "\u2022 Same as the original: the picture is kept as it is when the file type allows it, which is fastest "
            "and loses nothing\n"
            "\u2022 2160p (4K), 1440p, 1080p, 720p, 480p: the picture is resized so its shorter side has that many "
            "pixels. It keeps its shape, and a tall video stays tall. Smaller sizes make smaller files.\n\n"
            "Bigger than the original only stretches the picture: it does not add any detail and makes the file "
            "larger. Resizing saves the picture again, which takes longer than keeping it.",
            self._video_size,
        ))

        self._parallel = _combo(PARALLEL_LABELS, "How many files are cleaned at the same time")
        layout.addWidget(_Row(
            "Files at once",
            "Cleaning several files at the same time finishes a batch sooner, but each file takes a bit longer and "
            f"uses more memory: about {sysinfo.job_memory_gb():.1f} GB for each file at once, "
            f"{sysinfo.job_memory_gb(subtitles=True):.1f} GB with subtitles and "
            f"{sysinfo.job_memory_gb(rebuild_voice=True):.1f} GB with Rebuild the voice.\n\n"
            "Automatic picks a number that suits your computer, and never more than its memory allows.",
            self._parallel,
        ))

        self._updates = WrapCheck("Check for updates when Audio Cleaner starts")
        self._updates.setToolTip("Looks for a newer version a few seconds after the app starts, at most once a day")
        updates_row = _Row(
            "Updates",
            "When this is on, Audio Cleaner looks for a newer version a few seconds after it starts (at most "
            "once a day) and downloads it in the background. It sends nothing but a normal web request: your IP "
            "address and the version of this app.\n\n"
            "\u2022 Windows: the new version installs by itself when you close the app, over the old one. Your settings, "
            "history and models stay.\n"
            "\u2022 Mac: a message offers to open the installer; it asks for your password as any installer does.\n\n"
            "Every update is checked against a digital signature and a checksum before it is used. Turn this off to "
            "update by hand (download the new installer from the releases page and run it).",
            self._updates,
        )
        update_line = QHBoxLayout()
        self._check_now = QPushButton("Check now")
        self._check_now.setToolTip("Look for a newer version right now")
        self._update_status = QLabel()
        self._update_status.setObjectName("Small")
        self._update_status.setWordWrap(True)
        update_line.addWidget(self._check_now)
        update_line.addWidget(self._update_status, 1)
        updates_row.layout_.addLayout(update_line)
        layout.addWidget(updates_row)
        layout.addStretch(1)

        self._load()
        self._auto.toggled.connect(lambda v: self._store("auto_settings", v))
        self._strength.changed.connect(lambda v: self._store("strength", v))
        self._leveling.toggled.connect(lambda v: self._store("leveling", v))
        self._loudness.currentIndexChanged.connect(lambda _: self._store("loudness", self._loudness.currentData()))
        self._noise.currentIndexChanged.connect(lambda _: self._store("noise", self._noise.currentData()))
        self._echo.toggled.connect(lambda v: self._store("echo_track", v))
        self._highs.toggled.connect(lambda v: self._store("restore_highs", v))
        self._rebuild.toggled.connect(lambda v: self._store("rebuild_voice", v))
        self._subtitles.currentIndexChanged.connect(lambda _: self._store("subtitles", self._subtitles.currentData()))
        self._language.currentIndexChanged.connect(lambda _: self._store("language", self._language.currentData()))
        self._format.currentIndexChanged.connect(lambda _: self._store("audio_format", self._format.currentData()))
        self._video_size.currentIndexChanged.connect(lambda _: self._store("video_size", self._video_size.currentData()))
        self._parallel.currentIndexChanged.connect(lambda _: self._store("parallel", self._parallel.currentData()))
        self._where.changed.connect(self._where_changed)
        self._updates.toggled.connect(lambda v: self._store("check_updates", v))
        self._check_now.clicked.connect(self.updater.check_now)
        self.updater.statusChanged.connect(self._show_update_status)
        self._show_update_status(self.updater.status)

    def _show_update_status(self, text: str) -> None:
        self._update_status.setText(text)
        self._update_status.setVisible(bool(text))
        self._check_now.setEnabled(not self.updater.busy)  # one check at a time

    def _load(self) -> None:
        p = self._prefs
        self._auto.setChecked(p.auto_settings)
        self._strength.set_value(p.strength)
        self._leveling.setChecked(p.leveling)
        _select(self._loudness, p.loudness)
        _select(self._noise, p.noise)
        self._echo.setChecked(p.echo_track)
        self._highs.setChecked(p.restore_highs)
        self._rebuild.setChecked(p.rebuild_voice)
        _select(self._subtitles, p.subtitles)
        _select(self._language, p.language)
        _select(self._format, p.audio_format)
        _select(self._video_size, p.video_size)
        _select(self._parallel, p.parallel)
        self._updates.setChecked(p.check_updates)
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

    def full_height(self) -> int:
        """Height with every optional line showing (subtitle language, output folder), so the window can be
        sized once to fit them all."""
        spacing = self.layout().spacing() // 2 + 6
        optional = (self._language, self._folder_box, self.speech_panel, self.highs_panel, self.voice_panel,
                    self._rebuild_note, self._auto_note, self._auto_extra_note, self._update_status)
        extra = sum(w.sizeHint().height() + spacing for w in optional if w.isHidden())
        return self.sizeHint().height() + extra

    def _refresh_enabled(self) -> None:
        self._loudness.setEnabled(self._leveling.isChecked())
        auto = self._auto.isChecked()
        rebuilt = self._rebuild.isChecked() and not auto  # always rebuilt (automatic rebuilds only some files)
        self._strength.setEnabled(not rebuilt and not auto)
        self._noise.setEnabled(not rebuilt and not auto)
        self._highs.setEnabled(not auto)
        self._rebuild.setEnabled(not auto)
        self._rebuild_note.setVisible(rebuilt)
        self._auto_note.setVisible(auto)
        self._auto_extra_note.setVisible(auto)
        wants_subs = self._subtitles.currentData() != "off"
        self._language.setVisible(wants_subs)
        dl = self.downloader
        self.speech_panel.setVisible(dl.running or bool(dl.last_error) or (wants_subs and not dl.ready))
        hd = self.highs_downloader
        self.highs_panel.setVisible(hd.running or bool(hd.last_error) or ((auto or self._highs.isChecked()) and not hd.ready))
        vd = self.voice_downloader
        self.voice_panel.setVisible(vd.running or bool(vd.last_error) or ((auto or rebuilt) and not vd.ready))
        folder_mode = self._where.value() == "folder"
        self._folder_box.setVisible(folder_mode)
        self._folder.setText(self._prefs.output_folder)
