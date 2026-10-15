"""Preview: play a file's original and cleaned audio (and video) and switch between them.

QtMultimedia is imported lazily. Where it cannot load (missing system audio libraries,
no sound device), the pane says so and the rest of the app keeps working.
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QStackedWidget, QVBoxLayout, QWidget

from . import icons
from .widgets import ElidedLabel, IconButton, Segmented, SeekSlider, format_time

log = logging.getLogger(__name__)

_multimedia: SimpleNamespace | None | bool = False  # False = not tried yet


def load_multimedia() -> SimpleNamespace | None:
    """The QtMultimedia classes, or None when they cannot be loaded here."""
    global _multimedia
    if _multimedia is False:
        try:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
            from PySide6.QtMultimediaWidgets import QVideoWidget
        except (ImportError, OSError):
            log.warning("QtMultimedia is not available; previews are turned off", exc_info=True)
            _multimedia = None
        else:
            _multimedia = SimpleNamespace(QAudioOutput=QAudioOutput, QMediaPlayer=QMediaPlayer, QVideoWidget=QVideoWidget)
    return _multimedia


class PreviewPane(QFrame):
    closed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.key: int | None = None  # whatever the caller uses to say which file is shown
        self._original: Path | None = None
        self._cleaned: Path | None = None
        self._player = None
        self._resume_at = 0
        self._resume_playing = False
        self._scrubbing = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 14)
        layout.setSpacing(10)
        head = QHBoxLayout()
        title = QLabel("Preview")
        title.setObjectName("CardTitle")
        self._name = ElidedLabel()
        self._name.setObjectName("Small")
        close = IconButton("close", "Close the preview")
        close.clicked.connect(self._close)
        head.addWidget(title)
        head.addWidget(self._name, 1)
        head.addWidget(close)
        layout.addLayout(head)

        self._stage = QStackedWidget()
        self._stage.setObjectName("Preview")
        self._stage.setMinimumHeight(130)
        self._art = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self._note = QLabel("Preview unavailable", alignment=Qt.AlignmentFlag.AlignCenter)
        self._note.setWordWrap(True)
        self._stage.addWidget(self._art)
        self._stage.addWidget(self._note)
        layout.addWidget(self._stage, 1)

        controls = QHBoxLayout()
        controls.setSpacing(10)
        self._play = IconButton("play", "Play or pause (Space)", size=22)
        self._play.setFixedSize(38, 38)
        self._slider = SeekSlider()
        self._slider.setToolTip("Drag to jump to another moment")
        self._slider.setAccessibleName("Position")
        self._time = QLabel("0:00 / 0:00")
        self._time.setObjectName("Small")
        self._which = Segmented([("original", "Original"), ("cleaned", "Cleaned")])
        self._which.setToolTip("Switch between the recording as it was and the cleaned version; the position stays")
        controls.addWidget(self._play)
        controls.addWidget(self._slider, 1)
        controls.addWidget(self._time)
        self._which.setMinimumWidth(190)
        controls.addWidget(self._which)
        layout.addLayout(controls)

        self._play.clicked.connect(self.toggle)
        self._which.changed.connect(self._switch_source)
        self._slider.sliderPressed.connect(lambda: setattr(self, "_scrubbing", True))
        self._slider.sliderReleased.connect(self._scrub_done)
        self._slider.sliderMoved.connect(self._show_position)

        mm = load_multimedia()
        if mm is None:
            self._stage.setCurrentWidget(self._note)
            self._set_controls(False)
        else:
            self._build_player(mm)
        self._which.set_enabled_key("cleaned", False)
        self._art.setPixmap(icons.pixmap("waveform", QColor("#aab3c5"), 64, self.devicePixelRatioF()))

    # -- public -------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._player is not None

    def show_files(self, key: int, original: Path, cleaned: Path | None, has_video: bool, name: str) -> None:
        self.stop()
        self.key = key
        self._original, self._cleaned = original, cleaned
        self._name.setText(name)
        self._which.set_enabled_key("cleaned", cleaned is not None)
        self._which.blockSignals(True)
        self._which.set_value("cleaned" if cleaned else "original")
        self._which.blockSignals(False)
        if self._player is not None:
            self._video.setVisible(has_video)
            self._stage.setCurrentWidget(self._video if has_video else self._art)
            self._resume_at, self._resume_playing = 0, False
            self._load(cleaned or original)

    def set_cleaned(self, path: Path) -> None:
        self._cleaned = path
        self._which.set_enabled_key("cleaned", True)

    def stop(self) -> None:
        if self._player is not None:
            self._player.stop()

    def release(self) -> None:
        """Stop and close the file, so it can be moved or renamed (Windows locks files that are open)."""
        if self._player is not None:
            self._player.stop()
            self._player.setSource(QUrl())

    def toggle(self) -> None:
        if self._player is None:
            return
        if self._player.playbackState() == self._mm.QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    # -- internals ----------------------------------------------------------

    def _build_player(self, mm: SimpleNamespace) -> None:
        self._mm = mm
        self._player = mm.QMediaPlayer(self)
        self._audio = mm.QAudioOutput(self)
        self._player.setAudioOutput(self._audio)
        self._video = mm.QVideoWidget()
        self._stage.insertWidget(0, self._video)
        self._player.setVideoOutput(self._video)
        self._player.durationChanged.connect(lambda d: self._slider.setRange(0, d))
        self._player.positionChanged.connect(self._on_position)
        self._player.playbackStateChanged.connect(self._on_state)
        self._player.mediaStatusChanged.connect(self._on_status)
        self._player.errorOccurred.connect(self._on_error)

    def _set_controls(self, on: bool) -> None:
        for w in (self._play, self._slider, self._which):
            w.setEnabled(on)

    def _load(self, path: Path) -> None:
        self._player.setSource(QUrl.fromLocalFile(str(path)))

    def _switch_source(self, which: str) -> None:
        target = self._cleaned if which == "cleaned" else self._original
        if self._player is None or target is None:
            return
        self._resume_at = self._player.position()
        self._resume_playing = self._player.playbackState() == self._mm.QMediaPlayer.PlaybackState.PlayingState
        self._load(target)

    def _on_status(self, status) -> None:
        loaded = self._mm.QMediaPlayer.MediaStatus
        if status == loaded.LoadedMedia:
            self._set_controls(True)
            self._which.set_enabled_key("cleaned", self._cleaned is not None)
            if self._resume_at:
                self._player.setPosition(self._resume_at)
                self._resume_at = 0
            if self._resume_playing:
                self._resume_playing = False
                self._player.play()
        elif status == loaded.EndOfMedia:
            self._player.setPosition(0)
            self._player.pause()

    def _on_error(self, _error, message: str) -> None:
        log.warning("preview failed: %s", message)
        self._note.setText("This file can't be played in the preview.")
        self._stage.setCurrentWidget(self._note)
        self._set_controls(False)
        self._which.setEnabled(True)

    def _on_state(self, state) -> None:
        playing = state == self._mm.QMediaPlayer.PlaybackState.PlayingState
        self._play.set_icon("pause" if playing else "play")

    def _on_position(self, pos: int) -> None:
        if not self._scrubbing:
            self._slider.setValue(pos)
        self._show_position(pos)

    def _show_position(self, pos: int) -> None:
        total = self._player.duration() if self._player is not None else 0
        self._time.setText(f"{format_time(pos / 1000)} / {format_time(total / 1000)}")

    def _scrub_done(self) -> None:
        self._scrubbing = False
        if self._player is not None:
            self._player.setPosition(self._slider.value())

    def _close(self) -> None:
        self.stop()
        self.closed.emit()
        self.hide()
