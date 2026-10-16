"""Preview: play a file's original and cleaned audio (and video) and switch between them.

QtMultimedia is imported lazily. Where it cannot load (missing system audio libraries,
no sound device), the pane says so and the rest of the app keeps working.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from types import SimpleNamespace

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QStackedWidget, QVBoxLayout, QWidget

from . import icons, subtitles, video_preview
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


_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".flv", ".avi"}

class PreviewPane(QFrame):
    closed = Signal()
    _subtitles_read = Signal(object, object)  # key, cues (read on a worker thread)

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
        # How the picture is shown: "none" (audio file), "own" (our FFmpeg, timed to the sound: video_preview) or
        # "qt" (Qt's player, for video formats our FFmpeg does not decode).
        self._mode = "none"
        self._synced: video_preview.SyncedVideo | None = None
        self._video = None

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
        self._stage.setMinimumHeight(195)
        self._art = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self._note = QLabel("Preview unavailable", alignment=Qt.AlignmentFlag.AlignCenter)
        self._note.setWordWrap(True)
        self._stage.addWidget(self._art)
        self._stage.addWidget(self._note)
        # Subtitles over the bottom of the picture (or the waveform, for audio), timed to the sound.
        self._subs = subtitles.Track([])
        self._caption = QLabel(self._stage)
        self._caption.setWordWrap(True)
        self._caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._caption.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._caption.setStyleSheet(
            "QLabel { color: white; background: rgba(0, 0, 0, 170); border-radius: 4px; padding: 3px 8px; font-size: 13px; }"
        )
        self._caption.hide()
        self._stage.installEventFilter(self)
        self._caption_timer = QTimer(self, interval=100, timeout=self._update_caption)
        self._subtitles_read.connect(self._on_subtitles_read)
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
        self._cc = QPushButton("CC")
        self._cc.setCheckable(True)
        self._cc.setChecked(True)
        self._cc.setToolTip("Show the subtitles")
        self._cc.setFixedWidth(42)
        self._cc.toggled.connect(lambda _: self._update_caption())
        self._cc.hide()
        controls.addWidget(self._cc)
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

    def show_files(self, key: int, original: Path, cleaned: Path | None, has_video: bool, name: str,
                   subtitle_file: Path | None = None) -> None:
        """`subtitle_file`: the .srt/.vtt made with the result; without one, a subtitle track inside the cleaned video
        is used when there is one."""
        self.stop()
        self.key = key
        self._set_subtitles([])
        if subtitle_file is not None and subtitle_file.suffix.lower() in (".srt", ".vtt"):
            self._set_subtitles(subtitles.read_file(subtitle_file))
        elif cleaned is not None and cleaned.suffix.lower() in _VIDEO_SUFFIXES:  # a video made from a sound has them too
            self._read_embedded_subtitles(key, cleaned)
        self._original, self._cleaned = original, cleaned
        self._name.setText(name)
        self._which.set_enabled_key("cleaned", cleaned is not None)
        self._which.blockSignals(True)
        self._which.set_value("cleaned" if cleaned else "original")
        self._which.blockSignals(False)
        if self._player is not None:
            self._resume_at, self._resume_playing = 0, False
            self._show_picture(original if has_video else None)
            self._load(cleaned or original)

    def set_cleaned(self, path: Path) -> None:
        self._cleaned = path
        self._which.set_enabled_key("cleaned", True)

    def stop(self) -> None:
        if self._player is not None:
            self._player.stop()
            if self._synced is not None:
                self._synced.close()

    def release(self) -> None:
        """Stop and close the file, so it can be moved or renamed (Windows locks files that are open)."""
        if self._player is not None:
            if self._synced is not None:
                self._synced.close()
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
        self._view = video_preview.VideoView()
        self._stage.insertWidget(0, self._view)
        self._player.durationChanged.connect(lambda d: self._slider.setRange(0, d))
        self._player.positionChanged.connect(self._on_position)
        self._player.playbackStateChanged.connect(self._on_state)
        self._player.mediaStatusChanged.connect(self._on_status)
        self._player.errorOccurred.connect(self._on_error)

    def _show_picture(self, video: Path | None) -> None:
        """Choose how this file's picture is shown (see _mode). `video`: the file whose picture to show, or None."""
        if self._synced is not None:
            self._synced.close()
        self._player.setVideoOutput(None)
        if video is None:
            self._mode = "none"
            self._stage.setCurrentWidget(self._art)
            return
        info = None
        try:
            from .. import resources

            ff = resources.ffmpeg_dir()
            info = video_preview.probe(ff / resources._FFPROBE, video)
        except Exception:  # noqa: BLE001 - no FFmpeg next to the app: Qt's player shows the picture
            log.warning("preview: could not probe %s", video, exc_info=True)
        if info is not None and info.codec in video_preview.DECODABLE:
            self._mode = "own"
            if self._synced is None:
                self._synced = video_preview.SyncedVideo(
                    self._view, lambda: self._player.position() / 1000, ff / resources._FFMPEG, self)
            self._view.show_image(None)
            self._stage.setCurrentWidget(self._view)
            self._synced.open(video, info, 0.0)
            return
        self._mode = "qt"
        if self._video is None:
            self._video = self._mm.QVideoWidget()
            self._stage.insertWidget(0, self._video)
        self._player.setVideoOutput(self._video)
        self._stage.setCurrentWidget(self._video)

    # -- subtitles ----------------------------------------------------------

    def _read_embedded_subtitles(self, key: int, video: Path) -> None:
        try:
            from .. import resources

            ffmpeg = resources.ffmpeg_dir() / resources._FFMPEG
        except Exception:  # noqa: BLE001
            return
        threading.Thread(
            target=lambda: self._subtitles_read.emit(key, subtitles.read_embedded(ffmpeg, video)),
            name="preview-subtitles", daemon=True,
        ).start()

    def _on_subtitles_read(self, key, cues) -> None:
        if key == self.key and cues:
            self._set_subtitles(cues)

    def _set_subtitles(self, cues: list) -> None:
        self._subs = subtitles.Track(cues)
        self._cc.setVisible(bool(self._subs))
        if self._subs:
            self._caption_timer.start()
        else:
            self._caption_timer.stop()
        self._update_caption()

    def _update_caption(self) -> None:
        text = ""
        if self._subs and self._cc.isChecked() and self._player is not None:
            text = self._subs.at(self._player.position() / 1000)
        if text != self._caption.text():
            self._caption.setText(text)
            self._place_caption()
        self._caption.setVisible(bool(text))
        if text:
            self._caption.raise_()

    def _place_caption(self) -> None:
        area = self._stage.rect()
        width = min(area.width() - 24, max(120, int(area.width() * 0.9)))
        self._caption.setFixedWidth(width)
        height = self._caption.heightForWidth(width)
        self._caption.setFixedHeight(height if height > 0 else self._caption.sizeHint().height())
        self._caption.move((area.width() - width) // 2, area.height() - self._caption.height() - 10)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if obj is self._stage and event.type() == QEvent.Type.Resize:
            self._place_caption()
        return super().eventFilter(obj, event)

    def _seek_picture(self, ms: int) -> None:
        if self._mode == "own" and self._synced is not None:
            self._synced.seek(ms / 1000)

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
            if self._mode == "own" and hasattr(self._player, "setActiveVideoTrack"):
                self._player.setActiveVideoTrack(-1)  # the sound player never decodes the picture
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
            self._seek_picture(0)

    def _on_error(self, _error, message: str) -> None:
        log.warning("preview failed: %s", message)
        self._note.setText("This file can't be played in the preview.")
        self._stage.setCurrentWidget(self._note)
        self._set_controls(False)
        self._which.setEnabled(True)

    def _on_state(self, state) -> None:
        playing = state == self._mm.QMediaPlayer.PlaybackState.PlayingState
        self._play.set_icon("pause" if playing else "play")
        if self._synced is not None:
            self._synced.set_playing(playing)

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
            self._seek_picture(self._slider.value())

    def _close(self) -> None:
        self.stop()
        self.closed.emit()
        self.hide()
