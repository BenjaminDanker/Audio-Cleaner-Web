"""Stand-ins for QtMultimedia (which needs system sound libraries the test machines do not all have): a media player that only
remembers what it was told, and a sound card that pulls the live sound the way QAudioSink does."""
from __future__ import annotations

import enum
from types import SimpleNamespace

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QWidget


class PlaybackState(enum.Enum):
    StoppedState = 0
    PlayingState = 1
    PausedState = 2


class MediaStatus(enum.Enum):
    NoMedia = 0
    LoadedMedia = 1
    EndOfMedia = 2


class Error(enum.Enum):
    NoError = 0
    ResourceError = 1


class FakePlayer(QObject):
    durationChanged = Signal(int)
    positionChanged = Signal(int)
    playbackStateChanged = Signal(object)
    mediaStatusChanged = Signal(object)
    errorOccurred = Signal(object, str)
    PlaybackState = PlaybackState
    MediaStatus = MediaStatus
    Error = Error

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source = None
        self.state = PlaybackState.StoppedState
        self.pos = 0

    def setAudioOutput(self, _):  # noqa: N802
        pass

    def setVideoOutput(self, _):  # noqa: N802
        pass

    def setActiveVideoTrack(self, _):  # noqa: N802
        pass

    def setSource(self, url):  # noqa: N802
        self.source = url.toLocalFile() if url.isValid() else None
        if self.source:
            self.durationChanged.emit(10_000)
            self.mediaStatusChanged.emit(MediaStatus.LoadedMedia)

    def play(self):
        self.state = PlaybackState.PlayingState
        self.playbackStateChanged.emit(self.state)

    def pause(self):
        self.state = PlaybackState.PausedState
        self.playbackStateChanged.emit(self.state)

    def stop(self):
        self.state = PlaybackState.StoppedState
        self.playbackStateChanged.emit(self.state)
        self.positionChanged.emit(0)

    def position(self):
        return self.pos

    def duration(self):
        return 10_000

    def setPosition(self, ms):  # noqa: N802
        self.pos = ms

    def playbackState(self):  # noqa: N802
        return self.state


class FakeOutput(QObject):
    pass


class FakeVideoWidget(QWidget):
    pass


def namespace() -> SimpleNamespace:
    return SimpleNamespace(QAudioOutput=FakeOutput, QMediaPlayer=FakePlayer, QVideoWidget=FakeVideoWidget)


class FakeSink:
    """A sound card: `pull(frames)` is the card asking for sound; it reads the device like QAudioSink in pull mode."""

    made: list["FakeSink"] = []

    def __init__(self, rate, channels, device):
        self.rate, self.channels, self.device = rate, channels, device
        self.started = self.stopped = False
        self.frames_played = 0
        self.heard: list[bytes] = []
        FakeSink.made.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def processed_ms(self):
        return self.frames_played * 1000.0 / self.rate

    def pull(self, frames: int) -> bytes:
        data = bytes(self.device.read(frames * 2 * self.channels))
        self.frames_played += len(data) // (2 * self.channels)
        self.heard.append(data)
        return data


def sink_factory(rate, channels, device) -> FakeSink:
    return FakeSink(rate, channels, device)
