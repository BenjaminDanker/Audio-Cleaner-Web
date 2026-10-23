"""Live listening: the player plays the editor's live mix (live_mix.LiveMix, which can change at any moment) straight to the sound card
in small pieces, so a change in the editor is heard within a fraction of a second.

`LivePlayer` has the part of QMediaPlayer's interface that the preview pane uses (position, duration, play, pause, stop, signals), so the pane
works with either. The sound card is reached through a small adapter (`QtSink`, QAudioSink); tests give a stand-in that pulls the
sound the same way (nothing else of this module needs QtMultimedia).
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from types import SimpleNamespace

from PySide6.QtCore import QIODevice, QObject, QTimer, Signal

from acw_common.pipeline import SAMPLE_RATE

from ..live_mix import LiveMix

log = logging.getLogger(__name__)

BUFFER_SECONDS = 0.1  # sound held ready for the card: the delay between a change and hearing it
TICK_MS = 50


class PullDevice(QIODevice):
    """The sound card pulls from this: 16-bit samples of the stream, from a frame on, mixed at the strength of that moment."""

    def __init__(self, stream: LiveMix, start_frame: int, parent: QObject | None = None):
        super().__init__(parent)
        self.stream = stream
        self.frame = int(start_frame)  # the next frame to hand out
        # Unbuffered: Qt must not read ahead, or a change of strength would be heard only after what it had read ahead.
        self.open(QIODevice.OpenModeFlag.ReadOnly | QIODevice.OpenModeFlag.Unbuffered)

    @property
    def frame_bytes(self) -> int:
        return 2 * self.stream.channels

    def readData(self, maxlen: int) -> bytes:  # noqa: N802 - Qt override
        frames = max(0, int(maxlen)) // self.frame_bytes
        data = self.stream.read_pcm16(self.frame, frames)
        self.frame += len(data) // self.frame_bytes
        return data

    def writeData(self, data) -> int:  # noqa: N802
        return 0

    def bytesAvailable(self) -> int:  # noqa: N802
        return max(0, self.stream.frames - self.frame) * self.frame_bytes + super().bytesAvailable()

    def isSequential(self) -> bool:  # noqa: N802
        return True


class QtSink:
    """QAudioSink on the default sound output, in pull mode."""

    def __init__(self, rate: int, channels: int, device: QIODevice):
        from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices

        fmt = QAudioFormat()
        fmt.setSampleRate(rate)
        fmt.setChannelCount(channels)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        out = QMediaDevices.defaultAudioOutput()
        if out.isNull() or not out.isFormatSupported(fmt):
            raise RuntimeError("This computer's sound output can't play this live.")
        self._sink = QAudioSink(out, fmt)
        self._sink.setBufferSize(int(BUFFER_SECONDS * rate) * channels * 2)
        self._device = device

    def start(self) -> None:
        self._sink.start(self._device)

    def stop(self) -> None:
        self._sink.stop()

    def processed_ms(self) -> float:
        return self._sink.processedUSecs() / 1000.0


SinkFactory = Callable[[int, int, QIODevice], object]


class LivePlayer(QObject):
    positionChanged = Signal(int)
    durationChanged = Signal(int)
    playbackStateChanged = Signal(object)
    mediaStatusChanged = Signal(object)
    errorOccurred = Signal(object, str)

    def __init__(self, mm: SimpleNamespace, parent: QObject | None = None, sink_factory: SinkFactory | None = None):
        super().__init__(parent)
        self._state = mm.QMediaPlayer.PlaybackState
        self._status = mm.QMediaPlayer.MediaStatus
        self._error = getattr(mm.QMediaPlayer, "Error", None)
        self._factory = sink_factory or QtSink
        self._stream: LiveMix | None = None
        self._sink = None
        self._device: PullDevice | None = None
        self._base = 0  # frame the sound card was last started at
        self._current = self._state.StoppedState
        self._timer = QTimer(self, interval=TICK_MS, timeout=self._tick)

    # -- what QMediaPlayer has that the pane uses -----------------------------------------------------------------

    def load(self, stream: LiveMix) -> None:
        self._halt()
        self._stream, self._base = stream, 0
        self._set_state(self._state.StoppedState)
        self.durationChanged.emit(int(stream.seconds * 1000))
        self.mediaStatusChanged.emit(self._status.LoadedMedia)

    def unload(self) -> None:
        self._halt()
        self._stream, self._base = None, 0
        self._set_state(self._state.StoppedState)
        self.durationChanged.emit(0)
        self.mediaStatusChanged.emit(self._status.NoMedia)

    def duration(self) -> int:
        return int(self._stream.seconds * 1000) if self._stream is not None else 0

    def position(self) -> int:
        return int(self._frame_now() * 1000 / SAMPLE_RATE)

    def playbackState(self):  # noqa: N802
        return self._current

    def setPosition(self, ms: int) -> None:  # noqa: N802
        if self._stream is None:
            return
        frame = min(max(0, int(ms * SAMPLE_RATE / 1000)), self._stream.frames)
        playing = self._current == self._state.PlayingState
        if playing:
            self._halt()
        self._base = frame
        if playing:
            self._begin()
        self.positionChanged.emit(self.position())

    def play(self) -> None:
        if self._stream is None or self._current == self._state.PlayingState:
            return
        if self._base >= self._stream.frames:
            self._base = 0
        self._begin()

    def pause(self) -> None:
        if self._current == self._state.PlayingState:
            self._base = self._frame_now()
            self._halt()
        if self._stream is not None:
            self._set_state(self._state.PausedState)

    def stop(self) -> None:
        self._halt()
        self._base = 0
        self._set_state(self._state.StoppedState)
        self.positionChanged.emit(0)

    def setSource(self, _url) -> None:  # noqa: N802 - the pane clears the source to let go of files; this player holds none
        pass

    def setVideoOutput(self, _widget) -> None:  # noqa: N802
        pass

    # -- internals ---------------------------------------------------------------------------------------

    def _frame_now(self) -> int:
        if self._stream is None:
            return 0
        if self._sink is not None and self._current == self._state.PlayingState:
            return min(self._stream.frames, self._base + int(self._sink.processed_ms() * SAMPLE_RATE / 1000))
        return min(self._base, self._stream.frames)

    def _begin(self) -> None:
        stream = self._stream
        try:
            self._device = PullDevice(stream, self._base, self)
            self._sink = self._factory(SAMPLE_RATE, stream.channels, self._device)
            self._sink.start()
        except Exception as e:  # noqa: BLE001 - no sound output, a format it does not take: the pane says so
            log.warning("live listening could not start: %s", e, exc_info=True)
            self._halt()
            self.errorOccurred.emit(self._error.ResourceError if self._error is not None else 0, str(e))
            return
        self._set_state(self._state.PlayingState)
        self._timer.start()

    def _halt(self) -> None:
        self._timer.stop()
        sink, device, self._sink, self._device = self._sink, self._device, None, None
        if sink is not None:
            try:
                sink.stop()
            except Exception:  # noqa: BLE001
                log.debug("stopping the sound output failed", exc_info=True)
        if device is not None:
            device.close()
            device.deleteLater()

    def _set_state(self, state) -> None:
        if state != self._current:
            self._current = state
            self.playbackStateChanged.emit(state)

    def _tick(self) -> None:
        if self._stream is None or self._current != self._state.PlayingState:
            return
        frame = self._frame_now()
        self.positionChanged.emit(int(frame * 1000 / SAMPLE_RATE))
        if frame >= self._stream.frames:
            self._halt()
            self._base = self._stream.frames
            self._set_state(self._state.StoppedState)
            self.mediaStatusChanged.emit(self._status.EndOfMedia)
