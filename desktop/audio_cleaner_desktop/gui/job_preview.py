"""The job view's player: the preview pane (gui/player.py) with a second way to play, live (gui/live_audio.py).

Given the editor's live mix (live_mix.LiveMix), "Cleaned" plays the sound of the draft as it is now (it can be changed while it plays), and
"Original" plays the original through the same stream, at the same place and loudness, so switching between them is instant. Without a
stream it is the ordinary pane: the original against the saved result file.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QUrl

from ..live_mix import LiveMix
from . import live_audio
from .player import PreviewPane

log = logging.getLogger(__name__)


class JobPreview(PreviewPane):
    sink_factory = None  # how live sound reaches the sound card (live_audio.QtSink); tests give a stand-in
    _live: live_audio.LivePlayer | None = None  # (both stay None where QtMultimedia is not available)
    _stream: LiveMix | None = None

    def _build_player(self, mm) -> None:
        super()._build_player(mm)
        self._file = self._player
        self._live = live_audio.LivePlayer(mm, self, sink_factory=type(self).sink_factory)
        self._stream: LiveMix | None = None
        live = self._live
        for signal, slot in (
            (live.positionChanged, self._on_position), (live.durationChanged, lambda d: self._slider.setRange(0, d)),
            (live.playbackStateChanged, self._on_state), (live.mediaStatusChanged, self._on_status),
            (live.errorOccurred, self._on_error),
        ):
            signal.connect(lambda *args, slot=slot: slot(*args) if self._player is live else None)

    # -- public -------------------------------------------------------------

    @property
    def live(self) -> bool:
        """The sound being played is the live mix."""
        return self._player is not None and self._player is self._live and self._stream is not None

    def show_job(self, key: int, original: Path, cleaned: Path | None, has_video: bool, name: str,
                 subtitle_file: Path | None = None, stream: LiveMix | None = None) -> None:
        """Show a job: `cleaned` is its saved result (may be None); with `stream` the cleaned sound is the live mix."""
        self._leave_live()
        self.show_files(key, original, cleaned, has_video, name, subtitle_file)
        if stream is not None and self._player is not None:
            self._enter_live(stream)

    def use_stream(self, stream: LiveMix | None) -> None:
        """The stream became ready (or went away) while the job is shown."""
        if stream is None:
            self._leave_live()
        elif self._player is not None:
            self._enter_live(stream)

    def show_files(self, key, original, cleaned, has_video, name, subtitle_file=None) -> None:  # noqa: ANN001
        self._leave_live()
        super().show_files(key, original, cleaned, has_video, name, subtitle_file)

    # -- internals ----------------------------------------------------------

    def _enter_live(self, stream: LiveMix) -> None:
        was = self._which.value() if self._cleaned is not None else "cleaned"
        if self._player is self._live:
            self._live.blockSignals(True)
            self._live.stop()
            self._live.blockSignals(False)
        else:
            self._file.blockSignals(True)
            self._file.stop()
            self._file.setSource(QUrl())
            self._file.blockSignals(False)
        self._player = self._live
        self._stream = stream
        self._cleaned = self._cleaned or self._original
        self._resume_at, self._resume_playing = 0, False
        self._which.set_enabled_key("cleaned", True)
        self._which.blockSignals(True)
        self._which.set_value(was)
        self._which.blockSignals(False)
        stream.set_original(was == "original")
        self._live.load(stream)

    def _leave_live(self) -> None:
        if self._live is None or self._player is not self._live:
            return
        self._live.blockSignals(True)
        self._live.unload()
        self._live.blockSignals(False)
        self._player = self._file
        self._stream = None

    def _switch_source(self, which: str) -> None:
        if self.live:
            self._stream.set_original(which == "original")
            return
        super()._switch_source(which)

    def _on_error(self, _error, message: str) -> None:
        if self._live is not None and self._player is self._live:
            log.warning("live listening failed: %s", message)
            self._note.setText("Live listening is not available on this computer's sound output, so only the saved files can be "
                               "played. Changing and saving still works.")
            self._stage.setCurrentWidget(self._note)
            self._set_controls(False)
            self._which.setEnabled(True)
            return
        super()._on_error(_error, message)
