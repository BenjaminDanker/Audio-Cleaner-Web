"""The settings the user chooses, persisted with QSettings."""
from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings

from .. import engine


class _Setting:
    """One persisted value of a fixed type; unknown stored values fall back to the default."""

    def __init__(self, default: Any, allowed: Collection | None = None):
        self._default = default
        self._allowed = allowed
        self._key = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self._key = name

    def __get__(self, obj: "Preferences | None", owner: type | None = None) -> Any:
        if obj is None:
            return self
        raw = obj._store.value(self._key, self._default)
        try:
            if isinstance(self._default, bool):
                value = raw in (True, "true", "1", 1)
            else:
                value = type(self._default)(raw)
        except (TypeError, ValueError):
            return self._default
        return value if self._allowed is None or value in self._allowed else self._default

    def __set__(self, obj: "Preferences", value: Any) -> None:
        obj._store.setValue(self._key, value)


class Preferences:
    strength = _Setting(engine.DEFAULT_STRENGTH, engine.STRENGTHS)
    polish = _Setting(True)
    loudness = _Setting(engine.DEFAULT_LOUDNESS, engine.LOUDNESS_TARGETS)
    subtitles = _Setting("off", engine.SUBTITLE_MODES)
    language = _Setting("auto", ("auto", *engine.LANGUAGES))
    noise = _Setting("auto", ("auto", "light", "heavy"))
    echo_track = _Setting(False)
    restore_highs = _Setting(True)
    rebuild_voice = _Setting(False)
    audio_format = _Setting("same", engine.AUDIO_CHOICES)
    video_size = _Setting("same", engine.VIDEO_SIZES)  # picture size of video results: the shorter side in pixels
    parallel = _Setting(0, range(5))  # files at once; 0 = Auto
    output_mode = _Setting("next", ("next", "folder"))
    output_folder = _Setting("")
    last_input_folder = _Setting("")
    check_updates = _Setting(True)  # look for a newer version when the app starts (updater.py)
    update_last_check = _Setting(0.0)  # time.time() of the last check that reached the update server
    update_install_tried = _Setting("")  # the version whose installer was started when the app last closed

    def __init__(self, store: QSettings | None = None):
        self._store = store if store is not None else QSettings()

    def job_options(self) -> engine.JobOptions:
        return engine.JobOptions(
            strength=self.strength, polish=self.polish, audio_format=self.audio_format, loudness=self.loudness,
            subtitles=self.subtitles, language=self.language, noise=self.noise, echo_track=self.echo_track,
            restore_highs=self.restore_highs, rebuild_voice=self.rebuild_voice, video_size=self.video_size,
        )

    def output_dir(self) -> Path | None:
        """Where results go; None means next to each original."""
        if self.output_mode == "folder" and self.output_folder:
            return Path(self.output_folder)
        return None
