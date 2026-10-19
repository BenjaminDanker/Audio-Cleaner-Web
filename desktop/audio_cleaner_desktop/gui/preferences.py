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
    # On: each file gets the noise reduction strength and the kind of noise removal that fit it and its voice rebuilt when it
    # needs it (measured once, when it is cleaned: the trained model of
    # acw_common.reco_model, the website's "Automatic best settings"); the choices below are kept for when it is off.
    auto_settings = _Setting(True)
    strength = _Setting(engine.DEFAULT_STRENGTH, engine.STRENGTHS)
    leveling = _Setting(True)
    loudness = _Setting(engine.DEFAULT_LOUDNESS, engine.LOUDNESS_TARGETS)
    subtitles = _Setting("off", engine.SUBTITLE_MODES)
    language = _Setting("auto", ("auto", *engine.LANGUAGES))
    noise = _Setting("auto", ("auto", "light", "heavy"))
    echo_track = _Setting(False)
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
        auto = self.auto_settings
        return engine.JobOptions(
            strength=engine.AUTO_STRENGTH if auto else self.strength, leveling=self.leveling, audio_format=self.audio_format,
            loudness=self.loudness, subtitles=self.subtitles, language=self.language, noise="auto" if auto else self.noise,
            echo_track=self.echo_track,
            rebuild_voice=False if auto else self.rebuild_voice, auto_rebuild=auto, video_size=self.video_size,
        )

    def output_dir(self) -> Path | None:
        """Where results go; None means next to each original."""
        if self.output_mode == "folder" and self.output_folder:
            return Path(self.output_folder)
        return None
