"""What the editor screen shows for one file: a file that has not been started, one that is running, or a finished job. No widgets in here.

A Doc is made at once from what is already known (the file's name and, for a job, the facts and the waveform it kept) and fills itself in on
background threads, each reporting as it finishes, so the screen never waits for any of it:

    reading the file    its facts (length, channels, tracks), then the waveform (quick_look.quick_peaks: seconds, not minutes, for an hour)
    Automatic           "listening to a little of the file" for the best settings (quick_look.decide: about a minute of it)
    moments worth a look  the suggestions (a long recording only when asked: it needs all of it decoded)
    the live sound      for a job: the recording decoded and laid out so changes can be heard while it plays (editor_session.Session.open)

Playback of the file itself and editing the parts do not wait for the live sound.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from acw_common.timeline import LAYERS, Timeline

from .. import copies, editor_job, editor_session, engine, labels, quick_look, redo, resources, speed
from ..editor_model import STRENGTH_DB, EditModel
from .history import Entry
from .items import FileItem
from .preferences import Preferences
from .workers import Worker


def say_clock(seconds: float) -> str:
    s = int(round(max(0.0, seconds)))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"

log = logging.getLogger(__name__)


# (looked up when called, so a test that replaces engine.probe or the like is heard)
def _probe(path, ff=None):
    return engine.probe(path, ff)


def _peaks(*a, **kw):
    return quick_look.quick_peaks(*a, **kw)


def _decide(*a, **kw):
    return quick_look.decide(*a, **kw)


def _open_session(*a, **kw):
    return editor_session.Session.open(*a, **kw)


def _open_plan(*a, **kw):
    return editor_session.PlanSession.open(*a, **kw)


_PROBES = threading.Semaphore(4)  # files added by the dozen are read a few at a time


@dataclass
class DocEnv:
    """What a Doc asks of the rest of the app (tests replace these)."""

    prefs: Preferences
    scratch: Callable[[], Path | None] = lambda: None
    voice_ready: Callable[[], bool] = lambda: resources.voice_model_dir() is not None
    speech_ready: Callable[[], bool] = lambda: resources.speech_ready()
    noise_ready: Callable[[], bool] = lambda: resources.noise_ready()
    probe: Callable = _probe
    peaks: Callable = _peaks
    decide: Callable = _decide
    open_session: Callable = _open_session  # a job's live sound
    open_plan: Callable = _open_plan  # a file that has not run: all of it decoded, for the suggestions
    auto_suggest: bool = True  # look for moments worth a look by itself (a short recording, Automatic on)
    threads: int = 2


@dataclass
class Facts:
    """What the screen says about the file (from the probe, or from what a job kept)."""

    duration: float = 0.0
    has_video: bool = False
    channels: int = 2
    tracks: int = 1
    ext: str = ""
    video_codec: str | None = None
    width: int = 0
    height: int = 0
    size: int = 0

    @classmethod
    def of(cls, info, path: Path) -> "Facts":
        try:
            size = Path(path).stat().st_size
        except OSError:
            size = 0
        return cls(float(info.duration), bool(info.has_video), int(info.audio_channels), int(getattr(info, "audio_tracks", 1) or 1),
                   getattr(info, "ext", "") or Path(path).suffix.lower(),
                   getattr(info, "video_codec", None), int(getattr(info, "video_width", 0) or 0), int(getattr(info, "video_height", 0) or 0), size)

    def source(self) -> engine.Source:
        return engine.Source(self.has_video, self.ext, self.video_codec, self.duration, self.channels)


@dataclass(frozen=True)
class FileSettings:
    """The choices of one file that another can be given ("Use these settings for other files"): about the file as a whole, since the
    parts of a recording are about its own moments. `layers` None: Automatic, which each file decides for itself; else per layer what most
    of the file has (None: off). `ending`: the type of the result when another than the usual one was picked."""

    layers: dict | None
    subtitles: str
    language: str
    video_size: str
    ending: str | None = None
    had_parts: bool = False
    # (Automatic only: what it does besides what it picks, from the remembered choices; for the words)
    auto_leveling: str | None = None  # the loudness, None: no leveling
    auto_echo: bool = False

    def describe(self, has_video: bool) -> str:
        """In words, as the job list says it ("Strong noise reduction, leveled (podcast), subtitles: ...")."""
        s: dict = {"subtitles": self.subtitles, "video_size": self.video_size}
        if self.layers is None:
            s.update(strength=engine.AUTO_STRENGTH, leveling=self.auto_leveling is not None, loudness=self.auto_leveling,
                     echo_track=self.auto_echo)
        else:
            noise, level = self.layers.get("noise"), self.layers.get("leveling")
            s["rebuild_voice"] = self.layers.get("rebuild") is not None
            if noise is not None:
                s["strength_db"] = noise.get("strength_db")
                s["noise"] = noise.get("remover", "auto")
            elif not s["rebuild_voice"]:
                s["strength_db"] = 0
            s["echo_track"] = self.layers.get("echo") is not None
            s["leveling"] = level is not None
            if level is not None:
                s["loudness"] = editor_job.loudness_key(level.get("loudness_lufs"))
        text = labels.describe_settings(s, has_video)
        if self.layers is not None and self.layers.get("noise") is None and self.layers.get("rebuild") is None:
            text = ", ".join(["no noise reduction", *text.split(", ")[1:]])  # (the first piece says the noise reduction)
        if self.ending:
            text += f", saved as {self.ending}"
        return text


def _most_of(tl: Timeline, layer: str) -> dict | None:
    """The settings that most of the recording has in this layer (None: most of it has the layer off)."""
    time: dict[str, float] = {}
    settings: dict[str, dict] = {}
    for part in tl.layers[layer]:
        key = repr(sorted((part.settings or {}).items()))
        time[key] = time.get(key, 0.0) + part.seconds
        settings[key] = dict(part.settings or {})
    if not time or sum(time.values()) < tl.seconds / 2:
        return None
    return settings[max(time, key=time.get)]


def facts_of_job(job: copies.JobCopies, path: Path) -> Facts:
    m = job.media
    try:
        size = Path(path).stat().st_size
    except OSError:
        size = 0
    return Facts(float(m.get("duration") or editor_job.seconds_of(job)), bool(m.get("has_video")), int(m.get("channels") or 2),
                 int(m.get("audio_tracks") or 1), m.get("ext") or Path(path).suffix.lower(), m.get("video_codec"), 0, 0, size)


class Doc(QObject):
    changed = Signal()  # something the screen shows is different (facts, the waveform, Automatic's line, the live sound, a note)
    modelChanged = Signal()  # the draft, the selection or the suggestions were edited
    peaksPartial = Signal(object, float)  # (waveform so far, fraction): from the reading thread
    sessionReady = Signal(object)  # the live sound of a job is ready
    unreadable = Signal(str)  # the file could not be read
    factsReady = Signal()  # the file was read: its length, its channels, its name for the result
    progressed = Signal()  # how far a run or a save is, or the live sound (only the bar and the line change)

    def __init__(self, env: DocEnv, path: Path | str, *, item: FileItem | None = None, parent: QObject | None = None):
        super().__init__(parent)
        self.env = env
        self.path = Path(path)
        self.name = self.path.name
        self.item = item
        self.entry: Entry | None = None
        self.job: copies.JobCopies | None = None
        self.original = copies.OriginalCheck("ok")
        self.facts: Facts | None = None
        self.error = ""  # the file could not be read
        self.note = ""  # a line the screen shows above the editor (a failure, a saved result)
        self.note_kind = "info"
        self.model: EditModel | None = None
        self.seconds = 0.0
        self.peaks: list[int] | None = None
        self.heard: list[int] | None = None  # the waveform of the saved result
        self.wave = "none"  # "none" | "reading" | "ready" | "failed"
        self.wave_fraction = 0.0
        self.cuts: list[float] = []  # where the parts were split (the rows show a boundary there)
        # Automatic
        self.auto = env.prefs.auto_settings
        self.auto_base: Timeline | None = None  # what Automatic made of the file
        self.choices = None  # engine's decision (reco_model.Choices)
        self.listening = False
        self.applied = False
        # the suggestions
        self.sg = "idle"  # "idle" | "running" | "canceled" | "error" | "done"
        self.sg_fraction = 0.0
        self.sg_note = ""
        # the choices that are about the file as a whole
        self.subtitles = env.prefs.subtitles
        self.language = env.prefs.language
        self.video_size = env.prefs.video_size
        self.out_name = ""
        self.out_custom = False
        self._given: FileSettings | None = None  # settings given from another file before this one was read: they are used once it is
        self._taking = False
        # the live sound of a job
        self.session: editor_session.Session | None = None
        self.session_state = "none"  # "none" | "opening" | "ready" | "failed"
        self.session_fraction = 0.0
        self.session_note = ""
        self.snap: tuple | None = None  # the output choices of the latest result (to tell whether something changed)
        self.stage = "ready"  # "ready" (not run) | "processing" (a run or a save goes) | "done" (a result)
        self.saving = False  # the run that goes is a save of changes to a finished job
        self.run_fraction: float | None = None
        self.run_text = ""
        self._workers: dict[str, Worker] = {}
        self._parked: list[Worker] = []  # stopped, not yet ended
        self._gen: dict[str, int] = {}
        self._force_apply = False
        self._find_when_ready = False
        self._info = None
        self.active = False  # the file in the editor: the others are only read (facts), the rest waits until they are opened
        self._after_started = False
        self._disposed = False
        self.peaksPartial.connect(self._on_peaks_partial)

    # ------------------------------------------------------------------------------------------------ what is known
    @property
    def ready(self) -> bool:
        return self.model is not None

    @property
    def is_job(self) -> bool:
        return self.entry is not None

    @property
    def has_audio(self) -> bool:
        return self.model is not None and self.seconds > 0

    @property
    def echo_available(self) -> bool:
        return bool(self.model and self.model.echo_available)

    @property
    def readonly(self) -> bool:
        """A job without saved copies: it can be listened to, not changed."""
        return self.entry is not None and self.job is None

    @property
    def locked(self) -> bool:
        """The choices cannot be changed: a run goes, or there are no saved copies to change."""
        return self.stage == "processing" or self.readonly

    @property
    def show_suggestions(self) -> bool:
        """The moments worth a look are shown (pins, the card for a part, the count): with Automatic on, and for a finished job."""
        return bool((self.auto or self.is_job) and not self.readonly)

    @property
    def can_offer_find(self) -> bool:
        """Moments worth a look are not looked for by themselves (a long recording, or switched off): there is a button for it."""
        return bool(self.show_suggestions and self.model is not None and self.sg == "idle" and not self.listening and not self.locked and not self.model.suggestions
                    and (self.seconds > editor_job.SUGGEST_AUTO_SECONDS or not self.env.auto_suggest))

    @property
    def kind(self) -> str:
        return "Video" if self.facts is not None and self.facts.has_video else "Audio"

    def options_for(self, *, auto: bool) -> engine.JobOptions:
        """The whole-file choices that are not about parts (what the engine's options say besides the sound)."""
        p = self.env.prefs
        base = p.job_options()
        return engine.JobOptions(
            strength=engine.AUTO_STRENGTH if auto else (base.strength if base.strength != engine.AUTO_STRENGTH else engine.DEFAULT_STRENGTH),
            leveling=p.leveling, audio_format=p.audio_format, loudness=p.loudness, subtitles=self.subtitles, language=self.language,
            noise="auto" if auto else (p.noise if p.noise else "auto"), echo_track=p.echo_track,
            rebuild_voice=False if auto else p.rebuild_voice, auto_rebuild=auto, video_size=self.video_size)

    # ------------------------------------------------------------------------------------------------ a file that has not run
    def open_file(self) -> None:
        """Read the file in the background: its facts first, then everything that depends on them."""
        def read(cancel: threading.Event, progress):
            with _PROBES:
                return self.env.probe(self.path)

        self._start("probe", read, self._probed, self._probe_failed)

    def _probe_failed(self, message: str) -> None:
        self.error = message or "This file could not be read."
        self.changed.emit()
        self.unreadable.emit(self.error)

    def _probed(self, info) -> None:
        facts = Facts.of(info, self.path)
        self.set_facts(facts, info)
        opts = self.options_for(auto=self.auto)
        self.seconds = facts.duration
        echo = opts.echo_track and facts.tracks >= 2
        base = editor_session.baseline_for(self.options_for(auto=False) if self.auto else opts, self.seconds, echo=echo)
        self._new_model(base, echo_available=facts.tracks >= 2, loudness=editor_job.timeline_loudness(base))
        if self.out_name == "" and self.item is not None:
            self.out_name = self.item.out_name
        item = self.item
        if item is not None:
            item.duration, item.has_video, item.channels, item.probed = facts.duration, facts.has_video, facts.channels, True
            item.source = facts.source()
            if not item.out_custom:
                item.out_name = engine.default_output_name(self.path, item.source, self.env.prefs.job_options())
            self.out_name, self.out_custom = item.out_name, item.out_custom
        if self._given is not None:
            given, self._given = self._given, None
            self.take_settings(given)
        self.changed.emit()
        self.factsReady.emit()
        if self.active:
            self._after_probe()

    def _after_probe(self) -> None:
        """What follows once the file is known and is the one in the editor: its waveform, then what Automatic picks."""
        if self._after_started or self._info is None or self.is_job:
            return
        self._after_started = True
        self._read_wave(self._info)
        if self.auto and not self.applied:
            self._listen(self._info)
        else:
            self.maybe_suggest()

    def activate(self) -> None:
        """This is the file in the editor now."""
        self.active = True
        if self.model is not None and not self.is_job:
            self._after_probe()

    def release(self) -> None:
        """Another file is in the editor: what runs for this one (and the live sound, with its scratch files) is let go; it starts again when it is
        opened."""
        self.active = False
        for name in ("find", "listen", "wave", "heard", "session"):
            self.stop(name)
        if self.listening or self.wave == "reading":
            self._after_started = False
        if self.listening:
            self.listening = False
        if self.wave == "reading":
            self.wave, self.peaks = "none", None
        if self.sg == "running":
            self.sg = "idle"
        if self.session_state == "opening":
            self.session_state = "none"
        self._wait_parked()
        session, self.session = self.session, None
        if session is not None:
            session.close()
            self.session_state = "none"

    def set_facts(self, facts: Facts, info=None) -> None:
        self.facts = facts
        self._info = info

    def _new_model(self, draft: Timeline, *, echo_available: bool, loudness, done: Timeline | None = None) -> EditModel:
        m = EditModel(self.seconds, draft, done, echo_available=echo_available, loudness=loudness)
        p = self.env.prefs
        m.default_noise = {"strength_db": STRENGTH_DB.get(p.strength, STRENGTH_DB["max"]), "remover": p.noise or "auto"}
        m.on_change(self._model_changed)
        self.model = m
        return m

    # -- the waveform ------------------------------------------------------------------------------------------------------
    def _read_wave(self, info=None) -> None:
        """The waveform of the file, from a quick read of its sound (shown as it fills)."""
        if self.wave == "reading":
            return
        info = info or self._info
        self.wave, self.wave_fraction = "reading", 0.0
        env, path = self.env, self.path

        def read(cancel: threading.Event, progress) -> tuple:
            ff = engine.ffmpeg(env.threads, cancel)
            probed = info if info is not None else env.probe(path, ff)
            return env.peaks(ff, path, probed, partial=lambda pk, f: self.peaksPartial.emit(pk, f))

        self._start("wave", read, self._wave_done, self._wave_failed)
        self.changed.emit()

    def _on_peaks_partial(self, peaks, fraction: float) -> None:
        if self.wave == "reading" and not self._disposed:
            self.peaks, self.wave_fraction = list(peaks), float(fraction)
            self.changed.emit()

    def _wave_done(self, result) -> None:
        peaks, _seconds = result
        self.peaks, self.wave, self.wave_fraction = list(peaks), "ready", 1.0
        self.changed.emit()

    def _wave_failed(self, message: str) -> None:
        self.wave = "failed"
        self.changed.emit()

    # -- Automatic ---------------------------------------------------------------------------------------------------------
    def set_auto(self, on: bool) -> None:
        if on == self.auto or self.model is None:
            return
        self.auto = bool(on)
        self.env.prefs.auto_settings = self.auto
        if on:
            self.applied, self._force_apply = False, True
            self._listen(self._info)
        else:
            self.stop("listen")
            self.listening = False
            self.model._remember()
            self._replace_draft(editor_session.baseline_for(self.options_for(auto=False), self.seconds, echo=self._echo_wanted()))
            self.applied, self.auto_base = False, None
        self.changed.emit()

    def _echo_wanted(self) -> bool:
        return bool(self.env.prefs.echo_track and self.facts is not None and self.facts.tracks >= 2)

    def _replace_draft(self, tl: Timeline) -> None:
        m = self.model
        m.draft = tl.copy()
        m._changed()

    def _listen(self, info) -> None:
        """Listen to a little of the file for the settings it needs."""
        if info is None:
            return
        self.listening, self.applied = True, False
        env, path = self.env, self.path
        opts = self.options_for(auto=True)

        def listen(cancel: threading.Event, progress):
            return ("choices", env.decide(path, info, opts, threads=env.threads, cancel=cancel, voice_model=env.voice_ready()))

        self._start("listen", listen, self._listened, self._listen_failed)
        self.changed.emit()

    def _listen_failed(self, message: str) -> None:
        self.listening, self._force_apply = False, False
        self.changed.emit()
        self.maybe_suggest()

    def _listened(self, result) -> None:
        self.listening = False
        if self._disposed or self.model is None or not self.auto:
            return
        self.choices = result[1]
        base = editor_session.baseline_for(self.options_for(auto=True), self.seconds, echo=self._echo_wanted(), choices=self.choices)
        self.auto_base = base.copy()
        if self._force_apply:  # Automatic was switched on: its choices replace the ones made by hand (Undo brings them back)
            self.model._remember()
            self._replace_draft(base)
        elif not self.model.can_undo:  # nothing was changed while it listened
            self._replace_draft(base)
            self.model.clear_history()
        self._force_apply, self.applied = False, True
        self.changed.emit()
        self.maybe_suggest()

    @property
    def auto_custom(self) -> bool:
        """Automatic was on and the choices were changed since."""
        return bool(self.auto and self.applied and self.auto_base is not None and self.model is not None and self.model.draft != self.auto_base)

    @property
    def untouched_auto(self) -> bool:
        """Automatic is on and nothing was changed by hand: the run decides the choices again on the whole file (the same measurement)."""
        m = self.model
        return bool(m is not None and self.auto and not m.can_undo and (self.auto_base is None or m.draft == self.auto_base))

    def run_plan(self) -> tuple[engine.JobOptions, dict | None]:
        """(the options, the timeline) a run is given: Automatic's own deciding while it is untouched, else the timeline as it stands."""
        if self.model is None or self.untouched_auto:
            return self.options_for(auto=True), None
        return self.options_for(auto=False), self.model.effective().to_json()

    def use_auto_again(self) -> None:
        if self.model is not None and self.auto_base is not None:
            self.model._remember()
            self._replace_draft(self.auto_base)

    def undo_auto(self) -> None:
        """Back to the choices the file had before Automatic: Automatic goes off."""
        self.set_auto(False)

    @property
    def reasons(self) -> list[str]:
        return list(self.choices.reasons) if self.choices is not None else []

    def auto_matches(self, *layers: str) -> bool:
        """The layers are still as Automatic made them (the cards say "Automatic" while they are)."""
        if not (self.auto and self.applied and self.auto_base is not None and self.model is not None):
            return False
        return all(self.model.draft.layers[k] == self.auto_base.layers[k] for k in layers)

    # -- the model ---------------------------------------------------------------------------------------------------------
    def _model_changed(self) -> None:
        if self.stage_editable and not self.is_job:
            self._remember_choices()
        self.modelChanged.emit()

    @property
    def stage_editable(self) -> bool:
        return not self.locked

    def _remember_choices(self) -> None:
        """What was chosen for the whole file is what the next file starts with (not what Automatic picked: that is decided again)."""
        m = self.model
        if m is None or self.auto or self._taking or editor_job.has_parts(m.draft):
            return
        p, tl = self.env.prefs, m.draft
        p.leveling = bool(tl.layers["leveling"])
        if tl.layers["leveling"]:
            key = editor_job.loudness_key(editor_job.timeline_loudness(tl))
            p.loudness = key
        p.echo_track = bool(tl.layers["echo"])
        p.rebuild_voice = bool(tl.layers["rebuild"])
        if tl.layers["noise"]:
            st = tl.layers["noise"][0].settings
            key = next((k for k, db in engine.STRENGTHS.items() if abs(db - st["strength_db"]) < 0.5), None)
            if key:
                p.strength = key
            p.noise = st["remover"]

    # -- the same settings for other files ---------------------------------------------------------------------------------------
    def settings(self) -> FileSettings | None:
        """What this file is set to, for other files (None: nothing to give yet, the file is still being read)."""
        m = self.model
        if m is None:
            return None
        tl = m.draft
        auto = not self.is_job and self.auto and (self.untouched_auto or not self.applied)
        usual = engine.split_name(engine.default_output_name(self.path, self.facts.source(), self.env.prefs.job_options()))[1] \
            if self.facts is not None else ""
        ending = engine.split_name(self.out_name)[1] if self.out_name else ""
        p = self.env.prefs
        return FileSettings(None if auto else {name: _most_of(tl, name) for name in LAYERS}, self.subtitles, self.language, self.video_size,
                            ending if ending and ending != usual else None, had_parts=not auto and editor_job.has_parts(tl),
                            auto_leveling=p.loudness if p.leveling else None, auto_echo=bool(p.echo_track))

    def take_settings(self, s: FileSettings) -> None:
        """Take another file's settings (a file that has not been started). Its own name stays; a file still being read takes them once it
        is read. Undo brings back what it had, except after Automatic (which decides afresh for this file)."""
        if self.is_job or self.stage != "ready":
            return
        m = self.model
        if m is None or self.facts is None:
            self._given = s
            return
        self._taking = True  # (what this file is given is not remembered as the choices new files start with: that file's own were)
        try:
            self._take(s, m)
        finally:
            self._taking = False

    def _take(self, s: FileSettings, m: EditModel) -> None:
        self.subtitles, self.language, self.video_size = s.subtitles, s.language, s.video_size
        m._remember()
        if s.layers is None:  # Automatic: this file decides for itself (when it is opened, or by the run)
            self.stop("listen")
            self.listening, self.auto, self.applied, self.auto_base, self.choices = False, True, False, None, None
            self._after_started = False
            m.draft = editor_session.baseline_for(self.options_for(auto=False), self.seconds, echo=self._echo_wanted())
            m.clear_history()
        else:
            self.stop("listen")
            self.listening, self.auto, self.applied, self.auto_base = False, False, False, None
            tl = Timeline(self.seconds)
            for name, st in s.layers.items():
                if st is not None and (name != "echo" or self.facts.tracks >= 2):
                    tl.paint(name, 0.0, tl.seconds, dict(st) or None)
            if s.layers.get("leveling") is not None:
                m._loudness = s.layers["leveling"].get("loudness_lufs", m._loudness)
            m.draft = tl
        source = self.facts.source()
        if s.ending and not self.out_custom and s.ending in engine.available_endings(source):
            usual = engine.default_output_name(self.path, source, self.env.prefs.job_options())
            name = engine.with_ending(usual, s.ending)
            self.set_out_name(name, custom=name != usual)
        if self.subtitles == "embed" and engine.split_name(self.out_name)[1] not in engine._EMBED_ENDINGS:
            self.subtitles = "srt"  # (only a video file can hold them)
        m._changed()
        self.changed.emit()

    # -- the choices about the whole file ---------------------------------------------------------------------------------------
    def set_subtitles(self, kind: str) -> None:
        if kind != self.subtitles:
            self.subtitles = kind
            self.env.prefs.subtitles = kind
            self.changed.emit()
            self.modelChanged.emit()

    def set_language(self, language: str) -> None:
        if language != self.language:
            self.language = language
            self.env.prefs.language = language
            self.changed.emit()
            self.modelChanged.emit()

    def set_video_size(self, size: str) -> None:
        if size != self.video_size:
            self.video_size = size
            self.env.prefs.video_size = size
            self.changed.emit()
            self.modelChanged.emit()

    def set_out_name(self, name: str, *, custom: bool = True) -> None:
        """What the result is called. `custom`: it is the person's own (a name that is the usual one again follows the choices again)."""
        if name != self.out_name or custom != self.out_custom:
            self.out_name, self.out_custom = name, custom
            if self.item is not None:
                self.item.out_name, self.item.out_custom = name, custom
            self.changed.emit()
            self.modelChanged.emit()

    def selection(self) -> tuple[float, float]:
        """The part that is chosen (the whole file when none is)."""
        m = self.model
        return m.selection if m is not None and m.selection is not None else (0.0, self.seconds)

    # -- suggestions -------------------------------------------------------------------------------------------------------
    def maybe_suggest(self) -> None:
        """Look for moments worth a look by itself, in a recording that is not long (a longer one only when asked)."""
        if (not self.env.auto_suggest or not self.show_suggestions or self.model is None or self.sg != "idle" or self.model.suggestions
                or self.seconds > editor_job.SUGGEST_AUTO_SECONDS or self.facts is None):
            return
        if self.is_job and editor_job.stored_suggestions(self.job) is not None:
            return
        self.find()

    def find(self) -> None:
        """Look for the parts of the recording that other settings would suit (it needs all of the recording decoded)."""
        if self.model is None or "find" in self._workers:
            return
        if self.is_job and self.session is None:
            if self.session_state == "opening":  # a job is looked at in its live sound: when that is ready
                self.sg, self.sg_fraction, self._find_when_ready = "running", 0.0, True
                self.changed.emit()
            else:
                self._not_found("Looking for moments needs the original, and it is not there.")
            return
        self.sg, self.sg_fraction, self.sg_note = "running", 0.0, ""
        env, path, session = self.env, self.path, self.session
        scratch = env.scratch()

        def find(cancel: threading.Event, progress):
            if session is not None:
                return session.suggest(cancel=cancel, progress=progress)
            plan = env.open_plan(path, cancel=cancel, scratch_parent=scratch, threads=env.threads)
            try:
                return plan.suggest(cancel=cancel, progress=lambda f: progress(0.1 + 0.9 * f))
            finally:
                plan.close()

        w = self._start("find", find, self._found, lambda m: self._not_found(m or "Couldn't look for moments in this file."),
                        progress=self._find_progress, stopped=lambda: self._not_found("Stopped looking.", "canceled"))
        if w is not None:
            self.changed.emit()

    def _find_progress(self, f: float) -> None:
        self.sg_fraction = f
        self.changed.emit()  # (the line under Automatic says how far it is)

    def _found(self, found: dict) -> None:
        self.sg, self.sg_note = "done", "" if found["items"] else "No moments found that need other settings."
        if self.model is not None:
            self.model.set_suggestions(found["items"])
            if self.job is not None and self.session is None:
                editor_job.keep_suggestions(self.job, found)
        self.changed.emit()

    def _not_found(self, message: str, state: str = "error") -> None:
        self.sg, self.sg_note = state, message
        self.changed.emit()

    def cancel_find(self) -> None:
        if self._find_when_ready:
            self._find_when_ready = False
            self._not_found("Stopped looking.", "canceled")
            return
        self.stop("find")
        self._not_found("Stopped looking.", "canceled")

    def keep_suggestions(self) -> None:
        """A suggestion was applied, undone or dismissed: a job keeps that."""
        if self.model is None or self.job is None:
            return
        stored = editor_job.stored_suggestions(self.job) or {}
        editor_job.keep_suggestions(self.job, {**stored, "items": self.model.suggestions_json()})

    # ------------------------------------------------------------------------------------------------ a finished job
    @classmethod
    def for_job(cls, env: DocEnv, entry: Entry, job: copies.JobCopies | None, check: copies.OriginalCheck, *, item: FileItem | None = None,
                parent: QObject | None = None) -> "Doc":
        doc = cls(env, entry.source, item=item, parent=parent)
        doc.name = entry.name
        doc.attach_job(entry, job, check)
        return doc

    def attach_job(self, entry: Entry, job: copies.JobCopies | None, check: copies.OriginalCheck) -> None:
        """This is the finished job `entry` (with its saved copies `job`, when it has them): the editor shows the result, from the facts and
        the waveform the job kept, at once; the live sound follows in the background."""
        self.entry, self.job, self.original = entry, job, check
        self.stop("probe")
        self.listening = False
        if job is not None:
            self.facts = facts_of_job(job, Path(entry.source))
            self.seconds = editor_job.seconds_of(job) or self.facts.duration
            done = editor_job.current_timeline(job, self.seconds)
            reference = bool((editor_job.job_facts(job) or {}).get("reference")) or self.facts.tracks >= 2
            m = self._new_model(done.copy(), echo_available=reference, loudness=editor_job.timeline_loudness(done), done=done)
            stored = editor_job.stored_suggestions(job)
            m.set_suggestions(stored["items"] if stored else None)
            if stored:
                self.sg = "done"
            self.peaks = list(job.editor.get("peaks") or []) or None
            self.wave = "ready" if self.peaks else "none"
            wanted = redo.wanted_from(job)
            self.subtitles, self.language, self.video_size = wanted.subtitles, wanted.language, wanted.video_size
            self.out_name = redo.suggest_name(job, self.facts.source())
        else:  # (no saved copies: it can be listened to and deleted, not changed)
            self.facts = Facts(float(entry.duration), bool(entry.has_video), 2, 1, Path(entry.source).suffix.lower(), None, 0, 0, 0)
            self.seconds = self.facts.duration
            tl = Timeline.whole_file(max(self.seconds, 1.0), echo=False, noise=False, rebuild=False, leveling=False)
            m = self._new_model(tl, echo_available=False, loudness=editor_job.timeline_loudness(tl), done=tl)
            self.out_name = Path(entry.outputs[0]).name if entry.outputs else ""
        self.auto, self.applied, self.auto_base = False, False, None
        self.out_custom = False
        self.snap = self._output_choices()
        self._read_result_wave()
        if self.peaks is None and Path(entry.source).exists():
            self._read_wave()  # (an old job kept no waveform: it is read from the original)
        self.changed.emit()

    def result_paths(self) -> list[Path]:
        """The files of the latest result (the sound or video, then the subtitles)."""
        if self.entry is None:
            return []
        return [Path(p) for p in (self.job.saves[-1].get("files", []) if self.job is not None and self.job.saves else self.entry.outputs)]

    def _result_file(self) -> Path | None:
        return next((p for p in self.result_paths() if p.suffix.lower() not in (".srt", ".vtt", ".txt") and p.exists()), None)

    def _read_result_wave(self) -> None:
        """The waveform of the saved result (shown over the original's: what the cleaning did)."""
        result = self._result_file()
        self.heard = None
        if result is None:
            return
        env = self.env

        def read(cancel: threading.Event, progress):
            ff = engine.ffmpeg(env.threads, cancel)
            info = env.probe(result, ff)
            return env.peaks(ff, result, info)

        self._start("heard", read, self._heard_done, lambda m: None)

    def _heard_done(self, result) -> None:
        self.heard = list(result[0])
        self.changed.emit()

    # -- the live sound ----------------------------------------------------------------------------------------------------
    def open_session(self) -> None:
        """Decode the original and lay out what the job has made, in the background, so changes can be heard while it plays."""
        job = self.job
        if job is None or self.session_state in ("opening", "ready") or not self.original.ok:
            return
        self.session_state, self.session_fraction, self.session_note = "opening", 0.0, ""
        env, scratch = self.env, self.env.scratch()
        self._start("session", lambda cancel, progress: env.open_session(job, cancel=cancel, progress=progress, scratch_parent=scratch, threads=env.threads),
                    self._session_ready, self._session_failed, progress=self._session_progress,
                    stopped=lambda: setattr(self, "session_state", "none"))
        self.changed.emit()

    def _session_progress(self, f: float) -> None:
        self.session_fraction = f
        self.progressed.emit()

    def _session_failed(self, message: str) -> None:
        self.session_state, self.session_note = "failed", message
        self.changed.emit()

    def _session_ready(self, session: editor_session.Session) -> None:
        if self._disposed or self.job is None:
            session.close()
            return
        self.session, self.session_state, self.session_fraction = session, "ready", 1.0
        self.sessionReady.emit(session)
        self.changed.emit()
        self.refresh_session()
        if self._find_when_ready:
            self._find_when_ready = False
            self.sg = "idle"
            self.find()
        elif self.sg == "idle":
            self.maybe_suggest()

    def refresh_session(self) -> None:
        """The draft changed: hear it (cheap: only blocks new to the cache are read)."""
        if self.session is None or self.model is None:
            return
        try:
            self.session.refresh(self.model.draft)
        except Exception:  # noqa: BLE001 - listening must never break editing
            log.warning("could not refresh the live sound", exc_info=True)

    def quote(self) -> editor_job.Quote | None:
        """What saving the draft would take: from the live sound's session, else from the facts the job kept (nothing is decoded)."""
        if self.model is None or self.job is None:
            return None
        try:
            if self.session is not None:
                return self.session.quote(self.model.draft)
            return editor_job.quote_for(self.job, self.model.draft)
        except Exception:  # noqa: BLE001
            log.warning("could not quote the changes", exc_info=True)
            return None

    def saved(self, entry: Entry, job: copies.JobCopies | None) -> None:
        """A save finished: the job's latest result is what the draft says now, and the blocks it made can be heard."""
        self.entry, self.job = entry, job
        if job is None or self.model is None:
            return
        done = editor_job.current_timeline(job, self.seconds)
        self.model.done = done.copy()
        self.model.draft = done.copy()
        self.model.selection = None
        self.model.clear_history()
        self.out_name = redo.suggest_name(job, self.facts.source()) if self.facts is not None else self.out_name  # the next result has a free name
        self.snap = self._output_choices()
        self._read_result_wave()
        self.model._changed()
        self.refresh_session()
        self.changed.emit()

    def finished(self, entry: Entry, job: copies.JobCopies | None, check: copies.OriginalCheck) -> None:
        """A run of this file finished: the same screen shows the result (the draft is what was made)."""
        draft = self.model.draft.copy() if self.model is not None else None
        peaks, facts = self.peaks, self.facts
        suggestions = self.model.suggestions_json() if self.model is not None else []
        name = self.out_name
        self.entry, self.job, self.original = entry, job, check
        if job is not None and draft is not None:
            done = editor_job.current_timeline(job, self.seconds)
            m = self._new_model(done.copy(), echo_available=self.model.echo_available, loudness=editor_job.timeline_loudness(done), done=done)
            m.set_suggestions(suggestions)
            self.peaks = peaks or list(job.editor.get("peaks") or []) or None
            self.out_name = redo.suggest_name(job, facts.source() if facts is not None else engine.Source(False))
        elif draft is not None:
            self.model.done = draft
            self.model.clear_history()
            self.out_name = name
        self.auto_base, self.applied = None, False
        self.auto = False
        self.snap = self._output_choices()
        self._read_result_wave()
        self.changed.emit()

    # ------------------------------------------------------------------------------------------------ what it takes
    def estimate(self) -> tuple[list[tuple[str, float]], float]:
        """([(what, about how many seconds)], the total): what a run of the draft would take on this computer. Nothing is decoded to say it:
        each step's time for what it covers, as this computer's runs measured it (speed.py); a finished job's changes are quoted from the
        blocks it already has (`quote`)."""
        m = self.model
        if m is None or self.facts is None:
            return [], 0.0
        ch = min(max(self.facts.channels, 1), 2)
        rows: list[tuple[str, float]] = []

        def cost(step: str, covered: float) -> float:
            # (Voice rebuild runs once on the mix of a stereo recording whose sides are the same voice, as recordings of one voice are)
            return speed.cost(step, covered * (1 if step == "rebuild" else ch)) if covered > 0 else 0.0

        def covered(layer: str, pred=lambda p: True) -> float:
            return sum(p.end - p.start for p in m.draft.layers[layer] if pred(p))

        light = covered("noise", lambda p: p.settings["remover"] in ("light", "auto"))
        heavy = covered("noise", lambda p: p.settings["remover"] == "heavy")
        if light or heavy:
            rows.append((f"Noise removal · {say_clock(light + heavy)}", cost("light", light) + cost("heavy", heavy)))
        if covered("rebuild"):
            rows.append((f"Voice rebuild · {say_clock(covered('rebuild'))}", cost("rebuild", covered("rebuild"))))
        if covered("echo"):
            rows.append((f"Echo removal · {say_clock(covered('echo'))}", cost("echo", covered("echo"))))
        if covered("leveling"):
            rows.append((f"Leveling · {say_clock(covered('leveling'))}", cost("leveling", covered("leveling"))))
        if self.subtitles != "off":
            rows.append((f"Subtitles · {say_clock(self.seconds)}", speed.cost("speech", self.seconds)))
        return rows, sum(s for _, s in rows)

    # ------------------------------------------------------------------------------------------------ output choices
    def _output_choices(self) -> tuple:
        return (self.out_name, self.subtitles, self.language, self.video_size)

    @property
    def changed_output(self) -> bool:
        return self.snap is not None and self._output_choices() != self.snap

    @property
    def changed_sound(self) -> bool:
        return bool(self.model is not None and self.model.done is not None and self.model.draft != self.model.done)

    # ------------------------------------------------------------------------------------------------ threads
    def _start(self, name: str, fn: Callable, done: Callable, failed: Callable | None = None, *, progress: Callable | None = None,
               stopped: Callable | None = None) -> Worker | None:
        if self._disposed:
            return None
        self.stop(name)
        gen = self._gen[name] = self._gen.get(name, 0) + 1
        w = Worker(fn, self)
        self._workers[name] = w

        def guard(callback):
            def call(*args):
                if gen != self._gen.get(name) or self._disposed:
                    return
                self._workers.pop(name, None)
                callback(*args)
            return call

        w.done.connect(guard(done))
        w.failed.connect(guard(failed or (lambda m: None)))
        w.stopped.connect(guard(stopped or (lambda: None)))
        if progress is not None:
            w.progress.connect(lambda f: gen == self._gen.get(name) and progress(f))
        w.finished.connect(w.deleteLater)
        w.start()
        return w

    def stop(self, name: str) -> None:
        w = self._workers.pop(name, None)
        self._gen[name] = self._gen.get(name, 0) + 1
        if w is not None:
            w.cancel.set()
            self._parked.append(w)

    def _wait_parked(self, stop: bool = False) -> None:
        """Wait for the threads that were told to stop (one that has ended is already gone)."""
        for w in self._parked:
            try:
                if stop:
                    w.cancel.set()
                w.wait(5000)
            except RuntimeError:  # (deleted when it finished)
                pass
        self._parked.clear()

    def busy_with(self) -> list[str]:
        return list(self._workers)

    def dispose(self) -> None:
        """Let go of the file: stop everything that runs for it, close the live sound (it deletes its scratch files)."""
        self._disposed = True
        for name in list(self._workers):
            self.stop(name)
        self._wait_parked(stop=True)
        session, self.session = self.session, None
        if session is not None:
            session.close()
        self.session_state = "none"
