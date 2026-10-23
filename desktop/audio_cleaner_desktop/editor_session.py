"""A job (or a file about to be cleaned) opened in the per-part editor: the recording decoded and laid out for listening, what the editor
needs to quote and to look for suggestions. No Qt in here; the job view runs `Session.open` / `PlanSession.open` on a worker thread."""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from acw_common import editor, pipeline
from acw_common.ffmpeg import MediaError
from acw_common.timeline import Timeline

from . import copies, editor_job, engine, live_mix, resources

log = logging.getLogger(__name__)

Progress = Callable[[float], None]


class Session:
    """A job opened for editing: `mix` is what the player plays; `quote` says what an edit would cost; `suggest` finds moments to fix."""

    def __init__(self, job: copies.JobCopies, ff, work: Path, inputs: editor.Inputs, info, threads: int):
        self.job, self.ff, self.work, self.inputs, self.info, self.threads = job, ff, work, inputs, info, threads
        self.cache = job.block_cache(ff)
        self.seconds = inputs.n / pipeline.SAMPLE_RATE
        self.layers = live_mix.Layers(inputs.source, inputs.plan, work)
        self.mix: live_mix.LiveMix | None = None
        self.echo_available = info.audio_tracks >= 2
        self._facts = editor_job.facts_inputs(editor_job.job_facts(job) or editor_job.facts_of(inputs), job.source_id)
        self._closed = False

    @classmethod
    def open(cls, job: copies.JobCopies, *, cancel: threading.Event | None = None, progress: Progress | None = None,
             scratch_parent: Path | None = None, threads: int = 2) -> "Session":
        """Decode the original, make sure the job keeps blocks (an old job's copies become blocks: editor_job.make_ready), remember the recording's facts and waveform,
        and lay out what the cache has for listening. Raises MediaError (fit for the user)."""
        cancel = cancel or threading.Event()
        check = job.check_original()
        if not check.ok:
            raise MediaError(check.text)
        ff = engine.ffmpeg(threads, cancel)
        src = job.original_path
        info = engine.probe(src, ff)
        folder = scratch_parent or job.dir.parent
        work = editor_job.scratch_dir(folder, engine.estimated_scratch_gb(info.duration, info.audio_channels) * 3)
        try:
            inputs = editor_job.decode(ff, src, info, work, reference=True, cancel=cancel, source_id=job.source_id)
            if progress:
                progress(0.3)
            if job.samples and inputs.n != job.samples:
                raise MediaError("The original no longer matches the saved copies (it is not the same length).")
            editor_job.make_ready(job, ff, inputs, work, cancel=cancel, progress=lambda f: progress and progress(0.3 + 0.4 * f))
            if editor_job.job_facts(job) is None or not job.editor.get("peaks"):
                editor_job.store_facts(job, inputs, inputs.source, reference=info.audio_tracks >= 2)
            session = cls(job, ff, work, inputs, info, threads)
            tl = editor_job.current_timeline(job, session.seconds)
            session.prepare(tl)
            if progress:
                progress(1.0)
            return session
        except BaseException:
            editor_job.remove_work(work)
            raise

    def prepare(self, tl: Timeline) -> None:
        """Lay out the blocks the timeline needs that are in the cache and start the live mix on it."""
        tl = self._for_listening(tl)
        plan = editor.plan(tl, self._facts, self.cache)
        self.layers.load(self.cache, plan)
        self.mix = live_mix.LiveMix(self.layers, tl)

    def _for_listening(self, tl: Timeline) -> Timeline:
        tl, _ = editor_job.drop_missing_echo(tl, self._facts)
        return editor_job.resolve_known(tl, self.job.editor.get("resolved"))

    def refresh(self, tl: Timeline) -> None:
        """The draft changed (or blocks were made): hear it. Cheap: nothing is decoded again, only blocks new to the cache are read."""
        if self.mix is None or self._closed:
            return
        tl = self._for_listening(tl)
        plan = editor.plan(tl, self._facts, self.cache)
        self.mix.set_timeline(tl)
        self.mix.refresh(self.cache, plan)

    def quote(self, tl: Timeline) -> editor_job.Quote:
        tl = self._for_listening(tl)
        return editor_job.quote(tl, self._facts, self.cache, editor_job.previous_of(self.job, self.seconds), self.threads)

    def suggest(self, *, cancel: threading.Event | None = None, progress: Progress | None = None) -> dict:
        """Look for moments that other settings would suit (editor_job.find_suggestions) and keep what was found with the job."""
        found = editor_job.find_suggestions(self.inputs.source, threads=self.threads, cancel=cancel, progress=progress)
        editor_job.keep_suggestions(self.job, found)
        return found

    def close(self) -> None:
        self._closed = True
        if self.mix is not None:
            self.mix.close()
        self.layers.close()
        self.inputs = None  # type: ignore[assignment]  # (the memory maps must be let go before the folder can be removed)
        editor_job.remove_work(self.work)


class PlanSession:
    """A file that has not been cleaned yet, opened in the editor to choose its parts: the waveform, the suggestions, a quote of everything
    (nothing is processed yet, so every block is missing)."""

    def __init__(self, path: Path, info, work: Path, inputs: editor.Inputs, threads: int):
        self.path, self.info, self.work, self.inputs, self.threads = path, info, work, inputs, threads
        self.seconds = inputs.n / pipeline.SAMPLE_RATE
        self.peaks = editor_job.peaks_of(inputs.source)
        self.echo_available = info.audio_tracks >= 2
        self.facts = editor_job.facts_of(inputs)
        self.facts["reference"] = self.echo_available
        self._facts_inputs = editor_job.facts_inputs(self.facts, copies.source_id_of(copies.fingerprint(path)))
        self.suggestions: dict | None = None

    @classmethod
    def open(cls, path: Path, *, cancel: threading.Event | None = None, scratch_parent: Path | None = None, threads: int = 2) -> "PlanSession":
        cancel = cancel or threading.Event()
        ff = engine.ffmpeg(threads, cancel)
        info = engine.probe(path, ff)
        work = editor_job.scratch_dir(scratch_parent or Path(path).parent, engine.estimated_scratch_gb(info.duration, info.audio_channels))
        try:
            x = pipeline.decode_for_clean(ff, path, info, engine.MAX_SECONDS, work_dir=work)
            inputs = editor.Inputs(x, copies.source_id_of(copies.fingerprint(path)))
            return cls(path, info, work, inputs, threads)
        except BaseException:
            editor_job.remove_work(work)
            raise

    def baseline(self, options: engine.JobOptions, *, cancel: threading.Event | None = None) -> Timeline:
        """What the file's settings stand for on the whole file. "Automatic best settings" are decided by listening to the file (the same
        measurement a normal run makes), so the editor shows what would be done."""
        choices = None
        if options.decides:
            choices = engine._decide(self.inputs.source, options, cancel or threading.Event(), resources.voice_model_dir() is not None)
        return baseline_for(options, self.seconds, echo=self.echo_available, choices=choices)

    def quote(self, tl: Timeline) -> editor_job.Quote:
        tl, _ = editor_job.drop_missing_echo(tl, self._facts_inputs)
        tl = editor_job.resolve_known(tl, None)
        return editor_job.quote(tl, self._facts_inputs, None, None, self.threads)

    def suggest(self, *, cancel: threading.Event | None = None, progress: Progress | None = None) -> dict:
        self.suggestions = editor_job.find_suggestions(self.inputs.source, threads=self.threads, cancel=cancel, progress=progress)
        return self.suggestions

    def close(self) -> None:
        self.inputs = None  # type: ignore[assignment]
        editor_job.remove_work(self.work)


def baseline_for(options: engine.JobOptions, seconds: float, *, echo: bool, choices=None) -> Timeline:
    """The timeline of the whole-file settings `options` (with what "automatic" decided, `choices`, when it was asked)."""
    atten, noise, rebuild = options.atten_lim_db, options.noise, options.rebuild_voice
    if choices is not None:
        if options.strength == engine.AUTO_STRENGTH:
            atten = choices.atten_lim_db
        if options.noise == "auto":
            noise = choices.noise
        rebuild = rebuild or bool(choices.rebuild)
    strength = 100.0 if atten is None else float(atten)
    return Timeline.whole_file(
        seconds, echo=options.echo_track and echo, noise=False if rebuild else {"strength_db": strength, "remover": noise},
        rebuild=rebuild, leveling={"loudness_lufs": options.loudness_lufs} if options.leveling else False)


def only_parts(tl: Timeline, parts: list[tuple[float, float]]) -> Timeline:
    """The timeline with everything outside `parts` left out: only those parts are processed."""
    out = tl.copy()
    edges = sorted((max(0.0, a), min(tl.seconds, b)) for a, b in parts if b > a)
    for layer in out.layers:
        keep = []
        for p in out.layers[layer]:
            for a, b in edges:
                lo, hi = max(a, p.start), min(b, p.end)
                if hi > lo:
                    keep.append((lo, hi, dict(p.settings)))
        out.layers[layer] = []
        for lo, hi, st in keep:
            out.paint(layer, lo, hi, st)
    return out


__all__ = ["Session", "PlanSession", "baseline_for", "only_parts"]
