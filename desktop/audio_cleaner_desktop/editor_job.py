"""The per-part editor of a desktop job: what a job remembers for it, what an edit would cost, and doing it. No Qt in here.

A job's folder holds the BLOCK CACHE (copies.py, `blocks/`, acw_common.blocks.FlacCache): every block a model step made, keyed by the step,
the model, the audio it read and where it is. A normal run (`engine.process`, which calls `pipeline.clean_audio` with this cache) fills it, so
an edit only makes the blocks it does not have yet: a new strength is a blend of blocks that are there, Voice rebuild on one part makes the
blocks of that part, and Leveling runs over what changed. The job's manifest (`job.editor`) also keeps

    facts        what planning needs without decoding the original: samples, channels, the channel plan, the stretches that are not
                 digital silence, whether Voice rebuild works on the mix, whether the file has a second track
    peaks        the waveform the editor draws (0..100 per bucket)
    timeline     the timeline of the latest result (JSON v1, acw_common.timeline), None for a result made by the whole-file defaults
    resolved     the noise remover each "automatic" part was decided to use (so a quote and its run agree)
    suggestions  the parts the app found that other settings would suit, and what the user did with them

Old jobs (made before the block cache) keep their saved copies and open as before; `migrate` turns those copies into blocks the first time
the editor is asked for.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from acw_common import blocks, editor, kept, pipeline, restore, suggest
from acw_common.ffmpeg import MediaError
from acw_common.timeline import DEFAULT_LOUDNESS, Timeline, TimelineError

from . import copies, engine, resources

log = logging.getLogger(__name__)

SR = pipeline.SAMPLE_RATE
REFERENCE_ID = "r"
PEAK_BUCKETS = 1600
#: Suggestions are looked for by themselves in recordings up to this long; for a longer one there is a button (it takes a while).
SUGGEST_AUTO_SECONDS = 20 * 60

#: Wall-clock seconds one second of audio takes on one core of an ordinary computer, and a fixed amount per run, step by step. The
#: light remover, echo removal and Leveling are measured (editor.STEP_COSTS: CPU); the heavy remover is 1.8 x the light one, and Voice
#: rebuild on a CPU about 30 x (docs/research); the server runs those two on a graphics card, this app on the processor.
CPU_COSTS = {
    "echo": (0.12, 0.034), "light": (0.27, 0.116), "heavy": (1.5, 0.21), "rebuild": (4.3, 3.5), "leveling": (1.28, 0.013),
}
THREAD_EFFICIENCY = 0.7


# ---------------------------------------------------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------------------------------------------------
class EchoWholeFile:
    """What `pipeline.clean_audio` is given as its cache after the whole file went through echo removal. It keys every block with one
    `input_id`, but the editor's identity of a block names the echo parts that touch what it read (here: all of the file, cut to the
    block's reach). `put`, `get` and `has` rewrite the key to the editor's own (editor._Identity), so a whole-file run fills the cache the
    editor plans against."""

    MARK = "\x1eecho"

    def __init__(self, cache, source_id: str, n: int):
        self.cache = cache
        self.n = n
        self.ident = editor._Identity(SimpleNamespace(source_id=source_id, reference_id=REFERENCE_ID), [(0, n)])

    def _key(self, key: str) -> str:
        parts = key.split("|")
        if self.MARK not in parts:
            return key
        i = parts.index(self.MARK)
        if parts[1] in ("light", "heavy"):
            lo, hi = int(parts[i + 2]), int(parts[i + 5])
        else:  # Voice rebuild: the block's number and the length of the recording
            lo, hi = restore.block_input_range(int(parts[i + 3]), self.n)
        parts[i] = self.ident(lo, hi)
        return "|".join(parts)

    def get(self, key: str):
        return self.cache.get(self._key(key))

    def put(self, key: str, block) -> None:
        self.cache.put(self._key(key), block)

    def has(self, key: str) -> bool:
        return self.cache.has(self._key(key))


def cache_args(cache, source_id: str, n: int, echo: bool) -> dict:
    """The `cache` / `input_id` arguments of `pipeline.clean_audio` that make a whole-file run fill the editor's cache."""
    if cache is None:
        return {}
    if echo:
        return {"cache": EchoWholeFile(cache, source_id, n), "input_id": EchoWholeFile.MARK}
    return {"cache": cache, "input_id": source_id}


def echo_input_id(source_id: str) -> str:
    """The `input_id` of `pipeline.remove_echo` (it names the recording and the speaker track together)."""
    return f"{source_id}~{REFERENCE_ID}"


# ---------------------------------------------------------------------------------------------------------------------
# What a job knows about its recording
# ---------------------------------------------------------------------------------------------------------------------
def facts_of(inputs: editor.Inputs) -> dict:
    """What planning needs about the recording (JSON-able)."""
    return {
        "n": inputs.n, "channels": inputs.channels, "plan": [None if c is None else int(c) for c in inputs.plan],
        "spans": {str(c): [[int(a), int(b)] for a, b in inputs.spans(c)] for c in inputs.todo},
        "rebuild_mid": bool(inputs.rebuild_mid), "reference": inputs.reference is not None,
    }


def peaks_of(x: np.ndarray, buckets: int = PEAK_BUCKETS) -> list[int]:
    """The waveform: the loudest sample of each of `buckets` equal stretches, as 0..100 of the loudest of them (read in pieces)."""
    n = len(x)
    if n == 0:
        return []
    edges = np.linspace(0, n, min(buckets, n) + 1).astype(np.int64)
    out = np.zeros(len(edges) - 1)
    for i in range(len(out)):
        a, b = int(edges[i]), int(edges[i + 1])
        seg = np.asarray(x[a:b])
        out[i] = float(np.max(np.abs(seg))) if seg.size else 0.0
    top = float(out.max()) or 1.0
    return [int(round(v / top * 100)) for v in out]


def facts_inputs(facts: dict, source_id: str) -> editor.Inputs:
    """An `editor.Inputs` that plans without audio: the facts stand in for the samples (the job view quotes without decoding)."""

    class FactsInputs(editor.Inputs):
        def __init__(self) -> None:  # (deliberately not Inputs.__init__: there are no samples)
            self.source = np.zeros((0, facts["channels"]), np.float32)
            self.source_id = source_id
            self.reference = np.zeros(0, np.float32) if facts.get("reference") else None
            self.reference_id = REFERENCE_ID if facts.get("reference") else None
            self._plan = list(facts["plan"])
            self._spans = {int(c): [tuple(r) for r in rows] for c, rows in facts["spans"].items()}
            self._mid = bool(facts["rebuild_mid"])

        @property
        def n(self) -> int:
            return facts["n"]

        @property
        def channels(self) -> int:
            return facts["channels"]

    return FactsInputs()


def store_facts(job: copies.JobCopies, inputs: editor.Inputs, x: np.ndarray, *, reference: bool | None = None, save: bool = True) -> None:
    ed = job.editor
    ed["facts"] = facts_of(inputs)
    if reference is not None:
        ed["facts"]["reference"] = bool(reference)
    ed["peaks"] = peaks_of(x)
    ed["version"] = 1
    if save:
        job.save()


def job_facts(job: copies.JobCopies) -> dict | None:
    f = job.editor.get("facts")
    ok = isinstance(f, dict) and all(k in f for k in ("n", "channels", "plan", "spans", "rebuild_mid"))
    return f if ok else None


def seconds_of(job: copies.JobCopies) -> float:
    f = job_facts(job)
    if f:
        return f["n"] / SR
    return job.samples / SR if job.samples else float(job.media.get("duration") or 0.0)


# ---------------------------------------------------------------------------------------------------------------------
# Timelines
# ---------------------------------------------------------------------------------------------------------------------
def loudness_key(lufs: float | None) -> str:
    for key, value in engine.LOUDNESS_TARGETS.items():
        if (value is None and lufs is None) or (value is not None and lufs is not None and abs(value - lufs) < 0.05):
            return key
    return "podcast"


def baseline_timeline(job: copies.JobCopies, seconds: float | None = None) -> Timeline:
    """The timeline of the whole-file result the job was made with (what its choices stand for on the whole file)."""
    seconds = seconds_of(job) if seconds is None else seconds
    m = job.made
    atten = m.get("atten_lim_db")
    strength = float(atten) if isinstance(atten, (int, float)) and not isinstance(atten, bool) and atten > 0 else 100.0
    remover = m.get("noise") if m.get("noise") in ("light", "heavy") else "auto"
    rebuilt = bool(m.get("rebuilt"))
    lufs = engine.LOUDNESS_TARGETS.get(m.get("loudness") if m.get("loudness") in engine.LOUDNESS_TARGETS else engine.DEFAULT_LOUDNESS)
    return Timeline.whole_file(
        seconds, echo=bool(job.echo), noise=False if rebuilt else {"strength_db": min(strength, 100.0), "remover": remover},
        rebuild=rebuilt, leveling={"loudness_lufs": lufs} if m.get("leveling", True) else False)


def stored_timeline(job: copies.JobCopies, seconds: float) -> Timeline | None:
    raw = job.editor.get("timeline")
    if not raw:
        return None
    try:
        return Timeline.from_json(raw, seconds=seconds)
    except TimelineError:
        return None


def current_timeline(job: copies.JobCopies, seconds: float | None = None) -> Timeline:
    """The timeline the job's latest result stands for: the one it was made from, else the whole-file defaults it was made with."""
    seconds = seconds_of(job) if seconds is None else seconds
    return stored_timeline(job, seconds) or baseline_timeline(job, seconds)


def timeline_loudness(tl: Timeline) -> float | None:
    parts = tl.layers["leveling"]
    return parts[0].settings["loudness_lufs"] if parts else DEFAULT_LOUDNESS


def covered(tl: Timeline, layer: str) -> float:
    return sum(p.seconds for p in tl.layers[layer])


def made_from_timeline(tl: Timeline, *, echo: bool, language: str | None = None) -> dict:
    """The choices a result made from a timeline stands for, in the shape `job.made` has (the whole-file view of it: the strength and
    the remover that cover the most of the file, Voice rebuild when it covers more than half)."""
    total = tl.seconds or 1.0
    rebuilt = covered(tl, "rebuild") > total / 2
    noise = sorted(tl.layers["noise"], key=lambda p: -p.seconds)
    top = noise[0].settings if noise else None
    return {
        "atten_lim_db": None if rebuilt or top is None else float(top["strength_db"]),
        "strength": None if rebuilt or top is None else (next((k for k, db in engine.STRENGTHS.items() if abs(db - top["strength_db"]) < 0.5), None)),
        "noise": "rebuilt" if rebuilt else (top["remover"] if top and top["remover"] in ("light", "heavy") else None),
        "rebuilt": rebuilt, "leveling": bool(tl.layers["leveling"]), "loudness": loudness_key(timeline_loudness(tl)),
        "echo": bool(tl.layers["echo"]) or echo, "parts": has_parts(tl),
    }


def part_key(start: float, end: float) -> str:
    return f"{start:.3f}-{end:.3f}"


def has_parts(tl: Timeline) -> bool:
    """The timeline is not just every layer on the whole file."""
    whole = [(0.0, tl.seconds)]
    return any([(p.start, p.end) for p in parts] not in ([], whole) for parts in tl.layers.values())


# ---------------------------------------------------------------------------------------------------------------------
# Quote
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Quote:
    """What an edit would take: the blocks that are missing, step by step, and about how long making them takes here."""

    seconds: float = 0.0
    steps: dict | None = None  # {step: blocks to make}
    misses: int = 0
    leveling: bool = False  # Leveling has to run again over something
    plan: editor.Plan | None = None

    @property
    def nothing(self) -> bool:
        return self.misses == 0


def estimate_seconds(plan: editor.Plan, threads: int = 2) -> float:
    """About how long the plan's missing blocks and Leveling take on this kind of computer (wall-clock seconds)."""
    cores = max(1.0, threads * THREAD_EFFICIENCY)
    total = 0.0
    for step, row in plan.summary().items():
        if row["blocks"]:
            fixed, per = CPU_COSTS[step]
            total += (fixed + per * row["compute_seconds"]) / cores
    return total


def quote(tl: Timeline, inputs: editor.Inputs, cache, previous: Timeline | None = None, threads: int = 2) -> Quote:
    p = editor.plan(tl, inputs, cache, previous)
    steps = {s: row["blocks"] for s, row in p.summary().items() if row["blocks"] and s != "leveling"}
    return Quote(estimate_seconds(p, threads), steps, len(p.misses), bool(p.leveling), p)


def say_time(seconds: float) -> str:
    """"about 40 s", "about 3 min": rounded to what is worth saying."""
    s = max(1.0, seconds)
    if s < 10:
        return f"about {round(s)} s"
    if s < 90:
        return f"about {int(round(s / 5) * 5)} s"
    minutes = s / 60
    if minutes < 10:
        return f"about {round(minutes)} min"
    if minutes < 90:
        return f"about {int(round(minutes / 5) * 5)} min"
    hours = minutes / 60
    return f"about {hours:.1f} h"


def quote_for(job: copies.JobCopies, tl: Timeline, threads: int = 2) -> Quote | None:
    """What it would take to make `tl` from the job's blocks, worked out from the facts it kept (None when it kept none)."""
    f = job_facts(job)
    if f is None or not job.has_blocks:
        return None
    try:
        inputs = facts_inputs(f, job.source_id)
        tl = Timeline.from_json(tl.to_json(), seconds=f["n"] / SR)
        tl, _ = drop_missing_echo(tl, inputs)
        tl = resolve_known(tl, job.editor.get("resolved"))
        return quote(tl, inputs, blocks.FlacCache(job.dir / copies.BLOCKS_DIR, None, create=False), previous_of(job, f["n"] / SR), threads)
    except (TimelineError, editor.EditorError):
        return None


def resolve_known(tl: Timeline, known: dict | None) -> Timeline:
    """The timeline with the decided removers filled in; the rest "automatic" stands for the light remover (a quote's guess)."""
    out = tl.copy()
    for p in out.layers["noise"]:
        if p.settings.get("remover") == "auto":
            r = (known or {}).get(part_key(p.start, p.end))
            p.settings["remover"] = r if r in ("light", "heavy") else "light"
    return out


def previous_of(job: copies.JobCopies, seconds: float) -> Timeline | None:
    return stored_timeline(job, seconds)


def runner(threads: int | None = None, device: str = "cpu") -> editor.Runner:
    return editor.Runner(
        lib_path=resources.libdf_path(), model_path=resources.model_path(), heavy_model_path=resources.heavy_noise_model_path(),
        restore_model_dir=resources.voice_model_dir(), device=device, threads=threads or engine.worker_threads())


# ---------------------------------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------------------------------
def decode(ff, src: Path, info, work: Path, *, reference: bool, cancel: threading.Event | None = None,
           source_id: str) -> editor.Inputs:
    """The original as the editor reads it (48 kHz, one or two channels), and the file's second track when `reference` and it has one."""
    x = pipeline.decode_for_clean(ff, src, info, engine.MAX_SECONDS, work_dir=work)
    ref = None
    if reference and info.audio_tracks >= 2:
        ref = ff.decode_to_file(src, info, SR, 1, engine.MAX_SECONDS, work / "reference.f32", track=1)[:, 0]
    return editor.Inputs(x, source_id, reference=ref, reference_id=REFERENCE_ID if ref is not None else None)


def settle(tl: Timeline, inputs: editor.Inputs, known: dict | None, cancel: threading.Event | None = None) -> tuple[Timeline, dict]:
    """The timeline with every "automatic" remover decided (the ones decided before come from `known`, so a quote and its run agree), and
    {part: remover} of the parts that were automatic."""
    out = tl.copy()
    known = known or {}
    autos = []
    for p in out.layers["noise"]:
        if p.settings.get("remover") == "auto":
            r = known.get(part_key(p.start, p.end))
            if r in ("light", "heavy"):
                p.settings["remover"] = r
            autos.append(part_key(p.start, p.end))
    if any(p.settings.get("remover") == "auto" for p in out.layers["noise"]):
        out = editor.resolve_removers(out, inputs, resources.libdf_path(), resources.model_path(), cancel)
    return out, {k: next(p.settings["remover"] for p in out.layers["noise"] if part_key(p.start, p.end) == k) for k in autos}


def drop_missing_echo(tl: Timeline, inputs: editor.Inputs) -> tuple[Timeline, str | None]:
    """A file with one audio track has nothing to remove echo with: the echo layer is left out, and the notice says so."""
    if tl.layers["echo"] and inputs.reference is None:
        out = tl.copy()
        out.layers["echo"] = []
        return out, "only one audio track, so no speaker echo was removed"
    return tl, None


def render_timeline(
    tl: Timeline, inputs: editor.Inputs, cache, *, work: Path, threads: int, known: dict | None = None,
    progress: Callable[[float], None] | None = None, cancel: threading.Event | None = None, out: np.ndarray | None = None,
    previous: Timeline | None = None,
) -> tuple[np.ndarray, Timeline, dict, str | None, dict]:
    """Decide the automatic removers, make the blocks the cache lacks and compose the result. Returns (the audio, the timeline as it ran,
    {part: remover} decided, a notice or None, {step: blocks made})."""
    tl, notice = drop_missing_echo(tl, inputs)
    if progress:
        progress(0.01)
    tl, decided = settle(tl, inputs, known, cancel)
    p = editor.plan(tl, inputs, cache, previous)
    made = {s: row["blocks"] for s, row in p.summary().items() if row["blocks"] and s != "leveling"}
    if p.misses:
        missing = [t.step for t in p.misses]
        if "rebuild" in missing and resources.voice_model_dir() is None:
            raise MediaError("The model that rebuilds the voice is not installed. Download it in Settings.")
        if not resources.noise_ready() and any(s in ("light", "heavy") for s in missing):
            raise MediaError("The noise removal models are not installed.")
    y = editor.render(tl, inputs, cache, runner(threads), out=out, work_dir=work,
                      progress=(lambda f: progress(0.02 + 0.98 * f)) if progress else None, cancel=cancel)
    return y, tl, decided, notice, made


def remember(job: copies.JobCopies, tl: Timeline, decided: dict | None = None) -> None:
    """The job's latest result was made from this timeline (and `decided` removers)."""
    ed = job.editor
    ed["timeline"] = tl.to_json()
    ed["resolved"] = {**(ed.get("resolved") or {}), **(decided or {})}
    job.save()


# ---------------------------------------------------------------------------------------------------------------------
# Old jobs
# ---------------------------------------------------------------------------------------------------------------------
def needs_migration(job: copies.JobCopies) -> bool:
    """The job was made before the block cache and still has saved copies of that kind."""
    return not job.has_blocks and (job.has(copies.NOISE) or job.has(copies.REBUILT))


def migrate(job: copies.JobCopies, ff, inputs: editor.Inputs, work: Path, *, cancel: threading.Event | None = None,
            progress: Callable[[float], None] | None = None) -> bool:
    """Turn an old job's saved copies (the noise removal at full strength, the rebuilt voice, the recording without the speaker echo) into
    blocks of the cache, then remove the copies. The blocks are the old copies' samples; they were made by the same models, cut to the
    grid. Nothing is removed unless every block was written. Returns whether the job was changed."""
    if not needs_migration(job):
        return False
    n, ch = inputs.n, inputs.channels
    if job.samples and job.samples != n:
        raise MediaError("The original no longer matches the saved copies (it is not the same length).")
    echo = bool(job.echo and job.has(copies.INPUT) and inputs.reference is not None)
    if job.echo and not echo:
        return False  # the recording without the echo was not kept: blocks made from it could not be found again
    cache = job.block_cache(ff)
    seconds = n / SR
    jobs: list[tuple[str, Timeline, tuple[str, ...]]] = []
    if job.has(copies.NOISE):
        entry = job.entry(copies.NOISE)
        profile = entry.get("profile") if entry.get("profile") in ("light", "heavy") else "light"
        jobs.append((copies.NOISE, Timeline.whole_file(seconds, echo=echo, noise={"strength_db": 100.0, "remover": profile}, leveling=False),
                     ("light", "heavy")))
    if job.has(copies.REBUILT):
        jobs.append((copies.REBUILT, Timeline.whole_file(seconds, echo=echo, noise=False, rebuild=True, leveling=False), ("rebuild",)))
    if echo:
        jobs.append((copies.INPUT, Timeline.whole_file(seconds, echo=True, noise=False, leveling=False), ("echo",)))
    for k, (kind, tl, steps) in enumerate(jobs):
        entry = job.entry(kind)
        if entry is None:
            continue
        if cancel is not None and cancel.is_set():
            raise engine.Cancelled()
        full = np.zeros((n, ch), np.float32)
        kept.load(ff, entry, job.dir, full, work / f"migrate-{kind}.f32")
        for t in editor.plan(tl, inputs, None).tasks:
            if t.step in steps:
                col = 0 if t.channel == "mid" else int(t.channel)
                cache.put(t.key, np.asarray(full[t.s:t.e, col]))
        del full
        if progress:
            progress((k + 1) / len(jobs))
    job.set_blocks(n, echo)
    for kind in (copies.NOISE, copies.REBUILT, copies.INPUT):
        job.drop(kind)
    return True


# ---------------------------------------------------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------------------------------------------------
def stored_suggestions(job: copies.JobCopies) -> dict | None:
    s = job.editor.get("suggestions")
    return s if isinstance(s, dict) and isinstance(s.get("items"), list) else None


def find_suggestions(x: np.ndarray, *, threads: int, cancel: threading.Event | None = None,
                     progress: Callable[[float], None] | None = None) -> dict:
    """Look for parts of the recording that other settings would suit (acw_common.suggest), as the JSON the editor keeps: every item has
    an id and a state ("open", "applied", "dismissed")."""
    s = suggest.suggest(x, resources.libdf_path(), resources.model_path(), threads=threads, cancel=cancel, progress=progress)
    items = [{**sg.to_json(), "id": f"s{i}", "state": "open"} for i, sg in enumerate(s.suggestions)]
    return {"seconds": s.seconds, "whole": s.to_json()["whole"], "items": items}


def keep_suggestions(job: copies.JobCopies, data: dict | None) -> None:
    job.editor["suggestions"] = data
    job.save()


def scratch_dir(parent: Path, need_gb: float) -> Path:
    """A work folder for the editor's scratch files (the decoded recording, the blocks laid out for listening)."""
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(dir=engine._work_parent(parent, need_gb), prefix=f"{engine._WORK_PREFIX}{os.getpid()}-"))


def remove_work(work: Path) -> None:
    try:
        shutil.rmtree(work)
    except FileNotFoundError:
        pass
    except OSError:
        threading.Thread(target=engine._remove_work_dir, args=(work,), name="scratch-cleanup", daemon=True).start()


__all__ = [
    "EchoWholeFile", "cache_args", "echo_input_id", "facts_of", "facts_inputs", "peaks_of", "store_facts", "job_facts", "seconds_of",
    "baseline_timeline", "current_timeline", "made_from_timeline", "quote", "say_time", "render_timeline", "remember", "migrate",
    "needs_migration", "find_suggestions", "keep_suggestions", "Quote",
]
