"""The core of the per-part editor: plan what a timeline needs, quote it, run it, compose it.

    inputs = editor.Inputs(x, source_id, reference=track2, reference_id=...)       # the recording (and the speaker track)
    tl = timeline.Timeline.whole_file(inputs.seconds, noise={"strength_db": 24}).paint("rebuild", 30, 60)
    p = editor.plan(tl, inputs, cache)            # every block the timeline needs; p.misses: the ones not in the cache yet
    p.misses[0].seconds, ...                      # each with its seconds, so the server can quote and run only those
    editor.billable(p)                            # billable seconds per step + per-run fixed amounts in equivalent seconds
    y = editor.render(tl, inputs, cache, editor.Runner(lib_path, model_path, ...))   # (samples, channels) float32

THE MODEL STEPS run on the block grid of blocks.py and every block is cached, keyed by the step, the model, the identity of the
audio it read and where it is. A block's identity includes which echo-removal parts touch the audio it read (the noise removers and
Voice rebuild read the echo-free recording), but not the strength: strength is a blend of a full-strength block with its
input, so a strength change never needs a model run. Where noise removal and Voice rebuild both cover a moment, Voice rebuild wins
and the noise removal there is not run.

JOINS where one part meets another (the measures are in the editor experiments, docs/research):

    kind A, both sides keep the waveform of the recording (echo removal, noise removal on / off / light vs heavy / another
        strength, Leveling on / off): a linear ramp of 50 ms, 100 ms where the noise floor changes (strength or on / off of the
        noise removal, and Leveling). For a strength change inside the noise layer this IS a ramp of the blend amount.
    kind B, Voice rebuild against anything else: an equal-power crossfade, the join moved to the quietest 60 ms of the rebuilt
        voice within +-0.3 s when that is a pause (12 dB under the speech) and then 100 ms long, else 250 ms; the loudness of
        the two sides is matched over 2 s each (the gain clamped to +-9 dB), as a smooth gain on the rebuilt side that fades
        out over 2 s.

LEVELING per part: the loudness (and the level match) is measured over the part plus max(1 x its length, 10 s) each side, clamped
to the file; the high-pass, compressor and limiter run over the same stretch (2 s of context is all they need); a part with
Leveling off stays as it is. Whole-file Leveling (the default) is `clarity.process` over the file: today's output.

Everything is streamed through pieces of at most 10 s into `out` (a memory map when `work_dir` is given), the cache holds the
blocks, so a multi-hour file needs a few buffers, not copies of the file per step.
"""
from __future__ import annotations

import math
import shutil
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import blocks, clarity, dfn, dpdfnet, pipeline, restore
from .aec import echo_block
from .timeline import Timeline, TimelineError

SR = blocks.SAMPLE_RATE
STEPS = ("echo", "light", "heavy", "rebuild", "leveling")  # the billing steps, in the order they run

# ---- joins
RAMP_S = 0.05  # kind A
RAMP_FLOOR_S = 0.10  # kind A where the noise floor changes
LEVEL_RAMP_S = 0.10  # Leveling on / off
SNAP_S = 0.3  # kind B: how far the join may move to a pause
PAUSE_D_S = 0.10  # ... the crossfade when a pause was found
HARD_D_S = 0.25  # ... and when not
MATCH_S = 2.0  # loudness matched over this much of each side
MATCH_MAX_DB = 9.0
EDGE_S = 2.0  # the matching gain fades out over this much of the rebuilt part
PAUSE_DROP_DB = 12.0  # a pause is this far under the speech around it
MARGIN_A = 2 * int(RAMP_FLOOR_S * SR)  # samples a kind A join reads beyond a part (twice what it needs)
MARGIN_B = int((SNAP_S + HARD_D_S / 2) * SR) + 2 * 960 + 4800  # samples a kind B join reads beyond a part (shift, half the crossfade, the pause search)
# ---- Leveling
LEVEL_CONTEXT_S = 2.0
LEVEL_MIN_SIDE_S = 10.0
PIECE = 10 * SR  # composition works in pieces of this many samples
FRAME = 960  # 20 ms, the join measures' frame


class EditorError(ValueError):
    """The timeline cannot be run on these inputs (the message says why)."""


# ---------------------------------------------------------------------------------------------------------------------
# Pricing inputs
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StepCost:
    resource: str  # "cpu" or "gpu"
    fixed: float  # seconds of that resource per run of the step (a fresh sandboxed stage)
    per_second: float  # seconds of it per second of audio

    @property
    def fixed_equivalent_seconds(self) -> float:
        """The per-run fixed amount as seconds of audio at the step's own rate: fixed / per-second cost."""
        return self.fixed / self.per_second


#: Measured in a fresh sandboxed stage (docs/research editor experiments, q5). Light remover + Leveling together: 1.55 CPU-s fixed
#: and 0.129 CPU-s per second; Leveling alone 1.28 / 0.013; so the light remover's own part is 0.27 / 0.116 (they add up). Echo
#: removal: +0.12 CPU-s fixed (the per-second figure is the stage difference, 1.02 CPU-s over 30 s of audio). Heavy remover on the
#: GPU: 2.9 GPU-s fixed, 0.386 per second; Voice rebuild on the GPU: 4.3 / 0.203.
STEP_COSTS = {
    "echo": StepCost("cpu", 0.12, 0.034),
    "light": StepCost("cpu", 1.55 - 1.28, 0.129 - 0.013),
    "heavy": StepCost("gpu", 2.9, 0.386),
    "rebuild": StepCost("gpu", 4.3, 0.203),
    "leveling": StepCost("cpu", 1.28, 0.013),
}


# ---------------------------------------------------------------------------------------------------------------------
# The recording
# ---------------------------------------------------------------------------------------------------------------------
class Inputs:
    """The audio an edit works on. `source`: the recording (samples, channels) float32 at 48 kHz (a memory map is fine);
    `source_id`: anything that identifies its samples (a hash of the file, a job id plus the upload's hash): it is part of every
    cache key, so two different recordings must not share one. `reference` / `reference_id`: what the speakers played (the file's
    second track, mono), for echo removal."""

    def __init__(self, source: np.ndarray, source_id: str, reference: np.ndarray | None = None, reference_id: str | None = None):
        if source.dtype != np.float32 or source.ndim not in (1, 2):
            raise ValueError("source must be float32, (samples,) or (samples, channels)")
        self.source = source[:, None] if source.ndim == 1 else source
        self.source_id = str(source_id)
        if reference is not None:
            reference = reference[:, 0] if reference.ndim == 2 else reference
            if reference_id is None:
                raise ValueError("a reference needs reference_id")
        self.reference = reference
        self.reference_id = None if reference_id is None else str(reference_id)
        self._plan: list[int | None] | None = None
        self._spans: dict[int, list[tuple[int, int]]] = {}
        self._mid: bool | None = None

    @property
    def n(self) -> int:
        return len(self.source)

    @property
    def channels(self) -> int:
        return self.source.shape[1]

    @property
    def seconds(self) -> float:
        return self.n / SR

    @property
    def plan(self) -> list[int | None]:
        """pipeline.channel_plan of the recording: which channel's result each output channel gets."""
        if self._plan is None:
            self._plan = pipeline.channel_plan(self.source)
        return self._plan

    @property
    def todo(self) -> list[int]:
        return sorted({c for c in self.plan if c is not None})

    def spans(self, ch: int) -> list[tuple[int, int]]:
        """pipeline.sound_spans of a channel: what the noise removers are run on (long digital silence is not)."""
        if ch not in self._spans:
            self._spans[ch] = pipeline.sound_spans(self.source[:, ch])
        return self._spans[ch]

    @property
    def rebuild_mid(self) -> bool:
        """Voice rebuild of a stereo recording whose sides are the same voice is done once, on the mix (restore.restore)."""
        if self._mid is None:
            self._mid = self.channels == 2 and restore._side_db(self.source) < restore._MONO_SIDE_DB
        return self._mid

    def rebuild_jobs(self) -> list:
        """The signals Voice rebuild runs on: "mid" (both channels get its result) or the channels one by one."""
        return ["mid"] if self.rebuild_mid else list(range(self.channels))


# ---------------------------------------------------------------------------------------------------------------------
# Layout: the timeline as segments and joins
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Seg:
    a: int
    b: int
    kind: str  # echo layer: "off" | "echo"; main: "pass" | "noise" | "rebuild"; leveling: "off" | "level"
    st: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.b - self.a


@dataclass
class Join:
    t: int  # the boundary
    kind: str  # "A" | "B"
    left: int
    right: int
    d: int = 0  # samples of the ramp / crossfade (A: set here; B: set when the join is made)
    z0: int = 0
    z1: int = 0
    shape: str = "lin"  # "lin" | "ep"
    gain_db: float = 0.0  # B: the loudness match applied to the rebuilt side
    rb_right: bool = False  # B: the rebuilt segment is the right one


def _fill(parts: list[tuple[int, int, dict]], n: int, on: str, off: str) -> list[Seg]:
    segs, pos = [], 0
    for a, b, st in parts:
        if a > pos:
            segs.append(Seg(pos, a, off))
        segs.append(Seg(a, b, on, dict(st)))
        pos = b
    if pos < n or not segs:
        segs.append(Seg(pos, n, off))
    return segs


def _main_segments(noise: list, rebuild: list, n: int) -> list[Seg]:
    pts = sorted({0, n, *(a for a, _, _ in noise + rebuild), *(b for _, b, _ in noise + rebuild)})
    segs: list[Seg] = []
    for a, b in zip(pts, pts[1:]):
        if b <= a:
            continue
        if any(x <= a < y for x, y, _ in rebuild):
            kind, st = "rebuild", {}  # Voice rebuild wins where both cover the moment
        else:
            nz = next((s for x, y, s in noise if x <= a < y), None)
            if nz is None:
                kind, st = "pass", {}
            elif nz.get("browser"):
                kind, st = "browser", {"browser": True}  # the browser's own noise removal, uploaded (BrowserAudio): no model runs
            else:
                if nz["remover"] not in ("light", "heavy"):
                    raise EditorError("the noise remover of every part must be light or heavy: resolve 'auto' first (resolve_removers)")
                kind, st = "noise", {"remover": nz["remover"], "strength_db": nz["strength_db"]}
        if segs and segs[-1].kind == kind and segs[-1].st == st:
            segs[-1].b = b
        else:
            segs.append(Seg(a, b, kind, st))
    return segs


def _a_ramp(l: Seg, r: Seg, layer: str) -> int:
    if layer == "echo":
        d = RAMP_S
    elif layer == "level":
        d = LEVEL_RAMP_S
    else:  # the noise floor changes with the noise removal on / off and with its strength
        floor = l.kind != r.kind or l.st.get("strength_db") != r.st.get("strength_db")
        d = RAMP_FLOOR_S if floor else RAMP_S
    d = min(int(round(d * SR)), l.length, r.length)
    return d - d % 2


def _a_joins(segs: list[Seg], layer: str) -> list[Join]:
    joins = []
    for i in range(len(segs) - 1):
        j = Join(segs[i].b, "A", i, i + 1, _a_ramp(segs[i], segs[i + 1], layer))
        j.z0 = j.t - j.d // 2
        j.z1 = j.z0 + j.d
        joins.append(j)
    return joins


def _main_joins(segs: list[Seg]) -> list[Join]:
    joins = []
    for i in range(len(segs) - 1):
        l, r = segs[i], segs[i + 1]
        if "rebuild" in (l.kind, r.kind):
            joins.append(Join(l.b, "B", i, i + 1, shape="ep", rb_right=(r.kind == "rebuild")))
        else:
            j = Join(l.b, "A", i, i + 1, _a_ramp(l, r, "main"))
            j.z0 = j.t - j.d // 2
            j.z1 = j.z0 + j.d
            joins.append(j)
    return joins


class Layout:
    """A timeline on a recording: the segments of each layer and the joins between them (sample positions)."""

    def __init__(self, tl: Timeline, inputs: Inputs):
        n = inputs.n
        if abs(tl.seconds - n / SR) > 0.01 and n:
            raise EditorError(f"the timeline is for {tl.seconds} s, the recording is {n / SR:.3f} s")
        self.n = n
        self.timeline = tl
        echo = tl.samples("echo", n)
        if echo and inputs.reference is None:
            raise EditorError("echo removal needs the file's second audio track (the reference)")
        try:
            tl.check()
        except TimelineError as e:  # (an echo part over a part the browser cleaned)
            raise EditorError(str(e)) from None
        self.echo_parts = [(a, b) for a, b, _ in echo]
        self.echo = _fill(echo, n, "echo", "off")
        self.echo_joins = _a_joins(self.echo, "echo")
        noise, rebuild = tl.samples("noise", n), tl.samples("rebuild", n)
        self.main = _main_segments(noise, rebuild, n)
        self.main_joins = _main_joins(self.main)
        self.level_parts = [(a, b, s) for a, b, s in tl.samples("leveling", n)]
        # parts that touch with the same settings are one part (their Leveling is one measurement)
        merged: list = []
        for a, b, s in self.level_parts:
            if merged and merged[-1][1] == a and merged[-1][2] == s:
                merged[-1] = (merged[-1][0], b, s)
            else:
                merged.append((a, b, s))
        self.level_parts = merged
        self.level = _fill(self.level_parts, n, "level", "off")
        self.level_joins = _a_joins(self.level, "level")

    # ---- what each step is asked for (billable ranges)
    def asked(self) -> dict[str, list[tuple[int, int]]]:
        out: dict[str, list[tuple[int, int]]] = {s: [] for s in STEPS}
        out["echo"] = list(self.echo_parts)
        for s in self.main:
            if s.kind == "noise":
                out[s.st["remover"]].append((s.a, s.b))
            elif s.kind == "rebuild":
                out["rebuild"].append((s.a, s.b))
        out["leveling"] = [(a, b) for a, b, _ in self.level_parts]
        return out

    def whole_file_leveling(self) -> bool:
        return len(self.level_parts) == 1 and self.level_parts[0][0] == 0 and self.level_parts[0][1] == self.n

    def measure_span(self, a: int, b: int) -> tuple[int, int]:
        """Where a leveled part [a, b) is measured: the part and max(its length, 10 s) each side, clamped to the file."""
        e = max(b - a, int(LEVEL_MIN_SIDE_S * SR))
        return max(0, a - e), min(self.n, b + e)


# ---------------------------------------------------------------------------------------------------------------------
# Plan: the blocks a timeline needs
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Task:
    """One block of one step for one channel. [s, e) is the block, [lo, hi) what the model reads for it (context)."""
    step: str  # "echo" | "light" | "heavy" | "rebuild"
    channel: int | str  # the channel, or "mid" (Voice rebuild of the mix)
    lo: int
    s: int
    e: int
    hi: int
    key: str
    index: int | None = None  # Voice rebuild: the block's number on its grid
    asked: tuple = ()  # the sample ranges inside [s, e) that parts asked for (the billable ones, not the join margins)

    @property
    def seconds(self) -> float:
        """The block: its whole length (what it adds to the cache)."""
        return (self.e - self.s) / SR

    @property
    def asked_seconds(self) -> float:
        return blocks.union_seconds(list(self.asked))

    @property
    def compute_seconds(self) -> float:
        """What the model reads for it, context included (the context is ours, not billed)."""
        return (self.hi - self.lo) / SR


class _Identity:
    """The identity of the audio a model step reads: the recording, plus the echo-removal parts that touch it."""

    PAD = 2 * int(RAMP_FLOOR_S * SR)

    def __init__(self, inputs: Inputs, echo_parts: list[tuple[int, int]]):
        self.inputs, self.parts = inputs, echo_parts

    def __call__(self, lo: int, hi: int) -> str:
        ps = [(max(a, lo - self.PAD), min(b, hi + self.PAD)) for a, b in self.parts if b > lo - self.PAD and a < hi + self.PAD]
        if not ps:
            return self.inputs.source_id
        return f"{self.inputs.source_id}~{self.inputs.reference_id}~e" + ",".join(f"{a}-{b}" for a, b in ps)

    def echo(self) -> str:
        return f"{self.inputs.source_id}~{self.inputs.reference_id}"


def _extend(a: int, b: int, n: int, ma: int, mb: int) -> tuple[int, int]:
    return max(0, a - ma), min(n, b + mb)


class _Keys:
    """Cache keys and tasks for a layout (one place, so planning and running cannot disagree)."""

    def __init__(self, lay: Layout, inputs: Inputs):
        self.lay, self.inputs = lay, inputs
        self.ident = _Identity(inputs, lay.echo_parts)

    def noise_key(self, step: str, ch: int, lo: int, s: int, e: int, hi: int) -> str:
        return blocks.block_key(step, self.ident(lo, hi), ch, lo, s, e, hi)

    def echo_key(self, ch: int, lo: int, s: int, e: int, hi: int) -> str:
        return blocks.block_key("echo", self.ident.echo(), ch, lo, s, e, hi)

    def rebuild_key(self, job, b: int) -> str:
        lo, hi = restore.block_input_range(b, self.inputs.n)
        return blocks.rebuild_key(self.ident(lo, hi), job, b, self.inputs.n, restore._WINDOW_SECONDS, restore._OVERLAP_SECONDS)

    def tasks(self) -> list[Task]:
        lay, inputs, n = self.lay, self.inputs, self.inputs.n
        asked = lay.asked()
        out: list[Task] = []
        # echo removal: every part, every channel
        for ch in range(inputs.channels if lay.echo_parts else 0):
            seen = set()
            for a, b in lay.echo_parts:
                ra, rb = _extend(a, b, n, MARGIN_A, MARGIN_A)
                for lo, s, e, hi in blocks.grid_windows(0, n, blocks.grid_of("echo"), ra, rb):
                    if (s, e) in seen:
                        continue
                    seen.add((s, e))
                    out.append(Task("echo", ch, lo, s, e, hi, self.echo_key(ch, lo, s, e, hi),
                                    asked=tuple(blocks.intersect(asked["echo"], s, e))))
        # noise removal where it is wanted and Voice rebuild does not win; Voice rebuild
        for i, seg in enumerate(lay.main):
            if seg.kind not in ("noise", "rebuild"):
                continue
            ma = MARGIN_B if i > 0 and lay.main[i - 1].kind == "rebuild" or seg.kind == "rebuild" else (MARGIN_A if i > 0 else 0)
            mb = MARGIN_B if i + 1 < len(lay.main) and lay.main[i + 1].kind == "rebuild" or seg.kind == "rebuild" else (MARGIN_A if i + 1 < len(lay.main) else 0)
            ra, rb = _extend(seg.a, seg.b, n, ma, mb)
            if seg.kind == "noise":
                step = seg.st["remover"]
                grid = blocks.grid_of(step)
                for ch in inputs.todo:
                    for sa, sb in inputs.spans(ch):
                        for lo, s, e, hi in blocks.grid_windows(sa, sb, grid, ra, rb):
                            out.append(Task(step, ch, lo, s, e, hi, self.noise_key(step, ch, lo, s, e, hi),
                                            asked=tuple(blocks.intersect(asked[step], s, e))))
            else:
                size = restore.block_samples()
                for job in inputs.rebuild_jobs():
                    for b in range(ra // size, (rb - 1) // size + 1):
                        s, e = restore.block_bounds(b, n)
                        lo, hi = restore.block_input_range(b, n)
                        out.append(Task("rebuild", job, lo, s, e, hi, self.rebuild_key(job, b), index=b,
                                        asked=tuple(blocks.intersect(asked["rebuild"], s, e))))
        # the same block asked for by two segments (a join margin) is one task; its asked ranges are joined
        merged: dict[str, Task] = {}
        for t in out:
            old = merged.get(t.key)
            if old is None:
                merged[t.key] = t
            else:
                merged[t.key] = Task(t.step, t.channel, t.lo, t.s, t.e, t.hi, t.key, t.index, tuple(sorted(set(old.asked) | set(t.asked))))
        return list(merged.values())


@dataclass
class Plan:
    """What a timeline needs: every block (`tasks`), the ones the cache does not have (`misses`), the Leveling parts to run."""
    n: int
    tasks: list[Task]
    misses: list[Task]
    asked: dict[str, list[tuple[int, int]]]
    leveling: list[tuple[int, int]]  # leveled parts that have to be (re)made
    layout: Layout | None = field(default=None, repr=False)

    def blocks(self, step: str | None = None, missing: bool = True) -> list[Task]:
        src = self.misses if missing else self.tasks
        return [t for t in src if step is None or t.step == step]

    def summary(self) -> dict:
        """Per step: blocks to compute, the seconds they add, the context they also read, the billable seconds."""
        out = {}
        for step in STEPS:
            ts = self.blocks(step)
            if step == "leveling":
                secs = sum(b - a for a, b in self.leveling) / SR
                out[step] = {"blocks": len(self.leveling), "seconds": secs, "compute_seconds": secs,
                             "billable_seconds": math.ceil(secs - 1e-9) if secs > 0 else 0}
                continue
            out[step] = {"blocks": len(ts), "seconds": sum(t.seconds for t in ts), "compute_seconds": sum(t.compute_seconds for t in ts),
                         "billable_seconds": _billable_seconds(ts)}
        return out


def _billable_seconds(tasks: list[Task]) -> int:
    ranges = [r for t in tasks for r in t.asked]
    secs = blocks.union_seconds(ranges)
    return math.ceil(secs - 1e-9) if secs > 0 else 0


def plan(tl: Timeline, inputs: Inputs, cache: blocks.BlockCache | None = None, previous: Timeline | None = None) -> Plan:
    """Every block the timeline needs on this recording and, of those, the cache misses (`cache` None: all of them).
    `previous`: the timeline that was rendered before; the Leveling parts that it already made the same way (same settings, the same
    audio under their measure span) are not to be made again. The blocks are decided without touching the audio except to find
    its digital silence (inputs.spans)."""
    lay = Layout(tl, inputs)
    keys = _Keys(lay, inputs)
    tasks = keys.tasks()
    order = {s: i for i, s in enumerate(STEPS)}
    tasks.sort(key=lambda t: (order[t.step], str(t.channel), t.s))
    misses = [t for t in tasks if cache is None or not cache.has(t.key)]
    level = [(a, b) for a, b, _ in lay.level_parts]
    if previous is not None and lay.n:
        try:
            old = Layout(previous, inputs)
        except EditorError:
            old = None
        if old is not None:
            same = {(a, b, tuple(sorted(s.items()))) for a, b, s in old.level_parts if _level_signature(old, a, b) == _level_signature(lay, a, b)}
            level = [(a, b) for a, b, s in lay.level_parts if (a, b, tuple(sorted(s.items()))) not in same]
    return Plan(inputs.n, tasks, misses, lay.asked(), level, lay)


def _level_signature(lay: Layout, a: int, b: int) -> tuple:
    """What a leveled part's result depends on: its settings are compared by the caller; this is the audio under its measure span
    (and the context of the filters): the parts of the layers before Leveling that reach it."""
    ma, mb = lay.measure_span(a, b)
    ma, mb = max(0, ma - int(LEVEL_CONTEXT_S * SR)), min(lay.n, mb + int(LEVEL_CONTEXT_S * SR))
    sig = []
    for name, segs in (("echo", [(s.a, s.b, ()) for s in lay.echo if s.kind == "echo"]),
                       ("main", [(s.a, s.b, (s.kind, tuple(sorted(s.st.items())))) for s in lay.main if s.kind != "pass"])):
        sig.append((name, tuple((max(x, ma), min(y, mb), st) for x, y, st in segs if y > ma and x < mb)))
    return tuple(sig)


def billable(p: Plan) -> dict[str, dict]:
    """What the plan's missing blocks cost, step by step, for the server to price as `rate x (seconds + fixed-equivalent seconds)`:

        {step: {"seconds": whole seconds of the parts asked for, rounded UP (the context the models also read is ours),
                "fixed_equivalent_seconds": the step's per-run fixed cost as seconds of audio at the step's own rate,
                "resource": "cpu" | "gpu", "fixed": resource-seconds per run, "per_second": resource-seconds per audio second}}

    Only steps that have something to run are in it; there is no minimum charge. Light and heavy noise removal are separate steps
    (they have different rates)."""
    out = {}
    for step, row in p.summary().items():
        secs = row["billable_seconds"]
        if secs <= 0:
            continue
        c = STEP_COSTS[step]
        out[step] = {"seconds": secs, "fixed_equivalent_seconds": c.fixed_equivalent_seconds, "resource": c.resource,
                     "fixed": c.fixed, "per_second": c.per_second}
    return out


def compute_estimate(p: Plan) -> dict[str, float]:
    """The resource-seconds (CPU or GPU, by step) the plan's missing blocks will take: per-run fixed + context included."""
    out = {}
    for step, row in p.summary().items():
        if row["blocks"]:
            c = STEP_COSTS[step]
            out[step] = c.fixed + c.per_second * row["compute_seconds"]
    return out


def resolve_removers(tl: Timeline, inputs: Inputs, lib_path: str | Path, model_path: str | Path,
                     cancel: threading.Event | None = None) -> Timeline:
    """A copy of the timeline with every "auto" remover decided: the light remover, or the heavy one where the part's noise
    figure (pipeline.noise_figure_db, measured on the part) says it does clearly better (pipeline.automatic_choices)."""
    out = tl.copy()
    for part in out.layers["noise"]:
        if part.settings.get("remover") == "auto":
            a, b = int(round(part.start * SR)), min(inputs.n, int(round(part.end * SR)))
            figure = pipeline.noise_figure_db(inputs.source[a:b], lib_path, model_path, cancel)
            part.settings["remover"] = pipeline.automatic_choices(figure)[0]
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Running the blocks
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Runner:
    """Where the models are and how to run them (the arguments of pipeline.clean_audio)."""
    lib_path: str | Path | None = None
    model_path: str | Path | None = None
    heavy_model_path: str | Path | None = None
    restore_model_dir: str | Path | None = None
    device: str = "cpu"
    threads: int = 2


def _run_tasks(tasks: list[Task], lay: Layout, inputs: Inputs, E: np.ndarray, cache: blocks.BlockCache, runner: Runner | None,
               progress: Callable[[float], None] | None, cancel: threading.Event | None) -> None:
    """Make the blocks of `tasks` (echo-free audio `E` for the models) and put them in the cache."""
    if not tasks:
        return
    if runner is None:
        raise EditorError("blocks have to be computed but no Runner was given")
    weight = {t.key: t.compute_seconds * STEP_COSTS[t.step].per_second for t in tasks}
    total = sum(weight.values()) or 1.0
    done = 0.0

    def advance(step_done: float, step_total: float, base: float, share: float) -> None:
        if progress:
            progress(min((base + share * step_done / max(step_total, 1e-9)) / total, 1.0))

    groups: dict[tuple, list[Task]] = {}
    for t in tasks:
        groups.setdefault((t.step, t.channel), []).append(t)
    for (step, ch), ts in sorted(groups.items(), key=lambda kv: (STEPS.index(kv[0][0]), str(kv[0][1]))):
        share = sum(weight[t.key] for t in ts)
        if cancel is not None and cancel.is_set():
            raise dfn.Cancelled()
        prog = lambda f, base=done, share=share: advance(f, 1.0, base, share)  # noqa: E731
        if step == "echo":
            for k, t in enumerate(ts):
                if cancel is not None and cancel.is_set():
                    raise dfn.Cancelled()
                cache.put(t.key, echo_block(inputs.source[:, ch], inputs.reference, t.lo, t.s, t.e, t.hi))
                prog((k + 1) / len(ts))
        elif step == "light":
            if runner.lib_path is None or runner.model_path is None:
                raise EditorError("the light noise remover needs lib_path and model_path")
            dfn.run_windows(E[:, ch], runner.lib_path, runner.model_path, [(t.lo, t.s, t.e, t.hi) for t in ts],
                            lambda i, y, ts=ts: cache.put(ts[i].key, y), threads=runner.threads, progress=prog, cancel=cancel)
        elif step == "heavy":
            dpdfnet.run_windows(E, [(t.lo, t.s, t.e, t.hi) for t in ts], lambda i, y, ts=ts: cache.put(ts[i].key, y), channel=ch,
                                model_path=runner.heavy_model_path, threads=runner.threads, device=runner.device, progress=prog, cancel=cancel)
        else:
            if runner.restore_model_dir is None or not restore.available(runner.restore_model_dir):
                raise FileNotFoundError(f"voice rebuilding needs Sidon's models (not in {runner.restore_model_dir})")
            src = restore._Mid(E) if ch == "mid" else E[:, ch]
            by_index = {t.index: t for t in ts}
            restore.restore_blocks(src, runner.restore_model_dir, list(by_index), lambda b, y: cache.put(by_index[b].key, y),
                                   threads=runner.threads, device=runner.device, progress=prog, cancel=cancel)
        done += share
    if progress:
        progress(1.0)


class _LRU:
    def __init__(self, cache: blocks.BlockCache, size: int = 6):
        self.cache, self.size, self.d = cache, size, {}

    def get(self, key: str) -> np.ndarray:
        if key in self.d:
            self.d[key] = self.d.pop(key)
            return self.d[key]
        v = self.cache.get(key)
        if v is None:
            raise EditorError("a block that should be in the cache is not (the cache lost it?)")
        self.d[key] = v
        while len(self.d) > self.size:
            self.d.pop(next(iter(self.d)))
        return v


class BrowserAudio:
    """What the browser made of the noise parts that say `browser: true` (the noise layer's output there, instead of a remover's): pieces
    (first sample, samples (n, channels) float32 at 48 kHz; a memory map is fine), each covering exactly its part. Outside the pieces the
    recording itself is what a signal reads."""

    def __init__(self, pieces: list[tuple[int, np.ndarray]]):
        self.pieces = sorted(((int(a), x) for a, x in pieces), key=lambda p: p[0])

    def covers(self, a: int, b: int) -> bool:
        pos = a
        for first, x in self.pieces:
            if first <= pos < first + len(x):
                pos = first + len(x)
        return pos >= b

    def fill(self, out: np.ndarray, a: int) -> None:
        """Copy the pieces' samples over `out` (the recording's samples from `a`)."""
        b = a + len(out)
        for first, x in self.pieces:
            x0, x1 = max(a, first), min(b, first + len(x))
            if x1 > x0:
                out[x0 - a:x1 - a] = x[x0 - first:x1 - first]


# ---------------------------------------------------------------------------------------------------------------------
# Joins: measures
# ---------------------------------------------------------------------------------------------------------------------
def _mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1, dtype=np.float64) if x.ndim == 2 else x.astype(np.float64)


def _frame_power(x: np.ndarray) -> np.ndarray:
    k = len(x) // FRAME
    return np.mean(x[:k * FRAME].reshape(k, FRAME) ** 2, axis=1) + 1e-12 if k else np.zeros(0)


def _active_level_db(x: np.ndarray) -> float:
    """The level of the voice in `x`: the mean power of the louder half of its 20 ms frames, in dB."""
    p = np.sort(_frame_power(_mono(x)))
    return float(10 * np.log10(np.mean(p[len(p) // 2:]))) if len(p) else -120.0


def find_pause(lead: np.ndarray, lead_start: int, t0: int, max_shift: int) -> tuple[int, bool]:
    """The join position for a kind B join at `t0`: the centre of the quietest 60 ms of `lead` (mono, whose first sample is at
    `lead_start` in the file) within +-max_shift, if it is a pause (PAUSE_DROP_DB under the loud frames around), else `t0`.
    Returns (position, found a pause)."""
    a, b = t0 - max_shift, t0 + max_shift
    lo, hi = a - 2400 - lead_start, b + 2400 - lead_start
    if lo < 0 or hi > len(lead) or hi - lo < 4 * FRAME:
        return t0, False
    p = _frame_power(lead[lo:hi])
    sm = np.convolve(p, np.ones(3) / 3, "valid")
    centers = a - 2400 + np.arange(len(sm)) * FRAME + FRAME + FRAME // 2
    near = np.abs(centers - t0) <= max_shift
    if not len(sm) or not near.any():
        return t0, False
    i = int(np.flatnonzero(near)[np.argmin(sm[near])])
    ref = np.sort(_frame_power(lead))[-100:]
    if 10 * np.log10(sm[i]) < 10 * np.log10(np.mean(ref)) - PAUSE_DROP_DB:
        return int(centers[i]), True
    return t0, False


# ---------------------------------------------------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------------------------------------------------
class _Render:
    def __init__(self, tl: Timeline, inputs: Inputs, cache: blocks.BlockCache, runner: Runner | None, out: np.ndarray | None,
                 work_dir: Path | None, progress, cancel, browser: BrowserAudio | None = None):
        self.tl, self.inputs, self.cache, self.runner = tl, inputs, cache, runner
        self.lay = Layout(tl, inputs)
        self.browser = browser
        for seg in self.lay.main:
            if seg.kind == "browser" and (browser is None or not browser.covers(seg.a, seg.b)):
                raise EditorError("a part cleaned in the browser has no uploaded audio")
        for _, x in (browser.pieces if browser else []):
            if x.ndim != 2 or x.shape[1] != inputs.channels:
                raise EditorError("the audio cleaned in the browser must have the recording's channels")
        self.keys = _Keys(self.lay, inputs)
        self.n, self.ch = inputs.n, inputs.channels
        self.work_dir, self.progress, self.cancel = work_dir, progress, cancel
        self.lru = _LRU(cache)
        self.temp: list[Path] = []  # (before the first `alloc`: it records the scratch files it makes here)
        self.out = out if out is not None else self.alloc("out")
        self._scale: float | None = None

    # ---- buffers
    def alloc(self, name: str, n: int | None = None) -> np.ndarray:
        shape = (self.n if n is None else n, self.ch)
        if self.work_dir is None:
            return np.zeros(shape, np.float32)
        p = Path(self.work_dir) / f"editor-{name}-{uuid.uuid4().hex[:8]}.npy"
        self.temp.append(p)
        return np.lib.format.open_memmap(p, mode="w+", dtype=np.float32, shape=shape)

    def check(self) -> None:
        if self.cancel is not None and self.cancel.is_set():
            raise dfn.Cancelled()

    # ---- reading blocks
    def noise_full(self, step: str, c: int, a: int, b: int) -> np.ndarray:
        """The full-strength noise remover's output for channel `c` over [a, b) (zero in the digital silence it skips)."""
        out = np.zeros(b - a, np.float32)
        grid = blocks.grid_of(step)
        for sa, sb in self.inputs.spans(c):
            for lo, s, e, hi in blocks.grid_windows(sa, sb, grid, a, b):
                blk = self.lru.get(self.keys.noise_key(step, c, lo, s, e, hi))
                x0, x1 = max(a, s), min(b, e)
                if x1 > x0:
                    out[x0 - a:x1 - a] = blk[x0 - s:x1 - s]
        return out

    def echo_full(self, c: int, a: int, b: int) -> np.ndarray:
        out = np.zeros(b - a, np.float32)
        for lo, s, e, hi in blocks.grid_windows(0, self.n, blocks.grid_of("echo"), a, b):
            blk = self.lru.get(self.keys.echo_key(c, lo, s, e, hi))
            x0, x1 = max(a, s), min(b, e)
            if x1 > x0:
                out[x0 - a:x1 - a] = blk[x0 - s:x1 - s]
        return out

    def rebuild_raw(self, job, a: int, b: int) -> np.ndarray:
        out = np.zeros(b - a, np.float32)
        size = restore.block_samples()
        for k in range(a // size, (b - 1) // size + 1):
            s, e = restore.block_bounds(k, self.n)
            blk = self.lru.get(self.keys.rebuild_key(job, k))
            x0, x1 = max(a, s), min(b, e)
            if x1 > x0:
                out[x0 - a:x1 - a] = blk[x0 - s:x1 - s]
        return out

    # ---- the signals of the segments
    def rebuild_scale(self) -> float:
        """restore.restore keeps the rebuilt file under 0.99: a gain over everything when it overshoots. Here: the peak of the
        rebuilt blocks the timeline uses (all of them for a whole-file rebuild: the same number)."""
        if self._scale is None:
            peak = 0.0
            for t in self.keys.tasks():
                if t.step == "rebuild":
                    peak = max(peak, float(np.max(np.abs(self.lru.get(t.key)))) if t.e > t.s else 0.0)
            self._scale = 0.99 / peak if peak > 0.99 else 1.0
        return self._scale

    def rebuild_sig(self, seg: Seg, a: int, b: int) -> np.ndarray:
        """The rebuilt voice over [a, b) (clamped to what the blocks of the segment cover), the model's level scaled by
        rebuild_scale, all channels; zero outside what was made."""
        lo, hi = max(0, seg.a - MARGIN_B), min(self.n, seg.b + MARGIN_B)
        out = np.zeros((b - a, self.ch), np.float32)
        x0, x1 = max(a, lo), min(b, hi)
        if x1 <= x0:
            return out
        sc = np.float32(self.rebuild_scale())
        for k, job in enumerate(self.inputs.rebuild_jobs()):
            part = self.rebuild_raw(job, x0, x1) * sc
            if job == "mid":
                out[x0 - a:x1 - a, :] = part[:, None]
            else:
                out[x0 - a:x1 - a, k] = part
        return out

    def noise_sig(self, seg: Seg, E: np.ndarray, a: int, b: int) -> np.ndarray:
        step = seg.st["remover"]
        al = pipeline.strength_mix(seg.st["strength_db"])
        y = np.zeros((b - a, self.ch), np.float32)
        xe = E[a:b]
        for c in self.inputs.todo:
            f = self.noise_full(step, c, a, b)
            y[:, c] = f if al <= 0.0 else ((1.0 - al) * f.astype(np.float64) + al * xe[:, c].astype(np.float64)).astype(np.float32)
        for c, src in enumerate(self.inputs.plan):
            if src is not None and src != c:
                y[:, c] = y[:, src]  # the channel that copies another
        return y

    # ---- composition
    def compose(self, segs: list[Seg], joins: list[Join], sig: Callable[[int, int, int], np.ndarray], out: np.ndarray) -> None:
        """Write the segments' signals into `out`: each segment's own signal between the join zones, the crossfade of the two
        neighbours inside them (zones are `z0:z1` of each join, linear or equal-power weights)."""
        pieces: list[tuple[str, int, int, int]] = []
        for i, seg in enumerate(segs):
            ca = joins[i - 1].z1 if i > 0 else 0
            cb = joins[i].z0 if i < len(joins) else self.n
            for a in range(ca, cb, PIECE):
                pieces.append(("core", i, a, min(cb, a + PIECE)))
            if i < len(joins) and joins[i].z1 > joins[i].z0:
                pieces.append(("zone", i, joins[i].z0, joins[i].z1))
        for kind, i, a, b in pieces:
            self.check()
            if b <= a:
                continue
            if kind == "core":
                out[a:b] = sig(i, a, b)
            else:
                j = joins[i]
                t = (np.arange(a, b) - j.z0) / max(j.d, 1)
                wl, wr = (1 - t, t) if j.shape == "lin" else (np.cos(t * np.pi / 2), np.sin(t * np.pi / 2))
                out[a:b] = (wl[:, None] * sig(i, a, b).astype(np.float64) + wr[:, None] * sig(i + 1, a, b).astype(np.float64)).astype(np.float32)

    def make_b_joins(self, E: np.ndarray, segs: list[Seg], joins: list[Join]) -> None:
        """Decide the kind B joins: where they move to (the nearest pause within +-0.3 s), how long they are, and the loudness
        match of the rebuilt side with the other."""
        for j in joins:
            if j.kind != "B":
                continue
            l, r = segs[j.left], segs[j.right]
            rb, other = (r, l) if j.rb_right else (l, r)
            small = min(l.length, r.length)
            max_shift = min(int(SNAP_S * SR), int(0.2 * small))
            # the rebuilt voice around the join, to find a pause in
            half = 3 * SR
            lo, hi = max(rb.a - MARGIN_B, j.t - half), min(rb.b + MARGIN_B, j.t + half)
            lead = _mono(self.rebuild_sig(rb, lo, hi)) if hi > lo else np.zeros(0)
            t, found = find_pause(lead, lo, j.t, max_shift) if max_shift >= 480 else (j.t, False)
            d = int(round((PAUSE_D_S if found else HARD_D_S) * SR))
            d = min(d, int(0.5 * small))
            d -= d % 2
            j.t, j.d = t, d
            j.z0, j.z1 = t - d // 2, t - d // 2 + d
            # loudness match, measured on each side's own part (up to 2 s, at least 0.5 s)
            m = int(MATCH_S * SR)
            if j.rb_right:
                ra, rb_ = t, min(rb.b, t + m)
                oa, ob = max(other.a, t - m), t
            else:
                ra, rb_ = max(rb.a, t - m), t
                oa, ob = t, min(other.b, t + m)
            if rb_ - ra >= SR // 2 and ob - oa >= SR // 2:
                lv_rb = _active_level_db(self.rebuild_sig(rb, ra, rb_))
                lv_ot = _active_level_db(self.pass_sig(other, E, oa, ob))
                j.gain_db = float(np.clip(lv_ot - lv_rb, -MATCH_MAX_DB, MATCH_MAX_DB))

    def browser_sig(self, E: np.ndarray, a: int, b: int) -> np.ndarray:
        """The browser's audio over [a, b) where it has some (all of its part), the recording elsewhere (a join reaches a little past a part)."""
        y = np.array(E[a:b], dtype=np.float32)
        if self.browser is not None:
            self.browser.fill(y, a)
        return y

    def pass_sig(self, seg: Seg, E: np.ndarray, a: int, b: int) -> np.ndarray:
        if seg.kind == "browser":
            return self.browser_sig(E, a, b)
        return self.noise_sig(seg, E, a, b) if seg.kind == "noise" else np.array(E[a:b])

    def rebuild_gain(self, segs: list[Seg], joins: list[Join], i: int, a: int, b: int) -> np.ndarray | None:
        """The smooth loudness-match gain of rebuilt segment `i` over [a, b): full at each join, fading out over EDGE_S."""
        g = None
        pos = np.arange(a, b)
        for j in joins:
            if j.kind != "B" or not j.gain_db or i not in (j.left, j.right):
                continue
            seg = segs[i]
            edge = min(int(EDGE_S * SR), seg.length)
            dist = (pos - j.t) if j.rb_right else (j.t - pos)  # samples into the rebuilt part from the join
            u = np.clip(dist / max(edge, 1), 0.0, 1.0)
            w = 0.5 * (1 + np.cos(np.pi * u))
            w = np.where(dist < 0, 1.0, w)
            db = j.gain_db * w
            g = db if g is None else g + db
        return None if g is None else (10.0 ** (g / 20.0)).astype(np.float32)

    # ---- the steps
    def run(self) -> np.ndarray:
        lay, inputs = self.lay, self.inputs
        tasks = self.keys.tasks()
        order = {s: i for i, s in enumerate(STEPS)}
        tasks.sort(key=lambda t: (order[t.step], str(t.channel), t.s))
        prog = self.progress
        scale = lambda lo, hi: (lambda f: prog(lo + (hi - lo) * f)) if prog else None  # noqa: E731
        # 1. echo removal, then the echo-free recording the models read
        echo_tasks = [t for t in tasks if t.step == "echo" and not self.cache.has(t.key)]
        _run_tasks(echo_tasks, lay, inputs, inputs.source, self.cache, self.runner, scale(0.0, 0.1) if echo_tasks else None, self.cancel)
        E = inputs.source
        if lay.echo_parts:
            E = self.alloc("echo-free")

            def esig(i: int, a: int, b: int) -> np.ndarray:
                seg = lay.echo[i]
                if seg.kind == "off":
                    return inputs.source[a:b]
                return np.stack([self.echo_full(c, a, b) for c in range(self.ch)], axis=1)

            self.compose(lay.echo, lay.echo_joins, esig, E)
        # 2. the models
        rest = [t for t in tasks if t.step != "echo" and not self.cache.has(t.key)]
        _run_tasks(rest, lay, inputs, E, self.cache, self.runner, scale(0.1, 0.8) if rest else None, self.cancel)
        # 3. the main layer (noise removal / Voice rebuild)
        full_level = lay.whole_file_leveling()
        no_level = not lay.level_parts
        P = self.out if (full_level or no_level) else self.alloc("pre-level")
        self.make_b_joins(E, lay.main, lay.main_joins)

        def msig(i: int, a: int, b: int) -> np.ndarray:
            seg = lay.main[i]
            if seg.kind == "pass":
                return E[a:b]
            if seg.kind == "noise":
                return self.noise_sig(seg, E, a, b)
            if seg.kind == "browser":
                return self.browser_sig(E, a, b)
            y = self.rebuild_sig(seg, a, b)
            g = self.rebuild_gain(lay.main, lay.main_joins, i, a, b)
            return y if g is None else y * g[:, None]

        self.compose(lay.main, lay.main_joins, msig, P)
        if prog:
            prog(0.9)
        # 4. Leveling
        if full_level:
            self.out = clarity.process(E, P, SR, clarity.ClarityParams(loudness_lufs=lay.level_parts[0][2]["loudness_lufs"]))
        elif not no_level:
            self.level(E, P)
        if prog:
            prog(1.0)
        for p in self.temp:
            p.unlink(missing_ok=True)
        return self.out

    def level(self, E: np.ndarray, P: np.ndarray) -> None:
        lay = self.lay
        bufs: dict[int, tuple[int, np.ndarray]] = {}  # leveled part: (first sample, its result with the margins of its joins)

        def lsig(i: int, a: int, b: int) -> np.ndarray:
            seg = lay.level[i]
            if seg.kind == "off":
                return P[a:b]
            if i not in bufs:
                for k in [k for k in bufs if k < i - 1]:
                    del bufs[k]
                bufs[i] = self.level_part(E, P, seg, lay.level_joins, i)
            first, y = bufs[i]
            return y[a - first:b - first]

        self.compose(lay.level, lay.level_joins, lsig, self.out)

    def level_part(self, E: np.ndarray, P: np.ndarray, seg: Seg, joins: list[Join], i: int) -> tuple[int, np.ndarray]:
        """The Leveling of one part, measured on the part and max(its length, 10 s) each side (clamped to the file)."""
        lay = self.lay
        ma, mb = lay.measure_span(seg.a, seg.b)
        ctx = int(LEVEL_CONTEXT_S * SR)
        assert ma <= max(0, seg.a - ctx) and mb >= min(self.n, seg.b + ctx)  # the measure span contains the 2 s of context
        y = self.alloc("level", mb - ma)
        y[:] = P[ma:mb]
        clarity.process(E[ma:mb], y, SR, clarity.ClarityParams(loudness_lufs=seg.st["loudness_lufs"]))
        # keep the part and what its joins read beyond it
        left = joins[i - 1].d // 2 + 1 if i > 0 else 0
        right = joins[i].d // 2 + 1 if i < len(joins) else 0
        a, b = max(ma, seg.a - left), min(mb, seg.b + right)
        return a, np.array(y[a - ma:b - ma])


def render(
    tl: Timeline,
    inputs: Inputs,
    cache: blocks.BlockCache | None = None,
    runner: Runner | None = None,
    *,
    out: np.ndarray | None = None,
    work_dir: str | Path | None = None,
    progress: Callable[[float], None] | None = None,
    cancel: threading.Event | None = None,
    browser: BrowserAudio | None = None,
) -> np.ndarray:
    """The audio the timeline stands for, (samples, channels) float32, channels as `pipeline.channel_plan` has them (a channel that
    duplicates another is the same result, a silent one stays silent). Blocks the cache lacks are run with `runner` and put in
    the cache (`cache` None: kept for this call only, in `work_dir` when given); `out`: written into this array (e.g. a memory
    map). `browser`: the audio of the noise parts the browser cleaned (they need it). The noise remover of every part must be decided ("light" / "heavy": resolve_removers). The default timeline
    (every layer on the whole file) is what `pipeline.clean_audio` makes of the same recording, and fills the same cache."""
    wd = Path(work_dir) if work_dir is not None else None
    own = None
    if cache is None:
        if wd is not None:
            own = wd / f"editor-blocks-{uuid.uuid4().hex[:8]}"
            cache = blocks.DirCache(own)
        else:
            cache = blocks.MemoryCache()
    if out is not None and out.shape != (inputs.n, inputs.channels):
        raise ValueError("out must be shaped (samples, channels) like the recording")
    r = _Render(tl, inputs, cache, runner, out, wd, progress, cancel, browser)
    try:
        return r.run()
    finally:
        if own is not None:
            shutil.rmtree(own, ignore_errors=True)


__all__ = ["Inputs", "Runner", "BrowserAudio", "Layout", "Plan", "Task", "StepCost", "STEP_COSTS", "EditorError", "plan", "billable",
           "compute_estimate", "resolve_removers", "render", "find_pause"]
