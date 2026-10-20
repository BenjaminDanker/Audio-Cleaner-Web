"""Live listening for the per-part editor: the sound of a timeline, mixed per part while it plays. No Qt in here.

`LiveMix` has what the player reads from a `blend.BlendStream` (`frames`, `channels`, `seconds`, `read`, `read_pcm16`, `set_original`), so
the job's player (gui/job_preview.py, gui/live_audio.py) plays it the same way. It holds the recording and what each step made of it, laid out
sample for sample (`Layers`, memory maps filled from the job's block cache), and a snapshot of the timeline:

    echo removal        a 50 ms ramp between the recording and the echo-free sound at the edges of each part
    noise removal       per part: (1 - a) x the remover's full-strength sound + a x the recording, a from the strength; 100 ms ramps
                        between parts (so Off, another strength and another remover all join smoothly)
    Voice rebuild       equal-power edges of 250 ms: cos x the sound without it + sin x the rebuilt voice; where it is on it wins over
                        the noise removal, as in the saved result
    Leveling            previewed by loudness: each leveled part is brought to the loudness target (measured on what is heard there,
                        at most 12 dB either way), with 100 ms ramps. The even-out of the compressor and the limiter come with the saved file

A part whose blocks are not in the cache yet cannot be heard: that layer is left out there (`missing`) until it has been processed. The
"Hear" switches take a layer out of what is heard without touching the timeline. All of this is for listening: the saved file is made by
acw_common.editor (joins that look for a pause, loudness matching of the rebuilt voice, the whole compressor and limiter).
"""
from __future__ import annotations

import math
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from acw_common import clarity, editor
from acw_common.timeline import Timeline

SR = 48_000
CELL = SR // 10  # availability is tracked per 100 ms
RAMP_ECHO = int(0.05 * SR)
RAMP_NOISE = int(0.10 * SR)
RAMP_LEVEL = int(0.10 * SR)
EDGE_REBUILD = int(0.25 * SR)
LEVEL_MAX_DB = 12.0
LEVEL_BUDGET_S = 60.0  # audio measured for all the leveled parts together (per part: at most 12 s)
LAYERS = ("echo", "noise", "rebuild", "leveling")


def ramp_gate(pos: np.ndarray, a: int, b: int, d: int, n: int) -> np.ndarray:
    """1 inside [a, b), a linear ramp of `d` samples centred on each edge (none at the ends of the file); adjacent parts get gates that
    add up to 1 across their join."""
    g = np.ones(len(pos), np.float64)
    d = max(1, d)
    if a > 0:
        g *= np.clip((pos - a) / d + 0.5, 0.0, 1.0)
    if b < n:
        g *= np.clip((b - pos) / d + 0.5, 0.0, 1.0)
    return g


class Layers:
    """The recording and what each step made, (n, channels) float32 each (memory maps in `work`, made when first needed), and where each has
    sound (`have`: one flag per 100 ms)."""

    def __init__(self, src: np.ndarray, plan: list[int | None], work: Path | None = None):
        self.src = src
        self.n, self.channels = len(src), src.shape[1]
        self.plan = list(plan)
        self.work = Path(work) if work is not None else None
        self.arrays: dict[str, np.ndarray] = {}
        cells = -(-self.n // CELL)
        self.have = {k: np.zeros(cells, bool) for k in ("echo", "light", "heavy", "rebuild")}
        self.loaded: dict[tuple, str] = {}

    def array(self, step: str) -> np.ndarray:
        if step not in self.arrays:
            shape = (self.n, self.channels)
            if self.work is None:
                self.arrays[step] = np.zeros(shape, np.float32)
            else:
                path = self.work / f"live-{step}-{uuid.uuid4().hex[:8]}.npy"
                self.arrays[step] = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
        return self.arrays[step]

    def mark(self, step: str, s: int, e: int) -> None:
        self.have[step][s // CELL:-(-e // CELL)] = True

    def available(self, step: str, pos: np.ndarray) -> np.ndarray:
        cells = np.clip(pos // CELL, 0, len(self.have[step]) - 1)
        return self.have[step][cells]

    def load(self, cache, plan: "editor.Plan") -> int:
        """Lay out the blocks of `plan` that the cache has (and forget the ones it no longer has under these keys). Returns how many were read."""
        for h in self.have.values():
            h[:] = False
        read = 0
        for t in plan.tasks:
            ident = (t.step, t.channel, t.s, t.e)
            if not cache.has(t.key):
                self.loaded.pop(ident, None)
                continue
            if self.loaded.get(ident) != t.key:
                blk = cache.get(t.key)
                if blk is None or len(blk) != t.e - t.s:
                    self.loaded.pop(ident, None)
                    continue
                arr = self.array(t.step)
                if t.step == "rebuild":
                    cols = list(range(self.channels)) if t.channel == "mid" else [int(t.channel)]
                    for c in cols:
                        arr[t.s:t.e, c] = blk
                else:
                    arr[t.s:t.e, int(t.channel)] = blk
                self.loaded[ident] = t.key
                read += 1
            self.mark(t.step, t.s, t.e)
        return read

    def close(self) -> None:
        self.arrays.clear()
        self.src = None  # (the memory maps must be let go before their folder can be removed)


@dataclass(frozen=True)
class Snapshot:
    """The timeline as the mix reads it (sample positions), with the Leveling gains worked out for it."""
    echo: tuple = ()
    noise: tuple = ()  # (a, b, share of the recording kept, remover)
    rebuild: tuple = ()
    level: tuple = ()  # (a, b, linear gain)
    missing: dict = field(default_factory=dict)  # {layer: [(a, b)]}: parts that have no sound yet


def _snapshot_of(tl: Timeline, n: int) -> Snapshot:
    from acw_common import pipeline

    return Snapshot(
        echo=tuple((a, b) for a, b, _ in tl.samples("echo", n)),
        noise=tuple((a, b, pipeline.strength_mix(s["strength_db"]), s["remover"]) for a, b, s in tl.samples("noise", n)),
        rebuild=tuple((a, b) for a, b, _ in tl.samples("rebuild", n)),
        level=tuple((a, b, s["loudness_lufs"]) for a, b, s in tl.samples("leveling", n)),
    )


class LiveMix:
    """The sound of a timeline for the player. Thread-safe: the timeline and the switches can change while the sound is read."""

    def __init__(self, layers: Layers, timeline: Timeline):
        self.layers = layers
        self.frames, self.channels = layers.n, layers.channels
        self._lock = threading.Lock()
        self._hear = {k: True for k in LAYERS}
        self._original = False
        self._timeline = timeline
        self._snap = Snapshot()
        self._gains: tuple = ()
        self.set_timeline(timeline)

    # -- what the player and the job view use ---------------------------------------------------------------------
    @property
    def seconds(self) -> float:
        return self.frames / SR

    @property
    def original(self) -> bool:
        return self._original

    def set_original(self, on: bool) -> None:
        with self._lock:
            self._original = bool(on)

    def set_strength(self, db: float) -> None:  # (the strength is a property of the parts here; the job preview calls this one)
        pass

    @property
    def exact(self) -> bool:
        return False

    @exact.setter
    def exact(self, value: bool) -> None:
        pass

    @property
    def hear(self) -> dict:
        return dict(self._hear)

    def set_hear(self, layer: str, on: bool) -> None:
        """Hear this layer (or not), without touching the timeline."""
        if layer not in self._hear:
            raise ValueError(f"unknown layer {layer!r}")
        with self._lock:
            self._hear[layer] = bool(on)
        self._level_gains()

    @property
    def timeline(self) -> Timeline:
        return self._timeline

    def set_timeline(self, tl: Timeline) -> None:
        """Hear this timeline (its parts, strengths and switches) from the next piece on."""
        snap = _snapshot_of(tl, self.frames)
        with self._lock:
            self._timeline = tl.copy()
            self._snap = snap
            self._gains = ()
        self._level_gains()

    def missing(self, layer: str) -> list[tuple[float, float]]:
        """The parts (seconds) of a layer that cannot be heard yet: their blocks are not in the cache."""
        snap = self._snap
        if layer == "noise":
            items = [(a, b, r if r in ("light", "heavy") else "light") for a, b, _, r in snap.noise]
        elif layer == "echo":
            items = [(a, b, "echo") for a, b in snap.echo]
        elif layer == "rebuild":
            items = [(a, b, "rebuild") for a, b in snap.rebuild]
        else:
            return []
        out = []
        for a, b, step in items:
            pos = np.arange(a, b, CELL)
            if pos.size and not self.layers.available(step, pos).all():
                out.append((a / SR, b / SR))
        return out

    def refresh(self, cache, plan: "editor.Plan") -> int:
        """The cache has more blocks (an edit was saved): lay them out and hear them."""
        n = self.layers.load(cache, plan)
        self._level_gains()
        return n

    # -- reading ------------------------------------------------------------------------------------------------------
    def read(self, start: int, count: int) -> np.ndarray:
        """`count` frames from frame `start` (fewer at the end): (n, channels) float32."""
        start = max(0, int(start))
        end = min(self.frames, start + max(0, int(count)))
        if end <= start:
            return np.zeros((0, self.channels), np.float32)
        with self._lock:
            original, snap, gains, hear = self._original, self._snap, self._gains, dict(self._hear)
        if original:
            return np.array(self.layers.src[start:end], dtype=np.float32)
        return self._heard(snap, gains, hear, start, end)

    def read_pcm16(self, start: int, count: int) -> bytes:
        block = self.read(start, count)
        return np.rint(np.clip(block, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    def _heard(self, snap: Snapshot, gains: tuple, hear: dict, a: int, b: int, leveling: bool = True) -> np.ndarray:
        L, n = self.layers, self.frames
        pos = np.arange(a, b)
        x = np.asarray(L.src[a:b], dtype=np.float32)
        e = x
        if hear["echo"] and snap.echo and "echo" in L.arrays:
            g = np.zeros(len(pos))
            for pa, pb in snap.echo:
                if pb > a and pa < b:
                    g = np.maximum(g, ramp_gate(pos, pa, pb, RAMP_ECHO, n))
            g *= L.available("echo", pos)
            if g.any():
                e = (x + g[:, None] * (np.asarray(L.arrays["echo"][a:b], dtype=np.float32) - x)).astype(np.float32)
        m = e
        if hear["noise"] and snap.noise:
            m = np.array(e, dtype=np.float64)
            for pa, pb, share, remover in snap.noise:
                if pb <= a - RAMP_NOISE or pa >= b + RAMP_NOISE:
                    continue
                step = remover if remover in ("light", "heavy") else "light"
                if step not in L.arrays or not L.available(step, pos).any():
                    other = "heavy" if step == "light" else "light"
                    if other in L.arrays and L.available(other, pos).any():
                        step = other  # the other remover's sound stands in until this one has run
                    else:
                        continue
                g = ramp_gate(pos, pa, pb, RAMP_NOISE, n) * L.available(step, pos)
                if not g.any():
                    continue
                full = L.arrays[step][a:b]
                nb = np.zeros((len(pos), self.channels))
                for c, src in enumerate(L.plan):
                    if src is not None:
                        f = full[:, src].astype(np.float64)
                        nb[:, c] = f if share <= 0.0 else (1.0 - share) * f + share * e[:, c]
                m += g[:, None] * (nb - e)
        if hear["rebuild"] and snap.rebuild and "rebuild" in L.arrays:
            th = np.zeros(len(pos))
            for pa, pb in snap.rebuild:
                if pb > a - EDGE_REBUILD and pa < b + EDGE_REBUILD:
                    th += ramp_gate(pos, pa, pb, EDGE_REBUILD, n)
            th = np.clip(th, 0.0, 1.0) * L.available("rebuild", pos)
            if th.any():
                ang = th * (math.pi / 2)
                m = np.cos(ang)[:, None] * np.asarray(m, dtype=np.float64) + np.sin(ang)[:, None] * np.asarray(L.arrays["rebuild"][a:b], dtype=np.float64)
        if leveling and hear["leveling"] and gains:
            lv = np.ones(len(pos))
            for pa, pb, g_lin in gains:
                if pb > a - RAMP_LEVEL and pa < b + RAMP_LEVEL:
                    lv += ramp_gate(pos, pa, pb, RAMP_LEVEL, n) * (g_lin - 1.0)
            m = np.asarray(m, dtype=np.float64) * lv[:, None]
        return np.asarray(m).astype(np.float32)

    # -- Leveling, previewed by loudness -------------------------------------------------------------------------------
    def _level_gains(self) -> None:
        """The gain that brings each leveled part to its loudness target, measured on what is heard there without Leveling."""
        with self._lock:
            snap, hear = self._snap, dict(self._hear)
        parts = snap.level
        if not parts or not hear["leveling"]:
            with self._lock:
                if snap is self._snap:
                    self._gains = ()
            return
        per = min(12.0, max(0.5, LEVEL_BUDGET_S / len(parts)))
        out = []
        for a, b, target in parts:
            if target is None or b - a < int(0.4 * SR):
                out.append((a, b, 1.0))
                continue
            take = min(b - a, int(per * SR))
            pieces = max(1, take // (2 * SR))
            win = take // pieces
            chunks = []
            for k in range(pieces):
                s = a + int((b - a - win) * (k / max(1, pieces - 1))) if pieces > 1 else a + (b - a - win) // 2
                chunks.append(self._heard(snap, (), hear, s, s + win, leveling=False))
            sample = np.concatenate(chunks) if chunks else np.zeros((0, self.channels), np.float32)
            gain_db = 0.0
            if len(sample) >= int(0.4 * SR):
                try:
                    lufs = clarity.integrated_loudness(sample, SR)
                except ValueError:
                    lufs = float("-inf")
                if math.isfinite(lufs) and lufs > -70.0:
                    gain_db = float(np.clip(target - lufs, -LEVEL_MAX_DB, LEVEL_MAX_DB))
            out.append((a, b, 10.0 ** (gain_db / 20.0)))
        with self._lock:
            if snap is self._snap:
                self._gains = tuple(out)

    def level_gains_db(self) -> list[tuple[float, float, float]]:
        """(start, end, gain in dB) of each leveled part as previewed."""
        return [(a / SR, b / SR, 20.0 * math.log10(g)) for a, b, g in self._gains]

    def close(self) -> None:
        self.layers.close()
