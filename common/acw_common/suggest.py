"""Automatic suggestions per part for the editor: where a recording changes, and what fits each part better than the whole file.

    s = suggest.suggest(x, lib_path, model_path)        # x: (samples, channels) float32 at 48 kHz
    s.whole                                             # the decision for the whole file (reco_model: strength, remover, rebuild)
    s.suggestions                                       # parts whose decision differs from it, each with plain-word reasons
    tl = suggest.apply(timeline, s.suggestions)         # the suggestions as parts of a timeline (editable like any other)

How (measured in the editor experiments, Q6; accuracy and cost: docs/research):

  1. CHANGE POINTS. The file is cut into 5 s windows that do not overlap. Each gets the features of analysis.py (floor, level, tilt,
     drop, bandwidth, speech share, clipping) and a noise figure measured on a 3 s stretch from the window's middle (one DeepFilterNet3
     run, 0.3 s of CPU). A boundary candidate is the distance between neighbouring windows (the largest of the features' changes, each
     in units of what counts as a change); local peaks from 1.5 up, the strongest first, no two closer than 20 s.
  2. REFINED on a 1 s grid: around each candidate, the 1 s step at which the 5 s before and after differ most.
  3. ONE DECISION per detected part, from the whole part (analysis + the trained model, as for a whole file), and only the parts whose
     decision differs from the whole file's are returned, as suggestions with the reasons in plain words ("sounds distorted", "is
     noisier", "is quiet").
  4. CLIPPING SPOTS: short bursts of clipped samples (the detector of the declipping experiments, run in windows of a minute) become
     Voice rebuild suggestions over the burst with half a second of context, where Voice rebuild is not already on.
"""
from __future__ import annotations

import math
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import analysis, dfn, pipeline, reco_model
from .timeline import Timeline

SR = dfn.SAMPLE_RATE
COARSE_S = 5.0  # window of the change detection (not overlapping)
NF_STRETCH_S = 3.0  # the noise figure of a window is measured on this much of its middle
TAU = 1.5  # a change of this many "units" is a candidate boundary
MIN_PART_S = 20.0
REFINE_S = 5.0  # the 1 s grid search reaches this far each side of a candidate
#: What counts as a change in each feature (the distance of two windows is the largest change in these units).
SCALE = {"floorDb": 6.0, "levelDb": 6.0, "noiseDb": 6.0, "tiltDb": 6.0, "dropDb": 4.0, "speechFraction": 0.25, "clipFraction": 0.002,
         "bwLog2": 0.6, "hasSpeech": 0.5}
QUIET_DB = 8.0  # a part this much quieter (louder) than the file is "quiet" ("loud")
NOISY_DB = 3.0  # ... this much noisier / cleaner
CLIP_WINDOW_S = 60.0
CLIP_OVERLAP_S = 1.0
CLIP_CONTEXT_S = 0.5
CLIP_MIN_SAMPLES = 8  # clipped samples in a burst cluster below this are not a suggestion

REASON_WORDS = {"echoey": "sounds echoey", "muffled": "sounds muffled", "phone": "sounds like a phone call",
                "distorted": "sounds distorted", "unclear": "has lost some of its clarity"}


@dataclass(frozen=True)
class Decision:
    """What the automatic settings say for a stretch: rebuild the voice, which noise remover, how strong."""
    rebuild: bool
    remover: str  # "light" | "heavy"
    strength: str  # "gentle" | "balanced" | "strong" | "max"
    noise_db: float | None = None
    level_db: float | None = None
    rebuild_reasons: tuple[str, ...] = ()

    @property
    def key(self) -> tuple:
        return (self.rebuild, self.remover, self.strength)

    @property
    def strength_db(self) -> float:
        return reco_model.STRENGTH_DB[self.strength]


@dataclass
class Suggestion:
    start: float
    end: float
    kind: str  # "part" (the decision differs from the whole file's) | "clipping" (a burst of clipped samples)
    rebuild: bool
    noise: dict | None  # {"strength_db", "remover"} for the noise removal there; None for a clipping spot
    reasons: list[str]

    def to_json(self) -> dict:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "kind": self.kind, "rebuild": self.rebuild,
                "noise": self.noise, "reasons": list(self.reasons)}


@dataclass
class Suggestions:
    seconds: float
    whole: Decision
    parts: list[tuple[float, float, Decision]]  # every detected part and its decision
    suggestions: list[Suggestion]
    cpu_seconds: float = 0.0  # what finding them cost (process CPU)

    def to_json(self) -> dict:
        return {"seconds": self.seconds, "whole": {"rebuild": self.whole.rebuild, "remover": self.whole.remover, "strength": self.whole.strength},
                "parts": [{"start": a, "end": b, "rebuild": d.rebuild, "remover": d.remover, "strength": d.strength} for a, b, d in self.parts],
                "suggestions": [s.to_json() for s in self.suggestions]}


# ---------------------------------------------------------------------------------------------------------------------
# Features and decisions
# ---------------------------------------------------------------------------------------------------------------------
def _decision(stats: dict) -> Decision:
    rec = reco_model.recommend(stats, min_seconds=0.0)
    if rec is None:
        return Decision(False, "light", "balanced")
    ch = reco_model.choose(rec)
    return Decision(bool(ch.rebuild_wanted), "heavy" if ch.heavy else "light", ch.strength, stats.get("noiseDb"), stats.get("levelDb"),
                    tuple(rec.rebuild_reasons or ((rec.rebuild_reason,) if rec.rebuild_reason else ())))


def _mono(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x.mean(axis=1, dtype=np.float32) if x.ndim == 2 else x, dtype=np.float32)


def _fvec(st: dict, nf: float) -> dict:
    v = {k: (st.get(k) if st.get(k) is not None else float("nan")) for k in ("floorDb", "levelDb", "tiltDb", "dropDb", "speechFraction", "clipFraction")}
    v["noiseDb"] = nf
    v["bwLog2"] = math.log2(max(st.get("bandwidthHz") or 1.0, 1.0))
    v["hasSpeech"] = float(bool(st.get("hasSpeech")))
    return v


def _dist(a: dict, b: dict) -> float:
    d = [abs(a[k] - b[k]) / SCALE[k] for k in SCALE if not (math.isnan(a[k]) or math.isnan(b[k]))]
    return max(d) if d else 0.0


class _Windows:
    """Features of 5 s windows (start in whole seconds), made on demand and kept."""

    def __init__(self, x: np.ndarray, inst: dfn._Instances, cancel: threading.Event | None):
        self.x, self.inst, self.cancel = x, inst, cancel
        self.cache: dict[float, dict] = {}

    def _nf(self, seg: np.ndarray) -> float:
        mid = len(seg) // 2
        half = int(NF_STRETCH_S * SR) // 2
        s = _mono(seg[max(0, mid - half):mid + half])
        y = dfn._denoise_contiguous(self.inst, s, self.cancel)
        v = float(np.sum(y.astype(np.float64) ** 2))
        r = float(np.sum((s.astype(np.float64) - y) ** 2))
        return 99.0 if v + r <= 0 else float(min(99.0, 10 * np.log10((v + 1e-12) / (r + 1e-12))))

    def at(self, start: float) -> dict:
        if start not in self.cache:
            a = int(round(start * SR))
            seg = self.x[a:a + int(COARSE_S * SR)]
            nf = self._nf(seg)
            self.cache[start] = _fvec(analysis.measure_features(seg, nf), nf)
        return self.cache[start]

    def many(self, starts: list[float], threads: int) -> None:
        todo = [s for s in starts if s not in self.cache]
        if threads <= 1 or len(todo) < 2:
            for s in todo:
                self.at(s)
            return
        with ThreadPoolExecutor(max_workers=threads) as pool:
            list(pool.map(self.at, todo))


def change_points(x: np.ndarray, lib_path, model_path, *, threads: int = 2, cancel: threading.Event | None = None,
                  tau: float = TAU, min_part: float = MIN_PART_S) -> list[float]:
    """Where the recording changes (seconds, refined to 1 s), at least `min_part` apart and from the ends."""
    total = len(x) / SR
    k = int(total // COARSE_S)
    if k < 2 or total < 2 * min_part:
        return []
    inst = dfn._Instances(lib_path, model_path, 100.0)
    try:
        w = _Windows(x, inst, cancel)
        starts = [i * COARSE_S for i in range(k)]
        w.many(starts, threads)
        cand = [((i + 1) * COARSE_S, _dist(w.at(starts[i]), w.at(starts[i + 1]))) for i in range(k - 1)]
        peaks = [(t, d) for j, (t, d) in enumerate(cand)
                 if d >= tau and (j == 0 or d >= cand[j - 1][1]) and (j == len(cand) - 1 or d >= cand[j + 1][1])]
        peaks.sort(key=lambda p: -p[1])
        kept: list[float] = []
        for t, _ in peaks:
            if min_part <= t <= total - min_part and all(abs(t - u) >= min_part for u in kept):
                kept.append(t)
        out = []
        for t in sorted(kept):  # the 1 s step where the 5 s before and the 5 s after differ most
            lo, hi = int(round(t)) - int(REFINE_S), int(round(t)) + int(REFINE_S)
            cs = [float(c) for c in range(lo, hi + 1) if c - COARSE_S >= 0 and c + COARSE_S <= k * COARSE_S + 1e-9 and (c + COARSE_S) * SR <= len(x)]
            w.many(sorted({c - COARSE_S for c in cs} | {float(c) for c in cs}), threads)
            best, bt = -1.0, t
            for c in cs:
                d = _dist(w.at(c - COARSE_S), w.at(float(c)))
                if d > best:
                    best, bt = d, c
            out.append(float(bt))
        merged: list[float] = []
        for t in sorted(out):  # (a refinement can bring two closer than a part may be)
            if (not merged or t - merged[-1] >= min_part) and total - t >= min_part and t >= min_part:
                merged.append(t)
        return merged
    finally:
        inst.close()


def _part_stats(x: np.ndarray, a: int, b: int, lib_path, model_path, cancel) -> dict:
    seg = x[a:b]
    figure = pipeline.noise_figure_db(seg, lib_path, model_path, cancel)
    return analysis.measure_features(seg, figure)


# ---------------------------------------------------------------------------------------------------------------------
# Reasons, in plain words
# ---------------------------------------------------------------------------------------------------------------------
def reasons_for(part: Decision, whole: Decision) -> list[str]:
    out: list[str] = []
    if part.rebuild and not whole.rebuild:
        out.extend(REASON_WORDS[r] for r in part.rebuild_reasons if r in REASON_WORDS)
        if not out:
            out.append("sounds like it needs rebuilding")
    elif whole.rebuild and not part.rebuild:
        out.append("sounds fine here, no rebuild needed")
    if part.noise_db is not None and whole.noise_db is not None:
        if part.noise_db <= whole.noise_db - NOISY_DB:
            out.append("is noisier")
        elif part.noise_db >= whole.noise_db + NOISY_DB:
            out.append("is cleaner")
    if part.level_db is not None and whole.level_db is not None and not (math.isnan(part.level_db) or math.isnan(whole.level_db)):
        if part.level_db <= whole.level_db - QUIET_DB:
            out.append("is quiet")
        elif part.level_db >= whole.level_db + QUIET_DB:
            out.append("is loud")
    if not part.rebuild and part.remover != whole.remover:
        out.append("needs the stronger noise remover" if part.remover == "heavy" else "needs only the lighter noise remover")
    if not part.rebuild and part.strength != whole.strength and not any(r in out for r in ("is noisier", "is cleaner")):
        order = reco_model.STRENGTH_CLASSES
        out.append("needs a stronger noise removal" if order.index(part.strength) > order.index(whole.strength) else "needs a gentler noise removal")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Clipping spots
# ---------------------------------------------------------------------------------------------------------------------
def _clip_level(a: np.ndarray, bw: float = 0.0125, min_count: int = 40, min_ratio: float = 2.5, keep: float = 0.3, floor: float = 0.45):
    """a = |x| / max|x|. A pile-up in the amplitude histogram near the top (hard clipping puts many samples at one level; a codec
    smears it into a bump below the maximum, so the test is local density): the lower amplitude bound of the clipped zone, or None."""
    edges = np.arange(0.0, 1.0 + bw, bw)
    h, _ = np.histogram(a, bins=edges)
    nb = len(h)
    b0, b1 = int(round(0.1 / bw)), int(round(0.2 / bw))
    best = None
    for i in range(int(floor / bw), nb):
        lo, hi = i - b1, i - b0
        if lo < 0:
            continue
        base = float(np.median(h[lo:hi])) + 1.0
        if h[i] >= min_count and h[i] >= min_ratio * base:
            sc = h[i] / base
            if best is None or sc > best[0]:
                best = (sc, i, base)
    if best is None:
        return None
    _, i, base = best
    j = i
    while j - 1 >= int(floor / bw) and h[j - 1] >= max(keep * h[i], 1.8 * base):
        j -= 1
    return float(edges[j])


def clipped_bursts(x: np.ndarray, *, min_run: int = 2, burst_gap: float = 0.10, min_count: int = 40, min_ratio: float = 1.6,
                   min_level: float = 0.25) -> list[tuple[int, int, int]]:
    """Bursts of clipped samples in a mono stretch: (first sample, end, clipped samples), events closer than `burst_gap` s one burst.
    The detector of the declipping experiments (docs/research/declipping): the clip level is found as a pile-up in the amplitude
    histogram, so a soft or codec-smeared clip is found too, and a quiet recording's natural peaks are not."""
    n = len(x)
    peak = float(np.max(np.abs(x))) if n else 0.0
    if peak < 0.05:
        return []
    a = np.abs(x) / peak
    thr = _clip_level(a, min_count=min_count, min_ratio=min_ratio)
    if thr is None or thr * peak < min_level:  # DC offsets / tiny signals pegged at a low rail are not clipping
        return []
    cand = a >= thr
    sign = np.sign(x)
    d = np.flatnonzero(np.diff(np.concatenate([[0], cand.astype(np.int8), [0]])))
    events = []
    for st, en in zip(d[0::2], d[1::2]):
        seg = sign[st:en]
        cuts = np.concatenate([[0], np.flatnonzero(seg[1:] != seg[:-1]) + 1, [len(seg)]])
        events += [(int(st + cuts[i]), int(st + cuts[i + 1])) for i in range(len(cuts) - 1) if cuts[i + 1] - cuts[i] >= min_run]
    if not events:
        return []
    gap = int(burst_gap * SR)
    bursts = [[events[0][0], events[0][1], events[0][1] - events[0][0]]]
    for st, en in events[1:]:
        if st - bursts[-1][1] <= gap:
            bursts[-1][1] = en
            bursts[-1][2] += en - st
        else:
            bursts.append([st, en, en - st])
    return [(a0, b0, c) for a0, b0, c in bursts]


def clipping_spots(x: np.ndarray, *, context: float = CLIP_CONTEXT_S, merge_gap: float = 0.25) -> list[tuple[float, float]]:
    """Spots (start, end in seconds) with clipped bursts and `context` seconds around them, nearby ones merged. The detector runs on
    windows of CLIP_WINDOW_S (a minute: its clip level is that of the window); a window that is clipped almost everywhere is left to
    the file's own decision."""
    n = len(x)
    size, hop = int(CLIP_WINDOW_S * SR), int((CLIP_WINDOW_S - CLIP_OVERLAP_S) * SR)
    spans = []
    for a in range(0, max(1, n - int(CLIP_OVERLAP_S * SR)), hop):
        b = min(n, a + size)
        seg = _mono(x[a:b])
        bursts = clipped_bursts(seg)
        if not bursts:
            continue
        if sum(c for _, _, c in bursts) > 0.08 * len(seg):  # heavily clipped throughout: not a spot
            continue
        for s0, e0, c in bursts:
            if c >= CLIP_MIN_SAMPLES:
                spans.append((max(0, a + s0 - int(context * SR)), min(n, a + e0 + int(context * SR))))
        if b >= n:
            break
    spans.sort()
    out: list[list[int]] = []
    for s0, e0 in spans:
        if out and s0 - out[-1][1] <= int(merge_gap * SR):
            out[-1][1] = max(out[-1][1], e0)
        else:
            out.append([s0, e0])
    return [(s0 / SR, e0 / SR) for s0, e0 in out]


# ---------------------------------------------------------------------------------------------------------------------
# The whole thing
# ---------------------------------------------------------------------------------------------------------------------
def suggest(x: np.ndarray, lib_path: str | Path, model_path: str | Path, *, threads: int = 2, cancel: threading.Event | None = None,
            progress: Callable[[float], None] | None = None, clipping: bool = True) -> Suggestions:
    """Suggestions for the recording `x` ((samples, channels) float32, 48 kHz; a memory map is read in short stretches).
    `lib_path` / `model_path`: DeepFilterNet3 (the noise figure is measured with it)."""
    import time

    t0 = time.process_time()
    x = x if x.ndim == 2 else x[:, None]
    n = len(x)
    total = n / SR
    whole_stats = _part_stats(x, 0, n, lib_path, model_path, cancel) if n else {"noiseDb": None}
    whole = _decision(whole_stats)
    if progress:
        progress(0.1)
    cuts = change_points(_mono_view(x), lib_path, model_path, threads=threads, cancel=cancel) if n else []
    if progress:
        progress(0.7)
    edges = [0.0] + cuts + [total]
    parts: list[tuple[float, float, Decision]] = []
    sugg: list[Suggestion] = []
    for a, b in zip(edges[:-1], edges[1:]):
        if len(cuts) == 0:
            d = whole
        else:
            d = _decision(_part_stats(x, int(round(a * SR)), min(n, int(round(b * SR))), lib_path, model_path, cancel))
        parts.append((a, b, d))
        if d.key != whole.key:
            sugg.append(Suggestion(a, b, "part", d.rebuild, {"strength_db": d.strength_db, "remover": d.remover}, reasons_for(d, whole)))
    if progress:
        progress(0.9)
    if clipping and n:
        def rebuilt_at(t: float) -> bool:
            return next((d.rebuild for a, b, d in parts if a <= t < b), whole.rebuild)

        for a, b in clipping_spots(x):
            if rebuilt_at((a + b) / 2) and rebuilt_at(a) and rebuilt_at(min(b, total - 1e-3)):
                continue  # the voice is rebuilt there anyway
            sugg.append(Suggestion(a, b, "clipping", True, None, ["sounds distorted here (clipped)"]))
    sugg.sort(key=lambda s: (s.start, s.end))
    if progress:
        progress(1.0)
    return Suggestions(total, whole, parts, sugg, time.process_time() - t0)


def _mono_view(x: np.ndarray):
    """A mono view for the 5 s windows: the mix is made per window (a memory map is never loaded whole)."""
    class V:
        def __len__(self):
            return len(x)

        def __getitem__(self, sl):
            return _mono(x[sl])

    return V()


def apply(tl: Timeline, suggestions: list[Suggestion]) -> Timeline:
    """A copy of the timeline with the suggestions painted in: a part that rebuilds the voice goes on the rebuild layer; a part that
    does not takes the rebuild off there and sets its noise removal (strength and remover)."""
    out = tl.copy()
    for s in suggestions:
        if s.rebuild:
            out.paint("rebuild", s.start, s.end)
        else:
            out.erase("rebuild", s.start, s.end)
            if s.noise:
                out.paint("noise", s.start, s.end, dict(s.noise))
    return out


def whole_timeline(seconds: float, whole: Decision, *, echo: bool = False, leveling: bool = True) -> Timeline:
    """The timeline for the file's own decision: the default every suggestion is a change of."""
    t = Timeline.whole_file(seconds, echo=echo, noise={"strength_db": whole.strength_db, "remover": whole.remover}, rebuild=whole.rebuild,
                            leveling=leveling)
    return t
