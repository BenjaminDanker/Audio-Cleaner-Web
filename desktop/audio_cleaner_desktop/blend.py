"""Strength is a blend, so a job's strength can be changed without running a model again (and heard while it plays).

The noise remover always runs at full strength (acw_common.pipeline); what "strength" means is how much of the recording is
mixed back into its output: result = (1 - a) * full + a * recording, a = 10 ** (-dB / 20) (100 dB and above: nothing is mixed
back). This module has the arithmetic of the strength slider (the same scale as the website's: Gentle 12, Balanced 24, Strong 40,
Maximum 100 dB, equally spaced, with the dB in between), the loudness match that keeps the slider from changing the volume, and
`BlendStream`: the saved full-strength copy and the original, lined up sample for sample, mixed block by block for the player.

No Qt in here.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from acw_common import pipeline

SAMPLE_RATE = pipeline.SAMPLE_RATE
# The four named strengths in dB, in the order of the slider (the same numbers as engine.STRENGTHS).
NAMES = (("gentle", "Gentle", 12.0), ("balanced", "Balanced", 24.0), ("strong", "Strong", 40.0), ("max", "Maximum", 100.0))
POINTS = tuple(db for _, _, db in NAMES)
SLIDER_STEPS = 300  # the slider runs 0..300: three equal stretches between the four names
_SUMS_BLOCK = SAMPLE_RATE * 30


# ---------------------------------------------------------------------------------------------------- the scale

def mix_of(db: float) -> float:
    """The share `a` of the recording that stays in the result at a limit of `db` (0 at 100 dB and above)."""
    return pipeline.strength_mix(db)


def db_at(pos: float) -> float:
    """The dB at a slider position 0..3 (the four names at 0, 1, 2, 3; in between, the dB between them)."""
    p = min(max(float(pos), 0.0), len(POINTS) - 1.0)
    i = min(int(p), len(POINTS) - 2)
    return POINTS[i] + (POINTS[i + 1] - POINTS[i]) * (p - i)


def pos_of(db: float) -> float:
    """The slider position 0..3 of a strength in dB."""
    d = min(max(float(db), POINTS[0]), POINTS[-1])
    i = 0
    while i < len(POINTS) - 2 and d > POINTS[i + 1]:
        i += 1
    return i + (d - POINTS[i]) / (POINTS[i + 1] - POINTS[i])


def snap(db: float) -> float:
    """A strength the slider is at: a named one when it is within half a dB of it."""
    for _, _, named in NAMES:
        if abs(named - db) < 0.5:
            return named
    return float(db)


def strength_key(db: float) -> str | None:
    """"gentle" ... "max" when `db` is one of the named strengths, else None."""
    return next((key for key, _, named in NAMES if abs(named - db) < 0.5), None)


def strength_name(db: float) -> str:
    """"Balanced", or "Between Balanced and Strong"."""
    for _, label, named in NAMES:
        if abs(named - db) < 0.5:
            return label
    i = min(int(pos_of(db)), len(NAMES) - 2)
    return f"Between {NAMES[i][1]} and {NAMES[i + 1][1]}"


def db_of_key(key: str) -> float:
    return next(db for k, _, db in NAMES if k == key)


# ---------------------------------------------------------------------------------------------------- loudness match

@dataclass(frozen=True)
class Sums:
    """Sums that give the loudness of any blend in closed form: E(a) = (1 - a)^2 ef + a^2 ei + 2 a (1 - a) cfi."""
    ef: float
    ei: float
    cfi: float


def energy_sums(full: np.ndarray, x: np.ndarray, channels: list[int]) -> Sums:
    """The sums over the kept `channels`, block by block (so memory-mapped copies are never loaded whole)."""
    ef = ei = cfi = 0.0
    for ch in channels:
        for s in range(0, len(full), _SUMS_BLOCK):
            f = np.asarray(full[s:s + _SUMS_BLOCK, ch], dtype=np.float64)
            r = np.asarray(x[s:s + _SUMS_BLOCK, ch], dtype=np.float64)
            ef += float(np.dot(f, f))
            ei += float(np.dot(r, r))
            cfi += float(np.dot(f, r))
    return Sums(ef, ei, cfi)


def match_gain(sums: Sums | None, a: float) -> float:
    """The gain that gives a blend with share `a` the loudness of the full-strength sound: while listening, moving the slider
    changes the noise, not the volume (the website does the same)."""
    if sums is None:
        return 1.0
    e = (1 - a) ** 2 * sums.ef + a * a * sums.ei + 2 * a * (1 - a) * sums.cfi
    if not e > 0 or not sums.ef > 0:
        return 1.0
    return float(max(0.1, min(3.0, np.sqrt(sums.ef / e))))


# ---------------------------------------------------------------------------------------------------- the blended sound

def blend_block(full: np.ndarray, x: np.ndarray, plan: list[int | None], a: float, gain: float = 1.0) -> np.ndarray:
    """One block of the result for a share `a`: for every channel of the result, the blend of the full-strength copy and the
    recording in the channel `plan` names (None: silent). At gain 1 this is, bit for bit, what pipeline.blend_strength puts in
    the file (float64 arithmetic stored as float32). `full` and `x` hold the same frames, one column per channel of the recording."""
    out = np.zeros((len(full), len(plan)), dtype=np.float32)
    for c, src in enumerate(plan):
        if src is None:
            continue
        f = np.asarray(full[:, src], dtype=np.float64)
        if a > 0.0:
            f = (1.0 - a) * f + a * np.asarray(x[:, src], dtype=np.float64)
        out[:, c] = (f * gain if gain != 1.0 else f).astype(np.float32)
    return out


class BlendStream:
    """What the player reads: the saved full-strength copy and the original (the same frames, lined up sample for sample,
    as the pipeline lines them up), mixed at a strength that can be changed at any moment, from any thread.

    `full` and `x`: (frames, channels) float32 arrays, usually memory-mapped. `plan`: per channel of the sound that plays, the
    channel of the copy it is (pipeline.channel_plan). `exact`: no loudness match (the saved file is this blend, nothing after it)."""

    def __init__(self, full: np.ndarray, x: np.ndarray, plan: list[int | None], *, exact: bool = False,
                 sums: Sums | None = None):
        if len(full) != len(x):
            raise ValueError("the saved copy and the original are not the same length")
        self.full, self.x, self.plan = full, x, list(plan)
        self.frames = len(full)
        self.channels = len(self.plan)
        self.sums = sums if sums is not None else energy_sums(full, x, sorted({c for c in self.plan if c is not None}))
        self._exact = bool(exact)
        self._lock = threading.Lock()
        self._db = 100.0
        self._original = False
        self.set_strength(100.0)

    @property
    def exact(self) -> bool:
        return self._exact

    @exact.setter
    def exact(self, value: bool) -> None:
        """True when the saved file is this blend and nothing after it (Leveling off): no loudness match then."""
        self._exact = bool(value)
        self.set_strength(self._db)

    @property
    def seconds(self) -> float:
        return self.frames / SAMPLE_RATE

    @property
    def db(self) -> float:
        return self._db

    @property
    def original(self) -> bool:
        return self._original

    def set_strength(self, db: float) -> None:
        a = mix_of(db)
        gain = 1.0 if self._exact else match_gain(self.sums, a)
        with self._lock:
            self._db, self._a, self._gain = float(db), a, gain

    def set_original(self, on: bool) -> None:
        """Play the recording as it is (at the volume of the cleaned sound, unless the saved file is exact)."""
        with self._lock:
            self._original = bool(on)

    def mix(self) -> tuple[float, float, bool]:
        """(share of the recording, gain, playing the original) as they are now."""
        with self._lock:
            return self._a, self._gain, self._original

    def read(self, start: int, count: int) -> np.ndarray:
        """`count` frames from frame `start` (fewer at the end): (n, channels) float32."""
        start = max(0, int(start))
        end = min(self.frames, start + max(0, int(count)))
        if end <= start:
            return np.zeros((0, self.channels), dtype=np.float32)
        a, gain, original = self.mix()
        if original:
            g = 1.0 if self._exact else match_gain(self.sums, 1.0)
            out = np.zeros((end - start, self.channels), dtype=np.float32)
            for c in range(self.channels):  # the recording as it is, every channel (a silent side stays what it was)
                col = min(c, self.x.shape[1] - 1)
                out[:, c] = np.asarray(self.x[start:end, col], dtype=np.float32) * np.float32(g)
            return out
        return blend_block(self.full[start:end], self.x[start:end], self.plan, a, gain)

    def read_pcm16(self, start: int, count: int) -> bytes:
        """The same, as 16-bit little-endian samples (clipped), interleaved: what the sound card is given."""
        block = self.read(start, count)
        return np.rint(np.clip(block, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
