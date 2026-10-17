"""Long pauses that need not go through the noise model.

`pipeline.sound_spans` already skips digital silence. A pause with room noise in it is not silent, but when the
noise model leaves almost nothing of it (broadband hiss: -50 dBFS in, below -75 dBFS out) running the model over
it is wasted work. This module cuts such pauses out of the signal the model sees and fills them afterwards:

  * a Silero VAD finds the stretches without speech (`speech_probs`); only the middle of a long one becomes a
    hole: the VAD's padding and `CONTEXT_S` of pause on each side stay in, so the model settles on the noise and
    nothing near speech is ever cut;
  * the holes are cut out of the audio (`Spliced`), so the model runs on one continuous signal with its state
    adapted (a cold restart after every hole gave 25 dB SNR against an uncut run, this 50 dB), on the 10 ms hop
    grid so the frames after a hole keep their phase;
  * a hole is only used when a probe shows the model really removes this recording's pause noise: its predicted
    output must lie `GATE_DB` below the speech level (`plan`). Otherwise nothing is skipped at all;
  * the hole is filled with the original scaled by the attenuation the model applied around it, with short fades
    (`fill_holes`).

Measured on 12 LibriSpeech recordings with synthetic room noise, the cleaned file differs from the uncut run by
48-50 dB SNR where the gate opens, and the gate stays shut where it would be 14-42 dB (docs/research/
skip-noisy-pauses-2026-10.md)."""
from __future__ import annotations

import math
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.signal import resample_poly

SAMPLE_RATE = 48_000
HOP = SAMPLE_RATE // 100            # the noise models work in 10 ms frames: holes sit on this grid
VAD_FRAME = 1536                    # samples at 48 kHz per Silero decision (512 at 16 kHz = 32 ms)
VAD_THRESHOLD = 0.1                 # speech probability above this counts as speech (0.5 missed whispers by up to 1 s)
VAD_PAD_S = 0.4                     # pause kept around anything the VAD calls speech
CONTEXT_S = 0.5                     # more pause kept on each side of a hole: the model settles on it
MIN_HOLE_S = 0.5                    # shorter holes are not worth the seam
MIN_FILE_S = 180.0                  # shorter recordings can not save what the VAD pass and the probes cost
MIN_CANDIDATE_S = 60.0              # candidate holes must add up to this before the probes are run
MIN_SAVING_S = 30.0                 # and the holes that pass the gate to this
GATE_DB = 52.0                      # predicted model output in a hole must be this far below the speech level
EVENT_DB = 12.0                     # a hole whose loudest 10 ms frame is this far over its median holds an event
FADE_S = 0.04                       # the model's own output fades into / out of the fill over this long
PROBES = 3
PROBE_LEAD_S = 6.0                  # audio before a probed hole, for the model to settle
PROBE_HOLE_S = 1.5                  # of the hole itself; the last second of it is measured
LEVEL_FLOOR = 1e-12                 # mean square floor (-120 dBFS)

_CHUNK_FRAMES = 256                 # VAD frames per independent stream (8.2 s) ...
_LEAD_FRAMES = 32                   # ... each run from 1 s earlier so its state is settled when its own part begins
_BATCH = 32

_sessions: dict[str, object] = {}
_sessions_lock = threading.Lock()


def _db(mean_square: float) -> float:
    return 10.0 * math.log10(max(float(mean_square), LEVEL_FLOOR))


def _session(model_path: str):
    with _sessions_lock:
        s = _sessions.get(model_path)
        if s is None:
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.intra_op_num_threads = 1  # a tiny network on 32 ms frames: one thread is all it can use
            so.inter_op_num_threads = 1
            s = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
            _sessions[model_path] = s
        return s


def speech_probs(x: np.ndarray, model_path: str | Path, cancel: threading.Event | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Silero speech probability and mean square per VAD frame (32 ms) of mono 48 kHz audio `x` (any 1-D float view,
    read block by block). The file is cut into streams of 8 s that run side by side as one batch, each from 1 s
    earlier so that its state has settled where its own part begins."""
    from acw_common.dfn import Cancelled

    sess = _session(str(model_path))
    n = len(x)
    n_frames = -(-n // VAD_FRAME)
    probs = np.zeros(n_frames, dtype=np.float32)
    energy = np.zeros(n_frames, dtype=np.float64)
    n_chunks = -(-n_frames // _CHUNK_FRAMES)
    sr = np.array([16000], dtype=np.int64)
    for c0 in range(0, n_chunks, _BATCH):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        chunks = list(range(c0, min(c0 + _BATCH, n_chunks)))
        width = (_LEAD_FRAMES + _CHUNK_FRAMES) * VAD_FRAME
        seg = np.zeros((len(chunks), width), dtype=np.float32)
        for k, c in enumerate(chunks):
            a = (c * _CHUNK_FRAMES - _LEAD_FRAMES) * VAD_FRAME
            lo, hi = max(a, 0), min(a + width, n)
            if hi > lo:
                seg[k, lo - a:hi - a] = x[lo:hi]
        a16 = resample_poly(seg, 1, 3, axis=1).astype(np.float32, copy=False)
        state = np.zeros((2, len(chunks), 128), dtype=np.float32)
        ctx = np.zeros((len(chunks), 64), dtype=np.float32)
        for t in range(_LEAD_FRAMES + _CHUNK_FRAMES):
            frame = a16[:, t * 512:(t + 1) * 512]
            out, state = sess.run(["output", "stateN"], {"input": np.concatenate([ctx, frame], axis=1), "state": state, "sr": sr})
            ctx = frame[:, -64:]
            if t >= _LEAD_FRAMES:
                for k, c in enumerate(chunks):
                    i = c * _CHUNK_FRAMES + t - _LEAD_FRAMES
                    if i < n_frames:
                        probs[i] = out[k, 0]
        ms = np.mean(np.square(seg[:, _LEAD_FRAMES * VAD_FRAME:].reshape(len(chunks), _CHUNK_FRAMES, VAD_FRAME), dtype=np.float64), axis=2)
        for k, c in enumerate(chunks):
            i0 = c * _CHUNK_FRAMES
            m = min(_CHUNK_FRAMES, n_frames - i0)
            energy[i0:i0 + m] = ms[k, :m]
    return probs, energy


def candidate_holes(probs: np.ndarray, n: int) -> list[tuple[int, int]]:
    """Holes (sample ranges on the HOP grid) in the stretches without speech: each such stretch, widened by VAD_PAD_S
    around every speech frame, loses CONTEXT_S on each side that touches speech. A stretch at the very start or end
    of the recording has no speech on that side, so keeps no context there."""
    speech = binary_dilation(probs > VAD_THRESHOLD, iterations=int(math.ceil(VAD_PAD_S * SAMPLE_RATE / VAD_FRAME)))
    holes: list[tuple[int, int]] = []
    ctx = int(CONTEXT_S * SAMPLE_RATE)
    i, nf = 0, len(speech)
    while i < nf:
        if speech[i]:
            i += 1
            continue
        j = i
        while j < nf and not speech[j]:
            j += 1
        a, b = i * VAD_FRAME, min(n, j * VAD_FRAME)
        s = 0 if i == 0 else -(-(a + ctx) // HOP) * HOP
        e = n if j >= nf else (b - ctx) // HOP * HOP
        if e - s >= MIN_HOLE_S * SAMPLE_RATE:
            holes.append((s, e))
        i = j
    return holes


_BLOCK = SAMPLE_RATE * 30


def _hole_level(x: np.ndarray, s: int, e: int) -> tuple[float, bool]:
    """(level in dBFS, whether it is steady: no 10 ms frame EVENT_DB over the median) of x[s:e], read in blocks."""
    total = 0.0
    frames: list[np.ndarray] = []
    for a in range(s, e, _BLOCK):
        b = np.asarray(x[a:min(e, a + _BLOCK)], dtype=np.float64)
        total += float(np.sum(np.square(b)))
        k = len(b) // HOP
        if k:
            frames.append(np.mean(np.square(b[:k * HOP]).reshape(k, HOP), axis=1))
    if not frames:
        return _db(0.0), False
    db = 10.0 * np.log10(np.concatenate(frames) + LEVEL_FLOOR)
    return _db(total / max(e - s, 1)), bool(np.max(db) - np.median(db) <= EVENT_DB)


def plan(
    x: np.ndarray,
    vad_model: str | Path,
    denoise: Callable[[np.ndarray], np.ndarray],
    *,
    cancel: threading.Event | None = None,
    probs_energy: tuple[np.ndarray, np.ndarray] | None = None,
    stats: dict | None = None,
) -> list[tuple[int, int]]:
    """The holes to cut out of mono 48 kHz `x` before the noise model, or [] (usually: nothing to save, or the model
    leaves too much of this recording's pauses for them to be skipped invisibly). `denoise(window)` runs the noise
    model on a short mono array and returns the result (used for the probes). `probs_energy` replaces the VAD run."""
    n = len(x)
    if n < MIN_FILE_S * SAMPLE_RATE:
        return []
    probs, energy = probs_energy if probs_energy is not None else speech_probs(x, vad_model, cancel)
    holes = candidate_holes(probs, n)
    if sum(e - s for s, e in holes) < MIN_CANDIDATE_S * SAMPLE_RATE:
        return []
    voiced = probs > 0.5
    if voiced.sum() < 5 * SAMPLE_RATE / VAD_FRAME:
        return []
    speech_db = _db(float(np.mean(energy[voiced])))  # (with the noise under it; the cleaned speech is within a dB or two)
    levels, kept = [], []
    for s, e in holes:
        level, steady = _hole_level(x, s, e)
        if steady:
            levels.append(level)
            kept.append((s, e))
    holes = kept
    if not holes:
        return []
    # Probe the loudest holes with the real model: audio from before the hole (so the model has settled) to a bit
    # into it, level of the last second in and out.
    order = sorted(range(len(holes)), key=lambda i: -levels[i])[:PROBES]
    ratio = None  # dB the model takes off the pause noise of this recording (smallest seen)
    worst_level = -200.0
    for i in order:
        s, e = holes[i]
        lead = int(PROBE_LEAD_S * SAMPLE_RATE)
        stop = min(e, s + int(PROBE_HOLE_S * SAMPLE_RATE))
        lo = max(0, s - lead)
        window = np.ascontiguousarray(x[lo:stop], dtype=np.float32)
        y = denoise(window)
        m = min(stop - s, SAMPLE_RATE) // HOP * HOP
        if m <= 0:
            continue
        lin = _db(float(np.mean(np.square(window[-m:], dtype=np.float64))))
        lout = _db(float(np.mean(np.square(np.asarray(y[-m:], dtype=np.float64)))))
        ratio = lin - lout if ratio is None else min(ratio, lin - lout)
        worst_level = max(worst_level, lin)
    if ratio is None:
        return []
    chosen = [h for h, lvl in zip(holes, levels) if lvl <= worst_level + 3.0 and lvl - ratio <= speech_db - GATE_DB]
    if stats is not None:
        stats.update(speech_db=speech_db, attenuation_db=ratio, candidates=len(holes), chosen=len(chosen),
                     worst_pause_db=worst_level)
    if sum(e - s for s, e in chosen) < MIN_SAVING_S * SAMPLE_RATE:
        return []
    return chosen


class Spliced:
    """A 1-D array with the sample ranges `holes` (sorted, disjoint) cut out, without copying: length, slices and
    slice assignment work on the shortened signal and read from / write to `base`. Enough for dfn.denoise_array."""

    ndim = 1

    def __init__(self, base: np.ndarray, holes: list[tuple[int, int]]):
        self.base = base
        self._parts: list[tuple[int, int, int]] = []  # (position here, start in base, end in base)
        pos = cur = 0
        for s, e in holes:
            if s > cur:
                self._parts.append((pos, cur, s))
                pos += s - cur
            cur = e
        if len(base) > cur:
            self._parts.append((pos, cur, len(base)))
            pos += len(base) - cur
        self._n = pos

    @property
    def shape(self) -> tuple[int]:
        return (self._n,)

    def __len__(self) -> int:
        return self._n

    def _bounds(self, key) -> tuple[int, int]:
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("Spliced takes plain slices")
        lo, hi, _ = key.indices(self._n)
        return lo, max(lo, hi)

    def __getitem__(self, key) -> np.ndarray:
        lo, hi = self._bounds(key)
        out = np.empty(hi - lo, dtype=self.base.dtype)
        for pos, a, b in self._parts:
            p0, p1 = max(lo, pos), min(hi, pos + b - a)
            if p1 > p0:
                out[p0 - lo:p1 - lo] = self.base[a + p0 - pos:a + p1 - pos]
        return out

    def __setitem__(self, key, value) -> None:
        lo, hi = self._bounds(key)
        v = np.broadcast_to(np.asarray(value), (hi - lo,))
        for pos, a, b in self._parts:
            p0, p1 = max(lo, pos), min(hi, pos + b - a)
            if p1 > p0:
                self.base[a + p0 - pos:a + p1 - pos] = v[p0 - lo:p1 - lo]

    def __array__(self, dtype=None, copy=None):
        out = self[:]
        return out if dtype is None else out.astype(dtype, copy=False)


def fill_holes(x: np.ndarray, y: np.ndarray, holes: list[tuple[int, int]]) -> None:
    """Fill the holes of the cleaned signal `y` (which holds the model's output everywhere else) with the original
    `x` scaled by the attenuation the model applied just outside (interpolated across the hole), and fade the
    model's output over FADE_S into and out of the fill: the seam is where the model's own output was disturbed by the
    cut, and the level steps from the model's output to the scaled original by the level the scaling matched."""
    fade = int(FADE_S * SAMPLE_RATE)
    q = int(0.25 * SAMPLE_RATE)
    n = len(y)
    for s, e in holes:
        gl = gr = None
        if s > 0:
            lo = max(0, s - fade - q)
            gl = math.sqrt(max(float(np.mean(np.square(y[lo:s - fade], dtype=np.float64))), LEVEL_FLOOR) /
                           max(float(np.mean(np.square(x[lo:s - fade], dtype=np.float64))), LEVEL_FLOOR))
        if e < n:
            hi = min(n, e + fade + q)
            gr = math.sqrt(max(float(np.mean(np.square(y[e + fade:hi], dtype=np.float64))), LEVEL_FLOOR) /
                           max(float(np.mean(np.square(x[e + fade:hi], dtype=np.float64))), LEVEL_FLOOR))
        gl = gr if gl is None else gl
        gr = gl if gr is None else gr
        for a in range(s, e, _BLOCK):  # in blocks: a hole can be many minutes
            b = min(e, a + _BLOCK)
            y[a:b] = x[a:b] * (gl + (gr - gl) * (np.arange(a, b, dtype=np.float64) - s) / max(e - s, 1)).astype(np.float32)
        if s > 0:
            w = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            y[s - fade:s] = y[s - fade:s] * (1 - w) + x[s - fade:s] * np.float32(gl) * w
        if e < n:
            w = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            y[e:e + fade] = x[e:e + fade] * np.float32(gr) * (1 - w) + y[e:e + fade] * w


def denoise_with_pauses(x: np.ndarray, lib_path, model_path, vad_model, *, threads: int = 4, skip: bool = True,
                        report: dict | None = None) -> np.ndarray:
    """What pipeline.clean_audio would do for one channel of light-profile audio if pauses were skipped: plan the
    holes (probes through the same model), run the model on the spliced signal, fill the holes."""
    from acw_common import dfn

    y = np.empty(len(x), dtype=np.float32)
    holes = []
    if skip:
        holes = plan(x, vad_model, lambda w: dfn.denoise_array(w, lib_path, model_path, threads=1))
    if holes:
        dfn.denoise_array(Spliced(x, holes), lib_path, model_path, threads=threads, out=Spliced(y, holes))
        fill_holes(x, y, holes)
        if report is not None:
            report["pauses_skipped_s"] = sum(e - s for s, e in holes) / SAMPLE_RATE
    else:
        dfn.denoise_array(x, lib_path, model_path, threads=threads, out=y)
    return y
