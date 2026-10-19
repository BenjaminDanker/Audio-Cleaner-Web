"""Measuring a recording the way the website does, so that the desktop app decides the same ("Automatic best settings").

This is a port of web/src/media/analysis.js (the Analyser class and what its result() returns) and of the way
web/src/media/analyse.js chooses what to measure: the whole of a short file, otherwise six stretches of ten seconds spread
evenly over it (pipeline.measure_windows, the same ones the noise figure is measured on). Everything works on 48 kHz audio
(pipeline.SAMPLE_RATE), mono or stereo (a stereo file is analysed as two channels and a mono mix).

The trained model (reco_model.py) splits on exact thresholds of 14 of these numbers, so they are computed the same way and
in the same order as in the browser, not merely "about the same": 10 ms frame levels of the mono mix and of the speech band
(150 Hz to 3.5 kHz, two Butterworth biquads), kept as 32-bit floats like the browser's arrays; 2048-point Hann spectra every
1024 samples with no frame across the join of two stretches; the long-term speech spectrum, its 200 Hz median smoothing,
the bandwidth edge and the tilt; the 95th percentile of how fast the level falls in 60 ms; peak and clipping; BS.1770
loudness with its 100 ms blocks and two gates. spectrum.effective_bandwidth, spectrum.tilt_db and advice.drop_db measure the same
things but differ in small ways (they read the stretches as one piece, in 32-bit floats); this module follows the browser
exactly, and tests/test_reco_agreement.py (server) and tests/test_analysis.py (here) keep it so.

    stats = analysis.measure(x, libdf_path, model_path)      # x: (samples, channels) float32 at 48 kHz
    stats = analysis.measure_features(x, noise_db=18.2)      # when the noise figure is already known
    stats["bandwidthHz"], stats["dropDb"], ...               # the names of the website's Analyser.result()

The noise figure ("noiseDb") is pipeline.noise_figure_db: DeepFilterNet3 at full strength over the same stretches, what it
keeps against what it takes out. The browser's NoiseFigure is the same sum, with the same model running in WebAssembly.
"""
from __future__ import annotations

import math
import threading
from pathlib import Path

import numpy as np

from . import dfn
from .pipeline import SILENCE_DB, measure_windows

RATE = dfn.SAMPLE_RATE  # 48000
FRAME = RATE // 100  # 10 ms
NFFT = 2048
HOP = 1024
BINS = NFFT // 2 + 1
BIN_HZ = RATE / NFFT
MAX_FFT_FRAMES = 6000
# The speech band's level moves at least this much (90th minus 10th percentile, dB) when someone speaks.
SPEECH_BAND_DYN_DB = 6.8
CLIP_LEVEL = 0.999
CLIP_RUN = 3
DROP_FRAMES = 6  # 60 ms
EDGE_DB = 20.0  # bandwidth: the edge is where the speech spectrum is this far below its passband level
CLIFF_DB = 18.0  # ... and the level falls by at least this much from just below to just above the edge
DEAD_CHANNEL_PEAK = 1e-4

#: The measurements the trained model uses, in its order (hasSpeech as 0/1; a missing one is NaN).
FEATURES = ("floorDb", "levelDb", "peakDb", "silentFraction", "speechFraction", "bandDynDb", "hasSpeech", "bandwidthHz",
            "tiltDb", "dropDb", "clipFraction", "peak", "loudnessLufs", "noiseDb")

# BS.1770 K-weighting at 48 kHz: shelf, then high-pass.
_K1_B = (1.53512485958697, -2.69169618940638, 1.19839281085285)
_K1_A = (1.0, -1.69065929318241, 0.73248077421585)
_K2_B = (1.0, -2.0, 1.0)
_K2_A = (1.0, -1.99004745483398, 0.99007225036621)


def _biquad(kind: str, fc: float) -> list[float]:
    """RBJ biquad (Butterworth, Q = 1/sqrt 2) at 48 kHz as a second-order section [b0, b1, b2, 1, a1, a2]."""
    w = 2 * math.pi * fc / RATE
    c = math.cos(w)
    alpha = math.sin(w) / (2 * math.sqrt(0.5))
    a0 = 1 + alpha
    b = [(1 + c) / 2, -(1 + c), (1 + c) / 2] if kind == "high" else [(1 - c) / 2, 1 - c, (1 - c) / 2]
    return [b[0] / a0, b[1] / a0, b[2] / a0, 1.0, -2 * c / a0, (1 - alpha) / a0]


_BAND_SOS = np.array([_biquad("high", 150), _biquad("low", 3500)])
_HANN = np.hanning(NFFT)  # 0.5 - 0.5 cos(2 pi i / (N - 1)), as the browser's
_LO_BIN = math.ceil(150 / BIN_HZ)
_HI_BIN = math.floor(3500 / BIN_HZ)


def percentile(sorted_values: np.ndarray, q: float) -> float:
    """numpy.percentile (linear interpolation) of an already sorted array, q in 0..100, with the browser's arithmetic."""
    n = len(sorted_values)
    if not n:
        return math.nan
    pos = (q / 100) * (n - 1)
    lo = math.floor(pos)
    hi = min(n - 1, lo + 1)
    a = float(sorted_values[lo])
    return a + (float(sorted_values[hi]) - a) * (pos - lo)


# ---- one stretch of the file -------------------------------------------------------------------------------------
def _frame_sums(sq: np.ndarray, size: int) -> np.ndarray:
    n = len(sq) // size
    return sq[:n * size].reshape(n, size).sum(axis=1) if n else np.zeros(0)


def _levels(mono: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """dB of the 10 ms frames of the mono mix and of its speech band, as 32-bit floats (the partial last frame is dropped)."""
    from scipy.signal import sosfilt

    x = mono.astype(np.float64)
    band = sosfilt(_BAND_SOS, x)
    lv = 10 * np.log10(_frame_sums(x * x, FRAME) / FRAME + 1e-12)
    bl = 10 * np.log10(_frame_sums(band * band, FRAME) / FRAME + 1e-12)
    return lv.astype(np.float32), bl.astype(np.float32)


def _spectra(mono: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Power spectra (float32, bins 0..1024) of the 2048-sample Hann frames every 1024 samples, and the energy of each one
    between 150 Hz and 3.5 kHz."""
    import scipy.fft

    n = len(mono)
    if n < NFFT:
        return np.zeros((0, BINS), np.float32), np.zeros(0)
    frames = np.lib.stride_tricks.sliding_window_view(mono, NFFT)[::HOP]
    spectra = np.empty((len(frames), BINS), np.float32)
    energy = np.empty(len(frames))
    step = 512
    for s in range(0, len(frames), step):
        block = frames[s:s + step].astype(np.float64) * _HANN
        f = scipy.fft.rfft(block, axis=1)
        p = (f.real * f.real + f.imag * f.imag).astype(np.float32)
        spectra[s:s + step] = p
        energy[s:s + step] = p[:, _LO_BIN:_HI_BIN + 1].astype(np.float64).sum(axis=1)
    return spectra, energy


def _clipped(x: np.ndarray) -> int:
    """Samples in runs of at least CLIP_RUN at or above CLIP_LEVEL (the run's first CLIP_RUN count when it reaches that length)."""
    hit = np.abs(x).astype(np.float64) >= CLIP_LEVEL
    if not hit.any():
        return 0
    d = np.diff(np.concatenate(([0], hit.view(np.int8), [0])))
    lengths = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
    return int(lengths[lengths >= CLIP_RUN].sum())


def _loudness_blocks(planes: list[np.ndarray]) -> np.ndarray:
    """Mean-square K-weighted power of the 400 ms blocks (every 100 ms) of one stretch; none shorter than 0.4 s."""
    from scipy.signal import lfilter

    power = None
    for p in planes:
        y = lfilter(_K2_B, _K2_A, lfilter(_K1_B, _K1_A, p.astype(np.float64)))
        power = y * y if power is None else power + y * y
    sub = RATE // 10
    subs = _frame_sums(power, sub) / sub
    if len(subs) < 4:
        return np.zeros(0)
    return (subs[:-3] + subs[1:-2] + subs[2:-1] + subs[3:]) / 4


class _Parts:
    """What was measured on each stretch, put together."""

    def __init__(self) -> None:
        self.levels: list[np.ndarray] = []
        self.band_levels: list[np.ndarray] = []
        self.spectra: list[np.ndarray] = []
        self.energy: list[np.ndarray] = []
        self.blocks: list[np.ndarray] = []
        self.samples = 0
        self.peak = [0.0, 0.0]
        self.clipped = [0, 0]


def _add_stretch(parts: _Parts, planes: list[np.ndarray]) -> None:
    n = len(planes[0])
    if not n:
        return
    mono = planes[0] if len(planes) == 1 else (planes[0] + planes[1]) * np.float32(0.5)
    parts.samples += n
    lv, bl = _levels(mono)
    parts.levels.append(lv)
    parts.band_levels.append(bl)
    sp, en = _spectra(mono)
    parts.spectra.append(sp)
    parts.energy.append(en)
    for c, p in enumerate(planes):
        parts.peak[c] = max(parts.peak[c], float(np.max(np.abs(p))))
        parts.clipped[c] += _clipped(p)
    parts.blocks.append(_loudness_blocks(planes))


# ---- the numbers --------------------------------------------------------------------------------------------------
def _ltas_db(spectra: np.ndarray, energy: np.ndarray) -> np.ndarray | None:
    """Long-term spectrum (dB per bin) of the active frames, or None when there is too little to say."""
    if not len(spectra):
        return None
    top = float(energy.max())
    if top <= 0:
        return None
    thr = max(percentile(np.sort(energy), 60), top * 1e-4)
    act = np.flatnonzero(energy >= thr)
    if len(act) < 8:
        return None
    if len(act) > MAX_FFT_FRAMES:
        m = len(act)
        act = act[(np.arange(MAX_FFT_FRAMES) * (m - 1)) // (MAX_FFT_FRAMES - 1)]
    total = spectra[act].astype(np.float64).sum(axis=0)
    return 10 * np.log10(total / len(act) + 1e-20)


_FREQ = np.arange(BINS) * BIN_HZ


def _band_mean(db: np.ndarray, lo: float, hi: float) -> float:
    sel = (_FREQ >= lo) & (_FREQ < hi)
    return float(db[sel].sum() / sel.sum()) if sel.any() else math.nan


def bandwidth_and_tilt(spectra: np.ndarray, energy: np.ndarray) -> tuple[float, float | None]:
    """Where the speech spectrum ends (Hz; RATE / 2 when there is no sharp cut-off) and how far the top of the speech band
    (4-8 kHz) sits below its middle (0.6-2.4 kHz, dB; None when there is too little speech). The browser's bandwidthOf."""
    from scipy.signal import medfilt

    nyq = RATE / 2
    raw = _ltas_db(spectra, energy)
    if raw is None:
        return nyq, None
    db = medfilt(raw, max(3, int(math.floor(200 / BIN_HZ + 0.5)) | 1))
    ref = _band_mean(db, 600, 2400)
    top = min(nyq * 0.95, 20000.0)
    tilt = _band_mean(db, 4000, 8000) - ref
    for i in range(BINS):
        f = _FREQ[i]
        if not (2400 <= f < top and db[i] < ref - EDGE_DB):
            continue
        fe = float(f)
        if _band_mean(db, fe * 1.15, nyq) >= ref - EDGE_DB:
            continue  # does not stay down: a dip, not an edge
        if _band_mean(db, fe * 0.8, fe * 0.9) - _band_mean(db, fe * 1.15, fe * 1.4) < CLIFF_DB:
            continue  # shallow slope: natural roll-off
        local = _band_mean(db, fe * 0.8, fe * 0.9)
        j = np.flatnonzero((_FREQ >= fe * 0.9) & (db < local - 12.0))
        if len(j):
            return float(_FREQ[j[0]]), tilt
    return nyq, tilt


def drop_of(levels: np.ndarray, floor_db: float, peak_db: float) -> float | None:
    """The 95th percentile of how far the level falls within 60 ms, from the loud parts of the 10 ms frame levels (dB). None
    when the background is too close to the voice to see it (under 30 dB between them) or there is too little loud sound."""
    if peak_db - floor_db < 30:
        return None
    lv = levels.astype(np.float64)
    if len(lv) <= DROP_FRAMES:
        return None
    head = lv[:-DROP_FRAMES]
    drops = (head - lv[DROP_FRAMES:])[head >= peak_db - 10]
    if len(drops) < 30:
        return None
    return percentile(np.sort(drops), 95)


def loudness(blocks: np.ndarray) -> float | None:
    """Integrated loudness (LUFS, BS.1770 gating) of the 400 ms blocks' powers, or None when there is not 0.4 s of sound."""
    blocks = blocks[blocks > 0]
    lk = -0.691 + 10 * np.log10(blocks) if len(blocks) else blocks
    gated = blocks[lk > -70]
    if not len(gated):
        return None
    rel = -0.691 + 10 * math.log10(float(gated.sum()) / len(gated)) - 10
    kept = gated[(-0.691 + 10 * np.log10(gated)) > rel]
    return -0.691 + 10 * math.log10(float(kept.sum()) / len(kept)) if len(kept) else None


def _result(parts: _Parts, channels: int, noise_db: float | None) -> dict:
    seconds = parts.samples / RATE
    lv = np.concatenate(parts.levels) if parts.levels else np.zeros(0, np.float32)
    bl = np.concatenate(parts.band_levels) if parts.band_levels else np.zeros(0, np.float32)
    loud_mask = lv.astype(np.float64) > SILENCE_DB
    loud = lv[loud_mask].astype(np.float64)
    silent = len(lv) - len(loud)
    srt = np.sort(loud)
    enough = len(loud) >= 10
    nan = math.nan
    p10 = percentile(srt, 10) if enough else nan
    p90 = percentile(srt, 90) if enough else nan
    p95 = percentile(srt, 95) if enough else nan
    # share of the frames that are clearly above the room: speech, as far as a level alone can tell
    gate = max(p10 + 12, p95 - 25) if enough else math.inf
    speech_fraction = float(np.count_nonzero(loud >= gate)) / len(loud) if enough else 0.0
    spectra = np.concatenate(parts.spectra)[:MAX_FFT_FRAMES * 4] if parts.spectra else np.zeros((0, BINS), np.float32)
    energy = np.concatenate(parts.energy)[:MAX_FFT_FRAMES * 4] if parts.energy else np.zeros(0)
    if len(spectra) >= 3:
        bandwidth, tilt = bandwidth_and_tilt(spectra, energy)
    else:
        bandwidth, tilt = RATE / 2, None
    clip_fraction = max(parts.clipped[:channels]) / parts.samples if parts.samples else 0.0
    drop = drop_of(lv, p10, p95) if enough else None
    band = np.sort(bl[loud_mask].astype(np.float64))
    band_dyn = percentile(band, 90) - percentile(band, 10) if len(band) >= 10 else 0.0
    blocks = np.concatenate(parts.blocks) if parts.blocks else np.zeros(0)
    return {
        "seconds": seconds,
        "noiseDb": noise_db,
        "floorDb": p10,
        "levelDb": p90,
        "peakDb": p95,
        "silentFraction": silent / len(lv) if len(lv) else 1.0,
        "speechFraction": speech_fraction,
        "bandDynDb": band_dyn,
        "hasSpeech": bool(enough and band_dyn >= SPEECH_BAND_DYN_DB),
        "bandwidthHz": bandwidth,
        "tiltDb": tilt,
        "dropDb": drop,
        "clipFraction": clip_fraction,
        "peak": max(parts.peak[:channels]),
        "loudnessLufs": loudness(blocks),
    }


# ---- the whole recording -------------------------------------------------------------------------------------------
def _planes(seg: np.ndarray) -> tuple[list[np.ndarray], int]:
    """The channels the browser analyses: two for a stereo recording, else the mono mix (float32 arrays of one length)."""
    seg = np.asarray(seg, dtype=np.float32)
    if seg.ndim == 1:
        return [np.ascontiguousarray(seg)], 1
    if seg.shape[1] == 2:
        return [np.ascontiguousarray(seg[:, 0]), np.ascontiguousarray(seg[:, 1])], 2
    if seg.shape[1] == 1:
        return [np.ascontiguousarray(seg[:, 0])], 1
    return [np.ascontiguousarray(seg.mean(axis=1, dtype=np.float32))], 1


def measure_features(x: np.ndarray, noise_db: float | None = None) -> dict:
    """The website's measurements (Analyser.result()) of the audio `x` ((samples,) or (samples, channels), 48 kHz): the
    stretches pipeline.measure_windows chooses (all of a short recording, else about a minute spread over it), read from
    `x` only there, so a memory-mapped file of any length is fine. `noise_db` goes into the result as "noiseDb"."""
    n = len(x)
    parts = _Parts()
    channels = 1
    for a, b in measure_windows(n):
        planes, channels = _planes(x[a:b])
        _add_stretch(parts, planes)
    return _result(parts, channels, noise_db)


def measure(x: np.ndarray, lib_path: str | Path, model_path: str | Path, cancel: threading.Event | None = None,
            keep: dict | None = None) -> dict:
    """measure_features plus the noise figure (pipeline.noise_figure_db: the noise remover runs over the same stretches).
    With `keep`, a measurement of the whole recording at once is kept there as keep["measured"] (pipeline.Measured), which
    clean_audio(measured=...) uses instead of running the noise remover again. Setting `cancel` raises dfn.Cancelled."""
    from . import pipeline

    figure = pipeline.noise_figure_db(x, lib_path, model_path, cancel, keep)
    return measure_features(x, figure)
