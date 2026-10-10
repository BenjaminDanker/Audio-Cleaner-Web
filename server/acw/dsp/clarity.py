"""Post-denoise "clarity" chain for file jobs (vectorized NumPy/SciPy).

    denoised -> level match -> high-pass -> leveling compressor
             -> loudness normalize (EBU R128 / BS.1770) -> look-ahead peak limiter

Each stage is conservative by default: the goal is clean, consistent speech,
not a "processed" sound. All functions take/return float32 arrays shaped
(samples,) or (samples, channels).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import butter, lfilter, sosfilt


@dataclass
class ClarityParams:
    match_level: bool = True
    match_max_boost_db: float = 12.0
    highpass_hz: float = 80.0
    compress: bool = True
    comp_threshold_db: float = -24.0
    comp_ratio: float = 2.5
    comp_attack_ms: float = 8.0
    comp_release_ms: float = 120.0
    loudness_lufs: float | None = -16.0  # None keeps the level
    limit_ceiling_db: float = -1.0
    limit_lookahead_ms: float = 5.0
    limit_release_ms: float = 60.0


def _as2d(x: np.ndarray) -> np.ndarray:
    return x[:, None] if x.ndim == 1 else x


def _db(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-12))


def _lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else x.mean(axis=1)


def match_speech_level(original: np.ndarray, denoised: np.ndarray, sr: int, max_boost_db: float) -> np.ndarray:
    """DFN tends to lower speech level slightly. Scale the denoised signal so
    its loud (speech) frames match the original's, never attenuating."""
    o, d = _mono(original), _mono(denoised)
    if len(o) < sr:
        return denoised
    sos = butter(2, 150, btype="highpass", fs=sr, output="sos")
    win = int(0.05 * sr)
    env_o = np.sqrt(uniform_filter1d(sosfilt(sos, o) ** 2, win) + 1e-12)
    env_d = np.sqrt(uniform_filter1d(sosfilt(sos, d) ** 2, win) + 1e-12)
    # Frames where the cleaned signal is loud are speech (noise is gone there).
    thresh = np.percentile(env_d, 80)
    mask = env_d >= max(thresh, _lin(-50))
    if mask.mean() < 0.02:
        return denoised
    gain_db = float(np.clip(_db(float(np.median(env_o[mask] / env_d[mask]))), 0.0, max_boost_db))
    return (denoised * _lin(gain_db)).astype(np.float32)


def highpass(x: np.ndarray, sr: int, hz: float) -> np.ndarray:
    if hz <= 0:
        return x
    sos = butter(2, hz, btype="highpass", fs=sr, output="sos")
    return sosfilt(sos, x, axis=0).astype(np.float32)


def compress(x: np.ndarray, sr: int, threshold_db: float, ratio: float, attack_ms: float, release_ms: float) -> np.ndarray:
    """Feed-forward RMS compressor, linked across channels.

    Gain reduction is computed per sample from a 10 ms RMS detector, then a
    running minimum over the attack window gives a fast attack and a one-pole
    filter gives the release."""
    x2 = _as2d(x)
    det = np.sqrt(uniform_filter1d((x2 ** 2).max(axis=1), max(1, int(0.01 * sr))) + 1e-12)
    level_db = 20.0 * np.log10(det)
    over = np.maximum(level_db - threshold_db, 0.0)
    gr_db = -over * (1.0 - 1.0 / max(ratio, 1.0))
    attack = max(1, int(attack_ms * 1e-3 * sr))
    gr_db = minimum_filter1d(gr_db, attack, origin=-(attack // 2))
    a = float(np.exp(-1.0 / (release_ms * 1e-3 * sr)))
    gr_db = lfilter([1 - a], [1, -a], gr_db)
    gain = (10.0 ** (gr_db / 20.0)).astype(np.float32)
    y = x2 * gain[:, None]
    return y[:, 0] if x.ndim == 1 else y


def normalize_loudness(x: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    import pyloudnorm as pyln

    meter = pyln.Meter(sr)
    try:
        loud = meter.integrated_loudness(_as2d(x).astype(np.float64))
    except ValueError:
        return x
    if not np.isfinite(loud) or loud < -70:
        return x
    gain_db = float(np.clip(target_lufs - loud, -20.0, 30.0))
    return (x * _lin(gain_db)).astype(np.float32)


def limit(x: np.ndarray, sr: int, ceiling_db: float, lookahead_ms: float, release_ms: float) -> np.ndarray:
    """Look-ahead brickwall-ish peak limiter (sample peak, linked channels).
    Output is time-aligned with the input."""
    x2 = _as2d(x)
    ceiling = _lin(ceiling_db)
    peak = np.abs(x2).max(axis=1)
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-12))
    la = max(1, int(lookahead_ms * 1e-3 * sr))
    # Gain must already be down when the peak arrives: min over the next `la` samples.
    g = minimum_filter1d(need, la, origin=-(la // 2))
    a = float(np.exp(-1.0 / (release_ms * 1e-3 * sr)))
    # Release smoothing that never rises above the required gain.
    smoothed = lfilter([1 - a], [1, -a], g - 1.0) + 1.0
    g = np.minimum(g, smoothed)
    # Short attack smoothing (max-filter keeps it from overshooting).
    g = minimum_filter1d(g, max(1, la // 2))
    y = x2 * g[:, None].astype(np.float32)
    y = np.clip(y, -ceiling, ceiling)
    return y[:, 0] if x.ndim == 1 else y


def process(original: np.ndarray, denoised: np.ndarray, sr: int, p: ClarityParams | None = None) -> np.ndarray:
    p = p or ClarityParams()
    y = denoised.astype(np.float32, copy=False)
    if p.match_level:
        y = match_speech_level(original, y, sr, p.match_max_boost_db)
    y = highpass(y, sr, p.highpass_hz)
    if p.compress:
        y = compress(y, sr, p.comp_threshold_db, p.comp_ratio, p.comp_attack_ms, p.comp_release_ms)
    if p.loudness_lufs is not None:
        y = normalize_loudness(y, sr, p.loudness_lufs)
    y = limit(y, sr, p.limit_ceiling_db, p.limit_lookahead_ms, p.limit_release_ms)
    return y.astype(np.float32, copy=False)


__all__ = ["ClarityParams", "process", "highpass", "compress", "normalize_loudness", "limit", "match_speech_level"]
