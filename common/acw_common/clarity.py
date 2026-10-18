"""Post-denoise "clarity" chain for file jobs.

    denoised -> level match -> high-pass -> leveling compressor
             -> loudness normalize (EBU R128 / BS.1770) -> look-ahead peak limiter

Each stage is conservative by default: the goal is clean, consistent speech,
not a "processed" sound. Arrays are float32 shaped (samples,) or
(samples, channels).

Everything runs in blocks of BLOCK_SECONDS with the filter state carried
across blocks (and enough neighbouring samples for the centred filters), so
memory stays flat however long the file is; the result is the same as
processing the whole signal at once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
# scipy is imported inside the functions that use it: it takes most of a second to load, which a job that never levels
# the sound need not pay.

BLOCK_SECONDS = 10.0


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


def _blocks(n: int, sr: int):
    step = max(1, int(BLOCK_SECONDS * sr))
    for s in range(0, n, step):
        yield s, min(n, s + step)


def _mono_block(x2: np.ndarray, s: int, e: int) -> np.ndarray:
    return x2[s:e].mean(axis=1, dtype=np.float64) if x2.shape[1] > 1 else x2[s:e, 0].astype(np.float64)


def _frame_rms(x2: np.ndarray, sr: int, sos: np.ndarray, frame: int) -> np.ndarray:
    """RMS of the high-passed mono mix in consecutive frames of `frame` samples."""
    from scipy.signal import sosfilt_zi, sosfilt
    zi = sosfilt_zi(sos) * 0.0
    out, carry = [], np.zeros(0)
    for s, e in _blocks(len(x2), sr):
        y, zi = sosfilt(sos, _mono_block(x2, s, e), zi=zi)
        y = np.concatenate([carry, y])
        k = len(y) // frame
        out.append(np.sqrt(np.mean(y[: k * frame].reshape(k, frame) ** 2, axis=1) + 1e-12))
        carry = y[k * frame:]
    return np.concatenate(out) if out else np.zeros(0)


def speech_level_gain_db(original: np.ndarray, denoised: np.ndarray, sr: int, max_boost_db: float) -> float:
    """DFN tends to lower speech level slightly. Gain (never negative) that makes
    the denoised signal's loud (speech) frames match the original's."""
    from scipy.signal import butter
    o, d = _as2d(original), _as2d(denoised)
    if len(o) < sr:
        return 0.0
    sos = butter(2, 150, btype="highpass", fs=sr, output="sos")
    frame = int(0.05 * sr)
    env_o, env_d = _frame_rms(o, sr, sos, frame), _frame_rms(d, sr, sos, frame)
    # Frames where the cleaned signal is loud are speech (noise is gone there).
    thresh = np.percentile(env_d, 80)
    mask = env_d >= max(thresh, _lin(-50))
    if mask.mean() < 0.02:
        return 0.0
    return float(np.clip(_db(float(np.median(env_o[mask] / env_d[mask]))), 0.0, max_boost_db))


def match_speech_level(original: np.ndarray, denoised: np.ndarray, sr: int, max_boost_db: float) -> np.ndarray:
    gain_db = speech_level_gain_db(original, denoised, sr, max_boost_db)
    return (denoised * _lin(gain_db)).astype(np.float32) if gain_db else denoised


def _highpass_inplace(x2: np.ndarray, sr: int, hz: float) -> None:
    from scipy.signal import butter, sosfilt
    sos = butter(2, hz, btype="highpass", fs=sr, output="sos")
    zi = np.zeros((sos.shape[0], 2, x2.shape[1]))
    for s, e in _blocks(len(x2), sr):
        y, zi = sosfilt(sos, x2[s:e], axis=0, zi=zi)
        x2[s:e] = y


def highpass(x: np.ndarray, sr: int, hz: float) -> np.ndarray:
    if hz <= 0:
        return x
    y = _as2d(np.array(x, dtype=np.float32))
    _highpass_inplace(y, sr, hz)
    return y[:, 0] if x.ndim == 1 else y


def _compress_inplace(x2: np.ndarray, sr: int, threshold_db: float, ratio: float, attack_ms: float, release_ms: float) -> None:
    """Feed-forward RMS compressor, linked across channels.

    Gain reduction comes from a 10 ms RMS detector; a running minimum over the
    attack window gives a fast attack and a one-pole filter the release."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    from scipy.signal import lfilter
    n = len(x2)
    win = max(1, int(0.01 * sr))
    attack = max(1, int(attack_ms * 1e-3 * sr))
    a = float(np.exp(-1.0 / (release_ms * 1e-3 * sr)))
    margin = win + attack
    zi = np.zeros(1)
    for s, e in _blocks(n, sr):
        lo, hi = max(0, s - margin), min(n, e + margin)
        power = (x2[lo:hi].astype(np.float64) ** 2).max(axis=1)
        det = np.sqrt(uniform_filter1d(power, win) + 1e-12)
        over = np.maximum(20.0 * np.log10(det) - threshold_db, 0.0)
        gr_db = minimum_filter1d(-over * (1.0 - 1.0 / max(ratio, 1.0)), attack, origin=-(attack // 2))
        gr_db, zi = lfilter([1 - a], [1, -a], gr_db[s - lo:e - lo], zi=zi)
        x2[s:e] *= (10.0 ** (gr_db / 20.0)).astype(np.float32)[:, None]


def compress(x: np.ndarray, sr: int, threshold_db: float, ratio: float, attack_ms: float, release_ms: float) -> np.ndarray:
    y = _as2d(np.array(x, dtype=np.float32))
    _compress_inplace(y, sr, threshold_db, ratio, attack_ms, release_ms)
    return y[:, 0] if x.ndim == 1 else y


def integrated_loudness(x: np.ndarray, sr: int) -> float:
    """BS.1770-4 gated loudness (LUFS), equal to pyloudnorm's Meter, computed
    from 100 ms sub-block energies so the signal is never copied whole. Returns
    -inf for silence; raises ValueError for input shorter than one 400 ms block."""
    from scipy.signal import lfilter
    import pyloudnorm as pyln

    x2 = _as2d(x)
    n, channels = x2.shape
    block, step = int(0.4 * sr), int(0.1 * sr)
    if n < block:
        raise ValueError("audio shorter than one gating block")
    filters = list(pyln.Meter(sr)._filters.values())  # K-weighting: high shelf, then high pass
    zis = [[np.zeros(max(len(f.a), len(f.b)) - 1) for _ in range(channels)] for f in filters]
    sub = np.zeros((channels, -(-n // step)))
    for s, e in _blocks(n, sr):
        for ch in range(channels):
            y = x2[s:e, ch].astype(np.float64)
            for k, f in enumerate(filters):
                y, zis[k][ch] = lfilter(f.b, f.a, y, zi=zis[k][ch])
                y *= f.passband_gain
            first = s // step  # blocks start on a multiple of `step`
            sq = y * y
            m = -(-len(sq) // step)
            padded = np.zeros(m * step)
            padded[: len(sq)] = sq
            sub[ch, first:first + m] += padded.reshape(m, step).sum(axis=1)
    count = int(np.round((n / sr - 0.4) / 0.1)) + 1
    per = block // step
    padded = np.concatenate([sub, np.zeros((channels, per))], axis=1)
    z = sum(padded[:, j:j + count] for j in range(per)) / block  # (channels, blocks) mean squares
    gains = np.array([1.0, 1.0, 1.0, 1.41, 1.41][:channels])[:, None]
    with np.errstate(divide="ignore"):
        loud = -0.691 + 10.0 * np.log10((gains * z).sum(axis=0))
    gated = loud >= -70.0
    if not gated.any():
        return float("-inf")
    rel = -0.691 + 10.0 * np.log10((gains[:, 0] * z[:, gated].mean(axis=1)).sum()) - 10.0
    gated = (loud > rel) & (loud > -70.0)
    if not gated.any():
        return float("-inf")
    with np.errstate(divide="ignore"):
        return float(-0.691 + 10.0 * np.log10((gains[:, 0] * z[:, gated].mean(axis=1)).sum()))


def loudness_gain_db(x: np.ndarray, sr: int, target_lufs: float) -> float:
    try:
        loud = integrated_loudness(x, sr)
    except ValueError:
        return 0.0
    if not np.isfinite(loud) or loud < -70:
        return 0.0
    return float(np.clip(target_lufs - loud, -20.0, 30.0))


def normalize_loudness(x: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    gain_db = loudness_gain_db(x, sr, target_lufs)
    return (x * _lin(gain_db)).astype(np.float32) if gain_db else x


def _limit_inplace(x2: np.ndarray, sr: int, ceiling_db: float, lookahead_ms: float, release_ms: float) -> None:
    """Look-ahead brickwall-ish peak limiter (sample peak, linked channels);
    output stays time-aligned with the input.

    Per sample: required gain -> min over the look-ahead window (gain is down
    before the peak arrives) -> release smoothing that never rises above the
    requirement -> short min filter against overshoot. In blocks, the release
    filter state is carried and the min filters see `margin` real neighbours."""
    from scipy.ndimage import minimum_filter1d
    from scipy.signal import lfilter
    n = len(x2)
    ceiling = _lin(ceiling_db)
    la = max(1, int(lookahead_ms * 1e-3 * sr))
    short = max(1, la // 2)
    a = float(np.exp(-1.0 / (release_ms * 1e-3 * sr)))
    margin = 2 * la + short
    zi = np.zeros(1)

    def required(lo: int, hi: int) -> np.ndarray:
        peak = np.abs(x2[lo:hi]).max(axis=1).astype(np.float64)
        return np.minimum(1.0, ceiling / np.maximum(peak, 1e-12))

    g2_tail = np.zeros(0)  # final pre-short-filter gain of the samples just before this block
    for s, e in _blocks(n, sr):
        lo, hi = max(0, s - margin), min(n, e + margin)
        g1 = minimum_filter1d(required(lo, hi), la, origin=-(la // 2))[s - lo:]  # [s, hi)
        smoothed, zi_next = lfilter([1 - a], [1, -a], g1[: e - s] - 1.0, zi=zi)
        # The samples after the block continue from the same state; the next
        # block recomputes them identically, so this look-ahead is exact.
        ahead, _ = lfilter([1 - a], [1, -a], g1[e - s:] - 1.0, zi=zi_next)
        g2 = np.minimum(g1, np.concatenate([smoothed, ahead]) + 1.0)  # [s, hi)
        left = g2_tail[-(s - lo):] if s > lo else np.zeros(0)
        g3 = minimum_filter1d(np.concatenate([left, g2]), short)[len(left):len(left) + e - s]
        x2[s:e] = np.clip(x2[s:e] * g3.astype(np.float32)[:, None], -ceiling, ceiling)
        g2_tail = np.concatenate([g2_tail, g2[: e - s]])[-margin:]
        zi = zi_next


def limit(x: np.ndarray, sr: int, ceiling_db: float, lookahead_ms: float, release_ms: float) -> np.ndarray:
    y = _as2d(np.array(x, dtype=np.float32))
    _limit_inplace(y, sr, ceiling_db, lookahead_ms, release_ms)
    return y[:, 0] if x.ndim == 1 else y


def process(original: np.ndarray, denoised: np.ndarray, sr: int, p: ClarityParams | None = None) -> np.ndarray:
    """Run the chain. Works in place on `denoised` when it is a float32 array
    (the caller's buffer is reused to keep memory flat) and returns it."""
    p = p or ClarityParams()
    y = denoised if denoised.dtype == np.float32 else denoised.astype(np.float32)
    y2 = _as2d(y)
    if p.match_level:
        gain_db = speech_level_gain_db(original, y2, sr, p.match_max_boost_db)
        if gain_db:
            y2 *= np.float32(_lin(gain_db))
    if p.highpass_hz > 0:
        _highpass_inplace(y2, sr, p.highpass_hz)
    if p.compress:
        _compress_inplace(y2, sr, p.comp_threshold_db, p.comp_ratio, p.comp_attack_ms, p.comp_release_ms)
    if p.loudness_lufs is not None:
        gain_db = loudness_gain_db(y2, sr, p.loudness_lufs)
        if gain_db:
            y2 *= np.float32(_lin(gain_db))
    _limit_inplace(y2, sr, p.limit_ceiling_db, p.limit_lookahead_ms, p.limit_release_ms)
    return y


__all__ = [
    "ClarityParams", "process", "highpass", "compress", "integrated_loudness", "normalize_loudness", "limit",
    "match_speech_level", "speech_level_gain_db", "loudness_gain_db",
]
