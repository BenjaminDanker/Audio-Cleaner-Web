"""Where the speech in a recording ends (its bandwidth) and how dull it sounds (its tilt): the spectrum measurements that tell a phone-quality
or a muffled recording from a normal one. A recording that sounds like a phone call or a very small file is a reason to rebuild the voice
(reco_model.PHONE_HZ); the website measures the same (web/src/media/analysis.js bandwidthOf) and acw_common/analysis.py follows the browser
exactly, as the trained model needs; these two are the simpler reading used by advice.py and the checks that compare the two.

    bw = effective_bandwidth(x)   # Hz where the speech spectrum really ends (24000 for full band)
    tilt = tilt_db(x)             # dB the top of the speech band sits below its middle (very negative: muffled), None: too little speech
"""
from __future__ import annotations

import numpy as np
import scipy.fft

SAMPLE_RATE = 48_000
EDGE_DB = 20.0  # the edge is where the speech spectrum is this far below its passband level
CLIFF_DB = 18.0  # ... and the level falls by at least this much from just below to just above the edge

_LTAS_CHUNK = 1024  # frames per FFT batch while measuring (about 8 MB of audio at a time)


def _read_frames(x: np.ndarray, first: np.ndarray, n_fft: int) -> np.ndarray:
    """The frames of `x` starting at the sample positions `first`, as float32 (k, n_fft). `x` is (n,) or
    (n, ch) (channels are averaged); only these samples are read, so `x` may be a memory map."""
    frames = np.asarray(x[first[:, None] + np.arange(n_fft)], dtype=np.float32)
    if frames.ndim == 3:
        frames = frames[:, :, 0] if frames.shape[2] == 1 else frames.mean(axis=2)
    return frames


def _ltas_db(x: np.ndarray, sr: int, max_frames: int = 6000):
    """Long-term average power spectrum (dB) over the active frames of `x` (mono, or (n, ch) with the channels
    averaged), bin spacing sr/2048 Hz. Works block by block on the frames it samples, so the whole signal is
    never copied or averaged in memory."""
    n_fft, hop = 2048, 1024
    if len(x) < n_fft * 4:
        return None
    n = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n) if n <= max_frames * 4 else np.linspace(0, n - 1, max_frames * 4).astype(int)
    win = np.hanning(n_fft).astype(np.float32)
    f = np.arange(n_fft // 2 + 1) * sr / n_fft
    speech = (f >= 150) & (f <= 3500)

    def power(frame_ids: np.ndarray) -> np.ndarray:
        return np.abs(scipy.fft.rfft(_read_frames(x, frame_ids * hop, n_fft) * win, axis=1)) ** 2

    e = np.concatenate([power(idx[i:i + _LTAS_CHUNK])[:, speech].sum(axis=1) for i in range(0, len(idx), _LTAS_CHUNK)])
    if e.max() <= 0:
        return None
    # active = clearly above the quiet frames and not far below the loud ones (speech, not the gaps)
    thr = max(np.percentile(e, 60), e.max() * 1e-4)
    act = np.flatnonzero(e >= thr)
    if len(act) < 8:
        return None
    if len(act) > max_frames:
        act = act[np.linspace(0, len(act) - 1, max_frames).astype(int)]
    total = np.zeros(len(f), dtype=np.float64)
    for i in range(0, len(act), _LTAS_CHUNK):
        total += power(idx[act[i:i + _LTAS_CHUNK]]).sum(axis=0, dtype=np.float64)
    ltas = total / len(act)
    return f, 10 * np.log10(ltas + 1e-20)


def effective_bandwidth(x: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Frequency (Hz) at which the spectrum of the speech in `x` ends; `sr/2` when there is no sharp cut-off.

    Works on the long-term average spectrum of the active (speech) frames. The edge is the first frequency
    above 2.4 kHz where that spectrum has fallen 20 dB below the speech passband (0.6-2.4 kHz) and stays
    there, accepted only if it really is a cliff: the level just below the edge is >= CLIFF_DB above the
    level one octave-third higher (natural speech roll-off, noisy or dark recordings fall by < 12 dB there).
    Anything else (no edge, shallow slope, silence, too little speech) gives sr/2: a normal recording is
    never called band-limited. `x` is (n,) or (n, ch), also a memory map of any length: it is read
    block by block, never copied. Channels are averaged, so call it per channel for tracks with different
    content.
    """
    nyq = sr / 2
    r = _smoothed_ltas(x, sr)
    if r is None:
        return nyq
    f, db = r

    def mean(lo, hi):
        return _band_mean(f, db, lo, hi)

    ref = mean(600, 2400)
    top = min(nyq * 0.95, 20_000.0)
    for i in np.where((f >= 2400) & (f < top) & (db < ref - EDGE_DB))[0]:
        fe = f[i]
        if mean(fe * 1.15, nyq) >= ref - EDGE_DB:  # does not stay down: a dip, not an edge
            continue
        if mean(fe * 0.8, fe * 0.9) - mean(fe * 1.15, fe * 1.4) < CLIFF_DB:
            continue  # shallow slope: natural roll-off; keep looking further up
        # a real cliff: its position is where the level has fallen 12 dB below the level just before it
        local = mean(fe * 0.8, fe * 0.9)
        j = np.where((f >= fe * 0.9) & (db < local - 12.0))[0]
        return float(f[j[0]])
    return nyq


def _smoothed_ltas(x: np.ndarray, sr: int):
    """(frequencies, dB) of the long-term spectrum of the speech in `x`, smoothed; None when there is too little."""
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    r = _ltas_db(x, sr)
    if r is None:
        return None
    f, db = r
    # smooth in the dB domain with a ~200 Hz running median (kills single-bin tones and gaps)
    import scipy.signal

    return f, scipy.signal.medfilt(db, max(3, int(round(200 / f[1])) | 1))


def _band_mean(f: np.ndarray, db: np.ndarray, lo: float, hi: float) -> float:
    sel = (f >= lo) & (f < hi)
    return db[sel].mean() if sel.any() else np.nan


def tilt_db(x: np.ndarray, sr: int = SAMPLE_RATE) -> float | None:
    """How far the top of the speech band (4-8 kHz) sits below its middle (0.6-2.4 kHz), dB, on the same smoothed
    spectrum as effective_bandwidth: very negative for a muffled voice. None when there is too little speech. The
    website measures the same (web/src/media/analysis.js bandwidthOf, tiltDb)."""
    r = _smoothed_ltas(x, sr)
    if r is None:
        return None
    f, db = r
    return float(_band_mean(f, db, 4000, 8000) - _band_mean(f, db, 600, 2400))
