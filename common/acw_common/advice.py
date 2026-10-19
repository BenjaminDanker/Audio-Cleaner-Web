"""What a recording needs, from how it sounds: the desktop app's "Automatic best settings" (the website decides the same in
web/src/recommend.js, with the same numbers; docs/research/recommend-2026-10.md has the calibration). The noise strength
is pipeline.strength_for; this is the rest: whether the voice is worth rebuilding (an echoey room, a muffled voice).
Measured on the same stretches as the noise figure (pipeline.measure_windows)."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from . import spectrum
from .pipeline import SAMPLE_RATE, SILENCE_DB, measure_windows

# The level can't fall faster than this within 60 ms (dB): the voice rings on, as in a large or bare room.
ECHO_DROP_DB = 10.5
# The top of the speech band sits this far below its middle (dB) or more: a muffled voice without a sharp cut-off.
DULL_TILT_DB = -19.0
# The speech stops at this frequency or lower: a phone call or a very small file (a reason of its own to rebuild the voice: reco_model.py).
PHONE_HZ = 6600.0

# The speech band's level moves at least this much (90th minus 10th percentile, dB) when someone speaks; steady noise,
# hum and tones stay below.
SPEECH_BAND_DYN_DB = 6.8

_FRAME = SAMPLE_RATE // 100  # 10 ms
# The speech band, 150 Hz to 3.5 kHz: what moves with words and pauses (second-order Butterworth each, as the website's).
_BAND = np.concatenate([butter(2, 150, "highpass", fs=SAMPLE_RATE, output="sos"),
                        butter(2, 3500, "lowpass", fs=SAMPLE_RATE, output="sos")])
_DROP_FRAMES = 6  # 60 ms


def _levels(x: np.ndarray) -> np.ndarray:
    """dB of 10 ms frames of the mono mix."""
    m = np.asarray(x, dtype=np.float32)
    m = m.mean(axis=1, dtype=np.float32) if m.ndim == 2 else m
    n = len(m) // _FRAME
    if not n:
        return np.zeros(0)
    return 10 * np.log10(np.mean(np.square(m[:n * _FRAME].reshape(n, _FRAME), dtype=np.float64), axis=1) + 1e-12)


def has_speech(x: np.ndarray) -> bool:
    """Someone speaks: the level of the speech band moves up and down with words and pauses."""
    band = []
    for a, b in measure_windows(len(x)):
        m = np.asarray(x[a:b], dtype=np.float32)
        m = m.mean(axis=1, dtype=np.float32) if m.ndim == 2 else m
        lv, bl = _levels(m), _levels(sosfilt(_BAND, m))
        band.append(bl[lv > SILENCE_DB])
    band = np.concatenate(band) if band else np.zeros(0)
    return len(band) >= 10 and float(np.percentile(band, 90) - np.percentile(band, 10)) >= SPEECH_BAND_DYN_DB


def drop_db(x: np.ndarray) -> float | None:
    """How fast the sound can stop: the 95th percentile of how far the level falls within 60 ms, from the loud parts.
    Close-up speech stops abruptly (20 dB and more); in an echoey room each word's tail fills the next moments (7 dB and
    less). None when the background is too close to the voice to see it (under 30 dB between them) or there is too
    little loud sound. The website's dropOf (web/src/media/analysis.js)."""
    parts = [_levels(x[a:b]) for a, b in measure_windows(len(x))]
    levels = np.concatenate(parts) if parts else np.zeros(0)
    loud = levels[levels > SILENCE_DB]
    if len(loud) < 10:
        return None
    floor, peak = np.percentile(loud, 10), np.percentile(loud, 95)
    if peak - floor < 30:
        return None
    drops = [lv[i] - lv[i + _DROP_FRAMES] for lv in parts for i in range(len(lv) - _DROP_FRAMES) if lv[i] >= peak - 10]
    if len(drops) < 30:
        return None
    return float(np.percentile(drops, 95))


def rebuild_reason(x: np.ndarray) -> str | None:
    """"echoey" or "muffled" when rebuilding the voice is the fix, else None (also when nobody speaks: music is never
    rebuilt). A phone-quality recording is not muffled in this sense (reco_model.py calls it phone)."""
    if not has_speech(x):
        return None
    drop = drop_db(x)
    if drop is not None and drop < ECHO_DROP_DB:
        return "echoey"
    sample = np.concatenate([np.asarray(x[a:b]) for a, b in measure_windows(len(x))]) if len(x) else np.asarray(x)
    if spectrum.effective_bandwidth(sample) <= PHONE_HZ:
        return None
    tilt = spectrum.tilt_db(sample)
    return "muffled" if tilt is not None and tilt <= DULL_TILT_DB else None
