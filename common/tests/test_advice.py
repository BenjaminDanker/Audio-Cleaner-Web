"""What a recording needs from how it sounds (advice.py): someone speaking, an echoey room, a muffled voice. Generated
audio only: words with pauses (a glottal pulse train and some hiss), put in a room or behind a gentle low-pass."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.signal as ss

from acw_common import advice, bwe, pipeline

SR = pipeline.SAMPLE_RATE


def talk(seconds: float = 12, seed: int = 0) -> np.ndarray:
    """Words of 0.2-0.5 s with 0.3-0.8 s pauses: a pulse train at a gliding ~120 Hz (harmonics to the top, falling above
    900 Hz) and fricative hiss, switched with 10 ms ramps, over a quiet room (-70 dB). Spectrally like close-up speech:
    the top of the speech band about 10 dB below its middle."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    f0 = 120 * (1 + 0.15 * np.sin(2 * np.pi * 0.4 * np.arange(n) / SR + seed))
    pulses = np.zeros(n)
    pulses[np.nonzero(np.diff(np.floor(np.cumsum(f0) / SR)))[0]] = 1.0
    voiced = ss.sosfilt(ss.butter(2, 900, fs=SR, output="sos"), pulses)
    hiss = ss.sosfilt(ss.butter(2, 3000, "high", fs=SR, output="sos"), rng.standard_normal(n))
    v = voiced / np.std(voiced) + 0.6 * hiss
    env = np.zeros(n)
    ramp = int(0.01 * SR)
    pos = int(0.2 * SR)
    while pos + SR // 2 < n:
        w = int(rng.uniform(0.2, 0.5) * SR)
        env[pos:pos + w] = np.minimum(1, np.minimum(np.arange(w), np.arange(w)[::-1]) / ramp)
        pos += w + int(rng.uniform(0.3, 0.8) * SR)
    x = v * env
    return (0.3 * x / np.max(np.abs(x)) + 3e-4 * rng.standard_normal(n)).astype(np.float32)


def room(x: np.ndarray, rt60: float, seed: int = 0) -> np.ndarray:
    """In a room that rings for `rt60` seconds (-60 dB): a decaying noise impulse response."""
    m = int(rt60 * 1.5 * SR)
    h = np.random.default_rng(seed + 50).standard_normal(m) * 10 ** (-3 * np.arange(m) / SR / rt60)
    h[0] = 1.0
    y = ss.fftconvolve(x, h)[:len(x)]
    return (0.3 * y / np.max(np.abs(y)) + 3e-4 * np.random.default_rng(seed).standard_normal(len(y))).astype(np.float32)


def muffle(x: np.ndarray, cutoff: float = 2500) -> np.ndarray:
    """A gentle (second-order) low-pass: the highs fade without a cliff, as through a mask or from a mic behind something."""
    return ss.sosfilt(ss.butter(2, cutoff, fs=SR, output="sos"), x).astype(np.float32)


def test_a_close_voice_needs_nothing():
    x = talk()
    assert advice.has_speech(x)
    assert advice.drop_db(x) > 30  # words stop abruptly
    assert -15 < bwe.tilt_db(x) < -5 and bwe.effective_bandwidth(x) > advice.PHONE_HZ
    assert advice.rebuild_reason(x) is None


@pytest.mark.parametrize("rt60", [0.5, 0.8])
def test_an_echoey_room(rt60):
    x = room(talk(), rt60)
    assert advice.drop_db(x) < advice.ECHO_DROP_DB - 1
    assert advice.rebuild_reason(x) == "echoey"


def test_a_short_ring_is_not_echoey():
    x = room(talk(), 0.2)
    assert advice.drop_db(x) > advice.ECHO_DROP_DB and advice.rebuild_reason(x) is None


def test_a_muffled_voice():
    x = muffle(talk(), 2000)
    assert bwe.effective_bandwidth(x) > advice.PHONE_HZ  # no cliff: not a phone call
    assert bwe.tilt_db(x) <= advice.DULL_TILT_DB - 3
    assert advice.rebuild_reason(x) == "muffled"


def test_echo_is_named_before_muffled():
    assert advice.rebuild_reason(room(muffle(talk()), 0.8)) == "echoey"


def test_a_phone_call_is_left_to_restoring_the_highs():
    phone = ss.sosfilt(ss.butter(10, 3400, fs=SR, output="sos"), muffle(talk())).astype(np.float32)
    assert bwe.effective_bandwidth(phone) <= advice.PHONE_HZ and bwe.tilt_db(phone) < advice.DULL_TILT_DB
    assert advice.rebuild_reason(phone) is None


def test_no_speech_is_never_rebuilt():
    t = np.arange(10 * SR) / SR
    rng = np.random.default_rng(4)
    tone = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)  # dull (all below 4 kHz) but no words
    hum_in_a_room = room(tone, 0.8)
    steady = (0.05 * rng.standard_normal(t.size)).astype(np.float32)
    for x in (tone, hum_in_a_room, steady, muffle(steady), np.zeros(10 * SR, np.float32)):
        assert not advice.has_speech(x)
        assert advice.rebuild_reason(x) is None
    assert bwe.tilt_db(tone) < advice.DULL_TILT_DB  # it would count as muffled if speech were not checked first


def test_too_little_to_judge():
    assert not advice.has_speech(np.zeros(SR // 20, np.float32)) and advice.drop_db(np.zeros(SR // 20, np.float32)) is None
    assert advice.rebuild_reason(np.zeros(0, np.float32)) is None
    assert bwe.tilt_db(np.zeros(SR, np.float32)) is None
    # noise only 20 dB under the voice hides how fast the words stop: no figure rather than a wrong one
    x = talk() + (0.03 * np.random.default_rng(5).standard_normal(12 * SR)).astype(np.float32)
    assert advice.has_speech(x) and advice.drop_db(x) is None


def test_two_channels_are_judged_from_their_mix():
    x = room(talk(), 0.8)
    both = np.stack([x, x], 1)
    assert advice.has_speech(both) and advice.drop_db(both) == pytest.approx(advice.drop_db(x), abs=0.01)
    assert advice.rebuild_reason(both) == "echoey"
    m = muffle(talk())
    assert advice.rebuild_reason(np.stack([m, m], 1)) == "muffled"


def test_a_long_recording_is_judged_from_the_measured_stretches(tmp_path):
    """Only pipeline.measure_windows are read (the same ~60 s as the noise figure), also from a memory map."""
    x = np.concatenate([talk(30, s) for s in range(5)])  # 150 s: six stretches of 10 s
    assert [(b - a) / SR for a, b in pipeline.measure_windows(len(x))] == [10.0] * 6
    mm = np.lib.format.open_memmap(tmp_path / "x.npy", mode="w+", dtype=np.float32, shape=(len(x), 1))
    mm[:, 0] = room(x, 0.8)
    assert advice.has_speech(mm) and advice.rebuild_reason(mm) == "echoey"
    mm[:, 0] = x
    assert advice.rebuild_reason(mm) is None
