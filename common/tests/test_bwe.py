"""Bandwidth extension (AP-BWE): band-edge detection, model choice, extension on synthetic speech-like audio.
The detection tests need no model; the extension tests are skipped when the ONNX files are missing
(ACW_BWE_MODELS or ~/audio-cleaner/models/ap-bwe)."""
import os
import threading
from pathlib import Path

import numpy as np
import pytest
import scipy.signal as ss

from acw_common import bwe

SR = 48_000
MODEL_DIR = Path(os.environ.get("ACW_BWE_MODELS", Path.home() / "audio-cleaner" / "models" / "ap-bwe"))
needs_models = pytest.mark.skipif(not bwe.available(MODEL_DIR), reason="AP-BWE ONNX models not installed")


def speech_like(seconds, seed=0, f0=130.0):
    """Voiced bursts (harmonics, -6 dB/oct tilt, vibrato) and fricative noise bursts, up to the Nyquist, 48 kHz."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.1 * np.sin(2 * np.pi * 0.7 * t))) / SR
    voiced = np.zeros(n)
    for k in range(1, int(22_000 / f0)):
        voiced += np.sin(k * phase) / k ** 1.3
    syll = np.clip(np.sin(2 * np.pi * 3.1 * t + 0.4), 0, None) ** 2
    fric = np.clip(np.sin(2 * np.pi * 1.3 * t + 2.0), 0.6, None) - 0.6
    noise = ss.sosfilt(ss.butter(2, 3_000, "high", fs=SR, output="sos"), rng.standard_normal(n))
    x = 0.2 * voiced * syll + 0.7 * noise * fric + 0.001 * rng.standard_normal(n)
    return x.astype(np.float32)


def lowpass(x, cutoff, seed=1):
    """Sharp (Kaiser FIR) low-pass like a decent resampler/codec, plus a 16-bit dither floor."""
    h = ss.firwin(1201, cutoff, fs=SR, window=("kaiser", 9))
    y = ss.fftconvolve(x, h, mode="same")
    y += 3e-5 * np.random.default_rng(seed).standard_normal(len(y))
    return y.astype(np.float32)


def band_db(x, lo, hi):
    f, p = ss.welch(x, SR, nperseg=4096)
    return 10 * np.log10(p[(f >= lo) & (f < hi)].sum() + 1e-20)


@pytest.fixture(scope="module")
def full():
    return speech_like(12.0)


def test_full_band_is_left_alone(full):
    for x in (full, full * 0.05, np.random.default_rng(3).standard_normal(SR * 8).astype(np.float32) * 0.1):
        bw = bwe.effective_bandwidth(x)
        assert bw > 13_000
        assert bwe.pick_model(bw) is None


@pytest.mark.parametrize("cutoff, model", [(3_400, "8k"), (4_000, "8k"), (6_000, "12k"), (8_000, "16k"), (12_000, "24k")])
def test_detects_cutoff_and_picks_model(full, cutoff, model):
    x = lowpass(full, cutoff)
    bw = bwe.effective_bandwidth(x)
    assert 0.85 * cutoff < bw < 1.12 * cutoff, bw
    assert bwe.pick_model(bw) == model
    # also on a second, different signal and at a lower level
    y = lowpass(speech_like(10.0, seed=5, f0=95.0), cutoff, seed=2) * 0.2
    assert bwe.pick_model(bwe.effective_bandwidth(y)) == model


def test_white_noise_lowpassed():
    n = np.random.default_rng(4).standard_normal(SR * 10).astype(np.float32) * 0.1
    assert bwe.pick_model(bwe.effective_bandwidth(lowpass(n, 8_000))) == "16k"
    assert bwe.effective_bandwidth(n) > 20_000


def test_silence_and_short_input_are_full_band():
    assert bwe.pick_model(bwe.effective_bandwidth(np.zeros(SR, np.float32))) is None
    assert bwe.pick_model(bwe.effective_bandwidth(np.zeros(100, np.float32))) is None
    stereo = np.stack([lowpass(speech_like(6.0), 4_000), speech_like(6.0)], 1)
    assert isinstance(bwe.effective_bandwidth(stereo), float)  # channels are averaged; must not raise


def test_pick_model_table():
    assert [bwe.pick_model(b) for b in (3_400, 4_000, 4_400, 5_000, 6_600, 7_000, 8_800, 10_000, 12_600, 12_700, 20_000, 24_000)] == \
        ["8k", "8k", "8k", "12k", "12k", "16k", "16k", "24k", "24k", None, None, None]


def test_stft_matches_torch_definition():
    # reflect-padded centred STFT with the 320-sample Hann inside a 1024 FFT: perfect reconstruction inside
    x = np.random.default_rng(0).standard_normal(8000).astype(np.float32)
    s = bwe._stft(x)
    assert s.shape == (513, 101)
    y = bwe._istft(s, len(x))
    assert np.abs(y[200:-200] - x[200:-200]).max() < 1e-5


@needs_models
@pytest.mark.parametrize("cutoff, model", [(4_000, "8k"), (8_000, "16k")])
def test_extend_adds_highs_keeps_lows(full, cutoff, model):
    x = lowpass(full, cutoff)
    y = bwe.extend(x, MODEL_DIR, threads=2)
    assert y.shape == x.shape and y.dtype == np.float32 and np.isfinite(y).all()
    assert np.abs(y).max() < 2.0
    # energy appears above the cut-off (the input has -90 dBFS dither only), by tens of dB
    assert band_db(y, cutoff * 1.3, 20_000) > band_db(x, cutoff * 1.3, 20_000) + 25
    # ... but the generated highs stay below the passband level (no blow-up)
    assert band_db(y, cutoff * 1.3, 20_000) < band_db(y, 300, cutoff * 0.8)
    # the recorded band is kept
    lows = ss.sosfiltfilt(ss.butter(8, cutoff * 0.75, fs=SR, output="sos"), (y - x).astype(np.float64))
    assert np.sqrt(np.mean(lows ** 2)) < 0.02 * np.sqrt(np.mean(x ** 2))


@needs_models
def test_full_band_input_is_untouched(full):
    y = bwe.extend(full, MODEL_DIR, threads=2)
    assert np.array_equal(y, full)
    assert y is not full


@needs_models
def test_blocks_match_single_shot_and_length_edge_cases(full):
    x = lowpass(full[: SR * 7 + 37], 4_000)
    a = bwe.extend(x, MODEL_DIR, threads=2, block_seconds=1000)
    b = bwe.extend(x, MODEL_DIR, threads=2, block_seconds=1.5)
    assert a.shape == b.shape == x.shape
    assert np.abs(a - b).max() < 1e-3 * np.abs(a).max()
    for n in (1, 7, 100, 4000, 4001, 9000):  # tiny inputs must not crash and keep their length
        t = bwe.extend(x[:n], MODEL_DIR, threads=1, model="8k")
        assert t.shape == (n,)


@needs_models
def test_stereo_progress_cancel_inplace(full):
    left = lowpass(full[: SR * 5], 4_000)
    right = left * 0.5
    st = np.stack([left, right], 1)
    seen = []
    y = bwe.extend(st, MODEL_DIR, threads=2, progress=seen.append, block_seconds=2)
    assert y.shape == st.shape
    assert seen == sorted(seen) and abs(seen[-1] - 1.0) < 1e-9
    solo = bwe.extend(left, MODEL_DIR, threads=2, block_seconds=2)
    assert np.abs(y[:, 0] - solo).max() < 1e-6
    # the other channel is a quieter copy: it is extended as well, and its level is kept
    assert band_db(y[:, 1], 5_000, 20_000) > band_db(right, 5_000, 20_000) + 25
    assert abs(band_db(y[:, 1], 300, 3_000) - band_db(right, 300, 3_000)) < 0.1
    # in place
    z = st.copy()
    r = bwe.extend(z, MODEL_DIR, threads=2, out=z, block_seconds=2)
    assert r is z and np.abs(z - y).max() < 1e-5
    # cancel
    ev = threading.Event()
    ev.set()
    with pytest.raises(bwe.Cancelled):
        bwe.extend(left, MODEL_DIR, threads=1, cancel=ev)


def test_bandwidth_of_a_two_channel_memory_map_matches_the_mono_mix(tmp_path):
    """Read block by block from the file: no mono copy of the whole recording is made."""
    mono = ss.sosfilt(ss.butter(8, 3_400, fs=SR, output="sos"), speech_like(40)).astype(np.float32)
    st = np.stack([mono, 0.5 * mono[::-1]], axis=1)
    mm = np.memmap(tmp_path / "x.f32", dtype=np.float32, mode="w+", shape=st.shape)
    mm[:] = st
    assert bwe.effective_bandwidth(mm) == bwe.effective_bandwidth(st) == bwe.effective_bandwidth(st.mean(axis=1))
    assert bwe.pick_model(bwe.effective_bandwidth(mm)) is not None
    assert bwe.effective_bandwidth(mono[:, None]) == bwe.effective_bandwidth(mono)  # one channel, 2-D


def test_bandwidth_does_not_copy_the_signal(tmp_path, monkeypatch):
    seen = []
    real = np.asarray

    def spy(a, *args, **kwargs):
        out = real(a, *args, **kwargs)
        seen.append(out.size)
        return out

    mm = np.memmap(tmp_path / "y.f32", dtype=np.float32, mode="w+", shape=(SR * 60, 2))
    mm[:] = speech_like(60)[:, None]
    monkeypatch.setattr(bwe.np, "asarray", spy)
    bwe.effective_bandwidth(mm)
    assert max(seen) <= mm.size  # wrapping the map is fine; no array bigger than the input is ever made


@needs_models
def test_extending_in_place_matches_extending_a_copy():
    st = np.stack([lowpass(speech_like(14), 3_400), lowpass(speech_like(14, seed=3), 3_400)], axis=1).astype(np.float32)
    expected = bwe.extend(st, MODEL_DIR, threads=2, block_seconds=2)
    inplace = st.copy()
    result = bwe.extend(inplace, MODEL_DIR, threads=2, out=inplace, block_seconds=2)
    assert result is inplace and np.array_equal(result, expected)
