"""Spectrum measurements (spectrum.py): where the speech ends (the bandwidth) and the tilt of the top of the speech band, on synthetic
speech-like audio: the cut-off of a phone-quality recording is found, a normal recording is not called band-limited."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.signal as ss

from acw_common import spectrum

SR = 48_000
PHONE_HZ = 6600.0  # reco_model.PHONE_HZ: the speech stops at this frequency or lower


def speech_like(seconds, seed=0, f0=130.0):
    """Voiced bursts (harmonics, -6 dB/oct tilt, vibrato) and fricative noise bursts, up to the Nyquist, 48 kHz."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.1 * np.sin(2 * np.pi * 0.7 * t))) / SR
    voiced = np.sin(phase)
    # sin(k * phase) for the next harmonic from the two before it (sin((k+1)a) = 2 cos(a) sin(ka) - sin((k-1)a)):
    # four times faster than a sin() per harmonic, for the same signal to float32 precision
    below, here, step = np.zeros(n), voiced.copy(), 2 * np.cos(phase)
    for k in range(2, int(22_000 / f0)):
        below, here = here, step * here - below
        voiced += here / k ** 1.3
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





@pytest.fixture(scope="module")
def full():
    return speech_like(8.0)


def test_full_band_is_left_alone(full):
    for x in (full, full * 0.05, np.random.default_rng(3).standard_normal(SR * 8).astype(np.float32) * 0.1):
        assert spectrum.effective_bandwidth(x) > 13_000


@pytest.mark.parametrize("cutoff", [3_400, 6_000, 8_000, 12_000])
def test_detects_the_cutoff(full, cutoff):
    x = lowpass(full, cutoff)
    bw = spectrum.effective_bandwidth(x)
    assert 0.85 * cutoff < bw < 1.12 * cutoff, bw
    # also on a second, different signal and at a lower level
    y = lowpass(speech_like(6.0, seed=5, f0=95.0), cutoff, seed=2) * 0.2
    assert 0.85 * cutoff < spectrum.effective_bandwidth(y) < 1.12 * cutoff


def test_phone_quality_is_below_the_phone_line_and_the_rest_is_not(full):
    assert spectrum.effective_bandwidth(lowpass(full, 3_400)) <= PHONE_HZ
    assert spectrum.effective_bandwidth(lowpass(full, 6_000)) <= PHONE_HZ
    assert spectrum.effective_bandwidth(lowpass(full, 12_000)) > PHONE_HZ


def test_white_noise_lowpassed():
    n = np.random.default_rng(4).standard_normal(SR * 10).astype(np.float32) * 0.1
    assert 6_800 < spectrum.effective_bandwidth(lowpass(n, 8_000)) < 9_000
    assert spectrum.effective_bandwidth(n) > 20_000


def test_silence_and_short_input_are_full_band(full):
    assert spectrum.effective_bandwidth(np.zeros(SR, np.float32)) == SR / 2
    assert spectrum.effective_bandwidth(np.zeros(100, np.float32)) == SR / 2
    stereo = np.stack([lowpass(full, 4_000), full], 1)
    assert isinstance(spectrum.effective_bandwidth(stereo), float)  # channels are averaged; must not raise


def test_the_tilt_is_very_negative_for_a_muffled_voice_and_missing_without_speech(full):
    assert spectrum.tilt_db(full) > spectrum.tilt_db(lowpass(full, 3_400))
    assert spectrum.tilt_db(np.zeros(SR, np.float32)) is None


def test_two_channel_memory_map_is_read_block_by_block(tmp_path, monkeypatch):
    """The bandwidth of a memory-mapped two-channel file equals the in-memory mix's, and no array bigger than the file is made."""
    # 60 s: the sampled frames (a fixed bound, ~4 M samples) must be smaller than the file for the spy to mean something
    mono = ss.sosfilt(ss.butter(8, 3_400, fs=SR, output="sos"), np.tile(speech_like(12), 5)).astype(np.float32)
    st = np.stack([mono, 0.5 * mono[::-1]], axis=1)
    mm = np.memmap(tmp_path / "x.f32", dtype=np.float32, mode="w+", shape=st.shape)
    mm[:] = st
    seen = []
    real = np.asarray

    def spy(a, *args, **kwargs):
        out = real(a, *args, **kwargs)
        seen.append(out.size)
        return out

    monkeypatch.setattr(spectrum.np, "asarray", spy)
    bw = spectrum.effective_bandwidth(mm)
    monkeypatch.undo()
    assert bw == spectrum.effective_bandwidth(st) == spectrum.effective_bandwidth(st.mean(axis=1))
    assert max(seen) <= mm.size  # wrapping the map is fine; no array bigger than the input is ever made
    assert bw <= PHONE_HZ
    assert spectrum.effective_bandwidth(mono[:, None]) == spectrum.effective_bandwidth(mono)  # one channel, 2-D


def test_scipy_signal_is_loaded_only_where_it_is_used():
    """It takes most of a second to import: importing the pipeline must not pay for it."""
    code = "import sys; import acw_common.pipeline; sys.exit('scipy.signal' in sys.modules)"
    r = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1])
    assert r.returncode == 0
