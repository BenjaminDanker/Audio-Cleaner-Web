"""Tests for the DPDFNet-8 (48 kHz) file denoiser. Needs the ONNX model
(deploy/fetch-dpdfnet.sh puts it in ~/audio-cleaner/models/dpdfnet/, or set
ACW_DPDFNET_MODEL); otherwise skipped. Run with PYTHONPATH=common."""
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acw_common import dpdfnet  # noqa: E402
from acw_common.dfn import Cancelled  # noqa: E402

pytestmark = pytest.mark.skipif(not dpdfnet.default_model_path().is_file(), reason="DPDFNet ONNX model not available")
SR = dpdfnet.SAMPLE_RATE
THREADS = 2


def syllables(seconds, seed=0):
    """Voiced harmonics gated on/off about twice a second (off = noise-only) plus white noise.
    Returns (clean, noisy, gate)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f0 = 140 + 20 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    voice = sum(np.sin(k * phase) / k for k in range(1, 12))
    gate = (np.sin(2 * np.pi * 1.0 * t) > 0.2).astype(np.float32)
    gate = np.convolve(gate, np.ones(480) / 480, mode="same").astype(np.float32)
    clean = (0.2 * voice * gate).astype(np.float32)
    noise = (0.03 * rng.standard_normal(len(t))).astype(np.float32)
    return clean, clean + noise, gate


def noise_only_energy(y, gate):
    """Mean power of y where the speech gate has been off for a while (skip 100 ms edges)."""
    off = np.convolve((gate < 1e-6).astype(np.float32), np.ones(9600), mode="same") >= 9599
    assert off.sum() > SR // 2
    return float(np.mean(y[off] ** 2))


@pytest.fixture(scope="module")
def sig():
    return syllables(4.0)


@pytest.fixture(scope="module")
def full(sig):
    return dpdfnet.denoise_array(sig[1], atten_lim_db=100, threads=THREADS)


def test_mono_length_finite_and_noise_reduced(sig, full):
    clean, noisy, gate = sig
    assert full.shape == noisy.shape and full.dtype == np.float32
    assert np.isfinite(full).all()
    ratio_db = 10 * np.log10(noise_only_energy(full, gate) / noise_only_energy(noisy, gate))
    assert ratio_db < -15, f"noise-only part only dropped {-ratio_db:.1f} dB"
    # the voiced part must still be there
    on = gate > 0.99
    assert np.mean(full[on] ** 2) > 0.1 * np.mean(clean[on] ** 2)


def test_atten_limit_scales_with_strength(sig, full):
    _, noisy, gate = sig
    weak = dpdfnet.denoise_array(noisy, atten_lim_db=12, threads=THREADS)
    mid = dpdfnet.denoise_array(noisy, atten_lim_db=24, threads=THREADS)
    drop = lambda y: 10 * np.log10(noise_only_energy(noisy, gate) / noise_only_energy(y, gate))  # noqa: E731
    assert 10 < drop(weak) < 14, drop(weak)  # limited to roughly 12 dB
    assert drop(weak) + 3 < drop(mid) < drop(full) + 1
    assert drop(full) > 30


def test_stereo_is_per_channel(sig):
    clean, noisy, gate = sig
    n = len(noisy)
    stereo = np.stack([noisy, np.roll(noisy, 4800)], axis=1).astype(np.float32)
    y = dpdfnet.denoise_array(stereo, threads=THREADS)
    assert y.shape == (n, 2) and np.isfinite(y).all()
    left = dpdfnet.denoise_array(noisy, threads=THREADS)
    assert np.allclose(y[:, 0], left, atol=1e-5)
    assert noise_only_energy(y[:, 1], np.roll(gate, 4800)) < 0.2 * noise_only_energy(stereo[:, 1], np.roll(gate, 4800))


def test_short_and_empty_inputs():
    assert dpdfnet.denoise_array(np.zeros(0, np.float32)).shape == (0,)
    for n in (1, 100, 480, 959, 1500):
        y = dpdfnet.denoise_array(np.random.default_rng(n).standard_normal(n).astype(np.float32) * 0.01, threads=1)
        assert y.shape == (n,) and np.isfinite(y).all()


def test_output_is_time_aligned(sig, full):
    clean, noisy, gate = sig
    # cross-correlation peak of the enhanced signal with the clean one is at lag 0
    seg = slice(SR // 2, 3 * SR)
    lags = range(-60, 61)
    corr = [np.dot(full[seg], np.roll(clean, k)[seg]) for k in lags]
    assert list(lags)[int(np.argmax(corr))] == 0


def test_cancel():
    ev = threading.Event()
    ev.set()
    with pytest.raises(Cancelled):
        dpdfnet.denoise_array(np.zeros(SR * 3, np.float32), cancel=ev, threads=1)
    ev2 = threading.Event()
    seen = []

    def prog(f):
        seen.append(f)
        if f > 0.1:
            ev2.set()

    with pytest.raises(Cancelled):
        dpdfnet.denoise_array(syllables(8.0)[1], progress=prog, cancel=ev2, threads=1)
    assert seen and max(seen) < 0.6  # stopped early


def test_long_input_is_processed_block_wise(monkeypatch, tmp_path):
    clean, noisy, gate = syllables(14.0, seed=3)
    pushed = []
    orig = dpdfnet._Stream.push

    def spy(self, x, cancel=None):
        pushed.append(len(x))
        return orig(self, x, cancel)

    monkeypatch.setattr(dpdfnet._Stream, "push", spy)
    seq = dpdfnet.denoise_array(noisy, threads=1)
    assert max(pushed) <= dpdfnet._BLOCK_FRAMES * dpdfnet.HOP < len(noisy) // 3
    assert len(pushed) >= 7  # many small blocks, not one big call
    pushed.clear()
    # write straight into a memory-mapped file, in parallel segments with warm-up + crossfade
    mm = np.lib.format.open_memmap(tmp_path / "o.npy", mode="w+", dtype=np.float32, shape=(len(noisy), 1))
    fractions = []
    ret = dpdfnet.denoise_array(noisy[:, None], threads=THREADS, out=mm, segment_seconds=4, overlap_seconds=1.5,
                                progress=fractions.append)
    assert ret is mm and len(pushed) > 10
    par = np.asarray(mm[:, 0])
    assert fractions and fractions[-1] == 1.0 and all(a <= b + 1e-9 for a, b in zip(fractions, fractions[1:]))
    # parallel segments agree with the single sequential pass (RNN warm-up converged)
    err = np.sqrt(np.mean((par - seq) ** 2)) / np.sqrt(np.mean(seq ** 2))
    assert err < 0.05, err
    assert noise_only_energy(par, gate) < 0.2 * noise_only_energy(noisy, gate)
