"""Tests for the optional Sidon restore step (acw_common.restore).

They need the Sidon model files and onnxruntime: set ACW_SIDON_MODEL to the folder, or have
it at ~/audio-cleaner/models/sidon (the server layout); otherwise the model tests are skipped. Each test
runs a few seconds of audio on 2 threads (the model is slow: about real time on 4 threads)."""
import os
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acw_common import restore as R  # noqa: E402

SR = R.SAMPLE_RATE
MODEL_DIR = Path(os.environ.get("ACW_SIDON_MODEL", Path.home() / "audio-cleaner" / "models" / "sidon"))
needs_model = pytest.mark.skipif(not R.available(MODEL_DIR), reason="Sidon model files not available")
THREADS = 2
# A real speech clip (48 kHz mono wav) for the tests that need natural speech: the model is unstable on synthetic
# buzz, which makes chunked-vs-whole comparisons meaningless.
SPEECH_WAV = Path(os.environ.get("ACW_SPEECH_WAV", Path.home() / "audio-cleaner" / "dev" / "live_opus.wav"))
needs_speech = pytest.mark.skipif(not SPEECH_WAV.is_file(), reason="no speech clip (set ACW_SPEECH_WAV)")


def speech_like(seconds, seed=0, f0=140.0):
    """Harmonics with vibrato, a syllable envelope and some noise: enough for a shape/continuity test."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.1 * np.sin(2 * np.pi * 0.7 * t))) / SR
    voice = sum(np.sin(k * phase) / k for k in range(1, 20))
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t + 1.0)  # always voiced: every seam is in audible speech
    return (0.2 * voice * env + 0.01 * rng.standard_normal(len(t))).astype(np.float32)


def test_front_end_matches_transformers():
    """The numpy fbank (no transformers dependency) must stay identical to the reference extractor."""
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    x = speech_like(2.0)
    w16 = np.pad(R._to_16k(x)[: 99 * 320], (160, 160))
    fx = transformers.SeamlessM4TFeatureExtractor(feature_size=80, num_mel_bins=80, padding_value=1, stride=2)
    ref = fx(torch.from_numpy(w16), return_tensors="np", sampling_rate=16000)["input_features"][0]
    mine = R._fbank(w16)
    assert mine.shape == ref.shape
    assert np.abs(mine - ref).max() < 1e-4


def test_unavailable_without_files(tmp_path):
    assert not R.available(tmp_path)


@needs_model
@pytest.mark.parametrize("seconds", [0.3, 3.7])
def test_mono_length_preserved_and_finite(seconds):
    x = speech_like(seconds)
    y = R.restore(x, MODEL_DIR, threads=THREADS)
    assert y.shape == x.shape and y.dtype == np.float32
    assert np.isfinite(y).all()
    assert np.abs(y).max() <= 0.99 + 1e-6
    if seconds > 1:
        assert np.sqrt(np.mean(y**2)) > 1e-3  # not silent


@needs_model
def test_stereo_shapes_out_and_mid_path():
    mono = speech_like(2.0)
    dual = np.stack([mono, mono * 0.9], axis=1)  # side signal is tiny: restored once, written to both channels
    out = np.zeros_like(dual)
    y = R.restore(dual, MODEL_DIR, threads=THREADS, out=out)
    assert y is out and y.shape == dual.shape and np.isfinite(y).all()
    assert np.array_equal(y[:, 0], y[:, 1])
    wide = np.stack([speech_like(2.0, seed=1), speech_like(2.0, seed=2, f0=200.0)], axis=1)
    z = R.restore(wide, MODEL_DIR, threads=THREADS)
    assert z.shape == wide.shape and np.isfinite(z).all()
    assert not np.array_equal(z[:, 0], z[:, 1])  # unrelated channels are restored separately


def real_speech(seconds):
    sf = pytest.importorskip("soundfile")
    x, sr = sf.read(SPEECH_WAV, dtype="float32", always_2d=True)
    assert sr == SR
    x = np.tile(x[:, 0], int(np.ceil(seconds * SR / len(x))))[: int(seconds * SR)]
    return np.ascontiguousarray(x)


@needs_model
@needs_speech
def test_chunk_boundaries_do_not_click():
    """Short windows (several overlap cross-fades and decode blocks) against one single window:
    the same spectral envelope, and no discontinuity at the seams."""
    seconds = 14.0
    x = real_speech(seconds)
    one = R.restore(x, MODEL_DIR, threads=THREADS, window_seconds=60.0)
    win, ov = 6.0, 1.5
    chunked = R.restore(x, MODEL_DIR, threads=THREADS, window_seconds=win, overlap_seconds=ov)
    assert chunked.shape == one.shape and np.isfinite(chunked).all()

    # 1) same loudness and spectral envelope (the waveforms themselves differ in phase: generative model)
    ratio = np.sqrt(np.mean(chunked**2) / np.mean(one**2))
    assert 0.8 < ratio < 1.25, ratio

    def logspec(a):
        n = 960
        f = np.abs(np.fft.rfft(a[: len(a) // n * n].reshape(-1, n) * np.hanning(n), axis=1)) ** 2
        return 10 * np.log10(np.stack([b.sum(1) for b in np.array_split(f, 24, axis=1)], 1) + 1e-9)

    active = logspec(one).mean(1) > -60
    assert active.mean() > 0.3
    assert np.abs(logspec(one) - logspec(chunked))[active].mean() < 5.0

    # 2) no gross discontinuity at the seams (a misaligned block or a dropped crossfade gives a jump of about the
    #    signal level): the peakiness of the second difference around each seam (local max over 5 ms against the
    #    rms over the surrounding 100 ms) stays within what the rest of the file shows. Small steps hide in the
    #    speech's own high-frequency content; the exact-decoder test below covers those.
    d2 = np.abs(np.diff(chunked.astype(np.float64), 2))
    cs = np.cumsum(np.r_[0.0, d2**2])

    def peakiness(i):
        lo, hi = max(0, i - 2400), min(len(d2), i + 2400)
        rms = np.sqrt((cs[hi] - cs[lo]) / (hi - lo))
        return d2[i - 120 : i + 120].max() / (rms + 1e-12), rms

    grid = [peakiness(i) for i in range(2400, len(d2) - 2400, 480)]
    normal = np.array([p for p, rms in grid if rms > 1e-3])
    assert len(normal) > 100
    limit = np.percentile(normal, 99)
    hop = win - ov
    window_seams = [k * hop + ov / 2 for k in (1, 2)]  # middle of each cross-fade
    decode_seams = [R._DECODE_SECONDS * k for k in (1, 2) if R._DECODE_SECONDS * k < seconds - 1]
    for s in window_seams + decode_seams:
        i = int(s * SR)
        p, rms = peakiness(i)
        print(f"seam {s:.2f} s: peakiness {p:.1f} (limit {limit:.1f}), max d2 {d2[i - 120 : i + 120].max():.2e}")
        # seams that fall in near-silence (max second difference below -66 dBFS) cannot click audibly
        assert p <= limit * 1.25 or d2[i - 120 : i + 120].max() < 5e-4, (s, p, limit)


@needs_model
def test_decoder_blocks_with_context_equal_one_shot():
    """The decoder is convolutional: decoding in blocks with context frames and cropping is exact."""
    be = R._backend(MODEL_DIR, THREADS)
    w16 = R._to_16k(speech_like(6.0))[: 300 * R.FEATURE_SAMPLES]
    feats = be.features(R._fbank(np.pad(w16, (R._FBANK_PAD, R._FBANK_PAD))))
    whole = be.decode(feats)
    ctx, a, b = R._DECODE_CONTEXT, 100, 220
    block = be.decode(feats[a - ctx : b + ctx])[ctx * R.DECODER_UPSAMPLE : (ctx + b - a) * R.DECODER_UPSAMPLE]
    assert np.abs(block - whole[a * R.DECODER_UPSAMPLE : b * R.DECODER_UPSAMPLE]).max() < 1e-4


def test_import_is_lazy():
    import subprocess

    code = "import sys; sys.path.insert(0, %r); import acw_common.restore; print('torch' in sys.modules, 'onnxruntime' in sys.modules)" % str(
        Path(__file__).resolve().parents[1]
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False"]


@needs_model
def test_progress_and_cancel():
    x = speech_like(14.0)
    seen = []
    R.restore(x[: 4 * SR], MODEL_DIR, threads=THREADS, progress=seen.append)
    assert seen and seen == sorted(seen) and 0 < seen[0] <= 1 and seen[-1] == 1.0

    cancel = threading.Event()
    calls = []

    def progress(f):
        calls.append(f)
        cancel.set()  # cancelled after the first decoded block

    with pytest.raises(R.Cancelled):
        R.restore(x, MODEL_DIR, threads=THREADS, progress=progress, cancel=cancel, window_seconds=6.0, overlap_seconds=1.0)
    assert len(calls) == 1  # stopped right away, not after the whole file

    pre = threading.Event()
    pre.set()
    with pytest.raises(R.Cancelled):
        R.restore(x, MODEL_DIR, threads=THREADS, cancel=pre)


def test_side_db_in_blocks_matches_the_whole_file_formula(monkeypatch):
    rng = np.random.default_rng(3)
    x = np.stack([rng.standard_normal(70_001), 0.3 * rng.standard_normal(70_001)], axis=1).astype(np.float32)
    mid = 0.5 * (x[:, 0].astype(np.float64) + x[:, 1])
    side = 0.5 * (x[:, 0].astype(np.float64) - x[:, 1])
    want = 10 * np.log10((np.mean(side**2) + 1e-20) / (np.mean(mid**2) + 1e-20))
    monkeypatch.setattr(R, "_BLOCK", 4096)  # many blocks, the last one short
    assert abs(R._side_db(x) - want) < 1e-9


def test_mid_view_is_the_float32_mid_signal():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((1000, 2)).astype(np.float32)
    mid = R._Mid(x)
    want = np.ascontiguousarray(0.5 * (x[:, 0] + x[:, 1]))
    assert len(mid) == 1000 and mid[10:700].dtype == np.float32
    assert np.array_equal(mid[10:700], want[10:700]) and np.array_equal(mid[0:1000], want)


@needs_model
def test_worker_pool_gives_the_same_samples_as_one_thread():
    """Windows and decoder blocks are independent: the pool of single-threaded sessions must not change the output."""
    x = np.stack([speech_like(5.0, seed=5), speech_like(5.0, seed=6, f0=190.0)], axis=1)
    one = R.restore(x, MODEL_DIR, threads=1)
    many = R.restore(x, MODEL_DIR, threads=3)
    assert np.array_equal(one, many)
