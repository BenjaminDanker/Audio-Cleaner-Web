"""Tests for the shared processing package. The libDF/FFmpeg tests need the
native builds: set ACW_LIBDF / ACW_DFN_MODEL / ACW_FFMPEG_DIR, or have them in
~/audio-cleaner/{bin,models} (the server layout); otherwise they are skipped."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acw_common import clarity, dfn, pipeline  # noqa: E402
from acw_common.ffmpeg import FFmpeg, MediaError  # noqa: E402

HOME = Path.home() / "audio-cleaner"
LIB = Path(os.environ.get("ACW_LIBDF", HOME / "bin" / dfn.LIB_NAME))
MODEL = Path(os.environ.get("ACW_DFN_MODEL", HOME / "models" / dfn.MODEL_FILE))
FF_DIR = Path(os.environ.get("ACW_FFMPEG_DIR", HOME / "bin"))
# Generates test videos (needs lavfi + a video encoder, which the desktop's minimal FFmpeg lacks).
FIXTURE_FFMPEG = os.environ.get("ACW_FIXTURE_FFMPEG", str(FF_DIR / "ffmpeg"))
SR = pipeline.SAMPLE_RATE

needs_dfn = pytest.mark.skipif(not (LIB.is_file() and MODEL.is_file()), reason="libDF build/model not available")
needs_ff = pytest.mark.skipif(not (FF_DIR / "ffmpeg").exists() and not (FF_DIR / "ffmpeg.exe").exists(), reason="ffmpeg not available")


def voiced_noise(seconds=4.0, seed=0):
    """A crude voiced signal (harmonics, syllable envelope) plus white noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f0 = 140 + 20 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    voice = sum(np.sin(k * phase) / k for k in range(1, 12))
    env = np.clip(np.sin(2 * np.pi * 2.0 * t), 0, None) ** 2
    clean = (0.2 * voice * env).astype(np.float32)
    noise = (0.03 * rng.standard_normal(len(t))).astype(np.float32)
    return clean, clean + noise


def test_limiter_respects_ceiling_and_alignment():
    x = np.zeros(SR, dtype=np.float32)
    x[SR // 2] = 2.0
    y = clarity.limit(x, SR, -1.0, 5.0, 60.0)
    assert np.abs(y).max() <= 10 ** (-1 / 20) + 1e-6
    assert len(y) == len(x)


def test_clarity_keeps_shape_and_is_finite():
    _, noisy = voiced_noise(2.0)
    stereo = np.stack([noisy, noisy * 0.5], axis=1)
    y = clarity.process(stereo, stereo, SR)
    assert y.shape == stereo.shape and y.dtype == np.float32 and np.isfinite(y).all()


@needs_dfn
def test_clean_audio_reduces_noise_in_pauses():
    clean, noisy = voiced_noise()
    progress = []
    y = pipeline.clean_audio(noisy, LIB, MODEL, pipeline.CleanOptions(polish=False), threads=1, progress=progress.append)
    assert y.shape == (len(noisy), 1) and np.isfinite(y).all()
    assert progress[-1] == 1.0 and progress == sorted(progress)
    pauses = np.abs(clean) < 1e-4
    before = np.sqrt(np.mean(noisy[pauses] ** 2))
    after = np.sqrt(np.mean(y[pauses, 0] ** 2))
    assert 20 * np.log10(before / max(after, 1e-9)) > 10  # at least 10 dB less noise between syllables


needs_reset = pytest.mark.skipif(
    not (LIB.is_file() and hasattr(dfn._load(str(LIB)), "df_reset")), reason="libDF build without df_reset")


@needs_dfn
@needs_reset
def test_reset_gives_the_output_of_a_fresh_instance():
    _, a = voiced_noise(seconds=3.0, seed=1)
    _, b = voiced_noise(seconds=3.0, seed=2)
    a, b = a[: len(a) // 480 * 480], b[: len(b) // 480 * 480]
    fresh = dfn.DeepFilter(LIB, MODEL)
    want = fresh.process(b)
    fresh.close()
    used = dfn.DeepFilter(LIB, MODEL)
    used.process(a)
    used.reset()
    assert np.array_equal(used.process(b), want)
    # and again, after a partial run that was cancelled half way
    used.process(a[: 480 * 37])
    used.reset()
    assert np.array_equal(used.process(b), want)
    used.close()


@needs_dfn
@needs_reset
def test_segments_reuse_one_instance_per_thread_with_identical_output(monkeypatch):
    _, noisy = voiced_noise(seconds=12.0)
    made = []
    orig = dfn.DeepFilter.__init__

    def counting(self, *a, **k):
        made.append(1)
        orig(self, *a, **k)

    kw = dict(threads=2, segment_seconds=2.0, overlap_seconds=0.5)
    monkeypatch.setattr(dfn.DeepFilter, "__init__", counting)
    reused = dfn.denoise_array(noisy, LIB, MODEL, **kw)
    assert len(made) <= 2  # six segments on two worker threads
    monkeypatch.setattr(dfn.DeepFilter, "can_reset", property(lambda self: False))
    made.clear()
    fresh = dfn.denoise_array(noisy, LIB, MODEL, **kw)
    assert len(made) == 6  # without df_reset: one per segment (as before)
    assert np.array_equal(reused, fresh)


@needs_dfn
def test_segmented_denoise_is_aligned_and_denoises():
    clean, noisy = voiced_noise(seconds=9.0)
    one = dfn.denoise_array(noisy, LIB, MODEL, threads=1)
    seg = dfn.denoise_array(noisy, LIB, MODEL, threads=3, segment_seconds=2.0, overlap_seconds=1.0)
    assert one.shape == seg.shape == noisy.shape
    # The first segment has the same history as a single pass: identical, so no delay offset.
    assert np.abs(one[: 2 * SR] - seg[: 2 * SR]).max() < 1e-6
    # (Later segments start from a fresh model state, so they differ in detail;
    # what matters is that every segment denoises.)
    pauses = np.abs(clean) < 1e-4
    for s in range(0, 9, 2):
        w = slice(s * SR, (s + 2) * SR)
        before = np.sqrt(np.mean(noisy[w][pauses[w]] ** 2))
        after = np.sqrt(np.mean(seg[w][pauses[w]] ** 2))
        assert 20 * np.log10(before / max(after, 1e-9)) > 10, s


@needs_ff
def test_ffmpeg_round_trip_and_rejects_garbage(tmp_path):
    ff = FFmpeg(FF_DIR, threads=1)
    _, noisy = voiced_noise(2.0)
    stereo = np.stack([noisy, -noisy], axis=1)
    out = ff.write_audio(stereo, SR, tmp_path / "x", "flac")
    probe = ff.probe(out)
    assert probe.audio_channels == 2 and abs(probe.duration - 2.0) < 0.05 and not probe.has_video
    back = pipeline.decode_for_clean(ff, out, probe, 60)
    assert back.shape[1] == 2 and abs(len(back) - len(stereo)) < SR // 100
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video" * 100)
    with pytest.raises(MediaError):
        ff.probe(junk)


@needs_ff
def test_mux_keeps_or_strips_metadata(tmp_path):
    import subprocess

    ff = FFmpeg(FF_DIR, threads=1)
    src = tmp_path / "in.mp4"
    subprocess.run(
        [FIXTURE_FFMPEG, "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=2",
         "-f", "lavfi", "-i", "sine=frequency=300:duration=2", "-c:v", "mpeg4", "-c:a", "aac",
         "-metadata", "title=hello", "-shortest", str(src)],
        check=True,
    )
    _, noisy = voiced_noise(2.0)
    audio = ff.write_audio(noisy, SR, tmp_path / "a", "wav")
    for keep in (True, False):
        out = ff.mux_video(src, audio, tmp_path / f"out-{keep}.mp4", keep_metadata=keep)
        tags = subprocess.run([ff.ffprobe, "-v", "error", "-show_entries", "format_tags=title", "-of", "csv=p=0", str(out)],
                              capture_output=True, text=True).stdout.strip()
        assert (tags == "hello") == keep
        assert ff.probe(out).has_video


@needs_ff
def test_asr_chunks_are_cut_exactly_and_in_batches(tmp_path, monkeypatch):
    from acw_common import ffmpeg as ffmod

    monkeypatch.setattr(ffmod, "ASR_CHUNKS_PER_RUN", 4)  # 10 chunks -> 3 runs, the last one holding 2
    ff = FFmpeg(FF_DIR, threads=1)
    sr = 16000
    rng = np.random.default_rng(7)
    x = (rng.standard_normal(sr * 40) * 0.02).astype(np.float32)
    spans = []
    for i in range(10):
        a = sr * 4 * i + int(rng.integers(0, 4000))
        spans.append((a, a + int(sr * rng.uniform(0.5, 3.0))))
        x[spans[-1][0]:spans[-1][1]] += 0.3 * np.sin(2 * np.pi * (300 + 150 * i) * np.arange(spans[-1][1] - spans[-1][0]) / sr).astype(np.float32)
    names = [f"{i:04d}" for i in range(10)]
    files = ff.write_audio_chunks(x, sr, spans, tmp_path, names)
    assert [f.name for f in files] == [f"{n}.mp3" for n in names]
    assert not (tmp_path / ".chunks-graph.txt").exists()
    for (a, b), f in zip(spans, files):
        got = ff.decode(f, ff.probe(f), sr, 1, 10.0)[:, 0]
        assert abs(len(got) - (b - a)) <= 1152  # mp3 frames pad the end
        # it is this chunk's audio, from its first sample: the tone's frequency is the chunk's own
        ref = x[a:b]
        n = min(len(got), len(ref))
        assert np.corrcoef(got[:n], ref[:n])[0, 1] > 0.9
    assert ff.write_audio_chunks(x, sr, [], tmp_path, []) == []
    with pytest.raises(ValueError):
        ff.write_audio_chunks(x, sr, spans, tmp_path, names[:3])


@needs_dfn
def test_cancel_stops_quickly():
    import threading
    import time

    _, noisy = voiced_noise(seconds=120.0)
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    t0 = time.monotonic()
    with pytest.raises(pipeline.Cancelled):
        pipeline.clean_audio(noisy, LIB, MODEL, threads=2, cancel=cancel)
    assert time.monotonic() - t0 < 5


@pytest.mark.skipif(not MODEL.is_file(), reason="model not available")
def test_model_delay_from_config():
    assert dfn.model_delay(MODEL) == dfn.MODEL_DELAY == 1440
    ll = MODEL.with_name("DeepFilterNet3_ll_onnx.tar.gz")
    if ll.is_file():
        assert dfn.model_delay(ll) == 480
