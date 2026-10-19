"""Tests for the shared processing package. The libDF/FFmpeg tests need the
native builds: set ACW_LIBDF / ACW_DFN_MODEL / ACW_FFMPEG_DIR, or have them in
~/audio-cleaner/{bin,models} (the server layout); otherwise they are skipped."""
import json
import os
import subprocess
import sys
import threading
import time
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


def probe_streams(path):
    """The streams of a media file as ffprobe reports them (a list of dicts)."""
    ffprobe = str(FF_DIR / ("ffprobe.exe" if (FF_DIR / "ffprobe.exe").exists() else "ffprobe"))
    out = subprocess.run([ffprobe, "-v", "error", "-print_format", "json", "-show_streams", str(path)], capture_output=True,
                         text=True, check=True).stdout
    return json.loads(out)["streams"]


def make_multi_track_video(tmp_path):
    """A 3 s Matroska file: a picture, two audio tracks (440 and 880 Hz) and one subtitle track ('hello').
    Returns (the video, the subtitle file)."""
    srt = tmp_path / "in.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n\n")
    src = tmp_path / "multi.mkv"
    subprocess.run([FIXTURE_FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=5:duration=3",
                    "-f", "lavfi", "-i", "sine=f=440:d=3", "-f", "lavfi", "-i", "sine=f=880:d=3", "-i", str(srt),
                    "-map", "0:v", "-map", "1:a", "-map", "2:a", "-map", "3:s", "-c:v", "mpeg4", "-c:a", "aac", "-c:s", "srt", str(src)],
                   check=True)
    return src, srt


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


def pause_gain_db(noisy, clean, y):
    """How much the noise between the syllables drops, in dB."""
    pauses = np.abs(clean) < 1e-4
    before = np.sqrt(np.mean(noisy[pauses] ** 2))
    after = np.sqrt(np.mean(y[pauses, 0] ** 2))
    return 20 * np.log10(before / max(after, 1e-9))


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
    clean, noisy = voiced_noise(3.0)
    progress = []
    y = pipeline.clean_audio(noisy, LIB, MODEL, pipeline.CleanOptions(leveling=False), threads=1, progress=progress.append)
    assert y.shape == (len(noisy), 1) and np.isfinite(y).all()
    assert progress[-1] == 1.0 and progress == sorted(progress)
    assert pause_gain_db(noisy, clean, y) > 10  # at least 10 dB less noise between syllables


def test_clean_audio_without_denoising_and_without_leveling_returns_the_input(tmp_path):
    _, noisy = voiced_noise(2.0)
    missing = tmp_path / "no-such-library"  # no noise remover is loaded at all
    report: dict = {}
    progress = []
    opts = pipeline.CleanOptions(denoise=False, leveling=False)
    y = pipeline.clean_audio(noisy, missing, missing, opts, threads=1, progress=progress.append, report=report)
    assert y.shape == (len(noisy), 1) and y.dtype == np.float32 and np.array_equal(y[:, 0], noisy)
    assert report["noise"] == "none" and "noise_db" not in report
    assert progress and progress[-1] == 1.0
    out = np.zeros(len(noisy), dtype=np.float32)  # written into the given array
    got = pipeline.clean_audio(noisy, missing, missing, opts, out=out)
    assert np.array_equal(out, noisy) and np.shares_memory(got, out)


def test_clean_audio_without_denoising_only_levels_and_keeps_the_noise(tmp_path):
    clean, noisy = voiced_noise(4.0)
    missing = tmp_path / "no-such-library"
    report: dict = {}
    y = pipeline.clean_audio(noisy, missing, missing, pipeline.CleanOptions(denoise=False, leveling=True, loudness_lufs=-16.0),
                             threads=1, report=report)
    assert y.shape == (len(noisy), 1) and np.isfinite(y).all() and report["noise"] == "none"
    assert not np.allclose(y[:, 0], noisy)  # evened out and set to the loudness
    pauses = np.abs(clean) < 1e-4
    voiced = np.abs(clean) > 0.05
    gap_before = 20 * np.log10(np.sqrt(np.mean(noisy[voiced] ** 2)) / np.sqrt(np.mean(noisy[pauses] ** 2)))
    gap_after = 20 * np.log10(np.sqrt(np.mean(y[voiced, 0] ** 2)) / np.sqrt(np.mean(y[pauses, 0] ** 2)))
    assert gap_after < gap_before + 6  # the noise between syllables is still there (nothing like the 10+ dB denoising takes off)


@needs_dfn
def test_segmented_denoise_is_aligned_reuses_instances_and_matches_fresh_ones(monkeypatch):
    clean, noisy = voiced_noise(seconds=12.0)
    one = dfn.denoise_array(noisy, LIB, MODEL, threads=1)
    made = []
    orig = dfn.DeepFilter.__init__

    def counting(self, *a, **k):
        made.append(1)
        orig(self, *a, **k)

    monkeypatch.setattr(dfn.DeepFilter, "__init__", counting)
    kw = dict(threads=2, segment_seconds=2.0, overlap_seconds=0.5)
    seg = dfn.denoise_array(noisy, LIB, MODEL, **kw)
    assert one.shape == seg.shape == noisy.shape
    assert len(made) <= 2  # the segments share one instance per worker thread
    # The first segment has the same history as a single pass: identical, so no delay offset.
    assert np.abs(one[: 2 * SR] - seg[: 2 * SR]).max() < 1e-6
    # (Later segments start from a fresh model state, so they differ in detail; each one must denoise.)
    pauses = np.abs(clean) < 1e-4
    for s in range(0, 12, 4):
        w = slice(s * SR, (s + 4) * SR)
        before = np.sqrt(np.mean(noisy[w][pauses[w]] ** 2))
        after = np.sqrt(np.mean(seg[w][pauses[w]] ** 2))
        assert 20 * np.log10(before / max(after, 1e-9)) > 10, s
    # a new instance per segment gives the same samples: reuse changes nothing
    monkeypatch.setattr(dfn._Instances, "acquire", lambda self: dfn.DeepFilter(*self._args))
    assert np.array_equal(seg, dfn.denoise_array(noisy, LIB, MODEL, **kw))


@needs_dfn
def test_reset_gives_the_output_of_a_fresh_instance():
    _, a = voiced_noise(seconds=2.0, seed=1)
    _, b = voiced_noise(seconds=2.0, seed=2)
    a, b = a[: len(a) // 480 * 480], b[: len(b) // 480 * 480]
    fresh = dfn.DeepFilter(LIB, MODEL)
    want = fresh.process(b)
    fresh.close()
    used = dfn.DeepFilter(LIB, MODEL)
    used.process(a[: 480 * 37])  # a partial run, as if a cancelled one
    used.reset()
    assert np.array_equal(used.process(b), want)
    used.close()


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
    _, noisy = voiced_noise(seconds=30.0)
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    t0 = time.monotonic()
    with pytest.raises(pipeline.Cancelled):
        pipeline.clean_audio(noisy, LIB, MODEL, threads=2, cancel=cancel)
    assert time.monotonic() - t0 < 5

