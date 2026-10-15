"""Echo removal with a second audio track (OBS and others record mic on track 1, desktop sound on track 2).
Synthetic recordings: a "speaker" signal, and a mic that hears it back through a small room."""
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("livekit")

from acw_common import pipeline  # noqa: E402
from acw_common.ffmpeg import FFmpeg  # noqa: E402

from test_common import FF_DIR, FIXTURE_FFMPEG, SR, needs_ff  # noqa: E402

SECONDS = 14.0
FAR_ONLY = slice(int(6 * SR), int(8 * SR))  # speaker talking, nobody in the room, after the canceller adapted
NEAR_ONLY = slice(int(10 * SR), int(13 * SR))  # the person talking, speakers quiet


def speech_like(seconds, f0, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.15 * np.sin(2 * np.pi * 0.9 * t))) / SR
    voice = sum(np.sin(k * phase) / k for k in range(1, 15))
    env = np.clip(np.sin(2 * np.pi * 2.3 * t + seed), 0, None) ** 2
    return (0.25 * voice * env + 0.02 * rng.standard_normal(len(t))).astype(np.float32)


def room_recording(seconds=SECONDS, extra=0):
    """(mic, speaker, near): the speaker plays for the first 9 s, the person talks from 9 s on; the mic hears
    the speaker 40 ms late, quieter, with a short reverb tail."""
    n = int(seconds * SR) + extra
    far = speech_like(seconds, 120, 1)
    far = np.pad(far, (0, n - len(far)))
    far[int(9 * SR):] = 0
    near = np.pad(speech_like(seconds, 210, 2), (0, n - int(seconds * SR)))
    near[: int(9 * SR)] = 0
    ir = np.zeros(int(0.12 * SR), np.float32)
    ir[int(0.04 * SR)] = 0.5
    ir[int(0.047 * SR)] = 0.25
    ir[int(0.06 * SR)] = -0.12
    ir[int(0.1 * SR)] = 0.05
    echo = np.convolve(far, ir)[:n].astype(np.float32)
    return near + echo, far, near


def rms_db(x):
    return 20 * np.log10(max(float(np.sqrt(np.mean(np.square(x)))), 1e-9))


def make_file(ff, path, tracks):
    """A Matroska file with one lossless audio track per array (and 24-bit WAV intermediates)."""
    inputs, maps = [], []
    for i, x in enumerate(tracks):
        wav = ff.write_audio(x, SR, path.with_name(f"{path.stem}-t{i}"), "wav")
        inputs += ["-i", str(wav)]
        maps += ["-map", f"{i}:a"]
    subprocess.run([FIXTURE_FFMPEG, "-v", "error", *inputs, *maps, "-c:a", "flac", "-y", str(path)], check=True)
    return path


@pytest.fixture(scope="module")
def ff():
    return FFmpeg(FF_DIR, threads=1)


@needs_ff
def test_second_track_cancels_the_echo_and_keeps_the_voice(ff, tmp_path):
    mic, far, near = room_recording(extra=123)  # not a whole number of 10 ms frames or blocks
    src = make_file(ff, tmp_path / "rec.mkv", [mic, far])
    probe = ff.probe(src)
    assert probe.audio_tracks == 2
    x = pipeline.decode_for_clean(ff, src, probe, 60, work_dir=tmp_path)
    assert x.shape[1] == 1
    y = pipeline.remove_echo(ff, src, probe, x, 60, work_dir=tmp_path)
    assert y.shape == x.shape and y.dtype == np.float32 and np.isfinite(y).all()  # same length
    assert rms_db(x[FAR_ONLY]) - rms_db(y[FAR_ONLY]) > 20  # echo mostly gone
    kept = y[NEAR_ONLY, 0]  # the person's own voice stays: same level, same spectrum (the canceller shifts phase a little)
    voice = near[NEAR_ONLY]
    assert abs(rms_db(kept) - rms_db(voice)) < 1.5
    assert np.corrcoef(np.abs(np.fft.rfft(kept)), np.abs(np.fft.rfft(voice)))[0, 1] > 0.95


@needs_ff
def test_the_result_lines_up_in_time_with_the_recording(ff, tmp_path):
    """The canceller delays its output by 9 ms; the result is shifted back so subtitles and video stay in step."""
    n = int(8 * SR)
    mic = np.zeros(n, np.float32)
    at = int(4 * SR) + 123
    burst = (0.4 * np.hanning(960) * np.sin(2 * np.pi * 2000 * np.arange(960) / SR)).astype(np.float32)
    mic[at:at + 960] = burst
    far = np.zeros(n, np.float32)  # nothing playing: the burst is the only thing to find
    src = make_file(ff, tmp_path / "click.mkv", [mic, far])
    probe = ff.probe(src)
    x = pipeline.decode_for_clean(ff, src, probe, 60, work_dir=tmp_path)
    y = pipeline.remove_echo(ff, src, probe, x, 60, work_dir=tmp_path)
    assert y.shape == x.shape
    peak = int(np.argmax(np.abs(y[:, 0])))
    assert abs(peak - (at + 480)) < 100  # the burst's centre is where it was (it came out 432 samples late before)
    assert np.abs(y[-300:, 0]).max() < 1e-3  # the end is padded with silence, the length is unchanged


@needs_ff
def test_block_boundaries_do_not_change_the_result(ff, tmp_path, monkeypatch):
    mic, far, _ = room_recording()
    src = make_file(ff, tmp_path / "rec.mkv", [mic, far])
    probe = ff.probe(src)
    x = pipeline.decode_for_clean(ff, src, probe, 60, work_dir=tmp_path)
    whole = pipeline.remove_echo(ff, src, probe, x, 60)
    monkeypatch.setattr(pipeline, "ECHO_BLOCK", 3 * SR + 17)
    blocks = pipeline.remove_echo(ff, src, probe, x, 60, work_dir=tmp_path)
    assert blocks.shape == whole.shape
    assert np.abs(blocks - whole).max() < 1e-6


@needs_ff
def test_one_track_files_are_left_alone(ff, tmp_path):
    mic, _, _ = room_recording()
    src = make_file(ff, tmp_path / "solo.mkv", [mic])
    probe = ff.probe(src)
    assert probe.audio_tracks == 1
    x = pipeline.decode_for_clean(ff, src, probe, 60, work_dir=tmp_path)
    assert pipeline.remove_echo(ff, src, probe, x, 60, work_dir=tmp_path) is x
    assert not (tmp_path / "reference.f32").exists()


@needs_ff
def test_decode_picks_the_requested_track(ff, tmp_path):
    mic, far, _ = room_recording()
    src = make_file(ff, tmp_path / "rec.mkv", [mic, far])
    probe = ff.probe(src)
    ref = ff.decode_to_file(src, probe, SR, 1, 60, tmp_path / "ref.f32", track=1)
    assert isinstance(ref, np.memmap) and len(ref) == len(far)
    assert np.abs(ref[:, 0] - far).max() < 1e-4
    assert np.abs(ff.decode(src, probe, SR, 1, 60, track=1)[:, 0] - far).max() < 1e-4
    assert np.abs(ff.decode(src, probe, SR, 1, 60)[:, 0] - mic).max() < 1e-4


@needs_ff
def test_cancel_stops_echo_removal(ff, tmp_path):
    mic, far, _ = room_recording()
    src = make_file(ff, tmp_path / "rec.mkv", [mic, far])
    probe = ff.probe(src)
    x = pipeline.decode_for_clean(ff, src, probe, 60, work_dir=tmp_path)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(pipeline.Cancelled):
        pipeline.remove_echo(ff, src, probe, x, 60, cancel=cancel)


def test_speech_16k_is_mono_third_length_and_seamless():
    n = SR * 75 + 5  # crosses a block boundary, not a multiple of 3
    t = np.arange(n) / SR
    tone = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    y = pipeline.to_speech_16k(np.stack([tone, tone], axis=1))
    assert y.shape == (-(-n // 3),) and y.dtype == np.float32
    cut = (pipeline.SAMPLE_RATE * 30) // 3
    seam = y[cut - 200:cut + 200]
    assert abs(np.sqrt(np.mean(seam**2)) - 0.3 / np.sqrt(2)) < 0.01
    assert np.abs(np.diff(y[cut - 200:cut + 200])).max() < 0.3 * 2 * np.pi * 1000 / 16000 * 1.05
