"""Echo removal with a second audio track (OBS and others record mic on track 1, desktop sound on track 2).
Synthetic recordings: a "speaker" signal, and a mic that hears it back through a small room."""
import subprocess
import threading

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


@pytest.fixture(scope="module")
def room(ff, tmp_path_factory):
    """The room recording as a file, its decoded mic track (x) and the echo-removed result (y), made once."""
    d = tmp_path_factory.mktemp("echo")
    mic, far, near = room_recording(extra=123)  # not a whole number of 10 ms frames or blocks
    src = make_file(ff, d / "rec.mkv", [mic, far])
    probe = ff.probe(src)
    x = pipeline.decode_for_clean(ff, src, probe, 60, work_dir=d)
    y = pipeline.remove_echo(ff, src, probe, x, 60, work_dir=d)
    return dict(ff=ff, dir=d, src=src, probe=probe, x=x, y=y, far=far, near=near)


@needs_ff
def test_second_track_cancels_the_echo_and_keeps_the_voice(room):
    x, y, near = room["x"], room["y"], room["near"]
    assert room["probe"].audio_tracks == 2 and x.shape[1] == 1
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
def test_blocks_of_the_grid_take_the_echo_out_like_one_pass(room, monkeypatch, tmp_path):
    """Echo removal runs on the block grid (blocks.GRID_ECHO: 30 s blocks, each with a fresh canceller that adapts on the 10 s before
    it). A short grid puts several blocks, and seams, into the 14 s file: the blocks whose context reaches the start of the file are the
    one pass bit for bit, the later ones adapt on their own context, and the result is as good (the waveform of a later block is
    not the one pass's: echo removal converges in quality, not in waveform). Changed from "block boundaries do not change the result"
    (1e-6) when whole-file echo removal went from one continuous canceller to this grid."""
    from acw_common import blocks

    monkeypatch.setattr(blocks, "GRID_ECHO", blocks.Grid(3 * SR / SR + 17 / SR, 6.0, 0.01))
    got = pipeline.remove_echo(room["ff"], room["src"], room["probe"], room["x"], 60, work_dir=tmp_path)  # (not room["dir"]: the result is a file there)
    x, y, near = room["x"], room["y"], room["near"]
    assert got.shape == y.shape and np.isfinite(got).all()
    assert not np.array_equal(got, y)  # the later blocks did adapt on their own context
    assert np.abs(got[: 6 * SR] - y[: 6 * SR]).max() < 1e-6  # the blocks whose 6 s of context reach the file start
    assert rms_db(x[FAR_ONLY]) - rms_db(got[FAR_ONLY]) > 20  # echo mostly gone, also in a block that adapted on 6 s
    kept, voice = got[NEAR_ONLY, 0], near[NEAR_ONLY]
    assert abs(rms_db(kept) - rms_db(voice)) < 1.5
    assert np.corrcoef(np.abs(np.fft.rfft(kept)), np.abs(np.fft.rfft(voice)))[0, 1] > 0.95
    snr = 10 * np.log10(np.sum(y.astype(np.float64) ** 2) / np.sum((got.astype(np.float64) - y) ** 2))
    assert snr > 25, snr  # (measured: 32 dB on this recording, with 6 s of context)
    print(f"echo removal, blocks of 3 s with 6 s of context vs one pass: SNR {snr:.1f} dB")


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
def test_cancel_stops_echo_removal(room):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(pipeline.Cancelled):
        pipeline.remove_echo(room["ff"], room["src"], room["probe"], room["x"], 60, cancel=cancel)
