"""Putting cleaned sound back into a video: the file's other audio tracks and its own subtitles come along."""
import json
import subprocess

import numpy as np
import pytest

from acw_common.ffmpeg import FFmpeg

from test_common import FF_DIR, FIXTURE_FFMPEG, needs_ff


def streams(path):
    ffprobe = str(FF_DIR / ("ffprobe.exe" if (FF_DIR / "ffprobe.exe").exists() else "ffprobe"))
    out = subprocess.run([ffprobe, "-v", "error", "-print_format", "json", "-show_streams", str(path)], capture_output=True, text=True, check=True).stdout
    return [(s["codec_type"], s["codec_name"]) for s in json.loads(out)["streams"]]


@pytest.fixture
def source(tmp_path):
    ff = [FIXTURE_FFMPEG, "-v", "error", "-y"]
    srt = tmp_path / "in.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n\n")
    src = tmp_path / "multi.mkv"
    subprocess.run([*ff, "-f", "lavfi", "-i", "testsrc=size=64x64:rate=5:duration=3", "-f", "lavfi", "-i", "sine=f=440:d=3",
                    "-f", "lavfi", "-i", "sine=f=880:d=3", "-i", str(srt), "-map", "0:v", "-map", "1:a", "-map", "2:a", "-map", "3:s",
                    "-c:v", "mpeg4", "-c:a", "aac", "-c:s", "srt", str(src)], check=True)
    clean = FFmpeg(FF_DIR).write_audio(np.zeros(3 * 48000, np.float32), 48000, tmp_path / "clean", "wav")
    return src, clean, srt


@needs_ff
@pytest.mark.parametrize("ext,subtitle", [(".mkv", "subrip"), (".mp4", "mov_text")])
def test_other_tracks_and_subtitles_are_kept(source, tmp_path, ext, subtitle):
    src, clean, srt = source
    report = {}
    out = FFmpeg(FF_DIR).mux_video(src, clean, tmp_path / f"out{ext}", subtitles=srt, sub_lang="eng", report=report)
    kinds = streams(out)
    assert [k for k, _ in kinds].count("audio") == 2  # the cleaned first track and the second one, as it was
    assert [c for k, c in kinds if k == "subtitle"] == [subtitle, subtitle]  # its own subtitles and the new ones
    assert report == {"audio_kept": 1, "audio_dropped": 0, "subtitles_kept": 1, "subtitles_dropped": 0}


@needs_ff
def test_what_the_container_cannot_hold_is_left_out_not_fatal(source, tmp_path):
    src, clean, srt = source
    report = {}
    out = FFmpeg(FF_DIR).mux_video(src, clean, tmp_path / "out.mp4", report=report)  # mpeg4 video in mp4 is fine
    assert [k for k, _ in streams(out)].count("audio") == 2
    # a codec the container can't take without re-encoding is dropped, and the rest still works
    from acw_common import ffmpeg as ffmod

    plan = ffmod._plan_extras(FFmpeg(FF_DIR).probe(src), ".webm")
    assert plan["audio"] == [] and plan["total_audio"] == 1  # aac does not go into webm


@needs_ff
def test_a_single_track_video_is_unchanged_in_shape(tmp_path):
    ff = [FIXTURE_FFMPEG, "-v", "error", "-y"]
    src = tmp_path / "one.mp4"
    subprocess.run([*ff, "-f", "lavfi", "-i", "testsrc=size=64x64:rate=5:duration=2", "-f", "lavfi", "-i", "sine=d=2",
                    "-c:v", "mpeg4", "-c:a", "aac", str(src)], check=True)
    clean = FFmpeg(FF_DIR).write_audio(np.zeros(2 * 48000, np.float32), 48000, tmp_path / "clean", "wav")
    report = {}
    out = FFmpeg(FF_DIR).mux_video(src, clean, tmp_path / "o.mp4", report=report)
    assert [k for k, _ in streams(out)] == ["video", "audio"]
    assert report["audio_kept"] == 0 and report["subtitles_kept"] == 0
