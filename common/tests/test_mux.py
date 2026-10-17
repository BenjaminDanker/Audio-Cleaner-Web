"""Putting cleaned sound back into a video: the file's other audio tracks and its own subtitles come along."""
import numpy as np
import pytest

from acw_common import ffmpeg as ffmod
from acw_common.ffmpeg import FFmpeg

from test_common import FF_DIR, make_multi_track_video, needs_ff, probe_streams


@pytest.fixture
def source(tmp_path):
    src, srt = make_multi_track_video(tmp_path)
    clean = FFmpeg(FF_DIR).write_audio(np.zeros(3 * 48000, np.float32), 48000, tmp_path / "clean", "wav")
    return src, clean, srt


@needs_ff
@pytest.mark.parametrize("ext,subtitle", [(".mkv", "subrip"), (".mp4", "mov_text")])
def test_other_tracks_and_subtitles_are_kept(source, tmp_path, ext, subtitle):
    src, clean, srt = source
    report = {}
    out = FFmpeg(FF_DIR).mux_video(src, clean, tmp_path / f"out{ext}", subtitles=srt, sub_lang="eng", report=report)
    kinds = [(s["codec_type"], s["codec_name"]) for s in probe_streams(out)]
    assert [k for k, _ in kinds].count("audio") == 2  # the cleaned first track and the second one, as it was
    assert [c for k, c in kinds if k == "subtitle"] == [subtitle, subtitle]  # its own subtitles and the new ones
    assert report == {"audio_kept": 1, "audio_dropped": 0, "subtitles_kept": 1, "subtitles_dropped": 0}


@needs_ff
def test_a_codec_the_container_cannot_take_is_left_out_not_fatal(source):
    src, _, _ = source
    # aac does not go into webm: the plan drops that track and keeps the cleaned one
    plan = ffmod._plan_extras(FFmpeg(FF_DIR).probe(src), ".webm")
    assert plan["audio"] == [] and plan["total_audio"] == 1
