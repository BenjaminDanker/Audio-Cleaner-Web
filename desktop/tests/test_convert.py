"""Choosing what a result is: its name, its type (the ending says which) and the size of the picture.

The engine part: which endings a file can become, what makes a name unusable, and every kind of conversion (sound to
sound, video to video with and without saving the picture again, video to just its sound, sound to a video with a
black picture). The noise removal itself is left out of most of these (quick_clean): it has its own tests."""
from __future__ import annotations

import sys

import pytest

from acw_common import captions
from acw_common.ffmpeg import MediaError
from audio_cleaner_desktop import engine

from conftest import displayed_size, ffprobe_json, make_video, picture_packets, stream_facts


def length(path) -> float:
    return float(ffprobe_json(path)["format"]["duration"])

# Windows and macOS try the system's H.264 encoder first (and fall back to FFmpeg's own when it does not work there).
OS_H264 = sys.platform in ("win32", "darwin")


def codecs(*plain: str) -> set[str]:
    return {*plain, "h264"} if OS_H264 else set(plain)


AUDIO_CODECS = {".wav": "pcm_s24le", ".flac": "flac", ".mp3": "mp3", ".m4a": "aac", ".opus": "opus", ".ogg": "opus"}


def audio_source(ext=".wav", duration=3.0, channels=1):
    return engine.Source(False, ext, None, duration, channels)


def video_source(ext=".mp4", codec="h264"):
    return engine.Source(True, ext, codec, 10.0, 2)


# -- which types a file can become, and which names are fine (no FFmpeg needed) --------------------------------


def test_a_sound_can_become_any_sound_or_a_video_and_a_video_any_video_or_its_sound():
    sound = engine.available_endings(audio_source())
    assert sound[:6] == [".mp3", ".wav", ".flac", ".m4a", ".opus", ".ogg"]
    assert set(sound[6:]) == {".mp4", ".mov", ".mkv", ".flv", ".webm"}
    video = engine.available_endings(video_source())
    assert video[:5] == [".mp4", ".mov", ".mkv", ".flv", ".webm"]
    assert set(video[5:]) == set(sound[:6])


def test_webm_is_offered_for_a_picture_it_can_hold_as_it_is_even_without_the_vp9_encoder(monkeypatch):
    monkeypatch.setattr(engine, "WEBM_ENCODE", False)
    assert ".webm" not in engine.available_endings(video_source(".mp4", "h264"))
    assert ".webm" not in engine.available_endings(audio_source())
    assert ".webm" in engine.available_endings(video_source(".webm", "vp9"))
    assert ".webm" in engine.available_endings(video_source(".mkv", "av1"))


def test_the_usual_name_and_ending_do_not_change_for_people_who_never_touch_the_box():
    opts = engine.JobOptions()
    assert engine.default_output_name("/x/talk.wav", audio_source(".wav"), opts) == "talk-cleaned.wav"
    assert engine.default_output_name("/x/talk.mp3", audio_source(".mp3"), opts) == "talk-cleaned.mp3"
    assert engine.default_output_name("/x/talk.aiff", audio_source(".aiff"), opts) == "talk-cleaned.wav"
    assert engine.default_output_name("/x/talk.wma", audio_source(".wma"), opts) == "talk-cleaned.flac"
    assert engine.default_output_name("/x/clip.mp4", video_source(".mp4"), opts) == "clip-cleaned.mp4"
    assert engine.default_output_name("/x/clip.avi", video_source(".avi"), opts) == "clip-cleaned.mkv"
    assert engine.default_output_name("/x/clip.m4v", video_source(".m4v"), opts) == "clip-cleaned.m4v"
    # the "Audio file type" setting is where the box starts for a sound
    assert engine.default_output_name("/x/talk.wav", audio_source(".wav"), engine.JobOptions(audio_format="opus")) == "talk-cleaned.opus"
    # ... and a video is not touched by it
    assert engine.default_output_name("/x/clip.mp4", video_source(".mp4"), engine.JobOptions(audio_format="opus")) == "clip-cleaned.mp4"
    assert engine.output_extension(type("P", (), {"has_video": False, "ext": ".mp3", "duration": 5.0, "audio_channels": 1})(), opts) == ".mp3"


def test_a_very_long_recording_cannot_be_a_wav_so_the_usual_name_says_flac_and_a_typed_wav_is_refused():
    long = audio_source(".wav", duration=4 * 3600, channels=2)
    assert engine.default_output_name("/x/a.wav", long, engine.JobOptions()) == "a-cleaned.flac"
    assert "too long" in engine.check_output_name("a.wav", long)[1]
    assert engine.check_output_name("a.flac", long) == (".flac", "")


@pytest.mark.parametrize("name,ending", [
    ("talk.mp3", ".mp3"), ("My Talk 2.FLAC", ".flac"), ("a.b.c.ogg", ".ogg"), ("  padded.wav  ", ".wav"), ("ünï cødé.opus", ".opus"),
    ("clip.mp4", ".mp4"), ("x.WebM", ".webm"),
])
def test_good_names(name, ending):
    assert engine.check_output_name(name, audio_source()) == (ending, "")


@pytest.mark.parametrize("name,part", [
    ("", "Type a name"), ("   ", "Type a name"), ("talk", "Add an ending"), ("talk.", "can't end"), (".mp3", "in front of"),
    ("talk.xyz", "can't be saved as .xyz"), ("v1.2", "can't be saved as .2"), ("a/b.mp3", "can't contain"),
    ("a\\b.mp3", "can't contain"), ("a:b.mp3", "can't contain"), ('a"b.mp3', "can't contain"), ("a?.mp3", "can't contain"),
    ("a\tb.mp3", "can't contain"), ("con.mp3", "Windows"), ("LPT1.wav", "Windows"), ("nul.txt.mp3", "Windows"),
    ("x" * 260 + ".mp3", "too long"), ("talk.mov.exe", "can't be saved as .exe"),
])
def test_unusable_names_get_a_short_reason(name, part):
    ending, problem = engine.check_output_name(name, audio_source())
    assert ending == "" and part in problem, problem


def test_the_dropdown_and_the_name_box_agree_on_endings():
    assert engine.with_ending("talk-cleaned.wav", ".mp3") == "talk-cleaned.mp3"
    assert engine.with_ending("talk-cleaned.WAV", ".flac") == "talk-cleaned.flac"
    assert engine.with_ending("talk-cleaned", ".mp3") == "talk-cleaned.mp3"
    assert engine.with_ending("episode v1.2", ".mp3") == "episode v1.2.mp3"  # ".2" is part of the name, not an ending
    assert engine.with_ending("  spaced  ", ".opus") == "spaced.opus"
    assert engine.split_name("a.b.MP3") == ("a.b", ".mp3") and engine.split_name("none") == ("none", "")


def test_video_size_is_a_checked_option():
    assert engine.JobOptions().video_size == "same" and engine.JobOptions().short_side is None
    assert [engine.JobOptions(video_size=v).short_side for v in engine.VIDEO_SIZES[1:]] == [2160, 1440, 1080, 720, 480]
    with pytest.raises(ValueError, match="video size"):
        engine.JobOptions(video_size="900")


def test_a_name_that_the_file_cannot_have_is_a_message_not_a_crash(tmp_path):
    with pytest.raises(MediaError, match="can't be saved as .xyz"):
        engine.output_name_for(tmp_path / "a.wav", audio_source(), engine.JobOptions(), "a.xyz")
    with pytest.raises(MediaError, match="does not end in .mp3"):
        engine.output_name_for(tmp_path / "a.wav", audio_source(), engine.JobOptions(), "a.wav", "mp3")
    assert engine.output_name_for(tmp_path / "a.wav", audio_source(), engine.JobOptions(), None, "mp3") == "a-cleaned.mp3"
    assert engine.output_name_for(tmp_path / "a.wav", audio_source(), engine.JobOptions(), None, ".OGG") == "a-cleaned.ogg"
    assert engine.output_name_for(tmp_path / "a.wav", audio_source(), engine.JobOptions(), "x.mp3", "mp3") == "x.mp3"


# -- sound to sound -----------------------------------------------------------------------------------------------


@pytest.mark.native
@pytest.mark.parametrize("ending", list(AUDIO_CODECS))
def test_sound_to_every_sound_type_by_its_ending(clip_wav, tmp_path, quick_clean, ending):
    result = engine.process(clip_wav, out_dir=tmp_path, output_name="Episode 1" + ending)
    assert result.output == tmp_path / ("Episode 1" + ending)
    assert not result.has_video
    assert [s["codec_name"] for s in stream_facts(result.output, "audio")] == [AUDIO_CODECS[ending]]
    assert not stream_facts(result.output, "video")
    assert [p.name for p in tmp_path.iterdir()] == ["Episode 1" + ending]  # nothing else left behind


@pytest.mark.native
def test_the_whole_pipeline_with_a_chosen_name(clip_wav, tmp_path):
    result = engine.process(clip_wav, engine.JobOptions(polish=False), out_dir=tmp_path, output_name="clean voice.mp3")
    assert result.output.name == "clean voice.mp3" and stream_facts(result.output, "audio")[0]["codec_name"] == "mp3"
    assert length(result.output) == pytest.approx(3.0, abs=0.3)


@pytest.mark.native
def test_a_format_alone_keeps_the_usual_name(clip_wav, tmp_path, quick_clean):
    out = engine.process_file(clip_wav, out_dir=tmp_path, output_format="m4a")
    assert out.name == "clip-cleaned.m4a" and stream_facts(out, "audio")[0]["codec_name"] == "aac"


# -- names that are taken, names that are refused ------------------------------------------------------------------


@pytest.mark.native
def test_a_taken_name_is_never_replaced_it_gets_a_number(clip_wav, tmp_path, quick_clean):
    first = engine.process_file(clip_wav, out_dir=tmp_path, output_name="mine.flac")
    first_bytes = first.read_bytes()
    second = engine.process_file(clip_wav, out_dir=tmp_path, output_name="mine.flac")
    third = engine.process_file(clip_wav, out_dir=tmp_path, output_name="mine.flac")
    assert (first.name, second.name, third.name) == ("mine.flac", "mine (2).flac", "mine (3).flac")
    assert first.read_bytes() == first_bytes
    # the usual name keeps its own rule
    a = engine.process_file(clip_wav, out_dir=tmp_path)
    b = engine.process_file(clip_wav, out_dir=tmp_path)
    assert (a.name, b.name) == ("clip-cleaned.wav", "clip-cleaned (2).wav")


@pytest.mark.native
def test_naming_the_result_like_the_original_does_not_touch_the_original(clip_wav, tmp_path, quick_clean):
    src = tmp_path / "talk.wav"
    src.write_bytes(clip_wav.read_bytes())
    before = src.read_bytes()
    out = engine.process_file(src, output_name="talk.wav")
    assert out.name == "talk (2).wav" and src.read_bytes() == before


@pytest.mark.native
@pytest.mark.parametrize("name", ["", "noending", "bad|name.mp3", "talk.exe", "con.wav", "../up.mp3", "a/b.mp3"])
def test_an_unusable_name_stops_before_anything_is_written(clip_wav, tmp_path, quick_clean, name):
    with pytest.raises(MediaError):
        engine.process_file(clip_wav, out_dir=tmp_path, output_name=name)
    assert list(tmp_path.iterdir()) == []


# -- video ---------------------------------------------------------------------------------------------------------


@pytest.mark.native
def test_a_video_in_another_container_keeps_its_picture_when_it_fits(clip_wav, tmp_path, quick_clean):
    src = make_video(tmp_path / "in.mp4", clip_wav)
    for ending in (".mkv", ".mov", ".mp4"):
        out = engine.process_file(src, out_dir=tmp_path / "out", output_name="again" + ending)
        assert stream_facts(out, "video")[0]["codec_name"] == "mpeg4", ending
        assert picture_packets(out) == picture_packets(src), ending  # the same packets: not saved again
        assert [s["codec_name"] for s in stream_facts(out, "audio")] == ["aac"]
        assert displayed_size(out) == (160, 90)


@pytest.mark.native
def test_a_video_is_saved_again_only_when_the_new_type_cannot_hold_its_picture(clip_wav, tmp_path, quick_clean):
    src = make_video(tmp_path / "in.mp4", clip_wav)
    flv = engine.process(src, out_dir=tmp_path, output_name="old.flv")
    assert {s["codec_name"] for s in stream_facts(flv.output, "video")} <= codecs("flv1")  # MPEG-4 is not a Flash video format
    assert [s["codec_name"] for s in stream_facts(flv.output, "audio")] == ["aac"]
    assert displayed_size(flv.output) == (160, 90)
    web = engine.process(src, out_dir=tmp_path, output_name="web.webm")
    assert [s["codec_name"] for s in stream_facts(web.output, "video")] == ["vp9"]
    assert [s["codec_name"] for s in stream_facts(web.output, "audio")] == ["opus"]
    if not OS_H264:
        assert web.notice is None and flv.notice is None  # the first-choice encoder worked, no fallback to say


@pytest.mark.native
def test_a_webm_with_a_picture_it_holds_stays_a_copy(clip_wav, tmp_path, quick_clean):
    src = make_video(tmp_path / "in.webm", clip_wav, vcodec="libvpx-vp9", acodec="libopus")
    out = engine.process_file(src, out_dir=tmp_path / "o")
    assert out.suffix == ".webm" and picture_packets(out) == picture_packets(src)
    mkv = engine.process_file(src, out_dir=tmp_path / "o", output_name="x.mkv")
    assert picture_packets(mkv) == picture_packets(src)


@pytest.mark.native
def test_a_video_can_become_just_its_sound(clip_wav, tmp_path, quick_clean):
    src = make_video(tmp_path / "in.mp4", clip_wav)
    for ending in AUDIO_CODECS:
        out = engine.process(src, out_dir=tmp_path / "out", output_name="sound" + ending)
        assert out.output.suffix == ending and out.has_video  # has_video says what the original was
        assert not stream_facts(out.output, "video"), ending
        assert [s["codec_name"] for s in stream_facts(out.output, "audio")] == [AUDIO_CODECS[ending]], ending
        assert length(out.output) == pytest.approx(2.0, abs=0.3)


@pytest.mark.native
def test_a_video_with_two_sounds_says_what_became_of_the_second_one_when_only_the_sound_is_kept(clip_wav, tmp_path, quick_clean):
    from conftest import bundled_ffmpeg, run_ffmpeg

    src = make_video(tmp_path / "in.mp4", clip_wav)
    two = tmp_path / "two.mkv"
    run_ffmpeg(bundled_ffmpeg(), "-i", str(src), "-i", str(clip_wav), "-map", "0:v", "-map", "0:a", "-map", "1:a", "-c", "copy", str(two))
    out = engine.process(two, out_dir=tmp_path / "out", output_name="sound.mp3")
    assert "only the first audio track was cleaned" in out.notice and "wasn't kept" in out.notice


# -- video size ----------------------------------------------------------------------------------------------------


@pytest.mark.native
@pytest.mark.parametrize("src_size,rotation,want", [
    ((1280, 720), 0, (854, 480)),    # smaller
    ((160, 90), 0, (854, 480)),      # bigger: stretched
    ((720, 1280), 0, (480, 854)),    # portrait stays portrait
    ((90, 160), 0, (480, 854)),
    ((1280, 720), 90, (480, 854)),   # stored sideways, shown upright: portrait
    ((640, 640), 0, (480, 480)),
])
def test_video_size_keeps_the_shape_and_the_orientation(clip_wav, tmp_path, quick_clean, src_size, rotation, want):
    src = make_video(tmp_path / "in.mp4", clip_wav, size=src_size, seconds=1, fps=5, rotation=rotation)
    out = engine.process_file(src, engine.JobOptions(video_size="480"), out_dir=tmp_path / "out")
    assert displayed_size(out) == want
    video = stream_facts(out, "video")[0]
    assert video.get("sample_aspect_ratio", "1:1") == "1:1"
    assert (int(video["width"]), int(video["height"])) == want  # turned upright, no rotation left to apply


@pytest.mark.native
def test_a_picture_that_already_has_the_size_is_not_saved_again(clip_wav, tmp_path, quick_clean):
    src = make_video(tmp_path / "in.mp4", clip_wav, size=(854, 480), seconds=1, fps=5)
    out = engine.process_file(src, engine.JobOptions(video_size="480"), out_dir=tmp_path / "out")
    assert picture_packets(out) == picture_packets(src)


@pytest.mark.native
def test_the_size_does_not_touch_a_sound_that_becomes_just_a_sound(clip_wav, tmp_path, quick_clean):
    out = engine.process_file(clip_wav, engine.JobOptions(video_size="480"), out_dir=tmp_path, output_name="x.mp3")
    assert not stream_facts(out, "video")


# -- sound to video -------------------------------------------------------------------------------------------------


@pytest.mark.native
@pytest.mark.parametrize("ending,vcodecs,acodec", [
    (".mp4", codecs("mpeg4"), "aac"), (".mov", codecs("mpeg4"), "aac"), (".mkv", codecs("mpeg4"), "aac"),
    (".flv", codecs("flv1"), "aac"), (".webm", {"vp9"}, "opus"),
])
def test_a_sound_becomes_a_small_video_with_a_black_picture(clip_wav, tmp_path, quick_clean, ending, vcodecs, acodec):
    result = engine.process(clip_wav, out_dir=tmp_path, output_name="black" + ending)
    assert result.output.name == "black" + ending
    video = stream_facts(result.output, "video")
    assert len(video) == 1 and video[0]["codec_name"] in vcodecs
    assert displayed_size(result.output) == (1280, 720)
    assert [a["codec_name"] for a in stream_facts(result.output, "audio")] == [acodec]
    assert length(result.output) == pytest.approx(3.0, abs=0.9)  # a picture frame a second: the file may run up to a second over
    assert result.output.stat().st_size < 400_000  # one frame a second of black: a few KB per second at most


@pytest.mark.native
def test_the_black_picture_takes_the_video_size_setting(clip_wav, tmp_path, quick_clean):
    out = engine.process_file(clip_wav, engine.JobOptions(video_size="480"), out_dir=tmp_path, output_name="black.mp4")
    assert displayed_size(out) == (854, 480)


@pytest.fixture
def two_cues(monkeypatch):
    """Subtitles without the speech model: the transcription gives two fixed lines."""
    segments = [captions.Segment(0.2, 1.4, "hello there", []), captions.Segment(1.6, 2.8, "this is a test", [])]
    monkeypatch.setattr(engine, "_transcribe", lambda *a, **k: (list(segments), "en"))


@pytest.mark.native
@pytest.mark.parametrize("ending,subtitle", [(".mp4", "mov_text"), (".mov", "mov_text"), (".mkv", "subrip"), (".webm", "webvtt")])
def test_a_sound_made_into_a_video_carries_embedded_subtitles(clip_wav, tmp_path, quick_clean, two_cues, ending, subtitle):
    result = engine.process(clip_wav, engine.JobOptions(subtitles="embed", language="en"), out_dir=tmp_path, output_name="talk" + ending)
    assert result.embedded and result.subtitles is None
    subs = stream_facts(result.output, "subtitle")
    assert [s["codec_name"] for s in subs] == [subtitle]
    assert subs[0]["tags"]["language"] == "eng"
    assert list(tmp_path.glob("*.srt")) == []


@pytest.mark.native
def test_a_flash_video_cannot_hold_subtitles_so_it_gets_an_srt_file(clip_wav, tmp_path, quick_clean, two_cues):
    result = engine.process(clip_wav, engine.JobOptions(subtitles="embed", language="en"), out_dir=tmp_path, output_name="talk.flv")
    assert not result.embedded and result.subtitles == tmp_path / "talk.srt"
    assert "-->" in result.subtitles.read_text(encoding="utf-8")
    assert not stream_facts(result.output, "subtitle")


@pytest.mark.native
def test_subtitles_next_to_a_result_with_a_chosen_name_share_its_name(clip_wav, tmp_path, quick_clean, two_cues):
    result = engine.process(clip_wav, engine.JobOptions(subtitles="srt", language="en"), out_dir=tmp_path, output_name="Talk 1.mp3")
    assert result.paths == [tmp_path / "Talk 1.mp3", tmp_path / "Talk 1.srt"]
    again = engine.process(clip_wav, engine.JobOptions(subtitles="srt", language="en"), out_dir=tmp_path, output_name="Talk 1.mp3")
    assert again.paths == [tmp_path / "Talk 1 (2).mp3", tmp_path / "Talk 1 (2).srt"]


@pytest.mark.native
def test_a_video_made_from_a_sound_reports_its_progress_and_can_be_stopped(clip_wav, tmp_path, quick_clean):
    import threading

    seen = []
    engine.process_file(clip_wav, out_dir=tmp_path, output_name="p.mp4", progress=seen.append)
    assert seen == sorted(seen) and seen[-1] == 1.0
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(engine.Cancelled):
        engine.process_file(clip_wav, out_dir=tmp_path, output_name="q.mp4", cancel=cancel)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["p.mp4"]


def test_audio_formats_keep_working_for_the_server_and_the_old_callers():
    from acw_common.ffmpeg import AUDIO_FORMATS

    assert AUDIO_FORMATS["opus"][0] == ".opus" and AUDIO_FORMATS["ogg"][0] == ".ogg"
