"""The picture of a video result: copied when the container takes it, re-encoded when it does not or when the size
changes, resized keeping its shape; and a black picture for a sound. The server never asks for any of this and keeps
copying."""
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from acw_common import ffmpeg as ffmod
from acw_common.dfn import Cancelled
from acw_common.ffmpeg import FFmpeg, MediaError, ProbeResult, VideoOut

from test_common import FF_DIR, FIXTURE_FFMPEG, needs_ff


def probed(**kw) -> ProbeResult:
    base = dict(format_name="mov,mp4,m4a,3gp,3g2,mj2", duration=5.0, audio_index=1, audio_channels=2, audio_codec="aac",
                has_video=True, video_codec="h264", ext=".mp4", video_width=1920, video_height=1080, video_fps=30.0,
                video_bitrate=8_000_000)
    base.update(kw)
    return ProbeResult(**base)


# -- sizes ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("w,h,sar,rot,side,want", [
    (1920, 1080, 1.0, 0, 720, (1280, 720)),
    (1920, 1080, 1.0, 0, 2160, (3840, 2160)),   # bigger: stretched
    (1280, 720, 1.0, 0, 480, (854, 480)),
    (1080, 1920, 1.0, 0, 720, (720, 1280)),     # portrait stays portrait
    (1080, 1920, 1.0, 0, 480, (480, 854)),
    (1920, 1080, 1.0, 90, 720, (720, 1280)),    # stored sideways, shown upright
    (1920, 1080, 1.0, 270, 480, (480, 854)),
    (1920, 1080, 1.0, 180, 480, (854, 480)),
    (640, 640, 1.0, 0, 480, (480, 480)),        # square stays square
    (720, 576, 64 / 45, 0, 480, (854, 480)),    # a wide picture made of narrow pixels: square pixels afterwards
    (1000, 1001, 1.0, 0, 480, (480, 480)),      # nearly square: even numbers
    (4000, 100, 1.0, 0, 480, (19200, 480)),
])
def test_picture_size_keeps_the_shape_and_the_orientation(w, h, sar, rot, side, want):
    got = ffmod.picture_size(w, h, sar, rot, side)
    assert got == want and got[0] % 2 == 0 and got[1] % 2 == 0
    assert (got[0] >= got[1]) == (want[0] >= want[1])


def test_the_black_picture_is_16_by_9():
    assert ffmod.black_size(None) == (1280, 720)
    assert ffmod.black_size(1080) == (1920, 1080)
    assert ffmod.black_size(480) == (854, 480)
    assert ffmod.black_size(2160) == (3840, 2160)


def test_bit_rates_scale_with_the_picture_and_stay_between_limits():
    small = ffmod.picture_bitrate("mpeg4", 640, 360, 30)
    big = ffmod.picture_bitrate("mpeg4", 1920, 1080, 30)
    assert small < big < 80_000_000
    assert ffmod.picture_bitrate("libvpx-vp9", 1920, 1080, 30) < ffmod.picture_bitrate("mpeg4", 1920, 1080, 30)
    assert ffmod.picture_bitrate("h264_mf", 16, 16, 1) == 250_000  # the floor
    assert ffmod.picture_bitrate("mpeg4", 7680, 4320, 120) == 80_000_000  # the ceiling
    assert ffmod.picture_bitrate("mpeg4", 16, 16, 1, floor=64_000) == 64_000
    # not much more than the original had, unless the picture gets bigger
    assert ffmod.picture_bitrate("mpeg4", 1920, 1080, 30, source_bps=2_000_000, source_pixels=1920 * 1080) == 3_000_000
    assert ffmod.picture_bitrate("mpeg4", 1920, 1080, 30, source_bps=2_000_000, source_pixels=640 * 360) > 3_000_000


def test_which_encoders_are_tried(monkeypatch):
    for platform, first in (("win32", "h264_mf"), ("darwin", "h264_videotoolbox")):
        monkeypatch.setattr(sys, "platform", platform)
        assert ffmod.picture_encoders(".mp4") == [first, "mpeg4"]
        assert ffmod.picture_encoders(".flv") == [first, "flv"]
        assert ffmod.picture_encoders(".webm") == ["libvpx-vp9"]
    monkeypatch.setattr(sys, "platform", "linux")
    assert ffmod.picture_encoders(".mkv") == ["mpeg4"] and ffmod.picture_encoders(".flv") == ["flv"]


# -- copy or encode ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("info,ext,keeps", [
    (dict(), ".mp4", True),                                   # same container
    (dict(), ".m4v", True),                                   # an mp4 by another name
    (dict(), ".mkv", True),                                   # anything goes into mkv
    (dict(video_codec="prores", ext=".mov"), ".mp4", False),  # mp4 does not take ProRes
    (dict(video_codec="prores", ext=".mov"), ".mov", True),
    (dict(video_codec="prores", ext=".mov"), ".mkv", True),
    (dict(), ".mov", True),
    (dict(), ".webm", False),                                 # H.264 is not a WebM picture
    (dict(), ".flv", True),                                   # but a Flash video one
    (dict(video_codec="hevc"), ".flv", False),
    (dict(video_codec="mpeg4"), ".flv", False),
    (dict(video_codec="vp9", ext=".webm"), ".webm", True),
    (dict(video_codec="av1", ext=".mkv"), ".webm", True),
    (dict(video_codec="vp9", ext=".webm"), ".mp4", True),
    (dict(video_codec="vp8", ext=".webm"), ".mp4", False),
    (dict(video_codec="msmpeg4v3", ext=".avi"), ".mkv", True),
    (dict(video_codec="msmpeg4v3", ext=".avi"), ".mp4", False),
])
def test_a_picture_is_kept_when_the_new_container_takes_it(info, ext, keeps):
    plan = ffmod._picture_plan(probed(**info), ext, VideoOut())
    assert (plan is None) == keeps
    if plan:
        assert plan["encoders"][-1] in ("mpeg4", "flv", "libvpx-vp9") and plan["size"] is None


def test_another_size_always_means_encoding_and_the_same_size_does_not():
    assert ffmod._picture_plan(probed(), ".mp4", VideoOut(1080)) is None  # it is 1080p already
    plan = ffmod._picture_plan(probed(), ".mp4", VideoOut(720))
    assert plan["size"] == (1280, 720) and plan["out"] == (1280, 720)
    assert ffmod._picture_plan(probed(video_width=1080, video_height=1920), ".mp4", VideoOut(1080)) is None
    # stored sideways: 1080 wide shown as 1080 tall
    assert ffmod._picture_plan(probed(video_width=1920, video_height=1080, video_rotation=90), ".mp4", VideoOut(1080)) is None
    assert ffmod._picture_plan(probed(video_width=1920, video_height=1080, video_rotation=90), ".mp4", VideoOut(720))["size"] == (720, 1280)
    # square pixels are made even when the shape is wide only because the pixels are
    assert ffmod._picture_plan(probed(video_width=720, video_height=480, video_sar=1.5), ".mp4", VideoOut(480))["size"] == (1080, 480)


def test_odd_sizes_are_made_even_when_the_picture_must_be_saved_again():
    plan = ffmod._picture_plan(probed(video_codec="hevc", ext=".mkv", video_width=1919, video_height=1079), ".webm", VideoOut())
    assert plan["size"] == (1918, 1078)


def test_a_sound_file_or_an_unreadable_one_has_no_picture_to_plan():
    assert ffmod._picture_plan(None, ".mp4", VideoOut(720)) is None
    assert ffmod._picture_plan(probed(has_video=False, video_codec=None), ".mp4", VideoOut(720)) is None


def test_the_picture_options_for_each_encoder():
    plan = {"fps": 25.0, "size": (1280, 720), "out": (1280, 720), "source_bps": 0, "source_pixels": 0}
    mpeg4 = ffmod._picture_args("mpeg4", plan)
    assert mpeg4[:2] == ["-vf", "scale=1280:720:flags=lanczos,setsar=1"]
    assert ["-c:v", "mpeg4"] == mpeg4[2:4] and "-pix_fmt" in mpeg4 and mpeg4[mpeg4.index("-g") + 1] == "50"
    assert "-vf" not in ffmod._picture_args("h264_mf", {**plan, "size": None})
    vt = ffmod._picture_args("h264_videotoolbox", plan)
    assert vt[vt.index("-allow_sw") + 1] == "1"
    vp9 = ffmod._picture_args("libvpx-vp9", plan)
    assert "-row-mt" in vp9 and vp9[vp9.index("-deadline") + 1] == "good"


# -- with FFmpeg ----------------------------------------------------------------------------------------------------


def run(*args):
    subprocess.run([FIXTURE_FFMPEG, "-v", "error", "-y", *args], check=True, stdin=subprocess.DEVNULL)


def streams(path):
    import json

    ffprobe = str(FF_DIR / ("ffprobe.exe" if (FF_DIR / "ffprobe.exe").exists() else "ffprobe"))
    out = subprocess.run([ffprobe, "-v", "error", "-print_format", "json", "-show_streams", str(path)], capture_output=True,
                         text=True, check=True).stdout
    return json.loads(out)["streams"]


def shown(path) -> tuple[int, int]:
    video = next(s for s in streams(path) if s["codec_type"] == "video")
    turn = next((abs(int(d["rotation"])) for d in video.get("side_data_list", []) if "rotation" in d), 0)
    return (video["height"], video["width"]) if turn % 180 == 90 else (video["width"], video["height"])


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    d = tmp_path_factory.mktemp("pic")
    out = {}
    run("-f", "lavfi", "-i", "testsrc=size=320x180:rate=10:duration=3", "-f", "lavfi", "-i", "sine=d=3", "-c:v", "mpeg4", "-c:a", "aac",
        str(d / "land.mp4"))
    run("-f", "lavfi", "-i", "testsrc=size=180x320:rate=10:duration=3", "-f", "lavfi", "-i", "sine=d=3", "-c:v", "mpeg4", "-c:a", "aac",
        str(d / "port.mp4"))
    run("-display_rotation", "90", "-i", str(d / "land.mp4"), "-c", "copy", str(d / "turned.mp4"))
    run("-f", "lavfi", "-i", "testsrc=size=320x180:rate=10:duration=3", "-f", "lavfi", "-i", "sine=d=3", "-c:v", "libvpx-vp9",
        "-b:v", "200k", "-deadline", "realtime", "-cpu-used", "8", "-c:a", "libopus", str(d / "web.webm"))
    out.update({p.stem: p for p in d.iterdir()})
    out["clean"] = FFmpeg(FF_DIR).write_audio(np.zeros(3 * 48000, np.float32), 48000, d / "clean", "flac")
    return out


@needs_ff
def test_the_picture_facts_come_with_the_probe(sources):
    ff = FFmpeg(FF_DIR)
    land = ff.probe(sources["land"])
    assert (land.video_width, land.video_height, land.video_rotation, land.video_sar) == (320, 180, 0, 1.0)
    assert land.video_fps == pytest.approx(10.0) and land.video_bitrate > 0
    turned = ff.probe(sources["turned"])
    assert (turned.video_width, turned.video_height) == (320, 180) and turned.video_rotation % 180 == 90
    assert ff.probe(sources["clean"]).video_width == 0  # a sound has no picture


@needs_ff
def test_without_asking_the_picture_is_copied_as_before(sources, tmp_path):
    ff = FFmpeg(FF_DIR)
    out = ff.mux_video(sources["land"], sources["clean"], tmp_path / "o.mkv")
    assert [s["codec_name"] for s in streams(out) if s["codec_type"] == "video"] == ["mpeg4"]
    with pytest.raises(MediaError, match="Could not rebuild the video with the cleaned audio"):
        ff.mux_video(sources["land"], sources["clean"], tmp_path / "o.webm")  # mpeg4 does not fit WebM: no guessing
    report = {}
    assert ff.mux_video(sources["land"], sources["clean"], tmp_path / "o2.mkv", report=report)
    assert report == {"audio_kept": 0, "audio_dropped": 0, "subtitles_kept": 0, "subtitles_dropped": 0}  # as it always was
    assert ff.mux_video(sources["land"], sources["clean"], tmp_path / "o3.mkv", report=report, video=VideoOut())
    assert report["encoder"] == "copy" and report["fallback"] is False


@needs_ff
def test_asking_for_a_container_that_does_not_fit_encodes(sources, tmp_path):
    ff = FFmpeg(FF_DIR)
    report = {}
    web = ff.mux_video(sources["land"], sources["clean"], tmp_path / "o.webm", video=VideoOut(), report=report)
    assert [s["codec_name"] for s in streams(web) if s["codec_type"] in ("video", "audio")] == ["vp9", "opus"]
    assert report["encoder"] == "libvpx-vp9" and report["fallback"] is False
    flv = ff.mux_video(sources["land"], sources["clean"], tmp_path / "o.flv", video=VideoOut())
    assert {s["codec_name"] for s in streams(flv) if s["codec_type"] == "video"} <= {"flv1", "h264"}


@needs_ff
def test_resizing_keeps_the_shape_and_orientation(sources, tmp_path):
    ff = FFmpeg(FF_DIR)
    assert shown(ff.mux_video(sources["land"], sources["clean"], tmp_path / "a.mp4", video=VideoOut(360))) == (640, 360)  # up
    assert shown(ff.mux_video(sources["land"], sources["clean"], tmp_path / "b.mp4", video=VideoOut(90))) == (160, 90)    # down
    assert shown(ff.mux_video(sources["port"], sources["clean"], tmp_path / "c.mp4", video=VideoOut(360))) == (360, 640)
    turned = ff.mux_video(sources["turned"], sources["clean"], tmp_path / "d.mp4", video=VideoOut(360))
    assert shown(turned) == (360, 640)  # shown upright and stays tall
    assert not any("rotation" in d for s in streams(turned) for d in s.get("side_data_list", []))


@needs_ff
def test_a_fallback_is_used_when_the_first_encoder_cannot_run(sources, tmp_path, monkeypatch):
    monkeypatch.setattr(ffmod, "picture_encoders", lambda ext: ["no_such_encoder", "mpeg4"])
    report = {}
    out = FFmpeg(FF_DIR).mux_video(sources["land"], sources["clean"], tmp_path / "o.mp4", video=VideoOut(90), report=report)
    assert report["encoder"] == "mpeg4" and report["fallback"] is True and shown(out) == (160, 90)
    monkeypatch.setattr(ffmod, "picture_encoders", lambda ext: ["no_such_encoder"])
    with pytest.raises(MediaError, match="Could not convert the picture"):
        FFmpeg(FF_DIR).mux_video(sources["land"], sources["clean"], tmp_path / "p.mp4", video=VideoOut(90))


@needs_ff
def test_extra_tracks_and_subtitles_survive_a_re_encode(tmp_path):
    srt = tmp_path / "in.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n\n")
    multi = tmp_path / "multi.mkv"
    run("-f", "lavfi", "-i", "testsrc=size=64x64:rate=5:duration=3", "-f", "lavfi", "-i", "sine=f=440:d=3", "-f", "lavfi", "-i", "sine=f=880:d=3",
        "-i", str(srt), "-map", "0:v", "-map", "1:a", "-map", "2:a", "-map", "3:s", "-c:v", "mpeg4", "-c:a", "aac", "-c:s", "srt", str(multi))
    clean = FFmpeg(FF_DIR).write_audio(np.zeros(3 * 48000, np.float32), 48000, tmp_path / "clean", "wav")
    report = {}
    out = FFmpeg(FF_DIR).mux_video(multi, clean, tmp_path / "o.mp4", video=VideoOut(32), report=report)
    kinds = [s["codec_type"] for s in streams(out)]
    assert kinds.count("audio") == 2 and kinds.count("subtitle") == 1 and shown(out) == (32, 32)
    assert report["audio_kept"] == 1 and report["subtitles_kept"] == 1 and report["encoder"] == "mpeg4"


@needs_ff
def test_encoding_reports_its_progress_and_stops_when_asked(sources, tmp_path):
    seen = []
    FFmpeg(FF_DIR).mux_video(sources["land"], sources["clean"], tmp_path / "o.mp4", video=VideoOut(90), progress=seen.append)
    assert seen and seen == sorted(seen) and 0.0 <= seen[0] and seen[-1] <= 1.0
    cancel = threading.Event()
    ff = FFmpeg(FF_DIR, cancel=cancel)
    with pytest.raises(Cancelled):
        ff.mux_video(sources["land"], sources["clean"], tmp_path / "p.mp4", video=VideoOut(90), progress=lambda f: cancel.set())


@needs_ff
@pytest.mark.parametrize("ext,vcodecs,acodec,sub", [
    (".mp4", {"mpeg4", "h264"}, "aac", "mov_text"), (".mov", {"mpeg4", "h264"}, "aac", "mov_text"),
    (".mkv", {"mpeg4", "h264"}, "aac", "subrip"), (".webm", {"vp9"}, "opus", "webvtt"),
])
def test_a_sound_gets_a_black_picture_and_can_carry_subtitles(sources, tmp_path, ext, vcodecs, acodec, sub):
    srt = tmp_path / "s.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n\n")
    report = {}
    out = FFmpeg(FF_DIR).make_video(sources["clean"], tmp_path / f"b{ext}", 3.0, subtitles=srt, sub_lang="eng", report=report)
    found = {s["codec_type"]: s for s in streams(out)}
    assert found["video"]["codec_name"] in vcodecs and found["audio"]["codec_name"] == acodec
    assert found["subtitle"]["codec_name"] == sub and found["subtitle"]["tags"]["language"] == "eng"
    assert (found["video"]["width"], found["video"]["height"]) == (1280, 720)
    assert out.stat().st_size < 300_000 and report["fallback"] is False


@needs_ff
def test_the_black_picture_can_be_another_size_and_flash_video_has_no_subtitles(sources, tmp_path):
    ff = FFmpeg(FF_DIR)
    assert shown(ff.make_video(sources["clean"], tmp_path / "a.mp4", 3.0, video=VideoOut(480))) == (854, 480)
    flv = ff.make_video(sources["clean"], tmp_path / "b.flv", 3.0)
    assert {s["codec_type"] for s in streams(flv)} == {"video", "audio"}
    with pytest.raises(MediaError, match="Could not make a video file"):
        ff.make_video(sources["clean"], tmp_path / "c.flv", 3.0, subtitles=tmp_path / "missing.srt")


@needs_ff
def test_ogg_is_an_audio_format_of_its_own(tmp_path):
    out = FFmpeg(FF_DIR).write_audio(np.zeros(48000, np.float32), 48000, tmp_path / "x", "ogg")
    assert out.suffix == ".ogg" and streams(out)[0]["codec_name"] == "opus"
    assert Path(FFmpeg(FF_DIR).write_audio(np.zeros(48000, np.float32), 48000, tmp_path / "y", "opus")).suffix == ".opus"
