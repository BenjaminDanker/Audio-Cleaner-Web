"""A video made in another file type or size on the server (VideoOut with its own encoders, `fast`; mux_video with no new sound):
which picture is made again and which is copied (needs_picture_encode), the picture options each encoder gets, and what the
file's own sound becomes (copied, or made in the container's own kind when it can't hold it). The files are real, small, generated
here; the card's NVENC is never run (arguments only), x264 and VP9 are."""
import subprocess

import pytest

from acw_common import ffmpeg as ffmod
from acw_common.ffmpeg import FFmpeg, MediaError, VideoOut, convert_factor, needs_picture_encode, picture_copies, picture_work

from test_common import FF_DIR, FIXTURE_FFMPEG, needs_ff, probe_streams
from test_video_out import probed, shown

X264 = VideoOut(encoders=("libx264",), fast=True)
VP9 = VideoOut(encoders=("libvpx-vp9",), fast=True)


def make(args, out):
    subprocess.run([FIXTURE_FFMPEG, "-v", "error", "-y", *args, str(out)], check=True, stdin=subprocess.DEVNULL)
    return out


def kinds(path):
    """{"video": codec, "audio": codec, ...} of the first stream of each kind."""
    out = {}
    for s in probe_streams(path):
        out.setdefault(s["codec_type"], s["codec_name"])
    return out


# -- what is made again ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("info,ext,size,encode", [
    (dict(), ".mp4", None, False),                                          # H.264 into MP4 at its size: copied
    (dict(), ".mkv", None, False),                                          # MKV holds any picture
    (dict(), ".mov", None, False),                                          # (H.264 goes into a MOV)
    (dict(), ".webm", None, True),                                          # WebM does not hold H.264
    (dict(video_codec="wmv2", ext=".wmv"), ".mkv", None, False),            # a WMV is copied into MKV, whatever its picture
    (dict(video_codec="wmv2", ext=".wmv"), ".mp4", None, True),             # but MP4 can't hold it
    (dict(video_codec="wmv2", ext=".wmv"), ".webm", None, True),
    (dict(video_codec="vp9", ext=".webm"), ".webm", None, False),
    (dict(video_codec="vp9", ext=".webm"), ".mp4", None, False),            # (MP4 holds VP9)
    (dict(), ".mp4", 720, True),                                            # another size: made again
    (dict(), ".mkv", 720, True),                                            # even into MKV
    (dict(), ".mp4", 1080, False),                                          # the size it already is (1080p): nothing to do
    (dict(video_codec="wmv2", ext=".wmv"), ".mkv", 1080, False),
    (dict(video_width=1080, video_height=1920), ".mp4", 1080, False),       # portrait: the shorter side is already 1080
    (dict(), ".mp4", 2160, True),                                           # a bigger size is only stretched, but it is made again
])
def test_which_pictures_are_made_again(info, ext, size, encode):
    assert needs_picture_encode(probed(**info), ext, VideoOut(short_side=size)) is encode


def test_a_file_with_no_picture_has_nothing_to_encode():
    assert needs_picture_encode(None, ".mp4", VideoOut(short_side=720)) is False
    assert needs_picture_encode(probed(has_video=False, video_codec=None), ".mkv", VideoOut(short_side=720)) is False


# -- can the new file type hold the picture as it is --------------------------------------------------------------------


@pytest.mark.parametrize("codec, src_ext, ext, copies", [
    ("h264", ".avi", ".mp4", True), ("hevc", ".mkv", ".mp4", True), ("mpeg4", ".avi", ".mp4", True), ("vp9", ".webm", ".mp4", True),
    ("wmv2", ".wmv", ".mp4", False), ("msmpeg4v3", ".avi", ".mp4", False), ("prores", ".mov", ".mp4", False), ("vp8", ".webm", ".mp4", False),
    ("prores", ".mp4", ".mov", True), ("hevc", ".mkv", ".mov", True), ("vp9", ".webm", ".mov", False),
    ("vp8", ".mkv", ".webm", True), ("vp9", ".mp4", ".webm", True), ("av1", ".mkv", ".webm", True), ("h264", ".mp4", ".webm", False),
    ("wmv2", ".wmv", ".mkv", True), ("anything_at_all", ".avi", ".mkv", True), ("h264", ".mp4", ".mkv", True),
    # the same kind of file holds its own picture, whatever it is (M4V is an MP4)
    ("wmv2", ".mp4", ".mp4", True), ("wmv2", ".m4v", ".mp4", True), ("wmv2", ".mp4", ".m4v", True), ("h264", ".webm", ".webm", True),
    ("wmv2", ".mov", ".mov", True), (None, ".mp4", ".mp4", True), (None, ".m4v", ".mp4", True),
    # a file type not listed takes anything; FLV takes H.264 and Flash video
    ("wmv2", ".avi", ".avi", True), ("h264", ".mp4", ".flv", True), ("flv1", ".avi", ".flv", True), ("hevc", ".mp4", ".flv", False),
    # the codec not known: only the file types that need no codec to answer
    (None, ".wmv", ".mp4", None), (None, ".wmv", ".webm", None), (None, ".avi", ".mov", None), ("", ".avi", ".webm", None), (None, None, ".mp4", None),
    (None, ".wmv", ".mkv", True), (None, None, ".mkv", True), (None, ".wmv", ".avi", True), (None, ".mp4", ".flv", None),
    ("h264", None, ".mkv", True), ("h264", None, ".mp4", True), ("wmv2", None, ".mp4", False), ("wmv2", None, ".webm", False),
])
def test_a_picture_goes_into_a_file_type_as_it_is_when_that_type_holds_its_codec(codec, src_ext, ext, copies):
    assert picture_copies(codec, src_ext, ext) is copies


# -- how much work making the picture again is -------------------------------------------------------------------------


@pytest.mark.parametrize("width, height, fps, out, factor", [
    (1920, 1080, 30, None, 1.0),                  # the unit: a 1080p picture at 30 frames a second
    (1920, 1080, 24, None, 1.0), (1280, 720, 30, None, 1.0), (640, 360, 60, None, 1.0), (16, 16, 30, None, 1.0),  # (never less than 1)
    (1920, 1080, 60, None, 2.0), (1920, 1080, 50, None, 1.67), (1920, 1080, 120, None, 4.0), (1920, 1080, 240, None, 8.0),
    (2560, 1440, 30, None, 1.78), (2560, 1440, 60, None, 3.56),
    (3840, 2160, 30, None, 4.0), (3840, 2160, 60, None, 8.0), (7680, 4320, 30, None, 16.0),
    (1280, 720, 60, None, 1.0), (1280, 720, 120, None, 1.78), (1280, 720, 240, None, 3.56),
    (3840, 2160, 30, (1920, 1080), 2.0),          # a big picture made smaller: half the original's pixels a second still count
    (3840, 2160, 30, (1280, 720), 2.0), (3840, 2160, 60, (1280, 720), 4.0), (3840, 2160, 30, (2560, 1440), 2.0),
    (1920, 1080, 30, (3840, 2160), 4.0),          # made bigger: the result's pixels count
    (1280, 720, 30, (1920, 1080), 1.0), (1280, 720, 60, (3840, 2160), 8.0), (1920, 1080, 30, (2560, 1440), 1.78),
    (1920, 1080, 30, (1280, 720), 1.0),
    (1600, 1200, 30, (960, 720), 1.0),
])
def test_the_work_is_counted_in_units_of_1080p_at_30_frames_a_second(width, height, fps, out, factor):
    got = convert_factor(width, height, fps, out)
    assert got == factor and isinstance(got, float)


@pytest.mark.parametrize("fps", [None, 0, -1, -30.0, 241, 500, 1e9, float("nan")])
def test_a_frame_rate_outside_zero_to_240_counts_as_30(fps):
    assert convert_factor(3840, 2160, fps) == convert_factor(3840, 2160, 30) == 4.0
    assert convert_factor(1920, 1080, fps, (3840, 2160)) == 4.0
    assert convert_factor(640, 360, fps) == 1.0
    assert convert_factor(3840, 2160, 240) == 32.0


@pytest.mark.parametrize("info, ext, size, work", [
    (dict(), ".mp4", None, (False, 1.0)),                                    # copied: nothing is made, the factor is 1
    (dict(), ".mkv", None, (False, 1.0)), (dict(video_codec="wmv2", ext=".wmv"), ".mkv", None, (False, 1.0)),
    (dict(video_width=3840, video_height=2160, video_fps=60.0), ".mkv", None, (False, 1.0)),
    (dict(), ".webm", None, (True, 1.0)),
    (dict(video_codec="wmv2", ext=".wmv"), ".mp4", None, (True, 1.0)),
    (dict(video_codec="wmv2", ext=".wmv", video_width=3840, video_height=2160), ".mp4", None, (True, 4.0)),
    (dict(video_codec="wmv2", ext=".wmv", video_fps=60.0), ".mp4", None, (True, 2.0)),
    (dict(video_width=3840, video_height=2160, video_fps=60.0), ".webm", None, (True, 8.0)),
    (dict(), ".mp4", 720, (True, 1.0)),
    (dict(), ".mp4", 1080, (False, 1.0)),                                    # the size it already is
    (dict(video_width=3840, video_height=2160), ".mp4", 720, (True, 2.0)),
    (dict(video_width=3840, video_height=2160), ".mp4", 2160, (False, 1.0)),
    (dict(), ".mp4", 2160, (True, 4.0)),
    (dict(video_width=3840, video_height=2160, video_rotation=90), ".webm", None, (True, 4.0)),   # stored sideways: the same work
    (dict(video_width=3840, video_height=2160, video_rotation=90), ".mp4", 720, (True, 2.0)),
    (dict(video_width=3840, video_height=2160, video_rotation=90), ".mp4", 2160, (False, 1.0)),
    (dict(video_width=1080, video_height=1920), ".mp4", 1080, (False, 1.0)),                      # portrait: the short side is the width
    (dict(video_width=1080, video_height=1920), ".mp4", 720, (True, 1.0)),
    (dict(video_width=0, video_height=0), ".webm", None, (True, 1.0)),
])
def test_picture_work_says_whether_the_picture_is_made_again_and_how_much_work_that_is(info, ext, size, work):
    assert picture_work(probed(**info), ext, VideoOut(short_side=size)) == work
    assert needs_picture_encode(probed(**info), ext, VideoOut(short_side=size)) is work[0]


def test_there_is_nothing_to_make_for_no_picture():
    assert picture_work(None, ".mp4", VideoOut(short_side=720)) == (False, 1.0)
    assert picture_work(probed(has_video=False, video_codec=None), ".webm", VideoOut(short_side=720)) == (False, 1.0)


# -- with FFmpeg: the files that come out ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    d = tmp_path_factory.mktemp("conv")
    out = {
        "wmv": make(["-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=3", "-f", "lavfi", "-i", "sine=f=440:d=3",
                     "-c:v", "wmv2", "-c:a", "wmav2"], d / "clip.wmv"),
        "mp4": make(["-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=3", "-f", "lavfi", "-i", "sine=f=440:d=3",
                     "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac"], d / "clip.mp4"),
        "silent": make(["-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=3", "-c:v", "wmv2"], d / "silent.wmv"),
        "two": make(["-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=3", "-f", "lavfi", "-i", "sine=f=440:d=3",
                     "-f", "lavfi", "-i", "sine=f=880:d=3", "-map", "0:v", "-map", "1:a", "-map", "2:a", "-c:v", "wmv2",
                     "-c:a", "wmav2"], d / "two.mkv"),
    }
    return out


@pytest.fixture
def ff():
    return FFmpeg(FF_DIR, threads=2)


@pytest.fixture(scope="module")
def big(tmp_path_factory):
    """A 4K picture at 60 frames a second, a few frames of it (a plain colour), and the same stored sideways and shown upright."""
    d = tmp_path_factory.mktemp("big")
    make(["-f", "lavfi", "-i", "color=c=0x336699:size=3840x2160:rate=60:duration=0.25", "-f", "lavfi", "-i", "sine=f=440:d=0.25", "-c:v", "mpeg4",
          "-q:v", "20", "-c:a", "aac"], d / "big.mp4")
    make(["-display_rotation", "90", "-i", str(d / "big.mp4"), "-c", "copy"], d / "turned.mp4")
    return {"big": d / "big.mp4", "turned": d / "turned.mp4"}


@needs_ff
def test_the_work_of_real_files_is_read_from_what_the_probe_says(sources, big, ff):
    wmv, mp4 = ff.probe(sources["wmv"]), ff.probe(sources["mp4"])
    assert (wmv.video_codec, wmv.video_width, wmv.video_height) == ("wmv2", 320, 240) and wmv.video_fps == pytest.approx(10.0)
    assert picture_work(wmv, ".mkv", VideoOut()) == (False, 1.0) and picture_work(wmv, ".mp4", VideoOut()) == (True, 1.0)
    assert picture_work(wmv, ".webm", VideoOut(short_side=720)) == (True, 1.0)
    assert picture_work(mp4, ".mp4", VideoOut()) == (False, 1.0) and picture_work(mp4, ".webm", VideoOut()) == (True, 1.0)
    assert picture_work(mp4, ".mp4", VideoOut(short_side=240)) == (False, 1.0)
    clip, turned = ff.probe(big["big"]), ff.probe(big["turned"])
    assert (clip.video_codec, clip.video_width, clip.video_height, clip.video_rotation) == ("mpeg4", 3840, 2160, 0) and clip.video_fps == pytest.approx(60.0)
    assert turned.video_rotation % 180 == 90 and (turned.video_width, turned.video_height) == (3840, 2160)
    for info in (clip, turned):
        assert picture_work(info, ".mp4", VideoOut()) == (False, 1.0)                  # MPEG-4 in an MP4: copied
        assert picture_work(info, ".mkv", VideoOut()) == (False, 1.0)
        assert picture_work(info, ".webm", VideoOut()) == (True, 8.0)                  # 4K at 60 frames a second made again: 8 units
        assert picture_work(info, ".mp4", VideoOut(short_side=720)) == (True, 4.0)     # ... made 720p: half the original's pixels count
        assert picture_work(info, ".mkv", VideoOut(short_side=1080)) == (True, 4.0)
        assert picture_work(info, ".mp4", VideoOut(short_side=2160)) == (False, 1.0)   # the size it is
        assert picture_work(info, ".mp4", VideoOut(short_side=1440)) == (True, 4.0)


@needs_ff
def test_a_4k_clip_made_720p_comes_out_720p_and_the_turned_one_stays_upright(big, ff, tmp_path):
    out = ff.mux_video(big["big"], None, tmp_path / "o.mp4", video=VideoOut(short_side=720, encoders=("libx264",), fast=True))
    assert shown(out) == (1280, 720) and kinds(out) == {"video": "h264", "audio": "aac"}
    up = ff.mux_video(big["turned"], None, tmp_path / "t.mp4", video=VideoOut(short_side=720, encoders=("libx264",), fast=True))
    assert shown(up) == (720, 1280)


@needs_ff
def test_a_wmv_into_mp4_gets_h264_and_aac(sources, ff, tmp_path):
    report = {}
    out = ff.mux_video(sources["wmv"], None, tmp_path / "o.mp4", video=X264, report=report)
    assert kinds(out) == {"video": "h264", "audio": "aac"}
    assert report["encoder"] == "libx264" and report["fallback"] is False
    assert ff.probe(out).duration == pytest.approx(3.0, abs=0.4)
    assert shown(out) == (320, 240)  # the size stays


@needs_ff
def test_a_wmv_into_mkv_is_copied_picture_and_sound(sources, ff, tmp_path):
    report = {}
    out = ff.mux_video(sources["wmv"], None, tmp_path / "o.mkv", video=X264, report=report)
    assert kinds(out) == {"video": "wmv2", "audio": "wmav2"}
    assert report["encoder"] == "copy"


@needs_ff
def test_a_wmv_into_webm_gets_vp9_and_opus(sources, ff, tmp_path):
    report = {}
    out = ff.mux_video(sources["wmv"], None, tmp_path / "o.webm", video=VP9, report=report)
    assert kinds(out) == {"video": "vp9", "audio": "opus"}
    assert report["encoder"] == "libvpx-vp9" and report["fallback"] is False
    assert ff.probe(out).duration == pytest.approx(3.0, abs=0.4)


@needs_ff
def test_a_size_resizes_and_keeps_the_shape(sources, ff, tmp_path):
    big = ff.mux_video(sources["mp4"], None, tmp_path / "big.mp4", video=VideoOut(short_side=720, encoders=("libx264",), fast=True))
    assert shown(big) == (960, 720) and kinds(big) == {"video": "h264", "audio": "aac"}
    small = ff.mux_video(sources["wmv"], None, tmp_path / "small.mkv", video=VideoOut(short_side=120, encoders=("libx264",), fast=True))
    assert shown(small) == (160, 120) and kinds(small)["video"] == "h264" and kinds(small)["audio"] == "wmav2"  # (MKV: the sound is copied)
    web = ff.mux_video(sources["mp4"], None, tmp_path / "w.webm", video=VideoOut(short_side=480, encoders=("libvpx-vp9",), fast=True))
    assert shown(web) == (640, 480) and kinds(web) == {"video": "vp9", "audio": "opus"}


@needs_ff
def test_the_same_container_and_size_copy_everything(sources, ff, tmp_path):
    report = {}
    out = ff.mux_video(sources["mp4"], None, tmp_path / "o.mp4", video=VideoOut(short_side=240, encoders=("libx264",), fast=True), report=report)
    assert kinds(out) == {"video": "h264", "audio": "aac"} and report["encoder"] == "copy"


@needs_ff
def test_a_video_with_no_sound_is_copied_without_one_not_refused_by_the_map(sources, ff, tmp_path):
    # (the probe refuses a file with no sound, so the job never gets here; the "0:a:0?" map is what keeps the command itself from failing)
    with pytest.raises(MediaError, match="no audio track"):
        ff.probe(sources["silent"])
    out = ff.mux_video(sources["silent"], None, tmp_path / "s.mkv", video=X264)
    assert kinds(out) == {"video": "wmv2"}


@needs_ff
def test_only_the_first_sound_is_made_again_the_other_is_copied_where_it_fits(sources, ff, tmp_path):
    report = {}
    out = ff.mux_video(sources["two"], None, tmp_path / "o.mp4", video=X264, report=report)
    # wmav2 does not go into MP4: the first track is made as AAC, the second can't be held and is left out (the report says so)
    assert [s["codec_name"] for s in probe_streams(out) if s["codec_type"] == "audio"] == ["aac"]
    assert report["audio_kept"] == 0 and report["audio_dropped"] == 1
    mkv = ff.mux_video(sources["two"], None, tmp_path / "o.mkv", video=X264)
    assert [s["codec_name"] for s in probe_streams(mkv) if s["codec_type"] == "audio"] == ["wmav2", "wmav2"]


@needs_ff
def test_the_converted_sound_is_the_files_own(sources, ff, tmp_path):
    import numpy as np

    out = ff.mux_video(sources["wmv"], None, tmp_path / "o.mp4", video=X264)
    p = ff.probe(out)
    heard = ff.decode(out, p, 16000, 1, 10.0)[:, 0]
    assert len(heard) / 16000 == pytest.approx(3.0, abs=0.4)
    spectrum = np.abs(np.fft.rfft(heard[: 16000 * 2]))
    assert abs(np.argmax(spectrum) / 2 - 440) < 10  # the 440 Hz tone it had, not silence and not another sound


@needs_ff
def test_the_cards_encoder_that_cannot_run_falls_back_to_x264(sources, ff, tmp_path):
    # (the encoder list is the server's own: the card's NVENC first, x264 after it; a name FFmpeg doesn't know stands for a refusing card)
    report = {}
    out = ff.mux_video(sources["wmv"], None, tmp_path / "o.mp4", video=VideoOut(encoders=("no_such_encoder", "libx264"), fast=True), report=report)
    assert report["encoder"] == "libx264" and report["fallback"] is True and kinds(out)["video"] == "h264"
    with pytest.raises(MediaError, match="Could not convert the picture"):
        ff.mux_video(sources["wmv"], None, tmp_path / "p.mp4", video=VideoOut(encoders=("no_such_encoder",)))


@needs_ff
def test_a_new_sound_still_replaces_the_first_track(sources, ff, tmp_path):
    import numpy as np

    clean = ff.write_audio(np.zeros(3 * 48000, np.float32), 48000, tmp_path / "clean", "flac")
    out = ff.mux_video(sources["wmv"], clean, tmp_path / "o.mkv")
    assert kinds(out) == {"video": "wmv2", "audio": "aac"}  # (cleaning a video: the cleaned sound is made in the container's kind, as before)


@needs_ff
def test_a_black_picture_for_a_sound_is_made_with_the_encoders_asked_for(sources, ff, tmp_path):
    import numpy as np

    sound = ff.write_audio(0.2 * np.sin(2 * np.pi * 440 * np.arange(3 * 48000) / 48000).astype(np.float32), 48000, tmp_path / "s", "flac")
    report = {}
    mp4 = ff.make_video(sound, tmp_path / "a.mp4", 3.0, video=VideoOut(encoders=("libx264",)), report=report)
    assert kinds(mp4) == {"video": "h264", "audio": "aac"} and shown(mp4) == (1280, 720)
    assert report == {"encoder": "libx264", "fallback": False}
    # the first encoder that cannot run: the next one is used, and the report says so
    mkv = ff.make_video(sound, tmp_path / "b.mkv", 3.0, video=VideoOut(short_side=480, encoders=("no_such_encoder", "libx264")), report=report)
    assert kinds(mkv)["video"] == "h264" and shown(mkv) == (854, 480) and report == {"encoder": "libx264", "fallback": True}
    web = ff.make_video(sound, tmp_path / "c.webm", 3.0, video=VideoOut(encoders=("libvpx-vp9",)), report=report)
    assert kinds(web) == {"video": "vp9", "audio": "opus"} and report["encoder"] == "libvpx-vp9"
    with pytest.raises(MediaError, match="Could not make a video file"):
        ff.make_video(sound, tmp_path / "d.mp4", 3.0, video=VideoOut(encoders=("no_such_encoder",)))
    # no encoders named (or no video at all): the app's own choice, as before (this build of FFmpeg has no x264 of its own to rely on)
    for video in (None, VideoOut(), VideoOut(short_side=480)):
        out = ff.make_video(sound, tmp_path / "e.mp4", 3.0, video=video, report=report)
        assert kinds(out)["video"] in ("mpeg4", "h264") and report["encoder"] == ffmod.picture_encoders(".mp4")[0]

