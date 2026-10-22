"""A video made in another file type or size on our side (options.video_out, kind "convert" for a conversion alone; docs/api.md):
the request's checks (video_out and the source the client read of the video), the label, the card's encode stage, the encoders and file
name of the runner's finishing step; what a conversion costs: whether the picture is copied (free) or made again and its size factor
(convert_params), the price and its line (convert_cost, convert_line), the estimates, the holds, Automatic settings, the worker deciding
after the probe (the hold grows or is made then, a short balance fails the job uncharged, a copy is free, a sound file has no video to
make), and the runner making real files from small
generated videos (no models are needed for a conversion)."""
from __future__ import annotations

import json
import math
import re
import subprocess

import pytest

from acw import automatic, broker, db, editing, jobs
from acw.billing import ledger, pricing
from acw.worker import main as worker
from acw.worker import runner

from .job_support import count, make_done_job, params_of, rebuilt, roomy_disk, row, set_env, standard, step  # noqa: F401 - fixtures
from .test_automatic_settings import Stages, measurement
from .test_job_pipeline import HOME

pytestmark = pytest.mark.usefixtures("roomy_disk")

FFMPEG = HOME / "bin" / "ffmpeg"
FFPROBE = HOME / "bin" / "ffprobe"
CONVERT = 6_000  # micros a minute of 1080p at 30 fps


def per(micros_per_minute, seconds):
    """A price per minute for `seconds`, counted by the second rounded up (pricing.cost_micros)."""
    return math.ceil(micros_per_minute * math.ceil(seconds) / 60)


def src(codec="wmv2", width=1920, height=1080, fps=30):
    """What a client read of the video (video_out.source); None leaves a key out."""
    return {k: v for k, v in (("codec", codec), ("width", width), ("height", height), ("fps", fps)) if v is not None}


def vo(fmt="mp4", size=None, source=None):
    return {"video_out": {"format": fmt, "size": size, **({"source": source} if source is not None else {})}}


def funded(signed_in, micros=5_000_000):
    uid, h = signed_in
    ledger.credit(uid, micros, "grant", "t", "g")
    return uid, h


def held(uid):
    return ledger.balance(uid)["held_micros"]


def make(uid, kind="convert", seconds=600.0, filename="talk.wmv", options=None, size=1000, **kw):
    """A job made through jobs.create_job: kind convert (no tiers), or a denoise with Denoising unless told otherwise."""
    tiers = {} if kind == "convert" else {"denoise_tier": "standard"}
    args = dict(kind=kind, denoise_tier=None, transcribe_tier=None, filename=filename, size_bytes=size, claimed_seconds=seconds,
                options=options if options is not None else {})
    return jobs.create_job(uid, **{**args, **tiers, **kw})


# --------------------------------------------------------------------------------------------------- the request

@pytest.mark.parametrize("fmt", ["mp4", "mov", "mkv", "webm"])
@pytest.mark.parametrize("size", [None, 480, 720, 1080, 1440, 2160])
def test_a_file_type_and_a_size_are_taken(fmt, size):
    assert jobs.video_out_of({"video_out": {"format": fmt, "size": size}}) == {"format": fmt, "size": size}
    assert jobs.validate_options("convert", None, None, vo(fmt, size))["video_out"] == {"format": fmt, "size": size}


def test_the_size_may_be_left_out_and_no_video_out_is_none():
    assert jobs.video_out_of({"video_out": {"format": "mkv"}}) == {"format": "mkv", "size": None}
    assert jobs.video_out_of({}) is None and jobs.video_out_of({"video_out": None}) is None
    assert jobs.validate_options("denoise", "standard", None, {})["video_out"] is None


@pytest.mark.parametrize("bad, message", [
    ("mp4", "video_out must be {format, size"),
    (["mp4"], "video_out must be {format, size"),
    ({}, "video_out.format must be one of mp4, mov, mkv, webm"),
    ({"size": 720}, "video_out.format must be one of"),
    ({"format": "avi", "size": None}, "video_out.format must be one of"),
    ({"format": "MP4", "size": None}, "video_out.format must be one of"),
    ({"format": "mp4", "size": 999}, "video_out.size must be one of 480, 720, 1080, 1440, 2160, or null"),
    ({"format": "mp4", "size": "720"}, "video_out.size must be one of"),
    ({"format": "mp4", "size": True}, "video_out.size must be one of"),
    ({"format": "mp4", "size": 0}, "video_out.size must be one of"),
    ({"format": "mp4", "size": 720, "bitrate": 5}, "video_out must be {format, size"),
    ({"format": "mp4", "fps": 30}, "video_out must be {format, size"),
])
def test_a_wrong_video_out_is_refused_in_plain_words(bad, message):
    with pytest.raises(jobs.JobError, match=message.replace("{", r"\{").replace("}", r"\}").replace("(", r"\(")):
        jobs.validate_options("denoise", "standard", None, {"video_out": bad})


def test_video_out_needs_the_video_result():
    with pytest.raises(jobs.JobError, match="video_out needs output_video"):
        jobs.validate_options("denoise", "standard", None, {"video_out": {"format": "mp4", "size": None}, "output_video": False})
    assert jobs.validate_options("denoise", "standard", None, {**vo(), "output_video": True})["video_out"]["format"] == "mp4"


def test_a_conversion_needs_video_out_and_has_no_tiers(client, signed_in):
    _, h = signed_in
    with pytest.raises(jobs.JobError, match="needs options.video_out"):
        jobs.validate_options("convert", None, None, {})
    for tiers in ({"denoise_tier": "standard"}, {"denoise_tier": "off"}, {"transcribe_tier": "local"}):
        with pytest.raises(jobs.JobError, match="no denoiseTier or transcribeTier"):
            jobs.validate_options("convert", tiers.get("denoise_tier"), tiers.get("transcribe_tier"), vo())
    base = {"kind": "convert", "durationSeconds": 60, "options": vo("mkv")}
    assert client.post("/api/jobs/estimate", json=base, headers=h).status_code == 200
    assert client.post("/api/jobs/estimate", json={**base, "denoiseTier": "standard"}, headers=h).status_code == 400
    assert client.post("/api/jobs/estimate", json={**base, "options": {}}, headers=h).status_code == 400
    assert "convert" in jobs.KINDS


def test_a_conversion_has_no_timeline(client, signed_in):
    uid, h = funded(signed_in)
    tl = {"version": 1, "seconds": 60, "layers": {"noise": [{"start": 0, "end": 60, "strength_db": 40, "remover": "light"}]}}
    r = client.post("/api/jobs", json={"kind": "convert", "filename": "a.wmv", "sizeBytes": 1000, "durationSeconds": 60, "options": vo("mkv"),
                                       "timeline": tl}, headers=h)
    assert r.status_code == 400 and "has no timeline" in r.json()["detail"]
    assert count("jobs") == 0 and held(uid) == 0


def test_unknown_kinds_are_still_refused(client, signed_in):
    _, h = signed_in
    r = client.post("/api/jobs/estimate", json={"kind": "convertx", "durationSeconds": 60, "options": vo()}, headers=h)
    assert r.status_code == 400 and "unknown job type" in r.json()["detail"]


# ------------------------------------------------------------------------------------- what the client read of the video

def checked_source(source):
    return jobs.video_out_of({"video_out": {"format": "mp4", "size": None, "source": source}}).get("source")


@pytest.mark.parametrize("source, want", [
    ({"codec": "h264", "width": 1920, "height": 1080, "fps": 30}, {"codec": "h264", "width": 1920, "height": 1080, "fps": 30.0}),
    ({"codec": "wmv2"}, {"codec": "wmv2"}),
    ({"codec": "pcm_s16le_planar2"}, {"codec": "pcm_s16le_planar2"}),
    ({"width": 640, "height": 480}, {"width": 640, "height": 480}),
    ({"fps": 29.97}, {"fps": 29.97}),
    ({"width": 1920.0, "height": 1080.0, "fps": 25}, {"width": 1920, "height": 1080, "fps": 25.0}),
    ({"width": 16, "height": 16384, "fps": 1}, {"width": 16, "height": 16384, "fps": 1.0}),   # the ends of the ranges are in
    ({"width": 16384, "height": 16, "fps": 240}, {"width": 16384, "height": 16, "fps": 240.0}),
    ({}, {}),
    ({"codec": None, "width": None, "height": None, "fps": None}, {}),                         # (null is "not said")
])
def test_what_the_client_read_of_the_video_is_taken_as_numbers(source, want):
    got = checked_source(source)
    assert got == want
    assert all(type(got[k]) is int for k in ("width", "height") if k in got) and all(type(got[k]) is float for k in ("fps",) if k in got)


def test_no_source_means_none_is_kept():
    assert jobs.video_out_of({"video_out": {"format": "mp4", "size": None}}) == {"format": "mp4", "size": None}
    assert jobs.video_out_of({"video_out": {"format": "mp4", "size": None, "source": None}}) == {"format": "mp4", "size": None}
    opts = jobs.validate_options("convert", None, None, vo("webm", 720, src("h264")))
    assert opts["video_out"] == {"format": "webm", "size": 720, "source": {"codec": "h264", "width": 1920, "height": 1080, "fps": 30.0}}


CODEC = "video_out.source.codec must be FFmpeg's codec name, like h264"


@pytest.mark.parametrize("codec", ["H264", "h 264", "", "h264;ls", "x" * 33, "h.264", "../etc", "é", "h264\n", "\nh264", "h264 ", 264, True, ["h264"], {"a": 1}])
def test_a_codec_name_is_a_plain_ffmpeg_name(codec):
    with pytest.raises(jobs.JobError) as e:
        checked_source({"codec": codec})
    assert str(e.value) == CODEC


@pytest.mark.parametrize("key, bad, lo, hi", [
    ("width", [15, 16385, 0, -1, True, False, "1920", [1920], None.__class__, float("nan"), float("inf"), -float("inf"), 15.9, 1e9], 16, 16384),
    ("height", [15, 16385, 0, -1, True, "1080", float("nan"), float("inf")], 16, 16384),
    ("fps", [0, 0.5, 0.99, 241, 1000, -30, True, "30", [30], float("nan"), float("inf")], 1, 240),
])
def test_a_size_or_frame_rate_outside_its_range_is_refused(key, bad, lo, hi):
    others = {"width": 1920, "height": 1080, "fps": 30}
    others.pop(key)
    for value in bad:
        if value is None.__class__:
            continue
        with pytest.raises(jobs.JobError) as e:
            checked_source({**others, key: value})
        assert str(e.value) == f"video_out.source.{key} must be a number from {lo} to {hi}", (key, value)


def test_a_size_needs_both_numbers_or_neither():
    for one in ({"width": 1920}, {"height": 1080}, {"codec": "h264", "width": 1920, "fps": 30}, {"height": 16, "fps": 30}):
        with pytest.raises(jobs.JobError, match=r"video_out\.source needs both width and height, or neither"):
            checked_source(one)
    assert checked_source({"codec": "h264", "fps": 30}) == {"codec": "h264", "fps": 30.0}


@pytest.mark.parametrize("bad", ["h264", ["h264"], 5, True, [], {"bitrate": 5}, {"codec": "h264", "extra": 1}, {"Codec": "h264"}, {"size": 720}])
def test_the_source_is_a_small_object_with_its_four_keys_only(bad):
    with pytest.raises(jobs.JobError, match=re.escape("video_out.source must be {codec?, width?, height?, fps?}")):
        checked_source(bad)


def test_the_other_keys_of_video_out_are_still_refused_and_the_source_is_checked_through_the_api(client, signed_in):
    _, h = signed_in
    body = {"kind": "convert", "durationSeconds": 60}
    ok = client.post("/api/jobs/estimate", json={**body, "options": vo("mp4", 720, src())}, headers=h)
    assert ok.status_code == 200, ok.text
    for bad in ({"format": "mp4", "size": None, "bitrate": 5}, {"format": "mp4", "size": None, "source": {"codec": "H.264"}},
                {"format": "mp4", "size": None, "source": {"width": 1920}}, {"format": "mp4", "size": None, "source": {"fps": 0}}):
        r = client.post("/api/jobs/estimate", json={**body, "options": {"video_out": bad}}, headers=h)
        assert r.status_code == 400 and r.json()["detail"].startswith("video_out"), (bad, r.text)


# ------------------------------------------------------------ a sound asked for as a video gets a plain black picture, free

def test_a_sound_file_asked_for_as_a_video_is_taken_and_the_video_is_free(client, signed_in, monkeypatch):
    set_env(monkeypatch, ACW_MAX_ACTIVE_JOBS=500)  # (a hundred and more jobs are made here)
    uid, h = funded(signed_in, 50_000_000)
    assert jobs.AUDIO_EXTS <= jobs.ALLOWED_EXTS and not jobs.AUDIO_EXTS & {".mp4", ".mkv", ".avi", ".wmv", ".webm", ".mov", ".m4v", ".mxf", ".flv"}
    assert {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".wma", ".aiff", ".mka", ".aac", ".ac3"} <= jobs.AUDIO_EXTS
    asked = (vo("mkv"), vo("webm", 2160, src("h264", 3840, 2160, 60)))  # (a size and a source that would cost 8 times the work for a picture)
    for ext in sorted(jobs.AUDIO_EXTS):
        for options in asked:
            j = make(uid, filename=f"talk{ext}", options=options)  # a conversion alone
            p = params_of(j["id"])
            assert (p["picture_encode"], p["convert_factor"], p["convert_rate"]) == (False, 1.0, CONVERT), (ext, options)
            assert j["estCostMicros"] == 0 and row(j["id"])["hold_id"] is None and jobs.convert_cost(p, 600) == 0
            d = make(uid, kind="denoise", filename=f"talk{ext}", options=options)  # with cleaning: held for the cleaning alone
            assert d["estCostMicros"] == standard(600) and params_of(d["id"])["picture_encode"] is False
            assert jobs.label_of("denoise", params_of(d["id"])) == "Denoising + Leveling"  # (no "+ Video conversion": nothing is made again)
    assert held(uid) == count("jobs", "kind = 'denoise'") * standard(600)
    # an uppercase ending counts the same, and through the API
    assert params_of(make(uid, filename="TALK.WAV", options=vo("mp4", 720))["id"])["picture_encode"] is False
    for kind, extra in (("convert", {}), ("denoise", {"denoiseTier": "standard"})):
        r = client.post("/api/jobs", json={"kind": kind, "filename": "talk.mp3", "sizeBytes": 1000, "durationSeconds": 60, "options": vo("mp4", 720), **extra}, headers=h)
        assert r.status_code == 200, r.text
        assert r.json()["estCostMicros"] == (0 if kind == "convert" else standard(60))


def test_a_video_ending_is_judged_by_the_file_when_it_is_read_not_by_its_name(signed_in):
    uid, _ = funded(signed_in)
    j = make(uid, filename="talk.mkv", options=vo("mp4", 720))  # (it may hold only sound: the probe will say)
    assert params_of(j["id"])["picture_encode"] is True and j["estCostMicros"] == 60_000
    assert not hasattr(jobs, "NO_PICTURE")  # (no refusal any more)


# ------------------------------------------------------------------------------------------------ the label, the card

def test_what_a_conversion_is_called():
    assert jobs.label_of("convert", {}) == "Video conversion"
    assert jobs.label_of("convert", {"options": vo("mkv")}) == "Video conversion"
    assert jobs.job_label("convert", None, None) == "Video conversion"
    clean = {"denoise_tier": "standard", "transcribe_tier": None, "rates": {"denoise": 9_000, "leveling": 1_000}}
    assert jobs.label_of("denoise", clean) == "Denoising + Leveling"  # (a cleaning job keeps its name)


def test_the_job_page_names_a_conversion(client, signed_in):
    uid, h = funded(signed_in)
    jid = make(uid, options=vo("mkv"))["id"]
    shown = client.get(f"/api/jobs/{jid}", headers=h).json()
    assert shown["kind"] == "convert" and shown["label"] == "Video conversion"
    assert shown["options"]["video_out"] == {"format": "mkv", "size": None} and shown["denoiseTier"] is None and shown["transcribeTier"] is None
    assert "edit" not in shown  # (nothing of the per-part editor is for a conversion)
    assert client.get("/api/jobs", headers=h).status_code == 200



def stage_row(**p):
    """A job row as the broker reads it (a conversion unless a kind is told: the other kinds are made with the timeline their options stand for)."""
    nxt = p.pop("next_stage", None)
    kind = p.pop("kind", "convert")
    tier, transcribe, options = p.pop("tier", None), p.pop("transcribe", None), p.pop("options", {})
    params = {"denoise_tier": None, "transcribe_tier": None, **p}
    if kind == "convert":
        params["options"] = options
    else:
        raw, rest = editing.timeline_from_options(kind, tier, transcribe, jobs.validate_options(kind, tier, transcribe, options), 600.0)
        params.update(options=rest, edit={"first_run": True, "timeline": raw})
    return {"kind": kind, "next_stage": nxt, "params_json": json.dumps(params)}


@pytest.mark.parametrize("row_, stages, card_only", [
    (stage_row(options=vo("mp4", 720), picture_encode=True), ["encode"], False),          # the picture made again: the card's encoder
    (stage_row(options=vo("mp4"), picture_encode=None), ["encode"], False),               # (it may be: the probe will say)
    (stage_row(options=vo("mkv"), picture_encode=False), [], False),                      # copied: nothing for the card
    (stage_row(options=vo("mp4", 720), picture_encode=True, next_stage="finalize"), [], False),  # (the finishing step is not given to the card)
    (stage_row(options={}), [], False),
    (stage_row(kind="denoise", tier="standard", options={"noise": "heavy", **vo("mp4", 720)}, picture_encode=True), ["heavy", "encode"], False),
    (stage_row(kind="denoise", tier="standard", options={"noise": "light", **vo("mp4", 720)}, picture_encode=True), ["encode"], False),
    (stage_row(kind="denoise", tier="standard", options={"noise": "heavy", **vo("mp4", 720)}, picture_encode=False), ["heavy"], False),
    (stage_row(kind="denoise", tier="standard", options={"noise": "heavy", "echo_reference_track": True, **vo("webm")}, picture_encode=None), [], False),  # (echo removal: the whole job, its picture too, stays with the CPU workers)
    (stage_row(kind="denoise", tier="rebuild", options={"noise": "light", "echo_reference_track": True, **vo("mp4", 720)}, picture_encode=True), ["rebuild", "encode"], True),
    (stage_row(kind="denoise", tier="rebuild", options={"noise": "light", **vo("mp4", 720)}, picture_encode=True), ["rebuild", "encode"], True),
    (stage_row(kind="denoise_transcribe", tier="standard", transcribe="local", options={"noise": "light", **vo("mp4", 720)}, picture_encode=True),
     ["transcribe", "encode"], False),
])
def test_which_stages_go_to_the_card(row_, stages, card_only):
    assert broker.gpu_stages(row_) == stages
    assert broker.gpu_only(row_) is card_only



def test_the_encoders_of_a_video_made_on_our_side():
    assert runner._encoders({"gpu_devices": ["/dev/nvidia0"]}, ".mp4") == ("h264_nvenc", "libx264")  # the card's NVENC, x264 if it refuses
    assert runner._encoders({"gpu_devices": ["/dev/nvidia0"]}, ".mkv") == ("h264_nvenc", "libx264")
    assert runner._encoders({}, ".mp4") == runner._encoders({"gpu_devices": []}, ".mov") == ("libx264",)  # the CPU workers: x264
    assert runner._encoders({"gpu_devices": ["/dev/nvidia0"]}, ".webm") == runner._encoders({}, ".webm") == ("libvpx-vp9",)  # (the card has no VP9)



# ------------------------------------------------------------------------------ what a job stores: copied or made again, how big

def cp(fmt, size, source, ext=None):
    opts = jobs.validate_options("convert", None, None, vo(fmt, size, source))
    return jobs.convert_params(opts, ext)


@pytest.mark.parametrize("fmt, size, source, ext, encode, factor", [
    # the picture is copied when the new file type holds its codec (MKV holds any; the same kind of file holds its own), else made again
    ("mp4", None, src("wmv2", 320, 240, 10), ".wmv", True, 1.0),
    ("mp4", None, src("h264"), ".mp4", False, 1.0),                       # the same container
    ("mp4", None, src("h264"), ".m4v", False, 1.0),                       # (M4V and MP4 are one kind of file)
    ("mp4", None, src("wmv2"), ".m4v", False, 1.0),                       # ... whatever the picture is
    ("mp4", None, src("wmv2"), ".mp4", False, 1.0),
    ("mp4", None, src("h264"), ".avi", False, 1.0),                       # MP4 holds H.264, HEVC, MPEG-4, AV1, VP9, MPEG-2, MJPEG
    ("mp4", None, src("mpeg4"), ".avi", False, 1.0),
    ("mp4", None, src("vp9"), ".webm", False, 1.0),
    ("mp4", None, src("msmpeg4v3"), ".avi", True, 1.0),
    ("mp4", None, src("prores"), ".mov", True, 1.0),
    ("mov", None, src("hevc"), ".mkv", False, 1.0),
    ("mov", None, src("vp9"), ".webm", True, 1.0),
    ("mov", None, src("prores"), ".mp4", False, 1.0),
    ("webm", None, src("h264"), ".mp4", True, 1.0),                       # WebM holds VP8, VP9 and AV1 only
    ("webm", None, src("vp8"), ".mkv", False, 1.0),
    ("webm", None, src("av1"), ".mp4", False, 1.0),
    ("webm", None, src("wmv2"), ".webm", False, 1.0),
    ("mkv", None, src("wmv2"), ".wmv", False, 1.0),                       # anything goes into MKV
    ("mkv", None, src("h264"), ".mp4", False, 1.0),
    ("mkv", None, src("some_odd_codec"), ".avi", False, 1.0),
    # no codec said: a change of type alone can't be told (None) unless the container is the same or MKV
    ("mp4", None, None, ".wmv", None, 1.0),
    ("mp4", None, {"width": 1280, "height": 720, "fps": 30}, ".avi", None, 1.0),
    ("webm", None, src(None, 1920, 1080), ".mp4", None, 1.0),
    ("mov", None, None, None, None, 1.0),
    ("mp4", None, None, ".mp4", False, 1.0),
    ("mkv", None, None, ".wmv", False, 1.0),
    # a new size makes the picture again (an unknown codec does not matter)
    ("mp4", 720, src("h264", 1920, 1080, 30), ".mp4", True, 1.0),
    ("mkv", 720, src("wmv2", 1920, 1080, 30), ".wmv", True, 1.0),
    ("mp4", 480, src("h264", 1280, 720, 30), ".mp4", True, 1.0),
    ("mp4", 1080, src("h264", 1280, 720, 30), ".mp4", True, 1.0),         # (made bigger: only stretched, but made again)
    ("mp4", 720, None, ".wmv", True, 1.0),
    ("mp4", 720, src(None, 1280, 800), ".mov", True, 1.0),
    # a size equal to the source's short side is no resize: copied where the type holds the codec, made again where it does not
    ("mp4", 1080, src("h264", 1920, 1080, 30), ".mp4", False, 1.0),
    ("mp4", 1080, src("h264", 1080, 1920, 30), ".mp4", False, 1.0),       # (portrait: the short side is the width)
    ("mp4", 720, src("h264", 1280, 720, 30), ".avi", False, 1.0),
    ("mp4", 1080, src("wmv2", 1920, 1080, 30), ".wmv", True, 1.0),
    ("mkv", 1080, src("wmv2", 1920, 1080, 30), ".wmv", False, 1.0),
    ("mp4", 480, src("h264", 640, 480, 30), ".mp4", False, 1.0),
    ("mp4", 480, src("h264", 854, 480, 30), ".mp4", False, 1.0),
    ("webm", 720, src("h264", 1280, 720, 30), ".mp4", True, 1.0),
    # the size factor, in units of making a 1080p picture at 30 frames a second: the result's pixels a second or half the original's
    ("mp4", None, src("wmv2", 1920, 1080, 30), ".wmv", True, 1.0),
    ("mp4", None, src("wmv2", 1920, 1080, 24), ".wmv", True, 1.0),        # (never less than 1)
    ("mp4", None, src("wmv2", 640, 360, 30), ".wmv", True, 1.0),
    ("mp4", None, src("wmv2", 1920, 1080, 60), ".wmv", True, 2.0),        # 1080p60
    ("mp4", None, src("wmv2", 1920, 1080, 50), ".wmv", True, 1.67),
    ("mp4", None, src("wmv2", 1920, 1080, 120), ".wmv", True, 4.0),
    ("mp4", None, src("wmv2", 2560, 1440, 30), ".wmv", True, 1.78),       # 1440p
    ("mp4", None, src("wmv2", 3840, 2160, 30), ".wmv", True, 4.0),        # 4K made again, the same size
    ("mp4", None, src("wmv2", 3840, 2160, 60), ".wmv", True, 8.0),
    ("mp4", 2160, src("h264", 1920, 1080, 30), ".mp4", True, 4.0),        # 1080p made 4K
    ("mp4", 720, src("h264", 3840, 2160, 30), ".mp4", True, 2.0),         # 4K made 720p: half the original's pixels count
    ("mp4", 1080, src("h264", 3840, 2160, 30), ".mp4", True, 2.0),
    ("mp4", 1440, src("h264", 3840, 2160, 30), ".mp4", True, 2.0),
    ("mkv", 720, src("h264", 3840, 2160, 60), ".mp4", True, 4.0),
    ("mp4", 2160, src("h264", 1280, 720, 60), ".mp4", True, 8.0),
    ("mp4", 720, src("h264", 1600, 1200, 30), ".mp4", True, 1.0),         # 4:3 keeps its shape
    ("mp4", 1080, src("wmv2", 1920, 1080, 60), ".wmv", True, 2.0),        # no resize, made again (the codec): the factor still counts
    # a copy has a factor too (the price line says it) but is not charged for it
    ("mkv", None, src("h264", 3840, 2160, 60), ".mp4", False, 8.0),
    # a size with no dimensions said: 16:9 at that size, at the frame rate if said
    ("mp4", 2160, src("wmv2", None, None, None), ".wmv", True, 4.0),
    ("mp4", 1440, src("wmv2", None, None, None), ".wmv", True, 1.78),
    ("mp4", 1080, src("wmv2", None, None, None), ".wmv", True, 1.0),
    ("mp4", 720, None, ".wmv", True, 1.0),
    ("mkv", 480, None, None, True, 1.0),
    ("mp4", 1080, {"fps": 60}, ".wmv", True, 2.0),
    ("mp4", 2160, {"fps": 24}, ".wmv", True, 4.0 * 24 / 30),
    # nothing said, no size: 1
    ("mp4", None, src("wmv2", None, None, None), ".wmv", True, 1.0),
    ("mp4", None, src("wmv2", None, None, 60), ".wmv", True, 1.0),
])
def test_the_picture_is_copied_or_made_again_and_counts_by_its_size(fmt, size, source, ext, encode, factor):
    got = cp(fmt, size, source, ext)
    assert got == {"convert_rate": CONVERT, "picture_encode": encode, "convert_factor": factor}
    assert type(got["convert_factor"]) is float


def test_a_rate_is_pinned_with_the_job_and_nothing_is_stored_without_video_out():
    assert jobs.convert_params(jobs.validate_options("denoise", "standard", None, {})) == {}
    assert jobs.convert_params(jobs.validate_options("denoise", "standard", None, {}), ".wmv") == {}
    assert jobs.convert_params({"video_out": None}) == {} and jobs.convert_params({}) == {}
    assert cp("mp4", None, None, ".wmv")["convert_rate"] == pricing.rate("convert", "standard").micros_per_minute == CONVERT


def test_a_frame_rate_outside_one_to_240_counts_as_30():
    """(a request can't say one: the checks refuse it; a value that gets here some other way counts as 30 frames a second)"""
    for fps in (500, 0, -5, None, float("nan"), 240.5):
        opts = {"video_out": {"format": "mp4", "size": None, "source": {"codec": "wmv2", "width": 3840, "height": 2160, "fps": fps}}}
        assert jobs.convert_params(opts, ".wmv")["convert_factor"] == 4.0, fps
    opts = {"video_out": {"format": "mp4", "size": None, "source": {"codec": "wmv2", "width": 3840, "height": 2160, "fps": 240}}}
    assert jobs.convert_params(opts, ".wmv")["convert_factor"] == 32.0  # (240 fps is a real rate)
    opts = {"video_out": {"format": "mp4", "size": 2160, "source": {"fps": 600}}}
    assert jobs.convert_params(opts, ".wmv")["convert_factor"] == 4.0


# -------------------------------------------------------------------------------------------------- the price and its line

def conv(encode, factor=None, rate=CONVERT):
    p = {"convert_rate": rate, "picture_encode": encode}
    if factor is not None:
        p["convert_factor"] = factor
    return p


@pytest.mark.parametrize("params, seconds, per_minute, want", [
    (conv(True, 1.0), 600, 6_000, 60_000),
    (conv(True, 2.0), 600, 12_000, 120_000),
    (conv(True, 4.0), 600, 24_000, 240_000),
    (conv(True, 1.5), 600, 9_000, 90_000),
    (conv(True, 8.0), 90.5, 48_000, per(48_000, 91)),                    # (by the second, rounded up)
    (conv(True, 2.0), 61.2, 12_000, per(12_000, 62)),
    (conv(True, 4.0), 1, 24_000, 400),
    (conv(True, 2.0), 0, 12_000, 0),
    (conv(True), 600, 6_000, 60_000),                                    # (a job from before the factor: 1)
    (conv(True, 0), 600, 6_000, 60_000),
    (conv(True, 2.0, rate=12_000), 600, 24_000, 240_000),                # (the rate the job was made with, times its factor)
])
def test_the_conversion_costs_the_pinned_rate_times_the_factor_by_the_second(params, seconds, per_minute, want):
    assert jobs.convert_cost(params, seconds) == want
    line = jobs.convert_line(params, seconds)
    assert line == {"item": "convert", "label": "Video conversion", "quantity": 1, "usdPerMinute": per_minute / 1e6,
                    "factor": float(params.get("convert_factor") or 1.0), "micros": want}


@pytest.mark.parametrize("params", [conv(False, 4.0), conv(None, 4.0), conv(False), conv(None), {"picture_encode": True}, {"picture_encode": True, "convert_factor": 4.0}, {}])
def test_a_copy_an_unknown_picture_and_a_job_without_the_rate_cost_nothing(params):
    assert jobs.convert_cost(params, 600) == 0


def test_the_lines_say_copied_or_maybe_and_only_a_sure_price_adds_to_the_total():
    assert jobs.convert_line({}, 600) is None and jobs.convert_line({"picture_encode": True, "convert_factor": 2.0}, 600) is None
    copied = jobs.convert_line(conv(False, 4.0), 600)
    assert copied == {"item": "convert", "label": "Video conversion", "quantity": 1, "usdPerMinute": 0.024, "factor": 4.0, "micros": 0, "copied": True}
    maybe = jobs.convert_line(conv(None, 4.0), 600)  # (what it would cost, if the picture has to be made again; not in the total)
    assert maybe == {"item": "convert", "label": "Video conversion", "quantity": 1, "usdPerMinute": 0.024, "factor": 4.0, "micros": 240_000, "maybe": True}
    sure = jobs.convert_line(conv(True, 4.0), 600)
    assert sure["micros"] == 240_000 and "maybe" not in sure and "copied" not in sure
    for encode in (True, False, None):
        params = conv(encode, 4.0)
        line = jobs.convert_line(params, 600)
        counted = 0 if line.get("maybe") else line["micros"]
        assert counted == jobs.convert_cost(params, 600), encode


@pytest.mark.parametrize("factor, per_minute", [(1.1, 6_600), (1.12, 6_720), (1.35, 8_100), (2.2, 13_200), (1.09, 6_540), (2.18, 13_080), (1.78, 10_680), (3.56, 21_360)])
def test_the_rate_a_minute_is_exact_for_a_factor_with_two_decimals(factor, per_minute):
    assert jobs.convert_line(conv(True, factor), 60)["usdPerMinute"] == per_minute / 1e6
    assert jobs.convert_cost(conv(True, factor), 60) == per_minute


def test_the_rate_is_exact_for_every_factor_the_probe_can_store():
    """(whole micros: the factor has two decimals and the rate is a whole number, so the product is exact)"""
    from fractions import Fraction

    for hundredths in range(100, 6401):
        factor = hundredths / 100
        want = math.ceil(Fraction(CONVERT) * hundredths / 100)
        assert jobs.convert_cost(conv(True, factor), 60) == want, factor


# ------------------------------------------------------------------------------------------------------ estimates

def estimate(fmt, size=None, source=None, seconds=600, kind="convert", tier=None, extra=None, timeline=None):
    return jobs.estimate(kind, tier, None, seconds, {**vo(fmt, size, source), **(extra or {})}, timeline)


def lines_add_up(e):
    """The lines that are not "maybe" add up to the total (every estimate)."""
    return e["estCostMicros"] == sum(line["micros"] for line in e["lines"] if not line.get("maybe"))


def test_the_estimate_of_a_conversion_alone(client, signed_in):
    _, h = signed_in

    def ask(fmt, size=None, source=None, seconds=600):
        r = client.post("/api/jobs/estimate", json={"kind": "convert", "durationSeconds": seconds, "options": vo(fmt, size, source)}, headers=h)
        assert r.status_code == 200, r.text
        return r.json()

    a = ask("mp4", None, src("wmv2"))  # made again, 1080p30: known up front now
    assert (a["estCostMicros"], a["estCostUsd"]) == (60_000, 0.06) and a["lines"] == [conv_line(60_000, 1.0)]
    assert a["options"]["video_out"] == {"format": "mp4", "size": None, "source": {"codec": "wmv2", "width": 1920, "height": 1080, "fps": 30.0}}
    b = ask("mp4", 720, src("wmv2", 3840, 2160, 30))  # 4K made 720p counts twice
    assert b["estCostMicros"] == 120_000 and b["lines"] == [conv_line(120_000, 2.0)]
    c = ask("webm", None, src("h264", 3840, 2160, 60))  # 4K at 60 fps counts eight times
    assert c["estCostMicros"] == 480_000 and c["lines"] == [conv_line(480_000, 8.0)]
    d = ask("mkv", None, src("wmv2", 3840, 2160, 60))  # copied, free (the line says so; the factor is only informative)
    assert d["estCostMicros"] == 0 and d["lines"] == [{**conv_line(0, 8.0), "copied": True}]
    e = ask("mp4", None, src("h264"))  # (an MP4 holds H.264)
    assert e["estCostMicros"] == 0 and e["lines"][0]["copied"] is True
    f = ask("mp4", None, {"width": 3840, "height": 2160, "fps": 30})  # the codec not said: maybe, not in the total
    assert f["estCostMicros"] == 0 and f["lines"] == [{**conv_line(240_000, 4.0), "maybe": True}]
    g = ask("mp4", None)  # nothing said
    assert g["estCostMicros"] == 0 and g["lines"] == [{**conv_line(60_000, 1.0), "maybe": True}]
    h2 = ask("mp4", 2160)  # a size, nothing else said: 16:9 at that size
    assert h2["estCostMicros"] == 240_000 and h2["lines"] == [conv_line(240_000, 4.0)]
    i = ask("mp4", 720, src("wmv2"), seconds=61.2)
    assert i["estCostMicros"] == per(CONVERT, 62)
    for e in (a, b, c, d, e, f, g, h2, i):
        assert lines_add_up(e)


def conv_line(micros, factor):
    return {"item": "convert", "label": "Video conversion", "quantity": 1, "usdPerMinute": math.ceil(CONVERT * factor) / 1e6, "factor": factor, "micros": micros}


def test_the_conversion_goes_on_top_of_the_other_work():
    plain = estimate("mp4", kind="denoise", tier="standard")
    assert plain["estCostMicros"] == standard(600)
    # the estimate has no file name: the picture is judged by its codec alone (MP4 holds H.264)
    made = estimate("mp4", 720, src("h264", 3840, 2160, 30), kind="denoise", tier="standard")
    assert made["estCostMicros"] == standard(600) + 120_000 and [x["item"] for x in made["lines"]] == ["denoise", "leveling", "convert"]
    assert made["lines"][-1] == conv_line(120_000, 2.0) and lines_add_up(made)
    copied = estimate("mkv", None, src("wmv2"), kind="denoise", tier="standard")
    assert copied["estCostMicros"] == standard(600) and copied["lines"][-1]["copied"] is True and lines_add_up(copied)
    maybe = estimate("mp4", None, None, kind="denoise", tier="standard")
    assert maybe["estCostMicros"] == standard(600) and maybe["lines"][-1]["maybe"] is True and lines_add_up(maybe)
    both = jobs.estimate("denoise_transcribe", "standard", "local", 600, vo("webm", None, src("h264", 1920, 1080, 60)))
    base = jobs.estimate("denoise_transcribe", "standard", "local", 600, {})
    assert both["estCostMicros"] == base["estCostMicros"] + 120_000 and both["lines"][-1] == conv_line(120_000, 2.0) and lines_add_up(both)


def test_the_estimate_of_a_job_made_with_a_timeline_has_the_conversion_too(client, signed_in):
    _, h = signed_in
    tl = {"version": 1, "seconds": 600, "layers": {"noise": [{"start": 0, "end": 300, "strength_db": 40, "remover": "light"}]}}

    def ask(options):
        r = client.post("/api/jobs/estimate", json={"kind": "denoise", "durationSeconds": 600, "options": options, "timeline": tl}, headers=h)
        assert r.status_code == 200, r.text
        return r.json()

    plain = ask({})
    made = ask(vo("mp4", 1080, src("h264", 3840, 2160, 30)))
    assert made["estCostMicros"] == plain["estCostMicros"] + 120_000 and made["lines"][-1] == conv_line(120_000, 2.0) and lines_add_up(made)
    copied = ask(vo("mkv", None, src("wmv2", 3840, 2160, 30)))
    assert copied["estCostMicros"] == plain["estCostMicros"] and copied["lines"][-1] == {**conv_line(0, 4.0), "copied": True} and lines_add_up(copied)
    maybe = ask(vo("webm"))
    assert maybe["estCostMicros"] == plain["estCostMicros"] and maybe["lines"][-1]["maybe"] is True and lines_add_up(maybe)
    unknown = ask(vo("webm", None, {"width": 3840, "height": 2160, "fps": 30}))
    assert unknown["lines"][-1] == {**conv_line(240_000, 4.0), "maybe": True} and lines_add_up(unknown)


def test_the_estimate_of_automatic_settings_is_the_most_it_can_cost_with_the_conversion(client, signed_in, gpu_up):
    _, h = signed_in
    body = {"kind": "denoise", "denoiseTier": "standard", "durationSeconds": 600}
    base = client.post("/api/jobs/estimate", json={**body, "options": {"automatic": True}}, headers=h).json()
    sized = client.post("/api/jobs/estimate", json={**body, "options": {"automatic": True, **vo("mp4", 720, src("wmv2", 3840, 2160, 30))}}, headers=h).json()
    assert base["maximum"] is True and sized["maximum"] is True
    assert sized["estCostMicros"] == base["estCostMicros"] + 120_000 and sized["lines"][-1] == conv_line(120_000, 2.0)


# --------------------------------------------------------------------------------------------------- the hold

def test_a_picture_made_again_is_held_for_when_the_job_is_made(signed_in):
    uid, _ = funded(signed_in)
    j = make(uid, options=vo("mp4", None, src("wmv2")))  # a WMV with a source said: known to be made again
    assert j["estCostMicros"] == 60_000 == held(uid) == row(j["id"])["est_cost_micros"]
    p = params_of(j["id"])
    assert (p["convert_rate"], p["picture_encode"], p["convert_factor"], p["input_ext"]) == (CONVERT, True, 1.0, ".wmv")
    assert p["options"]["video_out"]["source"] == {"codec": "wmv2", "width": 1920, "height": 1080, "fps": 30.0}
    assert (p["denoise_tier"], p["transcribe_tier"], p["rates"]) == (None, None, {}) and row(j["id"])["kind"] == "convert"
    assert db.one("SELECT amount_micros FROM holds WHERE ref_id = ?", (j["id"],))["amount_micros"] == 60_000
    big = make(uid, options=vo("mp4", 720, src("wmv2", 3840, 2160, 60)), filename="big.wmv")  # 4K60 made 720p: 4x
    assert big["estCostMicros"] == 240_000 and params_of(big["id"])["convert_factor"] == 4.0 and held(uid) == 60_000 + 240_000
    unsaid = make(uid, options=vo("mp4", 2160), filename="c.wmv")  # a size with nothing said: 16:9 at that size, 4x
    assert unsaid["estCostMicros"] == 240_000 and params_of(unsaid["id"])["picture_encode"] is True


def test_a_copy_is_free_and_holds_nothing_even_with_no_credit(signed_in):
    uid, _ = signed_in
    mkv = make(uid, options=vo("mkv", None, src("wmv2", 3840, 2160, 60)))
    same = make(uid, options=vo("mp4", None, src("wmv2")), filename="talk.mp4")  # the same container: copied whatever the codec
    holds = make(uid, options=vo("mp4", None, src("h264")), filename="b.avi")
    for jid, factor in ((mkv["id"], 8.0), (same["id"], 1.0), (holds["id"], 1.0)):
        r = row(jid)
        assert r["hold_id"] is None and r["est_cost_micros"] == 0
        p = params_of(jid)
        assert (p["picture_encode"], p["convert_factor"]) == (False, factor)
    assert held(uid) == count("holds") == 0 and mkv["estCostMicros"] == same["estCostMicros"] == 0


def test_a_type_change_alone_without_the_codec_holds_nothing_yet(signed_in):
    uid, _ = signed_in
    for source in (None, {"width": 3840, "height": 2160, "fps": 30}):
        j = make(uid, options=vo("webm", None, source))
        assert row(j["id"])["hold_id"] is None and row(j["id"])["est_cost_micros"] == 0 and params_of(j["id"])["picture_encode"] is None
    assert params_of(j["id"])["convert_factor"] == 4.0 and held(uid) == 0


def test_a_conversion_needs_the_balance_for_its_factor(signed_in):
    uid, _ = signed_in
    ledger.credit(uid, 239_999, "grant", "t", "g")
    with pytest.raises(jobs.JobError, match=r"costs about \$0\.24; your available balance is \$0\.24") as e:
        make(uid, options=vo("mp4", None, src("wmv2", 3840, 2160, 30)))
    assert e.value.status == 402 and count("jobs") == 0 and held(uid) == 0
    ledger.credit(uid, 1, "grant", "t", "g2")
    assert make(uid, options=vo("mp4", None, src("wmv2", 3840, 2160, 30)))["estCostMicros"] == 240_000
    assert make(uid, options=vo("mkv", None, src("wmv2", 3840, 2160, 30)), filename="c.wmv")["estCostMicros"] == 0  # the copy needs none


def test_a_conversion_on_top_of_cleaning_is_held_with_it(signed_in):
    uid, _ = funded(signed_in)
    j = make(uid, kind="denoise", options=vo("mp4", 1080, src("h264", 3840, 2160, 30)), filename="a.mp4")
    assert j["estCostMicros"] == standard(600) + 120_000 == held(uid)
    k = make(uid, kind="denoise", options=vo("mp4"), filename="b.avi")  # (nothing said: the cleaning alone is held)
    assert k["estCostMicros"] == standard(600) and params_of(k["id"])["picture_encode"] is None
    c = make(uid, kind="denoise", options=vo("mkv", None, src("wmv2")), filename="c.wmv")
    assert c["estCostMicros"] == standard(600) and params_of(c["id"])["picture_encode"] is False


def test_the_whole_upload_again_must_fit_on_the_disk(signed_in, monkeypatch):
    from types import SimpleNamespace

    from acw import hostload

    uid, _ = funded(signed_in)
    size = 2 * 1024**3
    need = size + jobs.DISK_RESERVE  # a conversion keeps no scratch files or copies: the upload and the video made from it
    free = {"n": need}
    monkeypatch.setattr(hostload, "get", lambda d: SimpleNamespace(snapshot=lambda: SimpleNamespace(disk_free=free["n"])))
    with pytest.raises(jobs.JobError, match="short on storage") as e:
        make(uid, options=vo("mkv"), size=size)  # (room for the upload alone is not enough)
    assert e.value.status == 507 and count("jobs") == 0
    free["n"] = need + size - 1
    with pytest.raises(jobs.JobError, match="short on storage"):
        make(uid, options=vo("mkv"), size=size)
    free["n"] = need + size
    assert make(uid, options=vo("mkv"), size=size)["id"]


# ------------------------------------------------------------------------------------------- the worker's flow

class VideoStages(Stages):
    """The runner's stages, replaced: the probe says how long the file is and what making its picture again takes
    (`picture_encode`, `convert_factor`, as the real probe does); the main stage writes the video a conversion makes. A file with no
    picture reports `has_video` False and nothing about the picture. `during` is what the main stage saw of the money."""

    def __init__(self, monkeypatch, duration=120.0, picture_encode=True, factor=1.0, has_video=True, stats=None, audio_tracks=None):
        super().__init__(monkeypatch, stats, duration=duration)
        self.picture_encode, self.factor, self.has_video, self.audio_tracks = picture_encode, factor, has_video, audio_tracks
        self.during: dict = {}

    def __call__(self, job, d, stage, timeout):
        spec = json.loads((d / "spec.json").read_text()) if (d / "spec.json").exists() else {}
        if stage == "probe":
            self.calls.append((stage, spec, job.get("worker_id")))
            # (a file with no picture: nothing is made again, as the real probe says)
            picture = {"picture_encode": self.picture_encode, "convert_factor": self.factor} if self.has_video else {"picture_encode": False, "convert_factor": 1.0}
            tracks = {} if self.audio_tracks is None else {"audio_tracks": self.audio_tracks}
            return {"ok": True, "duration": self.duration, "channels": 1, "has_video": self.has_video, **picture, **tracks}
        if stage == "main":
            self.during = {"held": ledger.balance(job["user_id"])["held_micros"], "est": row(job["id"])["est_cost_micros"]}
            if spec["kind"] == "convert":
                self.calls.append((stage, spec, job.get("worker_id")))
                name = f"{spec['output_stem']}-converted.{spec['options']['video_out']['format']}"
                (d / name).write_bytes(b"video")
                return {"ok": True, "outputs": [{"file": name, "kind": "video"}], "kept": {}, "notice": None}
        return super().__call__(job, d, stage, timeout)


def start(client, h, kind="convert", seconds=120, filename="talk.wmv", **options):
    body = {"kind": kind, "filename": filename, "sizeBytes": 3000, "durationSeconds": seconds, "options": options}
    if kind != "convert":
        body["denoiseTier"] = "standard"
    r = client.post("/api/jobs", json=body, headers=h)
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    assert client.put(f"/api/jobs/{jid}/input", content=bytes(3000), headers=h).status_code == 200
    return jid


def test_a_picture_that_is_copied_is_free_and_nothing_is_ever_held(client, signed_in, monkeypatch):
    uid, h = signed_in
    stages = VideoStages(monkeypatch, picture_encode=False, factor=1.0)
    jid = start(client, h, **vo("mkv", None, src("wmv2")))
    take()
    done = client.get(f"/api/jobs/{jid}", headers=h).json()
    assert done["status"] == "done" and done["costMicros"] == 0 and done["label"] == "Video conversion"
    assert [o["name"] for o in done["outputs"]] == ["talk-converted.mkv"] and done["outputs"][0]["kind"] == "video"
    assert stages.stages == ["probe", "main"] and stages.spec("probe")["options"]["video_out"]["format"] == "mkv"
    assert stages.during == {"held": 0, "est": 0}
    p = params_of(jid)
    assert (p["picture_encode"], p["convert_factor"]) == (False, 1.0)
    assert row(jid)["hold_id"] is None and count("holds") == 0 and count("ledger") == 0


def test_a_picture_that_needs_making_gets_its_hold_when_the_file_is_read_and_is_charged_by_its_factor(client, signed_in, monkeypatch):
    """The client did not say the codec: nothing is held at first. The probe finds the picture is made again, and twice the work."""
    uid, h = funded(signed_in)
    stages = VideoStages(monkeypatch, duration=120.0, picture_encode=True, factor=2.0)
    jid = start(client, h, **vo("webm"))
    assert row(jid)["hold_id"] is None and held(uid) == 0 and params_of(jid)["picture_encode"] is None
    take()
    cost = per(12_000, 120)
    assert cost == 24_000
    assert stages.during == {"held": cost, "est": cost}  # the hold was made before the main stage ran
    done = client.get(f"/api/jobs/{jid}", headers=h).json()
    assert done["status"] == "done" and done["costMicros"] == cost and done["label"] == "Video conversion"
    p = params_of(jid)
    assert (p["picture_encode"], p["convert_factor"]) == (True, 2.0) and held(uid) == 0
    assert ledger.balance(uid)["balance_micros"] == 5_000_000 - cost
    line = db.one("SELECT * FROM ledger WHERE ref_id = ?", (jid,))
    assert (-line["amount_micros"], line["description"]) == (cost, "Video conversion: talk.wmv (2.0 min)")


def test_the_hold_grows_when_the_file_is_bigger_or_longer_than_the_client_said(client, signed_in, monkeypatch):
    """Said to be a minute of 1080p; read as two minutes of 4K: the hold goes from 6 000 to what that costs, and that is the charge."""
    uid, h = funded(signed_in)
    stages = VideoStages(monkeypatch, duration=120.0, picture_encode=True, factor=4.0)
    jid = start(client, h, seconds=60, **vo("mp4", None, src("wmv2")))
    assert held(uid) == per(CONVERT, 60) == 6_000
    take()
    assert stages.during == {"held": per(24_000, 120), "est": per(24_000, 120)} and per(24_000, 120) == 48_000
    done = client.get(f"/api/jobs/{jid}", headers=h).json()
    assert done["costMicros"] == 48_000 and params_of(jid)["convert_factor"] == 4.0
    assert ledger.balance(uid)["balance_micros"] == 5_000_000 - 48_000 and held(uid) == 0


def test_a_picture_smaller_than_the_client_said_is_charged_for_what_it_is(client, signed_in, monkeypatch):
    """Said to be 4K (held 24 000 for a minute); read as 1080p: it stays held while it runs and the charge is the real 6 000."""
    uid, h = funded(signed_in)
    stages = VideoStages(monkeypatch, duration=60.0, picture_encode=True, factor=1.0)
    jid = start(client, h, seconds=60, **vo("mp4", None, src("wmv2", 3840, 2160, 30)))
    assert held(uid) == 24_000
    take()
    assert stages.during == {"held": 24_000, "est": 24_000}
    done = client.get(f"/api/jobs/{jid}", headers=h).json()
    assert done["costMicros"] == 6_000 and held(uid) == 0 and ledger.balance(uid)["balance_micros"] == 5_000_000 - 6_000


def test_a_size_the_picture_already_has_is_held_for_then_charged_nothing(client, signed_in, monkeypatch):
    """1080p asked of a video that says nothing about its size: held for a resize; the probe finds nothing to make again."""
    uid, h = funded(signed_in)
    VideoStages(monkeypatch, duration=60.0, picture_encode=False, factor=1.0)
    jid = start(client, h, seconds=60, **vo("mp4", 1080))
    assert held(uid) == 6_000
    take()
    done = client.get(f"/api/jobs/{jid}", headers=h).json()
    assert done["status"] == "done" and done["costMicros"] == 0 and params_of(jid)["picture_encode"] is False
    assert held(uid) == 0 and ledger.balance(uid)["balance_micros"] == 5_000_000


def test_a_picture_the_hold_cannot_grow_for_fails_the_job_uncharged_with_the_hint(client, signed_in, monkeypatch):
    uid, h = signed_in
    ledger.credit(uid, 47_999, "grant", "t", "g")  # one micro short of two minutes of 4K
    stages = VideoStages(monkeypatch, duration=120.0, picture_encode=True, factor=4.0)
    jid = start(client, h, seconds=60, **vo("mp4", None, src("wmv2")))  # held 6 000 of it
    assert held(uid) == 6_000
    take()
    failed = client.get(f"/api/jobs/{jid}", headers=h).json()
    assert failed["status"] == "failed" and not failed["costMicros"]
    assert failed["error"] == ("This file is 2.0 min and costs $0.05, more than your available balance. The picture of this video has to be "
                               "made again for the file type or size you chose (Video conversion), more than the video first said; "
                               "MKV at the original size keeps it as it is, for free.")
    assert stages.stages == ["probe"] and held(uid) == 0 and ledger.balance(uid)["balance_micros"] == 47_999
    assert count("ledger", "kind = 'usage'") == 0 and not folder_of(jid).exists()


def folder_of(job_id):
    return jobs.job_dir(job_id)


def test_a_picture_that_turns_out_to_need_making_with_no_hold_and_too_little_balance_fails_uncharged(client, signed_in, monkeypatch):
    uid, h = signed_in
    ledger.credit(uid, 23_999, "grant", "t", "g")  # one micro short of 2 minutes at twice the work
    stages = VideoStages(monkeypatch, duration=120.0, picture_encode=True, factor=2.0)
    jid = start(client, h, **vo("webm"))
    assert row(jid)["hold_id"] is None
    take()
    failed = client.get(f"/api/jobs/{jid}", headers=h).json()
    assert failed["status"] == "failed" and failed["error"].startswith("This file is 2.0 min and costs $0.02, more than your available balance.")
    assert "MKV at the original size keeps it as it is, for free." in failed["error"]
    assert stages.stages == ["probe"] and row(jid)["hold_id"] is None and held(uid) == 0 and ledger.balance(uid)["balance_micros"] == 23_999


def test_the_short_balance_of_a_cleaning_job_keeps_its_old_words_when_no_conversion_is_in_it(client, signed_in, monkeypatch):
    uid, h = signed_in
    ledger.credit(uid, standard(600), "grant", "t", "g")
    VideoStages(monkeypatch, duration=1200.0, picture_encode=False, factor=1.0)
    jid = start(client, h, kind="denoise", seconds=600, **vo("mkv", None, src("wmv2")))
    take()
    error = client.get(f"/api/jobs/{jid}", headers=h).json()["error"]
    assert error.startswith("This file is 20.0 min and costs $0.20, more than your available balance.") and "Video conversion" not in error


def test_the_conversion_grows_the_hold_of_a_cleaning_job_and_is_charged_with_it(client, signed_in, monkeypatch):
    uid, h = funded(signed_in)
    stages = VideoStages(monkeypatch, duration=600.0, picture_encode=True, factor=2.0, stats=measurement(strength="balanced", rebuild=False, heavy=False))
    jid = start(client, h, kind="denoise", seconds=300, **vo("mp4"))
    assert held(uid) == standard(300)  # (nothing said about the picture: the cleaning alone, for the minutes said)
    take()
    steps = standard(600)  # read as 10 minutes
    done = client.get(f"/api/jobs/{jid}", headers=h).json()
    assert done["status"] == "done" and done["costMicros"] == steps + 120_000 and held(uid) == 0
    assert ledger.balance(uid)["balance_micros"] == 5_000_000 - steps - 120_000
    assert done["label"] == "Denoising + Leveling + Video conversion" and "main" in stages.stages


def test_a_cleaning_job_with_the_picture_copied_is_charged_for_the_cleaning_alone(client, signed_in, monkeypatch):
    uid, h = funded(signed_in)
    VideoStages(monkeypatch, duration=600.0, picture_encode=False, factor=4.0, stats=measurement(strength="balanced", rebuild=False, heavy=False))
    jid = start(client, h, kind="denoise", seconds=600, **vo("mkv"))
    take()
    done = client.get(f"/api/jobs/{jid}", headers=h).json()
    assert done["status"] == "done" and done["costMicros"] == standard(600) and done["label"] == "Denoising + Leveling"


@pytest.mark.parametrize("encode", ["yes", 1, 0, "true", [], {}, None])
def test_a_probe_that_does_not_say_clearly_whether_the_picture_is_made_again_fails_the_job_uncharged(client, signed_in, monkeypatch, encode):
    uid, h = funded(signed_in)
    VideoStages(monkeypatch, picture_encode=encode, factor=1.0)
    jid = start(client, h, **vo("mp4", 720))
    take()
    failed = client.get(f"/api/jobs/{jid}", headers=h).json()
    assert failed["status"] == "failed" and failed["error"] == "Processing failed; you were not charged." and not failed["costMicros"]
    assert held(uid) == 0 and ledger.balance(uid)["balance_micros"] == 5_000_000


@pytest.mark.parametrize("factor", [None, "2", "2.0", True, False, 0, 0.99, 64.01, 1e9, -1.0, float("nan"), float("inf"), [], {}])
def test_a_probe_with_a_factor_that_makes_no_sense_fails_the_job_uncharged(client, signed_in, monkeypatch, factor):
    uid, h = funded(signed_in)
    VideoStages(monkeypatch, picture_encode=True, factor=factor)
    jid = start(client, h, **vo("mp4", 720))
    take()
    failed = client.get(f"/api/jobs/{jid}", headers=h).json()
    assert failed["status"] == "failed" and failed["error"] == "Processing failed; you were not charged." and not failed["costMicros"], factor
    assert held(uid) == 0 and ledger.balance(uid)["balance_micros"] == 5_000_000


@pytest.mark.parametrize("factor, stored", [(1, 1.0), (1.0, 1.0), (2, 2.0), (3.456, 3.46), (3.454, 3.45), (64, 64.0), (64.0, 64.0)])
def test_a_factor_from_one_to_sixty_four_is_kept_to_two_decimals(client, signed_in, monkeypatch, factor, stored):
    uid, h = funded(signed_in)
    VideoStages(monkeypatch, duration=60.0, picture_encode=True, factor=factor)
    jid = start(client, h, seconds=60, **vo("mp4", 720))
    take()
    assert client.get(f"/api/jobs/{jid}", headers=h).json()["status"] == "done" and params_of(jid)["convert_factor"] == stored


@pytest.mark.parametrize("kind, options, steps", [
    ("convert", vo("mp4", 720), 0), ("convert", vo("webm"), 0), ("denoise", vo("mp4", 720), standard(600)), ("denoise", vo("mkv"), standard(600)),
    ("denoise", {"automatic": True, **vo("webm")}, standard(600)),
])
def test_a_file_that_turns_out_to_have_no_picture_gets_a_black_one_and_the_video_costs_nothing(client, signed_in, monkeypatch, kind, options, steps):
    """A video ending that holds only sound: the probe says nothing is made again; whatever was held for the picture is let go, the cleaning
    (when asked for) is charged as usual."""
    uid, h = funded(signed_in)
    stages = VideoStages(monkeypatch, duration=600.0, has_video=False, stats=measurement(strength="balanced", rebuild=False, heavy=False))
    jid = start(client, h, kind=kind, seconds=600, filename="talk.mkv", **options)
    take()
    done = client.get(f"/api/jobs/{jid}", headers=h).json()
    assert done["status"] == "done" and done["costMicros"] == steps
    assert "main" in stages.stages and done["label"] == ("Video conversion" if kind == "convert" else "Denoising + Leveling")
    p = params_of(jid)
    assert (p["picture_encode"], p["convert_factor"]) == (False, 1.0) and jobs.convert_cost(p, 600) == 0
    assert held(uid) == 0 and ledger.balance(uid)["balance_micros"] == 5_000_000 - steps


def test_the_finishing_step_of_a_picture_made_again_gets_the_time_it_needs(signed_in, monkeypatch):
    uid, _ = funded(signed_in)
    for source, seconds, encode, want in ((src("wmv2"), 100.0, True, 1800), (src("wmv2"), 7200.0, True, int(7200 * 0.32 * 3)),
                                          (None, 6000.0, None, int(6000 * 0.32 * 3)), (src("wmv2"), 7200.0, False, 1800)):
        jid = make(uid, options=vo("mp4", None, source), seconds=seconds, filename="talk.wmv")["id"]
        db.execute("UPDATE jobs SET status = 'running', next_stage = 'finalize', duration_seconds = ? WHERE id = ?", (seconds, jid))
        if encode is not True:
            jobs.update_params(jid, picture_encode=encode)
        asked = []
        monkeypatch.setattr(worker, "run_stage", lambda job, d, stage, timeout, asked=asked: asked.append((stage, timeout)) or {"ok": True, "outputs": []})
        monkeypatch.setattr(worker, "_complete", lambda *a, **k: None)
        worker.process(dict(db.one("SELECT * FROM jobs WHERE id = ?", (jid,))), worker._settings_extra())
        assert asked == [("finalize", want)], (encode, seconds)
        db.execute("DELETE FROM jobs WHERE id = ?", (jid,))


# --------------------------------------------------------------------------------------------- automatic settings

def test_the_conversion_stays_in_the_hold_and_the_charge_when_automatic_settings_decide(client, signed_in, monkeypatch, gpu_up):
    uid, h = funded(signed_in, 20_000_000)
    stages = VideoStages(monkeypatch, duration=600.0, picture_encode=True, factor=4.0, stats=measurement(strength="gentle", rebuild=False, heavy=False))
    jid = start(client, h, kind="denoise", seconds=600, automatic=True, **vo("mp4", None, src("wmv2", 3840, 2160, 30)))
    before = held(uid)
    assert before == rebuilt(600) + 240_000  # the most the decision could end in (a voice rebuild), and the picture at 4x
    seen = {}
    real = stages.__call__

    def run(job, d, stage, timeout):
        if stage == "main":  # the decision has been made and applied by now
            seen["held"], seen["est"] = held(uid), row(jid)["est_cost_micros"]
        return real(job, d, stage, timeout)

    monkeypatch.setattr(worker, "run_stage", run)
    take()
    after = standard(600) + 240_000  # Denoising + Leveling as decided, and the picture as it was
    assert seen == {"held": after, "est": after} and after < before
    done = client.get(f"/api/jobs/{jid}", headers=h).json()
    assert done["status"] == "done" and done["costMicros"] == after and done["label"] == "Denoising + Leveling + Video conversion"
    assert held(uid) == 0 and ledger.balance(uid)["balance_micros"] == 20_000_000 - after


def test_apply_adds_the_conversion_to_what_it_holds(signed_in, gpu_up):
    uid, _ = funded(signed_in, 20_000_000)
    jid = make(uid, kind="denoise", seconds=600, options={"automatic": True, **vo("mp4", 720, src("wmv2", 3840, 2160, 30))})["id"]
    db.execute("UPDATE jobs SET status = 'running', duration_seconds = 600 WHERE id = ?", (jid,))
    decision = automatic.decide(params_of(jid), measurement(strength="gentle", rebuild=False, heavy=False), 600.0, card_up=True)
    params = automatic.apply(jid, decision, 600.0)
    steps = standard(600)
    assert row(jid)["est_cost_micros"] == steps + 120_000 == held(uid)
    assert params["picture_encode"] is True and params["convert_factor"] == 2.0
    # with the picture copied there is nothing to add
    other = make(uid, kind="denoise", seconds=600, options={"automatic": True, **vo("mkv", None, src("wmv2", 3840, 2160, 60))}, filename="b.wmv")["id"]
    db.execute("UPDATE jobs SET status = 'running', duration_seconds = 600 WHERE id = ?", (other,))
    automatic.apply(other, automatic.decide(params_of(other), measurement(strength="gentle", rebuild=False, heavy=False), 600.0, card_up=True), 600.0)
    assert row(other)["est_cost_micros"] == steps


# ------------------------------------------------------------------------------------ the runner, with real files

NATIVE = pytest.mark.skipif(not (FFMPEG.exists() and FFPROBE.exists()), reason="ffmpeg not available")


def ffmpeg(*args):
    subprocess.run([str(FFMPEG), "-v", "error", "-y", *args], check=True, stdin=subprocess.DEVNULL)


def streams(path):
    out = subprocess.run([str(FFPROBE), "-v", "error", "-print_format", "json", "-show_streams", str(path)], capture_output=True, text=True, check=True).stdout
    found = {}
    for s in json.loads(out)["streams"]:
        found.setdefault(s["codec_type"], s)
    return found


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("convert")
    ffmpeg("-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=3", "-f", "lavfi", "-i", "sine=f=440:d=3", "-c:v", "wmv2", "-c:a", "wmav2", str(d / "talk.wmv"))
    ffmpeg("-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=3", "-f", "lavfi", "-i", "sine=f=440:d=3",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(d / "talk.mp4"))
    ffmpeg("-f", "lavfi", "-i", "sine=f=440:d=3", "-c:a", "pcm_s16le", str(d / "talk.wav"))
    # a 4K picture at 60 frames a second, a second of it (a plain colour: it encodes fast), also stored sideways and shown upright
    ffmpeg("-f", "lavfi", "-i", "color=c=0x336699:size=3840x2160:rate=60:duration=1", "-f", "lavfi", "-i", "sine=f=440:d=1", "-c:v", "mpeg4", "-q:v", "20",
           "-c:a", "aac", str(d / "big.mp4"))
    ffmpeg("-display_rotation", "90", "-i", str(d / "big.mp4"), "-c", "copy", str(d / "turned.mp4"))
    # files whose ending says video and that hold only sound
    ffmpeg("-f", "lavfi", "-i", "sine=f=440:d=3", "-c:a", "flac", str(d / "sound.mkv"))
    ffmpeg("-f", "lavfi", "-i", "sine=f=440:d=3", "-c:a", "aac", str(d / "sound.mp4"))
    return {p.name: p.read_bytes() for p in d.iterdir()}


def real_job(client, h, media, name, seconds=3, **options):
    data = media[name]
    r = client.post("/api/jobs", json={"kind": "convert", "filename": name, "sizeBytes": len(data), "durationSeconds": seconds, "options": options}, headers=h)
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    assert client.put(f"/api/jobs/{jid}/input", content=data, headers=h).status_code == 200
    return jid


def take(worker_id="test"):
    """A worker takes the next queued job and processes it; returns the job's id."""
    job = worker.claim(worker_id)
    assert job is not None
    worker.process(job, worker._settings_extra())
    return job["id"]


def download(client, h, jid, name, tmp_path):
    r = client.get(f"/api/jobs/{jid}/outputs/{name}", headers=h)
    assert r.status_code == 200, r.text
    p = tmp_path / name
    p.write_bytes(r.content)
    return p



@NATIVE
def test_a_wmv_is_copied_into_mkv_and_its_sound_comes_along(client, signed_in, media, tmp_path):
    uid, h = funded(signed_in)
    jid = real_job(client, h, media, "talk.wmv", **vo("mkv"))
    take()
    done = client.get(f"/api/jobs/{jid}", headers=h).json()
    assert done["status"] == "done", done["error"]
    assert [o["name"] for o in done["outputs"]] == ["talk-converted.mkv"] and done["outputs"][0]["kind"] == "video" and done["label"] == "Video conversion"
    s = streams(download(client, h, jid, "talk-converted.mkv", tmp_path))
    assert (s["video"]["codec_name"], s["audio"]["codec_name"]) == ("wmv2", "wmav2") and (s["video"]["width"], s["video"]["height"]) == (320, 240)
    p = params_of(jid)
    assert (done["costMicros"], p["picture_encode"], p["convert_factor"], row(jid)["hold_id"]) == (0, False, 1.0, None) and held(uid) == 0
    assert ledger.balance(uid)["balance_micros"] == 5_000_000


@NATIVE
def test_a_wmv_made_into_mp4_is_h264_and_aac(client, signed_in, media, tmp_path):
    uid, h = funded(signed_in)
    jid = real_job(client, h, media, "talk.wmv", **vo("mp4"))
    take()
    done = client.get(f"/api/jobs/{jid}", headers=h).json()
    assert done["status"] == "done", done["error"]
    assert [o["name"] for o in done["outputs"]] == ["talk-converted.mp4"]
    s = streams(download(client, h, jid, "talk-converted.mp4", tmp_path))
    assert (s["video"]["codec_name"], s["audio"]["codec_name"]) == ("h264", "aac") and (s["video"]["width"], s["video"]["height"]) == (320, 240)
    # nothing was said about the picture: no hold at first; the probe found it made again (1x), so the charge is the minutes at the rate
    seconds = row(jid)["duration_seconds"]
    assert seconds == pytest.approx(3.0, abs=0.4) and done["costMicros"] == per(CONVERT, seconds) > 0 and held(uid) == 0
    p = params_of(jid)
    assert (p["picture_encode"], p["convert_factor"]) == (True, 1.0) and ledger.balance(uid)["balance_micros"] == 5_000_000 - done["costMicros"]


@NATIVE
def test_a_size_resizes_and_keeps_the_shape(client, signed_in, media, tmp_path):
    uid, h = funded(signed_in)
    jid = real_job(client, h, media, "talk.mp4", **vo("mp4", 480))
    take()
    done = client.get(f"/api/jobs/{jid}", headers=h).json()
    assert done["status"] == "done", done["error"]
    s = streams(download(client, h, jid, "talk-converted.mp4", tmp_path))
    assert (s["video"]["width"], s["video"]["height"]) == (640, 480) and s["video"]["codec_name"] == "h264" and s["audio"]["codec_name"] == "aac"
    assert done["costMicros"] == per(CONVERT, row(jid)["duration_seconds"]) and params_of(jid)["convert_factor"] == 1.0 and held(uid) == 0


@NATIVE
def test_a_wmv_made_into_webm_is_vp9_and_opus(client, signed_in, media, tmp_path):
    uid, h = funded(signed_in)
    jid = real_job(client, h, media, "talk.wmv", **vo("webm"))
    take()
    done = client.get(f"/api/jobs/{jid}", headers=h).json()
    assert done["status"] == "done", done["error"]
    s = streams(download(client, h, jid, "talk-converted.webm", tmp_path))
    assert (s["video"]["codec_name"], s["audio"]["codec_name"]) == ("vp9", "opus")
    assert done["costMicros"] == per(CONVERT, row(jid)["duration_seconds"])


@NATIVE
def test_an_mp4_into_mkv_copies_everything(client, signed_in, media, tmp_path):
    uid, h = signed_in  # (no credit: a copy is free)
    jid = real_job(client, h, media, "talk.mp4", **vo("mkv"))
    take()
    done = client.get(f"/api/jobs/{jid}", headers=h).json()
    assert done["status"] == "done" and done["costMicros"] == 0
    s = streams(download(client, h, jid, "talk-converted.mkv", tmp_path))
    assert (s["video"]["codec_name"], s["audio"]["codec_name"]) == ("h264", "aac")


@NATIVE
def test_a_4k_picture_made_smaller_is_held_and_charged_by_its_work(client, signed_in, media, tmp_path):
    """4K at 60 frames a second made 720p counts 4 times (half the original's pixels a second): 24 000 micros a minute."""
    uid, h = funded(signed_in)
    jid = real_job(client, h, media, "big.mp4", seconds=1, **vo("mp4", 720, src("mpeg4", 3840, 2160, 60)))
    assert held(uid) == per(24_000, 1) == 400 and params_of(jid)["convert_factor"] == 4.0
    take()
    done = client.get(f"/api/jobs/{jid}", headers=h).json()
    assert done["status"] == "done", done["error"]
    p = params_of(jid)
    assert (p["picture_encode"], p["convert_factor"]) == (True, 4.0)
    assert done["costMicros"] == per(24_000, row(jid)["duration_seconds"]) and held(uid) == 0
    s = streams(download(client, h, jid, "big-converted.mp4", tmp_path))
    assert (s["video"]["width"], s["video"]["height"]) == (1280, 720) and s["video"]["codec_name"] == "h264"


@NATIVE
def test_what_the_client_said_is_checked_against_the_file(client, signed_in, media):
    """Said to be 320x240 (1x); it is 4K at 60 frames (8x): the hold grows to the file's price, which is what is charged."""
    uid, h = funded(signed_in)
    jid = real_job(client, h, media, "big.mp4", seconds=1, **vo("webm", None, src("mpeg4", 320, 240, 10)))
    assert held(uid) == per(CONVERT, 1) == 100
    job = worker.claim("test")
    seen = {}
    real = worker.run_stage

    def run(j, d, stage, timeout):
        if stage == "main":
            seen["held"], seen["est"] = held(uid), row(jid)["est_cost_micros"]
            return {"ok": True, "outputs": [], "kept": {}}  # (the encode itself is not run here)
        return real(j, d, stage, timeout)

    worker.run_stage = run
    try:
        worker.process(job, worker._settings_extra())
    finally:
        worker.run_stage = real
    duration = row(jid)["duration_seconds"]
    assert params_of(jid)["convert_factor"] == 8.0 and seen == {"held": per(48_000, duration), "est": per(48_000, duration)}


@NATIVE
def test_the_first_stage_says_whether_the_picture_is_made_again_and_how_big_the_work_is(media, tmp_path):
    d = tmp_path / "job"
    d.mkdir()
    names = {"talk.wmv": "input.wmv", "talk.mp4": "input.mp4", "talk.wav": "input.wav", "big.mp4": "big.mp4", "turned.mp4": "turned.mp4"}
    for name, to in names.items():
        (d / to).write_bytes(media[name])
    base = {"bin_dir": str(HOME / "bin"), "threads": 1, "max_seconds": 3600, "denoise_tier": None}

    def probe(name, fmt, size=None):
        got = runner.stage_probe(d, {**base, "input": name, "options": {"video_out": {"format": fmt, "size": size}}})
        return got["picture_encode"], got["convert_factor"]

    cases = [
        ("input.wmv", "mkv", None, (False, 1.0)), ("input.wmv", "mp4", None, (True, 1.0)), ("input.wmv", "webm", None, (True, 1.0)),
        ("input.wmv", "mkv", 720, (True, 1.0)),
        ("input.mp4", "mp4", None, (False, 1.0)), ("input.mp4", "mkv", None, (False, 1.0)), ("input.mp4", "mov", None, (False, 1.0)),
        ("input.mp4", "webm", None, (True, 1.0)), ("input.mp4", "mp4", 240, (False, 1.0)), ("input.mp4", "mp4", 480, (True, 1.0)),
        # a 4K picture at 60 frames a second (MPEG-4 in an MP4): a copy counts 1 (nothing is made); made again it counts by its work
        ("big.mp4", "mp4", None, (False, 1.0)), ("big.mp4", "mkv", None, (False, 1.0)), ("big.mp4", "webm", None, (True, 8.0)),
        ("big.mp4", "mp4", 720, (True, 4.0)), ("big.mp4", "mkv", 1080, (True, 4.0)), ("big.mp4", "mp4", 2160, (False, 1.0)),
        ("big.mp4", "mp4", 1440, (True, 4.0)),
        # stored sideways and shown upright: the work is the same, the size asked is of the shown picture
        ("turned.mp4", "webm", None, (True, 8.0)), ("turned.mp4", "mp4", 720, (True, 4.0)), ("turned.mp4", "mp4", 2160, (False, 1.0)),
    ]
    for name, fmt, size, want in cases:
        assert probe(name, fmt, size) == want, (name, fmt, size)
    got = runner.stage_probe(d, {**base, "input": "input.wmv", "options": {"video_out": {"format": "mp4", "size": None}}})
    assert got["has_video"] is True and got["duration"] == pytest.approx(3.0, abs=0.4)
    # no video_out, or no picture: nothing to say
    assert runner.stage_probe(d, {**base, "input": "input.wmv", "options": {}})["picture_encode"] is None
    sound = runner.stage_probe(d, {**base, "input": "input.wav", "options": {"video_out": {"format": "mp4", "size": 720}}})
    assert (sound["has_video"], sound["picture_encode"], sound["convert_factor"]) == (False, False, 1.0)  # (a sound: nothing is made again)
    assert sound["audio_tracks"] == 1



@NATIVE
def test_the_finishing_step_names_the_file_and_gives_the_card_its_encoders(media, tmp_path, monkeypatch):
    """stage_finalize on its own: a conversion alone is `<stem>-converted.<type>`; the encoders are the spec's (NVENC first on the card)."""
    from acw_common import ffmpeg as ffmod

    d = tmp_path / "job"
    d.mkdir()
    (d / "input.wmv").write_bytes(media["talk.wmv"])
    (d / "main-state.json").write_text(json.dumps({"outputs": [], "cleaned": None, "has_video": True}))
    spec = {"bin_dir": str(HOME / "bin"), "threads": 1, "input": "input.wmv", "input_ext": ".wmv", "output_stem": "clip",
            "options": {"video_out": {"format": "mp4", "size": 480}, "output_video": True, "embed_subtitles": True}}
    seen = []
    real = ffmod.FFmpeg.mux_video

    def spy(self, video_src, audio, out, *a, **kw):
        seen.append((out.name, audio, kw.get("video")))
        return real(self, video_src, audio, out, *a, **kw)

    monkeypatch.setattr(ffmod.FFmpeg, "mux_video", spy)
    got = runner.stage_finalize(d, spec)
    assert got["outputs"] == [{"file": "clip-converted.mp4", "kind": "video"}]
    name, audio, video = seen[0]
    assert (name, audio) == ("clip-converted.mp4", None)
    assert (video.short_side, video.encoders, video.fast) == (480, ("libx264",), True)
    s = streams(d / "clip-converted.mp4")
    assert (s["video"]["width"], s["video"]["height"]) == (640, 480)
    # on the card's worker: NVENC first, x264 after it (NVENC itself is not run here; a name FFmpeg does not know stands for it)
    (d / "clip-converted.mp4").unlink()
    monkeypatch.setattr(runner, "_encoders", lambda spec, ext: ("no_such_encoder", "libx264"))
    runner.stage_finalize(d, {**spec, "gpu_devices": ["/dev/nvidia0"]})
    assert seen[1][2].encoders == ("no_such_encoder", "libx264") and streams(d / "clip-converted.mp4")["video"]["codec_name"] == "h264"




def black_frame_max(path):
    """The brightest value (0 to 255) of the first picture of a video file."""
    out = subprocess.run([str(FFMPEG), "-v", "error", "-i", str(path), "-frames:v", "1", "-pix_fmt", "gray", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return max(out) if out else -1


@NATIVE
@pytest.mark.parametrize("name", ["talk.wav", "sound.mkv", "sound.mp4"])
@pytest.mark.parametrize("fmt, size, vcodec, acodec, dims", [
    ("mp4", None, "h264", "aac", (1280, 720)), ("mkv", 480, "h264", "aac", (854, 480)), ("webm", None, "vp9", "opus", (1280, 720)),
])
def test_a_sound_asked_for_as_a_video_comes_out_with_a_plain_black_picture_and_its_sound_for_nothing(client, signed_in, media, tmp_path, name, fmt, size, vcodec, acodec, dims):
    uid, h = funded(signed_in)
    jid = real_job(client, h, media, name, **vo(fmt, size))
    take()
    done = client.get(f"/api/jobs/{jid}", headers=h).json()
    assert done["status"] == "done", done["error"]
    stem = name.rsplit(".", 1)[0]
    assert [(o["name"], o["kind"]) for o in done["outputs"]] == [(f"{stem}-converted.{fmt}", "video")]
    out = download(client, h, jid, f"{stem}-converted.{fmt}", tmp_path)
    s = streams(out)
    assert (s["video"]["codec_name"], s["audio"]["codec_name"]) == (vcodec, acodec) and (s["video"]["width"], s["video"]["height"]) == dims
    assert black_frame_max(out) < 40  # (black: 16 is black in video levels)
    assert abs(float(s["audio"].get("duration") or 0) - 3.0) < 1.0 or abs(float(s["video"].get("duration") or 3.0) - 3.0) < 1.5
    # nothing is charged for the video, and whatever was held for it (a size asked of a video ending) is let go
    p = params_of(jid)
    assert (p["picture_encode"], p["convert_factor"]) == (False, 1.0) and done["costMicros"] == 0
    assert held(uid) == 0 and ledger.balance(uid)["balance_micros"] == 5_000_000


@NATIVE
def test_the_finishing_step_makes_the_black_picture_with_the_cleaned_sound_or_the_files_own(media, tmp_path, monkeypatch):
    """stage_finalize for a sound asked for as a video: `<stem>-cleaned.<type>` with the cleaned sound, `-captioned` with only subtitles,
    `-converted` with the file's own; the encoders are the spec's."""
    import numpy as np

    from acw_common import ffmpeg as ffmod
    from acw_common.ffmpeg import FFmpeg

    ff = FFmpeg(HOME / "bin")
    d = tmp_path / "job"
    d.mkdir()
    (d / "input.wav").write_bytes(media["talk.wav"])  # 440 Hz
    cleaned = ff.write_audio(0.3 * np.sin(2 * np.pi * 880 * np.arange(3 * 48000) / 48000).astype(np.float32), 48000, d / "clip-cleaned", "flac")  # 880 Hz
    spec = {"bin_dir": str(HOME / "bin"), "threads": 1, "input": "input.wav", "input_ext": ".wav", "output_stem": "clip",
            "options": {"video_out": {"format": "mp4", "size": None}, "output_video": True, "embed_subtitles": True, "caption_formats": ["srt"]}}

    def peak_hz(path):
        x = ff.decode(path, ff.probe(path), 16000, 1, 10.0)[:, 0]
        return np.argmax(np.abs(np.fft.rfft(x[: 16000 * 2]))) / 2

    def state(**kw):
        (d / "main-state.json").write_text(json.dumps({"outputs": [], "cleaned": None, "has_video": False, "duration": 3.0, **kw}))

    seen = []
    real = ffmod.FFmpeg.make_video

    def spy(self, audio, out, duration, *a, **kw):
        seen.append((audio.name, out.name, round(duration, 1), kw.get("video")))
        return real(self, audio, out, duration, *a, **kw)

    monkeypatch.setattr(ffmod.FFmpeg, "make_video", spy)
    # the file's own sound
    state()
    got = runner.stage_finalize(d, spec)
    assert got["outputs"] == [{"file": "clip-converted.mp4", "kind": "video"}]
    assert seen[0][:3] == ("input.wav", "clip-converted.mp4", 3.0) and (seen[0][3].short_side, seen[0][3].encoders) == (None, ("libx264",))
    s = streams(d / "clip-converted.mp4")
    assert (s["video"]["codec_name"], s["video"]["width"], s["video"]["height"], s["audio"]["codec_name"]) == ("h264", 1280, 720, "aac")
    assert abs(peak_hz(d / "clip-converted.mp4") - 440) < 10 and black_frame_max(d / "clip-converted.mp4") < 40
    # the cleaned sound, at the size asked for
    state(cleaned="clip-cleaned.flac", outputs=[{"file": "clip-cleaned.flac", "kind": "audio"}])
    got = runner.stage_finalize(d, {**spec, "options": {**spec["options"], "video_out": {"format": "mkv", "size": 480}}})
    assert [o["file"] for o in got["outputs"]] == ["clip-cleaned.flac", "clip-cleaned.mkv"]
    assert seen[1][:3] == ("clip-cleaned.flac", "clip-cleaned.mkv", 3.0) and seen[1][3].short_side == 480
    assert abs(peak_hz(d / "clip-cleaned.mkv") - 880) < 10 and (streams(d / "clip-cleaned.mkv")["video"]["width"], streams(d / "clip-cleaned.mkv")["video"]["height"]) == (854, 480)
    # only subtitles: the file's own sound with them in
    state()
    (d / "segments.json").write_text(json.dumps({"language": "en", "segments": [{"start": 0.2, "end": 2.0, "text": "hello there", "words": []}]}))
    got = runner.stage_finalize(d, spec)
    assert [o["file"] for o in got["outputs"] if o["kind"] == "video"] == ["clip-captioned.mp4"]
    assert streams(d / "clip-captioned.mp4")["subtitle"]["codec_name"] == "mov_text"
    # the card's worker: NVENC first, x264 after it (a name FFmpeg does not know stands for the refusing card)
    (d / "segments.json").unlink()
    monkeypatch.setattr(runner, "_encoders", lambda spec, ext: ("no_such_encoder", "libx264"))
    (d / "clip-converted.mp4").unlink()
    runner.stage_finalize(d, {**spec, "gpu_devices": ["/dev/nvidia0"]})
    assert seen[-1][3].encoders == ("no_such_encoder", "libx264") and streams(d / "clip-converted.mp4")["video"]["codec_name"] == "h264"
