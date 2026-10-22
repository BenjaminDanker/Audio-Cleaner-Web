"""A video made in another file type or size on our side (options.video_out, kind "convert" for a conversion alone; docs/api.md):
the request's checks, the label, the card's encode stage, the encoders and file name of the runner's finishing step, a re-run making
the video as its source did, and the runner making real files from small generated videos (no models are needed for a conversion).
What a conversion costs (its price, hold and charge) is not covered here."""
from __future__ import annotations

import json
import subprocess

import pytest

from acw import broker, jobs, rerun
from acw.billing import ledger
from acw.worker import main as worker
from acw.worker import runner

from .rerun_support import count, make_done_job, roomy_disk, row  # noqa: F401 - fixtures
from .test_job_pipeline import HOME

pytestmark = pytest.mark.usefixtures("roomy_disk")

FFMPEG = HOME / "bin" / "ffmpeg"
FFPROBE = HOME / "bin" / "ffprobe"


def vo(fmt="mp4", size=None):
    return {"video_out": {"format": fmt, "size": size}}


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
    """A job row as the broker reads it (kind denoise unless told otherwise)."""
    nxt = p.pop("next_stage", None)
    kind = p.pop("kind", "convert")
    return {"kind": kind, "next_stage": nxt, "params_json": json.dumps({"denoise_tier": p.pop("tier", None), "transcribe_tier": p.pop("transcribe", None), **p})}


@pytest.mark.parametrize("row_, stages, card_only", [
    (stage_row(options=vo("mp4", 720), picture_encode=True), ["encode"], False),          # the picture made again: the card's encoder
    (stage_row(options=vo("mp4"), picture_encode=None), ["encode"], False),               # (it may be: the probe will say)
    (stage_row(options=vo("mkv"), picture_encode=False), [], False),                      # copied: nothing for the card
    (stage_row(options=vo("mp4", 720), picture_encode=True, next_stage="finalize"), [], False),  # (the finishing step is not given to the card)
    (stage_row(options={}), [], False),
    (stage_row(kind="denoise", tier="standard", options={"noise": "heavy", **vo("mp4", 720)}, picture_encode=True), ["heavy", "encode"], False),
    (stage_row(kind="denoise", tier="standard", options={"noise": "light", **vo("mp4", 720)}, picture_encode=True), ["encode"], False),
    (stage_row(kind="denoise", tier="standard", options={"noise": "heavy", **vo("mp4", 720)}, picture_encode=False), ["heavy"], False),
    (stage_row(kind="denoise", tier="standard", options={"noise": "heavy", "echo_reference_track": True, **vo("webm")}, picture_encode=None), ["encode"], False),
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



# ------------------------------------------------------------------------------------------------------ re-runs

def test_a_re_run_makes_the_video_as_the_source_did_whatever_it_is_asked():
    src = {"options": jobs.validate_options("denoise", "standard", None, vo("mkv", 480))}
    merged = rerun.merged_options(src, {"atten_lim_db": 24, "video_out": {"format": "webm", "size": None}})
    assert merged["video_out"] == {"format": "mkv", "size": 480} and merged["atten_lim_db"] == 24 and merged["automatic"] is False
    none = {"options": jobs.validate_options("denoise", "standard", None, {})}
    assert rerun.merged_options(none, {"video_out": {"format": "mp4", "size": 720}})["video_out"] is None  # (it cannot be added later)
    assert rerun.merged_options(src, None)["video_out"] == {"format": "mkv", "size": 480}


def test_a_re_run_through_the_api_keeps_the_source_s_video_out(client, signed_in):
    uid, h = funded(signed_in)
    src = make_done_job(uid, seconds=60.0, options=vo("mp4", 720))
    ask = {"options": {"atten_lim_db": 24, "video_out": {"format": "webm", "size": None}}}  # (the video's choices are not for changing here)
    e = client.post(f"/api/jobs/{src}/rerun/estimate", json=ask, headers=h)
    assert e.status_code == 200, e.text
    assert e.json()["options"]["video_out"] == {"format": "mp4", "size": 720}
    made = client.post(f"/api/jobs/{src}/rerun", json=ask, headers=h)
    assert made.status_code == 200, made.text
    assert json.loads(row(made.json()["id"])["params_json"])["options"]["video_out"] == {"format": "mp4", "size": 720}
    # a job made without one cannot get one by a re-run
    plain = make_done_job(uid, seconds=60.0)
    e = client.post(f"/api/jobs/{plain}/rerun/estimate", json={"options": {"atten_lim_db": 24, "video_out": {"format": "mp4", "size": 720}}}, headers=h)
    assert e.status_code == 200 and e.json()["options"]["video_out"] is None



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


@NATIVE
def test_a_size_resizes_and_keeps_the_shape(client, signed_in, media, tmp_path):
    uid, h = funded(signed_in)
    jid = real_job(client, h, media, "talk.mp4", **vo("mp4", 480))
    take()
    done = client.get(f"/api/jobs/{jid}", headers=h).json()
    assert done["status"] == "done", done["error"]
    s = streams(download(client, h, jid, "talk-converted.mp4", tmp_path))
    assert (s["video"]["width"], s["video"]["height"]) == (640, 480) and s["video"]["codec_name"] == "h264" and s["audio"]["codec_name"] == "aac"


@NATIVE
def test_a_wmv_made_into_webm_is_vp9_and_opus(client, signed_in, media, tmp_path):
    uid, h = funded(signed_in)
    jid = real_job(client, h, media, "talk.wmv", **vo("webm"))
    take()
    done = client.get(f"/api/jobs/{jid}", headers=h).json()
    assert done["status"] == "done", done["error"]
    s = streams(download(client, h, jid, "talk-converted.webm", tmp_path))
    assert (s["video"]["codec_name"], s["audio"]["codec_name"]) == ("vp9", "opus")


@NATIVE
def test_an_mp4_into_mkv_copies_everything(client, signed_in, media, tmp_path):
    uid, h = funded(signed_in)
    jid = real_job(client, h, media, "talk.mp4", **vo("mkv"))
    take()
    assert client.get(f"/api/jobs/{jid}", headers=h).json()["status"] == "done"
    s = streams(download(client, h, jid, "talk-converted.mkv", tmp_path))
    assert (s["video"]["codec_name"], s["audio"]["codec_name"]) == ("h264", "aac")



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


