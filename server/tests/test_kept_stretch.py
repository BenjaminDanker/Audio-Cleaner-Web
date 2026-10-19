"""The strength slider's stretches: GET /api/jobs/{id}/kept/{noise|input}?start=&seconds= hands out short, sample-exact pieces of
the two kept copies of a job of any length (acw/stretch.py, the confined cutter acw/worker/cutter.py).

Real 24-bit FLAC copies made with the real FFmpeg (~/audio-cleaner/bin); everything else is the fake finished jobs of
rerun_support (no worker, no models). Skipped when FFmpeg is not there."""
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from acw import db, jobs, stretch
from acw.config import get_settings
from acw.security import LIMITS
from acw_common import kept
from acw_common.ffmpeg import AUDIO_FORMATS, FFmpeg

from .conftest import ORIGIN
from .rerun_support import (SR, account, api_key, clock, folder, make_done_job, other_browser, params_of, roomy_disk, row)  # noqa: F401

HOME = Path(os.environ.get("ACW_HOME", Path.home() / "audio-cleaner"))
pytestmark = [
    pytest.mark.usefixtures("roomy_disk"),
    pytest.mark.skipif(not (HOME / "bin/ffmpeg").exists(), reason="FFmpeg is not available"),
]


def ff() -> FFmpeg:
    return FFmpeg(HOME / "bin")


def signals(seconds: float, channels: int, seed: int = 1):
    """A noise-removed signal and the recording it came from, as float32 (samples, channels): different on every sample and channel."""
    n = int(seconds * SR)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    voice = np.stack([0.3 * np.sin(2 * np.pi * (220 + 70 * c) * t) * np.sin(2 * np.pi * 0.7 * t) for c in range(channels)], axis=1)
    noise = 0.02 * rng.standard_normal((n, channels))
    return (voice * 0.95).astype(np.float32), (voice + noise).astype(np.float32)


def real_job(uid, seconds=20.0, channels=2, **kw) -> str:
    """A finished job whose two kept copies are real FLAC files (24-bit, 48 kHz) made by kept.save."""
    jid = make_done_job(uid, seconds=seconds, channels=channels, **kw)
    y, x = signals(seconds, channels)
    d = folder(jid)
    params = params_of(jid)
    for key, name, arr in (("noise", "kept-noise", y), ("input", "kept-input", x)):
        entry = kept.save(ff(), arr, None, d / name)
        params["kept"][key].update({"channels": entry["channels"], "scale": entry["scale"], "samples": entry["samples"]})
    db.execute("UPDATE jobs SET params_json = ? WHERE id = ?", (__import__("json").dumps(params), jid))
    return jid


def decode(data: bytes, channels: int) -> np.ndarray:
    """The samples of a FLAC given as bytes, float32 (samples, channels)."""
    r = subprocess.run([str(HOME / "bin/ffmpeg"), "-v", "error", "-f", "flac", "-i", "pipe:0", "-f", "f32le", "-"], input=data, capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32).reshape(-1, channels)


def whole(jid: str, name: str, channels: int) -> np.ndarray:
    path = folder(jid) / f"kept-{name}.flac"
    return decode(path.read_bytes(), channels)


def stretch_of(client, jid, which, start=None, seconds=None, headers=None):
    q = "&".join(f"{k}={v}" for k, v in (("start", start), ("seconds", seconds)) if v is not None)
    return client.get(f"/api/jobs/{jid}/kept/{which}" + (f"?{q}" if q else ""), headers=headers or {})


# ------------------------------------------------------------------------------------------- the numbers

def test_span_rounds_to_the_nearest_sample_caps_at_thirty_seconds_and_stops_at_the_end():
    total = 100 * SR
    assert stretch.span(0, 10, total) == (0, 10 * SR)
    assert stretch.span(None, None, total) == (0, 30 * SR)  # defaults: from the start, the longest allowed
    assert stretch.span(12.5, None, total) == (int(12.5 * SR), 30 * SR)
    assert stretch.span(1 / 3, 0.001, total) == (16000, 48)
    assert stretch.span(5, 500, total) == (5 * SR, 30 * SR)  # more than the cap is cut to it
    assert stretch.span(95, 30, total) == (95 * SR, 5 * SR)  # cut short by the end
    assert stretch.span(100 - 1 / SR, 1, total) == (total - 1, 1)
    assert stretch.span(0, 1e-9, total)[1] == 1  # never empty
    for start in (100, 100.5, 5000):
        with pytest.raises(stretch.OutOfRange):
            stretch.span(start, 1, total)


def test_the_cutter_writes_the_same_kind_of_flac_the_copies_are():
    from acw.worker import cutter

    assert cutter.FLAC24 == AUDIO_FORMATS["flac24"][1] and cutter.SAMPLE_RATE == kept.SAMPLE_RATE == stretch.SAMPLE_RATE
    assert stretch.MAX_SECONDS == jobs.STRETCH_MAX_SECONDS == 30


# ------------------------------------------------------------------------------------------- exact cuts

@pytest.mark.parametrize("channels", [2, 1])
def test_a_stretch_is_exactly_the_samples_of_the_whole_copy_and_the_two_copies_line_up(client, signed_in, channels):
    """Bit for bit what the whole file decodes to at those positions, for both copies, at frame edges, whole seconds, one sample,
    thirty seconds and the very end; the same numbers cut from the two copies are the same positions."""
    uid, _ = signed_in
    jid = real_job(uid, seconds=75.0, channels=channels)
    total = params_of(jid)["kept"]["noise"]["samples"]
    full = {k: whole(jid, k, channels) for k in ("noise", "input")}
    assert len(full["noise"]) == len(full["input"]) == total == 75 * SR
    assert not np.array_equal(full["noise"], full["input"])
    cases = [(0.0, 10.0), (1.0, 0.5), (4608 / SR, 1.0), (4096 * 7 / SR, 2.0), (59.999979166, 30.0), (12.345678, 7.25), (30.0, 30.0),
             (74.5, 30.0), ((total - 1) / SR, 5.0), (0.0, 1 / SR), (33.333333, 1 / SR), (17.0, 0.0173)]
    for start, seconds in cases:
        first, count = stretch.span(start, seconds, total)
        got = {}
        for which in ("noise", "input"):
            r = stretch_of(client, jid, which, f"{start:.9f}", f"{seconds:.9f}")
            assert r.status_code == 200, (start, seconds, r.text)
            assert r.headers["content-type"] == "audio/flac"
            assert (int(r.headers["x-stretch-start"]), int(r.headers["x-stretch-samples"]), int(r.headers["x-total-samples"])) == (first, count, total)
            got[which] = decode(r.content, channels)
            assert got[which].shape == (count, channels), (start, seconds, got[which].shape)
            assert np.array_equal(got[which], full[which][first:first + count]), (which, start, seconds)
        assert got["noise"].shape == got["input"].shape


def test_many_random_stretches_match_the_whole_copy(client, signed_in):
    """Any position and length: the rounding of `start` to a sample is the only thing between the request and the samples."""
    uid, _ = signed_in
    jid = real_job(uid, seconds=40.0, channels=2)
    total = params_of(jid)["kept"]["noise"]["samples"]
    full = whole(jid, "noise", 2)
    rng = np.random.default_rng(7)
    for _ in range(25):
        first = int(rng.integers(0, total))
        count = int(rng.choice([1, 7, 4096, 4608, 48_000, 123_457, 480_000, 1_440_000]))
        r = stretch_of(client, jid, "noise", f"{first / SR:.7f}", f"{count / SR:.7f}")
        assert r.status_code == 200
        got = decode(r.content, 2)
        want = full[first:first + count]
        assert got.shape == want.shape and np.array_equal(got, want), (first, count)


def test_a_loud_copy_is_cut_as_it_is_kept(client, signed_in):
    """The scale a loud sample was brought down by stays in the job's record; the stretch holds the kept values (the page divides)."""
    uid, _ = signed_in
    jid = make_done_job(uid, seconds=12.0, channels=1)
    y, x = signals(12.0, 1)
    y = y * 6.0  # louder than full scale: kept.save scales it into the file
    params = params_of(jid)
    for key, name, arr in (("noise", "kept-noise", y), ("input", "kept-input", x)):
        e = kept.save(ff(), arr, None, folder(jid) / name)
        params["kept"][key].update({"channels": e["channels"], "scale": e["scale"], "samples": e["samples"]})
    db.execute("UPDATE jobs SET params_json = ? WHERE id = ?", (__import__("json").dumps(params), jid))
    scale = params["kept"]["noise"]["scale"]
    assert scale < 1.0
    r = stretch_of(client, jid, "noise", 3, 4)
    got = decode(r.content, 1)
    assert np.array_equal(got, whole(jid, "noise", 1)[3 * SR:7 * SR])
    assert np.max(np.abs(got)) <= kept.CEILING + 1e-6
    shown = client.get(f"/api/jobs/{jid}").json()["redo"]["preview"]
    assert shown["scale"]["noise"] == scale
    assert abs(float(np.max(np.abs(y[3 * SR:7 * SR]))) - float(np.max(np.abs(got))) / scale) < 1e-4


# ------------------------------------------------------------------------------------------- bounds

def test_the_length_of_a_stretch_is_capped_and_cut_short_at_the_end(client, signed_in):
    uid, _ = signed_in
    jid = real_job(uid, seconds=75.0, channels=1)
    r = stretch_of(client, jid, "noise", 0, 500)
    assert int(r.headers["x-stretch-samples"]) == 30 * SR and decode(r.content, 1).shape == (30 * SR, 1)
    r = stretch_of(client, jid, "input", 70, 30)
    assert int(r.headers["x-stretch-samples"]) == 5 * SR and decode(r.content, 1).shape == (5 * SR, 1)
    r = stretch_of(client, jid, "noise", 10)  # only start: the longest allowed
    assert int(r.headers["x-stretch-samples"]) == 30 * SR and int(r.headers["x-stretch-start"]) == 10 * SR
    r = stretch_of(client, jid, "noise", seconds=2)  # only seconds: from the start
    assert (int(r.headers["x-stretch-start"]), int(r.headers["x-stretch-samples"])) == (0, 2 * SR)


def test_a_start_past_the_end_or_a_nonsense_number_is_refused(client, signed_in):
    uid, _ = signed_in
    jid = real_job(uid, seconds=10.0, channels=1)
    for start in (10, 10.5, 999):
        r = stretch_of(client, jid, "noise", start, 1)
        assert r.status_code == 400 and "past the end" in r.json()["detail"], start
    for q in ("start=-1", "seconds=0", "seconds=-3", "start=abc", "seconds=", "start=nan", "start=inf", "seconds=inf", "seconds=1e12", "start=1e12"):
        r = client.get(f"/api/jobs/{jid}/kept/noise?{q}")
        assert r.status_code == 422, (q, r.status_code)
    assert client.get(f"/api/jobs/{jid}/kept/noise?start=0&seconds=1").status_code == 200


def test_without_start_or_seconds_the_whole_copy_comes_as_before(client, signed_in):
    uid, _ = signed_in
    jid = real_job(uid, seconds=8.0, channels=2)
    for which in ("noise", "input"):
        r = client.get(f"/api/jobs/{jid}/kept/{which}")
        assert r.status_code == 200 and r.headers["content-type"] == "audio/flac"
        assert r.content == (folder(jid) / f"kept-{which}.flac").read_bytes()
        assert "x-stretch-start" not in r.headers


def test_the_answers_are_private_and_not_stored(client, signed_in):
    uid, _ = signed_in
    jid = real_job(uid, seconds=8.0, channels=1)
    r = stretch_of(client, jid, "noise", 1, 2)
    assert r.headers["cache-control"] == "private, no-store" and r.headers["x-content-type-options"] == "nosniff"
    assert r.content.startswith(b"fLaC") and 0 < len(r.content) < 2 * SR * 4  # a few seconds of 24-bit FLAC is small


# ------------------------------------------------------------------------------------------- who may ask

def test_only_the_owner_and_only_while_the_files_exist(client, signed_in, other_browser, clock):
    uid, _ = signed_in
    other, other_uid, _ = other_browser
    jid = real_job(uid, seconds=10.0, channels=1)
    assert stretch_of(client, jid, "noise", 0, 1).status_code == 200
    assert stretch_of(other, jid, "noise", 0, 1).status_code == 404  # another account sees the same as "no such job"
    assert stretch_of(other, jid, "input", 0, 1).status_code == 404
    from fastapi.testclient import TestClient

    from acw.app import create_app

    with TestClient(create_app(), base_url=ORIGIN) as anon:
        assert stretch_of(anon, jid, "noise", 0, 1).status_code == 401
    assert stretch_of(client, "0" * 32, "noise", 0, 1).status_code == 404
    assert stretch_of(client, "short", "noise", 0, 1).status_code == 404
    for bad in ("rebuilt", "kept-noise.flac", "NOISE", "x"):
        assert stretch_of(client, jid, bad, 0, 1).status_code == 404, bad
    # past the 2 hours the sweep has deleted the files; before it has run the job still answers
    clock.advance(2 * 3600 + 5)
    jobs.sweep()
    assert row(jid)["status"] == "expired" and not folder(jid).exists()
    assert stretch_of(client, jid, "noise", 0, 1).status_code == 404
    assert stretch_of(client, jid, "input", 0, 1).status_code == 404


def test_a_job_that_is_not_done_or_has_no_copies_gives_nothing(client, signed_in):
    uid, _ = signed_in
    rebuild = make_done_job(uid, denoise_tier="rebuild", rebuilt=True)
    leveling_only = make_done_job(uid, denoise_tier="off")
    no_input = real_job(uid, seconds=6.0, channels=1)
    os.unlink(folder(no_input) / "kept-input.flac")
    running = real_job(uid, seconds=6.0, channels=1)
    db.execute("UPDATE jobs SET status = 'running' WHERE id = ?", (running,))
    for jid in (rebuild, leveling_only, no_input, running):
        for which in ("noise", "input"):
            assert stretch_of(client, jid, which, 0, 1).status_code == 404, (jid, which)


def test_a_link_in_place_of_a_copy_is_never_followed(client, signed_in, tmp_path):
    uid, _ = signed_in
    jid = real_job(uid, seconds=6.0, channels=1)
    secret = tmp_path / "secret.flac"
    secret.write_bytes((folder(jid) / "kept-input.flac").read_bytes())
    os.unlink(folder(jid) / "kept-input.flac")
    os.symlink(secret, folder(jid) / "kept-input.flac")
    assert stretch_of(client, jid, "input", 0, 1).status_code == 404
    assert stretch_of(client, jid, "noise", 0, 1).status_code == 404  # no preview without both copies


def test_an_api_key_of_the_account_can_ask_another_account_and_a_key_without_the_jobs_scope_cannot(client, signed_in, other_browser):
    from fastapi.testclient import TestClient

    from acw.app import create_app

    uid, _ = signed_in
    _, other_uid, _ = other_browser
    jid = real_job(uid, seconds=10.0, channels=1)
    _, mine = api_key(uid, "jobs")
    _, stream_only = api_key(uid, "stream")
    _, theirs = api_key(other_uid, "jobs")
    with TestClient(create_app(), base_url=ORIGIN) as keyed:  # no cookies: only the key speaks
        ok = stretch_of(keyed, jid, "noise", 1, 2, headers=mine)
        assert ok.status_code == 200 and int(ok.headers["x-stretch-samples"]) == 2 * SR
        assert stretch_of(keyed, jid, "input", 1, 2, headers=mine).status_code == 200
        assert stretch_of(keyed, jid, "noise", 1, 2, headers=stream_only).status_code == 403
        assert stretch_of(keyed, jid, "noise", 1, 2, headers=theirs).status_code == 404  # another account's key
        assert stretch_of(keyed, jid, "noise", 1, 2, headers={"Authorization": "Bearer nonsense"}).status_code == 401
        assert keyed.get(f"/api/jobs/{jid}/kept/noise", headers=mine).status_code == 200  # the whole copy too


# ------------------------------------------------------------------------------------------- limits and failures

def test_stretches_have_a_rate_limit_of_their_own_and_the_whole_copy_does_not_use_it(client, signed_in, monkeypatch):
    uid, _ = signed_in
    jid = real_job(uid, seconds=6.0, channels=1)
    assert LIMITS["kept_cut"][0] >= 60  # the real limit leaves room for seeking about
    monkeypatch.setitem(LIMITS, "kept_cut", (3, 0.001))
    assert [stretch_of(client, jid, "noise", 0, 1).status_code for _ in range(3)] == [200, 200, 200]
    r = stretch_of(client, jid, "noise", 0, 1)
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert client.get(f"/api/jobs/{jid}/kept/noise").status_code == 200  # no start / seconds: not a cut


def test_when_every_place_is_taken_the_answer_is_to_come_back_in_a_moment(client, signed_in, monkeypatch):
    uid, _ = signed_in
    jid = real_job(uid, seconds=6.0, channels=1)
    monkeypatch.setattr(stretch, "_places", threading.BoundedSemaphore(1))
    monkeypatch.setattr(stretch, "WAIT_SECONDS", 0.2)
    assert stretch._places.acquire()
    r = stretch_of(client, jid, "noise", 0, 1)
    assert r.status_code == 503 and r.headers["retry-after"] == "2"
    stretch._places.release()
    assert stretch_of(client, jid, "noise", 0, 1).status_code == 200


def test_a_damaged_copy_is_a_plain_error_and_frees_its_place(client, signed_in):
    uid, _ = signed_in
    jid = real_job(uid, seconds=6.0, channels=1)
    (folder(jid) / "kept-noise.flac").write_bytes(b"this is not audio" * 500)
    r = stretch_of(client, jid, "noise", 0, 1)
    assert r.status_code == 500 and "Couldn" in r.json()["detail"]
    assert stretch_of(client, jid, "input", 0, 1).status_code == 200
    for _ in range(stretch.CONCURRENT_CUTS + 1):  # nothing leaks a place
        assert stretch_of(client, jid, "noise", 0, 1).status_code == 500


def test_the_cutter_is_confined_and_writes_nothing(tmp_path):
    """It is run the way stretch.cut runs it; from inside, the folder is read-only and the rest of the disk can not be read."""
    jid_dir = tmp_path / "job"
    jid_dir.mkdir()
    secret = tmp_path / "other" / "secret.txt"
    secret.parent.mkdir()
    secret.write_text("secret")
    code = (
        "import sys, os\n"
        "from pathlib import Path\n"
        "from acw.worker import sandbox\n"
        f"sandbox.apply_landlock(sandbox.default_read_paths([{str(HOME / 'bin')!r}, {str(jid_dir)!r}]), ['/dev/null'])\n"
        "out = []\n"
        f"for p, mode in (({str(jid_dir / 'new.txt')!r}, 'w'), ({str(secret)!r}, 'r')):\n"
        "    try:\n"
        "        open(p, mode).close(); out.append('opened')\n"
        "    except PermissionError:\n"
        "        out.append('denied')\n"
        "print(','.join(out))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=stretch._environment(), cwd=str(jid_dir))
    assert r.stdout.strip() == "denied,denied", r.stderr
    assert not (jid_dir / "new.txt").exists()
    # and the program itself refuses nonsense
    for bad in ([], ["x", "relative/path.flac", "0", "10"], [str(HOME / "bin"), str(jid_dir / "a.flac"), "-1", "10"], [str(HOME / "bin"), str(jid_dir / "a.flac"), "0", "0"]):
        p = subprocess.run([sys.executable, "-m", "acw.worker.cutter", *bad], capture_output=True, env=stretch._environment(), cwd=str(jid_dir))
        assert p.returncode == 2 and p.stdout == b""


def test_a_long_recording_is_cut_as_quickly_as_a_short_one(client, signed_in, tmp_path):
    """Twenty minutes of mono (more than the page ever held whole): the stretch from near the end is the same size and exact."""
    uid, _ = signed_in
    seconds = 1200
    jid = make_done_job(uid, seconds=float(seconds), channels=1)
    d = folder(jid)
    params = params_of(jid)
    for key, name, expr in (("noise", "kept-noise", "sine=f=330:r=48000"), ("input", "kept-input", "anoisesrc=c=pink:a=0.1:r=48000")):
        subprocess.run([str(HOME / "bin/ffmpeg"), "-v", "error", "-f", "lavfi", "-i", f"{expr}:d={seconds}", "-ac", "1", *AUDIO_FORMATS["flac24"][1], "-y", str(d / f"{name}.flac")], check=True)
        params["kept"][key].update({"channels": [0], "scale": 1.0, "samples": seconds * SR})
    params["kept"]["noise"]["plan"] = [0]
    db.execute("UPDATE jobs SET params_json = ? WHERE id = ?", (__import__("json").dumps(params), jid))
    shown = client.get(f"/api/jobs/{jid}").json()["redo"]["preview"]
    assert shown["samples"] == seconds * SR and shown["stretchMaxSeconds"] == 30
    full = {k: whole(jid, k, 1) for k in ("noise", "input")}
    for start in (0.0, 600.25, 1169.999, 1199.5):
        for which in ("noise", "input"):
            r = stretch_of(client, jid, which, f"{start:.6f}", 10)
            first, count = int(r.headers["x-stretch-start"]), int(r.headers["x-stretch-samples"])
            got = decode(r.content, 1)
            assert np.array_equal(got, full[which][first:first + count]), (which, start)
    assert count == 48_000 // 2 and len(r.content) < 400_000
