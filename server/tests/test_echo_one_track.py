"""Echo removal on a file with one audio track (acw/worker/main.py, editing.py, rerun.py), without the native pieces: the whole job fails as soon as
the file is read, before any work, uncharged; a re-run asking for it on such a file is refused. The real files are in test_job_echo.py."""
import json

import pytest

from acw import db, editing, jobs
from acw.billing import ledger
from acw.worker import main as worker

from .rerun_support import count, make_done_job, params_of, roomy_disk  # noqa: F401 - fixtures
from .test_automatic_settings import Stages

pytestmark = pytest.mark.usefixtures("roomy_disk")


class EchoStages(Stages):
    """The runner's stages, replaced: the probe says how many audio tracks the file has (None: it does not say)."""

    def __init__(self, monkeypatch, tracks, duration=60.0):
        super().__init__(monkeypatch, None, duration=duration)
        self.tracks = tracks

    def __call__(self, job, d, stage, timeout):
        out = super().__call__(job, d, stage, timeout)
        if stage == "probe" and self.tracks is not None:
            out["audio_tracks"] = self.tracks
        return out


def _start(client, h, kind="denoise", seconds=60, **body):
    tiers = {"denoiseTier": "standard"} if kind != "transcribe" and "timeline" not in body else {}
    if kind in ("transcribe", "denoise_transcribe"):
        tiers["transcribeTier"] = "local"
    r = client.post("/api/jobs", json={"kind": kind, "filename": "rec.mkv", "sizeBytes": 3000, "durationSeconds": seconds, **tiers, **body}, headers=h)
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    assert client.put(f"/api/jobs/{jid}/input", content=bytes(3000), headers=h).status_code == 200
    return jid


def _take():
    job = worker.claim("test")
    worker.process(job, worker._settings_extra())
    return job["id"]


def _params(jid):
    return json.loads(db.one("SELECT params_json FROM jobs WHERE id = ?", (jid,))["params_json"])


@pytest.mark.parametrize("kind", ["denoise", "denoise_transcribe"])
def test_one_track_found_when_the_file_is_read_fails_the_job_before_any_work_uncharged(client, signed_in, monkeypatch, kind):
    uid, h = signed_in
    ledger.credit(uid, 2_000_000, "grant", "t", "g")
    stages = EchoStages(monkeypatch, tracks=1)
    jid = _start(client, h, kind=kind, options={"echo_reference_track": True})
    assert ledger.balance(uid)["held_micros"] > 0
    _take()
    job = client.get(f"/api/jobs/{jid}", headers=h).json()
    assert job["status"] == "failed" and job["error"] == jobs.NO_SECOND_TRACK and not job["costMicros"]
    assert stages.stages == ["probe"]  # (no measuring, no main stage)
    assert ledger.balance(uid) == {"balance_micros": 2_000_000, "held_micros": 0, "available_micros": 2_000_000}
    assert "audio_tracks" not in _params(jid)


def test_a_timeline_with_echo_parts_fails_the_same_way_and_one_without_them_does_not(client, signed_in, monkeypatch):
    uid, h = signed_in
    ledger.credit(uid, 2_000_000, "grant", "t", "g")
    layers = {"noise": [{"start": 0, "end": 60, "strength_db": 40, "remover": "light"}]}
    stages = EchoStages(monkeypatch, tracks=1)
    jid = _start(client, h, timeline={"version": 1, "seconds": 60, "layers": {**layers, "echo": [{"start": 0, "end": 30}]}})
    _take()
    failed = client.get(f"/api/jobs/{jid}", headers=h).json()
    assert failed["status"] == "failed" and failed["error"] == jobs.NO_SECOND_TRACK and stages.stages == ["probe"]
    assert ledger.balance(uid)["held_micros"] == 0 and ledger.balance(uid)["balance_micros"] == 2_000_000
    # no echo part: the single track is no reason to fail, and it is remembered
    jid = _start(client, h, timeline={"version": 1, "seconds": 60, "layers": layers})
    monkeypatch.setattr(worker, "_complete", lambda *a, **k: None)
    _take()
    assert "main" in stages.stages and _params(jid)["audio_tracks"] == 1


def test_two_tracks_unknown_tracks_and_no_echo_all_go_on(client, signed_in, monkeypatch):
    uid, h = signed_in
    ledger.credit(uid, 2_000_000, "grant", "t", "g")
    for tracks, echo in ((2, True), (3, True), (None, True), (1, False), (2, False), (None, False)):
        stages = EchoStages(monkeypatch, tracks=tracks)
        jid = _start(client, h, options={"echo_reference_track": echo})
        monkeypatch.setattr(worker, "_complete", lambda *a, **k: None)
        _take()
        assert "main" in stages.stages, (tracks, echo)
        assert ("audio_tracks" in _params(jid)) == (tracks is not None) and _params(jid).get("audio_tracks") == tracks
        db.execute("DELETE FROM jobs WHERE id = ?", (jid,))


@pytest.mark.parametrize("tracks", ["2", True, -1, 1.5, [2], {"n": 2}])
def test_an_audio_track_count_that_makes_no_sense_fails_the_job_uncharged(client, signed_in, monkeypatch, tracks):
    uid, h = signed_in
    ledger.credit(uid, 2_000_000, "grant", "t", "g")
    EchoStages(monkeypatch, tracks=tracks)
    jid = _start(client, h)
    _take()
    job = client.get(f"/api/jobs/{jid}", headers=h).json()
    assert job["status"] == "failed" and job["error"] == "Processing failed; you were not charged." and not job["costMicros"]
    assert ledger.balance(uid)["balance_micros"] == 2_000_000 and ledger.balance(uid)["held_micros"] == 0


def test_a_job_that_wants_echo_removal_is_known_from_its_options_or_from_the_timeline_it_was_made_with():
    assert editing.wants_echo({"options": {"echo_reference_track": True}}) is True
    assert editing.wants_echo({"options": {"echo_reference_track": False}}) is False
    assert editing.wants_echo({"options": {}}) is False and editing.wants_echo({}) is False
    first = {"edit": {"first_run": True, "timeline": {"version": 1, "seconds": 60, "layers": {"echo": [{"start": 0, "end": 10}]}}}}
    assert editing.wants_echo({"options": {}, **first}) is True
    assert editing.wants_echo({"options": {}, "edit": {"first_run": True, "timeline": {"version": 1, "seconds": 60, "layers": {"noise": [{"start": 0, "end": 10}]}}}}) is False
    assert editing.wants_echo({"options": {}, "edit": {"first_run": True, "timeline": {"version": 1, "seconds": 60, "layers": {"echo": []}}}}) is False
    # (the timeline of a job that is not a first run is its saved one, not what it asked for)
    assert editing.wants_echo({"options": {}, "edit": {"timeline": first["edit"]["timeline"]}}) is False


def test_a_file_is_known_to_have_one_track_from_the_probe_or_the_editors_facts():
    assert editing.one_track({"audio_tracks": 1}) is True
    assert editing.one_track({"edit": {"facts": {"reference": False}}}) is True
    assert editing.one_track({"audio_tracks": 1, "edit": {"facts": {"reference": True}}}) is True
    assert editing.one_track({"audio_tracks": 2}) is False and editing.one_track({"audio_tracks": 3}) is False
    assert editing.one_track({"edit": {"facts": {"reference": True}}}) is False
    assert editing.one_track({}) is False and editing.one_track({"edit": {}}) is False and editing.one_track({"audio_tracks": None}) is False


# ------------------------------------------------------------------------------------------------------- re-runs

def _source(uid, **changes):
    jid = make_done_job(uid, seconds=60.0)
    p = params_of(jid)
    p.update(changes)
    db.execute("UPDATE jobs SET params_json = ? WHERE id = ?", (json.dumps(p), jid))
    return jid


@pytest.mark.parametrize("changes", [{"audio_tracks": 1}, {"edit": {"facts": {"reference": False}}}, {"audio_tracks": 1, "edit": {"facts": {"reference": False}}}])
def test_a_re_run_asking_for_echo_removal_on_a_file_known_to_have_one_track_is_refused(client, signed_in, changes):
    uid, h = signed_in
    ledger.credit(uid, 2_000_000, "grant", "t", "g")
    src = _source(uid, **changes)
    ask = {"options": {"echo_reference_track": True}}
    for suffix in ("/estimate", ""):
        r = client.post(f"/api/jobs/{src}/rerun{suffix}", json=ask, headers=h)
        assert r.status_code == 400 and r.json()["detail"] == jobs.NO_SECOND_TRACK, (suffix, r.text)
    assert count("jobs") == 1 and ledger.balance(uid)["held_micros"] == 0
    # without echo removal the same job can still be changed
    r = client.post(f"/api/jobs/{src}/rerun/estimate", json={"options": {"atten_lim_db": 24}}, headers=h)
    assert r.status_code == 200, r.text


@pytest.mark.parametrize("changes", [{}, {"audio_tracks": 2}, {"audio_tracks": 3}, {"edit": {"facts": {"reference": True}}}])
def test_a_re_run_may_ask_for_echo_removal_when_the_file_has_a_second_track_or_nothing_is_known(client, signed_in, changes):
    uid, h = signed_in
    ledger.credit(uid, 2_000_000, "grant", "t", "g")
    src = _source(uid, **changes)
    r = client.post(f"/api/jobs/{src}/rerun/estimate", json={"options": {"echo_reference_track": True}}, headers=h)
    assert r.status_code == 200, r.text
    assert [x["item"] for x in r.json()["lines"]][:1] == ["denoise"] and any(x["item"] == "echo" for x in r.json()["lines"])
