"""Two-hour retention of a finished job's files and everything around it, without native binaries: when a job's expiry is set, what the
sweep deletes, what the job list shows, and "Delete now" (per job: there are no chains of jobs made from each other any more)."""
import json  # noqa: F401
import os  # noqa: F401

import pytest

from acw import db, jobs
from acw.billing import ledger

from .job_support import (RETENTION, account, add_job, api_key, clock, complete_like_the_worker, count, files_of, folder,  # noqa: F401 - fixtures
                          make_done_job, other_browser, params_of, roomy_disk, row, set_env)  # noqa: F401

pytestmark = pytest.mark.usefixtures("roomy_disk")


def new_upload(client, h, *, seconds=60, size=3000, **options):
    """A job made through the API with its upload sent: it is queued (no worker runs in these tests)."""
    r = client.post("/api/jobs", json={"kind": "denoise", "denoiseTier": "standard", "filename": "rec.wav", "sizeBytes": size,
                                       "durationSeconds": seconds, "options": {"output_video": False, **options}}, headers=h)
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    assert client.put(f"/api/jobs/{jid}/input", content=bytes(size), headers=h).status_code == 200
    return jid


# ------------------------------------------------------------------------------ when a job's files expire

@pytest.mark.parametrize("minutes, seconds", [(None, 7200), (30, 1800), (5, 300)])
def test_a_finished_job_keeps_its_upload_and_results_and_they_expire_together(client, signed_in, clock, monkeypatch, minutes, seconds):
    """expires_at = finished_at + the retention (2 hours, or ACW_JOB_RETENTION_MINUTES); scratch files are deleted at once."""
    if minutes:
        set_env(monkeypatch, ACW_JOB_RETENTION_MINUTES=minutes)
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    jid = new_upload(client, h)
    clock.advance(40)
    done = complete_like_the_worker(jid, outputs=["rec-cleaned.flac"], scratch=("decoded.f32", "spec.json", "result-main.json"))
    assert done["status"] == "done" and done["finished_at"] == clock.t
    assert done["expires_at"] == done["finished_at"] + seconds
    # the upload and the result stay together; scratch files and folders the runner made are gone
    assert files_of(jid) == {"input.wav", "rec-cleaned.flac"}
    shown = client.get(f"/api/jobs/{jid}").json()
    assert shown["expiresAt"] == clock.t + seconds


def test_failed_and_canceled_jobs_keep_no_files_and_the_money_comes_back(client, signed_in):
    """A job that fails or is canceled loses its upload and everything else at once."""
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    failed, canceled = new_upload(client, h), new_upload(client, h)
    assert ledger.balance(uid)["held_micros"] > 0 and folder(failed).is_dir()
    (folder(failed) / "scratch.f32").write_bytes(b"x")
    assert jobs.finish_failed(row(failed), "Processing failed; you were not charged.")
    assert client.post(f"/api/jobs/{canceled}/cancel", headers=h).status_code == 200
    for jid, status in ((failed, "failed"), (canceled, "canceled")):
        assert row(jid)["status"] == status and not folder(jid).exists()
    assert ledger.balance(uid)["held_micros"] == 0 and ledger.balance(uid)["balance_micros"] == 5_000_000


def test_the_sweep_deletes_a_job_exactly_when_its_expiry_is_reached(client, signed_in, clock):
    """expires_at <= now: folder gone, status expired, results forgotten; one second earlier nothing happens."""
    uid, h = signed_in
    due = make_done_job(uid, expires_in=0, cost=1234)
    later = make_done_job(uid, expires_in=1)
    jobs.sweep()
    assert row(due)["status"] == "expired" and row(due)["outputs_json"] is None and not folder(due).exists()
    assert row(due)["cost_micros"] == 1234  # what was charged is not forgotten
    assert row(later)["status"] == "done" and folder(later).is_dir() and "talk-cleaned.flac" in files_of(later)
    clock.advance(1)
    jobs.sweep()
    assert row(later)["status"] == "expired" and not folder(later).exists()
    gone = client.get(f"/api/jobs/{later}").json()
    assert gone["status"] == "expired" and gone["outputs"] == []
    assert client.get(f"/api/jobs/{later}/outputs/talk-cleaned.flac", headers=h).status_code == 404


def test_the_sweep_shortens_an_expiry_that_was_set_when_the_retention_was_longer(clock, monkeypatch):
    """Retention is a ceiling: an old 24 hour expiry ends at finished_at + retention (and is deleted if that has passed)."""
    uid = account()
    young = make_done_job(uid, finished_ago=100, expires_in=86400 - 100)
    old = make_done_job(uid, finished_ago=7300, expires_in=86400 - 7300)
    jobs.sweep()
    assert row(young)["status"] == "done" and row(young)["expires_at"] == row(young)["finished_at"] + RETENTION
    assert row(old)["status"] == "expired" and not folder(old).exists()
    set_env(monkeypatch, ACW_JOB_RETENTION_MINUTES=30)  # a shorter retention ends the young one too
    clock.advance(1800)
    jobs.sweep()
    assert row(young)["status"] == "expired"


def test_the_sweep_leaves_jobs_in_progress_and_other_jobs_alone(clock):
    """Only done jobs past their expiry are deleted; their neighbours' folders stay."""
    uid = account()
    running = add_job(uid, "running")
    keep = make_done_job(uid)
    drop = make_done_job(uid, expires_in=-5)
    jobs.sweep()
    assert row(running)["status"] == "running"
    assert row(keep)["status"] == "done" and folder(keep).is_dir()
    assert row(drop)["status"] == "expired" and not folder(drop).exists()


# --------------------------------------------------------------------------------------- the job list

def test_the_job_list_shows_what_is_still_within_its_two_hours(client, signed_in, clock):
    """In progress: always. Finished (done, failed, canceled): for 2 hours after they finished, before the sweep too."""
    uid, h = signed_in
    other = account("b@example.com")
    shown = {
        "uploading": add_job(uid, "uploading", created_ago=5 * 3600),
        "queued": add_job(uid, "queued", created_ago=9 * 3600),
        "running": add_job(uid, "running", created_ago=3 * 3600),
        "cloud_asr": add_job(uid, "cloud_asr", created_ago=3 * 3600),
        "done": make_done_job(uid, finished_ago=60),
        "done-7199": make_done_job(uid, finished_ago=7199),
        "failed": add_job(uid, "failed", finished_ago=100),
        "canceled": add_job(uid, "canceled", finished_ago=100),
    }
    hidden = {
        "done-7200": make_done_job(uid, finished_ago=7200),  # exactly two hours: its time is up
        "done-7201": make_done_job(uid, finished_ago=7201),
        "failed-old": add_job(uid, "failed", finished_ago=7300),
        "expired": add_job(uid, "expired", finished_ago=100),
        "someone-else": make_done_job(other, finished_ago=60),
    }
    body = client.get("/api/jobs").json()
    ids = [j["id"] for j in body["jobs"]]
    assert set(ids) == set(shown.values()), {k for k, v in {**shown, **hidden}.items() if (v in ids) != (k in shown)}
    assert not set(ids) & set(hidden.values())
    assert body["limits"] == {"maxActiveJobs": 3, "retentionSeconds": 7200, "maxRetentionSeconds": 21600}
    done = next(j for j in body["jobs"] if j["id"] == shown["done"])
    assert done["status"] == "done" and done["expiresAt"] == clock.t - 60 + 7200
    assert next(j for j in body["jobs"] if j["id"] == shown["failed"])["expiresAt"] is None


def test_the_job_list_limits_and_window_follow_the_settings(client, signed_in, clock, monkeypatch):
    """limits.maxActiveJobs / retentionSeconds are the configured values, and the listing window is the configured one."""
    uid, _ = signed_in
    set_env(monkeypatch, ACW_MAX_ACTIVE_JOBS=5, ACW_JOB_RETENTION_MINUTES=30)
    # (jobs finished under this retention have the deadline it gave them)
    inside, outside = make_done_job(uid, finished_ago=1799, expires_in=1), make_done_job(uid, finished_ago=1800, expires_in=0)
    body = client.get("/api/jobs").json()
    assert body["limits"] == {"maxActiveJobs": 5, "retentionSeconds": 1800, "maxRetentionSeconds": 21600}
    assert [j["id"] for j in body["jobs"]] == [inside]
    assert outside not in [j["id"] for j in body["jobs"]]


def test_the_job_list_is_newest_first_and_at_most_fifty(client, signed_in, clock):
    """Up to 50 of the account's jobs, newest first."""
    uid, _ = signed_in
    ids = [add_job(uid, "done", created_ago=1000 - i, finished_ago=500) for i in range(55)]
    got = [j["id"] for j in client.get("/api/jobs").json()["jobs"]]
    assert len(got) == 50 and got == ids[::-1][:50]


# ------------------------------------------------------------------------------------ "Delete now"

def test_delete_now_removes_the_upload_and_the_results_of_that_job_and_nothing_else(client, signed_in):
    uid, h = signed_in
    other = account("b@example.com")
    one, two = make_done_job(uid), make_done_job(uid)
    stranger = make_done_job(other)
    assert folder(one).is_dir()
    r = client.delete(f"/api/jobs/{one}", headers=h)
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert row(one)["status"] == "expired" and row(one)["outputs_json"] is None and not folder(one).exists()
    assert client.get(f"/api/jobs/{one}").json()["status"] == "expired"
    assert row(two)["status"] == "done" and folder(two).is_dir() and "talk-cleaned.flac" in files_of(two)
    assert row(stranger)["status"] == "done" and folder(stranger).is_dir()
    assert one not in {j["id"] for j in client.get("/api/jobs").json()["jobs"]} and two in {j["id"] for j in client.get("/api/jobs").json()["jobs"]}


@pytest.mark.parametrize("state", ["uploading", "queued", "running", "cloud_asr"])
def test_delete_is_refused_while_the_job_is_in_progress(client, signed_in, state):
    """409 and nothing deleted until it is done or canceled."""
    uid, h = signed_in
    busy = add_job(uid, state)
    other = make_done_job(uid)
    r = client.delete(f"/api/jobs/{busy}", headers=h)
    assert r.status_code == 409, r.text
    assert row(busy)["status"] == state
    # a finished job of the same account is not held up by it
    assert client.delete(f"/api/jobs/{other}", headers=h).status_code == 200 and row(other)["status"] == "expired"
    db.execute("UPDATE jobs SET status = 'canceled', finished_at = ? WHERE id = ?", (db.now(), busy))
    assert client.delete(f"/api/jobs/{busy}", headers=h).status_code == 200


def test_delete_of_another_accounts_job_or_a_missing_one_is_a_404_and_touches_nothing(client, signed_in, other_browser):
    """Jobs are looked up by id and account."""
    uid, h = signed_in
    _, other_uid, other_h = other_browser
    mine, theirs = make_done_job(uid), make_done_job(other_uid)
    assert client.delete(f"/api/jobs/{theirs}", headers=h).status_code == 404
    assert client.delete(f"/api/jobs/{'0' * 32}", headers=h).status_code == 404
    assert client.delete("/api/jobs/short", headers=h).status_code == 404
    assert all(row(j)["status"] == "done" and folder(j).is_dir() for j in (mine, theirs))


def test_after_delete_now_the_job_cannot_be_downloaded_or_changed_again(client, signed_in):
    """Everything about a deleted job answers as gone, and the routes that were removed with the kept copies are not there."""
    uid, h = signed_in
    jid = make_done_job(uid)
    assert client.get(f"/api/jobs/{jid}/outputs/talk-cleaned.flac", headers=h).status_code == 200
    assert client.delete(f"/api/jobs/{jid}", headers=h).status_code == 200
    assert client.get(f"/api/jobs/{jid}/outputs/talk-cleaned.flac", headers=h).status_code == 404
    assert client.get(f"/api/jobs/{jid}/timeline", headers=h).status_code in (404, 409)


def test_the_removed_routes_are_not_there_for_a_live_job_either(client, signed_in):
    uid, h = signed_in
    jid = make_done_job(uid)
    assert client.get(f"/api/jobs/{jid}/kept/noise", headers=h).status_code in (404, 405)
    assert client.get(f"/api/jobs/{jid}/kept/input?start=0&seconds=5", headers=h).status_code in (404, 405)
    for suffix in ("/rerun", "/rerun/estimate"):
        assert client.post(f"/api/jobs/{jid}{suffix}", json={"options": {"atten_lim_db": 24}}, headers=h).status_code in (404, 405)
    shown = client.get(f"/api/jobs/{jid}").json()
    assert "redo" not in shown and "sourceJobId" not in shown
