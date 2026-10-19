"""Two-hour retention of a finished job's files and everything around it, without native binaries: when a job's expiry is
set, what the sweep deletes, what the job list shows, "Delete now" for a whole chain of re-runs, the `redo` block of a
job and the kept-copy downloads the strength slider uses."""
import json
import os

import pytest

from acw import db, jobs
from acw.billing import ledger

from .rerun_support import (RETENTION, account, add_job, api_key, clock, complete_like_the_worker, count, file_bytes,  # noqa: F401 - fixtures
                            files_of, folder, kept_record, make_done_job, other_browser, params_of, roomy_disk, row, set_env)  # noqa: F401

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
def test_a_finished_job_keeps_upload_copies_and_results_and_all_expire_together(client, signed_in, clock, monkeypatch, minutes, seconds):
    """expires_at = finished_at + the retention (2 hours, or ACW_JOB_RETENTION_MINUTES); scratch files are deleted at once."""
    if minutes:
        set_env(monkeypatch, ACW_JOB_RETENTION_MINUTES=minutes)
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    jid = new_upload(client, h)
    kept = kept_record(60.0, channels=1)
    for n in ("kept-noise.flac", "kept-input.flac"):
        (folder(jid) / n).write_bytes(b"flac")
    clock.advance(40)
    done = complete_like_the_worker(jid, outputs=["rec-cleaned.flac"], kept=kept, scratch=("decoded.f32", "spec.json", "result-main.json"))
    assert done["status"] == "done" and done["finished_at"] == clock.t
    assert done["expires_at"] == done["finished_at"] + seconds
    # the upload, the kept copies and the result stay together; scratch files and folders the runner made are gone
    assert files_of(jid) == {"input.wav", "kept-noise.flac", "kept-input.flac", "rec-cleaned.flac"}
    shown = client.get(f"/api/jobs/{jid}").json()
    assert shown["expiresAt"] == clock.t + seconds
    assert shown["redo"]["available"] is True and shown["redo"]["noise"] is True and "preview" in shown["redo"]


def test_failed_and_canceled_jobs_keep_no_files_and_the_money_comes_back(client, signed_in):
    """A job that fails or is canceled loses its upload and everything else at once."""
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    failed, canceled = new_upload(client, h), new_upload(client, h)
    assert ledger.balance(uid)["held_micros"] > 0 and folder(failed).is_dir()
    (folder(failed) / "kept-noise.flac").write_bytes(b"x")
    assert jobs.finish_failed(row(failed), "Processing failed; you were not charged.")
    assert client.post(f"/api/jobs/{canceled}/cancel", headers=h).status_code == 200
    for jid, status in ((failed, "failed"), (canceled, "canceled")):
        assert row(jid)["status"] == status and not folder(jid).exists()
    assert ledger.balance(uid)["held_micros"] == 0 and ledger.balance(uid)["balance_micros"] == 5_000_000
    assert "redo" not in client.get(f"/api/jobs/{failed}").json()


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
    assert gone["status"] == "expired" and gone["outputs"] == [] and "redo" not in gone
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
    assert body["limits"] == {"maxActiveJobs": 3, "retentionSeconds": 7200}
    done = next(j for j in body["jobs"] if j["id"] == shown["done"])
    assert done["status"] == "done" and done["expiresAt"] == clock.t - 60 + 7200 and done["redo"]["available"] is True
    assert next(j for j in body["jobs"] if j["id"] == shown["failed"])["expiresAt"] is None


def test_the_job_list_limits_and_window_follow_the_settings(client, signed_in, clock, monkeypatch):
    """limits.maxActiveJobs / retentionSeconds are the configured values, and the listing window is the configured one."""
    uid, _ = signed_in
    set_env(monkeypatch, ACW_MAX_ACTIVE_JOBS=5, ACW_JOB_RETENTION_MINUTES=30)
    inside, outside = make_done_job(uid, finished_ago=1799), make_done_job(uid, finished_ago=1800)
    body = client.get("/api/jobs").json()
    assert body["limits"] == {"maxActiveJobs": 5, "retentionSeconds": 1800}
    assert [j["id"] for j in body["jobs"]] == [inside]
    assert outside not in [j["id"] for j in body["jobs"]]


def test_the_job_list_is_newest_first_and_at_most_fifty(client, signed_in, clock):
    """Up to 50 of the account's jobs, newest first."""
    uid, _ = signed_in
    ids = [add_job(uid, "done", created_ago=1000 - i, finished_ago=500) for i in range(55)]
    got = [j["id"] for j in client.get("/api/jobs").json()["jobs"]]
    assert len(got) == 50 and got == ids[::-1][:50]


# ------------------------------------------------------------------------------------ "Delete now"

def chain_of_three(uid, **kw):
    root = make_done_job(uid, **kw)
    first = make_done_job(uid, source=root, root=root, **kw)
    second = make_done_job(uid, source=first, root=root, **kw)
    return root, first, second


def test_delete_now_removes_every_job_of_the_chain_and_nothing_else(client, signed_in):
    """DELETE on any job of the chain deletes the upload, the kept copies and the results of all of them."""
    uid, h = signed_in
    other = account("b@example.com")
    root, first, second = chain_of_three(uid)
    failed_sibling = add_job(uid, "failed", finished_ago=10, root=root, source=root)
    unrelated = make_done_job(uid)
    strangers = chain_of_three(other)
    assert all(folder(j).is_dir() for j in (root, first, second))
    r = client.delete(f"/api/jobs/{first}", headers=h)  # the middle one
    assert r.status_code == 200 and r.json() == {"ok": True}
    for jid in (root, first, second):
        assert row(jid)["status"] == "expired" and row(jid)["outputs_json"] is None and not folder(jid).exists(), jid
        assert client.get(f"/api/jobs/{jid}").json()["status"] == "expired"
    assert row(failed_sibling)["status"] == "failed"
    assert row(unrelated)["status"] == "done" and folder(unrelated).is_dir()
    assert all(row(j)["status"] == "done" and folder(j).is_dir() for j in strangers)
    assert not {root, first, second} & {j["id"] for j in client.get("/api/jobs").json()["jobs"]}


def test_deleting_the_root_job_also_takes_the_reruns(client, signed_in):
    """The chain is the same whichever job is asked for."""
    uid, h = signed_in
    root, first, second = chain_of_three(uid)
    assert client.delete(f"/api/jobs/{root}", headers=h).status_code == 200
    assert {row(j)["status"] for j in (root, first, second)} == {"expired"}
    assert count("jobs", "status = 'done'") == 0


@pytest.mark.parametrize("state", ["uploading", "queued", "running", "cloud_asr"])
def test_delete_is_refused_while_any_job_of_the_chain_is_in_progress(client, signed_in, state):
    """409 and nothing deleted (not even the finished jobs) until the active one is done or canceled."""
    uid, h = signed_in
    root = make_done_job(uid)
    busy = add_job(uid, state, root=root, source=root)
    for jid in (root, busy):
        r = client.delete(f"/api/jobs/{jid}", headers=h)
        assert r.status_code == 409, (jid, r.text)
    assert row(root)["status"] == "done" and folder(root).is_dir() and row(root)["outputs_json"]
    db.execute("UPDATE jobs SET status = 'canceled', finished_at = ? WHERE id = ?", (db.now(), busy))
    assert client.delete(f"/api/jobs/{root}", headers=h).status_code == 200
    assert row(root)["status"] == "expired" and not folder(root).exists()


def test_delete_of_another_accounts_job_or_a_missing_one_is_a_404_and_touches_nothing(client, signed_in, other_browser):
    """Jobs are looked up by id and account."""
    uid, h = signed_in
    _, other_uid, other_h = other_browser
    mine = chain_of_three(uid)
    theirs = chain_of_three(other_uid)
    assert client.delete(f"/api/jobs/{theirs[0]}", headers=h).status_code == 404
    assert client.delete(f"/api/jobs/{'0' * 32}", headers=h).status_code == 404
    assert client.delete("/api/jobs/short", headers=h).status_code == 404
    assert all(row(j)["status"] == "done" and folder(j).is_dir() for j in (*mine, *theirs))


def test_after_delete_now_the_job_cannot_be_downloaded_kept_or_changed_again(client, signed_in):
    """Everything about a deleted job answers as gone."""
    uid, h = signed_in
    jid = make_done_job(uid)
    assert client.get(f"/api/jobs/{jid}/kept/noise").status_code == 200
    assert client.delete(f"/api/jobs/{jid}", headers=h).status_code == 200
    assert client.get(f"/api/jobs/{jid}/kept/noise").status_code == 404
    assert client.get(f"/api/jobs/{jid}/outputs/talk-cleaned.flac").status_code == 404
    r = client.post(f"/api/jobs/{jid}/rerun", json={"options": {"atten_lim_db": 24}}, headers=h)
    assert r.status_code == 409 and r.json()["code"] == "job_files_gone" and r.headers["X-Error-Code"] == "job_files_gone"


# ----------------------------------------------------------------------------------- the `redo` block

def redo_of(client, jid):
    return client.get(f"/api/jobs/{jid}").json()["redo"]


def test_redo_of_a_job_with_the_noise_removal_and_the_recording_kept(client, signed_in):
    """available, noise, preview with what the page needs to blend (sizes of the real files)."""
    uid, _ = signed_in
    jid = make_done_job(uid, seconds=60.0)
    redo = redo_of(client, jid)
    sizes = {k: (folder(jid) / f"kept-{k}.flac").stat().st_size for k in ("noise", "input")}
    assert redo == {"available": True, "noise": True, "rebuilt": False, "transcript": False, "translations": [],
                    "preview": {"samples": 60 * 48_000, "channels": [0, 1], "plan": [0, 1], "scale": {"noise": 1.0, "input": 1.0}, "bytes": sizes,
                                "stretchMaxSeconds": 30}}


def test_redo_lists_the_transcript_and_its_translations(client, signed_in):
    """transcript is true while segments.json exists; translations are the kept languages."""
    uid, _ = signed_in
    jid = make_done_job(uid, kind="denoise_transcribe", transcribe_tier="local",
                        segments={"translations": ["fr", "es"]}, options={"language": "en"})
    redo = redo_of(client, jid)
    assert redo["transcript"] is True and redo["translations"] == ["fr", "es"] and redo["noise"] is True
    (folder(jid) / "segments.json").unlink()
    assert redo_of(client, jid)["transcript"] is False
    only_text = make_done_job(uid, kind="transcribe", denoise_tier=None, transcribe_tier="local", segments={})
    redo = redo_of(client, only_text)
    assert (redo["available"], redo["noise"], redo["rebuilt"], redo["transcript"]) == (True, False, False, True)
    assert "preview" not in redo


def test_the_preview_needs_the_noise_removal_and_the_recording_and_standard_denoising(client, signed_in):
    """preview only with both copies kept and a Denoising (standard) job; a rebuild or a missing copy has none."""
    uid, _ = signed_in
    no_input = make_done_job(uid, input_copy=False)
    redo = redo_of(client, no_input)
    assert redo["noise"] is True and "preview" not in redo
    rebuild = make_done_job(uid, denoise_tier="rebuild", rebuilt=True)
    redo = redo_of(client, rebuild)
    assert (redo["noise"], redo["rebuilt"]) == (False, True) and "preview" not in redo
    leveling_only = make_done_job(uid, denoise_tier="off")
    redo = redo_of(client, leveling_only)
    assert redo["available"] is True and redo["noise"] is False and redo["rebuilt"] is False and "preview" not in redo
    # a copy named in the record but missing (or not a regular file) does not count
    gone = make_done_job(uid)
    (folder(gone) / "kept-input.flac").unlink()
    redo = redo_of(client, gone)
    assert redo["noise"] is True and "preview" not in redo
    (folder(gone) / "kept-noise.flac").unlink()
    os.symlink(folder(no_input) / "kept-noise.flac", folder(gone) / "kept-noise.flac")
    assert redo_of(client, gone)["noise"] is False


def test_redo_is_unavailable_without_the_upload_and_absent_for_jobs_that_are_not_done(client, signed_in):
    """{"available": false} when the upload is gone; no block at all unless the job is done."""
    uid, _ = signed_in
    jid = make_done_job(uid)
    (folder(jid) / "input.wav").unlink()
    assert redo_of(client, jid) == {"available": False}
    queued, expired = add_job(uid, "queued"), make_done_job(uid, status="expired")
    for other in (queued, expired):
        assert "redo" not in client.get(f"/api/jobs/{other}").json()
    listed = {j["id"]: j for j in client.get("/api/jobs").json()["jobs"]}
    assert "redo" not in listed[queued] and listed[jid]["redo"] == {"available": False}


# ----------------------------------------------------------------------- the kept copies for the slider

def test_the_owner_can_download_both_kept_copies_as_flac(client, signed_in):
    """GET /kept/noise and /kept/input: the files themselves, audio/flac, not cached."""
    uid, h = signed_in
    jid = make_done_job(uid)
    for which in ("noise", "input"):
        r = client.get(f"/api/jobs/{jid}/kept/{which}")
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "audio/flac"
        assert r.content == file_bytes(jid, f"kept-{which}.flac")
        assert r.headers["cache-control"] == "private, no-store" and r.headers["x-content-type-options"] == "nosniff"


def test_a_key_can_read_the_kept_copies_of_a_job_with_the_jobs_scope_only(client, signed_in):
    """The route takes the same sign-in as the other job routes: an API key needs the jobs scope."""
    uid, _ = signed_in
    jid = make_done_job(uid)
    _, wrong = api_key(uid, "stream")
    _, right = api_key(uid, "jobs")
    client.cookies.clear()
    assert client.get(f"/api/jobs/{jid}/kept/noise", headers=wrong).status_code == 403
    assert client.get(f"/api/jobs/{jid}/kept/noise", headers=right).status_code == 200
    assert client.get(f"/api/jobs/{jid}/kept/noise").status_code == 401


def test_the_kept_copies_are_not_given_to_another_account(client, signed_in, other_browser):
    """404 for someone else's job, whichever copy is asked for; the owner still gets both."""
    uid, _ = signed_in
    jid = make_done_job(uid)
    other_client, _, _ = other_browser
    for which in ("noise", "input"):
        assert other_client.get(f"/api/jobs/{jid}/kept/{which}").status_code == 404
        assert client.get(f"/api/jobs/{jid}/kept/{which}").status_code == 200


def test_the_kept_copies_are_404_for_bad_names_unfinished_jobs_and_after_expiry(client, signed_in, clock):
    """Only noise and input, only while the job is done and its files exist."""
    uid, _ = signed_in
    jid = make_done_job(uid, rebuilt=True)
    for bad in ("rebuilt", "kept-noise.flac", "noise.flac", "%2e%2e", "input%2F..", "NOISE", "", "x"):
        assert client.get(f"/api/jobs/{jid}/kept/{bad}").status_code == 404, bad
    assert client.get(f"/api/jobs/{'0' * 32}/kept/noise").status_code == 404
    assert client.get(f"/api/jobs/{add_job(uid, 'queued')}/kept/noise").status_code == 404
    expiring = make_done_job(uid, expires_in=0)
    assert client.get(f"/api/jobs/{expiring}/kept/noise").status_code == 200  # the sweep has not run yet
    jobs.sweep()
    assert client.get(f"/api/jobs/{expiring}/kept/noise").status_code == 404
    assert client.get(f"/api/jobs/{expiring}/kept/input").status_code == 404


def test_the_kept_copies_are_404_without_a_preview_or_when_the_file_is_a_link(client, signed_in):
    """A job without both copies (a rebuild, one made before the recording was kept) has none to give; a link in their place is never followed."""
    uid, _ = signed_in
    rebuild = make_done_job(uid, denoise_tier="rebuild", rebuilt=True)
    no_recording = make_done_job(uid, input_copy=False)
    for jid in (rebuild, no_recording):
        assert client.get(f"/api/jobs/{jid}/kept/noise").status_code == 404
        assert client.get(f"/api/jobs/{jid}/kept/input").status_code == 404
    planted = make_done_job(uid)
    (folder(planted) / "kept-input.flac").unlink()
    os.symlink(folder(no_recording) / "kept-noise.flac", folder(planted) / "kept-input.flac")
    assert client.get(f"/api/jobs/{planted}/kept/input").status_code == 404
    assert client.get(f"/api/jobs/{planted}/kept/noise").status_code == 404  # no preview without both copies


def _standard_job_with_both_copies(uid):
    return make_done_job(uid, seconds=30.0)


def _tweak(jid, **changes):
    """Change a job's stored parameters (denoise tier, kept record)."""
    params = params_of(jid)
    params.update(changes)
    db.execute("UPDATE jobs SET params_json = ? WHERE id = ?", (json.dumps(params), jid))
    return params


def _drop_from_record(jid, key):
    kept = params_of(jid)["kept"]
    kept.pop(key)
    _tweak(jid, kept=kept)


def _noise_file_gone(jid):
    (folder(jid) / "kept-noise.flac").unlink()


def _input_file_gone(jid):
    (folder(jid) / "kept-input.flac").unlink()


def _input_is_a_link(jid):
    (folder(jid) / "kept-input.flac").unlink()
    os.symlink(folder(jid) / "kept-noise.flac", folder(jid) / "kept-input.flac")


@pytest.mark.parametrize("change, noise, preview", [
    pytest.param(lambda j: None, True, True, id="noise-and-input-standard"),
    pytest.param(_input_file_gone, True, False, id="input-copy-file-missing"),
    pytest.param(lambda j: _drop_from_record(j, "input"), True, False, id="input-copy-not-in-the-record"),
    pytest.param(_input_is_a_link, True, False, id="input-copy-is-a-link"),
    pytest.param(_noise_file_gone, False, False, id="noise-copy-file-missing"),
    pytest.param(lambda j: _drop_from_record(j, "noise"), False, False, id="noise-copy-not-in-the-record"),
    pytest.param(lambda j: _tweak(j, denoise_tier="rebuild"), True, False, id="both-copies-but-a-voice-rebuild-job"),
    pytest.param(lambda j: _tweak(j, denoise_tier="off"), True, False, id="both-copies-but-leveling-only-job"),
])
def test_get_job_has_a_redo_preview_only_with_both_copies_and_standard_denoising(client, signed_in, change, noise, preview):
    """GET /api/jobs/{id} of a done job: `redo.preview` is there exactly when the noise copy and the input copy exist
    and the job's denoiseTier is standard (the list shows the same)."""
    uid, _ = signed_in
    jid = _standard_job_with_both_copies(uid)
    change(jid)
    redo = client.get(f"/api/jobs/{jid}").json()["redo"]
    assert redo["available"] is True and redo["noise"] is noise
    assert ("preview" in redo) is preview
    listed = next(j for j in client.get("/api/jobs").json()["jobs"] if j["id"] == jid)
    assert listed["redo"] == redo
    if preview:
        assert redo["preview"]["samples"] == 30 * 48_000 and set(redo["preview"]) == {"samples", "channels", "plan", "scale", "bytes", "stretchMaxSeconds"}
        assert client.get(f"/api/jobs/{jid}/kept/noise").status_code == client.get(f"/api/jobs/{jid}/kept/input").status_code == 200
    else:
        assert client.get(f"/api/jobs/{jid}/kept/noise").status_code == 404 and client.get(f"/api/jobs/{jid}/kept/input").status_code == 404


def test_a_transcript_only_job_has_no_noise_copy_and_no_preview(client, signed_in):
    """A job without noise removal has nothing for the strength slider."""
    uid, _ = signed_in
    jid = make_done_job(uid, kind="transcribe", denoise_tier=None, transcribe_tier="local", segments={})
    redo = client.get(f"/api/jobs/{jid}").json()["redo"]
    assert redo["noise"] is False and "preview" not in redo and redo["transcript"] is True


def test_the_preview_carries_the_energy_sums_when_the_runner_reported_them_and_only_sane_ones(client, signed_in):
    """The sums that keep a preview's loudness level while the strength moves (kept.energy): passed on as reported, dropped when not numbers."""
    uid, _ = signed_in
    jid = make_done_job(uid, seconds=30.0)
    assert "energy" not in redo_of(client, jid)["preview"]
    good = {"full": 12.5, "input": 20.0, "cross": -3.25}
    for energy, want in ((good, good), ({**good, "cross": 7}, {**good, "cross": 7.0}), ({"full": -1, "input": 2, "cross": 0}, None), ({"full": 1, "input": 2}, None),
                         ({"full": float("inf"), "input": 2, "cross": 0}, None), ("x", None), (None, None)):
        params = params_of(jid)
        params["kept"]["noise"]["energy"] = energy
        db.execute("UPDATE jobs SET params_json = ? WHERE id = ?", (json.dumps(params), jid))
        shown = redo_of(client, jid)["preview"]
        assert shown.get("energy") == want, energy
    from acw import rerun

    entry = {**kept_record(30.0)["noise"], "energy": good, "plan": [0, 1]}
    clean = rerun.sanitize_kept({"noise": entry}, folder(jid))
    assert clean["noise"]["energy"] == good
    assert "energy" not in rerun.sanitize_kept({"noise": {**entry, "energy": {"full": "x", "input": 1, "cross": 1}}}, folder(jid))["noise"]
