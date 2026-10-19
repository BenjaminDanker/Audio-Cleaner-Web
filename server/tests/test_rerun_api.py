"""Re-runs ("Change and save again") through the API and the database, without native binaries: who may change which job,
every error code, the estimate against the hold, the new job's row and hard-linked folder, the deletion time a re-run ends
at, the credit history line, the active-job limit and the API key's spending cap. The worker's completion step is run
by hand with the files a runner would have made (rerun_support.complete_like_the_worker)."""
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from acw import db, jobs, rerun
from acw.billing import ledger, pricing
from acw.config import get_settings

from .rerun_support import (GB, account, add_job, api_key, clock, complete_like_the_worker, count, files_of, folder, kept_record,  # noqa: F401
                            make_done_job, other_browser, params_of, roomy_disk, row, same_file, set_env)  # noqa: F401

pytestmark = pytest.mark.usefixtures("roomy_disk")

STRENGTH = {"options": {"atten_lim_db": 24}}  # blend again (and Leveling, which is on by default)
FREE_STRENGTH = {"options": {"atten_lim_db": 24, "leveling": False}}


def post(client, h, jid, estimate=False, **body):
    return client.post(f"/api/jobs/{jid}/rerun" + ("/estimate" if estimate else ""), json=body, headers=h)


def assert_error(r, status, code=None):
    """The status, and the code both in the body and in X-Error-Code."""
    assert r.status_code == status, r.text
    if code:
        assert r.json()["code"] == code and r.headers["X-Error-Code"] == code, (r.json(), dict(r.headers))
    return r.json()


def funded(client, signed_in, micros=5_000_000):
    uid, h = signed_in
    ledger.credit(uid, micros, "grant", "t", "g")
    return uid, h


def jobs_in_folders():
    return {p.name for p in get_settings().jobs_dir.iterdir()} if get_settings().jobs_dir.exists() else set()


# ----------------------------------------------------------------------------------------------- ownership

@pytest.mark.parametrize("estimate", [True, False])
def test_who_may_change_which_job(client, signed_in, other_browser, estimate):
    """The account's browser session can change any job of the account; a key only the jobs it made; anyone else gets a 404."""
    uid, h = funded(client, signed_in)
    other_client, other_uid, other_h = other_browser
    ledger.credit(other_uid, 5_000_000, "grant", "t", "g2")
    k1, kh1 = api_key(uid)
    k2, kh2 = api_key(uid)
    ok1, okh1 = api_key(other_uid)
    by_browser = make_done_job(uid)
    by_k1 = make_done_job(uid, api_key_id=k1)
    #            caller client   headers  job        expected
    matrix = [(client, h, by_browser, 200), (client, h, by_k1, 200),
              (client, kh1, by_k1, 200), (client, kh1, by_browser, 404),
              (client, kh2, by_k1, 404), (client, kh2, by_browser, 404),
              (other_client, other_h, by_browser, 404), (other_client, other_h, by_k1, 404),
              (client, okh1, by_browser, 404), (client, okh1, by_k1, 404)]
    for caller, headers, jid, want in matrix:
        r = post(caller, headers, jid, estimate, **STRENGTH)
        assert r.status_code == want, (headers.get("Authorization", "session")[-6:], jid == by_k1, r.text)
        if want == 404:
            assert r.json() == {"detail": "Job not found.", "code": "job_not_found"} and r.headers["X-Error-Code"] == "job_not_found"
    assert count("jobs") == 2 + (0 if estimate else 3)  # only the three allowed calls made a job
    if not estimate:
        # the key that asked is recorded on the new job; a browser session leaves it empty
        made = {(r_["source_job_id"], r_["api_key_id"]) for r_ in db.all_rows("SELECT * FROM jobs WHERE source_job_id IS NOT NULL")}
        assert made == {(by_browser, None), (by_k1, None), (by_k1, k1)}
        assert {r_["user_id"] for r_ in db.all_rows("SELECT * FROM jobs")} == {uid}


def test_a_key_needs_the_jobs_scope_and_a_browser_needs_its_csrf_token(client, signed_in):
    """Same sign-in rules as the other job routes."""
    uid, h = funded(client, signed_in)
    jid = make_done_job(uid)
    _, stream_only = api_key(uid, "stream")
    _, jobs_key = api_key(uid, "jobs")
    for estimate in (True, False):
        assert post(client, stream_only, jid, estimate, **STRENGTH).status_code == 403
        assert post(client, {"Origin": h["Origin"]}, jid, estimate, **STRENGTH).status_code == 403  # no CSRF token
        assert post(client, jobs_key, jid, estimate, **STRENGTH).status_code == 404  # a valid key, but not the key that made it
    client.cookies.clear()
    assert client.post(f"/api/jobs/{jid}/rerun", json=STRENGTH).status_code == 401
    assert count("jobs") == 1


def test_a_malformed_or_unknown_job_id_is_a_404_job_not_found(client, signed_in):
    """Neither a short id, nor an unknown 32 character one, nor one that is not hexadecimal."""
    funded(client, signed_in)
    for jid in ("abc", "0" * 32, "z" * 32, "a" * 33):
        for estimate in (True, False):
            assert_error(post(client, signed_in[1], jid, estimate, **STRENGTH), 404, "job_not_found")


# ---------------------------------------------------------------------------------------- the error codes

def test_job_files_gone_when_the_job_expired_was_deleted_or_lost_its_upload(client, signed_in):
    """409 job_files_gone for an expired job, a missing folder, a missing upload, and an upload that is a link."""
    uid, h = funded(client, signed_in)
    expired = make_done_job(uid, status="expired")
    no_folder = make_done_job(uid, files=False)
    no_upload = make_done_job(uid)
    os.unlink(folder(no_upload) / "input.wav")
    linked = make_done_job(uid)
    os.unlink(folder(linked) / "input.wav")
    os.symlink(folder(no_upload) / "kept-noise.flac", folder(linked) / "input.wav")
    for jid in (expired, no_folder, no_upload, linked):
        for estimate in (True, False):
            body = assert_error(post(client, h, jid, estimate, **STRENGTH), 409, "job_files_gone")
            assert "cleared away" in body["detail"]
    assert count("jobs", "source_job_id IS NOT NULL") == 0


@pytest.mark.parametrize("status", ["uploading", "queued", "running", "cloud_asr", "failed", "canceled"])
def test_job_not_finished_for_a_job_that_is_not_done(client, signed_in, status):
    """409 job_not_finished: only a finished job can be changed and saved again."""
    uid, h = funded(client, signed_in)
    jid = add_job(uid, status, finished_ago=10 if status in ("failed", "canceled") else None)
    for estimate in (True, False):
        body = assert_error(post(client, h, jid, estimate, **STRENGTH), 409, "job_not_finished")
        assert body["detail"] == "Only a finished job can be changed and saved again."


def test_job_expiring_when_there_is_too_little_time_left_for_the_work(client, signed_in, clock):
    """409 job_expiring: the re-run must be able to finish (twice its rough time and five minutes) before the job is cleared."""
    uid, h = funded(client, signed_in)
    nearly = make_done_job(uid, expires_in=100, seconds=60)
    for estimate in (True, False):
        body = assert_error(post(client, h, nearly, estimate, **STRENGTH), 409, "job_expiring")
        assert "about to be cleared away" in body["detail"]
    roomy = make_done_job(uid, expires_in=3000, seconds=60)
    assert post(client, h, roomy, True, **STRENGTH).status_code == 200
    # the longer the work, the more time it needs: with 30 minutes left a blend of an hour of audio fits, noise removal does not
    hour = make_done_job(uid, expires_in=1800, seconds=3600)
    assert post(client, h, hour, True, options={"atten_lim_db": 24}).status_code == 200
    assert_error(post(client, h, hour, True, options={"noise": "heavy"}), 409, "job_expiring")
    assert_error(post(client, h, hour, False, options={"noise": "heavy"}), 409, "job_expiring")
    assert count("jobs", "source_job_id IS NOT NULL") == 0 and ledger.balance(uid)["held_micros"] == 0


def test_price_changed_when_the_price_is_above_the_quoted_maximum(client, signed_in):
    """maxCostMicros: the quoted price; equal is fine, higher is 409 price_changed and nothing is made or held."""
    uid, h = funded(client, signed_in)
    jid = make_done_job(uid, seconds=3600)  # Leveling for an hour: 60,000 micros
    quoted = post(client, h, jid, True, **STRENGTH).json()["estCostMicros"]
    assert quoted == 60_000
    body = assert_error(post(client, h, jid, maxCostMicros=30_000, **STRENGTH), 409, "price_changed")
    assert "$0.06" in body["detail"] and "$0.03" in body["detail"]
    assert_error(post(client, h, jid, maxCostMicros=0, **STRENGTH), 409, "price_changed")
    assert_error(post(client, h, jid, maxCostMicros=quoted - 1, **STRENGTH), 409, "price_changed")
    assert count("jobs", "source_job_id IS NOT NULL") == 0 and ledger.balance(uid)["held_micros"] == 0
    assert post(client, h, jid, maxCostMicros=quoted, **STRENGTH).status_code == 200
    free = make_done_job(uid)
    assert post(client, h, free, maxCostMicros=0, **FREE_STRENGTH).status_code == 200  # a free re-run is within any quote
    assert post(client, h, make_done_job(uid), maxCostMicros=-1, **STRENGTH).status_code == 422


def test_nothing_to_change_when_the_choices_give_the_files_the_job_already_has(client, signed_in):
    """400 nothing_to_change, before any money is held: same choices, and subtitles dropped from a clean-only job."""
    uid, h = funded(client, signed_in)
    jid = make_done_job(uid)
    same = {"options": {"atten_lim_db": 100, "leveling": True}}
    for estimate in (True, False):
        assert_error(post(client, h, jid, estimate), 400, "nothing_to_change")
        assert_error(post(client, h, jid, estimate, **same), 400, "nothing_to_change")
        assert_error(post(client, h, jid, estimate, denoiseTier="standard"), 400, "nothing_to_change")
    text_only = make_done_job(uid, kind="transcribe", denoise_tier=None, transcribe_tier="local", segments={})
    for estimate in (True, False):
        body = assert_error(post(client, h, text_only, estimate, transcribeTier="off"), 400, "nothing_to_change")
        assert "Choose noise removal" in body["detail"]
    with_subs = make_done_job(uid, kind="denoise_transcribe", transcribe_tier="local", segments={"language": "en"}, options={"language": "en"})
    assert_error(post(client, h, with_subs, transcribeTier="off"), 400, "nothing_to_change")  # the page just leaves them out
    assert count("jobs", "source_job_id IS NOT NULL") == 0 and count("holds") == 0


def test_bad_choices_and_bodies_are_refused_before_anything_is_made(client, signed_in, gpu_up):
    """Unknown fields are a 422; a choice the job form would refuse is a 400; the rebuild needs the card and a short file."""
    uid, h = funded(client, signed_in)
    jid = make_done_job(uid)
    for estimate in (True, False):
        assert post(client, h, jid, estimate, nonsense=1).status_code == 422
        assert post(client, h, jid, estimate, options={"atten_lim_db": "strong"}).status_code == 400
        assert post(client, h, jid, estimate, options={"audio_format": "mp4"}).status_code == 400
        assert post(client, h, jid, estimate, options={"leveling": "yes"}).status_code == 400
        assert post(client, h, jid, estimate, options={"translate": ["fr"]}).status_code == 400  # a clean-only job has no subtitles
        assert post(client, h, jid, estimate, options={"x": "y" * 2100}).status_code == 400
        assert post(client, h, jid, estimate, denoiseTier="bogus").status_code == 400
        assert post(client, h, jid, estimate, denoiseTier="off", options={"leveling": False}).status_code == 400
    long_job = make_done_job(uid, seconds=100 * 60)
    body = post(client, h, long_job, True, denoiseTier="rebuild")
    assert body.status_code == 400 and "up to 90 minutes" in body.json()["detail"]
    assert count("jobs", "source_job_id IS NOT NULL") == 0 and count("holds") == 0


def test_the_voice_rebuild_waits_for_the_card(client, signed_in):
    """Rebuild from the upload runs only on the GPU worker: refused while it is down, accepted when it is up."""
    from acw import broker

    uid, h = funded(client, signed_in)
    jid = make_done_job(uid)
    r = post(client, h, jid, denoiseTier="rebuild")
    assert r.status_code == 400 and "paused" in r.json()["detail"]
    broker.mark_gpu_worker("alive")
    assert post(client, h, jid, True, denoiseTier="rebuild").status_code == 200
    # a rebuild that only levels the kept rebuilt voice needs no card
    broker.mark_gpu_worker("stopped")
    rebuilt = make_done_job(uid, denoise_tier="rebuild", rebuilt=True)
    assert post(client, h, rebuilt, True, options={"loudness_lufs": -20.0}).status_code == 200


def test_not_enough_balance_is_a_402_and_a_free_rerun_needs_no_balance(client, signed_in):
    """The price is held like any job's: a paid re-run with too little balance is refused, a free one is not."""
    uid, h = signed_in
    jid = make_done_job(uid, seconds=600)
    before = jobs_in_folders()
    r = post(client, h, jid, **STRENGTH)
    assert r.status_code == 402 and "available balance is $0.00" in r.json()["detail"] and "$0.01" in r.json()["detail"]
    assert count("jobs", "source_job_id IS NOT NULL") == 0 and count("holds") == 0 and jobs_in_folders() == before
    assert os.stat(folder(jid) / "input.wav").st_nlink == 1  # the links made for the attempt were taken away
    free = post(client, h, jid, **FREE_STRENGTH)
    assert free.status_code == 200 and free.json()["free"] is True and free.json()["estCostMicros"] == 0
    ledger.credit(uid, 10_000, "grant", "t", "g")  # exactly the price: enough
    assert post(client, h, jid, **STRENGTH).status_code == 200


def test_a_full_disk_is_a_507_and_leaves_nothing_behind(client, signed_in, monkeypatch):
    """Same disk check as a new job: refused before anything is made."""
    from acw import hostload

    uid, h = funded(client, signed_in)
    jid = make_done_job(uid)
    monkeypatch.setattr(hostload, "get", lambda d: SimpleNamespace(snapshot=lambda: SimpleNamespace(disk_free=1 * GB)))
    before = jobs_in_folders()
    assert post(client, h, jid, **STRENGTH).status_code == 507
    assert jobs_in_folders() == before and count("jobs", "source_job_id IS NOT NULL") == 0 and count("holds") == 0


# ------------------------------------------------------------------------------------ job limit and key cap

def test_the_active_job_limit_counts_rerun_and_refuses_with_the_exact_message(client, signed_in):
    """Three jobs in progress: a re-run (even a free one) is 429 too_many_active_jobs with Retry-After, and leaves nothing."""
    uid, h = funded(client, signed_in)
    jid = make_done_job(uid)
    for state in ("uploading", "queued", "running"):
        add_job(uid, state)
    before = jobs_in_folders()
    for body in (STRENGTH, FREE_STRENGTH):
        r = post(client, h, jid, **body)
        data = assert_error(r, 429, "too_many_active_jobs")
        assert data["detail"] == "You have 3 files being processed. Start this one when one of them is done."
        assert r.headers["Retry-After"] == "60"
    assert jobs_in_folders() == before and count("holds") == 0 and count("jobs", "source_job_id IS NOT NULL") == 0
    assert all(os.stat(folder(jid) / n).st_nlink == 1 for n in ("input.wav", "kept-noise.flac", "kept-input.flac"))
    assert post(client, h, jid, True, **STRENGTH).status_code == 200  # an estimate is only a question
    db.execute("UPDATE jobs SET status = 'canceled', finished_at = ? WHERE status = 'uploading'", (db.now(),))
    assert post(client, h, jid, **STRENGTH).status_code == 200  # room again


def test_a_rerun_in_progress_counts_toward_the_limit_of_new_jobs(client, signed_in):
    """2 jobs queued + 1 re-run = 3: a new file is refused with the real count; the limit follows ACW_MAX_ACTIVE_JOBS."""
    uid, h = funded(client, signed_in)
    jid = make_done_job(uid)
    add_job(uid, "queued"), add_job(uid, "running")
    assert post(client, h, jid, **STRENGTH).status_code == 200
    made = client.post("/api/jobs", json={"kind": "denoise", "denoiseTier": "standard", "filename": "a.wav", "sizeBytes": 1000,
                                          "durationSeconds": 60}, headers=h)
    data = assert_error(made, 429, "too_many_active_jobs")
    assert data["detail"] == "You have 3 files being processed. Start this one when one of them is done."
    with pytest.raises(jobs.JobError) as e:
        jobs.create_job(uid, kind="denoise", denoise_tier="standard", transcribe_tier=None, filename="a.wav", size_bytes=1000,
                        claimed_seconds=60.0, options={})
    assert (e.value.status, e.value.code) == (429, "too_many_active_jobs")
    assert count("jobs", "status = 'uploading'") == 0


def test_the_limit_end_to_end_three_uploads_then_a_fourth_and_a_rerun_are_refused(client, signed_in):
    """Through the API: three uploaded jobs wait in the queue; a new file and a re-run both get the same 429 until one ends."""
    uid, h = funded(client, signed_in)
    source = make_done_job(uid)
    queued = []
    for _ in range(3):
        r = client.post("/api/jobs", json={"kind": "denoise", "denoiseTier": "standard", "filename": "a.wav", "sizeBytes": 2000,
                                           "durationSeconds": 30}, headers=h)
        assert r.status_code == 200, r.text
        assert client.put(f"/api/jobs/{r.json()['id']}/input", content=bytes(2000), headers=h).status_code == 200
        queued.append(r.json()["id"])
    assert {row(j)["status"] for j in queued} == {"queued"}
    msg = "You have 3 files being processed. Start this one when one of them is done."
    fourth = client.post("/api/jobs", json={"kind": "denoise", "denoiseTier": "standard", "filename": "a.wav", "sizeBytes": 2000,
                                            "durationSeconds": 30}, headers=h)
    assert assert_error(fourth, 429, "too_many_active_jobs")["detail"] == msg and fourth.headers["Retry-After"] == "60"
    again = post(client, h, source, **STRENGTH)
    assert assert_error(again, 429, "too_many_active_jobs")["detail"] == msg
    assert client.post(f"/api/jobs/{queued[0]}/cancel", headers=h).status_code == 200
    assert post(client, h, source, **STRENGTH).status_code == 200  # one ended: a place is free
    assert_error(post(client, h, source, **STRENGTH), 429, "too_many_active_jobs")  # and now it is taken by the re-run
    assert client.get("/api/jobs").json()["limits"]["maxActiveJobs"] == 3


def test_the_limit_can_be_changed_with_the_environment(client, signed_in, monkeypatch):
    """ACW_MAX_ACTIVE_JOBS=1: one file at a time, and the message says "1 file"."""
    set_env(monkeypatch, ACW_MAX_ACTIVE_JOBS=1)
    uid, h = funded(client, signed_in)
    jid = make_done_job(uid)
    assert post(client, h, jid, **STRENGTH).status_code == 200
    data = assert_error(post(client, h, jid, **STRENGTH), 429, "too_many_active_jobs")
    assert data["detail"] == "You have 1 file being processed. Start this one when one of them is done."


def test_parallel_reruns_cannot_pass_the_active_job_limit(client, signed_in):
    """Eight re-runs at once with room for three: exactly three are made, the rest are 429, and nothing is left from them."""
    uid, h = funded(client, signed_in)
    src = make_done_job(uid)

    def one(_):
        try:
            return rerun.create_rerun(uid, src, options={"atten_lim_db": 24})["id"]
        except jobs.JobError as e:
            return e.status

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(one, range(8)))
    made = [r for r in results if isinstance(r, str)]
    assert len(made) == 3 and results.count(429) == 5, results
    assert count("holds", "status = 'active'") == 3 and ledger.balance(uid)["held_micros"] == 3_000
    assert jobs_in_folders() == {src, *made}
    assert os.stat(folder(src) / "input.wav").st_nlink == 4  # the source and the three re-runs, no stray links


def test_a_kept_copy_that_is_gone_means_the_noise_removal_runs_again_and_is_charged(client, signed_in):
    """The plan trusts the folder, not the record: with the kept noise removal deleted a strength change runs it again."""
    uid, h = funded(client, signed_in)
    src = make_done_job(uid, seconds=600)
    assert post(client, h, src, True, **STRENGTH).json()["estCostMicros"] == 10_000  # Leveling only
    os.unlink(folder(src) / "kept-noise.flac")
    e = post(client, h, src, True, **STRENGTH).json()
    assert e["steps"]["clean"] == "full" and e["estCostMicros"] == 100_000 and [x["item"] for x in e["lines"]] == ["denoise", "leveling"]
    assert client.get(f"/api/jobs/{src}").json()["redo"]["noise"] is False


def test_a_paid_rerun_counts_against_the_keys_monthly_cap_and_a_free_one_does_not(client, signed_in):
    """403 key_spending_cap for a paid re-run over the cap; free re-runs and the exact cap still work; the first counts."""
    uid, h = funded(client, signed_in)
    key_id, kh = api_key(uid, cap=5_000)
    jid = make_done_job(uid, seconds=600, api_key_id=key_id)  # Leveling for ten minutes: 10,000 micros
    before = jobs_in_folders()
    r = post(client, kh, jid, **STRENGTH)
    assert r.status_code == 403 and r.headers["X-Error-Code"] == "key_spending_cap"
    assert "monthly spending limit" in r.json()["detail"] and "$0.01" in r.json()["detail"]
    assert jobs_in_folders() == before and count("holds") == 0 and count("jobs", "source_job_id IS NOT NULL") == 0
    assert os.stat(folder(jid) / "input.wav").st_nlink == 1
    assert post(client, kh, jid, **FREE_STRENGTH).status_code == 200  # free work is never capped
    db.execute("UPDATE api_keys SET monthly_cap_micros = 10000 WHERE id = ?", (key_id,))
    assert post(client, kh, jid, estimate=True, **STRENGTH).json()["keyLimit"]["monthlyCapMicros"] == 10_000
    assert post(client, kh, jid, **STRENGTH).status_code == 200  # exactly the cap
    again = post(client, kh, jid, **STRENGTH)
    assert again.status_code == 403 and again.headers["X-Error-Code"] == "key_spending_cap"  # the first re-run's hold counts
    assert "keyLimit" not in post(client, h, jid, True, **STRENGTH).json()  # a browser session is not capped


# ------------------------------------------------------------------------------------- estimate and hold

def _items(rates, seconds, languages=0):
    """What the price should be, worked out from the per-minute rates, a started second counting."""
    secs = math.ceil(seconds)
    total = sum(math.ceil(rates[k] * secs / 60) for k in ("denoise", "leveling", "transcribe") if k in rates)
    return total + (languages * math.ceil(rates["translate"] * secs / 60) if "translate" in rates else 0)


TEXT = dict(kind="denoise_transcribe", transcribe_tier="local", segments={"language": "en"}, options={"language": "en"})
CASES = [
    pytest.param(dict(), STRENGTH, {"leveling": 1_000}, "denoised", 0, id="strength-with-leveling"),
    pytest.param(dict(), {"options": {"noise": "heavy"}}, {"denoise": 9_000, "leveling": 1_000}, "full", 0, id="other-noise-kind"),
    pytest.param(dict(options={"leveling": False}), {"options": {"noise": "heavy"}}, {"denoise": 9_000}, "full", 0, id="other-noise-kind-no-leveling"),
    pytest.param(dict(), {"denoiseTier": "rebuild"}, {"denoise": 30_000, "leveling": 1_000}, "full", 0, id="voice-rebuild"),
    pytest.param(dict(denoise_tier="off"), {"options": {"loudness_lufs": -20}}, {"leveling": 1_000}, "full", 0, id="leveling-only-new-loudness"),
    pytest.param(TEXT, {"transcribeTier": "premium"}, {"transcribe": 15_000}, "reuse", 0, id="better-subtitles"),
    pytest.param(TEXT, {"options": {"translate": ["fr", "es"]}}, {"translate": 4_000}, "reuse", 2, id="two-new-languages"),
    pytest.param(dict(options={"leveling": False}), {"options": {"atten_lim_db": 24}}, {}, "denoised", 0, id="free-strength"),
    pytest.param(dict(), {"options": {"audio_format": "wav"}}, {}, "reuse", 0, id="free-file-type"),
    pytest.param(TEXT, {"options": {"caption_formats": ["srt"]}}, {}, "reuse", 0, id="free-caption-files"),
]


@pytest.mark.parametrize("src_kw, body, rates, clean, languages", CASES)
def test_the_estimate_is_the_price_the_hold_is_made_for(client, signed_in, gpu_up, src_kw, body, rates, clean, languages):
    """estCostMicros of /rerun/estimate == the new job's est_cost_micros == the hold; a free re-run holds nothing."""
    uid, h = funded(client, signed_in)
    src = make_done_job(uid, seconds=90.5, **src_kw)
    spent_before, jobs_before = ledger.balance(uid), count("jobs")
    est = post(client, h, src, True, **body)
    assert est.status_code == 200, est.text
    e = est.json()
    price = _items(rates, 90.5, languages)
    assert e["estCostMicros"] == price and e["estCostUsd"] == price / 1e6 and e["free"] is (price == 0)
    assert sum(line["micros"] for line in e["lines"]) == price
    assert [line["item"] for line in e["lines"]] == [k for k in ("denoise", "leveling", "transcribe", "translate") if k in rates]
    assert e["steps"]["clean"] == clean and e["durationSeconds"] == 90.5 and e["expiresAt"] == row(src)["expires_at"]
    assert e["availableMicros"] == 5_000_000 and e["affordable"] is True
    # estimating makes and holds nothing
    assert count("jobs") == jobs_before and ledger.balance(uid) == spent_before and count("holds") == 0
    made = post(client, h, src, **body)
    assert made.status_code == 200, made.text
    out = made.json()
    assert out["estCostMicros"] == e["estCostMicros"] and out["free"] is e["free"] and out["expiresAt"] == row(src)["expires_at"]
    new = row(out["id"])
    assert new["est_cost_micros"] == price and new["status"] == "queued"
    if price:
        hold = db.one("SELECT * FROM holds WHERE id = ?", (new["hold_id"],))
        assert hold["amount_micros"] == price and hold["ref_type"] == "job" and hold["ref_id"] == out["id"] and hold["status"] == "active"
        assert ledger.balance(uid)["held_micros"] == price
        assert params_of(out["id"])["rates"] == rates
    else:
        assert new["hold_id"] is None and count("holds") == 0 and ledger.balance(uid)["held_micros"] == 0
        assert params_of(out["id"])["rates"] == {}


def test_estimate_lines_name_each_step_and_a_poor_account_is_not_affordable(client, signed_in):
    """lines carry the price list's words and per-minute price; affordable is false when the balance is short."""
    uid, h = signed_in
    ledger.credit(uid, 5_000, "grant", "t", "g")
    src = make_done_job(uid, seconds=600)
    e = post(client, h, src, True, options={"noise": "heavy"}).json()
    assert e["affordable"] is False and e["availableMicros"] == 5_000 and e["estCostMicros"] == 100_000
    assert e["lines"] == [
        {"item": "denoise", "label": "Denoising", "quantity": 1, "usdPerMinute": 0.009, "micros": 90_000},
        {"item": "leveling", "label": "Leveling", "quantity": 1, "usdPerMinute": 0.001, "micros": 10_000},
    ]
    assert e["denoiseTier"] == "standard" and e["transcribeTier"] is None
    assert e["options"]["noise"] == "heavy" and e["options"]["atten_lim_db"] == 100.0  # the options the re-run would run with


def test_estimate_tells_the_whole_story_of_translations(client, signed_in):
    """New languages only: quantity is the number of new ones; steps lists them."""
    uid, h = funded(client, signed_in)
    src = make_done_job(uid, seconds=120, **TEXT)
    db_params = params_of(src)
    db_params["kept"]["segments"]["translations"] = ["fr"]
    db.execute("UPDATE jobs SET params_json = ? WHERE id = ?", (json.dumps(db_params), src))
    (folder(src) / "segments.fr.json").write_bytes(b"{}")
    e = post(client, h, src, True, options={"translate": ["fr", "es", "de"]}).json()
    assert e["steps"] == {"clean": "reuse", "transcribe": False, "translate": ["es", "de"]}
    assert e["lines"] == [{"item": "translate", "label": "Translated subtitles, per language", "quantity": 2, "usdPerMinute": 0.004, "micros": 2 * 8_000}]
    assert e["estCostMicros"] == 16_000


# --------------------------------------------------------------------------------- the new job and its folder

def test_the_new_job_row_params_and_hard_linked_folder(client, signed_in):
    """A queued job of its own: linked to its source, with the kept copies hard-linked (same file, not a copy)."""
    uid, h = funded(client, signed_in)
    src = make_done_job(uid, seconds=60.0)
    r = post(client, h, src, **STRENGTH)
    assert r.status_code == 200 and set(r.json()) == {"id", "estCostMicros", "free", "expiresAt"}
    new_id = r.json()["id"]
    old, new = row(src), row(new_id)
    assert (new["kind"], new["tier"], new["status"], new["next_stage"], new["user_id"]) == ("denoise", "standard", "queued", "probe", uid)
    assert (new["source_job_id"], new["root_job_id"]) == (src, src)
    assert (new["input_name"], new["input_bytes"], new["received_bytes"]) == (old["input_name"], old["input_bytes"], old["received_bytes"])
    assert new["claimed_seconds"] == new["duration_seconds"] == 60.0 and new["api_key_id"] is None
    assert new["finished_at"] is None and new["outputs_json"] is None and new["created_at"] == new["queued_at"]
    p, sp = params_of(new_id), params_of(src)
    assert p["options"] == {**sp["options"], "atten_lim_db": 24.0}
    assert (p["denoise_tier"], p["transcribe_tier"], p["input_ext"], p["channels"]) == ("standard", None, ".wav", 2)
    assert p["rates"] == {"leveling": 1_000} and p["noise_profile"] == "light"
    assert p["kept"] == {k: sp["kept"][k] for k in ("noise", "input")}
    assert p["rerun"] == {"source": src, "root": src, "deadline": old["expires_at"], "clean": "denoised", "transcribe": False,
                          "reuse_audio": None, "translate_new": [], "paid": ["leveling"],
                          "linked": ["input.wav", "kept-input.flac", "kept-noise.flac"]}
    assert files_of(new_id) == {"input.wav", "kept-input.flac", "kept-noise.flac"}
    assert oct(folder(new_id).stat().st_mode & 0o777) == "0o700"
    for name in files_of(new_id):
        assert same_file(folder(src) / name, folder(new_id) / name) and os.stat(folder(new_id) / name).st_nlink >= 2
        assert (folder(new_id) / name).read_bytes() == (folder(src) / name).read_bytes()
    assert files_of(src) >= {"talk-cleaned.flac", "kept-noise.flac"}  # the source is as it was
    shown = client.get(f"/api/jobs/{new_id}").json()
    assert shown["sourceJobId"] == src and shown["label"] == "Changed and saved again (Leveling)" and shown["status"] == "queued"
    assert shown["denoiseTier"] == "standard" and shown["estCostMicros"] == 1_000 and "redo" not in shown and shown["expiresAt"] is None


def test_what_a_rerun_links_follows_the_plan(client, signed_in, gpu_up, monkeypatch):
    """The upload always; the kept copies that fit; the audio to reuse; the transcript when it is reused."""
    set_env(monkeypatch, ACW_MAX_ACTIVE_JOBS=20)
    uid, h = funded(client, signed_in)
    text = make_done_job(uid, **TEXT)
    caption_only = post(client, h, text, options={"caption_formats": ["srt"]}).json()["id"]
    assert files_of(caption_only) == {"input.wav", "kept-noise.flac", "kept-input.flac", "segments.json", "talk-cleaned.flac"}
    assert params_of(caption_only)["kept"]["segments"] == params_of(text)["kept"]["segments"]
    assert params_of(caption_only)["rerun"]["reuse_audio"] == "talk-cleaned.flac"
    assert params_of(caption_only)["rerun"]["clean"] == "reuse" and params_of(caption_only)["rerun"]["transcribe"] is False
    retranscribe = post(client, h, text, transcribeTier="premium").json()["id"]
    assert files_of(retranscribe) == {"input.wav", "kept-noise.flac", "kept-input.flac", "talk-cleaned.flac"}  # no transcript: it is made again
    assert "segments" not in params_of(retranscribe)["kept"] and params_of(retranscribe)["rerun"]["transcribe"] is True
    other_noise = make_done_job(uid)
    full = post(client, h, other_noise, options={"noise": "heavy"}).json()["id"]
    assert files_of(full) == {"input.wav"}  # noise removal runs again and writes its own copies
    assert params_of(full)["kept"] == {} and params_of(full)["rerun"]["linked"] == ["input.wav"]
    rebuild = make_done_job(uid)
    rb = post(client, h, rebuild, denoiseTier="rebuild").json()["id"]
    assert files_of(rb) == {"input.wav", "kept-noise.flac", "kept-input.flac"} and params_of(rb)["rates"] == {"denoise": 30_000, "leveling": 1_000}
    echo = make_done_job(uid)
    ec = post(client, h, echo, options={"echo_reference_track": True}).json()["id"]
    assert files_of(ec) == {"input.wav"}  # kept without echo removal: not for a run with it
    translations = make_done_job(uid, **TEXT)
    sp = params_of(translations)
    sp["kept"]["segments"]["translations"] = ["fr"]
    db.execute("UPDATE jobs SET params_json = ? WHERE id = ?", (json.dumps(sp), translations))
    (folder(translations) / "segments.fr.json").write_bytes(b"{}")
    tr = post(client, h, translations, options={"translate": ["fr", "es"]}).json()["id"]
    assert {"segments.json", "segments.fr.json"} <= files_of(tr) and params_of(tr)["rerun"]["translate_new"] == ["es"]


def test_a_rerun_of_a_rerun_keeps_the_root_and_ends_with_the_job_it_came_from(client, signed_in):
    """root_job_id is the first job of the chain, source_job_id the one it was made from, the deadline the latter's expiry."""
    uid, h = funded(client, signed_in)
    root = make_done_job(uid)
    first = make_done_job(uid, source=root, root=root, expires_in=3000)
    r = post(client, h, first, **STRENGTH)
    assert r.status_code == 200
    new = row(r.json()["id"])
    assert (new["source_job_id"], new["root_job_id"]) == (first, root)
    assert params_of(new["id"])["rerun"]["root"] == root and params_of(new["id"])["rerun"]["deadline"] == row(first)["expires_at"]
    assert r.json()["expiresAt"] == row(first)["expires_at"]


def test_deleting_or_sweeping_the_source_does_not_break_a_linked_rerun(client, signed_in, clock):
    """Hard links: the re-run's files survive the source's folder being deleted."""
    uid, h = funded(client, signed_in)
    src = make_done_job(uid, expires_in=4000)
    new_id = post(client, h, src, **STRENGTH).json()["id"]
    expected = {n: (folder(new_id) / n).read_bytes() for n in files_of(new_id)}
    clock.advance(4000)
    jobs.sweep()  # the source is due: its folder goes
    assert row(src)["status"] == "expired" and not folder(src).exists()
    assert {n: (folder(new_id) / n).read_bytes() for n in files_of(new_id)} == expected
    assert all(os.stat(folder(new_id) / n).st_nlink == 1 for n in expected)


def test_a_rerun_never_changes_the_source_files(client, signed_in):
    """The kept copies and the upload of the source stay byte for byte as they were."""
    uid, h = funded(client, signed_in)
    src = make_done_job(uid)
    before = {n: (folder(src) / n).read_bytes() for n in files_of(src)}
    post(client, h, src, **STRENGTH)
    post(client, h, src, **FREE_STRENGTH)
    assert {n: (folder(src) / n).read_bytes() for n in files_of(src)} == before and row(src)["status"] == "done"


# ------------------------------------------------------------------------- deletion time and money at the end

OUT = "talk-cleaned-2.flac"  # the new audio result of a re-run (the runner writes it next to the linked files)


@pytest.mark.parametrize("left, waited", [(7200, 60), (3600, 60), (1800, 600), (400, 0)])
def test_a_rerun_never_ends_later_than_the_job_it_came_from(client, signed_in, clock, left, waited):
    """expires_at of the finished re-run <= the source's; with plenty of time left it is exactly the source's."""
    uid, h = funded(client, signed_in)
    src = make_done_job(uid, expires_in=left, finished_ago=7200 - left)
    created = post(client, h, src, **FREE_STRENGTH)
    assert created.status_code == 200, created.text
    new_id = created.json()["id"]
    assert params_of(new_id)["rerun"]["deadline"] == row(src)["expires_at"]
    clock.advance(waited)
    done = complete_like_the_worker(new_id, outputs=[OUT])
    assert done["status"] == "done" and done["finished_at"] == clock.t
    assert done["expires_at"] == row(src)["expires_at"]  # <= the source's, and as late as allowed
    # the upload and the copies are kept for the same time, and the job can be changed again
    assert files_of(new_id) == {"input.wav", "kept-noise.flac", "kept-input.flac", OUT}
    shown = client.get(f"/api/jobs/{new_id}").json()
    assert shown["expiresAt"] == row(src)["expires_at"] and shown["redo"]["available"] and shown["redo"]["noise"] and "preview" in shown["redo"]
    assert [o["name"] for o in shown["outputs"]] == [OUT]
    clock.advance(left)  # the source's time is up: the sweep takes both together
    jobs.sweep()
    assert row(src)["status"] == "expired" and row(new_id)["status"] == "expired"


@pytest.mark.parametrize("advance", [1500, 940])
def test_a_rerun_that_is_only_done_when_its_source_is_about_to_go_is_stopped_and_not_charged(client, signed_in, clock, advance):
    """A long wait in the queue: its results could not be downloaded before they go and it never extends the deletion time, so
    it ends failed, the hold is given back, and no file of it stays (60 s or less before the end, or after it)."""
    uid, h = funded(client, signed_in)
    src = make_done_job(uid, seconds=60, expires_in=1000, finished_ago=6200)
    start = ledger.balance(uid)["balance_micros"]
    new_id = post(client, h, src, **STRENGTH).json()["id"]
    clock.advance(advance)
    done = complete_like_the_worker(new_id, outputs=[OUT])
    assert done["status"] == "failed" and "weren" in done["error"] and not done["cost_micros"]
    assert ledger.balance(uid)["balance_micros"] == start and ledger.balance(uid)["held_micros"] == 0
    assert not folder(new_id).exists() or files_of(new_id) == set()
    assert_error(post(client, h, new_id, **STRENGTH), 409, "job_not_finished")


def test_a_rerun_finishing_a_little_before_its_source_ends_with_it(client, signed_in, clock):
    """Never extends: with more than a minute left it keeps everything and ends exactly with the source."""
    uid, h = funded(client, signed_in)
    src = make_done_job(uid, expires_in=500, finished_ago=6700)
    new_id = post(client, h, src, **FREE_STRENGTH).json()["id"]
    clock.advance(400)  # 100 seconds left of the source's time
    done = complete_like_the_worker(new_id, outputs=[OUT])
    assert done["status"] == "done" and done["expires_at"] == row(src)["expires_at"] and "input.wav" in files_of(new_id)


def test_a_paid_rerun_is_charged_once_with_a_labelled_ledger_line_and_a_free_one_not_at_all(client, signed_in, clock):
    """"Changed and saved again (Leveling): talk.wav (1.0 min)" for exactly the price; nothing in the ledger for a free re-run."""
    uid, h = funded(client, signed_in)
    src = make_done_job(uid, seconds=60)
    start = ledger.balance(uid)["balance_micros"]
    lines_before = count("ledger")
    paid = post(client, h, src, **STRENGTH).json()["id"]
    assert ledger.balance(uid)["held_micros"] == 1_000 and count("ledger") == lines_before  # held, not yet charged
    done = complete_like_the_worker(paid, outputs=[OUT])
    assert done["cost_micros"] == pricing.rerun_cost_micros({"leveling": 1_000}, 60) == 1_000
    new_lines = db.all_rows("SELECT * FROM ledger WHERE id > ? ORDER BY id", (lines_before,))
    assert len(new_lines) == 1
    line = new_lines[0]
    assert (line["kind"], line["ref_type"], line["ref_id"], line["amount_micros"]) == ("usage", "job", paid, -1_000)
    assert line["description"] == "Changed and saved again (Leveling): talk.wav (1.0 min)"
    assert ledger.balance(uid) == {"balance_micros": start - 1_000, "held_micros": 0, "available_micros": start - 1_000}
    assert db.one("SELECT status FROM holds WHERE id = ?", (row(paid)["hold_id"],))["status"] == "captured"
    free = post(client, h, src, **FREE_STRENGTH).json()["id"]
    # (the first re-run finished, so there is room for another)
    again = complete_like_the_worker(free, outputs=["talk-cleaned-3.flac"])
    assert again["status"] == "done" and again["cost_micros"] == 0 and again["hold_id"] is None
    assert count("ledger") == lines_before + 1  # no line for it
    assert ledger.invariant_problems() == {"balance_mismatch": 0, "held_mismatch": 0}


@pytest.mark.parametrize("src_kw, body, label, micros", [
    pytest.param(dict(), {"options": {"noise": "heavy"}}, "Changed and saved again (Denoising + Leveling)", 10_000, id="noise-removal-again"),
    pytest.param(TEXT, {"transcribeTier": "premium"}, "Changed and saved again (Subtitles & transcript, extra accurate)", 15_000,
                 id="subtitles-again"),
    pytest.param(TEXT, {"options": {"translate": ["fr", "es"]}}, "Changed and saved again (Translated subtitles)", 8_000, id="new-languages"),
    pytest.param(dict(denoise_tier="off"), {"options": {"loudness_lufs": -20}}, "Changed and saved again (Leveling)", 1_000, id="leveling-only"),
])
def test_the_ledger_line_names_the_steps_that_ran_again_and_charges_their_price(client, signed_in, src_kw, body, label, micros):
    """One line per paid re-run, named by what ran again, for the price of those steps only (a minute of audio)."""
    uid, h = funded(client, signed_in)
    src = make_done_job(uid, seconds=60, **src_kw)
    new_id = post(client, h, src, **body).json()["id"]
    before = db.one("SELECT MAX(id) AS m FROM ledger")["m"]
    done = complete_like_the_worker(new_id, outputs=[OUT])
    line = db.one("SELECT * FROM ledger WHERE id > ?", (before,))
    assert db.one("SELECT COUNT(*) AS n FROM ledger WHERE id > ?", (before,))["n"] == 1
    assert line["amount_micros"] == -micros == -done["cost_micros"]
    assert line["description"] == f"{label}: talk.wav (1.0 min)"
    assert jobs.label_of(done["kind"], params_of(new_id)) == label


def _usage_line_of_a_rerun(client, signed_in):
    uid, h = funded(client, signed_in)
    src = make_done_job(uid, seconds=60)
    complete_like_the_worker(post(client, h, src, **STRENGTH).json()["id"], outputs=[OUT])
    usage = [e for e in client.get("/api/billing/history").json()["ledger"] if e["kind"] == "usage"]
    assert len(usage) == 1
    return usage[0]


def test_the_credit_history_page_has_the_rerun_line_with_its_description(client, signed_in):
    """GET /api/billing/history: the usage line of a re-run is there, described as changed and saved again."""
    line = _usage_line_of_a_rerun(client, signed_in)
    assert line["description"] == "Changed and saved again (Leveling): talk.wav (1.0 min)" and line["amount_micros"] == -1_000
    assert line["displayName"] == "talk.wav" and line["minutes"] == 1.0


def test_the_credit_history_page_labels_a_rerun_line_as_changed_and_saved_again(client, signed_in):
    """The `label` of the history line (what the page shows) names the re-run like its description does."""
    assert _usage_line_of_a_rerun(client, signed_in)["label"] == "Changed and saved again (Leveling)"


def test_a_failed_rerun_costs_nothing_and_its_hold_is_released(client, signed_in):
    """A re-run that fails is not charged; its linked files are removed from its folder but the source's stay."""
    uid, h = funded(client, signed_in)
    src = make_done_job(uid)
    new_id = post(client, h, src, **STRENGTH).json()["id"]
    assert ledger.balance(uid)["held_micros"] == 1_000
    assert jobs.finish_failed(row(new_id), "Processing failed; you were not charged.")
    assert ledger.balance(uid) == {"balance_micros": 5_000_000, "held_micros": 0, "available_micros": 5_000_000}
    assert not folder(new_id).exists() and files_of(src) >= {"input.wav", "kept-noise.flac", "kept-input.flac"}
    assert os.stat(folder(src) / "input.wav").st_nlink == 1
    assert post(client, h, src, **STRENGTH).status_code == 200  # the source is as good as before


def test_the_worker_prices_a_rerun_by_what_it_ran_and_never_extends_a_hold(client, signed_in):
    """A re-run is priced for the length the first job measured: the worker's re-price step is skipped for it."""
    uid, h = funded(client, signed_in)
    src = make_done_job(uid, seconds=75.3)
    new_id = post(client, h, src, **STRENGTH).json()["id"]
    assert row(new_id)["duration_seconds"] == 75.3 and row(new_id)["est_cost_micros"] == math.ceil(1_000 * 76 / 60)
    done = complete_like_the_worker(new_id, outputs=[OUT])
    assert done["cost_micros"] == done["est_cost_micros"] == 1_267
    assert db.one("SELECT amount_micros FROM holds WHERE id = ?", (done["hold_id"],))["amount_micros"] == 1_267


# ----------------------------------------------------------------------------------------- the API key flow

def key_job(client, kh, uid, seconds=60):
    """A job made through the API with a key (create, upload) and finished the way the worker finishes it, with the kept copies."""
    r = client.post("/api/jobs", json={"kind": "denoise", "denoiseTier": "standard", "filename": "talk.wav", "sizeBytes": 3000,
                                       "durationSeconds": seconds, "options": {"output_video": False}}, headers=kh)
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    assert client.put(f"/api/jobs/{jid}/input", content=bytes(3000), headers=kh).status_code == 200
    for n in ("kept-noise.flac", "kept-input.flac"):
        (folder(jid) / n).write_bytes(b"flac" + n.encode())
    db.execute("UPDATE jobs SET duration_seconds = ? WHERE id = ?", (seconds, jid))
    jobs.update_params(jid, channels=2)
    done = complete_like_the_worker(jid, outputs=["talk-cleaned.flac"], kept=kept_record(seconds, channels=2))
    assert done["status"] == "done" and done["api_key_id"] is not None
    return jid


def test_the_api_key_flow_quote_then_change_a_job_the_key_made(client, signed_in):
    """A key with the jobs scope: makes a job, quotes a change (dry run), changes it with the quoted maximum, polls and downloads."""
    uid, h = funded(client, signed_in)
    key_id, kh = api_key(uid, "jobs", cap=1_000_000)
    jid = key_job(client, kh, uid)
    assert row(jid)["api_key_id"] == key_id
    held, jobs_before = ledger.balance(uid), count("jobs")
    quote = client.post(f"/api/jobs/{jid}/rerun/estimate", json={"options": {"atten_lim_db": 24}}, headers=kh)
    assert quote.status_code == 200, quote.text
    q = quote.json()
    assert q["estCostMicros"] == 1_000 and q["free"] is False and q["keyLimit"]["monthlyCapMicros"] == 1_000_000
    assert count("jobs") == jobs_before and ledger.balance(uid) == held  # a quote makes and holds nothing
    made = client.post(f"/api/jobs/{jid}/rerun", json={"options": {"atten_lim_db": 24}, "maxCostMicros": q["estCostMicros"]}, headers=kh)
    assert made.status_code == 200, made.text
    new_id = made.json()["id"]
    assert made.json()["estCostMicros"] == q["estCostMicros"] and row(new_id)["api_key_id"] == key_id
    assert (row(new_id)["source_job_id"], row(new_id)["root_job_id"]) == (jid, jid)
    polled = client.get(f"/api/jobs/{new_id}", headers=kh).json()
    assert polled["status"] == "queued" and polled["sourceJobId"] == jid and polled["estCostMicros"] == 1_000
    done = complete_like_the_worker(new_id, outputs=[OUT])
    assert done["cost_micros"] == 1_000
    shown = client.get(f"/api/jobs/{new_id}", headers=kh).json()
    assert shown["status"] == "done" and [o["name"] for o in shown["outputs"]] == [OUT] and shown["redo"]["available"] is True
    assert client.get(f"/api/jobs/{new_id}/outputs/{OUT}", headers=kh).status_code == 200
    # the key can change its own re-run again (a chain made by one key)
    again = client.post(f"/api/jobs/{new_id}/rerun", json={"options": {"atten_lim_db": 12}}, headers=kh)
    assert again.status_code == 200 and row(again.json()["id"])["root_job_id"] == jid


def test_a_key_never_changes_another_keys_job_or_a_website_job_but_the_website_can_change_a_keys_job(client, signed_in):
    """404 job_not_found (body code and X-Error-Code) for the other key and the website's job; the session may redo the key's job."""
    uid, h = funded(client, signed_in)
    k1, kh1 = api_key(uid)
    k2, kh2 = api_key(uid)
    by_k1 = key_job(client, kh1, uid)
    by_website = make_done_job(uid)
    for path in ("rerun/estimate", "rerun"):
        for caller, jid in ((kh2, by_k1), (kh1, by_website), (kh2, by_website)):
            r = client.post(f"/api/jobs/{jid}/{path}", json=STRENGTH, headers=caller)
            assert r.status_code == 404, (path, r.text)
            assert r.json() == {"detail": "Job not found.", "code": "job_not_found"} and r.headers["X-Error-Code"] == "job_not_found"
    assert count("jobs", "source_job_id IS NOT NULL") == 0 and count("holds", "status = 'active'") == 0
    # the website session of the same account can change the key's job; the new job is the website's own (no key)
    quote = client.post(f"/api/jobs/{by_k1}/rerun/estimate", json=STRENGTH, headers=h)
    made = client.post(f"/api/jobs/{by_k1}/rerun", json={**STRENGTH, "maxCostMicros": quote.json()["estCostMicros"]}, headers=h)
    assert quote.status_code == made.status_code == 200
    assert row(made.json()["id"])["api_key_id"] is None and row(made.json()["id"])["source_job_id"] == by_k1
    # and that website-made re-run is out of reach of the key that made the first job
    assert client.post(f"/api/jobs/{made.json()['id']}/rerun/estimate", json=STRENGTH, headers=kh1).status_code == 404


def test_the_three_job_limit_counts_reruns_whoever_started_them(client, signed_in):
    """A key's re-run, the website's re-run and another key's new file are three jobs of one account: the next is a 429."""
    uid, h = funded(client, signed_in)
    k1, kh1 = api_key(uid)
    k2, kh2 = api_key(uid)
    by_k1 = key_job(client, kh1, uid)
    by_website = make_done_job(uid)
    assert client.post(f"/api/jobs/{by_k1}/rerun", json=STRENGTH, headers=kh1).status_code == 200  # 1: a key's re-run
    assert client.post(f"/api/jobs/{by_website}/rerun", json=STRENGTH, headers=h).status_code == 200  # 2: the website's
    new = client.post("/api/jobs", json={"kind": "denoise", "denoiseTier": "standard", "filename": "b.wav", "sizeBytes": 2000,
                                         "durationSeconds": 30}, headers=kh2)
    assert new.status_code == 200, new.text  # 3: another key's new file
    msg = "You have 3 files being processed. Start this one when one of them is done."
    for caller, jid in ((kh1, by_k1), (h, by_website), (h, by_k1)):
        r = client.post(f"/api/jobs/{jid}/rerun", json=FREE_STRENGTH, headers=caller)
        assert assert_error(r, 429, "too_many_active_jobs")["detail"] == msg and r.headers["Retry-After"] == "60"
    fourth = client.post("/api/jobs", json={"kind": "denoise", "denoiseTier": "standard", "filename": "c.wav", "sizeBytes": 2000,
                                            "durationSeconds": 30}, headers=kh1)
    assert assert_error(fourth, 429, "too_many_active_jobs")["detail"] == msg
    assert count("jobs", "status IN ('uploading', 'queued', 'running', 'cloud_asr')") == 3


def test_the_price_guard_and_the_error_shape_for_a_key(client, signed_in):
    """maxCostMicros below the quote: 409 price_changed with code in the body and X-Error-Code; the quote itself is accepted."""
    uid, h = funded(client, signed_in)
    key_id, kh = api_key(uid)
    jid = key_job(client, kh, uid, seconds=600)
    quote = client.post(f"/api/jobs/{jid}/rerun/estimate", json={"options": {"noise": "heavy"}}, headers=kh).json()
    assert quote["estCostMicros"] == 100_000 and quote["steps"]["clean"] == "full"
    held = ledger.balance(uid)["held_micros"]
    r = client.post(f"/api/jobs/{jid}/rerun", json={"options": {"noise": "heavy"}, "maxCostMicros": quote["estCostMicros"] - 1}, headers=kh)
    body = assert_error(r, 409, "price_changed")
    assert set(body) == {"detail", "code"} and "$0.10" in body["detail"]
    assert ledger.balance(uid)["held_micros"] == held and count("jobs", "source_job_id IS NOT NULL") == 0
    ok = client.post(f"/api/jobs/{jid}/rerun", json={"options": {"noise": "heavy"}, "maxCostMicros": quote["estCostMicros"]}, headers=kh)
    assert ok.status_code == 200 and ok.json()["estCostMicros"] == quote["estCostMicros"]
    # other errors keep the same shape for a key
    assert_error(client.post(f"/api/jobs/{jid}/rerun", json={}, headers=kh), 400, "nothing_to_change")
    assert_error(client.post(f"/api/jobs/{'0' * 32}/rerun", json=STRENGTH, headers=kh), 404, "job_not_found")
    stream_only = api_key(uid, "stream")[1]
    assert client.post(f"/api/jobs/{jid}/rerun/estimate", json=STRENGTH, headers=stream_only).status_code == 403
    assert client.post(f"/api/jobs/{jid}/rerun", json=STRENGTH, headers={"Authorization": "Bearer nope"}).status_code == 401


def test_an_old_restore_highs_option_is_ignored_in_a_rerun_and_in_the_options_it_carries(client, signed_in, gpu_up):
    """The step is gone: a job stored with the option (from before), or a request that still sends it, changes nothing: it is not
    carried into the new job, not priced and not reported."""
    uid, h = funded(client, signed_in)
    src = make_done_job(uid, seconds=60)
    jobs.update_params(src, options={**params_of(src)["options"], "restore_highs": True})  # stored by an older version
    e = post(client, h, src, True, denoiseTier="rebuild", options={"restore_highs": True}).json()
    assert [x["item"] for x in e["lines"]] == ["denoise", "leveling"] and e["estCostMicros"] == 31_000
    assert "restore_highs" not in e["options"]
    new_id = post(client, h, src, denoiseTier="rebuild").json()["id"]
    assert "restore_highs" not in params_of(new_id)["options"] and params_of(new_id)["rates"] == {"denoise": 30_000, "leveling": 1_000}
    assert client.get(f"/api/jobs/{new_id}").json()["label"] == "Changed and saved again (Voice rebuild + Leveling)"
    assert_error(post(client, h, src, True, options={"restore_highs": False}), 400, "nothing_to_change")  # (nothing that exists changed)


def test_a_rerun_with_leveling_turned_on_is_held_charged_and_named_with_it(client, signed_in):
    """The hold is the estimate, the ledger line names the step, and the amount is its price."""
    uid, h = funded(client, signed_in)
    src = make_done_job(uid, seconds=60, options={"leveling": False})
    quote = post(client, h, src, True, options={"leveling": True}).json()
    assert quote["estCostMicros"] == 1_000 and [x["item"] for x in quote["lines"]] == ["leveling"]
    assert quote["lines"][0] == {"item": "leveling", "label": "Leveling", "quantity": 1, "usdPerMinute": 0.001, "micros": 1_000}
    new_id = post(client, h, src, maxCostMicros=1_000, options={"leveling": True}).json()["id"]
    assert db.one("SELECT amount_micros FROM holds WHERE ref_id = ?", (new_id,))["amount_micros"] == 1_000
    assert client.get(f"/api/jobs/{new_id}").json()["label"] == "Changed and saved again (Leveling)"
    complete_like_the_worker(new_id, outputs=[OUT])
    line = db.one("SELECT * FROM ledger WHERE ref_id = ?", (new_id,))
    assert (line["amount_micros"], line["description"]) == (-1_000, "Changed and saved again (Leveling): talk.wav (1.0 min)")
    assert_error(post(client, h, src, maxCostMicros=999, options={"leveling": True}), 409, "price_changed")
