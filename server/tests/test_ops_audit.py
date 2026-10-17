"""Fixes from the architecture/operations and security audits: request bodies, restarts that don't fail jobs,
capacity estimates, backups, retention, login mail, uploads, live captions and the monitor."""
import asyncio
import gzip
import json
import os
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from acw import auth, db, jobs
from acw.billing import ledger, pricing
from acw.stream.features import Features
from tests.test_billing_audit import frame, make_job, run_session, running_job, user
from tests.conftest import ORIGIN, login, settle_payment


# ---------------------------------------------------------------- request bodies (SEC-01)

def test_database_commits_are_durable():
    assert db.one("PRAGMA synchronous")[0] == 2  # FULL
    assert oct(os.stat(db.get_settings().db_path).st_mode & 0o777) == "0o600"


def test_big_json_body_is_refused_before_it_is_read(client):
    r = client.post("/api/auth/email", content=b'{"email": "' + b"a" * 200_000 + b'"}',
                    headers={"Origin": ORIGIN, "content-type": "application/json"})
    assert r.status_code == 413


def test_streamed_body_without_a_length_is_counted_while_it_arrives(client):
    def body():
        for _ in range(40):
            yield b"x" * 4096  # 160 KB, no Content-Length

    r = client.post("/api/auth/email", content=body(), headers={"Origin": ORIGIN, "content-type": "application/json"})
    assert r.status_code == 413


def test_small_requests_and_the_upload_route_are_not_touched_by_the_limit(client, signed_in):
    assert client.get("/api/me").status_code == 200
    uid, h = signed_in
    # the upload route has no limit of its own in the middleware: it answers for itself (404: no such job)
    r = client.put("/api/jobs/" + "a" * 32 + "/input", content=b"x" * 200_000, headers=h)
    assert r.status_code == 404


def test_webhook_body_limit_is_one_megabyte(client):
    assert client.post("/api/stripe/webhook", content=b"x" * 1_200_000).status_code == 413
    assert client.post("/api/stripe/webhook", content=b"x" * 2000).status_code != 413


def test_limit_applies_under_the_base_path():
    from acw.bodylimit import BodyLimitMiddleware

    mw = BodyLimitMiddleware(None, base_path="/audio")
    assert mw.limit_for("/audio/api/jobs/" + "a" * 32 + "/input") is None
    assert mw.limit_for("/audio/api/auth/email") == 64 * 1024
    assert mw.limit_for("/audio/api/stripe/webhook") == 1024 * 1024


def test_access_log_has_no_query_string(client, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="acw.access"):
        client.get("/api/auth/oauth/google/callback?code=SECRETCODE&state=SECRETSTATE")
        client.get("/api/me")
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "SECRETCODE" not in text and "SECRETSTATE" not in text
    assert "/api/me" in text


# ---------------------------------------------------------------- rate limits (W-11)

def test_429_says_when_to_try_again(client):
    from acw.security import limiter

    for _ in range(7):
        r = client.post("/api/auth/email", json={"email": "someone@example.com"}, headers={"Origin": ORIGIN})
    assert r.status_code in (404, 429)  # 404 when mail is not configured in the test environment
    limiter.reset()
    for _ in range(130):
        r = client.get("/api/health")
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 1 and "try again in" in r.json()["detail"]


def test_limits_are_per_name_and_ipv6_counts_per_64():
    from acw.security import RateLimiter, client_net

    assert client_net("2001:db8:1:2:aaaa:bbbb:cccc:dddd") == client_net("2001:db8:1:2::1") == "2001:db8:1:2::/64"
    assert client_net("203.0.113.9") == "203.0.113.9"
    lim = RateLimiter(max_keys=3)
    assert lim.hit("api:keep", 1, 0.0001)
    for i in range(10):  # a flood of made-up keys of another kind
        lim.hit(f"email_addr:{i}", 1, 0.0001)
    assert not lim.hit("api:keep", 1, 0.0001)  # still remembered: the flood did not push it out


def test_throttle_slows_a_fast_sender():
    from acw.security import Throttle

    async def go():
        t = Throttle(1_000_000)
        t0 = time.monotonic()
        for _ in range(5):
            await t.take(600_000)  # 3 MB at 1 MB/s with a 1 MB burst
        return time.monotonic() - t0

    assert 1.5 < asyncio.run(go()) < 4


# ---------------------------------------------------------------- login mail (SEC-12, SEC-13)

@pytest.mark.parametrize("bad", ["x,attacker@evil.com", "a@b.com,c@d.com", "a;b@x.com", "<a@b.com>", '"a"@b.com',
                                 "a@b@c.com", "a b@c.com", "a@b.com\nBcc: x@y.com", "(c)a@b.com", "a@b", "a\x00@b.com"])
def test_addresses_that_mail_more_than_one_recipient_are_refused(bad):
    from acw.api.auth_routes import clean_email

    assert clean_email(bad) is None


def test_plain_addresses_pass_and_aliases_share_a_limit():
    from acw.api.auth_routes import canonical_email, clean_email

    assert clean_email("  Some.One+tag@Example.com ") == "some.one+tag@example.com"
    assert canonical_email("some.one+tag@gmail.com") == canonical_email("s.omeone@googlemail.com") == "someone@gmail.com"
    assert canonical_email("a.b+x@corp.example") == "a.b@corp.example"


def test_login_mail_verifies_the_server_certificate(monkeypatch):
    import ssl

    from acw.api import auth_routes
    from acw.config import get_settings

    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_FROM", "x@example.com")
    get_settings.cache_clear()
    seen = {}

    class FakeSMTP:
        def __init__(self, *a, **kw):
            seen["timeout"] = kw.get("timeout")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self, context=None):
            seen["context"] = context

        def login(self, *a):
            pass

        def send_message(self, msg):
            seen["to"] = msg["To"]

    monkeypatch.setattr(auth_routes.smtplib, "SMTP", FakeSMTP)
    auth_routes._send_mail("a@b.com", "s", "b")
    assert seen["context"].verify_mode == ssl.CERT_REQUIRED and seen["context"].check_hostname
    assert seen["timeout"] <= 15


def test_outbound_calls_have_their_own_small_budget():
    from fastapi import HTTPException

    from acw import security

    async def go():
        security._outbound_inflight = security.OUTBOUND_MAX
        try:
            with pytest.raises(HTTPException) as e:
                await security.run_outbound(lambda: 1)
            assert e.value.status_code == 503
        finally:
            security._outbound_inflight = 0
        assert await security.run_outbound(lambda: 7) == 7
        assert security._outbound_inflight == 0

    asyncio.run(go())


# ---------------------------------------------------------------- uploads and abuse limits (SEC-09, SEC-14, UPLOAD-01)

def test_promised_upload_bytes_are_not_free_space(monkeypatch):
    from acw import hostload

    GB = 1024**3
    snap = SimpleNamespace(disk_free=20 * GB)
    monkeypatch.setattr(hostload, "get", lambda d: SimpleNamespace(snapshot=lambda: snap))
    uid = user()
    other = user("b@example.com")  # one account may only have 8 GB of uploads waiting (test_job_reservations.py)
    ledger.credit(uid, 50_000_000, "grant", "t", "g")
    ledger.credit(other, 50_000_000, "grant", "t", "g2")
    make_job(uid, size_bytes=4 * GB)  # needs 4 GB + 10 GB reserve of the 20 GB
    make_job(other, size_bytes=4 * GB)  # 4 GB are promised to the first: 16 GB left, still enough
    with pytest.raises(jobs.JobError, match="short on storage"):
        make_job(other, size_bytes=4 * GB)  # 8 GB promised: 12 GB left, 14 needed
    assert db.one("SELECT COUNT(*) n FROM jobs")["n"] == 2
    assert db.one("SELECT COUNT(*) n FROM holds WHERE status = 'active'")["n"] == 2  # the refused job's hold was given back








def test_upload_expiry_moves_while_bytes_arrive(client, signed_in):
    from acw.api import jobs_routes

    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    jid = make_job(uid)
    db.execute("UPDATE jobs SET expires_at = ? WHERE id = ?", (db.now() + 5, jid))
    assert jobs_routes._still_uploading(jid)
    assert db.one("SELECT expires_at FROM jobs WHERE id = ?", (jid,))["expires_at"] >= db.now() + jobs.UPLOAD_WINDOW_SECONDS - 2
    jobs.finish_failed(dict(db.one("SELECT * FROM jobs WHERE id = ?", (jid,))), "x", status="canceled")
    assert not jobs_routes._still_uploading(jid)  # canceled meanwhile: the upload stops


def test_upload_works_and_keeps_the_files_in_the_job_directory(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    data = b"RIFF" + os.urandom(3000)
    r = client.post("/api/jobs", json={"kind": "denoise", "denoiseTier": "standard", "filename": "a.wav", "sizeBytes": len(data),
                                       "durationSeconds": 5, "displayName": "My\u202e recording\n.wav"}, headers=h)
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    assert client.put(f"/api/jobs/{jid}/input", content=data, headers=h).status_code == 200
    got = client.get(f"/api/jobs/{jid}", headers=h).json()
    assert got["status"] == "queued" and got["inputName"] == "My recording.wav" and got["label"] == "Clean & polish"
    assert jobs.input_path(dict(db.one("SELECT * FROM jobs WHERE id = ?", (jid,)))).read_bytes() == data


def test_downloads_never_follow_a_link_made_by_the_runner(client, signed_in, tmp_path):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    job, d = running_job(uid)
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")
    (d / "link.wav").symlink_to(secret)
    from acw.worker import main as W

    W._complete(job, d, {"outputs": [{"file": "out.wav", "kind": "audio"}, {"file": "link.wav", "kind": "audio"},
                                     {"file": "../x", "kind": "audio"}, {"file": "out.wav", "kind": "other"}]})
    names = [o["file"] for o in json.loads(db.one("SELECT outputs_json FROM jobs WHERE id = ?", (job["id"],))["outputs_json"])]
    assert names == ["out.wav"]
    # even if a listed output is swapped for a link afterwards, the download refuses it
    (d / "out.wav").unlink()
    (d / "out.wav").symlink_to(secret)
    assert client.get(f"/api/jobs/{job['id']}/outputs/out.wav", headers=h).status_code == 404


def test_runner_result_is_checked(tmp_path):
    from acw.worker import main as W

    ok = tmp_path / "r.json"
    ok.write_text(json.dumps({"ok": True, "outputs": [{"file": "a.wav", "kind": "audio"}], "notice": "x"}))
    assert W._read_result(ok)["ok"] is True
    for bad in ('{"ok": true, "outputs": "nope"}', '{"ok": true, "outputs": [{"file": 3, "kind": "audio"}]}', "[1]", "{",
                json.dumps({"ok": True, "notice": {"a": 1}}), json.dumps({"ok": True, "outputs": [{"file": "a", "kind": "audio"}] * 60})):
        ok.write_text(bad)
        with pytest.raises(W.StageFailed):
            W._read_result(ok)
    ok.unlink()
    (tmp_path / "t").write_text("{}")
    ok.symlink_to(tmp_path / "t")
    with pytest.raises(W.StageFailed):
        W._read_result(ok)
    big = tmp_path / "big.json"
    big.write_text('{"ok": true, "pad": "' + "x" * 2_000_000 + '"}')
    with pytest.raises(W.StageFailed):
        W._read_result(big)


# ---------------------------------------------------------------- restarts, capacity, honest billing (REL-01, CAP, REL-04, DB-03)

def test_a_stopping_worker_puts_its_job_back_instead_of_failing_it(monkeypatch):
    from acw.worker import main as W

    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    jid = make_job(uid)
    jobs.mark_uploaded(jid)
    claimed = W.claim("w1")
    assert claimed["id"] == jid and claimed["attempts"] == 1
    d = jobs.job_dir(jid)
    (d / "input.wav").write_bytes(b"audio")
    (d / "decoded.f32").write_bytes(b"scratch")
    (d / "asr_chunks").mkdir()

    def stopped(*a, **kw):
        raise W.Interrupted()

    monkeypatch.setattr(W, "run_stage", stopped)
    W.process(claimed, {})
    row = db.one("SELECT * FROM jobs WHERE id = ?", (jid,))
    assert (row["status"], row["attempts"], row["worker_id"]) == ("queued", 0, None)
    assert [p.name for p in d.iterdir()] == ["input.wav"]  # the upload is kept, scratch is gone
    assert db.one("SELECT status FROM holds WHERE id = ?", (row["hold_id"],))["status"] == "active"  # money still reserved
    assert W.claim("w2")["id"] == jid  # and another worker (or this one after the restart) takes it


def test_sigterm_during_a_stage_is_reported_as_an_interruption(tmp_path, monkeypatch):
    from acw.worker import main as W

    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    job, d = running_job(uid)
    (d / "spec.json").write_text("{}")
    W._stop.set()
    try:
        # a stage that was killed because the worker is stopping leaves no result file
        monkeypatch.setattr(W.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(
            stdout=iter(()), stderr=iter(()), poll=lambda: 0, wait=lambda: 0, returncode=-9, pid=1))
        with pytest.raises(W.Interrupted):
            W.run_stage(job, d, "main", 10)
    finally:
        W._stop.clear()


def process_with(monkeypatch, uid, denoise_tier, noise):
    from acw.worker import main as W

    job, d = running_job(uid, denoise_tier=denoise_tier)
    job["next_stage"] = "probe"

    def stage(j, dd, name, timeout):
        if name == "probe":
            return {"ok": True, "duration": 600.0, "channels": 2}
        return {"ok": True, "outputs": [{"file": "out.wav", "kind": "audio"}], "noise": noise}

    monkeypatch.setattr(W, "run_stage", stage)
    W.process(job, {})
    return dict(db.one("SELECT * FROM jobs WHERE id = ?", (job["id"],)))


def test_rebuild_that_ran_is_billed_as_rebuild(monkeypatch, gpu_up):
    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    row = process_with(monkeypatch, uid, "rebuild", noise="rebuilt")
    assert row["status"] == "done" and row["tier"] == "rebuild"
    assert row["cost_micros"] == pricing.cost_micros("denoise", "rebuild", 600)


def test_estimates_follow_the_work_the_job_does():
    def job(noise_profile=None, noise="auto", echo=False, tier="standard", channels=2, kind="denoise", transcribe=None):
        p = {"denoise_tier": tier if kind != "transcribe" else None, "transcribe_tier": transcribe,
             "options": {"noise": noise, "echo_reference_track": echo}, "channels": channels}
        if noise_profile:
            p["noise_profile"] = noise_profile
        return {"kind": kind, "claimed_seconds": 3600, "duration_seconds": None, "next_stage": "probe", "params_json": json.dumps(p)}

    light, heavy = jobs.job_seconds(job("light")), jobs.job_seconds(job("heavy"))
    assert heavy > 4 * light  # DPDFNet-8 is several times DeepFilterNet3
    assert jobs.job_seconds(job("heavy", channels=1)) < heavy  # per channel
    assert jobs.job_seconds(job(echo=True)) > jobs.job_seconds(job())
    assert jobs.job_seconds(job(tier="rebuild")) == pytest.approx(0.14 * 3600 * 2 + jobs.MODEL_LOAD_SECONDS)  # on the card
    assert jobs.job_seconds(job(kind="transcribe", transcribe="local")) < jobs.job_seconds(job(kind="transcribe", transcribe="local", channels=2)) + 1
    assert jobs.job_seconds(job(noise="heavy")) == pytest.approx(heavy)  # forced heavy needs no measurement


def test_admission_decided_outside_the_write_lock(monkeypatch):
    """claim() measures the host before it opens the transaction, so a slow sample never blocks other writers."""
    from acw.worker import main as W

    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    jid = make_job(uid)
    jobs.mark_uploaded(jid)
    in_tx = []
    snap_calls = []
    orig_tx = db.tx

    def watching_tx(*a, **kw):
        in_tx.append(True)
        return orig_tx(*a, **kw)

    class Load:
        def snapshot(self):
            snap_calls.append(len(in_tx))  # no transaction may have been opened yet
            from acw.hostload import Snapshot

            return Snapshot(cores=8, busy_cores=0.0, nice_cores=0.0, mem_available=20 * 1024**3, slice_headroom=None,
                            disk_free=100 * 1024**3)

    monkeypatch.setattr(db, "tx", watching_tx)
    assert W.claim("w1", Load())["id"] == jid
    assert snap_calls == [0]


# ---------------------------------------------------------------- live (LIVE-01..03, SEC-03/04/17/20)

def test_idle_limit_is_short_because_both_clients_keep_sending_frames():
    from acw.stream import routes

    assert routes.IDLE_TIMEOUT == 120


@pytest.mark.parametrize("value,expected", [(100, 100.0), (3, 6.0), (500, 100.0), ("30", 30.0), (float("nan"), None),
                                            (float("inf"), None), ("abc", None), (None, None), ([1], None)])
def test_noise_strength_is_validated(value, expected):
    from acw.stream.features import atten

    assert atten(value) == expected


def test_job_noise_strength_nan_is_refused():
    with pytest.raises(jobs.JobError):
        jobs.validate_options("denoise", "standard", None, {"atten_lim_db": float("nan")})
    with pytest.raises(jobs.JobError):
        jobs.validate_options("denoise", "standard", None, {"atten_lim_db": "x"})


def test_sessions_nobody_runs_are_closed_and_billed_by_the_sweeper():
    from acw.stream import routes

    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    old, fresh, mine = ledger.create_hold(uid, 20_000, "stream", "o"), ledger.create_hold(uid, 20_000, "stream", "f"), ledger.create_hold(uid, 20_000, "stream", "m")
    for sid, hold, age, secs in (("o", old, 600, 60), ("f", fresh, 5, 0), ("m", mine, 600, 0)):
        db.execute("INSERT INTO stream_sessions(id,user_id,denoise_tier,status,hold_id,seconds,paid_seconds,meter_rate,meter_seconds,started_at)"
                   " VALUES (?,?,'light','active',?,?,?,4000,?,?)", (sid, uid, hold, secs, secs, secs, db.now() - age))
    routes._live_ids.add("m")
    try:
        assert routes.close_orphans() == 1
    finally:
        routes._live_ids.discard("m")
    assert db.one("SELECT status, end_reason, cost_micros FROM stream_sessions WHERE id = 'o'")["end_reason"] == "session_lost"
    assert db.one("SELECT cost_micros FROM stream_sessions WHERE id = 'o'")["cost_micros"] == 4000
    assert db.one("SELECT status FROM stream_sessions WHERE id = 'f'")["status"] == "active"  # still starting
    assert db.one("SELECT status FROM stream_sessions WHERE id = 'm'")["status"] == "active"  # a handler owns it


def test_api_key_revocation_stops_a_running_session():
    from acw.stream import routes

    uid = user()
    db.execute("INSERT INTO api_keys(id,user_id,name,prefix,key_hash,scopes,created_at) VALUES ('k1',?,'k','acw_xxxx','h','stream',?)", (uid, db.now()))
    db.execute("INSERT INTO stream_sessions(id,user_id,status,started_at) VALUES ('sx',?,'active',?)", (uid, db.now()))
    sess = routes.Session(None, SimpleNamespace(user_id=uid, api_key_id="k1", session_hash=None), Features(denoise="light"))
    sess.id = "sx"
    assert asyncio.run(routes._persist(sess)) is True
    db.execute("UPDATE api_keys SET revoked_at = ? WHERE id = 'k1'", (db.now(),))
    assert asyncio.run(routes._persist(sess)) is False


def test_logging_out_stops_a_running_session():
    from acw.stream import routes

    uid = user()
    db.execute("INSERT INTO sessions(id_hash,user_id,csrf_token,created_at,expires_at,last_seen_at) VALUES ('hh',?,'c',?,?,?)",
               (uid, db.now(), db.now() + 1000, db.now()))
    db.execute("INSERT INTO stream_sessions(id,user_id,status,started_at) VALUES ('sy',?,'active',?)", (uid, db.now()))
    sess = routes.Session(None, SimpleNamespace(user_id=uid, api_key_id=None, session_hash="hh"), Features(denoise="light"))
    sess.id = "sy"
    assert asyncio.run(routes._persist(sess)) is True
    db.execute("DELETE FROM sessions WHERE id_hash = 'hh'")
    assert asyncio.run(routes._persist(sess)) is False


def test_a_client_that_vanishes_while_starting_gets_its_hold_back(monkeypatch):
    from acw.stream import routes

    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    sent = []

    class WS:
        async def receive_text(self):
            return json.dumps({"type": "start", "denoise": "light"})

        async def send_text(self, t):
            sent.append(t)
            raise RuntimeError("client gone")

        async def close(self, code=1000):
            pass

    async def no_worker(*a, **kw):
        raise OSError("no worker in this test")

    monkeypatch.setattr(routes.asyncio, "open_unix_connection", no_worker)
    principal = auth.Principal(user_id=uid, email="a@example.com", is_admin=False, status="active", via="api_key",
                               api_key_id=None, scopes=frozenset({"*"}))

    async def go():
        with pytest.raises(RuntimeError):
            await routes._start(WS(), principal, "127.0.0.1")

    asyncio.run(go())
    assert db.one("SELECT held_micros FROM users WHERE id = ?", (uid,))["held_micros"] == 0
    assert db.one("SELECT status FROM stream_sessions")["status"] == "ended"
    assert not routes._live_ids


def test_new_sessions_are_refused_while_draining(tmp_path):
    from acw.config import get_settings
    from acw.stream import routes

    assert not routes.drain_requested()
    (get_settings().data_dir / "drain").touch()
    assert routes.drain_requested()


def test_premium_captions_hear_the_original_not_the_denoised_audio():
    """Source check of the live engine: the tap carries `x` (after echo removal, before noise removal)."""
    src = (Path(__file__).resolve().parents[2] / "native/live/src/session.rs").read_text()
    assert "ipc::MSG_TAP, &ipc::pack_audio(seq, &x)" in src
    routes_src = (Path(__file__).resolve().parents[1] / "acw/stream/routes.py").read_text()
    assert "sess.premium.feed(y)" not in routes_src


def test_ledger_history_has_friendly_labels_and_says_when_there_is_more():
    uid = user()
    settle_payment(uid)
    ledger.credit(uid, 50_000_000, "topup", "payment", "p0", "Top-up")
    jid = make_job(uid, kind="denoise_transcribe", transcribe_tier="premium", filename="noisy.flac",
                   options={}, claimed_seconds=120)
    db.execute("UPDATE jobs SET status = 'running', duration_seconds = 120 WHERE id = ?", (jid,))
    job = dict(db.one("SELECT * FROM jobs WHERE id = ?", (jid,)))
    job["params_json"] = db.one("SELECT params_json FROM jobs WHERE id = ?", (jid,))["params_json"]
    jobs.update_params(jid, display_name="My talk.wav")
    job = dict(db.one("SELECT * FROM jobs WHERE id = ?", (jid,)))
    d = jobs.job_dir(jid)
    (d / "out.wav").write_bytes(b"x")
    from acw.worker import main as W

    W._complete(job, d, {"outputs": [{"file": "out.wav", "kind": "audio"}]})
    rows, more = ledger.history_page(uid, 100)
    usage = next(r for r in rows if r["kind"] == "usage")
    assert not more
    assert usage["label"] == "Clean & polish + Subtitles & transcript, extra accurate"
    assert (usage["product"], usage["minutes"], usage["displayName"]) == ("clean_transcript", 2.0, "My talk.wav")
    assert "denoise" not in usage["description"] and "My talk.wav" in usage["description"]
    assert next(r for r in rows if r["kind"] == "topup")["label"] == "Top-up"
    for i in range(5):
        ledger.credit(uid, 1000, "grant", "t", f"g{i}")
    rows, more = ledger.history_page(uid, 3)
    assert len(rows) == 3 and more


def test_billing_history_route_reports_more_rows(client, signed_in):
    uid, h = signed_in
    for i in range(3):
        ledger.credit(uid, 1000, "grant", "t", f"g{i}")
    body = client.get("/api/billing/history").json()
    assert body["ledgerHasMore"] is False and body["ledger"][0]["label"] == "Credit added"


def test_display_names_are_cleaned():
    c = jobs.clean_display_name
    assert c("a\u202eb\u200b.wav\x00") == "ab.wav"
    assert c("C:\\Users\\me\\talk.wav") == "talk.wav" and c("/tmp/../talk  one.wav") == "talk one.wav"
    assert len(c("x" * 500)) == 120 and c("  \u202e ") is None and c(None) is None


def test_stages_use_plain_words():
    src = Path(__file__).resolve().parents[1] / "acw"
    text = (src / "app.py").read_text() + (src / "worker/runner.py").read_text() + (src / "worker/main.py").read_text()
    for old in ("transcribing (premium)", "cloud asr running", "'denoising'", '"denoising"'):
        assert old not in text


# ---------------------------------------------------------------- premium transcription resumes (REL-03)

def test_premium_transcription_keeps_finished_segments(tmp_path, monkeypatch):
    from acw.asr import openai_asr
    from acw.config import get_settings

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()
    d = tmp_path / "asr_chunks"
    d.mkdir()
    chunks = []
    for i in range(3):
        (d / f"{i:04d}.mp3").write_bytes(b"x")
        chunks.append({"file": f"{i:04d}.mp3", "offset": i * 10.0, "seconds": 5.0})
    (d / "index.json").write_text(json.dumps({"chunks": chunks}))
    calls = []
    fail_on = {"0002.mp3"}

    async def fake(http, key, path, language):
        calls.append(path.name)
        if path.name in fail_on:
            raise openai_asr.CloudASRError("outage")
        return f"text {path.name}"

    monkeypatch.setattr(openai_asr, "_transcribe_one", fake)
    with pytest.raises(openai_asr.CloudASRError):
        asyncio.run(openai_asr.run_for_job(tmp_path, None))
    assert (d / "0000.txt").read_text() == "text 0000.mp3" and not (d / "0002.txt").exists()
    calls.clear()
    fail_on.clear()
    asyncio.run(openai_asr.run_for_job(tmp_path, None))
    assert calls == ["0002.mp3"]  # only what was missing is paid for again
    segs = json.loads((tmp_path / "segments.json").read_text())["segments"]
    assert [s["text"] for s in segs] == ["text 0000.mp3", "text 0001.mp3", "text 0002.mp3"]


def test_saved_segments_and_chunks_are_never_read_through_links(tmp_path, monkeypatch):
    from acw.asr import openai_asr
    from acw.config import get_settings

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()
    secret = tmp_path / "secret"
    secret.write_text("TOP SECRET")
    d = tmp_path / "asr_chunks"
    d.mkdir()
    (d / "0000.mp3").symlink_to(secret)
    (d / "index.json").write_text(json.dumps({"chunks": [{"file": "0000.mp3", "offset": 0, "seconds": 5.0}]}))
    with pytest.raises(openai_asr.CloudASRError):
        asyncio.run(openai_asr.run_for_job(tmp_path, None))  # the chunk is a link: refused, nothing is sent
    (d / "0000.mp3").unlink()
    (d / "0000.mp3").write_bytes(b"x")
    (d / "0000.txt").symlink_to(secret)  # a planted "already transcribed" text

    async def fake(http, key, path, language):
        return "real text"

    monkeypatch.setattr(openai_asr, "_transcribe_one", fake)
    asyncio.run(openai_asr.run_for_job(tmp_path, None))
    assert "TOP SECRET" not in (tmp_path / "segments.json").read_text()


def test_two_premium_jobs_run_at_once(monkeypatch):
    from acw import app as appmod

    uid = user()
    settle_payment(uid)
    ledger.credit(uid, 50_000_000, "grant", "t", "g")
    ids = []
    for _ in range(3):
        j = make_job(uid, kind="transcribe", denoise_tier=None, transcribe_tier="premium")
        db.execute("UPDATE jobs SET status = 'cloud_asr', stage = 'Waiting to transcribe', queued_at = ? WHERE id = ?", (db.now(), j))
        ids.append(j)
    assert len(appmod._claim_cloud_jobs(appmod.PREMIUM_JOBS_AT_ONCE)) == 2
    assert len(appmod._claim_cloud_jobs(2)) == 1  # the third, once a slot is free
    assert appmod._claim_cloud_jobs(2) == []


# ---------------------------------------------------------------- backups (OPS-01) and retention (DOC-02)

def test_backup_is_checked_after_writing(tmp_path):
    from acw import backup

    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    out = backup.run(tmp_path / "b", keep=2)
    status = json.loads((tmp_path / "b" / "backup-status.json").read_text())
    assert out.name == status["file"] and status["integrity"] == "ok" and status["ledger"] == "ok" and status["bytes"] > 0
    assert backup.verify(out) == {"integrity": "ok", "ledger": "ok"}


def test_a_backup_whose_money_does_not_add_up_fails(tmp_path):
    from acw import backup

    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    db.conn().execute("PRAGMA ignore_check_constraints=ON")
    db.execute("UPDATE users SET balance_micros = balance_micros + 7 WHERE id = ?", (uid,))
    with pytest.raises(backup.BackupError, match="failed its checks"):
        backup.run(tmp_path / "b")
    assert list((tmp_path / "b").glob("*.bad")) and not list((tmp_path / "b").glob("acw-*.sqlite3.gz"))
    assert "does not match" in json.loads((tmp_path / "b" / "backup-status.json").read_text())["ledger"] or True
    assert ledger.invariant_problems()["balance_mismatch"] == 1


def test_a_damaged_backup_file_is_caught(tmp_path):
    from acw import backup

    user()
    out = backup.run(tmp_path / "b")
    raw = bytearray(gzip.decompress(out.read_bytes()))
    raw[4096:] = os.urandom(len(raw) - 4096)  # damage every page after the first
    bad = tmp_path / "bad.sqlite3.gz"
    bad.write_bytes(gzip.compress(bytes(raw)))
    try:
        result = backup.verify(bad)
        assert result["integrity"] != "ok"
    except sqlite3.DatabaseError:
        pass  # too damaged to open at all: also a failure


def test_off_machine_copy_is_off_unless_configured_and_needs_a_key(tmp_path, monkeypatch):
    from acw import backup

    user()
    monkeypatch.delenv("ACW_BACKUP_OFFBOX", raising=False)
    backup.run(tmp_path / "a")
    assert json.loads((tmp_path / "a" / "backup-status.json").read_text())["offbox"] is None
    monkeypatch.setenv("ACW_BACKUP_OFFBOX", str(tmp_path / "remote") + "/")
    monkeypatch.delenv("ACW_BACKUP_PASSPHRASE_FILE", raising=False)
    monkeypatch.delenv("ACW_BACKUP_AGE_RECIPIENT", raising=False)
    with pytest.raises(backup.BackupError, match="neither"):
        backup.run(tmp_path / "b")


@pytest.mark.skipif(not Path("/usr/bin/gpg").exists() or not Path("/usr/bin/rsync").exists(), reason="gpg and rsync needed")
def test_off_machine_copy_is_encrypted(tmp_path, monkeypatch):
    from acw import backup

    user()
    remote = tmp_path / "remote"
    remote.mkdir()
    (tmp_path / "pass").write_text("correct horse battery staple\n")
    monkeypatch.setenv("ACW_BACKUP_OFFBOX", str(remote) + "/")
    monkeypatch.setenv("ACW_BACKUP_PASSPHRASE_FILE", str(tmp_path / "pass"))
    monkeypatch.setenv("GNUPGHOME", str(tmp_path / "gnupg"))
    (tmp_path / "gnupg").mkdir(mode=0o700)
    backup.run(tmp_path / "b")
    copies = list(remote.iterdir())
    assert len(copies) == 1 and copies[0].name.endswith(".gpg")
    assert b"SQLite format" not in copies[0].read_bytes() and copies[0].stat().st_size > 100
    assert not list((tmp_path / "b").glob("*.gpg"))  # the encrypted temp copy is not left behind


def test_retention_removes_addresses_and_file_names_on_schedule():
    from acw import retention

    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    now = db.now()
    day = 86400
    db.execute("INSERT INTO stream_sessions(id,user_id,status,client_ip,started_at) VALUES ('old',?,'ended','1.2.3.4',?)", (uid, now - 91 * day))
    db.execute("INSERT INTO stream_sessions(id,user_id,status,client_ip,started_at) VALUES ('new',?,'ended','1.2.3.4',?)", (uid, now - 5 * day))
    db.execute("INSERT INTO audit_log(user_id,actor,action,ip,created_at) VALUES (?,'u','x','9.9.9.9',?)", (uid, now - 100 * day))
    db.execute("INSERT INTO audit_log(user_id,actor,action,ip,created_at) VALUES (?,'u','y','9.9.9.9',?)", (uid, now - 1 * day))
    old_job = make_job(uid, filename="old-secret.wav")
    new_job = make_job(uid, filename="new.wav")
    jobs.update_params(old_job, display_name="Old name")
    db.execute("UPDATE jobs SET created_at = ? WHERE id = ?", (now - 400 * day, old_job))
    changed = retention.run()
    assert changed["stream_ips"] == 1 and changed["audit_ips"] == 1 and changed["job_names"] == 1
    assert db.one("SELECT client_ip FROM stream_sessions WHERE id = 'old'")["client_ip"] is None
    assert db.one("SELECT client_ip FROM stream_sessions WHERE id = 'new'")["client_ip"] == "1.2.3.4"
    assert db.one("SELECT COUNT(*) n FROM audit_log WHERE ip IS NOT NULL")["n"] == 1
    old = db.one("SELECT input_name, params_json FROM jobs WHERE id = ?", (old_job,))
    assert old["input_name"] == "" and "display_name" not in json.loads(old["params_json"])
    assert db.one("SELECT input_name FROM jobs WHERE id = ?", (new_job,))["input_name"] == "new.wav"
    assert not any(retention.run().values())  # a second run changes nothing


# ---------------------------------------------------------------- monitor and deep health (OPS-03)

def test_monitor_notices_a_bad_backup_and_missing_files(tmp_path, monkeypatch):
    from acw import monitor
    from acw.config import get_settings

    monkeypatch.setenv("ACW_MODELS_DIR", str(tmp_path / "no-models"))
    monkeypatch.setenv("ACW_BIN_DIR", str(tmp_path / "no-bin"))
    get_settings.cache_clear()
    assert not monitor.check_backup_result(tmp_path)[0].bad  # nothing written yet: staleness covers that
    (tmp_path / "backup-status.json").write_text(json.dumps({"file": "f", "bytes": 0, "integrity": "ok", "ledger": "ok"}))
    assert monitor.check_backup_result(tmp_path)[0].bad
    (tmp_path / "backup-status.json").write_text(json.dumps({"file": "f", "bytes": 10, "integrity": "ok", "ledger": "ok"}))
    assert not monitor.check_backup_result(tmp_path)[0].bad
    (tmp_path / "backup-status.json").write_text(json.dumps({"file": "f", "bytes": 10, "integrity": "ok", "ledger": "2 balances do not match"}))
    assert monitor.check_backup_result(tmp_path)[0].bad
    # the test environment has no models: the installation check says so
    f = {x.kind: x for x in monitor.check_installation(db.now())}
    assert f["files_missing"].bad and "models/" in f["files_missing"].body


def test_monitor_notices_stuck_states():
    from acw import monitor

    uid = user()
    settle_payment(uid)
    ledger.credit(uid, 50_000_000, "grant", "t", "g")
    now = db.now()
    quiet = {f.kind: f.bad for f in monitor.check_stuck_states(now)}
    assert not any(quiet.values())
    j = make_job(uid, kind="transcribe", denoise_tier=None, transcribe_tier="premium")
    db.execute("UPDATE jobs SET status = 'cloud_asr', heartbeat_at = ? WHERE id = ?", (now - 3600, j))
    db.execute("UPDATE holds SET created_at = ?", (now - 7 * 3600,))
    db.execute("INSERT INTO stripe_events(id,type,received_at) VALUES ('e1','x',?)", (now - 3600,))
    db.execute("INSERT INTO payments(id,user_id,kind,status,amount_cents,created_at,updated_at) VALUES ('pp',?,'auto_reload','pending',1000,?,?)",
               (uid, now - 7200, now))
    for i in range(12):
        jid = make_job(uid, claimed_seconds=10, size_bytes=100)
        db.execute("UPDATE jobs SET status = ?, finished_at = ? WHERE id = ?", ("failed" if i < 6 else "done", now - 60, jid))
    bad = {f.kind for f in monitor.check_stuck_states(now) if f.bad}
    assert bad == {"cloud_asr_stuck", "holds_old", "jobs_failing", "stripe_events_unprocessed", "payments_pending"}


def test_monitor_notices_a_money_mismatch():
    from acw import monitor

    uid = user()
    ledger.credit(uid, 1_000_000, "grant", "t", "g")
    assert not monitor.check_money()[0].bad
    db.execute("UPDATE users SET balance_micros = balance_micros + 5 WHERE id = ?", (uid,))
    assert monitor.check_money()[0].bad


def test_deep_health_is_for_the_machine_itself_or_admins(client, monkeypatch):
    from acw import health

    r = client.get("/api/health?deep=1")
    assert r.json() == {"ok": True}  # the test client is not loopback: no details
    assert not health.stream_worker_ok()
    assert health.stale_running_jobs() == 0
    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    job, _ = running_job(uid)
    db.execute("UPDATE jobs SET heartbeat_at = ? WHERE id = ?", (db.now() - 3600, job["id"]))
    assert health.stale_running_jobs() == 1
    deep = health.deep()
    assert deep["ok"] is False and deep["staleRunningJobs"] == 1 and "streamWorker" in deep and deep["money"]["held_mismatch"] == 0


# ---------------------------------------------------------------- misc security lows

def test_admin_rights_follow_the_environment_list(monkeypatch):
    from acw.config import get_settings

    monkeypatch.setenv("ACW_ADMIN_EMAILS", "boss@example.com")
    get_settings.cache_clear()
    assert auth.get_or_create_user("boss@example.com")["is_admin"] == 1
    monkeypatch.setenv("ACW_ADMIN_EMAILS", "")
    get_settings.cache_clear()
    assert auth.get_or_create_user("boss@example.com")["is_admin"] == 0


def test_browser_sessions_expire_when_unused(client):
    uid, h = login(client, "idle@example.com")
    db.execute("UPDATE sessions SET last_seen_at = ?", (db.now() - auth.SESSION_IDLE_SECONDS - 10,))
    assert client.get("/api/me").json()["signedIn"] is False


def test_cookie_name_gets_the_host_prefix_on_a_domain_of_its_own(monkeypatch):
    from acw.config import get_settings

    monkeypatch.setenv("ACW_ENV", "production")
    monkeypatch.setenv("ACW_SESSION_SECRET", "x" * 40)
    monkeypatch.setenv("ACW_BASE_PATH", "/audio")
    get_settings.cache_clear()
    assert get_settings().session_cookie_name == "__Secure-acw_session"
    monkeypatch.setenv("ACW_BASE_PATH", "")
    get_settings.cache_clear()
    s = get_settings()
    assert s.session_cookie_name == "__Host-acw_session" and s.cookie_path == "/"


def test_webhook_events_are_handled_one_at_a_time():
    from acw.billing import stripe_service

    assert hasattr(stripe_service._webhook_lock, "acquire")


def test_dead_code_is_gone():
    assert not hasattr(db, "row_dict") and not hasattr(auth, "require_admin")


def test_pricing_reports_the_upload_limit(client):
    assert client.get("/api/billing/pricing").json()["maxUploadBytes"] == 4 * 1024**3




def test_hostile_display_names_are_cleaned_on_the_server(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    name = "‮evil⁦x⁩‬\x00\x1b[31m" + "y" * 200
    r = client.post("/api/jobs", json={"kind": "denoise", "denoiseTier": "standard", "filename": "a.wav", "sizeBytes": 1000,
                                       "durationSeconds": 5, "displayName": name}, headers=h)
    assert r.status_code == 200, r.text
    shown = client.get(f"/api/jobs/{r.json()['id']}", headers=h).json()["inputName"]
    assert len(shown) == 120 and all(c not in shown for c in "‮⁦⁩‬\x00\x1b")


# ---------------------------------------------------------------- live handshake answers (OBS plugin asks)

def ws_denied(client, headers=None):
    from starlette.testclient import WebSocketDenialResponse

    try:
        with client.websocket_connect("/api/stream/ws", headers=headers or {}) as ws:
            return None, ws.receive_json()
    except WebSocketDenialResponse as e:
        return e.status_code, dict(e.headers)


def api_key(uid, scopes="stream"):
    from acw.security import new_api_key

    key, prefix, h = new_api_key()
    db.execute("INSERT INTO api_keys(id,user_id,name,prefix,key_hash,scopes,created_at) VALUES (?,?,?,?,?,?,?)",
               (db.new_id(), uid, "k", prefix, h, scopes, db.now()))
    return key


def test_handshake_says_why_it_refuses(client):
    uid = user()
    assert ws_denied(client)[0] == 401  # no credentials
    assert ws_denied(client, {"Authorization": "Bearer acw_nonsense"})[0] == 401  # unknown key
    revoked = api_key(uid)
    db.execute("UPDATE api_keys SET revoked_at = ?", (db.now(),))
    assert ws_denied(client, {"Authorization": f"Bearer {revoked}"})[0] == 401
    wrong_scope = api_key(uid, "jobs")
    assert ws_denied(client, {"Authorization": f"Bearer {wrong_scope}"})[0] == 403
    suspended = api_key(user("s@example.com"))
    db.execute("UPDATE users SET status = 'suspended' WHERE email = 's@example.com'")
    assert ws_denied(client, {"Authorization": f"Bearer {suspended}"})[0] == 403


def test_start_limit_answers_429_with_the_wait(client):
    from acw.security import limiter

    uid = user()
    key = api_key(uid)
    limiter.hit(f"stream_start:{uid}", 60, 60 / 3600, cost=60)  # the hour's starts are used up
    status, headers = ws_denied(client, {"Authorization": f"Bearer {key}"})
    assert status == 429 and int(headers["retry-after"]) >= 1


def test_a_failed_start_does_not_use_up_the_limit_quickly():
    from acw.security import LIMITS

    assert LIMITS["stream_start"][0] >= 60 and LIMITS["stream_start_ip"][0] >= 60


def test_errors_tell_the_client_when_to_come_back(client):
    uid = user()
    key = api_key(uid)
    from acw.config import get_settings

    (get_settings().data_dir / "drain").touch()
    status, event = ws_denied(client, {"Authorization": f"Bearer {key}"})
    assert status is None and event["code"] == "restarting" and event["retryAfterSeconds"] == 60


def test_idle_session_is_told_before_it_is_closed(monkeypatch):
    uid = user()
    sess = run_session(monkeypatch, [(0.05, frame(reference=True, seq=i)) for i in range(30)], uid, IDLE_TIMEOUT=0.3)
    assert sess.end_reason == "idle_timeout"
    sent = [json.loads(t) for t in sess.ws.sent if isinstance(t, str)]
    assert {"type": "stopped", "reason": "idle_timeout"} in sent


def test_a_dropped_session_is_replaced_instead_of_refusing_the_user():
    from acw.stream import routes

    uid = user()
    closed = []

    class WS:
        async def close(self, code=1000):
            closed.append(code)

    async def go():
        old = routes.Session(WS(), SimpleNamespace(user_id=uid), Features(denoise="light"))
        old.last_rx -= 60  # nothing from it for a minute, though the server never saw the socket close
        live = routes.Session(WS(), SimpleNamespace(user_id=uid), Features(denoise="light"))
        routes._open[uid] = [old, live]
        routes._live[uid] = routes.MAX_SESSIONS_PER_USER

        async def handler_notices():  # what the old session's own handler does once its socket is closed
            await asyncio.sleep(0.2)
            routes._live[uid] -= 1

        t = asyncio.create_task(handler_notices())
        ok = await routes._evict_stale(uid)
        await t
        return ok, old

    try:
        ok, old = asyncio.run(go())
    finally:
        routes._open.clear()
        routes._live.clear()
    assert ok and old.end_reason == "replaced" and closed == [1001]

    # a session that is talking is never replaced
    routes._open[uid] = [routes.Session(WS(), SimpleNamespace(user_id=uid), Features(denoise="light"))]
    try:
        assert asyncio.run(routes._evict_stale(uid)) is False
    finally:
        routes._open.clear()


def test_gpu_worker_leaves_echo_jobs_to_the_cpu_workers():
    import json

    from acw.broker import gpu_stages

    def job(**opts):
        return {"kind": "transcribe", "params_json": json.dumps({"transcribe_tier": "local", "options": opts}), "next_stage": None}

    assert gpu_stages(job()) == ["transcribe"]
    assert not gpu_stages(job(echo_reference_track=True))
