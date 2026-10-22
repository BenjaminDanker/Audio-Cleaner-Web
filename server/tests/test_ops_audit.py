"""Operations and security fixes: request body limits, log privacy, rate limits, login mail, uploads and downloads, worker
restarts that don't fail jobs, premium transcription that resumes, backups, retention, the monitor, deep health, and the
live handshake answers. Money paths are in test_billing_audit.py; uploads' abuse limits in test_abuse_limits.py."""
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

from .conftest import ORIGIN, login, settle_payment
from .job_support import queue_row, rebuilt
from .test_billing_audit import make_job, running_job, user


# ---------------------------------------------------------------- request bodies and the log

def test_database_file_is_private_and_commits_are_durable(client):
    assert db.one("PRAGMA synchronous")[0] == 2  # FULL
    assert oct(os.stat(db.get_settings().db_path).st_mode & 0o777) == "0o600"


def test_oversized_bodies_are_refused_before_they_are_read_and_the_limit_follows_the_base_path(client):
    from acw.bodylimit import BodyLimitMiddleware

    json_h = {"Origin": ORIGIN, "content-type": "application/json"}
    assert client.post("/api/auth/email", content=b'{"email": "' + b"a" * 200_000 + b'"}', headers=json_h).status_code == 413

    def streamed():  # 160 KB with no Content-Length: counted while it arrives
        for _ in range(40):
            yield b"x" * 4096

    assert client.post("/api/auth/email", content=streamed(), headers=json_h).status_code == 413
    assert client.post("/api/stripe/webhook", content=b"x" * 1_200_000).status_code == 413
    assert client.post("/api/stripe/webhook", content=b"x" * 2000).status_code != 413
    for base in ("", "/audio"):  # its own domain (production) and a shared host
        mw = BodyLimitMiddleware(None, base_path=base)
        assert mw.limit_for(f"{base}/api/auth/email") == 64 * 1024
        assert mw.limit_for(f"{base}/api/stripe/webhook") == 1024 * 1024
        assert mw.limit_for(f"{base}/api/jobs/" + "a" * 32 + "/input") is None  # uploads have their own rules


def test_the_access_log_never_has_a_query_string(client, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="acw.access"):
        client.get("/api/auth/oauth/google/callback?code=SECRETCODE&state=SECRETSTATE")
        client.get("/api/me")
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "SECRETCODE" not in text and "SECRETSTATE" not in text
    assert "/api/me" in text


# ---------------------------------------------------------------- rate limits

def test_429_says_when_to_try_again(client):
    from acw.security import limiter

    for _ in range(130):
        r = client.get("/api/health")
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 1 and "try again in" in r.json()["detail"]
    limiter.reset()


def test_limits_are_per_name_ipv6_counts_per_64_and_a_flood_of_keys_does_not_forget_a_limit():
    from acw.security import RateLimiter, client_net

    assert client_net("2001:db8:1:2:aaaa:bbbb:cccc:dddd") == client_net("2001:db8:1:2::1") == "2001:db8:1:2::/64"
    assert client_net("203.0.113.9") == "203.0.113.9"
    lim = RateLimiter(max_keys=3)
    assert lim.hit("api:keep", 1, 0.0001)
    for i in range(10):  # a flood of made-up keys of another kind
        lim.hit(f"email_addr:{i}", 1, 0.0001)
    assert not lim.hit("api:keep", 1, 0.0001)  # still remembered: the flood did not push it out


def test_a_fast_upload_sender_is_slowed_to_the_rate():
    from acw.security import Throttle

    async def go():
        t = Throttle(1_000_000)  # 1 MB/s with a 1 MB burst
        t0 = time.monotonic()
        for _ in range(2):
            await t.take(600_000)  # 1.2 MB: about 0.2 s after the burst
        return time.monotonic() - t0

    assert 0.1 < asyncio.run(go()) < 2


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


# ---------------------------------------------------------------- login mail

def test_addresses_that_would_mail_more_than_one_recipient_are_refused():
    from acw.api.auth_routes import clean_email

    for bad in ("x,attacker@evil.com", "a;b@x.com", "a@b.com\nBcc: x@y.com", "a@b@c.com", "a\x00@b.com"):
        assert clean_email(bad) is None, bad
    assert clean_email("  Some.One+tag@Example.com ") == "some.one+tag@example.com"


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


# ---------------------------------------------------------------- uploads, downloads and the runner's result

def test_promised_upload_bytes_are_not_free_space(monkeypatch):
    from acw import hostload

    GB = 1024**3
    snap = SimpleNamespace(disk_free=20 * GB)
    monkeypatch.setattr(hostload, "get", lambda d: SimpleNamespace(snapshot=lambda: snap))
    uid = user()
    other = user("b@example.com")
    ledger.credit(uid, 50_000_000, "grant", "t", "g")
    ledger.credit(other, 50_000_000, "grant", "t", "g2")
    make_job(uid, size_bytes=4 * GB)  # needs 4 GB + 10 GB reserve of the 20 GB
    make_job(other, size_bytes=4 * GB)  # 4 GB are promised to the first: 16 GB left, still enough
    with pytest.raises(jobs.JobError, match="short on storage"):
        make_job(other, size_bytes=4 * GB)  # 8 GB promised: 12 GB left, 14 needed
    assert db.one("SELECT COUNT(*) n FROM jobs")["n"] == 2
    assert db.one("SELECT COUNT(*) n FROM holds WHERE status = 'active'")["n"] == 2  # the refused job's hold was given back


def test_an_upload_is_kept_in_its_job_folder_under_a_cleaned_name(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    data = b"RIFF" + os.urandom(3000)
    r = client.post("/api/jobs", json={"kind": "denoise", "denoiseTier": "standard", "filename": "a.wav", "sizeBytes": len(data),
                                       "durationSeconds": 5, "displayName": "My\u202e recording\n.wav"}, headers=h)
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    assert client.put(f"/api/jobs/{jid}/input", content=data, headers=h).status_code == 200
    got = client.get(f"/api/jobs/{jid}", headers=h).json()
    assert got["status"] == "queued" and got["inputName"] == "My recording.wav" and got["label"] == "Denoising + Leveling"
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


def test_the_runners_result_file_is_checked_and_links_and_huge_files_are_refused(tmp_path):
    from acw.worker import main as W

    ok = tmp_path / "r.json"
    ok.write_text(json.dumps({"ok": True, "outputs": [{"file": "a.wav", "kind": "audio"}], "notice": "x"}))
    assert W._read_result(ok)["ok"] is True
    for bad in ('{"ok": true, "outputs": "nope"}', '{"ok": true, "outputs": [{"file": 3, "kind": "audio"}]}', "[1]", "{",
                json.dumps({"ok": True, "outputs": [{"file": "a", "kind": "audio"}] * 60})):
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


# ---------------------------------------------------------------- restarts, billing of a rebuild, admission

def test_a_stopping_worker_puts_its_job_back_instead_of_failing_it_and_the_money_stays_reserved(monkeypatch):
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


def test_a_rebuild_that_ran_is_billed_as_a_rebuild(monkeypatch, gpu_up):
    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    row = process_with(monkeypatch, uid, "rebuild", noise="rebuilt")
    assert row["status"] == "done"
    assert row["cost_micros"] == rebuilt(600)  # (the rebuild step and Leveling, each at its own rate)
    assert db.one("SELECT description FROM ledger WHERE user_id = ? AND kind = 'usage'", (uid,))["description"].startswith("Voice rebuild + Leveling")


def test_admission_is_decided_before_the_write_lock_opens(monkeypatch):
    """claim() measures the host before it opens the transaction, so a slow sample never blocks other writers."""
    from acw.hostload import Snapshot
    from acw.worker import main as W

    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    jid = make_job(uid)
    jobs.mark_uploaded(jid)
    in_tx, snap_calls = [], []
    orig_tx = db.tx

    def watching_tx(*a, **kw):
        in_tx.append(True)
        return orig_tx(*a, **kw)

    class Load:
        def snapshot(self):
            snap_calls.append(len(in_tx))  # no transaction may have been opened yet
            return Snapshot(cores=8, busy_cores=0.0, nice_cores=0.0, mem_available=20 * 1024**3, slice_headroom=None,
                            disk_free=100 * 1024**3)

    monkeypatch.setattr(db, "tx", watching_tx)
    assert W.claim("w1", Load())["id"] == jid
    assert snap_calls == [0]


def test_a_gpu_worker_leaves_echo_jobs_to_the_cpu_workers():
    from acw.broker import gpu_stages

    def job(**opts):
        return queue_row("denoise_transcribe", "standard", "local", {"noise": "light", **opts})

    assert gpu_stages(job()) == ["transcribe"]
    assert not gpu_stages(job(echo_reference_track=True))


# ---------------------------------------------------------------- live: sessions nobody runs, revoked keys, handshakes

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
    assert db.one("SELECT end_reason, cost_micros FROM stream_sessions WHERE id = 'o'")["end_reason"] == "session_lost"
    assert db.one("SELECT cost_micros FROM stream_sessions WHERE id = 'o'")["cost_micros"] == 4000
    assert db.one("SELECT status FROM stream_sessions WHERE id = 'f'")["status"] == "active"  # still starting
    assert db.one("SELECT status FROM stream_sessions WHERE id = 'm'")["status"] == "active"  # a handler owns it


def test_a_key_revoked_or_a_browser_signed_out_stops_its_running_session():
    from acw.stream import routes
    from acw.stream.features import Features

    uid = user()
    db.execute("INSERT INTO api_keys(id,user_id,name,prefix,key_hash,scopes,created_at) VALUES ('k1',?,'k','acw_xxxx','h','stream',?)", (uid, db.now()))
    db.execute("INSERT INTO sessions(id_hash,user_id,csrf_token,created_at,expires_at,last_seen_at) VALUES ('hh',?,'c',?,?,?)",
               (uid, db.now(), db.now() + 1000, db.now()))
    db.execute("INSERT INTO stream_sessions(id,user_id,status,started_at) VALUES ('sx',?,'active',?)", (uid, db.now()))
    db.execute("INSERT INTO stream_sessions(id,user_id,status,started_at) VALUES ('sy',?,'active',?)", (uid, db.now()))
    by_key = routes.Session(None, SimpleNamespace(user_id=uid, api_key_id="k1", session_hash=None), Features(denoise="light"))
    by_key.id = "sx"
    by_browser = routes.Session(None, SimpleNamespace(user_id=uid, api_key_id=None, session_hash="hh"), Features(denoise="light"))
    by_browser.id = "sy"
    assert asyncio.run(routes._persist(by_key)) is True and asyncio.run(routes._persist(by_browser)) is True
    db.execute("UPDATE api_keys SET revoked_at = ? WHERE id = 'k1'", (db.now(),))
    db.execute("DELETE FROM sessions WHERE id_hash = 'hh'")
    assert asyncio.run(routes._persist(by_key)) is False
    assert asyncio.run(routes._persist(by_browser)) is False


def test_a_client_that_vanishes_while_starting_gets_its_hold_back(monkeypatch):
    from acw.stream import routes

    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")

    class WS:
        async def receive_text(self):
            return json.dumps({"type": "start", "denoise": "light"})

        async def send_text(self, t):
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


def ws_denied(live_client, headers=None):
    from starlette.testclient import WebSocketDenialResponse

    try:
        with live_client.websocket_connect("/api/stream/ws", headers=headers or {}) as ws:
            return None, ws.receive_json()
    except WebSocketDenialResponse as e:
        return e.status_code, dict(e.headers)


def api_key(uid, scopes="stream"):
    from acw.security import new_api_key

    key, prefix, h = new_api_key()
    db.execute("INSERT INTO api_keys(id,user_id,name,prefix,key_hash,scopes,created_at) VALUES (?,?,?,?,?,?,?)",
               (db.new_id(), uid, "k", prefix, h, scopes, db.now()))
    return key


def test_the_live_handshake_says_why_it_refuses_and_when_to_come_back(client, live_client):
    from acw.config import get_settings
    from acw.security import limiter

    uid = user()
    assert ws_denied(live_client)[0] == 401  # no credentials
    assert ws_denied(live_client, {"Authorization": "Bearer acw_nonsense"})[0] == 401  # unknown key
    revoked = api_key(uid)
    db.execute("UPDATE api_keys SET revoked_at = ?", (db.now(),))
    assert ws_denied(live_client, {"Authorization": f"Bearer {revoked}"})[0] == 401
    assert ws_denied(live_client, {"Authorization": f"Bearer {api_key(uid, 'jobs')}"})[0] == 403  # wrong scope
    suspended = api_key(user("s@example.com"))
    db.execute("UPDATE users SET status = 'suspended' WHERE email = 's@example.com'")
    assert ws_denied(live_client, {"Authorization": f"Bearer {suspended}"})[0] == 403
    # the hour's starts are used up: 429 with the wait
    key = api_key(uid)
    limiter.hit(f"stream_start:{uid}", 60, 60 / 3600, cost=60)
    status, headers = ws_denied(live_client, {"Authorization": f"Bearer {key}"})
    assert status == 429 and int(headers["retry-after"]) >= 1
    limiter.reset()
    # draining for a restart: refused with a retry time
    (get_settings().data_dir / "drain").touch()
    status, event = ws_denied(live_client, {"Authorization": f"Bearer {api_key(uid)}"})
    assert status is None and event["code"] == "restarting" and event["retryAfterSeconds"] == 60


def test_a_dropped_session_is_replaced_instead_of_refusing_the_user():
    from acw.stream import routes
    from acw.stream.features import Features

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
    routes._open[uid] = [routes.Session(WS(), SimpleNamespace(user_id=uid), Features(denoise="light"))]  # a session that talks is never replaced
    try:
        assert asyncio.run(routes._evict_stale(uid)) is False
    finally:
        routes._open.clear()


# ---------------------------------------------------------------- premium transcription resumes and stays private

def test_premium_transcription_keeps_finished_segments_and_pays_only_for_the_missing_ones(tmp_path, monkeypatch):
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


def test_saved_chunks_and_texts_are_never_read_through_links(tmp_path, monkeypatch):
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


def test_premium_jobs_are_capped_at_a_few_at_once(monkeypatch):
    from acw import app as appmod

    uid = user()
    settle_payment(uid)
    ledger.credit(uid, 50_000_000, "grant", "t", "g")
    for _ in range(3):
        j = make_job(uid, kind="transcribe", denoise_tier=None, transcribe_tier="premium")
        db.execute("UPDATE jobs SET status = 'cloud_asr', stage = 'Waiting to transcribe', queued_at = ? WHERE id = ?", (db.now(), j))
    assert len(appmod._claim_cloud_jobs(appmod.PREMIUM_JOBS_AT_ONCE)) == 2
    assert len(appmod._claim_cloud_jobs(2)) == 1  # the third, once a slot is free
    assert appmod._claim_cloud_jobs(2) == []


def test_ledger_history_pages_and_labels_a_finished_job_in_plain_words():
    uid = user()
    settle_payment(uid)
    ledger.credit(uid, 50_000_000, "topup", "payment", "p0", "Top-up")
    jid = make_job(uid, kind="denoise_transcribe", transcribe_tier="premium", filename="noisy.flac", options={}, claimed_seconds=120)
    db.execute("UPDATE jobs SET status = 'running', duration_seconds = 120 WHERE id = ?", (jid,))
    jobs.update_params(jid, display_name="My talk.wav")
    job = dict(db.one("SELECT * FROM jobs WHERE id = ?", (jid,)))
    d = jobs.job_dir(jid)
    (d / "out.wav").write_bytes(b"x")
    from acw.worker import main as W

    W._complete(job, d, {"outputs": [{"file": "out.wav", "kind": "audio"}]})
    rows, more = ledger.history_page(uid, 100)
    usage = next(r for r in rows if r["kind"] == "usage")
    assert not more
    assert usage["label"] == "Denoising + Leveling + Subtitles, extra accurate"  # (named by the timeline's layers)
    assert (usage["product"], usage["minutes"], usage["displayName"]) == ("clean_transcript", 2.0, "My talk.wav")
    assert next(r for r in rows if r["kind"] == "topup")["label"] == "Top-up"
    for i in range(5):
        ledger.credit(uid, 1000, "grant", "t", f"g{i}")
    rows, more = ledger.history_page(uid, 3)
    assert len(rows) == 3 and more


def test_display_names_are_cleaned_of_control_and_direction_characters():
    c = jobs.clean_display_name
    assert c("a\u202eb\u200b.wav\x00") == "ab.wav"
    assert c("C:\\Users\\me\\talk.wav") == "talk.wav" and c("/tmp/../talk  one.wav") == "talk one.wav"
    assert len(c("x" * 500)) == 120 and c("  \u202e ") is None and c(None) is None


# ---------------------------------------------------------------- backups (OPS-01) and retention (DOC-02)

def test_backup_is_checked_after_writing_and_a_backup_whose_money_does_not_add_up_fails(tmp_path):
    from acw import backup

    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    out = backup.run(tmp_path / "b", keep=2)
    status = json.loads((tmp_path / "b" / "backup-status.json").read_text())
    assert out.name == status["file"] and status["integrity"] == "ok" and status["ledger"] == "ok" and status["bytes"] > 0
    assert backup.verify(out) == {"integrity": "ok", "ledger": "ok"}
    db.conn().execute("PRAGMA ignore_check_constraints=ON")
    db.execute("UPDATE users SET balance_micros = balance_micros + 7 WHERE id = ?", (uid,))
    with pytest.raises(backup.BackupError, match="failed its checks"):
        backup.run(tmp_path / "b")
    assert list((tmp_path / "b").glob("*.bad")) and not list((tmp_path / "b").glob("acw-*.sqlite3.gz"))
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
        assert backup.verify(bad)["integrity"] != "ok"
    except sqlite3.DatabaseError:
        pass  # too damaged to open at all: also a failure


def test_the_off_machine_copy_is_off_unless_configured_and_needs_a_key(tmp_path, monkeypatch):
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
def test_the_off_machine_copy_is_encrypted(tmp_path, monkeypatch):
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


def test_retention_removes_addresses_and_old_file_names_on_schedule():
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

def test_the_monitor_rejects_an_empty_or_unbalanced_backup_result(tmp_path):
    from acw import monitor

    assert not monitor.check_backup_result(tmp_path)[0].bad  # nothing written yet: staleness covers that
    (tmp_path / "backup-status.json").write_text(json.dumps({"file": "f", "bytes": 0, "integrity": "ok", "ledger": "ok"}))
    assert monitor.check_backup_result(tmp_path)[0].bad
    (tmp_path / "backup-status.json").write_text(json.dumps({"file": "f", "bytes": 10, "integrity": "ok", "ledger": "ok"}))
    assert not monitor.check_backup_result(tmp_path)[0].bad
    (tmp_path / "backup-status.json").write_text(json.dumps({"file": "f", "bytes": 10, "integrity": "ok", "ledger": "2 balances do not match"}))
    assert monitor.check_backup_result(tmp_path)[0].bad


def test_monitor_notices_stuck_states_and_a_money_mismatch():
    from acw import monitor

    uid = user()
    settle_payment(uid)
    ledger.credit(uid, 50_000_000, "grant", "t", "g")
    now = db.now()
    assert not any(f.bad for f in monitor.check_stuck_states(now))
    assert not monitor.check_money()[0].bad
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
    db.execute("UPDATE users SET balance_micros = balance_micros + 5 WHERE id = ?", (uid,))
    assert monitor.check_money()[0].bad


def test_deep_health_is_for_the_machine_itself_or_admins(client):
    from acw import health

    assert client.get("/api/health?deep=1").json() == {"ok": True}  # the test client is not loopback: no details
    assert not health.stream_worker_ok()
    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    job, _ = running_job(uid)
    db.execute("UPDATE jobs SET heartbeat_at = ? WHERE id = ?", (db.now() - 3600, job["id"]))
    assert health.stale_running_jobs() == 1
    deep = health.deep()
    assert deep["ok"] is False and deep["staleRunningJobs"] == 1 and "streamWorker" in deep and deep["money"]["held_mismatch"] == 0


# ---------------------------------------------------------------- accounts, sessions and cookies

def test_admin_rights_follow_the_environment_list(monkeypatch):
    from acw.config import get_settings

    monkeypatch.setenv("ACW_ADMIN_EMAILS", "boss@example.com")
    get_settings.cache_clear()
    assert auth.get_or_create_user("boss@example.com")["is_admin"] == 1
    monkeypatch.setenv("ACW_ADMIN_EMAILS", "")
    get_settings.cache_clear()
    assert auth.get_or_create_user("boss@example.com")["is_admin"] == 0


def test_browser_sessions_expire_when_unused(client):
    login(client, "idle@example.com")
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
