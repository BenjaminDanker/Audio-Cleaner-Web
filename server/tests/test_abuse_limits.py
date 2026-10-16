"""Fixes from the security / fraud audit: job reservations, cancel after the paid step, live duty cycle and message
rate, welcome-credit farming, first-top-up trust rules, premium captions that break down, malformed options,
and what an API key may learn from /api/me."""
import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from acw import auth, db, jobs
from acw.billing import ledger, pricing, stripe_service, trust
from acw.config import get_settings

from .conftest import ORIGIN, login, settle_payment
from .test_api_security import make_key

GB = 1024**3


def user(email="a@example.com", credit=50_000_000):
    uid = auth.get_or_create_user(email)["id"]
    if credit:
        ledger.credit(uid, credit, "grant", "t", "g")
    return uid


def make_job(uid, **kw):
    args = dict(kind="denoise", denoise_tier="standard", transcribe_tier=None, filename="a.wav", size_bytes=1000,
                claimed_seconds=600.0, options={}, free=False)
    return jobs.create_job(uid, **{**args, **kw})["id"]


@pytest.fixture
def roomy_disk(monkeypatch):
    from acw import hostload

    snap = SimpleNamespace(disk_free=500 * GB)
    monkeypatch.setattr(hostload, "get", lambda d: SimpleNamespace(snapshot=lambda: snap))


# ---------------------------------------------------------------- H1: job reservations

def test_parallel_creates_cannot_pass_the_active_job_cap():
    uid = user()

    def one(_):
        try:
            return make_job(uid)
        except jobs.JobError as e:
            return e.status

    with ThreadPoolExecutor(16) as pool:
        results = list(pool.map(one, range(16)))
    assert sum(isinstance(r, str) for r in results) == jobs.MAX_ACTIVE_JOBS_PER_USER
    assert db.one("SELECT COUNT(*) n FROM jobs")["n"] == jobs.MAX_ACTIVE_JOBS_PER_USER
    assert db.one("SELECT COUNT(*) n FROM holds WHERE status = 'active'")["n"] == jobs.MAX_ACTIVE_JOBS_PER_USER


def test_one_account_cannot_reserve_more_than_8_gb_of_uploads(roomy_disk):
    uid = user()
    make_job(uid, size_bytes=3 * GB, claimed_seconds=3600)
    make_job(uid, size_bytes=3 * GB, claimed_seconds=3600)
    with pytest.raises(jobs.JobError, match="large uploads") as e:
        make_job(uid, size_bytes=3 * GB, claimed_seconds=3600)  # 9 GB
    assert e.value.status == 429
    make_job(uid, size_bytes=1 * GB, claimed_seconds=3600)  # 7 GB: fine
    # a refused job leaves neither a row, a hold nor a folder behind
    assert db.one("SELECT COUNT(*) n FROM jobs")["n"] == 3
    assert db.one("SELECT COUNT(*) n FROM holds WHERE status = 'active'")["n"] == 3
    assert len(list(get_settings().jobs_dir.iterdir())) == 3


def test_a_size_that_does_not_fit_the_claimed_length_is_refused(roomy_disk):
    uid = user()
    with pytest.raises(jobs.JobError, match="larger than its length"):
        make_job(uid, size_bytes=4 * GB, claimed_seconds=0.5)  # 4 GiB in half a second
    with pytest.raises(jobs.JobError, match="larger than its length"):
        make_job(uid, size_bytes=1 * GB, claimed_seconds=20)  # 50 MB a second
    make_job(uid, size_bytes=150 * 1024**2, claimed_seconds=1)  # small files may be oddly dense
    make_job(uid, size_bytes=2 * GB, claimed_seconds=3600)  # an ordinary long video
    assert db.one("SELECT COUNT(*) n FROM jobs")["n"] == 2


def test_an_upload_that_sends_nothing_is_canceled_after_ten_minutes():
    uid = user()
    before = ledger.balance(uid)["held_micros"]
    jid = make_job(uid)
    row = db.one("SELECT created_at, expires_at FROM jobs WHERE id = ?", (jid,))
    assert row["expires_at"] - row["created_at"] == jobs.UPLOAD_FIRST_BYTE_SECONDS == 600
    assert ledger.balance(uid)["held_micros"] > before
    jobs.sweep()
    assert db.one("SELECT status FROM jobs WHERE id = ?", (jid,))["status"] == "uploading"  # still inside its ten minutes
    db.execute("UPDATE jobs SET expires_at = ? WHERE id = ?", (db.now() - 1, jid))
    jobs.sweep()
    assert db.one("SELECT status FROM jobs WHERE id = ?", (jid,))["status"] == "canceled"
    assert ledger.balance(uid)["held_micros"] == before  # the money set aside is free again


def test_the_first_bytes_of_an_upload_move_the_expiry_out(client, signed_in, monkeypatch):
    from acw.api import jobs_routes

    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    r = client.post("/api/jobs", json={"kind": "denoise", "denoiseTier": "standard", "filename": "a.wav", "sizeBytes": 3000,
                                       "durationSeconds": 5}, headers=h)
    assert r.status_code == 200, r.text
    calls = []
    real = jobs_routes._still_uploading
    monkeypatch.setattr(jobs_routes, "_still_uploading", lambda jid: calls.append(jid) or real(jid))
    assert client.put(r.json()["uploadUrl"].replace("api/", "/api/", 1), content=b"RIFF" + bytes(2996), headers=h).status_code == 200
    assert calls == [r.json()["id"]]  # called as soon as data arrived, not a minute later


# ---------------------------------------------------------------- H2, L7: cancel

def test_cancel_is_refused_once_the_paid_step_is_done(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    settle_payment(uid)
    jid = make_job(uid, kind="transcribe", denoise_tier=None, transcribe_tier="premium")
    db.execute("UPDATE jobs SET status = 'queued', next_stage = 'finalize' WHERE id = ?", (jid,))
    r = client.post(f"/api/jobs/{jid}/cancel", headers=h)
    assert r.status_code == 409 and "almost done" in r.json()["detail"]
    assert db.one("SELECT status FROM jobs WHERE id = ?", (jid,))["status"] == "queued"
    assert db.one("SELECT status FROM holds WHERE ref_id = ?", (jid,))["status"] == "active"
    db.execute("UPDATE jobs SET next_stage = 'probe' WHERE id = ?", (jid,))
    assert client.post(f"/api/jobs/{jid}/cancel", headers=h).status_code == 200  # before that, cancel still works


def test_cancel_says_so_when_it_did_not_cancel(client, signed_in, monkeypatch):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    jid = make_job(uid)
    db.execute("UPDATE jobs SET status = 'queued', next_stage = 'probe' WHERE id = ?", (jid,))
    monkeypatch.setattr(jobs, "finish_failed", lambda *a, **k: False)  # a worker took the job in between
    assert client.post(f"/api/jobs/{jid}/cancel", headers=h).status_code == 409
    # and for real: a job that is running now is not touched
    job = dict(db.one("SELECT * FROM jobs WHERE id = ?", (jid,)))
    db.execute("UPDATE jobs SET status = 'running' WHERE id = ?", (jid,))
    monkeypatch.undo()
    assert jobs.finish_failed(job, "Canceled.", status="canceled", only_from=("uploading", "queued")) is False
    assert db.one("SELECT status FROM jobs WHERE id = ?", (jid,))["status"] == "running"


# ---------------------------------------------------------------- H3: live duty cycle, M1: message rate

def frame(seq=0):
    from acw.stream import protocol as P

    return {"type": "websocket.receive", "bytes": P.HEADER.pack(P.VERSION, P.CODEC_S16, 0, seq, 0) + bytes(960)}


def ping():
    return {"type": "websocket.receive", "text": json.dumps({"type": "ping"})}


def drive(monkeypatch, messages, uid, **consts):
    """routes._run with a fake socket that delivers `messages` ((delay, message) pairs)."""
    from acw.stream import routes

    for k, v in consts.items():
        monkeypatch.setattr(routes, k, v)
    queue = list(messages)

    class FakeWS:
        def __init__(self):
            self.sent, self.closed = [], None

        async def receive(self):
            if not queue:
                await asyncio.sleep(3600)
            delay, msg = queue.pop(0)
            await asyncio.sleep(delay)
            return msg

        async def send_text(self, t):
            self.sent.append(json.loads(t))

        async def close(self, code=1000, reason=None):
            self.closed = self.closed or (code, reason)

    class R:
        async def readexactly(self, n):
            await asyncio.sleep(3600)

    async def never(r):
        await asyncio.sleep(3600)

    async def noop(*a, **k):
        return None

    monkeypatch.setattr(routes.P, "read_msg", never)
    monkeypatch.setattr(routes.P, "send_msg", noop)
    sess = routes.Session(FakeWS(), SimpleNamespace(user_id=uid), {"echo_reference": False}, 4000)
    sess.reader, sess.writer = R(), SimpleNamespace(close=lambda: None)
    sess.held_seconds = 300.0
    asyncio.run(routes._run(sess))
    return sess


def test_duty_cycle_counts_audio_against_wall_time(monkeypatch):
    from acw.stream import routes

    d = routes.DutyCycle(100.0)
    assert not d.starved(110.0, 0.0)  # inside the first 30 s nothing is judged
    assert not d.starved(129.0, 0.0)
    assert d.starved(131.0, 1.0)  # 1 s of audio in 31 s
    ok = routes.DutyCycle(0.0)
    audio = 0.0
    for t in range(1, 600):  # a normal session, then 10 s without data, then normal again
        audio += 0.0 if 300 <= t < 310 else 1.0
        assert not ok.starved(float(t), audio)
    squat = routes.DutyCycle(0.0)
    audio = 0.0
    hit = None
    for t in range(1, 600):  # 10 ms of audio every 10 s, an audio-less ping in between
        if t % 10 == 0:
            audio += 0.01
        if squat.starved(float(t), audio):
            hit = t
            break
    assert hit == 30
    resumed = routes.DutyCycle(0.0)  # a long healthy stretch does not excuse a squat that follows
    audio = 0.0
    for t in range(1, 3601):
        audio += 1.0
        assert not resumed.starved(float(t), audio)
    for t in range(3601, 3700):
        if resumed.starved(float(t), audio):
            break
    else:
        pytest.fail("a session that stopped sending was not ended")
    assert t < 3601 + routes.DUTY_WINDOW


def test_a_session_that_sends_too_little_audio_is_ended(monkeypatch):
    uid = user()
    msgs = [(0.05, ping()) for _ in range(40)]
    msgs[5] = (0.05, frame())  # one 10 ms frame, then pings only: not idle (audio came) but not live either
    sess = drive(monkeypatch, msgs, uid, DUTY_GRACE=0.5, DUTY_WINDOW=1.0, IDLE_TIMEOUT=60.0)
    assert sess.end_reason == "no_audio"
    assert {"type": "stopped", "reason": "no_audio"} in sess.ws.sent


def test_a_session_that_sends_every_frame_is_never_ended(monkeypatch):
    uid = user()
    msgs = [(0.01, frame(i)) for i in range(120)] + [(0.0, {"type": "websocket.disconnect"})]
    sess = drive(monkeypatch, msgs, uid, DUTY_GRACE=0.3, DUTY_WINDOW=0.6, IDLE_TIMEOUT=60.0)
    assert sess.end_reason == "client_closed"
    assert sess.seconds == pytest.approx(1.2)


def test_a_flood_of_messages_closes_the_connection(monkeypatch):
    uid = user()
    sess = drive(monkeypatch, [(0.0, ping())] * 400, uid)
    assert sess.end_reason == "rate_limited"
    assert sess.ws.closed == (1008, "too_many_messages")
    assert sum(1 for m in sess.ws.sent if m["type"] == "pong") <= 101  # the burst, then nothing
    assert sess.ws.sent[-1]["code"] == "rate_limited"


def test_ordinary_message_rates_are_fine(monkeypatch):
    uid = user()
    cfg = {"type": "config", "attenLimDb": 30}
    msgs = [(0.02, {"type": "websocket.receive", "text": json.dumps(cfg)}) for _ in range(5)]  # a slider moved a bit
    msgs += [(0.02, ping()) for _ in range(60)] + [(0.0, {"type": "websocket.disconnect"})]  # 50 a second
    sess = drive(monkeypatch, msgs, uid, IDLE_TIMEOUT=60.0, DUTY_GRACE=60.0)
    assert sess.end_reason == "client_closed" and sess.ws.closed is None


def test_settings_changes_have_a_tighter_limit(monkeypatch):
    uid = user()
    cfg = {"type": "websocket.receive", "text": json.dumps({"type": "config", "attenLimDb": 30})}
    sess = drive(monkeypatch, [(0.0, cfg)] * 12, uid, IDLE_TIMEOUT=60.0)
    assert sess.end_reason == "rate_limited" and sess.ws.closed[0] == 1008


# ---------------------------------------------------------------- H3/H4: one live session until the first payment

def test_live_sessions_at_once_depend_on_a_first_payment():
    uid = user()
    assert trust.max_live_sessions(uid) == 1
    settle_payment(uid)
    assert trust.max_live_sessions(uid) == 2


def test_an_unpaid_account_gets_one_live_session(client, signed_in):
    from acw.stream import routes

    uid, h = signed_in
    routes._live[uid] = 1
    try:
        with client.websocket_connect("/api/stream/ws", headers={"origin": ORIGIN}) as ws:
            ev = ws.receive_json()
        assert ev["code"] == "too_many_sessions" and "first top-up" in ev["message"]
        settle_payment(uid)
        with client.websocket_connect("/api/stream/ws", headers={"origin": ORIGIN}) as ws:
            ws.send_text(json.dumps({"type": "nonsense"}))  # past the session limit: the server now wants a start message
            with pytest.raises(Exception):
                ws.receive_json()
    finally:
        routes._live.pop(uid, None)


# ---------------------------------------------------------------- H4: welcome credit key, extra accurate

def _welcome_on(monkeypatch, secret="s" * 40):
    monkeypatch.delenv("ACW_WELCOME_CREDIT_MICROS", raising=False)
    if secret:
        monkeypatch.setenv("ACW_SESSION_SECRET", secret)
    else:
        monkeypatch.delenv("ACW_SESSION_SECRET", raising=False)
    get_settings.cache_clear()


def test_mailbox_aliases_get_one_welcome_credit(monkeypatch):
    _welcome_on(monkeypatch)
    aliases = ["john.doe@gmail.com", "johndoe@gmail.com", "john.doe+1@gmail.com", "j.o.h.n.d.o.e+x@gmail.com",
               "johndoe+y@googlemail.com", "JohnDoe@GMAIL.com", "other+a@corp.example", "other+b@corp.example", "other@corp.example"]
    got = [ledger.balance(auth.get_or_create_user(a)["id"])["balance_micros"] > 0 for a in aliases]
    assert got == [True] + [False] * 5 + [True, False, False]
    assert db.one("SELECT COUNT(*) n FROM welcome_claims")["n"] == 2


def test_the_welcome_claim_is_keyed_with_the_server_secret(monkeypatch):
    import hashlib

    _welcome_on(monkeypatch)
    auth.get_or_create_user("someone@example.com")
    key = db.one("SELECT email_hash FROM welcome_claims")["email_hash"]
    assert key != hashlib.sha256(b"someone@example.com").hexdigest()  # a copy of the database can't be checked by guessing addresses
    assert len(key) == 64
    # another secret gives another key: it can't be rebuilt without ours
    from acw.security import keyed_hash

    before = keyed_hash("welcome", "someone@example.com")
    assert before == key
    monkeypatch.setenv("ACW_SESSION_SECRET", "t" * 40)
    get_settings.cache_clear()
    assert keyed_hash("welcome", "someone@example.com") != key


def test_without_a_secret_the_claim_key_is_a_plain_hash_and_warns_once(monkeypatch, caplog):
    import hashlib

    from acw import security

    _welcome_on(monkeypatch, secret=None)
    monkeypatch.setattr(security, "_warned_unkeyed", False)
    with caplog.at_level(logging.WARNING, logger="acw.security"):
        auth.get_or_create_user("one@example.com")
        auth.get_or_create_user("two@example.com")
    assert db.one("SELECT COUNT(*) n FROM welcome_claims WHERE email_hash = ?", (hashlib.sha256(b"one@example.com").hexdigest(),))["n"] == 1
    assert len([r for r in caplog.records if "ACW_SESSION_SECRET" in r.getMessage()]) == 1


def test_canonical_mailbox():
    from acw.security import canonical_email

    assert canonical_email(" Some.One+tag@Googlemail.com ") == "someone@gmail.com"
    assert canonical_email("a.b+x@corp.example") == "a.b@corp.example"
    assert canonical_email("ＡＢ@example.com") == "ab@example.com"  # full-width letters (NFKC)


def test_extra_accurate_files_need_a_first_payment(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    body = {"kind": "transcribe", "transcribeTier": "premium", "filename": "a.wav", "sizeBytes": 1000, "durationSeconds": 60}
    r = client.post("/api/jobs", json=body, headers=h)
    assert r.status_code == 402 and r.json()["detail"].startswith("Extra accurate needs a top-up first")
    assert db.one("SELECT COUNT(*) n FROM jobs")["n"] == 0
    assert client.post("/api/jobs", json={**body, "transcribeTier": "local"}, headers=h).status_code == 200
    settle_payment(uid)
    assert client.post("/api/jobs", json=body, headers=h).status_code == 200


def test_a_refunded_payment_does_not_count_as_paid():
    uid = user()
    settle_payment(uid, cents=1000)
    assert trust.has_settled_payment(uid)
    db.execute("UPDATE payments SET refunded_cents = amount_cents WHERE user_id = ?", (uid,))
    assert not trust.has_settled_payment(uid)
    db.execute("UPDATE payments SET refunded_cents = 0, status = 'pending' WHERE user_id = ?", (uid,))
    assert not trust.has_settled_payment(uid)


def test_extra_accurate_live_captions_need_a_first_payment(client, signed_in, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    with client.websocket_connect("/api/stream/ws", headers={"origin": ORIGIN}) as ws:
        ws.send_text(json.dumps({"type": "start", "denoise": "light", "captions": "premium", "sampleRate": 48000}))
        ev = ws.receive_json()
    assert ev["code"] == "needs_topup" and ev["message"].startswith("Extra accurate needs a top-up first")
    assert db.one("SELECT COUNT(*) n FROM stream_sessions")["n"] == 0


def test_me_tells_the_pages_about_payment_state(client, signed_in):
    uid, h = signed_in
    me = client.get("/api/me").json()
    assert me["hasPaid"] is False and me["topupMaxCents"] == trust.NEW_ACCOUNT_TOPUP_MAX_CENTS
    settle_payment(uid)
    me = client.get("/api/me").json()
    assert me["hasPaid"] is True and me["topupMaxCents"] == pricing.TOPUP_MAX_CENTS


# ---------------------------------------------------------------- M4: first top-ups

def test_new_accounts_can_add_at_most_50_dollars_at_first(client, signed_in, monkeypatch):
    uid, h = signed_in
    made = []
    monkeypatch.setattr(stripe_service, "create_checkout", lambda u, c, s: made.append(c) or "https://checkout.example/x")
    r = client.post("/api/billing/checkout", json={"amountCents": 20_000}, headers=h)
    assert r.status_code == 400 and "$50" in r.json()["detail"] and "first top-up" in r.json()["detail"]
    assert client.post("/api/billing/checkout", json={"amountCents": 5_000}, headers=h).status_code == 200
    assert made == [5_000]
    # a first payment lifts it, and so does a 72-hour-old account
    settle_payment(uid)
    assert client.post("/api/billing/checkout", json={"amountCents": 20_000}, headers=h).status_code == 200


def test_old_accounts_have_the_usual_limit():
    uid = user()
    assert trust.topup_max_cents(uid) == 5_000
    db.execute("UPDATE users SET created_at = ? WHERE id = ?", (db.now() - 73 * 3600, uid))
    assert trust.topup_max_cents(uid) == pricing.TOPUP_MAX_CENTS


def test_the_checkout_function_checks_the_limit_too(monkeypatch):
    uid = user()
    monkeypatch.setattr(stripe_service, "ensure_customer", lambda _u: "cus_1")
    with pytest.raises(ValueError, match="up to \\$50"):
        stripe_service.create_checkout(uid, 10_000, False)


def test_auto_reload_amount_respects_the_new_account_limit(client, signed_in):
    uid, h = signed_in
    db.execute("UPDATE users SET default_payment_method = 'pm_1' WHERE id = ?", (uid,))
    body = {"enabled": True, "thresholdCents": 500, "amountCents": 10_000, "monthlyCapCents": 20_000, "consent": True}
    r = client.post("/api/billing/auto-reload", json=body, headers=h)
    assert r.status_code == 400 and "$50" in r.json()["detail"]
    assert client.post("/api/billing/auto-reload", json={**body, "amountCents": 5_000}, headers=h).status_code == 200
    # an old setting that is over the limit is not charged either
    db.execute("UPDATE users SET auto_reload_amount_cents = 10000, stripe_customer_id = 'cus_1' WHERE id = ?", (uid,))
    assert stripe_service.run_auto_reload(uid) == "cap"


# ---------------------------------------------------------------- M3: premium captions that break down

def premium_session(uid):
    from acw.stream import routes

    rate = routes._rate_micros_per_minute("light", "premium")
    sess = routes.Session(SimpleNamespace(), SimpleNamespace(user_id=uid), {"denoise": "light", "captions": "premium", "echo_reference": False}, rate)
    return sess, rate


def test_failed_premium_captions_stop_being_billed():
    uid = user()
    sess, rate = premium_session(uid)
    prem = pricing.rate("stream_captions", "premium").micros_per_minute
    den = pricing.rate("stream_denoise", "light").micros_per_minute
    assert rate == prem + den
    sess.samples = 48_000 * 60
    sess.premium_failed()  # the minute so far had them
    sess.samples += 48_000 * 60
    assert sess.micros_for(sess.seconds) == rate + den  # a minute with, a minute without
    assert sess.current_rate() == den
    sess.premium_failed()  # a second failure changes nothing
    assert sess.premium_off_seconds == 60


def test_the_hold_is_extended_at_the_reduced_rate(monkeypatch):
    uid = user()
    sess, rate = premium_session(uid)
    sess.open_hold()
    held = db.one("SELECT amount_micros FROM holds WHERE id = ?", (sess.hold_id,))["amount_micros"]
    assert held == rate * pricing.STREAM_HOLD_MINUTES
    sess.premium_failed()
    assert sess.extend_hold()
    den = pricing.rate("stream_denoise", "light").micros_per_minute
    assert db.one("SELECT amount_micros FROM holds WHERE id = ?", (sess.hold_id,))["amount_micros"] == held + den


def test_a_session_with_failed_captions_is_charged_only_for_what_it_got():
    from acw.stream import routes

    uid = user(credit=5_000_000)
    sess, rate = premium_session(uid)
    sess.id = "s-prem"
    db.execute("INSERT INTO stream_sessions(id,user_id,denoise_tier,caption_tier,status,started_at) VALUES ('s-prem',?,'light','premium','active',?)",
               (uid, db.now()))
    sess.open_hold()
    sess.writer = None
    sess.ws = SimpleNamespace(close=_noop)
    sess.samples = 48_000 * 120
    sess.premium_off_seconds = 60.0
    asyncio.run(routes._finish_inner(sess))
    den = pricing.rate("stream_denoise", "light").micros_per_minute
    assert db.one("SELECT cost_micros, premium_off_seconds FROM stream_sessions WHERE id = 's-prem'")["cost_micros"] == rate + den
    assert ledger.balance(uid)["balance_micros"] == 5_000_000 - rate - den
    assert ledger.balance(uid)["held_micros"] == 0


async def _noop(*a, **k):
    return None


def test_an_interrupted_session_is_billed_the_same_way_at_startup():
    from acw.stream import routes

    uid = user(credit=5_000_000)
    hold = ledger.create_hold(uid, 200_000, "stream", "s-old")
    db.execute(
        "INSERT INTO stream_sessions(id,user_id,denoise_tier,caption_tier,status,hold_id,seconds,premium_off_seconds,started_at)"
        " VALUES ('s-old',?,'light','premium','active',?,120,60,?)", (uid, hold, db.now() - 3600))
    asyncio.run(routes.startup())
    den = pricing.rate("stream_denoise", "light").micros_per_minute
    prem = pricing.rate("stream_captions", "premium").micros_per_minute
    assert db.one("SELECT cost_micros FROM stream_sessions WHERE id = 's-old'")["cost_micros"] == 2 * den + prem


def test_premium_captions_tell_the_session_when_they_fail():
    from acw.stream.openai_live import PremiumCaptions

    events, failed = [], []
    p = PremiumCaptions("en", events.append, on_fail=lambda: failed.append(1))
    p._fail()
    p._fail()
    assert failed == [1] and len(events) == 1 and events[0]["code"] == "captions_unavailable"
    assert "charged" in events[0]["message"] and "OpenAI" not in events[0]["message"]


def test_a_captions_only_session_ends_when_its_captions_break(monkeypatch):
    from acw.stream import routes

    uid = user()
    monkeypatch.setattr(routes, "IDLE_TIMEOUT", 60.0)
    q = [(0.05, frame(i)) for i in range(30)]

    class WS:
        sent = []

        async def receive(self):
            if not q:
                await asyncio.sleep(3600)
            d, m = q.pop(0)
            await asyncio.sleep(d)
            if len(q) == 20:
                sess.premium_failed()
            return m

        async def send_text(self, t):
            self.sent.append(json.loads(t))

        async def close(self, code=1000, reason=None):
            pass

    async def never(r):
        await asyncio.sleep(3600)

    monkeypatch.setattr(routes.P, "read_msg", never)
    monkeypatch.setattr(routes.P, "send_msg", _noop)
    rate = pricing.rate("stream_captions", "premium").micros_per_minute
    sess = routes.Session(WS(), SimpleNamespace(user_id=uid), {"denoise": None, "captions": "premium", "echo_reference": False}, rate)
    sess.reader, sess.writer, sess.held_seconds = SimpleNamespace(), SimpleNamespace(close=lambda: None), 300.0
    asyncio.run(routes._run(sess))
    assert sess.end_reason == "captions_unavailable"
    assert {"type": "stopped", "reason": "captions_unavailable"} in WS.sent
    assert sess.micros_for(sess.seconds) == routes._micros(0, sess.seconds) + routes._micros(rate, sess.premium_off_seconds)


# ---------------------------------------------------------------- L1: malformed options

@pytest.mark.parametrize("options", [
    {"audio_format": []},
    {"audio_format": {"a": 1}},
    {"caption_formats": [[1]]},
    {"caption_formats": [{"x": 1}]},
    {"caption_formats": "srt"},
    {"loudness_lufs": [1]},
    {"loudness_lufs": {"a": 1}},
    {"loudness_lufs": "loud"},
    {"atten_lim_db": [1]},
    {"language": ["en"]},
    {"noise": ["auto"]},
    {"noise": {"a": 1}},
    {"echo_reference_track": [True]},
])
def test_malformed_options_are_a_400_not_a_500(client, signed_in, options):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    r = client.post("/api/jobs", json={"kind": "denoise_transcribe", "denoiseTier": "standard", "transcribeTier": "local",
                                       "filename": "a.wav", "sizeBytes": 1000, "durationSeconds": 30, "options": options}, headers=h)
    assert r.status_code == 400, r.text
    assert db.one("SELECT COUNT(*) n FROM jobs")["n"] == 0


# ---------------------------------------------------------------- L6: /api/me for an API key

def test_an_api_key_does_not_learn_the_address_or_billing_settings(client):
    uid = user("owner@example.com")
    key = make_key(uid, "stream")
    me = client.get("/api/me", headers={"Authorization": f"Bearer {key}"}).json()
    assert me["signedIn"] is True and me["user"] == {"id": uid, "status": "active"}
    assert me["balance"]["available_micros"] == 50_000_000
    text = json.dumps(me)
    assert "owner@example.com" not in text and "autoReload" not in me and "csrfToken" not in me and "hasPaid" not in me
    # a browser session still gets all of it
    _, h = login(client, "owner@example.com")
    full = client.get("/api/me").json()
    assert full["user"]["email"] == "owner@example.com" and "autoReload" in full


def test_a_test_server_can_trust_every_account(monkeypatch):
    from acw import auth
    from acw.billing import trust
    from acw.config import get_settings

    uid = auth.get_or_create_user("trusted@example.com")["id"]
    assert not trust.has_settled_payment(uid)
    monkeypatch.setenv("ACW_TRUST_ALL_ACCOUNTS", "1")
    get_settings.cache_clear()
    assert trust.has_settled_payment(uid)
    get_settings.cache_clear()
