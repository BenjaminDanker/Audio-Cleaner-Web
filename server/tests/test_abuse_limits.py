"""Abuse and money limits: job reservations and uploads, cancelling after the paid step, live duty cycle and message
rate, free-trial farming, first top-ups, premium captions that break down, malformed options, and what an API key may
learn from /api/me."""
import asyncio
import json
import math
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from acw import auth, db, jobs
from acw.billing import ledger, pricing, stripe_service, trust
from acw.config import get_settings
from acw.stream.features import Features

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
                claimed_seconds=600.0, options={})
    return jobs.create_job(uid, **{**args, **kw})["id"]


@pytest.fixture
def roomy_disk(monkeypatch):
    from acw import hostload

    snap = SimpleNamespace(disk_free=500 * GB)
    monkeypatch.setattr(hostload, "get", lambda d: SimpleNamespace(snapshot=lambda: snap))


# ---------------------------------------------------------------- job reservations and uploads

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
    assert db.one("SELECT COUNT(*) n FROM holds WHERE status = 'active'")["n"] == jobs.MAX_ACTIVE_JOBS_PER_USER


def test_one_account_cannot_reserve_more_than_8_gb_of_uploads_and_a_size_must_fit_its_length(roomy_disk):
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
    with pytest.raises(jobs.JobError, match="larger than its length"):  # 4 GiB claimed as half a second: a cheap job
        make_job(uid, size_bytes=4 * GB, claimed_seconds=0.5)


def test_an_upload_that_sends_nothing_is_canceled_after_ten_minutes_and_the_money_comes_back():
    uid = user()
    before = ledger.balance(uid)["held_micros"]
    jid = make_job(uid)
    row = db.one("SELECT created_at, expires_at FROM jobs WHERE id = ?", (jid,))
    assert row["expires_at"] - row["created_at"] == jobs.UPLOAD_FIRST_BYTE_SECONDS == 600
    jobs.sweep()
    assert db.one("SELECT status FROM jobs WHERE id = ?", (jid,))["status"] == "uploading"  # still inside its ten minutes
    db.execute("UPDATE jobs SET expires_at = ? WHERE id = ?", (db.now() - 1, jid))
    jobs.sweep()
    assert db.one("SELECT status FROM jobs WHERE id = ?", (jid,))["status"] == "canceled"
    assert ledger.balance(uid)["held_micros"] == before  # the money set aside is free again


def test_the_first_bytes_of_an_upload_move_the_expiry_out_and_a_cancel_stops_the_upload(client, signed_in, monkeypatch):
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
    jid = make_job(uid)  # a second upload, still waiting for its bytes
    db.execute("UPDATE jobs SET expires_at = ? WHERE id = ?", (db.now() + 5, jid))
    assert jobs_routes._still_uploading(jid)
    assert db.one("SELECT expires_at FROM jobs WHERE id = ?", (jid,))["expires_at"] >= db.now() + jobs.UPLOAD_WINDOW_SECONDS - 2
    jobs.finish_failed(dict(db.one("SELECT * FROM jobs WHERE id = ?", (jid,))), "x", status="canceled")
    assert not jobs_routes._still_uploading(jid)  # canceled meanwhile: the upload stops


def test_cancel_is_refused_once_the_paid_step_is_done(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    settle_payment(uid)
    jid = make_job(uid, kind="transcribe", denoise_tier=None, transcribe_tier="premium")
    db.execute("UPDATE jobs SET status = 'queued', next_stage = 'finalize' WHERE id = ?", (jid,))
    r = client.post(f"/api/jobs/{jid}/cancel", headers=h)
    assert r.status_code == 409 and "almost done" in r.json()["detail"]
    assert db.one("SELECT status FROM holds WHERE ref_id = ?", (jid,))["status"] == "active"
    db.execute("UPDATE jobs SET next_stage = 'probe' WHERE id = ?", (jid,))
    assert client.post(f"/api/jobs/{jid}/cancel", headers=h).status_code == 200  # before that, cancel still works


# ---------------------------------------------------------------- live sessions: duty cycle, message rate, count

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
    sess = routes.Session(FakeWS(), SimpleNamespace(user_id=uid), Features(denoise="light"))
    sess.reader, sess.writer = R(), SimpleNamespace(close=lambda: None)
    sess.held_micros = sess.current_rate() * 5
    asyncio.run(routes._run(sess))
    return sess


def test_the_duty_cycle_counts_audio_against_wall_time(monkeypatch):
    from acw.stream import routes

    d = routes.DutyCycle(100.0)
    assert not d.starved(110.0, 0.0)  # inside the first 30 s nothing is judged
    assert d.starved(131.0, 1.0)  # 1 s of audio in 31 s
    ok = routes.DutyCycle(0.0)
    audio = 0.0
    for t in range(1, 600):  # a normal session, then 10 s without data, then normal again
        audio += 0.0 if 300 <= t < 310 else 1.0
        assert not ok.starved(float(t), audio)
    squat = routes.DutyCycle(0.0)
    audio, hit = 0.0, None
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
    msgs = [(0.02, ping()) for _ in range(40)]
    msgs[5] = (0.02, frame())  # one 10 ms frame, then pings only: not idle (audio came) but not live either
    sess = drive(monkeypatch, msgs, uid, DUTY_GRACE=0.2, DUTY_WINDOW=0.4, IDLE_TIMEOUT=60.0)
    assert sess.end_reason == "no_audio"
    assert {"type": "stopped", "reason": "no_audio"} in sess.ws.sent


def test_a_session_that_sends_every_frame_is_never_ended_and_is_billed_for_its_audio(monkeypatch):
    uid = user()
    msgs = [(0.01, frame(i)) for i in range(80)] + [(0.0, {"type": "websocket.disconnect"})]
    sess = drive(monkeypatch, msgs, uid, DUTY_GRACE=0.2, DUTY_WINDOW=0.4, IDLE_TIMEOUT=60.0)
    assert sess.end_reason == "client_closed" and sess.ws.closed is None
    assert sess.seconds == pytest.approx(0.8)


def test_a_flood_of_messages_closes_the_connection_and_settings_changes_have_a_tighter_limit(monkeypatch):
    uid = user()
    sess = drive(monkeypatch, [(0.0, ping())] * 400, uid)
    assert sess.end_reason == "rate_limited"
    assert sess.ws.closed == (1008, "too_many_messages")
    assert sum(1 for m in sess.ws.sent if m["type"] == "pong") <= 101  # the burst, then nothing
    assert sess.ws.sent[-1]["code"] == "rate_limited"
    cfg = {"type": "websocket.receive", "text": json.dumps({"type": "config", "attenLimDb": 30})}
    sess = drive(monkeypatch, [(0.0, cfg)] * 12, uid, IDLE_TIMEOUT=60.0)
    assert sess.end_reason == "rate_limited" and sess.ws.closed[0] == 1008


def test_four_live_sessions_at_most_per_account(client, signed_in):
    from acw.stream import routes

    uid, h = signed_in
    routes._live[uid] = 4
    try:
        with client.websocket_connect("/api/stream/ws", headers={"origin": ORIGIN}) as ws:
            ev = ws.receive_json()
        assert ev["code"] == "too_many_sessions" and "Four live sessions" in ev["message"]
        routes._live[uid] = 3  # one more is fine
        with client.websocket_connect("/api/stream/ws", headers={"origin": ORIGIN}) as ws:
            ws.send_text(json.dumps({"type": "nonsense"}))  # past the session limit: the server now wants a start message
            with pytest.raises(Exception):
                ws.receive_json()
    finally:
        routes._live.pop(uid, None)


# ---------------------------------------------------------------- free-trial farming and first top-ups

def _trial_on(monkeypatch, secret="s" * 40):
    monkeypatch.delenv("ACW_LIVE_TRIAL_SECONDS", raising=False)
    monkeypatch.setenv("ACW_SESSION_SECRET", secret)
    get_settings.cache_clear()


def test_mailbox_aliases_get_one_free_trial_and_the_claim_is_keyed_with_the_server_secret(monkeypatch):
    import hashlib

    from acw.security import keyed_hash

    _trial_on(monkeypatch)
    aliases = ["john.doe@gmail.com", "johndoe@gmail.com", "john.doe+1@gmail.com", "j.o.h.n.d.o.e+x@gmail.com",
               "johndoe+y@googlemail.com", "JohnDoe@GMAIL.com", "other+a@corp.example", "other+b@corp.example", "other@corp.example"]
    got = [auth.get_or_create_user(a)["trial_live_seconds"] > 0 for a in aliases]
    assert got == [True] + [False] * 5 + [True, False, False]
    assert db.one("SELECT COUNT(*) n FROM welcome_claims")["n"] == 2
    key = db.one("SELECT email_hash FROM welcome_claims WHERE email_hash = ?", (keyed_hash("welcome", "johndoe@gmail.com"),))["email_hash"]
    assert key != hashlib.sha256(b"john.doe@gmail.com").hexdigest()  # a copy of the database can't be checked by guessing addresses
    monkeypatch.setenv("ACW_SESSION_SECRET", "t" * 40)  # another secret gives another key
    get_settings.cache_clear()
    assert keyed_hash("welcome", "johndoe@gmail.com") != key


def test_canonical_mailbox():
    from acw.security import canonical_email

    assert canonical_email(" Some.One+tag@Googlemail.com ") == "someone@gmail.com"
    assert canonical_email("a.b+x@corp.example") == "a.b@corp.example"
    assert canonical_email("ＡＢ@example.com") == "ab@example.com"  # full-width letters (NFKC)


def test_a_refunded_payment_does_not_count_as_paid():
    uid = user()
    settle_payment(uid, cents=1000)
    assert trust.has_settled_payment(uid)
    db.execute("UPDATE payments SET refunded_cents = amount_cents WHERE user_id = ?", (uid,))
    assert not trust.has_settled_payment(uid)


def test_a_first_top_up_is_limited_to_50_dollars_until_a_top_up_has_gone_through(client, signed_in, monkeypatch):
    uid, h = signed_in
    made = []
    monkeypatch.setattr(stripe_service, "create_checkout", lambda u, c, s: made.append(c) or "https://checkout.example/x")
    assert trust.topup_max_cents(uid) == trust.FIRST_TOPUP_MAX_CENTS == 5_000
    db.execute("UPDATE users SET created_at = ? WHERE id = ?", (db.now() - 400 * 86400, uid))  # the age of the account does not matter
    assert trust.topup_max_cents(uid) == 5_000
    r = client.post("/api/billing/checkout", json={"amountCents": 20_000}, headers=h)
    assert r.status_code == 400 and "first top-up can be up to $50" in r.json()["detail"] and "$500" in r.json()["detail"]
    assert client.post("/api/billing/checkout", json={"amountCents": 5_000}, headers=h).status_code == 200
    assert made == [5_000]
    settle_payment(uid)  # a top-up has gone through
    assert trust.topup_max_cents(uid) == pricing.TOPUP_MAX_CENTS == 50_000
    assert client.post("/api/billing/checkout", json={"amountCents": 50_000}, headers=h).status_code == 200


def test_the_checkout_function_checks_the_first_top_up_limit_too(monkeypatch):
    uid = user()
    monkeypatch.setattr(stripe_service, "ensure_customer", lambda _u: "cus_1")
    with pytest.raises(ValueError, match="up to \\$50"):
        stripe_service.create_checkout(uid, 10_000, False)


def test_auto_reload_amount_respects_the_first_top_up_limit(client, signed_in):
    uid, h = signed_in
    db.execute("UPDATE users SET default_payment_method = 'pm_1' WHERE id = ?", (uid,))
    body = {"enabled": True, "thresholdCents": 500, "amountCents": 10_000, "monthlyCapCents": 20_000, "consent": True}
    r = client.post("/api/billing/auto-reload", json=body, headers=h)
    assert r.status_code == 400 and "$50" in r.json()["detail"]
    assert client.post("/api/billing/auto-reload", json={**body, "amountCents": 5_000}, headers=h).status_code == 200
    db.execute("UPDATE users SET auto_reload_amount_cents = 10000, stripe_customer_id = 'cus_1' WHERE id = ?", (uid,))
    assert stripe_service.run_auto_reload(uid) == "cap"  # an old setting over the limit is not charged either


# ---------------------------------------------------------------- premium captions that break down

def premium_session(uid):
    from acw.stream import routes

    f = Features(denoise="light", captions="premium")
    return routes.Session(SimpleNamespace(), SimpleNamespace(user_id=uid), f), f.rate()


def test_failed_premium_captions_stop_being_billed_and_the_hold_follows_at_the_reduced_rate():
    uid = user()
    sess, rate = premium_session(uid)
    prem = pricing.rate("stream_captions", "premium").micros_per_minute
    den = pricing.rate("stream_denoise", "light").micros_per_minute
    assert rate == prem + den
    sess.open_hold()
    held = db.one("SELECT amount_micros FROM holds WHERE id = ?", (sess.hold_id,))["amount_micros"]
    assert held == rate * pricing.STREAM_HOLD_MINUTES
    sess.count(48_000 * 60)
    sess.failed_premium()  # the minute so far had them
    assert sess.extend_hold()
    assert db.one("SELECT amount_micros FROM holds WHERE id = ?", (sess.hold_id,))["amount_micros"] == held + den
    sess.count(48_000 * 60)
    assert sess.meter.micros() == rate + den and sess.current_rate() == den  # a minute with, a minute without
    sess.failed_premium()  # a second failure changes nothing
    assert sess.meter.micros() == rate + den and sess.current_rate() == den


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
    sess.count(48_000 * 60)
    sess.failed_premium()
    sess.count(48_000 * 60)
    asyncio.run(routes._finish_inner(sess))
    den = pricing.rate("stream_denoise", "light").micros_per_minute
    assert db.one("SELECT cost_micros FROM stream_sessions WHERE id = 's-prem'")["cost_micros"] == rate + den
    assert ledger.balance(uid)["balance_micros"] == 5_000_000 - rate - den
    assert ledger.balance(uid)["held_micros"] == 0


async def _noop(*a, **k):
    return None


def test_an_interrupted_session_is_billed_the_same_way_at_startup_and_a_session_from_the_old_version_too():
    from acw.stream import routes

    uid = user(credit=5_000_000)
    den = pricing.rate("stream_denoise", "light").micros_per_minute
    prem = pricing.rate("stream_captions", "premium").micros_per_minute
    hold = ledger.create_hold(uid, 200_000, "stream", "s-old")
    db.execute(  # a minute with the captions, then they broke down: the meter as the session saved it
        "INSERT INTO stream_sessions(id,user_id,denoise_tier,caption_tier,status,hold_id,seconds,paid_seconds,meter_closed,"
        "meter_rate,meter_seconds,started_at) VALUES ('s-old',?,'light','premium','active',?,120,120,?,?,60,?)",
        (uid, hold, den + prem, den, db.now() - 3600))
    older_hold = ledger.create_hold(uid, 200_000, "stream", "s-older")
    db.execute("ALTER TABLE stream_sessions ADD COLUMN premium_off_seconds REAL")  # a session left by the version before the meter
    db.execute(
        "INSERT INTO stream_sessions(id,user_id,denoise_tier,caption_tier,status,hold_id,seconds,premium_off_seconds,started_at)"
        " VALUES ('s-older',?,'light','premium','active',?,120,60,?)", (uid, older_hold, db.now() - 3600))
    db.init_db()  # the new version starts: the meter is made from the old columns, which then go
    assert "premium_off_seconds" not in {r["name"] for r in db.all_rows("PRAGMA table_info(stream_sessions)")}
    asyncio.run(routes.startup())
    assert db.one("SELECT cost_micros FROM stream_sessions WHERE id = 's-old'")["cost_micros"] == 2 * den + prem
    assert db.one("SELECT cost_micros FROM stream_sessions WHERE id = 's-older'")["cost_micros"] == 2 * den + prem


def test_premium_captions_tell_the_session_once_when_they_fail():
    from acw.stream.openai_live import PremiumCaptions

    events, failed = [], []
    p = PremiumCaptions("en", events.append, on_fail=lambda: failed.append(1))
    p._fail()
    p._fail()
    assert failed == [1] and len(events) == 1 and events[0]["code"] == "captions_unavailable"


def test_a_captions_only_session_ends_when_its_captions_break(monkeypatch):
    from acw.stream import routes

    uid = user()
    monkeypatch.setattr(routes, "IDLE_TIMEOUT", 60.0)
    q = [(0.02, frame(i)) for i in range(30)]
    failed_at = []

    class WS:
        sent = []

        async def receive(self):
            if not q:
                await asyncio.sleep(3600)
            d, m = q.pop(0)
            await asyncio.sleep(d)
            if len(q) == 20:
                failed_at.append(sess.seconds)
                sess.failed_premium()
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
    sess = routes.Session(WS(), SimpleNamespace(user_id=uid), Features(captions="premium"))
    sess.reader, sess.writer = SimpleNamespace(), SimpleNamespace(close=lambda: None)
    sess.held_micros = sess.current_rate() * 5
    asyncio.run(routes._run(sess))
    assert sess.end_reason == "captions_unavailable"
    assert {"type": "stopped", "reason": "captions_unavailable"} in WS.sent
    assert sess.current_rate() == 0 and sess.meter.micros() == math.ceil(round(rate * failed_at[0] / 60, 6))


# ---------------------------------------------------------------- malformed options are a 400, not a 500

@pytest.mark.parametrize("options", [
    {"audio_format": []},
    {"caption_formats": "srt"},
    {"loudness_lufs": {"a": 1}},
    {"language": ["en"]},
    {"echo_reference_track": [True]},
])
def test_malformed_options_are_a_400_not_a_500(client, signed_in, options):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    r = client.post("/api/jobs", json={"kind": "denoise_transcribe", "denoiseTier": "standard", "transcribeTier": "local",
                                       "filename": "a.wav", "sizeBytes": 1000, "durationSeconds": 30, "options": options}, headers=h)
    assert r.status_code == 400, r.text
    assert db.one("SELECT COUNT(*) n FROM jobs")["n"] == 0


# ---------------------------------------------------------------- /api/me for an API key

def test_an_api_key_does_not_learn_the_address_or_billing_settings(client):
    uid = user("owner@example.com")
    key = make_key(uid, "stream")
    me = client.get("/api/me", headers={"Authorization": f"Bearer {key}"}).json()
    assert me["signedIn"] is True and me["user"] == {"id": uid, "status": "active"}
    assert me["balance"]["available_micros"] == 50_000_000
    text = json.dumps(me)
    assert "owner@example.com" not in text and "autoReload" not in me and "csrfToken" not in me
    _, h = login(client, "owner@example.com")  # a browser session still gets all of it
    full = client.get("/api/me").json()
    assert full["user"]["email"] == "owner@example.com" and "autoReload" in full
