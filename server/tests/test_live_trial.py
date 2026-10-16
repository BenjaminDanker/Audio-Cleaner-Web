"""The free live trial: every new account gets two hours of Standard live once per mailbox, ever; a trial session is
not charged, moves on to paid billing when the trial runs out (or stops), and is admitted only while live is not busy."""
import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from starlette.websockets import WebSocketDisconnect

from acw import auth, db
from acw.billing import ledger, pricing
from acw.config import get_settings
from acw.stream import protocol as P
from acw.stream import routes, worker

from .conftest import ORIGIN, login


def _trial_on(monkeypatch, value=None):
    if value is None:
        monkeypatch.delenv("ACW_LIVE_TRIAL_SECONDS", raising=False)
    else:
        monkeypatch.setenv("ACW_LIVE_TRIAL_SECONDS", str(value))
    get_settings.cache_clear()


def trial_left(uid):
    return db.one("SELECT trial_live_seconds t FROM users WHERE id = ?", (uid,))["t"]


# ---------------------------------------------------------------- the grant

def test_a_new_account_gets_two_hours_once(monkeypatch):
    _trial_on(monkeypatch)
    user = auth.get_or_create_user("new@example.com")
    assert user["trial_live_seconds"] == 7200
    assert ledger.balance(user["id"])["balance_micros"] == 0  # time, not money
    db.execute("UPDATE users SET trial_live_seconds = 100 WHERE id = ?", (user["id"],))
    auth.get_or_create_user("NEW@example.com")  # signing in again: nothing more
    assert trial_left(user["id"]) == 100
    assert db.one("SELECT COUNT(*) n FROM welcome_claims WHERE email_hash LIKE '%new%'")["n"] == 0  # only a hash


def test_aliases_of_one_mailbox_share_one_trial(monkeypatch):
    _trial_on(monkeypatch)
    got = [auth.get_or_create_user(a)["trial_live_seconds"] for a in
           ("some.one@gmail.com", "someone+x@gmail.com", "someone@googlemail.com", "other@example.com")]
    assert got == [7200, 0, 0, 7200]


def test_deleting_the_account_and_coming_back_does_not_give_it_twice(client, monkeypatch):
    _trial_on(monkeypatch)
    uid, h = login(client, "again@example.com")
    assert trial_left(uid) == 7200
    assert client.post("/api/account/delete", json={"confirm": "again@example.com"}, headers=h).status_code == 200
    uid2, _ = login(client, "again@example.com")
    assert uid2 != uid and trial_left(uid2) == 0


def test_the_length_can_be_set_or_switched_off(monkeypatch):
    _trial_on(monkeypatch, 600)
    assert auth.get_or_create_user("a@example.com")["trial_live_seconds"] == 600
    _trial_on(monkeypatch, 0)
    assert auth.get_or_create_user("b@example.com")["trial_live_seconds"] == 0
    assert db.one("SELECT COUNT(*) n FROM welcome_claims")["n"] == 1  # off: nobody is marked as having had it


def test_accounts_without_a_claim_get_nothing(monkeypatch):
    uid = auth.get_or_create_user("old@example.com")["id"]  # created while the trial was off
    _trial_on(monkeypatch)
    auth.get_or_create_user("old@example.com")
    assert trial_left(uid) == 0


def test_me_shows_what_is_left(client, monkeypatch):
    _trial_on(monkeypatch)
    uid, _ = login(client, "me@example.com")
    assert client.get("/api/me").json()["trialLiveSecondsLeft"] == 7200
    db.execute("UPDATE users SET trial_live_seconds = 1234.6 WHERE id = ?", (uid,))
    assert client.get("/api/me").json()["trialLiveSecondsLeft"] == 1234


# ---------------------------------------------------------------- sessions (a fake live worker behind the real endpoint)

@pytest.fixture
def fake_worker(monkeypatch):
    """Replaces the connection to the live worker: it answers `reply` to a start and ends when told to."""
    state = SimpleNamespace(reply={"type": "ready", "latencySamples": 0}, started=[])

    async def open_conn(_path):
        ended = asyncio.Event()
        return SimpleNamespace(first=True, ended=ended), SimpleNamespace(ended=ended, close=lambda: None)

    async def send(writer, kind, payload):
        if kind == P.MSG_START:
            state.started.append(json.loads(payload))
        elif kind == P.MSG_END:
            writer.ended.set()

    async def read(reader):
        if reader.first:
            reader.first = False
            return P.MSG_EVENT, P.pack_json(state.reply)
        await reader.ended.wait()
        return P.MSG_END, b""

    monkeypatch.setattr(routes.asyncio, "open_unix_connection", open_conn)
    monkeypatch.setattr(routes.P, "send_msg", send)
    monkeypatch.setattr(routes.P, "read_msg", read)
    monkeypatch.setattr(routes, "PERSIST_EVERY", 0)  # save the progress with every frame
    return state


def frame_bytes(seq):
    return P.HEADER.pack(P.VERSION, P.CODEC_S16, 0, seq, 0) + bytes(960)  # 10 ms


def run_session(client, seconds, start=None, stop=True):
    """Start a session, send `seconds` of audio, and return every message it got back until it was closed."""
    start = {"type": "start", "denoise": "light", "sampleRate": 48000, **(start or {})}
    got = []
    with client.websocket_connect("/api/stream/ws", headers={"origin": ORIGIN}) as ws:
        ws.send_text(json.dumps(start))
        got.append(ws.receive_json())
        if got[0]["type"] != "ready":
            return got
        for i in range(int(seconds * 100)):
            ws.send_bytes(frame_bytes(i))
        if not stop:
            return got  # the client just vanishes
        ws.send_text(json.dumps({"type": "stop"}))
        try:
            while True:
                got.append(ws.receive_json())
        except WebSocketDisconnect:
            pass
    return got


def the_session(wait=True):
    """The one session row, once its handler has finished (it settles after the client has gone)."""
    for _ in range(50 if wait else 1):
        row = db.one("SELECT * FROM stream_sessions")
        if row is None or row["status"] != "active":
            break
        time.sleep(0.1)
    return dict(row)


def test_a_trial_session_uses_trial_seconds_and_costs_nothing(client, signed_in, fake_worker):
    uid, _ = signed_in
    db.execute("UPDATE users SET trial_live_seconds = 600 WHERE id = ?", (uid,))
    got = run_session(client, 3)
    assert got[0]["type"] == "ready" and got[0]["trial"] is True and got[0]["trialSecondsLeft"] == 600
    assert not [m for m in got if m.get("type") in ("warning", "stopped", "error")]
    s = the_session()
    assert s["status"] == "ended" and s["seconds"] == pytest.approx(3) and s["trial_seconds"] == 3
    assert s["cost_micros"] == 0 and s["hold_id"] is None
    assert trial_left(uid) == 597
    assert db.one("SELECT COUNT(*) n FROM holds")["n"] == 0
    assert ledger.balance(uid) == {"balance_micros": 0, "held_micros": 0, "available_micros": 0}
    assert fake_worker.started[0]["trial"] is True


def test_standard_captions_are_covered_too(client, signed_in, fake_worker):
    uid, _ = signed_in
    db.execute("UPDATE users SET trial_live_seconds = 600 WHERE id = ?", (uid,))
    run_session(client, 1, {"denoise": None, "captions": "local", "language": "en"})
    assert the_session()["trial_seconds"] == 1 and trial_left(uid) == 599


def test_trial_seconds_are_taken_off_while_the_session_runs(client, signed_in, fake_worker):
    uid, _ = signed_in
    db.execute("UPDATE users SET trial_live_seconds = 600 WHERE id = ?", (uid,))
    run_session(client, 2, stop=False)  # the client just vanishes
    s = the_session(wait=False)  # a vanished client: the row stays "active" for the restart cleanup
    assert 0 < s["trial_seconds"] <= 2 and trial_left(uid) == pytest.approx(600 - s["trial_seconds"])  # saved with the frames


def test_the_trial_is_used_before_credit_and_credit_stays_untouched(client, signed_in, fake_worker):
    uid, _ = signed_in
    db.execute("UPDATE users SET trial_live_seconds = 600 WHERE id = ?", (uid,))
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    run_session(client, 2)
    assert ledger.balance(uid)["balance_micros"] == 5_000_000


def test_trial_low_warns_at_five_minutes_left(client, signed_in, fake_worker):
    uid, _ = signed_in
    db.execute("UPDATE users SET trial_live_seconds = 299.5 WHERE id = ?", (uid,))
    got = run_session(client, 2)
    warnings = [m for m in got if m.get("type") == "warning"]
    assert len(warnings) == 1 and warnings[0]["code"] == "trial_low" and 296 <= warnings[0]["secondsLeft"] <= 299
    # with plenty left there is no warning
    db.execute("UPDATE users SET trial_live_seconds = 3600 WHERE id = ?", (uid,))
    assert not [m for m in run_session(client, 1) if m.get("type") == "warning"]


def test_when_the_trial_ends_the_session_moves_on_to_credit(client, signed_in, fake_worker):
    uid, _ = signed_in
    db.execute("UPDATE users SET trial_live_seconds = 1 WHERE id = ?", (uid,))
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    got = run_session(client, 2.5)
    types = [(m["type"], m.get("code") or m.get("reason")) for m in got[1:]]
    assert ("warning", "trial_used_up_now_paid") in types and ("stopped", "trial_over") not in types
    assert all(t[0] != "error" for t in types)
    s = the_session()
    rate = routes._rate_micros_per_minute("light", None)
    assert s["seconds"] == pytest.approx(2.5) and s["trial_seconds"] == 1 and s["end_reason"] == "client_stop"
    assert s["cost_micros"] == routes._micros(rate, 1.5)  # only what came after the trial
    assert ledger.balance(uid)["balance_micros"] == 5_000_000 - s["cost_micros"]
    assert ledger.balance(uid)["held_micros"] == 0
    assert trial_left(uid) == 0


def test_when_the_trial_ends_without_credit_the_session_stops(client, signed_in, fake_worker):
    uid, _ = signed_in
    db.execute("UPDATE users SET trial_live_seconds = 1 WHERE id = ?", (uid,))
    got = run_session(client, 3)
    assert {"type": "stopped", "reason": "trial_over"} in got
    s = the_session()
    assert s["end_reason"] == "trial_over" and s["cost_micros"] == 0 and s["trial_seconds"] == 1
    assert s["seconds"] < 1.1  # no audio was processed past the trial
    assert trial_left(uid) == 0 and db.one("SELECT COUNT(*) n FROM holds")["n"] == 0


def test_a_new_session_after_the_trial_is_billed_normally(client, signed_in, fake_worker):
    uid, _ = signed_in
    got = run_session(client, 1)  # no trial seconds, no credit
    assert got[0]["code"] == "insufficient_funds"
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    run_session(client, 2)
    s = the_session()
    assert s["trial_seconds"] == 0 and s["cost_micros"] == routes._micros(routes._rate_micros_per_minute("light", None), 2)


def test_sessions_that_are_not_standard_are_billed_from_credit(client, signed_in, fake_worker, monkeypatch):
    uid, _ = signed_in
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")  # translations need the speech partner configured
    from acw.config import get_settings

    get_settings.cache_clear()
    db.execute("UPDATE users SET trial_live_seconds = 600 WHERE id = ?", (uid,))
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    run_session(client, 1, {"translate": ["de"], "captions": "local"})  # translations
    s = the_session()
    assert s["trial_seconds"] == 0 and s["cost_micros"] > 0 and trial_left(uid) == 600
    assert routes._trial_eligible({}, "light", "local") and routes._trial_eligible({}, None, "local")
    assert not routes._trial_eligible({}, "light", "premium")  # extra accurate
    assert not routes._trial_eligible({"translate": "de"}, "light", None)
    assert not routes._trial_eligible({"translate": ["de"]}, None, "local")
    assert routes._trial_eligible({"translate": ""}, "light", None) and routes._trial_eligible({"translate": []}, "light", None)


def test_extra_accurate_captions_are_billed_even_with_trial_seconds(client, signed_in, fake_worker, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()
    from acw.stream import openai_live

    async def connect(self):
        return None

    async def close(self):
        return None

    monkeypatch.setattr(openai_live.PremiumCaptions, "connect", connect)
    monkeypatch.setattr(openai_live.PremiumCaptions, "close", close)
    uid, _ = signed_in
    db.execute("UPDATE users SET trial_live_seconds = 600 WHERE id = ?", (uid,))
    assert run_session(client, 1, {"captions": "premium", "language": "en"})[0]["code"] == "insufficient_funds"
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    run_session(client, 1, {"captions": "premium", "language": "en"})
    s = the_session()
    assert s["trial_seconds"] == 0 and s["cost_micros"] > 0 and trial_left(uid) == 600


# ---------------------------------------------------------------- one at a time, and not while live is busy

def test_one_trial_session_at_a_time(client, signed_in, fake_worker, monkeypatch):
    uid, _ = signed_in
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from acw.config import get_settings

    get_settings.cache_clear()
    db.execute("UPDATE users SET trial_live_seconds = 600 WHERE id = ?", (uid,))
    other = routes.Session(SimpleNamespace(), SimpleNamespace(user_id=uid), {"echo_reference": False}, 4000, 600.0)
    routes._open[uid] = [other]
    try:
        got = run_session(client, 1)
        assert got[0]["code"] == "too_many_sessions" and "one live session at a time" in got[0]["message"]
        assert db.one("SELECT COUNT(*) n FROM stream_sessions")["n"] == 0
        # a paid session next to it is fine
        ledger.credit(uid, 5_000_000, "grant", "t", "g")
        assert run_session(client, 1, {"translate": ["de"], "captions": "local"})[0]["type"] == "ready"
        # and so is a trial session once the other one is gone
        routes._open[uid] = []
        assert run_session(client, 1)[0]["trial"] is True
    finally:
        routes._open.pop(uid, None)


def test_the_account_limit_still_counts_trial_sessions(client, signed_in, fake_worker):
    uid, _ = signed_in
    db.execute("UPDATE users SET trial_live_seconds = 600 WHERE id = ?", (uid,))
    routes._live[uid] = routes.MAX_SESSIONS_PER_USER
    try:
        assert run_session(client, 1)[0]["code"] == "too_many_sessions"
    finally:
        routes._live.pop(uid, None)


def test_the_worker_turns_trial_sessions_away_above_70_percent(monkeypatch):
    from acw import cpusplit

    monkeypatch.setattr(worker, "_monitor", None)
    monkeypatch.setattr(cpusplit, "units_per_core", lambda: 2.4)
    cap = worker.capacity_units()
    trial, paid = {"denoise": "light", "trial": True}, {"denoise": "light"}
    monkeypatch.setattr(worker, "_units", 0.69 * cap)
    assert not worker.trial_busy(trial)
    monkeypatch.setattr(worker, "_units", 0.71 * cap)
    assert worker.trial_busy(trial)
    assert not worker.trial_busy(paid) and worker.admit(paid)[0]  # paid sessions are unaffected, up to full
    monkeypatch.setattr(worker, "_units", 0.0)
    assert not worker.trial_busy(trial)


def test_a_busy_worker_gives_trial_busy_and_costs_no_trial_time(client, signed_in, fake_worker):
    uid, _ = signed_in
    db.execute("UPDATE users SET trial_live_seconds = 600 WHERE id = ?", (uid,))
    fake_worker.reply = {"type": "error", "code": "trial_busy"}
    got = run_session(client, 1)
    assert got[0]["code"] == "trial_busy" and got[0]["retryAfterSeconds"] > 0
    assert got[0]["message"] == ("Free trial sessions are paused while live is busy. "
                                 "Try again in a few minutes, or add credit to go live now.")
    assert db.one("SELECT trial_seconds FROM stream_sessions")["trial_seconds"] == 0
    assert trial_left(uid) == 600 and db.one("SELECT COUNT(*) n FROM holds")["n"] == 0


# ---------------------------------------------------------------- restarts

def test_an_interrupted_trial_session_keeps_the_seconds_that_were_saved():
    uid = auth.get_or_create_user("o@example.com")["id"]
    db.execute("UPDATE users SET trial_live_seconds = 500 WHERE id = ?", (uid,))
    db.execute("INSERT INTO stream_sessions(id,user_id,denoise_tier,status,seconds,trial_seconds,started_at)"
               " VALUES ('s-t',?,'light','active',100,100,?)", (uid, db.now() - 3600))
    sess = routes.Session(SimpleNamespace(), SimpleNamespace(user_id=uid), {}, 4000, 600.0)
    sess.id, sess.samples = "s-t", 48_000 * 100
    sess.save_progress()  # what the session did every 30 seconds
    sess.save_progress()  # twice is not charged twice
    assert trial_left(uid) == 500 - 100
    assert routes.close_orphans(0, "server_restart") == 1
    row = db.one("SELECT status, cost_micros, end_reason, trial_seconds FROM stream_sessions WHERE id = 's-t'")
    assert dict(row) == {"status": "ended", "cost_micros": 0, "end_reason": "server_restart", "trial_seconds": 100}
    assert trial_left(uid) == 400 and ledger.balance(uid)["balance_micros"] == 0


def test_an_interrupted_session_that_had_moved_on_is_billed_only_after_the_trial():
    uid = auth.get_or_create_user("o2@example.com")["id"]
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    hold = ledger.create_hold(uid, 200_000, "stream", "s-m")
    db.execute("INSERT INTO stream_sessions(id,user_id,denoise_tier,status,hold_id,seconds,trial_seconds,started_at)"
               " VALUES ('s-m',?,'light','active',?,130,100,?)", (uid, hold, db.now() - 3600))
    asyncio.run(routes.startup())
    rate = routes._rate_micros_per_minute("light", None)
    assert db.one("SELECT cost_micros FROM stream_sessions WHERE id = 's-m'")["cost_micros"] == routes._micros(rate, 30)
    assert ledger.balance(uid)["held_micros"] == 0


# ---------------------------------------------------------------- unchanged by the trial

def test_the_session_cleanup_runs():
    db.execute("INSERT INTO deleted_emails(email_hash, day, free_seconds) VALUES ('x', '2026-01-01', 1)")
    auth.purge_expired()
    assert db.one("SELECT COUNT(*) n FROM deleted_emails")["n"] == 0


def test_a_refund_never_pays_out_given_credit(monkeypatch):
    from acw.billing import stripe_service

    uid = auth.get_or_create_user("refund@example.com")["id"]
    ledger.credit(uid, 600_000, "grant", "admin", uid, "A grant")  # $0.60 given
    now = db.now()
    db.execute(
        "INSERT INTO payments(id, user_id, kind, status, amount_cents, refunded_cents, tax_cents, stripe_charge, created_at, updated_at)"
        " VALUES ('pay_w', ?, 'checkout', 'succeeded', 500, 0, 0, 'ch_w', ?, ?)", (uid, now, now))
    ledger.credit(uid, 500 * 10_000, "topup", "payment", "pay_w")  # $5 bought
    refunds = []

    class Fake:
        class v1:
            class charges:
                @staticmethod
                def retrieve(_id, params=None):
                    return SimpleNamespace(balance_transaction=SimpleNamespace(fee=45, currency="usd"))

            class refunds:
                @staticmethod
                def create(params=None, options=None):
                    refunds.append(params["amount"])
                    return SimpleNamespace(id="re_w")

    monkeypatch.setattr(stripe_service, "client", lambda: Fake)
    monkeypatch.setattr(stripe_service, "_reverse_tax", lambda *a: None)
    ledger.debit_available(uid, 4_000_000, "usage", "job", "j", "used $4")  # $1.60 left: $1 bought + $0.60 given
    assert stripe_service.refund_unused(uid, "pay_w") == 100 - 45  # only the $1 that was bought (less the card fee)
    assert refunds == [55]
    assert ledger.balance(uid)["balance_micros"] == 600_000  # the given credit stays to be used
