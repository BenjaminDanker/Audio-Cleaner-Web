"""The free live trial: every new account gets two hours of Standard live once per mailbox, ever; a trial session is
not charged, moves on to paid billing when the trial runs out (or stops), and is admitted only while live is not busy."""
import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from starlette.websockets import WebSocketDisconnect

from acw import auth, db
from acw.billing import ledger
from acw.config import get_settings
from acw.stream import protocol as P
from acw.stream import routes
from acw.stream.features import Features
from acw.stream.meter import micros as meter_micros

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

def test_a_new_account_gets_two_hours_once_per_mailbox_and_aliases_share_it(monkeypatch):
    _trial_on(monkeypatch)
    user = auth.get_or_create_user("new@example.com")
    assert user["trial_live_seconds"] == 7200
    assert ledger.balance(user["id"])["balance_micros"] == 0  # time, not money
    db.execute("UPDATE users SET trial_live_seconds = 100 WHERE id = ?", (user["id"],))
    auth.get_or_create_user("NEW@example.com")  # signing in again: nothing more
    assert trial_left(user["id"]) == 100
    assert db.one("SELECT COUNT(*) n FROM welcome_claims WHERE email_hash LIKE '%new%'")["n"] == 0  # only a hash
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


def test_switched_off_no_trial_is_given_and_none_is_given_later(monkeypatch):
    _trial_on(monkeypatch, 0)
    uid = auth.get_or_create_user("old@example.com")["id"]  # created while the trial was off
    assert db.one("SELECT COUNT(*) n FROM welcome_claims")["n"] == 0  # off: nobody is marked as having had it
    _trial_on(monkeypatch)
    auth.get_or_create_user("old@example.com")
    assert trial_left(uid) == 0


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


def run_session(live_client, seconds, start=None, stop=True):
    """Start a session, send `seconds` of audio, and return every message it got back until it was closed."""
    start = {"type": "start", "denoise": "light", "sampleRate": 48000, **(start or {})}
    got = []
    with live_client.websocket_connect("/api/stream/ws", headers={"origin": ORIGIN}) as ws:
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


def test_a_trial_session_uses_trial_seconds_costs_nothing_and_leaves_credit_alone(client, live_client, signed_in, fake_worker):
    uid, _ = signed_in
    db.execute("UPDATE users SET trial_live_seconds = 600 WHERE id = ?", (uid,))
    ledger.credit(uid, 5_000_000, "grant", "t", "g")  # the trial is used before credit, and credit stays untouched
    got = run_session(live_client, 2)
    assert got[0]["type"] == "ready" and got[0]["trial"] is True and got[0]["trialSecondsLeft"] == 600
    assert not [m for m in got if m.get("type") in ("warning", "stopped", "error")]
    s = the_session()
    assert s["status"] == "ended" and s["seconds"] == pytest.approx(2) and s["trial_seconds"] == 2
    assert s["cost_micros"] == 0 and s["hold_id"] is None
    assert trial_left(uid) == 598
    assert db.one("SELECT COUNT(*) n FROM holds")["n"] == 0
    assert ledger.balance(uid) == {"balance_micros": 5_000_000, "held_micros": 0, "available_micros": 5_000_000}
    assert fake_worker.started[0]["trial"] is True


def test_trial_seconds_are_taken_off_while_the_session_runs(client, live_client, signed_in, fake_worker):
    uid, _ = signed_in
    db.execute("UPDATE users SET trial_live_seconds = 600 WHERE id = ?", (uid,))
    run_session(live_client, 2, stop=False)  # the client just vanishes
    s = the_session(wait=False)  # a vanished client: the row stays "active" for the restart cleanup
    assert 0 < s["trial_seconds"] <= 2 and trial_left(uid) == pytest.approx(600 - s["trial_seconds"])  # saved with the frames


def test_trial_low_warns_at_five_minutes_left(client, live_client, signed_in, fake_worker):
    uid, _ = signed_in
    db.execute("UPDATE users SET trial_live_seconds = 299.5 WHERE id = ?", (uid,))
    got = run_session(live_client, 2)
    warnings = [m for m in got if m.get("type") == "warning"]
    assert len(warnings) == 1 and warnings[0]["code"] == "trial_low" and 296 <= warnings[0]["secondsLeft"] <= 299
    db.execute("UPDATE users SET trial_live_seconds = 3600 WHERE id = ?", (uid,))  # with plenty left there is no warning
    assert not [m for m in run_session(live_client, 1) if m.get("type") == "warning"]


def test_when_the_trial_ends_the_session_moves_on_to_credit(client, live_client, signed_in, fake_worker):
    uid, _ = signed_in
    db.execute("UPDATE users SET trial_live_seconds = 1 WHERE id = ?", (uid,))
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    got = run_session(live_client, 2.5)
    types = [(m["type"], m.get("code") or m.get("reason")) for m in got[1:]]
    assert ("warning", "trial_used_up_now_paid") in types and ("stopped", "trial_over") not in types
    assert all(t[0] != "error" for t in types)
    s = the_session()
    rate = Features(denoise="light").rate()
    assert s["seconds"] == pytest.approx(2.5) and s["trial_seconds"] == 1 and s["end_reason"] == "client_stop"
    assert s["cost_micros"] == meter_micros(0, rate, 1.5, 1.5)  # only what came after the trial
    assert ledger.balance(uid)["balance_micros"] == 5_000_000 - s["cost_micros"]
    assert ledger.balance(uid)["held_micros"] == 0
    assert trial_left(uid) == 0


def test_when_the_trial_ends_without_credit_the_session_stops(client, live_client, signed_in, fake_worker):
    uid, _ = signed_in
    db.execute("UPDATE users SET trial_live_seconds = 1 WHERE id = ?", (uid,))
    got = run_session(live_client, 3)
    assert {"type": "stopped", "reason": "trial_over"} in got
    s = the_session()
    assert s["end_reason"] == "trial_over" and s["cost_micros"] == 0 and s["trial_seconds"] == 1
    assert s["seconds"] < 1.1  # no audio was processed past the trial
    assert trial_left(uid) == 0 and db.one("SELECT COUNT(*) n FROM holds")["n"] == 0
    got = run_session(live_client, 1)  # and with no trial and no credit, the next one is refused
    assert got[0]["code"] == "insufficient_funds"


def test_a_session_after_the_trial_is_billed_from_credit(client, live_client, signed_in, fake_worker):
    uid, _ = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    run_session(live_client, 2)
    s = the_session()
    assert s["trial_seconds"] == 0 and s["cost_micros"] == meter_micros(0, Features(denoise="light").rate(), 2, 2)


@pytest.mark.parametrize("denoise, captions, trial", [
    ("light", "local", True), (None, "local", True), ("light", None, True),  # Standard noise removal and captions: covered
    ("light", "premium", False),  # extra accurate captions: paid
])
def test_only_standard_live_is_covered_by_the_trial(denoise, captions, trial):
    assert bool(routes._trial_eligible({}, denoise, captions)) is trial


@pytest.mark.parametrize("translate", [["de"], "de"])
def test_translations_are_not_covered_by_the_trial(translate):
    assert not routes._trial_eligible({"translate": translate}, "light", None)
    assert routes._trial_eligible({"translate": ""}, "light", None) and routes._trial_eligible({"translate": []}, "light", None)


def test_sessions_with_translations_are_billed_from_credit(client, live_client, signed_in, fake_worker, monkeypatch):
    uid, _ = signed_in
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")  # translations need the speech partner configured
    get_settings.cache_clear()
    db.execute("UPDATE users SET trial_live_seconds = 600 WHERE id = ?", (uid,))
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    run_session(live_client, 1, {"translate": ["de"], "captions": "local"})
    s = the_session()
    assert s["trial_seconds"] == 0 and s["cost_micros"] > 0 and trial_left(uid) == 600


def test_extra_accurate_captions_are_billed_even_with_trial_seconds(client, live_client, signed_in, fake_worker, monkeypatch):
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
    assert run_session(live_client, 1, {"captions": "premium", "language": "en"})[0]["code"] == "insufficient_funds"
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    run_session(live_client, 1, {"captions": "premium", "language": "en"})
    s = the_session()
    assert s["trial_seconds"] == 0 and s["cost_micros"] > 0 and trial_left(uid) == 600


# ---------------------------------------------------------------- one at a time, and not while live is busy

def test_one_trial_session_at_a_time_and_the_account_limit_counts_trial_sessions(client, live_client, signed_in, fake_worker, monkeypatch):
    uid, _ = signed_in
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()
    db.execute("UPDATE users SET trial_live_seconds = 600 WHERE id = ?", (uid,))
    other = routes.Session(SimpleNamespace(), SimpleNamespace(user_id=uid), Features(denoise="light"), 600.0)
    routes._open[uid] = [other]
    try:
        got = run_session(live_client, 1)
        assert got[0]["code"] == "too_many_sessions" and "one live session at a time" in got[0]["message"]
        assert db.one("SELECT COUNT(*) n FROM stream_sessions")["n"] == 0
        # a paid session next to it is fine
        ledger.credit(uid, 5_000_000, "grant", "t", "g")
        assert run_session(live_client, 1, {"translate": ["de"], "captions": "local"})[0]["type"] == "ready"
        # and so is a trial session once the other one is gone
        routes._open[uid] = []
        assert run_session(live_client, 1)[0]["trial"] is True
    finally:
        routes._open.pop(uid, None)
    routes._live[uid] = routes.MAX_SESSIONS_PER_USER  # the account limit still counts trial sessions
    try:
        assert run_session(live_client, 1)[0]["code"] == "too_many_sessions"
    finally:
        routes._live.pop(uid, None)


def test_a_busy_worker_gives_trial_busy_and_costs_no_trial_time(client, live_client, signed_in, fake_worker):
    uid, _ = signed_in
    db.execute("UPDATE users SET trial_live_seconds = 600 WHERE id = ?", (uid,))
    fake_worker.reply = {"type": "error", "code": "trial_busy"}
    got = run_session(live_client, 1)
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
    sess = routes.Session(SimpleNamespace(), SimpleNamespace(user_id=uid), Features(denoise="light"), 600.0)
    sess.id = "s-t"
    sess.count(48_000 * 100)
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
    rate = Features(denoise="light").rate()
    db.execute("INSERT INTO stream_sessions(id,user_id,denoise_tier,status,hold_id,seconds,trial_seconds,paid_seconds,meter_rate,"
               "meter_seconds,started_at) VALUES ('s-m',?,'light','active',?,130,100,30,?,30,?)", (uid, hold, rate, db.now() - 3600))
    asyncio.run(routes.startup())
    assert db.one("SELECT cost_micros FROM stream_sessions WHERE id = 's-m'")["cost_micros"] == meter_micros(0, rate, 30, 30)
    assert ledger.balance(uid)["held_micros"] == 0


# ---------------------------------------------------------------- unchanged by the trial

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
