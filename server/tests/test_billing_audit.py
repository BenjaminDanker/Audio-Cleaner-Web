"""Money on the job and live-session paths: billed lengths, holds and refunds, reconciliation with Stripe, price
snapshots and the session meter. Also the helpers other test modules import (frame, make_job, run_session, ...)."""
import asyncio
import json
import math
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from acw import auth, db, jobs
from acw.billing import ledger, pricing, stripe_service
from acw.stream.features import Features

HOME = Path(os.environ.get("ACW_HOME", Path.home() / "audio-cleaner"))
FFMPEG = HOME / "bin/ffmpeg"
NATIVE = pytest.mark.skipif(not FFMPEG.exists(), reason="ffmpeg not available")


def user(email="a@example.com"):
    return auth.get_or_create_user(email)["id"]


def make_job(uid, **kw):
    args = dict(kind="denoise", denoise_tier="standard", transcribe_tier=None, filename="a.wav", size_bytes=1000,
                claimed_seconds=600.0, options={})
    return jobs.create_job(uid, **{**args, **kw})["id"]


def running_job(uid, **kw):
    jid = make_job(uid, **kw)
    db.execute("UPDATE jobs SET status = 'running', duration_seconds = claimed_seconds WHERE id = ?", (jid,))
    d = jobs.job_dir(jid)
    (d / "out.wav").write_bytes(b"x")
    return dict(db.one("SELECT * FROM jobs WHERE id = ?", (jid,))), d


def frame(reference=False, seq=0):
    from acw.stream import protocol as P

    flags = P.FLAG_REFERENCE if reference else 0
    return {"type": "websocket.receive", "bytes": P.HEADER.pack(P.VERSION, P.CODEC_S16, flags, seq, 0) + bytes(960)}


# ---------------------------------------------------------------- measured length (a header can lie)

def forged_flac(tmp_path, real=120, claimed=10):
    """A FLAC of `real` seconds whose STREAMINFO says `claimed`."""
    path = tmp_path / "forged.flac"
    subprocess.run([str(FFMPEG), "-v", "error", "-f", "lavfi", "-i", f"sine=f=440:d={real}:sample_rate=8000", "-c:a", "flac",
                    "-y", str(path)], check=True)
    b = bytearray(path.read_bytes())
    field = int.from_bytes(b[18:26], "big")  # 20 bits rate, 3 channels, 5 bits depth, 36 bits total samples
    b[18:26] = ((field & ~((1 << 36) - 1)) | claimed * 8000).to_bytes(8, "big")
    path.write_bytes(bytes(b))
    return path


def runner_spec(tmp_path, **kw):
    return {"bin_dir": str(HOME / "bin"), "input": "forged.flac", "threads": 2, "max_seconds": pricing.MAX_JOB_SECONDS, "options": {}, **kw}


@NATIVE
def test_forged_header_is_billed_and_limited_by_the_measured_length(tmp_path):
    from acw.worker import runner
    from acw_common.ffmpeg import FFmpeg, MediaError

    path = forged_flac(tmp_path)
    ff = FFmpeg(HOME / "bin")
    probe = ff.probe(path)
    assert probe.duration == pytest.approx(10, abs=0.1)  # what the header claims
    assert ff.measure_duration(path, probe, 3600) == pytest.approx(120, abs=0.5)
    assert runner.stage_probe(tmp_path, runner_spec(tmp_path))["duration"] == pytest.approx(120, abs=0.5)
    with pytest.raises(MediaError, match="2.0 minutes"):  # the 10 s file is 2 minutes: over a 1 minute limit
        runner.stage_probe(tmp_path, runner_spec(tmp_path, max_seconds=60))


@NATIVE
def test_decode_stops_at_the_billed_length(tmp_path):
    from acw.worker import runner
    from acw_common.ffmpeg import FFmpeg

    path = forged_flac(tmp_path)
    ff = FFmpeg(HOME / "bin")
    probe = ff.probe(path)
    limit = runner.decode_limit(runner_spec(tmp_path, billed_seconds=10.0))
    assert limit == pytest.approx(12.2)
    assert len(ff.decode(path, probe, 16000, 1, limit)) <= 16000 * limit
    assert runner.decode_limit(runner_spec(tmp_path)) == pricing.MAX_JOB_SECONDS  # before the probe: the tier's limit


def test_cloud_asr_refuses_more_speech_than_was_billed(tmp_path, monkeypatch):
    from acw.asr import openai_asr
    from acw.config import get_settings

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()
    d = tmp_path / "asr_chunks"
    d.mkdir()
    (d / "0000.mp3").write_bytes(b"x")
    (d / "index.json").write_text(json.dumps({"chunks": [{"file": "0000.mp3", "offset": 0, "seconds": 15.0}]}))
    with pytest.raises(openai_asr.CloudASRError, match="billed"):
        asyncio.run(openai_asr.run_for_job(tmp_path, None, billed_seconds=10.0))


# ---------------------------------------------------------------- refunds

def refundable_user(email="refund@example.com"):
    uid = user(email)
    now = db.now()
    db.execute("INSERT INTO payments(id,user_id,kind,status,amount_cents,refunded_cents,tax_cents,stripe_charge,created_at,updated_at)"
               " VALUES ('p1',?,'checkout','succeeded',1000,0,0,'ch_1',?,?)", (uid, now, now))
    ledger.credit(uid, 10_000_000, "topup", "payment", "p1")
    return uid


def test_booking_a_refund_twice_takes_the_money_once():
    uid = refundable_user()
    for _ in range(2):  # a double click: two requests, the same Stripe refund
        hold = ledger.create_hold(uid, 5_000_000, "refund", "p1")
        stripe_service._book_refund(uid, "p1", hold, "re_1", 500, 0, "Refund")
    b = ledger.balance(uid)
    assert b["balance_micros"] == 5_000_000 and b["held_micros"] == 0
    assert db.one("SELECT refunded_cents FROM payments WHERE id='p1'")["refunded_cents"] == 500
    assert db.one("SELECT COUNT(*) n FROM holds WHERE status='active'")["n"] == 0


def test_refund_fee_part_is_booked_after_the_webhook_took_the_rest():
    uid = refundable_user()
    stripe_service._handle_charge_refunded({"id": "ch_1", "payment_intent": None, "amount_refunded": 441,
                                            "refunds": {"data": [{"id": "re_9", "status": "succeeded", "amount": 441}]}})
    hold = ledger.create_hold(uid, 5_000_000, "refund", "p1")
    stripe_service._book_refund(uid, "p1", hold, "re_9", 500, 0, "Refund")
    assert ledger.balance(uid)["balance_micros"] == 5_000_000
    assert db.one("SELECT COUNT(*) n FROM ledger WHERE kind='refund' AND ref_id='re_9'")["n"] == 2


def test_refund_hold_is_released_when_stripe_fails(monkeypatch):
    uid = refundable_user()

    class Fake:
        class v1:
            class charges:
                retrieve = staticmethod(lambda *a, **k: SimpleNamespace(balance_transaction=SimpleNamespace(fee=59, currency="usd")))

            class refunds:
                @staticmethod
                def create(params=None, options=None):
                    raise RuntimeError("stripe down")

    monkeypatch.setattr(stripe_service, "client", lambda: Fake)
    with pytest.raises(RuntimeError):
        stripe_service.refund_unused(uid, "p1")
    assert ledger.balance(uid)["held_micros"] == 0


def test_refund_of_spent_credit_is_recorded_as_unrecovered():
    uid = refundable_user()
    ledger.debit_available(uid, 9_900_000, "usage")  # nearly all spent
    refund = {"id": "ch_1", "payment_intent": None, "amount_refunded": 1000,
              "refunds": {"data": [{"id": "re_1", "status": "succeeded", "amount": 1000}]}}
    stripe_service._handle_charge_refunded(refund)
    stripe_service._handle_charge_refunded(refund)  # delivered twice
    rows = db.all_rows("SELECT detail FROM audit_log WHERE action = 'refund_unrecovered'")
    assert [r["detail"] for r in rows] == ["p1:re_1:9900000 micros"]


def test_dispute_of_spent_credit_is_recorded_as_unrecovered():
    uid = refundable_user()
    ledger.debit_available(uid, 9_000_000, "usage")
    stripe_service._handle_dispute_created({"id": "dp_1", "charge": "ch_1", "reason": "fraudulent"})
    rows = db.all_rows("SELECT detail FROM audit_log WHERE action = 'dispute_unrecovered'")
    assert [r["detail"] for r in rows] == ["p1:dp_1:9000000 micros"]


# ---------------------------------------------------------------- holds that nothing will close

def test_finish_failed_releases_the_hold_only_from_the_states_it_may_leave():
    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    jid = make_job(uid)
    job = dict(db.one("SELECT * FROM jobs WHERE id = ?", (jid,)))
    assert ledger.balance(uid)["held_micros"] > 0
    jobs.finish_failed(job, "Canceled.", status="canceled", only_from=("running",))  # not in that state: nothing happens
    assert db.one("SELECT status FROM jobs WHERE id = ?", (jid,))["status"] == "uploading"
    jobs.finish_failed(job, "Canceled.", status="canceled", only_from=("uploading", "queued"))
    assert db.one("SELECT status FROM jobs WHERE id = ?", (jid,))["status"] == "canceled"
    assert ledger.balance(uid)["held_micros"] == 0


def test_reaper_releases_holds_nothing_will_close():
    uid = user()
    ledger.credit(uid, 50_000_000, "grant", "t", "g")
    old = db.now() - 3600
    finished = make_job(uid)  # a hold on a job that is over
    db.execute("UPDATE jobs SET status = 'failed' WHERE id = ?", (finished,))
    running = make_job(uid, claimed_seconds=60)
    db.execute("UPDATE jobs SET status = 'running' WHERE id = ?", (running,))
    ended = ledger.create_hold(uid, 100_000, "stream", "s1")
    db.execute("INSERT INTO stream_sessions(id,user_id,status,hold_id,started_at) VALUES ('s1',?,'ended',?,?)", (uid, ended, old))
    live = ledger.create_hold(uid, 100_000, "stream", "s2")
    db.execute("INSERT INTO stream_sessions(id,user_id,status,hold_id,started_at) VALUES ('s2',?,'active',?,?)", (uid, live, db.now()))
    fresh_refund = ledger.create_hold(uid, 100_000, "refund", "p")
    stale_refund = ledger.create_hold(uid, 100_000, "refund", "p")
    db.execute("UPDATE holds SET created_at = ? WHERE id = ?", (db.now() - 700, stale_refund))
    orphan = ledger.create_hold(uid, 100_000, "stream", "nowhere")  # just created: its row may still be coming
    assert ledger.reap_stale_holds() == 3
    status = {r["id"]: r["status"] for r in db.all_rows("SELECT id, status FROM holds")}
    assert status[ended] == "released" and status[stale_refund] == "released"
    assert status[live] == status[fresh_refund] == status[orphan] == "active"
    assert db.one("SELECT status FROM holds WHERE ref_id = ?", (running,))["status"] == "active"
    db.execute("UPDATE holds SET created_at = ? WHERE id = ?", (old, orphan))
    assert ledger.reap_stale_holds() == 1
    assert ledger.balance(uid)["held_micros"] == db.one("SELECT SUM(amount_micros) s FROM holds WHERE status = 'active'")["s"]


# ---------------------------------------------------------------- payment reconciliation with Stripe

class StripeFake:
    """Only the network calls reconcile() makes; the rest of the client is real."""

    def __init__(self, monkeypatch, session=None, pi=None, found=None, capture_fails=False):
        self.session, self.pi, self.found = session, pi, found
        self.cancels, self.captures = [], []
        fake = self
        real = stripe_service.client

        def retrieve_pi(_id):
            return fake.pi

        def capture(_id, params=None, options=None):
            if capture_fails:
                raise RuntimeError("stripe down")
            fake.captures.append(_id)
            fake.pi = {**fake.pi, "status": "succeeded"}
            return fake.pi

        def cancel(_id, params=None, options=None):
            fake.cancels.append(_id)
            fake.pi = {**fake.pi, "status": "canceled"}
            return fake.pi

        def client():
            c = real()
            c.v1.checkout = SimpleNamespace(sessions=SimpleNamespace(retrieve=lambda _id: fake.session))
            c.v1.payment_intents = SimpleNamespace(
                retrieve=retrieve_pi, capture=capture, cancel=cancel,
                search=lambda params=None: SimpleNamespace(data=[fake.found] if fake.found else []))
            c.v1.charges = SimpleNamespace(retrieve=lambda _id, params=None: SimpleNamespace(
                payment_method_details=SimpleNamespace(card=SimpleNamespace(country="US"))))
            return c

        monkeypatch.setattr(stripe_service, "client", client)


def pending_payment(uid, kind="checkout", age=3600, **cols):
    created = db.now() - age
    values = {"id": "pay1", "user_id": uid, "kind": kind, "status": "pending", "amount_cents": 2000,
              "stripe_checkout_session": "cs_1" if kind == "checkout" else None, "created_at": created, "updated_at": created, **cols}
    db.execute(f"INSERT INTO payments({','.join(values)}) VALUES ({','.join('?' * len(values))})", tuple(values.values()))


def session_dict(uid, status="complete"):
    return {"id": "cs_1", "mode": "payment", "status": status, "payment_status": "unpaid", "payment_intent": "pi_1",
            "metadata": {"acw_user_id": uid, "acw_payment_id": "pay1"}, "total_details": {"amount_tax": 0}}


def pi_dict(uid, status="requires_capture", kind="checkout"):
    return {"id": "pi_1", "status": status, "currency": "usd", "payment_method": "pm_1", "latest_charge": "ch_1",
            "metadata": {"acw_user_id": uid, "acw_payment_id": "pay1", "acw_kind": kind}}


def test_reconcile_credits_a_checkout_whose_webhook_never_came(monkeypatch):
    uid = user()
    pending_payment(uid)
    fake = StripeFake(monkeypatch, session=session_dict(uid), pi=pi_dict(uid))
    assert stripe_service.reconcile() == 1
    assert ledger.balance(uid)["balance_micros"] == 20_000_000 and fake.captures == ["pi_1"]
    assert stripe_service.reconcile() == 0  # settled: nothing left to look at


def test_reconcile_leaves_young_and_still_open_payments_alone(monkeypatch):
    uid = user()
    pending_payment(uid, age=60)
    StripeFake(monkeypatch, session=session_dict(uid, "open"), pi=pi_dict(uid))
    assert stripe_service.reconcile() == 0  # too young to look at
    db.execute("UPDATE payments SET updated_at = ?", (db.now() - 3600,))
    assert stripe_service.reconcile() == 1
    assert db.one("SELECT status FROM payments")["status"] == "pending"  # still being paid
    assert stripe_service.reconcile() == 0  # and not asked about again on the next tick


def test_reconcile_closes_an_expired_checkout(monkeypatch):
    uid = user()
    pending_payment(uid)
    StripeFake(monkeypatch, session=session_dict(uid, "expired"))
    stripe_service.reconcile()
    p = db.one("SELECT status, failure_code FROM payments")
    assert (p["status"], p["failure_code"]) == ("failed", "expired")


@pytest.mark.parametrize("age_hours,cancelled", [(25, True), (3, False)])
def test_an_authorization_that_will_not_settle_is_cancelled_after_a_day(monkeypatch, age_hours, cancelled):
    uid = user()
    pending_payment(uid, age=age_hours * 3600)
    fake = StripeFake(monkeypatch, session=session_dict(uid), pi=pi_dict(uid), capture_fails=True)
    stripe_service.reconcile()
    p = db.one("SELECT status, failure_code FROM payments")
    if cancelled:
        assert fake.cancels == ["pi_1"]
        assert (p["status"], p["failure_code"]) == ("failed", "unsettled")
        assert ledger.balance(uid)["balance_micros"] == 0
    else:
        assert fake.cancels == [] and p["status"] == "pending"


def reload_user(uid, in_flight=1):
    db.execute("UPDATE users SET auto_reload_enabled=1, auto_reload_in_flight=?, default_payment_method='pm_1',"
               " stripe_customer_id='cus_1' WHERE id=?", (in_flight, uid))


def test_stuck_reload_is_credited_from_stripe_and_unblocked(monkeypatch):
    uid = user()
    reload_user(uid)
    pending_payment(uid, kind="auto_reload", age=2400)  # no PaymentIntent id saved: our request failed midway
    fake = StripeFake(monkeypatch, pi=pi_dict(uid, "requires_capture", "auto_reload"), found=pi_dict(uid, "requires_capture", "auto_reload"))
    stripe_service.reconcile()
    assert fake.captures == ["pi_1"]
    assert ledger.balance(uid)["balance_micros"] == 20_000_000
    assert db.one("SELECT auto_reload_in_flight FROM users WHERE id = ?", (uid,))["auto_reload_in_flight"] == 0


def test_reload_that_never_reached_stripe_expires_and_a_stale_flag_clears(monkeypatch):
    uid = user()
    reload_user(uid)
    pending_payment(uid, kind="auto_reload", age=600)
    StripeFake(monkeypatch)
    stripe_service.reconcile()  # 10 minutes: could still be on its way
    assert db.one("SELECT auto_reload_in_flight FROM users")["auto_reload_in_flight"] == 1
    db.execute("UPDATE payments SET created_at = ?, updated_at = ?", (db.now() - 2400,) * 2)
    stripe_service.reconcile()
    assert db.one("SELECT auto_reload_in_flight FROM users")["auto_reload_in_flight"] == 0
    assert db.one("SELECT status FROM payments")["status"] == "failed"
    db.execute("UPDATE users SET auto_reload_in_flight = 1")  # a flag with no payment behind it at all
    db.execute("DELETE FROM payments")
    stripe_service.reconcile()
    assert db.one("SELECT auto_reload_in_flight FROM users")["auto_reload_in_flight"] == 0


def test_failed_reload_request_keeps_the_reload_in_flight(monkeypatch):
    uid = user()
    reload_user(uid, in_flight=0)
    db.execute("UPDATE users SET auto_reload_threshold_micros=5000000, auto_reload_amount_cents=2000 WHERE id=?", (uid,))
    StripeFake(monkeypatch)

    def boom(*a, **k):
        raise ConnectionError("network")

    real = stripe_service.client

    def client():
        c = real()
        c.v1.payment_intents = SimpleNamespace(create=boom)
        return c

    monkeypatch.setattr(stripe_service, "client", client)
    assert stripe_service.run_auto_reload(uid) == "error"
    assert db.one("SELECT auto_reload_in_flight FROM users")["auto_reload_in_flight"] == 1
    assert db.one("SELECT status FROM payments")["status"] == "pending"


def test_reload_threshold_must_be_below_the_amount_and_bad_old_settings_do_nothing(client, signed_in):
    uid, h = signed_in
    db.execute("UPDATE users SET default_payment_method = 'pm_1' WHERE id = ?", (uid,))
    body = {"enabled": True, "thresholdCents": 5000, "amountCents": 1000, "monthlyCapCents": 5000, "consent": True}
    assert client.post("/api/billing/auto-reload", json=body, headers=h).status_code == 400
    assert client.post("/api/billing/auto-reload", json={**body, "thresholdCents": 500}, headers=h).status_code == 200
    db.execute("UPDATE users SET auto_reload_threshold_micros = ? WHERE id = ?", (5000 * 10_000, uid))  # an old, bad setting
    assert stripe_service.users_needing_reload() == [] and stripe_service.run_auto_reload(uid) == "skip"


# ---------------------------------------------------------------- jobs: prices fixed at creation, limits, stages

def test_a_job_that_is_no_longer_running_is_not_charged_again():
    from acw.worker import main as W

    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    job, d = running_job(uid)
    res = {"outputs": [{"file": "out.wav", "kind": "audio"}]}
    W._complete(job, d, res)
    first = ledger.balance(uid)["balance_micros"]
    assert first < 5_000_000 and db.one("SELECT status, cost_micros FROM jobs WHERE id = ?", (job["id"],))["cost_micros"] == 5_000_000 - first
    W._complete(job, d, res)  # e.g. the job was requeued by the sweeper and finished twice
    assert ledger.balance(uid)["balance_micros"] == first


def test_job_prices_are_fixed_at_creation(monkeypatch):
    from acw import editing
    from acw.worker import main as W

    from .job_support import standard

    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    job, d = running_job(uid)
    pinned = json.loads(job["params_json"])["edit_rates"]  # (every job is made with a timeline; its parts' prices are pinned with it)
    assert pinned["light"] == pinned["heavy"] == 9_000 and pinned["leveling"] == 1_000
    monkeypatch.setitem(pricing.RATES, ("denoise", "standard"), pricing.Rate("denoise", "standard", "x", 90_000))
    monkeypatch.setitem(pricing.RATES, ("leveling", "standard"), pricing.Rate("leveling", "standard", "x", 50_000))
    assert editing.step_rates()["light"] == 90_000  # (today's list is dearer)
    W._complete(job, d, {"outputs": [{"file": "out.wav", "kind": "audio"}]})
    assert db.one("SELECT cost_micros FROM jobs WHERE id = ?", (job["id"],))["cost_micros"] == standard(600, rates=pinned)


def test_a_job_made_without_leveling_never_pays_for_leveling():
    from acw.worker import main as W

    from .job_support import standard

    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    job, d = running_job(uid, options={"leveling": False})
    params = json.loads(job["params_json"])
    assert "leveling" not in params["edit"]["timeline"]["layers"] and params["edit_rates"]["leveling"] > 0
    W._complete(job, d, {"outputs": [{"file": "out.wav", "kind": "audio"}]})
    assert db.one("SELECT cost_micros FROM jobs WHERE id = ?", (job["id"],))["cost_micros"] == standard(600, leveling=False)


def test_rebuild_is_limited_to_90_minutes_and_standard_is_not(gpu_up):
    uid = user()
    ledger.credit(uid, 50_000_000, "grant", "t", "g")
    with pytest.raises(jobs.JobError, match="up to 90 minutes"):
        make_job(uid, denoise_tier="rebuild", claimed_seconds=91 * 60)
    make_job(uid, denoise_tier="rebuild", claimed_seconds=90 * 60)
    make_job(uid, denoise_tier="standard", claimed_seconds=3 * 3600)


def test_stage_time_limits_scale_with_the_length():
    cpu, wall = jobs.stage_limits(4 * 3600, 4 * 3600)
    assert wall == 4 * 3600 * 2 + 1800 and cpu == 4 * 3600 * 8 + 1800
    assert jobs.stage_limits(600, 3600)[1] == 600 * 3 + 1800


# ---------------------------------------------------------------- live sessions: meter, billing, saving

def live_session(uid):
    from acw.stream import routes

    return routes.Session(None, SimpleNamespace(user_id=uid), Features(denoise="light"))


def test_session_seconds_are_counted_in_whole_samples():
    s = live_session("u")
    for _ in range(3000):  # exactly 60 s of 20 ms Opus frames
        s.count(960)
    assert s.seconds == 60 and s.meter.micros() == 4000
    s.count(1)
    assert s.meter.micros() == 4067  # the 61st second has started


def test_changed_features_bill_each_stretch_at_its_rate_and_round_up_once():
    s = live_session("u")
    s.count(48_000 * 60)  # a minute of noise removal
    s.set_features(Features(denoise="light", captions="local"))
    s.count(48_000 * 30 + 1)  # half a minute and a sample with Standard captions too
    both = Features(denoise="light", captions="local").rate()
    assert s.meter.micros() == 4000 + math.ceil(both * 31 / 60)  # the started second at the rate it ended with
    s.set_features(Features(denoise="light"))
    assert s.meter.micros() == math.ceil(4000 + both * (30 + 1 / 48_000) / 60 + 4000 * (1 - 1 / 48_000) / 60)


def run_session(monkeypatch, messages, uid, **timeouts):
    """Drive routes._run with a fake socket that delivers `messages` ((delay, message) pairs)."""
    from acw.stream import routes

    for k, v in timeouts.items():
        monkeypatch.setattr(routes, k, v)
    queue = list(messages)

    class FakeWS:
        sent = []

        async def receive(self):
            if not queue:
                await asyncio.sleep(3600)
            delay, msg = queue.pop(0)
            await asyncio.sleep(delay)
            return msg

        async def send_text(self, t):
            self.sent.append(json.loads(t))

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
    sess.reader = R()
    sess.writer = SimpleNamespace(close=lambda: None)
    sess.held_micros = sess.current_rate() * 5
    asyncio.run(routes._run(sess))
    return sess


def test_echo_reference_frames_do_not_keep_a_session_alive_and_the_client_is_told(monkeypatch):
    uid = user()
    sess = run_session(monkeypatch, [(0.01, frame(reference=True, seq=i)) for i in range(20)], uid, IDLE_TIMEOUT=0.1)
    assert sess.end_reason == "idle_timeout" and sess.seconds == 0
    assert {"type": "stopped", "reason": "idle_timeout"} in sess.ws.sent


def test_microphone_audio_keeps_a_session_alive_and_is_billed_by_length(monkeypatch):
    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    sess = run_session(monkeypatch, [(0.02, frame(seq=i)) for i in range(4)] + [(0.02, {"type": "websocket.disconnect"})], uid,
                       IDLE_TIMEOUT=0.3)
    assert sess.end_reason == "client_closed" and sess.seconds == pytest.approx(0.04)


def test_session_ends_at_the_wall_clock_limit(monkeypatch):
    uid = user()
    sess = run_session(monkeypatch, [(0.02, {"type": "websocket.receive", "text": json.dumps({"type": "ping"})})] * 40, uid,
                       MAX_SESSION_SECONDS=0.2)
    assert sess.end_reason == "max_duration"


def test_suspension_stops_a_running_session(monkeypatch):
    uid = user()
    db.execute("UPDATE users SET status = 'suspended' WHERE id = ?", (uid,))
    sess = run_session(monkeypatch, [(0.02, frame(seq=i)) for i in range(20)], uid, PERSIST_EVERY=0.1)
    assert sess.end_reason == "account_closed"


def test_startup_bills_the_saved_seconds_of_an_interrupted_session():
    from acw.stream import routes

    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    hold = ledger.create_hold(uid, 20_000, "stream", "s1")
    ledger.create_hold(uid, 20_000, "stream", "s2")
    db.execute("INSERT INTO stream_sessions(id,user_id,denoise_tier,status,hold_id,seconds,paid_seconds,meter_rate,meter_seconds,started_at)"
               " VALUES ('s1',?,'light','active',?,90,90,4000,90,?)", (uid, hold, db.now()))
    idle = ledger.create_hold(uid, 20_000, "stream", "s3")
    db.execute("INSERT INTO stream_sessions(id,user_id,denoise_tier,status,hold_id,seconds,started_at) VALUES ('s3',?,'light','active',?,0,?)",
               (uid, idle, db.now()))
    asyncio.run(routes.startup())
    s = db.one("SELECT status, end_reason, cost_micros FROM stream_sessions WHERE id = 's1'")
    assert (s["status"], s["end_reason"], s["cost_micros"]) == ("ended", "server_restart", 6000)  # 90 s of light denoise
    assert db.one("SELECT cost_micros FROM stream_sessions WHERE id = 's3'")["cost_micros"] == 0
    assert ledger.balance(uid)["balance_micros"] == 5_000_000 - 6000
    assert db.one("SELECT status FROM holds WHERE id = ?", (idle,))["status"] == "released"


def test_session_progress_is_saved_for_the_billing_after_a_crash(monkeypatch):
    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    db.execute("INSERT INTO stream_sessions(id,user_id,status,started_at) VALUES ('s9',?,'active',?)", (uid, db.now()))
    from acw.stream import routes

    sess = live_session(uid)
    sess.id, sess.samples = "s9", 48_000 * 31
    assert asyncio.run(routes._persist(sess)) is True
    assert db.one("SELECT seconds FROM stream_sessions WHERE id = 's9'")["seconds"] == 31
