"""Fixes from the money audit (F1-F20): measured lengths, idempotent refunds, leaked holds, payment
reconciliation, live-session billing and per-tier limits."""
import asyncio
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from acw import auth, db, jobs
from acw.billing import ledger, pricing, stripe_service

HOME = Path(os.environ.get("ACW_HOME", Path.home() / "audio-cleaner"))
FFMPEG = HOME / "bin/ffmpeg"
NATIVE = pytest.mark.skipif(not FFMPEG.exists(), reason="ffmpeg not available")


def user(email="a@example.com"):
    return auth.get_or_create_user(email)["id"]


def make_job(uid, **kw):
    args = dict(kind="denoise", denoise_tier="standard", transcribe_tier=None, filename="a.wav", size_bytes=1000,
                claimed_seconds=600.0, options={}, free=False)
    return jobs.create_job(uid, **{**args, **kw})["id"]


# ---------------------------------------------------------------- F1: measured length

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
    return {"bin_dir": str(HOME / "bin"), "input": "forged.flac", "threads": 2, "max_seconds": pricing.MAX_JOB_SECONDS, **kw}


@NATIVE
def test_forged_header_is_billed_for_the_real_length(tmp_path):
    from acw.worker import runner
    from acw_common.ffmpeg import FFmpeg, MediaError

    path = forged_flac(tmp_path)
    ff = FFmpeg(HOME / "bin")
    probe = ff.probe(path)
    assert probe.duration == pytest.approx(10, abs=0.1)  # what the header claims
    assert ff.measure_duration(path, probe, 3600) == pytest.approx(120, abs=0.5)
    assert runner.stage_probe(tmp_path, runner_spec(tmp_path))["duration"] == pytest.approx(120, abs=0.5)
    # limits use the measured length: the 10 s file is 2 minutes, over a 1 minute limit
    with pytest.raises(MediaError, match="2.0 minutes"):
        runner.stage_probe(tmp_path, runner_spec(tmp_path, max_seconds=60))
    with pytest.raises(MediaError, match="Voice rebuilding takes files up to 1 minutes"):
        runner.stage_probe(tmp_path, runner_spec(tmp_path, max_seconds=60, denoise_tier="rebuild"))


@NATIVE
def test_decode_stops_at_the_billed_length(tmp_path):
    from acw.worker import runner
    from acw_common.ffmpeg import FFmpeg

    path = forged_flac(tmp_path)
    ff = FFmpeg(HOME / "bin")
    probe = ff.probe(path)
    spec = runner_spec(tmp_path, billed_seconds=10.0)
    limit = runner.decode_limit(spec)
    assert limit == pytest.approx(12.2)
    assert len(ff.decode(path, probe, 16000, 1, limit)) <= 16000 * limit
    assert runner.decode_limit(runner_spec(tmp_path)) == pricing.MAX_JOB_SECONDS  # before the probe: the tier's limit
    assert runner.decode_limit(runner_spec(tmp_path, billed_seconds=10_000_000)) == pricing.MAX_JOB_SECONDS


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


# ---------------------------------------------------------------- F2: refunds

def refundable_user():
    uid = user("refund@example.com")
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


# ---------------------------------------------------------------- F3: leaked holds

def test_finish_failed_releases_the_hold_and_free_allowance_with_the_status():
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

    # a free job from before free subtitles moved to the browser gives its reserved minutes back
    fid = make_job(uid, kind="transcribe", denoise_tier=None, transcribe_tier="local", claimed_seconds=600)
    params = json.loads(db.one("SELECT params_json FROM jobs WHERE id = ?", (fid,))["params_json"])
    db.execute("UPDATE jobs SET free = 1, params_json = ? WHERE id = ?", (json.dumps({**params, "free_day": jobs.utc_day()}), fid))
    db.execute("INSERT INTO free_usage(user_id, day, seconds) VALUES (?,?,600)", (uid, jobs.utc_day()))
    jobs.finish_failed(dict(db.one("SELECT * FROM jobs WHERE id = ?", (fid,))), "x")
    assert db.one("SELECT seconds FROM free_usage WHERE user_id = ?", (uid,))["seconds"] == 0


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
    held = ledger.balance(uid)["held_micros"]
    assert held == db.one("SELECT SUM(amount_micros) s FROM holds WHERE status = 'active'")["s"]


def test_reaper_logs_a_held_total_that_does_not_match(caplog):
    uid = user()
    ledger.credit(uid, 1_000_000, "grant", "t", "g")
    db.execute("UPDATE users SET held_micros = 5 WHERE id = ?", (uid,))
    with caplog.at_level("ERROR"):
        ledger.reap_stale_holds()
    assert "held_micros 5" in caplog.text


# ---------------------------------------------------------------- F4: payment reconciliation

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


def test_reconcile_waits_for_recent_and_open_payments(monkeypatch):
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


def test_unsettled_authorization_is_cancelled_after_a_day(monkeypatch):
    uid = user()
    pending_payment(uid, age=25 * 3600)
    fake = StripeFake(monkeypatch, session=session_dict(uid), pi=pi_dict(uid), capture_fails=True)
    stripe_service.reconcile()
    assert fake.cancels == ["pi_1"]
    p = db.one("SELECT status, failure_code FROM payments")
    assert (p["status"], p["failure_code"]) == ("failed", "unsettled")
    assert ledger.balance(uid)["balance_micros"] == 0


def test_unsettled_authorization_is_kept_for_a_day(monkeypatch):
    uid = user()
    pending_payment(uid, age=3 * 3600)
    fake = StripeFake(monkeypatch, session=session_dict(uid), pi=pi_dict(uid), capture_fails=True)
    stripe_service.reconcile()
    assert fake.cancels == [] and db.one("SELECT status FROM payments")["status"] == "pending"


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
    u = db.one("SELECT auto_reload_in_flight FROM users WHERE id = ?", (uid,))
    assert u["auto_reload_in_flight"] == 0


def test_reload_that_never_reached_stripe_expires(monkeypatch):
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


def test_reload_flag_without_a_pending_payment_is_cleared(monkeypatch):
    uid = user()
    reload_user(uid)
    StripeFake(monkeypatch)
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


def test_reload_threshold_must_be_below_the_amount(client, signed_in):
    uid, h = signed_in
    db.execute("UPDATE users SET default_payment_method = 'pm_1' WHERE id = ?", (uid,))
    body = {"enabled": True, "thresholdCents": 5000, "amountCents": 1000, "monthlyCapCents": 5000, "consent": True}
    assert client.post("/api/billing/auto-reload", json=body, headers=h).status_code == 400
    assert client.post("/api/billing/auto-reload", json={**body, "thresholdCents": 500}, headers=h).status_code == 200
    db.execute("UPDATE users SET auto_reload_threshold_micros = ? WHERE id = ?", (5000 * 10_000, uid))  # an old, bad setting
    assert stripe_service.users_needing_reload() == [] and stripe_service.run_auto_reload(uid) == "skip"


# ---------------------------------------------------------------- F13: unrecovered refunds and disputes

def test_refund_of_spent_credit_is_recorded():
    uid = refundable_user()
    ledger.debit_available(uid, 9_900_000, "usage")  # nearly all spent
    refund = {"id": "ch_1", "payment_intent": None, "amount_refunded": 1000,
              "refunds": {"data": [{"id": "re_1", "status": "succeeded", "amount": 1000}]}}
    stripe_service._handle_charge_refunded(refund)
    stripe_service._handle_charge_refunded(refund)  # delivered twice
    rows = db.all_rows("SELECT detail FROM audit_log WHERE action = 'refund_unrecovered'")
    assert [r["detail"] for r in rows] == ["p1:re_1:9900000 micros"]


def test_dispute_of_spent_credit_is_recorded():
    uid = refundable_user()
    ledger.debit_available(uid, 9_000_000, "usage")
    stripe_service._handle_dispute_created({"id": "dp_1", "charge": "ch_1", "reason": "fraudulent"})
    rows = db.all_rows("SELECT detail FROM audit_log WHERE action = 'dispute_unrecovered'")
    assert [r["detail"] for r in rows] == ["p1:dp_1:9000000 micros"]


# ---------------------------------------------------------------- F9, F11, F12, F18: job completion and prices

def running_job(uid, **kw):
    jid = make_job(uid, **kw)
    db.execute("UPDATE jobs SET status = 'running', duration_seconds = claimed_seconds WHERE id = ?", (jid,))
    d = jobs.job_dir(jid)
    (d / "out.wav").write_bytes(b"x")
    return dict(db.one("SELECT * FROM jobs WHERE id = ?", (jid,))), d


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
    from acw.worker import main as W

    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    job, d = running_job(uid)
    assert json.loads(job["params_json"])["rates"] == {"denoise": 10_000}
    monkeypatch.setitem(pricing.RATES, ("denoise", "standard"), pricing.Rate("denoise", "standard", "x", 90_000))
    W._complete(job, d, {"outputs": [{"file": "out.wav", "kind": "audio"}]})
    assert db.one("SELECT cost_micros FROM jobs WHERE id = ?", (job["id"],))["cost_micros"] == pricing.cost_micros(
        "denoise", "standard", 600, 10_000)


def test_rebuild_without_models_is_billed_as_standard(monkeypatch, tmp_path):
    from acw.config import get_settings
    from acw.worker import main as W

    monkeypatch.setenv("ACW_MODELS_DIR", str(tmp_path / "no-models"))
    get_settings.cache_clear()
    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    job, d = running_job(uid, denoise_tier="rebuild")
    assert json.loads(job["params_json"])["rates"] == {"denoise": 30_000}
    job = W._use_available_tier(job)
    params = json.loads(db.one("SELECT params_json FROM jobs WHERE id = ?", (job["id"],))["params_json"])
    assert params["denoise_tier"] == "standard" and params["rates"] == {"denoise": 10_000}
    assert "billed at its price" in params["notice"]
    W._complete(job, d, {"outputs": [{"file": "out.wav", "kind": "audio"}]})
    assert db.one("SELECT cost_micros FROM jobs WHERE id = ?", (job["id"],))["cost_micros"] == pricing.cost_micros("denoise", "standard", 600)


def test_rebuild_with_models_keeps_its_tier(monkeypatch, tmp_path):
    from acw.config import get_settings
    from acw.worker import main as W

    (tmp_path / "sidon").mkdir()
    for f in ("sidon_fe.onnx", "sidon_dec.onnx"):
        (tmp_path / "sidon" / f).write_bytes(b"x")
    monkeypatch.setenv("ACW_MODELS_DIR", str(tmp_path))
    get_settings.cache_clear()
    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    job, _ = running_job(uid, denoise_tier="rebuild")
    assert W._use_available_tier(job) is job


# ---------------------------------------------------------------- F6: limits per tier

def test_rebuild_is_limited_to_90_minutes():
    uid = user()
    ledger.credit(uid, 50_000_000, "grant", "t", "g")
    with pytest.raises(jobs.JobError, match="up to 90 minutes"):
        make_job(uid, denoise_tier="rebuild", claimed_seconds=91 * 60)
    make_job(uid, denoise_tier="rebuild", claimed_seconds=90 * 60)
    make_job(uid, denoise_tier="standard", claimed_seconds=3 * 3600)


def test_stage_limits_and_estimates_follow_the_tier():
    cpu, wall = jobs.stage_limits("rebuild", 90 * 60, 90 * 60)
    assert (cpu, wall) == (90 * 60 * 12 + 1800, 90 * 60 * 4 + 1800)
    cpu_std, wall_std = jobs.stage_limits("standard", 4 * 3600, 4 * 3600)
    assert wall_std == 4 * 3600 * 2 + 1800 and cpu_std < 4 * 3600 * 8 + 1800 + 1
    assert jobs.stage_limits("standard", 600, 3600)[1] == 600 * 3 + 1800

    def job(tier, noise="auto", channels=2):
        return {"kind": "denoise", "claimed_seconds": 3600, "duration_seconds": None, "next_stage": "probe",
                "params_json": json.dumps({"denoise_tier": tier, "transcribe_tier": None, "options": {"noise": noise},
                                           "channels": channels})}

    light, heavy, rebuild = (jobs.job_seconds(job("standard")), jobs.job_seconds(job("standard", "heavy")),
                             jobs.job_seconds(job("rebuild")))
    assert light < heavy < rebuild
    assert rebuild >= 3600  # about real time or slower
    assert jobs.job_needs(job("rebuild"))[0] == jobs.JOB_MEM_REBUILD > jobs.job_needs(job("standard"))[0]


# ---------------------------------------------------------------- F5, F7, F10, F20: live sessions

def live_session(uid, **kw):
    from acw.stream import routes

    sess = routes.Session(None, SimpleNamespace(user_id=uid), {"echo_reference": False}, 4000)
    return sess


def test_startup_bills_the_saved_seconds_of_an_interrupted_session():
    from acw.stream import routes

    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    hold = ledger.create_hold(uid, 20_000, "stream", "s1")
    ledger.create_hold(uid, 20_000, "stream", "s2")
    db.execute("INSERT INTO stream_sessions(id,user_id,denoise_tier,status,hold_id,seconds,started_at) VALUES ('s1',?,'light','active',?,90,?)",
               (uid, hold, db.now()))
    idle = ledger.create_hold(uid, 20_000, "stream", "s3")
    db.execute("INSERT INTO stream_sessions(id,user_id,denoise_tier,status,hold_id,seconds,started_at) VALUES ('s3',?,'light','active',?,0,?)",
               (uid, idle, db.now()))
    asyncio.run(routes.startup())
    s = db.one("SELECT status, end_reason, cost_micros FROM stream_sessions WHERE id = 's1'")
    assert (s["status"], s["end_reason"], s["cost_micros"]) == ("ended", "server_restart", 6000)  # 90 s of light denoise
    assert db.one("SELECT cost_micros FROM stream_sessions WHERE id = 's3'")["cost_micros"] == 0
    assert ledger.balance(uid)["balance_micros"] == 5_000_000 - 6000
    assert db.one("SELECT status FROM holds WHERE id = ?", (idle,))["status"] == "released"


def test_session_seconds_are_counted_in_whole_samples():
    s = live_session("u")
    for _ in range(3000):  # exactly 60 s of 20 ms Opus frames
        s.samples += 960
    assert s.seconds == 60 and s.micros_for(s.seconds) == 4000
    s.samples += 1
    assert s.micros_for(s.seconds) == 4067  # the 61st second has started


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
            self.sent.append(t)

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
    sess.reader = R()
    sess.writer = SimpleNamespace(close=lambda: None)
    sess.held_seconds = 300.0
    asyncio.run(routes._run(sess))
    return sess


def frame(reference=False, seq=0):
    from acw.stream import protocol as P

    return {"type": "websocket.receive", "bytes": P.HEADER.pack(P.VERSION, P.CODEC_S16, P.FLAG_REFERENCE if reference else 0, seq, 0) + bytes(960)}


def test_echo_reference_frames_do_not_keep_a_session_alive(monkeypatch):
    uid = user()
    sess = run_session(monkeypatch, [(0.05, frame(reference=True, seq=i)) for i in range(30)], uid, IDLE_TIMEOUT=0.3)
    assert sess.end_reason == "idle_timeout" and sess.seconds == 0


def test_microphone_audio_keeps_a_session_alive(monkeypatch):
    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    sess = run_session(monkeypatch, [(0.1, frame(seq=i)) for i in range(8)] + [(0.1, {"type": "websocket.disconnect"})], uid,
                       IDLE_TIMEOUT=0.3)
    assert sess.end_reason == "client_closed" and sess.seconds == pytest.approx(0.08)


def test_session_ends_at_the_wall_clock_limit(monkeypatch):
    uid = user()
    sess = run_session(monkeypatch, [(0.05, {"type": "websocket.receive", "text": json.dumps({"type": "ping"})})] * 40, uid,
                       MAX_SESSION_SECONDS=0.3)
    assert sess.end_reason == "max_duration"


def test_suspension_stops_a_running_session(monkeypatch):
    uid = user()
    db.execute("UPDATE users SET status = 'suspended' WHERE id = ?", (uid,))
    sess = run_session(monkeypatch, [(0.05, frame(seq=i)) for i in range(20)], uid, PERSIST_EVERY=0.2)
    assert sess.end_reason == "account_closed"


def test_session_progress_is_saved(monkeypatch):
    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    db.execute("INSERT INTO stream_sessions(id,user_id,status,started_at) VALUES ('s9',?,'active',?)", (uid, db.now()))
    from acw.stream import routes

    sess = live_session(uid)
    sess.id, sess.samples = "s9", 48_000 * 31
    assert asyncio.run(routes._persist(sess)) is True
    assert db.one("SELECT seconds FROM stream_sessions WHERE id = 's9'")["seconds"] == 31

