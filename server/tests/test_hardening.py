"""2026-10-18 hardening: what an API key may claim and spend at the worker, sign-in mail limits, the live Opus rate."""
import json
import logging

import pytest

from acw import db, jobs
from acw.billing import ledger
from acw.security import LIMITS, limiter

from .conftest import ORIGIN
from .test_api_security import make_key
from .test_billing_audit import make_job, user
from .test_live_trial import fake_worker, run_session  # noqa: F401 - a fixture


# ---------------------------------------------------------------- the worker's re-pricing of an API key's job

def _run_probe(monkeypatch, jid, measured):
    """Drive worker.process() through the probe stage with `measured` seconds; the main stage must not start."""
    from acw.worker import main as W

    ran = []
    monkeypatch.setattr(W, "_write_spec", lambda *a, **k: jobs.job_dir(jid))
    monkeypatch.setattr(W, "run_stage", lambda job, d, stage, t: ran.append(stage) or {"duration": measured, "channels": 1})
    db.execute("UPDATE jobs SET status = 'running' WHERE id = ?", (jid,))
    W.process(dict(db.one("SELECT * FROM jobs WHERE id = ?", (jid,))), {})
    return ran, db.one("SELECT status, error FROM jobs WHERE id = ?", (jid,))


def _keyed_job(cap=None, claimed=600.0):
    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    make_key(uid, "jobs")
    kid = db.one("SELECT id FROM api_keys")["id"]
    db.execute("UPDATE api_keys SET monthly_cap_micros = ? WHERE id = ?", (cap, kid))
    jid = jobs.create_job(uid, kind="denoise", denoise_tier="standard", transcribe_tier=None, filename="a.wav",
                          size_bytes=1000, claimed_seconds=claimed, options={}, api_key_id=kid)["id"]
    return uid, jid


def test_an_api_key_job_longer_than_it_claimed_fails_before_any_work_and_is_not_charged(monkeypatch):
    uid, jid = _keyed_job()
    ran, row = _run_probe(monkeypatch, jid, 600 * 1.1 + 5 + 1)
    assert ran == ["probe"] and row["status"] == "failed" and "durationSeconds" in row["error"]
    assert ledger.balance(uid)["balance_micros"] == 5_000_000 and ledger.balance(uid)["held_micros"] == 0


def test_a_browser_job_is_not_held_to_a_claim():
    from acw.worker import main as W

    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    job = dict(db.one("SELECT * FROM jobs WHERE id = ?", (make_job(uid),)))  # claimed 600 s, no key
    W._reprice(job, 3000)
    assert job["est_cost_micros"] > 0 and ledger.balance(uid)["held_micros"] == job["est_cost_micros"]


def test_the_keys_monthly_cap_holds_for_the_real_price_and_the_hold_is_released(monkeypatch):
    uid, jid = _keyed_job(cap=105_000)  # the 10-minute hold is $0.10; 11 minutes would be $0.11, within the 10% slack
    ran, row = _run_probe(monkeypatch, jid, 660)
    assert ran == ["probe"] and row["status"] == "failed" and "monthly spending limit" in row["error"]
    assert ledger.balance(uid)["balance_micros"] == 5_000_000 and ledger.balance(uid)["held_micros"] == 0
    uid2, jid2 = _keyed_job(cap=500_000)  # same job with room under the cap: the hold grows and the main stage starts
    ran, _ = _run_probe(monkeypatch, jid2, 660)
    assert ran[:2] == ["probe", "main"]


# ---------------------------------------------------------------- sign-in mail limits

@pytest.fixture
def mail(monkeypatch):
    from acw.api import auth_routes

    monkeypatch.setenv("SMTP_HOST", "smtp.test")
    monkeypatch.setenv("SMTP_FROM", "noreply@test")
    from acw.config import get_settings

    get_settings.cache_clear()
    sent = []
    monkeypatch.setattr(auth_routes, "_send_mail", lambda to, *a: sent.append(to))
    return sent


def _ask(client, email, ip="203.0.113.5"):
    return client.post("/api/auth/email", json={"email": email}, headers={"Origin": ORIGIN, "X-Forwarded-For": ip}).status_code


def test_sign_in_mails_are_limited_across_all_addresses(client, mail, monkeypatch):
    monkeypatch.setitem(LIMITS, "email_all", (2, 2 / 3600))
    assert [_ask(client, f"u{i}@example.org", f"203.0.113.{i}") for i in range(3)] == [200, 200, 429]
    assert len(mail) == 2


def test_two_ipv6_subscribers_of_one_48_share_a_bucket(client, mail, monkeypatch):
    monkeypatch.setitem(LIMITS, "email_link_wide", (2, 2 / 3600))
    limiter.reset()
    codes = [_ask(client, f"v{i}@example.org", f"2001:db8:1:{i}::1") for i in range(3)]  # three /64s, one /48
    assert codes == [200, 200, 429]
    assert _ask(client, "w@example.org", "2001:db8:2::1") == 200  # another /48


def test_a_failing_mail_send_logs_the_exception_type_not_the_address(client, mail, monkeypatch, caplog):
    from acw.api import auth_routes

    def boom(to, *a):
        raise RuntimeError(f"rejected {to}")

    monkeypatch.setattr(auth_routes, "_send_mail", boom)
    with caplog.at_level(logging.ERROR):
        assert _ask(client, "secret.person@example.org") == 503
    assert "RuntimeError" in caplog.text and "secret.person" not in caplog.text


# ---------------------------------------------------------------- live Opus rate

def test_an_opus_bitrate_outside_24_to_160_kbit_falls_back_to_96(client, live_client, signed_in, fake_worker):
    uid, _ = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    for asked in (24_000, 160_000, 23_999, 160_001, "abc", -5):
        run_session(live_client, 0.01, {"opusBitrate": asked})
    assert [s["opus_bitrate"] for s in fake_worker.started] == [24_000, 160_000, 96_000, 96_000, 96_000, 96_000]
