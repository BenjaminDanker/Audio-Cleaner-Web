"""The owner dashboard (acw/admin_app.py): who may reach it, that it only reads, and that its numbers are right."""
import hashlib
import json
import math
import sqlite3
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from acw import admin_app, db, samples
from acw.config import get_settings

DAY = 86400
H6 = 6 * 3600
S = 83_333 * H6  # a range start on a bucket edge (6 hours)
E = S + 10 * DAY
LOCAL = "http://127.0.0.1:8796"


def local_client(**kw):
    kw.setdefault("client", ("127.0.0.1", 50000))
    kw.setdefault("base_url", LOCAL)
    return TestClient(admin_app.app, **kw)


@pytest.fixture
def web():
    with local_client() as c:
        yield c


# ------------------------------------------------------------------------------------------ who may reach it

@pytest.mark.parametrize("host", ["127.0.0.1:8796", "localhost:8796", "127.0.0.1", "localhost"])
def test_loopback_hosts_are_allowed(host):
    with local_client() as c:
        r = c.get("/api/overview", headers={"host": host})
        assert r.status_code == 200, r.text


@pytest.mark.parametrize("host", [
    "evil.example", "evil.example:8796", "127.0.0.1:8797", "localhost:80", "127.0.0.1.evil.example:8796",
    "[::1]:8796", "0.0.0.0:8796", "192.168.1.5:8796", "primary-ram:8796", "",
])
def test_other_hosts_are_refused(host):
    """DNS rebinding: a web page reaching this machine under its own name gets a Host we don't know."""
    with local_client() as c:
        for path in ("/", "/api/overview", "/static/app.js"):
            r = c.get(path, headers={"host": host})
            assert r.status_code == 403, (host, path)
            assert "Not here" in r.text


@pytest.mark.parametrize("ip", ["10.0.0.5", "192.168.1.20", "203.0.113.9", "testclient", "::ffff:10.0.0.5", ""])
def test_only_loopback_clients_are_allowed(ip):
    with local_client(client=(ip, 4000)) as c:
        assert c.get("/api/overview").status_code == 403


def test_ipv6_loopback_client_is_allowed():
    with local_client(client=("::1", 4000)) as c:
        assert c.get("/api/overview").status_code == 200


@pytest.mark.parametrize("value", ["203.0.113.9", "127.0.0.1", "::1", ""])
def test_any_x_forwarded_for_is_refused(value):
    """Something forwarded this request (a proxy or tunnel put in front of it): never serve it, even for 127.0.0.1."""
    with local_client() as c:
        assert c.get("/api/overview", headers={"X-Forwarded-For": value}).status_code == 403
        assert c.get("/", headers={"x-forwarded-for": value}).status_code == 403


def test_refusals_do_not_leak_data(web):
    with local_client() as c:
        r = c.get("/api/users", headers={"host": "evil.example"})
        assert r.json() == {"detail": "Not here."}


def test_security_headers_on_every_answer(web):
    for path, status in (("/", 200), ("/api/overview", 200), ("/static/style.css", 200), ("/static/missing", 404),
                         ("/api/overview?start=9&end=1", 400), ("/nowhere", 404)):
        r = web.get(path)
        assert r.status_code == status, path
        assert r.headers["cache-control"] == "no-store", path
        assert r.headers["x-frame-options"] == "DENY", path
        assert r.headers["referrer-policy"] == "no-referrer", path
        csp = r.headers["content-security-policy"]
        assert "default-src 'self'" in csp and "frame-ancestors 'none'" in csp, path
        assert "script-src" not in csp and "unsafe-eval" not in csp and "http" not in csp, path


def test_there_are_no_routes_that_change_anything(web):
    methods = {m for route in admin_app.app.routes for m in getattr(route, "methods", None) or ()}
    assert methods <= {"GET", "HEAD"}
    for path in ("/", "/api/overview", "/api/series", "/api/users", "/api/errors"):
        assert web.post(path).status_code == 405
        assert web.delete(path).status_code == 405
    # the interactive API docs are off
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert web.get(path).status_code == 404


# ------------------------------------------------------------------------------------------ the page and /static

def test_index_and_the_three_static_files(web):
    r = web.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert r.content == (admin_app.UI / "index.html").read_bytes()
    for name in ("app.js", "chart.umd.js", "style.css"):
        r = web.get(f"/static/{name}")
        assert r.status_code == 200, name
        assert r.content == (admin_app.UI / name).read_bytes()
        assert r.content


def test_index_references_only_the_files_it_serves(web):
    html = web.get("/").text
    for name in ("app.js", "chart.umd.js", "style.css"):
        assert f"/static/{name}" in html or name in html
    assert "http://" not in html and "https://" not in html  # nothing loaded from elsewhere


@pytest.mark.parametrize("name", [
    "index.html", "chart.js-LICENSE.md", "nope.js", "App.js", "app.js%00", "../admin_app.py",
    "..%2Fadmin_app.py", "%2e%2e%2fadmin_app.py", "..%2F..%2Fconfig.py", "%2e%2e/%2e%2e/config.py", "/etc/passwd",
    "..\\admin_app.py", ".", "app.js/../../admin_app.py",
])
def test_static_serves_only_the_three_files(web, name):
    r = web.get(f"/static/{name}")
    assert r.status_code == 404  # (the HTTP client itself resolves a bare ".." or a trailing "/" before sending)
    secret = (Path(admin_app.__file__)).read_bytes()[:200]
    assert secret not in r.content


def test_static_endpoint_rejects_traversal_directly():
    for name in ("../admin_app.py", "index.html", "", "app.js ", "/app.js"):
        with pytest.raises(HTTPException) as e:
            admin_app.static(name)
        assert e.value.status_code == 404


def test_ui_directory_has_what_the_page_needs():
    for name in ("index.html", "app.js", "chart.umd.js", "style.css"):
        assert (admin_app.UI / name).is_file()


# ------------------------------------------------------------------------------------------ read-only database

def test_database_is_opened_read_only():
    db.execute("INSERT INTO meta(key, value) VALUES ('probe', 'x')")
    c = admin_app._db()
    try:
        assert c.execute("SELECT value FROM meta WHERE key = 'probe'").fetchone()[0] == "x"
        for sql in ("INSERT INTO meta(key, value) VALUES ('w', '1')", "UPDATE meta SET value = 'y'", "DELETE FROM meta",
                    "CREATE TABLE t (a)", "DROP TABLE meta", "PRAGMA user_version = 5"):
            with pytest.raises(sqlite3.OperationalError):
                c.execute(sql)
                c.commit()
    finally:
        c.close()
    assert db.one("SELECT value FROM meta WHERE key = 'probe'")["value"] == "x"


def test_database_must_exist_it_is_never_created(tmp_path, monkeypatch):
    monkeypatch.setenv("ACW_DB_PATH", str(tmp_path / "other" / "missing.sqlite3"))
    get_settings.cache_clear()
    with pytest.raises(sqlite3.OperationalError):
        admin_app._db()
    assert not (tmp_path / "other" / "missing.sqlite3").exists()


def test_api_calls_leave_the_database_untouched(web):
    seed()
    path = get_settings().db_path
    before = {t: db.one(f"SELECT COUNT(*) FROM {t}")[0] for t in ("users", "payments", "ledger", "jobs", "stream_sessions", "alert_state")}
    for path_ in ("/api/overview", "/api/series", "/api/users", "/api/errors"):
        assert web.get(path_, params={"start": S, "end": E}).status_code == 200
    after = {t: db.one(f"SELECT COUNT(*) FROM {t}")[0] for t in before}
    assert before == after and path.exists()


# ------------------------------------------------------------------------------------------ _range

@pytest.mark.parametrize("span,bucket", [
    (1, 300), (36000, 300), (36001, 1800), (60 * 3600, 1800), (60 * 3600 + 1, 3600), (120 * 3600, 3600),
    (120 * 3600 + 1, H6), (30 * DAY, H6), (30 * DAY + 1, DAY), (120 * DAY, DAY), (120 * DAY + 1, 7 * DAY),
    (840 * DAY, 7 * DAY), (840 * DAY + 1, 30 * DAY), (4000 * DAY, 30 * DAY),
])
def test_range_picks_a_bucket_for_a_readable_number_of_points(span, bucket):
    start, end, b = admin_app._range(1000, 1000 + span)
    assert (start, end, b) == (1000, 1000 + span, bucket)
    assert span / b <= 120 or b == 30 * DAY


def test_range_defaults_to_the_last_30_days(monkeypatch):
    monkeypatch.setattr(admin_app.time, "time", lambda: 1_900_000_000.9)
    assert admin_app._range(None, None) == (1_900_000_000 - 30 * DAY, 1_900_000_000, H6)
    assert admin_app._range(5, None)[0:2] == (5, 1_900_000_000)
    assert admin_app._range(None, 10_000_000)[0:2] == (10_000_000 - 30 * DAY, 10_000_000)


@pytest.mark.parametrize("start,end", [(10, 10), (11, 10), (-1, 10), (None, 0), (None, 5)])
def test_range_refuses_empty_reversed_and_negative(start, end):
    with pytest.raises(HTTPException) as e:
        admin_app._range(start, end)
    assert e.value.status_code == 400


@pytest.mark.parametrize("path", ["overview", "series", "users", "errors"])
def test_bad_range_is_400_on_every_endpoint(web, path):
    assert web.get(f"/api/{path}", params={"start": 100, "end": 100}).status_code == 400
    assert web.get(f"/api/{path}", params={"start": 200, "end": 100}).status_code == 400
    assert web.get(f"/api/{path}", params={"start": -5, "end": 100}).status_code == 400
    assert web.get(f"/api/{path}", params={"start": "abc"}).status_code == 422
    assert web.get(f"/api/{path}", params={"start": 1, "end": 100}).status_code == 200


def test_users_limit_is_bounded(web):
    assert web.get("/api/users", params={"limit": 0}).status_code == 422
    assert web.get("/api/users", params={"limit": 201}).status_code == 422
    assert web.get("/api/users", params={"limit": 200}).status_code == 200


# ------------------------------------------------------------------------------------------ seeded data

def add_user(name, created, status="active", balance=0):
    uid = db.new_id()
    db.execute("INSERT INTO users(id, email, created_at, status, balance_micros) VALUES (?,?,?,?,?)",
               (uid, f"{name}@example.com", created, status, balance))
    return uid


def add_payment(uid, status, cents, at, tax=0):
    db.execute("INSERT INTO payments(id, user_id, kind, status, amount_cents, tax_cents, created_at, updated_at)"
               " VALUES (?,?,?,?,?,?,?,?)", (db.new_id(), uid, "checkout", status, cents, tax, at, at))


def add_ledger(uid, kind, micros, at):
    db.execute("INSERT INTO ledger(user_id, amount_micros, kind, balance_after_micros, created_at) VALUES (?,?,?,?,?)",
               (uid, micros, kind, 0, at))


def add_job(uid, kind, status, params, at, *, cost=None, duration=None, claimed=60.0, error=None, wait=None):
    db.execute(
        "INSERT INTO jobs(id, user_id, kind, tier, status, params_json, input_name, input_bytes, claimed_seconds,"
        " duration_seconds, cost_micros, error, wait_seconds, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (db.new_id(), uid, kind, "x", status, json.dumps(params), "a.wav", 1, claimed, duration, cost, error, wait, at))


def add_session(uid, at, *, denoise=None, captions=None, seconds=0.0, trial=0.0, cost=0, reason=None):
    db.execute(
        "INSERT INTO stream_sessions(id, user_id, denoise_tier, caption_tier, status, seconds, trial_seconds, cost_micros,"
        " end_reason, started_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (db.new_id(), uid, denoise, captions, "ended" if reason else "active", seconds, trial, cost, reason, at))


def seed():
    ids = {}
    ids["alice"] = add_user("alice", S - 100 * DAY, balance=5_000_000)
    ids["bob"] = add_user("bob", S + 1000, balance=1_500_000)
    ids["carol"] = add_user("carol", S + 2000)
    ids["dave"] = add_user("dave", E)  # created exactly at the end: not in the range
    ids["erin"] = add_user("erin", S)  # created exactly at the start: in the range
    ids["frank"] = add_user("frank", S - 50 * DAY, status="deleted", balance=7_000_000)
    ids["gina"] = add_user("gina", S - 20 * DAY)
    u = ids

    add_payment(u["alice"], "succeeded", 1000, S - 10 * DAY, tax=80)  # before the range
    add_payment(u["alice"], "succeeded", 2000, S, tax=160)  # at the start: counts
    add_payment(u["bob"], "succeeded", 1000, S + DAY, tax=80)
    add_payment(u["bob"], "refunded", 500, S + 2 * DAY)
    add_payment(u["bob"], "failed", 700, S + 3 * DAY)
    add_payment(u["carol"], "succeeded", 9999, E)  # at the end: not in the range
    add_payment(u["gina"], "succeeded", 700, S + DAY)

    add_ledger(u["alice"], "usage", -3_000_000, S + DAY)
    add_ledger(u["bob"], "usage", -1_000_000, S + 2 * DAY)
    add_ledger(u["alice"], "usage", -5_000_000, E)  # outside
    add_ledger(u["alice"], "usage", -7_000_000, S - 1)  # outside
    add_ledger(u["bob"], "refund", -500_000, S + 3 * DAY)
    add_ledger(u["alice"], "dispute", -2_000_000, S + 4 * DAY)
    add_ledger(u["carol"], "grant", 250_000, S + DAY)
    add_ledger(u["carol"], "grant", -100_000, S + DAY)  # a grant taken back is not a grant
    add_ledger(u["alice"], "topup", 20_000_000, S)  # topups come from payments, not the ledger

    t = S + DAY
    # Denoising + Leveling: 10 minutes at 9000 and 1000 micros per minute
    add_job(u["alice"], "denoise", "done", {"denoise_tier": "standard", "rates": {"denoise": 9000, "leveling": 1000}}, t,
            cost=100_000, duration=600, wait=1)
    # Voice rebuild: 2 minutes at 30000
    add_job(u["bob"], "denoise", "done", {"denoise_tier": "rebuild", "rates": {"denoise": 30000}}, t,
            cost=60_000, duration=120, wait=3)
    # Leveling on its own: tier "off" must not be billed as noise removal even if a denoise rate were pinned
    add_job(u["bob"], "denoise", "done", {"denoise_tier": "off", "rates": {"denoise": 9000, "leveling": 1000}}, t,
            cost=5_000, duration=300)
    # Denoising + Subtitles + a translation (the translation takes what is left of the charge): 1 minute
    add_job(u["alice"], "denoise_transcribe", "done",
            {"denoise_tier": "standard", "transcribe_tier": "standard", "rates": {"denoise": 9000, "transcribe": 2000, "translate": 3000}},
            t, cost=20_000, duration=60)
    # Subtitles, extra accurate
    add_job(u["alice"], "transcribe", "done", {"transcribe_tier": "premium", "rates": {"transcribe": 4000}}, t,
            cost=4_000, duration=60)
    # failed jobs: not billed (cost None) but counted by type
    add_job(u["bob"], "denoise", "failed", {"denoise_tier": "standard"}, S + 2 * DAY, error="Out of memory. details", wait=5)
    add_job(u["alice"], "transcribe", "failed", {"transcribe_tier": "standard"}, S + 3 * DAY, error="Out of memory. again")
    add_job(u["alice"], "denoise", "failed", {"denoise_tier": "standard"}, S + 5 * DAY, error=None)
    # outside the range: ignored by everything
    add_job(u["carol"], "denoise", "done", {"denoise_tier": "standard", "rates": {"denoise": 99999}}, E, cost=99_999_999, duration=6000)
    add_job(u["carol"], "denoise", "failed", {"denoise_tier": "standard"}, S - 1, error="Old problem")

    add_session(u["alice"], t, denoise="standard", seconds=3600, cost=500_000, reason="client_stop")
    add_session(u["carol"], t, denoise="standard", seconds=1800, trial=1800, reason="trial_over")
    add_session(u["bob"], t, captions="premium", seconds=7200, cost=2_000_000, reason="error")
    add_session(u["bob"], t, denoise="standard")  # still running
    add_session(u["alice"], t, denoise="standard", reason="worker_lost")
    add_session(u["alice"], t, denoise="standard", captions="standard", cost=250_000, reason="client_closed")
    add_session(u["alice"], E, denoise="standard", seconds=36000, cost=9_000_000, reason="error")  # outside

    db.audit("system", "refund_unrecovered", u["bob"], "r1")
    db.execute("UPDATE audit_log SET created_at = ? WHERE action = 'refund_unrecovered'", (S + 2 * DAY,))
    for action, at in (("payment_uncredited", S + 3 * DAY), ("auto_reload_failed", S + 4 * DAY), ("payment_uncredited", E),
                       ("login", S + DAY), ("payment_refused_region", S - 1)):
        db.execute("INSERT INTO audit_log(actor, action, detail, created_at) VALUES ('system', ?, 'd', ?)", (action, at))
    db.execute("INSERT INTO alert_state(kind, active, streak, subject) VALUES ('disk_low', 1, 3, 'Disk space is low')")
    return ids


def fee(cents):
    return math.ceil(cents * admin_app.FEE_RATE + admin_app.FEE_FIXED_CENTS)


def get(web, path, start=S, end=E, **params):
    r = web.get(f"/api/{path}", params={"start": start, "end": end, **params})
    assert r.status_code == 200, r.text
    return r.json()


def test_overview_money(web):
    seed()
    m = get(web, "overview")["money"]
    assert fee(2000) == 88 and fee(1000) == 59 and fee(500) == 45 and fee(700) == 51  # (spelled out)
    fees = 88 + 59 + 45 + 51
    assert m["topupsUsd"] == 42.0  # 2000 + 1000 + 500 (refunded still counts) + 700, in cents
    assert m["payments"] == 4
    assert m["taxUsd"] == 2.4  # 160 + 80 (alice's earlier payment is outside)
    assert m["feesUsdEstimate"] == fees / 100
    assert m["refundsUsd"] == 0.5
    assert m["disputesUsd"] == 2.0
    assert m["usageUsd"] == 4.0
    assert m["grantsUsd"] == 0.25
    assert m["failedPayments"] == 1
    assert m["netUsd"] == pytest.approx(42.0 - 0.5 - 2.0 - fees / 100)
    assert m["outstandingCreditUsd"] == 6.5  # deleted accounts hold nothing that counts


def test_overview_users(web):
    seed()
    u = get(web, "overview")["users"]
    # alice, bob, carol, erin, gina (dave is created at the end, frank is deleted)
    assert u["accounts"] == 5
    assert u["signups"] == 3  # bob, carol, erin (erin at the very start; dave at the very end is out)
    assert u["active"] == 3  # alice, bob (jobs and sessions), carol (a live session)
    assert u["spenders"] == 2
    assert u["payersInRange"] == 3  # alice, bob, gina
    assert u["payersEver"] == 3
    assert u["neverPaid"] == 2  # carol, erin
    assert u["liveTrialUsers"] == 1


def test_overview_revenue_by_product_follows_the_pinned_rates(web):
    seed()
    o = get(web, "overview")
    rev = o["revenueByProduct"]
    assert rev == pytest.approx({
        "Live: Captions, extra accurate": 2.0,
        "Live: Noise removal": 0.5,
        "Live: Noise removal + Captions": 0.25,
        "Denoising": 0.099,  # 90000 + 9000
        "Voice rebuild": 0.06,
        "Leveling": 0.015,  # 10000 from the first job, 5000 from the "off" job
        "Translated subtitles": 0.009,  # 20000 - 9000 - 2000
        "Subtitles, extra accurate": 0.004,
        "Subtitles": 0.002,
    })
    assert list(rev) == sorted(rev, key=lambda k: -rev[k])  # biggest first
    assert "Other" not in rev and "Noise removal" not in rev
    assert o["usersByProduct"] == {"Live: Noise removal": 1, "Live: Captions, extra accurate": 1, "Live: Noise removal + Captions": 1,
                                   "Denoising": 1, "Leveling": 2, "Voice rebuild": 1, "Subtitles": 1, "Subtitles, extra accurate": 1,
                                   "Translated subtitles": 1}
    assert list(o["usersByProduct"].values()) == sorted(o["usersByProduct"].values(), reverse=True)


def test_overview_jobs_and_live(web):
    seed()
    o = get(web, "overview")
    assert o["jobs"]["byType"] == {
        "Denoising + Leveling": {"done": 1},
        "Voice rebuild": {"done": 1},
        "Leveling": {"done": 1},
        "Denoising + Subtitles & transcript": {"done": 1},
        "Subtitles & transcript, extra accurate": {"done": 1},
        "Denoising": {"failed": 2},
        "Subtitles & transcript": {"failed": 1},
    }
    assert o["jobs"]["minutesProcessed"] == 19.0  # (600 + 120 + 300 + 60 + 60) / 60, done jobs only
    assert o["jobs"]["waitP50"] == 3 and o["jobs"]["waitP90"] == 5  # waits 1, 3, 5
    live = o["live"]
    assert live["sessions"] == 6
    assert live["hours"] == 3.5  # (3600 + 1800 + 7200) / 3600
    assert live["trialHours"] == 0.5
    assert live["endReasons"] == {"client_stop": 1, "trial_over": 1, "error": 1, "still running": 1, "worker_lost": 1, "client_closed": 1}
    assert o["range"] == {"start": S, "end": E}


def test_overview_range_boundaries(web):
    """start is inclusive, end is exclusive, for every kind of row."""
    u = add_user("zed", S - DAY)
    add_user("edge_in", S)
    add_user("edge_out", E)
    add_payment(u, "succeeded", 1000, S)
    add_payment(u, "succeeded", 2000, E)
    add_ledger(u, "usage", -1_000_000, S)
    add_ledger(u, "usage", -2_000_000, E)
    add_job(u, "denoise", "done", {"denoise_tier": "standard", "rates": {"denoise": 6000}}, S, cost=6000, duration=60)
    add_job(u, "denoise", "done", {"denoise_tier": "standard", "rates": {"denoise": 6000}}, E, cost=6000, duration=60)
    add_session(u, S, seconds=3600, reason="client_stop")
    add_session(u, E, seconds=3600, reason="client_stop")
    o = get(web, "overview")
    assert (o["money"]["topupsUsd"], o["money"]["usageUsd"], o["money"]["payments"]) == (10.0, 1.0, 1)
    assert o["users"]["signups"] == 1 and o["users"]["accounts"] == 2  # zed and edge_in (edge_out is created at the end)
    assert o["jobs"]["byType"] == {"Denoising": {"done": 1}}
    assert o["live"]["sessions"] == 1 and o["live"]["hours"] == 1.0
    # shifting the range by one second swaps what is in
    o2 = get(web, "overview", start=S + 1, end=E + 1)
    assert (o2["money"]["topupsUsd"], o2["money"]["usageUsd"], o2["money"]["payments"]) == (20.0, 2.0, 1)
    assert o2["jobs"]["byType"] == {"Denoising": {"done": 1}} and o2["live"]["sessions"] == 1


def test_overview_with_nothing_in_it(web):
    o = get(web, "overview")
    assert o["money"]["topupsUsd"] == 0 and o["money"]["netUsd"] == 0
    assert o["users"]["accounts"] == 0 and o["users"]["neverPaid"] == 0
    assert o["revenueByProduct"] == {} and o["usersByProduct"] == {}
    assert o["jobs"] == {"byType": {}, "minutesProcessed": 0.0, "waitP50": None, "waitP90": None}
    assert o["live"]["sessions"] == 0 and o["live"]["endReasons"] == {}


def test_overview_default_range_is_the_last_30_days(web):
    uid = add_user("x", 1)
    now = db.now()
    add_payment(uid, "succeeded", 1000, now - 29 * DAY)
    add_payment(uid, "succeeded", 5000, now - 31 * DAY)
    r = web.get("/api/overview").json()
    assert r["money"]["topupsUsd"] == 10.0
    assert r["range"]["end"] - r["range"]["start"] == 30 * DAY


def test_job_without_a_pinned_rate_for_a_part_is_not_billed_for_it(web):
    """An old job (before Leveling had a price) has no leveling rate: nothing is attributed to Leveling."""
    u = add_user("old", S - DAY)
    add_job(u, "denoise", "done", {"denoise_tier": "standard", "rates": {"denoise": 6000}}, S + 5, cost=6000, duration=60)
    add_job(u, "denoise", "done", {"denoise_tier": "standard"}, S + 5, cost=7000, duration=60)  # no rates at all: "Other"
    rev = get(web, "overview")["revenueByProduct"]
    assert rev == pytest.approx({"Other": 0.007, "Denoising": 0.006})


# ------------------------------------------------------------------------------------------ series

def test_series_points_cover_the_range_and_values_land_in_their_bucket(web):
    seed()
    s = get(web, "series")
    assert s["bucket"] == H6
    pts = s["points"]
    assert len(pts) == 40 and pts[0]["t"] == S and pts[-1]["t"] == E - H6
    assert [p["t"] for p in pts] == list(range(S, E, H6))
    by = {p["t"]: p for p in pts}
    day1 = S + DAY  # a bucket edge
    assert by[S]["topupsUsd"] == 20.0  # alice's payment, at the very start
    assert by[S]["signups"] == 3  # erin (S), bob (S+1000) and carol (S+2000) share the first bucket
    assert by[day1]["topupsUsd"] == 17.0  # bob 1000 + gina 700
    assert by[day1]["usageUsd"] == 3.0
    assert by[day1]["jobsDone"] == 5 and by[day1]["jobsFailed"] == 0
    assert by[day1]["activeUsers"] == 3  # alice, bob, carol
    assert by[day1]["liveHours"] == 3.5
    assert by[S + 2 * DAY]["topupsUsd"] == 5.0 and by[S + 2 * DAY]["usageUsd"] == 1.0 and by[S + 2 * DAY]["jobsFailed"] == 1
    assert by[S + 3 * DAY]["refundsUsd"] == 0.5 and by[S + 4 * DAY]["refundsUsd"] == 2.0  # refund and dispute are both "refunds"
    assert by[S + 5 * DAY]["jobsFailed"] == 1
    # totals add up to the overview's
    assert sum(p["topupsUsd"] for p in pts) == 42.0
    assert sum(p["usageUsd"] for p in pts) == 4.0
    assert sum(p["jobsDone"] for p in pts) == 5 and sum(p["jobsFailed"] for p in pts) == 3
    assert sum(p["liveHours"] for p in pts) == pytest.approx(3.5)
    assert sum(p["signups"] for p in pts) == 3
    # buckets without anything are zero, not missing
    assert by[S + 7 * DAY] == {"t": S + 7 * DAY, "topupsUsd": 0, "usageUsd": 0, "refundsUsd": 0, "signups": 0,
                               "activeUsers": 0, "jobsDone": 0, "jobsFailed": 0, "liveHours": 0}


def test_series_with_a_start_between_bucket_edges_begins_at_the_edge_before(web):
    uid = add_user("x", S - DAY)
    add_payment(uid, "succeeded", 1000, S + 100)
    s = get(web, "series", start=S + 100, end=E)
    assert s["points"][0]["t"] == S and s["points"][0]["topupsUsd"] == 10.0
    assert s["points"][-1]["t"] < E <= s["points"][-1]["t"] + s["bucket"]
    # the row just before start is not counted even though it falls in the first bucket
    add_payment(uid, "succeeded", 5000, S + 99)
    assert get(web, "series", start=S + 100, end=E)["points"][0]["topupsUsd"] == 10.0


def test_series_bucket_sizes_follow_range(web):
    for span, bucket in ((10 * 3600, 300), (3 * DAY, 3600), (100 * DAY, DAY), (200 * DAY, 7 * DAY)):
        s = get(web, "series", start=1_000_000, end=1_000_000 + span)
        assert s["bucket"] == bucket
        assert len(s["points"]) <= 121 and s["points"][0]["t"] <= 1_000_000
        assert all(b["t"] - a["t"] == bucket for a, b in zip(s["points"], s["points"][1:]))
        assert s["points"][-1]["t"] + bucket >= 1_000_000 + span


def test_series_includes_the_machine_history(web):
    sample = {f: 1.0 for f in samples.FIELDS}
    samples.record({**sample, "cpu_busy": 2.0}, now=S + 100)
    samples.record({**sample, "cpu_busy": 4.0}, now=S + 200)
    samples.record({**sample, "cpu_busy": 8.0}, now=E)  # exclusive end
    m = get(web, "series")["machine"]
    assert len(m) == 1 and m[0]["t"] == S and m[0]["cpu_busy"] == 3.0


def test_series_without_machine_history_is_empty_there(web):
    assert not samples.path().exists()
    assert get(web, "series")["machine"] == []


# ------------------------------------------------------------------------------------------ users

def test_users_are_ordered_by_spending_then_top_ups_and_idle_accounts_are_left_out(web):
    seed()
    users = get(web, "users")["users"]
    assert [x["email"] for x in users] == ["alice@example.com", "bob@example.com", "gina@example.com", "carol@example.com"]
    a, b, g, c = users
    # alice's top-ups in range: 2000 at the start (the 1000 before the range doesn't count); 5 jobs (2 of them failed)
    assert (a["spentUsd"], a["topupsUsd"], a["jobs"], a["liveHours"], a["balanceUsd"]) == (3.0, 20.0, 5, 1.0, 5.0)
    assert (b["spentUsd"], b["topupsUsd"], b["jobs"], b["liveHours"], b["balanceUsd"]) == (1.0, 15.0, 3, 2.0, 1.5)
    assert (g["spentUsd"], g["topupsUsd"], g["jobs"]) == (0, 7.0, 0)
    assert (c["spentUsd"], c["topupsUsd"], c["jobs"], c["liveHours"]) == (0, 0, 0, 0.5)  # only live (the free trial)
    emails = {x["email"] for x in users}
    assert not emails & {"dave@example.com", "erin@example.com", "frank@example.com"}  # no activity / deleted
    assert users[0]["since"] == S - 100 * DAY
    assert [x["email"] for x in get(web, "users", limit=2)["users"]] == ["alice@example.com", "bob@example.com"]


def test_users_limit_applies_before_idle_accounts_are_dropped(web):
    seed()
    # ordering puts accounts with no activity after the active ones, so a short limit still lists the active ones
    assert len(get(web, "users", limit=4)["users"]) == 4


def test_users_outside_the_range_are_empty(web):
    seed()
    assert get(web, "users", start=S - 30 * DAY, end=S - 20 * DAY)["users"] == []


# ------------------------------------------------------------------------------------------ errors

def test_errors(web):
    seed()
    e = get(web, "errors")
    jobs_ = e["failedJobs"]
    assert len(jobs_) == 3  # the one before the range is not listed
    assert [j["created_at"] for j in jobs_] == sorted((j["created_at"] for j in jobs_), reverse=True)  # newest first
    assert jobs_[0]["email"] == "alice@example.com" and jobs_[0]["error"] is None and jobs_[0]["kind"] == "denoise"
    assert {j["email"] for j in jobs_} == {"alice@example.com", "bob@example.com"}
    assert e["failureReasons"] == {"Out of memory": 2, "unknown": 1}
    assert list(e["failureReasons"])[0] == "Out of memory"
    assert e["liveEndReasons"] == {"error": 1, "worker_lost": 1}  # normal ends, still-running and out-of-range sessions are not errors
    assert [m["action"] for m in e["moneyProblems"]] == ["auto_reload_failed", "payment_uncredited", "refund_unrecovered"]  # newest row first
    assert all(S <= m["created_at"] < E for m in e["moneyProblems"])
    assert [a["kind"] for a in e["alerts"]] == ["disk_low"] and e["alerts"][0]["active"] == 1


def test_live_end_reasons_exclude_exactly_the_normal_ones(web):
    u = add_user("x", S - DAY)
    for reason in admin_app.NORMAL_LIVE_ENDS:
        add_session(u, S + 10, reason=reason)
    for reason in ("error", "full", "NORMAL_LIVE_ENDS", "worker_lost", "error"):
        add_session(u, S + 10, reason=reason)
    add_session(u, S + 10)  # still running
    assert get(web, "errors")["liveEndReasons"] == {"error": 2, "full": 1, "NORMAL_LIVE_ENDS": 1, "worker_lost": 1}
    assert set(admin_app.NORMAL_LIVE_ENDS) == {"client_stop", "client_closed", "idle_timeout", "max_duration", "no_audio", "replaced", "trial_over"}


def test_errors_with_nothing_wrong(web):
    e = get(web, "errors")
    assert e == {"failedJobs": [], "failureReasons": {}, "moneyProblems": [], "liveEndReasons": {}, "alerts": []}


def test_failed_job_reason_is_the_first_sentence_and_bounded(web):
    u = add_user("x", S - DAY)
    add_job(u, "denoise", "failed", {}, S + 5, error="x" * 300 + ". tail")
    add_job(u, "denoise", "failed", {}, S + 6, error="")
    r = get(web, "errors")["failureReasons"]
    assert r == {"x" * 120: 1, "unknown": 1}


def test_errors_without_the_alert_table(web):
    db.execute("DROP TABLE alert_state")
    assert get(web, "errors")["alerts"] == []


# ------------------------------------------------------------------------------------------ the unit and the deploy

ROOT = Path(__file__).resolve().parents[2]


def test_unit_listens_on_loopback_only():
    unit = (ROOT / "deploy/systemd/acw-admin.service").read_text()
    exec_start = " ".join(line.strip().rstrip("\\") for line in unit.split("ExecStart=")[1].split("\n\n")[0].splitlines()
                          if not line.startswith("Restart"))
    assert "acw.admin_app:app" in exec_start
    assert "--host 127.0.0.1" in exec_start and "--port 8796" in exec_start
    assert "0.0.0.0" not in unit and "--host ::" not in unit and "[::]" not in unit
    assert admin_app.PORT == 8796
    assert "OnFailure=acw-alert@%n.service" in unit
    assert "NoNewPrivileges=yes" in unit


def test_installer_deploys_and_restarts_the_dashboard():
    sh = (ROOT / "deploy/install.sh").read_text()
    units = next(line for line in sh.splitlines() if line.startswith("UNITS="))
    assert "acw-admin.service" in units.split()[0:20] or "acw-admin.service" in units
    assert (ROOT / "deploy/systemd/acw-admin.service").is_file()
    block = sh.split('if [ "$APP_CHANGED" = 1 ] || [ "$LOCK_CHANGED" = 1 ]; then', 1)[1].split("\nfi", 1)[0]
    assert "add_restart acw-admin.service" in block
    assert "add_restart acw-web.service" in block


def test_nothing_forwards_to_the_dashboard():
    paths = [p for p in (ROOT / "deploy/vps").rglob("*") if p.is_file()]
    paths += [ROOT / "deploy/systemd" / n for n in ("acw-tunnel.service", "acw-tunnel-bulk.service")]
    paths += [p for p in (ROOT / "deploy/systemd").glob("*tunnel*") if p.is_file()]
    assert len(paths) >= 4
    for p in set(paths):
        text = p.read_text()
        assert "8796" not in text, p
        assert "acw-admin" not in text, p
        assert "admin_app" not in text, p
