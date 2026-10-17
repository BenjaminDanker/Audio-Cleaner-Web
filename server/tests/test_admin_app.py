"""The owner dashboard (acw/admin_app.py): who may reach it, that it only reads, and that its numbers are right."""
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
ROOT = Path(__file__).resolve().parents[2]


def local_client(**kw):
    kw.setdefault("client", ("127.0.0.1", 50000))
    kw.setdefault("base_url", LOCAL)
    return TestClient(admin_app.app, **kw)


@pytest.fixture
def web():
    with local_client() as c:
        yield c


# ------------------------------------------------------------------------------------------ who may reach it

@pytest.mark.parametrize("host", ["127.0.0.1:8796", "localhost"])
def test_loopback_host_names_are_served(host):
    with local_client() as c:
        assert c.get("/api/overview", headers={"host": host}).status_code == 200


@pytest.mark.parametrize("host", [
    "evil.example", "127.0.0.1:8797", "localhost:80", "127.0.0.1.evil.example:8796", "[::1]:8796", "0.0.0.0:8796",
    "primary-ram:8796", "",
])
def test_other_host_names_are_refused_without_data(host):
    """DNS rebinding: a web page reaching this machine under its own name gets a Host we don't know."""
    with local_client() as c:
        for path in ("/", "/api/users"):
            r = c.get(path, headers={"host": host})
            assert r.status_code == 403 and r.json() == {"detail": "Not here."}, (host, path)


@pytest.mark.parametrize("ip,status", [
    ("10.0.0.5", 403), ("203.0.113.9", 403), ("::ffff:10.0.0.5", 403), ("testclient", 403),
    ("127.0.0.1", 200), ("::1", 200),
])
def test_only_loopback_clients_are_allowed(ip, status):
    with local_client(client=(ip, 4000)) as c:
        assert c.get("/api/overview").status_code == status


@pytest.mark.parametrize("value", ["203.0.113.9", "127.0.0.1"])
def test_forwarded_requests_are_refused_even_from_loopback(value):
    """Something forwarded this request (a proxy or tunnel put in front of it): never serve it."""
    with local_client() as c:
        assert c.get("/api/overview", headers={"X-Forwarded-For": value}).status_code == 403
        assert c.get("/", headers={"x-forwarded-for": value}).status_code == 403


def test_security_headers_on_every_answer(web):
    for path, status in (("/", 200), ("/api/overview", 200), ("/api/overview?start=9&end=1", 400), ("/nowhere", 404)):
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
    for path in ("/", "/api/overview", "/api/users"):
        assert web.post(path).status_code == 405
        assert web.delete(path).status_code == 405
    for path in ("/docs", "/redoc", "/openapi.json"):  # the interactive API docs are off
        assert web.get(path).status_code == 404


# ------------------------------------------------------------------------------------------ the page and /static

def test_index_and_its_three_files(web):
    r = web.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert r.content == (admin_app.UI / "index.html").read_bytes()
    assert "http://" not in r.text and "https://" not in r.text  # nothing loaded from elsewhere
    for name in ("app.js", "chart.umd.js", "style.css"):
        r = web.get(f"/static/{name}")
        assert r.status_code == 200 and r.content == (admin_app.UI / name).read_bytes() and r.content, name


@pytest.mark.parametrize("name", [
    "index.html", "App.js", "app.js%00", "..%2F..%2Fconfig.py", "%2e%2e%2fadmin_app.py", "..\\admin_app.py", "/etc/passwd",
])
def test_static_serves_only_the_three_files(web, name):
    r = web.get(f"/static/{name}")
    assert r.status_code == 404
    assert Path(admin_app.__file__).read_bytes()[:200] not in r.content


def test_static_function_refuses_anything_but_the_three_files():
    for name in ("../admin_app.py", "index.html", "", "/app.js"):
        with pytest.raises(HTTPException) as e:
            admin_app.static(name)
        assert e.value.status_code == 404


# ------------------------------------------------------------------------------------------ read-only database

def test_database_is_read_only_and_a_missing_one_is_never_created(tmp_path, monkeypatch):
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

    monkeypatch.setenv("ACW_DB_PATH", str(tmp_path / "other" / "missing.sqlite3"))
    get_settings.cache_clear()
    with pytest.raises(sqlite3.OperationalError):
        admin_app._db()
    assert not (tmp_path / "other" / "missing.sqlite3").exists()


def test_api_calls_leave_the_database_untouched(web):
    seed()
    path = get_settings().db_path
    tables = ("users", "payments", "ledger", "jobs", "stream_sessions", "alert_state")
    before = {t: db.one(f"SELECT COUNT(*) FROM {t}")[0] for t in tables}
    for path_ in ("/api/overview", "/api/series", "/api/users", "/api/errors"):
        assert web.get(path_, params={"start": S, "end": E}).status_code == 200
    assert {t: db.one(f"SELECT COUNT(*) FROM {t}")[0] for t in tables} == before and path.exists()


# ------------------------------------------------------------------------------------------ _range

@pytest.mark.parametrize("span,bucket", [
    (1, 300), (36001, 1800), (120 * 3600 + 1, H6), (30 * DAY + 1, DAY), (840 * DAY + 1, 30 * DAY),
])
def test_range_picks_a_bucket_for_a_readable_number_of_points(span, bucket):
    start, end, b = admin_app._range(1000, 1000 + span)
    assert (start, end, b) == (1000, 1000 + span, bucket)
    assert span / b <= 120 or b == 30 * DAY


def test_range_defaults_to_the_last_30_days(monkeypatch):
    monkeypatch.setattr(admin_app.time, "time", lambda: 1_900_000_000.9)
    assert admin_app._range(None, None) == (1_900_000_000 - 30 * DAY, 1_900_000_000, H6)
    assert admin_app._range(5, None)[0:2] == (5, 1_900_000_000)


@pytest.mark.parametrize("start,end", [(10, 10), (11, 10), (-1, 10)])
def test_range_refuses_empty_reversed_and_negative(start, end):
    with pytest.raises(HTTPException) as e:
        admin_app._range(start, end)
    assert e.value.status_code == 400


@pytest.mark.parametrize("path", ["overview", "series", "users", "errors"])
def test_bad_range_is_refused_on_every_endpoint(web, path):
    assert web.get(f"/api/{path}", params={"start": 200, "end": 100}).status_code == 400
    assert web.get(f"/api/{path}", params={"start": -5, "end": 100}).status_code == 400
    assert web.get(f"/api/{path}", params={"start": "abc"}).status_code == 422


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


# ------------------------------------------------------------------------------------------ overview numbers (money)

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


def test_overview_counts_users_jobs_and_live(web):
    seed()
    o = get(web, "overview")
    u = o["users"]
    # alice, bob, carol, erin, gina (dave is created at the end, frank is deleted)
    assert (u["accounts"], u["signups"], u["active"], u["spenders"]) == (5, 3, 3, 2)
    assert (u["payersInRange"], u["payersEver"], u["neverPaid"], u["liveTrialUsers"]) == (3, 3, 2, 1)
    assert o["jobs"]["byType"] == {
        "Denoising + Leveling": {"done": 1}, "Voice rebuild": {"done": 1}, "Leveling": {"done": 1},
        "Denoising + Subtitles & transcript": {"done": 1}, "Subtitles & transcript, extra accurate": {"done": 1},
        "Denoising": {"failed": 2}, "Subtitles & transcript": {"failed": 1},
    }
    assert o["jobs"]["minutesProcessed"] == 19.0  # (600 + 120 + 300 + 60 + 60) / 60, done jobs only
    assert o["jobs"]["waitP50"] == 3 and o["jobs"]["waitP90"] == 5  # waits 1, 3, 5
    live = o["live"]
    assert (live["sessions"], live["hours"], live["trialHours"]) == (6, 3.5, 0.5)  # (3600 + 1800 + 7200) / 3600
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


def test_job_without_a_pinned_rate_for_a_part_is_not_billed_for_it(web):
    """An old job (before Leveling had a price) has no leveling rate: nothing is attributed to Leveling."""
    u = add_user("old", S - DAY)
    add_job(u, "denoise", "done", {"denoise_tier": "standard", "rates": {"denoise": 6000}}, S + 5, cost=6000, duration=60)
    add_job(u, "denoise", "done", {"denoise_tier": "standard"}, S + 5, cost=7000, duration=60)  # no rates at all: "Other"
    assert get(web, "overview")["revenueByProduct"] == pytest.approx({"Other": 0.007, "Denoising": 0.006})


# ------------------------------------------------------------------------------------------ series

def test_series_points_cover_the_range_and_values_land_in_their_bucket(web):
    seed()
    s = get(web, "series")
    assert s["bucket"] == H6
    pts = s["points"]
    assert [p["t"] for p in pts] == list(range(S, E, H6))  # 40 buckets from the start to the end
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


def test_series_includes_the_machine_history(web):
    sample = {f: 1.0 for f in samples.FIELDS}
    samples.record({**sample, "cpu_busy": 2.0}, now=S + 100)
    samples.record({**sample, "cpu_busy": 4.0}, now=S + 200)
    samples.record({**sample, "cpu_busy": 8.0}, now=E)  # exclusive end
    m = get(web, "series")["machine"]
    assert len(m) == 1 and m[0]["t"] == S and m[0]["cpu_busy"] == 3.0


# ------------------------------------------------------------------------------------------ users

def test_users_are_ordered_by_spending_and_idle_accounts_are_left_out(web):
    seed()
    users = get(web, "users")["users"]
    assert [x["email"] for x in users] == ["alice@example.com", "bob@example.com", "gina@example.com", "carol@example.com"]
    a, b, g, c = users
    # alice's top-ups in range: 2000 at the start (the 1000 before the range doesn't count); 5 jobs (2 of them failed)
    assert (a["spentUsd"], a["topupsUsd"], a["jobs"], a["liveHours"], a["balanceUsd"]) == (3.0, 20.0, 5, 1.0, 5.0)
    assert (b["spentUsd"], b["topupsUsd"], b["jobs"], b["liveHours"], b["balanceUsd"]) == (1.0, 15.0, 3, 2.0, 1.5)
    assert (g["spentUsd"], g["topupsUsd"], g["jobs"]) == (0, 7.0, 0)
    assert (c["spentUsd"], c["topupsUsd"], c["jobs"], c["liveHours"]) == (0, 0, 0, 0.5)  # only live (the free trial)
    assert not {x["email"] for x in users} & {"dave@example.com", "erin@example.com", "frank@example.com"}  # no activity / deleted
    assert users[0]["since"] == S - 100 * DAY
    assert [x["email"] for x in get(web, "users", limit=2)["users"]] == ["alice@example.com", "bob@example.com"]
    assert get(web, "users", start=S - 30 * DAY, end=S - 20 * DAY)["users"] == []
    assert web.get("/api/users", params={"limit": 201}).status_code == 422


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
    assert [m["action"] for m in e["moneyProblems"]] == ["auto_reload_failed", "payment_uncredited", "refund_unrecovered"]  # newest first
    assert all(S <= m["created_at"] < E for m in e["moneyProblems"])
    assert [a["kind"] for a in e["alerts"]] == ["disk_low"] and e["alerts"][0]["active"] == 1


def test_failed_job_reason_is_the_first_sentence_and_bounded(web):
    u = add_user("x", S - DAY)
    add_job(u, "denoise", "failed", {}, S + 5, error="x" * 300 + ". tail")
    add_job(u, "denoise", "failed", {}, S + 6, error="")
    assert get(web, "errors")["failureReasons"] == {"x" * 120: 1, "unknown": 1}


# ------------------------------------------------------------------------------------------ empty database

def test_every_endpoint_answers_with_nothing_in_the_database(web):
    o = get(web, "overview")
    assert o["money"]["topupsUsd"] == 0 and o["money"]["netUsd"] == 0 and o["users"]["accounts"] == 0
    assert o["revenueByProduct"] == {} and o["jobs"] == {"byType": {}, "minutesProcessed": 0.0, "waitP50": None, "waitP90": None}
    assert o["live"]["sessions"] == 0
    assert get(web, "series")["machine"] == [] and all(p["topupsUsd"] == 0 for p in get(web, "series")["points"])
    assert get(web, "users")["users"] == []
    assert get(web, "errors") == {"failedJobs": [], "failureReasons": {}, "moneyProblems": [], "liveEndReasons": {}, "alerts": []}


# ------------------------------------------------------------------------------------------ the unit file

def test_unit_listens_on_loopback_only_and_nothing_forwards_to_the_dashboard():
    unit = (ROOT / "deploy/systemd/acw-admin.service").read_text()
    exec_start = " ".join(line.strip().rstrip("\\") for line in unit.split("ExecStart=")[1].split("\n\n")[0].splitlines()
                          if not line.startswith("Restart"))
    assert "acw.admin_app:app" in exec_start
    assert "--host 127.0.0.1" in exec_start and "--port 8796" in exec_start
    assert "0.0.0.0" not in unit and "--host ::" not in unit and "[::]" not in unit
    assert "OnFailure=acw-alert@%n.service" in unit and "NoNewPrivileges=yes" in unit

    paths = [p for p in (ROOT / "deploy/vps").rglob("*") if p.is_file()]
    paths += [ROOT / "deploy/systemd" / n for n in ("acw-tunnel.service", "acw-tunnel-bulk.service")]
    paths += [p for p in (ROOT / "deploy/systemd").glob("*tunnel*") if p.is_file()]
    assert len(paths) >= 4
    for p in set(paths):
        text = p.read_text()
        assert "8796" not in text and "acw-admin" not in text and "admin_app" not in text, p
