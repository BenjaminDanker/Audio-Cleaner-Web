"""`acw-admin top` and `acw-admin problems` (acw/admin.py): the owner dashboard's lists on the command line, over the last days."""
import json
import time

import pytest

from acw import admin, db, owner_stats

from .test_owner_stats import add_job, add_ledger, add_payment, add_session, add_user

DAY = 86400


def run(capsys, *argv):
    try:
        admin.main(list(argv))
        code = None
    except SystemExit as e:
        code = e.code
    return capsys.readouterr().out, code


@pytest.fixture
def now(monkeypatch):
    t = 1_900_000_000
    monkeypatch.setattr(admin.time, "time", lambda: t + 0.7)
    return t


def seed(now):
    a = add_user("alice", now - 100 * DAY, balance=5_000_000)
    b = add_user("bob", now - 100 * DAY, balance=1_000_000)
    add_user("idle", now - 100 * DAY)
    add_ledger(a, "usage", -3_000_000, now - 2 * DAY)
    add_ledger(b, "usage", -1_000_000, now - 10 * DAY)
    add_ledger(a, "usage", -9_000_000, now - 40 * DAY)  # older than 30 days
    add_payment(b, "succeeded", 2000, now - 3 * DAY)
    add_job(b, "denoise", "done", {"denoise_tier": "standard"}, now - 3 * DAY, cost=1000, duration=60)
    add_session(a, now - DAY, denoise="standard", seconds=7200, cost=100, reason="client_stop")
    # problems
    add_job(a, "denoise", "failed", {"denoise_tier": "standard"}, now - 2 * DAY, error="Out of memory. details")
    add_job(a, "denoise", "failed", {"denoise_tier": "standard"}, now - 20 * DAY, error="Old trouble")
    add_session(b, now - DAY, denoise="standard", reason="worker_lost")
    db.execute("INSERT INTO audit_log(actor, action, detail, created_at) VALUES ('system', 'payment_uncredited', 'p1', ?)", (now - DAY,))
    db.execute("INSERT INTO audit_log(actor, action, detail, created_at) VALUES ('system', 'auto_reload_failed', 'old', ?)", (now - 20 * DAY,))
    return a, b


def test_top_lists_the_spenders_of_the_last_30_days_by_default_as_json(capsys, now):
    seed(now)
    out, code = run(capsys, "top", "--json")
    assert code is None and "\n" not in out.strip()
    users = json.loads(out)["users"]
    assert [u["email"] for u in users] == ["alice@example.com", "bob@example.com"]  # idle is left out; the 40-day-old usage is not counted
    a, b = users
    assert (a["spentUsd"], a["liveHours"], a["jobs"], a["balanceUsd"]) == (3.0, 2.0, 2, 5.0)  # (2 jobs: both failed)
    assert (b["spentUsd"], b["topupsUsd"], b["jobs"]) == (1.0, 20.0, 1)


def test_top_days_and_limit(capsys, now):
    seed(now)
    assert [u["spentUsd"] for u in json.loads(run(capsys, "top", "--days", "60", "--json")[0])["users"]] == [12.0, 1.0]
    day = json.loads(run(capsys, "top", "--days", "1", "--json")[0])["users"]  # only alice's live hours fall in the last day
    assert [(u["email"], u["spentUsd"], u["liveHours"]) for u in day] == [("alice@example.com", 0, 2.0)]
    users = json.loads(run(capsys, "top", "--days", "60", "--limit", "1", "--json")[0])["users"]
    assert [u["email"] for u in users] == ["alice@example.com"]
    assert json.loads(run(capsys, "top", "--days", "5", "--json")[0])["users"][0]["email"] == "alice@example.com"


def test_top_without_json_is_indented_json_and_the_range_ends_now(capsys, now):
    seed(now)
    out, code = run(capsys, "top")
    assert code is None and out.startswith("{\n ") and json.loads(out)["users"][0]["email"] == "alice@example.com"
    # an activity exactly at "now" is outside (the range is [now - days, now))
    add_ledger(db.one("SELECT id FROM users WHERE email = 'bob@example.com'")["id"], "usage", -50_000_000, now + 1)
    assert json.loads(run(capsys, "top", "--json")[0])["users"][0]["email"] == "alice@example.com"


@pytest.mark.parametrize("days,limit,want_days,want_limit", [
    (0, 0, 1, 1), (-5, -1, 1, 1), (1, 1, 1, 1), (30, 25, 30, 25), (3650, 200, 3650, 200), (99999, 5000, 3650, 200),
])
def test_top_clamps_days_and_limit(capsys, monkeypatch, now, days, limit, want_days, want_limit):
    seen = []
    monkeypatch.setattr(owner_stats, "top_users", lambda start, end, n: seen.append((start, end, n)) or {"users": []})
    run(capsys, "top", "--days", str(days), "--limit", str(limit), "--json")
    assert seen == [(now - want_days * DAY, now, want_limit)]


def test_top_default_arguments_are_30_days_and_25(capsys, monkeypatch, now):
    seen = []
    monkeypatch.setattr(owner_stats, "top_users", lambda start, end, n: seen.append((start, end, n)) or {"users": []})
    run(capsys, "top", "--json")
    assert seen == [(now - 30 * DAY, now, 25)]


def test_top_with_nothing_is_an_empty_list(capsys, now):
    assert json.loads(run(capsys, "top", "--json")[0]) == {"users": []}


def test_problems_lists_the_last_7_days_by_default(capsys, now):
    seed(now)
    out, code = run(capsys, "problems", "--json")
    assert code is None
    p = json.loads(out)
    assert [j["email"] for j in p["failedJobs"]] == ["alice@example.com"]  # the 20-day-old failure is not in 7 days
    assert p["failureReasons"] == {"Out of memory": 1}
    assert p["liveEndReasons"] == {"worker_lost": 1}
    assert [m["action"] for m in p["moneyProblems"]] == ["payment_uncredited"]
    assert p["alerts"] == []
    assert set(p) == {"failedJobs", "failureReasons", "moneyProblems", "liveEndReasons", "alerts"}


def test_problems_days_widens_the_range_and_alerts_are_listed(capsys, now):
    seed(now)
    db.execute("INSERT INTO alert_state(kind, active, streak, subject) VALUES ('disk_low', 1, 3, 'Disk space is low')")
    p = json.loads(run(capsys, "problems", "--days", "30", "--json")[0])
    assert len(p["failedJobs"]) == 2 and p["failureReasons"] == {"Out of memory": 1, "Old trouble": 1}
    assert sorted(m["action"] for m in p["moneyProblems"]) == ["auto_reload_failed", "payment_uncredited"]
    assert [a["kind"] for a in p["alerts"]] == ["disk_low"]


@pytest.mark.parametrize("days,want", [(0, 1), (-3, 1), (7, 7), (3650, 3650), (10**6, 3650)])
def test_problems_clamps_days(capsys, monkeypatch, now, days, want):
    seen = []
    monkeypatch.setattr(owner_stats, "errors", lambda start, end: seen.append((start, end)) or {})
    run(capsys, "problems", "--days", str(days), "--json")
    assert seen == [(now - want * DAY, now)]


def test_problems_with_nothing_is_empty_and_without_json_is_indented(capsys, now):
    empty = {"failedJobs": [], "failureReasons": {}, "moneyProblems": [], "liveEndReasons": {}, "alerts": []}
    assert json.loads(run(capsys, "problems", "--json")[0]) == empty
    out = run(capsys, "problems")[0]
    assert out.startswith("{\n ") and json.loads(out) == empty


def test_the_reports_change_nothing(capsys, now):
    seed(now)
    tables = ("users", "ledger", "jobs", "stream_sessions", "audit_log", "payments", "meta")
    before = {t: db.one(f"SELECT COUNT(*) FROM {t}")[0] for t in tables}
    for argv in (("top", "--json"), ("problems", "--json"), ("top",), ("problems",)):
        assert run(capsys, *argv)[1] is None
    assert {t: db.one(f"SELECT COUNT(*) FROM {t}")[0] for t in tables} == before


def test_non_numbers_are_refused(capsys, now):
    assert run(capsys, "top", "--days", "x")[1] == 2
    assert run(capsys, "problems", "--days", "1.5")[1] == 2
