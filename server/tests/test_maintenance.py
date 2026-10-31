"""Maintenance (acw/maintenance.py): the switch, who is an admin, which queued jobs may start, how a held job is shown, the worker's claim,
the live handshake and the `acw-admin maintenance` command. Only admin accounts' work runs while it is on."""
import json

import pytest

from acw import admin, auth, db, job_view, jobs, maintenance, security
from acw.billing import ledger
from acw.config import get_settings
from acw.worker import main as worker

from .job_support import roomy_disk  # noqa: F401 - fixture
from .test_billing_audit import make_job
from .test_ops_audit import api_key, ws_denied


pytestmark = pytest.mark.usefixtures("roomy_disk")


@pytest.fixture(autouse=True)
def many_active_jobs(monkeypatch):
    monkeypatch.setenv("ACW_MAX_ACTIVE_JOBS", "20")
    get_settings.cache_clear()


def run(capsys, *argv):
    try:
        admin.main(list(argv))
        code = None
    except SystemExit as e:
        code = e.code
    return capsys.readouterr().out, code


def user(email, *, is_admin=0) -> str:
    uid = auth.get_or_create_user(email)["id"]
    db.execute("UPDATE users SET is_admin = ? WHERE id = ?", (is_admin, uid))
    return uid


def admins_env(monkeypatch, *emails):
    monkeypatch.setenv("ACW_ADMIN_EMAILS", ",".join(emails))
    get_settings.cache_clear()


def queued(uid, at, *, local=False) -> str:
    """A queued, not yet started job of the account (queued_at orders the queue); `local`: sent from this machine itself."""
    ledger.credit(uid, 1_000_000, "grant", "t", f"g-{db.new_id()}")
    jid = make_job(uid)
    db.execute("UPDATE jobs SET status = 'queued', queued_at = ?, started_at = NULL, local_origin = ? WHERE id = ?", (at, 1 if local else 0, jid))
    return jid


def row(jid) -> dict:
    return dict(db.one("SELECT * FROM jobs WHERE id = ?", (jid,)))


def meta_raw():
    r = db.one("SELECT value FROM meta WHERE key = ?", (maintenance.KEY,))
    return None if r is None else r["value"]


def actions():
    return [(r["actor"], r["action"], r["detail"]) for r in db.all_rows("SELECT actor, action, detail FROM audit_log WHERE action LIKE 'maintenance%' ORDER BY id")]


# ------------------------------------------------------------------------------------------ the switch

def test_off_by_default_and_a_round_trip():
    assert maintenance.state() is None
    assert maintenance.switch_off("x") is False and actions() == []  # nothing to switch off: no audit row
    v = maintenance.switch_on("database move", "owner")
    assert v["reason"] == "database move" and isinstance(v["since"], int)
    assert maintenance.state() == v
    assert json.loads(meta_raw()) == v
    assert maintenance.switch_off("owner") is True
    assert maintenance.state() is None and meta_raw() is None
    assert actions() == [("owner", "maintenance_on", "database move"), ("owner", "maintenance_off", None)]


def test_switching_on_again_keeps_the_first_start_and_replaces_the_reason(monkeypatch):
    t = iter([1000, 2000, 3000, 4000])
    monkeypatch.setattr(db, "now", lambda: next(t))
    maintenance.switch_on("first")
    maintenance.switch_on("second")
    assert maintenance.state() == {"since": 1000, "reason": "second"}


def test_the_reason_is_cut_and_an_empty_one_is_audited_without_detail():
    maintenance.switch_on("x" * 500)
    assert len(maintenance.state()["reason"]) == 200
    maintenance.switch_off()
    maintenance.switch_on()
    assert maintenance.state()["reason"] == ""
    assert actions()[-1] == ("admin", "maintenance_on", None)


@pytest.mark.parametrize("garbage", ["not json", "", "null", "[1, 2]", "5", '"on"', "{"])
def test_a_garbage_meta_value_still_means_on_with_no_details(garbage):
    """The row exists, so maintenance is on: failing open would let everyone's work through by accident."""
    db.execute("INSERT INTO meta(key, value) VALUES (?, ?)", (maintenance.KEY, garbage))
    assert maintenance.state() == {}
    assert maintenance.blocks_user(user("someone@example.com")) is True
    assert maintenance.switch_off() is True and maintenance.state() is None


def test_switching_on_over_a_garbage_value_repairs_it():
    db.execute("INSERT INTO meta(key, value) VALUES (?, ?)", (maintenance.KEY, "garbage"))
    v = maintenance.switch_on("again")
    assert v["reason"] == "again" and "since" in v and maintenance.state() == v


# ------------------------------------------------------------------------------------------ admin accounts

def test_admin_ids_are_the_marked_accounts_and_the_listed_emails_case_insensitively(monkeypatch):
    marked = user("marked@example.com", is_admin=1)
    plain = user("plain@example.com")
    listed = user("Listed@Example.com")  # stored lower-cased by sign-in, but the match must not depend on it
    db.execute("UPDATE users SET email = 'MiXed@Example.com' WHERE id = ?", (listed,))
    assert maintenance.admin_ids() == {marked}
    admins_env(monkeypatch, "mixed@example.com")
    assert maintenance.admin_ids() == {marked, listed}
    assert plain not in maintenance.admin_ids()
    admins_env(monkeypatch, "mixed@example.com", "plain@example.com", "nobody@example.com")
    assert maintenance.admin_ids() == {marked, listed, plain}


def test_admin_ids_with_no_accounts_or_no_list_is_empty(monkeypatch):
    assert maintenance.admin_ids() == set()
    monkeypatch.delenv("ACW_ADMIN_EMAILS", raising=False)
    get_settings.cache_clear()
    user("a@example.com")
    assert maintenance.admin_ids() == set()


def test_blocks_user_only_while_on_and_never_an_admin(monkeypatch):
    admin_uid, other = user("boss@example.com", is_admin=1), user("other@example.com")
    assert not maintenance.blocks_user(admin_uid) and not maintenance.blocks_user(other)
    maintenance.switch_on("x")
    assert maintenance.blocks_user(other) and not maintenance.blocks_user(admin_uid)
    assert maintenance.blocks_user("no-such-account")  # (an unknown id is not an admin)
    admins_env(monkeypatch, "other@example.com")  # listed, though not yet marked at sign-in
    assert not maintenance.blocks_user(other)
    maintenance.switch_off()
    assert not maintenance.blocks_user("no-such-account")


# ------------------------------------------------------------------------------------------ runnable / paused

def _rows():
    a, b = user("admin@example.com", is_admin=1), user("b@example.com")
    ids = {
        "b_new": queued(b, 1),
        "a_new": queued(a, 2),
        "b_started": queued(b, 3),
        "b_running": queued(b, 4),
        "b_later": queued(b, 5),
    }
    db.execute("UPDATE jobs SET started_at = ? WHERE id = ?", (db.now(), ids["b_started"]))  # a later stage waiting again
    db.execute("UPDATE jobs SET status = 'running' WHERE id = ?", (ids["b_running"],))
    rows = [row(i) for i in ids.values()]
    return ids, rows


def test_runnable_returns_everything_while_off():
    ids, rows = _rows()
    assert maintenance.runnable(rows) == rows
    assert maintenance.runnable([]) == []


def test_runnable_keeps_admins_and_started_jobs_and_drops_the_others_unstarted_ones_while_on():
    ids, rows = _rows()
    maintenance.switch_on("x")
    got = [r["id"] for r in maintenance.runnable(rows)]
    assert got == [ids["a_new"], ids["b_started"], ids["b_running"]]  # order kept
    assert ids["b_new"] not in got and ids["b_later"] not in got
    assert maintenance.runnable([]) == []
    maintenance.switch_off()
    assert maintenance.runnable(rows) == rows


def test_runnable_follows_the_admin_email_list(monkeypatch):
    ids, rows = _rows()
    maintenance.switch_on("x")
    admins_env(monkeypatch, "B@Example.com")
    assert [r["id"] for r in maintenance.runnable(rows)] == [r["id"] for r in rows]


def test_runnable_does_not_change_the_rows_it_is_given():
    ids, rows = _rows()
    copy = [dict(r) for r in rows]
    maintenance.switch_on("x")
    maintenance.runnable(rows)
    assert rows == copy


def test_paused_is_a_queued_unstarted_job_of_a_non_admin_while_on():
    ids, rows = _rows()
    by = dict(zip(ids, rows))
    assert not any(maintenance.paused(r) for r in rows)  # off
    maintenance.switch_on("x")
    assert {k for k, r in by.items() if maintenance.paused(r)} == {"b_new", "b_later"}
    assert not maintenance.paused({"status": "done", "started_at": None, "user_id": "someone"})
    assert not maintenance.paused({"status": "queued", "started_at": 5, "user_id": "someone"})
    assert maintenance.paused({"status": "queued", "started_at": None, "user_id": "someone"})


# ------------------------------------------------------------------------------------------ the worker's claim

def test_claim_skips_a_non_admins_new_job_for_an_admins_behind_it_and_takes_all_after_switch_off(roomy_disk):
    a, b = user("admin@example.com", is_admin=1), user("b@example.com")
    first = queued(b, 1)  # first in line
    second = queued(a, 2)
    third = queued(b, 3)
    maintenance.switch_on("x")
    job = worker.claim("w1")
    assert job is not None and job["id"] == second  # the admin's job behind the other's is taken
    assert row(second)["status"] == "running"
    assert worker.claim("w2") is None  # nothing else may start
    assert row(first)["status"] == "queued" and row(third)["status"] == "queued"
    maintenance.switch_off()
    assert worker.claim("w2")["id"] == first  # first come, first served again
    assert worker.claim("w3")["id"] == third
    assert worker.claim("w4") is None


def test_claim_still_takes_a_started_job_of_a_non_admin_during_maintenance(roomy_disk):
    b = user("b@example.com")
    jid = queued(b, 1)
    db.execute("UPDATE jobs SET started_at = ? WHERE id = ?", (db.now(), jid))
    maintenance.switch_on("x")
    assert worker.claim("w1")["id"] == jid


def test_claim_with_only_held_jobs_and_maintenance_on_returns_none(roomy_disk):
    jid = queued(user("b@example.com"), 1)
    maintenance.switch_on("x")
    assert worker.claim("w1") is None and row(jid)["status"] == "queued"


# ------------------------------------------------------------------------------------------ the job as shown

def _shown(jid):
    return job_view.public_job(row(jid))


def test_a_held_job_is_shown_as_paused_with_no_wait_estimate(monkeypatch):
    monkeypatch.setattr(jobs, "queue_snapshot", lambda: ([jid], {jid: 120}))
    jid = queued(user("b@example.com"), 1)
    before = _shown(jid)
    assert "paused" not in before and before["queuePosition"] == 1 and before["queueWaitSeconds"] == 120
    maintenance.switch_on("x")
    shown = _shown(jid)
    assert shown["paused"] == maintenance.JOB_MESSAGE and shown["queueWaitSeconds"] is None and shown["queuePosition"] == 1
    maintenance.switch_off()
    assert "paused" not in _shown(jid) and _shown(jid)["queueWaitSeconds"] == 120


def test_an_admins_started_and_finished_jobs_are_not_shown_as_paused(monkeypatch):
    a, b = user("admin@example.com", is_admin=1), user("b@example.com")
    ja, jstarted, jdone = queued(a, 1), queued(b, 2), queued(b, 3)
    db.execute("UPDATE jobs SET started_at = ? WHERE id = ?", (db.now(), jstarted))
    db.execute("UPDATE jobs SET status = 'done' WHERE id = ?", (jdone,))
    monkeypatch.setattr(jobs, "queue_snapshot", lambda: ([ja, jstarted], {ja: 10, jstarted: 20}))
    maintenance.switch_on("x")
    for jid, eta in ((ja, 10), (jstarted, 20)):
        s = _shown(jid)
        assert "paused" not in s and s["queueWaitSeconds"] == eta
    s = _shown(jdone)
    assert "paused" not in s and s["queuePosition"] is None


def test_a_job_that_is_not_in_the_queue_snapshot_is_not_shown_as_paused(monkeypatch):
    jid = queued(user("b@example.com"), 1)
    monkeypatch.setattr(jobs, "queue_snapshot", lambda: ([], {}))
    maintenance.switch_on("x")
    s = _shown(jid)
    assert "paused" not in s and s["queuePosition"] is None


def test_the_api_shows_the_pause_to_the_owner_of_the_job(client, signed_in):
    uid, h = signed_in
    jid = queued(uid, 1)
    maintenance.switch_on("x")
    jobs.queue_snapshot.__globals__["_queue_cache"]["at"] = -1e9  # (the cached queue is from before)
    shown = client.get(f"/api/jobs/{jid}").json()
    assert shown["paused"] == maintenance.JOB_MESSAGE and shown["queueWaitSeconds"] is None


# ------------------------------------------------------------------------------------------ the live handshake

def test_live_handshake_refuses_a_blocked_user_with_the_maintenance_code_and_lets_an_admin_through(client, live_client, monkeypatch):
    from acw.stream import routes

    other, boss = user("other@example.com"), user("boss@example.com", is_admin=1)
    k_other, k_boss = api_key(other), api_key(boss)
    maintenance.switch_on("x")
    status, event = ws_denied(live_client, {"Authorization": f"Bearer {k_other}"})
    assert status is None
    assert event["type"] == "error" and event["code"] == "maintenance" and event["message"] == maintenance.MESSAGE
    assert event["retryAfterSeconds"] == 300
    # an admin is not refused for maintenance (the next check is stubbed to a different, recognisable refusal)
    monkeypatch.setattr(routes, "MAX_SESSIONS_PER_USER", 0)
    status, event = ws_denied(live_client, {"Authorization": f"Bearer {k_boss}"})
    assert status is None and event["code"] == "too_many_sessions"
    # after maintenance the other account gets past the check too
    maintenance.switch_off()
    status, event = ws_denied(live_client, {"Authorization": f"Bearer {k_other}"})
    assert event["code"] != "maintenance"


def test_the_maintenance_check_comes_after_the_drain_check(client, live_client):
    other = user("other@example.com")
    key = api_key(other)
    maintenance.switch_on("x")
    (get_settings().data_dir / "drain").touch()
    assert ws_denied(live_client, {"Authorization": f"Bearer {key}"})[1]["code"] == "restarting"


# ------------------------------------------------------------------------------------------ acw-admin maintenance

def test_cli_status_is_off_then_on_then_off_as_json(capsys):
    out, code = run(capsys, "maintenance", "status", "--json")
    assert code is None and json.loads(out) == {"on": False, "since": None, "reason": "", "waiting": 0}
    out, code = run(capsys, "maintenance", "status")
    assert code is None and out == "off\n"
    out, code = run(capsys, "maintenance", "on", "--reason", "disk swap", "--json")
    on = json.loads(out)
    assert code is None and on["on"] is True and on["reason"] == "disk swap" and on["since"] == maintenance.state()["since"] and on["waiting"] == 0
    assert json.loads(run(capsys, "maintenance", "status", "--json")[0]) == on
    out, code = run(capsys, "maintenance", "off", "--json")
    assert code is None and json.loads(out) == {"on": False, "since": None, "reason": "", "waiting": 0}
    assert maintenance.state() is None
    assert actions() == [("admin-cli", "maintenance_on", "disk swap"), ("admin-cli", "maintenance_off", None)]


def test_cli_waiting_counts_only_non_admins_unstarted_queued_jobs(capsys, monkeypatch):
    a, b, c = user("admin@example.com", is_admin=1), user("b@example.com"), user("c@example.com")
    queued(a, 1)
    queued(b, 2)
    queued(c, 3)
    started = queued(b, 4)
    db.execute("UPDATE jobs SET started_at = ? WHERE id = ?", (db.now(), started))
    running = queued(c, 5)
    db.execute("UPDATE jobs SET status = 'running' WHERE id = ?", (running,))
    done = queued(c, 6)
    db.execute("UPDATE jobs SET status = 'done' WHERE id = ?", (done,))
    assert json.loads(run(capsys, "maintenance", "status", "--json")[0])["waiting"] == 0  # off: nothing is held
    out, _ = run(capsys, "maintenance", "on", "--json")
    assert json.loads(out)["waiting"] == 2  # b's and c's new jobs
    admins_env(monkeypatch, "c@example.com")  # listed admins are counted out too
    assert json.loads(run(capsys, "maintenance", "status", "--json")[0])["waiting"] == 1


def test_cli_text_output_and_default_reason(capsys):
    out, code = run(capsys, "maintenance", "on")
    assert code is None and out.startswith("on since ") and out.strip().endswith("(0 job(s) waiting)")
    assert maintenance.state()["reason"] == ""
    assert run(capsys, "maintenance", "off")[0] == "off\n"
    assert run(capsys, "maintenance", "off")[0] == "off\n"  # already off: still fine
    assert [a[1] for a in actions()] == ["maintenance_on", "maintenance_off"]


def test_cli_refuses_an_unknown_action(capsys):
    assert run(capsys, "maintenance", "pause")[1] == 2
    assert run(capsys, "maintenance")[1] == 2


# ------------------------------------------------------------------------------------------ work sent from this machine

@pytest.mark.parametrize("host,headers,want", [
    ("127.0.0.1", {}, True), ("::1", {}, True),
    ("127.0.0.1", {"x-forwarded-for": "203.0.113.9"}, False),  # came through the tunnel's proxy
    ("127.0.0.1", {"x-forwarded-for": ""}, False), ("::1", {"x-forwarded-for": "127.0.0.1"}, False),
    ("203.0.113.9", {}, False), ("10.0.0.5", {}, False), ("127.0.0.2", {}, False), ("0.0.0.0", {}, False), ("localhost", {}, False),
    ("::ffff:127.0.0.1", {}, False), ("testclient", {}, False), ("", {}, False), (None, {}, False),
])
def test_is_local_is_the_loopback_without_a_forwarding_header(host, headers, want):
    assert security.is_local(host, headers) is want


def test_is_local_reads_the_header_name_case_insensitively_on_real_requests():
    from starlette.datastructures import Headers

    assert security.is_local("127.0.0.1", Headers({"X-Forwarded-For": "1.2.3.4"})) is False
    assert security.is_local("127.0.0.1", Headers({"user-agent": "x"})) is True


def test_local_jobs_are_never_held_and_are_not_shown_as_paused(monkeypatch):
    b = user("b@example.com")
    outside, local = queued(b, 1), queued(b, 2, local=True)
    rows = [row(outside), row(local)]
    assert maintenance.runnable(rows) == rows  # off
    maintenance.switch_on("x")
    assert [r["id"] for r in maintenance.runnable(rows)] == [local]
    assert maintenance.paused(rows[0]) and not maintenance.paused(rows[1])
    monkeypatch.setattr(jobs, "queue_snapshot", lambda: ([outside, local], {outside: 10, local: 20}))
    assert _shown(outside)["paused"] == maintenance.JOB_MESSAGE
    shown = _shown(local)
    assert "paused" not in shown and shown["queueWaitSeconds"] == 20


def test_a_local_flag_of_zero_none_or_missing_does_not_exempt():
    maintenance.switch_on("x")
    base = {"status": "queued", "started_at": None, "user_id": "someone"}
    for extra in ({}, {"local_origin": 0}, {"local_origin": None}):
        assert maintenance.paused({**base, **extra})
    assert not maintenance.paused({**base, "local_origin": 1})


def test_claim_takes_a_local_job_behind_a_held_one_during_maintenance(roomy_disk):
    b = user("b@example.com")
    held, local = queued(b, 1), queued(b, 2, local=True)
    maintenance.switch_on("x")
    assert worker.claim("w1")["id"] == local
    assert worker.claim("w2") is None and row(held)["status"] == "queued"
    maintenance.switch_off()
    assert worker.claim("w2")["id"] == held


def test_create_job_stores_where_it_was_sent_from():
    b = user("b@example.com")
    ledger.credit(b, 5_000_000, "grant", "t", "g")
    a_id = make_job(b)
    c_id = jobs.create_job(b, kind="denoise", denoise_tier="standard", transcribe_tier=None, filename="a.wav", size_bytes=1000,
                           claimed_seconds=60.0, options={}, local_origin=True)["id"]
    assert (row(a_id)["local_origin"], row(c_id)["local_origin"]) == (0, 1)


def _client_from(host, **headers):
    from fastapi.testclient import TestClient

    from acw.app import create_app

    from .conftest import ORIGIN

    return TestClient(create_app(), base_url=ORIGIN, client=(host, 50000), headers=headers)


@pytest.mark.parametrize("host,headers,want", [
    ("127.0.0.1", {}, 1), ("::1", {}, 1), ("127.0.0.1", {"X-Forwarded-For": "203.0.113.9"}, 0), ("203.0.113.9", {}, 0), ("testclient", {}, 0),
])
def test_the_jobs_api_marks_a_job_local_only_when_sent_from_this_machine_directly(host, headers, want):
    from .conftest import ORIGIN, login

    with _client_from(host, **headers) as c:
        uid, h = login(c, "poster@example.com")
        ledger.credit(uid, 5_000_000, "grant", "t", "g")
        r = c.post("/api/jobs", json={"kind": "denoise", "filename": "a.wav", "sizeBytes": 3000, "durationSeconds": 60, "denoiseTier": "standard"}, headers=h)
        assert r.status_code == 200, r.text
        assert row(r.json()["id"])["local_origin"] == want


def test_live_handshake_from_this_machine_is_not_refused_for_maintenance(monkeypatch):
    from fastapi.testclient import TestClient

    from acw.live_app import create_live_app
    from acw.stream import routes

    from .conftest import ORIGIN

    other = user("other@example.com")
    key = api_key(other)
    monkeypatch.setattr(routes, "MAX_SESSIONS_PER_USER", 0)  # the check after maintenance's: a recognisable refusal
    maintenance.switch_on("x")
    auth_h = {"Authorization": f"Bearer {key}"}
    cases = [("127.0.0.1", {}, "too_many_sessions"), ("::1", {}, "too_many_sessions"),
             ("127.0.0.1", {"X-Forwarded-For": "203.0.113.9"}, "maintenance"), ("203.0.113.9", {}, "maintenance")]
    for host, extra, code in cases:
        with TestClient(create_live_app(), base_url=ORIGIN, client=(host, 50000)) as c:
            status, event = ws_denied(c, {**auth_h, **extra})
            assert status is None and event["code"] == code, (host, extra, event)


def test_cli_waiting_does_not_count_local_jobs(capsys):
    b = user("b@example.com")
    queued(b, 1)
    queued(b, 2, local=True)
    queued(b, 3, local=True)
    out, _ = run(capsys, "maintenance", "on", "--json")
    assert json.loads(out)["waiting"] == 1
    assert json.loads(run(capsys, "maintenance", "status", "--json")[0])["waiting"] == 1


@pytest.mark.parametrize("reason", ["-x", "--now", "-5 min", "--help", "-"])
def test_cli_reason_equals_form_accepts_a_reason_starting_with_a_dash(capsys, reason):
    out, code = run(capsys, "maintenance", "on", f"--reason={reason}", "--json")
    assert code is None and json.loads(out)["reason"] == reason
    assert maintenance.state()["reason"] == reason
    assert actions()[-1] == ("admin-cli", "maintenance_on", reason)
