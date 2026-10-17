"""Retention and bounded storage: what the sweep deletes after 30 days, 90 days, a year and seven years; that money is
never touched by the sweep; the database compaction; the size alerts; backups kept to a count; the quiet access log."""
import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from acw import auth, backup, db, monitor, retention, storage
from acw.billing import ledger
from acw.config import get_settings

DAY = 86400
NOW = 2_000_000_000  # a fixed "now" far from the real clock so nothing depends on it


def user(email="a@example.com") -> str:
    return auth.get_or_create_user(email)["id"]


def job(uid, status="done", age_days=0, name="talk.wav", jid=None) -> str:
    jid = jid or db.new_id()
    db.execute(
        "INSERT INTO jobs(id,user_id,kind,tier,status,params_json,input_name,input_bytes,claimed_seconds,created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (jid, uid, "denoise", "standard", status, json.dumps({"display_name": name, "options": {}}), name, 1, 1.0, NOW - int(age_days * DAY)),
    )
    return jid


def count(table, where="1", params=()):
    return db.one(f"SELECT COUNT(*) n FROM {table} WHERE {where}", params)["n"]


# ---------------------------------------------------------------- a year: jobs, live sessions, holds, audit, keys

def test_finished_jobs_older_than_a_year_are_deleted_and_others_kept():
    uid = user()
    old = {s: job(uid, s, 400) for s in ("done", "failed", "canceled", "expired")}
    live_old = {s: job(uid, s, 400) for s in ("uploading", "queued", "running", "cloud_asr")}
    young = {s: job(uid, s, 364) for s in ("done", "failed")}
    assert retention.run(NOW)["jobs"] == 4
    for jid in old.values():
        assert db.one("SELECT 1 FROM jobs WHERE id = ?", (jid,)) is None
    for jid in (*live_old.values(), *young.values()):
        assert db.one("SELECT 1 FROM jobs WHERE id = ?", (jid,)) is not None  # not finished, or not old enough
    assert retention.run(NOW)["jobs"] == 0


def test_ended_live_sessions_holds_and_revoked_keys_older_than_a_year_go():
    uid = user()
    ins = "INSERT INTO stream_sessions(id,user_id,status,started_at) VALUES (?,?,?,?)"
    db.execute(ins, ("s-old", uid, "ended", NOW - 400 * DAY))
    db.execute(ins, ("s-new", uid, "ended", NOW - 10 * DAY))
    db.execute(ins, ("s-act", uid, "active", NOW - 400 * DAY))  # still marked running: close_orphans' business, not ours
    hold = "INSERT INTO holds(id,user_id,amount_micros,ref_type,ref_id,status,created_at,closed_at) VALUES (?,?,?,?,?,?,?,?)"
    db.execute(hold, ("h-cap", uid, 5, "job", "j", "captured", NOW - 401 * DAY, NOW - 400 * DAY))
    db.execute(hold, ("h-rel", uid, 5, "job", "j", "released", NOW - 400 * DAY, None))
    db.execute(hold, ("h-new", uid, 5, "job", "j", "captured", NOW - 3 * DAY, NOW - 3 * DAY))
    key = "INSERT INTO api_keys(id,user_id,name,prefix,key_hash,scopes,created_at,revoked_at) VALUES (?,?,?,?,?,?,?,?)"
    db.execute(key, ("k-old", uid, "n", "p", "h1", "stream", NOW - 500 * DAY, NOW - 400 * DAY))
    db.execute(key, ("k-new", uid, "n", "p", "h2", "stream", NOW - 500 * DAY, NOW - 5 * DAY))
    db.execute(key, ("k-live", uid, "n", "p", "h3", "stream", NOW - 500 * DAY, None))
    done = retention.run(NOW)
    assert (done["stream_sessions"], done["holds"], done["api_keys"]) == (1, 2, 1)
    assert {r["id"] for r in db.all_rows("SELECT id FROM stream_sessions")} == {"s-new", "s-act"}
    assert {r["id"] for r in db.all_rows("SELECT id FROM holds")} == {"h-new"}
    assert {r["id"] for r in db.all_rows("SELECT id FROM api_keys")} == {"k-new", "k-live"}


def test_an_active_hold_is_never_deleted_however_old():
    uid = user()
    ledger.credit(uid, 1_000_000, "grant", "t", "g")
    hid = ledger.create_hold(uid, 500_000, "job", "j")
    db.execute("UPDATE holds SET created_at = ? WHERE id = ?", (NOW - 900 * DAY, hid))
    retention.run(NOW)
    assert db.one("SELECT status FROM holds WHERE id = ?", (hid,))["status"] == "active"
    assert ledger.invariant_problems() == {"balance_mismatch": 0, "held_mismatch": 0}


def test_audit_rows_routine_after_a_year_money_ones_kept():
    uid = user()
    for action in (*retention.AUDIT_ROUTINE, *retention.AUDIT_WITH_MONEY, "something_new"):
        db.execute("INSERT INTO audit_log(user_id,actor,action,created_at) VALUES (?,?,?,?)", (uid, "t", action, NOW - 400 * DAY))
    db.execute("INSERT INTO audit_log(user_id,actor,action,created_at) VALUES (?,?,?,?)", (uid, "t", "login", NOW - 3 * DAY))
    retention.run(NOW)
    left = {r["action"] for r in db.all_rows("SELECT action FROM audit_log WHERE created_at < ?", (NOW - 300 * DAY,))}
    assert left == {*retention.AUDIT_WITH_MONEY, "something_new"}  # an unclassified action is kept, not deleted
    assert count("audit_log", "action = 'login' AND created_at = ?", (NOW - 3 * DAY,)) == 1


def test_every_audit_action_the_code_writes_is_classified_as_routine_or_money():
    """A new db.audit(...) action must say whether it is routine (deleted after 12 months) or evidence kept with the money."""
    root = Path(__file__).resolve().parents[1] / "acw"
    actions = set()
    for f in root.rglob("*.py"):
        text = f.read_text()
        actions |= set(re.findall(r"db\.audit\(\s*[^,]+,\s*[\"'](\w+)[\"']", text))
        actions |= set(re.findall(r"INSERT INTO audit_log\([^)]*\) VALUES \([^)]*'(\w+)'", text))
        actions |= set(re.findall(r"_audit_once\(\s*[\"'](\w+)[\"']", text))
        actions |= set(re.findall(r"_note_unrecovered\(\s*[\"'](\w+)[\"']", text))
    actions = {a for a in actions if not a.endswith("_")} | {"auto_reload_enabled", "auto_reload_disabled"}  # built as "auto_reload_" + ...
    assert {"login", "tos_accepted", "contact_sent", "refund", "grant", "account_deleted", "dispute_unrecovered"} <= actions
    unclassified = actions - set(retention.AUDIT_ROUTINE) - set(retention.AUDIT_WITH_MONEY)
    assert not unclassified, f"classify these audit actions in acw/retention.py: {sorted(unclassified)}"


# ---------------------------------------------------------------- 30 days: sessions, links, sign-in states; 90 days: events

def test_sessions_links_and_sign_in_states_go_after_30_days():
    uid = user()
    s = "INSERT INTO sessions(id_hash,user_id,csrf_token,created_at,expires_at,last_seen_at) VALUES (?,?,?,?,?,?)"
    db.execute(s, ("s-old", uid, "c", NOW - 90 * DAY, NOW - 31 * DAY, NOW - 40 * DAY))
    db.execute(s, ("s-recent", uid, "c", NOW - 90 * DAY, NOW - 29 * DAY, NOW - 40 * DAY))
    db.execute(s, ("s-live", uid, "c", NOW - 10 * DAY, NOW + 20 * DAY, NOW - 1 * DAY))
    t = "INSERT INTO login_tokens(token_hash,email,purpose,created_at,expires_at,used_at) VALUES (?,?,?,?,?,?)"
    db.execute(t, ("t-old", "x@e.com", "email_login", NOW - 50 * DAY, NOW - 50 * DAY + 900, None))
    db.execute(t, ("t-used", "x@e.com", "email_login", NOW - 50 * DAY, NOW + 900, NOW - 49 * DAY))
    db.execute(t, ("t-new", "x@e.com", "email_login", NOW - 100, NOW + 800, None))
    o = "INSERT INTO oauth_states(state_hash,provider,code_verifier,created_at,expires_at) VALUES (?,?,?,?,?)"
    db.execute(o, ("o-old", "google", "v", NOW - 60 * DAY, NOW - 60 * DAY + 600))
    db.execute(o, ("o-new", "google", "v", NOW - 100, NOW + 500))
    retention.run(NOW)
    assert {r["id_hash"] for r in db.all_rows("SELECT id_hash FROM sessions")} == {"s-recent", "s-live"}
    assert {r["token_hash"] for r in db.all_rows("SELECT token_hash FROM login_tokens")} == {"t-new"}
    assert {r["state_hash"] for r in db.all_rows("SELECT state_hash FROM oauth_states")} == {"o-new"}


def test_stripe_events_go_after_90_days_but_unprocessed_ones_stay_a_year():
    e = "INSERT INTO stripe_events(id,type,received_at,processed_at) VALUES (?,?,?,?)"
    db.execute(e, ("done-old", "t", NOW - 91 * DAY, NOW - 91 * DAY))
    db.execute(e, ("done-new", "t", NOW - 10 * DAY, NOW - 10 * DAY))
    db.execute(e, ("open-old", "t", NOW - 120 * DAY, None))  # not processed: stays for a look until a year
    db.execute(e, ("open-ancient", "t", NOW - 400 * DAY, None))
    retention.run(NOW)
    assert {r["id"] for r in db.all_rows("SELECT id FROM stripe_events")} == {"done-new", "open-old"}


# ---------------------------------------------------------------- file names leave the history; amounts stay

def test_file_names_leave_the_credit_history_after_a_year_amounts_stay_and_recent_names_stay():
    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    h = ledger.create_hold(uid, 1_000_000, "job", "j1")
    ledger.capture_hold(h, 400_000, "Clean & polish: my-secret-podcast.wav (12.5 min)")
    h2 = ledger.create_hold(uid, 1_000_000, "job", "j2")
    ledger.capture_hold(h2, 100_000, "denoise + transcribe: legacy: name (with) parens.flac (0.2 min)")
    h3 = ledger.create_hold(uid, 1_000_000, "stream", "s1")
    ledger.capture_hold(h3, 50_000, "Live session (3.0 min)")
    h4 = ledger.create_hold(uid, 1_000_000, "job", "j4")
    ledger.capture_hold(h4, 200_000, "Denoising: private-interview.wav (5.0 min)")
    db.execute("UPDATE ledger SET created_at = ?", (NOW - 400 * DAY,))
    h5 = ledger.create_hold(uid, 1_000_000, "job", "j5")
    ledger.capture_hold(h5, 400_000, "Denoising: recent.wav (1.0 min)")  # a line from this month keeps its name
    db.execute("UPDATE ledger SET created_at = ? WHERE description LIKE 'Denoising: recent%'", (NOW - 10 * DAY,))
    before = ledger.balance(uid)
    assert retention.run(NOW)["usage_names"] == 3
    rows = [r["description"] for r in db.all_rows("SELECT description FROM ledger WHERE kind = 'usage' ORDER BY id")]
    # Rows written before the rename keep their old label ("Clean & polish"); legacy kind names get today's label.
    assert rows[:4] == ["Clean & polish (12.5 min)", "Denoising + Subtitles & transcript (0.2 min)", "Live session (3.0 min)", "Denoising (5.0 min)"]
    assert rows[4] == "Denoising: recent.wav (1.0 min)"
    everything = str([dict(r) for r in db.all_rows("SELECT * FROM ledger")])
    assert "secret" not in everything and "legacy" not in everything and "private-interview" not in everything
    assert ledger.balance(uid) == before and ledger.invariant_problems() == {"balance_mismatch": 0, "held_mismatch": 0}
    assert retention.run(NOW)["usage_names"] == 0


def test_credit_history_still_renders_without_the_job_rows(client, signed_in):
    uid, h = signed_in
    now = db.now()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    jid = job(uid, "done", 0)
    hold = ledger.create_hold(uid, 1_000_000, "job", jid)
    ledger.capture_hold(hold, 400_000, "Denoising: song.wav (2.0 min)")
    db.execute("UPDATE ledger SET created_at = ?", (now - 400 * DAY,))
    db.execute("UPDATE jobs SET created_at = ?", (now - 400 * DAY,))
    retention.run(now)
    assert db.one("SELECT 1 FROM jobs WHERE id = ?", (jid,)) is None
    hist = client.get("/api/billing/history")
    assert hist.status_code == 200
    usage = [r for r in hist.json()["ledger"] if r["kind"] == "usage"]
    assert usage and usage[0]["description"] == "Denoising (2.0 min)" and "song" not in hist.text


# ---------------------------------------------------------------- money is never touched by the sweep

def _money_user(email, rows):
    """A user with ledger rows [(age_days, amount_micros, kind)], and a matching balance."""
    uid = user(email)
    for age, amount, kind in rows:
        if amount > 0:
            ledger.credit(uid, amount, kind, "t", f"{email}-{age}-{amount}")
        else:
            ledger.debit_available(uid, -amount, "usage", "job", f"{email}-{age}-{amount}", "Denoising: x.wav (1.0 min)")
        db.execute("UPDATE ledger SET created_at = ? WHERE id = (SELECT MAX(id) FROM ledger WHERE user_id = ?)", (NOW - int(age * DAY), uid))
    return uid


def _pay(uid, age_days, status="succeeded"):
    db.execute(
        "INSERT INTO payments(id,user_id,kind,status,amount_cents,created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
        (db.new_id(), uid, "checkout", status, 500, NOW - int(age_days * DAY), NOW - int(age_days * DAY)),
    )


def test_the_sweep_never_deletes_ledger_payments_or_money_audit_rows():
    uid = _money_user("old@example.com", [(3000, 5_000_000, "topup"), (2900, -1_000_000, "usage"), (10, 2_000_000, "topup")])
    _pay(uid, 3000)
    db.execute("INSERT INTO audit_log(user_id,actor,action,created_at) VALUES (?,?,?,?)", (uid, "stripe", "refund", NOW - 3000 * DAY))
    counts = [count("ledger"), count("payments"), count("audit_log", "action = 'refund'")]
    retention.run(NOW)
    assert [count("ledger"), count("payments"), count("audit_log", "action = 'refund'")] == counts
    assert ledger.invariant_problems() == {"balance_mismatch": 0, "held_mismatch": 0}


def test_big_backlogs_are_deleted_in_batches_over_several_sweeps(monkeypatch):
    uid = user()
    monkeypatch.setattr(retention, "BATCH", 10)
    monkeypatch.setattr(retention, "MAX_BATCHES", 3)
    for _ in range(100):
        job(uid, "done", 400)
    assert retention.run(NOW)["jobs"] == 30
    assert count("jobs") == 70
    for _ in range(3):
        retention.run(NOW)
    assert count("jobs") == 0


def test_sweep_runs_hourly_and_survives_a_failing_compaction(monkeypatch):
    calls = []
    monkeypatch.setattr(retention, "run", lambda: calls.append("run") or {})
    monkeypatch.setattr(retention, "maintain_storage", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(retention, "_last", None)
    retention.sweep()
    retention.sweep()
    assert calls == ["run"]


# ---------------------------------------------------------------- compaction of the file

def _fill_and_delete(n=300):
    uid = user()
    for _ in range(n):
        job(uid, "done", 400, name="x" * 200)
    db.execute("DELETE FROM jobs")  # leaves the pages free inside the file


def test_vacuum_gives_the_space_back_once_a_month_in_the_small_hours(monkeypatch):
    monkeypatch.setattr(retention, "VACUUM_MIN_FREE_PAGES", 5)
    _fill_and_delete()
    pages, free, _ = retention._page_facts()
    assert free >= retention.VACUUM_MIN_FREE_PAGES
    size = get_settings().db_path.stat().st_size
    now = db.now()
    assert retention.maintain_storage(now, hour=12) is None  # busy hours
    assert retention.maintain_storage(now, hour=3) == "vacuumed"
    assert get_settings().db_path.stat().st_size < size
    assert retention._page_facts()[1] == 0
    assert retention.maintain_storage(now + DAY, hour=3) is None  # not again within the month
    _fill_and_delete()
    assert retention.maintain_storage(now + 29 * DAY, hour=2) == "vacuumed"


def test_vacuum_waits_for_live_sessions_and_for_disk_space(monkeypatch):
    monkeypatch.setattr(retention, "VACUUM_MIN_FREE_PAGES", 5)
    uid = user()
    _fill_and_delete()
    db.execute("INSERT INTO stream_sessions(id,user_id,status,started_at) VALUES ('live',?,'active',?)", (uid, db.now()))
    assert retention.maintain_storage(hour=3) == "skipped: live session running"
    db.execute("UPDATE stream_sessions SET status = 'ended'")
    real_disk_usage = retention.shutil.disk_usage
    monkeypatch.setattr(retention.shutil, "disk_usage", lambda p: SimpleNamespace(free=1024))
    assert retention.maintain_storage(hour=3) == "skipped: low disk space"
    assert db.one("SELECT 1 FROM meta WHERE key = 'retention.vacuumed_at'") is None
    monkeypatch.setattr(retention.shutil, "disk_usage", real_disk_usage)
    assert retention.maintain_storage(hour=3) == "vacuumed"


# ---------------------------------------------------------------- money records after 7 years (by hand)

def _sum(uid):
    return db.one("SELECT COALESCE(SUM(amount_micros), 0) s FROM ledger WHERE user_id = ?", (uid,))["s"]


def test_compact_money_keeps_every_balance_and_the_chain_of_balances():
    a = _money_user("a@example.com", [(3000, 5_000_000, "topup"), (2900, -1_000_000, "usage"), (2700, 250_000, "grant"),
                                       (30, 1_000_000, "topup"), (5, -300_000, "usage")])
    b = _money_user("b@example.com", [(100, 2_000_000, "topup")])  # nothing old
    c = _money_user("c@example.com", [(2800, 3_000_000, "topup"), (2600, -3_000_000, "usage")])  # old and back to zero
    _pay(a, 3000)
    _pay(a, 2900, "pending")  # not settled: kept
    _pay(a, 10)
    db.execute("INSERT INTO audit_log(user_id,actor,action,created_at) VALUES (?,?,?,?)", (a, "x", "refund", NOW - 3000 * DAY))
    db.execute("INSERT INTO audit_log(user_id,actor,action,created_at) VALUES (?,?,?,?)", (a, "x", "refund", NOW - 20 * DAY))
    balances = {u: ledger.balance(u)["balance_micros"] for u in (a, b, c)}
    later = db.all_rows("SELECT id, balance_after_micros FROM ledger WHERE user_id = ? AND created_at > ?", (a, NOW - 100 * DAY))

    dry = retention.compact_money(NOW)
    assert dry["users"] == 2 and dry["ledger_rows"] == 2 + 2 and dry["payments"] == 1  # a: 3 old rows -> 1; c: 2 -> none
    assert count("ledger") == 8  # a dry run changes nothing

    res = retention.compact_money(NOW, apply=True)
    assert res["users"] == 2 and res["ledger_rows"] == 4 and res["payments"] == 1 and res["audit_log"] >= 1
    for u in (a, b, c):
        assert ledger.balance(u)["balance_micros"] == balances[u] == _sum(u)
    assert ledger.invariant_problems() == {"balance_mismatch": 0, "held_mismatch": 0}
    first = db.one("SELECT * FROM ledger WHERE user_id = ? ORDER BY id LIMIT 1", (a,))
    assert first["ref_type"] == "carry_forward" and first["kind"] == "adjustment"
    assert first["amount_micros"] == first["balance_after_micros"] == 5_000_000 - 1_000_000 + 250_000
    assert [dict(r) for r in db.all_rows("SELECT id, balance_after_micros FROM ledger WHERE user_id = ? AND created_at > ?", (a, NOW - 100 * DAY))] == [dict(r) for r in later]
    assert count("payments", "user_id = ?", (a,)) == 2  # the pending one and the recent one
    assert count("audit_log", "action = 'refund'") == 1
    again = retention.compact_money(NOW, apply=True)  # and it can be run again
    assert again["users"] == 0 and again["ledger_rows"] == 0
    ledger.credit(a, 1_000, "grant", "t", "more")  # new activity still works on top of the carried-forward row
    assert ledger.balance(a)["balance_micros"] == _sum(a)


def test_compact_money_rolls_back_a_user_whose_rows_do_not_add_up():
    a = _money_user("a@example.com", [(3000, 5_000_000, "topup"), (10, 1_000, "grant")])
    db.execute("UPDATE ledger SET amount_micros = amount_micros + 1 WHERE user_id = ? AND created_at < ?", (a, NOW - 100 * DAY))
    rows = count("ledger")
    with pytest.raises(RuntimeError, match="do not add up"):
        retention.compact_money(NOW, apply=True)
    assert count("ledger") == rows


def test_compact_money_removes_deleted_accounts_that_nothing_refers_to_any_more():
    gone = _money_user("gone@example.com", [(3000, 1_000_000, "topup"), (2900, -1_000_000, "usage")])
    kept = _money_user("kept@example.com", [(3000, 1_000_000, "topup"), (2900, -400_000, "usage"), (10, 1, "grant")])
    db.execute("UPDATE users SET status = 'deleted' WHERE id IN (?, ?)", (gone, kept))
    retention.run(NOW)  # the routine 'user_created' rows are long past their year
    retention.compact_money(NOW, apply=True)
    assert db.one("SELECT 1 FROM users WHERE id = ?", (gone,)) is None  # its old rows added up to nothing, and no balance is left
    assert db.one("SELECT 1 FROM users WHERE id = ?", (kept,)) is not None


def test_the_monitor_reminds_about_records_older_than_seven_years():
    _money_user("a@example.com", [(10, 1_000_000, "topup")])
    (f,) = monitor.check_old_money(NOW)
    assert f.kind == "old_money_records" and not f.bad
    db.execute("UPDATE ledger SET created_at = ?", (NOW - 7 * 365 * DAY - 10 * DAY,))
    assert monitor.check_old_money(NOW)[0].bad is False  # inside the month of grace after the 7 years
    db.execute("UPDATE ledger SET created_at = ?", (NOW - 7 * 365 * DAY - 40 * DAY,))
    (f,) = monitor.check_old_money(NOW)
    assert f.bad and "compact-money" in f.body


# ---------------------------------------------------------------- what the monitor watches

def test_size_alerts_for_the_data_and_backup_directories_skip_job_files(tmp_path, monkeypatch):
    data, bk = tmp_path / "sizes-data", tmp_path / "sizes-bk"
    (data / "jobs" / "abc").mkdir(parents=True)
    bk.mkdir()
    (data / "jobs" / "abc" / "input.wav").write_bytes(b"x" * 3_000_000)  # job files do not count towards the data limit
    (data / "acw.sqlite3").write_bytes(b"x" * 1000)
    (bk / "acw-1.sqlite3.gz").write_bytes(b"x" * 1000)
    monkeypatch.setenv("ACW_DATA_ALERT_GB", "0.000002")  # about 2 KB
    monkeypatch.setenv("ACW_BACKUPS_ALERT_GB", "0.000002")
    f = {x.kind: x for x in monitor.check_storage(data, bk)}
    assert not f["data_large"].bad and not f["backups_large"].bad
    (data / "stray.bin").write_bytes(b"x" * 5000)
    (bk / "acw-2.sqlite3.gz").write_bytes(b"x" * 5000)
    f = {x.kind: x for x in monitor.check_storage(data, bk)}
    assert f["data_large"].bad and f["backups_large"].bad
    assert "ACW_DATA_ALERT_GB" in f["data_large"].body
    s = storage.sizes(data, bk)
    assert s["jobs"] == 3_000_000 and s["data"] == 6000 and s["backups"] == 6000


# ---------------------------------------------------------------- backups stay at --keep, also when they fail

def test_backups_stay_at_keep_also_when_tonight_fails_and_failed_snapshots_are_capped(tmp_path, monkeypatch):
    user()
    out = tmp_path / "b"
    stamps = iter(f"20260101-{i:06d}" for i in range(100))  # snapshot names carry the second: give each run its own
    monkeypatch.setattr(backup, "time", SimpleNamespace(strftime=lambda fmt, t=None: next(stamps), gmtime=backup.time.gmtime,
                                                        time=backup.time.time))
    for _ in range(5):
        backup.run(out, keep=3)
    assert len(list(out.glob("acw-*.sqlite3.gz"))) == 3
    # tonight the off-machine copy fails: the snapshot is good but the run fails; the folder still stays at `keep`
    monkeypatch.setenv("ACW_BACKUP_OFFBOX", str(tmp_path / "remote") + "/")
    monkeypatch.delenv("ACW_BACKUP_PASSPHRASE_FILE", raising=False)
    monkeypatch.delenv("ACW_BACKUP_AGE_RECIPIENT", raising=False)
    for _ in range(2):
        with pytest.raises(backup.BackupError):
            backup.run(out, keep=3)
    assert len(list(out.glob("acw-*.sqlite3.gz"))) == 3
    for i in range(6):  # failed snapshots (*.bad): only the newest few are kept
        (out / f"acw-2020010{i}-000000.sqlite3.gz.bad").write_bytes(b"x")
    backup._prune(out, 3)
    assert len(list(out.glob("*.bad"))) == backup.KEEP_BAD


# ---------------------------------------------------------------- quiet access log

def test_polling_the_job_list_is_not_logged_but_errors_and_everything_else_are(client, signed_in, caplog):
    from acw.bodylimit import _quiet

    assert _quiet("GET", "/audio/api/jobs", 200) and _quiet("GET", "/api/health", 200)
    assert not _quiet("GET", "/api/jobs", 429) and not _quiet("GET", "/api/jobs", 401)
    assert not _quiet("POST", "/api/jobs", 200) and not _quiet("GET", "/api/jobs/" + "a" * 32, 200)
    assert not _quiet("GET", "/api/health", 503) and not _quiet("GET", "/api/me", 200)
    with caplog.at_level(logging.INFO, logger="acw.access"):
        for _ in range(5):
            assert client.get("/api/jobs").status_code == 200
        client.get("/api/me")
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "/api/me" in text and "/api/jobs" not in text
