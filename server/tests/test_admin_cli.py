"""The operator's command line (acw/admin.py), run through main(argv) against the test database: what each command prints and changes, and
the exit status the scripts rely on (check-money, verify-backup and compact-money exit 1 on a problem)."""
import gzip
import json
import shutil
import sqlite3

import pytest

from acw import admin, auth, db, monitor
from acw.billing import ledger

DAY = 86400
YEAR = 365 * DAY


def run(capsys, *argv):
    """(what it printed, its exit: None when it returned, else the code or message)."""
    try:
        admin.main(list(argv))
        code = None
    except SystemExit as e:
        code = e.code
    return capsys.readouterr().out, code


def user(email="a@example.com") -> str:
    return auth.get_or_create_user(email)["id"]


def balance(uid):
    return ledger.balance(uid)["balance_micros"]


# --------------------------------------------------------------------------------------------------------------------- accounts

def test_init_db_and_login_link(capsys):
    assert run(capsys, "init-db") == ("ok\n", None)
    out, code = run(capsys, "login-link", "New.Person@Example.com")
    assert code is None and "/login/verify#token=" in out and "valid once, for 15 minutes" in out
    assert db.one("SELECT email FROM users")["email"] == "new.person@example.com"
    assert db.one("SELECT COUNT(*) AS n FROM login_tokens")["n"] == 1


def test_grant_adds_credit_writes_the_ledger_line_and_the_audit_and_prints_the_balance(capsys):
    uid = user()
    out, code = run(capsys, "grant", "A@Example.com", "12.5", "support: thanks")
    assert code is None and balance(uid) == 12_500_000
    assert str(12_500_000) in out
    row = db.one("SELECT kind, amount_micros, description FROM ledger WHERE user_id = ?", (uid,))
    assert (row["kind"], row["amount_micros"]) == ("grant", 12_500_000) and "support: thanks" in ledger.history(uid)[0]["description"]
    audit = db.one("SELECT actor, detail FROM audit_log WHERE action = 'grant' AND user_id = ?", (uid,))
    assert audit["actor"] == "admin-cli" and audit["detail"] == "12.5 support: thanks"
    run(capsys, "grant", "a@example.com", "0.000001", "x" * 500)  # (the least; a long reason is cut)
    assert balance(uid) == 12_500_001 and len(ledger.history(uid)[0]["description"]) <= 200
    assert ledger.invariant_problems() == {"balance_mismatch": 0, "held_mismatch": 0}


@pytest.mark.parametrize("usd", ["0", "-5", "1000.01", "5000"])
def test_grant_refuses_an_amount_outside_zero_to_a_thousand_dollars_and_changes_nothing(capsys, usd):
    uid = user()
    out, code = run(capsys, "grant", "a@example.com", usd, "r")
    assert code == "amount must be between 0 and 1000 USD" and balance(uid) == 0
    assert db.one("SELECT COUNT(*) AS n FROM ledger")["n"] == 0


def test_an_unknown_account_stops_every_command_that_names_one(capsys):
    for argv in (("grant", "nobody@example.com", "5", "r"), ("user", "nobody@example.com"), ("suspend", "nobody@example.com", "r"),
                 ("unsuspend", "nobody@example.com")):
        assert run(capsys, *argv)[1] == "no user nobody@example.com", argv


def test_user_shows_the_account_the_balance_and_the_latest_lines(capsys):
    uid = user()
    for i in range(12):
        ledger.credit(uid, 1_000_000, "grant", "t", f"g{i}", f"line {i}")
    out, code = run(capsys, "user", "a@example.com")
    lines = out.strip().splitlines()
    assert code is None and f"'id': '{uid}'" in lines[0] and "'email': 'a@example.com'" in lines[0] and "'status': 'active'" in lines[0]
    assert "12000000" in lines[1] and len(lines) == 2 + 10  # (the ten latest)
    assert "line 11" in lines[2] and "line 2" in lines[-1] and "line 1'" not in out.replace("line 11", "")


def test_suspend_stops_the_account_ends_its_sessions_and_lets_go_of_its_holds_and_unsuspend_brings_it_back(capsys):
    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    hold = ledger.create_hold(uid, 2_000_000, "job", "j1")
    assert ledger.balance(uid)["held_micros"] == 2_000_000
    out, code = run(capsys, "suspend", "a@example.com", "abuse reported")
    assert (out, code) == ("suspended\n", None)
    assert db.one("SELECT status FROM users")["status"] == "suspended" and ledger.balance(uid)["held_micros"] == 0
    assert db.one("SELECT status FROM holds WHERE id = ?", (hold,))["status"] != "active" and balance(uid) == 5_000_000
    assert db.one("SELECT detail FROM audit_log WHERE action = 'suspend'")["detail"] == "abuse reported"
    out, code = run(capsys, "unsuspend", "a@example.com")
    assert (out, code) == ("active\n", None) and db.one("SELECT status FROM users")["status"] == "active"
    assert db.one("SELECT actor FROM audit_log WHERE action = 'unsuspend'")["actor"] == "admin-cli"


def test_suspending_ends_the_signed_in_sessions(client, signed_in, capsys):
    uid, h = signed_in
    assert client.get("/api/me").json()["user"]["id"] == uid
    run(capsys, "suspend", "user@example.com", "r")
    assert db.one("SELECT COUNT(*) AS n FROM sessions WHERE user_id = ?", (uid,))["n"] == 0
    assert client.get("/api/me").json().get("user") in (None, {})


def test_an_unknown_command_is_refused(capsys):
    with pytest.raises(SystemExit) as e:
        admin.main(["frobnicate"])
    assert e.value.code == 2


# --------------------------------------------------------------------------------------------------------------------- the checks

def test_check_money_prints_the_counts_and_exits_1_when_a_balance_or_a_hold_is_wrong(capsys):
    uid = user()
    ledger.credit(uid, 3_000_000, "grant", "t", "g")
    assert run(capsys, "check-money") == ("{'balance_mismatch': 0, 'held_mismatch': 0}\n", None)
    db.execute("UPDATE users SET balance_micros = balance_micros + 1 WHERE id = ?", (uid,))
    out, code = run(capsys, "check-money")
    assert code == 1 and "'balance_mismatch': 1" in out
    db.execute("UPDATE users SET balance_micros = balance_micros - 1, held_micros = 7 WHERE id = ?", (uid,))
    out, code = run(capsys, "check-money")
    assert code == 1 and "'held_mismatch': 1" in out and "'balance_mismatch': 0" in out


def test_verify_backup_passes_a_good_snapshot_and_fails_one_whose_ledger_is_wrong_or_that_is_not_a_database(capsys, tmp_path):
    from acw import backup

    uid = user()
    ledger.credit(uid, 3_000_000, "grant", "t", "g")
    good = backup.run(tmp_path / "bk")
    out, code = run(capsys, "verify-backup", str(good))
    assert code is None and "'integrity': 'ok'" in out and "'ledger': 'ok'" in out
    # a snapshot of a database whose balance does not match its ledger
    db.execute("UPDATE users SET balance_micros = balance_micros + 5 WHERE id = ?", (uid,))
    raw = tmp_path / "wrong.sqlite3"
    dst = sqlite3.connect(str(raw))
    db.conn().backup(dst)
    dst.close()
    wrong = tmp_path / "wrong.sqlite3.gz"
    with open(raw, "rb") as src, gzip.open(wrong, "wb") as gz:
        shutil.copyfileobj(src, gz)
    out, code = run(capsys, "verify-backup", str(wrong))
    assert code == 1 and "1 balances and 0 holds do not match" in out
    # not a database at all
    junk = tmp_path / "junk.sqlite3.gz"
    junk.write_bytes(gzip.compress(b"this is not a database" * 100))
    with pytest.raises(sqlite3.DatabaseError):
        admin.main(["verify-backup", str(junk)])
    capsys.readouterr()
    with pytest.raises(FileNotFoundError):
        admin.main(["verify-backup", str(tmp_path / "missing.gz")])


def test_retention_applies_the_rules_now_and_prints_the_counts(capsys):
    uid = user()
    old = db.now() - 100 * DAY
    db.execute("INSERT INTO stream_sessions(id,user_id,status,started_at,client_ip) VALUES ('s1',?,'ended',?,'203.0.113.9')", (uid, old))
    out, code = run(capsys, "retention")
    assert code is None and "'stream_ips': 1" in out
    assert db.one("SELECT client_ip FROM stream_sessions")["client_ip"] is None
    assert "'stream_ips': 0" in run(capsys, "retention")[0]


def test_storage_lists_the_sizes_and_the_limits(capsys, tmp_path, monkeypatch):
    from acw.config import get_settings

    monkeypatch.setattr(monitor, "BACKUP_DIR", tmp_path / "backups")
    (tmp_path / "backups").mkdir()
    (tmp_path / "backups" / "x.gz").write_bytes(b"x" * (3 * 1024**2))
    jobs = get_settings().data_dir / "jobs" / "j1"
    jobs.mkdir(parents=True)
    (jobs / "input.wav").write_bytes(b"x" * (2 * 1024**2))
    out, code = run(capsys, "storage")
    rows = {line.split()[0]: float(line.split()[1]) for line in out.splitlines() if line.endswith(" MB") and not line.startswith("alert")}
    assert code is None and set(rows) == {"data", "database", "jobs", "backups"}
    assert rows["backups"] == 3.0 and rows["jobs"] == 2.0 and rows["database"] > 0
    assert "alert limits: data" in out and "(without jobs)" in out


def test_activity_live_count_and_stats_print_without_changing_anything(capsys):
    uid = user()
    db.execute("INSERT INTO stream_sessions(id,user_id,status,started_at) VALUES ('s1',?,'active',?)", (uid, db.now()))
    assert run(capsys, "activity") == ("file jobs: none; live sessions: 1\n", None)
    assert run(capsys, "live-count") == ("1\n", None)
    out, code = run(capsys, "stats", "--days", "7")
    assert code is None and out.strip()
    assert run(capsys, "stats", "--days", "0")[1] == "--days must be between 1 and 3650"
    assert run(capsys, "stats", "--days", "4000")[1] == "--days must be between 1 and 3650"


# --------------------------------------------------------------------------------------------------------------------- compact-money

def money_user(email, rows):
    """A user with ledger rows [(age_days, amount_micros)], whose balance matches."""
    uid = user(email)
    for n, (age, amount) in enumerate(rows):
        if amount > 0:
            ledger.credit(uid, amount, "topup", "t", f"{email}-{n}")
        else:
            ledger.debit_available(uid, -amount, "usage", "job", f"{email}-{n}", "Denoising: x.wav (1.0 min)")
        db.execute("UPDATE ledger SET created_at = ? WHERE id = (SELECT MAX(id) FROM ledger WHERE user_id = ?)", (db.now() - int(age * DAY), uid))
    return uid


def old_and_new():
    old = money_user("old@example.com", [(8 * 365, 5_000_000), (7 * 365 + 30, -1_000_000), (3 * 365, 2_000_000), (10, -500_000)])
    young = money_user("young@example.com", [(6 * 365 + 300, 4_000_000)])  # (under 7 years: stays)
    now = db.now()
    for uid, age, status in ((old, 8 * 365, "succeeded"), (old, 8 * 365, "pending"), (young, 2 * 365, "succeeded")):
        db.execute("INSERT INTO payments(id,user_id,kind,status,amount_cents,created_at,updated_at) VALUES (?,?,'checkout',?,500,?,?)",
                   (db.new_id(), uid, status, now - age * DAY, now - age * DAY))
    db.execute("INSERT INTO audit_log(user_id,actor,action,created_at) VALUES (?,'x','refund',?)", (old, now - 8 * 365 * DAY))
    db.execute("INSERT INTO audit_log(user_id,actor,action,created_at) VALUES (?,'x','refund',?)", (old, now - 5 * DAY))
    return old, young


def counts():
    return {t: db.one(f"SELECT COUNT(*) AS n FROM {t}")["n"] for t in ("ledger", "payments", "audit_log")}


def test_compact_money_without_apply_only_says_what_would_go(capsys):
    old, young = old_and_new()
    before, bal = counts(), (balance(old), balance(young))
    out, code = run(capsys, "compact-money")
    assert code is None and "older than 7 years: {'ledger': 2, 'payments': 1," in out
    # (two old lines become one carried-forward line: one row goes; the pending payment is not counted)
    assert "would remove (add --apply to do it):" in out and "'users': 1, 'ledger_rows': 1, 'payments': 1" in out and "removed:" not in out and "check-money" not in out
    assert counts() == before and (balance(old), balance(young)) == bal


def test_compact_money_with_apply_removes_only_what_is_over_seven_years_and_every_balance_stays_right(capsys):
    old, young = old_and_new()
    bal = (balance(old), balance(young))
    after_ids = [r["id"] for r in db.all_rows("SELECT id FROM ledger WHERE user_id = ? AND created_at >= ? ORDER BY id", (old, db.now() - 7 * YEAR))]
    out, code = run(capsys, "compact-money", "--apply")
    assert code is None and "removed:  {'users': 1, 'ledger_rows': 1, 'payments': 1, 'audit_log': 1" in out and "check-money: {'balance_mismatch': 0, 'held_mismatch': 0}" in out
    assert (balance(old), balance(young)) == bal  # (balances are what they were)
    assert ledger.invariant_problems() == {"balance_mismatch": 0, "held_mismatch": 0}
    # the two old rows of the old account are one "carried forward" row; its newer rows and the young account are untouched
    rows = db.all_rows("SELECT id, ref_type, amount_micros FROM ledger WHERE user_id = ? ORDER BY id", (old,))
    assert rows[0]["ref_type"] == "carry_forward" and rows[0]["amount_micros"] == 4_000_000 and [r["id"] for r in rows[1:]] == after_ids
    assert db.one("SELECT COUNT(*) AS n FROM ledger WHERE user_id = ?", (young,))["n"] == 1
    assert counts()["payments"] == 2   # (the pending payment and the young one stay)
    assert db.one("SELECT COUNT(*) AS n FROM audit_log WHERE action = 'refund'")["n"] == 1  # (the old refund line went, the recent one stays)
    assert {r["status"] for r in db.all_rows("SELECT status FROM payments")} == {"pending", "succeeded"}
    # again: nothing left to remove
    out, code = run(capsys, "compact-money", "--apply")
    assert code is None and "'users': 0" in out and "'ledger_rows': 0" in out


def test_compact_money_exits_1_when_the_books_do_not_balance_afterwards(capsys):
    money_user("fine@example.com", [(8 * 365, 1_000_000)])
    other = money_user("broken@example.com", [(10, 1_000_000)])
    db.execute("UPDATE users SET balance_micros = balance_micros + 3 WHERE id = ?", (other,))  # (a mismatch that is no business of compaction)
    out, code = run(capsys, "compact-money", "--apply")
    assert code == 1 and "check-money: {'balance_mismatch': 1" in out
