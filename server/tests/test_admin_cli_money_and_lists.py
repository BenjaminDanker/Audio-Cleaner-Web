"""The operator's newer commands (acw/admin.py, run through main(argv) against the test database): grant --json, take (credit
removed as a negative grant, never below what is held), user --json and users (search the accounts)."""
import json

import pytest

from acw import admin, auth, db
from acw.billing import ledger

from .job_support import add_job


def run(capsys, *argv):
    """(what it printed, its exit: None when it returned, else the code or message)."""
    try:
        admin.main(list(argv))
        code = None
    except SystemExit as e:
        code = e.code
    return capsys.readouterr().out, code


def user(email="a@example.com", created_at=None) -> str:
    uid = auth.get_or_create_user(email)["id"]
    if created_at is not None:
        db.execute("UPDATE users SET created_at = ? WHERE id = ?", (created_at, uid))
    return uid


def bal(uid) -> dict:
    return ledger.balance(uid)


def give(uid, micros):
    ledger.credit(uid, micros, "grant", "t", f"seed-{micros}-{db.new_id()}", "seed")


# ------------------------------------------------------------------------------------------------------------------ grant --json

def test_grant_json_prints_the_balance_dict_as_json(capsys):
    uid = user()
    out, code = run(capsys, "grant", "a@example.com", "2.5", "thanks", "--json")
    assert code is None
    assert json.loads(out) == {"balance_micros": 2_500_000, "held_micros": 0, "available_micros": 2_500_000}
    assert out.strip().startswith('{"')  # (JSON with double quotes, not a Python repr)
    assert bal(uid)["balance_micros"] == 2_500_000


def test_grant_without_json_still_prints_the_python_dict(capsys):
    user()
    out, _ = run(capsys, "grant", "a@example.com", "1", "r")
    assert out.strip().startswith("{'balance_micros': 1000000")


# -------------------------------------------------------------------------------------------------------------------------- take

def test_take_removes_credit_as_a_negative_grant_and_audits(capsys):
    uid = user()
    give(uid, 10_000_000)
    out, code = run(capsys, "take", "A@Example.com", "4.25", "support: wrong grant", "--json")
    assert code is None
    assert json.loads(out) == {"balance_micros": 5_750_000, "held_micros": 0, "available_micros": 5_750_000, "taken_usd": 4.25}
    row = db.one("SELECT amount_micros, kind, ref_type, ref_id, description, balance_after_micros FROM ledger WHERE user_id = ? ORDER BY id DESC LIMIT 1", (uid,))
    assert (row["kind"], row["amount_micros"], row["ref_type"]) == ("grant", -4_250_000, "admin")
    assert row["ref_id"].startswith("take-") and row["ref_id"][5:].isdigit()
    assert row["description"] == "support: wrong grant" and row["balance_after_micros"] == 5_750_000
    audit = db.one("SELECT actor, detail FROM audit_log WHERE action = 'take' AND user_id = ?", (uid,))
    assert audit["actor"] == "admin-cli" and audit["detail"] == "4.25 of 4.25 support: wrong grant"
    assert ledger.invariant_problems() == {"balance_mismatch": 0, "held_mismatch": 0}


def test_take_without_json_prints_the_python_dict_with_taken_usd(capsys):
    uid = user()
    give(uid, 3_000_000)
    out, code = run(capsys, "take", "a@example.com", "1", "r")
    assert code is None and out.strip().startswith("{'balance_micros': 2000000") and "'taken_usd': 1.0" in out
    assert bal(uid)["balance_micros"] == 2_000_000


def test_take_lowers_what_counts_as_granted_credit(capsys):
    uid = user()
    give(uid, 5_000_000)
    run(capsys, "take", "a@example.com", "2", "r")
    assert ledger.granted_micros(uid) == 3_000_000


def test_take_more_than_is_held_takes_only_what_there_is(capsys):
    uid = user()
    give(uid, 1_500_000)
    out, code = run(capsys, "take", "a@example.com", "10", "r", "--json")
    d = json.loads(out)
    assert code is None and d["taken_usd"] == 1.5 and d["balance_micros"] == 0 and d["available_micros"] == 0
    row = db.one("SELECT amount_micros FROM ledger WHERE user_id = ? ORDER BY id DESC LIMIT 1", (uid,))
    assert row["amount_micros"] == -1_500_000
    audit = db.one("SELECT detail FROM audit_log WHERE action = 'take' AND user_id = ?", (uid,))
    assert audit["detail"] == "1.50 of 10.0 r"  # (what was taken, of what was asked)


def test_take_from_an_empty_account_takes_nothing_writes_no_ledger_line_but_audits(capsys):
    uid = user()
    out, code = run(capsys, "take", "a@example.com", "5", "r", "--json")
    d = json.loads(out)
    assert code is None and d["taken_usd"] == 0 and d["balance_micros"] == 0
    assert db.one("SELECT COUNT(*) AS n FROM ledger WHERE user_id = ?", (uid,))["n"] == 0
    assert db.one("SELECT detail FROM audit_log WHERE action = 'take' AND user_id = ?", (uid,))["detail"] == "0.00 of 5.0 r"


def test_take_never_touches_money_that_is_held(capsys):
    uid = user()
    give(uid, 10_000_000)
    ledger.create_hold(uid, 7_000_000, "job", "j1")
    out, code = run(capsys, "take", "a@example.com", "9", "r", "--json")
    d = json.loads(out)
    assert code is None and d["taken_usd"] == 3.0
    assert (d["balance_micros"], d["held_micros"], d["available_micros"]) == (7_000_000, 7_000_000, 0)
    # and with everything held, nothing is taken at all
    out, code = run(capsys, "take", "a@example.com", "1", "r", "--json")
    d = json.loads(out)
    assert code is None and d["taken_usd"] == 0 and d["balance_micros"] == 7_000_000 and d["held_micros"] == 7_000_000
    assert ledger.invariant_problems() == {"balance_mismatch": 0, "held_mismatch": 0}


def test_take_the_exact_available_amount(capsys):
    uid = user()
    give(uid, 4_000_000)
    out, _ = run(capsys, "take", "a@example.com", "4", "r", "--json")
    assert json.loads(out)["taken_usd"] == 4.0 and bal(uid)["balance_micros"] == 0


def test_take_the_most_a_thousand_dollars(capsys):
    uid = user()
    give(uid, 1_500_000_000)
    out, code = run(capsys, "take", "a@example.com", "1000", "r", "--json")
    assert code is None and json.loads(out)["taken_usd"] == 1000.0 and bal(uid)["balance_micros"] == 500_000_000


@pytest.mark.parametrize("usd", ["0", "-5", "-0.01", "1000.01", "5000", "nan", "inf"])
def test_take_refuses_an_amount_outside_zero_to_a_thousand_dollars_and_changes_nothing(capsys, usd):
    uid = user()
    give(uid, 10_000_000)
    n = db.one("SELECT COUNT(*) AS n FROM ledger")["n"]
    out, code = run(capsys, "take", "a@example.com", usd, "r")
    assert code == "amount must be between 0 and 1000 USD" and out == ""
    assert bal(uid)["balance_micros"] == 10_000_000 and db.one("SELECT COUNT(*) AS n FROM ledger")["n"] == n
    assert db.one("SELECT COUNT(*) AS n FROM audit_log WHERE action = 'take'")["n"] == 0


def test_take_a_negative_amount_given_after_the_double_dash_is_refused_too(capsys):
    uid = user()
    give(uid, 1_000_000)
    assert run(capsys, "take", "a@example.com", "--", "-1", "r")[1] == "amount must be between 0 and 1000 USD"
    assert bal(uid)["balance_micros"] == 1_000_000


def test_take_a_long_reason_is_cut_to_200_characters(capsys):
    uid = user()
    give(uid, 1_000_000)
    run(capsys, "take", "a@example.com", "1", "y" * 500)
    assert db.one("SELECT description FROM ledger WHERE user_id = ? ORDER BY id DESC LIMIT 1", (uid,))["description"] == "y" * 200


def test_two_takes_each_write_their_own_line(capsys):
    uid = user()
    give(uid, 10_000_000)
    run(capsys, "take", "a@example.com", "1", "one")
    run(capsys, "take", "a@example.com", "2", "two")
    rows = db.all_rows("SELECT amount_micros, description FROM ledger WHERE user_id = ? AND amount_micros < 0 ORDER BY id", (uid,))
    assert [(r["amount_micros"], r["description"]) for r in rows] == [(-1_000_000, "one"), (-2_000_000, "two")]
    assert bal(uid)["balance_micros"] == 7_000_000


def test_take_from_an_unknown_account_exits_with_no_user_and_changes_nothing(capsys):
    out, code = run(capsys, "take", "nobody@example.com", "5", "r")
    assert code == "no user nobody@example.com" and out == ""
    assert db.one("SELECT COUNT(*) AS n FROM audit_log WHERE action = 'take'")["n"] == 0


def test_take_only_affects_the_named_account(capsys):
    a, b = user("a@example.com"), user("b@example.com")
    give(a, 5_000_000)
    give(b, 5_000_000)
    run(capsys, "take", "a@example.com", "2", "r")
    assert bal(a)["balance_micros"] == 3_000_000 and bal(b)["balance_micros"] == 5_000_000


# ------------------------------------------------------------------------------------------------------------------ user --json

def test_user_json_has_the_account_balance_ledger_jobs_and_live_sessions(capsys):
    uid = user()
    give(uid, 2_000_000)
    jid = add_job(uid, "done", created_ago=100, finished_ago=50)
    db.execute("INSERT INTO stream_sessions(id,user_id,denoise_tier,caption_tier,status,seconds,cost_micros,started_at) VALUES ('s1',?,'light','premium','ended',12.5,34,?)",
               (uid, db.now() - 10))
    out, code = run(capsys, "user", "a@example.com", "--json")
    d = json.loads(out)
    assert code is None and set(d) == {"user", "balance", "ledger", "jobs", "live"}
    assert set(d["user"]) == {"id", "email", "status", "is_admin", "created_at", "stripe_customer_id", "auto_reload_enabled"}
    assert d["user"]["id"] == uid and d["user"]["email"] == "a@example.com" and d["user"]["status"] == "active"
    assert d["balance"] == {"balance_micros": 2_000_000, "held_micros": 0, "available_micros": 2_000_000}
    assert len(d["ledger"]) == 1 and d["ledger"][0]["kind"] == "grant" and d["ledger"][0]["amount_micros"] == 2_000_000
    assert len(d["jobs"]) == 1 and d["jobs"][0]["id"] == jid and d["jobs"][0]["status"] == "done"
    assert set(d["jobs"][0]) == {"id", "kind", "status", "created_at", "finished_at", "cost_micros", "duration_seconds"}
    assert len(d["live"]) == 1 and d["live"][0]["id"] == "s1" and d["live"][0]["seconds"] == 12.5 and d["live"][0]["cost_micros"] == 34
    assert set(d["live"][0]) == {"id", "denoise_tier", "caption_tier", "status", "started_at", "seconds", "cost_micros"}


def test_user_json_of_a_fresh_account_has_empty_lists(capsys):
    user()
    d = json.loads(run(capsys, "user", "a@example.com", "--json")[0])
    assert d["ledger"] == [] and d["jobs"] == [] and d["live"] == []


def test_user_json_limits_and_orders_ledger_jobs_and_live(capsys):
    uid = user()
    other = user("b@example.com")
    for i in range(35):
        ledger.credit(uid, 1_000_000, "grant", "t", f"g{i}", f"line {i}")
    ids = [add_job(uid, "done", created_ago=1000 - i, finished_ago=0) for i in range(25)]  # i larger = newer
    add_job(other, "done")  # (someone else's job is not listed)
    for i in range(13):
        db.execute("INSERT INTO stream_sessions(id,user_id,status,started_at) VALUES (?,?,'ended',?)", (f"s{i}", uid, 1_000_000 + i))
    db.execute("INSERT INTO stream_sessions(id,user_id,status,started_at) VALUES ('other',?,'ended',2000000)", (other,))
    d = json.loads(run(capsys, "user", "a@example.com", "--json")[0])
    assert [r["description"] for r in d["ledger"]] == [f"line {i}" for i in range(34, 4, -1)]  # 30, newest first
    assert len(d["jobs"]) == 20 and [j["id"] for j in d["jobs"]] == ids[::-1][:20]
    assert [j["created_at"] for j in d["jobs"]] == sorted((j["created_at"] for j in d["jobs"]), reverse=True)
    assert [s["id"] for s in d["live"]] == [f"s{i}" for i in range(12, 2, -1)]  # 10, newest first


def test_user_json_balance_shows_a_hold(capsys):
    uid = user()
    give(uid, 3_000_000)
    ledger.create_hold(uid, 1_000_000, "job", "j1")
    d = json.loads(run(capsys, "user", "a@example.com", "--json")[0])
    assert d["balance"] == {"balance_micros": 3_000_000, "held_micros": 1_000_000, "available_micros": 2_000_000}


def test_user_without_json_is_unchanged_python_text(capsys):
    uid = user()
    out, code = run(capsys, "user", "a@example.com")
    assert code is None and out.startswith("{'id': '" + uid) and not out.lstrip().startswith('{"')


def test_user_json_for_an_unknown_account_exits_with_no_user(capsys):
    assert run(capsys, "user", "nobody@example.com", "--json") == ("", "no user nobody@example.com")


# -------------------------------------------------------------------------------------------------------------------------- users

def seed_three():
    user("old@example.com", 1000)
    user("mid@Example.org", 2000)
    user("new@example.com", 3000)


def test_users_lists_newest_first_as_json_with_job_counts(capsys):
    seed_three()
    mid = db.one("SELECT id FROM users WHERE email = 'mid@example.org'")["id"]
    db.execute("UPDATE users SET balance_micros = 2500000 WHERE id = ?", (mid,))
    j1 = add_job(mid, "done", created_ago=500, finished_ago=0)
    j2 = add_job(mid, "done", created_ago=100, finished_ago=0)
    out, code = run(capsys, "users", "--json")
    rows = json.loads(out)
    assert code is None and [r["email"] for r in rows] == ["new@example.com", "mid@example.org", "old@example.com"]
    assert set(rows[0]) == {"email", "status", "created_at", "balance_micros", "held_micros", "jobs", "last_job"}
    assert [(r["jobs"], r["last_job"]) for r in rows[:1] + rows[2:]] == [(0, None), (0, None)]
    m = rows[1]
    assert m["jobs"] == 2 and m["balance_micros"] == 2_500_000 and m["held_micros"] == 0 and m["created_at"] == 2000 and m["status"] == "active"
    assert m["last_job"] == db.one("SELECT MAX(created_at) AS m FROM jobs WHERE id IN (?, ?)", (j1, j2))["m"]


def test_users_counts_only_each_accounts_own_jobs(capsys):
    a, b = user("a@example.com", 1), user("b@example.com", 2)
    add_job(a)
    add_job(a)
    add_job(b)
    rows = {r["email"]: r for r in json.loads(run(capsys, "users", "--json")[0])}
    assert (rows["a@example.com"]["jobs"], rows["b@example.com"]["jobs"]) == (2, 1)


def test_users_search_matches_an_email_fragment_case_insensitively(capsys):
    seed_three()
    emails = lambda *argv: [r["email"] for r in json.loads(run(capsys, "users", "--json", *argv)[0])]
    assert emails("--search", "EXAMPLE.com") == ["new@example.com", "old@example.com"]
    assert emails("--search", "mid") == ["mid@example.org"]
    assert emails("--search", "  OLD  ") == ["old@example.com"]  # (the fragment is trimmed)
    assert emails("--search", "@example") == ["new@example.com", "mid@example.org", "old@example.com"]
    assert emails("--search", "zzz") == []
    assert emails("--search", "") == ["new@example.com", "mid@example.org", "old@example.com"]


def test_users_search_only_looks_at_the_email(capsys):
    uid = user("a@example.com")
    db.execute("UPDATE users SET display_name = 'findme' WHERE id = ?", (uid,))
    assert json.loads(run(capsys, "users", "--json", "--search", "findme")[0]) == []


def test_users_leaves_out_deleted_accounts_but_lists_suspended_ones(capsys):
    seed_three()
    db.execute("UPDATE users SET status = 'deleted' WHERE email = 'mid@example.org'")
    db.execute("UPDATE users SET status = 'suspended' WHERE email = 'old@example.com'")
    rows = json.loads(run(capsys, "users", "--json")[0])
    assert [(r["email"], r["status"]) for r in rows] == [("new@example.com", "active"), ("old@example.com", "suspended")]
    assert json.loads(run(capsys, "users", "--json", "--search", "mid")[0]) == []


def test_users_limit_is_applied_and_clamped(capsys):
    for i in range(3):
        user(f"u{i}@example.com", 100 + i)
    n = lambda lim: len(json.loads(run(capsys, "users", "--json", "--limit", str(lim))[0]))
    assert n(2) == 2 and n(1) == 1 and n(3) == 3 and n(1000) == 3
    assert n(0) == 1 and n(-5) == 1  # (clamped up to 1)
    first = json.loads(run(capsys, "users", "--json", "--limit", "1")[0])
    assert [r["email"] for r in first] == ["u2@example.com"]  # (the newest)


def test_users_default_limit_is_fifty_and_the_most_is_five_hundred(capsys):
    db.execute("INSERT INTO users(id, email, created_at) VALUES ('seed-0', 'seed0@example.com', 0)")  # (a row to copy the defaults)
    with db.tx() as c:
        c.executemany("INSERT INTO users(id, email, created_at) VALUES (?,?,?)", [(f"x{i}", f"x{i}@example.com", 10 + i) for i in range(520)])
    assert len(json.loads(run(capsys, "users", "--json")[0])) == 50
    assert len(json.loads(run(capsys, "users", "--json", "--limit", "9999")[0])) == 500
    assert len(json.loads(run(capsys, "users", "--json", "--limit", "500")[0])) == 500
    assert len(json.loads(run(capsys, "users", "--json", "--limit", "499")[0])) == 499


def test_users_text_output_is_one_line_per_account(capsys):
    seed_three()
    db.execute("UPDATE users SET balance_micros = 1234500 WHERE email = 'mid@example.org'")
    out, code = run(capsys, "users")
    assert code is None and out == "new@example.com  active  $0.00\nmid@example.org  active  $1.23\nold@example.com  active  $0.00\n"
    assert not out.lstrip().startswith("[")


def test_users_text_output_with_no_match_prints_nothing_and_json_prints_an_empty_list(capsys):
    seed_three()
    assert run(capsys, "users", "--search", "zzz")[0].strip() == ""
    assert run(capsys, "users", "--json", "--search", "zzz") == ("[]\n", None)


def test_users_json_and_text_agree_on_the_accounts(capsys):
    seed_three()
    rows = json.loads(run(capsys, "users", "--json")[0])
    text = run(capsys, "users")[0].splitlines()
    assert [l.split("  ")[0] for l in text] == [r["email"] for r in rows]


@pytest.mark.xfail(strict=True, reason="product bug: admin.py users builds LIKE from the raw fragment; _ and % act as wildcards")
def test_users_search_treats_percent_and_underscore_as_plain_characters(capsys):
    """Known product bug (xfail until fixed): the fragment goes into LIKE unescaped, so '_' matches any character and '%' matches anything."""
    user("a_b@example.com", 1)
    user("axb@example.com", 2)
    emails = lambda s: [r["email"] for r in json.loads(run(capsys, "users", "--json", "--search", s)[0])]
    assert emails("a_b") == ["a_b@example.com"]
    assert emails("%") == []
