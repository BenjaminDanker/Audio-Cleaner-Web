"""Operator CLI: `python -m acw.admin <command>`.

  init-db                        create/upgrade the schema
  login-link EMAIL               print a one-time sign-in URL (creates the account)
  grant EMAIL USD REASON         add promotional/support credit
  user EMAIL                     show balance, status, recent activity
  suspend EMAIL REASON / unsuspend EMAIL
  stats [--days N]               plain-text report for the last N days (default 30): jobs by tier, queue wait and run
                                 time percentiles, failures, live sessions, usage (read-only)
  activity                       what is running now (file jobs, live sessions)
  live-count                     number of live sessions running now
  check-money                    balances and holds against the ledger (exit 1 on a mismatch)
  verify-backup FILE             integrity + ledger check of a .sqlite3.gz backup
  retention                      apply the retention rules now (the web process does it hourly) and print the counts
  storage                        sizes of what grows: database, job files, data directory, backups
  compact-money [--apply]        records older than 7 years: what would be removed (and, with --apply, remove it; yearly)
  mirror [--once] [--force] [--channel C] [--verify] [--status]
                                 copy the newest OBS plugin release from the build service, check it and publish it
                                 for download (acw-mirror.timer runs it every
                                 10 minutes); --status shows what is served, --verify hashes every served file
  backup DIR [--keep N]          consistent gzip'd DB snapshot, checked after writing; keeps the newest N
                                 (optional encrypted copy off the machine: see acw/backup.py)
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from . import auth, db
from .billing import ledger


def _user(email: str) -> dict:
    row = db.one("SELECT * FROM users WHERE email = ?", (email.strip().lower(),))
    if row is None:
        sys.exit(f"no user {email}")
    return dict(row)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="acw-admin")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db")
    p = sub.add_parser("login-link")
    p.add_argument("email")
    p = sub.add_parser("grant")
    p.add_argument("email")
    p.add_argument("usd", type=float)
    p.add_argument("reason")
    p.add_argument("--json", action="store_true", help="print the new balance as JSON")
    p = sub.add_parser("take", help="remove credit from an account (support correction); the reason shows in their history")
    p.add_argument("email")
    p.add_argument("usd", type=float)
    p.add_argument("reason")
    p.add_argument("--json", action="store_true", help="print the new balance as JSON")
    p = sub.add_parser("user")
    p.add_argument("email")
    p.add_argument("--json", action="store_true", help="the account, its balance, recent ledger and jobs as JSON")
    p = sub.add_parser("users", help="list accounts (newest first), optionally matching an email fragment")
    p.add_argument("--search", default="")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("suspend")
    p.add_argument("email")
    p.add_argument("reason")
    p = sub.add_parser("unsuspend")
    p.add_argument("email")
    p = sub.add_parser("stats", help="plain-text report from the database (read-only)")
    p.add_argument("--days", type=int, default=30, help="period in days (default 30)")
    sub.add_parser("activity", help="print what is running now: file jobs and live sessions")
    p = sub.add_parser("capacity", help="the whole machine in one place: CPU per core, memory, live engines, jobs, queue")
    p.add_argument("--json", action="store_true")
    sub.add_parser("live-count", help="print the number of live sessions running now")
    sub.add_parser("check-money", help="verify balances and holds against the ledger")
    p = sub.add_parser("verify-backup", help="check a backup file (.sqlite3.gz): integrity and ledger sums")
    p.add_argument("file")
    sub.add_parser("retention", help="apply the retention rules now and print how many rows each changed")
    sub.add_parser("storage", help="sizes of the database, job files, data directory and backups")
    p = sub.add_parser("compact-money", help="payment/ledger records older than 7 years: show, or with --apply remove (balances stay correct)")
    p.add_argument("--apply", action="store_true")
    p = sub.add_parser("mirror", help="mirror the release files from the build service and publish them for download")
    p.add_argument("--once", action="store_true", help="one pass (the default; the timer's unit says it explicitly)")
    p.add_argument("--force", action="store_true", help="fetch again even if the served version looks current or newer")
    p.add_argument("--channel", choices=("obs-plugin",), help="only this channel")
    p.add_argument("--verify", action="store_true", help="hash every served file against its recorded checksum; no network")
    p.add_argument("--status", action="store_true", help="print what is served and when it was last mirrored; no network")
    p = sub.add_parser("backup")
    p.add_argument("dir")
    p.add_argument("--keep", type=int, default=30)
    a = ap.parse_args(argv)
    db.init_db()

    if a.cmd == "init-db":
        print("ok")
    elif a.cmd == "login-link":
        from .config import get_settings

        auth.get_or_create_user(a.email)
        token = auth.create_login_token(a.email, "admin_login", "cli")
        print(get_settings().url(f"/login/verify#token={token}"))
        print("(valid once, for 15 minutes)")
    elif a.cmd == "grant":
        u = _user(a.email)
        if not 0 < a.usd <= 1000:
            sys.exit("amount must be between 0 and 1000 USD")
        ledger.credit(u["id"], int(round(a.usd * 1_000_000)), "grant", "admin", f"grant-{int(time.time() * 1000)}", a.reason[:200])
        db.audit("admin-cli", "grant", user_id=u["id"], detail=f"{a.usd} {a.reason}")
        print(json.dumps(ledger.balance(u["id"])) if a.json else ledger.balance(u["id"]))
    elif a.cmd == "take":
        u = _user(a.email)
        if not 0 < a.usd <= 1000:
            sys.exit("amount must be between 0 and 1000 USD")
        # a negative grant: it comes off what we gave (refund rules subtract grants), never below what is held for running work
        taken = ledger.debit_available(u["id"], int(round(a.usd * 1_000_000)), "grant", "admin", f"take-{int(time.time() * 1000)}",
                                       a.reason[:200], allow_partial=True)
        db.audit("admin-cli", "take", user_id=u["id"], detail=f"{taken / 1_000_000:.2f} of {a.usd} {a.reason}")
        out = {**ledger.balance(u["id"]), "taken_usd": taken / 1_000_000}
        print(json.dumps(out) if a.json else out)
    elif a.cmd == "user":
        u = _user(a.email)
        info = {k: u[k] for k in ("id", "email", "status", "is_admin", "created_at", "stripe_customer_id", "auto_reload_enabled")}
        if a.json:
            jobs = [dict(r) for r in db.all_rows(
                "SELECT id, kind, status, created_at, finished_at, cost_micros, duration_seconds FROM jobs WHERE user_id = ? ORDER BY created_at DESC LIMIT 20",
                (u["id"],))]
            live = [dict(r) for r in db.all_rows(
                "SELECT id, denoise_tier, caption_tier, status, started_at, seconds, cost_micros FROM stream_sessions WHERE user_id = ? ORDER BY started_at DESC LIMIT 10",
                (u["id"],))]
            print(json.dumps({"user": info, "balance": ledger.balance(u["id"]), "ledger": ledger.history(u["id"], 30), "jobs": jobs, "live": live}, default=str))
        else:
            print(info)
            print(ledger.balance(u["id"]))
            for r in ledger.history(u["id"], 10):
                print(r)
    elif a.cmd == "users":
        like = f"%{a.search.strip().lower()}%"
        rows = [dict(r) for r in db.all_rows(
            "SELECT u.email, u.status, u.created_at, u.balance_micros, u.held_micros,"
            " (SELECT COUNT(*) FROM jobs j WHERE j.user_id = u.id) AS jobs,"
            " (SELECT MAX(created_at) FROM jobs j WHERE j.user_id = u.id) AS last_job"
            " FROM users u WHERE u.status != 'deleted' AND u.email LIKE ? ORDER BY u.created_at DESC LIMIT ?", (like, max(1, min(a.limit, 500))))]
        print(json.dumps(rows) if a.json else "\n".join(f"{r['email']}  {r['status']}  ${r['balance_micros'] / 1e6:.2f}" for r in rows))
    elif a.cmd == "suspend":
        u = _user(a.email)
        with db.tx() as c:
            c.execute("UPDATE users SET status = 'suspended' WHERE id = ?", (u["id"],))
            ledger.release_all_holds(u["id"], c)
            c.execute("DELETE FROM sessions WHERE user_id = ?", (u["id"],))
        db.audit("admin-cli", "suspend", user_id=u["id"], detail=a.reason)
        print("suspended")
    elif a.cmd == "unsuspend":
        u = _user(a.email)
        db.execute("UPDATE users SET status = 'active' WHERE id = ?", (u["id"],))
        db.audit("admin-cli", "unsuspend", user_id=u["id"])
        print("active")
    elif a.cmd == "mirror":
        from . import mirror

        sys.exit(mirror.main(force=a.force, verify=a.verify, status=a.status, only=a.channel))
    elif a.cmd == "backup":
        from pathlib import Path

        from . import backup

        try:
            print(backup.run(Path(a.dir), a.keep))
        except backup.BackupError as e:
            sys.exit(f"backup problem: {e}")
    elif a.cmd == "activity":
        rows = {r["status"]: r["n"] for r in db.all_rows(
            "SELECT status, COUNT(*) AS n FROM jobs WHERE status IN ('uploading', 'queued', 'running', 'cloud_asr') GROUP BY status")}
        live = db.one("SELECT COUNT(*) AS n FROM stream_sessions WHERE status = 'active'")["n"]
        print("file jobs: " + (", ".join(f"{n} {s}" for s, n in sorted(rows.items())) or "none") + f"; live sessions: {live}")
    elif a.cmd == "capacity":
        from . import broker

        sn = broker.snapshot()
        print(json.dumps(sn, indent=1) if a.json else broker.format_snapshot(sn))
    elif a.cmd == "live-count":
        print(db.one("SELECT COUNT(*) AS n FROM stream_sessions WHERE status = 'active'")["n"])
    elif a.cmd == "check-money":
        problems = ledger.invariant_problems()
        print(problems)
        if any(problems.values()):
            sys.exit(1)
    elif a.cmd == "verify-backup":
        from pathlib import Path

        from . import backup

        result = backup.verify(Path(a.file))
        print(result)
        if result["integrity"] != "ok" or result["ledger"] != "ok":
            sys.exit(1)
    elif a.cmd == "retention":
        from . import retention

        print(retention.run())
    elif a.cmd == "storage":
        from . import monitor, storage
        from .config import get_settings

        for k, v in storage.sizes(get_settings().data_dir, monitor.BACKUP_DIR).items():
            print(f"{k:10s} {v / 1024**2:10.1f} MB")
        print(f"alert limits: data {storage.data_limit() / 1024**3:.0f} GB (without jobs), backups {storage.backups_limit() / 1024**3:.0f} GB")
    elif a.cmd == "compact-money":
        from . import retention

        before = retention.old_money_records()
        print("older than 7 years:", before)
        res = retention.compact_money(apply=a.apply)
        print(("removed: " if a.apply else "would remove (add --apply to do it): "), res)
        if a.apply:
            problems = ledger.invariant_problems()
            print("check-money:", problems)
            if any(problems.values()):
                sys.exit(1)
    elif a.cmd == "stats":
        from . import stats

        if not 1 <= a.days <= 3650:
            sys.exit("--days must be between 1 and 3650")
        print(stats.render(stats.report(a.days)))


if __name__ == "__main__":
    main()
