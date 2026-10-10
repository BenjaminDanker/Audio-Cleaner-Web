"""Operator CLI: `python -m acw.admin <command>`.

  init-db                        create/upgrade the schema
  login-link EMAIL               print a one-time sign-in URL (creates the account)
  grant EMAIL USD REASON         add promotional/support credit
  user EMAIL                     show balance, status, recent activity
  suspend EMAIL REASON / unsuspend EMAIL
  stats                          totals for the last 30 days
  backup DIR [--keep N]          consistent gzip'd DB snapshot, keeps the newest N
"""
from __future__ import annotations

import argparse
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
    p = sub.add_parser("user")
    p.add_argument("email")
    p = sub.add_parser("suspend")
    p.add_argument("email")
    p.add_argument("reason")
    p = sub.add_parser("unsuspend")
    p.add_argument("email")
    sub.add_parser("stats")
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
        print(ledger.balance(u["id"]))
    elif a.cmd == "user":
        u = _user(a.email)
        print({k: u[k] for k in ("id", "email", "status", "is_admin", "created_at", "stripe_customer_id", "auto_reload_enabled")})
        print(ledger.balance(u["id"]))
        for r in ledger.history(u["id"], 10):
            print(r)
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
    elif a.cmd == "backup":
        import gzip
        import os
        import shutil
        import sqlite3
        from pathlib import Path

        out_dir = Path(a.dir)
        out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        tmp = out_dir / f".acw-{stamp}.sqlite3"
        dst = sqlite3.connect(str(tmp))
        db.conn().backup(dst)  # online, consistent snapshot
        dst.close()
        final = out_dir / f"acw-{stamp}.sqlite3.gz"
        with open(tmp, "rb") as src, gzip.open(final, "wb", compresslevel=6) as gz:
            shutil.copyfileobj(src, gz)
        os.chmod(final, 0o600)
        tmp.unlink()
        backups = sorted(out_dir.glob("acw-*.sqlite3.gz"))
        for old in backups[: max(0, len(backups) - a.keep)]:
            old.unlink()
        print(final)
    elif a.cmd == "stats":
        since = db.now() - 30 * 86400
        print("users", db.one("SELECT COUNT(*) n FROM users")["n"])
        print("topups $", (db.one("SELECT COALESCE(SUM(amount_cents),0) s FROM payments WHERE status='succeeded' AND created_at>=?", (since,))["s"]) / 100)
        print("usage $", -(db.one("SELECT COALESCE(SUM(amount_micros),0) s FROM ledger WHERE kind='usage' AND created_at>=?", (since,))["s"]) / 1e6)
        print("jobs", dict(db.all_rows("SELECT status, COUNT(*) FROM jobs WHERE created_at>=? GROUP BY status", (since,))))
        print("stream minutes", (db.one("SELECT COALESCE(SUM(seconds),0) s FROM stream_sessions WHERE started_at>=?", (since,))["s"]) / 60)


if __name__ == "__main__":
    main()
