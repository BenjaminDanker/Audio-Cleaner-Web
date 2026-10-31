"""Maintenance: while it is on, only admin accounts (ACW_ADMIN_EMAILS) get work done.

Everyone else's file jobs are still taken and wait in the queue (shown to them as paused for maintenance) and start when it ends;
a job that had already started finishes, later stages included. New live sessions from everyone else are refused with a short
message (a live session can't wait); running ones continue.

The switch is the `maintenance` row of the meta table, so the web, live and worker processes all see the same one. It is set with
`acw-admin maintenance on|off` (the home page's admin side runs that).
"""
from __future__ import annotations

import json

from . import db
from .config import get_settings

KEY = "maintenance"
MESSAGE = "Audio Cleaner is down for a short maintenance. Please try again in a little while."
JOB_MESSAGE = "Paused for a short maintenance: it starts as soon as we're back"


def state() -> dict | None:
    """{"since": unix time, "reason": text} while maintenance is on, else None."""
    row = db.one("SELECT value FROM meta WHERE key = ?", (KEY,))
    if row is None:
        return None
    try:
        value = json.loads(row["value"])
    except ValueError:
        value = None
    return value if isinstance(value, dict) else {}


def switch_on(reason: str = "", actor: str = "admin") -> dict:
    value = state() or {"since": db.now()}
    value["reason"] = reason[:200]
    db.execute("INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (KEY, json.dumps(value)))
    db.audit(actor, "maintenance_on", detail=value["reason"] or None)
    return value


def switch_off(actor: str = "admin") -> bool:
    n = db.execute("DELETE FROM meta WHERE key = ?", (KEY,)).rowcount
    if n:
        db.audit(actor, "maintenance_off")
    return bool(n)


def admin_ids() -> set[str]:
    """Admin accounts: marked at sign-in, or listed in ACW_ADMIN_EMAILS (an account that hasn't signed in since it was listed)."""
    emails = list(get_settings().admin_emails)
    marks = ",".join("?" * len(emails))
    sql = "SELECT id FROM users WHERE is_admin = 1" + (f" OR lower(email) IN ({marks})" if emails else "")
    return {r["id"] for r in db.all_rows(sql, tuple(emails))}


def blocks_user(user_id: str) -> bool:
    """Whether maintenance keeps this account's new work from starting."""
    return state() is not None and user_id not in admin_ids()


def _held(job: dict, admins: set[str]) -> bool:
    return job.get("status") == "queued" and job.get("started_at") is None and job.get("user_id") not in admins


def runnable(rows: list[dict]) -> list[dict]:
    """The queued jobs a worker may take now: all of them, or during maintenance the admins' and the ones already started."""
    if not rows or state() is None:
        return rows
    admins = admin_ids()
    return [r for r in rows if not _held(r, admins)]


def paused(job: dict) -> bool:
    """A queued job that waits for maintenance to end."""
    return job.get("status") == "queued" and job.get("started_at") is None and state() is not None and job.get("user_id") not in admin_ids()
