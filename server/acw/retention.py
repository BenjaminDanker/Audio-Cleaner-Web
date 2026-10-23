"""Privacy retention and bounded storage, as the Privacy Policy promises it (web/src/pages/Privacy.jsx, section 8).

  * network addresses of live sessions, the audit log and browser sessions: removed after 90 days
  * file names (jobs, and the credit history's usage lines): removed after 12 months
  * job records, live-session records, closed holds, routine audit rows, revoked API keys: deleted after 12 months
  * expired browser sessions, login links and sign-in states: deleted 30 days after they ran out
  * processed Stripe events: deleted after 90 days (Stripe retries for about 3 days); any other after 12 months
  * payment, ledger and refund records, and the audit rows that go with them: kept 7 years. NOTHING here touches
    them. `compact_money()` (python -m acw.admin compact-money, a yearly job for the owner once records are 7 years
    old) replaces a user's old ledger rows with one carried-forward row, so balances still add up.

`sweep()` is cheap and idempotent; the web process runs it from its housekeeping loop (at most hourly). Deletes run
in small batches so a first run over a big backlog never holds the write lock for long. The database file is
compacted by `maintain_storage()` (a monthly VACUUM at a quiet hour, never while a live session runs), because
SQLite keeps the space of deleted rows for itself and never gives it back to the disk otherwise.
"""
from __future__ import annotations

import logging
import re
import shutil
import time

from . import db

log = logging.getLogger("acw.retention")

DAY = 86400
IP_DAYS = 90
FILENAME_DAYS = 365
RECORD_DAYS = 365  # jobs, live sessions, closed holds, routine audit rows, revoked API keys
TOKEN_DAYS = 30  # expired browser sessions, login links, OAuth states
STRIPE_EVENT_DAYS = 90
STRIPE_EVENT_ANY_DAYS = 365
MONEY_DAYS = 7 * 365 + 2  # payments, ledger, refunds and what goes with them (two days for leap years)

BATCH = 2000
MAX_BATCHES = 50  # per rule and per sweep; the rest follows in the next hourly sweep

TERMINAL_JOB_STATES = ("done", "failed", "canceled", "expired")

# Audit rows: what is only an activity log (deleted after RECORD_DAYS) and what is evidence kept with the money
# records (consent, payments, refunds, grants, suspensions, deletions). An action nobody has classified is kept, never
# deleted; a test (tests/test_retention.py) fails when the code writes an action that is in neither list.
AUDIT_ROUTINE = ("login", "user_created", "api_key_created", "api_key_revoked", "api_key_cap_changed", "webhook_secret_rotated", "contact_sent")
AUDIT_WITH_MONEY = (
    "tos_accepted", "checkout_started", "auto_reload_enabled", "auto_reload_disabled", "auto_reload_failed",
    "payment_refused_region", "suspended", "suspend", "unsuspend", "refund", "refund_unrecovered",
    "payment_uncredited", "dispute_unrecovered", "efw_unrecovered", "grant", "account_deleted",
)

_last: float | None = None  # monotonic time of the last sweep (None: not yet since start)
_USAGE_DESC = re.compile(r"^(?P<label>[^:]+): .* (?P<mins>\(\d+(?:\.\d+)?\s*min\))$", re.S)
_LEGACY_LABELS = {
    "denoise": "Denoising",
    "transcribe": "Subtitles & transcript",
    "denoise + transcribe": "Denoising + Subtitles & transcript",
}


def _delete(table: str, where: str, params: tuple = ()) -> int:
    """DELETE ... WHERE in batches (each its own short transaction). Returns the number of rows removed."""
    total = 0
    for _ in range(MAX_BATCHES):
        n = db.execute(
            f"DELETE FROM {table} WHERE rowid IN (SELECT rowid FROM {table} WHERE {where} LIMIT {BATCH})", params
        ).rowcount
        total += n
        if n < BATCH:
            break
    return total


def strip_file_name(description: str) -> str | None:
    """A usage line's description without the file name ("Denoising: talk.wav (1.2 min)" ->
    "Denoising (1.2 min)"), or None when there is nothing to remove."""
    m = _USAGE_DESC.match(description or "")
    if not m:
        return None
    label = _LEGACY_LABELS.get(m["label"], m["label"])
    return f"{label} {m['mins']}"


def _blank_usage_names(cut: int) -> int:
    n = 0
    after = 0
    for _ in range(MAX_BATCHES):
        rows = db.all_rows(
            "SELECT id, description FROM ledger WHERE id > ? AND kind = 'usage' AND ref_type = 'job' AND created_at < ?"
            " AND description LIKE '%: %' ORDER BY id LIMIT ?", (after, cut, BATCH),
        )
        for r in rows:
            after = r["id"]
            new = strip_file_name(r["description"])
            if new is not None:
                db.execute("UPDATE ledger SET description = ? WHERE id = ?", (new, r["id"]))
                n += 1
        if len(rows) < BATCH:
            break
    return n


def run(now: int | None = None) -> dict[str, int]:
    """Apply the limits now. Returns how many rows each rule changed or removed."""
    now = db.now() if now is None else now
    ip_cut = now - IP_DAYS * DAY
    name_cut = now - FILENAME_DAYS * DAY
    rec_cut = now - RECORD_DAYS * DAY
    tok_cut = now - TOKEN_DAYS * DAY
    states = ",".join("?" * len(TERMINAL_JOB_STATES))
    routine = ",".join("?" * len(AUDIT_ROUTINE))
    done = {
        "stream_ips": db.execute(
            "UPDATE stream_sessions SET client_ip = NULL WHERE client_ip IS NOT NULL AND started_at < ?", (ip_cut,)
        ).rowcount,
        "audit_ips": db.execute(
            "UPDATE audit_log SET ip = NULL WHERE ip IS NOT NULL AND created_at < ?", (ip_cut,)
        ).rowcount,
        "session_ips": db.execute(
            "UPDATE sessions SET ip = NULL, user_agent = NULL WHERE (ip IS NOT NULL OR user_agent IS NOT NULL) AND created_at < ?", (ip_cut,)
        ).rowcount,
        "login_token_ips": db.execute(
            "UPDATE login_tokens SET ip = NULL WHERE ip IS NOT NULL AND created_at < ?", (ip_cut,)
        ).rowcount,
        "job_names": db.execute(
            "UPDATE jobs SET input_name = '', params_json = json_remove(params_json, '$.display_name')"
            " WHERE input_name != '' AND created_at < ?", (name_cut,)
        ).rowcount,
        "usage_names": _blank_usage_names(name_cut),
        # Whole rows. Nothing else points at a job, a live session or a hold by a foreign key; the credit history
        # shows an old usage line from the ledger's own text when the job row is gone.
        "jobs": _delete("jobs", f"status IN ({states}) AND created_at < ?", (*TERMINAL_JOB_STATES, rec_cut)),
        "stream_sessions": _delete("stream_sessions", "status = 'ended' AND started_at < ?", (rec_cut,)),
        "holds": _delete("holds", "status != 'active' AND COALESCE(closed_at, created_at) < ?", (rec_cut,)),
        "audit_log": _delete("audit_log", f"action IN ({routine}) AND created_at < ?", (*AUDIT_ROUTINE, rec_cut)),
        "api_keys": _delete("api_keys", "revoked_at IS NOT NULL AND revoked_at < ?", (rec_cut,)),
        "sessions": _delete("sessions", "expires_at < ?", (tok_cut,)),
        "login_tokens": _delete("login_tokens", "COALESCE(used_at, expires_at) < ? OR expires_at < ?", (tok_cut, tok_cut)),
        "oauth_states": _delete("oauth_states", "expires_at < ?", (tok_cut,)),
        "stripe_events": _delete(
            "stripe_events",
            "(processed_at IS NOT NULL AND received_at < ?) OR received_at < ?",
            (now - STRIPE_EVENT_DAYS * DAY, now - STRIPE_EVENT_ANY_DAYS * DAY),
        ),
    }
    return done


# ---------------------------------------------------------------- the database file

VACUUM_EVERY_DAYS = 28
VACUUM_MIN_FREE_PAGES = 2000  # about 8 MB: below this the file is not worth rewriting
VACUUM_HOURS = (2, 3)  # machine-local hours; the nightly snapshot starts at 04:17
_VACUUM_KEY = "retention.vacuumed_at"


def _page_facts() -> tuple[int, int, int]:
    pages = db.one("PRAGMA page_count")[0]
    free = db.one("PRAGMA freelist_count")[0]
    size = db.one("PRAGMA page_size")[0]
    return pages, free, size


def maintain_storage(now: int | None = None, *, hour: int | None = None) -> str | None:
    """Give deleted rows' space back to the disk: a VACUUM about once a month, in the small hours, when at least
    ~8 MB (or a quarter of the file) is free inside it, no live session is running and the disk has room for the
    copy SQLite makes. Returns what it did ("vacuumed") or why not (None when it is simply not due)."""
    now = db.now() if now is None else now
    hour = time.localtime(now).tm_hour if hour is None else hour
    if hour not in VACUUM_HOURS:
        return None
    last = db.one("SELECT value FROM meta WHERE key = ?", (_VACUUM_KEY,))
    if last is not None and now - int(last["value"]) < VACUUM_EVERY_DAYS * DAY:
        return None
    pages, free, size = _page_facts()
    if free < VACUUM_MIN_FREE_PAGES and free * 4 < pages:
        return None
    if db.one("SELECT 1 FROM stream_sessions WHERE status = 'active' LIMIT 1") is not None:
        return "skipped: live session running"
    from .config import get_settings

    path = get_settings().db_path
    if shutil.disk_usage(path.parent).free < 3 * pages * size + 256 * 1024**2:
        log.warning("not vacuuming the database: not enough free disk space")
        return "skipped: low disk space"
    t0 = time.monotonic()
    db.execute("VACUUM")
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (_VACUUM_KEY, str(now)),
    )
    log.info("database vacuumed in %.1f s (%d of %d pages were free)", time.monotonic() - t0, free, pages)
    return "vacuumed"


def sweep() -> None:
    global _last
    if _last is not None and time.monotonic() - _last < 3600:
        return
    _last = time.monotonic()
    done = run()
    removed = {k: v for k, v in done.items() if v}
    if removed:
        log.info("retention: %s", ", ".join(f"{k}={v}" for k, v in removed.items()))
    try:
        maintain_storage()
    except Exception:  # noqa: BLE001
        log.exception("database compaction failed")


# ---------------------------------------------------------------- money records after 7 years (by hand)

def money_cutoff(now: int | None = None) -> int:
    return (db.now() if now is None else now) - MONEY_DAYS * DAY


def old_money_records(now: int | None = None) -> dict[str, int]:
    """How many payment, ledger and money-audit rows are older than 7 years (for the monitor and compact-money)."""
    cut = money_cutoff(now)

    def old(table: str, extra: str = "") -> int:
        # The monitor asks every 2 minutes: rows are in time order by id, so look at the oldest one first and only count
        # (a scan) when there is anything to count.
        first = db.one(f"SELECT created_at t FROM {table} ORDER BY id LIMIT 1") if table != "payments" else None
        if first is not None and first["t"] >= cut:
            return 0
        return db.one(f"SELECT COUNT(*) n FROM {table} WHERE created_at < ?{extra}", (cut,))["n"]

    return {"ledger": old("ledger"), "payments": old("payments", " AND status != 'pending'"), "audit_log": old("audit_log")}


def compact_money(now: int | None = None, *, apply: bool = False) -> dict[str, int]:
    """Remove payment, ledger and audit records older than 7 years without breaking any balance.

    Per user, the leading run of ledger rows older than 7 years is replaced by ONE (none if they add up to nothing) "carried forward" row (kind
    'adjustment', ref_type 'carry_forward') that takes the id and the closing balance of the last row it replaces, so
    sum(ledger) still equals the balance and every later row's balance_after still follows on. A run that is only
    such a row is left alone (running it again changes nothing). Old payments (settled ones), old audit rows and then
    deleted accounts with nothing left that refers to them are removed too. Without `apply` it only counts."""
    cut = money_cutoff(now)
    out = {"users": 0, "ledger_rows": 0, "payments": 0, "audit_log": 0, "deleted_users": 0}
    for u in db.all_rows("SELECT DISTINCT user_id FROM ledger WHERE created_at < ?", (cut,)):
        uid = u["user_id"]
        first_new = db.one("SELECT MIN(id) m FROM ledger WHERE user_id = ? AND created_at >= ?", (uid, cut))["m"]
        where, params = ("id < ?", (first_new,)) if first_new is not None else ("1", ())
        prefix = db.all_rows(f"SELECT id, amount_micros, kind, ref_type, balance_after_micros, created_at FROM ledger WHERE user_id = ? AND {where} ORDER BY id", (uid, *params))
        if not prefix or (len(prefix) == 1 and prefix[0]["ref_type"] == "carry_forward"):
            continue
        out["users"] += 1
        out["ledger_rows"] += len(prefix) - (1 if sum(r["amount_micros"] for r in prefix) else 0)  # a balance stays as one row
        if not apply:
            continue
        with db.tx() as c:
            total = c.execute(f"SELECT COALESCE(SUM(amount_micros), 0) s, MAX(id) last FROM ledger WHERE user_id = ? AND {where}", (uid, *params)).fetchone()
            last = c.execute("SELECT balance_after_micros b, created_at t FROM ledger WHERE id = ?", (total["last"],)).fetchone()
            if total["s"] != last["b"]:
                raise RuntimeError(f"user {uid}: ledger rows do not add up to their closing balance; nothing was changed")
            c.execute(f"DELETE FROM ledger WHERE user_id = ? AND {where}", (uid, *params))
            if total["s"]:  # rows that add up to nothing need no replacement
                c.execute(
                    "INSERT INTO ledger(id, user_id, amount_micros, kind, ref_type, ref_id, description, balance_after_micros, created_at)"
                    " VALUES (?,?,?, 'adjustment', 'carry_forward', NULL, 'Balance carried forward (older records removed after 7 years)', ?, ?)",
                    (total["last"], uid, total["s"], total["s"], last["t"]),
                )
            bal = c.execute("SELECT balance_micros b FROM users WHERE id = ?", (uid,)).fetchone()["b"]
            s = c.execute("SELECT COALESCE(SUM(amount_micros), 0) s FROM ledger WHERE user_id = ?", (uid,)).fetchone()["s"]
            if bal != s:
                raise RuntimeError(f"user {uid}: balance {bal} != ledger {s} after compaction; rolled back")
    if apply:
        out["payments"] = db.execute("DELETE FROM payments WHERE created_at < ? AND status != 'pending'", (cut,)).rowcount
        out["audit_log"] = db.execute("DELETE FROM audit_log WHERE created_at < ?", (cut,)).rowcount
        out["deleted_users"] = db.execute(
            "DELETE FROM users WHERE status = 'deleted' AND balance_micros = 0 AND held_micros = 0"
            " AND NOT EXISTS (SELECT 1 FROM ledger WHERE user_id = users.id)"
            " AND NOT EXISTS (SELECT 1 FROM payments WHERE user_id = users.id)"
            " AND NOT EXISTS (SELECT 1 FROM holds WHERE user_id = users.id)"
            " AND NOT EXISTS (SELECT 1 FROM jobs WHERE user_id = users.id)"
            " AND NOT EXISTS (SELECT 1 FROM stream_sessions WHERE user_id = users.id)"
            " AND NOT EXISTS (SELECT 1 FROM sessions WHERE user_id = users.id)"
            " AND NOT EXISTS (SELECT 1 FROM identities WHERE user_id = users.id)"
            " AND NOT EXISTS (SELECT 1 FROM api_keys WHERE user_id = users.id)"
            " AND NOT EXISTS (SELECT 1 FROM audit_log WHERE user_id = users.id)"
        ).rowcount
    else:
        old = old_money_records(now)
        out["payments"], out["audit_log"] = old["payments"], old["audit_log"]
    return out
