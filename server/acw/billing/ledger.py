"""Prepaid credit ledger.

Invariants (also enforced by CHECK constraints on users):
  * balance_micros >= 0 and held_micros <= balance_micros
  * every change to balance_micros has exactly one ledger row
  * available = balance - held; new holds and direct debits only use available

Usage is reserved up front with a hold, then captured at the real amount (or
released). That keeps a job or stream from spending money the account no
longer has, and lets several jobs run at once without overdrawing.
"""
from __future__ import annotations

import logging
import re
import sqlite3

from .. import db

log = logging.getLogger(__name__)


class InsufficientFunds(Exception):
    def __init__(self, needed: int, available: int):
        super().__init__(f"needed {needed} micros, available {available}")
        self.needed = needed
        self.available = available


class LedgerError(Exception):
    pass


def _user_row(c: sqlite3.Connection, user_id: str) -> sqlite3.Row:
    row = c.execute(
        "SELECT id, balance_micros, held_micros, status FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    if row is None:
        raise LedgerError("unknown user")
    return row


def _apply(
    c: sqlite3.Connection,
    user_id: str,
    amount: int,
    kind: str,
    ref_type: str | None,
    ref_id: str | None,
    description: str | None,
) -> int:
    row = _user_row(c, user_id)
    new_balance = row["balance_micros"] + amount
    if new_balance < row["held_micros"]:
        raise InsufficientFunds(-amount, row["balance_micros"] - row["held_micros"])
    c.execute("UPDATE users SET balance_micros = ? WHERE id = ?", (new_balance, user_id))
    c.execute(
        "INSERT INTO ledger(user_id, amount_micros, kind, ref_type, ref_id, description,"
        " balance_after_micros, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (user_id, amount, kind, ref_type, ref_id, description, new_balance, db.now()),
    )
    return new_balance


def balance(user_id: str) -> dict:
    row = db.one("SELECT balance_micros, held_micros FROM users WHERE id = ?", (user_id,))
    if row is None:
        raise LedgerError("unknown user")
    return {
        "balance_micros": row["balance_micros"],
        "held_micros": row["held_micros"],
        "available_micros": row["balance_micros"] - row["held_micros"],
    }


def granted_micros(user_id: str) -> int:
    """Credit given, not bought (admin grants, top-up bonuses, less any bonus taken back): never paid out by a refund."""
    row = db.one("SELECT COALESCE(SUM(amount_micros), 0) AS s FROM ledger WHERE user_id = ? AND kind = 'grant'", (user_id,))
    return max(0, int(row["s"]))


def credit(
    user_id: str,
    amount_micros: int,
    kind: str,
    ref_type: str | None = None,
    ref_id: str | None = None,
    description: str | None = None,
    *,
    c: sqlite3.Connection | None = None,
) -> bool:
    """Add money. Returns False if this (kind, ref) was already credited."""
    if amount_micros <= 0:
        raise LedgerError("credit amount must be positive")

    def run(cc: sqlite3.Connection) -> bool:
        if ref_id is not None:
            dup = cc.execute(
                "SELECT 1 FROM ledger WHERE kind = ? AND ref_type IS ? AND ref_id = ?",
                (kind, ref_type, ref_id),
            ).fetchone()
            if dup:
                return False
        _apply(cc, user_id, amount_micros, kind, ref_type, ref_id, description)
        return True

    if c is not None:
        return run(c)
    with db.tx() as cc:
        return run(cc)


def debit_available(
    user_id: str,
    amount_micros: int,
    kind: str,
    ref_type: str | None = None,
    ref_id: str | None = None,
    description: str | None = None,
    *,
    allow_partial: bool = False,
    c: sqlite3.Connection | None = None,
) -> int:
    """Take money from the unheld balance. Returns the amount actually taken."""
    if amount_micros < 0:
        raise LedgerError("debit amount must be non-negative")

    def run(cc: sqlite3.Connection) -> int:
        if ref_id is not None and kind in ("refund", "dispute"):
            dup = cc.execute(
                "SELECT 1 FROM ledger WHERE kind = ? AND ref_type IS ? AND ref_id = ?",
                (kind, ref_type, ref_id),
            ).fetchone()
            if dup:
                return 0
        row = _user_row(cc, user_id)
        available = row["balance_micros"] - row["held_micros"]
        take = amount_micros
        if take > available:
            if not allow_partial:
                raise InsufficientFunds(amount_micros, available)
            take = available
        if take == 0:
            return 0
        _apply(cc, user_id, -take, kind, ref_type, ref_id, description)
        return take

    if c is not None:
        return run(c)
    with db.tx() as cc:
        return run(cc)


def create_hold(user_id: str, amount_micros: int, ref_type: str, ref_id: str, *, c: sqlite3.Connection | None = None) -> str:
    """Reserve money. With `c` it joins the caller's transaction (a job's checks and its row are one step)."""
    if amount_micros <= 0:
        raise LedgerError("hold amount must be positive")

    def run(c: sqlite3.Connection) -> str:
        row = _user_row(c, user_id)
        if row["status"] != "active":
            raise LedgerError("account is not active")
        available = row["balance_micros"] - row["held_micros"]
        if amount_micros > available:
            raise InsufficientFunds(amount_micros, available)
        hold_id = db.new_id()
        c.execute(
            "INSERT INTO holds(id, user_id, amount_micros, ref_type, ref_id, status, created_at)"
            " VALUES (?,?,?,?,?, 'active', ?)",
            (hold_id, user_id, amount_micros, ref_type, ref_id, db.now()),
        )
        c.execute("UPDATE users SET held_micros = held_micros + ? WHERE id = ?", (amount_micros, user_id))
        return hold_id

    if c is not None:
        return run(c)
    with db.tx() as cc:
        return run(cc)


def _active_hold(c: sqlite3.Connection, hold_id: str) -> sqlite3.Row:
    hold = c.execute("SELECT * FROM holds WHERE id = ?", (hold_id,)).fetchone()
    if hold is None:
        raise LedgerError("unknown hold")
    if hold["status"] != "active":
        raise LedgerError(f"hold is {hold['status']}")
    return hold


def extend_hold(hold_id: str, extra_micros: int, *, c: sqlite3.Connection | None = None) -> None:
    """Grow an active hold, e.g. a live stream reserving its next minutes. With `c` it joins the caller's transaction."""
    if extra_micros <= 0:
        return

    def run(cc: sqlite3.Connection) -> None:
        hold = _active_hold(cc, hold_id)
        row = _user_row(cc, hold["user_id"])
        if row["status"] != "active":
            raise LedgerError("account is not active")
        available = row["balance_micros"] - row["held_micros"]
        if extra_micros > available:
            raise InsufficientFunds(extra_micros, available)
        cc.execute("UPDATE holds SET amount_micros = amount_micros + ? WHERE id = ?", (extra_micros, hold_id))
        cc.execute("UPDATE users SET held_micros = held_micros + ? WHERE id = ?", (extra_micros, hold["user_id"]))

    if c is not None:
        return run(c)
    with db.tx() as cc:
        return run(cc)


def reduce_hold(hold_id: str, new_amount_micros: int, *, c: sqlite3.Connection | None = None) -> int:
    """Shrink an active hold to `new_amount_micros` (a job held the most it could cost and turned out to need less); the rest is
    available again at once. A hold that is not larger than that is left as it is. Returns the amount now held. With `c` it joins
    the caller's transaction."""
    if new_amount_micros <= 0:
        raise LedgerError("hold amount must be positive")

    def run(cc: sqlite3.Connection) -> int:
        hold = _active_hold(cc, hold_id)
        held = hold["amount_micros"]
        if held <= new_amount_micros:
            return held
        cc.execute("UPDATE holds SET amount_micros = ? WHERE id = ?", (new_amount_micros, hold_id))
        cc.execute("UPDATE users SET held_micros = held_micros - ? WHERE id = ?", (held - new_amount_micros, hold["user_id"]))
        return new_amount_micros

    if c is not None:
        return run(c)
    with db.tx() as cc:
        return run(cc)


def capture_hold(
    hold_id: str,
    amount_micros: int,
    description: str | None = None,
    *,
    kind: str = "usage",
    ref_type: str | None = None,
    ref_id: str | None = None,
    c: sqlite3.Connection | None = None,
) -> int:
    """Charge up to the held amount and release the rest. Returns the amount charged.

    If the real cost exceeds the hold, the overage is taken from the available
    balance as far as it goes; whatever is still missing is written off (the
    caller should have extended the hold first). With `c` it joins the caller's
    transaction (e.g. the job's status change).
    """
    if amount_micros < 0:
        raise LedgerError("capture amount must be non-negative")

    def run(cc: sqlite3.Connection) -> int:
        hold = _active_hold(cc, hold_id)
        user_id = hold["user_id"]
        held = hold["amount_micros"]
        cc.execute("UPDATE users SET held_micros = held_micros - ? WHERE id = ?", (held, user_id))
        charge = amount_micros
        if charge > held:
            # The released hold is now part of `available` again.
            row = _user_row(cc, user_id)
            charge = min(charge, row["balance_micros"] - row["held_micros"])
        if charge > 0:
            _apply(
                cc,
                user_id,
                -charge,
                kind,
                ref_type or hold["ref_type"],
                ref_id or hold["ref_id"],
                description,
            )
        cc.execute(
            "UPDATE holds SET status = 'captured', amount_micros = ?, closed_at = ? WHERE id = ?",
            (charge, db.now(), hold_id),
        )
        return charge

    if c is not None:
        return run(c)
    with db.tx() as cc:
        return run(cc)


def release_hold(hold_id: str, *, c: sqlite3.Connection | None = None) -> None:
    """Give a hold back. Does nothing if it is already closed. With `c` it joins the caller's transaction."""

    def run(cc: sqlite3.Connection) -> None:
        hold = cc.execute("SELECT * FROM holds WHERE id = ?", (hold_id,)).fetchone()
        if hold is None or hold["status"] != "active":
            return
        cc.execute("UPDATE users SET held_micros = held_micros - ? WHERE id = ?", (hold["amount_micros"], hold["user_id"]))
        cc.execute("UPDATE holds SET status = 'released', closed_at = ? WHERE id = ?", (db.now(), hold_id))

    if c is not None:
        return run(c)
    with db.tx() as cc:
        return run(cc)


def release_all_holds(user_id: str, c: sqlite3.Connection) -> None:
    holds = c.execute("SELECT id, amount_micros FROM holds WHERE user_id = ? AND status = 'active'", (user_id,)).fetchall()
    for h in holds:
        c.execute("UPDATE users SET held_micros = held_micros - ? WHERE id = ?", (h["amount_micros"], user_id))
        c.execute("UPDATE holds SET status = 'released', closed_at = ? WHERE id = ?", (db.now(), h["id"]))


# Holds that outlive their purpose, by what they were opened for. A hold is young for a moment
# before the job or stream row it names exists, so "missing" only counts after a grace period.
ORPHAN_GRACE_SECONDS = 120
REFUND_HOLD_MAX_SECONDS = 600
STREAM_HOLD_MAX_SECONDS = 13 * 3600  # longer than any session (stream.routes.MAX_SESSION_SECONDS)
_reported_mismatches: set[tuple] = set()


def reap_stale_holds() -> int:
    """Release active holds nothing will ever close: for a finished (or missing) job, an ended (or missing)
    live session, or a refund that took too long. Also logs when users.held_micros stops matching the sum of
    the active holds. Returns how many holds it released."""
    now = db.now()
    stale = db.all_rows(
        "SELECT h.id, h.ref_type, h.ref_id FROM holds h"
        " LEFT JOIN jobs j ON h.ref_type = 'job' AND j.id = h.ref_id"
        " LEFT JOIN stream_sessions s ON h.ref_type = 'stream' AND s.id = h.ref_id"
        " WHERE h.status = 'active' AND ("
        "  (h.ref_type = 'job' AND ((j.id IS NULL AND h.created_at < :orphan)"
        "     OR j.status IN ('done', 'failed', 'canceled', 'expired')))"
        "  OR (h.ref_type = 'stream' AND ((s.id IS NULL AND h.created_at < :orphan) OR s.status = 'ended'"
        "     OR h.created_at < :stream_old))"
        "  OR (h.ref_type = 'refund' AND h.created_at < :refund_old))",
        {"orphan": now - ORPHAN_GRACE_SECONDS, "stream_old": now - STREAM_HOLD_MAX_SECONDS,
         "refund_old": now - REFUND_HOLD_MAX_SECONDS},
    )
    for h in stale:
        log.warning("releasing leaked %s hold %s (%s)", h["ref_type"], h["id"], h["ref_id"])
        release_hold(h["id"])
    for u in db.all_rows(
        "SELECT u.id, u.held_micros, COALESCE(SUM(h.amount_micros), 0) AS active FROM users u"
        " LEFT JOIN holds h ON h.user_id = u.id AND h.status = 'active' GROUP BY u.id"
        " HAVING u.held_micros != COALESCE(SUM(h.amount_micros), 0)"
    ):
        key = (u["id"], u["held_micros"], u["active"])
        if key not in _reported_mismatches:
            _reported_mismatches.add(key)
            log.error("user %s: held_micros %s != sum of active holds %s", u["id"], u["held_micros"], u["active"])
    return len(stale)


def invariant_problems() -> dict:
    """How many accounts break the money rules right now: balance != sum of ledger entries, or held != sum of
    active holds. Both should always be 0."""
    bad_balance = db.one(
        "SELECT COUNT(*) AS n FROM users u WHERE u.balance_micros != COALESCE((SELECT SUM(amount_micros) FROM ledger l WHERE l.user_id = u.id), 0)"
    )["n"]
    bad_held = db.one(
        "SELECT COUNT(*) AS n FROM users u WHERE u.held_micros != COALESCE((SELECT SUM(amount_micros) FROM holds h"
        " WHERE h.user_id = u.id AND h.status = 'active'), 0)"
    )["n"]
    return {"balance_mismatch": bad_balance, "held_mismatch": bad_held}


_KIND_LABELS = {
    "topup": "Top-up", "grant": "Credit added", "refund": "Refund", "dispute": "Payment dispute",
    "adjustment": "Adjustment", "usage": "Usage",
}


def history(user_id: str, limit: int = 100) -> list[dict]:
    rows = db.all_rows(
        "SELECT id, amount_micros, kind, ref_type, ref_id, description, balance_after_micros, created_at"
        " FROM ledger WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    )
    return [dict(r) for r in rows]


def history_page(user_id: str, limit: int = 100) -> tuple[list[dict], bool]:
    """The newest `limit` entries with customer-facing fields added, and whether older ones exist. Besides the
    raw columns every entry has `label` (what it was, in the words of the price list); usage entries also have
    `product` (clean | transcript | clean_transcript | live), `tier`, `minutes` and `displayName` (the file's
    name as the person chose it)."""
    from .. import job_view  # noqa: PLC0415 - jobs imports this module

    rows = history(user_id, limit + 1)
    more = len(rows) > limit
    rows = rows[:limit]
    job_ids = [r["ref_id"] for r in rows if r["kind"] == "usage" and r["ref_type"] == "job" and r["ref_id"]]
    stream_ids = [r["ref_id"] for r in rows if r["kind"] == "usage" and r["ref_type"] == "stream" and r["ref_id"]]
    job_rows = {
        r["id"]: r for r in db.all_rows(
            f"SELECT id, kind, params_json, input_name, duration_seconds FROM jobs WHERE id IN ({','.join('?' * len(job_ids))})", tuple(job_ids)
        )
    } if job_ids else {}
    stream_rows = {
        r["id"]: r for r in db.all_rows(
            f"SELECT id, denoise_tier, caption_tier, seconds FROM stream_sessions WHERE id IN ({','.join('?' * len(stream_ids))})", tuple(stream_ids)
        )
    } if stream_ids else {}
    import json

    for r in rows:
        r["label"] = _KIND_LABELS.get(r["kind"], r["kind"].capitalize())
        j = job_rows.get(r["ref_id"]) if r["ref_type"] == "job" else None
        s = stream_rows.get(r["ref_id"]) if r["ref_type"] == "stream" else None
        if j is not None:
            p = json.loads(j["params_json"])
            r["label"] = job_view.label_of(j["kind"], p)  # a re-run is named for the steps it did again
            r["product"] = {"denoise": "clean", "transcribe": "transcript", "denoise_transcribe": "clean_transcript"}.get(j["kind"])
            r["tier"] = p.get("denoise_tier") or p.get("transcribe_tier")
            r["minutes"] = round((j["duration_seconds"] or 0) / 60, 1)
            r["displayName"] = p.get("display_name") or j["input_name"]
            m = re.match(r"^(Edited[^:]*): .* \((\d+(?:\.\d+)?) min\)$", r["description"] or "", re.S)
            if m:  # a run of the per-part editor (editing.py): named for the steps it ran, for the minutes it processed
                r["label"], r["minutes"] = m.group(1), float(m.group(2))
        elif s is not None:
            parts = []
            if s["denoise_tier"]:
                parts.append("Live noise removal")
            if s["caption_tier"]:
                parts.append("Live captions" + (", extra accurate" if s["caption_tier"] == "premium" else ""))
            r["label"] = " + ".join(parts) or "Live session"
            r["product"] = "live"
            r["tier"] = s["denoise_tier"] or s["caption_tier"]
            r["minutes"] = round((s["seconds"] or 0) / 60, 1)
    return rows, more
