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

import sqlite3

from .. import db


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


def create_hold(user_id: str, amount_micros: int, ref_type: str, ref_id: str) -> str:
    if amount_micros <= 0:
        raise LedgerError("hold amount must be positive")
    with db.tx() as c:
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


def _active_hold(c: sqlite3.Connection, hold_id: str) -> sqlite3.Row:
    hold = c.execute("SELECT * FROM holds WHERE id = ?", (hold_id,)).fetchone()
    if hold is None:
        raise LedgerError("unknown hold")
    if hold["status"] != "active":
        raise LedgerError(f"hold is {hold['status']}")
    return hold


def extend_hold(hold_id: str, extra_micros: int) -> None:
    """Grow an active hold, e.g. a live stream reserving its next minutes."""
    if extra_micros <= 0:
        return
    with db.tx() as c:
        hold = _active_hold(c, hold_id)
        row = _user_row(c, hold["user_id"])
        if row["status"] != "active":
            raise LedgerError("account is not active")
        available = row["balance_micros"] - row["held_micros"]
        if extra_micros > available:
            raise InsufficientFunds(extra_micros, available)
        c.execute("UPDATE holds SET amount_micros = amount_micros + ? WHERE id = ?", (extra_micros, hold_id))
        c.execute("UPDATE users SET held_micros = held_micros + ? WHERE id = ?", (extra_micros, hold["user_id"]))


def capture_hold(
    hold_id: str,
    amount_micros: int,
    description: str | None = None,
    *,
    kind: str = "usage",
    ref_type: str | None = None,
    ref_id: str | None = None,
) -> int:
    """Charge up to the held amount and release the rest. Returns the amount charged.

    If the real cost exceeds the hold, the overage is taken from the available
    balance as far as it goes; whatever is still missing is written off (the
    caller should have extended the hold first).
    """
    if amount_micros < 0:
        raise LedgerError("capture amount must be non-negative")
    with db.tx() as c:
        hold = _active_hold(c, hold_id)
        user_id = hold["user_id"]
        held = hold["amount_micros"]
        c.execute("UPDATE users SET held_micros = held_micros - ? WHERE id = ?", (held, user_id))
        charge = amount_micros
        if charge > held:
            row = _user_row(c, user_id)
            available = row["balance_micros"] - row["held_micros"]
            charge = held + min(charge - held, available)
        if charge > 0:
            _apply(
                c,
                user_id,
                -charge,
                kind,
                ref_type or hold["ref_type"],
                ref_id or hold["ref_id"],
                description,
            )
        c.execute(
            "UPDATE holds SET status = 'captured', amount_micros = ?, closed_at = ? WHERE id = ?",
            (charge, db.now(), hold_id),
        )
        return charge


def release_hold(hold_id: str) -> None:
    with db.tx() as c:
        hold = c.execute("SELECT * FROM holds WHERE id = ?", (hold_id,)).fetchone()
        if hold is None or hold["status"] != "active":
            return
        c.execute("UPDATE users SET held_micros = held_micros - ? WHERE id = ?", (hold["amount_micros"], hold["user_id"]))
        c.execute("UPDATE holds SET status = 'released', closed_at = ? WHERE id = ?", (db.now(), hold_id))


def release_all_holds(user_id: str, c: sqlite3.Connection) -> None:
    holds = c.execute("SELECT id, amount_micros FROM holds WHERE user_id = ? AND status = 'active'", (user_id,)).fetchall()
    for h in holds:
        c.execute("UPDATE users SET held_micros = held_micros - ? WHERE id = ?", (h["amount_micros"], user_id))
        c.execute("UPDATE holds SET status = 'released', closed_at = ? WHERE id = ?", (db.now(), h["id"]))


def history(user_id: str, limit: int = 100) -> list[dict]:
    rows = db.all_rows(
        "SELECT id, amount_micros, kind, ref_type, ref_id, description, balance_after_micros, created_at"
        " FROM ledger WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    )
    return [dict(r) for r in rows]
