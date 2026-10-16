"""Monthly spending caps of API keys.

A key may carry a cap (`api_keys.monthly_cap_micros`, NULL = none) for the calendar month (UTC). What counts as spent
by a key this month:

  * file jobs it created that finished this month, at their real cost, plus jobs still running at their reserved
    estimate (the money held for them);
  * live sessions it started that ended this month, at their real cost, plus the money held for sessions still running.

The check runs inside the same transaction that reserves the money (job creation, the start of a live session and
each time a live session reserves its next minute), so parallel requests can't all pass a check none of them has
recorded yet. A job that is already accepted is never stopped for the cap; if its real cost turns out higher than its
estimate (the file was longer than claimed) it can finish slightly above the cap. Browser sessions have no cap.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from . import db

ACTIVE_JOB_STATES = ("uploading", "queued", "running", "cloud_asr")
MICROS_PER_CENT = 10_000
MAX_CAP_CENTS = 1_000_000  # $10,000 a month: a number that can't overflow anything and is above any real use


class KeyCapReached(Exception):
    def __init__(self, cap: int, used: int, wanted: int, resets_at: int):
        self.cap, self.used, self.wanted, self.resets_at = cap, used, wanted, resets_at
        left = max(0, cap - used)
        when = datetime.fromtimestamp(resets_at, timezone.utc).strftime("%Y-%m-%d")
        if wanted <= 0 or left <= 0:
            msg = f"This API key has used its monthly spending limit (${cap / 1e6:.2f})."
        else:
            msg = (f"This needs about ${wanted / 1e6:.2f}, but this API key has only ${left / 1e6:.2f} left of its monthly "
                   f"spending limit (${cap / 1e6:.2f}).")
        super().__init__(f"{msg} It starts over on {when} (UTC). You can raise the limit in the API keys card on the Live page, or use another key.")


def month_bounds(now: int | None = None) -> tuple[int, int]:
    """(start of this calendar month, start of the next one), UTC, as Unix seconds."""
    d = datetime.fromtimestamp(db.now() if now is None else now, timezone.utc)
    start = d.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    nxt = start.replace(year=start.year + 1, month=1) if start.month == 12 else start.replace(month=start.month + 1)
    return int(start.timestamp()), int(nxt.timestamp())


def used_micros(c: sqlite3.Connection, key_id: str, since: int) -> int:
    """Spent this month by a key (see the module text), counting money still held."""
    marks = ",".join("?" * len(ACTIVE_JOB_STATES))
    jobs_done = c.execute(
        "SELECT COALESCE(SUM(cost_micros), 0) AS n FROM jobs WHERE api_key_id = ? AND status IN ('done', 'expired') AND finished_at >= ?",
        (key_id, since),
    ).fetchone()["n"]
    jobs_open = c.execute(
        f"SELECT COALESCE(SUM(est_cost_micros), 0) AS n FROM jobs WHERE api_key_id = ? AND status IN ({marks})", (key_id, *ACTIVE_JOB_STATES)
    ).fetchone()["n"]
    live_done = c.execute(
        "SELECT COALESCE(SUM(cost_micros), 0) AS n FROM stream_sessions WHERE api_key_id = ? AND status = 'ended' AND ended_at >= ?",
        (key_id, since),
    ).fetchone()["n"]
    live_open = c.execute(
        "SELECT COALESCE(SUM(h.amount_micros), 0) AS n FROM stream_sessions s JOIN holds h ON h.id = s.hold_id"
        " WHERE s.api_key_id = ? AND s.status = 'active' AND h.status = 'active'",
        (key_id,),
    ).fetchone()["n"]
    return int(jobs_done + jobs_open + live_done + live_open)


def check(c: sqlite3.Connection, key_id: str | None, extra_micros: int) -> None:
    """Raise KeyCapReached when reserving `extra_micros` more would take the key over its cap. Call inside the
    transaction that reserves the money. Nothing happens for browser sessions (no key), keys without a cap, or free work."""
    if key_id is None or extra_micros <= 0:
        return
    row = c.execute("SELECT monthly_cap_micros FROM api_keys WHERE id = ?", (key_id,)).fetchone()
    if row is None or row["monthly_cap_micros"] is None:
        return
    start, end = month_bounds()
    used = used_micros(c, key_id, start)
    if used + extra_micros > row["monthly_cap_micros"]:
        raise KeyCapReached(int(row["monthly_cap_micros"]), used, extra_micros, end)


def usage_for(key_ids: list[str]) -> dict[str, int]:
    """Spent this month, per key (for the API keys card)."""
    start = month_bounds()[0]
    c = db.conn()
    return {k: used_micros(c, k, start) for k in key_ids}


def key_limit(key_id: str) -> dict | None:
    """A key's cap and how much of it is used, or None for a key without a cap."""
    row = db.one("SELECT monthly_cap_micros FROM api_keys WHERE id = ?", (key_id,))
    if row is None or row["monthly_cap_micros"] is None:
        return None
    start, end = month_bounds()
    used = used_micros(db.conn(), key_id, start)
    cap = int(row["monthly_cap_micros"])
    return {"monthlyCapMicros": cap, "usedMicros": used, "remainingMicros": max(0, cap - used), "resetsAt": end}
