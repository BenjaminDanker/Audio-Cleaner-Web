"""Machine samples for the admin dashboard's resource graphs: the monitor (every 2 minutes) records one row of what the
machine is doing (broker.snapshot), and the dashboard reads them back per time bucket. A database of its own
(data/samples.sqlite3), so the history never touches the main database. No personal data: counts and machine figures.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from .config import get_settings

KEEP_S = 400 * 86400  # a little over a year
FIELDS = (
    "cpu_busy", "cpu_cores", "mem_available", "slice_headroom", "disk_free", "gpu_busy", "gpu_mem_used", "gpu_mem_total",
    "jobs_queued", "jobs_running", "live_sessions", "live_trial", "live_units", "live_capacity",
)


def path() -> Path:
    return get_settings().data_dir / "samples.sqlite3"


def _connect(p: Path, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        c = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=5)
    else:
        c = sqlite3.connect(p, timeout=5)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("CREATE TABLE IF NOT EXISTS samples (ts INTEGER PRIMARY KEY, " + ", ".join(f"{f} REAL" for f in FIELDS) + ")")
    c.row_factory = sqlite3.Row
    return c


def from_snapshot(sn: dict) -> dict:
    """One sample from broker.snapshot()."""
    card = (sn.get("gpu") or {}).get("card") or {}
    engine = (sn.get("live") or {}).get("engine") or {}
    sessions = (sn.get("live") or {}).get("sessions") or {}
    return {
        "cpu_busy": sn["cpu"]["busy"], "cpu_cores": sn["cpu"]["cores"],
        "mem_available": sn["memory"]["available"], "slice_headroom": sn["memory"]["sliceHeadroom"],
        "disk_free": sn["disk"]["free"],
        "gpu_busy": card.get("busyPercent"), "gpu_mem_used": card.get("memoryUsedMb"), "gpu_mem_total": card.get("memoryTotalMb"),
        "jobs_queued": sn["jobs"]["queued"], "jobs_running": len(sn["jobs"]["running"]),
        "live_sessions": sum(sessions.values()), "live_trial": sessions.get("live_trial", 0),
        "live_units": engine.get("units"), "live_capacity": engine.get("capacityUnits"),
    }


def record(sample: dict, now: int | None = None, p: Path | None = None) -> None:
    now = int(now if now is not None else time.time())
    c = _connect(p or path())
    try:
        with c:
            c.execute(f"INSERT OR REPLACE INTO samples (ts, {', '.join(FIELDS)}) VALUES (?{', ?' * len(FIELDS)})",
                      (now, *(sample.get(f) for f in FIELDS)))
            c.execute("DELETE FROM samples WHERE ts < ?", (now - KEEP_S,))
    finally:
        c.close()


def series(start: int, end: int, bucket: int, p: Path | None = None) -> list[dict]:
    """Per bucket of `bucket` seconds: the average of each field (missing values left out), and the bucket's start."""
    p = p or path()
    if not p.exists():
        return []
    c = _connect(p, readonly=True)
    try:
        rows = c.execute(
            f"SELECT (ts / ?) * ? AS t, {', '.join(f'AVG({f}) AS {f}' for f in FIELDS)} FROM samples"
            " WHERE ts >= ? AND ts < ? GROUP BY t ORDER BY t", (bucket, bucket, start, end),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        c.close()
