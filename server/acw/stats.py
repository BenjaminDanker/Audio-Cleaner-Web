"""Read-only numbers from the database for `python -m acw.admin stats` and the Prometheus file (acw.metrics).

Everything here is SELECT only. Percentiles use the nearest-rank method (p95 of 10 jobs is the slowest one).
"""
from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

from . import db

_DONE = ("done", "expired")  # an expired job finished fine; its files are just gone
WAIT_SQL = "COALESCE(wait_seconds, CASE WHEN started_at >= queued_at THEN started_at - queued_at END)"


def percentile(values: list[float] | list[int], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[max(0, min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1))]


def final_status(status: str) -> str:
    return "done" if status in _DONE else status


def finished_jobs_by_status(since: int) -> dict[str, int]:
    """Jobs that ended at or after `since` (unix seconds; 0 = ever): done, failed, canceled."""
    out: dict[str, int] = {"done": 0, "failed": 0, "canceled": 0}
    for r in db.all_rows(
        "SELECT status, COUNT(*) AS n FROM jobs WHERE finished_at IS NOT NULL AND finished_at >= ? AND status IN ('done','expired','failed','canceled')"
        " GROUP BY status", (since,)):
        out[final_status(r["status"])] += r["n"]
    return out


def queue_waits(since: int) -> list[int]:
    """Seconds each job that started at or after `since` waited for its first worker."""
    return [r["w"] for r in db.all_rows(
        f"SELECT {WAIT_SQL} AS w FROM jobs WHERE started_at IS NOT NULL AND started_at >= ?", (since,)) if r["w"] is not None]


def live_end_reasons(since: int) -> dict[str, int]:
    return {r["reason"]: r["n"] for r in db.all_rows(
        "SELECT COALESCE(end_reason, 'running') AS reason, COUNT(*) AS n FROM stream_sessions WHERE status = 'ended' AND started_at >= ?"
        " GROUP BY reason ORDER BY n DESC", (since,))}


def live_totals(since: int) -> dict[str, float]:
    r = db.one(
        "SELECT COUNT(*) AS sessions, COALESCE(SUM(seconds), 0) AS seconds, COALESCE(SUM(frames), 0) AS frames,"
        " COALESCE(SUM(late_frames), 0) AS late_frames, COALESCE(SUM(dropped_frames), 0) AS dropped_frames"
        " FROM stream_sessions WHERE started_at >= ?", (since,))
    return dict(r)


def usage_micros(since: int) -> int:
    return -db.one("SELECT COALESCE(SUM(amount_micros), 0) AS s FROM ledger WHERE kind = 'usage' AND created_at >= ?", (since,))["s"]


# ------------------------------------------------------------------ the report

def report(days: int = 30, now: int | None = None) -> dict:
    now = int(time.time()) if now is None else now
    since = now - days * 86400
    rep: dict = {"days": days, "since": since, "now": now}

    # --- file jobs, grouped by tier
    groups: dict[str, dict] = defaultdict(lambda: {"status": Counter(), "wait": [], "run": [], "audio_done": 0.0, "run_done": 0.0})
    in_progress: Counter = Counter()
    errors: Counter = Counter()
    for r in db.all_rows(
        f"SELECT tier, status, error, started_at, finished_at, COALESCE(duration_seconds, claimed_seconds) AS audio,"
        f" {WAIT_SQL} AS wait FROM jobs WHERE created_at >= ?", (since,)):
        name = r["tier"]
        g = groups[name]
        status = final_status(r["status"])
        g["status"][status] += 1
        if status in ("uploading", "queued", "running", "cloud_asr"):
            in_progress[status] += 1
        if r["wait"] is not None:
            g["wait"].append(r["wait"])
        if status == "done" and r["started_at"] is not None and r["finished_at"] is not None and r["audio"]:
            run = max(0, r["finished_at"] - r["started_at"])
            g["run"].append(run)
            g["audio_done"] += r["audio"]
            g["run_done"] += run
        if status == "failed":
            errors[(r["error"] or "(no message)")[:100]] += 1
    rep["jobs"] = {
        name: {
            "status": dict(g["status"]),
            "audio_hours_done": g["audio_done"] / 3600,
            "wait_p50": percentile(g["wait"], 50), "wait_p95": percentile(g["wait"], 95),
            "run_p50": percentile(g["run"], 50), "run_p95": percentile(g["run"], 95),
            "rtf": (g["run_done"] / g["audio_done"]) if g["audio_done"] else None,
        }
        for name, g in sorted(groups.items())
    }
    rep["in_progress"] = dict(in_progress)
    done = sum(g["status"]["done"] for g in groups.values())
    failed = sum(g["status"]["failed"] for g in groups.values())
    rep["failure_ratio"] = failed / (done + failed) if done + failed else None
    rep["done"], rep["failed"] = done, failed
    rep["top_errors"] = errors.most_common(5)

    # --- live sessions
    tot = live_totals(since)
    tiers: dict[str, list] = defaultdict(lambda: [0, 0.0])
    for r in db.all_rows("SELECT denoise_tier, caption_tier, seconds FROM stream_sessions WHERE started_at >= ?", (since,)):
        key = "+".join(x for x in (r["denoise_tier"] and f"denoise {r['denoise_tier']}", r["caption_tier"] and f"captions {r['caption_tier']}") if x) or "no processing"
        tiers[key][0] += 1
        tiers[key][1] += r["seconds"] or 0
    rep["live"] = {**tot, "minutes": tot["seconds"] / 60, "end_reasons": live_end_reasons(since),
                   "tiers": {k: {"sessions": v[0], "minutes": v[1] / 60} for k, v in sorted(tiers.items())}}

    # --- money
    usage = {r["ref_type"] or "other": -r["s"] for r in db.all_rows(
        "SELECT ref_type, SUM(amount_micros) AS s FROM ledger WHERE kind = 'usage' AND created_at >= ? GROUP BY ref_type", (since,))}
    rep["usage_usd"] = {k: v / 1e6 for k, v in sorted(usage.items())}
    rep["usage_users"] = db.one("SELECT COUNT(DISTINCT user_id) AS n FROM ledger WHERE kind = 'usage' AND created_at >= ?", (since,))["n"]
    rep["topups_usd"] = db.one("SELECT COALESCE(SUM(amount_cents), 0) AS s FROM payments WHERE status = 'succeeded' AND created_at >= ?", (since,))["s"] / 100
    rep["users_total"] = db.one("SELECT COUNT(*) AS n FROM users")["n"]
    rep["users_new"] = db.one("SELECT COUNT(*) AS n FROM users WHERE created_at >= ?", (since,))["n"]
    return rep


def _dur(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    s = int(round(seconds))
    if s < 90:
        return f"{s}s"
    if s < 5400:
        return f"{s / 60:.1f}m"
    return f"{s / 3600:.1f}h"


def render(rep: dict) -> str:
    out: list[str] = []
    when = datetime.fromtimestamp(rep["since"], timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out.append(f"Audio Cleaner, last {rep['days']} day(s) (since {when})")
    out.append(f"accounts: {rep['users_total']} ({rep['users_new']} new)")

    out.append("")
    out.append("File jobs (created in the period)")
    if not rep["jobs"]:
        out.append("  none")
    else:
        out.append(f"  {'tier':<22}{'done':>6}{'failed':>8}{'cancel':>8}{'audio h':>9}{'wait p50':>10}{'wait p95':>10}{'run p50':>9}{'run p95':>9}{'RTF':>7}")
        for name, j in rep["jobs"].items():
            st = j["status"]
            rtf = "-" if j["rtf"] is None else f"{j['rtf']:.2f}"
            out.append(
                f"  {name:<22}{st.get('done', 0):>6}{st.get('failed', 0):>8}{st.get('canceled', 0):>8}{j['audio_hours_done']:>9.2f}"
                f"{_dur(j['wait_p50']):>10}{_dur(j['wait_p95']):>10}{_dur(j['run_p50']):>9}{_dur(j['run_p95']):>9}{rtf:>7}")
        out.append("  wait = queue wait before the first worker took the job; run = start to finish; RTF = run seconds per audio second.")
    if rep["in_progress"]:
        out.append("  in progress now: " + ", ".join(f"{n} {s}" for s, n in sorted(rep["in_progress"].items())))
    ratio = rep["failure_ratio"]
    out.append("  failure ratio: " + ("-" if ratio is None else f"{ratio * 100:.1f}% ({rep['failed']} of {rep['done'] + rep['failed']} finished)"))
    if rep["top_errors"]:
        out.append("  top errors:")
        for msg, n in rep["top_errors"]:
            out.append(f"    {n:>4}  {msg}")

    live = rep["live"]
    out.append("")
    out.append("Live sessions")
    out.append(f"  {live['sessions']} sessions, {live['minutes']:.1f} minutes of audio")
    if live["end_reasons"]:
        out.append("  end reasons: " + ", ".join(f"{k} {v}" for k, v in live["end_reasons"].items()))
    for name, t in live["tiers"].items():
        out.append(f"  {name}: {t['sessions']} sessions, {t['minutes']:.1f} min")
    if live["frames"]:
        out.append(f"  frames: {int(live['frames'])}, late (>100 ms) {int(live['late_frames'])} ({live['late_frames'] / live['frames'] * 100:.3f}%), "
                   f"never returned {int(live['dropped_frames'])} ({live['dropped_frames'] / live['frames'] * 100:.3f}%)")
    else:
        out.append("  frames: not recorded for this period")

    out.append("")
    out.append("Free tier and money")
    if rep["usage_usd"]:
        parts = ", ".join(f"{k} ${v:,.2f}" for k, v in rep["usage_usd"].items())
        out.append(f"  usage charged: ${sum(rep['usage_usd'].values()):,.2f} ({parts}) from {rep['usage_users']} account(s)")
    else:
        out.append("  usage charged: $0.00")
    out.append(f"  top-ups paid: ${rep['topups_usd']:,.2f}")
    return "\n".join(out)
