"""The owner's dashboard: money, users, usage, errors and the machine, for any time range.

    uvicorn acw.admin_app:app --host 127.0.0.1 --port 8796      (acw-admin.service)

It is reachable from this machine only: it listens on 127.0.0.1, nothing forwards to it (the tunnels carry 8790 and
8794, the VPS knows nothing of it), and requests whose Host isn't this machine's loopback address are refused (so a web
page can't reach it through DNS rebinding). Open it over SSH:  ssh -L 8796:127.0.0.1:8796 <you>@primary-ram  and
http://localhost:8796 . It only reads: the main database is opened read-only, and there is nothing here that changes
anything. The machine history comes from acw/samples.py (recorded by the monitor every 2 minutes).
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from . import jobs as job_names, samples
from .billing import pricing
from .config import get_settings

UI = Path(__file__).resolve().parent / "admin_ui"
PORT = 8796
ALLOWED_HOSTS = {f"127.0.0.1:{PORT}", f"localhost:{PORT}", "127.0.0.1", "localhost"}
DAY = 86400
MICROS = 1_000_000
# Live sessions that ended the usual way; every other end reason is listed under errors.
NORMAL_LIVE_ENDS = ("client_stop", "client_closed", "idle_timeout", "max_duration", "no_audio", "replaced", "trial_over")
# Stripe's standard US card rate, for the estimated fee of each payment (the real fee is on Stripe's dashboard).
FEE_RATE, FEE_FIXED_CENTS = 0.029, 30

app = FastAPI(title="Audio Cleaner dashboard", docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def local_only(request: Request, call_next):
    host = request.headers.get("host", "")
    client = request.client.host if request.client else ""
    if host not in ALLOWED_HOSTS or client not in ("127.0.0.1", "::1") or "x-forwarded-for" in request.headers:
        return JSONResponse({"detail": "Not here."}, status_code=403)
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'"
    resp.headers["Referrer-Policy"] = "no-referrer"
    return resp


def _db() -> sqlite3.Connection:
    c = sqlite3.connect(f"file:{get_settings().db_path}?mode=ro", uri=True, timeout=10)
    c.row_factory = sqlite3.Row
    return c


def _range(start: int | None, end: int | None) -> tuple[int, int, int]:
    """(start, end, bucket seconds): a bucket size that gives a readable number of points for the range."""
    end = int(end if end is not None else time.time())
    start = int(start if start is not None else end - 30 * DAY)
    if not 0 <= start < end:
        raise HTTPException(400, "start must be before end")
    span = end - start
    bucket = next(b for b in (300, 1800, 3600, 6 * 3600, DAY, 7 * DAY, 30 * DAY) if span / b <= 120) if span <= 120 * 30 * DAY else 30 * DAY
    return start, end, bucket


def _job_parts(kind: str, params: dict, cost: int, seconds: float) -> dict[str, int]:
    """A finished job's charge split by price list item (the pinned rates; translations take what is left)."""
    rates = params.get("rates") or {}
    tier = params.get("denoise_tier")
    t_tier = params.get("transcribe_tier")
    out: dict[str, int] = {}
    secs = max(0, math.ceil(seconds or 0))
    if "denoise" in rates and tier and tier != pricing.NO_DENOISE:
        out["Voice rebuild" if tier == "rebuild" else "Denoising"] = math.ceil(rates["denoise"] * secs / 60)
    if "highs" in rates:
        out["Restore highs"] = math.ceil(rates["highs"] * secs / 60)
    if "leveling" in rates:
        out["Leveling"] = math.ceil(rates["leveling"] * secs / 60)
    if "transcribe" in rates and t_tier:
        out["Subtitles, extra accurate" if t_tier == "premium" else "Subtitles"] = math.ceil(rates["transcribe"] * secs / 60)
    rest = max(0, (cost or 0) - sum(out.values()))
    if rest:
        out["Translated subtitles" if "translate" in rates else "Other"] = rest
    return out


def _live_label(denoise: str | None, captions: str | None) -> str:
    parts = (["Noise removal"] if denoise else []) + (
        ["Captions, extra accurate"] if captions == "premium" else ["Captions"] if captions else [])
    return "Live: " + (" + ".join(parts) or "nothing")


@app.get("/")
def index():
    return FileResponse(UI / "index.html")


@app.get("/static/{name}")
def static(name: str):
    if name not in ("app.js", "chart.umd.js", "style.css"):
        raise HTTPException(404)
    return FileResponse(UI / name)


@app.get("/api/overview")
def overview(start: int | None = Query(None), end: int | None = Query(None)) -> dict:
    start, end, _ = _range(start, end)
    c = _db()
    try:
        one = lambda sql, *a: c.execute(sql, a).fetchone()[0] or 0  # noqa: E731
        paid = ("succeeded", "refunded", "disputed")
        q = ",".join("?" * len(paid))
        topups_cents = one(f"SELECT SUM(amount_cents) FROM payments WHERE status IN ({q}) AND created_at >= ? AND created_at < ?", *paid, start, end)
        n_payments = one(f"SELECT COUNT(*) FROM payments WHERE status IN ({q}) AND created_at >= ? AND created_at < ?", *paid, start, end)
        tax_cents = one(f"SELECT SUM(tax_cents) FROM payments WHERE status IN ({q}) AND created_at >= ? AND created_at < ?", *paid, start, end)
        fees_cents = sum(math.ceil(r[0] * FEE_RATE + FEE_FIXED_CENTS) for r in c.execute(
            f"SELECT amount_cents FROM payments WHERE status IN ({q}) AND created_at >= ? AND created_at < ?", (*paid, start, end)))
        refunds = -one("SELECT SUM(amount_micros) FROM ledger WHERE kind = 'refund' AND created_at >= ? AND created_at < ?", start, end)
        disputes = -one("SELECT SUM(amount_micros) FROM ledger WHERE kind = 'dispute' AND created_at >= ? AND created_at < ?", start, end)
        usage = -one("SELECT SUM(amount_micros) FROM ledger WHERE kind = 'usage' AND created_at >= ? AND created_at < ?", start, end)
        grants = one("SELECT SUM(amount_micros) FROM ledger WHERE kind = 'grant' AND amount_micros > 0 AND created_at >= ? AND created_at < ?", start, end)
        failed_payments = one("SELECT COUNT(*) FROM payments WHERE status = 'failed' AND created_at >= ? AND created_at < ?", start, end)
        outstanding = one("SELECT SUM(balance_micros) FROM users WHERE status != 'deleted'")

        accounts = one("SELECT COUNT(*) FROM users WHERE status != 'deleted' AND created_at < ?", end)
        signups = one("SELECT COUNT(*) FROM users WHERE created_at >= ? AND created_at < ?", start, end)
        payers_ever = one(f"SELECT COUNT(DISTINCT user_id) FROM payments WHERE status IN ({q}) AND created_at < ?", *paid, end)
        payers = one(f"SELECT COUNT(DISTINCT user_id) FROM payments WHERE status IN ({q}) AND created_at >= ? AND created_at < ?", *paid, start, end)
        active = one(
            "SELECT COUNT(*) FROM (SELECT user_id FROM jobs WHERE created_at >= ? AND created_at < ?"
            " UNION SELECT user_id FROM stream_sessions WHERE started_at >= ? AND started_at < ?)", start, end, start, end)
        spenders = one("SELECT COUNT(DISTINCT user_id) FROM ledger WHERE kind = 'usage' AND created_at >= ? AND created_at < ?", start, end)
        trial_users = one("SELECT COUNT(DISTINCT user_id) FROM stream_sessions WHERE trial_seconds > 0 AND started_at >= ? AND started_at < ?", start, end)

        revenue: dict[str, int] = {}
        users_by_product: dict[str, set] = {}
        jobs_by: dict[str, dict[str, int]] = {}
        minutes = 0.0
        waits = []
        for r in c.execute("SELECT user_id, kind, status, params_json, cost_micros, duration_seconds, claimed_seconds, wait_seconds"
                           " FROM jobs WHERE created_at >= ? AND created_at < ?", (start, end)):
            params = json.loads(r["params_json"])
            parts = _job_parts(r["kind"], params, r["cost_micros"] or 0, r["duration_seconds"] or 0) if r["status"] == "done" else {}
            for label, micros in parts.items():
                revenue[label] = revenue.get(label, 0) + micros
                users_by_product.setdefault(label, set()).add(r["user_id"])
            name = job_names.job_label(r["kind"], params.get("denoise_tier"), params.get("transcribe_tier"),
                                       "leveling" in (params.get("rates") or {}), "highs" in (params.get("rates") or {})) or r["kind"]
            jobs_by.setdefault(name, {}).setdefault(r["status"], 0)
            jobs_by[name][r["status"]] += 1
            if r["status"] == "done":
                minutes += (r["duration_seconds"] or r["claimed_seconds"] or 0) / 60
            if r["wait_seconds"] is not None:
                waits.append(r["wait_seconds"])
        live_hours = trial_hours = 0.0
        live_sessions = 0
        end_reasons: dict[str, int] = {}
        for r in c.execute("SELECT user_id, denoise_tier, caption_tier, seconds, trial_seconds, cost_micros, end_reason FROM stream_sessions"
                           " WHERE started_at >= ? AND started_at < ?", (start, end)):
            live_sessions += 1
            live_hours += (r["seconds"] or 0) / 3600
            trial_hours += (r["trial_seconds"] or 0) / 3600
            end_reasons[r["end_reason"] or "still running"] = end_reasons.get(r["end_reason"] or "still running", 0) + 1
            if r["cost_micros"]:
                label = _live_label(r["denoise_tier"], r["caption_tier"])
                revenue[label] = revenue.get(label, 0) + r["cost_micros"]
                users_by_product.setdefault(label, set()).add(r["user_id"])
        waits.sort()
        return {
            "range": {"start": start, "end": end},
            "money": {
                "topupsUsd": topups_cents / 100, "payments": n_payments, "taxUsd": tax_cents / 100, "feesUsdEstimate": fees_cents / 100,
                "refundsUsd": refunds / MICROS, "disputesUsd": disputes / MICROS, "usageUsd": usage / MICROS, "grantsUsd": grants / MICROS,
                "netUsd": topups_cents / 100 - refunds / MICROS - disputes / MICROS - fees_cents / 100,
                "failedPayments": failed_payments, "outstandingCreditUsd": outstanding / MICROS,
            },
            "users": {"accounts": accounts, "signups": signups, "active": active, "spenders": spenders, "payersInRange": payers,
                      "payersEver": payers_ever, "neverPaid": max(0, accounts - payers_ever), "liveTrialUsers": trial_users},
            "revenueByProduct": {k: v / MICROS for k, v in sorted(revenue.items(), key=lambda kv: -kv[1])},
            "usersByProduct": {k: len(v) for k, v in sorted(users_by_product.items(), key=lambda kv: -len(kv[1]))},
            "jobs": {"byType": jobs_by, "minutesProcessed": round(minutes, 1),
                     "waitP50": waits[len(waits) // 2] if waits else None, "waitP90": waits[int(len(waits) * 0.9)] if waits else None},
            "live": {"sessions": live_sessions, "hours": round(live_hours, 2), "trialHours": round(trial_hours, 2), "endReasons": end_reasons},
        }
    finally:
        c.close()


@app.get("/api/series")
def series(start: int | None = Query(None), end: int | None = Query(None)) -> dict:
    start, end, bucket = _range(start, end)
    c = _db()
    try:
        def per_bucket(sql: str, *args) -> dict[int, float]:
            return {r[0]: r[1] or 0 for r in c.execute(sql, (bucket, bucket, *args))}

        paid = "('succeeded', 'refunded', 'disputed')"
        topups = per_bucket(f"SELECT (created_at / ?) * ? AS t, SUM(amount_cents) / 100.0 FROM payments WHERE status IN {paid}"
                            " AND created_at >= ? AND created_at < ? GROUP BY t", start, end)
        usage = per_bucket("SELECT (created_at / ?) * ? AS t, -SUM(amount_micros) / 1e6 FROM ledger WHERE kind = 'usage'"
                           " AND created_at >= ? AND created_at < ? GROUP BY t", start, end)
        refunds = per_bucket("SELECT (created_at / ?) * ? AS t, -SUM(amount_micros) / 1e6 FROM ledger WHERE kind IN ('refund', 'dispute')"
                             " AND created_at >= ? AND created_at < ? GROUP BY t", start, end)
        signups = per_bucket("SELECT (created_at / ?) * ? AS t, COUNT(*) FROM users WHERE created_at >= ? AND created_at < ? GROUP BY t", start, end)
        active = per_bucket(
            "SELECT t, COUNT(DISTINCT user_id) FROM (SELECT (created_at / ?) * ? AS t, user_id FROM jobs WHERE created_at >= ? AND created_at < ?"
            " UNION ALL SELECT (started_at / ?) * ? AS t, user_id FROM stream_sessions WHERE started_at >= ? AND started_at < ?) GROUP BY t",
            start, end, bucket, bucket, start, end)
        done = per_bucket("SELECT (created_at / ?) * ? AS t, COUNT(*) FROM jobs WHERE status = 'done' AND created_at >= ? AND created_at < ? GROUP BY t", start, end)
        failed = per_bucket("SELECT (created_at / ?) * ? AS t, COUNT(*) FROM jobs WHERE status = 'failed' AND created_at >= ? AND created_at < ? GROUP BY t", start, end)
        live = per_bucket("SELECT (started_at / ?) * ? AS t, SUM(seconds) / 3600.0 FROM stream_sessions WHERE started_at >= ? AND started_at < ? GROUP BY t", start, end)
        points = []
        t = (start // bucket) * bucket
        while t < end:
            points.append({"t": t, "topupsUsd": topups.get(t, 0), "usageUsd": usage.get(t, 0), "refundsUsd": refunds.get(t, 0),
                           "signups": signups.get(t, 0), "activeUsers": active.get(t, 0), "jobsDone": done.get(t, 0),
                           "jobsFailed": failed.get(t, 0), "liveHours": live.get(t, 0)})
            t += bucket
        return {"bucket": bucket, "points": points, "machine": samples.series(start, end, bucket)}
    finally:
        c.close()


@app.get("/api/users")
def top_users(start: int | None = Query(None), end: int | None = Query(None), limit: int = Query(25, ge=1, le=200)) -> dict:
    start, end, _ = _range(start, end)
    c = _db()
    try:
        rows = c.execute(
            "SELECT u.email, u.created_at, u.balance_micros,"
            " (SELECT -SUM(amount_micros) FROM ledger l WHERE l.user_id = u.id AND l.kind = 'usage' AND l.created_at >= ? AND l.created_at < ?) AS spent,"
            " (SELECT SUM(amount_cents) FROM payments p WHERE p.user_id = u.id AND p.status IN ('succeeded', 'refunded', 'disputed')"
            "   AND p.created_at >= ? AND p.created_at < ?) AS topups,"
            " (SELECT COUNT(*) FROM jobs j WHERE j.user_id = u.id AND j.created_at >= ? AND j.created_at < ?) AS jobs,"
            " (SELECT SUM(seconds) FROM stream_sessions s WHERE s.user_id = u.id AND s.started_at >= ? AND s.started_at < ?) AS live_seconds"
            " FROM users u WHERE u.status != 'deleted' ORDER BY spent DESC, topups DESC LIMIT ?",
            (start, end, start, end, start, end, start, end, limit),
        ).fetchall()
        return {"users": [{"email": r["email"], "since": r["created_at"], "balanceUsd": (r["balance_micros"] or 0) / MICROS,
                           "spentUsd": (r["spent"] or 0) / MICROS, "topupsUsd": (r["topups"] or 0) / 100, "jobs": r["jobs"] or 0,
                           "liveHours": round((r["live_seconds"] or 0) / 3600, 2)} for r in rows if (r["spent"] or r["topups"] or r["jobs"] or r["live_seconds"])]}
    finally:
        c.close()


@app.get("/api/errors")
def errors(start: int | None = Query(None), end: int | None = Query(None)) -> dict:
    start, end, _ = _range(start, end)
    c = _db()
    try:
        failed = [dict(r) for r in c.execute(
            "SELECT j.created_at, j.kind, j.error, j.duration_seconds, u.email FROM jobs j JOIN users u ON u.id = j.user_id"
            " WHERE j.status = 'failed' AND j.created_at >= ? AND j.created_at < ? ORDER BY j.created_at DESC LIMIT 100", (start, end))]
        reasons: dict[str, int] = {}
        for r in failed:
            key = (r["error"] or "unknown").split(".")[0][:120]
            reasons[key] = reasons.get(key, 0) + 1
        money_trouble = [dict(r) for r in c.execute(
            "SELECT created_at, action, detail FROM audit_log WHERE action IN ('refund_unrecovered', 'payment_uncredited', 'dispute_unrecovered',"
            " 'efw_unrecovered', 'auto_reload_failed', 'payment_refused_region') AND created_at >= ? AND created_at < ? ORDER BY id DESC LIMIT 100",
            (start, end))]
        bad_live = {r[0]: r[1] for r in c.execute(
            "SELECT end_reason, COUNT(*) FROM stream_sessions WHERE started_at >= ? AND started_at < ?"
            " AND end_reason IS NOT NULL AND end_reason NOT IN (" + ", ".join(f"'{r}'" for r in NORMAL_LIVE_ENDS) + ") GROUP BY end_reason", (start, end))}
        try:
            alerts = [dict(r) for r in c.execute("SELECT * FROM alert_state")]
        except sqlite3.OperationalError:
            alerts = []
        return {"failedJobs": failed, "failureReasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
                "moneyProblems": money_trouble, "liveEndReasons": bad_live, "alerts": alerts}
    finally:
        c.close()
