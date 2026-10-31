"""Numbers for the home page on the Raspberry Pi (Host-Ops home/): jobs running and waiting per feature, live sessions, the
GPU worker (every 10 s), and the money summary (every 5 min: this month and last month from the owner's dashboard
calculations in acw.owner_stats, and the last 30 days as a series).

The web process POSTs them as JSON to `ACW_DASHBOARD_URL` (e.g. http://192.168.1.200/api/acw; unset = off, so test and dev
instances send nothing). Features are named with the website's own words (job_view.label_of), so a new feature shows up
on the page without changes here. Best effort: a problem is logged once and never reaches a job.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import urllib.request

from . import broker, db, job_view

log = logging.getLogger("acw.dashboard_stats")

INTERVAL_S = 10  # live numbers
MONEY_S = 300  # the money summary
# Always sent (0 when idle) so their graphs are continuous; other features are added when a job has them.
FEATURES = ["Denoising", "Voice rebuild", "Leveling", "Subtitles & transcript", "Subtitles & transcript, extra accurate",
            "Echo removal", "Translation", "Video conversion", "Automatic settings"]
LIVE = ["Live noise removal", "Live captions", "Live captions, extra accurate"]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def features(kind: str, params: dict) -> list[str]:
    """What a job does, in the website's words, one entry per feature."""
    names = [part.strip() for part in job_view.label_of(kind, params).split(" + ") if part.strip()]
    opts = {**params, **(params.get("options") or {})}
    if opts.get("translate"):
        names.append("Translation")
    if opts.get("echo_reference_track"):
        names.append("Echo removal")
    return list(dict.fromkeys(names))


def live_features(denoise_tier: str | None, caption_tier: str | None) -> list[str]:
    out = []
    if denoise_tier and denoise_tier not in ("none", "off"):
        out.append("Live noise removal")
    if caption_tier and caption_tier not in ("none", "off"):
        out.append("Live captions, extra accurate" if caption_tier == "premium" else "Live captions")
    return out


def collect() -> dict[str, float]:
    gauges: dict[str, float] = {}
    for name in FEATURES:
        gauges[f"job_running_{slug(name)}"] = 0
        gauges[f"job_waiting_{slug(name)}"] = 0
    for name in LIVE:
        gauges[f"live_{slug(name)}"] = 0
    rows = db.all_rows("SELECT kind, status, params_json FROM jobs WHERE status IN ('queued', 'running', 'cloud_asr')")
    for r in rows:
        try:
            names = features(r["kind"], json.loads(r["params_json"]))
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
        state = "waiting" if r["status"] == "queued" else "running"
        for name in names:
            key = f"job_{state}_{slug(name)}"
            gauges[key] = gauges.get(key, 0) + 1
            gauges.setdefault(f"job_{'running' if state == 'waiting' else 'waiting'}_{slug(name)}", 0)
    gauges["jobs_running"] = sum(1 for r in rows if r["status"] != "queued")
    gauges["jobs_waiting"] = sum(1 for r in rows if r["status"] == "queued")
    sessions = db.all_rows("SELECT denoise_tier, caption_tier FROM stream_sessions WHERE status = 'active'")
    for s in sessions:
        for name in live_features(s["denoise_tier"], s["caption_tier"]):
            gauges[f"live_{slug(name)}"] = gauges.get(f"live_{slug(name)}", 0) + 1
    gauges["live_sessions"] = len(sessions)
    gauges["gpu_worker"] = 2 if broker.gpu_worker_up() and not broker.gpu_worker_idle() else 1 if broker.gpu_worker_up() else 0
    return gauges


def money() -> dict:
    """This month and last month (the owner's dashboard's own calculations), and the last 30 days as a series."""
    from . import owner_stats
    now = int(time.time())
    month = time.localtime(now)
    this = int(time.mktime((month.tm_year, month.tm_mon, 1, 0, 0, 0, 0, 0, -1)))
    prev_month = (month.tm_year - (month.tm_mon == 1), 12 if month.tm_mon == 1 else month.tm_mon - 1)
    prev = int(time.mktime((*prev_month, 1, 0, 0, 0, 0, 0, -1)))
    return {"month": owner_stats.overview(this, now), "prev": owner_stats.overview(prev, this),
            "series": owner_stats.series(now - 30 * 86400, now)["points"]}


def target() -> str | None:
    url = os.environ.get("ACW_DASHBOARD_URL", "").strip()
    return url or None


def post(url: str, body: dict) -> None:
    request = urllib.request.Request(url, data=json.dumps(body, default=str).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=5) as response:
        response.read()


async def send_loop() -> None:
    url = target()
    if url is None:
        return
    failing = False
    money_at = float("-inf")
    loop = asyncio.get_running_loop()
    while True:
        try:
            body = {"live": await asyncio.to_thread(collect)}
            with_money = loop.time() - money_at >= MONEY_S
            if with_money:
                body["money"] = await asyncio.to_thread(money)
            await asyncio.to_thread(post, url, body)
            if with_money:
                money_at = loop.time()  # only once it arrived: after an outage the money comes back on the next round
            failing = False
        except Exception:  # noqa: BLE001
            if not failing:
                log.exception("home page numbers could not be sent (will keep trying)")
            failing = True
        await asyncio.sleep(INTERVAL_S)
