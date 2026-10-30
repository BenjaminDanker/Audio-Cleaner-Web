"""Live numbers for the home dashboard: jobs running and waiting per feature, live sessions, the GPU worker.

Every 10 s the web process counts them, and every second it sends the latest count as statsd gauges to Netdata on this host (`ACW_DASHBOARD_STATSD`, e.g. 127.0.0.1:8125;
unset = off, so test and dev instances send nothing). Netdata streams them to the Raspberry Pi, whose home page (Host-Ops
netdata/home.html) shows them. Features are named with the website's own words (job_view.label_of), so a new feature shows up
on the page without changes here. Best effort: a problem is logged once and never reaches a job.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import socket

from . import broker, db, job_view

log = logging.getLogger("acw.dashboard_stats")

INTERVAL_S = 10  # how often the database is counted
SEND_S = 1  # how often the latest count is sent: Netdata draws a missed second as a gap
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


def target() -> tuple[str, int] | None:
    raw = os.environ.get("ACW_DASHBOARD_STATSD", "").strip()
    if not raw:
        return None
    host, _, port = raw.rpartition(":")
    try:
        return (host or "127.0.0.1", int(port))
    except ValueError:
        log.warning("ACW_DASHBOARD_STATSD=%r is not host:port; dashboard numbers are off", raw)
        return None


def send(gauges: dict[str, float], addr: tuple[str, int]) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        for key, value in gauges.items():
            udp.sendto(f"acw.{key}:{value}|g".encode(), addr)


async def send_loop() -> None:
    addr = target()
    if addr is None:
        return
    failing = False
    gauges: dict[str, float] | None = None
    counted_at = float("-inf")
    loop = asyncio.get_running_loop()
    while True:
        try:
            if loop.time() - counted_at >= INTERVAL_S:
                counted_at = loop.time()
                gauges = await asyncio.to_thread(collect)
                failing = False  # only a good count re-arms the log: rounds that just resend say nothing about it
            if gauges is not None:
                send(gauges, addr)
        except Exception:  # noqa: BLE001
            gauges = None  # a stale count is worse than a gap
            if not failing:
                log.exception("dashboard numbers could not be sent (will keep trying)")
            failing = True
        await asyncio.sleep(SEND_S)
