"""Metrics (OBS-01): one structured log line per finished job and live session, and the Prometheus text file the
monitor writes. Everything here is best effort: a metrics problem is logged and never reaches billing or a
session.

Log lines are single-line JSON on stderr (journald), no user ids, e-mail addresses or file names:

  {"event":"job_done","job":"…","kind":"denoise","tier":"standard","audio_s":612.4,"queue_wait_s":35,
   "run_s":71,"cost_micros":61240,"noise":"heavy","attempts":1,"input_bytes":98765432,…}
  {"event":"job_failed",…,"status":"failed","error":"Processing failed; you were not charged."}
  {"event":"live_session_end","session":"…","denoise":"light","captions":"local","audio_s":3601.2,"wall_s":3640,
   "cost_micros":…,"end_reason":"client_stop","frames":180060,"late_frames":12,"dropped_frames":0}

Find them with:  acw-logs --since -1d | grep '"event":"job_'
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

from . import db

_log = logging.getLogger("acw.metrics")
log = logging.getLogger("acw")

# A live frame answered later than this after it was sent counts as late (frames are 20 ms; the model adds 30 ms).
LATE_FRAME_SECONDS = 0.1


def _setup() -> None:
    if not _log.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(message)s"))
        _log.addHandler(h)
        _log.setLevel(logging.INFO)
        _log.propagate = False


def emit(event: str, **fields) -> None:
    """Write one JSON line. Fields with value None are left out."""
    _setup()
    rec = {"event": event, "ts": int(time.time()), **{k: v for k, v in fields.items() if v is not None}}
    _log.info(json.dumps(rec, separators=(",", ":"), default=str))


# ------------------------------------------------------------------ jobs

def noise_used(kind: str, params: dict, status: str) -> str:
    """Which noise remover a job got: light (DeepFilterNet3), heavy (DPDFNet-8), rebuilt (Sidon), none
    (transcription only), auto (chosen per file but not recorded: it did not get that far)."""
    if kind == "transcribe":
        return "none"
    if params.get("denoise_tier") == "rebuild" and status == "done":
        return "rebuilt"
    profile = params.get("noise_profile") or (params.get("options") or {}).get("noise")
    if profile in ("light", "heavy"):
        return profile
    return "auto" if status == "done" else "unknown"


def job_fields(row: dict) -> dict:
    params = json.loads(row["params_json"])
    started, finished = row.get("started_at"), row.get("finished_at")
    wait = row.get("wait_seconds")
    if wait is None and started is not None and row.get("queued_at") is not None and started >= row["queued_at"]:
        wait = started - row["queued_at"]
    return {
        "job": row["id"],
        "kind": row["kind"],
        "tier": row["tier"],
        "status": row["status"],
        "audio_s": round(float(row.get("duration_seconds") or row["claimed_seconds"] or 0), 1),
        "queue_wait_s": wait,
        "run_s": (finished - started) if started is not None and finished is not None else None,
        "cost_micros": row.get("cost_micros"),
        "noise": noise_used(row["kind"], params, row["status"]),
        "attempts": row.get("attempts"),
        "input_bytes": row.get("input_bytes"),
    }


def job_ended(job_id: str) -> None:
    """After a job reached done, failed or canceled (call it once the status change is committed)."""
    try:
        row = db.one("SELECT * FROM jobs WHERE id = ?", (job_id,))
        if row is None:
            return
        row = dict(row)
        fields = job_fields(row)
        if row["status"] == "done":
            emit("job_done", **fields)
        else:
            emit("job_failed", **fields, error=(row.get("error") or "")[:200] or None, stage=row.get("stage"))
    except Exception:  # noqa: BLE001
        log.exception("could not write the job metrics line")


# ------------------------------------------------------------------ live sessions

class FrameStats:
    """Round-trip bookkeeping for one live session: frames sent to the worker, frames answered, and how many
    answers came later than LATE_FRAME_SECONDS."""

    __slots__ = ("sent", "answered", "late", "worst")

    def __init__(self) -> None:
        self.sent = self.answered = self.late = 0
        self.worst = 0.0

    def answer(self, round_trip: float) -> None:
        self.answered += 1
        if round_trip > LATE_FRAME_SECONDS:
            self.late += 1
        if round_trip > self.worst:
            self.worst = round_trip

    @property
    def dropped(self) -> int:
        """Frames that never came back (the worker lost them, or the session ended with them in flight)."""
        return max(0, self.sent - self.answered)


def live_ended(sess_id: str, frames: FrameStats | None = None, wall_seconds: float | None = None) -> None:
    """After a live session was closed and billed."""
    try:
        row = db.one("SELECT * FROM stream_sessions WHERE id = ?", (sess_id,))
        if row is None:
            return
        row = dict(row)
        emit(
            "live_session_end",
            session=row["id"],
            denoise=row.get("denoise_tier"),
            captions=row.get("caption_tier"),
            audio_s=round(float(row.get("seconds") or 0), 1),
            wall_s=round(wall_seconds, 1) if wall_seconds is not None else (
                (row["ended_at"] - row["started_at"]) if row.get("ended_at") else None),
            cost_micros=row.get("cost_micros"),
            end_reason=row.get("end_reason"),
            frames=row.get("frames"),
            late_frames=row.get("late_frames"),
            dropped_frames=row.get("dropped_frames"),
            worst_round_trip_ms=round(frames.worst * 1000) if frames is not None and frames.sent else None,
        )
    except Exception:  # noqa: BLE001
        log.exception("could not write the live session metrics line")


# ------------------------------------------------------------------ Prometheus text file

def _esc(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class Prom:
    """Prometheus text exposition format (version 0.0.4), built in memory."""

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._seen: set[str] = set()

    def add(self, name: str, mtype: str, help_text: str, value: float | int | None, **labels) -> None:
        if value is None:
            return
        if name not in self._seen:
            self._seen.add(name)
            self._lines.append(f"# HELP {name} {help_text}")
            self._lines.append(f"# TYPE {name} {mtype}")
        lab = ",".join(f'{k}="{_esc(v)}"' for k, v in labels.items())
        v = repr(float(value)) if isinstance(value, float) else str(int(value))
        self._lines.append(f"{name}{{{lab}}} {v}" if lab else f"{name} {v}")

    def text(self) -> str:
        return "\n".join(self._lines) + "\n"


def collect(prom: Prom, *, now: int | None = None, data_dir: Path | None = None) -> None:
    """The database part of the metrics file."""
    from . import broker, stats

    now = db.now() if now is None else now
    q = db.one("SELECT COUNT(*) AS n FROM jobs WHERE status = 'queued'")["n"]
    prom.add("acw_queue_length", "gauge", "File jobs waiting for a worker.", q)
    for status in ("uploading", "running", "cloud_asr"):
        prom.add("acw_jobs_in_state", "gauge", "File jobs in this state right now.",
                 db.one("SELECT COUNT(*) AS n FROM jobs WHERE status = ?", (status,))["n"], status=status)
    live = db.one("SELECT COUNT(*) AS n FROM stream_sessions WHERE status = 'active'")["n"]
    prom.add("acw_live_sessions", "gauge", "Live sessions running now.", live)
    if data_dir is not None:
        engine = broker.live_engine_status(timeout=1.0)  # the live engine's own count, from its socket
        prom.add("acw_live_engine_up", "gauge", "The live engine answers (1) or not (0).", 0 if "down" in engine else 1)
        prom.add("acw_live_units", "gauge", "Live capacity units in use.", engine.get("units"))
        prom.add("acw_live_capacity_units", "gauge", "Live capacity units the engine accepts right now.", engine.get("capacityUnits"))
        prom.add("acw_live_caption_streams", "gauge", "Standard caption streams running.", engine.get("captionStreams"))
    day = now - 86400
    for status, n in stats.finished_jobs_by_status(day).items():
        prom.add("acw_jobs_finished_24h", "gauge", "File jobs that ended in the last 24 hours, by final status.", n, status=status)
    waits = stats.queue_waits(day)
    prom.add("acw_queue_wait_p95_seconds_24h", "gauge", "95th percentile of the queue wait of jobs that started in the last 24 hours.",
             stats.percentile(waits, 95) if waits else None)
    for status, n in stats.finished_jobs_by_status(0).items():
        prom.add("acw_jobs_finished_total", "counter", "File jobs that ever ended, by final status.", n, status=status)
    for reason, n in stats.live_end_reasons(0).items():
        prom.add("acw_live_sessions_ended_total", "counter", "Live sessions that ever ended, by reason.", n, end_reason=reason)
    tot = stats.live_totals(0)
    prom.add("acw_live_audio_seconds_total", "counter", "Seconds of live audio processed.", tot["seconds"])
    prom.add("acw_live_frames_total", "counter", "Live frames processed.", tot["frames"])
    prom.add("acw_live_late_frames_total", "counter", "Live frames answered more than 100 ms after they were sent.", tot["late_frames"])
    prom.add("acw_live_dropped_frames_total", "counter", "Live frames that never came back.", tot["dropped_frames"])
    prom.add("acw_usage_micros_total", "counter", "Customer usage charged, in micro-dollars (ledger).", stats.usage_micros(0))


def write_prom(path: Path, prom: Prom) -> None:
    """Atomically replace `path` (write next to it, then rename): a collector never reads a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        tmp.write_text(prom.text())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
