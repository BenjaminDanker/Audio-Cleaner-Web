"""File-job lifecycle shared by the API and the worker.

uploading -> queued -> running -> (cloud_asr -> queued[finalize] -> running) -> done
                                  `-> failed / canceled            done -> expired

Money: paid jobs hold the estimated cost (from the client's claimed
duration) at creation; the worker re-prices from the probed duration before
doing the heavy work and captures the real cost on success. Free jobs reserve
seconds of the daily allowance the same way.
"""
from __future__ import annotations

import json
import re
import shutil
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from . import db
from .billing import ledger, pricing
from .config import get_settings

KINDS = {"denoise", "transcribe", "denoise_transcribe"}
AUDIO_FORMATS = {"wav", "flac", "mp3", "m4a", "opus"}
CAPTION_FORMATS = {"srt", "vtt", "txt", "json"}
# Anything FFmpeg reads well; the sandboxed probe decides for real (ffmpeg.ALLOWED_FORMATS).
ALLOWED_EXTS = {
    # audio
    ".wav", ".rf64", ".w64", ".mp3", ".mp2", ".mpa", ".m4a", ".m4b", ".m4r", ".aac", ".flac", ".ogg", ".oga",
    ".opus", ".spx", ".wma", ".aiff", ".aif", ".aifc", ".caf", ".au", ".snd", ".amr", ".ac3", ".eac3", ".dts",
    ".mka", ".wv", ".ape", ".tta", ".mpc",
    # video
    ".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi", ".wmv", ".asf", ".flv", ".f4v", ".3gp", ".3g2", ".ts",
    ".mts", ".m2ts", ".mpg", ".mpeg", ".vob", ".mxf", ".dv", ".ogv", ".rm", ".rmvb",
}
LANG_RE = re.compile(r"^[a-z]{2,3}$")
ACTIVE = ("uploading", "queued", "running", "cloud_asr")
MAX_ACTIVE_JOBS_PER_USER = 3

# What a job needs before a worker may start it (measured 2026-10-14, see
# docs/operations.md). Decoded/cleaned audio are memory-mapped scratch files,
# so RAM is flat per job and disk grows with length.
GB = 1024**3
JOB_MEM = int(1.5 * GB)
LOCAL_ASR_MEM = int(4.5 * GB)  # Parakeet fp32 (Whisper turbo needs ~2 GB)
SCRATCH_PER_HOUR_PER_CHANNEL = int(1.4 * GB)
# Wall-clock seconds per second of audio at 4 threads, for queue estimates.
SPEED = {"denoise": 0.1, "transcribe_local": 0.12, "transcribe_premium": 0.08}
# Never let our work take the host below these.
MEM_RESERVE = 2 * GB
DISK_RESERVE = 10 * GB
# Order in which workers take queued jobs: paid before free, then the user with
# the fewest jobs already running (so one big batch can't block everyone), then
# first come, first served.
QUEUE_ORDER = (
    "j.free, (SELECT COUNT(*) FROM jobs r WHERE r.user_id = j.user_id AND r.status IN ('running', 'cloud_asr')),"
    " j.queued_at"
)


def scratch_bytes(kind: str, seconds: float, channels: int = 2, echo: bool = False) -> int:
    denoise = kind in ("denoise", "denoise_transcribe")
    if not (denoise or echo):
        return 0
    # Whole-file buffers, each a half "channel-hour": decoded + cleaned per channel; echo removal adds its
    # result per channel and the mono reference (without denoising it also needs the decoded mic).
    buffers = 2 * channels if denoise else 0
    if echo:
        buffers += (channels if denoise else 2 * channels) + 1
    return int(seconds / 3600 * buffers / 2 * SCRATCH_PER_HOUR_PER_CHANNEL)


def job_needs(job: dict) -> tuple[int, int]:
    """(memory, scratch disk) in bytes that `job`'s next stage needs."""
    if job.get("next_stage") == "finalize":
        return int(0.3 * GB), 0
    params = json.loads(job["params_json"])
    seconds = float(job.get("duration_seconds") or job["claimed_seconds"])
    mem = JOB_MEM + (LOCAL_ASR_MEM if params.get("transcribe_tier") == "local" else 0)
    return mem, scratch_bytes(job["kind"], seconds, echo=params.get("options", {}).get("echo_reference_track", False))


def job_seconds(job: dict) -> float:
    """Rough wall-clock time the whole job takes once started."""
    params = json.loads(job["params_json"])
    audio = float(job.get("duration_seconds") or job["claimed_seconds"])
    t = 0.0
    if job["kind"] in ("denoise", "denoise_transcribe"):
        t += SPEED["denoise"] * audio
    if params.get("transcribe_tier"):
        t += SPEED["transcribe_" + params["transcribe_tier"]] * audio
    return t + 10


def queued_ids() -> list[str]:
    return [r["id"] for r in db.all_rows(f"SELECT j.id FROM jobs j WHERE j.status = 'queued' ORDER BY {QUEUE_ORDER}")]


class JobError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def job_dir(job_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise JobError("bad job id", 404)
    return get_settings().jobs_dir / job_id


def safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip(".-_")[:60]
    return stem or "audio"


def validate_options(kind: str, denoise_tier: str | None, transcribe_tier: str | None, options: dict, free: bool) -> dict:
    if kind not in KINDS:
        raise JobError("unknown job type")
    if kind in ("denoise", "denoise_transcribe"):
        if free:
            raise JobError("Free jobs are transcription only; denoise runs free in your browser.")
        pricing.rate("denoise", denoise_tier or "")
    if kind in ("transcribe", "denoise_transcribe"):
        if free and transcribe_tier != "local":
            raise JobError("Free transcription uses the standard model.")
        pricing.rate("transcribe", transcribe_tier or "")
    out: dict = {}
    fmt = options.get("audio_format", "wav")
    if fmt not in AUDIO_FORMATS:
        raise JobError("unsupported output format")
    out["audio_format"] = fmt
    caps = options.get("caption_formats", ["srt", "vtt", "txt"])
    if not isinstance(caps, list) or not set(caps) <= CAPTION_FORMATS:
        raise JobError("unsupported caption format")
    out["caption_formats"] = list(dict.fromkeys(caps))
    lang = options.get("language")
    if lang is not None and not (isinstance(lang, str) and LANG_RE.match(lang)):
        raise JobError("bad language code")
    out["language"] = lang
    out["output_video"] = bool(options.get("output_video", True))
    out["embed_subtitles"] = bool(options.get("embed_subtitles", True))
    lufs = options.get("loudness_lufs", -16.0)
    if lufs is not None:
        lufs = float(lufs)
        if not -30.0 <= lufs <= -9.0:
            raise JobError("loudness target must be between -30 and -9 LUFS")
    out["loudness_lufs"] = lufs
    atten = float(options.get("atten_lim_db", 100.0))
    out["atten_lim_db"] = min(max(atten, 6.0), 100.0)
    # Cancel the speaker echo in the first audio track using the second (desktop sound); ignored for one-track files.
    echo = options.get("echo_reference_track", False)
    if not isinstance(echo, bool):
        raise JobError("echo_reference_track must be true or false")
    out["echo_reference_track"] = echo
    # Which noise remover: measured per recording ("auto"), or forced for light / heavy background noise.
    noise = options.get("noise", "auto")
    if noise not in ("auto", "light", "heavy"):
        raise JobError("noise must be auto, light or heavy")
    out["noise"] = noise
    return out


def free_seconds_left(user_id: str) -> int:
    """Free standard transcription left today (UTC)."""
    row = db.one("SELECT seconds FROM free_usage WHERE user_id = ? AND day = ?", (user_id, utc_day()))
    return int(max(0.0, pricing.FREE_TRANSCRIBE_SECONDS_PER_DAY - (row["seconds"] if row else 0.0)))


def _reserve_free(c, user_id: str, seconds: float) -> None:
    day = utc_day()
    row = c.execute("SELECT seconds FROM free_usage WHERE user_id = ? AND day = ?", (user_id, day)).fetchone()
    used = row["seconds"] if row else 0.0
    if used + seconds > pricing.FREE_TRANSCRIBE_SECONDS_PER_DAY:
        left = max(0, pricing.FREE_TRANSCRIBE_SECONDS_PER_DAY - used) / 60
        raise JobError(f"Daily free transcription limit reached ({left:.0f} min left today).", 402)
    c.execute(
        "INSERT INTO free_usage(user_id, day, seconds) VALUES (?,?,?)"
        " ON CONFLICT(user_id, day) DO UPDATE SET seconds = seconds + excluded.seconds",
        (user_id, day, seconds),
    )


def adjust_free(user_id: str, day: str, delta_seconds: float) -> None:
    db.execute(
        "UPDATE free_usage SET seconds = MAX(0, seconds + ?) WHERE user_id = ? AND day = ?",
        (delta_seconds, user_id, day),
    )


def create_job(
    user_id: str,
    *,
    kind: str,
    denoise_tier: str | None,
    transcribe_tier: str | None,
    filename: str,
    size_bytes: int,
    claimed_seconds: float,
    options: dict,
    free: bool,
) -> dict:
    settings = get_settings()
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise JobError("Unsupported file type.")
    if not 0 < size_bytes <= settings.max_upload_bytes:
        raise JobError(f"Files must be under {settings.max_upload_bytes // 1024**2} MB.", 413)
    if not 0.5 <= claimed_seconds <= pricing.MAX_JOB_SECONDS:
        raise JobError(f"Files must be under {pricing.MAX_JOB_SECONDS // 3600} hours.")
    if free and claimed_seconds > pricing.FREE_TRANSCRIBE_MAX_FILE_SECONDS:
        raise JobError(f"Free transcription is limited to {pricing.FREE_TRANSCRIBE_MAX_FILE_SECONDS // 60} minutes per file.")
    opts = validate_options(kind, denoise_tier, transcribe_tier, options, free)
    active = db.one("SELECT COUNT(*) AS n FROM jobs WHERE user_id = ? AND status IN (?,?,?,?)", (user_id, *ACTIVE))
    if active["n"] >= MAX_ACTIVE_JOBS_PER_USER:
        raise JobError("You already have several jobs in progress; wait for one to finish.", 429)
    # The one hard limit: the upload and its scratch files must fit on disk.
    from . import hostload

    disk_need = size_bytes + scratch_bytes(kind, claimed_seconds, echo=opts["echo_reference_track"]) + DISK_RESERVE
    if hostload.get(settings.data_dir).snapshot().disk_free < disk_need:
        raise JobError("We're short on storage space right now. Please try again a bit later.", 507)

    job_id = db.new_id()
    params = {
        "options": opts,
        "denoise_tier": denoise_tier if kind != "transcribe" else None,
        "transcribe_tier": transcribe_tier if kind != "denoise" else None,
        "input_ext": ext,
        "free_day": utc_day() if free else None,
    }
    est = 0 if free else pricing.job_cost_micros(kind, params["denoise_tier"], params["transcribe_tier"], claimed_seconds)
    hold_id = None
    if free:
        with db.tx() as c:
            _reserve_free(c, user_id, claimed_seconds)
    elif est > 0:
        try:
            hold_id = ledger.create_hold(user_id, est, "job", job_id)
        except ledger.InsufficientFunds as e:
            raise JobError(
                f"This job costs about ${est / 1e6:.2f}; your available balance is ${e.available / 1e6:.2f}.", 402
            ) from e
    now = db.now()
    tier = "/".join(t for t in (params["denoise_tier"], params["transcribe_tier"]) if t)
    try:
        d = job_dir(job_id)
        d.mkdir(parents=True, mode=0o700)
        db.execute(
            "INSERT INTO jobs(id, user_id, kind, tier, free, status, params_json, input_name, input_bytes,"
            " claimed_seconds, hold_id, est_cost_micros, created_at, expires_at)"
            " VALUES (?,?,?,?,?, 'uploading', ?,?,?,?,?,?,?,?)",
            (job_id, user_id, kind, tier, 1 if free else 0, json.dumps(params), filename[:200], size_bytes,
             claimed_seconds, hold_id, est, now, now + 3600),
        )
    except Exception:
        if hold_id:
            ledger.release_hold(hold_id)
        if free:
            adjust_free(user_id, params["free_day"], -claimed_seconds)
        raise
    return {"id": job_id, "estCostMicros": est}


def input_path(job: dict) -> Path:
    params = json.loads(job["params_json"])
    return job_dir(job["id"]) / f"input{params['input_ext']}"


def set_notice(job_id: str, notice: str) -> None:
    """A note for the person about how the job was handled (shown with the job; not an error)."""
    with db.tx() as c:
        row = c.execute("SELECT params_json FROM jobs WHERE id = ?", (job_id,)).fetchone()
        params = {**json.loads(row["params_json"]), "notice": notice}
        c.execute("UPDATE jobs SET params_json = ? WHERE id = ?", (json.dumps(params), job_id))


def mark_uploaded(job_id: str) -> None:
    db.execute(
        "UPDATE jobs SET status = 'queued', next_stage = 'probe', queued_at = ?, expires_at = NULL"
        " WHERE id = ? AND status = 'uploading'",
        (db.now(), job_id),
    )


def finish_failed(job: dict, message: str, status: str = "failed") -> None:
    """Release money/quota and delete files. Safe to call twice."""
    with db.tx() as c:
        cur = c.execute("SELECT status FROM jobs WHERE id = ?", (job["id"],)).fetchone()
        if cur is None or cur["status"] in ("done", "failed", "canceled", "expired"):
            return
        c.execute(
            "UPDATE jobs SET status = ?, error = ?, finished_at = ?, expires_at = ? WHERE id = ?",
            (status, message[:500], db.now(), db.now() + 86400, job["id"]),
        )
    if job.get("hold_id"):
        ledger.release_hold(job["hold_id"])
    if job.get("free"):
        params = json.loads(job["params_json"])
        adjust_free(job["user_id"], params["free_day"], -float(job.get("duration_seconds") or job["claimed_seconds"]))
    shutil.rmtree(job_dir(job["id"]), ignore_errors=True)


def queue_wait_seconds(ahead: list[str]) -> int:
    """Estimate until a job behind `ahead` starts: the work in front of it
    (including what is running now) spread over the workers."""
    rows = db.all_rows(
        "SELECT * FROM jobs WHERE status IN ('running', 'cloud_asr')"
        + (f" OR id IN ({','.join('?' * len(ahead))})" if ahead else ""),
        tuple(ahead),
    )
    now = db.now()
    work = 0.0
    for r in rows:
        r = dict(r)
        left = job_seconds(r)
        if r["status"] != "queued" and r.get("started_at"):
            left = max(10.0, left - (now - r["started_at"]))
        work += left
    return int(work / max(1, get_settings().workers))


def public_job(job: dict, queue: list[str] | None = None) -> dict:
    outputs = json.loads(job["outputs_json"]) if job.get("outputs_json") else []
    params = json.loads(job["params_json"])
    position = eta = None
    if job["status"] == "queued":
        queue = queued_ids() if queue is None else queue
        if job["id"] in queue:
            position = queue.index(job["id"]) + 1
            eta = queue_wait_seconds(queue[: position - 1])
    return {
        "queuePosition": position,
        "queueWaitSeconds": eta,
        "id": job["id"],
        "kind": job["kind"],
        "tier": job["tier"],
        "free": bool(job["free"]),
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"],
        "error": job["error"],
        "inputName": job["input_name"],
        "durationSeconds": job["duration_seconds"] or job["claimed_seconds"],
        "estCostMicros": job["est_cost_micros"],
        "costMicros": job["cost_micros"],
        "options": params["options"],
        "notice": params.get("notice"),
        "outputs": [{"name": o["file"], "kind": o["kind"], "bytes": o.get("bytes")} for o in outputs],
        "createdAt": job["created_at"],
        "finishedAt": job["finished_at"],
        "expiresAt": job["expires_at"] if job["status"] == "done" else None,
    }


def sweep() -> None:
    """Expire finished outputs and abandoned uploads; recover stuck jobs."""
    now = db.now()
    for job in db.all_rows("SELECT * FROM jobs WHERE status = 'uploading' AND expires_at <= ?", (now,)):
        finish_failed(dict(job), "Upload was not completed.", status="canceled")
    for job in db.all_rows("SELECT id FROM jobs WHERE status = 'done' AND expires_at <= ?", (now,)):
        shutil.rmtree(job_dir(job["id"]), ignore_errors=True)
        db.execute("UPDATE jobs SET status = 'expired', outputs_json = NULL WHERE id = ? AND status = 'done'", (job["id"],))
    # Running jobs whose worker stopped heartbeating (crash/restart): retry once.
    for job in db.all_rows("SELECT * FROM jobs WHERE status = 'running' AND heartbeat_at < ?", (now - 120,)):
        job = dict(job)
        if job["attempts"] >= 2:
            finish_failed(job, "Processing failed repeatedly; you were not charged.")
        else:
            db.execute(
                "UPDATE jobs SET status = 'queued', worker_id = NULL WHERE id = ? AND status = 'running' AND heartbeat_at < ?",
                (job["id"], now - 120),
            )
    # Remove directories of failed/canceled jobs that may have been left behind.
    jobs_root = get_settings().jobs_dir
    if jobs_root.exists():
        known = {r["id"] for r in db.all_rows("SELECT id FROM jobs WHERE status IN ('uploading','queued','running','cloud_asr','done')")}
        for d in jobs_root.iterdir():
            if d.is_dir() and re.fullmatch(r"[0-9a-f]{32}", d.name) and d.name not in known:
                if now - int(d.stat().st_mtime) > 3600:
                    shutil.rmtree(d, ignore_errors=True)
