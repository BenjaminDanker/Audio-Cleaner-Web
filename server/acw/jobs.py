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
import math
import re
import shutil
import threading
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from . import db, metrics
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
# Disk promised to uploads that are still arriving, per account, and the most a file may weigh per second of its
# claimed length (320 Mbit/s: high-end video) unless it is small anyway. Without these a claim of a 4 GiB file
# that is half a second long reserves 4 GiB of disk for an hour at no cost.
MAX_RESERVED_UPLOAD_BYTES = 8 * 1024**3
MAX_BYTES_PER_SECOND = 40_000_000
PLAUSIBLE_ANY_SIZE_BYTES = 200 * 1024**2

# What a job needs before a worker may start it (measured 2026-10-14, see
# docs/operations.md). Decoded/cleaned audio are memory-mapped scratch files,
# so RAM is flat per job and disk grows with length.
GB = 1024**3
JOB_MEM = int(1.5 * GB)
JOB_MEM_REBUILD = 3 * GB  # Sidon peaks at ~2.5 GB
LOCAL_ASR_MEM = int(4.5 * GB)  # Parakeet fp32 (Whisper turbo needs ~2 GB)
SCRATCH_PER_HOUR_PER_CHANNEL = int(1.4 * GB)
# Wall-clock seconds per second of audio and per channel, at 4 threads, for queue estimates and the slow-job
# monitor. Stereo is processed channel by channel. What a job costs depends on the work it does: the noise
# remover (DeepFilterNet3 vs the heavier DPDFNet-8; "auto" is measured on the recording, so until the runner
# reports its choice the estimate sits in between), the voice rebuild instead of both, restoring highs
# (phone-quality recordings only, so only a small allowance), echo removal, and transcription.
SPEED = {
    "denoise_light": 0.05,
    "denoise_auto": 0.15,
    "denoise_heavy": 0.33,
    "denoise_rebuild": 0.9,
    "highs": 0.02,
    "echo": 0.03,
    "transcribe_local": 0.12,
    "transcribe_premium": 0.08,
}
MODEL_LOAD_SECONDS = 20  # every stage starts a fresh sandboxed process that loads its models
# Ceilings for the sandboxed stages per tier: CPU-seconds per second of audio (worst path: heavy stereo
# cleaning, highs restoration and local transcription) and wall-clock seconds per second of audio.
CPU_PER_AUDIO_SECOND = {"standard": 8, "rebuild": 12}
WALL_PER_AUDIO_SECOND = {"standard": 3, "rebuild": 4}
STAGE_SLACK_SECONDS = 1800
# An upload that has not finished this long after its last bytes arrived is canceled (the clock restarts while data flows).
UPLOAD_WINDOW_SECONDS = 3600
# ... but a job whose upload has not sent a single byte this long after it was created is canceled (it only holds disk space).
UPLOAD_FIRST_BYTE_SECONDS = 600
# Never let our work take the host below these.
MEM_RESERVE = 2 * GB
DISK_RESERVE = 10 * GB
# Order in which workers take queued jobs: paid before free (a paid cleanup with
# free subtitles counts as paid), then the user with the fewest jobs already
# running (so one big batch can't block everyone), then first come, first served.
QUEUE_ORDER = (
    "(j.free AND j.hold_id IS NULL), (SELECT COUNT(*) FROM jobs r WHERE r.user_id = j.user_id AND r.status IN ('running', 'cloud_asr')),"
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
    mem = (JOB_MEM_REBUILD if params.get("denoise_tier") == "rebuild" else JOB_MEM) + (
        LOCAL_ASR_MEM if params.get("transcribe_tier") == "local" else 0
    )
    return mem, scratch_bytes(job["kind"], seconds, echo=params.get("options", {}).get("echo_reference_track", False))


def job_seconds(job: dict) -> float:
    """Rough wall-clock time the whole job takes once started, from the work it will do."""
    params = json.loads(job["params_json"])
    audio = float(job.get("duration_seconds") or job["claimed_seconds"])
    channels = max(1, min(2, int(params.get("channels") or 2)))
    opts = params.get("options", {})
    t = 0.0
    if job["kind"] in ("denoise", "denoise_transcribe"):
        if params.get("denoise_tier") == "rebuild":
            t += SPEED["denoise_rebuild"] * audio * channels
        else:
            profile = params.get("noise_profile") or opts.get("noise", "auto")
            t += (SPEED["denoise_" + profile] if profile in ("light", "heavy") else SPEED["denoise_auto"]) * audio * channels
            t += SPEED["highs"] * audio * channels
    if opts.get("echo_reference_track"):
        t += SPEED["echo"] * audio * channels
    if params.get("transcribe_tier"):
        t += SPEED["transcribe_" + params["transcribe_tier"]] * audio
    return t + MODEL_LOAD_SECONDS


def is_rebuild(job: dict) -> bool:
    """A voice-rebuild job that still has its heavy stage ahead (or is in it)."""
    return job.get("next_stage") != "finalize" and json.loads(job["params_json"]).get("denoise_tier") == "rebuild"


def stage_limits(denoise_tier: str | None, seconds: float, max_seconds: float) -> tuple[int, int]:
    """(CPU-seconds, wall-clock seconds) the main stage of a job of `seconds` of audio may use."""
    tier = "rebuild" if denoise_tier == "rebuild" else "standard"
    wall = seconds * WALL_PER_AUDIO_SECOND[tier] + STAGE_SLACK_SECONDS
    if tier == "standard":
        wall = min(wall, max_seconds * 2 + STAGE_SLACK_SECONDS)
    return int(seconds * CPU_PER_AUDIO_SECOND[tier] + STAGE_SLACK_SECONDS), int(wall)


def queued_ids() -> list[str]:
    return [r["id"] for r in db.all_rows(f"SELECT j.id FROM jobs j WHERE j.status = 'queued' ORDER BY {QUEUE_ORDER}")]


class JobError(Exception):
    def __init__(self, message: str, status: int = 400, retry_after: int | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after  # seconds, sent as Retry-After with a 429


def seconds_until_utc_midnight() -> int:
    now = datetime.now(timezone.utc)
    return max(1, 86400 - (now.hour * 3600 + now.minute * 60 + now.second))


def utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def job_dir(job_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise JobError("bad job id", 404)
    return get_settings().jobs_dir / job_id


def clean_display_name(raw: str | None) -> str | None:
    """A name to show people instead of the upload's file name (which can be a re-encoded copy): path removed,
    control, bidirectional and zero-width characters dropped, whitespace collapsed, at most 120 characters."""
    if not raw:
        return None
    s = unicodedata.normalize("NFC", str(raw)).replace("\\", "/").rsplit("/", 1)[-1]
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")  # Cc, Cf (bidi, zero-width), Cs, Co, Cn
    s = " ".join(s.split())[:120].strip()
    return s or None


def display_name(job: dict) -> str:
    """What people see for this job's file."""
    return json.loads(job["params_json"]).get("display_name") or job["input_name"]


def job_label(kind: str, denoise_tier: str | None, transcribe_tier: str | None) -> str:
    """Customer-facing name of what a job does (the same words as the price list)."""
    parts = []
    if kind in ("denoise", "denoise_transcribe"):
        parts.append("Clean & polish, voice rebuilt" if denoise_tier == "rebuild" else "Clean & polish")
    if kind in ("transcribe", "denoise_transcribe"):
        parts.append("Subtitles & transcript" + (", extra accurate" if transcribe_tier == "premium" else ""))
    return " + ".join(parts)


def safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip(".-_")[:60]
    return stem or "audio"


def validate_options(kind: str, denoise_tier: str | None, transcribe_tier: str | None, options: dict, free: bool) -> dict:
    if kind not in KINDS:
        raise JobError("unknown job type")
    if free:
        # Free noise removal and free subtitles both run in the visitor's browser now; jobs here are always paid.
        raise JobError("Free subtitles are made in your browser now (choose “Free, in your browser”). Jobs here are paid.")
    if kind in ("denoise", "denoise_transcribe"):
        pricing.rate("denoise", denoise_tier or "")
    if kind in ("transcribe", "denoise_transcribe"):
        pricing.rate("transcribe", transcribe_tier or "")
    out: dict = {}
    fmt = options.get("audio_format", "wav")
    if not isinstance(fmt, str) or fmt not in AUDIO_FORMATS:
        raise JobError("unsupported output format")
    out["audio_format"] = fmt
    caps = options.get("caption_formats", ["srt", "vtt", "txt"])
    if not isinstance(caps, list) or not all(isinstance(x, str) for x in caps) or not set(caps) <= CAPTION_FORMATS:
        raise JobError("unsupported caption format")
    out["caption_formats"] = list(dict.fromkeys(caps))
    lang = options.get("language")
    if lang is not None and not (isinstance(lang, str) and LANG_RE.match(lang)):
        raise JobError("bad language code")
    out["language"] = lang
    # Translations of the subtitles (up to 3 languages), made from the transcript after it is written.
    from . import translate

    try:
        out["translate"] = translate.check_targets(options.get("translate"), lang)
    except ValueError as e:
        raise JobError(str(e)) from None
    if out["translate"] and kind not in ("transcribe", "denoise_transcribe"):
        raise JobError("translations need subtitles or a transcript")
    out["output_video"] = bool(options.get("output_video", True))
    out["embed_subtitles"] = bool(options.get("embed_subtitles", True))
    lufs = options.get("loudness_lufs", -16.0)
    if lufs is not None:
        try:
            lufs = float(lufs)
        except (TypeError, ValueError):
            raise JobError("loudness target must be a number")
        if not -30.0 <= lufs <= -9.0:
            raise JobError("loudness target must be between -30 and -9 LUFS")
    out["loudness_lufs"] = lufs
    try:
        atten = float(options.get("atten_lim_db", 100.0))
    except (TypeError, ValueError):
        raise JobError("noise strength must be a number")
    if not math.isfinite(atten):
        raise JobError("noise strength must be a number")
    out["atten_lim_db"] = min(max(atten, 6.0), 100.0)
    # Cancel the speaker echo in the first audio track using the second (desktop sound); ignored for one-track files.
    echo = options.get("echo_reference_track", False)
    if not isinstance(echo, bool):
        raise JobError("echo_reference_track must be true or false")
    out["echo_reference_track"] = echo
    # Which noise remover: measured per recording ("auto"), or forced for light / heavy background noise.
    noise = options.get("noise", "auto")
    if not isinstance(noise, str) or noise not in ("auto", "light", "heavy"):
        raise JobError("noise must be auto, light or heavy")
    out["noise"] = noise
    return out


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
    display_name: str | None = None,
) -> dict:
    settings = get_settings()
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise JobError("Unsupported file type.")
    if not 0 < size_bytes <= settings.max_upload_bytes:
        raise JobError(f"Files must be under {settings.max_upload_bytes // 1024**2} MB.", 413)
    if not 0.5 <= claimed_seconds <= pricing.MAX_JOB_SECONDS:
        raise JobError(f"Files must be under {pricing.MAX_JOB_SECONDS // 3600} hours.")
    if denoise_tier == "rebuild" and kind != "transcribe" and claimed_seconds > pricing.MAX_REBUILD_SECONDS:
        raise JobError(
            f"Voice rebuilding takes files up to {pricing.MAX_REBUILD_SECONDS // 60} minutes; use the standard cleanup for longer files."
        )
    if size_bytes > max(PLAUSIBLE_ANY_SIZE_BYTES, claimed_seconds * MAX_BYTES_PER_SECOND):
        raise JobError("That file is much larger than its length suggests. Please check the file and try again.")
    opts = validate_options(kind, denoise_tier, transcribe_tier, options, free)
    # The one hard limit: the upload and its scratch files must fit on disk, on top of what other uploads
    # that are still arriving have been promised (checked again, with the other limits, when the job is stored).
    from . import hostload

    disk_need = size_bytes + scratch_bytes(kind, claimed_seconds, echo=opts["echo_reference_track"]) + DISK_RESERVE
    disk_free = hostload.get(settings.data_dir).snapshot().disk_free
    if disk_free < disk_need:
        raise JobError("We're short on storage space right now. Please try again a bit later.", 507)

    job_id = db.new_id()
    params = {
        "options": opts,
        "denoise_tier": denoise_tier if kind != "transcribe" else None,
        "transcribe_tier": transcribe_tier if kind != "denoise" else None,
        "input_ext": ext,
        "display_name": clean_display_name(display_name),
    }
    # Prices are fixed when the job is created (the Terms), whatever happens to the price list while it waits.
    n_translate = len(opts["translate"])
    params["rates"] = pricing.job_rates(kind, params["denoise_tier"], params["transcribe_tier"], n_translate)
    est = pricing.job_cost_micros(kind, params["denoise_tier"], params["transcribe_tier"], claimed_seconds, params["rates"], n_translate)
    now = db.now()
    tier = "/".join(t for t in (params["denoise_tier"], params["transcribe_tier"]) if t)
    try:
        d = job_dir(job_id)
        d.mkdir(parents=True, mode=0o700)
        # The limits, the money and the row are one step: parallel requests can't all pass a check that
        # none of them has recorded yet (the money rolls back with everything else).
        with db.tx() as c:
            active = c.execute("SELECT COUNT(*) AS n FROM jobs WHERE user_id = ? AND status IN (?,?,?,?)", (user_id, *ACTIVE)).fetchone()
            if active["n"] >= MAX_ACTIVE_JOBS_PER_USER:
                raise JobError("You already have several jobs in progress; wait for one to finish.", 429, retry_after=60)
            # Bytes promised to uploads that are still arriving are not free space any more.
            promised = c.execute("SELECT COALESCE(SUM(input_bytes), 0) AS n FROM jobs WHERE status = 'uploading'").fetchone()["n"]
            mine = c.execute(
                "SELECT COALESCE(SUM(input_bytes), 0) AS n FROM jobs WHERE status = 'uploading' AND user_id = ?", (user_id,)
            ).fetchone()["n"]
            if mine + size_bytes > MAX_RESERVED_UPLOAD_BYTES:
                raise JobError("You have large uploads waiting already; wait for them to finish.", 429, retry_after=60)
            if disk_free - promised < disk_need:
                raise JobError("We're short on storage space right now. Please try again a bit later.", 507)
            hold_id = None
            if est > 0:
                try:
                    hold_id = ledger.create_hold(user_id, est, "job", job_id, c=c)
                except ledger.InsufficientFunds as e:
                    raise JobError(
                        f"This job costs about ${est / 1e6:.2f}; your available balance is ${e.available / 1e6:.2f}.", 402
                    ) from e
            c.execute(
                "INSERT INTO jobs(id, user_id, kind, tier, free, status, params_json, input_name, input_bytes,"
                " claimed_seconds, hold_id, est_cost_micros, created_at, expires_at)"
                " VALUES (?,?,?,?,?, 'uploading', ?,?,?,?,?,?,?,?)",
                (job_id, user_id, kind, tier, 0, json.dumps(params), filename[:200], size_bytes,
                 claimed_seconds, hold_id, est, now, now + UPLOAD_FIRST_BYTE_SECONDS),
            )
    except Exception:
        shutil.rmtree(job_dir(job_id), ignore_errors=True)
        raise
    return {"id": job_id, "estCostMicros": est}


def input_path(job: dict) -> Path:
    params = json.loads(job["params_json"])
    return job_dir(job["id"]) / f"input{params['input_ext']}"


def update_params(job_id: str, **changes) -> dict:
    """Merge `changes` into a job's stored parameters; returns the result."""
    with db.tx() as c:
        row = c.execute("SELECT params_json FROM jobs WHERE id = ?", (job_id,)).fetchone()
        params = {**json.loads(row["params_json"]), **changes}
        c.execute("UPDATE jobs SET params_json = ? WHERE id = ?", (json.dumps(params), job_id))
    return params


def set_notice(job_id: str, notice: str) -> None:
    """A note for the person about how the job was handled (shown with the job; not an error).
    Adds to a note that is already there."""
    with db.tx() as c:
        row = c.execute("SELECT params_json FROM jobs WHERE id = ?", (job_id,)).fetchone()
        params = json.loads(row["params_json"])
        if params.get("notice") and notice not in params["notice"]:
            notice = f"{params['notice']} {notice}"
        params["notice"] = notice
        c.execute("UPDATE jobs SET params_json = ? WHERE id = ?", (json.dumps(params), job_id))


def mark_uploaded(job_id: str) -> None:
    db.execute(
        "UPDATE jobs SET status = 'queued', next_stage = 'probe', queued_at = ?, expires_at = NULL"
        " WHERE id = ? AND status = 'uploading'",
        (db.now(), job_id),
    )
    invalidate_queue_cache()


def finish_failed(
    job: dict, message: str, status: str = "failed", only_from: tuple[str, ...] | None = None, not_stage: str | None = None
) -> bool:
    """Release money/quota and delete files. Safe to call twice. The status change, the hold and the free
    allowance move in one transaction, so a crash can't leave money reserved for a finished job.
    `only_from` limits it to jobs still in one of those states (cancel: not once a worker has started) and
    `not_stage` to jobs whose next stage is another one (cancel: not once the paid-for transcription is done).
    True when the job was moved to `status` by this call."""
    with db.tx() as c:
        cur = c.execute("SELECT status, next_stage, free, claimed_seconds, params_json FROM jobs WHERE id = ?", (job["id"],)).fetchone()
        if cur is None or cur["status"] in ("done", "failed", "canceled", "expired"):
            return False
        if only_from is not None and cur["status"] not in only_from:
            return False
        if not_stage is not None and cur["next_stage"] == not_stage:
            return False
        c.execute(
            "UPDATE jobs SET status = ?, error = ?, finished_at = ?, expires_at = ? WHERE id = ?",
            (status, message[:500], db.now(), db.now() + 86400, job["id"]),
        )
        if job.get("hold_id"):
            ledger.release_hold(job["hold_id"], c=c)
        if cur["free"]:
            # claimed_seconds is what is reserved: creation reserves it, the probe corrects it.
            c.execute(
                "UPDATE free_usage SET seconds = MAX(0, seconds - ?) WHERE user_id = ? AND day = ?",
                (float(cur["claimed_seconds"]), job["user_id"], json.loads(cur["params_json"])["free_day"]),
            )
    invalidate_queue_cache()
    shutil.rmtree(job_dir(job["id"]), ignore_errors=True)
    metrics.job_ended(job["id"])
    return True


def requeue(job: dict) -> None:
    """Put a running job back in the queue untouched: the attempt it used is given back, its upload stays and
    its scratch files go (the stage starts again from the upload; after cloud transcription it keeps the
    segments and chunks it needs)."""
    with db.tx() as c:
        n = c.execute(
            "UPDATE jobs SET status = 'queued', worker_id = NULL, attempts = MAX(0, attempts - 1), progress = 0, stage = NULL,"
            " queued_at = ? WHERE id = ? AND status = 'running'",
            (db.now(), job["id"]),
        ).rowcount
    if not n:
        return
    d = job_dir(job["id"])
    if job.get("next_stage") == "finalize":
        return  # finalize only reads what the earlier stages left
    keep = f"input{json.loads(job['params_json'])['input_ext']}"
    for p in d.iterdir():
        if p.name == keep:
            continue
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


def _seconds_left(r: dict, now: int) -> float:
    left = job_seconds(r)
    if r["status"] != "queued" and r.get("started_at"):
        left = max(10.0, left - (now - r["started_at"]))
    return left


def queue_wait_seconds(ahead: list[str]) -> int:
    """Estimate until a job behind `ahead` starts: the work in front of it
    (including what is running now) spread over the workers."""
    rows = db.all_rows(
        "SELECT * FROM jobs WHERE status IN ('running', 'cloud_asr')"
        + (f" OR id IN ({','.join('?' * len(ahead))})" if ahead else ""),
        tuple(ahead),
    )
    now = db.now()
    work = sum(_seconds_left(dict(r), now) for r in rows)
    return int(work / max(1, get_settings().workers))


# Everyone looking at the job list sees the same queue. The ordering and the estimates cost a query per queued job
# and run in the process that also relays live audio, so they are computed once a second for all requests
# (a user's own changes drop the cache so they see them at once).
QUEUE_CACHE_SECONDS = 1.0
_queue_cache: dict = {"key": None, "at": 0.0, "ids": [], "waits": {}}
_queue_cache_lock = threading.Lock()


def invalidate_queue_cache() -> None:
    with _queue_cache_lock:
        _queue_cache["key"] = None


def queue_snapshot() -> tuple[list[str], dict[str, int]]:
    """(queued job ids in start order, estimated seconds until each starts), at most QUEUE_CACHE_SECONDS old.
    One pass for the whole queue: the estimate for a job is the work running now plus the work queued in front of it,
    spread over the workers (the same figure queue_wait_seconds gives)."""
    key = str(get_settings().db_path)
    with _queue_cache_lock:
        if _queue_cache["key"] == key and time.monotonic() - _queue_cache["at"] < QUEUE_CACHE_SECONDS:
            return _queue_cache["ids"], _queue_cache["waits"]
        # Computed under the lock: concurrent requests wait for one computation instead of each running their own.
        ids = queued_ids()
        now = db.now()
        workers = max(1, get_settings().workers)
        rows = {r["id"]: dict(r) for r in db.all_rows("SELECT * FROM jobs WHERE status IN ('running', 'cloud_asr', 'queued')")}
        work = sum(_seconds_left(r, now) for r in rows.values() if r["status"] != "queued")
        waits: dict[str, int] = {}
        for jid in ids:
            waits[jid] = int(work / workers)
            if jid in rows:
                work += _seconds_left(rows[jid], now)
        _queue_cache.update(key=key, at=time.monotonic(), ids=ids, waits=waits)
        return ids, waits


def public_job(job: dict, queue: tuple[list[str], dict[str, int]] | None = None) -> dict:
    """`queue` is queue_snapshot() (pass it once when listing many jobs)."""
    outputs = json.loads(job["outputs_json"]) if job.get("outputs_json") else []
    params = json.loads(job["params_json"])
    position = eta = None
    if job["status"] == "queued":
        ids, waits = queue_snapshot() if queue is None else queue
        if job["id"] in waits:
            position = ids.index(job["id"]) + 1
            eta = waits[job["id"]]
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
        "inputName": params.get("display_name") or job["input_name"],
        "label": job_label(job["kind"], params.get("denoise_tier"), params.get("transcribe_tier")),
        "durationSeconds": job["duration_seconds"] or job["claimed_seconds"],
        "estCostMicros": job["est_cost_micros"],
        "costMicros": job["cost_micros"],
        "options": params["options"],
        "notice": params.get("notice"),
        "outputs": [{"name": o["file"], "kind": o["kind"], "bytes": o.get("bytes"), **({"lang": o["lang"]} if o.get("lang") else {})} for o in outputs],
        "createdAt": job["created_at"],
        "finishedAt": job["finished_at"],
        "expiresAt": job["expires_at"] if job["status"] == "done" else None,
    }


def sweep() -> None:
    """Expire finished outputs and abandoned uploads; recover stuck jobs."""
    now = db.now()
    for job in db.all_rows("SELECT * FROM jobs WHERE status = 'uploading' AND expires_at <= ?", (now,)):
        finish_failed(dict(job), "Upload was not completed.", status="canceled")
    # 'canceling' was a transient state of older versions of the cancel request.
    for job in db.all_rows("SELECT * FROM jobs WHERE status = 'canceling'"):
        finish_failed(dict(job), "Canceled.", status="canceled")
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
