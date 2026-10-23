"""File-job lifecycle shared by the API and the worker.

uploading -> queued -> running -> (cloud_asr -> queued[finalize] -> running) -> done
                                  `-> failed / canceled            done -> expired

Money: paid jobs hold the estimated cost (from the client's claimed
duration) at creation; the worker re-prices from the probed duration before
doing the heavy work and captures the real cost on success. Free jobs reserve
seconds of the daily allowance the same way.

Every job is made one way: with a timeline (editing.py). A job asked for with whole-file options gets the timeline they stand for
(editing.timeline_from_options); a finished job is changed by editing its timeline, which runs only what is missing.

What a job keeps: its upload, the per-part editor's block cache and recording copy (editing.py) and its results stay in the job's
folder until `expiry`: 2 hours (`job_retention_seconds`) after its last edit, and never later than 6 hours
(`job_max_retention_seconds`) after the upload arrived. They are deleted together (`sweep`, or at once with "Delete now"). A failed or
canceled job's files go at once.
"""
from __future__ import annotations

import hmac
import json
import math
import re
import shutil
import threading
import time
import unicodedata
from pathlib import Path

from . import automatic, db, keycaps, metrics
from .billing import ledger, pricing
from .config import get_settings

# "convert": a video made in another file type or picture size, nothing else (options.video_out).
KINDS = {"denoise", "transcribe", "denoise_transcribe", "convert"}
AUDIO_FORMATS = {"wav", "flac", "mp3", "m4a", "opus"}
# options.video_out: the video result's file type and the shorter side of its picture (None: the original size).
VIDEO_OUT_FORMATS = ("mp4", "mov", "mkv", "webm")
VIDEO_OUT_SIZES = (480, 720, 1080, 1440, 2160)
CAPTION_FORMATS = {"srt", "vtt", "txt", "json"}
# Anything FFmpeg reads well; the sandboxed probe decides for real (ffmpeg.ALLOWED_FORMATS).
AUDIO_EXTS = {
    ".wav", ".rf64", ".w64", ".mp3", ".mp2", ".mpa", ".m4a", ".m4b", ".m4r", ".aac", ".flac", ".ogg", ".oga",
    ".opus", ".spx", ".wma", ".aiff", ".aif", ".aifc", ".caf", ".au", ".snd", ".amr", ".ac3", ".eac3", ".dts",
    ".mka", ".wv", ".ape", ".tta", ".mpc",
}
ALLOWED_EXTS = {
    *AUDIO_EXTS,
    # video
    ".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi", ".wmv", ".asf", ".flv", ".f4v", ".3gp", ".3g2", ".ts",
    ".mts", ".m2ts", ".mpg", ".mpeg", ".vob", ".mxf", ".dv", ".ogv", ".rm", ".rmvb",
}
LANG_RE = re.compile(r"^[a-z]{2,3}$")
ACTIVE = ("uploading", "queued", "running", "cloud_asr")
# A job that would only finish this close to the moment its files are cleared away (6 hours after the upload) is stopped instead: nobody
# could download its results in time, so it ends as failed and is not charged.
MIN_DELIVERY_SECONDS = 60
TOO_LATE = "This took longer than expected and your files were about to be cleared away, so it was stopped. You weren’t charged."
# An edit run needs this long before the files go, on top of what it takes (editing.start_run).
DEADLINE_MARGIN_SECONDS = 300
# The editor's preview plays short stretches of the recording and of what each step made (stretch.py). A stretch is at most this many
# seconds.
STRETCH_MAX_SECONDS = 30
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
# Re-measured 2026-10-17 (docs/research/heavy-2026-10.md): a heavy stereo clean peaks at 0.2 GB of private memory
# and 0.8 GB including the mapped scratch files; rounded up.
JOB_MEM = 1 * GB
# Parakeet fp32: 0.3 GB of private memory per job plus the 2.4 GB of weights, which are the model file's own page
# cache, shared by every concurrent Parakeet job (the first one pays for them); Whisper turbo needs ~1.7 GB.
LOCAL_ASR_MEM = int(3.2 * GB)
SCRATCH_PER_HOUR_PER_CHANNEL = int(1.4 * GB)
# Wall-clock seconds per second of audio and per channel, at 4 threads, for queue estimates and the slow-job
# monitor. Stereo is processed channel by channel. What a job costs depends on the work it does: the noise
# remover (DeepFilterNet3 vs the heavier DPDFNet-8; "auto" is measured on the recording, so until the runner
# reports its choice the estimate sits in between), the voice rebuild instead of both, echo removal, and transcription.
SPEED = {
    "denoise_light": 0.08,  # x1.54 since the 30 s block grid with 16 s of context (common/acw_common/blocks.py)
    "denoise_auto": 0.25,
    "denoise_heavy": 0.49,  # x1.8 since the 30 s block grid with 24 s of context
    "denoise_rebuild": 0.14,  # on the card (the only place it runs): 8.2 s per minute (broker.GPU_WORK)
    "leveling": 0.01,  # Leveling on its own (with noise removal it is part of the denoise figures)
    "echo": 0.03,
    "transcribe_local": 0.12,
    "transcribe_premium": 0.08,
    "suggest": 0.22,  # the search for suggestions (acw_common.suggest, 0.22 s per audio second measured at 2 threads), a first run's extra and an on-demand run's all
    "encode": 0.32,  # the picture made again (options.video_out): x264 on 4 threads, 1080p30 (the card's NVENC: 0.14)
}
MODEL_LOAD_SECONDS = 12  # a job runs two or three fresh sandboxed stages that each load their models (measured 1.1 to 4.8 s per stage, 2026-10-19)
# Ceilings for the sandboxed stages per tier: CPU-seconds per second of audio (worst path: heavy stereo
# cleaning and local transcription) and wall-clock seconds per second of audio.
CPU_PER_AUDIO_SECOND = 8
WALL_PER_AUDIO_SECOND = 3
STAGE_SLACK_SECONDS = 1800
# An upload that has not finished this long after its last bytes arrived is canceled (the clock restarts while data flows).
UPLOAD_WINDOW_SECONDS = 3600
# ... but a job whose upload has not sent a single byte this long after it was created is canceled (it only holds disk space).
UPLOAD_FIRST_BYTE_SECONDS = 600
# Never let our work take the host below these.
# Free system memory (MemAvailable) left after a job starts: the Minecraft memory controller closes new world starts at
# 4 GiB available and reopens at 6 GiB, so a job that took the host down to 2 GB shut players out until it ended.
MEM_RESERVE = 6 * GB
# Room kept under the acw slice's own MemoryMax (a separate limit, 12.5 GB: a 6 GB reserve there would stop a 6 GB
# local transcription from ever running next to the web and live processes).
SLICE_RESERVE = 2 * GB
DISK_RESERVE = 10 * GB
# Order in which workers take queued jobs: the user with the fewest jobs already running first (so one big batch
# can't block everyone), then first come, first served.
QUEUE_ORDER = (
    "(SELECT COUNT(*) FROM jobs r WHERE r.user_id = j.user_id AND r.status IN ('running', 'cloud_asr')), j.queued_at"
)


def params_of(job_id: str) -> dict | None:
    """A job's stored parameters as they are now (the copy in a claimed job dict can be older)."""
    row = db.one("SELECT params_json FROM jobs WHERE id = ?", (job_id,))
    return json.loads(row["params_json"]) if row else None


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
    if params.get("run"):  # an edit run decodes the recording and renders into scratch files (editing.py)
        run = params["run"]
        from . import edit_subtitles, editing

        return (JOB_MEM + (LOCAL_ASR_MEM if edit_subtitles.local_asr_wanted(params) else 0),
                scratch_bytes("denoise", seconds, int(params.get("channels") or 2), echo=bool(run.get("echo"))))
    from . import edit_subtitles, editing

    # (an automatic job not decided yet still has its whole-file subtitle tier: automatic.apply)
    local = edit_subtitles.local_asr_wanted(params) or params.get("transcribe_tier") == "local"
    return (JOB_MEM + (LOCAL_ASR_MEM if local else 0),
            scratch_bytes(job["kind"], seconds, int(params.get("channels") or 2), echo=editing.wants_echo(params)))


def job_seconds(job: dict) -> float:
    """Rough wall-clock time the whole job takes once started, from the work it will do."""
    params = json.loads(job["params_json"])
    audio = float(job.get("duration_seconds") or job["claimed_seconds"])
    channels = max(1, min(2, int(params.get("channels") or 2)))
    opts = params.get("options", {})
    if params.get("run"):  # an edit run: what its blocks take (editing.py), and the render
        return float(params["run"].get("est_seconds") or 0.0) + MODEL_LOAD_SECONDS
    t = SPEED["encode"] * audio if encodes(params) else 0.0
    if (params.get("edit") or {}).get("first_run"):  # every job but a conversion: only its timeline's parts
        from . import editing

        t += editing.timeline_seconds(params["edit"]["timeline"], audio, channels)
        return t + MODEL_LOAD_SECONDS
    if automatic.pending(params):  # not decided yet: the most it can be (Denoising or Voice rebuild on the whole file, and the rest)
        t += max(SPEED["denoise_auto"], SPEED["denoise_rebuild"]) * audio * channels
        if opts.get("echo_reference_track"):
            t += SPEED["echo"] * audio * channels
        if params.get("transcribe_tier"):
            t += SPEED["transcribe_" + params["transcribe_tier"]] * audio
    return t + MODEL_LOAD_SECONDS


def encodes(params: dict) -> bool:
    """Will the job make the picture of its video again (options.video_out)? Until its file is read it may (None counts)."""
    return bool(params.get("options", {}).get("video_out")) and params.get("picture_encode") is not False


def _convert_rate(params: dict) -> int:
    """Micros a minute of the video conversion: the pinned rate times the picture's size factor (convert_factor, two decimals), in whole
    numbers (a float product can land a hair above the exact one and round up a micro)."""
    hundredths = round(float(params.get("convert_factor") or 1.0) * 100)
    return -(-int(params["convert_rate"]) * hundredths // 100)


def convert_cost(params: dict, seconds: float) -> int:
    """What the video conversion of a job costs for `seconds` (its pinned `convert_rate`, times `convert_factor`: 1 up to 1080p at 30
    frames a second, more for a bigger or faster picture): only when the picture is made again (`picture_encode`). Both are worked out
    from what the client read of the video (video_out.source) and checked when the file is read."""
    if not params.get("convert_rate") or not params.get("picture_encode"):
        return 0
    return pricing.cost_micros("convert", "standard", seconds, _convert_rate(params))


def convert_line(params: dict, seconds: float) -> dict | None:
    """The price line of a video conversion (with its size `factor`): `maybe` when it can't be told yet whether the picture is made again
    (the client did not say the video's codec; then its micros, what it would cost, are not part of the total); `copied`: the picture is
    copied as it is, free (micros 0)."""
    if not params.get("convert_rate"):
        return None
    rate = _convert_rate(params)
    micros = 0 if params.get("picture_encode") is False else pricing.cost_micros("convert", "standard", seconds, rate)
    return {"item": "convert", "label": pricing.rate("convert", "standard").label, "quantity": 1, "usdPerMinute": rate / pricing.MICROS_PER_USD,
            "factor": float(params.get("convert_factor") or 1.0), "micros": micros,
            **({"maybe": True} if params.get("picture_encode") is None else {}), **({"copied": True} if params.get("picture_encode") is False else {})}


_CODEC_RE = re.compile(r"^[a-z0-9_]{1,32}$")


# Echo removal asked for on a file with one audio track: there is no desktop sound to remove the echo with. The whole job is refused (a job
# whose file is known) or fails before any work, uncharged, so nobody pays for part of what they asked for.
NO_SECOND_TRACK = ("This file has only one audio track, so there is no desktop sound to remove the echo with. Send it without echo removal "
                   "(nothing was charged).")


def video_out_of(options: dict) -> dict | None:
    """options.video_out checked: {"format": one of VIDEO_OUT_FORMATS, "size": one of VIDEO_OUT_SIZES or None, "source": optional
    {"codec": FFmpeg's codec name, "width", "height", "fps"}: the video as the client read it, so the price is known before the upload}."""
    vo = options.get("video_out")
    if vo is None:
        return None
    if not isinstance(vo, dict) or set(vo) - {"format", "size", "source"}:
        raise JobError("video_out must be {format, size, source?}")
    fmt = vo.get("format")
    if fmt not in VIDEO_OUT_FORMATS:
        raise JobError(f"video_out.format must be one of {', '.join(VIDEO_OUT_FORMATS)}")
    size = vo.get("size")
    if size is not None and (isinstance(size, bool) or size not in VIDEO_OUT_SIZES):
        raise JobError(f"video_out.size must be one of {', '.join(map(str, VIDEO_OUT_SIZES))}, or null for the original size")
    out: dict = {"format": fmt, "size": size}
    src = vo.get("source")
    if src is not None:
        if not isinstance(src, dict) or set(src) - {"codec", "width", "height", "fps"}:
            raise JobError("video_out.source must be {codec?, width?, height?, fps?}")
        clean: dict = {}
        if src.get("codec") is not None:
            if not isinstance(src["codec"], str) or not _CODEC_RE.fullmatch(src["codec"]):
                raise JobError("video_out.source.codec must be FFmpeg's codec name, like h264")
            clean["codec"] = src["codec"]
        for key, lo, hi in (("width", 16, 16384), ("height", 16, 16384), ("fps", 1, 240)):
            v = src.get(key)
            if v is None:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
                raise JobError(f"video_out.source.{key} must be a number from {lo} to {hi}")
            clean[key] = int(v) if key != "fps" else float(v)
        if ("width" in clean) != ("height" in clean):
            raise JobError("video_out.source needs both width and height, or neither")
        out["source"] = clean
    return out


def convert_params(opts: dict, input_ext: str | None = None) -> dict:
    """What a job with a video conversion stores: the rate (pinned like every price), whether the picture is made again
    (`picture_encode`: a new size, or a file type that can't hold the video's codec; MKV holds any; None when the codec is not known)
    and the size factor (`convert_factor`), from what the client read of the video (video_out.source); the worker checks both when it
    reads the file. `input_ext`: the upload's ending (".mp4"), when known."""
    from acw_common.ffmpeg import convert_factor, picture_copies, picture_size

    vo = opts.get("video_out")
    if not vo:
        return {}
    if input_ext in AUDIO_EXTS:  # a sound: its video gets a plain black picture, free
        return {"convert_rate": pricing.rate("convert", "standard").micros_per_minute, "picture_encode": False, "convert_factor": 1.0}
    src = vo.get("source") or {}
    w, h, fps = src.get("width"), src.get("height"), src.get("fps")
    resize = vo["size"] is not None and not (w and h and min(w, h) == vo["size"])
    copies = picture_copies(src.get("codec"), input_ext, "." + vo["format"])
    encode = True if resize else None if copies is None else not copies
    if w and h:
        factor = convert_factor(w, h, fps, picture_size(w, h, 1.0, 0, vo["size"]) if resize else None)
    elif vo["size"]:
        factor = convert_factor(*picture_size(16, 9, 1.0, 0, vo["size"]), fps)  # (the size of the original not said: 16:9 at the size asked)
    else:
        factor = 1.0
    return {"convert_rate": pricing.rate("convert", "standard").micros_per_minute, "picture_encode": encode, "convert_factor": factor}


def stage_limits(seconds: float, max_seconds: float) -> tuple[int, int]:
    """(CPU-seconds, wall-clock seconds) the main stage of a job of `seconds` of audio may use."""
    wall = min(seconds * WALL_PER_AUDIO_SECOND, max_seconds * 2) + STAGE_SLACK_SECONDS
    return int(seconds * CPU_PER_AUDIO_SECOND + STAGE_SLACK_SECONDS), int(wall)


# The wall-clock limit of a stage assumes the job runs at the speed it was claimed with. When live audio or a game server
# takes cores, the same work takes 2 to 3 times longer (a 90 min voice rebuild: 9,700 s at 4 threads, 20,000 to 29,000 s on
# one core against a limit of 23,400 s), and killing it at the limit throws away hours of work. So at the limit a stage
# that is still making progress gets more time, as much as its own progress says it needs (with some margin), but never
# more than STRETCH_LIMIT times the original limit; a stage whose progress has not moved for STALL_SECONDS is stopped.
STALL_SECONDS = 1800
STRETCH_LIMIT = 3.0
MIN_STRETCH_SECONDS = 60  # a stretch shorter than this is not worth it


def stretched_limit(base: float, elapsed: float, start_progress: int, progress: int, since_progress: float) -> float | None:
    """The new wall-clock limit of a stage (seconds from its start) for a stage that has run `elapsed` seconds, went from
    `start_progress` to `progress` percent and last moved `since_progress` seconds ago; None when it should be stopped now."""
    done = progress - start_progress
    if done <= 0 or progress >= 100 or since_progress > STALL_SECONDS:
        return None
    projected = elapsed * (100 - start_progress) / done
    limit = min(projected * 1.2 + STAGE_SLACK_SECONDS / 3, base * STRETCH_LIMIT)
    return limit if limit > elapsed + MIN_STRETCH_SECONDS else None


def queued_ids() -> list[str]:
    return [r["id"] for r in db.all_rows(f"SELECT j.id FROM jobs j WHERE j.status = 'queued' ORDER BY {QUEUE_ORDER}")]


class JobError(Exception):
    def __init__(self, message: str, status: int = 400, retry_after: int | None = None, code: str | None = None, extra: dict | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after  # seconds, sent as Retry-After with a 429
        self.code = code  # sent as X-Error-Code and "code" in the answer, for clients that want to tell this error from others
        self.extra = extra  # more fields of the answer (the quote of an export that needs processing)


TOO_MANY_JOBS = "too_many_active_jobs"


def too_many_jobs_message(n: int) -> str:
    return f"You have {n} {'file' if n == 1 else 'files'} being processed. Start this one when one of them is done."


def check_active_limit(c, user_id: str) -> None:
    """An account may have `max_active_jobs` file jobs at once (uploading, waiting, running; re-runs count): one more is
    refused with 429 and the code `too_many_active_jobs`. Called inside the transaction that stores the new job."""
    limit = get_settings().max_active_jobs
    n = c.execute("SELECT COUNT(*) AS n FROM jobs WHERE user_id = ? AND status IN (?,?,?,?)", (user_id, *ACTIVE)).fetchone()["n"]
    if n >= limit:
        raise JobError(too_many_jobs_message(n), 429, retry_after=60, code=TOO_MANY_JOBS)


def expiry(now: int, uploaded_at: int | None) -> int:
    """When a job's files are deleted if its last edit is at `now`: 2 hours later, and never later than 6 hours after the upload
    arrived (`uploaded_at`; unknown: as if it arrived now)."""
    settings = get_settings()
    return min(now + settings.job_retention_seconds, int(uploaded_at or now) + settings.job_max_retention_seconds)


def hard_deadline(job: dict) -> int:
    """The latest moment a job's files can still be there: 6 hours after the upload arrived."""
    return int(job.get("uploaded_at") or job["created_at"]) + get_settings().job_max_retention_seconds


def touch(job_id: str) -> None:
    """An edit (a quote, a timeline save, a process or an export): the job's files go 2 hours from now instead (never past the 6 hours
    after the upload). Only a finished job; one that is running gets its deadline when it finishes."""
    now = db.now()
    cap = get_settings().job_max_retention_seconds
    db.execute(
        "UPDATE jobs SET edited_at = ?, expires_at = MIN(?, COALESCE(uploaded_at, created_at) + ?) WHERE status = 'done' AND id = ?",
        (now, now + get_settings().job_retention_seconds, cap, job_id),
    )


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


def job_label(kind: str, denoise_tier: str | None, transcribe_tier: str | None, leveling: bool = False) -> str:
    """Customer-facing name of what a job does (the same words as the price list). `leveling`: Leveling is part of it
    (for a stored job: "leveling" in its pinned rates)."""
    parts = []
    if kind == "convert":
        return pricing.rate("convert", "standard").label
    if kind in ("denoise", "denoise_transcribe"):
        if denoise_tier != pricing.NO_DENOISE:
            parts.append("Voice rebuild" if denoise_tier == "rebuild" else "Denoising")
        if leveling or denoise_tier == pricing.NO_DENOISE:
            parts.append("Leveling")
    if kind in ("transcribe", "denoise_transcribe"):
        parts.append("Subtitles & transcript" + (", extra accurate" if transcribe_tier == "premium" else ""))
    return " + ".join(parts)


def safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip(".-_")[:60]
    return stem or "audio"


def validate_options(kind: str, denoise_tier: str | None, transcribe_tier: str | None, options: dict) -> dict:
    if kind not in KINDS:
        raise JobError("unknown job type")
    if kind in ("denoise", "denoise_transcribe") and denoise_tier != pricing.NO_DENOISE:
        pricing.rate("denoise", denoise_tier or "")
    if kind in ("transcribe", "denoise_transcribe"):
        pricing.rate("transcribe", transcribe_tier or "")
    if kind == "convert" and (denoise_tier or transcribe_tier):
        raise JobError("A conversion (kind convert) has no denoiseTier or transcribeTier.")
    out: dict = {}
    fmt = options.get("audio_format", "flac")  # lossless like WAV, about half the size to download
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
    # The video result in another file type or picture size (made from the whole video, so it has to be uploaded whole).
    out["video_out"] = video_out_of(options)
    if out["video_out"] and not out["output_video"]:
        raise JobError("video_out needs output_video")
    if kind == "convert" and not out["video_out"]:
        raise JobError("A conversion (kind convert) needs options.video_out: the file type and picture size of the video.")
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
    # Even out the voice and set the loudness (on by default; its own price).
    leveling = options.get("leveling", True)
    if not isinstance(leveling, bool):
        raise JobError("leveling must be true or false")
    out["leveling"] = leveling
    # Automatic settings (docs/api.md): the server measures the upload and decides what the request did not give itself. The values
    # above stand in for what is decided until then (automatic.py).
    auto = options.get("automatic", False)
    if not isinstance(auto, bool):
        raise JobError("automatic must be true or false")
    out["automatic"] = auto
    if auto and kind not in ("denoise", "denoise_transcribe"):
        raise JobError("Automatic settings choose how the sound is cleaned: use them with noise removal (kind denoise or denoise_transcribe).")
    if auto and denoise_tier != "standard":
        raise JobError("Automatic settings need denoiseTier standard: the server picks Voice rebuild itself where the voice needs it. "
                       "For Voice rebuild, or Leveling only, choose the settings yourself.")
    if kind in ("denoise", "denoise_transcribe") and denoise_tier == pricing.NO_DENOISE and not out["leveling"]:
        raise JobError("Choose Denoising, Voice rebuild or Leveling.")
    return out


REBUILD_PAUSED = "Voice rebuilding is paused right now. Try again later, or choose Denoising."


def check_duration(kind: str, denoise_tier: str | None, seconds: float) -> None:
    """The length of the file fits the job, and the job can be done now (the voice rebuild runs only on the GPU)."""
    from .broker import rebuild_available

    if denoise_tier == "rebuild" and kind != "transcribe" and not rebuild_available():
        raise JobError(REBUILD_PAUSED)
    if not 0.5 <= seconds <= pricing.MAX_JOB_SECONDS:
        raise JobError(f"Files must be under {pricing.MAX_JOB_SECONDS // 3600} hours.")
    if denoise_tier == "rebuild" and kind != "transcribe" and seconds > pricing.MAX_REBUILD_SECONDS:
        raise JobError(
            f"Voice rebuilding takes files up to {pricing.MAX_REBUILD_SECONDS // 60} minutes; use the standard cleanup for longer files."
        )


def price_lines(rates: dict[str, int], d_tier: str | None, t_tier: str | None, seconds: float, n_translate: int) -> list[dict]:
    """The parts of a price, in price-list order: a line for each step in the pinned `rates`."""
    lines = []
    if "denoise" in rates:
        r = pricing.rate("denoise", d_tier)
        lines.append({"item": "denoise", "label": r.label, "quantity": 1, "usdPerMinute": rates["denoise"] / pricing.MICROS_PER_USD,
                      "micros": pricing.cost_micros("denoise", d_tier, seconds, rates["denoise"])})
    if "echo" in rates:
        r = pricing.rate("echo", "standard")
        lines.append({"item": "echo", "label": r.label, "quantity": 1, "usdPerMinute": rates["echo"] / pricing.MICROS_PER_USD,
                      "micros": pricing.cost_micros("echo", "standard", seconds, rates["echo"])})
    if "leveling" in rates:
        r = pricing.rate("leveling", "standard")
        lines.append({"item": "leveling", "label": r.label, "quantity": 1, "usdPerMinute": rates["leveling"] / pricing.MICROS_PER_USD,
                      "micros": pricing.cost_micros("leveling", "standard", seconds, rates["leveling"])})
    if "transcribe" in rates:
        r = pricing.rate("transcribe", t_tier)
        lines.append({"item": "transcribe", "label": r.label, "quantity": 1, "usdPerMinute": rates["transcribe"] / pricing.MICROS_PER_USD,
                      "micros": pricing.cost_micros("transcribe", t_tier, seconds, rates["transcribe"])})
    if "translate" in rates:
        r = pricing.rate("transcribe_translate", "standard")
        lines.append({"item": "translate", "label": r.label, "quantity": n_translate, "usdPerMinute": rates["translate"] / pricing.MICROS_PER_USD,
                      "micros": n_translate * pricing.cost_micros("transcribe_translate", "standard", seconds, rates["translate"])})
    return lines


def automatic_plan(kind: str, t_tier: str | None, raw_options: dict, opts: dict) -> dict | None:
    """What a job made with `automatic` stores to be decided later (params["automatic"]): the options the request gave itself and every
    outcome the decision can end in, with the prices it would pin. None for any other job. `raw_options`: the request's options as
    sent, `opts`: validated."""
    if not opts["automatic"]:
        return None
    from . import edit_prices

    fixed = automatic.fixed_fields(raw_options)
    # (the prices of the timeline it becomes once decided are the ones of today, like any job's: automatic.apply)
    return {"fixed": fixed, "candidates": automatic.candidates(kind, t_tier, opts, fixed, len(opts["translate"])),
            "edit_rates": edit_prices.step_rates(), "subtitle_rates": edit_prices.subtitle_rates()}


def estimate(kind: str, denoise_tier: str | None, transcribe_tier: str | None, duration_seconds: float, options: dict,
             timeline: object | None = None) -> dict:
    """The price of a job before it exists (no hold, nothing stored): the same checks and the same price as creating
    it, with the parts of the price listed. POST /api/jobs/estimate. With `options.automatic` the price is the most the
    decision could end in (`maximum`), and `outcomes` lists the ways it can end."""
    if timeline is None and kind != "convert" and not (options or {}).get("automatic"):
        # every job is made one way: whole-file options are the timeline they stand for (editing.timeline_from_options)
        from . import editing

        check_duration(kind, denoise_tier, duration_seconds)
        asked = validate_options(kind, denoise_tier, transcribe_tier, options)
        timeline, options = editing.timeline_from_options(kind, denoise_tier, transcribe_tier, asked, duration_seconds)
        denoise_tier = transcribe_tier = None
    if timeline is not None:
        from . import editing

        timeline, options, transcribe_tier = editing.subtitles_into_timeline(timeline, transcribe_tier, options, duration_seconds)
        return estimate_timeline(kind, denoise_tier, transcribe_tier, duration_seconds, options, timeline)
    check_duration(kind, denoise_tier, duration_seconds)
    opts = validate_options(kind, denoise_tier, transcribe_tier, options)
    # (only an automatic job, whose settings are decided once its file is read, and a conversion get here)
    t_tier = transcribe_tier if kind != "denoise" else None
    plan = automatic_plan(kind, t_tier, options, opts)
    lines: list[dict] = []
    total = 0
    if plan:
        from . import editing

        cands = automatic.within(plan["candidates"], duration_seconds)
        top, total = automatic.dearest(kind, t_tier, cands, duration_seconds, opts)
        # the lines are the dearest outcome's: the timeline it would become (automatic.apply)
        lines = editing.options_price(kind, top["tier"], t_tier, {**opts, "leveling": "leveling" in top["rates"]}, duration_seconds)[1]
    cv = convert_params(opts)
    total += convert_cost(cv, duration_seconds)
    line = convert_line(cv, duration_seconds)
    out = {
        "estCostMicros": total,
        "estCostUsd": total / pricing.MICROS_PER_USD,
        "durationSeconds": duration_seconds,
        "maxSeconds": pricing.MAX_JOB_SECONDS,
        "lines": [*lines, *([line] if line else [])],
        "options": opts,
    }
    if plan:
        out["maximum"] = True  # what is held; the charge is for the steps that actually run, and can be less
        out["outcomes"] = [{"label": automatic.outcome_label(kind, c), "denoiseTier": c["tier"],
                            "estCostMicros": automatic.cost(kind, t_tier, c, duration_seconds, opts)} for c in cands]
        out["fixed"] = plan["fixed"]
    return out


def estimate_timeline(kind: str, denoise_tier: str | None, transcribe_tier: str | None, duration_seconds: float, options: dict,
                      timeline: object) -> dict:
    """The price of a job made with a timeline: each step's parts, from an empty cache (editing.py), and the subtitles."""
    from . import edit_prices, edit_subtitles, editing

    tl, d_tier = editing.prepare_timeline_job(kind, denoise_tier, options, duration_seconds, timeline, transcribe_tier)
    if not 0.5 <= duration_seconds <= pricing.MAX_JOB_SECONDS:
        raise JobError(f"Files must be under {pricing.MAX_JOB_SECONDS // 3600} hours.")
    opts = validate_options(editing.options_kind(kind, tl), d_tier, transcribe_tier, options)
    rates = edit_prices.step_rates()
    micros, lines, _ = editing.price_timeline(tl, duration_seconds, rates)
    n = len(opts["translate"])
    t_tier = transcribe_tier if kind != "denoise" and not tl.layers["subtitles"] else None
    t_rates = pricing.job_rates("transcribe", None, t_tier, n)
    total = micros + (pricing.job_cost_micros("transcribe", None, t_tier, duration_seconds, t_rates, n) if t_tier else 0)
    text_lines = price_lines(t_rates, None, t_tier, duration_seconds, n) if t_tier else []
    if tl.layers["subtitles"]:  # the subtitle parts: each at its quality's rate and its translations (free parts cost nothing)
        sub = edit_subtitles.subtitle_plan(None, tl, edit_prices.subtitle_rates())
        total += sub["micros"]
        text_lines = sub["lines"]
    cv = convert_params(opts)
    total += convert_cost(cv, duration_seconds)
    line = convert_line(cv, duration_seconds)
    if line:
        text_lines = [*text_lines, line]
    return {
        "estCostMicros": total,
        "estCostUsd": total / pricing.MICROS_PER_USD,
        "durationSeconds": duration_seconds,
        "maxSeconds": pricing.MAX_JOB_SECONDS,
        "lines": [*lines, *text_lines],
        "options": opts,
        "timeline": tl.to_json(),
    }


class Replay(Exception):
    """create_job found the Idempotency-Key already used by a job: nothing was created."""

    def __init__(self, job_id: str, est_cost_micros: int):
        super().__init__(job_id)
        self.job_id = job_id
        self.est_cost_micros = est_cost_micros


IDEMPOTENCY_SECONDS = 86400


def replay_of(user_id: str, key: str, body_hash: str, c=None) -> dict | None:
    """The job an Idempotency-Key of this account (used in the last 24 hours) already made, as the create answer;
    None for a new key. The same key with other job details is a conflict (409). `c`: inside a transaction."""
    q = c.execute if c is not None else db.conn().execute
    row = q(
        "SELECT k.body_hash, k.job_id, k.created_at, j.est_cost_micros, j.params_json, j.claimed_seconds, j.duration_seconds FROM idempotency_keys k LEFT JOIN jobs j ON j.id = k.job_id"
        " WHERE k.user_id = ? AND k.key = ?",
        (user_id, key),
    ).fetchone()
    if row is None or row["created_at"] <= db.now() - IDEMPOTENCY_SECONDS:
        return None
    if not hmac.compare_digest(row["body_hash"].encode(), body_hash.encode()):
        raise JobError("This Idempotency-Key was already used for a different job. Use a new key for a new job.", 409)
    params = json.loads(row["params_json"]) if row["params_json"] is not None else None
    maximum = params is not None and automatic.pending(params)
    from . import edit_subtitles

    needs = edit_subtitles.pending_needs({"id": row["job_id"], "claimed_seconds": row["claimed_seconds"], "duration_seconds": row["duration_seconds"]}, params) \
        if params is not None else None
    return {"id": row["job_id"], "estCostMicros": row["est_cost_micros"] or 0, **({"maximum": True} if maximum else {}),
            **({"needs": needs} if needs else {})}


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
    display_name: str | None = None,
    api_key_id: str | None = None,
    callback_url: str | None = None,
    idempotency: tuple[str, str] | None = None,
    timeline: object | None = None,
) -> dict:
    settings = get_settings()
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise JobError("Unsupported file type.")
    if not 0 < size_bytes <= settings.max_upload_bytes:
        raise JobError(f"Files must be under {settings.max_upload_bytes // 1024**2} MB.", 413)
    tl = None
    if timeline is not None and kind == "convert":
        raise JobError("A conversion (kind convert) has no timeline.")
    whole_file = None
    if timeline is None and kind != "convert" and not (options or {}).get("automatic"):
        # Every job is made one way: whole-file options are the timeline they stand for (editing.timeline_from_options), made again for the
        # file's real length once it is read (editing.refit_whole_file).
        from . import edit_prices, edit_store, edit_subtitles, editing

        check_duration(kind, denoise_tier, claimed_seconds)
        asked = validate_options(kind, denoise_tier, transcribe_tier, options)
        whole_file = {"kind": kind, "denoise_tier": denoise_tier, "transcribe_tier": transcribe_tier, "options": asked}
        timeline, options = editing.timeline_from_options(kind, denoise_tier, transcribe_tier, asked, claimed_seconds)
        denoise_tier = transcribe_tier = None
    elif timeline is not None:
        from . import edit_prices, edit_store, edit_subtitles, editing

        timeline, options, transcribe_tier = editing.subtitles_into_timeline(timeline, transcribe_tier, options, claimed_seconds)
    if timeline is not None:
        from . import edit_prices, edit_store, edit_subtitles, editing

        tl, denoise_tier = editing.prepare_timeline_job(kind, denoise_tier, options, claimed_seconds, timeline, transcribe_tier)
    check_duration(kind, denoise_tier, claimed_seconds)
    if size_bytes > max(PLAUSIBLE_ANY_SIZE_BYTES, claimed_seconds * MAX_BYTES_PER_SECOND):
        raise JobError("That file is much larger than its length suggests. Please check the file and try again.")
    opts = validate_options(editing.options_kind(kind, tl) if tl is not None else kind, denoise_tier, transcribe_tier, options)
    if whole_file:  # (the job's record says the sound choices that were asked for; its timeline is what is made of them)
        opts = {**opts, **{k: whole_file["options"][k] for k in editing.REPLACED_OPTIONS if k in whole_file["options"]}}
    # The one hard limit: the upload and its scratch files must fit on disk, on top of what other uploads
    # that are still arriving have been promised (checked again, with the other limits, when the job is stored).
    from . import hostload

    disk_need = size_bytes + scratch_bytes(kind, claimed_seconds, echo=opts["echo_reference_track"] or bool(tl and tl.layers["echo"])) + DISK_RESERVE
    if kind in ("denoise", "denoise_transcribe"):
        from . import edit_prices, edit_store, edit_subtitles, editing

        disk_need += edit_store.cache_bytes(claimed_seconds, 2, editing.steps_in(tl) if tl is not None else 1)
    if opts["video_out"]:
        disk_need += size_bytes  # the converted video, about as big as the upload
    disk_free = hostload.get(settings.data_dir).snapshot().disk_free
    if disk_free < disk_need:
        raise JobError("We're short on storage space right now. Please try again a bit later.", 507)

    job_id = db.new_id()
    params = {
        "options": opts,
        "denoise_tier": denoise_tier if kind != "transcribe" else None,
        "transcribe_tier": transcribe_tier if kind != "denoise" and not (tl is not None and tl.layers["subtitles"]) else None,
        "input_ext": ext,
        "display_name": clean_display_name(display_name),
        **convert_params(opts, ext),
        **({"whole_file": whole_file} if whole_file else {}),
    }
    # Prices are fixed when the job is created (the Terms), whatever happens to the price list while it waits.
    plan = automatic_plan(kind, params["transcribe_tier"], options, opts)
    if tl is not None:
        # A job made with a timeline: only its parts are processed and charged (editing.py); the subtitles are priced as always.
        from . import edit_prices, edit_store, edit_subtitles, editing

        params["edit"] = {"timeline": tl.to_json(), "first_run": True}
        params["edit_rates"] = edit_prices.step_rates()
        if tl.layers["subtitles"]:
            params["subtitle_rates"] = edit_prices.subtitle_rates()
        params["rates"] = {}  # (its subtitles are parts of the timeline too)
        est = editing.first_run_cost(kind, params, claimed_seconds)
    elif plan:
        # held for the most it can cost; the decision (automatic.py) shrinks the hold to what will be charged
        params["automatic"] = plan
        top, est = automatic.dearest(kind, params["transcribe_tier"], automatic.within(plan["candidates"], claimed_seconds), claimed_seconds,
                                     opts)
        params["rates"] = top["rates"]
    else:  # (a conversion: its price is the video conversion's, below)
        params["rates"] = {}
        est = 0
    est += convert_cost(params, claimed_seconds)  # (another file type only: known once the file is read, then held if it is made again)
    now = db.now()
    # (the record's tier, for the metrics and the ledger: a timeline job's stored tier is only a placeholder, so its Voice rebuild is named)
    tier = "rebuild" if tl is not None and tl.layers["rebuild"] else "/".join(t for t in (params["denoise_tier"], params["transcribe_tier"]) if t)
    try:
        d = job_dir(job_id)
        d.mkdir(parents=True, mode=0o700)
        # The limits, the money and the row are one step: parallel requests can't all pass a check that
        # none of them has recorded yet (the money rolls back with everything else).
        with db.tx() as c:
            if idempotency is not None:  # a parallel request with the same key waited for this transaction and now sees its row
                again = replay_of(user_id, *idempotency, c=c)
                if again is not None:
                    raise Replay(again["id"], again["estCostMicros"])
            check_active_limit(c, user_id)
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
                keycaps.check(c, api_key_id, est)  # the API key's monthly spending cap (KeyCapReached)
                try:
                    hold_id = ledger.create_hold(user_id, est, "job", job_id, c=c)
                except ledger.InsufficientFunds as e:
                    raise JobError(
                        f"This job costs about ${est / 1e6:.2f}; your available balance is ${e.available / 1e6:.2f}.", 402
                    ) from e
            c.execute(
                "INSERT INTO jobs(id, user_id, kind, tier, status, params_json, input_name, input_bytes,"
                " claimed_seconds, hold_id, est_cost_micros, created_at, expires_at, api_key_id)"
                " VALUES (?,?,?,?, 'uploading', ?,?,?,?,?,?,?,?,?)",
                (job_id, user_id, kind, tier, json.dumps(params), filename[:200], size_bytes,
                 claimed_seconds, hold_id, est, now, now + UPLOAD_FIRST_BYTE_SECONDS, api_key_id),
            )
            if callback_url:
                c.execute(
                    "INSERT INTO job_callbacks(job_id, user_id, url, created_at) VALUES (?,?,?,?)", (job_id, user_id, callback_url, now)
                )
            if idempotency is not None:
                c.execute(
                    "INSERT INTO idempotency_keys(user_id, key, body_hash, job_id, created_at) VALUES (?,?,?,?,?)"
                    " ON CONFLICT(user_id, key) DO UPDATE SET body_hash = excluded.body_hash, job_id = excluded.job_id,"
                    " created_at = excluded.created_at",
                    (user_id, idempotency[0], idempotency[1], job_id, now),
                )
    except Exception:
        shutil.rmtree(job_dir(job_id), ignore_errors=True)
        raise
    wanted = edit_subtitles.all_uploads(tl) if tl is not None else []
    return {"id": job_id, "estCostMicros": est, **({"maximum": True} if plan else {}), **({"needs": wanted} if wanted else {})}


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


def mark_uploaded(job_id: str, source: bool = True) -> None:
    """The job's file (`source`) or something else it waits for (the browser's work for a timeline's browser parts and free subtitles,
    editing.py) has arrived: when the file is in and nothing else is awaited it is queued, else it stays `uploading` and waits (for
    another UPLOAD_WINDOW_SECONDS after the last arrival, like an upload that is still coming). The one place a job leaves `uploading`."""
    now = db.now()
    if source:
        db.execute("UPDATE jobs SET uploaded_at = COALESCE(uploaded_at, ?) WHERE id = ? AND status = 'uploading'", (now, job_id))
    row = db.one("SELECT * FROM jobs WHERE id = ? AND status = 'uploading' AND uploaded_at IS NOT NULL", (job_id,))
    if row is None:
        return
    from . import edit_subtitles

    if edit_subtitles.pending_needs(dict(row), json.loads(row["params_json"])):
        db.execute("UPDATE jobs SET expires_at = ? WHERE id = ? AND status = 'uploading'", (now + UPLOAD_WINDOW_SECONDS, job_id))
        return
    db.execute(
        "UPDATE jobs SET status = 'queued', next_stage = 'probe', queued_at = ?, expires_at = NULL"
        " WHERE id = ? AND status = 'uploading' AND uploaded_at IS NOT NULL",
        (now, job_id),
    )
    invalidate_queue_cache()


def finish_failed(
    job: dict, message: str, status: str = "failed", only_from: tuple[str, ...] | None = None, not_stage: str | None = None
) -> bool:
    """Release money and delete files. Safe to call twice. The status change and the hold move in one
    transaction, so a crash can't leave money reserved for a finished job.
    `only_from` limits it to jobs still in one of those states (cancel: not once a worker has started) and
    `not_stage` to jobs whose next stage is another one (cancel: not once the paid-for transcription is done).
    True when the job was moved to `status` by this call."""
    reverted = False
    with db.tx() as c:
        cur = c.execute("SELECT status, next_stage, params_json FROM jobs WHERE id = ?", (job["id"],)).fetchone()
        if cur is None or cur["status"] in ("done", "failed", "canceled", "expired"):
            return False
        if only_from is not None and cur["status"] not in only_from:
            return False
        if not_stage is not None and cur["next_stage"] == not_stage:
            return False
        params = json.loads(cur["params_json"])
        run = params.get("run")
        if run:
            # An edit run that ends without its result (canceled, failed, too late): the job is as it was before it, its files stay
            # (they are the person's work), the money is released and the run's record says why (editing.py).
            reverted = True
            params.pop("run", None)
            params.pop("editor_reported", None)
            edit = dict(params.get("edit") or {})
            edit["lastRun"] = {"op": run.get("op"), "ok": False, "error": message[:300], "at": db.now()}
            params["edit"] = edit
            c.execute(
                "UPDATE jobs SET status = 'done', next_stage = 'probe', error = NULL, params_json = ?, progress = 100, stage = 'Done',"
                " worker_id = NULL, expires_at = ? WHERE id = ?",
                (json.dumps(params), expiry(db.now(), _uploaded_at(c, job["id"])), job["id"]),
            )
        else:
            c.execute(
                "UPDATE jobs SET status = ?, error = ?, finished_at = ?, expires_at = ? WHERE id = ?",
                (status, message[:500], db.now(), db.now() + get_settings().job_retention_seconds, job["id"]),
            )
        if job.get("hold_id"):
            ledger.release_hold(job["hold_id"], c=c)
    invalidate_queue_cache()
    if reverted:
        from . import edit_store

        edit_store.clear_scratch(job_dir(job["id"]))
    else:
        shutil.rmtree(job_dir(job["id"]), ignore_errors=True)
    metrics.job_ended(job["id"])
    return True


def _uploaded_at(c, job_id: str) -> int | None:
    row = c.execute("SELECT COALESCE(uploaded_at, created_at) AS t FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return row["t"] if row else None


def requeue(job: dict, next_stage: str | None = None, keep_place: bool = False) -> None:
    """Put a running job back in the queue untouched: the attempt it used is given back, its upload stays and
    its scratch files go (the stage starts again from the upload; after cloud transcription it keeps the
    segments and chunks it needs). `next_stage`: where the next worker goes on from (default: as it was); `keep_place`: it keeps
    its place in the queue instead of joining at the end."""
    sets, args = "status = 'queued', worker_id = NULL, attempts = MAX(0, attempts - 1), progress = 0, stage = NULL", []
    if not keep_place:
        sets, args = sets + ", queued_at = ?", [db.now()]
    if next_stage:
        sets, args = sets + ", next_stage = ?", [*args, next_stage]
    with db.tx() as c:
        n = c.execute(f"UPDATE jobs SET {sets} WHERE id = ? AND status = 'running'", (*args, job["id"])).rowcount
    if not n:
        return
    invalidate_queue_cache()
    d = job_dir(job["id"])
    if job.get("next_stage") == "finalize":
        return  # finalize only reads what the earlier stages left
    params = json.loads(job["params_json"])
    if params.get("run"):  # an edit run: the job's files are the person's work; only the run's scratch goes
        from . import edit_store, edit_subtitles

        edit_store.clear_scratch(d)
        return
    # (the blocks a stopped run had made are kept: they are exactly what the run starts again from)
    from . import edit_store, edit_subtitles

    keep = {f"input{params['input_ext']}", "blocks", edit_subtitles.BROWSER, edit_subtitles.SUBS}
    for p in d.iterdir():
        if p.name in keep:
            continue
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


def hand_over(job: dict) -> None:
    """A worker that has decided an automatic job needs the voice rebuilt, but is not the card's, gives it up: the job goes back in
    the queue (where it was, to be taken by the card's worker alone, as it is a rebuild now) and goes on at its main stage, as its
    file is probed and its settings decided."""
    requeue(job, next_stage="main", keep_place=True)


def _seconds_left(r: dict, now: int) -> float:
    left = job_seconds(r)
    if r["status"] != "queued" and r.get("started_at"):
        left = max(10.0, left - (now - r["started_at"]))
    return left


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
    One pass for the whole queue, in two lanes: a voice rebuild waits for the GPU worker's running job and the
    rebuilds in front of it; any other job for the work running and queued in front of it on the CPU workers."""
    key = str(get_settings().db_path)
    with _queue_cache_lock:
        if _queue_cache["key"] == key and time.monotonic() - _queue_cache["at"] < QUEUE_CACHE_SECONDS:
            return _queue_cache["ids"], _queue_cache["waits"]
        # Computed under the lock: concurrent requests wait for one computation instead of each running their own.
        ids = queued_ids()
        now = db.now()
        from .broker import gpu_only, is_gpu_worker

        workers = max(1, get_settings().workers)
        rows = {r["id"]: dict(r) for r in db.all_rows("SELECT * FROM jobs WHERE status IN ('running', 'cloud_asr', 'queued')")}
        running = [r for r in rows.values() if r["status"] != "queued"]
        gpu_work = sum(_seconds_left(r, now) for r in running if is_gpu_worker(r.get("worker_id")))
        cpu_work = sum(_seconds_left(r, now) for r in running if not is_gpu_worker(r.get("worker_id")))
        waits: dict[str, int] = {}
        for jid in ids:
            r = rows.get(jid)
            if r is not None and gpu_only(r):
                waits[jid] = int(gpu_work)
                gpu_work += _seconds_left(r, now)
            else:
                waits[jid] = int(cpu_work / workers)
                if r is not None:
                    cpu_work += _seconds_left(r, now)
        _queue_cache.update(key=key, at=time.monotonic(), ids=ids, waits=waits)
        return ids, waits


def callbacks_for(job_ids: list[str]) -> dict[str, dict]:
    """The state of each job's callback (the jobs that have one), as shown with the job."""
    if not job_ids:
        return {}
    cols = "SELECT job_id, url, state, attempts, next_attempt_at, last_status, last_error, delivered_at FROM job_callbacks"
    if len(job_ids) == 1:
        rows = [r for r in (db.one(f"{cols} WHERE job_id = ?", (job_ids[0],)),) if r is not None]
    else:
        rows = db.all_rows(f"{cols} WHERE job_id IN ({','.join('?' * len(job_ids))})", tuple(job_ids))
    return {
        r["job_id"]: {
            "url": r["url"].split("?", 1)[0],  # the query string often carries a token: not shown back
            "state": r["state"],
            "attempts": r["attempts"],
            "lastStatus": r["last_status"],
            "lastError": r["last_error"],
            "deliveredAt": r["delivered_at"],
            "nextAttemptAt": r["next_attempt_at"] if r["state"] == "pending" and r["attempts"] else None,
        }
        for r in rows
    }


def label_of(kind: str, params: dict) -> str:
    """The customer-facing name of what a job does: the price list's words; for a re-run, the steps it did again. A video conversion
    that makes the picture again is named too."""
    base = _label_of(kind, params)
    if kind != "convert" and not params.get("run") and params.get("convert_rate") and params.get("picture_encode"):
        return f"{base} + {pricing.rate('convert', 'standard').label}"
    return base


def _label_of(kind: str, params: dict) -> str:
    if params.get("run"):
        from . import editing

        return editing.run_label(params)
    edit = params.get("edit") or {}
    # a job made with a timeline (it alone has `edit_rates`; its stored tier is only a placeholder) is named for its layers, also once its first run is over
    if edit.get("first_run") or (edit.get("timeline") and params.get("edit_rates")):
        from . import editing

        return " + ".join([editing.timeline_label(params), *([job_label("transcribe", None, params["transcribe_tier"])] if params.get("transcribe_tier") else [])])
    if automatic.pending(params):  # not decided yet: what it will do is not known
        return " + ".join(["Automatic settings", *([job_label("transcribe", None, params["transcribe_tier"])] if params.get("transcribe_tier") else [])])
    rates = params.get("rates") or {}
    return job_label(kind, params.get("denoise_tier"), params.get("transcribe_tier"), "leveling" in rates)


def public_job(job: dict, queue: tuple[list[str], dict[str, int]] | None = None, callbacks: dict[str, dict] | None = None) -> dict:
    """`queue` is queue_snapshot() and `callbacks` callbacks_for() (pass them once when listing many jobs)."""
    outputs = json.loads(job["outputs_json"]) if job.get("outputs_json") else []
    params = json.loads(job["params_json"])
    position = eta = None
    if job["status"] == "queued":
        ids, waits = queue_snapshot() if queue is None else queue
        if job["id"] in waits:
            position = ids.index(job["id"]) + 1
            eta = waits[job["id"]]
    callback = (callbacks if callbacks is not None else callbacks_for([job["id"]])).get(job["id"])
    out = {
        "queuePosition": position,
        "queueWaitSeconds": eta,
        "id": job["id"],
        "kind": job["kind"],
        "tier": job["tier"],
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"],
        "error": job["error"],
        "inputName": params.get("display_name") or job["input_name"],
        "label": label_of(job["kind"], params),
        # (a job asked for with options: the tiers it was asked for; the timeline it became is what is made of them)
        "denoiseTier": (params.get("whole_file") or {}).get("denoise_tier", params.get("denoise_tier")),
        "transcribeTier": (params.get("whole_file") or {}).get("transcribe_tier", params.get("transcribe_tier")),
        "durationSeconds": job["duration_seconds"] or job["claimed_seconds"],
        "estCostMicros": job["est_cost_micros"],
        "costMicros": job["cost_micros"],
        "options": params["options"],
        "notice": params.get("notice"),
        "outputs": [{"name": o["file"], "kind": o["kind"], "bytes": o.get("bytes"), **({"lang": o["lang"]} if o.get("lang") else {})} for o in outputs],
        "createdAt": job["created_at"],
        "finishedAt": job["finished_at"],
        # when the files are deleted: 2 hours after the last edit, never later than `hardDeadlineAt` (6 hours after the upload)
        "expiresAt": job["expires_at"] if job["status"] == "done" else None,
        "editedAt": job.get("edited_at") or job["finished_at"],
        "hardDeadlineAt": hard_deadline(job),
    }
    from . import edit_subtitles, editing

    if editing.editable_kind(job):
        e = params.get("edit") or {}
        out["edit"] = {"available": job["status"] == "done", "hasTimeline": bool(e.get("timeline")), "run": (params.get("run") or {}).get("op"),
                       "lastRun": e.get("lastRun"), "suggestions": bool(e.get("suggestions"))}
        needs = edit_subtitles.pending_needs(job, params)  # what the browser still has to send for the timeline's browser parts and free subtitles
        if needs is not None:
            out["needs"] = needs
    if params.get("automatic"):
        out["chosen"] = params.get("chosen")  # null until the file has been listened to
        if automatic.pending(params):
            out["maximum"] = True  # estCostMicros is the most it can cost until then
    if job["status"] in ("queued", "running", "cloud_asr"):
        out["estimateSeconds"] = int(job_seconds(job))  # the whole job's rough length, for the wait before progress shows
    if callback is not None:
        out["callback"] = callback
    return out


def sweep() -> None:
    """Expire finished outputs and abandoned uploads; recover stuck jobs."""
    now = db.now()
    db.execute("DELETE FROM idempotency_keys WHERE created_at <= ?", (now - IDEMPOTENCY_SECONDS,))
    for job in db.all_rows("SELECT * FROM jobs WHERE status = 'uploading' AND expires_at <= ?", (now,)):
        finish_failed(dict(job), "Upload was not completed.", status="canceled")
    # 'canceling' was a transient state of older versions of the cancel request.
    for job in db.all_rows("SELECT * FROM jobs WHERE status = 'canceling'"):
        finish_failed(dict(job), "Canceled.", status="canceled")
    # The retention is a ceiling: files that were given a later deadline (a longer retention, an older rule) end with these two:
    # 2 hours after the last edit, and 6 hours after the upload.
    settings = get_settings()
    db.execute(
        "UPDATE jobs SET expires_at = MIN(COALESCE(edited_at, finished_at) + ?, COALESCE(uploaded_at, created_at) + ?) WHERE status = 'done'"
        " AND COALESCE(edited_at, finished_at) IS NOT NULL AND expires_at > MIN(COALESCE(edited_at, finished_at) + ?, COALESCE(uploaded_at, created_at) + ?)",
        (settings.job_retention_seconds, settings.job_max_retention_seconds) * 2,
    )
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
