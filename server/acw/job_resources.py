"""What a job needs before a worker may start it and how long it takes: memory, scratch disk, the wall-clock seconds per second of audio of
each step, the ceilings of the sandboxed stages, and the reserves our work never takes the host below."""
from __future__ import annotations

import json

from . import automatic, job_convert


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


# Never let our work take the host below these.
# Free system memory (MemAvailable) left after a job starts: the Minecraft memory controller closes new world starts at
# 4 GiB available and reopens at 6 GiB, so a job that took the host down to 2 GB shut players out until it ended.
MEM_RESERVE = 6 * GB


# Room kept under the acw slice's own MemoryMax (a separate limit, 12.5 GB: a 6 GB reserve there would stop a 6 GB
# local transcription from ever running next to the web and live processes).
SLICE_RESERVE = 2 * GB
DISK_RESERVE = 10 * GB


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
    t = SPEED["encode"] * audio if job_convert.encodes(params) else 0.0
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
