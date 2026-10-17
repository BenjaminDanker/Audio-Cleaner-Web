"""Job worker: `python -m acw.worker.main`.

Claims one queued job at a time and drives the sandboxed runner through its
stages. This process has database access but (under systemd) no network; the
runner it spawns has neither. Run several instances for parallelism.
"""
from __future__ import annotations

import json
import re
import logging
import math
import os
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import acw_common
from acw_common import restore

from .. import broker, cpusplit, db, hostload, jobs, metrics
from ..billing import ledger, pricing
from ..config import get_settings

log = logging.getLogger("acw.worker")
_stop = threading.Event()


# The GPU worker (acw-worker-gpu, its own venv with CUDA ONNX Runtime) takes the jobs whose heavy stages run well on the
# card (docs/research/gpu-2026-10.md): local transcription (Parakeet), the heavy noise remover (DPDFNet-8), the voice
# rebuild (Sidon) and band extension (AP-BWE). It runs one job at a time (Parakeet alone needs 3.7 of the 4 GB; the
# stages of one job run one after the other and give the card back). The CPU workers leave these jobs to it while it
# is idle, and the voice rebuild also while it is busy (on the CPU that is hours of four cores). See docs/operations.md.
GPU_MODE = os.environ.get("ACW_WORKER_GPU") == "1"
GPU_IDLE_FRESH_S = 6
GPU_ALIVE_FRESH_S = 90


def _gpu_idle_marker() -> Path:
    return get_settings().data_dir / "gpu-worker-idle"


def _gpu_alive_marker() -> Path:
    return get_settings().data_dir / "gpu-worker-alive"


def _fresh(marker: Path, seconds: float) -> bool:
    try:
        return time.time() - marker.stat().st_mtime < seconds
    except OSError:
        return False


def gpu_worker_idle() -> bool:
    return _fresh(_gpu_idle_marker(), GPU_IDLE_FRESH_S)


def gpu_worker_up() -> bool:
    """The GPU worker is running, idle or busy (it touches the marker at least every 20 s, also while a job runs)."""
    return _fresh(_gpu_alive_marker(), GPU_ALIVE_FRESH_S)


def is_local_transcription(job: dict) -> bool:
    """A job whose transcription the GPU worker does. Not one with echo removal: the GPU venv has no echo canceller,
    so the CPU workers keep those."""
    params = json.loads(job["params_json"])
    return (
        params.get("transcribe_tier") == "local"
        and not params.get("options", {}).get("echo_reference_track")
        and job.get("next_stage") != "finalize"
    )


def is_gpu_denoise(job: dict, models_dir: Path | None = None) -> bool:
    """A job whose cleaning has a stage on the card: the voice rebuild (Sidon), or the heavy noise remover, chosen
    by the user or measured on the recording ("auto"; a recording that turns out light is cleaned by
    DeepFilterNet3 on the worker's CPU threads, as everywhere). Heavy noise needs the batched DPDFNet file, which is
    optional (deploy/models.lock): without it those jobs stay with the CPU workers. Light cleaning stays on the CPU,
    band extension follows the job to the card where it goes."""
    params = json.loads(job["params_json"])
    opts = params.get("options", {})
    if job["kind"] not in ("denoise", "denoise_transcribe") or job.get("next_stage") == "finalize" or opts.get("echo_reference_track"):
        return False
    if params.get("denoise_tier") == "rebuild":
        return True
    if opts.get("noise", "auto") not in ("auto", "heavy"):
        return False
    return heavy_device(models_dir) == "cuda"


def is_gpu_job(job: dict, models_dir: Path | None = None) -> bool:
    """A job the GPU worker takes (and, while it is idle, the CPU workers leave to it)."""
    return is_local_transcription(job) or is_gpu_denoise(job, models_dir)


def gpu_only(job: dict) -> bool:
    """A GPU job the CPU workers do not take even while the GPU worker is busy (it is up): the voice rebuild takes
    five times longer on the CPU, on four cores, for hours. When no GPU worker is up they take it as before."""
    return jobs.is_rebuild(job) and is_gpu_denoise(job)


def _gpu_spec() -> dict:
    """Sandbox additions the CUDA driver needs (found with strace under
    Landlock): the GPU device files, driver info under /proc and /sys, and
    this process's own /proc/self (it names its threads there, which it treats
    as fatal if refused). Nothing about other processes."""
    devices = sorted(str(p) for p in Path("/dev").glob("nvidia*") if not p.is_dir())
    read = [p for p in ("/proc/driver/nvidia", "/proc/devices", "/proc/sys/vm/mmap_min_addr", "/sys/bus/pci",
                        "/sys/devices", "/sys/module/nvidia", "/sys/module/nvidia_uvm", "/dev/char") if Path(p).exists()]
    return {
        "asr_device": "cuda",
        "clean_device": "cuda",
        "gpu_devices": [*devices, "/proc/self"],
        "gpu_read": [*read, "/proc/self"],
        "address_space_bytes": 0,
    }


def heavy_device(models_dir: Path | None = None) -> str:
    """Where the heavy noise remover runs in the GPU worker: on the card when the batched DPDFNet file is installed
    (the card runs only that one), else on the worker's CPU threads like anywhere else. Looked at per job, so
    installing the file takes effect without a restart."""
    from acw_common import dpdfnet

    return "cuda" if ((models_dir or get_settings().models_dir) / "dpdfnet" / dpdfnet.BATCH_MODEL_FILE).is_file() else "cpu"


def _cuda_library_path() -> str:
    """CUDA/cuDNN come from pip wheels inside the GPU venv (site-packages/nvidia/*/lib)."""
    import site

    dirs = []
    for sp in site.getsitepackages():
        dirs += sorted(str(p) for p in Path(sp, "nvidia").glob("*/lib") if p.is_dir())
    return os.pathsep.join(dirs)


def _settings_extra() -> dict:
    s = get_settings()
    return {
        "threads": int(os.environ.get("ACW_JOB_THREADS", "4")),
        "whisper_model": os.environ.get("ACW_WHISPER_MODEL", "large-v3-turbo"),
        "lib_path": str(s.bin_dir / "libdf.so"),
        **(_gpu_spec() if GPU_MODE else {}),
        "dfn_model": str(s.models_dir / "DeepFilterNet3_onnx.tar.gz"),
        "bin_dir": str(s.bin_dir),
        "models_dir": str(s.models_dir),
    }


# The probe reads the whole file once (a fast decode that measures its real length).
PROBE_CPU_SECONDS = 3600
PROBE_TIMEOUT = 900

MAX_JOB_THREADS = int(os.environ.get("ACW_JOB_THREADS_MAX", "6"))


def admission(
    job: dict, snap: hostload.Snapshot, running: list[dict], cpus: set[int] | None = None, gpu: bool = False
) -> tuple[bool, str, int]:
    """Whether `job` may start now, why not, and how many threads it gets: the resource broker decides."""
    return broker.admit_job(job, snap, running, cpus, gpu=gpu, max_threads=MAX_JOB_THREADS)


_last_wait_log = 0.0


def claim(worker_id: str, load: hostload.HostLoad | None = None) -> dict | None:
    """Take the next queued job (jobs.QUEUE_ORDER) if the host has room for it.
    The head of the queue waits rather than letting smaller jobs overtake it. At most one voice-rebuild job
    runs at a time (it uses several cores for hours): further ones wait without holding up the rest.

    The host is measured outside the database's write lock (a snapshot can sleep); the lock is only taken to
    check once more with fresh facts and to take the job."""
    global _last_wait_log
    now = db.now()
    rows = [dict(r) for r in db.all_rows(
        f"SELECT j.* FROM jobs j WHERE j.status = 'queued' ORDER BY {jobs.QUEUE_ORDER} LIMIT 50"
    )]
    if GPU_MODE:
        rows = [r for r in rows if is_gpu_job(r)]
    elif gpu_worker_idle():
        rows = [r for r in rows if not is_gpu_job(r)]  # the GPU worker takes these
    elif gpu_worker_up():
        rows = [r for r in rows if not gpu_only(r)]  # busy: it takes these when it is free again
    if not rows:
        return None
    snap = cpus = None
    if load is not None:
        snap = load.snapshot()
        cpus = broker.job_cpus(get_settings().data_dir)

    def running_jobs(c) -> list[dict]:
        return [dict(r) for r in c.execute("SELECT * FROM jobs WHERE status = 'running'").fetchall()]

    def pick(running: list[dict]) -> tuple[dict | None, str, int | None]:
        candidates = rows
        if any(jobs.is_rebuild(r) for r in running):
            candidates = [r for r in rows if not jobs.is_rebuild(r)]
        if not candidates:
            return None, "another voice rebuild is running", None
        row = candidates[0]
        if snap is None:
            return row, "", None
        ok, why, threads = admission(row, snap, running, cpus, gpu=GPU_MODE)
        return (row, "", threads) if ok else (None, why, None)

    row, why, _ = pick(running_jobs(db.conn()))
    if row is None:
        if why and time.monotonic() - _last_wait_log > 60:
            _last_wait_log = time.monotonic()
            log.info("jobs wait (%s)", why)
        return None
    with db.tx() as c:
        row, why, threads = pick(running_jobs(c))  # same facts, current list of running jobs
        if row is None:
            return None
        n = c.execute(
            "UPDATE jobs SET status = 'running', worker_id = ?, heartbeat_at = ?, attempts = attempts + 1,"
            " started_at = COALESCE(started_at, ?),"
            " wait_seconds = COALESCE(wait_seconds, MAX(0, ? - COALESCE(queued_at, created_at)))"
            " WHERE id = ? AND status = 'queued'",
            (worker_id, now, now, now, row["id"]),
        ).rowcount
        if not n:
            return None  # another worker took it
        job = dict(c.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone())
        job["threads"] = threads
        return job


def _max_seconds(job: dict, params: dict) -> int:
    most = pricing.max_job_seconds(params["denoise_tier"])
    return min(most, pricing.FREE_TRANSCRIBE_MAX_FILE_SECONDS) if job["free"] else most


def _write_spec(job: dict, extra: dict, billed: float | None = None) -> Path:
    """Write the runner's instructions. `billed` is the measured length of the audio (known after the
    probe): the main stage never decodes more than that, and its CPU limit is sized from it and the tier."""
    d = jobs.job_dir(job["id"])
    params = json.loads(job["params_json"])
    max_seconds = _max_seconds(job, params)
    cpu_seconds = PROBE_CPU_SECONDS if billed is None else jobs.stage_limits(params["denoise_tier"], billed, max_seconds)[0]
    spec = {
        **extra,
        "job_id": job["id"],
        "kind": job["kind"],
        "denoise_tier": params["denoise_tier"],
        "transcribe_tier": params["transcribe_tier"],
        "input": f"input{params['input_ext']}",
        "input_ext": params["input_ext"],
        "output_stem": jobs.safe_stem(jobs.display_name(job)),
        "options": params["options"],
        "max_seconds": max_seconds,
        "cpu_seconds": cpu_seconds,  # generous; the wall-clock timeout is the real limit
        "address_space_bytes": 32 * 1024**3,  # memory-mapped scratch files count here, not in RAM
        "max_file_bytes": 16 * 1024**3,
        **({"billed_seconds": billed} if billed is not None else {}),
    }
    (d / "spec.json").write_text(json.dumps(spec))
    return d


class Interrupted(Exception):
    """The worker is stopping (a restart or a deploy): the job goes back in the queue instead of failing."""


class StageFailed(Exception):
    def __init__(self, user_message: str, internal: str | None = None):
        super().__init__(internal or user_message)
        self.user_message = user_message


OOM_MESSAGE = "This file needs more memory than we can give one job, so it could not be processed. Splitting it into shorter parts may help; you were not charged."
INTERRUPTED_MESSAGE = "Processing was interrupted several times, so we stopped trying; you were not charged. If the file is very long, splitting it into shorter parts may help."
# A job put back in the queue by a stopping worker keeps its attempt, so a deploy never costs it a retry. But a worker that
# keeps stopping on the same job (for example systemd stopping the unit after the kernel killed the runner for memory)
# must not run it forever: the job is failed at its MAX_INTERRUPTIONS-th interruption.
MAX_INTERRUPTIONS = int(os.environ.get("ACW_MAX_INTERRUPTIONS", "3"))


def oom_kills() -> int | None:
    """How many processes the kernel's OOM killer has killed in this worker's cgroup (runners included) so far; None if unreadable."""
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                for ev in (Path("/sys/fs/cgroup") / line[3:].strip().lstrip("/") / "memory.events").read_text().splitlines():
                    key, _, value = ev.partition(" ")
                    if key == "oom_kill":
                        return int(value)
    except (OSError, ValueError):
        pass
    return None


def run_stage(job: dict, d: Path, stage: str, timeout: float) -> dict:
    env = {
        "PATH": "/usr/bin:/bin",
        # acw and acw_common share one directory when deployed, not in a checkout.
        "PYTHONPATH": os.pathsep.join(
            dict.fromkeys(str(Path(m.__file__).resolve().parents[i]) for m, i in ((sys.modules["acw"], 1), (acw_common, 1)))
        ),
        "LANG": "C.UTF-8",
        "HOME": str(d),
        "HF_HUB_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if GPU_MODE:
        env["LD_LIBRARY_PATH"] = _cuda_library_path()
    result_file = d / f"result-{stage}.json"
    result_file.unlink(missing_ok=True)
    oom_before = oom_kills()
    proc = subprocess.Popen(
        [sys.executable, "-m", "acw.worker.runner", str(d), stage],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        env=env,
        cwd=str(d),
        start_new_session=True,
    )
    stderr_tail: list[bytes] = []

    def drain_stderr() -> None:
        for line in proc.stderr:  # type: ignore[union-attr]
            stderr_tail.append(line)
            del stderr_tail[:-40]

    threading.Thread(target=drain_stderr, daemon=True).start()
    started = time.monotonic()
    deadline = started + timeout
    last_update = 0.0
    prog = {"first": None, "now": 0, "at": started}  # the stage's reported progress: where it began, where it is, when it last moved
    killed_by_us = False

    data_dir = get_settings().data_dir
    pinned: set[int] | None = None

    def more_time() -> bool:
        """At the deadline: a main stage that is still making progress gets the time its progress says it needs
        (jobs.stretched_limit), so a job that ran slower than planned (live audio took its cores) is not thrown away."""
        nonlocal deadline
        if stage != "main" or _stop.is_set() or prog["first"] is None:
            return False
        now = time.monotonic()
        limit = jobs.stretched_limit(timeout, now - started, prog["first"], prog["now"], now - prog["at"])
        if limit is None or started + limit <= deadline:
            return False
        log.info("job %s stage %s still at %s%% after %.0f s: limit %.0f s -> %.0f s", job["id"], stage, prog["now"], now - started, deadline - started, limit)
        deadline = started + limit
        return True

    def watchdog() -> None:
        nonlocal pinned, killed_by_us
        beat = 0.0
        while proc.poll() is None:
            # Off the live-audio cores while live sessions run (acw.cpusplit).
            want = cpusplit.job_cpus_now(data_dir)
            if want != pinned:
                cpusplit.pin_tree(proc.pid, want)
                pinned = want
            if _stop.is_set() or (time.monotonic() > deadline and not more_time()):
                killed_by_us = True
                os.killpg(proc.pid, signal.SIGKILL)
                return
            if time.monotonic() - beat > 20:
                beat = time.monotonic()
                try:
                    db.execute("UPDATE jobs SET heartbeat_at = ? WHERE id = ?", (db.now(), job["id"]))
                    if GPU_MODE:
                        _gpu_alive_marker().touch()
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(1)

    threading.Thread(target=watchdog, daemon=True).start()
    for raw in proc.stdout:  # type: ignore[union-attr]
        try:
            msg = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(msg, dict):
            continue
        if msg.get("noise") in ("light", "heavy") and job.get("noise_profile") != msg["noise"]:
            # Which noise remover the recording got: the time estimate follows it.
            job["noise_profile"] = msg["noise"]
            jobs.update_params(job["id"], noise_profile=msg["noise"])
        now = time.monotonic()
        try:
            progress = max(0, min(100, int(msg.get("progress", 0))))
        except (TypeError, ValueError, OverflowError):
            progress = 0
        if prog["first"] is None:
            prog["first"] = prog["now"] = progress
        if progress > prog["now"]:
            prog["now"], prog["at"] = progress, now
        if now - last_update >= 1.0 or msg.get("progress") == 100:
            last_update = now
            db.execute(
                "UPDATE jobs SET progress = ?, stage = ?, heartbeat_at = ? WHERE id = ?",
                (progress, str(msg.get("stage", ""))[:40], db.now(), job["id"]),
            )
    proc.wait()
    if not result_file.exists() and not killed_by_us and proc.returncode == -signal.SIGKILL:
        # Killed from outside while we were not stopping it: the kernel's OOM killer when its counter moved. With
        # OOMPolicy=stop systemd then stops this whole worker (SIGTERM), which would read as a deploy and put the same
        # job back again and again.
        after = oom_kills()
        if oom_before is not None and after is not None and after > oom_before:
            log.error("job %s stage %s was killed for lack of memory", job["id"], stage)
            raise StageFailed(OOM_MESSAGE, "oom kill")
    if _stop.is_set() and not result_file.exists():
        raise Interrupted()
    if not result_file.exists():
        tail = b"".join(stderr_tail).decode(errors="replace")[-2000:]
        log.error("job %s stage %s died rc=%s: %s", job["id"], stage, proc.returncode, tail)
        if time.monotonic() > deadline:
            raise StageFailed("Processing took too long and was stopped; you were not charged.")
        raise StageFailed("Processing failed; you were not charged.", f"rc={proc.returncode}")
    res = _read_result(result_file)
    if not res.get("ok"):
        if res.get("user_error"):
            raise StageFailed(res["user_error"])
        log.error("job %s stage %s error: %s", job["id"], stage, res.get("error"))
        raise StageFailed("Processing failed; you were not charged.", res.get("error"))
    return res


RESULT_MAX_BYTES = 1_000_000
OUTPUT_KINDS = {"audio", "captions", "transcript", "segments", "video"}


def _read_result(path: Path) -> dict:
    """The runner's result file. The runner is sandboxed but is the part that handles untrusted input, so what it
    writes is checked: a regular file (never a link) of sane size, holding a JSON object of the expected shape."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as f:
            st = os.fstat(f.fileno())
            if not stat.S_ISREG(st.st_mode) or st.st_size > RESULT_MAX_BYTES:
                raise ValueError("bad result file")
            res = json.loads(f.read(RESULT_MAX_BYTES + 1))
    except (OSError, ValueError) as e:
        raise StageFailed("Processing failed; you were not charged.", f"unreadable result: {e}")
    if not isinstance(res, dict) or not isinstance(res.get("ok"), bool):
        raise StageFailed("Processing failed; you were not charged.", "malformed result")
    if res["ok"]:
        outs = res.get("outputs", [])
        if not isinstance(outs, list) or len(outs) > 50 or not all(
            isinstance(o, dict) and isinstance(o.get("file"), str) and isinstance(o.get("kind"), str) for o in outs
        ):
            raise StageFailed("Processing failed; you were not charged.", "malformed outputs")
        for key in ("notice", "noise", "language"):
            if res.get(key) is not None and not isinstance(res[key], str):
                raise StageFailed("Processing failed; you were not charged.", f"malformed {key}")
    elif res.get("user_error") is not None and not isinstance(res["user_error"], str):
        res["user_error"] = None
    return res


def _job_cost(job: dict, params: dict, seconds: float) -> int:
    """What the job costs for `seconds` of audio, at the prices it was created with."""
    # Translations: what was asked for until the job is done, then only the languages that were made.
    asked = params.get("options", {}).get("translate") or []
    made = params.get("translated", asked)
    return pricing.job_cost_micros(job["kind"], params["denoise_tier"], params["transcribe_tier"], seconds, params.get("rates"), len(made))


def _reprice(job: dict, duration: float) -> None:
    """Make the hold (or free allowance) match the real duration before the
    expensive work starts."""
    params = json.loads(job["params_json"])
    db.execute("UPDATE jobs SET duration_seconds = ? WHERE id = ?", (duration, job["id"]))
    if job["free"]:
        # claimed_seconds is what is reserved today; keep it in step with the allowance, so a retry of the
        # job (new attempt, job loaded again) doesn't take the difference a second time.
        with db.tx() as c:
            claimed = c.execute("SELECT claimed_seconds FROM jobs WHERE id = ?", (job["id"],)).fetchone()["claimed_seconds"]
            delta = duration - claimed
            if delta > 0:
                day = params["free_day"]
                row = c.execute("SELECT seconds FROM free_usage WHERE user_id = ? AND day = ?", (job["user_id"], day)).fetchone()
                if (row["seconds"] if row else 0) + delta > pricing.FREE_TRANSCRIBE_SECONDS_PER_DAY + 30:
                    raise StageFailed("This file is longer than your remaining free minutes today.")
            c.execute("UPDATE jobs SET claimed_seconds = ? WHERE id = ?", (duration, job["id"]))
            c.execute(
                "UPDATE free_usage SET seconds = MAX(0, seconds + ?) WHERE user_id = ? AND day = ?",
                (delta, job["user_id"], params["free_day"]),
            )
        job["claimed_seconds"] = duration
        if not job["hold_id"]:
            return  # free subtitles only; with a paid cleanup the hold is corrected below
    cost = _job_cost(job, params, duration)
    if cost > job["est_cost_micros"]:
        try:
            ledger.extend_hold(job["hold_id"], cost - job["est_cost_micros"])
        except (ledger.InsufficientFunds, ledger.LedgerError):
            raise StageFailed(
                f"This file is {duration / 60:.1f} min and costs ${cost / 1e6:.2f}, more than your available balance."
            )
        db.execute("UPDATE jobs SET est_cost_micros = ? WHERE id = ?", (cost, job["id"]))
        job["est_cost_micros"] = cost


def _complete(job: dict, d: Path, res: dict) -> None:
    outputs = []
    dev = os.stat(d).st_dev
    for o in res.get("outputs", []):
        name = o["file"]
        if "/" in name or name.startswith(".") or name in {x["file"] for x in outputs}:
            continue
        try:
            st = os.lstat(d / name)  # lstat: a link planted by the runner is not an output
        except OSError:
            continue
        if stat.S_ISREG(st.st_mode) and st.st_dev == dev and o["kind"] in OUTPUT_KINDS:
            lang = o.get("lang")
            outputs.append({"file": name, "kind": o["kind"], "bytes": st.st_size,
                            **({"lang": lang} if isinstance(lang, str) and re.fullmatch(r"[a-z]{2,3}", lang) else {})})
    params = json.loads(job["params_json"])
    duration = float(job.get("duration_seconds") or job["claimed_seconds"])
    keep = {o["file"] for o in outputs}
    for p in d.iterdir():
        if p.name not in keep:
            if p.is_dir():
                import shutil

                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
    now = db.now()
    # Charging and finishing are one step: a job is never "done" without its charge, and a failure
    # after the charge can't be reported as "not charged". Only a job that is still running completes.
    with db.tx() as c:
        n = c.execute(
            "UPDATE jobs SET status = 'done', progress = 100, stage = 'Done', outputs_json = ?, cost_micros = 0,"
            " finished_at = ?, expires_at = ? WHERE id = ? AND status = 'running'",
            (json.dumps(outputs), now, now + get_settings().job_output_ttl_hours * 3600, job["id"]),
        ).rowcount
        if not n:
            log.warning("job %s is no longer running; not completing it", job["id"])
            return
        if job["hold_id"]:
            cost = ledger.capture_hold(
                job["hold_id"], _job_cost(job, params, duration),
                f"{jobs.job_label(job['kind'], params['denoise_tier'], params['transcribe_tier'])}: {jobs.display_name(job)[:60]} ({duration / 60:.1f} min)", c=c,
            )
            c.execute("UPDATE jobs SET cost_micros = ? WHERE id = ?", (cost, job["id"]))
    metrics.job_ended(job["id"])


def _bill_as_standard(job: dict, notice: str) -> dict:
    """A voice-rebuilt job that did not get the rebuild is priced as the standard tier, at the standard tier's
    price as it was when the job was created, and says so. The job record is updated and returned."""
    params = json.loads(job["params_json"])
    params = jobs.update_params(
        job["id"], denoise_tier="standard", rates={**params.get("rates", {}), "denoise": pricing.rate("denoise", "standard").micros_per_minute}
    )
    tier = "/".join(t for t in (params["denoise_tier"], params["transcribe_tier"]) if t)
    db.execute("UPDATE jobs SET tier = ? WHERE id = ?", (tier, job["id"]))
    jobs.set_notice(job["id"], notice)
    return {**job, "params_json": json.dumps(params), "tier": tier}


NO_REBUILD_NOTICE = "Voice rebuilding wasn't available for this file, so it was cleaned with the standard option and billed at its price."


def _use_available_tier(job: dict) -> dict:
    """A voice-rebuilt job whose models are not installed would silently run the standard chain, so
    it is billed as the standard tier (and says so) rather than charging for something it didn't get."""
    params = json.loads(job["params_json"])
    if params.get("denoise_tier") != "rebuild" or restore.available(get_settings().models_dir / "sidon"):
        return job
    log.error("job %s asked for voice rebuilding but the models are missing; billing the standard tier", job["id"])
    return _bill_as_standard(job, NO_REBUILD_NOTICE)


def process(job: dict, extra: dict) -> None:
    try:
        if job["next_stage"] != "finalize":
            job = _use_available_tier(job)
        d = _write_spec(job, extra, job.get("duration_seconds"))
        if job["next_stage"] == "finalize":
            res = run_stage(job, d, "finalize", 1800)
            job["duration_seconds"] = db.one("SELECT duration_seconds FROM jobs WHERE id = ?", (job["id"],))["duration_seconds"]
            _complete(job, d, res)
            return
        probe = run_stage(job, d, "probe", PROBE_TIMEOUT)
        try:
            duration = float(probe["duration"])  # the longer of the header's and the measured length
            channels = min(max(int(probe.get("channels") or 1), 1), 2)
        except (KeyError, TypeError, ValueError):
            raise StageFailed("Processing failed; you were not charged.", "malformed probe result")
        if not (math.isfinite(duration) and 0 < duration <= _max_seconds(job, json.loads(job["params_json"])) + 61):
            raise StageFailed("Processing failed; you were not charged.", "implausible duration")
        _reprice(job, duration)
        job["duration_seconds"] = duration
        params = json.loads(job["params_json"])
        jobs.update_params(job["id"], channels=channels)
        d = _write_spec(job, extra, duration)
        wall = jobs.stage_limits(params["denoise_tier"], duration, _max_seconds(job, params))[1]
        res = run_stage(job, d, "main", wall)
        if res.get("notice"):
            jobs.set_notice(job["id"], res["notice"])
        if params.get("denoise_tier") == "rebuild" and job["kind"] != "transcribe" and res.get("noise") != "rebuilt":
            # The rebuild did not run (the models vanished, or the chain fell back): never charge its price.
            log.error("job %s asked for voice rebuilding but it did not run (%s); billing the standard tier", job["id"], res.get("noise"))
            job = _bill_as_standard(job, NO_REBUILD_NOTICE)
        if res.get("needs_cloud_asr"):
            db.execute(
                "UPDATE jobs SET status = 'cloud_asr', stage = 'Waiting to transcribe', progress = 85,"
                " heartbeat_at = ? WHERE id = ?",
                (db.now(), job["id"]),
            )
            return
        _complete(job, d, res)
    except Interrupted:
        # Stopped by a restart or deploy, not a failure: back in the queue with the upload kept, no attempt used.
        interruptions = int(json.loads(job["params_json"]).get("interruptions", 0)) + 1
        if interruptions >= MAX_INTERRUPTIONS:
            log.error("job %s interrupted %d times; failing it", job["id"], interruptions)
            jobs.finish_failed(job, INTERRUPTED_MESSAGE)
        else:
            log.info("job %s put back in the queue (worker stopping, interruption %d)", job["id"], interruptions)
            jobs.update_params(job["id"], interruptions=interruptions)
            jobs.requeue(job)
    except StageFailed as e:
        jobs.finish_failed(job, e.user_message)
    except Exception:  # noqa: BLE001
        log.exception("job %s crashed", job["id"])
        jobs.finish_failed(job, "Processing failed; you were not charged.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    db.init_db()
    worker_id = f"{socket.gethostname()}:{os.getpid()}"
    extra = _settings_extra()
    signal.signal(signal.SIGTERM, lambda *_: _stop.set())
    signal.signal(signal.SIGINT, lambda *_: _stop.set())
    log.info("worker %s started", worker_id)
    load = hostload.get(get_settings().data_dir)
    last_sweep = 0.0
    while not _stop.is_set():
        if time.monotonic() - last_sweep > 60:
            last_sweep = time.monotonic()
            try:
                jobs.sweep()
            except Exception:  # noqa: BLE001
                log.exception("sweep failed")
        if GPU_MODE:
            _gpu_alive_marker().touch()
        job = claim(worker_id, load)
        if job is None:
            if GPU_MODE:
                _gpu_idle_marker().touch()
            _stop.wait(2.0)
            continue
        if GPU_MODE:
            _gpu_idle_marker().unlink(missing_ok=True)
        log.info("job %s (%s) claimed with %s threads", job["id"], job["kind"], job["threads"])
        process(job, {**extra, "threads": job["threads"] or extra["threads"], **({"heavy_device": heavy_device()} if GPU_MODE else {})})
    if GPU_MODE:  # stopping: the CPU workers must not wait for a worker that is gone
        _gpu_idle_marker().unlink(missing_ok=True)
        _gpu_alive_marker().unlink(missing_ok=True)


if __name__ == "__main__":
    main()
