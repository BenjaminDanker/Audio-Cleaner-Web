"""Job worker: `python -m acw.worker.main`.

Claims one queued job at a time and drives the sandboxed runner through its
stages. This process has database access but (under systemd) no network; the
runner it spawns has neither. Run several instances for parallelism.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import acw_common

from .. import cpusplit, db, hostload, jobs
from ..billing import ledger, pricing
from ..config import get_settings

log = logging.getLogger("acw.worker")
_stop = threading.Event()


# The GPU worker (acw-worker-gpu, its own venv with CUDA ONNX Runtime) only
# takes local transcription jobs and runs Parakeet on the GPU; the CPU workers
# leave those to it while it is idle. See docs/operations.md.
GPU_MODE = os.environ.get("ACW_WORKER_GPU") == "1"
GPU_IDLE_FRESH_S = 6


def _gpu_idle_marker() -> Path:
    return get_settings().data_dir / "gpu-worker-idle"


def gpu_worker_idle() -> bool:
    try:
        return time.time() - _gpu_idle_marker().stat().st_mtime < GPU_IDLE_FRESH_S
    except OSError:
        return False


def is_local_transcription(job: dict) -> bool:
    return json.loads(job["params_json"]).get("transcribe_tier") == "local" and job.get("next_stage") != "finalize"


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
        "gpu_devices": [*devices, "/proc/self"],
        "gpu_read": [*read, "/proc/self"],
        "address_space_bytes": 0,
    }


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


# A job joins others already running only if this many cores are idle.
MIN_IDLE_CORES = 2.0
MAX_JOB_THREADS = int(os.environ.get("ACW_JOB_THREADS_MAX", "6"))


def admission(
    job: dict, snap: hostload.Snapshot, running: list[dict], cpus: set[int] | None = None, gpu: bool = False
) -> tuple[bool, str, int]:
    """Whether `job` may start now, why not, and how many threads it gets.

    Memory and disk must fit with a reserve left for the rest of the host.
    CPU only decides when other jobs are already running: with nothing running
    the queue always moves (niced, so it yields to everything else)."""
    mem, disk = jobs.job_needs(job)
    if gpu:  # the model lives in GPU memory
        mem = min(mem, jobs.JOB_MEM + jobs.GB)
    now = db.now()
    # Jobs started in the last minute may not have allocated their memory yet.
    mem += sum(jobs.job_needs(r)[0] for r in running if (r["started_at"] or 0) > now - 60)
    if snap.memory_headroom - mem < jobs.MEM_RESERVE:
        return False, f"memory: {snap.memory_headroom / jobs.GB:.1f} GB free, job needs {mem / jobs.GB:.1f} GB", 0
    if snap.disk_free - disk < jobs.DISK_RESERVE:
        return False, f"disk: {snap.disk_free / jobs.GB:.0f} GB free, job needs {disk / jobs.GB:.1f} GB", 0
    # Idle CPU among the cores jobs may use now (live audio may have taken some).
    idle = snap.idle_cores
    if cpus is not None and snap.busy_per_cpu:
        idle = sum(1.0 - snap.busy_per_cpu[c] for c in cpus if c < len(snap.busy_per_cpu))
    if running and idle < MIN_IDLE_CORES and not gpu:
        return False, f"cpu: {idle:.1f} job cores idle", 0
    return True, "", max(2, min(MAX_JOB_THREADS, int(idle) - 1))


_last_wait_log = 0.0


def claim(worker_id: str, load: hostload.HostLoad | None = None) -> dict | None:
    """Take the next queued job (jobs.QUEUE_ORDER) if the host has room for it.
    The head of the queue waits rather than letting smaller jobs overtake it."""
    global _last_wait_log
    now = db.now()
    with db.tx() as c:
        rows = [dict(r) for r in c.execute(
            f"SELECT j.* FROM jobs j WHERE j.status = 'queued' ORDER BY {jobs.QUEUE_ORDER} LIMIT 50"
        ).fetchall()]
        if GPU_MODE:
            rows = [r for r in rows if is_local_transcription(r)]
        elif gpu_worker_idle():
            rows = [r for r in rows if not is_local_transcription(r)]  # the GPU worker takes these
        if not rows:
            return None
        row = rows[0]
        threads = None
        if load is not None:
            running = [dict(r) for r in c.execute("SELECT * FROM jobs WHERE status = 'running'").fetchall()]
            units, per_core = cpusplit.live_units(get_settings().data_dir)
            cpus = cpusplit.job_cpus(units, per_core) if units > 0 else cpusplit.all_cpus()
            ok, why, threads = admission(row, load.snapshot(), running, cpus, gpu=GPU_MODE)
            if not ok:
                if time.monotonic() - _last_wait_log > 60:
                    _last_wait_log = time.monotonic()
                    log.info("job %s waits for resources (%s)", row["id"], why)
                return None
        c.execute(
            "UPDATE jobs SET status = 'running', worker_id = ?, heartbeat_at = ?, attempts = attempts + 1,"
            " started_at = COALESCE(started_at, ?) WHERE id = ?",
            (worker_id, now, now, row["id"]),
        )
        job = dict(c.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone())
        job["threads"] = threads
        return job


def _write_spec(job: dict, extra: dict) -> Path:
    d = jobs.job_dir(job["id"])
    params = json.loads(job["params_json"])
    max_seconds = (
        pricing.FREE_TRANSCRIBE_MAX_FILE_SECONDS if job["free"] else pricing.MAX_JOB_SECONDS
    )
    spec = {
        **extra,
        "job_id": job["id"],
        "kind": job["kind"],
        "denoise_tier": params["denoise_tier"],
        "transcribe_tier": params["transcribe_tier"],
        "input": f"input{params['input_ext']}",
        "input_ext": params["input_ext"],
        "output_stem": jobs.safe_stem(job["input_name"]),
        "options": params["options"],
        "max_seconds": max_seconds,
        # Generous ceilings; the wall-clock timeout below is the real limit.
        "cpu_seconds": int(max_seconds * 6 + 1800),
        "address_space_bytes": 32 * 1024**3,  # memory-mapped scratch files count here, not in RAM
        "max_file_bytes": 16 * 1024**3,
    }
    (d / "spec.json").write_text(json.dumps(spec))
    return d


class StageFailed(Exception):
    def __init__(self, user_message: str, internal: str | None = None):
        super().__init__(internal or user_message)
        self.user_message = user_message


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
    deadline = time.monotonic() + timeout
    last_update = 0.0

    data_dir = get_settings().data_dir
    pinned: set[int] | None = None

    def watchdog() -> None:
        nonlocal pinned
        beat = 0.0
        while proc.poll() is None:
            # Off the live-audio cores while live sessions run (acw.cpusplit).
            units, per_core = cpusplit.live_units(data_dir)
            want = cpusplit.job_cpus(units, per_core) if units > 0 else cpusplit.all_cpus()
            if want != pinned:
                cpusplit.pin_tree(proc.pid, want)
                pinned = want
            if time.monotonic() > deadline or _stop.is_set():
                os.killpg(proc.pid, signal.SIGKILL)
                return
            if time.monotonic() - beat > 20:
                beat = time.monotonic()
                try:
                    db.execute("UPDATE jobs SET heartbeat_at = ? WHERE id = ?", (db.now(), job["id"]))
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(1)

    threading.Thread(target=watchdog, daemon=True).start()
    for raw in proc.stdout:  # type: ignore[union-attr]
        try:
            msg = json.loads(raw)
        except ValueError:
            continue
        now = time.monotonic()
        if now - last_update >= 1.0 or msg.get("progress") == 100:
            last_update = now
            db.execute(
                "UPDATE jobs SET progress = ?, stage = ?, heartbeat_at = ? WHERE id = ?",
                (max(0, min(100, int(msg.get("progress", 0)))), str(msg.get("stage", ""))[:40], db.now(), job["id"]),
            )
    proc.wait()
    if not result_file.exists():
        tail = b"".join(stderr_tail).decode(errors="replace")[-2000:]
        log.error("job %s stage %s died rc=%s: %s", job["id"], stage, proc.returncode, tail)
        if time.monotonic() > deadline:
            raise StageFailed("Processing took too long and was stopped; you were not charged.")
        raise StageFailed("Processing failed; you were not charged.", f"rc={proc.returncode}")
    res = json.loads(result_file.read_text())
    if not res.get("ok"):
        if res.get("user_error"):
            raise StageFailed(res["user_error"])
        log.error("job %s stage %s error: %s", job["id"], stage, res.get("error"))
        raise StageFailed("Processing failed; you were not charged.", res.get("error"))
    return res


def _reprice(job: dict, duration: float) -> None:
    """Make the hold (or free allowance) match the real duration before the
    expensive work starts."""
    params = json.loads(job["params_json"])
    db.execute("UPDATE jobs SET duration_seconds = ? WHERE id = ?", (duration, job["id"]))
    if job["free"]:
        delta = duration - job["claimed_seconds"]
        if delta > 0:
            day = params["free_day"]
            row = db.one("SELECT seconds FROM free_usage WHERE user_id = ? AND day = ?", (job["user_id"], day))
            if (row["seconds"] if row else 0) + delta > pricing.FREE_TRANSCRIBE_SECONDS_PER_DAY + 30:
                raise StageFailed("This file is longer than your remaining free minutes today.")
        jobs.adjust_free(job["user_id"], params["free_day"], delta)
        job["claimed_seconds"] = duration
        return
    cost = pricing.job_cost_micros(job["kind"], params["denoise_tier"], params["transcribe_tier"], duration)
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
    for o in res.get("outputs", []):
        p = d / o["file"]
        if p.parent == d and p.is_file():
            outputs.append({**o, "bytes": p.stat().st_size})
    params = json.loads(job["params_json"])
    duration = float(job.get("duration_seconds") or job["claimed_seconds"])
    cost = 0
    if not job["free"] and job["hold_id"]:
        cost = pricing.job_cost_micros(job["kind"], params["denoise_tier"], params["transcribe_tier"], duration)
        cost = ledger.capture_hold(job["hold_id"], cost, f"{job['kind'].replace('_', ' + ')}: {job['input_name'][:60]} ({duration / 60:.1f} min)")
    keep = {o["file"] for o in outputs}
    for p in d.iterdir():
        if p.name not in keep:
            if p.is_dir():
                import shutil

                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
    now = db.now()
    db.execute(
        "UPDATE jobs SET status = 'done', progress = 100, stage = 'done', outputs_json = ?, cost_micros = ?,"
        " finished_at = ?, expires_at = ? WHERE id = ?",
        (json.dumps(outputs), cost, now, now + get_settings().job_output_ttl_hours * 3600, job["id"]),
    )


def process(job: dict, extra: dict) -> None:
    d = _write_spec(job, extra)
    budget = pricing.MAX_JOB_SECONDS * 2 + 1800
    try:
        if job["next_stage"] == "finalize":
            res = run_stage(job, d, "finalize", 1800)
            job["duration_seconds"] = db.one("SELECT duration_seconds FROM jobs WHERE id = ?", (job["id"],))["duration_seconds"]
            _complete(job, d, res)
            return
        probe = run_stage(job, d, "probe", 120)
        _reprice(job, float(probe["duration"]))
        job["duration_seconds"] = float(probe["duration"])
        res = run_stage(job, d, "main", min(budget, probe["duration"] * 3 + 1800))
        if res.get("needs_cloud_asr"):
            db.execute(
                "UPDATE jobs SET status = 'cloud_asr', stage = 'transcribing (premium)', progress = 85,"
                " heartbeat_at = ? WHERE id = ?",
                (db.now(), job["id"]),
            )
            return
        _complete(job, d, res)
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
        job = claim(worker_id, load)
        if job is None:
            if GPU_MODE:
                _gpu_idle_marker().touch()
            _stop.wait(2.0)
            continue
        if GPU_MODE:
            _gpu_idle_marker().unlink(missing_ok=True)
        log.info("job %s (%s) claimed with %s threads", job["id"], job["kind"], job["threads"])
        process(job, {**extra, "threads": job["threads"] or extra["threads"]})


if __name__ == "__main__":
    main()
