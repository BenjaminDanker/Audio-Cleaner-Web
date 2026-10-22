"""The resource broker: one place that knows who may use the machine, in which order, and how full it is.

Everything that uses our CPU, memory or GPU belongs to a class, and the classes are ranked in one table (CLASSES)
instead of rules spread over the code:

  live        paid live audio: real time, so it is placed first; file jobs move off its cores (below)
  job         paid file jobs: batch work; it slows down (fewer cores) rather than stops when live needs room
  live_trial  free-trial live: real time too, but it never takes room from paid work: it only joins cores live
              already uses, or opens a new one while no paid job is running, and it stops at 70% of live capacity

A class outranks another when its rank is lower. The decisions read from the table:
  - job_cpus(): the cores jobs may use = everything minus the cores of classes that outrank jobs (plus a spare).
  - live_options(cls): what a new live session of that class may take (sent to the live engine with the start).
  - admit_job(): memory, disk and CPU room for a file job (worker/main.py claims through it).
  - gpu_stages() / gpu_only(): which file jobs the GPU worker takes, and which the CPU workers leave to it.
  - snapshot(): the whole picture in one place: live engine, per-core use, the card, jobs, queue, memory, disk
    (`python -m acw.admin capacity`, and the metrics file).

Scaling is linear: a live session costs the same CPU however many run (the native engine has no shared lock and
shares the model weights), and a job's cost depends on its audio only, so capacity grows with cores.
"""
from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import db, hostload, jobs
from .config import get_settings


@dataclass(frozen=True)
class Class:
    name: str
    rank: int  # lower = first
    realtime: bool
    paid: bool


CLASSES = {
    "live": Class("live", 0, True, True),
    "job": Class("job", 1, False, True),
    "live_trial": Class("live_trial", 2, True, False),
}


def outranks(a: str, b: str) -> bool:
    return CLASSES[a].rank < CLASSES[b].rank


# ----------------------------------------------------------------------------------------------- the cores
#
# Live audio and file jobs keep off each other's physical cores: a job thread on a live core's other hyper-thread
# halves the live thread's speed (one niced 6-thread job pushed 12 live sessions from ~5 ms to over 0.5 s; priorities
# don't help). The live engine places each session on one core, lowest first, and says which in its `ready`; the web
# process writes the cores in use to a small marker file; job workers read it every second and keep off those cores
# and the next one (so a new session starts on a free core). With no live session jobs use every core. No privileges
# needed: processes set their own CPU affinity.

_CPU = Path("/sys/devices/system/cpu")


@lru_cache(maxsize=1)
def physical_cores() -> list[list[int]]:
    """Logical CPUs grouped by physical core, in core order (the live engine computes the same)."""
    cores: dict[tuple[int, int], list[int]] = {}
    for cpu in sorted(os.sched_getaffinity(0) | set(range(os.cpu_count() or 1))):
        topo = _CPU / f"cpu{cpu}" / "topology"
        try:
            key = (int((topo / "physical_package_id").read_text()), int((topo / "core_id").read_text()))
        except (OSError, ValueError):
            key = (0, cpu)
        cores.setdefault(key, []).append(cpu)
    return [sorted(v) for _, v in sorted(cores.items())]


def max_live_cores() -> int:
    """Physical cores live may use: all but ACW_MIN_JOB_CORES (default 1 per 8 cores), which file jobs keep."""
    cores = len(physical_cores())
    keep = int(os.environ.get("ACW_MIN_JOB_CORES", max(1, round(cores / 8))))
    return cores - max(1, min(cores - 1, keep))


def cpus_of(core_indexes) -> set[int]:
    cores = physical_cores()
    return {c for i in core_indexes if 0 <= i < len(cores) for c in cores[i]}


def all_cpus() -> set[int]:
    return {c for core in physical_cores() for c in core}


def _marker(data_dir: Path) -> Path:
    return data_dir / "live-cores"


def publish_live_cores(cores, data_dir: Path | None = None) -> None:
    """Written by the live process whenever the set of cores carrying live sessions changes."""
    p = _marker(data_dir or get_settings().data_dir)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(",".join(str(c) for c in sorted(set(cores))))
    tmp.replace(p)


def live_cores(data_dir: Path | None = None) -> list[int]:
    try:
        text = _marker(data_dir or get_settings().data_dir).read_text().strip()
        return [int(c) for c in text.split(",")] if text else []
    except (OSError, ValueError):
        return []


def job_cpus(data_dir: Path | None = None) -> set[int]:
    """The logical CPUs file jobs may use now: every core but those carrying live sessions and the next live one."""
    busy = set(live_cores(data_dir))
    if not busy:
        return all_cpus()
    spare = next((c for c in range(max_live_cores()) if c not in busy), None)
    if spare is not None:
        busy.add(spare)
    return cpus_of(i for i in range(len(physical_cores())) if i not in busy) or all_cpus()


def _descendants(pid: int) -> list[int]:
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        try:
            for task in os.listdir(f"/proc/{p}/task"):
                todo.extend(int(c) for c in Path(f"/proc/{p}/task/{task}/children").read_text().split())
        except OSError:
            continue
    return out


def pin_tree(pid: int, cpus: set[int]) -> None:
    """Set the CPU affinity of every thread of `pid` and its descendants (new threads and children inherit it)."""
    for p in _descendants(pid):
        try:
            tasks = os.listdir(f"/proc/{p}/task")
        except OSError:
            continue
        for tid in tasks:
            try:
                os.sched_setaffinity(int(tid), cpus)
            except OSError:
                pass  # thread exited


# ------------------------------------------------------------------------------------------------ the GPU
#
# One card (GTX 960, 4 GB). What runs well on it was measured (docs/research/gpu-2026-10.md): the big models of file
# jobs. Not live audio (one 10 ms frame costs 8.7 ms of card latency, a caption chunk more than the CPU spends on it),
# not Whisper (the pinned CTranslate2 has no kernels for this card and turbo does not fit), not DeepFilterNet3 (its
# runtime is CPU only), not echo removal (no echo canceller in the GPU venv). Every workload is compute bound (two
# DPDFNet streams together are only 16 % faster than one) and Parakeet alone takes 3.3 of the 4 GB, so the card runs
# one job at a time: the GPU worker (acw-worker-gpu, its own venv) claims one, its stages use the card one after
# another. The CPU workers leave the card's jobs to it while it is idle (when it is busy they take them: the card is
# faster, the CPU is not slow for these). The voice rebuild runs only on the card: 7.4x real time there against 0.74x
# on four cores, which would hold a quarter of the machine for longer than the recording; while no GPU worker is up it
# can't be ordered (creating one says so) and queued ones wait.


@dataclass(frozen=True)
class GpuWork:
    name: str
    card_mb: int  # peak of the whole card while it runs
    seconds_per_minute: float  # wall seconds per minute of audio (per channel cleaned)
    host_cores: float  # CPU it still needs (decoding, framing, the decoder loop)


GPU_WORK = {
    "transcribe": GpuWork("Parakeet transcription", 3300, 2.0, 0.9),
    "heavy": GpuWork("DPDFNet-8 heavy noise removal", 360, 9.7, 0.1),  # 5.5 s on 30 min files, 16 s on 2.5 min
    "rebuild": GpuWork("Sidon voice rebuild", 2600, 8.2, 0.13),
    # A video's picture made again (options.video_out): NVENC, the card's own encoder (it leaves the CUDA cores alone); the host still
    # decodes the original (1.7 cores while it runs, 1080p30 H.264). On the CPU workers x264 does it (jobs.SPEED["encode"]).
    "encode": GpuWork("NVENC video encoding", 250, 8.2, 1.7),
}
GPU_IDLE_FRESH_S = 6
GPU_ALIVE_FRESH_S = 90


def gpu_stages(job: dict) -> list[str]:
    """The GPU_WORK stages of `job` that make it a GPU job (empty: the CPU workers' job). A recording that turns out
    lightly noisy under "auto" is cleaned by DeepFilterNet3 on the GPU worker's own CPU threads. A video whose picture is made
    again adds "encode"."""
    stages = _sound_stages(job)
    if job.get("next_stage") != "finalize" and jobs.encodes(json.loads(job["params_json"])):
        stages.append("encode")
    return stages


def _sound_stages(job: dict) -> list[str]:
    params = json.loads(job["params_json"])
    if job.get("next_stage") == "finalize":
        return []
    if params.get("run"):  # an edit run (editing.py): the steps its missing blocks need, found when it was quoted
        return list(params["run"].get("gpu") or [])
    if (params.get("edit") or {}).get("first_run"):  # a job made with a timeline: by the layers it has
        layers = (params["edit"].get("timeline") or {}).get("layers") or {}
        stages = ["transcribe"] if params.get("transcribe_tier") == "local" or any(p.get("quality") == "standard" for p in layers.get("subtitles") or []) else []
        if layers.get("rebuild"):
            stages.append("rebuild")
        elif any(p.get("remover", "auto") in ("auto", "heavy") for p in layers.get("noise") or [] if not p.get("browser")):
            stages.append("heavy")
        return stages
    # an automatic job not decided yet (its file is measured on any worker first: automatic.py) or a conversion: the card only for what
    # it decides, or for the picture
    return ["transcribe"] if params.get("transcribe_tier") == "local" else []


def gpu_only(job: dict) -> bool:
    """A job only the GPU worker takes: the voice rebuild."""
    return "rebuild" in gpu_stages(job)


GPU_WORKER_PREFIX = "gpu:"  # the worker_id of jobs the GPU worker runs starts with it


def is_gpu_worker(worker_id: str | None) -> bool:
    return bool(worker_id) and worker_id.startswith(GPU_WORKER_PREFIX)


def _gpu_marker(state: str) -> Path:
    return get_settings().data_dir / f"gpu-worker-{state}"


def mark_gpu_worker(state: str) -> None:
    """The GPU worker says how it is: "alive" (at least every 20 s, also while a job runs), "idle" (every claim
    that found nothing), "busy" (it took a job) or "stopped"."""
    if state in ("alive", "idle"):
        _gpu_marker(state).touch()
    if state in ("busy", "stopped"):
        _gpu_marker("idle").unlink(missing_ok=True)
    if state == "stopped":
        _gpu_marker("alive").unlink(missing_ok=True)


def _fresh(marker: Path, seconds: float) -> bool:
    try:
        return time.time() - marker.stat().st_mtime < seconds
    except OSError:
        return False


def gpu_worker_idle() -> bool:
    return _fresh(_gpu_marker("idle"), GPU_IDLE_FRESH_S)


def gpu_worker_up() -> bool:
    return _fresh(_gpu_marker("alive"), GPU_ALIVE_FRESH_S)


def gpu_card() -> dict:
    """The card's memory and load now (nvidia-smi); {"down": why} without one."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip().splitlines()[0]
        name, used, total, util = (v.strip() for v in out.split(","))
        return {"name": name, "memoryUsedMb": int(used), "memoryTotalMb": int(total), "busyPercent": int(util)}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as e:
        return {"down": str(e)[:80]}


# ------------------------------------------------------------------------------------------ admission

# A job joins others already running only if this many cores are idle.
MIN_IDLE_CORES = 2.0


def running_paid_jobs() -> int:
    return db.one("SELECT COUNT(*) AS n FROM jobs WHERE status = 'running'")["n"]


def live_options(cls: str) -> dict:
    """What a new live session of class `cls` may take, for the live engine's start message. A class that paid
    work outranks may not open a new core while that work runs (it would push jobs off it)."""
    if outranks("job", cls) and running_paid_jobs() > 0:
        return {"may_open_core": False}
    return {}


def admit_job(job: dict, snap: hostload.Snapshot, running: list[dict], cpus: set[int] | None = None,
              gpu: bool = False, max_threads: int = 6) -> tuple[bool, str, int]:
    """Whether `job` may start now, why not, and how many threads it gets. Memory and disk must fit with a reserve
    left for the rest of the host; CPU only decides when other jobs already run (with nothing running the queue
    always moves, niced, so it yields to everything that outranks it)."""
    mem, disk = jobs.job_needs(job)
    if gpu:  # the model lives in GPU memory
        mem = min(mem, jobs.JOB_MEM + jobs.GB)
    now = db.now()
    # Jobs started in the last minute may not have allocated their memory yet.
    mem += sum(jobs.job_needs(r)[0] for r in running if (r["started_at"] or 0) > now - 60)
    if snap.mem_available - mem < jobs.MEM_RESERVE or (snap.slice_headroom is not None and snap.slice_headroom - mem < jobs.SLICE_RESERVE):
        return False, f"memory: {snap.memory_headroom / jobs.GB:.1f} GB free, job needs {mem / jobs.GB:.1f} GB", 0
    if snap.disk_free - disk < jobs.DISK_RESERVE:
        return False, f"disk: {snap.disk_free / jobs.GB:.0f} GB free, job needs {disk / jobs.GB:.1f} GB", 0
    # Idle CPU among the cores jobs may use now (live audio, which outranks jobs, may have taken some).
    idle = snap.idle_cores
    if cpus is not None and snap.busy_per_cpu:
        idle = sum(1.0 - snap.busy_per_cpu[c] for c in cpus if c < len(snap.busy_per_cpu))
    if running and idle < MIN_IDLE_CORES and not gpu:
        return False, f"cpu: {idle:.1f} job cores idle", 0
    return True, "", max(2, min(max_threads, int(idle) - 1))


# --------------------------------------------------------------------------------------------- the whole picture

def live_engine_status(timeout: float = 2.0) -> dict:
    """Ask the live engine how it is doing (a MSG_QUERY on its socket); {"down": why} if it doesn't answer."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(str(get_settings().stream_worker_socket))
        s.sendall(b"Q" + struct.pack("<I", 0))
        data = b""
        while len(data) < 5 or len(data) < 5 + struct.unpack("<I", data[1:5])[0]:
            chunk = s.recv(65536)
            if not chunk:
                raise ConnectionError("closed")
            data += chunk
        return json.loads(data[5:])
    except (OSError, ValueError) as e:
        return {"down": str(e)[:80]}
    finally:
        s.close()


def snapshot() -> dict:
    """Everything that competes for the machine, in one place."""
    s = get_settings()
    snap = hostload.get(s.data_dir).snapshot()
    cores = live_cores(s.data_dir)
    running = [dict(r) for r in db.all_rows(
        "SELECT id, kind, json_extract(params_json, '$.denoise_tier') AS tier, started_at, worker_id FROM jobs WHERE status = 'running'")]
    queued = db.one("SELECT COUNT(*) AS n FROM jobs WHERE status = 'queued'")["n"]
    live = db.all_rows("SELECT trial_seconds > 0 AS trial, COUNT(*) AS n FROM stream_sessions WHERE status = 'active' GROUP BY 1")
    by_class = {("live_trial" if r["trial"] else "live"): r["n"] for r in live}
    return {
        "classes": [{"name": c.name, "rank": c.rank, "realtime": c.realtime, "paid": c.paid} for c in sorted(CLASSES.values(), key=lambda c: c.rank)],
        "cpu": {"cores": snap.cores, "busy": round(snap.busy_cores, 2), "niced": round(snap.nice_cores, 2),
                "perCpu": [round(b, 2) for b in snap.busy_per_cpu]},
        "memory": {"available": snap.mem_available, "sliceHeadroom": snap.slice_headroom, "reserve": jobs.MEM_RESERVE},
        "disk": {"free": snap.disk_free, "reserve": jobs.DISK_RESERVE},
        "live": {"sessions": by_class, "liveCores": cores, "engine": live_engine_status()},
        "gpu": {"worker": "idle" if gpu_worker_idle() else "busy" if gpu_worker_up() else "down", "card": gpu_card()},
        "jobs": {"running": running, "queued": queued, "jobCpus": sorted(job_cpus(s.data_dir))},
    }


def format_snapshot(sn: dict) -> str:
    """A plain-text view of snapshot() for the admin command."""
    gb = 1024**3
    lines = [f"CPU: {sn['cpu']['busy']} of {sn['cpu']['cores']} logical CPUs busy ({sn['cpu']['niced']} niced)"]
    lines.append("  per CPU: " + " ".join(f"{b:.0%}" for b in sn["cpu"]["perCpu"]))
    m = sn["memory"]
    lines.append(f"Memory: {m['available'] / gb:.1f} GB available (reserve {m['reserve'] / gb:.1f} GB)"
                 + (f", {m['sliceHeadroom'] / gb:.1f} GB under the service limit" if m["sliceHeadroom"] is not None else ""))
    lines.append(f"Disk: {sn['disk']['free'] / gb:.0f} GB free")
    lv = sn["live"]
    lines.append(f"Live: {lv['sessions'] or 'none'}; on cores {lv['liveCores'] or '-'}")
    e = lv["engine"]
    if "down" in e:
        lines.append(f"  live engine NOT ANSWERING ({e['down']})")
    else:
        lines.append(f"  live engine: {e['sessions']} sessions, {e['units']:.1f} of {e['capacityUnits']:.1f} units, "
                     f"captions {e['captionStreams']}/{e['captionCap']}")
    g = sn["gpu"]
    card = g["card"]
    lines.append(f"GPU worker: {g['worker']}; " + (f"card NOT ANSWERING ({card['down']})" if "down" in card else
                 f"{card['name']}: {card['memoryUsedMb']} of {card['memoryTotalMb']} MB, {card['busyPercent']}% busy"))
    jb = sn["jobs"]
    lines.append(f"Jobs: {len(jb['running'])} running, {jb['queued']} queued; job CPUs {jb['jobCpus']}")
    for r in jb["running"]:
        lines.append(f"  {r['id'][:8]} {r['kind']}/{r['tier']} on {r['worker_id']}")
    lines.append("Priority: " + " > ".join(c["name"] for c in sn["classes"]))
    return "\n".join(lines)
