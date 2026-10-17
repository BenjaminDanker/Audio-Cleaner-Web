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
  - snapshot(): the whole picture in one place: live engine(s), per-core use, jobs, queue, memory, disk
    (`python -m acw.admin capacity`, and the metrics file).

Scaling is linear: a live session costs the same CPU however many run (the native engine has no shared lock and
shares the model weights), and a job's cost depends on its audio only, so capacity grows with cores.
"""
from __future__ import annotations

import json
import os
import socket
import struct
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
    """Written by the web process whenever the set of cores carrying live sessions changes."""
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
    jb = sn["jobs"]
    lines.append(f"Jobs: {len(jb['running'])} running, {jb['queued']} queued; job CPUs {jb['jobCpus']}")
    for r in jb["running"]:
        lines.append(f"  {r['id'][:8]} {r['kind']}/{r['tier']} on {r['worker_id']}")
    lines.append("Priority: " + " > ".join(c["name"] for c in sn["classes"]))
    return "\n".join(lines)
