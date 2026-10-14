"""Keep live audio and file jobs off each other's physical cores.

Measured on the Ryzen 7 1700X (8 cores / 16 threads): one niced 6-thread file
job running next to 12 live sessions pushed their frame latency from ~5 ms to
over 0.5 s. Priorities don't help, because a job thread on the other
hyper-thread of the same physical core still halves the live thread's speed.
So while any server-processed live session is running, file jobs are pinned
to their own physical cores; otherwise they may use every core.

No privileges needed: processes set their own CPU affinity (cpuset isn't
delegated to user services). The web process, which knows the live sessions,
writes a small marker file the job workers read.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_CPU = Path("/sys/devices/system/cpu")


@lru_cache(maxsize=1)
def physical_cores() -> list[list[int]]:
    """Logical CPUs grouped by physical core, in core order."""
    cores: dict[tuple[int, int], list[int]] = {}
    for cpu in sorted(os.sched_getaffinity(0) | set(range(os.cpu_count() or 1))):
        topo = _CPU / f"cpu{cpu}" / "topology"
        try:
            key = (int((topo / "physical_package_id").read_text()), int((topo / "core_id").read_text()))
        except (OSError, ValueError):
            key = (0, cpu)
        cores.setdefault(key, []).append(cpu)
    return [sorted(v) for _, v in sorted(cores.items())]


def live_core_count() -> int:
    cores = len(physical_cores())
    return max(1, min(cores - 1, int(os.environ.get("ACW_LIVE_PHYSICAL_CORES", round(cores * 5 / 8)))))


def live_cpus() -> set[int]:
    return {c for core in physical_cores()[: live_core_count()] for c in core}


def job_cpus() -> set[int]:
    rest = {c for core in physical_cores()[live_core_count():] for c in core}
    return rest or set(range(os.cpu_count() or 1))


def all_cpus() -> set[int]:
    return {c for core in physical_cores() for c in core}


# --------------------------------------------------------------- live marker

def marker(data_dir: Path) -> Path:
    return data_dir / "live-sessions"


def set_live_sessions(data_dir: Path, n: int) -> None:
    """Called by the web process whenever server-processed live sessions start/stop."""
    p = marker(data_dir)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(str(max(0, n)))
    tmp.replace(p)


def live_active(data_dir: Path) -> bool:
    try:
        return int(marker(data_dir).read_text() or 0) > 0
    except (OSError, ValueError):
        return False


# ------------------------------------------------------------------ pinning

def _descendants(pid: int) -> list[int]:
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        try:
            for task in os.listdir(f"/proc/{p}/task"):
                children = Path(f"/proc/{p}/task/{task}/children").read_text().split()
                todo.extend(int(c) for c in children)
        except OSError:
            continue
    return out


def pin_tree(pid: int, cpus: set[int]) -> None:
    """Set the CPU affinity of every thread of `pid` and its descendants
    (new threads and children inherit it)."""
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
