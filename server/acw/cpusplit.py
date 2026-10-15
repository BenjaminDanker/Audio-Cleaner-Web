"""Keep live audio and file jobs off each other's physical cores.

Measured on the Ryzen 7 1700X (8 cores / 16 threads): one niced 6-thread file
job running next to 12 live sessions pushed their frame latency from ~5 ms to
over 0.5 s. Priorities don't help, because a job thread on the other
hyper-thread of the same physical core still halves the live thread's speed.
So live audio gets whole physical cores, as many as its sessions need plus a
spare (up to all but `min_job_cores()`), and file jobs are re-pinned to the
rest; with no live sessions, jobs use every core. Live comes first: jobs slow
down, live sessions are only refused once live would need more than that.

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


# Capacity is counted in "units": one standard live-denoise session = 1.
UNITS = {"light": 1.0, "low_latency": 3.6}
CAPTIONS_LOCAL_UNITS = 0.7  # streaming English caption model
# Units one physical core carries with good latency while file jobs run on the
# other cores: 12 units on 5 cores, measured 2026-10-14 on the Ryzen 7 1700X.
# Other CPUs: scaled at start-up by calibrate_units_per_core().
REFERENCE_UNITS_PER_CORE = 2.4
REFERENCE_FRAME_COST = 0.110  # cores one standard session used single-threaded there


def session_units(cfg: dict) -> float:
    return UNITS.get(cfg.get("denoise") or "", 0.0) + (CAPTIONS_LOCAL_UNITS if cfg.get("captions") == "local" else 0.0)


def min_job_cores() -> int:
    """Physical cores file jobs always keep, however busy live gets."""
    cores = len(physical_cores())
    return max(1, min(cores - 1, int(os.environ.get("ACW_MIN_JOB_CORES", max(1, round(cores / 8))))))


def max_live_cores() -> int:
    return len(physical_cores()) - min_job_cores()


def live_cores_for(units: float, units_per_core: float) -> int:
    """Physical cores reserved for live audio carrying `units`: what the
    sessions need plus one spare (so the next session starts on a free core)."""
    if units <= 0:
        return 0
    return min(max_live_cores(), int(-(-units // units_per_core)) + 1)


def live_cpus(units: float | None = None, units_per_core: float = REFERENCE_UNITS_PER_CORE) -> set[int]:
    """Logical CPUs of the reserved live cores (all possible live cores if `units` is None)."""
    n = max_live_cores() if units is None else live_cores_for(units, units_per_core)
    return {c for core in physical_cores()[:n] for c in core}


def job_cpus(units: float, units_per_core: float = REFERENCE_UNITS_PER_CORE) -> set[int]:
    n = live_cores_for(units, units_per_core)
    return {c for core in physical_cores()[n:] for c in core}


def all_cpus() -> set[int]:
    return {c for core in physical_cores() for c in core}


@lru_cache(maxsize=1)
def units_per_core() -> float:
    """Calibrated once per process (web and stream worker measure the same CPU)."""
    from .config import get_settings

    s = get_settings()
    try:
        return calibrate_units_per_core(s.bin_dir / "libdf.so", s.models_dir / "DeepFilterNet3_onnx.tar.gz")
    except Exception:  # noqa: BLE001 - no model (tests, dev): assume the reference machine
        return REFERENCE_UNITS_PER_CORE


def calibrate_units_per_core(lib_path: Path, model_path: Path, seconds: float = 4.0) -> float:
    """Scale the reference capacity by how fast this CPU runs one session's model."""
    import time

    import numpy as np

    from acw_common.dfn import DeepFilter

    if os.environ.get("ACW_LIVE_UNITS_PER_CORE"):
        return float(os.environ["ACW_LIVE_UNITS_PER_CORE"])
    df = DeepFilter(lib_path, model_path)
    try:
        x = (0.05 * np.random.default_rng(0).standard_normal(int(48_000 * seconds))).astype(np.float32)
        df.process(x[:48_000])  # warm up
        t = time.thread_time()
        df.process(x)
        cost = (time.thread_time() - t) / seconds
    finally:
        df.close()
    return REFERENCE_UNITS_PER_CORE * REFERENCE_FRAME_COST / max(cost, 1e-3)


# --------------------------------------------------------------- live marker

def marker(data_dir: Path) -> Path:
    return data_dir / "live-units"


def set_live_units(data_dir: Path, units: float, units_per_core: float) -> None:
    """Written by the web process whenever server-processed live sessions start/stop."""
    p = marker(data_dir)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(f"{max(0.0, units):.2f} {units_per_core:.3f}")
    tmp.replace(p)


def live_units(data_dir: Path) -> tuple[float, float]:
    """(units in use, units per core) from the marker; (0, reference) if none."""
    try:
        units, per_core = marker(data_dir).read_text().split()
        return float(units), float(per_core)
    except (OSError, ValueError):
        return 0.0, REFERENCE_UNITS_PER_CORE


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
