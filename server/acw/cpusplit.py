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
UNITS = {"light": 1.0}
CAPTIONS_LOCAL_UNITS = 0.7  # streaming English caption model
ECHO_UNITS = 0.2  # echo cancellation: 0.020 cores per stream next to 0.096 for the denoiser (2026-10-15)
# Units one physical core carries with good latency while file jobs run on the
# other cores. Other CPUs: scaled at start-up by calibrate_units_per_core().
# * libdf with one copy of the model weights per session (no df_shares_model): 12 units on 5 cores, measured
#   2026-10-14 on the Ryzen 7 1700X. Each session's 8 MB of weights evicted the others' from the caches, so a
#   session cost 0.14 cores alone and 0.40 at 28 sessions.
# * libdf sharing one copy (df_shares_model, native/deepfilter/tract-0.23.patch), the Opus codec through libopus and
#   one worker process per pair of cores: 30 Opus sessions on 6 cores at p99 15 ms, 36 fell behind (2026-10-16).
#   4.05 at the reference cost is 4.3 per core on the 1700X (whose single-session cost is 0.104).
REFERENCE_UNITS_PER_CORE = 2.4
SHARED_MODEL_UNITS_PER_CORE = 4.05
REFERENCE_FRAME_COST = 0.110  # cores one standard session used single-threaded there
# Live audio runs in several worker processes, one per pair of live cores: one Python process falls behind at about
# ten sessions however many cores it has (each session's thread needs the GIL around every native call).
CORES_PER_STREAM_WORKER = 2
# Free-trial sessions only start while live is less than this full, so a trial never takes the last places from
# people who pay.
TRIAL_CAPACITY_FRACTION = 0.7


def session_units(cfg: dict) -> float:
    return (UNITS.get(cfg.get("denoise") or "", 0.0) + (CAPTIONS_LOCAL_UNITS if cfg.get("captions") == "local" else 0.0)
            + (ECHO_UNITS if cfg.get("echo_reference") else 0.0))


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


def stream_workers() -> int:
    """How many live worker processes run (acw.stream.worker): one per CORES_PER_STREAM_WORKER live cores, or
    ACW_STREAM_WORKERS. The web process finds them by their sockets, so only the worker side reads this."""
    env = os.environ.get("ACW_STREAM_WORKERS", "").strip()
    n = int(env) if env.isdigit() else -(-max_live_cores() // CORES_PER_STREAM_WORKER)
    return max(1, min(n, max_live_cores()))


def stream_worker_cores(index: int, workers: int) -> list[int]:
    """Physical core indexes (into physical_cores()) of live worker `index` (1-based) of `workers`: the live cores
    split into contiguous, nearly equal parts, the first workers taking the extra ones."""
    base, extra = divmod(max_live_cores(), workers)
    i = index - 1
    start = i * base + min(i, extra)
    return list(range(start, start + base + (1 if i < extra else 0)))


def cpus_of(core_indexes) -> set[int]:
    cores = physical_cores()
    return {c for i in core_indexes if 0 <= i < len(cores) for c in cores[i]}


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
    """Calibrated once per CPU model and remembered in the data dir, so a
    restart while the machine is busy doesn't measure (and keep) a lower value."""
    import json

    from .config import get_settings

    if os.environ.get("ACW_LIVE_UNITS_PER_CORE"):
        return float(os.environ["ACW_LIVE_UNITS_PER_CORE"])
    s = get_settings()
    cache = s.data_dir / "live-calibration.json"
    cpu = _cpu_model()
    shared = shares_model(s.bin_dir / "libdf.so")
    try:
        saved = json.loads(cache.read_text())
        if saved.get("cpu") == cpu and saved.get("shared_model", False) == shared:
            return float(saved["units_per_core"])
    except (OSError, ValueError, KeyError):
        pass
    try:
        value = calibrate_units_per_core(s.bin_dir / "libdf.so", s.models_dir / "DeepFilterNet3_onnx.tar.gz")
    except Exception:  # noqa: BLE001 - no model (tests, dev): assume the reference machine
        return REFERENCE_UNITS_PER_CORE
    try:
        cache.write_text(json.dumps({"cpu": cpu, "shared_model": shared, "units_per_core": round(value, 3)}))
    except OSError:
        pass  # sandboxed process: the next unsandboxed one writes it
    return value


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def calibrate_units_per_core(lib_path: Path, model_path: Path, runs: int = 3, seconds: float = 2.0) -> float:
    """Scale the reference capacity by how fast this CPU runs one session's
    model: the best of a few short runs (others' load only ever slows it)."""
    import time

    import numpy as np

    from acw_common.dfn import DeepFilter

    df = DeepFilter(lib_path, model_path)
    try:
        x = (0.05 * np.random.default_rng(0).standard_normal(int(48_000 * seconds))).astype(np.float32)
        df.process(x[:48_000])  # warm up
        costs = []
        for _ in range(runs):
            t = time.thread_time()
            df.process(x)
            costs.append((time.thread_time() - t) / seconds)
    finally:
        df.close()
    reference = SHARED_MODEL_UNITS_PER_CORE if shares_model(lib_path) else REFERENCE_UNITS_PER_CORE
    return reference * REFERENCE_FRAME_COST / max(min(costs), 1e-3)


def shares_model(lib_path: Path) -> bool:
    """Whether this libdf build keeps one copy of the model weights for all sessions (df_shares_model)."""
    import ctypes

    try:
        return hasattr(ctypes.CDLL(str(lib_path)), "df_shares_model")
    except OSError:
        return False


# --------------------------------------------------------------- live marker

def marker(data_dir: Path) -> Path:
    return data_dir / "live-units"


def set_live_units(data_dir: Path, units: float, units_per_core: float, cores=None) -> None:
    """Written by the web process whenever server-processed live sessions start/stop. `cores`: the physical core
    indexes of the live workers that have sessions using our CPU (file jobs keep off them and one spare core)."""
    p = marker(data_dir)
    tmp = p.with_suffix(".tmp")
    text = f"{max(0.0, units):.2f} {units_per_core:.3f}"
    if cores is not None:
        text += " " + (",".join(str(c) for c in sorted(set(cores))) or "-")
    tmp.write_text(text)
    tmp.replace(p)


def _read_marker(data_dir: Path) -> tuple[float, float, list[int] | None]:
    try:
        parts = marker(data_dir).read_text().split()
        cores = None
        if len(parts) > 2:
            cores = [] if parts[2] == "-" else [int(c) for c in parts[2].split(",")]
        return float(parts[0]), float(parts[1]), cores
    except (OSError, ValueError, IndexError):
        return 0.0, REFERENCE_UNITS_PER_CORE, None


def live_units(data_dir: Path) -> tuple[float, float]:
    """(units in use, units per core) from the marker; (0, reference) if none."""
    units, per_core, _ = _read_marker(data_dir)
    return units, per_core


def job_cpus_now(data_dir: Path) -> set[int]:
    """The logical CPUs file jobs may use right now: off the cores of the live workers that have sessions, plus the
    next live core (where the next session's worker would start), or by units from an older marker."""
    units, per_core, cores = _read_marker(data_dir)
    if cores is None:
        return job_cpus(units, per_core) if units > 0 else all_cpus()
    if not cores:
        return all_cpus()
    reserved = set(cores)
    spare = next((c for c in range(max_live_cores()) if c not in reserved), None)
    if spare is not None:
        reserved.add(spare)
    return cpus_of(i for i in range(len(physical_cores())) if i not in reserved) or all_cpus()


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
