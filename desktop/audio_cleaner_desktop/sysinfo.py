"""How many files the machine can clean at once. No Qt in here."""
from __future__ import annotations

import ctypes
import functools
import os
import subprocess
import sys

# Peak memory of one cleaning job, and of the speech model when subtitles are on.
CLEAN_JOB_GB = 1.5
SUBTITLE_EXTRA_GB = 1.0
# Share of the installed memory the app may plan to use.
RAM_BUDGET = 0.6
MAX_PARALLEL = 4


@functools.cache
def physical_cores() -> int:
    logical = os.cpu_count() or 2
    if sys.platform == "darwin":
        try:
            return int(subprocess.run(["sysctl", "-n", "hw.physicalcpu"], capture_output=True, text=True, timeout=5).stdout)
        except (OSError, ValueError, subprocess.SubprocessError):
            return logical
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                blocks = f.read().split("\n\n")
            cores = set()
            for block in blocks:
                fields = dict(map(str.strip, line.split(":", 1)) for line in block.splitlines() if ":" in line)
                if "core id" in fields:
                    cores.add((fields.get("physical id"), fields["core id"]))
            if cores:
                return len(cores)
        except OSError:
            pass
    return max(1, logical // 2)


@functools.cache
def total_ram_gb() -> float:
    """Installed memory in GB (0 when it cannot be read)."""
    if os.name == "nt":

        class _Status(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                ("avail", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong), ("page_avail", ctypes.c_ulonglong),
                ("virt_total", ctypes.c_ulonglong), ("virt_avail", ctypes.c_ulonglong), ("ext", ctypes.c_ulonglong),
            ]

        status = _Status()
        status.length = ctypes.sizeof(_Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.total / 2**30
        return 0.0
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
    except (ValueError, OSError, AttributeError):
        return 0.0


def max_parallel(requested: int, subtitles: bool, *, cores: int | None = None, ram_gb: float | None = None) -> int:
    """Files to clean at once. `requested` is 1-4, or 0 for Auto (a third of the physical cores).
    Never more than the memory budget allows."""
    cores = physical_cores() if cores is None else cores
    ram_gb = total_ram_gb() if ram_gb is None else ram_gb
    wanted = requested or min(MAX_PARALLEL, max(1, cores // 3))
    per_job = CLEAN_JOB_GB + (SUBTITLE_EXTRA_GB if subtitles else 0.0)
    by_ram = max(1, int(ram_gb * RAM_BUDGET // per_job)) if ram_gb else wanted
    return max(1, min(wanted, MAX_PARALLEL, by_ram))


def threads_per_job(parallel: int) -> int:
    """CPU threads each of `parallel` jobs may use."""
    return max(2, (os.cpu_count() or 2) // max(1, parallel))
