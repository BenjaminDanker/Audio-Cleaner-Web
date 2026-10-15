"""How many files the machine can clean at once, and how politely the app shares the computer. No Qt in here."""
from __future__ import annotations

import ctypes
import functools
import os
import subprocess
import sys

# Peak memory of one cleaning job (measured: 0.5 GB for a normal file, with headroom), of the speech model when
# subtitles are on (measured 1.7 GB in all), of a job that rebuilds the voice (measured 2.6 GB in all) and of
# the extra track that echo removal works with.
CLEAN_JOB_GB = 1.5
SUBTITLE_EXTRA_GB = 1.0
REBUILD_JOB_GB = 2.6
ECHO_EXTRA_GB = 0.3
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
    if sys.platform == "darwin":  # SC_PHYS_PAGES does not exist there
        try:
            return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout) / 2**30
        except (OSError, ValueError, subprocess.SubprocessError):
            return 0.0
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
    except (ValueError, OSError, AttributeError):
        return 0.0


def job_memory_gb(subtitles: bool = False, rebuild_voice: bool = False, echo_track: bool = False) -> float:
    """Planned peak memory of one job with these options."""
    return ((REBUILD_JOB_GB if rebuild_voice else CLEAN_JOB_GB) + (SUBTITLE_EXTRA_GB if subtitles else 0.0)
            + (ECHO_EXTRA_GB if echo_track else 0.0))


def max_parallel(requested: int, subtitles: bool, *, cores: int | None = None, ram_gb: float | None = None,
                 rebuild_voice: bool = False, echo_track: bool = False) -> int:
    """Files to clean at once. `requested` is 1-4, or 0 for Auto (a third of the physical cores).
    Never more than the memory budget allows."""
    cores = physical_cores() if cores is None else cores
    ram_gb = total_ram_gb() if ram_gb is None else ram_gb
    wanted = requested or min(MAX_PARALLEL, max(1, cores // 3))
    per_job = job_memory_gb(subtitles, rebuild_voice, echo_track)
    by_ram = max(1, int(ram_gb * RAM_BUDGET // per_job)) if ram_gb else wanted
    return max(1, min(wanted, MAX_PARALLEL, by_ram))


def threads_per_job(parallel: int) -> int:
    """CPU threads each of `parallel` jobs may use; one core stays free for the window and the rest of the system."""
    return max(2, ((os.cpu_count() or 2) - 1) // max(1, parallel))


def lower_priority() -> None:
    """Run below normal priority, so a long batch does not make the rest of the computer sluggish."""
    try:
        if os.name == "nt":
            below_normal = 0x00004000
            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.SetPriorityClass(ctypes.c_void_p(kernel32.GetCurrentProcess()), below_normal)
        else:
            os.nice(5)
    except (OSError, AttributeError):
        pass


class SleepInhibitor:
    """Keeps the computer from going to sleep while files are being cleaned (a laptop that sleeps in the
    middle of a long file just stalls). Call acquire() when work starts and release() when it ends, from the
    same thread (Windows ties the request to the calling thread)."""

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._held = False

    @property
    def held(self) -> bool:
        return self._held

    def acquire(self) -> None:
        if self._held:
            return
        self._held = True
        try:
            if os.name == "nt":
                ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED)
            elif sys.platform == "darwin":
                # caffeinate stops by itself when this process ends
                self._proc = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())], stdin=subprocess.DEVNULL,
                                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, AttributeError):
            pass

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        try:
            if os.name == "nt":
                ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
            if self._proc is not None:
                self._proc.terminate()
                self._proc.wait(timeout=5)
        except (OSError, AttributeError, subprocess.SubprocessError):
            pass
        self._proc = None
