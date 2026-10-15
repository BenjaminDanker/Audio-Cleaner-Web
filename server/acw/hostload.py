"""What the host can spare right now, measured rather than assumed.

The machine also runs other heavy services (game servers), so queued work
starts when the CPU, memory and disk actually have room for it: never on fixed
session counts. Readings come from /proc and from the cgroup of the systemd
slice the services run in, averaged over a few seconds so one busy moment
doesn't decide anything.

CPU is split by priority: file jobs run at `Nice=10`, so time spent in niced
processes is CPU that live audio (normal priority) can take back at once.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

GB = 1024**3
_SAMPLE_S = 1.0
_ALPHA = 0.2  # EWMA weight per sample: ~10 s memory


@dataclass(frozen=True)
class Snapshot:
    cores: int
    busy_cores: float  # all non-idle CPU, in cores
    nice_cores: float  # of which: niced processes (our file jobs, ...)
    mem_available: int  # bytes the kernel can hand out without swapping
    slice_headroom: int | None  # bytes left under our slice's MemoryMax (None: no limit / not in a slice)
    disk_free: int  # bytes free on the data volume
    busy_per_cpu: tuple[float, ...] = ()  # 0..1 per logical CPU
    own_cores: float = 0.0  # CPU this process itself uses, in cores

    @property
    def idle_cores(self) -> float:
        return max(0.0, self.cores - self.busy_cores)

    @property
    def cores_for_live(self) -> float:
        """Cores live audio could use now: idle plus what niced work would yield."""
        return max(0.0, self.cores - (self.busy_cores - self.nice_cores))

    @property
    def memory_headroom(self) -> int:
        if self.slice_headroom is None:
            return self.mem_available
        return min(self.mem_available, self.slice_headroom)


def _read(fd: int) -> str:
    """Re-read a /proc or cgroup file through an fd opened at start-up (the
    stream worker can't open these once it has sandboxed itself)."""
    return os.pread(fd, 65536, 0).decode()


def _parse(line: str) -> tuple[int, int, int]:
    v = [int(x) for x in line.split()[1:]]
    user, nice, system, idle, iowait, irq, softirq, steal = (v + [0] * 8)[:8]
    return user + nice + system + idle + iowait + irq + softirq + steal, idle + iowait, nice


def _cpu_times(fd: int) -> tuple[tuple[int, int, int], list[tuple[int, int, int]]]:
    """(total, idle, nice) jiffies since boot: for the whole machine and per CPU."""
    lines = _read(fd).splitlines()
    per = [_parse(line) for line in lines[1:] if line.startswith("cpu") and line[3:4].isdigit()]
    return _parse(lines[0]), per


def _mem_available(fd: int) -> int:
    for line in _read(fd).splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def _slice_dir() -> Path | None:
    """cgroup v2 directory of the slice holding this process's service."""
    try:
        with open("/proc/self/cgroup") as f:
            rel = next(line.split("::", 1)[1].strip() for line in f if line.startswith("0::"))
    except (OSError, StopIteration):
        return None
    d = Path("/sys/fs/cgroup") / rel.lstrip("/")
    while d != Path("/sys/fs/cgroup") and not d.name.endswith(".slice"):
        d = d.parent
    return d if d.name.endswith(".slice") and d.name != "user.slice" else None


def _inactive_file(fd: int | None) -> int:
    """Bytes of reclaimable file cache in the slice (memory.stat): counted in memory.current, but the kernel
    gives them back at once, so they are not "used" for admission."""
    if fd is None:
        return 0
    try:
        for line in _read(fd).splitlines():
            if line.startswith("inactive_file "):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 0


def _slice_headroom(fds: tuple[int, int, int | None] | None) -> int | None:
    if fds is None:
        return None
    try:
        limit = _read(fds[0]).strip()
        if limit == "max":
            return None
        used = max(0, int(_read(fds[1])) - _inactive_file(fds[2]))
        return max(0, int(limit) - used)
    except (OSError, ValueError):
        return None


class HostLoad:
    def __init__(self, data_dir: Path):
        self._stat = os.open("/proc/stat", os.O_RDONLY)
        self._meminfo = os.open("/proc/meminfo", os.O_RDONLY)
        self._data = os.open(data_dir, os.O_RDONLY | os.O_DIRECTORY)
        d = _slice_dir()
        try:
            self._slice = None
            if d:
                try:
                    stat_fd = os.open(d / "memory.stat", os.O_RDONLY)
                except OSError:
                    stat_fd = None
                self._slice = (os.open(d / "memory.max", os.O_RDONLY), os.open(d / "memory.current", os.O_RDONLY), stat_fd)
        except OSError:
            self._slice = None
        self._cores = os.cpu_count() or 1
        self._lock = threading.Lock()
        self._prev = _cpu_times(self._stat)
        self._prev_own = (time.monotonic(), sum(os.times()[:2]))
        self._busy = self._nice = None
        self._per: list[float] = []
        self._own = 0.0
        self._thread = threading.Thread(target=self._run, name="hostload", daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        (cur, cur_per), (prev, prev_per) = _cpu_times(self._stat), self._prev
        now, own = time.monotonic(), sum(os.times()[:2])
        dt = cur[0] - prev[0]
        if dt > 0:
            busy = self._cores * (1 - (cur[1] - prev[1]) / dt)
            nice = self._cores * (cur[2] - prev[2]) / dt
            per = [1 - (c[1] - p[1]) / (c[0] - p[0]) if c[0] > p[0] else 0.0 for c, p in zip(cur_per, prev_per)]
            own_cores = (own - self._prev_own[1]) / max(1e-6, now - self._prev_own[0])
            with self._lock:
                if self._busy is None:
                    self._busy, self._nice, self._per, self._own = busy, nice, per, own_cores
                else:
                    self._busy += _ALPHA * (busy - self._busy)
                    self._nice += _ALPHA * (nice - self._nice)
                    self._per = [a + _ALPHA * (b - a) for a, b in zip(self._per, per)] if self._per else per
                    self._own += _ALPHA * (own_cores - self._own)
        self._prev = (cur, cur_per)
        self._prev_own = (now, own)

    def _run(self) -> None:
        while True:
            time.sleep(_SAMPLE_S)
            try:
                self._sample()
            except OSError:
                pass

    def snapshot(self) -> Snapshot:
        with self._lock:
            busy = self._busy
        if busy is None:  # first second after start: take a short direct reading
            time.sleep(0.25)
            self._sample()
        with self._lock:
            busy, nice, per, own = self._busy or 0.0, self._nice or 0.0, tuple(self._per), self._own
        try:
            st = os.fstatvfs(self._data)
            disk = st.f_bavail * st.f_frsize
        except OSError:
            disk = 0
        return Snapshot(
            cores=self._cores,
            busy_cores=busy,
            nice_cores=min(nice, busy),
            mem_available=_mem_available(self._meminfo),
            slice_headroom=_slice_headroom(self._slice),
            disk_free=disk,
            busy_per_cpu=per,
            own_cores=own,
        )


_instance: HostLoad | None = None
_instance_lock = threading.Lock()


def get(data_dir: Path) -> HostLoad:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = HostLoad(data_dir)
        return _instance
