"""Health monitor: `python -m acw.monitor` (a oneshot run every 2 minutes by acw-monitor.timer).

Reads the database and system state only; the only network it touches is
loopback HTTP, one HTTPS GET to our own public health URL, and SMTP (through
acw.alerts). Each check yields a Finding; acw.alerts.observe turns findings
into de-duplicated mails and "resolved" mails. One failing check never stops
the others. See docs/operations.md#alerts for the list.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import alerts, db, jobs
from .config import get_settings

log = logging.getLogger("acw.monitor")

GB = 1024**3
QUEUE_HEAD_WAIT_S = 20 * 60
QUEUE_ESTIMATE_S = 45 * 60
SLOW_FACTOR = 3.0
SLOW_MIN_ELAPSED_S = 10 * 60
DISK_MIN = 15 * GB
MEM_MIN = 2 * GB
SWAP_MIN_USED = 512 * 1024**2
SWAP_GROWTH = 256 * 1024**2
SWAP_WINDOW_S = 30 * 60
RESTART_WINDOW_S = 30 * 60
RESTART_ALERT_AT = 2
FULL_PER_HOUR = int(os.environ.get("ACW_ALERT_FULL_THRESHOLD", "10"))
BACKUP_MAX_AGE_S = 36 * 3600
BACKUP_DIR = Path(os.environ.get("ACW_BACKUP_DIR", "/mnt/mc-storage/audio-cleaner-backups"))
UNIT_DIR = Path.home() / ".config/systemd/user"


@dataclass
class Finding:
    kind: str
    bad: bool
    subject: str
    body: str = ""
    confirm: int = 1  # consecutive bad runs before mailing
    clear: int = 2  # consecutive good runs before "resolved"


# --- system facts -----------------------------------------------------------

def meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, v = line.partition(":")
            out[k] = int(v.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return out


def _fmt_gb(n: float) -> str:
    return f"{n / GB:.1f} GB"


def acw_slice_dir() -> Path | None:
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                d = Path("/sys/fs/cgroup") / line.split("::", 1)[1].strip().lstrip("/")
                while d != Path("/sys/fs/cgroup"):
                    if d.name == "acw.slice":
                        return d
                    d = d.parent
    except OSError:
        pass
    uid = os.getuid()
    d = Path(f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/acw.slice")
    return d if d.is_dir() else None


def _slice_usec(d: Path | None) -> int | None:
    try:
        for line in (d / "cpu.stat").read_text().splitlines():  # type: ignore[operator]
            if line.startswith("usage_usec"):
                return int(line.split()[1])
    except (OSError, TypeError, ValueError):
        pass
    return None


def _proc_cpu() -> dict[int, tuple[str, int]]:
    out: dict[int, tuple[str, int]] = {}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            s = (p / "stat").read_text()
            comm = s[s.index("(") + 1 : s.rindex(")")]
            f = s[s.rindex(")") + 2 :].split()
            out[int(p.name)] = (comm, int(f[11]) + int(f[12]))  # utime + stime
        except (OSError, ValueError, IndexError):
            continue
    return out


def _total_busy_jiffies() -> tuple[int, int]:
    v = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    total = sum(v[:8])
    return total, total - v[3] - (v[4] if len(v) > 4 else 0)


def describe_host(sample_s: float = 2.0) -> str:
    """How loaded the machine is, split into Audio Cleaner vs everything else
    (so the owner can see when the game servers are what slows the queue)."""
    cores = os.cpu_count() or 1
    try:
        slice_dir = acw_slice_dir()
        t0, b0 = _total_busy_jiffies()
        u0, p0, w0 = _slice_usec(slice_dir), _proc_cpu(), time.monotonic()
        time.sleep(sample_s)
        t1, b1 = _total_busy_jiffies()
        u1, p1, w1 = _slice_usec(slice_dir), _proc_cpu(), time.monotonic()
        busy = cores * (b1 - b0) / (t1 - t0) if t1 > t0 else 0.0
        ours = (u1 - u0) / 1e6 / max(1e-6, w1 - w0) if u0 is not None and u1 is not None else None
        other = max(0.0, busy - ours) if ours is not None else busy
        hz = os.sysconf("SC_CLK_TCK")
        per: dict[str, float] = {}
        for pid, (comm, j1) in p1.items():
            if pid in p0 and j1 > p0[pid][1]:
                try:
                    if "acw.slice" in Path(f"/proc/{pid}/cgroup").read_text():
                        continue
                except OSError:
                    continue
                per[comm] = per.get(comm, 0.0) + (j1 - p0[pid][1]) / hz / max(1e-6, w1 - w0)
        top = ", ".join(f"{name} {c:.1f}" for name, c in sorted(per.items(), key=lambda kv: -kv[1])[:4] if c >= 0.2)
        lines = [
            f"CPU: {cores} logical cores, {busy:.1f} busy in total.",
            f"  Audio Cleaner services: {ours:.1f} cores" if ours is not None else "  Audio Cleaner services: unknown",
            f"  Everything else on the machine: {other:.1f} cores" + (f" (busiest: {top})" if top else ""),
        ]
    except Exception as e:  # noqa: BLE001
        lines = [f"CPU: could not be measured ({type(e).__name__})."]
    m = meminfo()
    if m:
        swap = m.get("SwapTotal", 0) - m.get("SwapFree", 0)
        lines.append(f"Memory available: {_fmt_gb(m.get('MemAvailable', 0))} of {_fmt_gb(m.get('MemTotal', 0))}; swap in use {_fmt_gb(swap)}.")
    return "\n".join(lines)


def _state_get(name: str, default):
    row = db.one("SELECT value FROM meta WHERE key = ?", (f"monitor.{name}",))
    try:
        return json.loads(row["value"]) if row else default
    except ValueError:
        return default


def _state_set(name: str, value) -> None:
    db.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (f"monitor.{name}", json.dumps(value)),
    )


# --- checks -----------------------------------------------------------------

def expected_units() -> list[str]:
    names = sorted(p.name for p in (UNIT_DIR / "default.target.wants").glob("acw-*.service"))
    if (UNIT_DIR / "timers.target.wants" / "acw-backup.timer").exists():
        names.append("acw-backup.timer")
    return names


def unit_states(units: list[str]) -> dict[str, dict[str, str]]:
    if not units:
        return {}
    out = subprocess.run(
        ["systemctl", "--user", "show", *units, "-p", "Id", "-p", "ActiveState", "-p", "SubState", "-p", "Result", "-p", "NRestarts"],
        capture_output=True, text=True, timeout=20, check=True,
    ).stdout
    states: dict[str, dict[str, str]] = {}
    for block in out.strip().split("\n\n"):
        d = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
        if "Id" in d:
            states[d["Id"]] = d
    return states


def check_services(states: dict[str, dict[str, str]], prev: dict, now: int) -> tuple[list[Finding], dict]:
    """Not running, or restarted by systemd repeatedly. `prev` is last run's NRestarts history."""
    findings: list[Finding] = []
    hist: dict[str, dict] = {}
    for unit, st in sorted(states.items()):
        active = st.get("ActiveState") == "active"
        findings.append(Finding(
            f"service:{unit}", not active, f"{unit} is not running",
            f"{unit} is {st.get('ActiveState')} ({st.get('SubState')}), last result: {st.get('Result')}.\n"
            f"Check: systemctl --user status {unit}; journalctl --user -u {unit} -n 80 --no-pager\n",
            confirm=2,
        ))
        n = int(st.get("NRestarts") or 0)
        p = prev.get(unit) or {}
        events = [t for t in p.get("events", []) if t > now - RESTART_WINDOW_S]
        if p and n > p.get("n", 0):
            events += [now] * min(20, n - p["n"])
        hist[unit] = {"n": n, "events": events[-20:]}
        findings.append(Finding(
            f"restarts:{unit}", len(events) >= RESTART_ALERT_AT, f"{unit} keeps restarting",
            f"systemd restarted {unit} {len(events)} times in the last {RESTART_WINDOW_S // 60} minutes "
            "(it crashes or exits and is brought back automatically).\n"
            f"Check: journalctl --user -u {unit} --since '-1h' --no-pager\n",
        ))
    return findings, hist


def http_ok(url: str, timeout: float = 10.0) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "acw-monitor"})
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            body = r.read(2048)
            if r.status == 200 and b'"ok"' in body:
                return True, "ok"
            return False, f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001
        return False, type(e).__name__


def check_health(ok_fn: Callable[[str], tuple[bool, str]] = http_ok) -> list[Finding]:
    s = get_settings()
    port = os.environ.get("ACW_PORT", "8790")
    local_url = f"http://127.0.0.1:{port}{s.base_path}/api/health"
    ok, why = ok_fn(local_url)
    findings = [Finding(
        "health_local", not ok, "Audio Cleaner API is not answering locally",
        f"{local_url} failed ({why}). acw-web is down or stuck; users cannot reach the service at all.\n"
        "Check: systemctl --user status acw-web; journalctl --user -u acw-web -n 80 --no-pager\n",
        confirm=2,
    )]
    if ok and s.public_origin.startswith("https://"):
        url = s.url("/api/health")
        pub_ok, pub_why = ok_fn(url)
        findings.append(Finding(
            "health_public", not pub_ok, "Audio Cleaner is not reachable from the internet",
            f"The local API is healthy, but {url} failed from this machine ({pub_why}).\n"
            "The problem is on the public path: the acw-tunnel SSH tunnel, the VPS, or its Caddy route.\n"
            "Check: systemctl --user status acw-tunnel; ssh vps 'curl -s 127.0.0.1:18790/audio/api/health'\n",
            confirm=2,
        ))
    return findings


def check_queue(ctx: Callable[[], str]) -> list[Finding]:
    ids = jobs.queued_ids()
    now = db.now()
    head_wait = est = 0
    if ids:
        row = db.one("SELECT queued_at FROM jobs WHERE id = ?", (ids[0],))
        head_wait = max(0, now - (row["queued_at"] or now)) if row else 0
        est = jobs.queue_wait_seconds(ids[:-1])
    bad = bool(ids) and (head_wait > QUEUE_HEAD_WAIT_S or est > QUEUE_ESTIMATE_S)
    body = ""
    if bad:
        running = db.one("SELECT COUNT(*) AS n FROM jobs WHERE status = 'running'")["n"]
        body = (
            f"The file queue is backing up: {len(ids)} job(s) waiting, {running} running, "
            f"{get_settings().workers} CPU workers configured.\n"
            f"The job at the head of the queue (id {ids[0][:8]}) has waited {head_wait // 60} minutes; "
            f"the last job in line is estimated to wait {est // 60} minutes.\n\n"
            f"{ctx()}\n\n"
            "If 'everything else' is using most cores, the game servers (or something else) are what slows the queue.\n"
        )
    return [Finding("queue_long", bad, "File queue is long", body)]


def check_slow_jobs(ctx: Callable[[], str]) -> list[Finding]:
    now = db.now()
    slow: list[str] = []
    for r in db.all_rows("SELECT * FROM jobs WHERE status = 'running' AND started_at IS NOT NULL"):
        job = dict(r)
        elapsed = now - job["started_at"]
        est = jobs.job_seconds(job)
        frac = (job["progress"] or 0) / 100
        projected = elapsed / frac if frac >= 0.05 else None
        if elapsed >= SLOW_MIN_ELAPSED_S and ((projected or elapsed) > SLOW_FACTOR * est):
            slow.append(f"{job['id'][:8]}: running {elapsed // 60} min, progress {job['progress']}%, expected about {int(est // 60)} min total")
    body = ""
    if slow:
        body = (f"{len(slow)} running job(s) are more than {SLOW_FACTOR:.0f}x slower than expected:\n  "
                + "\n  ".join(slow[:10]) + f"\n\n{ctx()}\n")
    return [Finding("jobs_slow", bool(slow), "File jobs are running far slower than expected", body)]


def check_resources(data_dir: Path, mem: dict[str, int], swap_hist: list, now: int, ctx: Callable[[], str]) -> tuple[list[Finding], list]:
    findings: list[Finding] = []
    try:
        free = shutil.disk_usage(data_dir).free
    except OSError:
        free = 0
    findings.append(Finding(
        "disk_low", free < DISK_MIN, f"Disk space is low ({_fmt_gb(free)} free)",
        f"Only {_fmt_gb(free)} is free on the volume holding {data_dir} (alert below {_fmt_gb(DISK_MIN)}). "
        "New uploads are refused below the job reserve and running jobs can fail.\n"
        "Check: du -sh ~/audio-cleaner/data/jobs; df -h\n",
    ))
    avail = mem.get("MemAvailable", 0)
    findings.append(Finding(
        "mem_low", bool(mem) and avail < MEM_MIN, f"Memory is low ({_fmt_gb(avail)} available)",
        f"Only {_fmt_gb(avail)} of RAM is available (alert below {_fmt_gb(MEM_MIN)}).\n\n{ctx()}\n", confirm=2,
    ))
    used = max(0, mem.get("SwapTotal", 0) - mem.get("SwapFree", 0))
    swap_hist = [s for s in swap_hist if s[0] > now - SWAP_WINDOW_S] + [[now, used]]
    grew = used - swap_hist[0][1]
    findings.append(Finding(
        "swap_growing", used >= SWAP_MIN_USED and grew >= SWAP_GROWTH,
        f"Swap use is growing ({_fmt_gb(used)} in use)",
        f"Swap in use grew by {_fmt_gb(grew)} in the last {SWAP_WINDOW_S // 60} minutes and is now {_fmt_gb(used)}: "
        f"the machine is short of RAM.\n\n{ctx()}\n", confirm=2,
    ))
    return findings, swap_hist


def check_full() -> list[Finding]:
    n = db.one("SELECT COUNT(*) AS n FROM stream_sessions WHERE end_reason = 'full' AND started_at > ?", (db.now() - 3600,))["n"]
    return [Finding(
        "live_full", n > FULL_PER_HOUR, f"Live sessions refused for capacity ({n} in the last hour)",
        f"{n} live sessions were refused because live capacity was full in the last hour (alert above {FULL_PER_HOUR}).\n"
        "Live capacity follows the CPU cores the game servers leave free; see docs/operations.md (Capacity).\n",
    )]


def check_backup(directory: Path = BACKUP_DIR, now: float | None = None) -> list[Finding]:
    now = time.time() if now is None else now
    try:
        newest = max((p.stat().st_mtime for p in directory.glob("acw-*.sqlite3*")), default=None)
    except OSError:
        newest = None
    if newest is None:
        why = f"No database snapshot found in {directory} (is the backup disk mounted?)."
        bad = True
    else:
        age = now - newest
        bad = age > BACKUP_MAX_AGE_S
        why = f"The newest database snapshot in {directory} is {age / 3600:.0f} hours old (alert after {BACKUP_MAX_AGE_S // 3600} h)."
    return [Finding(
        "backup_stale", bad, "Database backup is overdue",
        why + "\nCheck: systemctl --user status acw-backup.timer acw-backup.service; journalctl --user -u acw-backup\n",
    )]


# --- driver -----------------------------------------------------------------

def run(*, ctx: Callable[[], str] | None = None, http: Callable[[str], tuple[bool, str]] = http_ok) -> list[Finding]:
    """One monitoring pass. Returns all findings (for tests); mails via acw.alerts."""
    settings = get_settings()
    now = db.now()
    cache: list[str] = []

    def host_ctx() -> str:
        if not cache:
            cache.append((ctx or describe_host)())
        return cache[0]

    findings: list[Finding] = []

    def guarded(name: str, fn: Callable[[], list[Finding]]) -> None:
        try:
            findings.extend(fn())
        except Exception:  # noqa: BLE001
            log.exception("check %s failed", name)

    def services() -> list[Finding]:
        states = unit_states(expected_units())
        f, hist = check_services(states, _state_get("restarts", {}), now)
        _state_set("restarts", hist)
        return f

    def resources() -> list[Finding]:
        f, hist = check_resources(settings.data_dir, meminfo(), _state_get("swap", []), now, host_ctx)
        _state_set("swap", hist)
        return f

    guarded("services", services)
    guarded("health", lambda: check_health(http))
    guarded("queue", lambda: check_queue(host_ctx))
    guarded("slow_jobs", lambda: check_slow_jobs(host_ctx))
    guarded("resources", resources)
    guarded("live_full", check_full)
    guarded("backup", check_backup)

    for f in findings:
        try:
            alerts.observe(f.kind, f.bad, f.subject, f.body, confirm=f.confirm, clear=f.clear)
        except Exception:  # noqa: BLE001
            log.exception("could not record %s", f.kind)
    try:
        alerts.expire_stale("openai:", 30 * 60)
    except Exception:  # noqa: BLE001
        log.exception("openai alert expiry failed")
    return findings


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    db.init_db()
    findings = run()
    bad = [f.kind for f in findings if f.bad]
    log.info("monitor: %d checks, problems: %s", len(findings), ", ".join(bad) or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
