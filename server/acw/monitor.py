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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import alerts, db, health, jobs, metrics, retention, storage
from .billing import ledger
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
    for timer in ("acw-backup.timer", "acw-mirror.timer"):
        if (UNIT_DIR / "timers.target.wants" / timer).exists():
            names.append(timer)
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


def check_bulk(ok_fn: Callable[[str], tuple[bool, str]] = http_ok, unit_dir: Path | None = None) -> list[Finding]:
    """The bulk-transfer process (uploads and downloads, acw-bulk) answers on its own port. Only checked where
    that unit is enabled. Its tunnel is a unit like the others (service:acw-tunnel-bulk.service)."""
    unit_dir = UNIT_DIR if unit_dir is None else unit_dir
    if not (unit_dir / "default.target.wants" / "acw-bulk.service").exists():
        return []
    s = get_settings()
    url = f"http://127.0.0.1:{os.environ.get('ACW_BULK_PORT', '8794')}{s.base_path}/api/health"
    ok, why = ok_fn(url)
    return [Finding(
        "health_bulk", not ok, "Audio Cleaner upload/download process is not answering",
        f"{url} failed ({why}). acw-bulk is down or stuck: uploads and result downloads fall back to the main process "
        "once the VPS notices (a few seconds to a minute), or fail if the VPS is not set up for the fallback.\n"
        "Check: systemctl --user status acw-bulk; journalctl --user -u acw-bulk -n 80 --no-pager\n",
        confirm=2,
    )]


def check_mirror(unit_dir: Path | None = None, now: float | None = None) -> list[Finding]:
    """The release mirror (acw-mirror.timer, every 10 minutes) keeps the downloads current: no successful run for
    2 hours, or a newer release on the build service not served for an hour. Only checked where the timer is enabled."""
    unit_dir = UNIT_DIR if unit_dir is None else unit_dir
    if not (unit_dir / "timers.target.wants" / "acw-mirror.timer").exists():
        return []
    from . import mirror

    problems = mirror.stale_problems(now)
    return [Finding(
        "mirror_stale", bool(problems), "Release downloads are out of date",
        "\n".join(problems) + "\nWhat people can download now is the last good version (nothing wrong is ever served); "
        "new releases are not reaching the website and the apps.\n"
        "Check: python -m acw.admin mirror --status; journalctl --user -u acw-mirror -n 60 --no-pager; "
        "docs/operations.md, \"Release downloads\".\n",
        confirm=2,
    )]


def check_queue(ctx: Callable[[], str]) -> list[Finding]:
    ids = jobs.queued_ids()
    now = db.now()
    head_wait = est = 0
    if ids:
        row = db.one("SELECT queued_at FROM jobs WHERE id = ?", (ids[0],))
        head_wait = max(0, now - (row["queued_at"] or now)) if row else 0
        est = max(jobs.queue_snapshot()[1].values(), default=0)
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


def newest_backup(directory: Path = BACKUP_DIR) -> float | None:
    """Modification time of the newest snapshot, None when there is none (or the disk is not mounted)."""
    try:
        return max((p.stat().st_mtime for p in directory.glob("acw-*.sqlite3*")), default=None)
    except OSError:
        return None


def check_backup(directory: Path = BACKUP_DIR, now: float | None = None) -> list[Finding]:
    now = time.time() if now is None else now
    newest = newest_backup(directory)
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


def check_backup_result(directory: Path = BACKUP_DIR) -> list[Finding]:
    """The last snapshot was written, is not empty, passed SQLite's integrity check and its ledger sums."""
    try:
        status = json.loads((directory / "backup-status.json").read_text())
    except (OSError, ValueError):
        return [Finding("backup_bad", False, "Database backup failed its check")]  # no status yet: staleness covers it
    problems = []
    if not status.get("bytes"):
        problems.append("the snapshot is empty")
    if status.get("integrity") != "ok":
        problems.append(f"SQLite integrity check: {status.get('integrity')}")
    if status.get("ledger") != "ok":
        problems.append(f"ledger check: {status.get('ledger')}")
    if status.get("error"):
        problems.append(str(status["error"])[:200])
    bad = bool(problems)
    return [Finding(
        "backup_bad", bad, "Database backup failed its check",
        "The newest snapshot (" + str(status.get("file")) + ") is not good: " + "; ".join(problems) + ".\n"
        "Check: journalctl --user -u acw-backup -n 40 --no-pager; python -m acw.admin check-money\n",
    )]


def check_storage(data_dir: Path, backup_dir: Path = BACKUP_DIR) -> list[Finding]:
    """The data directory (without job files) and the backup directory stay small; a leak shows up here early."""
    s = storage.sizes(data_dir, backup_dir)
    d_lim, b_lim = storage.data_limit(), storage.backups_limit()
    return [
        Finding("data_large", s["data"] > d_lim, f"The data directory is larger than expected ({_fmt_gb(s['data'])})",
                f"{data_dir} without jobs/ holds {_fmt_gb(s['data'])} (alert above {_fmt_gb(d_lim)}; the database file and its log are "
                f"{_fmt_gb(s['database'])}).\nSomething that should be bounded is growing: python -m acw.admin storage; "
                "docs/operations.md, \"What grows and what bounds it\". ACW_DATA_ALERT_GB changes the limit.\n"),
        Finding("backups_large", s["backups"] > b_lim, f"The backup directory is larger than expected ({_fmt_gb(s['backups'])})",
                f"{backup_dir} holds {_fmt_gb(s['backups'])} (alert above {_fmt_gb(b_lim)}). Thirty nightly snapshots should be far smaller "
                "than that: check the database size and for stray files in the folder (python -m acw.admin storage). "
                "ACW_BACKUPS_ALERT_GB changes the limit.\n"),
    ]


def check_old_money(now: int) -> list[Finding]:
    """Payment and ledger records are kept 7 years and never deleted by the program; this reminds the owner when
    the first ones are older, so the yearly clean-up (acw.admin compact-money) gets done."""
    old = retention.old_money_records(now - 30 * 86400)  # a month of grace after the 7 years
    n = sum(old.values())
    return [Finding("old_money_records", n > 0, "Payment records older than 7 years can be cleaned up",
                    f"{old['ledger']} ledger rows, {old['payments']} payments and {old['audit_log']} audit rows are older than 7 years.\n"
                    "Nothing is deleted automatically. When you are ready: python -m acw.admin compact-money (shows what it would do), "
                    "then compact-money --apply. It keeps every balance correct. See docs/operations.md.\n", confirm=1)]


def check_stuck_states(now: int) -> list[Finding]:
    out: list[Finding] = []
    n = db.one("SELECT COUNT(*) AS n FROM jobs WHERE status = 'cloud_asr' AND COALESCE(heartbeat_at, queued_at, created_at) < ?", (now - 1800,))["n"]
    out.append(Finding("cloud_asr_stuck", n > 0, "Extra-accurate transcription is stuck",
                       f"{n} job(s) have been waiting on the OpenAI transcription for over 30 minutes (outage, quota or a bug).\n"
                       "Check: journalctl --user -u acw-web | grep -i 'cloud ASR'\n"))
    up = db.one("SELECT COUNT(*) AS n, COALESCE(SUM(input_bytes), 0) AS b FROM jobs WHERE status = 'uploading'")
    out.append(Finding("uploads_piling", up["n"] > 15 or up["b"] > 40 * GB, "Many uploads in progress",
                       f"{up['n']} uploads are in progress ({_fmt_gb(up['b'])} promised). Disk space and the home link are shared with live audio.\n"))
    holds = db.one("SELECT COUNT(*) AS n FROM holds WHERE status = 'active' AND ref_type != 'refund' AND created_at < ?", (now - 6 * 3600,))["n"]
    out.append(Finding("holds_old", holds > 0, "Credit has been held for hours",
                       f"{holds} hold(s) have been active for over 6 hours (a job or live session that is over should have released them).\n"
                       "They are released by the housekeeping sweep; if this persists, look at acw-web logs.\n"))
    rows = db.one(
        "SELECT SUM(status = 'failed') AS failed, SUM(status IN ('done', 'expired')) AS ok FROM jobs WHERE finished_at > ?", (now - 86400,)
    )
    failed, ok = int(rows["failed"] or 0), int(rows["ok"] or 0)
    out.append(Finding("jobs_failing", failed + ok >= 10 and failed / (failed + ok) > 0.3, "Many file jobs are failing",
                       f"{failed} of the {failed + ok} file jobs that ended in the last 24 hours failed.\n"
                       "Check: journalctl --user -u 'acw-worker@*' --since '-1d' | grep -i 'died\\|crashed'\n"))
    ev = db.one("SELECT COUNT(*) AS n FROM stripe_events WHERE processed_at IS NULL AND received_at < ?", (now - 600,))["n"]
    out.append(Finding("stripe_events_unprocessed", ev > 0, "Stripe events were received but not processed",
                       f"{ev} Stripe event(s) older than 10 minutes were stored without being handled: a payment may not be credited.\n"
                       "Check: journalctl --user -u acw-web | grep -i webhook\n"))
    pend = db.one(
        "SELECT COUNT(*) AS n FROM payments WHERE status = 'pending' AND ((kind = 'auto_reload' AND created_at < ?) OR created_at < ?)",
        (now - 3600, now - 26 * 3600),
    )["n"]
    out.append(Finding("payments_pending", pend > 0, "Payments are stuck as pending",
                       f"{pend} payment(s) are still pending long after they should have settled (auto top-ups: 1 hour, checkouts: 26 hours).\n"))
    return out


def check_money() -> list[Finding]:
    p = ledger.invariant_problems()
    bad = any(p.values())
    return [Finding("ledger_invariant", bad, "Balances do not match the ledger",
                    f"{p['balance_mismatch']} account(s) have a balance different from the sum of their ledger entries and "
                    f"{p['held_mismatch']} have a held amount different from their active holds.\n"
                    "Check: python -m acw.admin check-money; do not restart anything until you have looked.\n", confirm=1)]


def check_installation(now: int) -> list[Finding]:
    out: list[Finding] = []
    missing = health.missing_files()
    out.append(Finding("files_missing", bool(missing), "Model or program files are missing",
                       "Missing under ~/audio-cleaner: " + ", ".join(missing[:12]) + ".\nJobs that need them fail.\n"))
    out.append(Finding("stream_socket", not health.stream_worker_ok(), "The live audio worker does not answer",
                       "Connecting to the live worker's socket failed: live sessions cannot start.\n"
                       "Check: systemctl --user status acw-stream; journalctl --user -u acw-stream -n 60 --no-pager\n", confirm=2))
    age = health.gpu_worker_age()
    running = db.one("SELECT COUNT(*) AS n FROM jobs WHERE status = 'running'")["n"]
    states = {}
    try:
        states = unit_states(["acw-worker-gpu.service"])
    except Exception:  # noqa: BLE001
        pass
    gpu_unit_active = states.get("acw-worker-gpu.service", {}).get("ActiveState") == "active"
    out.append(Finding("gpu_worker_silent", gpu_unit_active and age is not None and age > 45 * 60 and running == 0,
                       "The GPU transcription worker is not reporting in",
                       "The GPU worker is running but has not marked itself idle for over 45 minutes while nothing else is running; "
                       "CPU workers are taking the transcription jobs it should.\nCheck: journalctl --user -u acw-worker-gpu -n 60 --no-pager\n"))
    return out


# --- metrics file -----------------------------------------------------------

def metrics_path() -> Path:
    return get_settings().data_dir / "metrics.prom"


def write_metrics(findings: list[Finding], states: dict[str, dict[str, str]], now: int | None = None,
                  path: Path | None = None, backup_dir: Path | None = None) -> Path:
    """data/metrics.prom in Prometheus text format, replaced atomically on every run. A file for a collector
    (for example node_exporter's textfile directory) to read: there is no HTTP endpoint."""
    settings = get_settings()
    now = db.now() if now is None else now
    path = metrics_path() if path is None else path
    prom = metrics.Prom()
    metrics.collect(prom, now=now, data_dir=settings.data_dir)
    for unit, st in sorted(states.items()):
        prom.add("acw_unit_active", "gauge", "1 when the systemd user unit is active.", 1 if st.get("ActiveState") == "active" else 0, unit=unit)
    try:
        prom.add("acw_disk_free_bytes", "gauge", "Free bytes on the volume holding the data directory.", shutil.disk_usage(settings.data_dir).free)
    except OSError:
        pass
    newest = newest_backup(BACKUP_DIR if backup_dir is None else backup_dir)
    try:
        sz = storage.sizes(settings.data_dir, BACKUP_DIR if backup_dir is None else backup_dir)
        for what, n in sz.items():
            prom.add("acw_storage_bytes", "gauge", "Bytes used by the data directory (without jobs), the database, job files and the backups.", n, what=what)
    except Exception:  # noqa: BLE001
        log.exception("could not size the data directories")
    try:
        from . import downloads, mirror

        for ch, c in sorted((mirror.load_status().get("channels") or {}).items()):
            if isinstance(c.get("last_ok"), (int, float)):
                prom.add("acw_mirror_last_ok_timestamp_seconds", "gauge", "When the release mirror last finished a channel without a problem.", int(c["last_ok"]), channel=ch)
        for ch in downloads.CHANNELS:
            if downloads.root().joinpath(ch).is_dir():
                prom.add("acw_downloads_bytes", "gauge", "Bytes of release files kept for download.",
                         sum(p.stat().st_size for p in downloads.channel_dir(ch).rglob("*") if p.is_file()), channel=ch)
    except Exception:  # noqa: BLE001
        log.exception("could not read the mirror state")
    prom.add("acw_backup_age_seconds", "gauge", "Age of the newest database snapshot (absent when there is none).",
             None if newest is None else max(0, int(time.time() - newest)))
    for f in findings:
        prom.add("acw_check_problem", "gauge", "1 when the monitor check of this kind is failing.", 1 if f.bad else 0, kind=f.kind)
    prom.add("acw_metrics_written_timestamp_seconds", "gauge", "When this file was written.", now)
    metrics.write_prom(path, prom)
    return path


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

    unit_state: dict[str, dict[str, str]] = {}

    def services() -> list[Finding]:
        states = unit_states(expected_units())
        unit_state.update(states)
        f, hist = check_services(states, _state_get("restarts", {}), now)
        _state_set("restarts", hist)
        return f

    def resources() -> list[Finding]:
        f, hist = check_resources(settings.data_dir, meminfo(), _state_get("swap", []), now, host_ctx)
        _state_set("swap", hist)
        return f

    guarded("services", services)
    guarded("health", lambda: check_health(http))
    guarded("bulk", lambda: check_bulk(http))
    guarded("mirror", check_mirror)
    guarded("queue", lambda: check_queue(host_ctx))
    guarded("slow_jobs", lambda: check_slow_jobs(host_ctx))
    guarded("resources", resources)
    guarded("live_full", check_full)
    guarded("backup", check_backup)
    guarded("backup_result", check_backup_result)
    guarded("stuck_states", lambda: check_stuck_states(now))
    guarded("money", check_money)
    guarded("storage", lambda: check_storage(settings.data_dir))
    guarded("old_money", lambda: check_old_money(now))
    guarded("installation", lambda: check_installation(now))

    for f in findings:
        try:
            alerts.observe(f.kind, f.bad, f.subject, f.body, confirm=f.confirm, clear=f.clear)
        except Exception:  # noqa: BLE001
            log.exception("could not record %s", f.kind)
    try:
        alerts.expire_stale("openai:", 30 * 60)
    except Exception:  # noqa: BLE001
        log.exception("openai alert expiry failed")
    try:
        write_metrics(findings, unit_state, now)
    except Exception:  # noqa: BLE001
        log.exception("could not write the metrics file")
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
