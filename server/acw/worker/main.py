"""Job worker: `python -m acw.worker.main`.

Claims one queued job at a time and drives the sandboxed runner through its
stages. This process has database access but (under systemd) no network; the
runner it spawns has neither. Run several instances for parallelism.
"""
from __future__ import annotations

import json
import re
import logging
import math
import os
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import acw_common

from .. import automatic, broker, db, editing, hostload, jobs, keycaps, metrics, rerun
from ..billing import ledger, pricing
from ..config import get_settings

log = logging.getLogger("acw.worker")
_stop = threading.Event()


# This process is the GPU worker (acw-worker-gpu): it takes the jobs broker.gpu_stages() names, one at a time.
GPU_MODE = os.environ.get("ACW_WORKER_GPU") == "1"


def _gpu_spec() -> dict:
    """Sandbox additions the CUDA driver needs (found with strace under
    Landlock): the GPU device files, driver info under /proc and /sys, and
    this process's own /proc/self (it names its threads there, which it treats
    as fatal if refused). Nothing about other processes."""
    devices = sorted(str(p) for p in Path("/dev").glob("nvidia*") if not p.is_dir())
    read = [p for p in ("/proc/driver/nvidia", "/proc/devices", "/proc/sys/vm/mmap_min_addr", "/sys/bus/pci",
                        "/sys/devices", "/sys/module/nvidia", "/sys/module/nvidia_uvm", "/dev/char") if Path(p).exists()]
    return {
        "asr_device": "cuda",
        "clean_device": "cuda",
        "gpu_devices": [*devices, "/proc/self"],
        "gpu_read": [*read, "/proc/self"],
        "address_space_bytes": 0,
    }


def _cuda_library_path() -> str:
    """CUDA/cuDNN come from pip wheels inside the GPU venv (site-packages/nvidia/*/lib)."""
    import site

    dirs = []
    for sp in site.getsitepackages():
        dirs += sorted(str(p) for p in Path(sp, "nvidia").glob("*/lib") if p.is_dir())
    return os.pathsep.join(dirs)


def _settings_extra() -> dict:
    s = get_settings()
    return {
        "threads": int(os.environ.get("ACW_JOB_THREADS", "4")),
        "whisper_model": os.environ.get("ACW_WHISPER_MODEL", "large-v3-turbo"),
        "lib_path": str(s.bin_dir / "libdf.so"),
        **(_gpu_spec() if GPU_MODE else {}),
        "dfn_model": str(s.models_dir / "DeepFilterNet3_onnx.tar.gz"),
        "bin_dir": str(s.bin_dir),
        "models_dir": str(s.models_dir),
    }


# The probe reads the whole file once (a fast decode that measures its real length).
PROBE_CPU_SECONDS = 3600
PROBE_TIMEOUT = 900
# The measurement of an automatic job (automatic.py): a decode, about a minute of the file spread over it measured by the noise remover
# and the browser's measurements; minutes of CPU even for a file of hours, mostly the decode.
ANALYSE_TIMEOUT = 1800

MAX_JOB_THREADS = int(os.environ.get("ACW_JOB_THREADS_MAX", "6"))


def admission(
    job: dict, snap: hostload.Snapshot, running: list[dict], cpus: set[int] | None = None, gpu: bool = False
) -> tuple[bool, str, int]:
    """Whether `job` may start now, why not, and how many threads it gets: the resource broker decides."""
    return broker.admit_job(job, snap, running, cpus, gpu=gpu, max_threads=MAX_JOB_THREADS)


_last_wait_log = 0.0


def claim(worker_id: str, load: hostload.HostLoad | None = None) -> dict | None:
    """Take the next queued job (jobs.QUEUE_ORDER) if the host has room for it.
    The head of the queue waits rather than letting smaller jobs overtake it.

    The host is measured outside the database's write lock (a snapshot can sleep); the lock is only taken to
    check once more with fresh facts and to take the job."""
    global _last_wait_log
    now = db.now()
    rows = [dict(r) for r in db.all_rows(
        f"SELECT j.* FROM jobs j WHERE j.status = 'queued' ORDER BY {jobs.QUEUE_ORDER} LIMIT 50"
    )]
    if GPU_MODE:
        rows = [r for r in rows if broker.gpu_stages(r)]
    elif broker.gpu_worker_idle():
        rows = [r for r in rows if not broker.gpu_stages(r)]  # the GPU worker takes these
    else:
        rows = [r for r in rows if not broker.gpu_only(r)]  # the voice rebuild waits for the card
    if not rows:
        return None
    snap = cpus = None
    if load is not None:
        snap = load.snapshot()
        cpus = broker.job_cpus(get_settings().data_dir)

    def running_jobs(c) -> list[dict]:
        return [dict(r) for r in c.execute("SELECT * FROM jobs WHERE status = 'running'").fetchall()]

    def pick(running: list[dict]) -> tuple[dict | None, str, int | None]:
        row = rows[0]
        if snap is None:
            return row, "", None
        ok, why, threads = admission(row, snap, running, cpus, gpu=GPU_MODE)
        return (row, "", threads) if ok else (None, why, None)

    row, why, _ = pick(running_jobs(db.conn()))
    if row is None:
        if why and time.monotonic() - _last_wait_log > 60:
            _last_wait_log = time.monotonic()
            log.info("jobs wait (%s)", why)
        return None
    with db.tx() as c:
        row, why, threads = pick(running_jobs(c))  # same facts, current list of running jobs
        if row is None:
            return None
        n = c.execute(
            "UPDATE jobs SET status = 'running', worker_id = ?, heartbeat_at = ?, attempts = attempts + 1,"
            " started_at = COALESCE(started_at, ?),"
            " wait_seconds = COALESCE(wait_seconds, MAX(0, ? - COALESCE(queued_at, created_at)))"
            " WHERE id = ? AND status = 'queued'",
            (worker_id, now, now, now, row["id"]),
        ).rowcount
        if not n:
            return None  # another worker took it
        job = dict(c.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone())
        job["threads"] = threads
        return job


def _max_seconds(job: dict, params: dict) -> int:
    return pricing.max_job_seconds(params["denoise_tier"])


def _options_for_spec(job: dict, params: dict) -> dict:
    """The job's options as the runner reads them. A job whose subtitles are made per part (a subtitles layer in its timeline) has no
    whole-file language or translations of its own: the languages the parts translate into (all of them: the files are put together from
    every part) are what its caption files are written in."""
    spec = editing.spec_for(job, params)
    if not spec or "subtitles" not in spec:
        return params["options"]
    langs = list(dict.fromkeys(t for p in spec["subtitles"]["parts"] for t in p["translate"]))
    return {**params["options"], "translate": langs}


def _write_spec(job: dict, extra: dict, billed: float | None = None) -> Path:
    """Write the runner's instructions. `billed` is the measured length of the audio (known after the
    probe): the main stage never decodes more than that, and its CPU limit is sized from it and the tier."""
    d = jobs.job_dir(job["id"])
    params = json.loads(job["params_json"])
    max_seconds = _max_seconds(job, params)
    cpu_seconds = PROBE_CPU_SECONDS if billed is None else jobs.stage_limits(billed, max_seconds)[0]
    spec = {
        **extra,
        "job_id": job["id"],
        "kind": job["kind"],
        "denoise_tier": params["denoise_tier"],
        "transcribe_tier": params["transcribe_tier"],
        "input": f"input{params['input_ext']}",
        "input_ext": params["input_ext"],
        "output_stem": jobs.safe_stem(jobs.display_name(job)),
        "options": _options_for_spec(job, params),
        "max_seconds": max_seconds,
        "cpu_seconds": cpu_seconds,  # generous; the wall-clock timeout is the real limit
        "address_space_bytes": 32 * 1024**3,  # memory-mapped scratch files count here, not in RAM
        "max_file_bytes": 16 * 1024**3,
        **({"billed_seconds": billed} if billed is not None else {}),
        # a re-run starts from the copies its source job kept (jobs.py, rerun.py); a first run keeps its own (the noise removal
        # and, next to it, the recording as the remover heard it, for the strength slider: any length, stretch.py)
        **({"rerun": {"clean": params["rerun"]["clean"], "transcribe": params["rerun"]["transcribe"],
                      "reuse_audio": params["rerun"].get("reuse_audio"), "kept": params.get("kept") or {}}}
           if params.get("rerun") else {}),
        # the per-part editor (editing.py): what the run is, the timeline it makes, and what identifies the recording in the block cache
        **({"editor": ed} if (ed := editing.spec_for(job, params)) else {}),
    }
    (d / "spec.json").write_text(json.dumps(spec))
    return d


class Interrupted(Exception):
    """The worker is stopping (a restart or a deploy): the job goes back in the queue instead of failing."""


class StageFailed(Exception):
    def __init__(self, user_message: str, internal: str | None = None):
        super().__init__(internal or user_message)
        self.user_message = user_message


OOM_MESSAGE = "This file needs more memory than we can give one job, so it could not be processed. Splitting it into shorter parts may help; you were not charged."
INTERRUPTED_MESSAGE = "Processing was interrupted several times, so we stopped trying; you were not charged. If the file is very long, splitting it into shorter parts may help."
# A job put back in the queue by a stopping worker keeps its attempt, so a deploy never costs it a retry. But a worker that
# keeps stopping on the same job (for example systemd stopping the unit after the kernel killed the runner for memory)
# must not run it forever: the job is failed at its MAX_INTERRUPTIONS-th interruption.
MAX_INTERRUPTIONS = int(os.environ.get("ACW_MAX_INTERRUPTIONS", "3"))


def oom_kills() -> int | None:
    """How many processes the kernel's OOM killer has killed in this worker's cgroup (runners included) so far; None if unreadable."""
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                for ev in (Path("/sys/fs/cgroup") / line[3:].strip().lstrip("/") / "memory.events").read_text().splitlines():
                    key, _, value = ev.partition(" ")
                    if key == "oom_kill":
                        return int(value)
    except (OSError, ValueError):
        pass
    return None


def run_stage(job: dict, d: Path, stage: str, timeout: float) -> dict:
    env = {
        "PATH": "/usr/bin:/bin",
        # acw and acw_common share one directory when deployed, not in a checkout.
        "PYTHONPATH": os.pathsep.join(
            dict.fromkeys(str(Path(m.__file__).resolve().parents[i]) for m, i in ((sys.modules["acw"], 1), (acw_common, 1)))
        ),
        "LANG": "C.UTF-8",
        "HOME": str(d),
        "HF_HUB_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        # numpy's own thread pool: one thread. The heavy work sets its threads itself (ONNX Runtime, CTranslate2, libDF);
        # a pool sized to the machine only costs CPU, starting up in every stage.
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    if GPU_MODE:
        env["LD_LIBRARY_PATH"] = _cuda_library_path()
    result_file = d / f"result-{stage}.json"
    result_file.unlink(missing_ok=True)
    oom_before = oom_kills()
    proc = subprocess.Popen(
        [sys.executable, "-m", "acw.worker.runner", str(d), stage],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        env=env,
        cwd=str(d),
        start_new_session=True,
    )
    stderr_tail: list[bytes] = []

    def drain_stderr() -> None:
        for line in proc.stderr:  # type: ignore[union-attr]
            stderr_tail.append(line)
            del stderr_tail[:-40]

    threading.Thread(target=drain_stderr, daemon=True).start()
    started = time.monotonic()
    deadline = started + timeout
    last_update = 0.0
    prog = {"first": None, "now": 0, "at": started}  # the stage's reported progress: where it began, where it is, when it last moved
    killed_by_us = False

    data_dir = get_settings().data_dir
    pinned: set[int] | None = None

    def more_time() -> bool:
        """At the deadline: a main stage that is still making progress gets the time its progress says it needs
        (jobs.stretched_limit), so a job that ran slower than planned (live audio took its cores) is not thrown away."""
        nonlocal deadline
        if stage != "main" or _stop.is_set() or prog["first"] is None:
            return False
        now = time.monotonic()
        limit = jobs.stretched_limit(timeout, now - started, prog["first"], prog["now"], now - prog["at"])
        if limit is None or started + limit <= deadline:
            return False
        log.info("job %s stage %s still at %s%% after %.0f s: limit %.0f s -> %.0f s", job["id"], stage, prog["now"], now - started, deadline - started, limit)
        deadline = started + limit
        return True

    def watchdog() -> None:
        nonlocal pinned, killed_by_us
        beat = 0.0
        while proc.poll() is None:
            # Off the live-audio cores while live sessions run (resource broker).
            want = broker.job_cpus(data_dir)
            if want != pinned:
                broker.pin_tree(proc.pid, want)
                pinned = want
            if _stop.is_set() or (time.monotonic() > deadline and not more_time()):
                killed_by_us = True
                os.killpg(proc.pid, signal.SIGKILL)
                return
            if time.monotonic() - beat > 20:
                beat = time.monotonic()
                try:
                    db.execute("UPDATE jobs SET heartbeat_at = ? WHERE id = ?", (db.now(), job["id"]))
                    if GPU_MODE:
                        broker.mark_gpu_worker("alive")
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(1)

    threading.Thread(target=watchdog, daemon=True).start()
    for raw in proc.stdout:  # type: ignore[union-attr]
        try:
            msg = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(msg, dict):
            continue
        if msg.get("noise") in ("light", "heavy") and job.get("noise_profile") != msg["noise"]:
            # Which noise remover the recording got: the time estimate follows it.
            job["noise_profile"] = msg["noise"]
            jobs.update_params(job["id"], noise_profile=msg["noise"])
        now = time.monotonic()
        try:
            progress = max(0, min(100, int(msg.get("progress", 0))))
        except (TypeError, ValueError, OverflowError):
            progress = 0
        if prog["first"] is None:
            prog["first"] = prog["now"] = progress
        if progress > prog["now"]:
            prog["now"], prog["at"] = progress, now
        if now - last_update >= 1.0 or msg.get("progress") == 100:
            last_update = now
            db.execute(
                "UPDATE jobs SET progress = ?, stage = ?, heartbeat_at = ? WHERE id = ?",
                (progress, str(msg.get("stage", ""))[:40], db.now(), job["id"]),
            )
    proc.wait()
    if not result_file.exists() and not killed_by_us and proc.returncode == -signal.SIGKILL:
        # Killed from outside while we were not stopping it: the kernel's OOM killer when its counter moved. With
        # OOMPolicy=stop systemd then stops this whole worker (SIGTERM), which would read as a deploy and put the same
        # job back again and again.
        after = oom_kills()
        if oom_before is not None and after is not None and after > oom_before:
            log.error("job %s stage %s was killed for lack of memory", job["id"], stage)
            raise StageFailed(OOM_MESSAGE, "oom kill")
    if _stop.is_set() and not result_file.exists():
        raise Interrupted()
    if not result_file.exists():
        tail = b"".join(stderr_tail).decode(errors="replace")[-2000:]
        log.error("job %s stage %s died rc=%s: %s", job["id"], stage, proc.returncode, tail)
        if time.monotonic() > deadline:
            raise StageFailed("Processing took too long and was stopped; you were not charged.")
        raise StageFailed("Processing failed; you were not charged.", f"rc={proc.returncode}")
    res = _read_result(result_file)
    if not res.get("ok"):
        if res.get("user_error"):
            raise StageFailed(res["user_error"])
        log.error("job %s stage %s error: %s", job["id"], stage, res.get("error"))
        raise StageFailed("Processing failed; you were not charged.", res.get("error"))
    return res


RESULT_MAX_BYTES = 1_000_000
OUTPUT_KINDS = {"audio", "captions", "transcript", "segments", "video"}
TEXT_KINDS = ("captions", "transcript", "segments")  # the caption files, the transcript and the segments: what the subtitles make


def _read_result(path: Path) -> dict:
    """The runner's result file. The runner is sandboxed but is the part that handles untrusted input, so what it
    writes is checked: a regular file (never a link) of sane size, holding a JSON object of the expected shape."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as f:
            st = os.fstat(f.fileno())
            if not stat.S_ISREG(st.st_mode) or st.st_size > RESULT_MAX_BYTES:
                raise ValueError("bad result file")
            res = json.loads(f.read(RESULT_MAX_BYTES + 1))
    except (OSError, ValueError) as e:
        raise StageFailed("Processing failed; you were not charged.", f"unreadable result: {e}")
    if not isinstance(res, dict) or not isinstance(res.get("ok"), bool):
        raise StageFailed("Processing failed; you were not charged.", "malformed result")
    if res["ok"]:
        outs = res.get("outputs", [])
        if not isinstance(outs, list) or len(outs) > 50 or not all(
            isinstance(o, dict) and isinstance(o.get("file"), str) and isinstance(o.get("kind"), str) for o in outs
        ):
            raise StageFailed("Processing failed; you were not charged.", "malformed outputs")
        if res.get("kept") is not None and not isinstance(res["kept"], dict):
            raise StageFailed("Processing failed; you were not charged.", "malformed kept")
        if not isinstance(res.get("echo_skipped", False), bool):
            raise StageFailed("Processing failed; you were not charged.", "malformed echo_skipped")
        for key in ("notice", "noise", "language"):
            if res.get(key) is not None and not isinstance(res[key], str):
                raise StageFailed("Processing failed; you were not charged.", f"malformed {key}")
    elif res.get("user_error") is not None and not isinstance(res["user_error"], str):
        res["user_error"] = None
    return res


def _job_cost(job: dict, params: dict, seconds: float, billed: dict | None = None) -> int:
    """What the job costs for `seconds` of audio, at the prices it was created with. `billed`: what a job made with a timeline reports
    having run (editing.billable_from), else the whole timeline is charged. A video conversion counts when the picture is made again."""
    return _steps_cost(job, params, seconds, billed) + jobs.convert_cost(params, seconds)


def _steps_cost(job: dict, params: dict, seconds: float, billed: dict | None = None) -> int:
    if job["kind"] == "convert":
        return 0
    if (params.get("edit") or {}).get("first_run"):
        return editing.first_run_cost(job["kind"], params, seconds, billed, job["id"])
    # Translations: what was asked for until the job is done, then only the languages that were made.
    asked = params.get("options", {}).get("translate") or []
    made = params.get("translated", asked)
    if automatic.pending(params):
        # (not decided: it ends up charged for the most it can cost, which is what was held)
        return automatic.pending_cost(job["kind"], params, seconds)
    rates = dict(params.get("rates") or {})
    if params.get("echo_skipped"):
        rates.pop("echo", None)  # (a file with one audio track had no echo removal: it is not charged)
    if params.get("rerun"):
        # A re-run pays only for the steps it did again (its pinned rates are just those), and for the new languages it made.
        new = params["rerun"].get("translate_new") or []
        return pricing.rerun_cost_micros(rates, seconds, len([lang for lang in made if lang in new]))
    return pricing.job_cost_micros(job["kind"], params["denoise_tier"], params["transcribe_tier"], seconds, rates, len(made),
                                   "leveling" in rates, "echo" in rates)


# A job made with an API key must say how long its file is (an API client knows): the measured length may be this
# much longer, no more. (The website estimates the length of a file it can't read itself from its size, so its jobs
# are held to the measured price only.)
CLAIM_SLACK, CLAIM_SLACK_SECONDS = 1.1, 5.0


def _reprice(job: dict, duration: float) -> None:
    """Make the hold match the real duration before the expensive work starts: within the account's balance and, for a
    job made with an API key, within the key's monthly spending limit and the length it said the file has."""
    params = json.loads(job["params_json"])
    claimed = float(job.get("claimed_seconds") or 0)
    key_id = job.get("api_key_id")
    if key_id and duration > claimed * CLAIM_SLACK + CLAIM_SLACK_SECONDS:
        raise StageFailed(
            f"This file is {duration / 60:.1f} min long, but the job said {claimed / 60:.1f} min. "
            "Send its real length in durationSeconds."
        )
    db.execute("UPDATE jobs SET duration_seconds = ? WHERE id = ?", (duration, job["id"]))
    cost = _job_cost(job, params, duration)
    if cost > job["est_cost_micros"]:
        try:
            with db.tx() as c:
                keycaps.check(c, key_id, cost - job["est_cost_micros"])  # the key's limit holds for the real price too
                if job.get("hold_id"):
                    ledger.extend_hold(job["hold_id"], cost - job["est_cost_micros"], c=c)
                else:  # (a video conversion that was free until the file showed its picture has to be made again)
                    job["hold_id"] = ledger.create_hold(job["user_id"], cost, "job", job["id"], c=c)
                    c.execute("UPDATE jobs SET hold_id = ? WHERE id = ?", (job["hold_id"], job["id"]))
                c.execute("UPDATE jobs SET est_cost_micros = ? WHERE id = ?", (cost, job["id"]))
        except keycaps.KeyCapReached as e:
            raise StageFailed(str(e))
        except (ledger.InsufficientFunds, ledger.LedgerError):
            why = (" The picture of this video has to be made again for the file type or size you chose (Video conversion), more than"
                   " the video first said; MKV at the original size keeps it as it is, for free." if jobs.convert_cost(params, duration) else "")
            raise StageFailed(
                f"This file is {duration / 60:.1f} min and costs ${cost / 1e6:.2f}, more than your available balance.{why}"
            )
        job["est_cost_micros"] = cost


def _verified_outputs(d: Path, res: dict) -> list[dict]:
    """The result files the runner names that are regular files on the job's own device (a link it planted is not an output)."""
    outputs: list[dict] = []
    dev = os.stat(d).st_dev
    for o in res.get("outputs", []):
        name = o["file"]
        if "/" in name or name.startswith(".") or name in {x["file"] for x in outputs}:
            continue
        try:
            st = os.lstat(d / name)  # lstat: a link planted by the runner is not an output
        except OSError:
            continue
        if stat.S_ISREG(st.st_mode) and st.st_dev == dev and o["kind"] in OUTPUT_KINDS:
            lang = o.get("lang")
            outputs.append({"file": name, "kind": o["kind"], "bytes": st.st_size,
                            **({"lang": lang} if isinstance(lang, str) and re.fullmatch(r"[a-z]{2,3}", lang) else {})})
    return outputs


def _remove(d: Path, p: Path) -> None:
    if p.is_dir() and not p.is_symlink():
        import shutil

        shutil.rmtree(p, ignore_errors=True)
    else:
        p.unlink(missing_ok=True)


def _complete(job: dict, d: Path, res: dict) -> None:
    outputs = _verified_outputs(d, res)
    fresh = db.one("SELECT params_json FROM jobs WHERE id = ?", (job["id"],))
    params = json.loads(fresh["params_json"] if fresh is not None else job["params_json"])
    duration = float(job.get("duration_seconds") or job["claimed_seconds"])
    # What stays in the folder besides the results, until the job's deadline (jobs.expiry): the upload, the copies a later
    # "change and save again" starts from (rerun.py) and the per-part editor's (editing.py: the block cache, the recording,
    # the suggestions).
    now = db.now()
    if rerun.too_late(params, now) or now + rerun.MIN_DELIVERY_SECONDS >= jobs.hard_deadline(job):  # nobody could download these in time
        jobs.finish_failed(job, rerun.TOO_LATE)
        return
    report = params.pop("editor_reported", None) or editing.sanitize_report(res.get("editor"), d, duration)
    kept = rerun.collect_kept(params, d, res.get("kept") or params.get("kept_reported"))
    keep = ({o["file"] for o in outputs} | {f"input{params['input_ext']}"} | rerun.kept_files(kept)
            | {editing.BLOCKS, f"{editing.SOURCE}.flac", editing.SUGGESTIONS, editing.BROWSER, editing.SUBS})
    for p in d.iterdir():
        if p.name not in keep:
            _remove(d, p)
    params.pop("kept_reported", None)
    params["kept"] = kept
    editing.clear_chunks(job["id"])
    first_run = bool((params.get("edit") or {}).get("first_run"))  # a job made with a timeline: charged for what ran
    if job["kind"] in ("denoise", "denoise_transcribe"):
        params = editing.absorb(job, params, report)
        e = params["edit"]
        if not e.get("rendered") and e.get("facts"):
            e["rendered"] = editing.derived_rendered(job, params)  # (None when it cannot be known: everything is then made again)
            if e["rendered"] is None:
                e.pop("rendered")
        if report.get("suggestions"):
            editing.write_suggestions(job["id"], report["suggestions"])
        if report.get("suggestions") or report.get("has_suggestions"):
            e["suggestions"] = True
    # Charging and finishing are one step: a job is never "done" without its charge, and a failure
    # after the charge can't be reported as "not charged". Only a job that is still running completes.
    with db.tx() as c:
        uploaded = c.execute("SELECT COALESCE(uploaded_at, created_at) AS t FROM jobs WHERE id = ?", (job["id"],)).fetchone()["t"]
        n = c.execute(
            "UPDATE jobs SET status = 'done', progress = 100, stage = 'Done', outputs_json = ?, cost_micros = 0,"
            " finished_at = ?, edited_at = ?, expires_at = ? WHERE id = ? AND status = 'running'",
            (json.dumps(outputs), now, now, jobs.expiry(now, uploaded), job["id"]),
        ).rowcount
        if not n:
            log.warning("job %s is no longer running; not completing it", job["id"])
            return
        # (read again inside the transaction: the notice and the translated languages are written by other processes)
        costing = json.loads(c.execute("SELECT params_json FROM jobs WHERE id = ?", (job["id"],)).fetchone()["params_json"])
        cur = dict(costing)
        cur.pop("kept_reported", None)
        cur.pop("editor_reported", None)
        cur["kept"] = kept
        if "edit" in params:
            cur["edit"] = params["edit"]
        c.execute("UPDATE jobs SET params_json = ? WHERE id = ?", (json.dumps(cur), job["id"]))
        if job["hold_id"]:
            cost = _job_cost(job, costing, duration, report.get("billed") if first_run else None)
            if first_run:
                cost = min(cost, int(job.get("est_cost_micros") or cost))  # (a job made with a timeline never pays more than it was held for)
            cost = ledger.capture_hold(
                job["hold_id"], cost,
                f"{jobs.label_of(job['kind'], costing)}: {jobs.display_name(job)[:60]} ({duration / 60:.1f} min)", c=c,
            )
            c.execute("UPDATE jobs SET cost_micros = ? WHERE id = ?", (cost, job["id"]))
    jobs.touch(job["id"])  # finishing is an edit of the job and of the chain it belongs to: they keep their files together
    metrics.job_ended(job["id"])


def _complete_edit(job: dict, d: Path, res: dict) -> None:
    """The end of an edit run (editing.py): the new audio replaces the old, the cache and the report are kept, the run is charged for
    what it ran (never more than the quote it was held at), and the job is finished again with a new deadline."""
    fresh = db.one("SELECT params_json, outputs_json FROM jobs WHERE id = ?", (job["id"],))
    params = json.loads(fresh["params_json"])
    run = params.get("run")
    if not run:
        log.warning("job %s has no run to complete", job["id"])
        return
    now = db.now()
    if now + rerun.MIN_DELIVERY_SECONDS >= jobs.hard_deadline(job):
        jobs.finish_failed(job, rerun.TOO_LATE)
        return
    duration = float(job["duration_seconds"])
    report = params.pop("editor_reported", None) or editing.sanitize_report(res.get("editor"), d, duration)
    outputs = _verified_outputs(d, res)
    # The audio of the run is written under a temporary name next to the old (editing.STAGED); it takes the old one's place only when the
    # run is complete, so the files on disk always are what the stored record says.
    staged = {o["file"]: editing.final_name(o["file"]) for o in outputs if editing.final_name(o["file"]) != o["file"]}
    outputs = [{**o, "file": editing.final_name(o["file"])} for o in outputs]
    old = json.loads(fresh["outputs_json"] or "[]")
    audio_changed = run["op"] != "suggest" and run.get("audio", True)  # (a run that only changes the subtitles leaves the sound alone)
    text_changed = run["op"] != "suggest" and run.get("subs") is not None  # the caption files are put together again
    if audio_changed and not any(o["kind"] == "audio" for o in outputs):
        raise StageFailed("Processing failed; you were not charged.", "the run made no audio")
    if audio_changed or text_changed:
        # the other results stay (captions and transcript unless they were made again); a video made with the old sound or captions does
        # not fit any more
        gone = {"video", *(["audio"] if audio_changed else []), *(TEXT_KINDS if text_changed else ())}
        keep_old = [o for o in old if o["kind"] not in gone and o["file"] not in {x["file"] for x in outputs}]
        outputs = [*(o for o in outputs if (o["kind"] == "audio" and audio_changed) or (o["kind"] in TEXT_KINDS and text_changed)), *keep_old]
    else:
        outputs = old
    kept = params.get("kept") or {}
    if audio_changed:
        kept = {k: v for k, v in kept.items() if k == "segments"}  # the copies of the old strength slider no longer fit the audio
    keep = ({o["file"] for o in outputs} | {f"input{params['input_ext']}", editing.BLOCKS, f"{editing.SOURCE}.flac", editing.SUGGESTIONS,
                                            editing.BROWSER, editing.SUBS} | rerun.kept_files(kept))
    # (until the run is committed the old results and the new audio's temporary names stay where they are)
    for p in d.iterdir():
        if p.name not in keep | set(staged) | {o["file"] for o in json.loads(fresh["outputs_json"] or "[]")}:
            _remove(d, p)
    editing.clear_chunks(job["id"])
    params = editing.absorb(job, params, report, op=run["op"], processed=run["timeline"])
    params["kept"] = kept
    if report.get("suggestions"):
        editing.write_suggestions(job["id"], report["suggestions"])
        params["edit"]["suggestions"] = True
    if report.get("rendered") is None and audio_changed:
        params["edit"].pop("rendered", None)  # (should not happen: the runner always says what it rendered)
    params.pop("run", None)
    est = int(job["est_cost_micros"] or 0)
    billed = report.get("billed")
    subs_charge = editing.subtitle_charge(job["id"], run["subs"], run["subtitle_rates"]) if run.get("subs") else 0  # (what was made, translations that failed excepted)
    charge = est if billed is None or run["op"] == "suggest" else min(est, editing.price(billed, run["rates"])[0] + subs_charge)
    if run["op"] == "suggest":
        charge = 0
    swapped = editing.swap_in(d, staged)
    try:
        with db.tx() as c:
            uploaded = c.execute("SELECT COALESCE(uploaded_at, created_at) AS t, COALESCE(cost_micros, 0) AS spent FROM jobs WHERE id = ?", (job["id"],)).fetchone()
            n = c.execute(
                "UPDATE jobs SET status = 'done', next_stage = 'probe', progress = 100, stage = 'Done', outputs_json = ?, params_json = ?,"
                " finished_at = ?, edited_at = ?, expires_at = ?, error = NULL, worker_id = NULL WHERE id = ? AND status = 'running'",
                (json.dumps(outputs), json.dumps(params), now, now, jobs.expiry(now, uploaded["t"]), job["id"]),
            ).rowcount
            if not n:
                log.warning("job %s is no longer running; not completing it", job["id"])
                editing.undo_swap(swapped)
                return
            if job["hold_id"]:
                asked = max([0.0, *(run.get("asked") or {}).values()])
                charged = ledger.capture_hold(
                    job["hold_id"], charge,
                    f"{editing.run_label({'run': run})}: {jobs.display_name(job)[:60]} ({asked / 60:.1f} min)", c=c,
                )
            else:
                charged = 0
            c.execute("UPDATE jobs SET cost_micros = ?, est_cost_micros = ? WHERE id = ?", (uploaded["spent"] + charged, charged, job["id"]))
    except BaseException:
        editing.undo_swap(swapped)  # (the old audio is back: the record, which did not change, and the files agree)
        raise
    editing.commit_swap(swapped)
    for p in d.iterdir():  # (and now the old results that the new ones replaced, under other names, can go)
        if p.name not in keep:
            _remove(d, p)
    jobs.touch(job["id"])
    metrics.job_ended(job["id"])


def _decide(job: dict, extra: dict, duration: float) -> bool:
    """The settings of an automatic job (automatic.py): its file is measured in a sandboxed stage, the decision is made here, and
    the job becomes the job that was decided (tier, options, prices, estimate, hold). True when the decision is a voice rebuild and
    this worker is not the card's: the job goes back in the queue for it and this worker is done with it."""
    d = _write_spec(job, extra, duration)
    res = run_stage(job, d, "analyse", ANALYSE_TIMEOUT)
    stats = automatic.clean_stats(res.get("stats"))
    if stats is None:
        raise StageFailed("Processing failed; you were not charged.", "malformed measurements")
    params = json.loads(db.one("SELECT params_json FROM jobs WHERE id = ?", (job["id"],))["params_json"])
    # The card is up when this is its worker, or when its marker says so (the same fact that lets a rebuild be ordered).
    decision = automatic.decide(params, stats, duration, card_up=GPU_MODE or broker.gpu_worker_up())
    if automatic.apply(job["id"], decision, duration) is None:
        raise StageFailed("Processing failed; you were not charged.", "the job was no longer waiting for its settings")
    chosen = decision["chosen"]
    log.info("job %s decided: %s", job["id"], {k: chosen[k] for k in ("denoiseTier", "strength", "noise", "leveling")})
    if decision["notice"]:
        jobs.set_notice(job["id"], decision["notice"])
    if decision["tier"] == "rebuild" and not GPU_MODE:
        jobs.hand_over(job)
        return True
    return False


def process(job: dict, extra: dict) -> None:
    try:
        d = _write_spec(job, extra, job.get("duration_seconds"))
        params = json.loads(job["params_json"])
        if job["next_stage"] == "finalize":
            # (a video whose picture is made again takes as long as that takes)
            seconds = float(job.get("duration_seconds") or job["claimed_seconds"])
            res = run_stage(job, d, "finalize", max(1800, int(seconds * jobs.SPEED["encode"] * 3)) if jobs.encodes(params) else 1800)
            job["duration_seconds"] = db.one("SELECT duration_seconds FROM jobs WHERE id = ?", (job["id"],))["duration_seconds"]
            (_complete_edit if params.get("run") else _complete)(job, d, res)
            return
        if job["next_stage"] == "edit":
            # An edit run (editing.py): the recording is known, measured and priced already; one stage makes the missing blocks of the
            # timeline, renders it and writes the audio.
            duration = float(job["duration_seconds"])
            d = _write_spec(job, extra, duration)
            res = run_stage(job, d, "edit", jobs.stage_limits(duration, _max_seconds(job, params))[1])
            if res.get("needs_cloud_asr"):
                # Subtitle parts that need the internet (Accurate transcription, translations): the web process makes them, then the
                # runner puts the files together (the same way a job made with a timeline and subtitles goes on after its first stage)
                jobs.update_params(job["id"], editor_reported=editing.sanitize_report(res.get("editor"), d, duration))
                db.execute("UPDATE jobs SET status = 'cloud_asr', stage = 'Waiting to transcribe', progress = 85, heartbeat_at = ? WHERE id = ?",
                           (db.now(), job["id"]))
                return
            _complete_edit(job, d, res)
            return
        if params.get("rerun") or job["next_stage"] == "main":
            # A re-run knows its file: the source job measured it, and priced it (the hold is the price). So does an automatic job
            # a worker handed over to the card once it had decided on a rebuild: probed, measured, decided and priced already.
            duration = float(job["duration_seconds"])
        else:
            probe = run_stage(job, d, "probe", PROBE_TIMEOUT)
            try:
                duration = float(probe["duration"])  # the longer of the header's and the measured length
                channels = min(max(int(probe.get("channels") or 1), 1), 2)
            except (KeyError, TypeError, ValueError):
                raise StageFailed("Processing failed; you were not charged.", "malformed probe result")
            if not (math.isfinite(duration) and 0 < duration <= _max_seconds(job, params) + 61):
                raise StageFailed("Processing failed; you were not charged.", "implausible duration")
            if params.get("options", {}).get("video_out"):
                if not probe.get("has_video"):
                    if job["kind"] == "convert":
                        raise StageFailed("This file has no picture, so there is no video to convert. You weren't charged.")
                    pe, factor = False, 1.0  # (a sound file: there is no video to make, the rest of the job goes on)
                else:
                    # another file type or size: now it is known whether the picture is made again (charged) or copied (free)
                    pe, factor = probe.get("picture_encode"), probe.get("convert_factor")
                if not isinstance(pe, bool) or isinstance(factor, bool) or not isinstance(factor, (int, float)) or not 1.0 <= factor <= 64.0:
                    raise StageFailed("Processing failed; you were not charged.", "malformed picture_encode / convert_factor")
                params = jobs.update_params(job["id"], picture_encode=pe, convert_factor=round(float(factor), 2))
                job["params_json"] = json.dumps(params)
            _reprice(job, duration)
            job["duration_seconds"] = duration
            params = json.loads(job["params_json"])
            jobs.update_params(job["id"], channels=channels)
        if automatic.pending(params):
            if _decide(job, extra, duration):
                return  # handed over to the card's worker
            fresh = db.one("SELECT params_json, est_cost_micros FROM jobs WHERE id = ?", (job["id"],))  # the job it has become
            job["params_json"], job["est_cost_micros"] = fresh["params_json"], fresh["est_cost_micros"]
            params = json.loads(fresh["params_json"])
        d = _write_spec(job, extra, duration)
        wall = jobs.stage_limits(duration, _max_seconds(job, params))[1]
        res = run_stage(job, d, "main", wall)
        if res.get("notice"):
            jobs.set_notice(job["id"], res["notice"])
        if res.get("echo_skipped"):
            jobs.update_params(job["id"], echo_skipped=True)  # (no second track: the echo removal that was asked for is not charged)
        if res.get("kept"):
            jobs.update_params(job["id"], kept_reported=res["kept"])  # the finishing step may run after the cloud stage
        if res.get("editor"):
            # What the per-part editor needs to know (editing.py), checked now: the suggestions go to their file at once, the rest
            # waits in the job's record for the finishing step (which may run after the cloud stage).
            rep = editing.sanitize_report(res["editor"], d, duration)
            if rep.get("suggestions"):
                editing.write_suggestions(job["id"], rep.pop("suggestions"))
                rep["has_suggestions"] = True
            jobs.update_params(job["id"], editor_reported=rep)
        if res.get("needs_cloud_asr"):
            db.execute(
                "UPDATE jobs SET status = 'cloud_asr', stage = 'Waiting to transcribe', progress = 85,"
                " heartbeat_at = ? WHERE id = ?",
                (db.now(), job["id"]),
            )
            return
        _complete(job, d, res)
    except Interrupted:
        # Stopped by a restart or deploy, not a failure: back in the queue with the upload kept, no attempt used.
        interruptions = int(json.loads(job["params_json"]).get("interruptions", 0)) + 1
        if interruptions >= MAX_INTERRUPTIONS:
            log.error("job %s interrupted %d times; failing it", job["id"], interruptions)
            jobs.finish_failed(job, INTERRUPTED_MESSAGE)
        else:
            log.info("job %s put back in the queue (worker stopping, interruption %d)", job["id"], interruptions)
            jobs.update_params(job["id"], interruptions=interruptions)
            jobs.requeue(job)
    except StageFailed as e:
        jobs.finish_failed(job, e.user_message)
    except Exception:  # noqa: BLE001
        log.exception("job %s crashed", job["id"])
        jobs.finish_failed(job, "Processing failed; you were not charged.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    db.init_db()
    worker_id = f"{broker.GPU_WORKER_PREFIX if GPU_MODE else ''}{socket.gethostname()}:{os.getpid()}"
    extra = _settings_extra()
    signal.signal(signal.SIGTERM, lambda *_: _stop.set())
    signal.signal(signal.SIGINT, lambda *_: _stop.set())
    log.info("worker %s started", worker_id)
    load = hostload.get(get_settings().data_dir)
    last_sweep = 0.0
    while not _stop.is_set():
        if time.monotonic() - last_sweep > 60:
            last_sweep = time.monotonic()
            try:
                jobs.sweep()
            except Exception:  # noqa: BLE001
                log.exception("sweep failed")
        if GPU_MODE:
            broker.mark_gpu_worker("alive")
        job = claim(worker_id, load)
        if job is None:
            if GPU_MODE:
                broker.mark_gpu_worker("idle")
            _stop.wait(2.0)
            continue
        if GPU_MODE:
            broker.mark_gpu_worker("busy")
        log.info("job %s (%s) claimed with %s threads", job["id"], job["kind"], job["threads"])
        process(job, {**extra, "threads": job["threads"] or extra["threads"]})
    if GPU_MODE:  # stopping: the CPU workers must not wait for a worker that is gone
        broker.mark_gpu_worker("stopped")


if __name__ == "__main__":
    main()
