"""Short, sample-exact stretches of the copies a finished job kept, for the strength slider.

`GET /api/jobs/{id}/kept/{noise|input}?start=<seconds>&seconds=<n>` answers with a 24-bit FLAC of just those samples of the
noise removal at full strength (`noise`) or of the recording as the remover heard it (`input`). Both kept files have the same
sample grid, so the same `start` and `seconds` give two stretches that line up sample for sample, and a page (or a program) can
blend them at any strength (`(1 - a) * noise + a * input`, `a = 10^(-dB/20)`) and play a recording of any length while holding
only a few seconds of it. A stretch decodes to the very values the whole kept file decodes to at those positions (24-bit FLAC
is cut and written again without a loss).

The cutting is done by `acw.worker.cutter`, a confined program of its own (FFmpeg inside the same kind of sandbox as the job
runner, with nothing writable), because the kept files were made by the runner, which handles untrusted uploads: this process
never opens them with a decoder itself. A few cuts run at once; the rest wait a moment, then are told to come back.

The per-part editor's preview (`GET /api/jobs/{id}/preview/{source|echo|noise|rebuild}`, acw/editing.py) uses the same helper for the
recording (`source`, cut like a kept copy) and a second one for what the block cache holds of a step (`assemble`: the stretch is put
together from the 30 s blocks it overlaps by `acw.worker.assembler`, confined the same way). Both share the places above.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

import acw_common

from . import jobs
from .config import get_settings

log = logging.getLogger("acw.stretch")

SAMPLE_RATE = 48_000
MAX_SECONDS = jobs.STRETCH_MAX_SECONDS
DEFAULT_SECONDS = MAX_SECONDS
CONCURRENT_CUTS = 4  # each is a process of about 40 ms of work; the web process keeps its threads for everything else
WAIT_SECONDS = 8  # how long a request waits for a free place before it is told to try again
TIMEOUT_SECONDS = 20


class Busy(Exception):
    """All places are taken and stayed taken."""


class Failed(Exception):
    """The cut did not work out (a damaged copy, or no sandbox)."""


class OutOfRange(ValueError):
    """`start` is not inside the recording."""


_places = threading.BoundedSemaphore(CONCURRENT_CUTS)


def span(start: float | None, seconds: float | None, total: int) -> tuple[int, int]:
    """(first sample, number of samples) for a request of `seconds` from `start` seconds (the nearest sample), in a copy of `total`
    samples: at most MAX_SECONDS long, and cut short by the end of the recording. Raises OutOfRange when `start` is at or past the end."""
    first = int(round((start or 0.0) * SAMPLE_RATE))
    if first >= total:
        raise OutOfRange(f"start is past the end of the recording ({total / SAMPLE_RATE:.3f} seconds)")
    wanted = int(round((DEFAULT_SECONDS if seconds is None else seconds) * SAMPLE_RATE))
    count = min(max(wanted, 1), MAX_SECONDS * SAMPLE_RATE, total - first)
    return first, count


def _environment() -> dict[str, str]:
    modules = (sys.modules["acw"], acw_common)  # acw and acw_common share one directory when deployed, not in a checkout
    return {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": os.pathsep.join(dict.fromkeys(str(Path(m.__file__).resolve().parents[1]) for m in modules)),
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def cut(path: Path, first: int, count: int) -> bytes:
    """The FLAC of `count` samples from sample `first` of the kept copy `path` (a regular file, already checked)."""
    if not _places.acquire(timeout=WAIT_SECONDS):
        raise Busy()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "acw.worker.cutter", str(get_settings().bin_dir), str(path), str(first), str(count)],
            capture_output=True, stdin=subprocess.DEVNULL, timeout=TIMEOUT_SECONDS, env=_environment(), cwd=str(path.parent),
            start_new_session=True,
        )
    except subprocess.TimeoutExpired:
        log.warning("cutting %s timed out", path)
        raise Failed("timed out")
    finally:
        _places.release()
    if proc.returncode != 0 or not proc.stdout.startswith(b"fLaC"):
        log.warning("cutting %s failed (%s): %s", path, proc.returncode, proc.stderr[-300:].decode("utf-8", "replace"))
        raise Failed("the cut failed")
    return proc.stdout


def assemble(job_dir: Path, first: int, count: int, channels: list) -> tuple[bytes, float]:
    """(the FLAC of the stretch [first, first + count) of a step made from its cached blocks, the scale it was brought down by when
    it was louder than a 24-bit file holds: the samples are the stretch times it). `channels`: per output channel the block files that
    overlap it (editing.plan_stretch), or None for a channel that is silent. Samples the cache does not hold are silence."""
    import json as _json

    if not _places.acquire(timeout=WAIT_SECONDS):
        raise Busy()
    try:
        spec = _json.dumps({"first": first, "count": count, "channels": channels}, separators=(",", ":"))
        proc = subprocess.run(
            [sys.executable, "-m", "acw.worker.assembler", str(get_settings().bin_dir), str(job_dir), spec],
            capture_output=True, stdin=subprocess.DEVNULL, timeout=TIMEOUT_SECONDS, env=_environment(), cwd=str(job_dir),
            start_new_session=True,
        )
    except subprocess.TimeoutExpired:
        log.warning("assembling a stretch of %s timed out", job_dir)
        raise Failed("timed out")
    finally:
        _places.release()
    if proc.returncode != 0 or not proc.stdout.startswith(b"fLaC"):
        log.warning("assembling a stretch of %s failed (%s): %s", job_dir, proc.returncode, proc.stderr[-300:].decode("utf-8", "replace"))
        raise Failed("the stretch could not be made")
    scale = 1.0
    for line in proc.stderr.decode("utf-8", "replace").splitlines()[::-1]:
        if line.startswith("scale="):
            try:
                scale = float(line[6:])
            except ValueError:
                pass
            break
    return proc.stdout, scale if 0.0 < scale <= 1.0 else 1.0


def peaks(job_dir: Path, spec: dict, timeout: float = 60.0) -> dict:
    """The loudness outline the confined `acw.worker.peaker` reads from cached blocks (`{"blocks": [{"name", "at"}]}`) or from the recording's
    copy (`{"file", "channels", "scale"}`): its JSON answer (see peaker.py). Shares the places of the cuts."""
    import json as _json

    if not _places.acquire(timeout=WAIT_SECONDS):
        raise Busy()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "acw.worker.peaker", str(get_settings().bin_dir), str(job_dir), _json.dumps(spec, separators=(",", ":"))],
            capture_output=True, stdin=subprocess.DEVNULL, timeout=timeout, env=_environment(), cwd=str(job_dir), start_new_session=True,
        )
    except subprocess.TimeoutExpired:
        log.warning("the outline of %s timed out", job_dir)
        raise Failed("timed out")
    finally:
        _places.release()
    if proc.returncode != 0:
        log.warning("the outline of %s failed (%s): %s", job_dir, proc.returncode, proc.stderr[-300:].decode("utf-8", "replace"))
        raise Failed("the outline could not be made")
    try:
        return _json.loads(proc.stdout)
    except ValueError:
        raise Failed("the outline could not be read")
