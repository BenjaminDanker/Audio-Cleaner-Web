"""What an edited job keeps in its folder, and what the web process knows about it without decoding (acw/editing.py has the whole
picture): the FACTS of the recording and the planning stand-in made from them, the BLOCK CACHE, the SUGGESTIONS, the job's timelines as
stored in its record, the disk it takes, and the staged files of a run until it is complete."""
from __future__ import annotations

import json
import math
import os
import shutil
import stat
from pathlib import Path

import numpy as np

from . import db, jobs
from .billing import pricing


MAX_PARTS = 500  # per layer
SUGGESTIONS = "suggestions.json"
SOURCE = "kept-source"  # the recording as 24-bit FLAC (acw_common.kept), for the preview stretches
BLOCKS = "blocks"
MAX_SPANS = 5000
SR = 48_000
GB = 1024**3


# Disk (measured on speech, docs/operations.md "What a job keeps"): the recording as FLAC and each cached step, per hour of audio and
# channel. A first run keeps the recording and one or two steps; a job that is edited a lot can hold the other steps too.
SOURCE_PER_HOUR_PER_CHANNEL = int(0.40 * GB)
BLOCKS_PER_HOUR_PER_CHANNEL = int(0.35 * GB)


def editor_lib():
    from acw_common import editor

    return editor


def timeline_lib():
    from acw_common import timeline

    return timeline


def read_json(path: Path, limit: int = 2_000_000):
    """A small JSON file of the job's folder, never through a link; None when it is missing or not sane."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return None
    try:
        with os.fdopen(fd, "rb") as f:
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode) or os.fstat(f.fileno()).st_size > limit:
                return None
            return json.loads(f.read().decode("utf-8", errors="replace"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------------------------------
# Facts and the planning stand-in for the recording
# ---------------------------------------------------------------------------------------------------------------------
def sanitize_facts(raw: object, d: Path) -> dict | None:
    """The facts a runner reports about the recording, checked: sane numbers only, the recording's copy only when it is a regular file."""
    try:
        n, ch = int(raw["n"]), int(raw["channels"])  # type: ignore[index]
        if not (SR // 2 <= n <= pricing.MAX_JOB_SECONDS * SR * 2 and ch in (1, 2)):
            return None
        plan = [None if c is None else int(c) for c in raw["plan"]]  # type: ignore[index]
        if len(plan) != ch or not all(c is None or 0 <= c < ch for c in plan):
            return None
        spans: dict[str, list[list[int]]] = {}
        todo = sorted({c for c in plan if c is not None})
        for c in todo:
            rows = raw["spans"][str(c)]  # type: ignore[index]
            out, last = [], 0
            for a, b in rows:
                a, b = int(a), int(b)
                if not (last <= a < b <= n):
                    return None
                out.append([a, b])
                last = b
            if not out or len(out) > MAX_SPANS:
                return None
            spans[str(c)] = out
        facts = {"n": n, "channels": ch, "plan": plan, "spans": spans, "rebuild_mid": bool(raw["rebuild_mid"]),  # type: ignore[index]
                 "reference": bool(raw.get("reference", False))}  # type: ignore[union-attr]
        src = raw.get("source")  # type: ignore[union-attr]
        if isinstance(src, dict):
            entry = {"file": f"{SOURCE}.flac", "channels": [int(c) for c in src["channels"]], "scale": float(src["scale"]), "samples": int(src["samples"])}
            if (src.get("file") == entry["file"] and regular_file(d / entry["file"]) and 0.0 < entry["scale"] <= 1.0
                    and entry["samples"] > 0 and entry["channels"] == list(range(ch))):
                facts["source"] = entry
        return facts
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except OSError:
        return False


def default_facts(job: dict, params: dict) -> dict:
    """What planning assumes about a job whose runner never reported facts (an old job, or a subtitles-only job's first edit): the
    length, the channels it has, every channel cleaned over all of it. Such a job has no cache either, so every block misses."""
    seconds = float(job.get("duration_seconds") or job["claimed_seconds"])
    ch = max(1, min(2, int(params.get("channels") or 2)))
    n = int(round(seconds * SR))
    return {"n": n, "channels": ch, "plan": list(range(ch)), "spans": {str(c): [[0, n]] for c in range(ch)}, "rebuild_mid": False, "reference": False}


def facts_of(job: dict, params: dict) -> dict:
    return (params.get("edit") or {}).get("facts") or default_facts(job, params)


def source_id(job: dict) -> str:
    """What identifies the samples of the recording in every cache key: the first job of the chain (the upload never changes)."""
    return "j" + (job.get("root_job_id") or job["id"])


def inputs_for(job: dict, params: dict):
    """An `editor.Inputs` that plans without audio: the facts stand in for the samples (the web process never decodes an upload)."""
    ed = editor_lib()
    f = facts_of(job, params)

    class FactsInputs(ed.Inputs):
        def __init__(self) -> None:  # noqa: D107 (deliberately not Inputs.__init__: there are no samples)
            self.source = np.zeros((0, f["channels"]), np.float32)
            self.source_id = source_id(job)
            self.reference = np.zeros(0, np.float32) if f.get("reference") else None
            self.reference_id = "r" if f.get("reference") else None
            self._plan = list(f["plan"])
            self._spans = {int(c): [tuple(r) for r in rows] for c, rows in f["spans"].items()}
            self._mid = bool(f["rebuild_mid"])

        @property
        def n(self) -> int:
            return f["n"]

        @property
        def channels(self) -> int:
            return f["channels"]

    return FactsInputs()


def blocks_dir(job_id: str) -> Path:
    return jobs.job_dir(job_id) / BLOCKS


def cache_for(job_id: str):
    """The job's block cache for looking (never opened with a decoder here)."""
    from acw_common import blocks

    return blocks.FlacCache(blocks_dir(job_id), None, create=False)


def file_seconds(job: dict, params: dict) -> float:
    """The recording's length as the editor counts it: its samples, else the measured length."""
    f = (params.get("edit") or {}).get("facts")
    return f["n"] / SR if f else float(job.get("duration_seconds") or job["claimed_seconds"])


def saved_timeline(job: dict, params: dict, seconds: float):
    """The job's working timeline: what was last saved, or the one it was made with (every job has one); re-based on the recording's
    length."""
    return timeline_lib().Timeline.from_json((params.get("edit") or {}).get("timeline") or {"version": 1, "seconds": seconds, "layers": {}}, seconds=seconds)


def rendered_timeline(job: dict, params: dict, seconds: float):
    """The timeline whose audio the job's result files hold (None: not known, so everything is made again)."""
    e = params.get("edit") or {}
    return timeline_lib().Timeline.from_json(e["rendered"], seconds=seconds) if e.get("rendered") else None


def part_key(start: float, end: float) -> str:
    return f"{start:.3f}-{end:.3f}"


# ---------------------------------------------------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------------------------------------------------
def sanitize_suggestions(raw: object, seconds: float) -> dict | None:
    """The suggestions a runner reports (suggest.Suggestions.to_json), checked and reduced to what the API says."""
    try:
        whole = raw["whole"]  # type: ignore[index]
        out_whole = {"rebuild": bool(whole["rebuild"]), "remover": whole["remover"], "strength": whole["strength"]}
        if out_whole["remover"] not in ("light", "heavy") or out_whole["strength"] not in ("gentle", "balanced", "strong", "max"):
            return None
        items = []
        for s in raw["suggestions"][:MAX_PARTS]:  # type: ignore[index]
            a, b = float(s["start"]), float(s["end"])
            if not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= seconds + 1.0) or s["kind"] not in ("part", "clipping"):
                continue
            noise = s.get("noise")
            if noise is not None:
                noise = {"strength_db": float(noise["strength_db"]), "remover": noise["remover"]}
                if noise["remover"] not in ("light", "heavy") or not 0 < noise["strength_db"] <= 100:
                    continue
            reasons = [str(r)[:160] for r in s.get("reasons", [])][:6]
            items.append({"start": round(a, 3), "end": round(min(b, seconds), 3), "kind": s["kind"], "rebuild": bool(s["rebuild"]),
                          "noise": noise, "reasons": reasons})
        return {"seconds": round(seconds, 3), "whole": out_whole, "suggestions": items}
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def write_suggestions(job_id: str, data: dict) -> None:
    d = jobs.job_dir(job_id)
    tmp = d / f"{SUGGESTIONS}.tmp"
    tmp.write_text(json.dumps({**data, "madeAt": db.now()}))
    os.chmod(tmp, 0o600)
    os.replace(tmp, d / SUGGESTIONS)


def read_suggestions(job_id: str) -> dict | None:
    p = jobs.job_dir(job_id) / SUGGESTIONS
    try:
        if not regular_file(p) or p.stat().st_size > 400_000:
            return None
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------------------------------------------------
def cache_bytes(seconds: float, channels: int = 2, steps: int = 1) -> int:
    """Disk the recording's FLAC and `steps` cached steps of a job take."""
    return int(seconds / 3600 * channels * (SOURCE_PER_HOUR_PER_CHANNEL + steps * BLOCKS_PER_HOUR_PER_CHANNEL))


def growth_bytes(plan, channels: int) -> int:
    """Disk the blocks a plan has to make add to the cache."""
    return int(sum(t.seconds for t in plan.misses) / 3600 * BLOCKS_PER_HOUR_PER_CHANNEL)


def cache_size(job_id: str) -> int:
    try:
        return cache_for(job_id).bytes_used()
    except OSError:
        return 0
SCRATCH_NAMES = {"decoded.f32", "reference.f32", "echo-free.npy", "cleaned.npy", "kept-decoded.f32", "spec.json", "main-state.json"}
STAGED = ".run"  # the audio an edit run made, under this extra ending, until the run is complete (`swap_in`)
BACKUP = ".bak"  # ... and the audio it replaces, while the run is being committed


def final_name(name: str) -> str:
    """The name a file of an edit run's result has once the run is complete: its temporary ending taken off."""
    return name[:-len(STAGED)] if name.endswith(STAGED) else name


def swap_in(d: Path, staged: dict[str, str]) -> list[tuple[Path, Path, Path | None]]:
    """Put the audio of a finished run in its place: each temporary file (`staged` maps its name to its final name) replaces the file with the
    final name, which is kept as `<name>.bak` until `commit_swap`. Returns what `undo_swap` / `commit_swap` need."""
    done: list[tuple[Path, Path, Path | None]] = []
    try:
        for tmp_name, final_name_ in staged.items():
            final, tmp = d / final_name_, d / tmp_name
            bak = d / (final_name_ + BACKUP) if final.exists() else None
            if bak is not None:
                os.replace(final, bak)
            os.replace(tmp, final)
            done.append((final, tmp, bak))
    except BaseException:
        undo_swap(done)
        raise
    return done


def undo_swap(done: list[tuple[Path, Path, Path | None]]) -> None:
    """The run did not complete: the old audio is back, the new is dropped."""
    for final, tmp, bak in reversed(done):
        try:
            if bak is not None:
                os.replace(bak, final)
            else:
                final.unlink(missing_ok=True)
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def commit_swap(done: list[tuple[Path, Path, Path | None]]) -> None:
    for _final, _tmp, bak in done:
        if bak is not None:
            bak.unlink(missing_ok=True)


def clear_scratch(d: Path) -> None:
    """Remove what a run leaves behind besides the job's files (the memory-mapped scratch, the spec and the results of its stages, the
    audio it made under a temporary name; audio that a run was putting in its place when it stopped goes back to what it was)."""
    try:
        for p in d.iterdir():
            n = p.name
            if n.endswith(BACKUP) and not p.is_dir():
                try:
                    os.replace(p, p.with_name(n[:-len(BACKUP)]))
                except OSError:
                    pass
                continue
            if n.endswith(STAGED) and not p.is_dir():
                p.unlink(missing_ok=True)
                continue
            if n in SCRATCH_NAMES or n.startswith(("result-", "editor-", "browser-")) or n == "asr_chunks":
                if p.is_dir() and not p.is_symlink():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)
    except OSError:
        pass
