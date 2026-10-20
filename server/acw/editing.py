"""The per-part editor on the server: a job holds a timeline, a block cache and suggestions; quotes, process and export runs.

A job that has an audio step (Denoising, Voice rebuild, Leveling, echo removal) is EDITABLE. It holds

  * a TIMELINE (acw_common.timeline, format v1): which step is applied to which parts of the recording, each part with its own
    settings. A job made without one is the default timeline of its options (every step on the whole file); a job made with one
    (`POST /api/jobs` `timeline`) only has its parts processed, and charged. Subtitles are whole-file job options, not part of it.
  * a BLOCK CACHE (`blocks/` in the job's folder, 24-bit FLAC, acw_common.blocks.FlacCache): every block a model step made, keyed by
    the step, the model, the audio it read and where it is. A whole-file first run fills it, an edit only runs what it lacks.
  * the recording as 24-bit FLAC (`kept-source.flac`, for the preview stretches) and the SUGGESTIONS (`suggestions.json`).
  * the FACTS about the audio that planning needs and the web process must not measure itself (it never decodes uploads): length,
    channels, which channels are cleaned, the stretches the noise removers run on, whether the file has a second track.

An EDIT RUN is the job going back through the queue (`next_stage` "edit", `params["run"]`):

  quote     what a timeline needs and costs, from the cache: `editor.plan` -> the cache misses -> `editor.billable`. Nothing is held.
  process   the misses are run through the normal queue, sandbox and workers (Voice rebuild and the heavy remover on the card), then
            the timeline is rendered (joins, Leveling per part) into the result audio. Held at the quote, charged for what ran.
  export    the render alone, from the cache, for a quote of nothing (a strength change, a layer toggle, anything the cache holds).
  suggest   the suggestions of a job that has none (free).

PRICE of a run = for each step that runs, its rate per second x (the whole seconds, rounded up, of the parts asked for in the blocks
that have to be made + the step's fixed per-run cost as equivalent seconds, `editor.STEP_COSTS`). The context a model also reads is
never billed and there is no minimum. Leveling is charged where it runs on audio that changed since the last render.

RETENTION: the job's files are deleted 2 hours after its last edit and never later than 6 hours after the upload (`jobs.expiry`).
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
from pathlib import Path

import numpy as np

from . import db, jobs, keycaps
from .billing import ledger, pricing
from .config import get_settings

# The `code` of the errors a client may want to tell apart (X-Error-Code and "code" in the answer; docs/api.md).
JOB_NOT_FOUND = "job_not_found"  # no such job, or not yours: an API key can only edit the jobs it made itself
NOT_FINISHED = "job_not_finished"
FILES_GONE = "job_files_gone"  # the files are cleared away (expired or deleted): start again from the file
NOTHING_TO_CHANGE = "nothing_to_change"
EXPIRING = "job_expiring"  # too close to the end of the job's 6 hours to run again
PRICE_CHANGED = "price_changed"  # the price is above `maxCostMicros`
NOT_EDITABLE = "job_not_editable"  # a job without audio steps (subtitles only), or an old job whose audio facts were never recorded
BAD_TIMELINE = "bad_timeline"
BUSY = "job_busy"  # the job is being processed: wait for it
NEEDS_PROCESSING = "needs_processing"  # an export of something that has to be processed (and paid for) first
NOT_CACHED = "not_cached"  # a preview stretch of something the cache does not hold
EDITED = "job_edited"  # the old re-run does not apply to a job that has a timeline of its own

TIMELINE_MAX_BYTES = 200_000  # the JSON of a timeline (about 2,300 parts)
MAX_PARTS = 500  # per layer
SUGGESTIONS = "suggestions.json"
SOURCE = "kept-source"  # the recording as 24-bit FLAC (acw_common.kept), for the preview stretches
BLOCKS = "blocks"
KEEP_RUNS = 40  # edit runs remembered in the job's record
MAX_SPANS = 5000
MAX_RESOLVED = 2000

SR = 48_000
GB = 1024**3
# Disk (measured on speech, docs/operations.md "What a job keeps"): the recording as FLAC and each cached step, per hour of audio and
# channel. A first run keeps the recording and one or two steps; a job that is edited a lot can hold the other steps too.
SOURCE_PER_HOUR_PER_CHANNEL = int(0.40 * GB)
BLOCKS_PER_HOUR_PER_CHANNEL = int(0.35 * GB)

# Customer-facing names of the steps (the price list's words: what you get, never how it is made).
ITEM = {"echo": ("echo", "Echo removal"), "light": ("denoise", "Denoising"), "heavy": ("denoise", "Denoising"),
        "rebuild": ("rebuild", "Voice rebuild"), "leveling": ("leveling", "Leveling")}


def _ed():
    from acw_common import editor

    return editor


def _tl():
    from acw_common import timeline

    return timeline


# ---------------------------------------------------------------------------------------------------------------------
# Rates and prices
# ---------------------------------------------------------------------------------------------------------------------
def step_rates() -> dict[str, int]:
    """Micros per minute of audio of each step that can run, at today's price list (pinned with every run)."""
    denoise = pricing.rate("denoise", "standard").micros_per_minute
    return {"echo": pricing.rate("echo", "standard").micros_per_minute, "light": denoise, "heavy": denoise,
            "rebuild": pricing.rate("denoise", "rebuild").micros_per_minute,
            "leveling": pricing.rate("leveling", "standard").micros_per_minute}


def step_micros(rate_per_minute: int, seconds: int, fixed_equivalent_seconds: float) -> int:
    """rate per second x (whole seconds + the step's fixed amount as equivalent seconds), rounded up to a micro."""
    return math.ceil(rate_per_minute * (seconds + fixed_equivalent_seconds) / 60 - 1e-9)


def price(billable: dict, rates: dict[str, int]) -> tuple[int, list[dict]]:
    """(total micros, the lines a person sees) of `editor.billable(plan)` at `rates`. Light and heavy noise removal are one line,
    Denoising; each internal step is rounded up on its own."""
    total = 0
    items: dict[str, dict] = {}
    for step in ("echo", "light", "heavy", "rebuild", "leveling"):
        b = billable.get(step)
        if not b or b["seconds"] <= 0:
            continue
        micros = step_micros(rates[step], int(b["seconds"]), float(b["fixed_equivalent_seconds"]))
        item, label = ITEM[step]
        line = items.setdefault(item, {"item": item, "label": label, "quantity": 1, "seconds": 0, "fixedSeconds": 0.0, "usdPerMinute": rates[step] / pricing.MICROS_PER_USD,
                                       "micros": 0})
        line["seconds"] += int(b["seconds"])
        line["fixedSeconds"] = round(line["fixedSeconds"] + float(b["fixed_equivalent_seconds"]), 3)
        line["micros"] += micros
        total += micros
    return total, list(items.values())


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
            if (src.get("file") == entry["file"] and _regular(d / entry["file"]) and 0.0 < entry["scale"] <= 1.0
                    and entry["samples"] > 0 and entry["channels"] == list(range(ch))):
                facts["source"] = entry
        return facts
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def _regular(path: Path) -> bool:
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
    ed = _ed()
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


# ---------------------------------------------------------------------------------------------------------------------
# Timelines of a job
# ---------------------------------------------------------------------------------------------------------------------
def bad(message: str, status: int = 400, code: str = BAD_TIMELINE) -> jobs.JobError:
    return jobs.JobError(message, status, code=code)


def parse_timeline(raw: object, seconds: float, *, slack: float = 0.5):
    """Check a timeline from a client strictly and read it for a recording of `seconds`: the size, the format, at most MAX_PARTS parts
    in a layer, and every part inside the file (up to `slack` seconds past its end, clamped). Raises JobError 400 `bad_timeline`."""
    tl = _tl()
    try:
        size = len(json.dumps(raw, separators=(",", ":")))
    except (TypeError, ValueError):
        raise bad("The timeline is not valid JSON.") from None
    if size > TIMELINE_MAX_BYTES:
        raise bad(f"The timeline is too big ({size // 1000} KB; at most {TIMELINE_MAX_BYTES // 1000} KB).")
    if not isinstance(raw, dict):
        raise bad("The timeline must be an object.")
    layers = raw.get("layers", {})
    if isinstance(layers, dict):
        for name, parts in layers.items():
            if isinstance(parts, list):
                if len(parts) > MAX_PARTS:
                    raise bad(f"At most {MAX_PARTS} parts per layer ({name} has {len(parts)}).")
                for i, p in enumerate(parts):
                    if not isinstance(p, dict):
                        continue  # (from_json says what is wrong)
                    a, b = p.get("start"), p.get("end")
                    if all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (a, b)):
                        if not (0 <= a < b):
                            raise bad(f"Part {i + 1} of {name}: it must start at 0 or later and end after it starts.")
                        if b > seconds + slack:
                            raise bad(f"Part {i + 1} of {name} ends at {b:g} s, after the end of the recording ({seconds:.3f} s).")
    sec = raw.get("seconds")
    if isinstance(sec, (int, float)) and not isinstance(sec, bool) and abs(sec - seconds) > max(slack, 0.002 * seconds):
        raise bad(f"The timeline is for a recording of {sec:g} s; this one is {seconds:.3f} s.")
    try:
        return tl.Timeline.from_json(raw, seconds=seconds)
    except tl.TimelineError as e:
        raise bad(str(e)) from None


def editable_kind(job: dict) -> bool:
    return job["kind"] in ("denoise", "denoise_transcribe")


def default_timeline(job: dict, params: dict, seconds: float):
    """The timeline a job made without one stands for: its options, every step on the whole file (`Timeline.from_clean_options`;
    a noise remover that was left to "auto" is the one the runner reported using). Empty for a job without audio steps."""
    from acw_common import pipeline

    tl = _tl()
    if not editable_kind(job):
        return tl.Timeline(seconds)
    opts = params["options"]
    tier = params.get("denoise_tier")
    noise = opts.get("noise", "auto")
    if noise == "auto" and params.get("noise_profile") in ("light", "heavy"):
        noise = params["noise_profile"]
    c = pipeline.CleanOptions(atten_lim_db=float(opts.get("atten_lim_db", 100.0)), loudness_lufs=opts.get("loudness_lufs", -16.0),
                              leveling=bool(opts.get("leveling", True)), noise=noise, restore_voice=tier == "rebuild",
                              denoise=tier != pricing.NO_DENOISE)
    return tl.Timeline.from_clean_options(seconds, c, echo=bool(opts.get("echo_reference_track")))


def saved_timeline(job: dict, params: dict, seconds: float):
    """The job's working timeline: what was last saved (or given at creation), else the default; re-based on the recording's length."""
    raw = (params.get("edit") or {}).get("timeline")
    if raw:
        return _tl().Timeline.from_json(raw, seconds=seconds)
    return default_timeline(job, params, seconds)


def rendered_timeline(job: dict, params: dict, seconds: float):
    """The timeline whose audio the job's result files hold (None: not known, so everything is made again)."""
    e = params.get("edit") or {}
    if e.get("rendered"):
        return _tl().Timeline.from_json(e["rendered"], seconds=seconds)
    if e.get("first_run"):
        return None
    t = default_timeline(job, params, seconds)  # a job made before the editor: what its options say
    return None if any(p.settings.get("remover") == "auto" for _, p in t) else t


def part_key(start: float, end: float) -> str:
    return f"{start:.3f}-{end:.3f}"


def with_removers(tl, resolved: dict | None):
    """A copy of the timeline with every "auto" remover that was decided for exactly that part set; the others stay "auto".
    Returns (timeline, True when some part still has to be decided)."""
    out = tl.copy()
    left = False
    for p in out.layers["noise"]:
        if p.settings.get("remover") == "auto":
            r = (resolved or {}).get(part_key(p.start, p.end))
            if r in ("light", "heavy"):
                p.settings["remover"] = r
            else:
                left = True
    return out, left


def for_planning(tl, resolved: dict | None, provisional: str = "light"):
    """The timeline as the planner needs it (every remover named): the parts not decided yet count as the light remover, which is
    what most recordings get; the quote says so (`approximate`)."""
    out, left = with_removers(tl, resolved)
    for p in out.layers["noise"]:
        if p.settings.get("remover") == "auto":
            p.settings["remover"] = provisional
    return out, left


# ---------------------------------------------------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------------------------------------------------
def asked_summary(plan) -> dict:
    """Seconds of the recording each customer-facing step is asked for (union over the plan's parts), for the quote."""
    out: dict[str, float] = {}
    blocks = _blocks()
    for step, ranges in plan.asked.items():
        item = ITEM[step][0]
        out[item] = out.get(item, 0.0) + blocks.union_seconds(ranges)
    return {k: round(v, 3) for k, v in out.items() if v > 0}


def _blocks():
    from acw_common import blocks

    return blocks


def make_quote(job: dict, params: dict, tl, *, rates: dict[str, int] | None = None, provisional: str = "light") -> dict:
    """What `tl` needs on this job and costs, from its cache. {"free", "unchanged", "approximate", "micros", "lines", "steps",
    "billable", "misses", "asked", "plan": the planner's Plan (not JSON)}. Raises JobError for a timeline this recording can't have."""
    ed = _ed()
    seconds = file_seconds(job, params)
    edit = params.get("edit") or {}
    rates = rates or step_rates()
    tl = _tl().Timeline.from_json(tl.to_json(), seconds=seconds)
    rendered = rendered_timeline(job, params, seconds)
    planned, approx = for_planning(tl, edit.get("resolved"), provisional)
    prev, _ = for_planning(rendered, edit.get("resolved")) if rendered is not None else (None, False)
    inputs = inputs_for(job, params)
    try:
        plan = ed.plan(planned, inputs, cache_for(job["id"]), prev)
    except ed.EditorError as e:
        raise bad(str(e)) from None
    unchanged = rendered is not None and with_removers(tl, edit.get("resolved"))[0] == rendered
    if unchanged:
        billable, total, lines = {}, 0, []  # the result files already hold exactly this
    else:
        billable = ed.billable(plan)
        total, lines = price(billable, rates)
    rebuild_s = _blocks().union_seconds(plan.asked.get("rebuild", []))
    if rebuild_s > pricing.MAX_REBUILD_SECONDS:
        raise bad(f"Voice rebuild takes up to {pricing.MAX_REBUILD_SECONDS // 60} minutes of a recording at a time; this timeline asks for {rebuild_s / 60:.0f}.")
    summary = plan.summary()
    return {
        "free": total == 0,
        "unchanged": unchanged,
        "approximate": approx,
        "micros": total,
        "lines": lines,
        "billable": billable,
        "steps": {s: {"blocks": r["blocks"], "seconds": billable.get(s, {}).get("seconds", 0)} for s, r in summary.items()
                  if r["blocks"] or billable.get(s)},
        "misses": len(plan.misses),
        "asked": asked_summary(plan),
        "needsProcess": bool(plan.misses or billable),
        "rates": rates,
        "plan": plan,
        "timeline": tl,
        "gpu": gpu_stages_of(plan),
    }


def gpu_stages_of(plan) -> list[str]:
    """The GPU steps a plan's missing blocks need: the card runs Voice rebuild and the heavy remover."""
    return [s for s in ("rebuild", "heavy") if any(t.step == s for t in plan.misses)]


def public_quote(q: dict) -> dict:
    """The quote as the API says it."""
    return {
        "estCostMicros": q["micros"], "estCostUsd": q["micros"] / pricing.MICROS_PER_USD, "free": q["free"], "unchanged": q["unchanged"],
        "needsProcess": q["needsProcess"], "approximate": q["approximate"], "lines": q["lines"], "steps": q["steps"], "asked": q["asked"],
        "timeline": q["timeline"].to_json(),
    }


# ---------------------------------------------------------------------------------------------------------------------
# What is cached, and the stretches of it a preview plays
# ---------------------------------------------------------------------------------------------------------------------
def probe_timeline(tl, layer: str, remover: str | None = None):
    """A timeline that asks for `layer` on the whole recording (with the echo parts of `tl`, which the blocks' identity includes):
    its plan lists every block of that layer, the cache says which of them exist."""
    t = _tl().Timeline(tl.seconds)
    for p in tl.layers["echo"]:
        t.paint("echo", p.start, p.end)
    if layer == "noise":
        t.paint("noise", 0.0, t.seconds, {"strength_db": 100.0, "remover": remover or "light"})
    elif layer == "rebuild":
        t.paint("rebuild", 0.0, t.seconds)
    elif layer == "echo":
        t.paint("echo", 0.0, t.seconds)
    return t


def _merge(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in sorted(r for r in ranges if r[1] > r[0]):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _intersect(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _subtract(whole: tuple[int, int], covered: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out, pos = [], whole[0]
    for a, b in _merge(covered):
        if b <= pos or a >= whole[1]:
            continue
        if a > pos:
            out.append((pos, a))
        pos = max(pos, b)
    if pos < whole[1]:
        out.append((pos, whole[1]))
    return out


def _plan_of(job: dict, params: dict, probe, inputs=None):
    ed = _ed()
    try:
        return ed.plan(probe, inputs or inputs_for(job, params), cache_for(job["id"]))
    except ed.EditorError:
        return None  # (echo removal on a file without a second track: nothing of it exists)


def layer_coverage(job: dict, params: dict, tl, step: str, plan=None) -> dict:
    """What the cache holds of one step on the whole recording: {"channels": {channel: [(a, b)]}, "silent": {channel: [(a, b)]}}
    in samples (silent: stretches the noise removers skip, digital silence, that count as done)."""
    inputs = inputs_for(job, params)
    layer = {"echo": "echo", "light": "noise", "heavy": "noise", "rebuild": "rebuild"}[step]
    plan = plan or _plan_of(job, params, probe_timeline(tl, layer, step if layer == "noise" else None), inputs)
    cache = cache_for(job["id"])
    out: dict = {"channels": {}, "silent": {}}
    if plan is None:
        return out
    for t in plan.tasks:
        if t.step != step:
            continue
        out["channels"].setdefault(t.channel, [])
        if cache.has(t.key):
            out["channels"][t.channel].append((t.s, t.e))
    if layer == "noise":
        for c in out["channels"]:
            spans = inputs.spans(c)
            out["silent"][c] = _subtract((0, inputs.n), list(spans))
    return out


def covered(cov: dict) -> list[tuple[int, int]]:
    """The samples where every channel of the step has its block (or is silent there)."""
    sets = [_merge(cov["channels"][c] + cov["silent"].get(c, [])) for c in cov["channels"]]
    if not sets:
        return []
    out = sets[0]
    for s in sets[1:]:
        out = _intersect(out, s)
    return out


def cached_ranges(job: dict, params: dict, tl) -> dict[str, list[list[float]]]:
    """What the cache holds, per layer, as time ranges in seconds: `echo`, `noise` (either remover; `light` and `heavy` apart) and
    `rebuild`. Leveling is not cached (it is made again at every render)."""
    if not blocks_dir(job["id"]).is_dir():
        return {"echo": [], "noise": [], "light": [], "heavy": [], "rebuild": []}
    secs = lambda rs: [[round(a / SR, 3), round(b / SR, 3)] for a, b in rs]  # noqa: E731
    by = {s: covered(layer_coverage(job, params, tl, s)) for s in ("echo", "light", "heavy", "rebuild")}
    return {"echo": secs(by["echo"]), "noise": secs(_merge(by["light"] + by["heavy"])), "light": secs(by["light"]), "heavy": secs(by["heavy"]),
            "rebuild": secs(by["rebuild"])}


def stretch_removers(tl, params: dict, resolved: dict | None, first: int, end: int, has: dict[str, list[tuple[int, int]]]) -> list[tuple[int, int, str]]:
    """Which remover's output a preview of [first, end) plays at each moment: the one of the noise part there (decided, or for a part
    still on "auto" the one whose blocks exist, light first), and outside the noise parts the one the job used (else light)."""
    fallback = params.get("noise_profile") if params.get("noise_profile") in ("light", "heavy") else "light"
    segs: list[tuple[int, int, str]] = []
    pos = first
    for p in sorted(tl.layers["noise"], key=lambda p: p.start):
        a, b = max(first, int(round(p.start * SR))), min(end, int(round(p.end * SR)))
        if b <= a:
            continue
        if a > pos:
            segs.append((pos, a, fallback))
        r = p.settings.get("remover")
        if r == "auto":
            r = (resolved or {}).get(part_key(p.start, p.end))
        if r not in ("light", "heavy"):
            r = "light" if _intersect(has.get("light", []), [(a, b)]) or not _intersect(has.get("heavy", []), [(a, b)]) else "heavy"
        segs.append((a, b, r))
        pos = b
    if pos < end:
        segs.append((pos, end, fallback))
    merged: list[tuple[int, int, str]] = []
    for a, b, r in segs:
        if merged and merged[-1][1] == a and merged[-1][2] == r:
            merged[-1] = (merged[-1][0], b, r)
        else:
            merged.append((a, b, r))
    return merged


def plan_stretch(job: dict, params: dict, layer: str, first: int, count: int) -> dict:
    """What the assembler needs to make the stretch [first, first + count) samples of `layer` ("echo", "noise", "rebuild") from the
    cache, per output channel: the block files that overlap it (each with the part of the stretch it is used for: `from`, `to`), and
    which parts of the stretch the cache does not hold.
    {"channels": [[{"name", "at", "len", "from", "to"}] | None per output channel], "missing": [(a, b)] relative to `first`}"""
    f = facts_of(job, params)
    seconds = file_seconds(job, params)
    tl = saved_timeline(job, params, seconds)
    cache = cache_for(job["id"])
    end = first + count
    inputs = inputs_for(job, params)
    chans: list[list[dict] | None] = [[] for _ in range(f["channels"])]
    have: list[list[tuple[int, int]]] = [[] for _ in range(f["channels"])]  # what each output channel holds of the stretch

    def take(plan, step: str, ranges: list[tuple[int, int]]) -> None:
        for t in plan.tasks:
            if t.step != step or not cache.has(t.key):
                continue
            for a, b in _intersect([(t.s, t.e)], ranges):
                outs = range(f["channels"]) if t.channel == "mid" else [c for c in range(f["channels"]) if (f["plan"][c] if step in ("light", "heavy") else c) == t.channel]
                for c in outs:
                    if chans[c] is not None:
                        chans[c].append({"name": cache.name(t.key), "at": t.s, "len": t.e - t.s, "from": a, "to": b})
                        have[c].append((a, b))

    if layer == "echo":
        plan = _plan_of(job, params, probe_timeline(tl, "echo"), inputs)
        if plan is not None:
            take(plan, "echo", [(first, end)])
    elif layer == "rebuild":
        plan = _plan_of(job, params, probe_timeline(tl, "rebuild"), inputs)
        if plan is not None:
            take(plan, "rebuild", [(first, end)])
    else:
        has = {s: covered(layer_coverage(job, params, tl, s)) for s in ("light", "heavy")}
        for a, b, r in stretch_removers(tl, params, (params.get("edit") or {}).get("resolved"), first, end, has):
            plan = _plan_of(job, params, probe_timeline(tl, "noise", r), inputs)
            if plan is not None:
                take(plan, r, [(a, b)])
        for c in range(f["channels"]):  # a channel with nothing to clean (silent), or digital silence the removers skip
            src = f["plan"][c]
            if src is None:
                chans[c] = None
            else:
                have[c] += _intersect(_subtract((0, inputs.n), list(inputs.spans(src))), [(first, end)])
    active = [c for c in range(f["channels"]) if chans[c] is not None]
    got = [(first, end)] if not active else _merge(have[active[0]])
    for c in active[1:]:
        got = _intersect(got, _merge(have[c]))
    missing = _subtract((first, end), got) if active else []
    return {"channels": chans, "missing": [(a - first, b - first) for a, b in missing]}


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
        if not _regular(p) or p.stat().st_size > 400_000:
            return None
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------------------------------
# What a run reports
# ---------------------------------------------------------------------------------------------------------------------
def billable_from(seconds_by_step: object) -> dict:
    """The `editor.billable` shape from the whole seconds a runner reports per step; the fixed amounts are ours, not the runner's."""
    ed = _ed()
    out: dict = {}
    if isinstance(seconds_by_step, dict):
        for step, row in seconds_by_step.items():
            try:
                secs = int(row["seconds"] if isinstance(row, dict) else row)
            except (KeyError, TypeError, ValueError):
                continue
            if step in ed.STEP_COSTS and 0 < secs <= pricing.MAX_JOB_SECONDS:
                c = ed.STEP_COSTS[step]
                out[step] = {"seconds": secs, "fixed_equivalent_seconds": c.fixed_equivalent_seconds, "resource": c.resource,
                             "fixed": c.fixed, "per_second": c.per_second}
    return out


def sanitize_report(raw: object, d: Path, seconds: float) -> dict:
    """The record of a run the runner reports (`result["editor"]`), checked: {"facts", "resolved", "rendered", "billed",
    "suggestions", "computed"}; whatever is not sane is left out."""
    out: dict = {}
    if not isinstance(raw, dict):
        return out
    facts = sanitize_facts(raw.get("facts"), d) if raw.get("facts") else None
    if facts:
        out["facts"] = facts
    res = raw.get("resolved")
    if isinstance(res, dict):
        clean = {k: v for k, v in res.items() if isinstance(k, str) and re.fullmatch(r"\d+(\.\d{1,3})?-\d+(\.\d{1,3})?", k) and v in ("light", "heavy")}
        out["resolved"] = dict(list(clean.items())[:MAX_RESOLVED])
    if raw.get("rendered") is not None:
        try:
            rt = _tl().Timeline.from_json(raw["rendered"], seconds=facts["n"] / SR if facts else seconds)
            if all(p.settings.get("remover") in ("light", "heavy") for p in rt.layers["noise"]):
                out["rendered"] = rt.to_json()
        except (_tl().TimelineError, KeyError, TypeError):
            pass
    if "billed" in raw:
        out["billed"] = billable_from(raw["billed"])
    if raw.get("suggestions"):
        sug = sanitize_suggestions(raw["suggestions"], facts["n"] / SR if facts else seconds)
        if sug:
            out["suggestions"] = sug
    if isinstance(raw.get("computed"), dict):
        out["computed"] = {k: int(v) for k, v in raw["computed"].items() if k in ("echo", "light", "heavy", "rebuild", "leveling") and isinstance(v, int)}
    return out


def absorb(job: dict, params: dict, report: dict, *, op: str | None = None) -> dict:
    """The job's parameters with a run's report in them: the facts, the removers that were decided, the timeline that was rendered
    (the one the result files hold), and a line in its history."""
    edit = dict(params.get("edit") or {})
    if report.get("facts"):
        edit["facts"] = report["facts"]
    if report.get("resolved"):
        merged = {**(edit.get("resolved") or {}), **report["resolved"]}
        edit["resolved"] = dict(list(merged.items())[-MAX_RESOLVED:])
    if report.get("rendered"):
        edit["rendered"] = report["rendered"]
    edit.pop("first_run", None)
    if op:
        runs = list(edit.get("runs") or [])
        runs.append({"op": op, "at": db.now(), "computed": report.get("computed") or {}})
        edit["runs"] = runs[-KEEP_RUNS:]
        edit["lastRun"] = {"op": op, "ok": True, "at": db.now()}
    return {**params, "edit": edit}


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


def clear_scratch(d: Path) -> None:
    """Remove what a run leaves behind besides the job's files (the memory-mapped scratch, the spec and the results of its stages)."""
    try:
        for p in d.iterdir():
            n = p.name
            if n in SCRATCH_NAMES or n.startswith("result-") or n.startswith("editor-") or n == "asr_chunks":
                if p.is_dir() and not p.is_symlink():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------------------------------------------------
# A job made with a timeline
# ---------------------------------------------------------------------------------------------------------------------
REPLACED_OPTIONS = ("atten_lim_db", "noise", "leveling", "loudness_lufs", "echo_reference_track", "automatic")


def steps_in(tl) -> int:
    """How many cached steps a timeline can fill (for the disk a job is promised)."""
    return max(1, sum(1 for n in ("echo", "noise", "rebuild") if tl.layers[n]))


def prepare_timeline_job(kind: str, denoise_tier: str | None, options: dict, claimed_seconds: float, raw: object):
    """The checks of a job made with a timeline: it needs audio steps, replaces the choices the timeline makes (the tier, strength,
    remover, Leveling, loudness, echo removal, automatic settings), and every part is inside the file. Returns (Timeline, the tier the
    job stores)."""
    if kind not in ("denoise", "denoise_transcribe"):
        raise bad("A timeline is for jobs with noise removal or Leveling: use kind denoise or denoise_transcribe.")
    if denoise_tier is not None:
        raise bad("A timeline says what is done to each part: leave denoiseTier out.", code="timeline_options")
    clash = [k for k in REPLACED_OPTIONS if k in (options or {})]
    if clash:
        raise bad(f"A timeline replaces these options, leave them out: {', '.join(clash)}.", code="timeline_options")
    tl = parse_timeline(raw, claimed_seconds, slack=0.1 * claimed_seconds + 5.0)
    if not any(tl.layers[n] for n in ("echo", "noise", "rebuild", "leveling")):
        raise bad("The timeline has no parts: it would not do anything.")
    rebuild_s = sum(p.seconds for p in tl.layers["rebuild"])
    if rebuild_s > pricing.MAX_REBUILD_SECONDS:
        raise bad(f"Voice rebuild takes up to {pricing.MAX_REBUILD_SECONDS // 60} minutes of a recording at a time; this timeline asks for {rebuild_s / 60:.0f}.")
    if tl.layers["rebuild"]:
        from .broker import gpu_worker_up

        if not gpu_worker_up():  # (the card is the only place it runs)
            raise jobs.JobError(jobs.REBUILD_PAUSED)
    return tl, "standard"


def price_timeline(tl, seconds: float, rates: dict[str, int]) -> tuple[int, list[dict], dict]:
    """The price of running `tl` from nothing (an empty cache) on a recording of about `seconds`, as the hold of a job made with
    one: (micros, lines, billable). Every part of every step has to be made."""
    ed = _ed()
    n = int(round(seconds * SR))
    f = {"n": n, "channels": 2, "plan": [0, 1], "spans": {"0": [[0, n]], "1": [[0, n]]}, "rebuild_mid": False, "reference": True}
    job = {"id": "0" * 32, "claimed_seconds": seconds, "duration_seconds": seconds, "root_job_id": None}
    inputs = inputs_for(job, {"edit": {"facts": f}})
    t = _tl().Timeline.from_json(tl.to_json(), seconds=n / SR)
    planned, _ = for_planning(t, None)
    try:
        plan = ed.plan(planned, inputs, None)
    except ed.EditorError as e:
        raise bad(str(e)) from None
    billable = ed.billable(plan)
    micros, lines = price(billable, rates)
    return micros, lines, billable


def first_run_cost(kind: str, params: dict, seconds: float, billed: dict | None = None) -> int:
    """What a job made with a timeline costs for a recording of `seconds`: its parts (`billed`: the runner's report of what ran, else the
    whole timeline) at the rates pinned with the job, and the subtitles."""
    rates = params.get("edit_rates") or step_rates()
    if billed is None:
        tl = _tl().Timeline.from_json(params["edit"]["timeline"], seconds=seconds)
        edit = price_timeline(tl, seconds, rates)[0]
    else:
        edit = price(billed, rates)[0]
    text = pricing.job_cost_micros("transcribe", None, params.get("transcribe_tier"), seconds, params.get("rates"),
                                   len((params.get("options") or {}).get("translate") or [])) if kind != "denoise" else 0
    return edit + text


def run_label(params: dict) -> str:
    """The customer-facing name of an edit run: what it does."""
    run = params.get("run") or {}
    if run.get("op") == "suggest":
        return "Finding suggestions"
    names = list(dict.fromkeys(l["label"] for l in run.get("lines") or []))
    return "Edited" + (f" ({' + '.join(names)})" if names else "")


def timeline_label(params: dict) -> str:
    """The name of a first run made with a timeline (the layers it has)."""
    tl = (params.get("edit") or {}).get("timeline") or {}
    names = {"echo": "Echo removal", "noise": "Denoising", "rebuild": "Voice rebuild", "leveling": "Leveling"}
    have = [names[k] for k in ("rebuild", "noise", "echo", "leveling") if (tl.get("layers") or {}).get(k)]
    return " + ".join(have) or "Edited"


def timeline_seconds(raw: dict, seconds: float, channels: int) -> float:
    """A rough wall-clock time for running a timeline from nothing (the queue estimate): each layer's parts at the step's speed."""
    S = jobs.SPEED
    per = {"echo": S["echo"], "noise": S["denoise_auto"], "rebuild": S["denoise_rebuild"], "leveling": S["leveling"]}
    total = 0.0
    for name, parts in ((raw or {}).get("layers") or {}).items():
        if name in per:
            total += per[name] * sum(max(0.0, min(seconds, p["end"]) - max(0.0, p["start"])) for p in parts) * channels
    return total


def derived_rendered(job: dict, params: dict) -> dict | None:
    """The timeline a first run's result holds, from the job's options (every step on the whole file), when it is fully decided."""
    f = (params.get("edit") or {}).get("facts")
    if not f:
        return None
    try:
        t = default_timeline(job, params, f["n"] / SR)
    except _tl().TimelineError:
        return None
    return None if any(p.settings.get("remover") == "auto" for _, p in t) else t.to_json()


def spec_for(job: dict, params: dict) -> dict | None:
    """What the runner is told about the per-part editor: the run (a first run, or an edit run's operation, timeline and the one it
    renders from), what identifies the recording in every cache key, and the removers already decided."""
    if not editable_kind(job):
        return None
    e = params.get("edit") or {}
    run = params.get("run")
    out = {"source_id": source_id(job), "resolved": e.get("resolved") or {}, "facts": e.get("facts")}
    if run:
        out.update(op=run["op"], timeline=run["timeline"], previous=run.get("previous"), audio_format=run.get("audio_format"),
                   suggest=run["op"] == "suggest")
    else:
        out.update(op="first", timeline=e.get("timeline") if e.get("first_run") else None, previous=None, audio_format=None,
                   suggest=not params.get("rerun") and 0 < float(job.get("duration_seconds") or job["claimed_seconds"]) <= get_settings().suggest_max_seconds)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# The API's operations
# ---------------------------------------------------------------------------------------------------------------------
def job_for_edit(user_id: str, job_id: str, api_key_id: str | None, *, need_done: bool = True) -> tuple[dict, dict]:
    """The job and its parameters, for an operation of the editor: the account's own job (an API key: only a job it made itself, the
    re-run's rule), one that has audio steps and files. `need_done`: not while it is still being made or being processed."""
    row = db.one("SELECT * FROM jobs WHERE id = ? AND user_id = ?", (job_id, user_id)) if len(job_id) == 32 else None
    if row is None or (api_key_id is not None and row["api_key_id"] != api_key_id):
        raise jobs.JobError("Job not found.", 404, code=JOB_NOT_FOUND)
    job = dict(row)
    params = json.loads(job["params_json"])
    if not editable_kind(job):
        raise jobs.JobError("This job has no sound to edit (it makes subtitles only).", 409, code=NOT_EDITABLE)
    gone = "This job’s files are already cleared away, so it can’t be edited. Start it again from your file."
    if job["status"] in ("expired", "failed", "canceled"):
        raise jobs.JobError(gone, 409, code=FILES_GONE)
    if need_done:
        if job["status"] != "done":
            if params.get("run"):
                raise jobs.JobError("This job is being processed. Wait for it to finish.", 409, code=BUSY)
            raise jobs.JobError("This job is not finished yet.", 409, code=NOT_FINISHED)
        if not jobs.job_dir(job_id).is_dir() or not _regular(jobs.input_path(job)):
            raise jobs.JobError(gone, 409, code=FILES_GONE)
    return job, params


def timeline_info(job: dict, params: dict) -> dict:
    """The answer of GET /api/jobs/{id}/timeline: the timeline, what is cached of each layer, the suggestions, how to read the preview."""
    seconds = file_seconds(job, params)
    e = params.get("edit") or {}
    f = e.get("facts")
    tl = saved_timeline(job, params, seconds)
    rendered = rendered_timeline(job, params, seconds)
    out: dict = {
        "timeline": tl.to_json(),
        "custom": bool(e.get("timeline")),
        "rendered": rendered.to_json() if rendered is not None else None,
        "seconds": round(seconds, 3),
        "channels": (f or default_facts(job, params))["channels"],
        "secondTrack": bool(f and f.get("reference")),
        "cached": cached_ranges(job, params, tl) if job["status"] == "done" else {"echo": [], "noise": [], "light": [], "heavy": [], "rebuild": []},
        "suggestions": read_suggestions(job["id"]),
        "run": (params.get("run") or {}).get("op"),
        "lastRun": e.get("lastRun"),
        "expiresAt": job["expires_at"] if job["status"] == "done" else None,
        "editedAt": job.get("edited_at") or job["finished_at"],
        "hardDeadlineAt": jobs.hard_deadline(job),
        "cacheBytes": cache_size(job["id"]),
        "preview": None,
    }
    if f and f.get("source") and job["status"] == "done":
        out["preview"] = {"layers": ["source", "echo", "noise", "rebuild"], "samples": f["n"], "channels": f["channels"],
                          "sourceScale": f["source"]["scale"], "stretchMaxSeconds": jobs.STRETCH_MAX_SECONDS}
    return out


def save_timeline(user_id: str, job_id: str, api_key_id: str | None, raw: object) -> dict:
    """PUT /api/jobs/{id}/timeline: check and keep the working timeline. An edit: the job's files are kept 2 hours from now."""
    job, params = job_for_edit(user_id, job_id, api_key_id)
    seconds = file_seconds(job, params)
    tl = parse_timeline(raw, seconds)
    with db.tx() as c:
        row = c.execute("SELECT params_json, status FROM jobs WHERE id = ?", (job_id,)).fetchone()
        cur = json.loads(row["params_json"])
        if row["status"] != "done" or cur.get("run"):
            raise jobs.JobError("This job is being processed. Wait for it to finish.", 409, code=BUSY)
        edit = dict(cur.get("edit") or {})
        edit["timeline"] = tl.to_json()
        cur["edit"] = edit
        c.execute("UPDATE jobs SET params_json = ? WHERE id = ?", (json.dumps(cur), job_id))
    jobs.touch(job_id)
    fresh = db.one("SELECT expires_at FROM jobs WHERE id = ?", (job_id,))
    return {"timeline": tl.to_json(), "expiresAt": fresh["expires_at"]}


def estimate_seconds(q: dict, seconds: float, channels: int) -> float:
    """A rough wall-clock time for the run of a quote: what its missing blocks take, a decode and the render."""
    ed = _ed()
    work = sum(ed.compute_estimate(q["plan"]).values()) / 1.5
    return work + jobs.SPEED["leveling"] * seconds * channels


def start_run(user_id: str, job_id: str, api_key_id: str | None, op: str, raw: object | None = None, *, max_cost_micros: int | None = None,
              audio_format: str | None = None) -> dict:
    """Queue an edit run on a finished job: `process` (what the quote says: the missing blocks, then the render), `export` (the render
    alone, only for a quote of nothing) or `suggest`. The same limits as any job: the account's simultaneous jobs (quotes do not
    count), the API key's spending cap, the disk; the money is held at the quote and the run is charged for what it ran."""
    from . import hostload

    settings = get_settings()
    job, params = job_for_edit(user_id, job_id, api_key_id)
    jobs.check_active_limit(db.conn(), user_id)  # (again in the transaction that stores the run; this is the cheap early answer)
    seconds = file_seconds(job, params)
    edit = params.get("edit") or {}
    f = facts_of(job, params)
    fmt = audio_format or params["options"].get("audio_format", "flac")
    if fmt not in jobs.AUDIO_FORMATS:
        raise jobs.JobError("unsupported output format")
    rates = step_rates()
    if op == "suggest":
        tl = saved_timeline(job, params, seconds)
        q = None
        micros = hold = 0
        run = {"op": op, "timeline": tl.to_json(), "previous": None, "lines": [], "asked": {}, "gpu": [], "echo": False}
        est_seconds = jobs.SPEED["suggest"] * seconds
    else:
        tl = parse_timeline(raw, seconds) if raw is not None else saved_timeline(job, params, seconds)
        q = make_quote(job, params, tl, rates=rates)
        if q["unchanged"] and fmt == params["options"].get("audio_format", "flac"):
            raise jobs.JobError("Nothing would change: the result files already hold this timeline.", 400, code=NOTHING_TO_CHANGE)
        if op == "export" and q["needsProcess"] and not q["unchanged"]:
            raise jobs.JobError("This has to be processed first (the quote has something to make); export is for changes that need nothing.",
                                409, code=NEEDS_PROCESSING, extra={"quote": public_quote(q)})
        micros = hold = q["micros"]
        if q["approximate"]:
            hold = max(micros, make_quote(job, params, tl, rates=rates, provisional="heavy")["micros"])
        if max_cost_micros is not None and (micros if q["approximate"] else hold) > max_cost_micros:
            raise jobs.JobError(f"This now costs about ${hold / 1e6:.2f}, more than the ${max_cost_micros / 1e6:.2f} you were quoted.", 409, code=PRICE_CHANGED)
        if q["gpu"] and "rebuild" in q["gpu"]:
            from .broker import gpu_worker_up

            if not gpu_worker_up():
                raise jobs.JobError(jobs.REBUILD_PAUSED)
        est_seconds = estimate_seconds(q, seconds, f["channels"])
        rendered = rendered_timeline(job, params, seconds)
        prev, _ = for_planning(rendered, edit.get("resolved")) if rendered is not None else (None, False)
        run = {"op": op, "timeline": tl.to_json(), "previous": prev.to_json() if prev is not None else None, "lines": q["lines"],
               "asked": q["asked"], "gpu": q["gpu"], "echo": bool(tl.layers["echo"]), "rates": rates}
    need_seconds = 2 * (est_seconds + jobs.MODEL_LOAD_SECONDS)
    from . import rerun

    if jobs.hard_deadline(job) - db.now() < need_seconds + rerun.DEADLINE_MARGIN_SECONDS:
        raise jobs.JobError("This one is about to be cleared away, so there isn’t time to process it again. Start it again from your file.", 409, code=EXPIRING)
    channels = f["channels"]
    disk_need = (jobs.scratch_bytes("denoise", seconds, channels, echo=run["echo"]) + (growth_bytes(q["plan"], channels) if q else 0)
                 + jobs.DISK_RESERVE)
    if hostload.get(settings.data_dir).snapshot().disk_free < disk_need:
        raise jobs.JobError("We're short on storage space right now. Please try again a bit later.", 507)
    now = db.now()
    run.update(id=db.new_id()[:12], audio_format=fmt, hold=hold, est_seconds=int(est_seconds), startedAt=now)
    new_params = {**params, "run": run, "options": {**params["options"], "audio_format": fmt}}
    if op != "suggest":  # the timeline that is processed or exported is the job's working timeline from now on
        new_params["edit"] = {**edit, "timeline": tl.to_json()}
    with db.tx() as c:
        jobs.check_active_limit(c, user_id)
        hold_id = None
        if hold > 0:
            keycaps.check(c, api_key_id, hold)
            try:
                hold_id = ledger.create_hold(user_id, hold, "job", job_id, c=c)
            except ledger.InsufficientFunds as e:
                raise jobs.JobError(f"This costs about ${hold / 1e6:.2f}; your available balance is ${e.available / 1e6:.2f}.", 402) from e
        n = c.execute(
            "UPDATE jobs SET status = 'queued', next_stage = 'edit', params_json = ?, hold_id = ?, est_cost_micros = ?, progress = 0, stage = NULL,"
            " error = NULL, attempts = 0, worker_id = NULL, started_at = NULL, wait_seconds = NULL, queued_at = ?, edited_at = ?, expires_at = ?"
            " WHERE id = ? AND status = 'done'",
            (json.dumps(new_params), hold_id, hold, now, now, jobs.expiry(now, job.get("uploaded_at") or job["created_at"]), job_id),
        ).rowcount
        if not n:
            raise jobs.JobError("This job is being processed. Wait for it to finish.", 409, code=BUSY)
    jobs.invalidate_queue_cache()
    return {"id": job_id, "run": {"id": run["id"], "op": op}, "estCostMicros": micros, "holdMicros": hold, "free": micros == 0,
            "approximate": bool(q and q["approximate"]), "expiresAt": jobs.expiry(now, job.get("uploaded_at") or job["created_at"])}
