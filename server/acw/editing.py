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
# Subtitles per part, and the parts the browser does itself (timeline version 2)
# ---------------------------------------------------------------------------------------------------------------------
BROWSER = "browser"  # the job's folder of what the browser uploaded: the cleaned noise parts (FLAC) and the cues of free subtitles (JSON)
SUBS = "subs"  # ... and of the subtitle parts' results, one folder per part (segments.json, segments.<lang>.json, asr_chunks/)
NEEDS_UPLOADS = "needs_uploads"
CUES_MAX_BYTES = 1_000_000  # the JSON of one free part's cues
NOISE_BYTES_PER_SECOND = 48_000 * 2 * 4  # most a FLAC of 48 kHz audio can take per second and channel (32-bit samples, no compression)
TRANSCRIBE_SPEED = {"standard": "transcribe_local", "accurate": "transcribe_premium"}


def subtitle_rates() -> dict[str, int]:
    """Micros per minute of the subtitle steps at today's price list (pinned with the job, like `step_rates`): Standard and Accurate
    transcription per part, and a translation per part and language."""
    return {"transcribe_standard": pricing.rate("transcribe", "local").micros_per_minute,
            "transcribe_accurate": pricing.rate("transcribe", "premium").micros_per_minute,
            "translate": pricing.rate("transcribe_translate", "standard").micros_per_minute}


def _ms(t: float) -> int:
    return int(round(float(t) * 1000))


def noise_name(start: float, end: float) -> str:
    return f"noise-{_ms(start)}-{_ms(end)}.flac"


def cues_name(start: float, end: float) -> str:
    return f"cues-{_ms(start)}-{_ms(end)}.json"


def browser_dir(job_id: str) -> Path:
    return jobs.job_dir(job_id) / BROWSER


def subtitle_key(part) -> str:
    """The folder name of a subtitle part's results: where it is, its quality and language (what its transcription depends on)."""
    st = part.settings
    return f"{_ms(part.start)}-{_ms(part.end)}-{st['quality'][0]}-{st['language'] or 'auto'}"


def subtitle_dir(job_id: str, part) -> Path:
    return jobs.job_dir(job_id) / SUBS / subtitle_key(part)


def _read_json(path: Path, limit: int = 2_000_000):
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


def wants_uploads(raw: object) -> bool:
    """A stored timeline (JSON) has parts that need something from the browser: a noise part it cleans, subtitles it makes."""
    layers = (raw or {}).get("layers") if isinstance(raw, dict) else None
    layers = layers or {}
    return (any(isinstance(p, dict) and p.get("browser") for p in layers.get("noise") or [])
            or any(isinstance(p, dict) and p.get("quality") == "free" for p in layers.get("subtitles") or []))


def all_uploads(tl) -> list[dict]:
    """What the browser has to send for a timeline: the cleaned audio of each of its browser noise parts and the cues of each free
    subtitle part, `{kind: "noise" | "cues", start, end}` in time order."""
    out = [{"kind": "noise", "start": p.start, "end": p.end} for p in tl.browser_parts()]
    out += [{"kind": "cues", "start": p.start, "end": p.end} for p in tl.free_parts()]
    return sorted(out, key=lambda n: (n["start"], n["end"], n["kind"]))


def cues_of(job_id: str, part) -> list[dict] | None:
    """The cues the browser uploaded for a free part, as they were checked; None when none (or they were for another language)."""
    data = _read_json(browser_dir(job_id) / cues_name(part.start, part.end), CUES_MAX_BYTES + 1000)
    if not isinstance(data, dict) or data.get("language") != part.settings.get("language") or not isinstance(data.get("cues"), list):
        return None
    return data["cues"]


def needs_for(job_id: str, tl) -> list[dict]:
    """What the timeline still needs the browser to upload for this job: its browser noise parts and free subtitle parts that have nothing
    stored yet (or, for a free part, only cues made for another language)."""
    have_dir = browser_dir(job_id)
    out = []
    for n in all_uploads(tl):
        if n["kind"] == "noise":
            if not _regular(have_dir / noise_name(n["start"], n["end"])):
                out.append(n)
        elif cues_of(job_id, next(p for p in tl.free_parts() if p.start == n["start"] and p.end == n["end"])) is None:
            out.append(n)
    return out


def pending_needs(job: dict, params: dict) -> list[dict] | None:
    """What the job's working timeline still needs the browser to send; None when its timeline has no such part (so there is nothing to
    report, and the job is exactly the job it always was)."""
    raw = (params.get("edit") or {}).get("timeline")
    if not wants_uploads(raw):
        return None
    try:
        tl = _tl().Timeline.from_json(raw, seconds=file_seconds(job, params))
    except _tl().TimelineError:
        return None
    return needs_for(job["id"], tl)


def uploads_fingerprint(job_id: str, tl) -> tuple[str, str]:
    """(the audio the browser cleaned, the subtitles with the cues the browser made) a timeline stands on, as short texts; "" for none.
    A render is remembered with them (`edit.browser_rendered`, `edit.subs_rendered`): what a quote compares to know the result files are
    current."""
    import hashlib

    def stat_of(name: str) -> list:
        try:
            st = os.stat(browser_dir(job_id) / name)
            return [name, st.st_size, st.st_mtime_ns]
        except OSError:
            return [name, None]

    def digest(obj) -> str:
        return hashlib.sha1(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]

    audio = [stat_of(noise_name(p.start, p.end)) for p in tl.browser_parts()]
    subs = [p.to_json() for p in tl.parts(_tl().SUBTITLES)]
    cues = [stat_of(cues_name(p.start, p.end)) for p in tl.free_parts()]
    return (digest(audio) if audio else "", digest([subs, cues]) if subs else "")


def _cost(rate_per_minute: int, seconds: int) -> int:
    return step_micros(rate_per_minute, seconds, 0.0)


def subtitle_plan(job_id: str | None, tl, rates: dict[str, int]) -> dict:
    """What the subtitles layer needs made, and what that costs: for each part `{key, start, end, quality, language, translate, free,
    transcribe (a paid transcription has to run), translate_run (the translations that have to run)}`. `job_id` None: from nothing (a
    job's first run); else whatever the job's folder already holds is not made again and not charged again. Free parts cost nothing.
    Price of a part = its quality's rate x its whole seconds (rounded up) + the translation rate x the same seconds for each language
    to make. `rates`: `subtitle_rates()`, as pinned."""
    parts, lines_by = [], {"standard": [0, 0], "accurate": [0, 0]}
    tr_seconds, tr_micros, langs = 0, 0, set()
    micros = 0
    for p in tl.parts(_tl().SUBTITLES):
        st = p.settings
        free = st["quality"] == "free"
        have = None if free or job_id is None else _read_json(subtitle_dir(job_id, p) / "segments.json", 20_000_000)
        spoken = have.get("language") if isinstance(have, dict) else None
        e = {"key": subtitle_key(p), "start": p.start, "end": p.end, "quality": st["quality"], "language": st["language"],
             "translate": list(st["translate"]), "free": free, "transcribe": not free and not isinstance(have, dict), "translate_run": []}
        if not free:
            e["translate_run"] = [t for t in st["translate"] if t != spoken
                                  and (job_id is None or not _regular(subtitle_dir(job_id, p) / f"segments.{t}.json"))]
        secs = max(1, math.ceil(p.seconds - 1e-9))
        if e["transcribe"]:
            m = _cost(rates[f"transcribe_{st['quality']}"], secs)
            lines_by[st["quality"]][0] += secs
            lines_by[st["quality"]][1] += m
            micros += m
        for t in e["translate_run"]:
            m = _cost(rates["translate"], secs)
            tr_seconds, tr_micros = tr_seconds + secs, tr_micros + m
            langs.add(t)
            micros += m
        parts.append(e)
    lines = []
    for quality, (secs, m) in lines_by.items():
        if secs:
            r = pricing.rate("transcribe", "local" if quality == "standard" else "premium")
            lines.append({"item": "transcribe" if quality == "standard" else "transcribe_accurate", "label": r.label, "quantity": 1,
                          "seconds": secs, "fixedSeconds": 0.0, "usdPerMinute": rates[f"transcribe_{quality}"] / pricing.MICROS_PER_USD, "micros": m})
    if tr_seconds:
        lines.append({"item": "translate", "label": pricing.rate("transcribe_translate", "standard").label, "quantity": len(langs),
                      "seconds": tr_seconds, "fixedSeconds": 0.0, "usdPerMinute": rates["translate"] / pricing.MICROS_PER_USD, "micros": tr_micros})
    work = sum(e["transcribe"] or bool(e["translate_run"]) for e in parts)
    return {"parts": parts, "micros": micros, "lines": lines, "work": work,
            "seconds": {"standard": lines_by["standard"][0], "accurate": lines_by["accurate"][0], "translate": tr_seconds}}


def subtitle_charge(job_id: str, plan: dict, rates: dict[str, int]) -> int:
    """What the subtitle work of `plan` costs once it has run: the transcriptions and translations that were planned and exist now (a
    translation that could not be made is not charged)."""
    total = 0
    for e in plan["parts"]:
        if e["free"]:
            continue
        d = jobs.job_dir(job_id) / SUBS / e["key"]
        secs = max(1, math.ceil(e["end"] - e["start"] - 1e-9))
        if e["transcribe"] and _regular(d / "segments.json"):
            total += _cost(rates[f"transcribe_{e['quality']}"], secs)
        total += sum(_cost(rates["translate"], secs) for t in e["translate_run"] if _regular(d / f"segments.{t}.json"))
    return total


def subtitle_seconds(plan: dict) -> float:
    """A rough wall-clock time for the subtitle work of a plan (the queue estimate)."""
    S = jobs.SPEED
    return sum(S[TRANSCRIBE_SPEED[e["quality"]]] * (e["end"] - e["start"]) for e in plan["parts"] if e["transcribe"])


def subtitle_spec(tl) -> dict:
    """What the runner is told about the subtitles layer: every part (it makes what its folder lacks and puts them all together)."""
    return {"parts": [{"key": subtitle_key(p), "start": p.start, "end": p.end, **{k: p.settings[k] for k in ("quality", "language", "translate")}}
                      for p in tl.parts(_tl().SUBTITLES)]}


def browser_spec(job_id: str, tl) -> list[dict]:
    """The browser-cleaned noise parts the runner reads: where each is and the file the browser sent for it."""
    return [{"start": p.start, "end": p.end, "file": f"{BROWSER}/{noise_name(p.start, p.end)}"} for p in tl.browser_parts()]


def has_subtitle_layer(params: dict) -> bool:
    """The job's working timeline has subtitle parts (so its subtitles are made per part, not for the whole file)."""
    layers = ((params.get("edit") or {}).get("timeline") or {}).get("layers") or {}
    return bool(layers.get("subtitles"))


def local_asr_wanted(params: dict) -> bool:
    """The job's next run transcribes with the local recogniser (which needs its memory): Standard subtitle parts that are still to make."""
    run = params.get("run")
    if run:
        return any(e["transcribe"] and e["quality"] == "standard" for e in (run.get("subs") or {}).get("parts", []))
    layers = ((params.get("edit") or {}).get("timeline") or {}).get("layers") or {}
    return any(p.get("quality") == "standard" for p in layers.get("subtitles") or [])


def subtitle_cloud_phase(job_id: str, params: dict) -> str | None:
    """What the subtitle parts still need from the web process, which alone reaches the internet: "asr" (an Accurate part whose speech
    chunks wait for the transcription), "translate" (a translation still to make) or None (nothing: the files can be put together)."""
    spec = subtitle_cloud_work(job_id, params)
    if spec["asr"]:
        return "asr"
    return "translate" if spec["translate"] else None


def subtitle_cloud_work(job_id: str, params: dict) -> dict:
    """{"asr": [part dicts whose chunks wait], "translate": [(part dict, [languages]) still to make]}, from the job's folder."""
    parts = (params.get("run") or {}).get("subs", {}).get("parts") if params.get("run") else None
    if parts is None:
        tl = _tl().Timeline.from_json(params["edit"]["timeline"], seconds=float(params["edit"]["timeline"]["seconds"]))
        parts = subtitle_plan(None, tl, params.get("subtitle_rates") or subtitle_rates())["parts"]
    out: dict = {"asr": [], "translate": []}
    base = jobs.job_dir(job_id) / SUBS
    for e in parts:
        if e["free"]:
            continue
        d = base / e["key"]
        done = _regular(d / "segments.json")
        if e["quality"] == "accurate" and not done and (d / "asr_chunks" / "index.json").is_file():
            out["asr"].append(e)
            continue
        if not done:
            continue
        spoken = (_read_json(d / "segments.json", 20_000_000) or {}).get("language")
        todo = [t for t in e["translate"] if t != spoken and not _regular(d / f"segments.{t}.json")]
        if todo:
            out["translate"].append((e, todo))
    return out


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


def parse_for_job(params: dict, raw: object, seconds: float):
    """`parse_timeline` for a timeline of an existing job: one that already has its subtitles made for the whole file (a transcription
    tier of its own) cannot also have subtitle parts. (Echo parts on a file without a second track are refused when it is priced: editor.Plan.)"""
    tl = parse_timeline(raw, seconds)
    if tl.layers["subtitles"] and params.get("transcribe_tier"):
        raise bad("This job's subtitles were set for the whole file, so a timeline can't have subtitle parts. Make the subtitles in a new job.",
                  code="timeline_options")
    return tl


def wants_echo(params: dict) -> bool:
    """Does the job ask for echo removal: the whole-file option, or echo parts in the timeline it was made with?"""
    if (params.get("options") or {}).get("echo_reference_track"):
        return True
    layers = (((params.get("edit") or {}).get("timeline") or {}).get("layers") or {}) if (params.get("edit") or {}).get("first_run") else {}
    return bool(layers.get("echo"))


def editable_kind(job: dict) -> bool:
    """Every job made with a timeline (all of them are: the API's options become one, timeline_from_options); a conversion has none."""
    return job["kind"] in ("denoise", "denoise_transcribe", "transcribe")


# The subtitle quality of a timeline part for each transcription tier of the API's options.
QUALITY_OF_TIER = {"local": "standard", "premium": "accurate"}


def timeline_from_options(kind: str, denoise_tier: str | None, transcribe_tier: str | None, options: dict, seconds: float) -> tuple[dict, dict]:
    """A job asked for with options (the API's whole-file choices, already checked: jobs.validate_options) as the timeline they stand for, so
    every job is made one way (editrun.render, and the subtitles part by part): the sound steps on the whole file
    (Timeline.from_clean_options), and the subtitles as one part over the whole file with the quality, language and
    translations asked for. Returns (the timeline as JSON, the options the timeline does not hold)."""
    from acw_common import pipeline

    raw: dict = {"version": 1, "seconds": seconds, "layers": {}}
    if kind in ("denoise", "denoise_transcribe"):
        c = pipeline.CleanOptions(atten_lim_db=float(options.get("atten_lim_db", 100.0)), loudness_lufs=options.get("loudness_lufs", -16.0),
                                  leveling=bool(options.get("leveling", True)), noise=options.get("noise", "auto"),
                                  restore_voice=denoise_tier == "rebuild", denoise=denoise_tier != pricing.NO_DENOISE)
        raw = _tl().Timeline.from_clean_options(seconds, c, echo=bool(options.get("echo_reference_track"))).to_json()
    subs = kind in ("transcribe", "denoise_transcribe") and transcribe_tier in QUALITY_OF_TIER
    if subs:
        raw = {**raw, "version": _tl().VERSION_2, "layers": {**raw["layers"], "subtitles": [
            {"start": 0.0, "end": raw["seconds"], "quality": QUALITY_OF_TIER[transcribe_tier], "language": options.get("language"),
             "translate": list(options.get("translate") or [])}]}}
    rest = {k: v for k, v in options.items() if k not in REPLACED_OPTIONS and not (subs and k in ("language", "translate"))}
    return raw, rest


def subtitles_into_timeline(raw: object, transcribe_tier: str | None, options: dict, seconds: float) -> tuple[object, dict, str | None]:
    """A timeline sent with whole-file subtitle choices (a transcription tier, options.language and .translate) and no subtitle parts of its
    own: the subtitles become one part over the whole file, so they are made the one way (part by part). Returns (the timeline, the options
    without those choices, the tier left: None when it went into the timeline)."""
    layers = raw.get("layers") if isinstance(raw, dict) else None
    if transcribe_tier not in QUALITY_OF_TIER or not isinstance(layers, dict) or layers.get("subtitles"):
        return raw, options, transcribe_tier
    end = raw.get("seconds") if isinstance(raw.get("seconds"), (int, float)) and not isinstance(raw.get("seconds"), bool) else seconds
    part = {"start": 0.0, "end": end, "quality": QUALITY_OF_TIER[transcribe_tier], "language": (options or {}).get("language"),
            "translate": list((options or {}).get("translate") or [])}
    rest = {k: v for k, v in (options or {}).items() if k not in ("language", "translate")}
    return {**raw, "version": _tl().VERSION_2, "layers": {**layers, "subtitles": [part]}}, rest, None


def options_price(kind: str, denoise_tier: str | None, transcribe_tier: str | None, options: dict, seconds: float) -> tuple[int, list[dict]]:
    """What a job asked for with options costs for `seconds`, at today's prices, and its lines: the price of the timeline they stand for (the
    same as a job made with that timeline)."""
    raw, _ = timeline_from_options(kind, denoise_tier, transcribe_tier, options, seconds)
    tl = parse_timeline(raw, seconds)
    micros, lines = price_timeline(tl, seconds, step_rates())[:2] if any(tl.layers[n] for n in _tl().LAYERS) else (0, [])
    if tl.layers["subtitles"]:
        sub = subtitle_plan(None, tl, subtitle_rates())
        micros, lines = micros + sub["micros"], [*lines, *sub["lines"]]
    return micros, lines


def refit_whole_file(job_id: str, params: dict, seconds: float) -> dict:
    """A job asked for with options covers the whole file, whatever its length turns out to be: once the file is read, its timeline is made
    again for the length it has (`params["whole_file"]`: what was asked for). Returns the parameters."""
    w = params.get("whole_file")
    if not w:
        return params
    raw, _ = timeline_from_options(w["kind"], w["denoise_tier"], w["transcribe_tier"], w["options"], seconds)
    return jobs.update_params(job_id, edit={**(params.get("edit") or {}), "timeline": raw})


def saved_timeline(job: dict, params: dict, seconds: float):
    """The job's working timeline: what was last saved, or the one it was made with (every job has one); re-based on the recording's
    length."""
    return _tl().Timeline.from_json((params.get("edit") or {}).get("timeline") or {"version": 1, "seconds": seconds, "layers": {}}, seconds=seconds)


def rendered_timeline(job: dict, params: dict, seconds: float):
    """The timeline whose audio the job's result files hold (None: not known, so everything is made again)."""
    e = params.get("edit") or {}
    return _tl().Timeline.from_json(e["rendered"], seconds=seconds) if e.get("rendered") else None


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
    browser_fp, subs_fp = uploads_fingerprint(job["id"], tl)
    subs = subtitle_plan(job["id"], tl, params.get("subtitle_rates") or subtitle_rates())
    # (what the sound depends on is compared without the subtitles; the audio a browser sent counts as part of it)
    audio_same = (rendered is not None and with_removers(tl.audio(), edit.get("resolved"))[0] == rendered.audio()
                  and browser_fp == (edit.get("browser_rendered") or ""))
    subs_same = subs_fp == (edit.get("subs_rendered") or "") and not subs["work"]
    unchanged = audio_same and subs_same
    if audio_same:
        billable, total, lines = {}, 0, []  # the result files already hold exactly this sound
    else:
        billable = ed.billable(plan)
        total, lines = price(billable, rates)
    total, lines = total + subs["micros"], [*lines, *subs["lines"]]
    # A video made on our side (the whole video was uploaded, or a sound asked for as a video) is made again with a new sound or new
    # subtitles, the way the first run made it: its picture made again is charged again (a video conversion, jobs.convert_cost).
    opts = params.get("options") or {}
    video = (bool(params.get("has_video")) or bool(opts.get("video_out"))) and bool(opts.get("output_video", True)) and not unchanged
    convert = jobs.convert_cost(params, seconds) if video else 0
    if convert:
        total, lines = total + convert, [*lines, jobs.convert_line(params, seconds)]
    rebuild_s = _blocks().union_seconds(plan.asked.get("rebuild", []))
    if rebuild_s > pricing.MAX_REBUILD_SECONDS:
        raise bad(f"Voice rebuild takes up to {pricing.MAX_REBUILD_SECONDS // 60} minutes of a recording at a time; this timeline asks for {rebuild_s / 60:.0f}.")
    summary = plan.summary()
    steps = {s: {"blocks": r["blocks"], "seconds": billable.get(s, {}).get("seconds", 0)} for s, r in summary.items()
             if r["blocks"] or billable.get(s)}
    asked = asked_summary(plan)
    if subs["work"]:
        todo = [e for e in subs["parts"] if e["transcribe"] or e["translate_run"]]
        steps["subtitles"] = {"parts": len(todo), "seconds": sum(subs["seconds"].values())}
        asked["subtitles"] = round(sum(e["end"] - e["start"] for e in todo), 3)
    return {
        "free": total == 0,
        "unchanged": unchanged,
        "audioSame": audio_same,
        "approximate": approx,
        "micros": total,
        "lines": lines,
        "billable": billable,
        "subs": subs,
        "subsSame": subs_same,
        "needs": needs_for(job["id"], tl) if wants_uploads(tl.to_json()) else None,
        "steps": steps,
        "misses": len(plan.misses),
        "asked": asked,
        "needsProcess": bool(plan.misses or billable or subs["work"]),
        "rates": rates,
        "plan": plan,
        "timeline": tl,
        "gpu": gpu_stages_of(plan),
        "video": video,
        "convertMicros": convert,
    }


def gpu_stages_of(plan) -> list[str]:
    """The GPU steps a plan's missing blocks need: the card runs Voice rebuild and the heavy remover."""
    return [s for s in ("rebuild", "heavy") if any(t.step == s for t in plan.misses)]


def public_quote(q: dict) -> dict:
    """The quote as the API says it."""
    out = {
        "estCostMicros": q["micros"], "estCostUsd": q["micros"] / pricing.MICROS_PER_USD, "free": q["free"], "unchanged": q["unchanged"],
        "needsProcess": q["needsProcess"], "approximate": q["approximate"], "lines": q["lines"], "steps": q["steps"], "asked": q["asked"],
        "timeline": q["timeline"].to_json(),
    }
    if q.get("needs") is not None:
        out["needs"] = q["needs"]  # what the timeline still needs the browser to upload (empty: everything is in)
    return out


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
        r = "browser" if p.settings.get("browser") else p.settings.get("remover")  # (a browser part has no model blocks: not cached here)
        if r == "auto":
            r = (resolved or {}).get(part_key(p.start, p.end))
        if r != "browser" and r not in ("light", "heavy"):
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
            if r == "browser":
                continue
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
# Outlines: the loudness of what a layer holds, for drawing it
# ---------------------------------------------------------------------------------------------------------------------
PEAKS_RATE = 4  # values per second of the recording
PEAKS_WINDOW = SR // PEAKS_RATE
PEAKS_BUDGET_SECONDS = 25.0  # how long one request works on blocks that have no outline yet (the rest follow on the next call)
PEAKS_PER_RUN = 24  # blocks one confined run reads


def _read_outline(path: Path) -> dict | None:
    data = _read_json(path, 4_000_000)
    if isinstance(data, dict) and isinstance(data.get("first"), int) and isinstance(data.get("p"), list) and all(isinstance(v, (int, float)) for v in data["p"]):
        return data
    return None


def _write_outline(path: Path, data: dict) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")))
    os.replace(tmp, path)


def peaks_of(job: dict, params: dict, layer: str) -> dict | None:
    """The outline of a layer (`source`, `echo`, `noise` or `rebuild`, the stems of the preview, on its sample grid): `{"rate": 4,
    "seconds", "peaks": [...]}`, one value per quarter second, the largest absolute sample over all channels (0 to 1, three decimals),
    `None` where the layer is not in the cache for that moment. Made from the cached blocks (decoded by the confined helper, never here)
    and kept per block next to it (`blocks/<name>.peaks`, the recording's in `kept-source.peaks`), so a block is read once. None when
    nothing of the layer is cached; `pending` is added when the time of this call ran out before every block had its outline."""
    from . import stretch

    f = (params.get("edit") or {}).get("facts")
    n = int(f["n"])
    windows = -(-n // PEAKS_WINDOW)
    d = jobs.job_dir(job["id"])
    out: list[float | None] = [None] * windows
    pending = False
    if layer == "source":
        entry = f.get("source")
        if not entry or not _regular(d / entry["file"]):
            return None
        side = d / (Path(entry["file"]).stem + ".peaks")
        got = _read_outline(side)
        if got is None:
            got = stretch.peaks(d, {"file": entry["file"], "channels": f["channels"], "scale": float(entry["scale"])}, timeout=240.0)
            if not isinstance(got.get("p"), list):
                raise stretch.Failed("the outline could not be read")
            _write_outline(side, {"first": 0, "p": got["p"]})
        for i, v in enumerate(got["p"][:windows]):
            out[i] = v
    else:
        plan = plan_stretch(job, params, layer, 0, n)
        wanted: dict[str, dict] = {}
        for chan in plan["channels"]:
            for b in chan or []:
                wanted.setdefault(b["name"], b)
        have: dict[str, dict] = {}
        todo: list[dict] = []
        for name, b in wanted.items():
            got = _read_outline(d / BLOCKS / f"{name}.peaks")
            if got is None:
                todo.append({"name": name, "at": b["at"]})
            else:
                have[name] = got
        import time as _time

        t0 = _time.monotonic()
        for i in range(0, len(todo), PEAKS_PER_RUN):
            if _time.monotonic() - t0 > PEAKS_BUDGET_SECONDS:
                pending = True
                break
            made = stretch.peaks(d, {"blocks": todo[i:i + PEAKS_PER_RUN]})
            for name, got in made.items():
                if name in wanted and isinstance(got, dict) and isinstance(got.get("first"), int) and isinstance(got.get("p"), list):
                    _write_outline(d / BLOCKS / f"{name}.peaks", {"first": got["first"], "p": got["p"]})
                    have[name] = got
        unknown = [False] * windows  # under a block that has no outline yet (the time ran out)
        for chan in plan["channels"]:
            for b in chan or []:
                got = have.get(b["name"])
                for w in range(b["from"] // PEAKS_WINDOW, min(windows, -(-b["to"] // PEAKS_WINDOW))):
                    if got is None:
                        unknown[w] = True
                        continue
                    i = w - got["first"]
                    if 0 <= i < len(got["p"]):
                        out[w] = got["p"][i] if out[w] is None else max(out[w], got["p"][i])
        # a window the cache lacks (in all of it) is absent; one with nothing missing but no block under it is silence the removers skip
        absent = [False] * windows
        for a, z in plan["missing"]:
            for w in range(-(-a // PEAKS_WINDOW), min(windows, z // PEAKS_WINDOW)):
                absent[w] = True
        for w in range(windows):
            if absent[w] or unknown[w]:
                out[w] = None
            elif out[w] is None:
                out[w] = 0.0
    if all(v is None for v in out):
        return None
    result = {"rate": PEAKS_RATE, "seconds": round(n / SR, 3), "peaks": [None if v is None else round(min(1.0, float(v)), 3) for v in out]}
    if pending:
        result["pending"] = True
    return result


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
            if all(p.settings.get("browser") or p.settings.get("remover") in ("light", "heavy") for p in rt.layers["noise"]):
                out["rendered"] = rt.audio().to_json()
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


def absorb(job: dict, params: dict, report: dict, *, op: str | None = None, processed: object | None = None) -> dict:
    """The job's parameters with a run's report in them: the facts, the removers that were decided, the timeline that was rendered
    (the one the result files hold), a line in its history, and the browser's uploads and subtitles the result files stand on
    (`processed`: the timeline that was run, default the job's working one)."""
    edit = dict(params.get("edit") or {})
    raw = processed if processed is not None else edit.get("timeline")
    if op != "suggest":  # (a search for suggestions renders nothing: the result files stand on what they stood on)
        for k in ("browser_rendered", "subs_rendered"):
            edit.pop(k, None)
    if op != "suggest" and raw and (wants_uploads(raw) or ((raw.get("layers") or {}).get("subtitles") if isinstance(raw, dict) else None)):
        try:
            f = report.get("facts") or edit.get("facts")
            fingerprints = uploads_fingerprint(job["id"], _tl().Timeline.from_json(raw, seconds=f["n"] / SR if f else file_seconds(job, params)))
        except (_tl().TimelineError, KeyError, TypeError):
            fingerprints = ("", "")
        for k, v in zip(("browser_rendered", "subs_rendered"), fingerprints):
            if v:
                edit[k] = v
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


# ---------------------------------------------------------------------------------------------------------------------
# A job made with a timeline
# ---------------------------------------------------------------------------------------------------------------------
REPLACED_OPTIONS = ("atten_lim_db", "noise", "leveling", "loudness_lufs", "echo_reference_track", "automatic")


def steps_in(tl) -> int:
    """How many cached steps a timeline can fill (for the disk a job is promised)."""
    return max(1, sum(1 for n in ("echo", "noise", "rebuild") if tl.layers[n]))


def options_kind(kind: str, tl) -> str:
    """The kind whose options a job made with `tl` is checked as: with subtitle parts the subtitle choices are the parts' own (no tier, no
    whole-file language or translations), so its options are those of a job that cleans the sound."""
    return "denoise" if tl.layers[_tl().SUBTITLES] else kind


def prepare_timeline_job(kind: str, denoise_tier: str | None, options: dict, claimed_seconds: float, raw: object,
                         transcribe_tier: str | None = None):
    """The checks of a job made with a timeline: it needs audio steps or subtitles, replaces the choices the timeline makes (the tier,
    strength, remover, Leveling, loudness, echo removal, automatic settings; with subtitle parts also the subtitle quality, language and
    translations), and every part is inside the file. Returns (Timeline, the tier the job stores)."""
    if kind not in ("denoise", "denoise_transcribe", "transcribe"):
        raise bad("A timeline is for jobs with noise removal, Leveling or subtitles: use kind denoise, denoise_transcribe or transcribe.")
    if denoise_tier is not None:
        raise bad("A timeline says what is done to each part: leave denoiseTier out.", code="timeline_options")
    clash = [k for k in REPLACED_OPTIONS if k in (options or {})]
    if clash:
        raise bad(f"A timeline replaces these options, leave them out: {', '.join(clash)}.", code="timeline_options")
    tl = parse_timeline(raw, claimed_seconds, slack=0.1 * claimed_seconds + 5.0)
    if not any(tl.layers[n] for n in ("echo", "noise", "rebuild", "leveling", "subtitles")):
        raise bad("The timeline has no parts: it would not do anything.")
    if kind == "transcribe" and any(tl.layers[n] for n in _tl().LAYERS):
        raise bad("A transcribe job only has subtitles: use kind denoise_transcribe for sound steps too.")
    if tl.layers["subtitles"]:
        clash = [name for name, given in (("transcribeTier", transcribe_tier is not None), ("options.language", (options or {}).get("language") is not None),
                                          ("options.translate", bool((options or {}).get("translate")))) if given]
        if clash:
            raise bad(f"A timeline with subtitles says the quality, language and translations of each part: leave these out: {', '.join(clash)}.",
                      code="timeline_options")
    rebuild_s = sum(p.seconds for p in tl.layers["rebuild"])
    if rebuild_s > pricing.MAX_REBUILD_SECONDS:
        raise bad(f"Voice rebuild takes up to {pricing.MAX_REBUILD_SECONDS // 60} minutes of a recording at a time; this timeline asks for {rebuild_s / 60:.0f}.")
    if tl.layers["rebuild"]:
        from .broker import rebuild_available

        if not rebuild_available():  # (the card is the only place it runs)
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


def first_run_cost(kind: str, params: dict, seconds: float, billed: dict | None = None, job_id: str | None = None) -> int:
    """What a job made with a timeline costs for a recording of `seconds`: its parts (`billed`: the runner's report of what ran, else the
    whole timeline) at the rates pinned with the job, and the subtitles: the whole-file ones as always, or its subtitle parts (`job_id`
    given and `billed`: the ones that were made, else all of them)."""
    rates = params.get("edit_rates") or step_rates()
    tl = _tl().Timeline.from_json(params["edit"]["timeline"], seconds=seconds)
    if billed is None:
        edit = price_timeline(tl, seconds, rates)[0]
    else:
        edit = price(billed, rates)[0]
    if tl.layers["subtitles"]:
        sr = params.get("subtitle_rates") or subtitle_rates()
        plan = subtitle_plan(None, tl, sr)
        edit += plan["micros"] if billed is None or job_id is None else subtitle_charge(job_id, plan, sr)
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
    layers = tl.get("layers") or {}
    accurate = any(p.get("quality") == "accurate" for p in layers.get("subtitles") or [])
    names = {"echo": "Echo removal", "noise": "Denoising", "rebuild": "Voice rebuild", "leveling": "Leveling",
             "subtitles": "Subtitles, extra accurate" if accurate else "Subtitles"}
    have = [names[k] for k in ("rebuild", "noise", "echo", "leveling", "subtitles") if layers.get(k)]
    return " + ".join(have) or "Edited"


def timeline_seconds(raw: dict, seconds: float, channels: int) -> float:
    """A rough wall-clock time for running a timeline from nothing (the queue estimate): each layer's parts at the step's speed."""
    S = jobs.SPEED
    per = {"echo": S["echo"], "noise": S["denoise_auto"], "rebuild": S["denoise_rebuild"], "leveling": S["leveling"]}
    total = 0.0
    for name, parts in ((raw or {}).get("layers") or {}).items():
        if name == "subtitles":
            total += sum(S[TRANSCRIBE_SPEED[p["quality"]]] * max(0.0, min(seconds, p["end"]) - max(0.0, p["start"])) for p in parts
                         if p.get("quality") in TRANSCRIBE_SPEED)
        elif name in per:
            total += per[name] * sum(max(0.0, min(seconds, p["end"]) - max(0.0, p["start"])) for p in parts if not p.get("browser")) * channels
    return total


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
                   suggest=run["op"] == "suggest", video=bool(run.get("video")), audio_file=run.get("audio_file"))
    else:
        out.update(op="first", timeline=e.get("timeline") if e.get("first_run") else None, previous=None, audio_format=None,
                   suggest=0 < float(job.get("duration_seconds") or job["claimed_seconds"]) <= get_settings().suggest_max_seconds)
    raw = out.get("timeline")
    if raw and (wants_uploads(raw) or (raw.get("layers") or {}).get("subtitles") or run and run.get("subs") is not None):
        tl = _tl().Timeline.from_json(raw, seconds=file_seconds(job, params))
        out["browser"] = browser_spec(job["id"], tl)
        out["audio"] = bool(run["audio"]) if run and "audio" in run else any(tl.layers[n] for n in _tl().LAYERS)
        if (not run and tl.layers["subtitles"]) or (run and run.get("subs") is not None):  # the files put together from the parts
            out["subtitles"] = subtitle_spec(tl)
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


def job_for_uploads(user_id: str, job_id: str, api_key_id: str | None) -> tuple[dict, dict]:
    """The job and its parameters for an upload of the browser's work (cleaned noise parts, cues of free subtitles): the account's own job
    (an API key: one it made itself) that is still waiting for its uploads (`uploading`) or is finished and idle (`done`), with a
    timeline that has such parts."""
    row = db.one("SELECT * FROM jobs WHERE id = ? AND user_id = ?", (job_id, user_id)) if len(job_id) == 32 else None
    if row is None or (api_key_id is not None and row["api_key_id"] != api_key_id):
        raise jobs.JobError("Job not found.", 404, code=JOB_NOT_FOUND)
    job = dict(row)
    params = json.loads(job["params_json"])
    if not editable_kind(job):
        raise jobs.JobError("This job has no sound to edit (it makes subtitles only).", 409, code=NOT_EDITABLE)
    if job["status"] in ("expired", "failed", "canceled"):
        raise jobs.JobError("This job’s files are already cleared away. Start it again from your file.", 409, code=FILES_GONE)
    if job["status"] not in ("uploading", "done") or params.get("run"):
        raise jobs.JobError("This job is being processed. Wait for it to finish.", 409, code=BUSY)
    return job, params


def upload_target(job: dict, params: dict, kind: str, start: float, end: float):
    """The part of the job's current (working) timeline an upload is for, found by its exact range: a browser noise part (`noise`) or a free
    subtitle part (`cues`). Raises JobError 400 when the timeline has no such part."""
    raw = (params.get("edit") or {}).get("timeline")
    what = "cleaned in the browser" if kind == "noise" else "free subtitles"
    if not wants_uploads(raw):
        raise jobs.JobError(f"This job's timeline has no part with {what}.", 400)
    tl = _tl().Timeline.from_json(raw, seconds=file_seconds(job, params))
    for p in (tl.browser_parts() if kind == "noise" else tl.free_parts()):
        if _ms(p.start) == _ms(start) and _ms(p.end) == _ms(end):
            return tl, p
    raise jobs.JobError(f"No part of this job's timeline from {start:g} to {end:g} s has {what}. The parts are listed in `needs`.", 400)


def noise_expected(job: dict, params: dict, part) -> int:
    """How many samples (48 kHz) the browser's FLAC for a noise part must hold: the part's range, as the editor counts it."""
    n = None
    f = (params.get("edit") or {}).get("facts")
    if f:
        n = int(f["n"])
    a, b = int(round(part.start * SR)), int(round(part.end * SR))
    return (min(b, n) if n is not None else b) - a


def job_channels(job: dict, params: dict) -> int | None:
    """The channels of the job's audio (as the editor works on it) when the job knows them (after its file was probed), else None."""
    f = (params.get("edit") or {}).get("facts")
    return int(f["channels"]) if f else (int(params["channels"]) if params.get("channels") else None)


def flac_header(head: bytes) -> tuple[int, int, int]:
    """(sample rate, channels, samples) from the first bytes of a FLAC file (its STREAMINFO block); ValueError for anything else. Only the
    header is read here: the web process never decodes an upload, the sandboxed runner does when the audio is used."""
    if len(head) < 42 or head[:4] != b"fLaC" or (head[4] & 0x7F) != 0 or int.from_bytes(head[5:8], "big") < 34:
        raise ValueError("That is not a FLAC file.")
    info = int.from_bytes(head[18:26], "big")
    rate, channels, samples = info >> 44, ((info >> 41) & 0x7) + 1, info & ((1 << 36) - 1)
    return rate, channels, samples


def store_cues(job_id: str, part, cues: list[dict]) -> None:
    d = browser_dir(job_id)
    d.mkdir(mode=0o700, exist_ok=True)
    final = d / cues_name(part.start, part.end)
    tmp = final.with_name(final.name + ".tmp")
    tmp.write_text(json.dumps({"language": part.settings.get("language"), "cues": cues}, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, final)


def clear_chunks(job_id: str) -> None:
    """The speech chunks of Accurate parts (cut from the upload for the transcription) once the files are made."""
    root = jobs.job_dir(job_id) / SUBS
    try:
        for d in root.iterdir():
            shutil.rmtree(d / "asr_chunks", ignore_errors=True)
    except OSError:
        pass


def timeline_info(job: dict, params: dict) -> dict:
    """The answer of GET /api/jobs/{id}/timeline: the timeline, what is cached of each layer, the suggestions, how to read the preview."""
    seconds = file_seconds(job, params)
    e = params.get("edit") or {}
    f = e.get("facts")
    tl = saved_timeline(job, params, seconds)
    rendered = rendered_timeline(job, params, seconds)
    out: dict = {
        "timeline": tl.to_json(),
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
    if wants_uploads(tl.to_json()):
        out["needs"] = needs_for(job["id"], tl)
    if tl.layers["subtitles"]:  # the subtitle parts whose text exists (made, or uploaded by the browser)
        out["cached"]["subtitles"] = [[p.start, p.end] for p in tl.layers["subtitles"]
                                      if (cues_of(job["id"], p) is not None if p.settings["quality"] == "free"
                                          else _regular(subtitle_dir(job["id"], p) / "segments.json"))]
    if f and f.get("source") and job["status"] == "done":
        out["preview"] = {"layers": ["source", "echo", "noise", "rebuild"], "samples": f["n"], "channels": f["channels"],
                          "sourceScale": f["source"]["scale"], "stretchMaxSeconds": jobs.STRETCH_MAX_SECONDS}
    return out


def save_timeline(user_id: str, job_id: str, api_key_id: str | None, raw: object) -> dict:
    """PUT /api/jobs/{id}/timeline: check and keep the working timeline. An edit: the job's files are kept 2 hours from now."""
    job, params = job_for_edit(user_id, job_id, api_key_id)
    seconds = file_seconds(job, params)
    tl = parse_for_job(params, raw, seconds)
    store_timeline(job_id, tl)
    fresh = db.one("SELECT expires_at FROM jobs WHERE id = ?", (job_id,))
    out = {"timeline": tl.to_json(), "expiresAt": fresh["expires_at"]}
    if wants_uploads(out["timeline"]):
        out["needs"] = needs_for(job_id, tl)
    return out


def store_timeline(job_id: str, tl) -> None:
    """Keep `tl` as the working timeline of a finished, idle job (an edit: its files are kept 2 hours from now)."""
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


def estimate_seconds(q: dict, seconds: float, channels: int) -> float:
    """A rough wall-clock time for the run of a quote: what its missing blocks take, a decode and the render."""
    ed = _ed()
    work = sum(ed.compute_estimate(q["plan"]).values()) / 1.5
    return work + jobs.SPEED["leveling"] * seconds * channels + subtitle_seconds(q["subs"])


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
        tl = parse_for_job(params, raw, seconds) if raw is not None else saved_timeline(job, params, seconds)
        missing = needs_for(job_id, tl)
        if missing:  # the timeline becomes the working one, so the uploads that are asked for can be matched with it
            if raw is not None:
                store_timeline(job_id, tl)
            raise jobs.JobError("The browser's work for some parts of this timeline has to be uploaded first.", 409, code=NEEDS_UPLOADS,
                                extra={"needs": missing})
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
            from .broker import rebuild_available

            if not rebuild_available():
                raise jobs.JobError(jobs.REBUILD_PAUSED)
        est_seconds = estimate_seconds(q, seconds, f["channels"])
        rendered = rendered_timeline(job, params, seconds)
        prev, _ = for_planning(rendered, edit.get("resolved")) if rendered is not None else (None, False)
        run = {"op": op, "timeline": tl.to_json(), "previous": prev.to_json() if prev is not None else None, "lines": q["lines"],
               "asked": q["asked"], "gpu": q["gpu"], "echo": bool(tl.layers["echo"]), "rates": rates}
        if q["subs"]["parts"] or not q["subsSame"]:  # the subtitle work and what it costs (the rates pinned with the job)
            run["subs"] = None if q["subsSame"] else {"parts": q["subs"]["parts"]}
            run["subtitle_rates"] = params.get("subtitle_rates") or subtitle_rates()
        run["audio"] = (not q["audioSame"]) or fmt != params["options"].get("audio_format", "flac")  # the sound is rendered again
        if q["video"]:  # the video is made again too (with the new sound, or the job's sound as it is and the new subtitles)
            current = next((o["file"] for o in json.loads(job.get("outputs_json") or "[]") if o["kind"] == "audio"), None)
            run.update(video=True, audio_file=current, convert_micros=q["convertMicros"])
        if run.get("subs") and any(e["transcribe"] and e["quality"] == "standard" for e in run["subs"]["parts"]) and "transcribe" not in run["gpu"]:
            run["gpu"] = [*run["gpu"], "transcribe"]  # (the local recogniser has the card's places too)
    need_seconds = 2 * (est_seconds + jobs.MODEL_LOAD_SECONDS)
    if jobs.hard_deadline(job) - db.now() < need_seconds + jobs.DEADLINE_MARGIN_SECONDS:
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
