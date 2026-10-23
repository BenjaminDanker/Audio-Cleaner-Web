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

WHERE IT IS: this module has the timelines of a job, the quotes, what a run reports, jobs made with a timeline and the API's operations;
around it are

  edit_store      what the job keeps in its folder: the facts and the planning stand-in, the block cache, the suggestions, the disk, the
                  staged files of a run
  edit_prices     the rates of the steps and the lines a person sees
  edit_subtitles  the subtitle parts and the parts the browser does itself (its uploads)
  edit_preview    what the cache holds, the preview stretches and the outlines
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from . import db, edit_preview, edit_prices, edit_store, edit_subtitles, jobs, keycaps
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
TIMELINE_MAX_BYTES = 200_000  # the JSON of a timeline (about 2,300 parts)
KEEP_RUNS = 40  # edit runs remembered in the job's record
MAX_RESOLVED = 2000
NEEDS_UPLOADS = "needs_uploads"


# ---------------------------------------------------------------------------------------------------------------------
# Timelines of a job
# ---------------------------------------------------------------------------------------------------------------------
def bad(message: str, status: int = 400, code: str = BAD_TIMELINE) -> jobs.JobError:
    return jobs.JobError(message, status, code=code)


def parse_timeline(raw: object, seconds: float, *, slack: float = 0.5):
    """Check a timeline from a client strictly and read it for a recording of `seconds`: the size, the format, at most MAX_PARTS parts
    in a layer, and every part inside the file (up to `slack` seconds past its end, clamped). Raises JobError 400 `bad_timeline`."""
    tl = edit_store.timeline_lib()
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
                if len(parts) > edit_store.MAX_PARTS:
                    raise bad(f"At most {edit_store.MAX_PARTS} parts per layer ({name} has {len(parts)}).")
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
        raw = edit_store.timeline_lib().Timeline.from_clean_options(seconds, c, echo=bool(options.get("echo_reference_track"))).to_json()
    subs = kind in ("transcribe", "denoise_transcribe") and transcribe_tier in QUALITY_OF_TIER
    if subs:
        raw = {**raw, "version": edit_store.timeline_lib().VERSION_2, "layers": {**raw["layers"], "subtitles": [
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
    return {**raw, "version": edit_store.timeline_lib().VERSION_2, "layers": {**layers, "subtitles": [part]}}, rest, None


def options_price(kind: str, denoise_tier: str | None, transcribe_tier: str | None, options: dict, seconds: float) -> tuple[int, list[dict]]:
    """What a job asked for with options costs for `seconds`, at today's prices, and its lines: the price of the timeline they stand for (the
    same as a job made with that timeline)."""
    raw, _ = timeline_from_options(kind, denoise_tier, transcribe_tier, options, seconds)
    tl = parse_timeline(raw, seconds)
    micros, lines = price_timeline(tl, seconds, edit_prices.step_rates())[:2] if any(tl.layers[n] for n in edit_store.timeline_lib().LAYERS) else (0, [])
    if tl.layers["subtitles"]:
        sub = edit_subtitles.subtitle_plan(None, tl, edit_prices.subtitle_rates())
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


def with_removers(tl, resolved: dict | None):
    """A copy of the timeline with every "auto" remover that was decided for exactly that part set; the others stay "auto".
    Returns (timeline, True when some part still has to be decided)."""
    out = tl.copy()
    left = False
    for p in out.layers["noise"]:
        if p.settings.get("remover") == "auto":
            r = (resolved or {}).get(edit_store.part_key(p.start, p.end))
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
        item = edit_prices.ITEM[step][0]
        out[item] = out.get(item, 0.0) + blocks.union_seconds(ranges)
    return {k: round(v, 3) for k, v in out.items() if v > 0}


def _blocks():
    from acw_common import blocks

    return blocks


def make_quote(job: dict, params: dict, tl, *, rates: dict[str, int] | None = None, provisional: str = "light") -> dict:
    """What `tl` needs on this job and costs, from its cache. {"free", "unchanged", "approximate", "micros", "lines", "steps",
    "billable", "misses", "asked", "plan": the planner's Plan (not JSON)}. Raises JobError for a timeline this recording can't have."""
    ed = edit_store.editor_lib()
    seconds = edit_store.file_seconds(job, params)
    edit = params.get("edit") or {}
    rates = rates or edit_prices.step_rates()
    tl = edit_store.timeline_lib().Timeline.from_json(tl.to_json(), seconds=seconds)
    rendered = edit_store.rendered_timeline(job, params, seconds)
    planned, approx = for_planning(tl, edit.get("resolved"), provisional)
    prev, _ = for_planning(rendered, edit.get("resolved")) if rendered is not None else (None, False)
    inputs = edit_store.inputs_for(job, params)
    try:
        plan = ed.plan(planned, inputs, edit_store.cache_for(job["id"]), prev)
    except ed.EditorError as e:
        raise bad(str(e)) from None
    browser_fp, subs_fp = edit_subtitles.uploads_fingerprint(job["id"], tl)
    subs = edit_subtitles.subtitle_plan(job["id"], tl, params.get("subtitle_rates") or edit_prices.subtitle_rates())
    # (what the sound depends on is compared without the subtitles; the audio a browser sent counts as part of it)
    audio_same = (rendered is not None and with_removers(tl.audio(), edit.get("resolved"))[0] == rendered.audio()
                  and browser_fp == (edit.get("browser_rendered") or ""))
    subs_same = subs_fp == (edit.get("subs_rendered") or "") and not subs["work"]
    unchanged = audio_same and subs_same
    if audio_same:
        billable, total, lines = {}, 0, []  # the result files already hold exactly this sound
    else:
        billable = ed.billable(plan)
        total, lines = edit_prices.price(billable, rates)
    total, lines = total + subs["micros"], [*lines, *subs["lines"]]
    # A video made on our side (the whole video was uploaded, or a sound asked for as a video) is made again with a new sound, or new
    # subtitles when they go inside it, the way the first run made it: its picture made again is charged again (jobs.convert_cost).
    opts = params.get("options") or {}
    embedded = bool(opts.get("embed_subtitles", True))
    video = ((bool(params.get("has_video")) or bool(opts.get("video_out"))) and bool(opts.get("output_video", True))
             and (not audio_same or (not subs_same and embedded)))
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
        "needs": edit_subtitles.needs_for(job["id"], tl) if edit_subtitles.wants_uploads(tl.to_json()) else None,
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
# What a run reports
# ---------------------------------------------------------------------------------------------------------------------
def billable_from(seconds_by_step: object) -> dict:
    """The `editor.billable` shape from the whole seconds a runner reports per step; the fixed amounts are ours, not the runner's."""
    ed = edit_store.editor_lib()
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
    facts = edit_store.sanitize_facts(raw.get("facts"), d) if raw.get("facts") else None
    if facts:
        out["facts"] = facts
    res = raw.get("resolved")
    if isinstance(res, dict):
        clean = {k: v for k, v in res.items() if isinstance(k, str) and re.fullmatch(r"\d+(\.\d{1,3})?-\d+(\.\d{1,3})?", k) and v in ("light", "heavy")}
        out["resolved"] = dict(list(clean.items())[:MAX_RESOLVED])
    if raw.get("rendered") is not None:
        try:
            rt = edit_store.timeline_lib().Timeline.from_json(raw["rendered"], seconds=facts["n"] / edit_store.SR if facts else seconds)
            if all(p.settings.get("browser") or p.settings.get("remover") in ("light", "heavy") for p in rt.layers["noise"]):
                out["rendered"] = rt.audio().to_json()
        except (edit_store.timeline_lib().TimelineError, KeyError, TypeError):
            pass
    if "billed" in raw:
        out["billed"] = billable_from(raw["billed"])
    if raw.get("suggestions"):
        sug = edit_store.sanitize_suggestions(raw["suggestions"], facts["n"] / edit_store.SR if facts else seconds)
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
    if op != "suggest" and raw and (edit_subtitles.wants_uploads(raw) or ((raw.get("layers") or {}).get("subtitles") if isinstance(raw, dict) else None)):
        try:
            f = report.get("facts") or edit.get("facts")
            fingerprints = edit_subtitles.uploads_fingerprint(job["id"], edit_store.timeline_lib().Timeline.from_json(raw, seconds=f["n"] / edit_store.SR if f else edit_store.file_seconds(job, params)))
        except (edit_store.timeline_lib().TimelineError, KeyError, TypeError):
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
# A job made with a timeline
# ---------------------------------------------------------------------------------------------------------------------
REPLACED_OPTIONS = ("atten_lim_db", "noise", "leveling", "loudness_lufs", "echo_reference_track", "automatic")


def steps_in(tl) -> int:
    """How many cached steps a timeline can fill (for the disk a job is promised)."""
    return max(1, sum(1 for n in ("echo", "noise", "rebuild") if tl.layers[n]))


def options_kind(kind: str, tl) -> str:
    """The kind whose options a job made with `tl` is checked as: with subtitle parts the subtitle choices are the parts' own (no tier, no
    whole-file language or translations), so its options are those of a job that cleans the sound."""
    return "denoise" if tl.layers[edit_store.timeline_lib().SUBTITLES] else kind


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
    if kind == "transcribe" and any(tl.layers[n] for n in edit_store.timeline_lib().LAYERS):
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
    ed = edit_store.editor_lib()
    n = int(round(seconds * edit_store.SR))
    f = {"n": n, "channels": 2, "plan": [0, 1], "spans": {"0": [[0, n]], "1": [[0, n]]}, "rebuild_mid": False, "reference": True}
    job = {"id": "0" * 32, "claimed_seconds": seconds, "duration_seconds": seconds, "root_job_id": None}
    inputs = edit_store.inputs_for(job, {"edit": {"facts": f}})
    t = edit_store.timeline_lib().Timeline.from_json(tl.to_json(), seconds=n / edit_store.SR)
    planned, _ = for_planning(t, None)
    try:
        plan = ed.plan(planned, inputs, None)
    except ed.EditorError as e:
        raise bad(str(e)) from None
    billable = ed.billable(plan)
    micros, lines = edit_prices.price(billable, rates)
    return micros, lines, billable


def first_run_cost(kind: str, params: dict, seconds: float, billed: dict | None = None, job_id: str | None = None) -> int:
    """What a job made with a timeline costs for a recording of `seconds`: its parts (`billed`: the runner's report of what ran, else the
    whole timeline) at the rates pinned with the job, and the subtitles: the whole-file ones as always, or its subtitle parts (`job_id`
    given and `billed`: the ones that were made, else all of them)."""
    rates = params.get("edit_rates") or edit_prices.step_rates()
    tl = edit_store.timeline_lib().Timeline.from_json(params["edit"]["timeline"], seconds=seconds)
    if billed is None:
        edit = price_timeline(tl, seconds, rates)[0]
    else:
        edit = edit_prices.price(billed, rates)[0]
    if tl.layers["subtitles"]:
        sr = params.get("subtitle_rates") or edit_prices.subtitle_rates()
        plan = edit_subtitles.subtitle_plan(None, tl, sr)
        edit += plan["micros"] if billed is None or job_id is None else edit_subtitles.subtitle_charge(job_id, plan, sr)
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
            total += sum(S[edit_subtitles.TRANSCRIBE_SPEED[p["quality"]]] * max(0.0, min(seconds, p["end"]) - max(0.0, p["start"])) for p in parts
                         if p.get("quality") in edit_subtitles.TRANSCRIBE_SPEED)
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
    out = {"source_id": edit_store.source_id(job), "resolved": e.get("resolved") or {}, "facts": e.get("facts")}
    if run:
        out.update(op=run["op"], timeline=run["timeline"], previous=run.get("previous"), audio_format=run.get("audio_format"),
                   suggest=run["op"] == "suggest", video=bool(run.get("video")), audio_file=run.get("audio_file"))
    else:
        out.update(op="first", timeline=e.get("timeline") if e.get("first_run") else None, previous=None, audio_format=None,
                   suggest=0 < float(job.get("duration_seconds") or job["claimed_seconds"]) <= get_settings().suggest_max_seconds)
    raw = out.get("timeline")
    if raw and (edit_subtitles.wants_uploads(raw) or (raw.get("layers") or {}).get("subtitles") or run and run.get("subs") is not None):
        tl = edit_store.timeline_lib().Timeline.from_json(raw, seconds=edit_store.file_seconds(job, params))
        out["browser"] = edit_subtitles.browser_spec(job["id"], tl)
        out["audio"] = bool(run["audio"]) if run and "audio" in run else any(tl.layers[n] for n in edit_store.timeline_lib().LAYERS)
        if (not run and tl.layers["subtitles"]) or (run and run.get("subs") is not None):  # the files put together from the parts
            out["subtitles"] = edit_subtitles.subtitle_spec(tl)
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
        if not jobs.job_dir(job_id).is_dir() or not edit_store.regular_file(jobs.input_path(job)):
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


def timeline_info(job: dict, params: dict) -> dict:
    """The answer of GET /api/jobs/{id}/timeline: the timeline, what is cached of each layer, the suggestions, how to read the preview."""
    seconds = edit_store.file_seconds(job, params)
    e = params.get("edit") or {}
    f = e.get("facts")
    tl = edit_store.saved_timeline(job, params, seconds)
    rendered = edit_store.rendered_timeline(job, params, seconds)
    out: dict = {
        "timeline": tl.to_json(),
        "rendered": rendered.to_json() if rendered is not None else None,
        "seconds": round(seconds, 3),
        "channels": (f or edit_store.default_facts(job, params))["channels"],
        "secondTrack": bool(f and f.get("reference")),
        "cached": edit_preview.cached_ranges(job, params, tl) if job["status"] == "done" else {"echo": [], "noise": [], "light": [], "heavy": [], "rebuild": []},
        "suggestions": edit_store.read_suggestions(job["id"]),
        "run": (params.get("run") or {}).get("op"),
        "lastRun": e.get("lastRun"),
        "expiresAt": job["expires_at"] if job["status"] == "done" else None,
        "editedAt": job.get("edited_at") or job["finished_at"],
        "hardDeadlineAt": jobs.hard_deadline(job),
        "cacheBytes": edit_store.cache_size(job["id"]),
        "preview": None,
    }
    if edit_subtitles.wants_uploads(tl.to_json()):
        out["needs"] = edit_subtitles.needs_for(job["id"], tl)
    if tl.layers["subtitles"]:  # the subtitle parts whose text exists (made, or uploaded by the browser)
        out["cached"]["subtitles"] = [[p.start, p.end] for p in tl.layers["subtitles"]
                                      if (edit_subtitles.cues_of(job["id"], p) is not None if p.settings["quality"] == "free"
                                          else edit_store.regular_file(edit_subtitles.subtitle_dir(job["id"], p) / "segments.json"))]
    if f and f.get("source") and job["status"] == "done":
        out["preview"] = {"layers": ["source", "echo", "noise", "rebuild"], "samples": f["n"], "channels": f["channels"],
                          "sourceScale": f["source"]["scale"], "stretchMaxSeconds": jobs.STRETCH_MAX_SECONDS}
    return out


def save_timeline(user_id: str, job_id: str, api_key_id: str | None, raw: object) -> dict:
    """PUT /api/jobs/{id}/timeline: check and keep the working timeline. An edit: the job's files are kept 2 hours from now."""
    job, params = job_for_edit(user_id, job_id, api_key_id)
    seconds = edit_store.file_seconds(job, params)
    tl = parse_for_job(params, raw, seconds)
    store_timeline(job_id, tl)
    fresh = db.one("SELECT expires_at FROM jobs WHERE id = ?", (job_id,))
    out = {"timeline": tl.to_json(), "expiresAt": fresh["expires_at"]}
    if edit_subtitles.wants_uploads(out["timeline"]):
        out["needs"] = edit_subtitles.needs_for(job_id, tl)
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
    ed = edit_store.editor_lib()
    work = sum(ed.compute_estimate(q["plan"]).values()) / 1.5
    return work + jobs.SPEED["leveling"] * seconds * channels + edit_subtitles.subtitle_seconds(q["subs"])


def start_run(user_id: str, job_id: str, api_key_id: str | None, op: str, raw: object | None = None, *, max_cost_micros: int | None = None,
              audio_format: str | None = None) -> dict:
    """Queue an edit run on a finished job: `process` (what the quote says: the missing blocks, then the render), `export` (the render
    alone, only for a quote of nothing) or `suggest`. The same limits as any job: the account's simultaneous jobs (quotes do not
    count), the API key's spending cap, the disk; the money is held at the quote and the run is charged for what it ran."""
    from . import hostload

    settings = get_settings()
    job, params = job_for_edit(user_id, job_id, api_key_id)
    jobs.check_active_limit(db.conn(), user_id)  # (again in the transaction that stores the run; this is the cheap early answer)
    seconds = edit_store.file_seconds(job, params)
    edit = params.get("edit") or {}
    f = edit_store.facts_of(job, params)
    fmt = audio_format or params["options"].get("audio_format", "flac")
    if fmt not in jobs.AUDIO_FORMATS:
        raise jobs.JobError("unsupported output format")
    rates = edit_prices.step_rates()
    if op == "suggest":
        tl = edit_store.saved_timeline(job, params, seconds)
        q = None
        micros = hold = 0
        run = {"op": op, "timeline": tl.to_json(), "previous": None, "lines": [], "asked": {}, "gpu": [], "echo": False}
        est_seconds = jobs.SPEED["suggest"] * seconds
    else:
        tl = parse_for_job(params, raw, seconds) if raw is not None else edit_store.saved_timeline(job, params, seconds)
        missing = edit_subtitles.needs_for(job_id, tl)
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
        rendered = edit_store.rendered_timeline(job, params, seconds)
        prev, _ = for_planning(rendered, edit.get("resolved")) if rendered is not None else (None, False)
        run = {"op": op, "timeline": tl.to_json(), "previous": prev.to_json() if prev is not None else None, "lines": q["lines"],
               "asked": q["asked"], "gpu": q["gpu"], "echo": bool(tl.layers["echo"]), "rates": rates}
        if q["subs"]["parts"] or not q["subsSame"]:  # the subtitle work and what it costs (the rates pinned with the job)
            run["subs"] = None if q["subsSame"] else {"parts": q["subs"]["parts"]}
            run["subtitle_rates"] = params.get("subtitle_rates") or edit_prices.subtitle_rates()
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
    disk_need = (jobs.scratch_bytes("denoise", seconds, channels, echo=run["echo"]) + (edit_store.growth_bytes(q["plan"], channels) if q else 0)
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
