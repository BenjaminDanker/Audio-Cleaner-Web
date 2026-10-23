"""Subtitles per part, and the parts the browser does itself (timeline version 2; acw/editing.py has the whole picture): where each
part's results and the browser's uploads go, what is still missing, and what the subtitle parts cost."""
from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

from . import edit_prices, edit_store, job_resources, jobs
from .billing import pricing


# ---------------------------------------------------------------------------------------------------------------------
# Subtitles per part, and the parts the browser does itself (timeline version 2)
# ---------------------------------------------------------------------------------------------------------------------
BROWSER = "browser"  # the job's folder of what the browser uploaded: the cleaned noise parts (FLAC) and the cues of free subtitles (JSON)
SUBS = "subs"  # ... and of the subtitle parts' results, one folder per part (segments.json, segments.<lang>.json, asr_chunks/)
CUES_MAX_BYTES = 1_000_000  # the JSON of one free part's cues
TRANSCRIBE_SPEED = {"standard": "transcribe_local", "accurate": "transcribe_premium"}


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
    data = edit_store.read_json(browser_dir(job_id) / cues_name(part.start, part.end), CUES_MAX_BYTES + 1000)
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
            if not edit_store.regular_file(have_dir / noise_name(n["start"], n["end"])):
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
        tl = edit_store.timeline_lib().Timeline.from_json(raw, seconds=edit_store.file_seconds(job, params))
    except edit_store.timeline_lib().TimelineError:
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
    subs = [p.to_json() for p in tl.parts(edit_store.timeline_lib().SUBTITLES)]
    cues = [stat_of(cues_name(p.start, p.end)) for p in tl.free_parts()]
    return (digest(audio) if audio else "", digest([subs, cues]) if subs else "")


def subtitle_plan(job_id: str | None, tl, rates: dict[str, int]) -> dict:
    """What the subtitles layer needs made, and what that costs: for each part `{key, start, end, quality, language, translate, free,
    transcribe (a paid transcription has to run), translate_run (the translations that have to run)}`. `job_id` None: from nothing (a
    job's first run); else whatever the job's folder already holds is not made again and not charged again. Free parts cost nothing.
    Price of a part = its quality's rate x its whole seconds (rounded up) + the translation rate x the same seconds for each language
    to make. `rates`: `subtitle_rates()`, as pinned."""
    parts, lines_by = [], {"standard": [0, 0], "accurate": [0, 0]}
    tr_seconds, tr_micros, langs = 0, 0, set()
    micros = 0
    for p in tl.parts(edit_store.timeline_lib().SUBTITLES):
        st = p.settings
        free = st["quality"] == "free"
        have = None if free or job_id is None else edit_store.read_json(subtitle_dir(job_id, p) / "segments.json", 20_000_000)
        spoken = have.get("language") if isinstance(have, dict) else None
        e = {"key": subtitle_key(p), "start": p.start, "end": p.end, "quality": st["quality"], "language": st["language"],
             "translate": list(st["translate"]), "free": free, "transcribe": not free and not isinstance(have, dict), "translate_run": []}
        if not free:
            e["translate_run"] = [t for t in st["translate"] if t != spoken
                                  and (job_id is None or not edit_store.regular_file(subtitle_dir(job_id, p) / f"segments.{t}.json"))]
        secs = max(1, math.ceil(p.seconds - 1e-9))
        if e["transcribe"]:
            m = edit_prices.micros_at(rates[f"transcribe_{st['quality']}"], secs)
            lines_by[st["quality"]][0] += secs
            lines_by[st["quality"]][1] += m
            micros += m
        for t in e["translate_run"]:
            m = edit_prices.micros_at(rates["translate"], secs)
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
        if e["transcribe"] and edit_store.regular_file(d / "segments.json"):
            total += edit_prices.micros_at(rates[f"transcribe_{e['quality']}"], secs)
        total += sum(edit_prices.micros_at(rates["translate"], secs) for t in e["translate_run"] if edit_store.regular_file(d / f"segments.{t}.json"))
    return total


def subtitle_seconds(plan: dict) -> float:
    """A rough wall-clock time for the subtitle work of a plan (the queue estimate)."""
    S = job_resources.SPEED
    return sum(S[TRANSCRIBE_SPEED[e["quality"]]] * (e["end"] - e["start"]) for e in plan["parts"] if e["transcribe"])


def subtitle_spec(tl) -> dict:
    """What the runner is told about the subtitles layer: every part (it makes what its folder lacks and puts them all together)."""
    return {"parts": [{"key": subtitle_key(p), "start": p.start, "end": p.end, **{k: p.settings[k] for k in ("quality", "language", "translate")}}
                      for p in tl.parts(edit_store.timeline_lib().SUBTITLES)]}


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
        tl = edit_store.timeline_lib().Timeline.from_json(params["edit"]["timeline"], seconds=float(params["edit"]["timeline"]["seconds"]))
        parts = subtitle_plan(None, tl, params.get("subtitle_rates") or edit_prices.subtitle_rates())["parts"]
    out: dict = {"asr": [], "translate": []}
    base = jobs.job_dir(job_id) / SUBS
    for e in parts:
        if e["free"]:
            continue
        d = base / e["key"]
        done = edit_store.regular_file(d / "segments.json")
        if e["quality"] == "accurate" and not done and (d / "asr_chunks" / "index.json").is_file():
            out["asr"].append(e)
            continue
        if not done:
            continue
        spoken = (edit_store.read_json(d / "segments.json", 20_000_000) or {}).get("language")
        todo = [t for t in e["translate"] if t != spoken and not edit_store.regular_file(d / f"segments.{t}.json")]
        if todo:
            out["translate"].append((e, todo))
    return out


def upload_target(job: dict, params: dict, kind: str, start: float, end: float):
    """The part of the job's current (working) timeline an upload is for, found by its exact range: a browser noise part (`noise`) or a free
    subtitle part (`cues`). Raises JobError 400 when the timeline has no such part."""
    raw = (params.get("edit") or {}).get("timeline")
    what = "cleaned in the browser" if kind == "noise" else "free subtitles"
    if not wants_uploads(raw):
        raise jobs.JobError(f"This job's timeline has no part with {what}.", 400)
    tl = edit_store.timeline_lib().Timeline.from_json(raw, seconds=edit_store.file_seconds(job, params))
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
    a, b = int(round(part.start * edit_store.SR)), int(round(part.end * edit_store.SR))
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
