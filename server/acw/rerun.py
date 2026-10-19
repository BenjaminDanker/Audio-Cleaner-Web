"""Cheap re-runs of a finished job ("Change and save again" for paid jobs).

A finished job keeps, in its folder, until it is deleted (2 hours after it finished, jobs.sweep): the upload, the noise
removal at full strength (`kept-noise.flac`) or the rebuilt voice (`kept-rebuilt.flac`) when those ran, the transcript
(`segments.json`, `segments.<lang>.json`) and its results. A re-run is a new job (`source_job_id`, `root_job_id`) that
starts from the latest of those copies and runs only the steps whose inputs changed:

  strength, Leveling, loudness                  from the kept noise removal: blend, Leveling (no noise model); Leveling is paid
                                                whenever it runs
  Voice rebuild on                              the rebuild from the upload, at its full price (it is the expensive step)
  Voice rebuild off                             the kept noise removal, or noise removal from the upload (paid) when the job
                                                never had one
  Leveling or loudness with Voice rebuild       Leveling from the kept rebuilt voice
  another noise type or echo removal            noise removal from the upload (paid)
  another subtitle kind, or the language        transcription again from the upload (paid)
  more translation languages                    only the new ones (paid)

Everything else (file type, name, which subtitle files, subtitles in the video) is done in the browser and is free
(web/src/finish.js). The steps that run are the paid ones listed in the price; encoding and the strength blend are free.

The files a re-run needs are hard-linked into its folder (no copy: disk grows only by what it makes), so deleting one job of
a chain never breaks another, and "Delete now" deletes the whole chain (the upload with it). A re-run ends at the same time
as the job it came from and never extends it (`deadline`).
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
from pathlib import Path

from . import db, jobs, keycaps
from .billing import ledger, pricing
from .config import get_settings

RERUN_LABEL = "Changed and saved again"
# The `code` of the errors a client may want to tell apart (X-Error-Code and "code" in the answer; docs/api.md).
JOB_NOT_FOUND = "job_not_found"  # no such job, or not yours: an API key can only change the jobs it made itself
NOT_FINISHED = "job_not_finished"
FILES_GONE = "job_files_gone"  # the job's files are cleared away (expired or deleted): start again from the file
NOTHING_TO_CHANGE = "nothing_to_change"  # the choices would give the files the job already has
EXPIRING = "job_expiring"  # too close to the end of the job's 2 hours to run again
PRICE_CHANGED = "price_changed"  # the price is above `maxCostMicros`
# A re-run must be able to finish, and its results be downloaded, before the job it came from is cleared away.
DEADLINE_MARGIN_SECONDS = 300
# A re-run that is only done this close to the end of the job it came from (or later) is not kept: see too_late.
MIN_DELIVERY_SECONDS = 60
TOO_LATE = "This took longer than expected and your files were about to be cleared away, so it was stopped. You weren’t charged."
SEGMENTS = "segments.json"
TRANSLATION = re.compile(r"^segments\.([a-z]{2})\.json$")


def _regular(path: Path) -> bool:
    """A regular file (never a link) in the job's folder: files the sandboxed runner made are checked before they are used."""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISREG(st.st_mode)


def sanitize_kept(raw: object, d: Path) -> dict:
    """The copies the runner reports having kept, checked: known keys and types only, and only files that exist as regular files in
    `d`. Anything else is dropped (the job just can't be redone from it)."""
    out: dict = {}
    if not isinstance(raw, dict):
        return out
    for key, name in (("noise", jobs.KEPT_NOISE), ("rebuilt", jobs.KEPT_REBUILT), ("input", jobs.KEPT_INPUT)):
        e = raw.get(key)
        if not isinstance(e, dict):
            continue
        try:
            entry = {
                "file": f"{name}.flac",
                "channels": [int(c) for c in e["channels"]],
                "scale": float(e["scale"]),
                "samples": int(e["samples"]),
                "echo": bool(e.get("echo", False)),
            }
            if e.get("file") != entry["file"] or not _regular(d / entry["file"]):
                continue
            if not (1 <= len(entry["channels"]) <= 2 and all(c in (0, 1) for c in entry["channels"])):
                continue
            if not (0.0 < entry["scale"] <= 1.0 and entry["samples"] > 0):
                continue
            if key == "noise":
                entry["profile"] = e["profile"] if e["profile"] in ("light", "heavy") else None
                entry["noise_opt"] = e["noise_opt"] if e["noise_opt"] in ("auto", "light", "heavy") else None
                entry["plan"] = [None if c is None else int(c) for c in e["plan"]]
                if entry["profile"] is None or entry["noise_opt"] is None or not (1 <= len(entry["plan"]) <= 2):
                    continue
                if not all(c is None or c in (0, 1) for c in entry["plan"]):
                    continue
                energy = _energy(e.get("energy"))
                if energy:
                    entry["energy"] = energy
        except (KeyError, TypeError, ValueError):
            continue
        out[key] = entry
    return out


def _energy(raw: object) -> dict | None:
    """The sums of squares and of the products of the two copies (acw_common.kept.energy), when they are sane numbers."""
    try:
        out = {k: float(raw[k]) for k in ("full", "input", "cross")}  # type: ignore[index]
    except (KeyError, TypeError, ValueError):
        return None
    ok = all(math.isfinite(v) for v in out.values()) and out["full"] >= 0 and out["input"] >= 0
    return out if ok else None


def translations_in(d: Path) -> list[str]:
    return sorted(m.group(1) for p in d.iterdir() if (m := TRANSLATION.match(p.name)) and _regular(p))


def collect_kept(params: dict, d: Path, reported: object) -> dict:
    """The `kept` record of a job that just finished: the copies that are in its folder (the ones the runner made now, and
    the ones a re-run inherited and linked), and what is known about the transcript."""
    inherited = params.get("kept") or {}
    kept = sanitize_kept({**{k: v for k, v in inherited.items() if k in ("noise", "rebuilt", "input")}, **(reported if isinstance(reported, dict) else {})}, d)
    if "input" in kept and not ("noise" in kept and kept["input"]["channels"] == kept["noise"]["channels"]
                                and kept["input"]["samples"] == kept["noise"]["samples"] and kept["input"]["echo"] == kept["noise"]["echo"]):
        kept.pop("input")  # only ever used with the noise removal it was kept with
    opts = params.get("options", {})
    seg = d / SEGMENTS
    if _regular(seg):
        rr = params.get("rerun")
        ran = rr is None or rr.get("transcribe", True)
        old = inherited.get("segments")
        if ran or not isinstance(old, dict):
            entry = {"tier": params.get("transcribe_tier"), "language": opts.get("language"), "echo": bool(opts.get("echo_reference_track"))}
        else:
            entry = {k: old.get(k) for k in ("tier", "language", "echo")}
        entry["translations"] = translations_in(d)
        if entry["tier"] in ("local", "premium"):
            kept["segments"] = entry
    return kept


def kept_files(kept: dict) -> set[str]:
    """Names of the files a `kept` record stands for."""
    names = {kept[k]["file"] for k in ("noise", "rebuilt", "input") if k in kept}
    seg = kept.get("segments")
    if seg:
        names.add(SEGMENTS)
        names.update(f"segments.{lang}.json" for lang in seg.get("translations", []))
    return names


def audio_output(job: dict) -> dict | None:
    for o in json.loads(job["outputs_json"] or "[]"):
        if o.get("kind") == "audio":
            return o
    return None


def redo_info(job: dict) -> dict:
    """What `public_job` says about changing a finished job again: whether it can still be done (its files exist) and
    which copies it can start from."""
    out = {"available": False}
    if job["status"] != "done":
        return out
    params = json.loads(job["params_json"])
    d = jobs.job_dir(job["id"])
    if not _regular(jobs.input_path(job)):
        return out
    kept = params.get("kept") or {}
    has = lambda key: key in kept and _regular(d / kept[key]["file"])  # noqa: E731
    out = {
        "available": True,
        "noise": has("noise"),
        "rebuilt": has("rebuilt"),
        "transcript": "segments" in kept and _regular(d / SEGMENTS),
        "translations": list((kept.get("segments") or {}).get("translations", [])),
    }
    if out["noise"] and has("input") and params.get("denoise_tier") == "standard":
        # what the page needs to blend the strength itself (the strength slider, and saving a new strength without a job):
        # the two copies sample for sample, any length (GET /api/jobs/{id}/kept/{noise|input}?start=&seconds= hands out short
        # stretches of them, stretch.py), and how to read them
        n = kept["noise"]
        out["preview"] = {"samples": n["samples"], "channels": n["channels"], "plan": n["plan"],
                          "scale": {"noise": n["scale"], "input": kept["input"]["scale"]},
                          "bytes": {k: (d / kept[k]["file"]).stat().st_size for k in ("noise", "input")},
                          "stretchMaxSeconds": jobs.STRETCH_MAX_SECONDS}
        if energy := _energy(n.get("energy")):
            out["preview"]["energy"] = energy  # keeps the loudness of a preview level while the strength moves
    return out


# ---------------------------------------------------------------------------------------------------- planning

def _levels_same(new: dict, old: dict) -> bool:
    if not new["leveling"] and not old["leveling"]:
        return True
    return new["leveling"] == old["leveling"] and new["loudness_lufs"] == old["loudness_lufs"]


def plan_rerun(src_params: dict, src_outputs_audio: dict | None, d_tier: str | None, t_tier: str | None, new: dict, files: set[str]) -> dict:
    """Which steps a re-run with the choices `new` (validated options, with the tiers) runs again, and from what. `files`:
    the files that exist in the source job's folder.

    Returns {"clean": "none" | "full" | "denoised" | "rebuilt" | "reuse", "transcribe": bool, "reuse_audio": name | None,
    "paid": [steps from pricing.RERUN_ITEMS], "translate_new": [languages], "work": bool, "carry": [kept keys to carry], "reuse_segments": bool}."""
    old = src_params["options"]
    kept = {k: v for k, v in (src_params.get("kept") or {}).items() if (k not in ("noise", "rebuilt", "input") or v["file"] in files)}
    d_old = src_params.get("denoise_tier")
    paid: list[str] = []
    clean = "none"
    reuse_audio = None
    can_reuse = src_outputs_audio is not None and src_outputs_audio["file"] in files
    if d_tier:
        levels_same = _levels_same(new, old)
        same_audio_file = can_reuse and new["echo_reference_track"] == old["echo_reference_track"]
        if d_tier == pricing.NO_DENOISE:
            if d_old == pricing.NO_DENOISE and same_audio_file and levels_same:
                clean = "reuse"
            else:
                clean = "full"
                if new["leveling"]:
                    paid.append("leveling")
        elif d_tier == "rebuild":
            entry = kept.get("rebuilt")
            if entry and entry["echo"] == new["echo_reference_track"]:
                if d_old == "rebuild" and same_audio_file and levels_same:
                    clean = "reuse"
                else:
                    clean = "rebuilt"
                    if new["leveling"]:
                        paid.append("leveling")
            else:
                clean = "full"
                paid.append("denoise")
                if new["leveling"]:
                    paid.append("leveling")
        else:
            entry = kept.get("noise")
            if entry and entry["echo"] == new["echo_reference_track"] and new["noise"] in (entry["noise_opt"], entry["profile"]):
                same_result = (d_old == "standard" and same_audio_file and levels_same and new["atten_lim_db"] == old["atten_lim_db"]
                               and new["noise"] == old["noise"])
                if same_result:
                    clean = "reuse"
                else:
                    clean = "denoised"
                    if new["leveling"]:
                        paid.append("leveling")
            else:
                clean = "full"
                paid.append("denoise")
                if new["leveling"]:
                    paid.append("leveling")
        if clean == "reuse":
            reuse_audio = src_outputs_audio["file"]
    transcribe = False
    translate_new: list[str] = []
    linked_seg = False
    if t_tier:
        seg = kept.get("segments")
        if (seg and SEGMENTS in files and seg.get("tier") == t_tier and seg.get("language") == new["language"]
                and seg.get("echo") == new["echo_reference_track"]):
            linked_seg = True
            have = set(seg.get("translations", []))
            translate_new = [t for t in new["translate"] if t not in have]
        else:
            transcribe = True
            paid.append("transcribe")
            translate_new = list(new["translate"])
        if translate_new:
            paid.append("translate")
    # What is carried forward (hard-linked into the re-run's folder) besides the upload: the kept copies that still fit the
    # echo choice, and the transcript when it is reused.
    made_again = {"rebuild": {"rebuilt"}, "standard": {"noise", "input"}}.get(d_tier, set()) if clean == "full" else set()
    carry = [key for key in ("noise", "rebuilt", "input")
             if key in kept and kept[key]["echo"] == new["echo_reference_track"] and key not in made_again]
    # Only a different file type (the audio is written again from the result at hand) or different caption files (written again
    # from the kept transcript) are work too, and free.
    fmt_changed = bool(d_tier) and new["audio_format"] != old["audio_format"]
    caps_changed = bool(t_tier) and set(new["caption_formats"]) != set(old["caption_formats"])
    work = (clean in ("full", "denoised", "rebuilt") or transcribe or bool(translate_new) or (clean == "reuse" and fmt_changed)
            or (linked_seg and caps_changed))
    return {"clean": clean, "transcribe": transcribe, "reuse_audio": reuse_audio, "paid": paid, "translate_new": translate_new,
            "work": work, "carry": carry, "reuse_segments": linked_seg}


AUTOMATIC_RERUN = ("Automatic settings are for a first run. The choices they made are this job’s settings now: change them like any "
                   "other (strength, Leveling, or the Voice rebuild tier).")


def merged_options(src_params: dict, options: dict | None) -> dict:
    """The source's choices with the changes asked for on top. A job made with automatic settings carries the choices they made
    (its options are the decided ones); the automatic mode itself does not go on: asking for it again is refused."""
    if (options or {}).get("automatic"):
        raise jobs.JobError(AUTOMATIC_RERUN)
    return {**src_params["options"], **(options or {}), "automatic": False}


def _tiers(src_params: dict, denoise_tier, transcribe_tier) -> tuple[str, str | None, str | None]:
    """(kind, denoise tier, transcribe tier) of the re-run: what is asked for, else what the source had. "off" as the
    transcribe tier drops the subtitles."""
    d = denoise_tier if denoise_tier is not None else src_params.get("denoise_tier")
    t = src_params.get("transcribe_tier") if transcribe_tier is None else (None if transcribe_tier == "off" else transcribe_tier)
    if d and t:
        return "denoise_transcribe", d, t
    if d:
        return "denoise", d, None
    if t:
        return "transcribe", None, t
    raise jobs.JobError("Choose noise removal, Leveling, subtitles, or more than one of them.", code=NOTHING_TO_CHANGE)


GONE = "This job’s files are already cleared away, so it can’t be changed and saved again. Start it again from your file."


def _source(user_id: str, job_id: str, api_key_id: str | None) -> dict:
    """The finished job to start from; an API key can only change jobs it made itself."""
    row = db.one("SELECT * FROM jobs WHERE id = ? AND user_id = ?", (job_id, user_id)) if len(job_id) == 32 else None
    if row is None or (api_key_id is not None and row["api_key_id"] != api_key_id):
        raise jobs.JobError("Job not found.", 404, code=JOB_NOT_FOUND)
    src = dict(row)
    if src["status"] != "done":
        raise jobs.JobError(GONE if src["status"] == "expired" else "Only a finished job can be changed and saved again.", 409,
                            code=FILES_GONE if src["status"] == "expired" else NOT_FINISHED)
    d = jobs.job_dir(src["id"])
    if not d.is_dir() or not _regular(jobs.input_path(src)):
        raise jobs.JobError(GONE, 409, code=FILES_GONE)
    return src


def _prepare(user_id: str, job_id: str, api_key_id: str | None, denoise_tier, transcribe_tier, options: dict | None):
    src = _source(user_id, job_id, api_key_id)
    sp = json.loads(src["params_json"])
    kind, d_tier, t_tier = _tiers(sp, denoise_tier, transcribe_tier)
    # The upload's own length, as it was measured.
    seconds = float(src["duration_seconds"] or src["claimed_seconds"])
    new = jobs.validate_options(kind, d_tier, t_tier, merged_options(sp, options))
    d = jobs.job_dir(src["id"])
    files = {p.name for p in d.iterdir() if _regular(p)}
    plan = plan_rerun(sp, audio_output(src), d_tier, t_tier, new, files)
    return src, sp, kind, d_tier, t_tier, new, seconds, plan, files


def estimate(user_id: str, job_id: str, api_key_id: str | None, denoise_tier: str | None, transcribe_tier: str | None, options: dict | None) -> dict:
    """What a re-run with these choices does and costs, before it exists (nothing is held or stored): the same checks and the
    same price as `create_rerun`. POST /api/jobs/{id}/rerun/estimate."""
    src, sp, kind, d_tier, t_tier, new, seconds, plan, _ = _prepare(user_id, job_id, api_key_id, denoise_tier, transcribe_tier, options)
    _check(plan, d_tier, t_tier, src, sp, kind, seconds)
    rates = pricing.rerun_rates(d_tier, t_tier, plan["paid"])
    n = len(plan["translate_new"])
    total = pricing.rerun_cost_micros(rates, seconds, n)
    return {
        "estCostMicros": total,
        "estCostUsd": total / pricing.MICROS_PER_USD,
        "free": total == 0,
        "durationSeconds": seconds,
        "lines": jobs.price_lines(rates, d_tier, t_tier, seconds, n),
        "steps": {"clean": plan["clean"], "transcribe": plan["transcribe"], "translate": plan["translate_new"]},
        "expiresAt": src["expires_at"],
        "options": new,
        "denoiseTier": d_tier,
        "transcribeTier": t_tier,
    }


def _check(plan: dict, d_tier: str | None, t_tier: str | None, src: dict, sp: dict, kind: str, seconds: float) -> None:
    if not plan["work"]:
        raise jobs.JobError("Nothing would change: these choices give the files this job already has.", code=NOTHING_TO_CHANGE)
    if plan["clean"] == "full" and d_tier == "rebuild":
        jobs.check_duration(kind, d_tier, seconds)  # the card must be up, and the file short enough for the rebuild
    deadline = src["expires_at"] or 0
    need = _seconds_needed(plan, kind, d_tier, t_tier, sp, seconds)
    if deadline and deadline - db.now() < need + DEADLINE_MARGIN_SECONDS:
        raise jobs.JobError("This one is about to be cleared away, so there isn’t time to change it again. Start it again from your file.", 409, code=EXPIRING)


def _seconds_needed(plan: dict, kind: str, d_tier: str | None, t_tier: str | None, sp: dict, seconds: float) -> float:
    """Rough wall-clock time of the re-run once it starts, doubled for the wait in the queue."""
    probe = {"kind": kind, "duration_seconds": seconds, "claimed_seconds": seconds, "params_json": json.dumps({
        "options": sp["options"], "denoise_tier": d_tier, "transcribe_tier": t_tier, "channels": sp.get("channels") or 2,
        "rerun": {"clean": plan["clean"], "transcribe": plan["transcribe"]}})}
    return 2 * jobs.job_seconds(probe)


def rerun_label(rates: dict[str, int], d_tier: str | None, t_tier: str | None) -> str:
    parts = []
    if "denoise" in rates:
        parts.append("Voice rebuild" if d_tier == "rebuild" else "Denoising")
    if "leveling" in rates:
        parts.append("Leveling")
    if "transcribe" in rates:
        parts.append("Subtitles & transcript" + (", extra accurate" if t_tier == "premium" else ""))
    elif "translate" in rates:
        parts.append("Translated subtitles")
    return f"{RERUN_LABEL} ({' + '.join(parts)})" if parts else RERUN_LABEL


def link(src: Path, dst: Path) -> None:
    """A hard link of a regular file (no disk used); a copy when the folders are on different devices."""
    if not _regular(src):
        raise FileNotFoundError(src.name)
    try:
        os.link(src, dst, follow_symlinks=False)
    except OSError:
        shutil.copyfile(src, dst)
        os.chmod(dst, 0o600)


def create_rerun(user_id: str, job_id: str, *, denoise_tier: str | None = None, transcribe_tier: str | None = None,
                 options: dict | None = None, api_key_id: str | None = None, max_cost_micros: int | None = None) -> dict:
    """Make the re-run job: checks, the money (held like any job), its folder (the files it starts from, linked) and its
    row, queued. Same limits as a new job: the account's simultaneous jobs, the API key's spending cap, the disk.
    `max_cost_micros`: the price the caller was quoted (estimate); a higher price now is refused (409, price_changed)."""
    settings = get_settings()
    src, sp, kind, d_tier, t_tier, new, seconds, plan, files = _prepare(user_id, job_id, api_key_id, denoise_tier, transcribe_tier, options)
    _check(plan, d_tier, t_tier, src, sp, kind, seconds)
    rates = pricing.rerun_rates(d_tier, t_tier, plan["paid"])
    n_translate = len(plan["translate_new"])
    est = pricing.rerun_cost_micros(rates, seconds, n_translate)
    if max_cost_micros is not None and est > max_cost_micros:
        raise jobs.JobError(f"This now costs about ${est / 1e6:.2f}, more than the ${max_cost_micros / 1e6:.2f} you were quoted.", 409, code=PRICE_CHANGED)
    from . import hostload

    channels = max(1, min(2, int(sp.get("channels") or 2)))
    disk_need = (jobs.scratch_bytes(kind, seconds, channels, echo=new["echo_reference_track"])
                 + jobs.kept_bytes(kind, seconds, channels, d_tier) + jobs.DISK_RESERVE)
    if hostload.get(settings.data_dir).snapshot().disk_free < disk_need:
        raise jobs.JobError("We're short on storage space right now. Please try again a bit later.", 507)

    new_id = db.new_id()
    root = src["root_job_id"] or src["id"]
    kept = {k: v for k, v in (sp.get("kept") or {}).items() if k in plan["carry"]}
    if plan["reuse_segments"] and "segments" in (sp.get("kept") or {}):
        kept["segments"] = sp["kept"]["segments"]
    src_dir = jobs.job_dir(src["id"])
    dst = jobs.job_dir(new_id)
    linked: list[str] = []
    params = {
        "options": new,
        "denoise_tier": d_tier,
        "transcribe_tier": t_tier,
        "input_ext": sp["input_ext"],
        "display_name": sp.get("display_name"),
        "channels": sp.get("channels"),
        "rates": rates,
        "kept": kept,
        "rerun": {
            "source": src["id"], "root": root, "deadline": src["expires_at"], "clean": plan["clean"], "transcribe": plan["transcribe"],
            "reuse_audio": plan["reuse_audio"], "translate_new": plan["translate_new"], "paid": plan["paid"], "linked": linked,
        },
    }
    if sp.get("noise_profile"):
        params["noise_profile"] = sp["noise_profile"]
    now = db.now()
    tier = "/".join(t for t in (d_tier, t_tier) if t)
    try:
        dst.mkdir(parents=True, mode=0o700)
        wanted = [jobs.input_path(src).name, *sorted(kept_files(kept))]
        if plan["reuse_audio"]:
            wanted.append(plan["reuse_audio"])
        for name in dict.fromkeys(wanted):
            link(src_dir / name, dst / name)
            linked.append(name)
        with db.tx() as c:
            jobs.check_active_limit(c, user_id)
            hold_id = None
            if est > 0:
                keycaps.check(c, api_key_id, est)
                try:
                    hold_id = ledger.create_hold(user_id, est, "job", new_id, c=c)
                except ledger.InsufficientFunds as e:
                    raise jobs.JobError(f"This costs about ${est / 1e6:.2f}; your available balance is ${e.available / 1e6:.2f}.", 402) from e
            c.execute(
                "INSERT INTO jobs(id, user_id, kind, tier, status, params_json, next_stage, input_name, input_bytes, received_bytes,"
                " claimed_seconds, duration_seconds, hold_id, est_cost_micros, created_at, queued_at, api_key_id, source_job_id, root_job_id)"
                " VALUES (?,?,?,?, 'queued', ?, 'probe', ?,?,?,?,?,?,?,?,?,?,?,?)",
                (new_id, user_id, kind, tier, json.dumps(params), src["input_name"], src["input_bytes"], src["received_bytes"],
                 seconds, seconds, hold_id, est, now, now, api_key_id, src["id"], root),
            )
    except BaseException:
        shutil.rmtree(dst, ignore_errors=True)
        raise
    jobs.invalidate_queue_cache()
    return {"id": new_id, "estCostMicros": est, "free": est == 0, "expiresAt": src["expires_at"]}


def chain(job: dict) -> list[dict]:
    """Every job made from the same upload as `job` (itself included)."""
    root = job.get("root_job_id") or job["id"]
    return [dict(r) for r in db.all_rows("SELECT * FROM jobs WHERE id = ? OR root_job_id = ?", (root, root))]


def delete_chain(job: dict) -> None:
    """"Delete now": the files of the job and of every re-run made from the same upload, the upload included."""
    for j in chain(job):
        if j["status"] in ("uploading", "queued", "running", "cloud_asr"):
            continue
        shutil.rmtree(jobs.job_dir(j["id"]), ignore_errors=True)
        db.execute("UPDATE jobs SET status = 'expired', outputs_json = NULL WHERE id = ? AND status = 'done'", (j["id"],))


def chain_active(job: dict) -> bool:
    return any(j["status"] in ("uploading", "queued", "running", "cloud_asr") for j in chain(job))


def too_late(params: dict, now: int) -> bool:
    """A re-run that is only finishing when the job it came from is about to be cleared away (it waited in the queue longer than
    planned): its results could not be downloaded before they go, and a re-run never extends the deletion time, so it ends as
    failed and is not charged (`TOO_LATE`)."""
    rr = params.get("rerun")
    return bool(rr and rr.get("deadline") and now + MIN_DELIVERY_SECONDS >= int(rr["deadline"]))


def completion_expiry(job: dict, params: dict, now: int) -> int:
    """When the results of a job that has just finished are deleted: 2 hours from now, and for a re-run never later than
    the job it came from (its upload, its kept copies and all the jobs made from the same upload go together)."""
    expires = now + get_settings().job_retention_seconds
    rr = params.get("rerun")
    if rr and rr.get("deadline"):
        expires = min(expires, int(rr["deadline"]))
    return expires
