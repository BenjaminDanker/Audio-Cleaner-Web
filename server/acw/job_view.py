"""How a job is shown: its label, the callbacks it reports to, and the job as the API and the website see it (`public_job`)."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from . import automatic, db, job_resources, jobs
from .billing import pricing


def job_label(kind: str, denoise_tier: str | None, transcribe_tier: str | None, leveling: bool = False) -> str:
    """Customer-facing name of what a job does (the same words as the price list). `leveling`: Leveling is part of it
    (for a stored job: "leveling" in its pinned rates)."""
    parts = []
    if kind == "convert":
        return pricing.rate("convert", "standard").label
    if kind in ("denoise", "denoise_transcribe"):
        if denoise_tier != pricing.NO_DENOISE:
            parts.append("Voice rebuild" if denoise_tier == "rebuild" else "Denoising")
        if leveling or denoise_tier == pricing.NO_DENOISE:
            parts.append("Leveling")
    if kind in ("transcribe", "denoise_transcribe"):
        parts.append("Subtitles & transcript" + (", extra accurate" if transcribe_tier == "premium" else ""))
    return " + ".join(parts)


def safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip(".-_")[:60]
    return stem or "audio"


def callbacks_for(job_ids: list[str]) -> dict[str, dict]:
    """The state of each job's callback (the jobs that have one), as shown with the job."""
    if not job_ids:
        return {}
    cols = "SELECT job_id, url, state, attempts, next_attempt_at, last_status, last_error, delivered_at FROM job_callbacks"
    if len(job_ids) == 1:
        rows = [r for r in (db.one(f"{cols} WHERE job_id = ?", (job_ids[0],)),) if r is not None]
    else:
        rows = db.all_rows(f"{cols} WHERE job_id IN ({','.join('?' * len(job_ids))})", tuple(job_ids))
    return {
        r["job_id"]: {
            "url": r["url"].split("?", 1)[0],  # the query string often carries a token: not shown back
            "state": r["state"],
            "attempts": r["attempts"],
            "lastStatus": r["last_status"],
            "lastError": r["last_error"],
            "deliveredAt": r["delivered_at"],
            "nextAttemptAt": r["next_attempt_at"] if r["state"] == "pending" and r["attempts"] else None,
        }
        for r in rows
    }


def label_of(kind: str, params: dict) -> str:
    """The customer-facing name of what a job does: the price list's words; for a re-run, the steps it did again. A video conversion
    that makes the picture again is named too."""
    base = _label_of(kind, params)
    if kind != "convert" and not params.get("run") and params.get("convert_rate") and params.get("picture_encode"):
        return f"{base} + {pricing.rate('convert', 'standard').label}"
    return base


def _label_of(kind: str, params: dict) -> str:
    if params.get("run"):
        from . import editing

        return editing.run_label(params)
    edit = params.get("edit") or {}
    # a job made with a timeline (it alone has `edit_rates`; its stored tier is only a placeholder) is named for its layers, also once its first run is over
    if edit.get("first_run") or (edit.get("timeline") and params.get("edit_rates")):
        from . import editing

        return " + ".join([editing.timeline_label(params), *([job_label("transcribe", None, params["transcribe_tier"])] if params.get("transcribe_tier") else [])])
    if automatic.pending(params):  # not decided yet: what it will do is not known
        return " + ".join(["Automatic settings", *([job_label("transcribe", None, params["transcribe_tier"])] if params.get("transcribe_tier") else [])])
    rates = params.get("rates") or {}
    return job_label(kind, params.get("denoise_tier"), params.get("transcribe_tier"), "leveling" in rates)


def public_job(job: dict, queue: tuple[list[str], dict[str, int]] | None = None, callbacks: dict[str, dict] | None = None) -> dict:
    """`queue` is queue_snapshot() and `callbacks` callbacks_for() (pass them once when listing many jobs)."""
    outputs = json.loads(job["outputs_json"]) if job.get("outputs_json") else []
    params = json.loads(job["params_json"])
    position = eta = None
    if job["status"] == "queued":
        ids, waits = jobs.queue_snapshot() if queue is None else queue
        if job["id"] in waits:
            position = ids.index(job["id"]) + 1
            eta = waits[job["id"]]
    callback = (callbacks if callbacks is not None else callbacks_for([job["id"]])).get(job["id"])
    out = {
        "queuePosition": position,
        "queueWaitSeconds": eta,
        "id": job["id"],
        "kind": job["kind"],
        "tier": job["tier"],
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"],
        "error": job["error"],
        "inputName": params.get("display_name") or job["input_name"],
        "label": label_of(job["kind"], params),
        # (a job asked for with options: the tiers it was asked for; the timeline it became is what is made of them)
        "denoiseTier": (params.get("whole_file") or {}).get("denoise_tier", params.get("denoise_tier")),
        "transcribeTier": (params.get("whole_file") or {}).get("transcribe_tier", params.get("transcribe_tier")),
        "durationSeconds": job["duration_seconds"] or job["claimed_seconds"],
        "estCostMicros": job["est_cost_micros"],
        "costMicros": job["cost_micros"],
        "options": params["options"],
        "notice": params.get("notice"),
        "outputs": [{"name": o["file"], "kind": o["kind"], "bytes": o.get("bytes"), **({"lang": o["lang"]} if o.get("lang") else {})} for o in outputs],
        "createdAt": job["created_at"],
        "finishedAt": job["finished_at"],
        # when the files are deleted: 2 hours after the last edit, never later than `hardDeadlineAt` (6 hours after the upload)
        "expiresAt": job["expires_at"] if job["status"] == "done" else None,
        "editedAt": job.get("edited_at") or job["finished_at"],
        "hardDeadlineAt": jobs.hard_deadline(job),
    }
    from . import edit_subtitles, editing

    if editing.editable_kind(job):
        e = params.get("edit") or {}
        out["edit"] = {"available": job["status"] == "done", "hasTimeline": bool(e.get("timeline")), "run": (params.get("run") or {}).get("op"),
                       "lastRun": e.get("lastRun"), "suggestions": bool(e.get("suggestions"))}
        needs = edit_subtitles.pending_needs(job, params)  # what the browser still has to send for the timeline's browser parts and free subtitles
        if needs is not None:
            out["needs"] = needs
    if params.get("automatic"):
        out["chosen"] = params.get("chosen")  # null until the file has been listened to
        if automatic.pending(params):
            out["maximum"] = True  # estCostMicros is the most it can cost until then
    if job["status"] in ("queued", "running", "cloud_asr"):
        out["estimateSeconds"] = int(job_resources.job_seconds(job))  # the whole job's rough length, for the wait before progress shows
    if callback is not None:
        out["callback"] = callback
    return out
