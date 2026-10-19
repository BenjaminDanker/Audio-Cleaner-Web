"""Automatic settings for API jobs: `options.automatic: true` (docs/api.md).

The server decides the way the website's "Automatic best settings" and the desktop app do: it listens to the upload (the same
measurements as the browser's, acw_common/analysis.py, in a sandboxed `analyse` stage of the runner) and asks the one shared function
(acw_common/reco_model.py `automatic`) for the strength, the noise remover, Voice rebuild and Leveling. Anything the
request gives itself wins over the decision (`fixed_fields`).

The job is created with `denoiseTier: "standard"`, held for the most it could cost (`candidates`: Denoising + Leveling,
or Voice rebuild + Leveling, whichever is more: the rebuild, wherever it is possible) and queued like any job. A worker that takes it measures it after the probe, once
(`decide`), and `apply` turns the job into the job that was decided: the tier, the options, the prices pinned with it, the estimate
and the hold (shrunk to what will be charged). From there it is an ordinary job: charged for what ran, re-run like any other (the
choices made are its settings now). The voice rebuild runs only on the card, so a worker that is not the card's hands the job over
once it has decided on a rebuild (`jobs.hand_over`); where a rebuild is wanted but cannot be done (the file is too long, the card is
not up, the job names Denoising settings or echo removal), the recording is cleaned with Denoising and `chosen` says why.

This module has no heavy imports: the API process imports it for prices; the model comes in only where a decision is made.
"""
from __future__ import annotations

import json
import math

from . import db
from .billing import ledger, pricing

# The options that are decided when `automatic` is on. Giving one yourself (`fixed_fields`) keeps your value.
DECIDED = ("atten_lim_db", "noise", "leveling")
FIXED_WORDS = {"atten_lim_db": "the noise strength", "noise": "the noise type", "leveling": "Leveling"}
# The longest the voice rebuild takes (pricing.MAX_REBUILD_SECONDS), in the words of the reasons.
REBUILD_MINUTES = pricing.MAX_REBUILD_SECONDS // 60


def fixed_fields(options: dict) -> list[str]:
    """Which of the decided options the request gave itself: those it names (`noise: "auto"` is the automatic choice, not a choice)."""
    out = []
    for key in DECIDED:
        if key in options and not (key == "noise" and options[key] == "auto"):
            out.append(key)
    return out


def own_denoising(options: dict, fixed: list[str]) -> bool:
    """Did the request choose settings that only Denoising has (a strength, a noise type)? Then the voice is not rebuilt: it asked
    for Denoising."""
    return "atten_lim_db" in fixed or "noise" in fixed


def rebuild_blocker(options: dict, fixed: list[str], seconds: float | None = None, card_up: bool | None = None) -> str | None:
    """Why the voice cannot be rebuilt for this job, in plain words (None: it can). `seconds`, `card_up`: not asked when None."""
    if own_denoising(options, fixed):
        return "you chose Denoising settings yourself"
    if options.get("echo_reference_track"):
        return "voice rebuilding is not done together with echo removal"
    if seconds is not None and seconds > pricing.MAX_REBUILD_SECONDS:
        return f"voice rebuilding takes files up to {REBUILD_MINUTES} minutes"
    if card_up is False:
        return "voice rebuilding is paused right now"
    return None


def candidates(kind: str, t_tier: str | None, options: dict, fixed: list[str], n_translate: int) -> list[dict]:
    """Every way the decision can end, as {"tier", "rates"} (the prices pinned with the job if it does): Denoising with whatever the
    request did not fix switched on (Leveling), and Voice rebuild with Leveling where a rebuild is possible for this
    job at all (`rebuild_blocker` without the length and the card: those are asked when the price is worked out, `within`, and when
    the decision is made). The hold covers the dearest of them."""
    leveling = bool(options.get("leveling")) if "leveling" in fixed else True
    out = [{"tier": "standard", "rates": pricing.job_rates(kind, "standard", t_tier, n_translate, leveling)}]
    if rebuild_blocker(options, fixed) is None:
        out.append({"tier": "rebuild", "rates": pricing.job_rates(kind, "rebuild", t_tier, n_translate, leveling)})
    return out


def within(cands: list[dict], seconds: float) -> list[dict]:
    """The candidates that a file of `seconds` can end as (a voice rebuild takes files up to pricing.MAX_REBUILD_SECONDS)."""
    return [c for c in cands if c["tier"] != "rebuild" or seconds <= pricing.MAX_REBUILD_SECONDS]


def cost(kind: str, t_tier: str | None, cand: dict, seconds: float, n_translate: int) -> int:
    """What a job that ends as `cand` costs for `seconds` of audio."""
    rates = cand["rates"]
    return pricing.job_cost_micros(kind, cand["tier"], t_tier, seconds, rates, n_translate, "leveling" in rates)


def dearest(kind: str, t_tier: str | None, cands: list[dict], seconds: float, n_translate: int) -> tuple[dict, int]:
    """(the dearest candidate, its price) at `seconds`."""
    priced = [(cost(kind, t_tier, c, seconds, n_translate), c) for c in cands]
    top = max(priced, key=lambda p: p[0])
    return top[1], top[0]


def pending(params: dict) -> bool:
    """An automatic job whose settings are not decided yet."""
    return bool(params.get("automatic")) and "chosen" not in params


def pending_cost(kind: str, params: dict, seconds: float) -> int:
    """The most an undecided automatic job can cost for `seconds` of audio."""
    return dearest(kind, params.get("transcribe_tier"), within(params["automatic"]["candidates"], seconds), seconds,
                   len(params["options"].get("translate") or []))[1]


def outcome_label(kind: str, cand: dict) -> str:
    from . import jobs

    rates = cand["rates"]
    return jobs.job_label(kind, cand["tier"], None, "leveling" in rates)


# ----------------------------------------------------------------------------------------------------- the decision

def clean_stats(raw: object) -> dict | None:
    """The measurements the runner reports, checked (the runner is sandboxed but handles untrusted input): the numbers the model
    reads, finite or missing, and nothing else. None when the noise figure or the length is missing."""
    from acw_common import reco_model

    if not isinstance(raw, dict):
        return None
    out: dict = {}
    for key in (*reco_model.FEATURES, "seconds"):
        v = raw.get(key)
        if key == "hasSpeech":
            out[key] = v is True or v == 1
        elif isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            out[key] = None
        else:
            out[key] = float(v)
    return None if out["noiseDb"] is None or out["seconds"] is None else out


def strength_name(db_value: float) -> str | None:
    from acw_common import reco_model

    return next((k for k, v in reco_model.STRENGTH_DB.items() if v == db_value), None)


def decide(params: dict, stats: dict, seconds: float, *, card_up: bool) -> dict:
    """Settle the settings of an automatic job from the measurements of its upload (checked: `clean_stats`): the shared decision
    (reco_model.automatic), with what the request fixed kept and a voice rebuild only where it can be done.

    Returns {"tier": "standard" | "rebuild", "options": the decided options, "chosen": what the job reports, "notice": a note when a
    wanted rebuild could not be done (else None)}."""
    from acw_common import reco_model

    opts = params["options"]
    fixed = params["automatic"]["fixed"]
    blocker = rebuild_blocker(opts, fixed, seconds, card_up)
    auto = reco_model.automatic(stats, rebuild_available=blocker is None, rebuild_unavailable=blocker or "", min_seconds=0.0)
    if auto is None:  # (only without a noise figure, which clean_stats refuses)
        raise ValueError("no decision for these measurements")
    rebuild = auto.rebuild
    decided = {
        "atten_lim_db": opts["atten_lim_db"] if "atten_lim_db" in fixed else auto.atten_lim_db,
        "noise": opts["noise"] if "noise" in fixed else auto.noise,
        "leveling": opts["leveling"] if "leveling" in fixed else auto.leveling,
    }
    reasons = list(auto.reasons)
    # What the request chose itself is said after what was heard: the reasons above are the decision on the recording alone.
    mine = [k for k in fixed if k in DECIDED]
    if mine:
        reasons.append("You chose " + _and([FIXED_WORDS[k] for k in mine]) + " yourself, so " + ("your choice is" if len(mine) == 1 else "your choices are") + " used.")
    if rebuild:
        strength = None
    else:
        strength = auto.strength if "atten_lim_db" not in fixed else strength_name(decided["atten_lim_db"])
    chosen = {
        "strength": strength,
        "attenLimDb": None if rebuild else decided["atten_lim_db"],
        "noise": None if rebuild else decided["noise"],
        "heavy": False if rebuild else decided["noise"] == "heavy",
        "rebuild": rebuild,
        "rebuildWanted": auto.rebuild_wanted,
        "leveling": decided["leveling"],
        "denoiseTier": "rebuild" if rebuild else "standard",
        "fixed": mine,
        "reasons": reasons,
    }
    notice = None
    if auto.rebuild_wanted and not rebuild:
        notice = f"It sounds like it needs the voice rebuilt, but {blocker}, so it was cleaned with Denoising instead."
    return {"tier": chosen["denoiseTier"], "options": decided, "chosen": chosen, "notice": notice}


def _and(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def apply(job_id: str, decision: dict, seconds: float) -> dict | None:
    """Turn a running automatic job into the job that was decided, in one step: its tier and options, the prices pinned with it
    (so it is charged for what runs), the estimate, and the hold, shrunk to the estimate. Returns the new parameters, or None when
    the job is no longer running or was decided already."""
    with db.tx() as c:
        row = c.execute("SELECT kind, status, params_json, hold_id, est_cost_micros FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None or row["status"] != "running":
            return None
        params = json.loads(row["params_json"])
        if not pending(params):
            return None
        kind, t_tier, tier = row["kind"], params.get("transcribe_tier"), decision["tier"]
        opts = {**params["options"], **decision["options"]}
        n = len(opts.get("translate") or [])
        # the prices pinned when the job was made: those of the candidate it ended as, without the parts that were not chosen
        cand = next((c for c in params["automatic"]["candidates"] if c["tier"] == tier), None)
        if cand is None:
            raise ValueError(f"no {tier} outcome was priced for this job")
        rates = {k: v for k, v in cand["rates"].items() if k != "leveling" or opts["leveling"]}
        new_cost = pricing.job_cost_micros(kind, tier, t_tier, seconds, rates, n, "leveling" in rates)
        params.update(options=opts, denoise_tier=tier, rates=rates, chosen=decision["chosen"])
        params["automatic"] = {"fixed": params["automatic"]["fixed"]}  # (the candidates were only for the hold)
        est = row["est_cost_micros"]
        if row["hold_id"] and 0 < new_cost < est:
            ledger.reduce_hold(row["hold_id"], new_cost, c=c)
            est = new_cost
        c.execute(
            "UPDATE jobs SET params_json = ?, tier = ?, est_cost_micros = ? WHERE id = ?",
            (json.dumps(params), "/".join(t for t in (tier, t_tier) if t), est, job_id),
        )
        return params
