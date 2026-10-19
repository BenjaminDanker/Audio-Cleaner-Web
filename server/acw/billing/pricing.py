"""Price list. Amounts are USD micros (1 USD = 1_000_000) per minute of audio.

These are launch placeholders; change them here only. The public price page
reads them through /api/billing/pricing so the site never disagrees with billing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

MICROS_PER_USD = 1_000_000
MICROS_PER_CENT = 10_000


@dataclass(frozen=True)
class Rate:
    product: str
    tier: str
    label: str
    micros_per_minute: int
    available: bool = True


# Labels are shown to customers: describe what they get, never how it's made.
_RATE_LIST = [
    # File processing
    Rate("denoise", "standard", "Denoising", 9_000),
    # The voice is rebuilt (also removes room echo and brings back missing high sounds), on the GPU only, one job at a time.
    Rate("denoise", "rebuild", "Voice rebuild", 30_000),
    # Leveling: evens out the voice and sets the loudness; with Denoising or Voice rebuild, or on its own ("off" noise tier).
    # About 5 % of Denoising's CPU; Denoising + Leveling stays at the 0.60 USD an hour Denoising cost before.
    Rate("leveling", "standard", "Leveling", 1_000),
    Rate("transcribe", "local", "Transcription", 4_000),
    Rate("transcribe", "premium", "Transcription, extra accurate", 15_000),
    # Per extra language, on top of the transcription (text only goes to the speech partner; a few cents an hour).
    Rate("transcribe_translate", "standard", "Translated subtitles, per language", 4_000),
    # Live streaming
    Rate("stream_denoise", "light", "Live noise removal", 4_000),
    Rate("stream_captions", "local", "Live captions", 6_000),
    # The extra-accurate caption engine bills per minute of audio sent; we only send speech.
    Rate("stream_captions", "premium", "Live captions, extra accurate", 30_000),
    Rate("stream_translate", "standard", "Live caption translation, per language", 4_000),
]
RATES: dict[tuple[str, str], Rate] = {(r.product, r.tier): r for r in _RATE_LIST}

# No minimum: jobs are billed per second of audio (rounded up), as advertised.
MIN_JOB_CHARGE_MICROS = 0

# Top-up rules (cents). $5 minimum (the owner's choice: Stripe's 30c fixed fee makes small top-ups cost us more,
# 8.9% of $5, but the free trial covers trying things out).
TOPUP_MIN_CENTS = 500
TOPUP_MAX_CENTS = 50_000
TOPUP_PRESETS_CENTS = [1_000, 2_000, 5_000, 10_000]
# Bonus credit on bigger top-ups, roughly what Stripe's fixed fee saves on them (it is 5.9% of $10, 3.5% of $50, 3.2%
# of $100): (from cents, per mille). Bonus credit is given credit: never refunded as money, and taken back when its
# payment is refunded or disputed.
TOPUP_BONUS_TIERS = [(10_000, 25), (5_000, 20)]


def topup_bonus_micros(amount_cents: int) -> int:
    for from_cents, per_mille in TOPUP_BONUS_TIERS:
        if amount_cents >= from_cents:
            return amount_cents * MICROS_PER_CENT * per_mille // 1000
    return 0


# Hard limits
MAX_JOB_SECONDS = 4 * 60 * 60
# The voice-rebuilt tier needs ~7 CPU-seconds per second of stereo audio (docs/research/restore-sidon.md):
# longer files would tie up the machine for hours and still risk timing out uncharged.
MAX_REBUILD_SECONDS = 90 * 60
# Streams reserve this many minutes ahead and top the hold up as they go.
STREAM_HOLD_MINUTES = 5


class PricingError(ValueError):
    pass


def rate(product: str, tier: str) -> Rate:
    r = RATES.get((product, tier))
    if r is None or not r.available:
        raise PricingError(f"unknown or unavailable tier {product}/{tier}")
    return r


def max_job_seconds(denoise_tier: str | None) -> int:
    """Longest file a job with this noise-removal tier accepts."""
    return MAX_REBUILD_SECONDS if denoise_tier == "rebuild" else MAX_JOB_SECONDS


def cost_micros(product: str, tier: str, seconds: float, micros_per_minute: int | None = None) -> int:
    """Price for `seconds` of audio, rounded up to the next whole second."""
    secs = max(0, math.ceil(seconds))
    return math.ceil((rate(product, tier).micros_per_minute if micros_per_minute is None else micros_per_minute) * secs / 60)


# The noise tier of a job that only levels (no noise removal).
NO_DENOISE = "off"


def job_rates(kind: str, denoise_tier: str | None, transcribe_tier: str | None, translate_languages: int = 0,
              leveling: bool = False) -> dict[str, int]:
    """Micros per minute of each part of a job at today's prices. Stored with the job, so a price
    change never applies to a job that was already created. `translate` is per language."""
    out: dict[str, int] = {}
    if kind in ("denoise", "denoise_transcribe") and denoise_tier and denoise_tier != NO_DENOISE:
        out["denoise"] = rate("denoise", denoise_tier).micros_per_minute
    if kind in ("denoise", "denoise_transcribe") and leveling:
        out["leveling"] = rate("leveling", "standard").micros_per_minute
    if kind in ("transcribe", "denoise_transcribe") and transcribe_tier:
        out["transcribe"] = rate("transcribe", transcribe_tier).micros_per_minute
        if translate_languages:
            out["translate"] = rate("transcribe_translate", "standard").micros_per_minute
    return out


def job_cost_micros(
    kind: str, denoise_tier: str | None, transcribe_tier: str | None, seconds: float, rates: dict[str, int] | None = None,
    translate_languages: int = 0, leveling: bool = False,
) -> int:
    """`rates` (from job_rates) pins the prices; without it the current price list applies. `translate_languages`:
    how many translations are charged (asked for when estimating, made when the job is done); `leveling`: whether
    Leveling is part of it (a job's charge: "leveling" in its pinned rates, so a job made before it had a price never pays for it)."""
    rates = rates or {}
    total = 0
    if kind in ("denoise", "denoise_transcribe") and denoise_tier and denoise_tier != NO_DENOISE:
        total += cost_micros("denoise", denoise_tier, seconds, rates.get("denoise"))
    if kind in ("denoise", "denoise_transcribe") and leveling:
        total += cost_micros("leveling", "standard", seconds, rates.get("leveling"))
    if kind in ("transcribe", "denoise_transcribe") and transcribe_tier:
        total += cost_micros("transcribe", transcribe_tier, seconds, rates.get("transcribe"))
        if translate_languages:
            total += translate_languages * cost_micros("transcribe_translate", "standard", seconds, rates.get("translate"))
    return max(total, MIN_JOB_CHARGE_MICROS) if total else 0


RERUN_ITEMS = ("denoise", "leveling", "transcribe", "translate")


def rerun_rates(denoise_tier: str | None, transcribe_tier: str | None, paid: list[str]) -> dict[str, int]:
    """Micros per minute of each step a re-run does again (and so pays for), at today's prices; stored with the job. `paid`:
    the steps from RERUN_ITEMS that run. Steps that don't run (the strength blend and encoding never have a price) are not here."""
    out: dict[str, int] = {}
    if "denoise" in paid and denoise_tier and denoise_tier != NO_DENOISE:
        out["denoise"] = rate("denoise", denoise_tier).micros_per_minute
    if "leveling" in paid:
        out["leveling"] = rate("leveling", "standard").micros_per_minute
    if "transcribe" in paid and transcribe_tier:
        out["transcribe"] = rate("transcribe", transcribe_tier).micros_per_minute
    if "translate" in paid:
        out["translate"] = rate("transcribe_translate", "standard").micros_per_minute
    return out


def rerun_cost_micros(rates: dict[str, int], seconds: float, translate_languages: int = 0) -> int:
    """What a re-run costs: each step it does again, per second of audio rounded up, at the prices pinned with the job."""
    secs = max(0, math.ceil(seconds))
    per = lambda micros: math.ceil(micros * secs / 60)  # noqa: E731
    total = sum(per(rates[k]) for k in ("denoise", "leveling", "transcribe") if k in rates)
    if "translate" in rates:
        total += translate_languages * per(rates["translate"])
    return total


def public_price_list() -> list[dict]:
    return [
        {
            "product": r.product,
            "tier": r.tier,
            "label": r.label,
            "usdPerMinute": r.micros_per_minute / MICROS_PER_USD,
            "usdPerHour": r.micros_per_minute * 60 / MICROS_PER_USD,
            "available": r.available,
        }
        for r in _RATE_LIST
    ]
