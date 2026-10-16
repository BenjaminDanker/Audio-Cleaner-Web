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
    Rate("denoise", "standard", "Clean & polish", 10_000),
    # The voice is rebuilt (also removes room echo): ~30x the CPU of the standard tier. Placeholder price.
    Rate("denoise", "rebuild", "Clean & polish, voice rebuilt", 30_000),
    Rate("transcribe", "local", "Transcription", 4_000),
    Rate("transcribe", "premium", "Transcription, extra accurate", 15_000),
    # Live streaming
    Rate("stream_denoise", "light", "Live noise removal", 4_000),
    Rate("stream_captions", "local", "Live captions", 6_000),
    # The extra-accurate caption engine bills per minute of audio sent; we only send speech.
    Rate("stream_captions", "premium", "Live captions, extra accurate", 30_000),
]
RATES: dict[tuple[str, str], Rate] = {(r.product, r.tier): r for r in _RATE_LIST}

# No minimum: jobs are billed per second of audio (rounded up), as advertised.
MIN_JOB_CHARGE_MICROS = 0

# Top-up rules (cents). $10 minimum keeps Stripe's fixed fee and dispute
# exposure reasonable (see docs/payments.md).
TOPUP_MIN_CENTS = 500
TOPUP_MAX_CENTS = 50_000
TOPUP_PRESETS_CENTS = [1_000, 2_000, 5_000, 10_000]

# The free subtitle minutes on our side are gone (free subtitles run in the browser); a job created free before that
# change still finishes under the limits it was created with (acw/worker/main.py).
FREE_TRANSCRIBE_SECONDS_PER_DAY = 60 * 60
FREE_TRANSCRIBE_MAX_FILE_SECONDS = 30 * 60

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


def job_rates(kind: str, denoise_tier: str | None, transcribe_tier: str | None) -> dict[str, int]:
    """Micros per minute of each part of a job at today's prices. Stored with the job, so a price
    change never applies to a job that was already created."""
    out: dict[str, int] = {}
    if kind in ("denoise", "denoise_transcribe") and denoise_tier:
        out["denoise"] = rate("denoise", denoise_tier).micros_per_minute
    if kind in ("transcribe", "denoise_transcribe") and transcribe_tier:
        out["transcribe"] = rate("transcribe", transcribe_tier).micros_per_minute
    return out


def job_cost_micros(
    kind: str, denoise_tier: str | None, transcribe_tier: str | None, seconds: float, rates: dict[str, int] | None = None
) -> int:
    """`rates` (from job_rates) pins the prices; without it the current price list applies."""
    rates = rates or {}
    total = 0
    if kind in ("denoise", "denoise_transcribe") and denoise_tier:
        total += cost_micros("denoise", denoise_tier, seconds, rates.get("denoise"))
    if kind in ("transcribe", "denoise_transcribe") and transcribe_tier:
        total += cost_micros("transcribe", transcribe_tier, seconds, rates.get("transcribe"))
    return max(total, MIN_JOB_CHARGE_MICROS) if total else 0


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
