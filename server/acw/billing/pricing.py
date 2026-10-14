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
    Rate("denoise", "standard", "Studio clean", 10_000),
    Rate("transcribe", "local", "Transcription", 4_000),
    Rate("transcribe", "premium", "Transcription, premium accuracy", 15_000),
    # Live streaming
    Rate("stream_denoise", "light", "Live noise removal", 4_000),
    Rate("stream_denoise", "low_latency", "Live noise removal, low latency", 8_000),
    Rate("stream_captions", "local", "Live captions", 6_000),
    # The premium caption engine bills per minute of audio sent; we only send speech.
    Rate("stream_captions", "premium", "Live captions, premium accuracy", 25_000),
]
RATES: dict[tuple[str, str], Rate] = {(r.product, r.tier): r for r in _RATE_LIST}

# No minimum: jobs are billed per second of audio (rounded up), as advertised.
MIN_JOB_CHARGE_MICROS = 0

# Top-up rules (cents). $10 minimum keeps Stripe's fixed fee and dispute
# exposure reasonable (see docs/payments.md).
TOPUP_MIN_CENTS = 1_000
TOPUP_MAX_CENTS = 50_000
TOPUP_PRESETS_CENTS = [1_000, 2_000, 5_000, 10_000]

# Free tier
FREE_TRANSCRIBE_SECONDS_PER_DAY = 60 * 60
FREE_TRANSCRIBE_MAX_FILE_SECONDS = 30 * 60

# Hard limits
MAX_JOB_SECONDS = 4 * 60 * 60
# Streams reserve this many minutes ahead and top the hold up as they go.
STREAM_HOLD_MINUTES = 5


class PricingError(ValueError):
    pass


def rate(product: str, tier: str) -> Rate:
    r = RATES.get((product, tier))
    if r is None or not r.available:
        raise PricingError(f"unknown or unavailable tier {product}/{tier}")
    return r


def cost_micros(product: str, tier: str, seconds: float) -> int:
    """Price for `seconds` of audio, rounded up to the next whole second."""
    secs = max(0, math.ceil(seconds))
    return math.ceil(rate(product, tier).micros_per_minute * secs / 60)


def job_cost_micros(kind: str, denoise_tier: str | None, transcribe_tier: str | None, seconds: float) -> int:
    total = 0
    if kind in ("denoise", "denoise_transcribe") and denoise_tier:
        total += cost_micros("denoise", denoise_tier, seconds)
    if kind in ("transcribe", "denoise_transcribe") and transcribe_tier:
        total += cost_micros("transcribe", transcribe_tier, seconds)
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
