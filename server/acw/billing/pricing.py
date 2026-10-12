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


_RATE_LIST = [
    # File processing
    Rate("denoise", "standard", "Studio clean (DeepFilterNet3 + clarity chain)", 10_000),
    Rate("transcribe", "local", "Transcription - standard (on-prem model)", 4_000),
    Rate("transcribe", "premium", "Transcription - premium (OpenAI)", 15_000),
    # Live streaming
    Rate("stream_denoise", "light", "Live denoise - light (DeepFilterNet3)", 4_000),
    Rate("stream_denoise", "medium", "Live denoise - medium", 8_000, available=False),
    Rate("stream_denoise", "high", "Live denoise - high", 15_000, available=False),
    Rate("stream_captions", "local", "Live captions - standard (on-prem model)", 6_000),
    # OpenAI gpt-live-transcribe is $0.017/min of audio sent; we only send speech.
    Rate("stream_captions", "premium", "Live captions - premium (OpenAI realtime)", 25_000),
]
RATES: dict[tuple[str, str], Rate] = {(r.product, r.tier): r for r in _RATE_LIST}

# Smallest charge for one file job, so a 2-second clip still covers overhead.
MIN_JOB_CHARGE_MICROS = 10_000  # $0.01

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
