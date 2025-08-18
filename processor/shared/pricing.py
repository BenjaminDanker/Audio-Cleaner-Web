from __future__ import annotations

"""Shared pricing constants for processor-time deductions.

Architecture:
- API deducts base cost + transcription upfront (batch jobs) or initial fee (streaming)
- Processor handles per-minute charges for streaming and additional language costs
- Batch processor only processes, no billing (API handles all costs upfront)

Streaming pricing:
- API charges initial session fee when auth succeeds
- Processor handles per-minute streaming transcription costs
- Additional languages charged per-minute as processed
"""

# Streaming costs (different from batch - real-time processing overhead)
STREAMING_COST_PER_MINUTE = 15  # $0.15 per minute for real-time transcription (higher than batch $0.003/min)
STREAMING_LANG_COST_PER_MINUTE = 8  # $0.08 per minute per additional language (higher than batch $0.02/min)

# Legacy batch language cost (for backwards compatibility, but batch shouldn't use this anymore)
EXTRA_LANG_CENTS_PER_MINUTE = 5  # $0.05 per minute per extra language (deprecated - API handles this now)


def streaming_transcription_charge_cents(minutes: float) -> int:
    """Calculate real-time transcription cost for streaming sessions."""
    if minutes <= 0:
        return 0
    return int(round(minutes * STREAMING_COST_PER_MINUTE))


def streaming_language_charge_cents(minutes: float, additional_languages: int) -> int:
    """Calculate additional language cost for streaming (real-time translation overhead)."""
    if additional_languages <= 0 or minutes <= 0:
        return 0
    return int(round(minutes * additional_languages * STREAMING_LANG_COST_PER_MINUTE))


def extra_language_charge_cents(minutes: float, additional_languages: int) -> int:
    """Legacy function for batch language charges - should not be used anymore.
    
    API now handles all batch costs upfront. This is kept for backwards compatibility.
    """
    if additional_languages <= 0 or minutes <= 0:
        return 0
    return int(round(minutes * additional_languages * EXTRA_LANG_CENTS_PER_MINUTE))
