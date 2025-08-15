from __future__ import annotations

"""Shared pricing constants for processor-time deductions.

Notes:
- The web/API currently deducts base cost for file jobs upfront based on file size.
- For optional translations, we add an extra per-language, per-minute charge at completion.
- Streaming should deduct based on minutes processed (to be implemented in streaming service).
"""

# Cents per minute per additional language for subtitles/translations
EXTRA_LANG_CENTS_PER_MINUTE = 5  # $0.05 per minute per extra language


def extra_language_charge_cents(minutes: float, additional_languages: int) -> int:
    if additional_languages <= 0 or minutes <= 0:
        return 0
    return int(round(minutes * additional_languages * EXTRA_LANG_CENTS_PER_MINUTE))
