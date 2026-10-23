"""The named strengths of the noise reduction, the same scale as the website's: Gentle 12, Balanced 24, Strong 40, Maximum 100 dB,
equally spaced, with the dB in between (what a strength is: acw_common.pipeline.strength_mix).

No Qt in here.
"""
from __future__ import annotations

# The four named strengths in dB, in order (the same numbers as engine.STRENGTHS).
NAMES = (("gentle", "Gentle", 12.0), ("balanced", "Balanced", 24.0), ("strong", "Strong", 40.0), ("max", "Maximum", 100.0))
POINTS = tuple(db for _, _, db in NAMES)


def pos_of(db: float) -> float:
    """Where a strength in dB lies on the scale, 0..3 (the four names at 0, 1, 2, 3)."""
    d = min(max(float(db), POINTS[0]), POINTS[-1])
    i = 0
    while i < len(POINTS) - 2 and d > POINTS[i + 1]:
        i += 1
    return i + (d - POINTS[i]) / (POINTS[i + 1] - POINTS[i])


def strength_key(db: float) -> str | None:
    """"gentle" ... "max" when `db` is one of the named strengths, else None."""
    return next((key for key, _, named in NAMES if abs(named - db) < 0.5), None)


def strength_name(db: float) -> str:
    """"Balanced", or "Between Balanced and Strong"."""
    for _, label, named in NAMES:
        if abs(named - db) < 0.5:
            return label
    i = min(int(pos_of(db)), len(NAMES) - 2)
    return f"Between {NAMES[i][1]} and {NAMES[i + 1][1]}"
