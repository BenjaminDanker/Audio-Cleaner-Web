"""Rates and prices of the per-part editor (acw/editing.py): what each step costs per second at today's price list, and the lines a
person sees."""
from __future__ import annotations

import math

from .billing import pricing


# Customer-facing names of the steps (the price list's words: what you get, never how it is made).
ITEM = {"echo": ("echo", "Echo removal"), "light": ("denoise", "Denoising"), "heavy": ("denoise", "Denoising"),
        "rebuild": ("rebuild", "Voice rebuild"), "leveling": ("leveling", "Leveling")}


# ---------------------------------------------------------------------------------------------------------------------
# Rates and prices
# ---------------------------------------------------------------------------------------------------------------------
def step_rates() -> dict[str, int]:
    """Micros per minute of audio of each step that can run, at today's price list (pinned with every run)."""
    denoise = pricing.rate("denoise", "standard").micros_per_minute
    return {"echo": pricing.rate("echo", "standard").micros_per_minute, "light": denoise, "heavy": denoise,
            "rebuild": pricing.rate("denoise", "rebuild").micros_per_minute,
            "leveling": pricing.rate("leveling", "standard").micros_per_minute}


def step_micros(rate_per_minute: int, seconds: int, fixed_equivalent_seconds: float) -> int:
    """rate per second x (whole seconds + the step's fixed amount as equivalent seconds), rounded up to a micro."""
    return math.ceil(rate_per_minute * (seconds + fixed_equivalent_seconds) / 60 - 1e-9)


def price(billable: dict, rates: dict[str, int]) -> tuple[int, list[dict]]:
    """(total micros, the lines a person sees) of `editor.billable(plan)` at `rates`. Light and heavy noise removal are one line,
    Denoising; each internal step is rounded up on its own."""
    total = 0
    items: dict[str, dict] = {}
    for step in ("echo", "light", "heavy", "rebuild", "leveling"):
        b = billable.get(step)
        if not b or b["seconds"] <= 0:
            continue
        micros = step_micros(rates[step], int(b["seconds"]), float(b["fixed_equivalent_seconds"]))
        item, label = ITEM[step]
        line = items.setdefault(item, {"item": item, "label": label, "quantity": 1, "seconds": 0, "fixedSeconds": 0.0, "usdPerMinute": rates[step] / pricing.MICROS_PER_USD,
                                       "micros": 0})
        line["seconds"] += int(b["seconds"])
        line["fixedSeconds"] = round(line["fixedSeconds"] + float(b["fixed_equivalent_seconds"]), 3)
        line["micros"] += micros
        total += micros
    return total, list(items.values())


def subtitle_rates() -> dict[str, int]:
    """Micros per minute of the subtitle steps at today's price list (pinned with the job, like `step_rates`): Standard and Accurate
    transcription per part, and a translation per part and language."""
    return {"transcribe_standard": pricing.rate("transcribe", "local").micros_per_minute,
            "transcribe_accurate": pricing.rate("transcribe", "premium").micros_per_minute,
            "translate": pricing.rate("transcribe_translate", "standard").micros_per_minute}


def micros_at(rate_per_minute: int, seconds: int) -> int:
    return step_micros(rate_per_minute, seconds, 0.0)
