"""The price of a live session's paid audio. Its features can change while it runs (stream/features.py), so it is
billed in stretches: each stretch of audio at the rate of the features it was processed with. As promised on the
site, the paid seconds of the whole session are rounded up to the next second once, at the end (at the rate then)."""
from __future__ import annotations

import math

from . import protocol as P


class Meter:
    def __init__(self, rate: int):
        self.rate = rate  # micros per minute now
        self.samples = 0  # paid samples processed at this rate
        self.total = 0  # paid samples of the whole session
        self.closed = 0.0  # micros of the stretches before (exact)

    def add(self, n: int) -> None:
        self.samples += n
        self.total += n

    def remove(self, n: int) -> int:
        """Take back up to `n` samples of the current stretch (audio that needed no work); returns how many."""
        n = min(max(0, n), self.samples)
        self.samples -= n
        self.total -= n
        return n

    def set_rate(self, rate: int) -> None:
        """The features changed: the audio so far keeps its rate, what follows gets `rate`."""
        self.closed += self.rate * self.samples / P.SAMPLE_RATE / 60
        self.samples = 0
        self.rate = rate

    @property
    def seconds(self) -> float:
        """Paid seconds of the current stretch."""
        return self.samples / P.SAMPLE_RATE

    def micros(self) -> int:
        return micros(self.closed, self.rate, self.seconds, self.total / P.SAMPLE_RATE)


def micros(closed: float, rate: int, seconds: float, total_seconds: float) -> int:
    """A meter's price from what it saved (stream_sessions.meter_closed, meter_rate, meter_seconds, paid_seconds):
    the stretches before, the current one, and the rest of the session's last second at the current rate."""
    total = round(total_seconds, 6)
    rest = math.ceil(total) - total
    return math.ceil(round(closed + rate * (seconds + rest) / 60, 6))
