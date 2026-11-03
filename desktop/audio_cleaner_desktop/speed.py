"""About how long each step takes on this computer: what the time shown on Start and in Details is worked out from.

A first guess comes from the cost of each step on an ordinary computer (CPU_COSTS), spread over the threads a run gets here (all the
processor's threads but one when the file runs by itself). Every run then measures how long its steps really took (editor.Runner.timings,
and the speech for subtitles) and keeps, step by step, how far that was from the guess; later guesses are corrected by it. So the times
follow this computer's processor, its memory and whatever else runs on it, after one run of each step. The ratios are kept in a small
file in the app's data folder (`use_file`); without one (the command line, tests) they last until the app closes.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from . import sysinfo

log = logging.getLogger(__name__)

#: Wall-clock seconds one second of audio takes on one core of an ordinary computer, and a fixed amount per run, step by step. The
#: light remover, echo removal and Leveling are measured (editor.STEP_COSTS: CPU); the heavy remover is 1.8 x the light one, and Voice
#: rebuild on a CPU about 30 x (docs/research); the server runs those two on a graphics card, this app on the processor.
CPU_COSTS = {
    "echo": (0.12, 0.034), "light": (0.27, 0.116), "heavy": (1.5, 0.21), "rebuild": (4.3, 3.5), "leveling": (1.28, 0.013),
}
#: How much of each thread a step really gets (the models do not use many threads perfectly).
THREAD_EFFICIENCY = 0.7
#: Seconds a second of speech takes to write down for subtitles on an ordinary computer.
SPEECH_COST = 0.5
#: A run's measurement counts only when the guess for it was at least this long (a few seconds are mostly loading the model).
LEARN_FROM_SECONDS = 5.0

_lock = threading.Lock()
_ratios: dict[str, float] = {}
_file: Path | None = None


def use_file(path: Path) -> None:
    """Keep the learned speeds in `path` (read now, written after every run)."""
    global _file
    with _lock:
        _file = path
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            _ratios.clear()
            _ratios.update({k: float(v) for k, v in raw.items() if k in (*CPU_COSTS, "speech") and 0.01 <= float(v) <= 100})
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError, AttributeError) as e:
            log.warning("the learned speeds could not be read: %s", e)


def run_threads() -> int:
    """The threads a file gets when it runs by itself (worker.py gives the files that run together a share each)."""
    return sysinfo.threads_per_job(1)


def guess(step: str, seconds: float, threads: int) -> float:
    """The first guess: wall-clock seconds `step` takes for `seconds` of audio (per channel it runs on) with `threads` threads."""
    if seconds <= 0:
        return 0.0
    if step == "speech":
        return SPEECH_COST * seconds
    fixed, per = CPU_COSTS[step]
    return (fixed + per * seconds) / max(1.0, threads * THREAD_EFFICIENCY)


def cost(step: str, seconds: float, threads: int | None = None) -> float:
    """About how long `step` takes here for `seconds` of audio: the guess, corrected by what this computer's runs measured."""
    with _lock:
        ratio = _ratios.get(step, 1.0)
    return guess(step, seconds, threads or run_threads()) * ratio


def learn(timings: dict, threads: int) -> None:
    """A run's measured steps ({step: [wall seconds, seconds of audio]}) correct the guesses (half the old correction, half the new)."""
    changed = False
    with _lock:
        for step, (wall, seconds) in timings.items():
            if step not in CPU_COSTS and step != "speech":
                continue
            g = guess(step, seconds, threads)
            if g < LEARN_FROM_SECONDS or wall <= 0:
                continue
            measured = min(100.0, max(0.01, wall / g))
            old = _ratios.get(step)
            _ratios[step] = measured if old is None else 0.5 * old + 0.5 * measured
            changed = True
        if changed and _file is not None:
            try:
                _file.write_text(json.dumps(_ratios, indent=1), encoding="utf-8")
            except OSError as e:
                log.warning("the learned speeds could not be kept: %s", e)
