"""Headless modes: --cli (clean files) and --selftest (check the bundled native pieces)."""
from __future__ import annotations

import signal
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

from acw_common import pipeline
from acw_common.ffmpeg import MediaError

from . import engine, resources


class _ProgressLine:
    """Rewrites one line on a terminal; prints a line every 10% otherwise (logs, CI)."""

    def __init__(self, label: str):
        self._label = label
        self._tty = sys.stdout.isatty()
        self._last = -1

    def __call__(self, fraction: float) -> None:
        pct = int(fraction * 100)
        if self._tty:
            print(f"\r{self._label}  {pct:3d}%", end="", flush=True)
        elif pct // 10 > self._last // 10:
            print(f"{self._label}  {pct}%", flush=True)
        self._last = pct

    def finish(self) -> None:
        if self._tty:
            print("\r" + " " * (len(self._label) + 7) + "\r", end="")


def run(inputs: list[str], out_dir: str | None, options: engine.JobOptions) -> int:
    """Clean each input in turn. Returns 0 when all succeed, 1 when any fails, 130 on Ctrl+C."""
    missing = resources.check()
    if missing:
        print("Missing components: " + ", ".join(missing), file=sys.stderr)
        return 1
    cancel = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: cancel.set())
    failures = 0
    for n, name in enumerate(inputs, 1):
        label = f"[{n}/{len(inputs)}] {Path(name).name}"
        line = _ProgressLine(label)
        try:
            out = engine.process_file(name, options, out_dir=out_dir, progress=line, cancel=cancel)
        except engine.Cancelled:
            line.finish()
            print(f"{label}  cancelled", file=sys.stderr)
            return 130
        except (MediaError, OSError) as e:
            line.finish()
            print(f"{label}  failed: {e}", file=sys.stderr)
            failures += 1
        else:
            line.finish()
            print(f"{label}  ->  {out}")
    return 1 if failures else 0


def selftest() -> int:
    """Denoise 3 s of noisy tone and round-trip it through FFmpeg, using the resolved native pieces."""
    try:
        missing = resources.check()
        if missing:
            raise RuntimeError("missing components: " + ", ".join(missing))
        sr = pipeline.SAMPLE_RATE
        t = np.arange(3 * sr) / sr
        rng = np.random.default_rng(0)
        x = (0.2 * np.sin(2 * np.pi * 220 * t) + 0.03 * rng.standard_normal(t.size)).astype(np.float32)
        y = pipeline.clean_audio(x, resources.libdf_path(), resources.model_path(), threads=1)
        if y.shape != (x.size, 1) or not np.isfinite(y).all():
            raise RuntimeError("the noise reduction engine returned unexpected audio")
        ff = engine.ffmpeg()
        with tempfile.TemporaryDirectory(prefix="acw-selftest-") as tmp:
            out = ff.write_audio(pipeline.as_output(y), sr, Path(tmp) / "selftest", "flac")
            info = ff.probe(out)
            decoded = ff.decode(out, info, sr, 1, 10)
        if abs(info.duration - 3.0) > 0.2 or info.audio_channels != 1 or decoded.size < 2.8 * sr:
            raise RuntimeError(f"FFmpeg round-trip mismatch (duration {info.duration:.2f} s)")
    except Exception as e:  # noqa: BLE001 - any failure is a failed selftest
        print(f"selftest failed: {e}")
        return 1
    print("selftest ok")
    return 0
