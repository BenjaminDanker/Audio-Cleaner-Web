"""The "Studio clean" chain, shared by the server's job runner and the desktop app:

    decode (48 kHz, mono or stereo) -> DeepFilterNet3 per channel -> clarity chain

Keeping it in one place means a file cleaned on the desktop sounds exactly
like the same file cleaned on the website.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import clarity, dfn
from .dfn import Cancelled
from .ffmpeg import FFmpeg, ProbeResult

SAMPLE_RATE = dfn.SAMPLE_RATE
MAX_CHANNELS = 2

Progress = Callable[[float], None]


@dataclass
class CleanOptions:
    # Maximum noise attenuation in dB; 100 = no limit. Lower keeps more ambience.
    atten_lim_db: float = 100.0
    # Run the clarity chain (level match, high-pass, compressor, loudness, limiter).
    polish: bool = True
    # Loudness target for the clarity chain; None keeps the level.
    loudness_lufs: float | None = -16.0


def clean_audio(
    x: np.ndarray,
    lib_path: str | Path,
    model_path: str | Path,
    opts: CleanOptions | None = None,
    *,
    threads: int = 2,
    progress: Progress | None = None,
    cancel: threading.Event | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Clean float32 audio shaped (samples, channels) at SAMPLE_RATE.

    `progress` gets a fraction in [0, 1]; denoising is ~95% of the work.
    Setting `cancel` aborts with `Cancelled`. The result is written into `out`
    when given (same shape, float32; e.g. a memory-mapped file, so long inputs
    need little RAM), otherwise into a new array."""
    opts = opts or CleanOptions()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    channels = x.shape[1]
    y = np.empty(x.shape, dtype=np.float32) if out is None else (out[:, None] if out.ndim == 1 else out)
    for ch in range(channels):
        def prog(f: float, ch: int = ch) -> None:
            if progress:
                progress(0.95 * (ch + f) / channels)

        dfn.denoise_array(
            x[:, ch], lib_path, model_path, atten_lim_db=opts.atten_lim_db, threads=threads, progress=prog, cancel=cancel,
            out=y[:, ch],
        )
    if cancel is not None and cancel.is_set():
        raise Cancelled()
    if opts.polish:
        y = clarity.process(x, y, SAMPLE_RATE, clarity.ClarityParams(loudness_lufs=opts.loudness_lufs))
    if progress:
        progress(1.0)
    return y


def decode_for_clean(
    ff: FFmpeg, src: Path, probe: ProbeResult, max_seconds: float, work_dir: Path | None = None
) -> np.ndarray:
    """Decode the first audio track at SAMPLE_RATE, keeping stereo (downmixing more).

    With `work_dir`, the samples go to a raw file there and come back
    memory-mapped (read-only), so RAM use stays flat for long files."""
    channels = min(probe.audio_channels, MAX_CHANNELS)
    if work_dir is None:
        return ff.decode(src, probe, SAMPLE_RATE, channels, max_seconds)
    return ff.decode_to_file(src, probe, SAMPLE_RATE, channels, max_seconds, Path(work_dir) / "decoded.f32")


def output_buffer(like: np.ndarray, work_dir: Path | None = None) -> np.ndarray:
    """A float32 array shaped like `like`: memory-mapped in `work_dir` when given."""
    if work_dir is None:
        return np.empty(like.shape, dtype=np.float32)
    return np.lib.format.open_memmap(Path(work_dir) / "cleaned.npy", mode="w+", dtype=np.float32, shape=like.shape)


__all__ = ["SAMPLE_RATE", "CleanOptions", "Cancelled", "clean_audio", "decode_for_clean", "output_buffer", "as_output"]


def as_output(y: np.ndarray) -> np.ndarray:
    """Mono results are written as 1-D arrays."""
    return y[:, 0] if y.ndim == 2 and y.shape[1] == 1 else y
