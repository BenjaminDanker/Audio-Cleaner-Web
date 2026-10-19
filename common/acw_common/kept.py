"""Intermediate results kept so a later change of the choices does not run the expensive step again.

The server keeps, per paid job, the noise removal's full-strength output (before strength, highs and leveling) or the
rebuilt voice (before leveling): `pipeline.clean_audio(keep=...)` hands them over, `save` writes them as 24-bit FLAC and
`load` brings them back as float32, to within 1.2e-7 of full scale (one 24-bit step; a sample is scaled down first when it would
clip, and the scale is kept with the file). The 32-bit float they are made of would take twice the disk; FLAC takes
a third to a half of that (docs/operations.md, "What a job keeps").
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .ffmpeg import FFmpeg

SAMPLE_RATE = 48_000
FORMAT = "flac24"
CEILING = 0.98  # a sample louder than this is scaled into it before the 24-bit file is written
BLOCK = SAMPLE_RATE * 30


def _peak(y: np.ndarray) -> float:
    peak = 0.0
    for s in range(0, len(y), BLOCK):
        peak = max(peak, float(np.max(np.abs(y[s:s + BLOCK])))) if len(y[s:s + BLOCK]) else peak
    return peak


def save(ff: FFmpeg, y: np.ndarray, channels: list[int] | None, dest: Path) -> dict:
    """Write the channels `channels` of y ((samples, channels); None: all) to `dest` (its suffix is replaced with .flac).
    Returns what `load` needs: {"file", "channels", "scale", "samples"}. Channels must be next to each other."""
    y2 = y[:, None] if y.ndim == 1 else y
    cols = list(range(y2.shape[1])) if channels is None else sorted(channels)
    if cols != list(range(cols[0], cols[-1] + 1)):
        raise ValueError("kept channels must be next to each other")
    part = y2[:, cols[0]:cols[-1] + 1]
    peak = _peak(part)
    scale = 1.0 if peak <= CEILING else CEILING / peak
    # Written under another name and moved into place: the folder may hold hard links of a job's files (a re-run's), and
    # writing into a linked file would change the other job's copy too.
    path = ff.write_audio(part, SAMPLE_RATE, dest.with_name(dest.name + "-part"), FORMAT, gain=scale)
    final = dest.with_suffix(path.suffix)
    os.replace(path, final)
    return {"file": final.name, "channels": cols, "scale": scale, "samples": int(len(y2))}


def load(ff: FFmpeg, entry: dict, directory: Path, out: np.ndarray, scratch: Path) -> np.ndarray:
    """Read a kept copy into `out` ((samples, channels), float32, any writable array: the kept channels are set, the others
    left as they are). `scratch` is a file name for the decoded samples (removed afterwards). Returns `out`."""
    path = directory / entry["file"]
    probe = ff.probe(path)
    cols = list(entry["channels"])
    n = int(entry["samples"])
    raw = ff.decode_to_file(path, probe, SAMPLE_RATE, len(cols), n / SAMPLE_RATE + 2.0, scratch)
    inv = np.float32(1.0 / float(entry["scale"]))
    try:
        for s in range(0, min(n, len(raw), len(out)), BLOCK):
            e = min(n, len(raw), len(out), s + BLOCK)
            for k, ch in enumerate(cols):
                out[s:e, ch] = raw[s:e, k] * inv
        done = min(n, len(raw), len(out))
        if done < len(out):
            for ch in cols:
                out[done:, ch] = 0.0
    finally:
        del raw
        scratch.unlink(missing_ok=True)
    return out
