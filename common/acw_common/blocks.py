"""The block grid and the block cache of the per-part editor (docs/research: editor experiments, 2026-10).

Every model step runs on a FIXED grid of blocks, each block with some context before and after it that the model sees
but that is cropped away, and every processed block can be cached. A block is identified by the step, the settings that
change the model run, the input it was made from and where it is; strength is not a model setting (it is a blend of the
full-strength block with the input, pipeline.strength_mix), so a strength change never needs a model run.

    light remover (DeepFilterNet3)  30 s grid, 16 s before, 0.25 s after   (>= 58 dB against one contiguous run)
    heavy remover (DPDFNet-8)       30 s grid, 24 s before, 0.25 s after   (about 41 dB, PESQ >= 4.4)
    echo removal (AEC3)             30 s grid, 10 s before                 (quality converges; the waveform does not)
    Voice rebuild (Sidon)           its own windows (restore.py): 14 s step, each block made from the window before it,
                                    its own and the one after it, dither seeded per window: equal to a whole-file run bit for bit

A whole-file run of a step is "all blocks" of the same grid (editor.render, and dfn.denoise_array, dpdfnet.denoise_array,
restore.restore on their own), so the seams of a long file get the context the cache has.

Blocks are numpy float32 arrays of the block's core ([start, end), no context). `BlockCache` is anything with `get(key)`,
`put(key, array)` and `has(key)`: `MemoryCache`, `DirCache` (float32 .npy files) and `FlacCache` (24-bit FLAC files, the server's) are here.
"""
from __future__ import annotations

import hashlib
import os
import stat
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

SAMPLE_RATE = 48_000
#: Part of every cache key: change it when a step's output changes (a model, the grid or its context), and every cached block misses.
VERSION = 1
#: What a step's cache keys call the model that made the block (a new model file or version = a new tag).
MODEL_TAGS = {"echo": "aec3", "light": "dfn3", "heavy": "dpdfnet8", "rebuild": "sidon-v0.1"}


@dataclass(frozen=True)
class Grid:
    """A step's block grid: the blocks are `seconds` long and start at multiples of it (from the start of the file);
    the model also gets `before` / `after` seconds of the recording around each block, cropped from the result."""
    seconds: float
    before: float
    after: float

    @property
    def block(self) -> int:
        return int(round(self.seconds * SAMPLE_RATE))

    @property
    def pre(self) -> int:
        return int(round(self.before * SAMPLE_RATE))

    @property
    def post(self) -> int:
        return int(round(self.after * SAMPLE_RATE))


GRID_LIGHT = Grid(30.0, 16.0, 0.25)
GRID_HEAVY = Grid(30.0, 24.0, 0.25)
GRID_ECHO = Grid(30.0, 10.0, 0.01)  # 0.01 s after: the canceller delays its output by 432 samples (aec.DELAY_SAMPLES)
GRIDS = {"light": "GRID_LIGHT", "heavy": "GRID_HEAVY", "echo": "GRID_ECHO"}


def grid_of(step: str) -> Grid:
    """The grid of a step ("light", "heavy", "echo"), read at call time (tests make the grids smaller)."""
    return globals()[GRIDS[step]]


#: One run of a model: (lo, s, e, hi) in samples: the model reads [lo, hi), the block is [s, e).
Window = tuple[int, int, int, int]


def grid_windows(a: int, b: int, grid: Grid, ra: int | None = None, rb: int | None = None) -> list[Window]:
    """The windows of the grid blocks that touch [ra, rb) (default: all of [a, b)) of the stretch [a, b) of a file: the
    blocks are the file's grid cells cut to the stretch (a stretch that starts inside a cell has a short first block), and
    each block's context is clamped to the stretch (nothing is read before `a` or after `b`). A stretch is a part of
    the recording the model is run on in one piece: all of it, or one of pipeline.sound_spans."""
    ra = a if ra is None else max(a, ra)
    rb = b if rb is None else min(b, rb)
    if rb <= ra:
        return []
    size = grid.block
    out: list[Window] = []
    for k in range(ra // size, (rb - 1) // size + 1):
        s, e = max(a, k * size), min(b, (k + 1) * size)
        if e > s:
            out.append((max(a, s - grid.pre), s, e, min(b, e + grid.post)))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------------------------------------------------
class BlockCache(Protocol):
    def get(self, key: str) -> np.ndarray | None: ...
    def put(self, key: str, block: np.ndarray) -> None: ...
    def has(self, key: str) -> bool: ...


class MemoryCache:
    """Blocks in a dict (tests, short files)."""

    def __init__(self) -> None:
        self.blocks: dict[str, np.ndarray] = {}
        self.puts = 0

    def get(self, key: str) -> np.ndarray | None:
        return self.blocks.get(key)

    def put(self, key: str, block: np.ndarray) -> None:
        self.puts += 1
        self.blocks[key] = np.array(block, dtype=np.float32)

    def has(self, key: str) -> bool:
        return key in self.blocks


class DirCache:
    """Blocks as .npy files in a folder (one per key, named by the key's hash): they survive restarts and a multi-hour
    file's blocks do not sit in RAM. float32, so a block read back is bit for bit the block that was stored."""

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.puts = 0
        self._lock = threading.Lock()

    def path(self, key: str) -> Path:
        return self.dir / (hashlib.sha1(key.encode()).hexdigest() + ".npy")

    def get(self, key: str) -> np.ndarray | None:
        try:
            return np.load(self.path(key))
        except (FileNotFoundError, ValueError, EOFError):
            return None

    def put(self, key: str, block: np.ndarray) -> None:
        p = self.path(key)
        tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        with open(tmp, "wb") as f:
            np.save(f, np.asarray(block, dtype=np.float32))
        os.replace(tmp, p)
        with self._lock:
            self.puts += 1

    def has(self, key: str) -> bool:
        return self.path(key).is_file()


class FlacCache:
    """Blocks as 24-bit FLAC files in a folder (one per key, named by the key's hash): the server's block cache, about a third of
    the .npy's size (a 30 s mono block of speech is about 1.3 MB). A block read back is within one 24-bit step (1.2e-7 of full
    scale) of the block that was stored; a sample louder than CEILING is scaled down into it when written and the scale is kept in
    a small file next to it (`<name>.scale`, only then), so a block of any level comes back at its level.

    `ff` is an acw_common.ffmpeg.FFmpeg: needed to `put` and `get`; without one the cache can only say what it `has` (the web
    process plans with it: it never opens a file the sandboxed runner wrote). Only regular files count (never a link)."""

    CEILING = 0.98
    SUFFIX = ".flac"

    def __init__(self, directory: str | Path, ff=None, *, create: bool = True) -> None:
        self.dir = Path(directory)
        if create:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.ff = ff
        self.puts = 0
        self._lock = threading.Lock()

    def name(self, key: str) -> str:
        return hashlib.sha1(key.encode()).hexdigest()

    def path(self, key: str) -> Path:
        return self.dir / (self.name(key) + self.SUFFIX)

    def has(self, key: str) -> bool:
        try:
            return stat.S_ISREG(os.lstat(self.path(key)).st_mode)
        except OSError:
            return False

    def _scale(self, path: Path) -> float:
        try:
            v = float(path.with_suffix(".scale").read_text())
        except (OSError, ValueError):
            return 1.0
        return v if 0.0 < v <= 1.0 else 1.0

    def put(self, key: str, block: np.ndarray) -> None:
        if self.ff is None:
            raise ValueError("a FlacCache without an FFmpeg can only look")
        x = np.asarray(block, dtype=np.float32).reshape(-1)
        peak = float(np.max(np.abs(x))) if len(x) else 0.0
        scale = 1.0 if peak <= self.CEILING else self.CEILING / peak
        final = self.path(key)
        tmp = self.dir / f"{self.name(key)}-{os.getpid()}-{threading.get_ident()}"  # (no dot in it: the encoder sets the suffix)
        written = self.ff.write_audio(x, SAMPLE_RATE, tmp, "flac24", gain=scale)  # <tmp>.flac
        side = final.with_suffix(".scale")
        if scale != 1.0:
            side_tmp = side.with_name(side.name + ".tmp")
            side_tmp.write_text(repr(scale))
            os.replace(side_tmp, side)
        else:
            side.unlink(missing_ok=True)
        os.replace(written, final)
        with self._lock:
            self.puts += 1

    def get(self, key: str) -> np.ndarray | None:
        if not self.has(key):
            return None
        if self.ff is None:
            raise ValueError("a FlacCache without an FFmpeg can only look")
        path = self.path(key)
        r = self.ff._run([self.ff.ffmpeg, "-nostdin", "-hide_banner", "-v", "error", "-protocol_whitelist", "file", "-f", "flac", "-i",
                          str(path), "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"], timeout=300)
        if r.returncode != 0 or not r.stdout:
            return None
        y = np.frombuffer(r.stdout, dtype=np.float32)
        scale = self._scale(path)
        return (y / np.float32(scale)).astype(np.float32) if scale != 1.0 else y.copy()

    def bytes_used(self) -> int:
        """Disk the folder's blocks take."""
        total = 0
        try:
            for p in self.dir.iterdir():
                try:
                    st = os.lstat(p)
                    total += st.st_size if stat.S_ISREG(st.st_mode) else 0
                except OSError:
                    pass
        except OSError:
            pass
        return total


def block_key(step: str, input_id: str, *parts: object) -> str:
    """The cache key of a block: the step, the model that made it, the identity of the audio it was made from
    (`input_id`: anything that identifies the samples the step read, see editor.Inputs) and where it is (`parts`: channel,
    sample positions). Settings that do not change the model run (strength, loudness) are not in it."""
    return "|".join([f"acwb{VERSION}", step, MODEL_TAGS[step], input_id, *(str(p) for p in parts)])


def rebuild_key(input_id: str, job: object, b: int, n: int, window_seconds: float, overlap_seconds: float) -> str:
    """The cache key of Voice-rebuild block `b` of a recording of `n` samples (job: "mid" or the channel)."""
    return block_key("rebuild", input_id, job, f"w{window_seconds:g}o{overlap_seconds:g}", b, n)


@dataclass
class BlockHooks:
    """What a model run needs to use a cache without knowing about keys: `get(desc)` returns the block or None, `put(desc, block)`
    stores one. `desc` is a tuple of ints the model function names: (lo, s, e, hi) in the coordinates of the whole file for the
    noise removers, (job, block index) for Voice rebuild."""
    get: Callable[[tuple], np.ndarray | None]
    put: Callable[[tuple, np.ndarray], None]


def union_seconds(ranges: list[tuple[int, int]]) -> float:
    """Total length (seconds) of the union of sample ranges."""
    total, end = 0, -1
    for a, b in sorted(r for r in ranges if r[1] > r[0]):
        if a >= end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total / SAMPLE_RATE


def intersect(ranges: list[tuple[int, int]], a: int, b: int) -> list[tuple[int, int]]:
    return [(max(a, x), min(b, y)) for x, y in ranges if min(b, y) > max(a, x)]
