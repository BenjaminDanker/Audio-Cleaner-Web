"""DeepFilterNet3 via libDF's C API (tract runtime, no torch).

`DeepFilter` wraps one stateful model instance (one mono channel, 48 kHz,
480-sample hops). ctypes drops the GIL during foreign calls, so separate
instances in separate threads run in parallel; `denoise_array` uses that to
split long files into overlapping segments.
"""
from __future__ import annotations

import configparser
import ctypes
import functools
import math
import sys
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import blocks as _blocks

SAMPLE_RATE = 48_000
# DeepFilterNet3 outputs audio delayed by this many samples (STFT + 2 hops of
# lookahead). Other models (e.g. the low-latency DFN3-LL): see model_delay().
MODEL_DELAY = 1440

# File name of the libDF C-API build per platform (see native/deepfilter/build.sh).
LIB_NAME = {"win32": "df.dll", "darwin": "libdf.dylib"}.get(sys.platform, "libdf.so")
MODEL_FILE = "DeepFilterNet3_onnx.tar.gz"

_FP = ctypes.POINTER(ctypes.c_float)
# How often (in hops, 10 ms each) a running segment checks for cancellation.
_CANCEL_CHECK_HOPS = 500


class Cancelled(Exception):
    """Raised when the `cancel` event passed to a denoise call is set."""


@functools.lru_cache(maxsize=8)
def model_delay(model_path: str | Path) -> int:
    """Output delay in samples of a libDF model tarball, from its config.ini:
    the STFT overlap (fft_size - hop_size) plus the model's lookahead in hops."""
    with tarfile.open(model_path) as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("config.ini"))
        cfg = configparser.ConfigParser()
        cfg.read_string(tar.extractfile(member).read().decode())
    df = cfg["df"]
    hop = df.getint("hop_size")
    lookahead = max(df.getint("df_lookahead", 0), cfg.getint("deepfilternet", "conv_lookahead", fallback=0))
    return df.getint("fft_size") - hop + lookahead * hop


_lib_lock = threading.Lock()
_libs: dict[str, ctypes.CDLL] = {}


def _load(lib_path: str) -> ctypes.CDLL:
    with _lib_lock:
        lib = _libs.get(lib_path)
        if lib is not None:
            return lib
        lib = ctypes.CDLL(lib_path)
        lib.df_create_with_params.restype = ctypes.c_void_p
        lib.df_create_with_params.argtypes = [
            ctypes.c_char_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_char_p,
        ]
        lib.df_get_frame_length.restype = ctypes.c_size_t
        lib.df_get_frame_length.argtypes = [ctypes.c_void_p]
        lib.df_process_frame.restype = ctypes.c_float
        lib.df_process_frame.argtypes = [ctypes.c_void_p, _FP, _FP]
        lib.df_set_atten_lim.restype = None
        lib.df_set_atten_lim.argtypes = [ctypes.c_void_p, ctypes.c_float]
        lib.df_set_post_filter_beta.restype = None
        lib.df_set_post_filter_beta.argtypes = [ctypes.c_void_p, ctypes.c_float]
        lib.df_free.restype = None
        lib.df_free.argtypes = [ctypes.c_void_p]
        lib.df_reset.restype = ctypes.c_int
        lib.df_reset.argtypes = [ctypes.c_void_p]
        _libs[lib_path] = lib
        return lib


class DeepFilter:
    def __init__(self, lib_path: str | Path, model_path: str | Path, atten_lim_db: float = 100.0, post_filter_beta: float = 0.0):
        self._lib = _load(str(lib_path))
        # Always run both decoder stages: skipping them on "clean" frames
        # measurably hurts quality (PESQ 3.2 -> 2.1 in WorkAdventure's tests).
        st = self._lib.df_create_with_params(
            str(model_path).encode(), float(atten_lim_db), -math.inf, math.inf, math.inf, float(post_filter_beta), None
        )
        if not st:
            raise RuntimeError("could not load DeepFilterNet model")
        self._st = ctypes.c_void_p(st)
        self.hop = int(self._lib.df_get_frame_length(self._st))
        self.delay = model_delay(str(model_path))
        self._in = np.zeros(self.hop, dtype=np.float32)
        self._out = np.zeros(self.hop, dtype=np.float32)

    def reset(self) -> None:
        """Back to the state right after creation (the output for the next signal is then bit for bit what a
        new instance gives, without re-reading the model)."""
        if self._lib.df_reset(self._st) != 0:
            raise RuntimeError("could not reset DeepFilterNet state")

    def set_atten_lim(self, db: float) -> None:
        self._lib.df_set_atten_lim(self._st, float(db))

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """One hop of float32 mono audio in, one hop out (delayed by `self.delay`)."""
        self._in[:] = frame
        lsnr = self._lib.df_process_frame(self._st, self._in.ctypes.data_as(_FP), self._out.ctypes.data_as(_FP))
        return self._out.copy(), float(lsnr)

    def process(self, x: np.ndarray, cancel: threading.Event | None = None) -> np.ndarray:
        """Run a whole buffer (len multiple of hop) through the stateful model."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        out = np.empty_like(x)
        hop = self.hop
        fn, st = self._lib.df_process_frame, self._st
        check = hop * _CANCEL_CHECK_HOPS
        for i in range(0, len(x), hop):
            if cancel is not None and i % check == 0 and cancel.is_set():
                raise Cancelled()
            fn(st, x[i:i + hop].ctypes.data_as(_FP), out[i:i + hop].ctypes.data_as(_FP))
        return out

    def close(self) -> None:
        if self._st:
            self._lib.df_free(self._st)
            self._st = ctypes.c_void_p(0)

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


class _Instances:
    """One DeepFilter per worker thread for the segments of one `denoise_array` call, reset between segments
    instead of created per segment: creation re-reads the model file and hashes its weights each time."""

    def __init__(self, lib_path, model_path, atten_lim_db: float):
        self._args = (lib_path, model_path, atten_lim_db)
        self._local = threading.local()
        self._all: list[DeepFilter] = []
        self._lock = threading.Lock()

    def acquire(self) -> DeepFilter:
        df = getattr(self._local, "df", None)
        if df is not None:
            df.reset()
            return df
        df = self._local.df = DeepFilter(*self._args)
        with self._lock:
            self._all.append(df)
        return df

    def close(self) -> None:
        with self._lock:
            instances, self._all = self._all, []
        for df in instances:
            df.close()


def _denoise_contiguous(inst: _Instances, x: np.ndarray, cancel=None) -> np.ndarray:
    if cancel is not None and cancel.is_set():
        raise Cancelled()
    df = inst.acquire()
    hop, delay = df.hop, df.delay
    n = len(x)
    total = int(math.ceil((n + delay) / hop)) * hop
    padded = np.zeros(total, dtype=np.float32)
    padded[:n] = x
    return df.process(padded, cancel)[delay:delay + n]


def run_windows(
    x: np.ndarray,
    lib_path: str | Path,
    model_path: str | Path,
    windows: list[_blocks.Window],
    deliver,
    *,
    atten_lim_db: float = 100.0,
    threads: int = 4,
    progress=None,
    cancel: threading.Event | None = None,
) -> None:
    """Run the model over each window (lo, s, e, hi) of mono `x` (sample positions in x) on its own, from a fresh state:
    it reads x[lo:hi] and `deliver(i, block)` gets the result for [s, e) (float32, the model's delay removed). The windows
    are independent, so they run in `threads` threads; `deliver` is called under a lock. This is the block run of the
    editor's cache and of every whole-file run (`denoise_array`)."""
    if not windows:
        return
    inst = _Instances(lib_path, model_path, atten_lim_db)
    total = sum(e - s for _, s, e, _ in windows) or 1
    done = 0
    lock = threading.Lock()

    def work(i: int) -> None:
        nonlocal done
        lo, s, e, hi = windows[i]
        y = _denoise_contiguous(inst, np.ascontiguousarray(x[lo:hi], dtype=np.float32), cancel)
        core = y[s - lo:e - lo]
        with lock:
            deliver(i, core)
            done += e - s
            if progress:
                progress(min(done / total, 1.0))

    try:
        workers = max(1, min(threads, len(windows)))
        if workers == 1:
            for i in range(len(windows)):
                work(i)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(work, range(len(windows))))
    finally:
        inst.close()


def denoise_array(
    x: np.ndarray,
    lib_path: str | Path,
    model_path: str | Path,
    *,
    atten_lim_db: float = 100.0,
    threads: int = 4,
    segment_seconds: float | None = None,
    overlap_seconds: float | None = None,
    progress=None,
    cancel: threading.Event | None = None,
    out: np.ndarray | None = None,
    grid: _blocks.Grid | None = None,
    grid_offset: int = 0,
    hooks: _blocks.BlockHooks | None = None,
) -> np.ndarray:
    """Denoise mono 48 kHz float32 audio.

    The default is the editor's block run (blocks.GRID_LIGHT): the file is cut into 30 s blocks at multiples of 30 s from the
    start of the file, each run on its own from a fresh model state with 16 s of the recording before it and 0.25 s after
    it (>= 58 dB against one contiguous run, so no audible seams; compute x1.54), and the blocks are exact: no
    crossfade. The grid does not depend on `threads`, so the same input always gives the same blocks. `grid_offset`: the
    sample position of x[0] in the whole file, when x is only a stretch of it (the grid is the whole file's). `hooks`: a
    block cache (blocks.BlockHooks: the cached blocks are not run again, the others are stored) for full-strength runs.

    With `segment_seconds` and/or `overlap_seconds` the older cut is used instead: segments of that length (60 s) with
    `overlap_seconds` (0.5 s) of warm-up each side, run in parallel and joined with a 50 ms equal-power crossfade; a single
    thread runs the file contiguously.

    Results go straight into `out` (any writable 1-D float32 array or view of
    len(x), e.g. a column of a memory-mapped file), so memory use doesn't grow
    with the length of the input. Setting `cancel` makes the call raise
    `Cancelled` within ~5 s of audio."""
    n = len(x)
    if out is None:
        out = np.empty(n, dtype=np.float32)
    if segment_seconds is None and overlap_seconds is None:
        return _denoise_blocks(x, lib_path, model_path, out, atten_lim_db, threads, progress, cancel,
                               grid or _blocks.grid_of("light"), grid_offset, hooks)
    segment_seconds = 60.0 if segment_seconds is None else segment_seconds
    overlap_seconds = 0.5 if overlap_seconds is None else overlap_seconds
    seg = int(segment_seconds * SAMPLE_RATE)
    ov = int(overlap_seconds * SAMPLE_RATE)
    inst = _Instances(lib_path, model_path, atten_lim_db)
    if n <= seg + 2 * ov or threads <= 1:
        try:
            out[:] = _denoise_contiguous(inst, np.ascontiguousarray(x, dtype=np.float32), cancel)
        finally:
            inst.close()
        if progress:
            progress(1.0)
        return out

    bounds = [(s, min(s + seg, n)) for s in range(0, n, seg)]
    fade = min(int(0.05 * SAMPLE_RATE), ov)  # 50 ms crossfade inside the warm overlap
    heads: dict[int, np.ndarray] = {}  # first `fade` samples of segment i (i > 0), blended below
    tails: dict[int, np.ndarray] = {}  # `fade` samples segment i produced past its end
    done = 0
    lock = threading.Lock()

    def work(i: int) -> None:
        nonlocal done
        s, e = bounds[i]
        lo, hi = max(0, s - ov), min(n, e + ov)
        y = _denoise_contiguous(inst, np.ascontiguousarray(x[lo:hi], dtype=np.float32), cancel)
        core = y[s - lo:e - lo]
        head = fade if i > 0 else 0
        out[s + head:e] = core[head:]
        with lock:
            if head:
                heads[i] = core[:head].copy()
            if i + 1 < len(bounds):
                tails[i] = y[e - lo:e - lo + fade].copy()
            done += 1
            if progress:
                progress(done / len(bounds))

    try:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            list(pool.map(work, range(len(bounds))))
    finally:
        inst.close()

    if fade:
        ramp = np.sin(np.linspace(0, np.pi / 2, fade, dtype=np.float32)) ** 2
        for i in range(1, len(bounds)):
            s = bounds[i][0]
            out[s:s + fade] = tails[i - 1] * (1 - ramp) + heads[i] * ramp
    return out


def _denoise_blocks(x, lib_path, model_path, out, atten_lim_db, threads, progress, cancel, grid, offset, hooks) -> np.ndarray:
    n = len(x)
    if hooks is not None and atten_lim_db < 100.0:
        raise ValueError("cached blocks are full-strength runs: the strength is a blend afterwards")
    wins = _blocks.grid_windows(offset, offset + n, grid)
    todo: list[_blocks.Window] = []
    hit_samples = 0
    for lo, s, e, hi in wins:
        w = (lo - offset, s - offset, e - offset, hi - offset)
        hit = hooks.get((lo, s, e, hi)) if hooks is not None else None
        if hit is not None and len(hit) == e - s:
            out[w[1]:w[2]] = hit
            hit_samples += e - s
        else:
            todo.append(w)

    def deliver(i: int, core: np.ndarray) -> None:
        lo, s, e, hi = todo[i]
        out[s:e] = core
        if hooks is not None:
            hooks.put((lo + offset, s + offset, e + offset, hi + offset), core)

    base = hit_samples / n if n else 1.0

    def prog(f: float) -> None:
        if progress:
            progress(base + f * (1.0 - base))

    run_windows(x, lib_path, model_path, todo, deliver, atten_lim_db=atten_lim_db, threads=threads, progress=prog, cancel=cancel)
    if progress:
        progress(1.0)
    return out
