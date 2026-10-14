"""DeepFilterNet3 via libDF's C API (tract runtime, no torch).

`DeepFilter` wraps one stateful model instance (one mono channel, 48 kHz,
480-sample hops). ctypes drops the GIL during foreign calls, so separate
instances in separate threads run in parallel; `denoise_array` uses that to
split long files into overlapping segments.
"""
from __future__ import annotations

import ctypes
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SAMPLE_RATE = 48_000
# The model outputs audio delayed by this many samples (STFT + lookahead).
MODEL_DELAY = 1440

# File name of the libDF C-API build per platform (see native/deepfilter/build.sh).
LIB_NAME = {"win32": "df.dll", "darwin": "libdf.dylib"}.get(sys.platform, "libdf.so")
MODEL_FILE = "DeepFilterNet3_onnx.tar.gz"

_FP = ctypes.POINTER(ctypes.c_float)
_lib_lock = threading.Lock()
_libs: dict[str, ctypes.CDLL] = {}


def _load(lib_path: str) -> ctypes.CDLL:
    with _lib_lock:
        lib = _libs.get(lib_path)
        if lib is not None:
            return lib
        lib = ctypes.CDLL(lib_path)
        lib.df_create.restype = ctypes.c_void_p
        lib.df_create.argtypes = [ctypes.c_char_p, ctypes.c_float, ctypes.c_char_p]
        if hasattr(lib, "df_create_with_params"):
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
        _libs[lib_path] = lib
        return lib


class DeepFilter:
    def __init__(self, lib_path: str | Path, model_path: str | Path, atten_lim_db: float = 100.0, post_filter_beta: float = 0.0):
        self._lib = _load(str(lib_path))
        if hasattr(self._lib, "df_create_with_params"):
            # Always run both decoder stages: skipping them on "clean" frames
            # measurably hurts quality (PESQ 3.2 -> 2.1 in WorkAdventure's tests).
            st = self._lib.df_create_with_params(
                str(model_path).encode(), float(atten_lim_db), -math.inf, math.inf, math.inf, float(post_filter_beta), None
            )
        else:
            st = self._lib.df_create(str(model_path).encode(), float(atten_lim_db), None)
            if st and post_filter_beta:
                self._lib.df_set_post_filter_beta(st, float(post_filter_beta))
        if not st:
            raise RuntimeError("could not load DeepFilterNet model")
        self._st = ctypes.c_void_p(st)
        self.hop = int(self._lib.df_get_frame_length(self._st))
        self._in = np.zeros(self.hop, dtype=np.float32)
        self._out = np.zeros(self.hop, dtype=np.float32)

    def set_atten_lim(self, db: float) -> None:
        self._lib.df_set_atten_lim(self._st, float(db))

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """One hop of float32 mono audio in, one hop out (delayed by MODEL_DELAY)."""
        self._in[:] = frame
        lsnr = self._lib.df_process_frame(self._st, self._in.ctypes.data_as(_FP), self._out.ctypes.data_as(_FP))
        return self._out.copy(), float(lsnr)

    def process(self, x: np.ndarray) -> np.ndarray:
        """Run a whole buffer (len multiple of hop) through the stateful model."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        out = np.empty_like(x)
        hop = self.hop
        fn, st = self._lib.df_process_frame, self._st
        for i in range(0, len(x), hop):
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


def _denoise_contiguous(lib_path, model_path, x: np.ndarray, atten_lim_db: float) -> np.ndarray:
    df = DeepFilter(lib_path, model_path, atten_lim_db)
    try:
        hop = df.hop
        n = len(x)
        total = int(math.ceil((n + MODEL_DELAY) / hop)) * hop
        padded = np.zeros(total, dtype=np.float32)
        padded[:n] = x
        y = df.process(padded)
        return y[MODEL_DELAY:MODEL_DELAY + n]
    finally:
        df.close()


def denoise_array(
    x: np.ndarray,
    lib_path: str | Path,
    model_path: str | Path,
    *,
    atten_lim_db: float = 100.0,
    threads: int = 4,
    segment_seconds: float = 60.0,
    overlap_seconds: float = 2.0,
    progress=None,
) -> np.ndarray:
    """Denoise mono 48 kHz float32 audio. Long inputs are cut into segments
    with `overlap_seconds` of warm-up context on each side, processed in
    parallel, and joined with a short equal-power crossfade."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = len(x)
    seg = int(segment_seconds * SAMPLE_RATE)
    ov = int(overlap_seconds * SAMPLE_RATE)
    if n <= seg + 2 * ov or threads <= 1:
        out = _denoise_contiguous(lib_path, model_path, x, atten_lim_db)
        if progress:
            progress(1.0)
        return out

    bounds = [(s, min(s + seg, n)) for s in range(0, n, seg)]
    results: list[np.ndarray | None] = [None] * len(bounds)
    done = 0
    lock = threading.Lock()

    def work(i: int) -> None:
        nonlocal done
        s, e = bounds[i]
        lo, hi = max(0, s - ov), min(n, e + ov)
        y = _denoise_contiguous(lib_path, model_path, x[lo:hi], atten_lim_db)
        results[i] = y
        with lock:
            done += 1
            if progress:
                progress(done / len(bounds))

    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(work, range(len(bounds))))

    out = np.zeros(n, dtype=np.float32)
    fade = min(int(0.05 * SAMPLE_RATE), ov)  # 50 ms crossfade inside the warm overlap
    ramp = np.sin(np.linspace(0, np.pi / 2, fade, dtype=np.float32)) ** 2 if fade else None
    for i, (s, e) in enumerate(bounds):
        lo = max(0, s - ov)
        y = results[i]
        assert y is not None
        core = y[s - lo:s - lo + (e - s)]
        if i == 0 or ramp is None:
            out[s:e] = core
            continue
        # Blend the first `fade` samples of this segment with the previous
        # segment's tail, which extends past `s` inside its own overlap.
        prev_s, _prev_e = bounds[i - 1]
        prev_lo = max(0, prev_s - ov)
        prev = results[i - 1]
        tail = prev[s - prev_lo:s - prev_lo + fade]
        out[s:s + fade] = tail * (1 - ramp) + core[:fade] * ramp
        out[s + fade:e] = core[fade:]
    return out
