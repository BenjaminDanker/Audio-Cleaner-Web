#!/usr/bin/env python3
"""Smoke test + benchmark for a libDF shared library (ctypes).

    ~/audio-cleaner/venv/bin/python native/deepfilter/test_native.py <libdf.so> <model.tar.gz> \
        [--old <old-libdf.so>] [--seconds 20] [--save PREFIX]

Loads the model through df_create_with_params (always-on stages), processes synthetic noisy
speech-like audio frame by frame, prints load time / ms per frame / real-time factor, checks the
output is finite and, if an older library is given (default: ~/audio-cleaner/bin/libdf-tract021.so,
or libdf.so when that does not exist and is not the library under test), runs it on the same input
via df_create and prints its timing and the difference to the new output.
"""
import argparse
import ctypes as C
import os
import sys
import time

import numpy as np

SR = 48000
BIN = os.path.expanduser("~/audio-cleaner/bin")
INF = float("inf")


def synth(seconds: float, seed: int = 1) -> np.ndarray:
    """Deterministic 'speech-like' signal in noise: gliding harmonic voice with syllabic
    amplitude modulation and pauses, plus coloured noise and a few clicks."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    f0 = 120 + 40 * np.sin(2 * np.pi * 0.3 * t) + 15 * np.sin(2 * np.pi * 2.1 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    voice = sum(np.sin(k * phase) / k for k in range(1, 25))
    # crude formant-ish shaping: slowly moving low-pass via a mix of two envelopes
    env = np.clip(np.sin(2 * np.pi * 3.5 * t) + 0.3, 0, None) ** 1.5
    env *= (np.sin(2 * np.pi * 0.4 * t) > -0.3)  # pauses
    voice = 0.25 * voice / np.abs(voice).max() * env
    white = rng.standard_normal(n)
    # pink-ish noise: 1/f via FFT shaping
    spec = np.fft.rfft(white)
    spec /= np.sqrt(np.maximum(np.arange(spec.size), 1))
    noise = np.fft.irfft(spec, n)
    noise = 0.05 * noise / noise.std()
    x = voice + noise
    for c in rng.integers(0, n - 200, 6):  # keyboard-click like transients
        x[c:c + 100] += 0.2 * rng.standard_normal(100) * np.hanning(100)
    return np.clip(x, -0.98, 0.98).astype(np.float32)


def load(path: str):
    lib = C.CDLL(path)
    lib.df_create.restype = C.c_void_p
    lib.df_create.argtypes = [C.c_char_p, C.c_float, C.c_char_p]
    lib.df_get_frame_length.restype = C.c_size_t
    lib.df_get_frame_length.argtypes = [C.c_void_p]
    lib.df_process_frame.restype = C.c_float
    lib.df_process_frame.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p]
    lib.df_free.argtypes = [C.c_void_p]
    lib.df_free.restype = None
    if hasattr(lib, "df_create_with_params"):
        lib.df_create_with_params.restype = C.c_void_p
        lib.df_create_with_params.argtypes = [C.c_char_p, C.c_float, C.c_float, C.c_float,
                                              C.c_float, C.c_float, C.c_char_p]
        lib.df_last_error.restype = C.c_char_p
        lib.df_last_error.argtypes = []
    return lib


def run(lib, model: bytes, x: np.ndarray, label: str):
    t0 = time.perf_counter()
    if hasattr(lib, "df_create_with_params"):
        st = lib.df_create_with_params(model, 100.0, -INF, INF, INF, 0.0, None)
        if not st:
            sys.exit(f"{label}: df_create_with_params failed: {lib.df_last_error()!r}")
    else:
        st = lib.df_create(model, 100.0, None)
        if not st:
            sys.exit(f"{label}: df_create failed")
    load_s = time.perf_counter() - t0
    hop = lib.df_get_frame_length(st)
    nfr = len(x) // hop
    x = np.ascontiguousarray(x[: nfr * hop])
    y = np.zeros_like(x)
    lsnr = np.zeros(nfr, np.float32)
    per = np.zeros(nfr)
    xp, yp = x.ctypes.data, y.ctypes.data
    cpu0 = time.process_time()
    wall0 = time.perf_counter()
    for i in range(nfr):
        a = time.perf_counter()
        lsnr[i] = lib.df_process_frame(st, xp + 4 * i * hop, yp + 4 * i * hop)
        per[i] = time.perf_counter() - a
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    lib.df_free(st)
    ms = 1e3 * wall / nfr
    print(f"[{label}] load {load_s * 1e3:7.1f} ms | hop {hop} | {nfr} frames | "
          f"{ms:.3f} ms/frame (p50 {1e3 * np.median(per):.3f}, p99 {1e3 * np.percentile(per, 99):.3f}, "
          f"max {1e3 * per.max():.3f}) | RTF {ms / (1e3 * hop / SR):.4f} | cpu/wall {cpu / wall:.2f}")
    return y, lsnr, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lib")
    ap.add_argument("model")
    ap.add_argument("--old", default=None, help="older libdf.so to compare against")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--save", metavar="PREFIX",
                    help="write PREFIX.in.f32 / PREFIX.out.f32 (raw f32) for test_wasm.mjs --input/--ref")
    a = ap.parse_args()

    model = os.path.abspath(os.path.expanduser(a.model)).encode()
    x = synth(a.seconds)
    print(f"input: {len(x) / SR:.1f} s synthetic noisy speech-like signal, rms {np.sqrt((x ** 2).mean()):.4f}")

    # error path: a bad model path must give NULL, not a crash
    lib = load(a.lib)
    if hasattr(lib, "df_create_with_params"):
        bad = lib.df_create_with_params(b"/nonexistent/model.tar.gz", 100.0, -INF, INF, INF, 0.0, None)
        assert not bad, "expected NULL for a missing model"
        print("missing model -> NULL, df_last_error:", lib.df_last_error())
        assert not lib.df_create_with_params(model, 100.0, -INF, INF, INF, -1.0, None), "negative beta must fail"
        assert not lib.df_create_with_params(model, 100.0, -INF, INF, INF, 0.0, b"loud"), "bad log level must fail"
        assert lib.df_get_frame_length(None) == 0
        buf = np.zeros(480, np.float32)
        assert np.isnan(lib.df_process_frame(None, buf.ctypes.data, buf.ctypes.data)), "NULL state -> NaN"
        print("bad args (missing model, negative beta, bad log level, NULL state) handled without crashing")

    y, lsnr, ms = run(lib, model, x, "new")
    ok = bool(np.isfinite(y).all() and np.isfinite(lsnr).all())
    rms_out = float(np.sqrt((y ** 2).mean()))
    print(f"output finite: {ok} | out rms {rms_out:.4f} | lsnr {lsnr.min():.1f}..{lsnr.max():.1f} dB "
          f"(mean {lsnr.mean():.1f})")
    if not ok or rms_out == 0.0:
        sys.exit("FAIL: non-finite or silent output")

    if a.save:
        hop = len(x) // len(lsnr)
        x[: len(lsnr) * hop].tofile(a.save + ".in.f32")
        y.tofile(a.save + ".out.f32")
        print(f"saved {a.save}.in.f32 / {a.save}.out.f32")

    old_path = a.old
    if old_path is None:
        for cand in ("libdf-tract021.so", "libdf.so"):
            p = os.path.join(BIN, cand)
            if os.path.exists(p) and not os.path.samefile(p, a.lib):
                old_path = p
                break
    if old_path and os.path.exists(old_path):
        old = load(old_path)
        if hasattr(old, "df_create_with_params"):
            print(f"(skipping comparison: {old_path} is not an old build)")
        else:
            yo, lo, ms_old = run(old, model, x, "old")
            n = min(len(y), len(yo))
            # old build hard-codes thresholds -15/35/35 dB; only frames inside that range are comparable
            hop = len(x) // len(lsnr)
            inside = (lsnr > -15) & (lsnr < 35)
            mask = np.repeat(inside, hop)[:n]
            d = y[:n][mask] - yo[:n][mask]
            ref = yo[:n][mask]
            snr = 10 * np.log10((ref ** 2).sum() / max((d ** 2).sum(), 1e-20))
            print(f"new vs old output (frames with lsnr in [-15,35] dB: {inside.mean() * 100:.0f}%): "
                  f"SNR {snr:.1f} dB, max |diff| {np.abs(d).max():.2e}")
            print(f"speed: old {ms_old:.3f} ms/frame -> new {ms:.3f} ms/frame ({ms_old / ms:.2f}x)")
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
