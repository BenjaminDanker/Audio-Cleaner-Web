#!/usr/bin/env python3
"""Clarity pipeline diagnostics tool (local).

Runs the same stages as audio_clarity_pipeline step-by-step, writes a WAV after
each stage, and prints metrics so you ca        entry = {
            "name": name,
            "rms_dbfs": _dbfs(rms),
            "peak_dbfs": _dbfs_peak(peak),
            "spectral_low": low,
            "spectral_high": high,
            "spectral_high_minus_low_db": _dbfs(max(high,1e-12)) - _dbfs(max(low,1e-12)),
            "noise_floor_proxy": noise,
            "snr_proxy_db": _dbfs(rms / max(noise, 1e-12)),
            "lufs": lufs,
            "ess_ratio_5k9k_to_300_3k": ess_ratio(sig, fs),
        }y verify changes.

Usage (Windows PowerShell):
  python .\processor\tools\clarity_diagnostics.py -i path\to\audio_or_video.ext -o .\diag_out \
    --denoise-atten 30 --dereverb-strength 0.15 --gate-threshold-db -48 --shelf-gain-db 3 --lufs-target -14

Notes:
- Requires Python deps from processor/requirements.txt
- Uses FFmpeg if available to extract/normalize via MediaExtractor; otherwise falls back to soundfile + resample.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple
import time
import shutil

import numpy as np
import torch


def _add_paths():
    here = Path(__file__).resolve()
    processor_dir = here.parents[1]
    shared_dir = processor_dir / "shared"
    batch_dir = processor_dir / "batch"
    for p in (processor_dir, shared_dir, batch_dir):
        if str(p) not in sys.path:
            sys.path.append(str(p))


_add_paths()

# Now safe to import project modules
from shared.ai.audio_clarity_pipeline import (  # type: ignore
    ClarityParams,
    _ensure_mono_f32,
    _dereverb_spectral_subtraction,
    _apply_noise_gate,
    _butter_highpass,
    _simple_high_shelf,
    _apply_compressor,
    _apply_limiter,
)
from shared.ai.audio_denoise_dfnet import _get_enhancer  # type: ignore
from df.enhance import enhance  # type: ignore

try:
    import soundfile as sf
except Exception as e:  # pragma: no cover
    print("soundfile import failed:", e)
    raise

try:
    import pyloudnorm as pyln  # type: ignore
except Exception:
    pyln = None

try:
    from batch.media_extractor import MediaExtractor  # type: ignore
except Exception:
    MediaExtractor = None  # type: ignore

try:
    from scipy.signal import resample_poly  # type: ignore
except Exception:
    resample_poly = None

import torch
from df.enhance import enhance  # type: ignore


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def _dbfs_peak(val: float) -> float:
    # Only for *peak* on a normalized [-1, 1] waveform
    v = max(min(val, 1.0), 1e-12)
    return 20.0 * math.log10(v)

def _dbfs(val: float) -> float:
    # Generic dB for arbitrary linear magnitudes/ratios (no clamp!)
    return 20.0 * math.log10(max(val, 1e-12))


def _spectral_bands(x: np.ndarray, fs: int) -> Tuple[float, float]:
    # Rough band energies: <1kHz and >3kHz
    n = int(2 ** math.ceil(math.log2(len(x) + 1)))
    spec = np.fft.rfft(x, n=n)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    low = float(np.mean(mag[(freqs < 1000)])) if np.any(freqs < 1000) else 0.0
    high = float(np.mean(mag[(freqs > 3000)])) if np.any(freqs > 3000) else 0.0
    return low, high


def _noise_floor_proxy(x: np.ndarray) -> float:
    # 10th percentile absolute amplitude as noise proxy
    return float(np.percentile(np.abs(x), 10))


def ess_ratio(x: np.ndarray, fs: int, lo: int = 5500, hi: int = 9000, voice_lo: int = 300, voice_hi: int = 3500) -> float:
    """Compute ess_ratio for backward compatibility - returns just the ratio."""
    from shared.ai.audio_clarity_pipeline import _ess_ratio  # type: ignore
    ratio, _, _ = _ess_ratio(x, fs, lo, hi, voice_lo, voice_hi)
    return ratio


def _loudness_lufs(x: np.ndarray, fs: int) -> float | None:
    if pyln is None:
        return None
    try:
        meter = pyln.Meter(fs)
        return float(meter.integrated_loudness(x.astype(np.float32)))
    except Exception:
        return None


def _write_wav(path: Path, x: np.ndarray, fs: int):
    sf.write(str(path), x.astype(np.float32, copy=False), fs, subtype="PCM_16")


def _extract_or_read(input_path: Path, work: Path, target_sr: int) -> Tuple[np.ndarray, int, Path, float]:
    """Return mono f32 audio at target_sr and path used as source for diagnostics, plus duration (ms)."""
    t0 = time.perf_counter()
    if MediaExtractor is not None:
        # Normalize via extractor (uses system FFmpeg)
        try:
            extr = MediaExtractor(target_sr)
            res = extr.extract(str(input_path), str(work))
            audio, sr = sf.read(res.extracted_wav_path, dtype="float32")
            x = _ensure_mono_f32(audio)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            return x, sr, Path(res.extracted_wav_path), dt_ms
        except Exception:
            pass
    # Fallback: direct read + optional resample
    audio, sr = sf.read(str(input_path), dtype="float32")
    x = _ensure_mono_f32(audio)
    if sr != target_sr:
        if resample_poly is None:
            raise RuntimeError("Need scipy to resample when ffmpeg/extractor not available")
        from math import gcd
        g = gcd(sr, target_sr)
        up, down = target_sr // g, sr // g
        x = resample_poly(x, up, down).astype(np.float32, copy=False)
        sr = target_sr
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return x, sr, input_path, dt_ms


def run_diagnostics(input_file: Path, out_dir: Path, params: ClarityParams) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Pre-initialize enhancer (including model loading) before any timing
    enh = _get_enhancer()
    fs = enh.sample_rate
    
    # Warm up the model with a tiny dummy array to ensure it's fully loaded
    dummy = torch.zeros((1, 1000), dtype=torch.float32)
    with torch.no_grad():
        _ = enhance(enh.model, enh.df_state, dummy, atten_lim_db=30)
    
    # Stage: extraction/normalization
    x, sr, src_path, extract_ms = _extract_or_read(input_file, out_dir, fs)

    report: Dict = {
        "source": str(input_file),
        "normalized_from": str(src_path),
        "fs": sr,
        "extract_ms": extract_ms,
        "stages": [],
    }

    total_start = time.perf_counter()

    def stage(name: str, sig: np.ndarray, proc_ms: float | None = None, io_ms: float | None = None) -> np.ndarray:
        rms = _rms(sig)
        peak = float(np.max(np.abs(sig)))
        low, high = _spectral_bands(sig, fs)
        noise = _noise_floor_proxy(sig)
        snr_proxy = _dbfs(rms / max(noise, 1e-12))
        lufs = _loudness_lufs(sig, fs)
        entry = {
            "name": name,
            "rms_dbfs": _dbfs(rms),
            "peak_dbfs": _dbfs(peak),
            "spectral_low": low,
            "spectral_high": high,
            "spectral_high_minus_low_db": _dbfs(max(high,1e-12)) - _dbfs(max(low,1e-12)),
            "noise_floor_proxy": noise,
            "snr_proxy_db": snr_proxy,
            "lufs": lufs,
            "ess_ratio_5k9k_to_300_3k": ess_ratio(sig, fs),
        }
        # Back-compat: duration_ms remains the processing time
        if proc_ms is not None:
            entry["duration_ms"] = proc_ms
            entry["proc_ms"] = proc_ms
        if io_ms is not None:
            entry["io_ms"] = io_ms
        report["stages"].append(entry)
        _write_wav(out_dir / f"{len(report['stages']):02d}_{name}.wav", sig, fs)
        return sig

    # 1) Denoise (DFNet)
    x = _ensure_mono_f32(x)
    t0 = time.perf_counter()
    x_t = torch.from_numpy(x.astype(np.float32, copy=False))
    if x_t.ndim == 1:
        x_t = x_t.unsqueeze(0)
    x_t = x_t.contiguous()
    x_enh = enhance(enh.model, enh.df_state, x_t, atten_lim_db=enh.clamp_atten(params.denoise_atten_db))
    if x_enh.ndim > 1:
        x_enh = x_enh.squeeze(0)
    x = x_enh.detach().cpu().numpy().astype(np.float32, copy=False)
    proc_ms = (time.perf_counter() - t0) * 1000.0
    t_io = time.perf_counter()
    x = stage("denoise_dfnet", x, proc_ms=proc_ms, io_ms=None)
    io_ms = (time.perf_counter() - t_io) * 1000.0
    report["stages"][-1]["io_ms"] = io_ms

    # 2) Noise gate
    from shared.ai.audio_clarity_pipeline import StreamState  # type: ignore
    st = StreamState()
    t0 = time.perf_counter()
    x = _apply_noise_gate(x, fs, params.gate_threshold_db, params.gate_ratio, params.gate_attack_ms, params.gate_release_ms, st)
    proc_ms = (time.perf_counter() - t0) * 1000.0
    t_io = time.perf_counter()
    x = stage("noise_gate", x, proc_ms=proc_ms, io_ms=None)
    report["stages"][-1]["io_ms"] = (time.perf_counter() - t_io) * 1000.0

    # 3) EQ: HPF + high-shelf
    t0 = time.perf_counter()
    b, a = _butter_highpass(params.highpass_hz, fs, order=2)
    from scipy.signal import lfilter  # type: ignore
    x = lfilter(b, a, x)
    x = _simple_high_shelf(x, fs, params.shelf_freq_hz, params.shelf_gain_db)
    proc_ms = (time.perf_counter() - t0) * 1000.0
    t_io = time.perf_counter()
    x = stage("eq_tilt", x, proc_ms=proc_ms, io_ms=None)
    report["stages"][-1]["io_ms"] = (time.perf_counter() - t_io) * 1000.0

    # 4) Loudness normalize BEFORE compression (fixed threshold behavior)
    t0 = time.perf_counter()
    from shared.ai.audio_clarity_pipeline import _loudness_normalize_file  # type: ignore
    x = _loudness_normalize_file(x, fs, params.lufs_target, params.limit_ceiling_dbfs)
    proc_ms = (time.perf_counter() - t0) * 1000.0
    t_io = time.perf_counter()
    x = stage("loudness_normalize", x, proc_ms=proc_ms, io_ms=None)
    report["stages"][-1]["io_ms"] = (time.perf_counter() - t_io) * 1000.0

    # 5) Compressor AFTER normalization with fixed dialog settings
    t0 = time.perf_counter()
    # Compute and log GR from detector; then apply compressor with same settings
    from shared.ai.audio_clarity_pipeline import _compressor_gain_trace  # type: ignore
    env, gr_s, gain = _compressor_gain_trace(
        x, fs,
        thr_db=float(params.comp_threshold_db),
        ratio=float(params.comp_ratio),
    )
    # Gain reduction in dB (negative numbers) using smoothed GR trace
    gr_db = gr_s.astype(np.float32, copy=False)
    # Compute positive "reduction" from the smoothed GR trace
    reduction_db = -gr_db  # positive numbers = amount of reduction
    
    # Use reduction values for all stats (positive = amount of reduction)
    avg_gr_db = float(np.percentile(reduction_db, 50))  # median reduction
    max_gr_db = float(np.max(reduction_db))             # deepest reduction
    x = _apply_compressor(
        x, fs,
        thr_db=float(params.comp_threshold_db),
        ratio=float(params.comp_ratio),
        attack_ms=float(params.comp_attack_ms),
        release_ms=float(params.comp_release_ms),
        makeup_db=float(params.comp_makeup_db),
        state=st,
    )
    proc_ms = (time.perf_counter() - t0) * 1000.0
    t_io = time.perf_counter()
    x = stage("compressor", x, proc_ms=proc_ms, io_ms=None)
    report["stages"][-1]["avg_gr_db"] = avg_gr_db
    report["stages"][-1]["max_gr_db"] = max_gr_db
    # Also log percentiles of reduction (positive numbers, dB of reduction)
    p50, p90, p99 = np.percentile(reduction_db, [50, 90, 99])
    report["stages"][-1]["comp_gr_db_p50"] = float(p50)
    report["stages"][-1]["comp_gr_db_p90"] = float(p90)
    report["stages"][-1]["comp_gr_db_p99"] = float(p99)
    report["stages"][-1]["comp_gr_db_min"] = float(np.min(reduction_db))  # usually 0
    report["stages"][-1]["comp_gr_db_max"] = float(np.max(reduction_db))  # deepest
    report["stages"][-1]["io_ms"] = (time.perf_counter() - t_io) * 1000.0

    # 6) Dynamic presence notch after compressor (driven by ess_ratio)
    t0 = time.perf_counter()
    from shared.ai.audio_clarity_pipeline import _apply_dynamic_presence_notch  # type: ignore
    x, applied_depth_db = _apply_dynamic_presence_notch(x, fs)
    proc_ms = (time.perf_counter() - t0) * 1000.0
    t_io = time.perf_counter()
    x = stage("presence_notch_dyn", x, proc_ms=proc_ms, io_ms=None)
    report["stages"][-1]["applied_depth_db"] = applied_depth_db
    report["stages"][-1]["ess_ratio_after_notch"] = report["stages"][-1]["ess_ratio_5k9k_to_300_3k"]
    report["stages"][-1]["io_ms"] = (time.perf_counter() - t_io) * 1000.0

    # 7) Final limiter
    t0 = time.perf_counter()
    x = _apply_limiter(x, params.limit_ceiling_dbfs)
    proc_ms = (time.perf_counter() - t0) * 1000.0
    t_io = time.perf_counter()
    x = stage("limiter", x, proc_ms=proc_ms, io_ms=None)
    report["stages"][-1]["io_ms"] = (time.perf_counter() - t_io) * 1000.0

    # Totals
    report["total_wall_ms"] = (time.perf_counter() - total_start) * 1000.0
    # Sum processing-only times
    report["total_proc_ms"] = sum(float(s.get("proc_ms", s.get("duration_ms", 0.0))) for s in report["stages"]) + float(extract_ms)

    # Final metrics JSON
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main():
    ap = argparse.ArgumentParser(description="Run clarity diagnostics")
    ap.add_argument("-i", "--input", required=True, help="Path to input audio/video file")
    ap.add_argument("-o", "--outdir", default="./diag_out", help="Output directory for stage WAVs and metrics.json")
    # Parameter overrides
    ap.add_argument("--denoise-atten", type=int, default=30)
    ap.add_argument("--dereverb-strength", type=float, default=0.0, help="Deprecated; DFNet3 post-filter handles mild dereverb. Default off.")
    ap.add_argument("--gate-threshold-db", type=float, default=-55.0)
    ap.add_argument("--gate-ratio", type=float, default=0.1)
    ap.add_argument("--gate-attack-ms", type=float, default=5.0)
    ap.add_argument("--gate-release-ms", type=float, default=50.0)
    ap.add_argument("--highpass-hz", type=float, default=150.0)
    ap.add_argument("--shelf-freq-hz", type=float, default=3500.0)
    ap.add_argument("--shelf-gain-db", type=float, default=0.0)
    # De-esser removed
    ap.add_argument("--comp-threshold-db", type=float, default=-25.0)
    ap.add_argument("--comp-ratio", type=float, default=2.0)
    ap.add_argument("--comp-attack-ms", type=float, default=2.0)
    ap.add_argument("--comp-release-ms", type=float, default=60.0)
    ap.add_argument("--comp-makeup-db", type=float, default=0.0)
    ap.add_argument("--limit-ceiling-dbfs", type=float, default=-1.0)
    ap.add_argument("--lufs-target", type=float, default=-14.0)

    args = ap.parse_args()
    inp = Path(args.input).resolve()
    outd = Path(args.outdir).resolve()

    params = ClarityParams(
        denoise_atten_db=args.denoise_atten,
        dereverb_strength=args.dereverb_strength,
        gate_threshold_db=args.gate_threshold_db,
        gate_ratio=args.gate_ratio,
        gate_attack_ms=args.gate_attack_ms,
        gate_release_ms=args.gate_release_ms,
        highpass_hz=args.highpass_hz,
        shelf_freq_hz=args.shelf_freq_hz,
        shelf_gain_db=args.shelf_gain_db,
    # de-esser removed
        comp_threshold_db=args.comp_threshold_db,
        comp_ratio=args.comp_ratio,
        comp_attack_ms=args.comp_attack_ms,
        comp_release_ms=args.comp_release_ms,
        comp_makeup_db=args.comp_makeup_db,
        limit_ceiling_dbfs=args.limit_ceiling_dbfs,
        lufs_target=args.lufs_target,
    )

    rep = run_diagnostics(inp, outd, params)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    import shutil  # after function defs
    main()
