#!/usr/bin/env python3
"""Clarity pipeline diagnostics tool (local).

Runs the same stages as audio_clarity_pipeline step-by-step, writes a WAV after
each stage, and prints metrics so you can objectively verify changes.

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


def _dbfs(val: float) -> float:
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
    if MediaExtractor is not None and (shutil.which("ffmpeg") or os.environ.get("FFMPEG_PATH")):
        # Normalize via extractor to match production path
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
    enh = _get_enhancer()
    fs = enh.sample_rate

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

    def stage(name: str, sig: np.ndarray, duration_ms: float | None = None) -> np.ndarray:
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
        }
        if duration_ms is not None:
            entry["duration_ms"] = duration_ms
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
    x = stage("denoise_dfnet", x, (time.perf_counter() - t0) * 1000.0)

    # 2) Dereverb
    t0 = time.perf_counter()
    x = _dereverb_spectral_subtraction(x, fs, params.dereverb_strength)
    x = stage("dereverb", x, (time.perf_counter() - t0) * 1000.0)

    # 3) Noise gate
    from shared.ai.audio_clarity_pipeline import StreamState  # type: ignore
    st = StreamState()
    t0 = time.perf_counter()
    x = _apply_noise_gate(x, fs, params.gate_threshold_db, params.gate_ratio, params.gate_attack_ms, params.gate_release_ms, st)
    x = stage("noise_gate", x, (time.perf_counter() - t0) * 1000.0)

    # 4) EQ: HPF + high-shelf
    t0 = time.perf_counter()
    b, a = _butter_highpass(params.highpass_hz, fs, order=2)
    from scipy.signal import lfilter  # type: ignore
    x = lfilter(b, a, x)
    x = _simple_high_shelf(x, fs, params.shelf_freq_hz, params.shelf_gain_db)
    x = stage("eq_tilt", x, (time.perf_counter() - t0) * 1000.0)

    # 5) Compressor
    t0 = time.perf_counter()
    x = _apply_compressor(x, fs, params.comp_threshold_db, params.comp_ratio, params.comp_attack_ms, params.comp_release_ms, params.comp_makeup_db, st)
    x = stage("compressor", x, (time.perf_counter() - t0) * 1000.0)

    # 6) Loudness normalize
    t0 = time.perf_counter()
    from shared.ai.audio_clarity_pipeline import _loudness_normalize_file  # type: ignore
    x = _loudness_normalize_file(x, fs, params.lufs_target, params.limit_ceiling_dbfs)
    x = stage("loudness_normalize", x, (time.perf_counter() - t0) * 1000.0)

    # 7) Final limiter after normalization
    t0 = time.perf_counter()
    x = _apply_limiter(x, params.limit_ceiling_dbfs)
    x = stage("limiter", x, (time.perf_counter() - t0) * 1000.0)

    report["total_time_ms"] = (time.perf_counter() - total_start) * 1000.0

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
    ap.add_argument("--dereverb-strength", type=float, default=0.15)
    ap.add_argument("--gate-threshold-db", type=float, default=-48.0)
    ap.add_argument("--gate-ratio", type=float, default=0.2)
    ap.add_argument("--gate-attack-ms", type=float, default=5.0)
    ap.add_argument("--gate-release-ms", type=float, default=50.0)
    ap.add_argument("--highpass-hz", type=float, default=150.0)
    ap.add_argument("--shelf-freq-hz", type=float, default=3500.0)
    ap.add_argument("--shelf-gain-db", type=float, default=3.0)
    ap.add_argument("--comp-threshold-db", type=float, default=-18.0)
    ap.add_argument("--comp-ratio", type=float, default=3.0)
    ap.add_argument("--comp-attack-ms", type=float, default=5.0)
    ap.add_argument("--comp-release-ms", type=float, default=100.0)
    ap.add_argument("--comp-makeup-db", type=float, default=3.0)
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
