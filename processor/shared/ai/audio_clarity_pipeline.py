"""CPU-only speech clarity pipeline for file and streaming use.

Stages (in order):
 1) Denoise (DeepFilterNet3 wrapper)
 2) Dereverb (light spectral subtraction)
 3) Noise Gate (level-based attenuation between words)
 4) EQ Tilt (HPF ~150 Hz + gentle high-shelf 2–5 kHz)
 5) Compression/Leveler (fast attack, moderate ratio)
 6) Limiter (ceiling -1 dBFS)
 7) Loudness normalization (-14 LUFS target)

Interfaces:
  - process_file(in_path, work_dir, params) -> (processed_wav_path, sample_rate)
  - process_stream_chunk(mono_f32, sr, state, params) -> (processed_chunk, updated_state)

Notes:
  - All DSP is CPU-friendly (NumPy + SciPy). DeepFilterNet runs on CPU via torch cpu wheels.
  - Keep per-2s chunk latency small by using lightweight vectorized operations and reusing state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import os

import numpy as np
import torch
import soundfile as sf
try:
    from scipy.signal import butter, lfilter
except Exception:  # pragma: no cover - optional dependency during dev
    butter = None  # type: ignore
    def lfilter(b, a, x):  # type: ignore
        return x

from df.enhance import enhance, init_df, load_audio, save_audio  # type: ignore

# Global enhancer for streaming (no media_extractor dependency)
_GLOBAL_ENHANCER = None

def _get_global_enhancer():
    """Get singleton enhancer without media_extractor dependency."""
    global _GLOBAL_ENHANCER
    if _GLOBAL_ENHANCER is None:
        from pathlib import Path
        import os
        base_dir = Path(__file__).parent.parent.parent
        root = getattr(__import__('sys'), "_MEIPASS", str(base_dir))
        models_root = os.path.join(root, "models/DeepFilterNet3")
        model, df_state, _ = init_df(models_root, post_filter=True)
        
        class SimpleEnhancer:
            def __init__(self, model, df_state):
                self.model = model
                self.df_state = df_state
            
            def clamp_atten(self, atten_db):
                if atten_db is None:
                    return None
                try:
                    val = int(atten_db)
                except (ValueError, TypeError):
                    val = 30
                return max(-10, min(80, val))
            
            @property
            def sample_rate(self):
                return self.df_state.sr()
        
        _GLOBAL_ENHANCER = SimpleEnhancer(model, df_state)
    return _GLOBAL_ENHANCER


# ---------------------------- Utilities ----------------------------

def _db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def _lin_to_db(lin: float, eps: float = 1e-12) -> float:
    return 20.0 * float(np.log10(max(lin, eps)))


def _ensure_mono_f32(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        # average channels if stereo
        x = x.mean(axis=1)
    return x.astype(np.float32, copy=False)


# ---------------------------- Parameters ----------------------------


@dataclass
class ClarityParams:
    # Denoise
    denoise_atten_db: Optional[int] = 30

    # Dereverb
    dereverb_strength: float = 0.15  # 0..1, spectral floor subtraction amount

    # Noise gate
    gate_threshold_db: float = -48.0
    gate_ratio: float = 0.2  # linear attenuation when below threshold
    gate_attack_ms: float = 5.0
    gate_release_ms: float = 50.0

    # EQ tilt
    highpass_hz: float = 150.0
    shelf_freq_hz: float = 3500.0
    shelf_gain_db: float = 3.0

    # Compressor
    comp_threshold_db: float = -18.0
    comp_ratio: float = 3.0
    comp_attack_ms: float = 5.0
    comp_release_ms: float = 100.0
    comp_makeup_db: float = 3.0

    # Limiter
    limit_ceiling_dbfs: float = -1.0

    # Loudness normalization (file mode only by default)
    lufs_target: float = -14.0
    normalize_streaming: bool = False  # avoid LUFS on every chunk; apply optionally


# ---------------------------- Stateful processing helpers ----------------------------


@dataclass
class StreamState:
    # Smoothing envelopes for gate/compressor
    gate_env: float = 0.0
    comp_env: float = 0.0

    # DeepFilter state is stored in enhancer df_state; we reuse the singleton enhancer
    # No explicit fields needed here for DFNet


def _butter_highpass(cutoff: float, fs: int, order: int = 2):
    nyq = 0.5 * fs
    normal_cutoff = max(min(cutoff / nyq, 0.99), 0.0001)
    if butter is None:
        # Fallback: no-op filter
        return np.array([1.0], dtype=np.float32), np.array([1.0], dtype=np.float32)
    b, a = butter(order, normal_cutoff, btype="highpass")
    return b, a


def _simple_high_shelf(x: np.ndarray, fs: int, freq: float, gain_db: float) -> np.ndarray:
    # Cheap shelf via first-order tilt: y = x + g * HPF(x) around corner freq
    if abs(gain_db) < 0.1:
        return x
    b, a = _butter_highpass(freq, fs, order=1)
    hp = lfilter(b, a, x)
    g = _db_to_lin(gain_db) - 1.0
    return x + g * hp


def _apply_noise_gate(x: np.ndarray, fs: int, thr_db: float, ratio: float, attack_ms: float, release_ms: float, state: StreamState) -> np.ndarray:
    # Level detection using abs with simple envelope follower
    attack = np.exp(-1.0 / (attack_ms * 0.001 * fs))
    release = np.exp(-1.0 / (release_ms * 0.001 * fs))
    env = state.gate_env
    out = np.empty_like(x)
    thr_lin = _db_to_lin(thr_db)
    for i, s in enumerate(np.abs(x)):
        if s > env:
            env = attack * env + (1 - attack) * s
        else:
            env = release * env + (1 - release) * s
        gain = 1.0
        if env < thr_lin:
            gain = ratio
        out[i] = x[i] * gain
    state.gate_env = env
    return out


def _apply_compressor(x: np.ndarray, fs: int, thr_db: float, ratio: float, attack_ms: float, release_ms: float, makeup_db: float, state: StreamState) -> np.ndarray:
    # Simple feed-forward compressor on absolute level with soft knee (very light)
    attack = np.exp(-1.0 / (attack_ms * 0.001 * fs))
    release = np.exp(-1.0 / (release_ms * 0.001 * fs))
    env = state.comp_env
    out = np.empty_like(x)
    thr_lin = _db_to_lin(thr_db)
    makeup = _db_to_lin(makeup_db)
    for i, s in enumerate(np.abs(x)):
        if s > env:
            env = attack * env + (1 - attack) * s
        else:
            env = release * env + (1 - release) * s
        gain = 1.0
        if env > thr_lin:
            # above threshold: reduce by ratio
            over = env / max(thr_lin, 1e-12)
            desired = over ** (1.0 - 1.0 / max(ratio, 1e-6))
            gain = 1.0 / max(desired, 1e-6)
        out[i] = x[i] * gain
    state.comp_env = env
    out *= makeup
    return out


def _apply_limiter(x: np.ndarray, ceiling_dbfs: float) -> np.ndarray:
    ceiling = _db_to_lin(ceiling_dbfs)
    return np.clip(x, -ceiling, ceiling)


def _dereverb_spectral_subtraction(x: np.ndarray, fs: int, strength: float = 0.15, frame: int = 512, hop: int = 256) -> np.ndarray:
    """Light dereverb via per-band noise floor subtraction.

    Not a true WPE; we estimate a spectral floor by the 10th percentile magnitude across time
    and subtract a fraction of it. Keeps CPU cost low.
    """
    if strength <= 0.0:
        return x
    # STFT
    win = np.hanning(frame).astype(np.float32)
    n_frames = 1 + (len(x) - frame) // hop if len(x) >= frame else 1
    # Pad if needed
    pad = (n_frames * hop + frame) - len(x)
    if pad > 0:
        x_p = np.pad(x, (0, pad))
    else:
        x_p = x
    frames = np.lib.stride_tricks.sliding_window_view(x_p, frame)[::hop]
    frames = frames * win[None, :]
    # FFT
    spec = np.fft.rfft(frames, axis=1)
    mag = np.abs(spec)
    phase = np.angle(spec)
    floor = np.percentile(mag, 10, axis=0)
    mag_d = np.maximum(mag - strength * floor[None, :], 0.0)
    spec_d = mag_d * np.exp(1j * phase)
    # iSTFT (overlap-add)
    time_frames = np.fft.irfft(spec_d, axis=1)
    out = np.zeros_like(x_p)
    wsum = np.zeros_like(x_p)
    for i, fr in enumerate(time_frames):
        start = i * hop
        out[start : start + frame] += fr * win
        wsum[start : start + frame] += win ** 2
    wsum = np.where(wsum < 1e-6, 1.0, wsum)
    out = out / wsum
    return out[: len(x)]


def _loudness_normalize_file(y: np.ndarray, fs: int, target_lufs: float, ceiling_dbfs: Optional[float] = None) -> np.ndarray:
    try:
        import pyloudnorm as pyln  # lazy import
        meter = pyln.Meter(fs)  # EBU R128
        loudness = meter.integrated_loudness(y.astype(np.float32))
        loudness = float(loudness)
        gain_db = target_lufs - loudness
        # Respect ceiling if provided by capping gain to avoid predicted peak overs
        if ceiling_dbfs is not None:
            peak = float(np.max(np.abs(y)) + 1e-12)
            peak_dbfs = _lin_to_db(peak)
            predicted_peak_dbfs = peak_dbfs + gain_db
            if predicted_peak_dbfs > ceiling_dbfs:
                gain_db = ceiling_dbfs - peak_dbfs
        return y * _db_to_lin(gain_db)
    except Exception:
        # fallback: simple RMS normalization roughly toward target
        rms = np.sqrt(np.mean(y**2) + 1e-12)
        target_rms = _db_to_lin(target_lufs) * 0.5  # crude mapping
        if rms > 0:
            y = y * (target_rms / rms)
        if ceiling_dbfs is not None:
            # Ensure ceiling by scaling if needed
            peak = float(np.max(np.abs(y)) + 1e-12)
            peak_dbfs = _lin_to_db(peak)
            if peak_dbfs > ceiling_dbfs:
                y = y * _db_to_lin(ceiling_dbfs - peak_dbfs)
        return y


# ---------------------------- Public APIs ----------------------------


def process_file(in_path: str, work_dir: str, params: Optional[Dict[str, Any]] = None) -> Tuple[str, int]:
    """Run clarity pipeline on a media file path and return wav artifact path + sample rate."""
    p = ClarityParams(**(params or {}))
    
    # For file processing, we need the full extractor - lazy import to avoid circular dependency
    from .audio_denoise_dfnet import _get_enhancer_and_extractor
    enhancer, extractor = _get_enhancer_and_extractor()

    # 1) Extract to mono wav at model SR
    extraction = extractor.extract(in_path, work_dir)
    wav_path = extraction.extracted_wav_path
    audio, sr = sf.read(wav_path, dtype="float32")
    x = _ensure_mono_f32(audio)

    # 2) Denoise (DeepFilterNet)
    x = _ensure_mono_f32(x)
    x = np.ascontiguousarray(x)
    x = _ensure_mono_f32(x)
    x = x.astype(np.float32, copy=False)
    x = _ensure_mono_f32(x)
    x = _ensure_mono_f32(x)
    x = _ensure_mono_f32(x)
    x = _ensure_mono_f32(x)
    x = _ensure_mono_f32(x)
    # Actual enhance
    from df.enhance import enhance  # type: ignore
    # DFNet expects [C, N] (2D tensor). Add channel dim, run, then squeeze back.
    if isinstance(x, np.ndarray):
        x_t = torch.from_numpy(x.astype(np.float32, copy=False))
    else:
        x_t = torch.tensor(x, dtype=torch.float32)
    if x_t.ndim == 1:
        x_t = x_t.unsqueeze(0)  # [1, N]
    x_t = x_t.contiguous()
    x_enh = enhance(enhancer.model, enhancer.df_state, x_t, atten_lim_db=enhancer.clamp_atten(p.denoise_atten_db))
    if x_enh.ndim > 1:
        x_enh = x_enh.squeeze(0)
    x = x_enh.detach().cpu().numpy().astype(np.float32, copy=False)

    # 3) Dereverb
    x = _dereverb_spectral_subtraction(x, sr, p.dereverb_strength)

    st = StreamState()
    # 4) Noise Gate
    x = _apply_noise_gate(x, sr, p.gate_threshold_db, p.gate_ratio, p.gate_attack_ms, p.gate_release_ms, st)
    # 5) EQ tilt
    b, a = _butter_highpass(p.highpass_hz, sr, order=2)
    x = lfilter(b, a, x)
    x = _simple_high_shelf(x, sr, p.shelf_freq_hz, p.shelf_gain_db)
    # 6) Compression/Leveler
    x = _apply_compressor(x, sr, p.comp_threshold_db, p.comp_ratio, p.comp_attack_ms, p.comp_release_ms, p.comp_makeup_db, st)
    # 7) Loudness normalization (-14 LUFS by default) with ceiling awareness
    x = _loudness_normalize_file(x, sr, p.lufs_target, p.limit_ceiling_dbfs)
    # 8) Final limiter to enforce ceiling post-normalization
    x = _apply_limiter(x, p.limit_ceiling_dbfs)

    out_wav = os.path.join(work_dir, "clarity_output.wav")
    sf.write(out_wav, x, sr, subtype="PCM_16")
    return out_wav, sr


def process_stream_chunk(chunk_mono_f32: np.ndarray, sr: int, state: Optional[StreamState], params: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, StreamState]:
    """Process a streaming chunk. Returns processed chunk and updated state.

    chunk_mono_f32: mono float32 PCM array
    sr: sample rate
    state: keeps envelopes; if None, initialized.
    params: optional overrides matching ClarityParams
    """
    p = ClarityParams(**(params or {}))
    if state is None:
        state = StreamState()
    # Streaming does NOT require media extraction; get enhancer only to avoid batch dependency
    enhancer = _get_global_enhancer()
    x = _ensure_mono_f32(chunk_mono_f32)
    from df.enhance import enhance  # type: ignore

    # 1) Denoise on chunk (DFNet keeps internal state). Convert to [1, N] torch and back.
    if isinstance(x, np.ndarray):
        x_t = torch.from_numpy(x.astype(np.float32, copy=False))
    else:
        x_t = torch.tensor(x, dtype=torch.float32)
    if x_t.ndim == 1:
        x_t = x_t.unsqueeze(0)
    x_t = x_t.contiguous()
    x_enh = enhance(enhancer.model, enhancer.df_state, x_t, atten_lim_db=enhancer.clamp_atten(p.denoise_atten_db))
    if x_enh.ndim > 1:
        x_enh = x_enh.squeeze(0)
    x = x_enh.detach().cpu().numpy().astype(np.float32, copy=False)
    # 2) Dereverb (lightweight)
    x = _dereverb_spectral_subtraction(x, sr, p.dereverb_strength)
    # 3) Gate
    x = _apply_noise_gate(x, sr, p.gate_threshold_db, p.gate_ratio, p.gate_attack_ms, p.gate_release_ms, state)
    # 4) EQ
    b, a = _butter_highpass(p.highpass_hz, sr, order=2)
    x = lfilter(b, a, x)
    x = _simple_high_shelf(x, sr, p.shelf_freq_hz, p.shelf_gain_db)
    # 5) Compressor
    x = _apply_compressor(x, sr, p.comp_threshold_db, p.comp_ratio, p.comp_attack_ms, p.comp_release_ms, p.comp_makeup_db, state)
    # 6) Loudness normalization (optional in streaming for latency)
    if p.normalize_streaming:
        x = _loudness_normalize_file(x, sr, p.lufs_target, p.limit_ceiling_dbfs)
    # 7) Final limiter to enforce ceiling post-normalization
    x = _apply_limiter(x, p.limit_ceiling_dbfs)
    return x.astype(np.float32, copy=False), state
