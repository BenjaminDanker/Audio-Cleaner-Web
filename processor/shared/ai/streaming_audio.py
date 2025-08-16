"""Streaming-only audio processing utilities.

This module provides audio enhancement for real-time streaming without dependencies 
on batch processing utilities like media_extractor.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import os
import logging

import numpy as np
try:
    from scipy.signal import butter, lfilter
except Exception:  # pragma: no cover
    butter = None
    def lfilter(b, a, x):  # type: ignore
        return x

from df.enhance import enhance, init_df  # type: ignore

logger = logging.getLogger(__name__)

# Global streaming enhancer instance
_STREAMING_ENHANCER = None

class StreamingEnhancer:
    """DeepFilterNet enhancer optimized for streaming (no file I/O dependencies)."""
    
    def __init__(self, models_root: str):
        self.models_root = models_root
        logger.info("Initializing streaming DeepFilterNet model from %s", models_root)
        self.model, self.df_state, _ = init_df(models_root, post_filter=True)
    
    def clamp_atten(self, atten_db: Optional[int]) -> Optional[int]:
        if atten_db is None:
            return None
        try:
            val = int(atten_db)
        except (ValueError, TypeError):
            val = 30  # default
        return max(-10, min(80, val))  # clamp to valid range
    
    @property
    def sample_rate(self) -> int:
        return self.df_state.sr()

def _get_streaming_enhancer():
    """Get singleton streaming enhancer."""
    global _STREAMING_ENHANCER
    if _STREAMING_ENHANCER is None:
        from pathlib import Path
        base_dir = Path(__file__).parent.parent.parent
        root = getattr(__import__('sys'), "_MEIPASS", str(base_dir))
        models_root = os.path.join(root, "models/DeepFilterNet3")
        _STREAMING_ENHANCER = StreamingEnhancer(models_root)
    return _STREAMING_ENHANCER

@dataclass
class StreamState:
    """State for streaming audio processing."""
    gate_env: float = 0.0
    comp_env: float = 0.0

@dataclass 
class StreamingParams:
    """Parameters for streaming audio processing."""
    denoise_atten_db: Optional[int] = 30
    dereverb_strength: float = 0.15
    gate_threshold_db: float = -48.0
    gate_ratio: float = 0.2
    gate_attack_ms: float = 5.0
    gate_release_ms: float = 50.0
    highpass_hz: float = 150.0
    shelf_freq_hz: float = 3500.0
    shelf_gain_db: float = 3.0
    comp_threshold_db: float = -18.0
    comp_ratio: float = 3.0
    comp_attack_ms: float = 5.0
    comp_release_ms: float = 100.0
    comp_makeup_db: float = 3.0
    limit_ceiling_dbfs: float = -1.0

def _db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)

def _ensure_mono_f32(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float32, copy=False)

def _butter_highpass(cutoff: float, fs: int, order: int = 2):
    nyq = 0.5 * fs
    normal_cutoff = max(min(cutoff / nyq, 0.99), 0.0001)
    if butter is None:
        return np.array([1.0], dtype=np.float32), np.array([1.0], dtype=np.float32)
    b, a = butter(order, normal_cutoff, btype="highpass")
    return b, a

def _simple_high_shelf(x: np.ndarray, fs: int, freq: float, gain_db: float) -> np.ndarray:
    if abs(gain_db) < 0.1:
        return x
    b, a = _butter_highpass(freq, fs, order=1)
    hp = lfilter(b, a, x)
    g = _db_to_lin(gain_db) - 1.0
    return x + g * hp

def _apply_noise_gate(x: np.ndarray, fs: int, thr_db: float, ratio: float, 
                     attack_ms: float, release_ms: float, state: StreamState) -> np.ndarray:
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
        gain = 1.0 if env >= thr_lin else ratio
        out[i] = x[i] * gain
    
    state.gate_env = env
    return out

def _apply_compressor(x: np.ndarray, fs: int, thr_db: float, ratio: float,
                     attack_ms: float, release_ms: float, makeup_db: float, 
                     state: StreamState) -> np.ndarray:
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

def _dereverb_spectral_subtraction(x: np.ndarray, fs: int, strength: float = 0.15, 
                                  frame: int = 512, hop: int = 256) -> np.ndarray:
    """Light dereverb via spectral subtraction."""
    if strength <= 0.0:
        return x
    
    win = np.hanning(frame).astype(np.float32)
    n_frames = 1 + (len(x) - frame) // hop if len(x) >= frame else 1
    pad = (n_frames * hop + frame) - len(x)
    if pad > 0:
        x_p = np.pad(x, (0, pad))
    else:
        x_p = x
    
    frames = np.lib.stride_tricks.sliding_window_view(x_p, frame)[::hop]
    frames = frames * win[None, :]
    spec = np.fft.rfft(frames, axis=1)
    mag = np.abs(spec)
    phase = np.angle(spec)
    floor = np.percentile(mag, 10, axis=0)
    mag_d = np.maximum(mag - strength * floor[None, :], 0.0)
    spec_d = mag_d * np.exp(1j * phase)
    
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

def process_stream_chunk(chunk_mono_f32: np.ndarray, sr: int, state: Optional[StreamState], 
                        params: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, StreamState]:
    """Process a streaming audio chunk with full clarity pipeline.
    
    Args:
        chunk_mono_f32: mono float32 PCM array
        sr: sample rate
        state: processing state (created if None)
        params: optional parameter overrides
    
    Returns:
        (processed_chunk, updated_state)
    """
    p = StreamingParams(**(params or {}))
    if state is None:
        state = StreamState()
    
    enhancer = _get_streaming_enhancer()
    x = _ensure_mono_f32(chunk_mono_f32)
    
    # 1) Denoise with DeepFilterNet
    x = enhance(enhancer.model, enhancer.df_state, x, 
                atten_lim_db=enhancer.clamp_atten(p.denoise_atten_db))
    
    # 2) Dereverb
    x = _dereverb_spectral_subtraction(x, sr, p.dereverb_strength)
    
    # 3) Noise gate
    x = _apply_noise_gate(x, sr, p.gate_threshold_db, p.gate_ratio, 
                         p.gate_attack_ms, p.gate_release_ms, state)
    
    # 4) EQ tilt
    b, a = _butter_highpass(p.highpass_hz, sr, order=2)
    x = lfilter(b, a, x)
    x = _simple_high_shelf(x, sr, p.shelf_freq_hz, p.shelf_gain_db)
    
    # 5) Compressor
    x = _apply_compressor(x, sr, p.comp_threshold_db, p.comp_ratio,
                         p.comp_attack_ms, p.comp_release_ms, p.comp_makeup_db, state)
    
    # 6) Limiter
    x = _apply_limiter(x, p.limit_ceiling_dbfs)
    
    return x.astype(np.float32, copy=False), state
