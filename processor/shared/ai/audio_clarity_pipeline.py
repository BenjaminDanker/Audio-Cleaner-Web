"""CPU-only speech clarity pipeline for file and streaming use.

Stages (in order):
 1) Denoise (DeepFilterNet3 wrapper)
 2) VAD Gate (Silero VAD-based speech detection)
 3) EQ Tilt (HPF ~150 Hz + optional gentle high-shelf)
 4) Compression/Leveler (fast attack, moderate ratio)
 5) Limiter (ceiling -1 dBFS)

Interfaces:
    - process_file(in_path, work_dir, params) -> (processed_wav_path, sample_rate)
    - process_stream_chunk(mono_f32, sr, state, params) -> (processed_chunk, updated_state)

Notes:
    - All DSP is CPU-friendly (NumPy + SciPy). DeepFilterNet runs on CPU via torch cpu wheels.
    - Silero VAD provides intelligent speech/non-speech detection for gating.
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
    from scipy.signal import butter, lfilter, resample
    from scipy.ndimage import uniform_filter1d
except Exception:  # pragma: no cover - optional dependency during dev
    butter = None  # type: ignore
    uniform_filter1d = None  # type: ignore
    resample = None  # type: ignore
    def lfilter(b, a, x):  # type: ignore
        return x

from df.enhance import enhance, init_df, load_audio, save_audio  # type: ignore

# Global enhancer for streaming (no media_extractor dependency)
_GLOBAL_ENHANCER = None

# Global Silero VAD model for streaming
_GLOBAL_VAD_MODEL = None
_GLOBAL_VAD_UTILS = None

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


def _get_global_vad():
    """Get singleton Silero VAD model from local path."""
    global _GLOBAL_VAD_MODEL, _GLOBAL_VAD_UTILS
    if _GLOBAL_VAD_MODEL is None:
        import torch
        from pathlib import Path
        
        # Try to load from local path first
        base_dir = Path(__file__).parent.parent.parent
        root = getattr(__import__('sys'), "_MEIPASS", str(base_dir))
        local_vad_path = os.path.join(root, "models/silero-vad/hub/hub/snakers4_silero-vad_master")
        
        if os.path.exists(local_vad_path):
            # Load from local path
            _GLOBAL_VAD_MODEL, _GLOBAL_VAD_UTILS = torch.hub.load(
                repo_or_dir=local_vad_path,
                model='silero_vad',
                source='local'
            )
        else:
            raise ModuleNotFoundError("silvero folder not found")
    return _GLOBAL_VAD_MODEL, _GLOBAL_VAD_UTILS


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


def _make_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _dump_stage(y: np.ndarray, fs: int, stages_dir: Optional[str], idx: int, name: str, enabled: bool) -> None:
    """Write a stage WAV to stages_dir as NN_name.wav when enabled."""
    if not enabled or not stages_dir:
        return
    _make_dir(stages_dir)
    fname = f"{idx:02d}_{name}.wav"
    path = os.path.join(stages_dir, fname)
    try:
        sf.write(path, y.astype(np.float32, copy=False), fs, subtype="PCM_16")
    except Exception:
        # Best-effort; don't fail pipeline on debug dump errors
        pass


def _dump_stream_stage(y: np.ndarray, fs: int, state: "StreamState", idx: int, name: str, enabled: bool) -> None:
    """Dump per-chunk stage WAVs into debug dir, grouped by ordered stage folder."""
    if not enabled or not state.debug_dir:
        return
    stage_dir = os.path.join(state.debug_dir, f"{idx:02d}_{name}")
    _make_dir(stage_dir)
    fname = f"{state.chunk_index:05d}.wav"
    path = os.path.join(stage_dir, fname)
    try:
        sf.write(path, y.astype(np.float32, copy=False), fs, subtype="PCM_16")
    except Exception:
        pass


def _moving_rms(x: np.ndarray, fs: int, win_ms: float) -> np.ndarray:
    """Fast moving RMS using convolution. Returns same-length envelope."""
    win_len = max(1, int((win_ms * 0.001) * fs))
    if win_len > len(x):
        win_len = max(1, len(x))
    w = np.ones(win_len, dtype=np.float32) / float(win_len)
    env = np.convolve((x.astype(np.float32) ** 2), w, mode="same")
    return np.sqrt(env + 1e-12).astype(np.float32)


def _apply_denoise_gain_comp(orig: np.ndarray, denoised: np.ndarray, fs: int, *, hp_hz: float, thr_db: float, win_ms: float, max_boost_db: float, min_voice_ratio: float = 0.05) -> Tuple[np.ndarray, float]:
    """Estimate a single scalar gain to match speech-level RMS of denoised to original.

    - High-pass to voice band edge to avoid DC/rumble bias
    - Compute moving RMS envelopes
    - Select frames where original env > threshold (speech-active)
    - Use median(orig_env / den_env) as ratio; cap boost to max_boost_db; never attenuate
    - If too few speech frames, return identity
    """
    if len(orig) != len(denoised) or len(orig) == 0:
        return denoised, 0.0
    # HPF both
    b, a = _butter_highpass(hp_hz, fs, order=2)
    o = lfilter(b, a, orig.astype(np.float32, copy=False))
    d = lfilter(b, a, denoised.astype(np.float32, copy=False))
    # Envelopes
    env_o = _moving_rms(o, fs, win_ms)
    env_d = _moving_rms(d, fs, win_ms)
    thr_lin = _db_to_lin(thr_db)
    mask = env_o > thr_lin
    voice_ratio = float(np.mean(mask)) if env_o.size else 0.0
    if voice_ratio < float(min_voice_ratio):
        return denoised, 0.0
    ratios = env_o[mask] / np.maximum(env_d[mask], 1e-6)
    # Robust central tendency
    r = float(np.median(np.clip(ratios, 1e-6, 1e6)))
    gain_db = max(0.0, min(max_boost_db, 20.0 * np.log10(r)))
    g = _db_to_lin(gain_db)
    return (denoised * g).astype(np.float32, copy=False), float(gain_db)


# ---------------------------- Parameters ----------------------------


@dataclass
class ClarityParams:
    # Denoise
    denoise_atten_db: Optional[int] = 50

    # Dereverb (disabled by default; kept for backward compatibility)
    dereverb_strength: float = 0.0  # 0..1, spectral floor subtraction amount

    # VAD gate (using Silero VAD instead of level-based gate)
    vad_enabled: bool = True  # Can disable VAD entirely
    vad_threshold: float = 0.5  # VAD confidence threshold (0-1) - reasonable threshold
    vad_min_speech_duration_ms: int = 100   # Reasonable speech detection time
    vad_min_silence_duration_ms: int = 500  # Reasonable silence time before closing gate
    vad_speech_pad_ms: int = 200  # Reasonable padding around speech
    vad_attenuation_ratio: float = 0.1  # More significant reduction (90% attenuation)

    # EQ tilt
    highpass_hz: float = 150.0
    shelf_freq_hz: float = 3500.0
    shelf_gain_db: float = 0.0

    # Compressor
    comp_threshold_db: float = -25.0
    comp_ratio: float = 2.0
    comp_attack_ms: float = 2.0
    comp_release_ms: float = 60.0
    comp_makeup_db: float = 0.0

    # Limiter
    limit_ceiling_dbfs: float = -1.0

    # Debug: dump intermediate stage WAVs
    debug_save_stages: bool = True
    debug_stages_dir: Optional[str] = None  # defaults: work_dir/stages (file), ./stream_stages (stream)

    # Post-denoise gain compensation (speech-aware, single scalar)
    denoise_gain_max_db: float = 12.0
    denoise_gain_ref_db: float = -45.0
    denoise_gain_win_ms: float = 50.0
    denoise_gain_hp_hz: float = 150.0
    denoise_min_voice_ratio: float = 0.05


# ---------------------------- Stateful processing helpers ----------------------------


@dataclass
class StreamState:
    # Smoothing envelopes for gate/compressor
    gate_env: float = 0.0
    comp_env: float = 0.0
    # Current applied gate gain (smoothed) to avoid zipper/popping when toggling
    gate_gain: float = 1.0
    
    # VAD state
    vad_speech_frames: int = 0  # consecutive frames of detected speech
    vad_silence_frames: int = 0  # consecutive frames of detected silence
    vad_is_speech_active: bool = False  # current gate state
    vad_window_buffer: Optional[np.ndarray] = None  # buffer for VAD window processing

    # DeepFilter state is stored in enhancer df_state; we reuse the singleton enhancer
    # No explicit fields needed here for DFNet

    # Debug streaming stage dumping
    debug_save_stages: bool = False
    debug_dir: Optional[str] = None
    chunk_index: int = 0


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

def _biquad_peaking(fs: int, f0: float, Q: float, gain_db: float):
    """Return biquad peaking (EQ) filter coefficients (b, a)."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * (f0 / float(fs))
    alpha = float(np.sin(w0) / (2.0 * Q))
    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / A
    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float32)
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float32)
    return b, a

def _ess_ratio(x: np.ndarray, fs: int, lo: int = 5500, hi: int = 9000, voice_lo: int = 300, voice_hi: int = 3500) -> Tuple[float, float, float]:
    """Compute simple sibilance-to-voice band energy ratio.
    
    Returns (ratio, ess_band_rms, voice_band_rms) for adaptive thresholding.
    """
    L = len(x)
    if L <= 16:
        return 0.0, 1e-12, 1e-12
    # Next power-of-two for efficient FFT
    n = 1
    while n < L:
        n <<= 1
    win = np.hanning(L).astype(np.float32)
    spec = np.fft.rfft(x[:L] * win, n=n)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    sel_s = (freqs >= lo) & (freqs <= hi)
    sel_v = (freqs >= voice_lo) & (freqs <= voice_hi)
    if not np.any(sel_s) or not np.any(sel_v):
        return 0.0, 1e-12, 1e-12
    s = float(np.sqrt(np.mean((mag[sel_s] ** 2)) + 1e-12))
    v = float(np.sqrt(np.mean((mag[sel_v] ** 2)) + 1e-12))
    return float(s / max(v, 1e-12)), s, v


def _spectral_centroid(x: np.ndarray, fs: int, lo_hz: float = 4000.0, hi_hz: float = 10000.0) -> float:
    """Compute spectral centroid in the given frequency range."""
    L = len(x)
    if L <= 16:
        return (lo_hz + hi_hz) * 0.5
    n = 1
    while n < L:
        n <<= 1
    win = np.hanning(L).astype(np.float32)
    spec = np.fft.rfft(x[:L] * win, n=n)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    
    sel = (freqs >= lo_hz) & (freqs <= min(hi_hz, 0.49 * fs))
    if not np.any(sel):
        return (lo_hz + hi_hz) * 0.5
    
    f_sel = freqs[sel]
    m_sel = mag[sel]
    
    # Weighted centroid
    if np.sum(m_sel) < 1e-12:
        return (lo_hz + hi_hz) * 0.5
    
    centroid = float(np.sum(f_sel * m_sel) / np.sum(m_sel))
    return np.clip(centroid, lo_hz, hi_hz)

def _apply_dynamic_presence_notch(x: np.ndarray, fs: int, *, base_threshold: float = 0.05, scale: float = 40.0, max_depth_db: float = 8.0, fallback_f0_hz: float = 6650.0, Q: float = 9.0) -> Tuple[np.ndarray, float]:
    """Apply an adaptive presence notch that centers on the actual sibilance peak.

    - Adaptive trigger: normalizes threshold to voice band strength
    - Adaptive center: uses spectral centroid of 4-10 kHz when triggered
    - Adaptive depth: scales with (ess_ratio - threshold) * scale
    
    Returns (processed, applied_depth_db). If bypassed, applied_depth_db = 0.
    """
    ratio, ess_rms, voice_rms = _ess_ratio(x, fs)
    
    # Adaptive threshold based on voice band strength
    # If voice is weak (-40 dBFS RMS), be more aggressive
    # If voice is strong (-20 dBFS RMS), require higher ratio
    voice_dbfs = 20.0 * np.log10(max(voice_rms, 1e-12))
    # Normalize threshold: weaker voice → lower threshold needed
    adaptive_threshold = base_threshold * (1.0 + max(0.0, (voice_dbfs + 35.0) / 15.0))
    
    if ratio <= adaptive_threshold:
        return x, 0.0

    # Adaptive depth based on how much we exceed threshold
    excess_ratio = ratio - adaptive_threshold
    depth_db = -float(min(max_depth_db, excess_ratio * scale))
    if depth_db > -0.25:
        return x, 0.0

    # Adaptive center: find centroid in sibilance band
    f_center = _spectral_centroid(x, fs, 4000.0, 10000.0)
    # Fall back to default if centroid seems off
    if f_center < 4000.0 or f_center > 10000.0:
        f_center = fallback_f0_hz

    # Adaptive Q: deeper cuts get slightly wider for smoothness
    depth_factor = abs(depth_db) / max_depth_db
    q = float(np.clip(Q * (1.0 - 0.2 * depth_factor), 6.0, 12.0))

    b, a = _biquad_peaking(fs, f_center, q, depth_db)
    from scipy.signal import lfilter  # type: ignore
    y = lfilter(b, a, x)
    return y.astype(np.float32, copy=False), float(depth_db)


def _butter_lowpass(cutoff: float, fs: int, order: int = 1):
    # Kept for compatibility if needed elsewhere; not used in current pipeline
    nyq = 0.5 * fs
    normal_cutoff = max(min(cutoff / nyq, 0.99), 0.0001)
    if butter is None:
        return np.array([1.0], dtype=np.float32), np.array([1.0], dtype=np.float32)
    b, a = butter(order, normal_cutoff, btype="lowpass")
    return b, a


def _apply_vad_gate(x: np.ndarray, fs: int, threshold: float, min_speech_ms: int, 
                   min_silence_ms: int, pad_ms: int, ratio: float, state: StreamState) -> np.ndarray:
    """VAD-based gate using Silero VAD for speech detection.
    
    - Uses Silero VAD to detect speech vs non-speech
    - Much more conservative approach - biased toward keeping audio
    - Applies hysteresis with minimum durations to avoid rapid switching
    - Smooth gain transitions to avoid clicks/pops
    - Maintains state across chunks for streaming consistency
    """
    if ratio >= 0.999:
        return x  # effectively bypass
    
    vad_model, utils = _get_global_vad()
    
    # Silero VAD expects 16kHz, so we may need to resample for analysis
    vad_sr = 16000
    if fs != vad_sr and resample is not None:
        # Simple linear interpolation resampling for VAD analysis only
        x_vad = resample(x, int(len(x) * vad_sr / fs))
    else:
        x_vad = x.copy()
    
    # Convert to tensor for VAD
    x_tensor = torch.from_numpy(x_vad.astype(np.float32))
    
    # Get VAD probabilities - Silero VAD processes in chunks
    vad_window_size = 512  # samples at 16kHz (~32ms windows)
    speech_probs = []
    
    # Process in overlapping windows for smooth detection
    hop_size = vad_window_size // 4  # More overlap for smoother detection
    for i in range(0, len(x_tensor) - vad_window_size + 1, hop_size):
        window = x_tensor[i:i + vad_window_size]
        if len(window) == vad_window_size:
            prob = vad_model(window, vad_sr).item()
            speech_probs.append(prob)
        else:
            # Handle last partial window
            if len(window) > 0:
                speech_probs.append(speech_probs[-1] if speech_probs else 0.5)  # Default to speech
    
    if not speech_probs:
        speech_probs = [0.5]  # Default to speech if no analysis possible
    
    # Much more conservative decision making
    # Use a lower effective threshold and bias toward speech
    effective_threshold = max(threshold - 0.1, 0.2)  # Lower threshold
    
    # Consider it speech if ANY significant portion shows speech
    speech_ratio = np.mean([p > effective_threshold for p in speech_probs])
    overall_speech_confidence = np.max(speech_probs)  # Peak confidence
    
    # Multiple criteria for speech detection (OR logic - any can trigger speech)
    current_decision = (
        speech_ratio > 0.3 or  # 30% of windows show speech, OR
        overall_speech_confidence > threshold or  # Peak confidence exceeds threshold, OR
        np.mean(speech_probs) > effective_threshold  # Average confidence is decent
    )
    
    # Update state counters with bias toward speech
    if current_decision:  # speech detected
        state.vad_speech_frames += 1
        state.vad_silence_frames = 0
        # Quick to open gate
        if state.vad_speech_frames >= max(1, min_speech_ms // 50):  # Much faster opening
            state.vad_is_speech_active = True
    else:  # silence detected
        state.vad_silence_frames += 1
        state.vad_speech_frames = 0
        # Slow to close gate
        frames_per_ms = vad_sr / 1000.0 / hop_size
        required_silence_frames = max(5, int(min_silence_ms * frames_per_ms))
        if state.vad_silence_frames >= required_silence_frames:
            state.vad_is_speech_active = False
    
    # Create gain envelope - much more conservative
    # Default to speech (gain = 1.0) unless very confident it's silence
    if state.vad_is_speech_active or current_decision:
        # Speech mode - apply minimal attenuation or none
        base_gain = 1.0
    else:
        # Only attenuate during confirmed long silences
        base_gain = ratio
    
    # Create smooth gain envelope
    gain_envelope = np.full(len(x), base_gain, dtype=np.float32)
    
    # Add very generous padding around ANY potential speech
    if pad_ms > 0 and base_gain < 1.0:
        # If we're attenuating, add lots of padding to be safe
        pad_samples = int(pad_ms * 0.001 * fs)
        
        # Look for any frames that might be speech and pad them generously
        frame_size = len(x) // max(1, len(speech_probs))
        for i, prob in enumerate(speech_probs):
            if prob > effective_threshold:  # Any hint of speech
                start_idx = max(0, i * frame_size - pad_samples)
                end_idx = min(len(x), (i + 1) * frame_size + pad_samples)
                gain_envelope[start_idx:end_idx] = 1.0  # Full gain around speech
    
    # Very gentle smoothing to avoid artifacts
    alpha = 0.99  # Very slow changes
    for i in range(1, len(gain_envelope)):
        gain_envelope[i] = alpha * gain_envelope[i-1] + (1 - alpha) * gain_envelope[i]
    
    # Apply gain
    out = x * gain_envelope
    
    # Update state for continuity
    state.gate_gain = float(gain_envelope[-1]) if len(gain_envelope) > 0 else state.gate_gain
    
    return out


def _compressor_gain_trace(
    x: np.ndarray,
    fs: int,
    thr_db: float,
    ratio: float,
    *,
    sidechain_hpf_hz: float | None = 3500.0,
    sidechain_shelf_db: float = 2.0,
    sidechain_mix: float = 0.5,
    win_ms: float = 3.0,
    attack_ms: float = 0.5,
    release_ms: float = 80.0,
):
    """Compute sidechain envelope, smoothed GR in dB, and linear gain for the compressor."""
    sc_base = x.astype(np.float32, copy=False)
    sc_tilt = sc_base
    if sidechain_hpf_hz is not None and sidechain_hpf_hz > 0:
        b_hp, a_hp = _butter_highpass(float(sidechain_hpf_hz), fs, order=2)
        sc_tilt = lfilter(b_hp, a_hp, sc_tilt)
    if abs(sidechain_shelf_db) > 0.1:
        sc_tilt = _simple_high_shelf(sc_tilt, fs, 4000.0, float(sidechain_shelf_db))
    # Blend original with tilted detector so peaks are still seen
    mix = float(np.clip(sidechain_mix, 0.0, 1.0))
    sc = (1.0 - mix) * sc_base + mix * sc_tilt

    # Fast vectorized RMS envelope (much faster than convolution for large windows)
    win_len = max(1, int((win_ms * 0.001) * fs))  # ~3 ms
    if win_len > len(sc):
        win_len = max(1, len(sc))
    
    # Use uniform_filter1d for fast moving average instead of convolution
    if uniform_filter1d is not None:
        env_sq = uniform_filter1d((sc ** 2).astype(np.float64), win_len, mode='constant')
    else:
        # Fallback to convolution if scipy not available
        w = np.ones(win_len, dtype=np.float32) / float(win_len)
        env_sq = np.convolve(sc ** 2, w, mode="same")
    env = np.sqrt(env_sq + 1e-12).astype(np.float32)
    
    # Level in dBFS (vectorized)
    lvl_db = 20.0 * np.log10(np.clip(env, 1e-12, None))
    r = max(float(ratio), 1.0)
    # Target output dB when above threshold (vectorized)
    above = np.maximum(lvl_db - float(thr_db), 0.0)
    out_db = float(thr_db) + above / r
    gr_target_db = out_db - lvl_db  # negative or 0
    
    # Fast vectorized smoothing using scipy's lfilter instead of Python loop
    aA = float(np.exp(-1.0 / (max(attack_ms, 0.05) * 0.001 * fs)))
    aR = float(np.exp(-1.0 / (max(release_ms, 0.1) * 0.001 * fs)))
    
    # Create adaptive alpha array (vectorized comparison)
    gr_diff = np.diff(gr_target_db, prepend=0.0)
    alpha_arr = np.where(gr_diff < 0, aA, aR)  # attack when going more negative
    
    # Use lfilter for much faster smoothing than Python loop
    # Convert to IIR filter form: y[n] = alpha*y[n-1] + (1-alpha)*x[n]
    # For varying alpha, we approximate with average alpha (close enough for audio)
    alpha_avg = float(np.mean(alpha_arr))
    gr_s = lfilter([1.0 - alpha_avg], [1.0, -alpha_avg], gr_target_db).astype(np.float32)
    
    gain = (10.0 ** (gr_s / 20.0)).astype(np.float32, copy=False)
    return env, gr_s, gain


def _apply_compressor(
    x: np.ndarray,
    fs: int,
    thr_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    makeup_db: float,
    state: StreamState,
    *,
    sidechain_hpf_hz: float | None = 3500.0,
    sidechain_shelf_db: float = 2.0,
    sidechain_mix: float = 0.5,
    lookahead_ms: float = 2.5,
) -> np.ndarray:
    """Fast feed-forward compressor using moving RMS envelope (vectorized).

    - Envelope: moving RMS (~10 ms) for smooth level estimation
    - Static curve: above threshold, gain = over^(1/ratio - 1)
    - Makeup applied at the end
    - Updates state.comp_env with last RMS for continuity across chunks
    """
    # Detector and gain
    env, gr_s, gain = _compressor_gain_trace(
        x, fs, thr_db, ratio,
        sidechain_hpf_hz=sidechain_hpf_hz,
        sidechain_shelf_db=sidechain_shelf_db,
        sidechain_mix=sidechain_mix,
        win_ms=3.0,
        attack_ms=attack_ms,
        release_ms=release_ms,
    )
    # Look-ahead: delay program, apply undelayed gain
    la = max(1, int(2.5 * 0.001 * fs))  # Fixed 2.5 ms lookahead for consistency
    la = min(la, max(1, len(x) - 1)) if len(x) > 1 else 1
    x_del = np.concatenate([np.zeros(la, np.float32), x[:-la]]).astype(np.float32, copy=False)
    # Apply gain with optional makeup (in dB) folded in
    # Safety: never amplify from compressor
    gain = np.minimum(gain, 1.0).astype(np.float32, copy=False)
    # y = program delayed, multiplied by safe gain and optional makeup
    y = (x_del * (gain * _db_to_lin(makeup_db))).astype(np.float32, copy=False)
    state.comp_env = float(env[-1]) if env.size else state.comp_env
    return y


def _apply_limiter(x: np.ndarray, ceiling_dbfs: float) -> np.ndarray:
    ceiling = _db_to_lin(ceiling_dbfs)
    return np.clip(x, -ceiling, ceiling)


# De-esser removed


# FFmpeg de-esser removed


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


# ---------------------------- Public APIs ----------------------------


def process_file(in_path: str, work_dir: str, params: Optional[Dict[str, Any]] = None) -> Tuple[str, int]:
    """Run clarity pipeline on a media file path and return wav artifact path + sample rate."""
    p = ClarityParams(**(params or {}))
    
    # For file processing, we need the full extractor - lazy import to avoid circular dependency
    try:
        from .audio_denoise_dfnet import _get_enhancer_and_extractor
    except ImportError:
        # Fallback for direct script execution
        from processor.shared.ai.audio_denoise_dfnet import _get_enhancer_and_extractor
    enhancer, extractor = _get_enhancer_and_extractor()

    # 1) Extract to mono wav at model SR
    extraction = extractor.extract(in_path, work_dir)
    wav_path = extraction.extracted_wav_path
    audio, sr = sf.read(wav_path, dtype="float32")
    x = _ensure_mono_f32(audio)
    orig_mono = x.copy()

    # Prepare debug stages dir
    stages_dir = p.debug_stages_dir or os.path.join(work_dir, "stages")
    _dump_stage(x, sr, stages_dir, 0, "original", p.debug_save_stages)

    # 2) Denoise (DeepFilterNet)
    x = _ensure_mono_f32(x)
    x = np.ascontiguousarray(x)
    # Actual enhance
    from df.enhance import enhance  # type: ignore
    # DFNet expects [C, N] (2D tensor). Add channel dim, run, then squeeze back.
    # Optimize: avoid unnecessary copies and use the most direct path
    if isinstance(x, np.ndarray):
        # Use direct tensor creation without copy when possible
        x_t = torch.from_numpy(x)
    else:
        x_t = torch.tensor(x, dtype=torch.float32, copy=False)
    if x_t.ndim == 1:
        x_t = x_t.unsqueeze(0)  # [1, N]
    # Ensure contiguous for best performance without unnecessary copy
    if not x_t.is_contiguous():
        x_t = x_t.contiguous()
    
    # Set torch to use single thread for this operation to avoid overhead
    with torch.no_grad():  # Disable gradient computation for inference
        x_enh = enhance(enhancer.model, enhancer.df_state, x_t, atten_lim_db=enhancer.clamp_atten(p.denoise_atten_db))
    
    if x_enh.ndim > 1:
        x_enh = x_enh.squeeze(0)
    # Direct numpy conversion without extra copy
    x = x_enh.detach().cpu().numpy()
    _dump_stage(x, sr, stages_dir, 1, "denoised", p.debug_save_stages)

    # Post-denoise gain compensation (speech-aware, single scalar)
    x, boost_db = _apply_denoise_gain_comp(
        orig_mono, x, sr,
        hp_hz=float(p.denoise_gain_hp_hz),
        thr_db=float(p.denoise_gain_ref_db),
        win_ms=float(p.denoise_gain_win_ms),
        max_boost_db=float(p.denoise_gain_max_db),
        min_voice_ratio=float(p.denoise_min_voice_ratio),
    )
    _dump_stage(x, sr, stages_dir, 2, "denoise_boost", p.debug_save_stages)

    # Optional light dereverb
    stage_idx = 3
    if p.dereverb_strength > 0.0:
        x = _dereverb_spectral_subtraction(x, sr, strength=float(p.dereverb_strength))
        _dump_stage(x, sr, stages_dir, stage_idx, "dereverb", p.debug_save_stages)
        stage_idx += 1

    st = StreamState()
    
    # VAD gate - moved to position 2 in optimized order
    if p.vad_enabled:
        x = _apply_vad_gate(x, sr, p.vad_threshold, p.vad_min_speech_duration_ms, 
                           p.vad_min_silence_duration_ms, p.vad_speech_pad_ms, 
                           p.vad_attenuation_ratio, st)
        _dump_stage(x, sr, stages_dir, stage_idx, "vad_gate", p.debug_save_stages)
    else:
        _dump_stage(x, sr, stages_dir, stage_idx, "vad_disabled", p.debug_save_stages)
    stage_idx += 1
    
    # EQ - position 3 in optimized order
    b, a = _butter_highpass(p.highpass_hz, sr, order=2)
    x = lfilter(b, a, x)
    x = _simple_high_shelf(x, sr, p.shelf_freq_hz, p.shelf_gain_db)
    _dump_stage(x, sr, stages_dir, stage_idx, "eq", p.debug_save_stages)
    stage_idx += 1
    
    # Compression/Leveler - position 4 in optimized order
    x = _apply_compressor(x, sr, p.comp_threshold_db, p.comp_ratio, p.comp_attack_ms, p.comp_release_ms, p.comp_makeup_db, st)
    _dump_stage(x, sr, stages_dir, stage_idx, "compressor", p.debug_save_stages)
    stage_idx += 1
    
    # Dynamic presence notch
    x, notch_db = _apply_dynamic_presence_notch(x, sr)
    _dump_stage(x, sr, stages_dir, stage_idx, "presence_notch", p.debug_save_stages)
    stage_idx += 1
    
    # Limiter - position 5 in optimized order (final step)
    x = _apply_limiter(x, p.limit_ceiling_dbfs)
    _dump_stage(x, sr, stages_dir, stage_idx, "limiter", p.debug_save_stages)
    stage_idx += 1

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

    # Setup debug stream dumping
    p_debug = bool(getattr(p, "debug_save_stages", False))
    if p_debug and not state.debug_dir:
        # default to ./stream_stages relative to CWD if not provided
        state.debug_dir = p.debug_stages_dir or os.path.abspath(os.path.join(os.getcwd(), "stream_stages"))
        state.debug_save_stages = True
        _make_dir(state.debug_dir)
    # Original
    _dump_stream_stage(x, sr, state, 0, "original", state.debug_save_stages)

    # 1) Denoise on chunk (DFNet keeps internal state). Convert to [1, N] torch and back.
    # Optimize: avoid unnecessary copies and use the most direct path
    if isinstance(x, np.ndarray):
        x_t = torch.from_numpy(x)
    else:
        x_t = torch.tensor(x, dtype=torch.float32, copy=False)
    if x_t.ndim == 1:
        x_t = x_t.unsqueeze(0)
    if not x_t.is_contiguous():
        x_t = x_t.contiguous()
    
    with torch.no_grad():  # Disable gradient computation for inference
        x_enh = enhance(enhancer.model, enhancer.df_state, x_t, atten_lim_db=enhancer.clamp_atten(p.denoise_atten_db))
    
    if x_enh.ndim > 1:
        x_enh = x_enh.squeeze(0)
    x = x_enh.detach().cpu().numpy()
    _dump_stream_stage(x, sr, state, 1, "denoised", state.debug_save_stages)

    # 2) Post-denoise gain compensation
    x, _ = _apply_denoise_gain_comp(
        _ensure_mono_f32(chunk_mono_f32), x, sr,
        hp_hz=float(p.denoise_gain_hp_hz),
        thr_db=float(p.denoise_gain_ref_db),
        win_ms=float(p.denoise_gain_win_ms),
        max_boost_db=float(p.denoise_gain_max_db),
        min_voice_ratio=float(p.denoise_min_voice_ratio),
    )
    _dump_stream_stage(x, sr, state, 2, "denoise_boost", state.debug_save_stages)
    
    # 3) VAD gate - position 2 in optimized order
    if p.vad_enabled:
        x = _apply_vad_gate(x, sr, p.vad_threshold, p.vad_min_speech_duration_ms, 
                           p.vad_min_silence_duration_ms, p.vad_speech_pad_ms, 
                           p.vad_attenuation_ratio, state)
        _dump_stream_stage(x, sr, state, 3, "vad_gate", state.debug_save_stages)
    else:
        _dump_stream_stage(x, sr, state, 3, "vad_disabled", state.debug_save_stages)
    
    # 4) EQ - position 3 in optimized order
    b, a = _butter_highpass(p.highpass_hz, sr, order=2)
    x = lfilter(b, a, x)
    x = _simple_high_shelf(x, sr, p.shelf_freq_hz, p.shelf_gain_db)
    _dump_stream_stage(x, sr, state, 4, "eq", state.debug_save_stages)
    
    # 5) Compressor - position 4 in optimized order
    x = _apply_compressor(x, sr, p.comp_threshold_db, p.comp_ratio, p.comp_attack_ms, p.comp_release_ms, p.comp_makeup_db, state)
    _dump_stream_stage(x, sr, state, 5, "compressor", state.debug_save_stages)
    
    # 6) Dynamic presence notch
    x, _ = _apply_dynamic_presence_notch(x, sr)
    _dump_stream_stage(x, sr, state, 6, "presence_notch", state.debug_save_stages)
    
    # 7) Limiter - final step
    x = _apply_limiter(x, p.limit_ceiling_dbfs)
    _dump_stream_stage(x, sr, state, 7, "limiter", state.debug_save_stages)
    
    # Advance chunk index once per call
    if state.debug_save_stages:
        state.chunk_index += 1
    return x.astype(np.float32, copy=False), state
