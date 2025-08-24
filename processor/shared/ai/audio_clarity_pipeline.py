"""CPU-only speech clarity pipeline for file and streaming use.

Stages (in order):
 1) Denoise (DeepFilterNet3 wrapper)
 2) Noise Gate (level-based attenuation between words)
 3) EQ Tilt (HPF ~150 Hz + optional gentle high-shelf)
 4) Compression/Leveler (fast attack, moderate ratio)
 5) Loudness normalization (-14 LUFS target, file mode by default)
 6) Limiter (ceiling -1 dBFS)

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
    from scipy.ndimage import uniform_filter1d
except Exception:  # pragma: no cover - optional dependency during dev
    butter = None  # type: ignore
    uniform_filter1d = None  # type: ignore
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

    # Dereverb (disabled by default; kept for backward compatibility)
    dereverb_strength: float = 0.0  # 0..1, spectral floor subtraction amount

    # Noise gate
    gate_threshold_db: float = -55.0
    gate_ratio: float = 0.1  # linear attenuation when below threshold
    gate_attack_ms: float = 5.0
    gate_release_ms: float = 50.0

    # EQ tilt
    highpass_hz: float = 150.0
    shelf_freq_hz: float = 3500.0
    shelf_gain_db: float = 0.0

    # De-esser: removed

    # Compressor
    comp_threshold_db: float = -25.0
    comp_ratio: float = 2.0
    comp_attack_ms: float = 2.0
    comp_release_ms: float = 60.0
    comp_makeup_db: float = 0.0

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
    # Current applied gate gain (smoothed) to avoid zipper/popping when toggling
    gate_gain: float = 1.0
    # (de-esser removed)

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


def _apply_noise_gate(x: np.ndarray, fs: int, thr_db: float, ratio: float, attack_ms: float, release_ms: float, state: StreamState) -> np.ndarray:
    """Fast, pop-free gate using moving RMS and smoothed gain (vectorized).

    - Envelope: moving RMS over ~10 ms window (no Python loops)
    - Target gain: 1.0 above threshold, `ratio` below
    - Gain smoothing: one-pole IIR with time constant ~= release_ms
    """
    if ratio >= 0.999:
        return x  # effectively bypass

    # Moving RMS envelope (~10 ms)
    win_len = max(1, int(0.01 * fs))
    if win_len > len(x):
        win_len = max(1, len(x))
    w = np.ones(win_len, dtype=np.float32) / float(win_len)
    # Compute moving average of squared signal, then sqrt to get RMS
    # Pad to maintain 'same' length
    x2 = x.astype(np.float32, copy=False) ** 2
    env = np.convolve(x2, w, mode="same")
    env = np.sqrt(env + 1e-12)

    # Target gain based on threshold
    thr_lin = _db_to_lin(thr_db)
    tgt = np.where(env < thr_lin, float(ratio), 1.0).astype(np.float32, copy=False)

    # Smooth gain with one-pole low-pass (vectorized). Prefer slower opening (release_ms)
    # y[n] = (1-alpha)*x[n] + alpha*y[n-1]
    alpha = float(np.exp(-1.0 / (max(release_ms, 1e-3) * 0.001 * fs)))
    b = np.array([1.0 - alpha], dtype=np.float32)
    a = np.array([1.0, -alpha], dtype=np.float32)

    from scipy.signal import lfilter  # type: ignore
    y = lfilter(b, a, tgt)
    # Blend first sample with previous state to maintain continuity
    y[0] = (1.0 - alpha) * tgt[0] + alpha * float(state.gate_gain)

    out = x * y.astype(np.float32, copy=False)
    # Update state with last values
    state.gate_env = float(env[-1]) if env.size else state.gate_env
    state.gate_gain = float(y[-1]) if y.size else state.gate_gain
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

    st = StreamState()
    # 3) Noise Gate
    x = _apply_noise_gate(x, sr, p.gate_threshold_db, p.gate_ratio, p.gate_attack_ms, p.gate_release_ms, st)
    # 4) EQ tilt
    b, a = _butter_highpass(p.highpass_hz, sr, order=2)
    x = lfilter(b, a, x)
    x = _simple_high_shelf(x, sr, p.shelf_freq_hz, p.shelf_gain_db)
    # 5) Loudness normalization (-14 LUFS by default) with ceiling awareness
    x = _loudness_normalize_file(x, sr, p.lufs_target, p.limit_ceiling_dbfs)
    # 6) Compression/Leveler AFTER normalization (fixed threshold behavior)
    x = _apply_compressor(x, sr, p.comp_threshold_db, p.comp_ratio, p.comp_attack_ms, p.comp_release_ms, p.comp_makeup_db, st)
    # Dynamic presence notch before limiter, driven by ess_ratio
    x, _ = _apply_dynamic_presence_notch(x, sr)
    # 7) Final limiter to enforce ceiling post-normalization
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
    # 2) Gate
    x = _apply_noise_gate(x, sr, p.gate_threshold_db, p.gate_ratio, p.gate_attack_ms, p.gate_release_ms, state)
    # 3) EQ
    b, a = _butter_highpass(p.highpass_hz, sr, order=2)
    x = lfilter(b, a, x)
    x = _simple_high_shelf(x, sr, p.shelf_freq_hz, p.shelf_gain_db)
    # 4) Loudness normalization (optional in streaming for latency)
    if p.normalize_streaming:
        x = _loudness_normalize_file(x, sr, p.lufs_target, p.limit_ceiling_dbfs)
    # 5) Compressor (with gentle HF-tilted sidechain), placed after optional normalization
    x = _apply_compressor(x, sr, p.comp_threshold_db, p.comp_ratio, p.comp_attack_ms, p.comp_release_ms, p.comp_makeup_db, state)
    # Dynamic presence notch
    x, _ = _apply_dynamic_presence_notch(x, sr)
    # 6) Final limiter to enforce ceiling post-normalization
    x = _apply_limiter(x, p.limit_ceiling_dbfs)
    return x.astype(np.float32, copy=False), state
