"""Bandwidth extension for band-limited recordings (phone calls, VoIP, old tapes) with AP-BWE.

AP-BWE (Lu et al., IEEE/ACM TASLP 2024; github.com/yxlu-0102/AP-BWE; code and weights MIT) is a small
all-convolutional GAN generator that works on 48 kHz STFT frames (n_fft 1024, win 320, hop 80). Its input is
the narrow-band signal *resampled back to 48 kHz*; it predicts the wide-band log-magnitude (as a residual on
the input) and the wide-band phase. There are four official 48 kHz generators, named after the input
sampling rate they were trained for: 8k (content up to 4 kHz), 12k (6 kHz), 16k (8 kHz), 24k (12 kHz).

The product path needs only onnxruntime + numpy + scipy; the networks were converted once with
tools/export_apbwe_onnx.py (STFT/iSTFT are done here in numpy and checked against torch).

    bw = effective_bandwidth(x)        # Hz where the speech spectrum really ends (24000 for full band)
    name = pick_model(bw)              # "8k" | "12k" | "16k" | "24k" | None (None: leave the file alone)
    y = extend(x, model_dir, threads=4)   # 48 kHz float32, same length/shape as x

What is kept: the network regenerates the complete band, including the low band. `extend` therefore blends
the *original* spectrum back in below a crossover (just under the real cut-off of the recording, linear ramp
375 Hz wide) so everything the recording actually contains is kept (changes are STFT round-off, about -60 dB)
and only the missing highs are synthesised.

Long input is processed in blocks (default 6 s) with `CONTEXT` samples of overlap on each side; the model's
receptive field is only +-27 frames (2160 samples), so overlap-save is exact up to float round-off and no
crossfade between blocks is needed; peak memory does not depend on the file length.
"""
from __future__ import annotations

import functools
import math
import threading
from pathlib import Path

import numpy as np
import scipy.fft

from .dfn import Cancelled

SAMPLE_RATE = 48_000
N_FFT, HOP, WIN = 1024, 80, 320
N_BINS = N_FFT // 2 + 1
_OFF = (N_FFT - WIN) // 2  # position of the Hann window inside the (zero padded) FFT frame
CONTEXT = 2_560  # samples of overlap per block side (a multiple of HOP; receptive field + STFT is ~2.4k: output identical to 4000)
BLOCK_SECONDS = 6.0  # longer blocks re-run less context (CONTEXT 2560 + 6 s: 7 % less CPU than 4000 + 3 s, same samples)
CROSS_RAMP_BINS = 8  # width of the crossover between original and generated spectrum
EDGE_DB = 20.0  # the edge is where the speech spectrum is this far below its passband level
CLIFF_DB = 18.0  # ... and the level falls by at least this much from just below to just above the edge
GATE_LO, GATE_HI = 5e-4, 3e-3  # input band rms (STFT magnitude units, ~ -75 dBFS) between 'silent' and 'signal'
FILL_GAIN = 0.6  # level of the mirrored spectrum used to fill the gap between the real cut-off and the model's band

#: name -> (input sampling rate the generator was trained for, ONNX file)
MODELS = {
    "8k": (8_000, "8kto48k.onnx"),
    "12k": (12_000, "12kto48k.onnx"),
    "16k": (16_000, "16kto48k.onnx"),
    "24k": (24_000, "24kto48k.onnx"),
}
#: effective-bandwidth ceilings (Hz) up to which each generator is used: its nominal cut-off plus ~10 %.
#: Anything wider than the last one is treated as full band.
_LIMITS = (("8k", 4_400.0), ("12k", 6_600.0), ("16k", 8_800.0), ("24k", 12_600.0))

# periodic Hann window, == torch.hann_window (and scipy.signal.get_window("hann", WIN), bit for bit; scipy.signal itself is
# imported only where it is used: it takes most of a second to load)
_HANN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)).astype(np.float32)


# --------------------------------------------------------------------------------------------------------
# Bandwidth detection
# --------------------------------------------------------------------------------------------------------
_LTAS_CHUNK = 1024  # frames per FFT batch while measuring (about 8 MB of audio at a time)


def _read_frames(x: np.ndarray, first: np.ndarray, n_fft: int) -> np.ndarray:
    """The frames of `x` starting at the sample positions `first`, as float32 (k, n_fft). `x` is (n,) or
    (n, ch) (channels are averaged); only these samples are read, so `x` may be a memory map."""
    frames = np.asarray(x[first[:, None] + np.arange(n_fft)], dtype=np.float32)
    if frames.ndim == 3:
        frames = frames[:, :, 0] if frames.shape[2] == 1 else frames.mean(axis=2)
    return frames


def _ltas_db(x: np.ndarray, sr: int, max_frames: int = 6000):
    """Long-term average power spectrum (dB) over the active frames of `x` (mono, or (n, ch) with the channels
    averaged), bin spacing sr/2048 Hz. Works block by block on the frames it samples, so the whole signal is
    never copied or averaged in memory."""
    n_fft, hop = 2048, 1024
    if len(x) < n_fft * 4:
        return None
    n = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n) if n <= max_frames * 4 else np.linspace(0, n - 1, max_frames * 4).astype(int)
    win = np.hanning(n_fft).astype(np.float32)
    f = np.arange(n_fft // 2 + 1) * sr / n_fft
    speech = (f >= 150) & (f <= 3500)

    def power(frame_ids: np.ndarray) -> np.ndarray:
        return np.abs(scipy.fft.rfft(_read_frames(x, frame_ids * hop, n_fft) * win, axis=1)) ** 2

    e = np.concatenate([power(idx[i:i + _LTAS_CHUNK])[:, speech].sum(axis=1) for i in range(0, len(idx), _LTAS_CHUNK)])
    if e.max() <= 0:
        return None
    # active = clearly above the quiet frames and not far below the loud ones (speech, not the gaps)
    thr = max(np.percentile(e, 60), e.max() * 1e-4)
    act = np.flatnonzero(e >= thr)
    if len(act) < 8:
        return None
    if len(act) > max_frames:
        act = act[np.linspace(0, len(act) - 1, max_frames).astype(int)]
    total = np.zeros(len(f), dtype=np.float64)
    for i in range(0, len(act), _LTAS_CHUNK):
        total += power(idx[act[i:i + _LTAS_CHUNK]]).sum(axis=0, dtype=np.float64)
    ltas = total / len(act)
    return f, 10 * np.log10(ltas + 1e-20)


def effective_bandwidth(x: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Frequency (Hz) at which the spectrum of the speech in `x` ends; `sr/2` when there is no sharp cut-off.

    Works on the long-term average spectrum of the active (speech) frames. The edge is the first frequency
    above 2.4 kHz where that spectrum has fallen 20 dB below the speech passband (0.6-2.4 kHz) and stays
    there, accepted only if it really is a cliff: the level just below the edge is >= CLIFF_DB above the
    level one octave-third higher (natural speech roll-off, noisy or dark recordings fall by < 12 dB there).
    Anything else (no edge, shallow slope, silence, too little speech) gives sr/2, which keeps normal
    recordings out of the extension. `x` is (n,) or (n, ch), also a memory map of any length: it is read
    block by block, never copied. Channels are averaged, so call it per channel for tracks with different
    content.
    """
    nyq = sr / 2
    r = _smoothed_ltas(x, sr)
    if r is None:
        return nyq
    f, db = r

    def mean(lo, hi):
        return _band_mean(f, db, lo, hi)

    ref = mean(600, 2400)
    top = min(nyq * 0.95, 20_000.0)
    for i in np.where((f >= 2400) & (f < top) & (db < ref - EDGE_DB))[0]:
        fe = f[i]
        if mean(fe * 1.15, nyq) >= ref - EDGE_DB:  # does not stay down: a dip, not an edge
            continue
        if mean(fe * 0.8, fe * 0.9) - mean(fe * 1.15, fe * 1.4) < CLIFF_DB:
            continue  # shallow slope: natural roll-off; keep looking further up
        # a real cliff: its position is where the level has fallen 12 dB below the level just before it
        local = mean(fe * 0.8, fe * 0.9)
        j = np.where((f >= fe * 0.9) & (db < local - 12.0))[0]
        return float(f[j[0]])
    return nyq


def _smoothed_ltas(x: np.ndarray, sr: int):
    """(frequencies, dB) of the long-term spectrum of the speech in `x`, smoothed; None when there is too little."""
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    r = _ltas_db(x, sr)
    if r is None:
        return None
    f, db = r
    # smooth in the dB domain with a ~200 Hz running median (kills single-bin tones and gaps)
    import scipy.signal

    return f, scipy.signal.medfilt(db, max(3, int(round(200 / f[1])) | 1))


def _band_mean(f: np.ndarray, db: np.ndarray, lo: float, hi: float) -> float:
    sel = (f >= lo) & (f < hi)
    return db[sel].mean() if sel.any() else np.nan


def tilt_db(x: np.ndarray, sr: int = SAMPLE_RATE) -> float | None:
    """How far the top of the speech band (4-8 kHz) sits below its middle (0.6-2.4 kHz), dB, on the same smoothed
    spectrum as effective_bandwidth: very negative for a muffled voice. None when there is too little speech. The
    website measures the same (web/src/media/analysis.js bandwidthOf, tiltDb)."""
    r = _smoothed_ltas(x, sr)
    if r is None:
        return None
    f, db = r
    return float(_band_mean(f, db, 4000, 8000) - _band_mean(f, db, 600, 2400))


def pick_model(bandwidth_hz: float) -> str | None:
    """Generator for a recording whose speech ends at `bandwidth_hz`; None when it is (nearly) full band."""
    for name, limit in _LIMITS:
        if bandwidth_hz <= limit:
            return name
    return None


# --------------------------------------------------------------------------------------------------------
# Model runtime
# --------------------------------------------------------------------------------------------------------
_session_lock = threading.Lock()
_sessions: dict[tuple[str, int], object] = {}


def unload() -> None:
    """Drop the cached model session (about 120 MB of RAM)."""
    with _session_lock:
        _sessions.clear()


# CUDA provider options: the default exhaustive cuDNN search re-runs for every new input length, and the default
# arena rounds every growth up to a power of two.
_CUDA_OPTIONS = {"cudnn_conv_algo_search": "HEURISTIC", "arena_extend_strategy": "kSameAsRequested"}


def _session(path: str, threads: int, device: str = "cpu"):
    """One cached onnxruntime session (the 119 MB graphs are not kept around: at most one at a time)."""
    import onnxruntime as ort
    if device not in ("cpu", "cuda"):
        raise ValueError(f"unknown device {device!r}")
    if device == "cuda":
        from .gpu import prepare_cuda

        prepare_cuda()
    with _session_lock:
        s = _sessions.get((path, threads, device))
        if s is None:
            _sessions.clear()
            so = ort.SessionOptions()
            so.intra_op_num_threads = max(1, int(threads))
            so.inter_op_num_threads = 1
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.log_severity_level = 3
            # idle worker threads sleep instead of spinning while the numpy STFT part runs (and next to other services)
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
            providers = [("CUDAExecutionProvider", dict(_CUDA_OPTIONS))] if device == "cuda" else ["CPUExecutionProvider"]
            s = ort.InferenceSession(path, so, providers=providers)
            if device == "cuda" and s.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError(f"CUDA is not usable for {Path(path).name} (providers: {s.get_providers()})")
            _sessions[(path, threads, device)] = s
        return s


def model_path(model_dir: str | Path, name: str) -> Path:
    return Path(model_dir) / MODELS[name][1]


def available(model_dir: str | Path) -> bool:
    return all(model_path(model_dir, n).is_file() for n in MODELS)


def _stft(x: np.ndarray):
    """torch.stft(x, 1024, hop=80, win_length=320, hann, center=True, reflect) for len(x) % 80 == 0."""
    n = len(x)
    xp = np.pad(x, (WIN // 2, WIN // 2), mode="reflect")  # only the 320 window samples ever see the padding
    frames = np.lib.stride_tricks.sliding_window_view(xp, WIN)[::HOP]  # (n/80 + 1, 320)
    buf = np.zeros((len(frames), N_FFT), np.float32)
    buf[:, _OFF:_OFF + WIN] = frames * _HANN
    return scipy.fft.rfft(buf, axis=1).T  # (513, T) complex64


def _istft(spec: np.ndarray, n: int) -> np.ndarray:
    """Inverse of _stft: windowed overlap-add normalised by the window envelope (torch.istft), n samples."""
    t = spec.shape[1]
    seg = scipy.fft.irfft(spec.T, n=N_FFT, axis=1)[:, _OFF:_OFF + WIN].astype(np.float32) * _HANN  # (T, 320)
    acc = np.zeros((t + 3, HOP), np.float32)
    for j in range(WIN // HOP):
        acc[j:j + t] += seg[:, j * HOP:(j + 1) * HOP]
    # the envelope sum of hann^2 at 4x overlap is exactly 1.5 away from the two ends (which are context)
    y = acc[2:2 + n // HOP].reshape(-1) / 1.5
    return y


@functools.lru_cache(maxsize=8)
def _resample_filter(ratio: int, rolloff: float = 0.99, width: int = 6):
    """torchaudio's default sinc_interp_hann kernel (what AP-BWE was trained with) at the 48 kHz rate."""
    half = int(math.ceil(width * ratio / rolloff))
    n = np.arange(-half, half + 1, dtype=np.float64)
    t = np.clip(n * rolloff / ratio, -width, width)
    h = np.cos(t * np.pi / width / 2) ** 2 * np.sinc(t) * (rolloff / ratio)
    return h


def _band_limit(x: np.ndarray, lr_rate: int) -> np.ndarray:
    """48 kHz -> lr_rate -> 48 kHz like the training data (everything above lr_rate/2 removed)."""
    ratio = SAMPLE_RATE // lr_rate
    h = _resample_filter(ratio)
    import scipy.signal

    lo = scipy.signal.resample_poly(x, 1, ratio, window=h.copy())
    y = scipy.signal.resample_poly(lo, ratio, 1, window=h.copy())
    y = y[:len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y.astype(np.float32)


def _ramp(lr_rate: int, crossover_hz: float) -> np.ndarray:
    """Weight (513, 1) of the original spectrum: 1 below the crossover, 0 above, linear ramp in between."""
    k0 = crossover_hz / (SAMPLE_RATE / N_FFT)
    k = np.arange(N_BINS, dtype=np.float32)
    w = np.clip((k0 + CROSS_RAMP_BINS / 2 - k) / CROSS_RAMP_BINS, 0, 1)
    return w[:, None].astype(np.float32)


def _run_block(sess, blk: np.ndarray, lr_rate: int, w_orig: np.ndarray, fill_hz: float | None) -> np.ndarray:
    """blk: float32, len % 80 == 0 (context included). Returns the extended block, same length."""
    nb = _band_limit(blk, lr_rate)
    s = _stft(nb)
    if fill_hz is not None:
        # The generator was trained on inputs that are full up to lr_rate/2. If the recording ends lower
        # (3.4 kHz phone speech for the 8k model) it sees a hole and hardly generates anything, so the
        # hole is filled with the mirrored spectrum just below the real edge first (spectral folding).
        kc = int(round(fill_hz / (SAMPLE_RATE / N_FFT)))
        kt = min(N_BINS - 1, int(lr_rate / 2 / (SAMPLE_RATE / N_FFT)) + 2)
        if 8 < kc < kt:
            k = np.arange(kc, kt)
            s[k] = FILL_GAIN * s[np.maximum(2 * kc - k, 1)]
    mag = np.log(np.abs(s) + 1e-4)[None].astype(np.float32)
    pha = np.angle(s)[None].astype(np.float32)
    mag_wb, pr, pi_ = sess.run(None, {"mag": mag, "pha": pha})
    amp = np.exp(mag_wb[0])
    # Two guards against the network inventing sound: never louder than twice the loudest input bin of the
    # frame (out-of-distribution input such as noise or clicks), and faded out where the input itself is
    # (nearly) digital silence, where it would otherwise add a hiss of about -75 dBFS.
    amp = np.minimum(amp, 2.0 * np.abs(s).max(axis=0))
    inband = np.sqrt(np.mean(np.square(np.abs(s[:max(8, int(lr_rate / 2 / (SAMPLE_RATE / N_FFT)))])), axis=0))
    amp = amp * np.clip((inband - GATE_LO) / (GATE_HI - GATE_LO), 0.0, 1.0)
    norm = np.hypot(pr[0], pi_[0])
    norm = np.where(norm > 0, norm, 1.0)
    gen = (amp / norm) * (pr[0] + 1j * pi_[0])  # amp * exp(j atan2(im, re))
    spec = (w_orig * _stft(blk) + (1.0 - w_orig) * gen).astype(np.complex64)
    return _istft(spec, len(blk))


def extend(x: np.ndarray, model_dir: str | Path, threads: int = 4, progress=None, cancel=None, out=None, *,
           model: str | None = None, crossover_hz: float | None = None,
           block_seconds: float = BLOCK_SECONDS, device: str = "cpu") -> np.ndarray:
    """Restore the highs of band-limited speech. `x` is float32 (n,) or (n, ch) at 48 kHz, ~[-1, 1].

    Returns an array of the same shape (written into `out` if given; `out` may be `x` itself). With
    model=None each channel gets the generator `pick_model(effective_bandwidth(channel))` chooses and a
    full-band channel is passed through unchanged; pass model="8k".."24k" to force one. `crossover_hz`
    (default: 0.9 x the channel's measured cut-off, at most 0.95 x the generator's input Nyquist) is where
    the original spectrum hands over to the generated one. `progress(fraction)` is called after each block;
    `cancel` is a threading.Event (or anything with is_set()) -> dfn.Cancelled. `device` "cuda" runs the generator
    on the GPU (onnxruntime-gpu; raises when CUDA cannot be used); the STFT and resampling around it stay on the CPU.
    """
    x = np.asarray(x)
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    if out is None:
        out = x.copy()
    elif out is not x:
        out[...] = x
    mono = x.ndim == 1
    xs = x[:, None] if mono else x
    os_ = out[:, None] if mono else out
    n, ch = xs.shape
    total = n * ch
    done = 0
    block = max(HOP * 100, int(block_seconds * SAMPLE_RATE) // HOP * HOP)
    shared = out is x or np.shares_memory(out, x)
    for c in range(ch):
        xc = xs[:, c]
        name = model
        bw = effective_bandwidth(xc)
        if name is None:
            name = pick_model(bw)
        if name is None or n == 0:
            done += n
            continue
        lr_rate = MODELS[name][0]
        cross = crossover_hz
        if cross is None:
            cross = min(0.9 * bw, 0.95 * lr_rate / 2) if bw < 0.95 * lr_rate / 2 else 0.85 * lr_rate / 2
        fill_hz = bw if bw < 0.92 * lr_rate / 2 else None
        w_orig = _ramp(lr_rate, cross)
        sess = _session(str(model_path(model_dir, name)), threads, device)
        tail = None  # when writing into x itself: its original samples just before the block (the left context)
        for s in range(0, n, block):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            e = min(s + block, n)
            if tail is not None:
                blk = np.concatenate([tail, _slice(xc, s, e + CONTEXT)])
            else:
                blk = _slice(xc, s - CONTEXT, e + CONTEXT)
            if shared and e < n:
                tail = np.array(xc[e - CONTEXT:e], dtype=np.float32)
            y = _run_block(sess, blk, lr_rate, w_orig, fill_hz)
            os_[s:e, c] = y[CONTEXT:CONTEXT + (e - s)]
            done += e - s
            if progress is not None:
                progress(done / total)
    return out


def _slice(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """x[lo:hi] with reflection beyond the ends (like the STFT's centre padding), padded to a multiple of HOP."""
    n = len(x)
    seg = np.array(x[max(lo, 0):min(hi, n)], dtype=np.float32)
    left, right = max(0, -lo), max(0, hi - n)
    right += (-(hi - lo)) % HOP
    if (left or right) and n > 1:
        seg = np.pad(seg, (left, right), mode="reflect")
    elif left or right:
        seg = np.pad(seg, (left, right), mode="edge")
    return np.ascontiguousarray(seg)
