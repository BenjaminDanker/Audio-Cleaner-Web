"""Optional, slow "restore" step: Sidon speech restoration (sarulab-speech/sidon-v0.1, MIT).

Sidon re-synthesises the voice from w2v-BERT 2.0 features, so room echo, reverb and noise
are removed and the band is rebuilt up to 24 kHz. Input and output are 48 kHz float32. It is
a generative model: use it only on recorded speech files (never live), expect a cleaner,
slightly different-sounding voice, and note that on badly reverberant speech it can change
words. About 0.8x real time on 4 threads (see docs/research/restore-sidon.md).

Pipeline (the official demo recipe, made chunk-safe):
  48 kHz -> 50 Hz high-pass -> 16 kHz -> Kaldi fbank (80 mel, povey window, 2-frame stack)
  -> feature extractor (8 w2v-BERT layers + LoRA, 50 features/s)
  -> DAC decoder (960 samples per feature, i.e. 48 kHz).

The two networks run on onnxruntime (already a dependency of the app) from an ONNX conversion
of the official TorchScript files, or, as a fallback, on torch from the TorchScript files.
The feature extractor is windowed (`window_seconds`, linear cross-fade over the overlap) and
the convolutional decoder runs on short blocks with context frames that are cropped away, so
memory stays bounded for any file length. Per-window mean/variance normalisation of the fbank
follows the official demo, which normalises per chunk.

onnxruntime / torch are imported lazily, only when a model is loaded.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.util
import threading
from pathlib import Path

import numpy as np

from .dfn import Cancelled

SAMPLE_RATE = 48_000
MODEL_RATE = 16_000
FEATURE_FILE = "feature_extractor_cpu.pt"
DECODER_FILE = "decoder_cpu.pt"
# ONNX conversion of the same weights (see docs/research/restore-sidon.md); the int8 file is optional.
ONNX_FE = "sidon_fe.onnx"
ONNX_FE_INT8 = "sidon_fe_q8.onnx"
ONNX_DEC = "sidon_dec.onnx"
# Pinned upstream revision of sarulab-speech/sidon-v0.1 and the sha256 of the CPU TorchScript files.
HF_REVISION = "b3b02d8bbd55fdbc410e6e46e76ef95ace4fbf52"
SHA256 = {
    FEATURE_FILE: "fd9abc906a9048b3c047bd3d246a25f6485d09c2d6bb85e098f527f844efc019",
    DECODER_FILE: "34dcc80fab75bd1336369ba5b2063f6da475887aa824db94b495c4059c5fdea4",
}

FEATURE_SAMPLES = 320  # 16 kHz samples per feature frame (20 ms)
DECODER_UPSAMPLE = 960  # 48 kHz samples per feature frame
FEATURE_RATE = MODEL_RATE / FEATURE_SAMPLES  # 50 per second
_FBANK_PAD = 160  # the official demo pads each chunk by 160 samples per side so frames are centred
_TAIL_FEATURES = 32  # zero tail so the last real frame is not at the very edge of the decoder input
# Dither floor (about -100 dBFS): keeps digital silence from hitting the log-mel floor, which would
# skew the per-window statistics. Deterministic so repeated runs give the same result.
_DITHER = 1e-5


# ---------------------------------------------------------------------------------------------
# Front end: numpy re-implementation of transformers' SeamlessM4TFeatureExtractor (no transformers).
# ---------------------------------------------------------------------------------------------

def _kaldi_mel(f):
    return 1127.0 * np.log(1.0 + f / 700.0)


@functools.lru_cache(maxsize=1)
def _mel_filters() -> np.ndarray:
    """(257, 80) Kaldi mel filter bank, triangular in mel space, 20 Hz .. 8 kHz."""
    n_bins, n_mel = 257, 80
    mel_pts = np.linspace(_kaldi_mel(20.0), _kaldi_mel(MODEL_RATE / 2), n_mel + 2)
    bin_mel = _kaldi_mel(MODEL_RATE / ((n_bins - 1) * 2) * np.arange(n_bins))
    diff = np.diff(mel_pts)
    slopes = mel_pts[None, :] - bin_mel[:, None]
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    return np.maximum(0.0, np.minimum(down, up))


@functools.lru_cache(maxsize=1)
def _povey_window() -> np.ndarray:
    return np.power(np.hanning(400), 0.85)


@functools.lru_cache(maxsize=1)
def _highpass_sos() -> np.ndarray:
    """50 Hz RBJ biquad high-pass at 48 kHz, Q = 0.707 (what torchaudio.highpass_biquad does)."""
    w0 = 2 * np.pi * 50.0 / SAMPLE_RATE
    alpha = np.sin(w0) / (2 * 0.707)
    c = np.cos(w0)
    b = np.array([(1 + c) / 2, -(1 + c), (1 + c) / 2])
    a = np.array([1 + alpha, -2 * c, 1 - alpha])
    return np.concatenate([b / a[0], a / a[0]])[None, :]


def _to_16k(x48: np.ndarray) -> np.ndarray:
    """High-pass and resample a (short, e.g. one window with context) 48 kHz block to 16 kHz."""
    from scipy.signal import resample_poly, sosfilt

    y = sosfilt(_highpass_sos(), x48.astype(np.float64))
    return resample_poly(y, 1, 3).astype(np.float32)


def _fbank(wave: np.ndarray) -> np.ndarray:
    """Normalised stacked fbank features for a 16 kHz block whose length is `k * 320 + 320`
    (the extra 320 being the 160-sample pad per side): returns (k, 160) float32."""
    w = wave.astype(np.float64) * 32768.0
    n_frames = 1 + (len(w) - 400) // 160
    frames = np.lib.stride_tricks.sliding_window_view(w, 400)[::160][:n_frames].copy()
    frames -= frames.mean(axis=1, keepdims=True)
    pre = np.empty_like(frames)
    pre[:, 1:] = frames[:, 1:] - 0.97 * frames[:, :-1]
    pre[:, 0] = frames[:, 0] * (1 - 0.97)
    pre *= _povey_window()
    spec = np.abs(np.fft.rfft(pre, n=512, axis=1)) ** 2
    mel = np.log(np.maximum(1.192092955078125e-07, spec @ _mel_filters()))
    mel = (mel - mel.mean(axis=0)) / np.sqrt(mel.var(axis=0, ddof=1) + 1e-7)
    n = (len(mel) // 2) * 2
    return mel[:n].reshape(n // 2, 160).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Models: onnxruntime (preferred, already shipped with the app) or the official TorchScript files
# ---------------------------------------------------------------------------------------------

_lock = threading.Lock()
_backends: dict[tuple, "_Backend"] = {}


class _Backend:
    """features(feats (T, 160)) -> (T, 1024); decode(feats (T, 1024)) -> (T * 960,) float32."""

    name = ""

    def features(self, feats: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def decode(self, feats: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class _OnnxBackend(_Backend):
    name = "onnx"

    def __init__(self, model_dir: Path, threads: int, int8: bool):
        import onnxruntime as ort

        def session(path: Path):
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
            so.log_severity_level = 3
            so.enable_mem_pattern = False  # shapes vary with the file
            # idle worker threads sleep instead of spinning: this runs next to other services
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
            return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

        fe_file = ONNX_FE_INT8 if int8 and (model_dir / ONNX_FE_INT8).is_file() else ONNX_FE
        self.fe = session(model_dir / fe_file)
        self.dec = session(model_dir / ONNX_DEC)

    def features(self, feats):
        return self.fe.run(None, {"x": feats[None]})[0][0]

    def decode(self, feats):
        return self.dec.run(None, {"f": np.ascontiguousarray(feats.T)[None]})[0].reshape(-1)


class _TorchBackend(_Backend):
    name = "torchscript"

    def __init__(self, model_dir: Path, threads: int):
        import torch

        self.torch = torch
        torch.set_num_threads(threads)
        self.fe = torch.jit.load(str(model_dir / FEATURE_FILE), map_location="cpu").eval()
        self.dec = torch.jit.load(str(model_dir / DECODER_FILE), map_location="cpu").eval()

    def features(self, feats):
        with self.torch.inference_mode():
            return self.fe(self.torch.from_numpy(feats)[None])["last_hidden_state"][0].numpy()

    def decode(self, feats):
        with self.torch.inference_mode():
            return self.dec(self.torch.from_numpy(np.ascontiguousarray(feats.T))[None]).reshape(-1).numpy()


def _has_onnx(d: Path) -> bool:
    return (d / ONNX_FE).is_file() and (d / ONNX_DEC).is_file() and importlib.util.find_spec("onnxruntime") is not None


def _has_torch(d: Path) -> bool:
    return (d / FEATURE_FILE).is_file() and (d / DECODER_FILE).is_file() and importlib.util.find_spec("torch") is not None


def is_available(model_dir: str | Path) -> bool:
    """True if the model files and a runtime (onnxruntime or torch) are present. Imports nothing heavy."""
    d = Path(model_dir)
    return _has_onnx(d) or _has_torch(d)


def _backend(model_dir: str | Path, threads: int, backend: str, int8: bool) -> _Backend:
    d = Path(model_dir).resolve()
    if backend == "auto":
        backend = "onnx" if _has_onnx(d) else "torch"
    key = (str(d), backend, threads if backend == "onnx" else 0, bool(int8) and backend == "onnx")
    with _lock:
        b = _backends.get(key)
        if b is None:
            if backend == "onnx":
                b = _OnnxBackend(d, threads, int8)
            elif backend == "torch":
                b = _TorchBackend(d, threads)
            else:
                raise ValueError(f"unknown backend {backend!r}")
            _backends.clear()  # only ever keep one model set in RAM (~1 GB)
            _backends[key] = b
        if backend == "torch":
            b.torch.set_num_threads(threads)
        return b


def unload() -> None:
    """Drop the cached models (about 1 GB of RAM)."""
    with _lock:
        _backends.clear()


def verify_files(model_dir: str | Path) -> dict[str, bool]:
    """sha256-check the TorchScript files against the pinned values (reads ~1 GB)."""
    out = {}
    for name, want in SHA256.items():
        h = hashlib.sha256()
        with open(Path(model_dir) / name, "rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                h.update(block)
        out[name] = h.hexdigest() == want
    return out


# ---------------------------------------------------------------------------------------------
# Restoration of one mono channel
# ---------------------------------------------------------------------------------------------

def _restore_mono(
    x: np.ndarray,
    out: np.ndarray,
    models,
    *,
    window_seconds: float,
    overlap_seconds: float,
    decode_seconds: float,
    decode_context: int,
    progress,
    cancel,
) -> None:
    be = models
    n = len(x)
    n_feat = -(-n // DECODER_UPSAMPLE)  # features covering the signal
    total = n_feat + _TAIL_FEATURES
    win = max(100, int(window_seconds * FEATURE_RATE))
    ov = min(max(10, int(overlap_seconds * FEATURE_RATE)), win // 2 - 1)
    dec_n = max(50, int(decode_seconds * FEATURE_RATE))
    ctx = decode_context
    pad48 = 2 * SAMPLE_RATE  # real audio context for the 48 kHz -> 16 kHz filters
    rng = np.random.default_rng(0)

    def check_cancel():
        if cancel is not None and cancel.is_set():
            raise Cancelled()

    def window_feats(s: int, e: int) -> np.ndarray:
        """w2v-BERT features for frames [s, e), using real audio on both sides as context."""
        check_cancel()
        a48 = s * DECODER_UPSAMPLE - pad48
        b48 = e * DECODER_UPSAMPLE + pad48
        lo, hi = max(a48, 0), min(b48, n)
        seg = np.zeros(b48 - a48, np.float32)
        if hi > lo:
            seg[lo - a48 : hi - a48] = x[lo:hi]
        w16 = _to_16k(seg)
        keep = pad48 // 3
        w16 = w16[keep - _FBANK_PAD : keep + (e - s) * FEATURE_SAMPLES + _FBANK_PAD].copy()
        if s == 0:  # file start: the official recipe zero-pads (no audio before sample 0)
            w16[:_FBANK_PAD] = 0
        w16 += _DITHER * rng.standard_normal(len(w16)).astype(np.float32)
        return be.features(_fbank(w16))

    def feature_blocks():
        """Yield (start_frame, features) finished blocks in order; windows are cross-faded over `ov` frames."""
        tail = None
        s = 0
        while True:
            e = min(s + win, total)
            cur = window_feats(s, e)
            if tail is not None:
                t = ((np.arange(len(tail), dtype=np.float32) + 1) / (len(tail) + 1))[:, None]
                cur = np.concatenate([tail * (1 - t) + cur[: len(tail)] * t, cur[len(tail) :]])
            if e >= total:
                yield s, cur
                return
            yield s, cur[:-ov]
            tail = cur[-ov:]
            s = e - ov

    buf = np.zeros((0, 1024), np.float32)
    buf_start = 0  # feature index of buf[0]
    written = 0  # features decoded and written so far
    for start, block in feature_blocks():
        check_cancel()
        if len(buf) == 0:
            buf, buf_start = block, start
        else:
            buf = np.concatenate([buf, block])
        while written < n_feat:
            end = min(written + dec_n, n_feat)
            if buf_start + len(buf) < min(end + ctx, total):
                break
            a = max(0, written - ctx)
            b = min(buf_start + len(buf), end + ctx)
            y = be.decode(buf[a - buf_start : b - buf_start])
            lo = (written - a) * DECODER_UPSAMPLE
            seg = y[lo : lo + (end - written) * DECODER_UPSAMPLE]
            o0 = written * DECODER_UPSAMPLE
            m = min(len(seg), n - o0)
            out[o0 : o0 + m] = seg[:m]
            written = end
            drop = (written - ctx) - buf_start
            if drop > 0:
                buf, buf_start = buf[drop:], buf_start + drop
            if progress:
                progress(written / n_feat)
            check_cancel()


# Side energy this far below the mid energy: the recording is effectively mono (dual mic, mono source).
_MONO_SIDE_DB = -20.0
# Tuned on speech: ~30 s windows keep the quadratic attention cheap and cancel latency low.
_WINDOW_SECONDS = 16.0
_OVERLAP_SECONDS = 2.0
_DECODE_SECONDS = 6.0
_DECODE_CONTEXT = 12  # frames (0.24 s) of decoder context per side, cropped away (10 already gives bit-identical output)


def _side_db(x: np.ndarray) -> float:
    mid = 0.5 * (x[:, 0].astype(np.float64) + x[:, 1])
    side = 0.5 * (x[:, 0].astype(np.float64) - x[:, 1])
    return 10 * np.log10((np.mean(side**2) + 1e-20) / (np.mean(mid**2) + 1e-20))


def available(model_dir: str | Path) -> bool:
    """The ONNX models are there (the product never needs torch: TorchScript is a research fallback)."""
    d = Path(model_dir)
    return (d / ONNX_FE).is_file() and (d / ONNX_DEC).is_file()


def restore(
    x: np.ndarray,
    model_dir: str | Path,
    threads: int = 4,
    progress=None,
    cancel: threading.Event | None = None,
    out: np.ndarray | None = None,
    *,
    int8: bool = False,
    backend: str = "auto",
    stereo: str = "auto",
    window_seconds: float = _WINDOW_SECONDS,
    overlap_seconds: float = _OVERLAP_SECONDS,
) -> np.ndarray:
    """Restore 48 kHz float32 speech `x` of shape (n,) or (n, channels) with Sidon.

    Returns an array of the same shape (written into `out` when given, e.g. a memory-mapped
    file; `out` may not alias `x`). The output level is the model's own (roughly speech
    normalised), so run loudness normalisation after this step.

    * `progress(fraction)` is called after every ~6 s of audio; setting `cancel` raises
      `Cancelled` within one window (a few seconds of processing).
    * `int8=True` uses the int8 feature extractor if `sidon_fe_q8.onnx` exists: 277 MB instead of
      398 MB on disk and ~0.5 GB less RAM, same speed, ~5% feature error (no word changes in tests).
    * `stereo`: "auto" restores the mid signal once and writes it to both channels when the
      side signal is more than 20 dB below it, otherwise every channel separately; "channels"
      and "mid" force one or the other. More than two channels are always done separately.
    * `backend`: "auto" (onnxruntime if the .onnx files exist, else torch + TorchScript),
      "onnx" or "torch". Peak memory is bounded by the window size, not by the file length
      (plus ~1 GB for the models).
    """
    x = np.asarray(x)
    if x.dtype != np.float32 or x.ndim not in (1, 2):
        raise ValueError("x must be float32 with shape (n,) or (n, channels)")
    if out is None:
        out = np.empty(x.shape, np.float32)
    elif out.shape != x.shape or out.dtype != np.float32:
        raise ValueError("out must be float32 with the same shape as x")
    if len(x) == 0:
        return out
    be = _backend(model_dir, threads, backend, int8)
    kw = dict(window_seconds=window_seconds, overlap_seconds=overlap_seconds, decode_seconds=_DECODE_SECONDS,
              decode_context=_DECODE_CONTEXT)

    if x.ndim == 1:
        jobs = [(x, out)]
    else:
        ch = x.shape[1]
        mode = stereo
        if mode == "auto":
            mode = "mid" if ch == 2 and _side_db(x) < _MONO_SIDE_DB else "channels"
        if mode == "mid" and ch == 2:
            jobs = [(np.ascontiguousarray(0.5 * (x[:, 0] + x[:, 1])), None)]
        else:
            jobs = [(np.ascontiguousarray(x[:, c]), out[:, c]) for c in range(ch)]
    for i, (src, dst) in enumerate(jobs):
        target = dst if dst is not None else np.empty(len(src), np.float32)
        base = i / len(jobs)

        def prog(f, base=base):
            if progress:
                progress(base + f / len(jobs))

        if dst is not None and not dst.flags.c_contiguous:  # a column view of (n, ch): work in a temp
            tmp = np.empty(len(src), np.float32)
            _restore_mono(src, tmp, be, progress=prog, cancel=cancel, **kw)
            dst[:] = tmp
        else:
            _restore_mono(src, target, be, progress=prog, cancel=cancel, **kw)
        if dst is None:  # mid-only: same restored signal on both channels
            out[:] = target[:, None]
    if progress:
        progress(1.0)
    # restoration can overshoot a little; keep the file free of clipping
    step = 1 << 22
    peak = max((float(np.max(np.abs(out[a : a + step]))) for a in range(0, len(out), step)), default=0.0)
    if peak > 0.99:
        g = np.float32(0.99 / peak)
        for a in range(0, len(out), step):
            out[a : a + step] *= g
    return out


__all__ = ["restore", "is_available", "unload", "verify_files", "Cancelled", "SAMPLE_RATE"]
