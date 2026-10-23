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

The two networks run on onnxruntime (already a dependency of the app) from our ONNX conversion
of the official TorchScript files (tools/convert_sidon_onnx.py). The feature extractor is windowed (`window_seconds`, linear cross-fade over the overlap) and
the convolutional decoder runs on short blocks with context frames that are cropped away, so
memory stays bounded for any file length. Per-window mean/variance normalisation of the fbank
follows the official demo, which normalises per chunk.

onnxruntime is imported lazily, only when a model is loaded.
"""
from __future__ import annotations

import functools
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .dfn import Cancelled

SAMPLE_RATE = 48_000
MODEL_RATE = 16_000
# ONNX conversion of sarulab-speech/sidon-v0.1 (see docs/research/restore-sidon.md).
ONNX_FE = "sidon_fe.onnx"
ONNX_DEC = "sidon_dec.onnx"

FEATURE_SAMPLES = 320  # 16 kHz samples per feature frame (20 ms)
DECODER_UPSAMPLE = 960  # 48 kHz samples per feature frame
FEATURE_RATE = MODEL_RATE / FEATURE_SAMPLES  # 50 per second
_FBANK_PAD = 160  # the official demo pads each chunk by 160 samples per side so frames are centred
_TAIL_FEATURES = 32  # zero tail so the last real frame is not at the very edge of the decoder input
# Dither floor (about -100 dBFS): keeps digital silence from hitting the log-mel floor, which would
# skew the per-window statistics. Deterministic: the noise of a window is seeded by the window's index (_dither), so a
# window gives the same result in a whole-file run and in a block run of the editor (blocks.py).
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
# Models
# ---------------------------------------------------------------------------------------------

_lock = threading.Lock()
_backends: dict[tuple, "_Backend"] = {}


# onnxruntime session settings (measured in docs/research/heavy-2026-10.md): the memory arena stays on (off: -0.3 GB
# peak, +28 % CPU), and so does the default graph optimisation (the only level that saves memory, none, changes
# the samples by up to 4e-5; basic and extended keep the memory and the samples).
_ARENA = True
_MAX_WORKERS = 4  # concurrent window / block calls; each holds up to ~0.7 GB of activations
_OPT_LEVEL = None
# CUDA provider options: the default exhaustive cuDNN search re-runs for every new input length, and the default arena
# rounds every growth up to a power of two (peak 3.7 GB on a 4 GB card against 2.6 GB when it grows by what is asked).
_CUDA_OPTIONS = {"cudnn_conv_algo_search": "HEURISTIC", "arena_extend_strategy": "kSameAsRequested"}
# One call at a time: the card is the bottleneck (82 % busy), two in flight only add 1 GB of activations (docs/research/gpu-2026-10.md).
_GPU_WORKERS = 1


class _Backend:
    """features(feats (T, 160)) -> (T, 1024); decode(feats (T, 1024)) -> (T * 960,) float32."""

    def __init__(self, model_dir: Path, threads: int, device: str):
        import onnxruntime as ort

        # `threads` is the size of the worker pool the caller runs the two networks on (restore._restore_mono):
        # blocks and windows are independent, so the sessions themselves are single-threaded (docs/research/heavy-2026-10.md)
        self.workers = max(1, min(threads, _MAX_WORKERS))
        if device == "cuda":
            from .gpu import prepare_cuda

            prepare_cuda()
            self.workers = _GPU_WORKERS  # the card does the arithmetic, one call at a time
            providers = [("CUDAExecutionProvider", dict(_CUDA_OPTIONS))]
        elif device == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            raise ValueError(f"unknown device {device!r}")

        def session(path: Path):
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            so.log_severity_level = 3
            so.enable_mem_pattern = False  # shapes vary with the file
            # idle worker threads sleep instead of spinning: this runs next to other services
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
            if not _ARENA:
                so.enable_cpu_mem_arena = False  # the arena keeps the largest activations of every window alive
            if _OPT_LEVEL is not None:
                so.graph_optimization_level = _OPT_LEVEL(ort)
            sess = ort.InferenceSession(str(path), so, providers=providers)
            if device == "cuda" and sess.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError(f"CUDA is not usable for {path.name} (providers: {sess.get_providers()})")
            return sess

        self.fe = session(model_dir / ONNX_FE)
        self.dec = session(model_dir / ONNX_DEC)

    def features(self, feats):
        return self.fe.run(None, {"x": feats[None]})[0][0]

    def decode(self, feats):
        return self.dec.run(None, {"f": np.ascontiguousarray(feats.T)[None]})[0].reshape(-1)


def _backend(model_dir: str | Path, threads: int, device: str = "cpu") -> _Backend:
    key = (str(Path(model_dir).resolve()), threads if device == "cpu" else 0, device)
    with _lock:
        b = _backends.get(key)
        if b is None:
            _backends.clear()  # only ever keep one model set in RAM (~1 GB)
            b = _backends[key] = _Backend(Path(key[0]), threads, device)
        return b


def unload() -> None:
    """Drop the cached models (about 1 GB of RAM)."""
    with _lock:
        _backends.clear()


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
    first_win: int = 0,
    write_from: int = 0,
    hi_frame: int | None = None,
    origin: int = 0,
) -> None:
    """Restore `x` into `out`. By default all of it. A block run (`restore_blocks`) starts at window `first_win`, writes the
    feature frames [write_from, hi_frame) only (the windows after the first are made as in a whole-file run, so the
    samples are the whole-file run's bit for bit) and `out` holds the samples from position `origin` on."""
    be = models
    n = len(x)
    n_feat = -(-n // DECODER_UPSAMPLE)  # features covering the signal
    total = n_feat + _TAIL_FEATURES
    win = max(100, int(window_seconds * FEATURE_RATE))
    ov = min(max(10, int(overlap_seconds * FEATURE_RATE)), win // 2 - 1)
    dec_n = max(50, int(decode_seconds * FEATURE_RATE))
    ctx = decode_context
    pad48 = 2 * SAMPLE_RATE  # real audio context for the 48 kHz -> 16 kHz filters
    hi = n_feat if hi_frame is None else min(hi_frame, n_feat)

    def check_cancel():
        if cancel is not None and cancel.is_set():
            raise Cancelled()

    def window_feats(idx: int, s: int, e: int) -> np.ndarray:
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
        w16 += _DITHER * _dither(idx, len(w16))
        return be.features(_fbank(w16))

    def decode_block(feats: np.ndarray, skip: int, o0: int, count: int) -> None:
        """Decode `feats` (a block with `skip` frames of context before it), keep `count` frames of the block."""
        check_cancel()
        y = be.decode(feats)
        lo = skip * DECODER_UPSAMPLE
        seg = y[lo : lo + count * DECODER_UPSAMPLE]
        m = min(len(seg), n - o0)
        out[o0 - origin : o0 - origin + m] = seg[:m]

    def windows():
        """The (index, start, end) frames of the feature windows, in order; they overlap by `ov` frames. A block run stops
        when the windows have made every feature the decoder needs up to `hi` (its context included)."""
        s, idx = first_win * (win - ov), first_win
        while True:
            e = min(s + win, total)
            yield idx, s, e
            if e >= total or e - ov >= min(hi + ctx, total):
                return
            s, idx = e - ov, idx + 1

    workers = be.workers
    pool = ThreadPoolExecutor(workers) if workers > 1 else _InlinePool()
    try:
        win_iter = windows()
        pending_windows: deque = deque()  # (start, end, future of its features), in order
        pending_decodes: deque = deque()  # (frames, future)
        tail = None
        buf = np.zeros((0, 1024), np.float32)
        buf_start = 0  # feature index of buf[0]
        written = write_from  # features handed to the decoder so far
        finished = 0  # features decoded and written
        # Work in flight bounds the extra memory: a 16 s feature window needs ~0.7 GB of activations and a decoder
        # block ~50 MB per second, all of it kept by onnxruntime's arena (turning the arena off costs 28 % CPU). The
        # feature extractor is a fifth of the work, so two windows at a time keep the workers busy.
        ahead = workers + 1
        windows_ahead = min(_FE_AHEAD, ahead)

        def drain(limit: int) -> None:
            nonlocal finished
            while len(pending_decodes) > limit:
                frames, fut = pending_decodes.popleft()
                fut.result()
                finished += frames
                if progress:
                    progress(finished / max(1, hi - write_from))
                check_cancel()

        while True:
            while len(pending_windows) < windows_ahead:
                nxt = next(win_iter, None)
                if nxt is None:
                    break
                wi, ws, we = nxt
                pending_windows.append((ws, we, pool.submit(window_feats, wi, ws, we)))
            if not pending_windows:
                break
            start, e, fut = pending_windows.popleft()
            cur = fut.result()
            check_cancel()
            if tail is not None:
                t = ((np.arange(len(tail), dtype=np.float32) + 1) / (len(tail) + 1))[:, None]
                cur = np.concatenate([tail * (1 - t) + cur[: len(tail)] * t, cur[len(tail) :]])
            if e >= total:
                block = cur
            else:
                block = cur[:-ov]
                tail = cur[-ov:]
            if len(buf) == 0:
                buf, buf_start = block, start
            else:
                buf = np.concatenate([buf, block])
            while written < hi:
                end = min(written + dec_n, hi)
                if hi - end < _MIN_DECODE:  # (a very short decoder call is not the same arithmetic to the last bit: no stub at the end)
                    end = hi
                if buf_start + len(buf) < min(end + ctx, total):
                    break
                a = max(buf_start, written - ctx)
                b = min(buf_start + len(buf), end + ctx)
                pending_decodes.append((end - written, pool.submit(
                    decode_block, buf[a - buf_start : b - buf_start], written - a, written * DECODER_UPSAMPLE,
                    end - written)))
                written = end
                drop = (written - ctx) - buf_start
                if drop > 0:
                    buf, buf_start = buf[drop:], buf_start + drop
                drain(ahead)
                check_cancel()
        drain(0)
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)


def _dither(window_index: int, count: int) -> np.ndarray:
    """The dither noise of one feature window: seeded by the window's index, so it does not depend on which run made it."""
    return np.random.default_rng(window_index).standard_normal(count).astype(np.float32)


def _grid(window_seconds: float, overlap_seconds: float) -> tuple[int, int]:
    """(window, overlap) in feature frames, as `_restore_mono` cuts them."""
    win = max(100, int(window_seconds * FEATURE_RATE))
    return win, min(max(10, int(overlap_seconds * FEATURE_RATE)), win // 2 - 1)


def block_samples(window_seconds: float | None = None, overlap_seconds: float | None = None) -> int:
    """The editor's Voice-rebuild block in samples: the step between windows (14 s with 16 s windows and 2 s of overlap),
    from the start of the file."""
    win, ov = _grid(_WINDOW_SECONDS if window_seconds is None else window_seconds,
                    _OVERLAP_SECONDS if overlap_seconds is None else overlap_seconds)
    return (win - ov) * DECODER_UPSAMPLE


def block_count(n: int, window_seconds: float | None = None, overlap_seconds: float | None = None) -> int:
    return -(-n // block_samples(window_seconds, overlap_seconds)) if n > 0 else 0


def block_bounds(b: int, n: int, window_seconds: float | None = None, overlap_seconds: float | None = None) -> tuple[int, int]:
    """Samples [start, end) of block `b` of a file of `n` samples."""
    size = block_samples(window_seconds, overlap_seconds)
    return b * size, min(n, (b + 1) * size)


def block_input_range(b: int, n: int, window_seconds: float | None = None, overlap_seconds: float | None = None) -> tuple[int, int]:
    """The samples of the recording that block `b` is made from: its window, the one before it and the one after it, each
    with the 2 s the filters read around it (clamped to the file)."""
    win, ov = _grid(_WINDOW_SECONDS if window_seconds is None else window_seconds,
                    _OVERLAP_SECONDS if overlap_seconds is None else overlap_seconds)
    step = win - ov
    lo = max(0, ((b - 1) * step if b > 0 else 0) * DECODER_UPSAMPLE - 2 * SAMPLE_RATE)
    hi = min(n, ((b + 1) * step + win) * DECODER_UPSAMPLE + 2 * SAMPLE_RATE)
    return lo, hi


def restore_blocks(
    x,
    model_dir: str | Path,
    blocks: list[int],
    deliver,
    *,
    threads: int = 4,
    device: str = "cpu",
    progress=None,
    cancel: threading.Event | None = None,
    window_seconds: float | None = None,
    overlap_seconds: float | None = None,
) -> None:
    """Rebuild the voice for the blocks `blocks` (indices of the grid `block_samples`) of mono `x` (1-D float32, or anything
    with len() and slicing that gives float32, like a column of a stereo array); `deliver(b, samples)` gets each block, the
    model's own level (before restore()'s peak limit). Each block is the whole-file run's block bit for bit: it is made
    from the window before it, its own and the one after it, with the dither of those windows. Neighbouring blocks share
    their windows, so a run of k blocks costs k + 2 windows. Peak memory is bounded by the window size."""
    wanted = sorted(set(int(b) for b in blocks))
    if not wanted:
        return
    ws = _WINDOW_SECONDS if window_seconds is None else window_seconds
    ovs = _OVERLAP_SECONDS if overlap_seconds is None else overlap_seconds
    be = _backend(model_dir, threads, device)
    n = len(x)
    win, ov = _grid(ws, ovs)
    step = win - ov
    n_feat = -(-n // DECODER_UPSAMPLE)
    runs: list[list[int]] = []
    for b in wanted:
        if b < 0 or b * step >= n_feat:
            raise ValueError(f"block {b} is outside the recording")
        if runs and runs[-1][1] == b - 1:
            runs[-1][1] = b
        else:
            runs.append([b, b])
    total_blocks = len(wanted)
    done_blocks = 0
    for b0, b1 in runs:
        s0 = b0 * step * DECODER_UPSAMPLE
        e1 = min(n, (b1 + 1) * step * DECODER_UPSAMPLE)
        buf = np.empty(e1 - s0, np.float32)
        count = b1 - b0 + 1

        def prog(f, base=done_blocks, count=count):
            if progress:
                progress(min((base + f * count) / total_blocks, 1.0))

        _restore_mono(x, buf, be, window_seconds=ws, overlap_seconds=ovs, decode_seconds=_DECODE_SECONDS,
                      decode_context=_DECODE_CONTEXT, progress=prog, cancel=cancel, first_win=max(b0 - 1, 0),
                      write_from=b0 * step, hi_frame=min((b1 + 1) * step, n_feat), origin=s0)
        for b in range(b0, b1 + 1):
            a, e = block_bounds(b, n, ws, ovs)
            deliver(b, buf[a - s0:e - s0].copy())
        done_blocks += count


class _InlinePool:
    """The ThreadPoolExecutor interface for one worker: runs the call in the caller's thread."""

    def submit(self, fn, *args):
        f: Future = Future()
        try:
            f.set_result(fn(*args))
        except BaseException as exc:  # noqa: BLE001 - re-raised by result()
            f.set_exception(exc)
        return f

    def shutdown(self, wait=True, cancel_futures=False):
        pass


# Side energy this far below the mid energy: the recording is effectively mono (dual mic, mono source).
_MONO_SIDE_DB = -20.0
# Tuned on speech: ~30 s windows keep the quadratic attention cheap and cancel latency low.
_WINDOW_SECONDS = 16.0
_OVERLAP_SECONDS = 2.0
_DECODE_SECONDS = 10.0  # 6 s re-decoded 24 context frames per 300 (8 %); 10 s with 10 context frames re-decode 4 %, same samples
_MIN_DECODE = 100  # frames: the decoder is never called on fewer than this (its result differs from a long call's in the last bits)
_DECODE_CONTEXT = 10  # frames (0.2 s) of decoder context per side, cropped away (12 gives bit-identical output too)
_FE_AHEAD = 2  # feature windows in flight (see _restore_mono)
_BLOCK = 1 << 20  # samples per step of the whole-file statistics: memory must not grow with the length of the file


def _side_db(x: np.ndarray) -> float:
    """Side energy over mid energy in dB (stereo `x`, any length: works in blocks, so a memory-mapped hour-long
    file needs a few MB, not the 8 bytes per sample and temporary of a whole-file computation)."""
    mid_e = side_e = 0.0
    for a in range(0, len(x), _BLOCK):
        b = x[a : a + _BLOCK].astype(np.float64)
        mid = 0.5 * (b[:, 0] + b[:, 1])
        side = 0.5 * (b[:, 0] - b[:, 1])
        mid_e += float(np.dot(mid, mid))
        side_e += float(np.dot(side, side))
    n = max(len(x), 1)
    return 10 * np.log10((side_e / n + 1e-20) / (mid_e / n + 1e-20))


class _Mid:
    """The mid signal 0.5 * (left + right) of a stereo array, computed on demand for the slices that are read
    (float32 like the whole-file version it replaces) instead of held as a second copy of the file."""

    def __init__(self, x: np.ndarray):
        self._x = x

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, sl: slice) -> np.ndarray:
        b = self._x[sl]
        return np.float32(0.5) * (b[:, 0] + b[:, 1])


def available(model_dir: str | Path) -> bool:
    """The models are there (the desktop app downloads them when voice rebuilding is first wanted)."""
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
    device: str = "cpu",
    stereo: str = "auto",
    window_seconds: float | None = None,
    overlap_seconds: float | None = None,
) -> np.ndarray:
    """Restore 48 kHz float32 speech `x` of shape (n,) or (n, channels) with Sidon.

    Returns an array of the same shape (written into `out` when given, e.g. a memory-mapped
    file; `out` may not alias `x`). The output level is the model's own (roughly speech
    normalised), so run loudness normalisation after this step.

    * `progress(fraction)` is called after every ~6 s of audio; setting `cancel` raises
      `Cancelled` within one window (a few seconds of processing).
    * `stereo`: "auto" restores the mid signal once and writes it to both channels when the
      side signal is more than 20 dB below it, otherwise every channel separately; "channels"
      and "mid" force one or the other. More than two channels are always done separately.
    * `device`: "cpu", or "cuda" (onnxruntime-gpu; raises when CUDA cannot be used, the output equals the CPU's up
      to float round-off).
    Peak memory is bounded by the window size, not by the file length (plus ~1 GB for the models).
    """
    x = np.asarray(x)
    window_seconds = _WINDOW_SECONDS if window_seconds is None else window_seconds  # (read now: tests make the window smaller)
    overlap_seconds = _OVERLAP_SECONDS if overlap_seconds is None else overlap_seconds
    if x.dtype != np.float32 or x.ndim not in (1, 2):
        raise ValueError("x must be float32 with shape (n,) or (n, channels)")
    if out is None:
        out = np.empty(x.shape, np.float32)
    elif out.shape != x.shape or out.dtype != np.float32:
        raise ValueError("out must be float32 with the same shape as x")
    if len(x) == 0:
        return out
    be = _backend(model_dir, threads, device)
    kw = dict(window_seconds=window_seconds, overlap_seconds=overlap_seconds, decode_seconds=_DECODE_SECONDS,
              decode_context=_DECODE_CONTEXT)

    # (source, destination) pairs, made one at a time: the sources are views (or the on-demand mid), never copies
    if x.ndim == 1:
        jobs = [lambda: (x, out)]
    else:
        ch = x.shape[1]
        mode = stereo
        if mode == "auto":
            mode = "mid" if ch == 2 and _side_db(x) < _MONO_SIDE_DB else "channels"
        if mode == "mid" and ch == 2:
            jobs = [lambda: (_Mid(x), out[:, 0])]
        else:
            jobs = [(lambda c=c: (x[:, c], out[:, c])) for c in range(ch)]
    for i, job in enumerate(jobs):
        src, dst = job()
        base = i / len(jobs)

        def prog(f, base=base):
            if progress:
                progress(base + f / len(jobs))

        _restore_mono(src, dst, be, progress=prog, cancel=cancel, **kw)
        if x.ndim == 2 and len(jobs) == 1 and x.shape[1] == 2:  # mid-only: same restored signal on both channels
            out[:, 1] = out[:, 0]
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


__all__ = ["restore", "restore_blocks", "block_samples", "block_count", "block_bounds", "block_input_range", "available", "unload",
           "Cancelled", "SAMPLE_RATE"]
