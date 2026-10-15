"""DPDFNet-8 (48 kHz "HR" variant) speech enhancement via onnxruntime (CPU), no torch.

Model: Ceva-IP/DPDFNet `onnx/dpdfnet8_48khz_hr.onnx` (Apache-2.0). It is a
causal streaming model: one call takes one STFT frame plus the recurrent state
and returns the enhanced frame plus the new state:

    spec     (1, 1, 481, 2)  float32   real/imag of one rfft frame
    state_in (90228,)        float32   packed RNN + running-norm state
    -> spec_e (1, 1, 481, 2), state_out (90228,)

Pre/post-processing (as in the official `dpdfnet` package, v0.6.0):
  * 48 kHz, n_fft = win = 960 (20 ms), hop = 480 (10 ms), 481 bins;
  * periodic-free Vorbis window w[n] = sin(pi/2 * sin^2(pi/2 * (n + .5) / 480)),
    used for analysis AND synthesis (w^2[n] + w^2[n+480] = 1, so plain
    overlap-add reconstructs exactly), no scaling, rfft/irfft unnormalised;
  * centre=True framing with reflect padding at the start (we pad 480 samples
    of reflection in front instead of librosa's);
  * initial state: zeros(state_size) with the first `erb_norm_state_size`
    entries set from the model metadata `erb_norm_init` and the following
    `spec_norm_state_size` from `spec_norm_init`;
  * the enhanced output lags the input by 4 hops = 1920 samples (40 ms), which
    is dropped (the official offline path drops `2 * win_len` samples).

The model has no attenuation-limit input. `atten_lim_db` is implemented like
DeepFilterNet's: out = a * noisy + (1 - a) * enhanced with a = 10 ** (-dB / 20)
(time-aligned; the official package blends in the STFT domain, which is
equivalent because the transform is linear and reconstructs perfectly). So
noise is reduced by roughly at most `atten_lim_db`; >= 100 means no blending.

`denoise_array` mirrors `dfn.denoise_array` (minus lib/model path arguments,
which default to the server layout). Audio is processed block-wise so memory
stays flat for long files; with `threads > 1` the input is cut into segments
that run in parallel (one onnxruntime call per frame is tiny, so intra-op
threading does not help) with 2 s of warm-up context and a short equal-power
crossfade, exactly like the DeepFilterNet path.
"""
from __future__ import annotations

import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import scipy.fft

from .dfn import Cancelled  # same exception, so callers catch one type

SAMPLE_RATE = 48_000
WIN = 960
HOP = 480
BINS = WIN // 2 + 1
# Output lags input by 4 hops (2 windows): STFT overlap + model look-ahead.
MODEL_DELAY = 4 * HOP
MODEL_FILE = "dpdfnet8_48khz_hr.onnx"

_BLOCK_FRAMES = 200  # frames (2 s) per FFT batch / cancel check / write
_WINDOW = np.sin(0.5 * np.pi * np.sin(0.5 * np.pi * (np.arange(WIN) + 0.5) / HOP) ** 2).astype(np.float32)


def default_model_path() -> Path:
    env = os.environ.get("ACW_DPDFNET_MODEL")
    if env:
        return Path(env)
    return Path.home() / "audio-cleaner" / "models" / "dpdfnet" / MODEL_FILE


_lock = threading.Lock()
_models: dict[str, tuple] = {}


def unload() -> None:
    """Drop the cached model sessions."""
    with _lock:
        _models.clear()


def _load(path: str):
    """One shared InferenceSession per model file (run() is thread-safe) and its initial state."""
    import onnxruntime as ort

    with _lock:
        hit = _models.get(path)
        if hit is not None:
            return hit
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        meta = sess.get_modelmeta().custom_metadata_map
        size = int(meta["state_size"])
        ne, ns = int(meta["erb_norm_state_size"]), int(meta["spec_norm_state_size"])
        state = np.zeros(size, dtype=np.float32)
        state[:ne] = [float(v) for v in meta["erb_norm_init"].split(",")]
        state[ne:ne + ns] = [float(v) for v in meta["spec_norm_init"].split(",")]
        hit = (sess, state, sess.get_inputs()[0].name, sess.get_inputs()[1].name,
               sess.get_outputs()[0].name, sess.get_outputs()[1].name)
        _models[path] = hit
        return hit


def _col(x, ch, a, b) -> np.ndarray:
    return np.ascontiguousarray(x[a:b, ch] if x.ndim == 2 else x[a:b], dtype=np.float32)


class _Stream:
    """Stateful frame-by-frame enhancer for one mono channel. `push` takes any
    number of input samples and returns the finished (not yet delay-trimmed)
    time-domain samples, a multiple of HOP per frame consumed. The first
    HOP samples of the input stream are preceded by a HOP-long reflection."""

    def __init__(self, model, first: np.ndarray):
        self._sess, self._init, self._in_n, self._st_n, self._out_n, self._ost_n = model
        self.state = self._init.copy()
        n = min(HOP, max(len(first) - 1, 0))
        pad = np.zeros(HOP, dtype=np.float32)
        if n:
            pad[HOP - n:] = first[n:0:-1]  # x[n], ..., x[1] (reflect, edge excluded)
        self._buf = pad
        self._ola = np.zeros(WIN, dtype=np.float32)  # pending overlap-add tail (HOP valid + HOP not yet complete)

    def push(self, x: np.ndarray, cancel: threading.Event | None = None) -> np.ndarray:
        buf = np.concatenate([self._buf, x]) if len(self._buf) else x
        k = (len(buf) - WIN) // HOP + 1 if len(buf) >= WIN else 0
        if k <= 0:
            self._buf = buf
            return np.zeros(0, dtype=np.float32)
        frames = np.lib.stride_tricks.as_strided(buf, (k, WIN), (buf.strides[0] * HOP, buf.strides[0]))
        spec = scipy.fft.rfft(frames * _WINDOW, axis=1)
        inp = np.empty((k, 1, 1, BINS, 2), dtype=np.float32)
        inp[:, 0, 0, :, 0] = spec.real
        inp[:, 0, 0, :, 1] = spec.imag
        enh = np.empty((k, BINS), dtype=np.complex64)
        run, state = self._sess.run, self.state
        names = [self._out_n, self._ost_n]
        for i in range(k):
            if cancel is not None and i % 50 == 0 and cancel.is_set():
                raise Cancelled()
            o, state = run(names, {self._in_n: inp[i], self._st_n: state})
            enh[i] = o[0, 0, :, 0] + 1j * o[0, 0, :, 1]
        self.state = state
        td = scipy.fft.irfft(enh, n=WIN, axis=1).astype(np.float32) * _WINDOW
        out = np.empty(k * HOP, dtype=np.float32)
        ola = self._ola
        for i in range(k):
            ola[:] += td[i]
            out[i * HOP:(i + 1) * HOP] = ola[:HOP]
            ola[:HOP] = ola[HOP:]
            ola[HOP:] = 0.0
        self._buf = buf[k * HOP:].copy()
        return out


class _Sink:
    """Routes a stream's output (absolute sample positions, delay removed) into
    `out` / small head+tail crossfade buffers, applying the atten-lim blend."""

    def __init__(self, x, ch, out, lo, s, e, head, tail, alpha):
        self.x, self.ch, self.out = x, ch, out
        # Absolute sample index of the next stream output sample: the stream starts
        # HOP samples before x[lo] (reflection pad) and lags by MODEL_DELAY.
        self.pos = lo - HOP - MODEL_DELAY
        self.s, self.e, self.head, self.tail, self.alpha = s, e, head, tail, alpha
        self.head_buf = np.zeros(head, dtype=np.float32)
        self.tail_buf = np.zeros(tail, dtype=np.float32)

    def write(self, y: np.ndarray) -> None:
        a, b = self.pos, self.pos + len(y)
        self.pos = b
        lo, hi = max(a, self.s), min(b, self.e + self.tail)
        if hi <= lo:
            return
        seg = y[lo - a:hi - a]
        if self.alpha:
            ref = _col(self.x, self.ch, lo, hi)
            seg = self.alpha * ref + (1.0 - self.alpha) * seg
        # head region [s, s+head) and tail region [e, e+tail) go to side buffers
        h0, h1 = lo, min(hi, self.s + self.head)
        if h1 > h0:
            self.head_buf[h0 - self.s:h1 - self.s] = seg[:h1 - h0]
        m0, m1 = max(lo, self.s + self.head), min(hi, self.e)
        if m1 > m0:
            self.out[m0:m1, self.ch] = seg[m0 - lo:m1 - lo]
        t0, t1 = max(lo, self.e), hi
        if t1 > t0:
            self.tail_buf[t0 - self.e:t1 - self.e] = seg[t0 - lo:t1 - lo]


def _run_segment(model, x, ch, out, lo, hi, s, e, head, tail, alpha, cancel, report) -> _Sink:
    """Stream x[lo:hi] (channel ch) through the model; results for [s, e) go to `out`
    (plus `head`/`tail` crossfade samples kept in the returned sink)."""
    st = _Stream(model, _col(x, ch, lo, min(hi, lo + HOP + 1)))
    sink = _Sink(x, ch, out, lo, s, e, head, tail, alpha)
    block = _BLOCK_FRAMES * HOP
    need = (hi - lo) + HOP + MODEL_DELAY  # stream samples needed to cover x[lo:hi]
    produced, p = 0, lo
    zeros = np.zeros(block, dtype=np.float32)
    while produced < need:
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        if p < hi:
            q = min(hi, p + block)
            chunk = _col(x, ch, p, q)
            p = q
        else:
            chunk = zeros  # flush the look-ahead
        y = st.push(chunk, cancel)
        produced += len(y)
        a = sink.pos
        sink.write(y)
        report(max(0, min(sink.pos, e) - max(a, s)))
    return sink


def denoise_array(
    x: np.ndarray,
    *,
    atten_lim_db: float = 100.0,
    threads: int = 4,
    progress=None,
    cancel: threading.Event | None = None,
    out: np.ndarray | None = None,
    model_path: str | Path | None = None,
    segment_seconds: float = 60.0,
    overlap_seconds: float = 2.0,
) -> np.ndarray:
    """Denoise 48 kHz float32 audio, (n,) or (n, channels), each channel on its
    own. Returns an array shaped like `x`; results go straight into `out` when
    given (any writable float32 array/memmap of x's shape), so memory use does
    not grow with the file length. `progress(fraction)` is called regularly;
    setting `cancel` makes the call raise `Cancelled` within ~2 s of audio."""
    x = np.asarray(x) if not isinstance(x, np.ndarray) else x
    mono = x.ndim == 1
    if x.ndim not in (1, 2):
        raise ValueError("expected (n,) or (n, channels) audio")
    n = x.shape[0]
    chans = 1 if mono else x.shape[1]
    if out is None:
        out = np.empty(x.shape, dtype=np.float32)
    out2 = out[:, None] if mono else out  # view; writes land in `out`
    if n == 0:
        return out
    path = str(model_path or default_model_path())
    model = _load(path)
    alpha = 0.0 if atten_lim_db >= 100 else min(float(10.0 ** (-max(float(atten_lim_db), 0.0) / 20.0)), 1.0)

    total_work = n * chans
    done = 0
    lock = threading.Lock()

    def report(k: int) -> None:
        nonlocal done
        if k and progress:
            with lock:
                done += k
                f = min(done / total_work, 1.0)
            progress(f)

    seg = int(segment_seconds * SAMPLE_RATE)
    ov = int(overlap_seconds * SAMPLE_RATE)
    if threads > 1:
        # make sure the work actually spreads over the threads
        seg = max(10 * SAMPLE_RATE, min(seg, math.ceil(n * chans / threads / HOP) * HOP))
    if threads <= 1 or n <= seg + ov:
        for c in range(chans):
            _run_segment(model, x, c, out2, 0, n, 0, n, 0, 0, alpha, cancel, report)
        if progress:
            progress(1.0)
        return out

    bounds = [(s, min(s + seg, n)) for s in range(0, n, seg)]
    fade = min(int(0.05 * SAMPLE_RATE), ov)
    heads: dict[tuple[int, int], np.ndarray] = {}
    tails: dict[tuple[int, int], np.ndarray] = {}

    def work(job: tuple[int, int]) -> None:
        c, i = job
        s, e = bounds[i]
        lo, hi = max(0, s - ov), min(n, e + ov)
        sink = _run_segment(model, x, c, out2, lo, hi, s, e,
                            fade if i > 0 else 0, fade if i + 1 < len(bounds) else 0, alpha, cancel, report)
        with lock:
            if i > 0:
                heads[c, i] = sink.head_buf
            if i + 1 < len(bounds):
                tails[c, i] = sink.tail_buf

    jobs = [(c, i) for c in range(chans) for i in range(len(bounds))]
    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(work, jobs))

    if fade:
        ramp = np.sin(np.linspace(0, np.pi / 2, fade, dtype=np.float32)) ** 2
        for c in range(chans):
            for i in range(1, len(bounds)):
                s = bounds[i][0]
                out2[s:s + fade, c] = tails[c, i - 1] * (1 - ramp) + heads[c, i] * ramp
    if progress:
        progress(1.0)
    return out
