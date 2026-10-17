"""DPDFNet-8 (48 kHz "HR" variant) speech enhancement via onnxruntime (CPU), no torch.

Model: Ceva-IP/DPDFNet `onnx/dpdfnet8_48khz_hr.onnx` (Apache-2.0), run as our
batched export of it (below). It is a causal streaming model: one call takes
one STFT frame plus the recurrent state and returns the enhanced frame plus the
new state (published batch-1 shapes):

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
stays flat for long files; the input is cut into segments with 2 s of warm-up
context and a short equal-power crossfade, exactly like the DeepFilterNet path.

Batching (docs/research/heavy-2026-10.md): one onnxruntime call per frame is
tiny and latency bound, so intra-op threading does not help, but the recurrent
layers are starved at batch 1. `dpdfnet8_48khz_hr_batch.onnx` (made by
tools/export_dpdfnet_batch.py from the same weights) takes a batch of
independent segments per call:

    spec     (B, 1, 481, 2)  float32
    state_in (B, 90228)      float32   one state row per segment
    -> spec_e (B, 1, 481, 2), state_out (B, 90228)

and each row is exactly the published batch-1 model on that segment, at 0.41
instead of 0.73 CPU-seconds per audio second on one Zen 1 core (batch 8). The
segments of a call are stepped in lock step; every worker thread owns one call
at a time.
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
MODEL_FILE = "dpdfnet8_48khz_hr_batch.onnx"  # the published weights with a dynamic batch (tools/export_dpdfnet_batch.py)
MAX_BATCH = 8  # segments per model call: 8 is within 3 % of the best per-frame cost measured (16)

_BLOCK_FRAMES = 200  # frames (2 s) per FFT batch / cancel check / write
_WINDOW = np.sin(0.5 * np.pi * np.sin(0.5 * np.pi * (np.arange(WIN) + 0.5) / HOP) ** 2).astype(np.float32)


def default_model_path() -> Path:
    env = os.environ.get("ACW_DPDFNET_MODEL")
    if env:
        return Path(env)
    return Path.home() / "audio-cleaner" / "models" / "dpdfnet" / MODEL_FILE


_lock = threading.Lock()
_models: dict[str, "_Model"] = {}


def unload() -> None:
    """Drop the cached model sessions."""
    with _lock:
        _models.clear()


class _Model:
    """One shared InferenceSession (run() is thread-safe) and the initial state of one stream."""

    def __init__(self, path: str):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        meta = sess.get_modelmeta().custom_metadata_map
        size = int(meta["state_size"])
        ne, ns = int(meta["erb_norm_state_size"]), int(meta["spec_norm_state_size"])
        state = np.zeros(size, dtype=np.float32)
        state[:ne] = [float(v) for v in meta["erb_norm_init"].split(",")]
        state[ne:ne + ns] = [float(v) for v in meta["spec_norm_init"].split(",")]
        self.init = state
        self.in_n, self.st_n = sess.get_inputs()[0].name, sess.get_inputs()[1].name
        self.out_names = [sess.get_outputs()[0].name, sess.get_outputs()[1].name]
        if len(sess.get_inputs()[1].shape) != 2:
            raise ValueError(f"{path}: not the batched model ({MODEL_FILE}, tools/export_dpdfnet_batch.py)")

    def new_state(self, rows: int) -> np.ndarray:
        return np.repeat(self.init[None, :], rows, axis=0)

    def step(self, spec: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """spec (rows, 1, BINS, 2) -> (enhanced (rows, 1, BINS, 2), new state)."""
        return self.sess.run(self.out_names, {self.in_n: spec, self.st_n: state})


def _load(path: str) -> _Model:
    with _lock:
        hit = _models.get(path)
        if hit is None:
            hit = _models[path] = _Model(path)
        return hit


def _col(x, ch, a, b) -> np.ndarray:
    return np.ascontiguousarray(x[a:b, ch] if x.ndim == 2 else x[a:b], dtype=np.float32)


class _Stream:
    """Frame-by-frame enhancer for one mono stream. `prepare` takes any number of input samples (a multiple of
    HOP once running) and returns their STFT frames, `finish` takes the model's frames for them back and returns
    the finished (not yet delay-trimmed) time-domain samples, HOP per frame. The first HOP samples of the input
    stream are preceded by a HOP-long reflection. `push` does both with the model for a stream on its own."""

    def __init__(self, model: _Model, first: np.ndarray):
        self._model = model
        self.state = model.new_state(1)
        n = min(HOP, max(len(first) - 1, 0))
        pad = np.zeros(HOP, dtype=np.float32)
        if n:
            pad[HOP - n:] = first[n:0:-1]  # x[n], ..., x[1] (reflect, edge excluded)
        self._buf = pad
        self._ola = np.zeros(WIN, dtype=np.float32)  # pending overlap-add tail (HOP valid + HOP not yet complete)

    def prepare(self, x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray | None:
        """STFT frames of the next samples as (k, 1, BINS, 2) float32 (into `out` when given), None if less than
        one frame is complete yet."""
        buf = np.concatenate([self._buf, x]) if len(self._buf) else x
        k = (len(buf) - WIN) // HOP + 1 if len(buf) >= WIN else 0
        if k <= 0:
            self._buf = buf
            return None
        frames = np.lib.stride_tricks.as_strided(buf, (k, WIN), (buf.strides[0] * HOP, buf.strides[0]))
        spec = scipy.fft.rfft(frames * _WINDOW, axis=1)
        inp = np.empty((k, 1, BINS, 2), dtype=np.float32) if out is None else out
        inp[:, 0, :, 0] = spec.real
        inp[:, 0, :, 1] = spec.imag
        self._buf = buf[k * HOP:].copy()
        return inp

    def finish(self, enhanced: np.ndarray) -> np.ndarray:
        """(k, 1, BINS, 2) model output -> k * HOP samples."""
        k = enhanced.shape[0]
        enh = enhanced[:, 0, :, 0] + 1j * enhanced[:, 0, :, 1]
        td = scipy.fft.irfft(enh.astype(np.complex64, copy=False), n=WIN, axis=1).astype(np.float32) * _WINDOW
        out = np.empty(k * HOP, dtype=np.float32)
        ola = self._ola
        for i in range(k):
            ola[:] += td[i]
            out[i * HOP:(i + 1) * HOP] = ola[:HOP]
            ola[:HOP] = ola[HOP:]
            ola[HOP:] = 0.0
        return out

    def push(self, x: np.ndarray, cancel: threading.Event | None = None) -> np.ndarray:
        inp = self.prepare(x)
        if inp is None:
            return np.zeros(0, dtype=np.float32)
        enh = np.empty_like(inp)
        state = self.state
        for i in range(len(inp)):
            if cancel is not None and i % 50 == 0 and cancel.is_set():
                raise Cancelled()
            o, state = self._model.step(inp[i:i + 1], state)
            enh[i] = o[0]
        self.state = state
        return self.finish(enh)


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


class _Row:
    """One segment: stream x[lo:hi] of channel `ch`; its results for [s, e) go to the output (plus `head` / `tail`
    crossfade samples kept in the sink)."""

    def __init__(self, ch, lo, hi, s, e, head, tail):
        self.ch, self.lo, self.hi, self.s, self.e, self.head, self.tail = ch, lo, hi, s, e, head, tail

    @property
    def length(self) -> int:
        return self.hi - self.lo


def _run_rows(model: _Model, x, rows: list[_Row], out, alpha, cancel, report) -> list[_Sink]:
    """Stream the segments `rows` through the model together, one batched call per frame, and return their sinks.
    The rows advance in lock step; a row that is shorter than the longest is fed zeros after its end (those frames
    are computed and dropped), so rows of similar length should share a call. A row's result does not depend on
    the others."""
    nb = len(rows)
    streams = [_Stream(model, _col(x, r.ch, r.lo, min(r.hi, r.lo + HOP + 1))) for r in rows]
    sinks = [_Sink(x, r.ch, out, r.lo, r.s, r.e, r.head, r.tail, alpha) for r in rows]
    block = _BLOCK_FRAMES * HOP
    need = max(r.length for r in rows) + HOP + MODEL_DELAY  # stream samples needed to cover every row
    pos = [r.lo for r in rows]
    state = model.new_state(nb)
    produced = 0
    while produced < need:
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        size = min(block, -(-(need - produced) // HOP) * HOP)  # the last push is as short as it can be
        k = size // HOP
        spec = np.zeros((k, nb, 1, BINS, 2), dtype=np.float32)
        for j, (r, st) in enumerate(zip(rows, streams)):
            chunk = np.zeros(size, dtype=np.float32)
            q = min(r.hi, pos[j] + size)
            if q > pos[j]:
                chunk[:q - pos[j]] = _col(x, r.ch, pos[j], q)
                pos[j] = q
            frames = st.prepare(chunk)
            if frames is None or len(frames) != k:  # cannot happen: a push of k * HOP samples yields k frames
                raise RuntimeError("DPDFNet framing out of step")
            spec[:, j] = frames
        enh = np.empty_like(spec)
        for i in range(k):
            if cancel is not None and i % 50 == 0 and cancel.is_set():
                raise Cancelled()
            o, state = model.step(spec[i], state)
            enh[i] = o
        produced += size
        for j, (st, sink) in enumerate(zip(streams, sinks)):
            y = st.finish(enh[:, j])
            a = sink.pos
            sink.write(y)
            report(max(0, min(sink.pos, rows[j].e) - max(a, rows[j].s)))
    return sinks


def _plan(n: int, chans: int, threads: int, seg: int, ov: int) -> tuple[int, bool]:
    """(segment length in samples, whether to cut the channels into segments at all). The segmentation does not
    depend on the model file: the segments, their warm-up and their crossfades are what they always were, so the
    output of the batched model equals that of the batch-1 model; batching only changes how many are stepped
    together. (Shorter segments would batch short files too, but every segmentation changes the output a little:
    the recurrent state after a 2 s warm-up is not the state of a continuous pass.)"""
    if threads > 1:
        # make sure the work actually spreads over the threads
        seg = max(10 * SAMPLE_RATE, min(seg, math.ceil(n * chans / threads / HOP) * HOP))
    return seg, threads > 1 and n > seg + ov


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

    ov = int(overlap_seconds * SAMPLE_RATE)
    seg, cut = _plan(n, chans, threads, int(segment_seconds * SAMPLE_RATE), ov)
    if not cut:
        # whole channels as they are, no seams; the channels share a call
        rows = [_Row(c, 0, n, 0, n, 0, 0) for c in range(chans)]
        groups = [rows[i:i + MAX_BATCH] for i in range(0, len(rows), MAX_BATCH)]
        if threads > 1 and len(groups) > 1:
            with ThreadPoolExecutor(max_workers=min(threads, len(groups))) as pool:
                list(pool.map(lambda g: _run_rows(model, x, g, out2, alpha, cancel, report), groups))
        else:
            for g in groups:
                _run_rows(model, x, g, out2, alpha, cancel, report)
        if progress:
            progress(1.0)
        return out

    bounds = [(s, min(s + seg, n)) for s in range(0, n, seg)]
    fade = min(int(0.05 * SAMPLE_RATE), ov)
    # What a segment needs after its end: the crossfade samples and the look-ahead of the model (the model and the
    # STFT are causal), nothing more; the full overlap is only needed before the start, to warm the state up.
    tail_ctx = min(ov, fade + MODEL_DELAY + 3 * WIN)
    heads: dict[tuple[int, int], np.ndarray] = {}
    tails: dict[tuple[int, int], np.ndarray] = {}
    rows = []
    index: dict[int, tuple[int, int]] = {}
    for c in range(chans):
        for i, (s, e) in enumerate(bounds):
            r = _Row(c, max(0, s - ov), min(n, e + tail_ctx), s, e, fade if i > 0 else 0, fade if i + 1 < len(bounds) else 0)
            index[id(r)] = (c, i)
            rows.append(r)
    # batches of similar length, one per call; at least one per thread so every thread has work
    rows.sort(key=lambda r: -r.length)
    nbatch = max(min(threads, len(rows)), math.ceil(len(rows) / MAX_BATCH))
    base, extra = divmod(len(rows), nbatch)
    groups, i = [], 0
    for g in range(nbatch):
        size = base + (g < extra)
        groups.append(rows[i:i + size])
        i += size

    def work(group: list[_Row]) -> None:
        sinks = _run_rows(model, x, group, out2, alpha, cancel, report)
        with lock:
            for r, sink in zip(group, sinks):
                c, i = index[id(r)]
                if i > 0:
                    heads[c, i] = sink.head_buf
                if i + 1 < len(bounds):
                    tails[c, i] = sink.tail_buf

    if threads > 1 and len(groups) > 1:
        with ThreadPoolExecutor(max_workers=min(threads, len(groups))) as pool:
            list(pool.map(work, groups))
    else:
        for g in groups:
            work(g)

    if fade:
        ramp = np.sin(np.linspace(0, np.pi / 2, fade, dtype=np.float32)) ** 2
        for c in range(chans):
            for i in range(1, len(bounds)):
                s = bounds[i][0]
                out2[s:s + fade, c] = tails[c, i - 1] * (1 - ramp) + heads[c, i] * ramp
    if progress:
        progress(1.0)
    return out
