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

GPU (docs/research/gpu-2026-10.md): `device="cuda"` runs the batched file on the card (onnxruntime-gpu, CUDA graphs:
one replay per 10 ms frame instead of ~600 kernel launches, 8.7 ms at 4 rows, 38 ms at 64) with the recurrent state
kept on the card. The card is latency bound, so it wants as many segments as there are in the file in one call (up to
64): same segmentation, same output (SNR > 90 dB against the CPU), about 0.11 s of wall time per second of audio on a
10 minute file against 0.27 on four cores, and 0.06 when it has 64 segments, using one CPU thread.
"""
from __future__ import annotations

import contextlib
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import scipy.fft

from . import blocks as _blocks
from .dfn import Cancelled  # same exception, so callers catch one type

SAMPLE_RATE = 48_000
WIN = 960
HOP = 480
BINS = WIN // 2 + 1
# Output lags input by 4 hops (2 windows): STFT overlap + model look-ahead.
MODEL_DELAY = 4 * HOP
MODEL_FILE = "dpdfnet8_48khz_hr_batch.onnx"  # the published weights with a dynamic batch (tools/export_dpdfnet_batch.py)
MAX_BATCH = 8  # segments per model call: 8 is within 3 % of the best per-frame cost measured (16)

# Rows per call on the card: every size is its own CUDA graph (~90 MB of the card's memory, made when first used), and a
# call is padded up to the next size. A call costs 8.7 ms a frame for one row and about 0.5 ms more for each further
# row up to 12 (GTX 960: 4 rows 9.6 ms, 8 rows 11.0, 16 rows 14.8, 32 rows 23.2, 64 rows 38.4), so the sizes are close
# together where a file usually lands (up to a few dozen segments) and a padded row is cheap.
GPU_ROWS = (2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64)
# The default cuDNN search is exhaustive; the arena grows by what is asked instead of by powers of two.
_CUDA_OPTIONS = {"cudnn_conv_algo_search": "HEURISTIC", "arena_extend_strategy": "kSameAsRequested", "enable_cuda_graph": "1"}

_BLOCK_FRAMES = 200  # frames (2 s) per FFT batch / cancel check / write
_WINDOW = np.sin(0.5 * np.pi * np.sin(0.5 * np.pi * (np.arange(WIN) + 0.5) / HOP) ** 2).astype(np.float32)


def default_model_path() -> Path:
    env = os.environ.get("ACW_DPDFNET_MODEL")
    if env:
        return Path(env)
    return Path.home() / "audio-cleaner" / "models" / "dpdfnet" / MODEL_FILE


_lock = threading.Lock()
_models: dict[tuple[str, str], "_Model"] = {}


def unload() -> None:
    """Drop the cached model sessions (and with them what the card holds)."""
    with _lock:
        _models.clear()


class _Model:
    """One shared InferenceSession (run() is thread-safe) and the initial state of one stream."""

    gpu = False

    def __init__(self, path: str):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._open(ort, path, opts, ["CPUExecutionProvider"])

    def _open(self, ort, path: str, opts, providers) -> None:
        self.sess = sess = ort.InferenceSession(path, sess_options=opts, providers=providers)
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
        self.max_batch = MAX_BATCH

    def exclusive(self):
        """Held while one group of segments is streamed through the model (the card's buffers are not shared)."""
        return contextlib.nullcontext()

    def new_state(self, rows: int) -> np.ndarray:
        return np.repeat(self.init[None, :], rows, axis=0)

    def step(self, spec: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """spec (rows, 1, BINS, 2) -> (enhanced (rows, 1, BINS, 2), new state)."""
        return self.sess.run(self.out_names, {self.in_n: spec, self.st_n: state})


class _Rung:
    """One call size: its own session (onnxruntime 1.19 crashes with an illegal memory access when a session replays a
    CUDA graph again after it captured one for another input size, so every size has a session of its own), the spec in
    and out, and two state buffers that the two graphs read and write alternately (graph 0: a -> b, graph 1: b -> a)."""

    def __init__(self, ort, sess, rows: int, state_size: int):
        self.rows = rows
        self.sess = sess
        self.host_spec = np.zeros((rows, 1, BINS, 2), dtype=np.float32)
        self.spec = ort.OrtValue.ortvalue_from_numpy(self.host_spec, "cuda", 0)
        self.out = ort.OrtValue.ortvalue_from_numpy(np.zeros_like(self.host_spec), "cuda", 0)
        zeros = np.zeros((rows, state_size), dtype=np.float32)
        self.state = [ort.OrtValue.ortvalue_from_numpy(zeros, "cuda", 0) for _ in range(2)]
        self.runs = []
        for k in range(2):
            io = sess.io_binding()
            io.bind_ortvalue_input("spec", self.spec)
            io.bind_ortvalue_input("state_in", self.state[k])
            io.bind_ortvalue_output("spec_e", self.out)
            io.bind_ortvalue_output("state_out", self.state[1 - k])
            ro = ort.RunOptions()
            ro.add_run_config_entry("gpu_graph_id", str(k))
            self.runs.append((io, ro))


class _GpuState:
    """The state of a group of rows: it lives in a rung's buffers on the card, `parity` says which of the two
    holds it now."""

    def __init__(self, rung: _Rung, rows: int):
        self.rung, self.rows, self.parity = rung, rows, 0


class _GpuModel(_Model):
    """The batched model on the card. Segments are streamed through it one group at a time (`exclusive`), the
    group's state never leaves the card, and each 10 ms frame is one replay of a CUDA graph."""

    gpu = True

    def __init__(self, path: str):
        import onnxruntime as ort

        from .gpu import prepare_cuda

        prepare_cuda()
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._ort, self._path, self._opts = ort, path, opts
        self._providers = [("CUDAExecutionProvider", dict(_CUDA_OPTIONS))]
        self._open(ort, path, opts, self._providers)  # reads the metadata; the rungs run on sessions of their own
        if self.sess.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"CUDA is not usable for DPDFNet (providers: {self.sess.get_providers()})")
        self.max_batch = GPU_ROWS[-1]
        self._rungs: dict[int, _Rung] = {}
        self._run_lock = threading.Lock()

    def exclusive(self):
        return self._run_lock

    def new_state(self, rows: int) -> _GpuState:
        rung_rows = next((r for r in GPU_ROWS if r >= rows), None)
        if rung_rows is None:
            raise ValueError(f"at most {GPU_ROWS[-1]} rows per call")
        rung = self._rungs.get(rung_rows)
        if rung is None:
            sess = self._ort.InferenceSession(self._path, sess_options=self._opts, providers=self._providers)
            rung = self._rungs[rung_rows] = _Rung(self._ort, sess, rung_rows, len(self.init))
        rung.host_spec[:] = 0.0  # rows beyond the group's are fed silence
        rung.state[0].update_inplace(np.repeat(self.init[None, :], rung_rows, axis=0))
        return _GpuState(rung, rows)

    def step(self, spec: np.ndarray, state: _GpuState) -> tuple[np.ndarray, _GpuState]:
        rung = state.rung
        if spec.shape[0] != state.rows:
            raise ValueError("the rows of a group are fixed")
        rung.host_spec[:state.rows] = spec
        rung.spec.update_inplace(rung.host_spec)
        io, ro = rung.runs[state.parity]
        rung.sess.run_with_iobinding(io, ro)
        state.parity ^= 1
        return rung.out.numpy()[:state.rows].copy(), state


def _load(path: str, device: str = "cpu") -> _Model:
    if device not in ("cpu", "cuda"):
        raise ValueError(f"unknown device {device!r}")
    with _lock:
        hit = _models.get((path, device))
        if hit is None:
            hit = _models[path, device] = (_GpuModel if device == "cuda" else _Model)(path)
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
        self.state = None  # made by `push` (a stream that is only prepared / finished has none)
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
        state = self.state if self.state is not None else self._model.new_state(1)
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

    def __init__(self, x, ch, out, lo, s, e, head, tail, alpha, origin=0, col=None):
        self.x, self.ch, self.out = x, ch, out
        # `out` holds the samples from position `origin` on, in column `col` (default: the channel itself)
        self.origin, self.col = origin, ch if col is None else col
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
            self.out[m0 - self.origin:m1 - self.origin, self.col] = seg[m0 - lo:m1 - lo]
        t0, t1 = max(lo, self.e), hi
        if t1 > t0:
            self.tail_buf[t0 - self.e:t1 - self.e] = seg[t0 - lo:t1 - lo]


class _Row:
    """One segment: stream x[lo:hi] of channel `ch`; its results for [s, e) go to the output (plus `head` / `tail`
    crossfade samples kept in the sink)."""

    def __init__(self, ch, lo, hi, s, e, head, tail, out=None, origin=0, col=0):
        self.ch, self.lo, self.hi, self.s, self.e, self.head, self.tail = ch, lo, hi, s, e, head, tail
        # a row with its own `out` (holding the samples from `origin` on, column `col`) is not written to the shared output
        self.out, self.origin, self.col = out, origin, col

    @property
    def length(self) -> int:
        return self.hi - self.lo


def _run_rows(model: _Model, x, rows: list[_Row], out, alpha, cancel, report) -> list[_Sink]:
    with model.exclusive():
        return _stream_rows(model, x, rows, out, alpha, cancel, report)


def _stream_rows(model: _Model, x, rows: list[_Row], out, alpha, cancel, report) -> list[_Sink]:
    """Stream the segments `rows` through the model together, one batched call per frame, and return their sinks.
    The rows advance in lock step; a row that is shorter than the longest is fed zeros after its end (those frames
    are computed and dropped), so rows of similar length should share a call. A row's result does not depend on
    the others."""
    nb = len(rows)
    streams = [_Stream(model, _col(x, r.ch, r.lo, min(r.hi, r.lo + HOP + 1))) for r in rows]
    sinks = [_Sink(x, r.ch, out, r.lo, r.s, r.e, r.head, r.tail, alpha) if r.out is None
             else _Sink(x, r.ch, r.out, r.lo, r.s, r.e, r.head, r.tail, alpha, r.origin, r.col) for r in rows]
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
    segment_seconds: float | None = None,
    overlap_seconds: float | None = None,
    device: str = "cpu",
    grid: _blocks.Grid | None = None,
) -> np.ndarray:
    """Denoise 48 kHz float32 audio, (n,) or (n, channels), each channel on its
    own.

    The default is the editor's block run (blocks.GRID_HEAVY): 30 s blocks at multiples of 30 s from the start of the file,
    each run on its own from a fresh model state with 24 s of the recording before it and 0.25 s after it (about 41 dB
    against one contiguous run, PESQ >= 4.4; it never reaches 50 dB; compute x1.8), exact blocks without crossfade. The
    grid does not depend on `threads`. With `segment_seconds` and/or
    `overlap_seconds` the older cut is used: segments (60 s) with that much warm-up (2 s) and a 50 ms crossfade, the
    segments made smaller to spread a short file over `threads`.

    Returns an array shaped like `x`; results go straight into `out` when
    given (any writable float32 array/memmap of x's shape), so memory use does
    not grow with the file length. `progress(fraction)` is called regularly;
    setting `cancel` makes the call raise `Cancelled` within ~2 s of audio. `device` "cuda" streams the segments
    through the batched model on the GPU, all of them in as few calls as fit (`threads` then only decides how a
    short file is cut, as on the CPU, so the segments and the output are the same; it raises when CUDA is
    missing)."""
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
    model = _load(path, device)
    workers = 1 if model.gpu else threads  # one stream of CUDA calls drives the card
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

    if segment_seconds is None and overlap_seconds is None:
        return _denoise_blocks(x, out2, model, alpha, workers, threads, progress, cancel, grid or _blocks.grid_of("heavy"), out)
    segment_seconds = 60.0 if segment_seconds is None else segment_seconds
    overlap_seconds = 2.0 if overlap_seconds is None else overlap_seconds
    ov = int(overlap_seconds * SAMPLE_RATE)
    seg, cut = _plan(n, chans, threads, int(segment_seconds * SAMPLE_RATE), ov)
    if not cut:
        # whole channels as they are, no seams; the channels share a call
        rows = [_Row(c, 0, n, 0, n, 0, 0) for c in range(chans)]
        groups = [rows[i:i + model.max_batch] for i in range(0, len(rows), model.max_batch)]
        if workers > 1 and len(groups) > 1:
            with ThreadPoolExecutor(max_workers=min(workers, len(groups))) as pool:
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
    if model.gpu:
        # as full as the card takes: a call costs nearly the same for 4 rows as for 16
        groups = [rows[i:i + model.max_batch] for i in range(0, len(rows), model.max_batch)]
    else:
        nbatch = max(min(threads, len(rows)), math.ceil(len(rows) / model.max_batch))
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

    if workers > 1 and len(groups) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(groups))) as pool:
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


def _groups(rows: list[_Row], model: _Model, threads: int) -> list[list[_Row]]:
    """Batches of similar length, one per call; at least one per thread so every thread has work."""
    if not rows:
        return []
    rows.sort(key=lambda r: -r.length)
    if model.gpu:
        # as full as the card takes: a call costs nearly the same for 4 rows as for 16
        return [rows[i:i + model.max_batch] for i in range(0, len(rows), model.max_batch)]
    nbatch = max(min(threads, len(rows)), math.ceil(len(rows) / model.max_batch))
    base, extra = divmod(len(rows), nbatch)
    groups, i = [], 0
    for g in range(nbatch):
        size = base + (g < extra)
        groups.append(rows[i:i + size])
        i += size
    return groups


def _drive(model: _Model, x, groups: list[list[_Row]], out, alpha, cancel, report, workers: int, after=None) -> None:
    def work(group: list[_Row]) -> None:
        _run_rows(model, x, group, out, alpha, cancel, report)
        if after is not None:
            after(group)

    if workers > 1 and len(groups) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(groups))) as pool:
            list(pool.map(work, groups))
    else:
        for g in groups:
            work(g)


def run_windows(
    x: np.ndarray,
    windows: list[_blocks.Window],
    deliver,
    *,
    channel: int = 0,
    model_path: str | Path | None = None,
    threads: int = 4,
    device: str = "cpu",
    progress=None,
    cancel: threading.Event | None = None,
) -> None:
    """Run the model over each window (lo, s, e, hi) of channel `channel` of `x` on its own, from a fresh state: it reads
    [lo, hi) and `deliver(i, block)` gets the result for [s, e) (float32, delay removed, full strength). The windows of a
    call are batched through the model together (up to MAX_BATCH rows on the CPU, as many as the card takes on the GPU), each
    row exactly the model on that window. `deliver` is called under a lock. This is the block run of the editor's cache."""
    if not windows:
        return
    model = _load(str(model_path or default_model_path()), device)
    workers = 1 if model.gpu else threads
    rows = []
    for i, (lo, s, e, hi) in enumerate(windows):
        r = _Row(channel, lo, hi, s, e, 0, 0, out=np.zeros((e - s, 1), dtype=np.float32), origin=s, col=0)
        r.index = i
        rows.append(r)
    total = sum(e - s for _, s, e, _ in windows) or 1
    done = 0
    lock = threading.Lock()

    def report(k: int) -> None:
        nonlocal done
        if k and progress:
            with lock:
                done += k
                f = min(done / total, 1.0)
            progress(f)

    def after(group: list[_Row]) -> None:
        with lock:
            for r in group:
                deliver(r.index, r.out[:, 0])

    _drive(model, x, _groups(rows, model, threads), None, 0.0, cancel, report, workers, after)
    if progress:
        progress(1.0)


def _denoise_blocks(x, out2, model, alpha, workers, threads, progress, cancel, grid, out) -> np.ndarray:
    """The block run of `denoise_array`: every channel's grid blocks as rows."""
    n = x.shape[0]
    chans = out2.shape[1]
    rows = [_Row(c, lo, hi, s, e, 0, 0) for c in range(chans) for lo, s, e, hi in _blocks.grid_windows(0, n, grid)]
    total = max(n * chans, 1)
    done = 0
    lock = threading.Lock()

    def report(k: int) -> None:
        nonlocal done
        if k and progress:
            with lock:
                done += k
                f = min(done / total, 1.0)
            progress(f)

    _drive(model, x, _groups(rows, model, threads), out2, alpha, cancel, report, workers)
    if progress:
        progress(1.0)
    return out
