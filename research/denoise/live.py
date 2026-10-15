"""Causal / streaming candidates ("live" tiers): DeepFilterNet3 (+ low-latency model) via libDF,
DPDFNet (ONNX, stateful per-frame) and GTCRN.

Each class exposes
  hop            samples per model frame at the model's native rate
  sr             native sample rate
  stream(x)      run x (native rate, float32) frame by frame, returns (y, per_frame_seconds)
                 y is the raw stream output (still delayed by `delay` samples)
  enhance(x48)   48 kHz in -> 48 kHz out, resampled as needed, delay-compensated, same length
  delay          measured input->output delay in native samples (see measure_delay)
"""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
from scipy.signal import resample_poly

from common import LIBDF, MODELS, SR, SRC, resample

GTCRN_ROOT = SRC / "gtcrn"


def _xcorr_lag(ref: np.ndarray, out: np.ndarray, max_lag: int) -> int:
    """Lag L >= 0 maximising correlation of out[n+L] with ref[n] (FFT based)."""
    n = min(len(ref), len(out))
    ref, out = ref[:n].astype(np.float64), out[:n].astype(np.float64)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    cc = np.fft.irfft(np.fft.rfft(out, nfft) * np.conj(np.fft.rfft(ref, nfft)), nfft)
    cc = cc[: max_lag + 1]
    return int(np.argmax(cc))


class _Live:
    name = "live"
    sr = SR
    hop = 480
    win = 960
    delay = 0

    def stream(self, x):
        raise NotImplementedError

    def reset(self):
        pass

    def _native(self, x48):
        return resample(x48, SR, self.sr) if self.sr != SR else x48.astype(np.float32)

    def measure_delay(self, clean48: np.ndarray, max_ms: float = 100.0) -> float:
        """Empirical input->output delay in ms, from clean speech through the (reset) stream."""
        self.delay = 0
        self.reset()
        xn = self._native(clean48)
        xn = xn[: len(xn) // self.hop * self.hop]
        y, _ = self.stream(xn)
        self.delay = _xcorr_lag(xn, y, int(max_ms / 1000 * self.sr))
        return 1000.0 * self.delay / self.sr

    def __call__(self, x48):
        return self.enhance(x48)

    def enhance(self, x48):
        xn = self._native(x48)
        self.reset()
        pad = np.zeros(self.delay + self.hop, dtype=np.float32)
        xp = np.concatenate([xn, pad])
        xp = xp[: int(np.ceil(len(xp) / self.hop)) * self.hop]
        y, _ = self.stream(xp)
        y = y[self.delay: self.delay + len(xn)]
        return resample(y, self.sr, SR)[: len(x48)] if self.sr != SR else y[: len(x48)]


class LiveDFN(_Live):
    """DeepFilterNet3 / DeepFilterNet3-LL through libDF (tract, single thread)."""

    sr, hop, win = SR, 480, 960

    def __init__(self, model_tar, name):
        sys.path.insert(0, "/home/silver/repos/Audio-Cleaner-Web/server")
        from acw.dsp import dfn
        self.name = name
        self._dfn = dfn
        self.model_tar = str(model_tar)
        self.df = dfn.DeepFilter(str(LIBDF), self.model_tar)
        self.hop = self.df.hop

    def reset(self):
        self.df.close()
        self.df = self._dfn.DeepFilter(str(LIBDF), self.model_tar)

    def stream(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        out = np.zeros_like(x)
        h = self.hop
        fn, st = self.df._lib.df_process_frame, self.df._st
        FP = self._dfn._FP
        times = np.empty(len(x) // h)
        pc = time.perf_counter
        for k in range(len(x) // h):
            a = x[k * h:(k + 1) * h]
            t0 = pc()
            fn(st, a.ctypes.data_as(FP), out[k * h:(k + 1) * h].ctypes.data_as(FP))
            times[k] = pc() - t0
        return out, times


class LiveDPDFNet(_Live):
    """DPDFNet streaming ONNX models (Apache-2.0), driven exactly like dpdfnet.StreamEnhancer
    (causal STFT, Vorbis window, 50 % overlap-add) but at the model's native rate."""

    def __init__(self, onnx_name, sr, intra_threads=1):
        import onnxruntime as ort
        self.name, self.sr = onnx_name, sr
        self.win = int(round(0.02 * sr))
        self.hop = self.win // 2
        n = np.arange(self.win)
        self.window = np.sin(np.pi / 2 * np.sin(np.pi * (n + 0.5) / self.win) ** 2).astype(np.float32)  # Vorbis
        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(MODELS / "dpdfnet" / f"{onnx_name}.onnx"), so, providers=["CPUExecutionProvider"])
        meta = self.sess.get_modelmeta().custom_metadata_map
        ssz, e, s = int(meta["state_size"]), int(meta["erb_norm_state_size"]), int(meta["spec_norm_state_size"])
        self.init_state = np.zeros(ssz, dtype=np.float32)
        self.init_state[:e] = [float(v) for v in meta["erb_norm_init"].split(",")]
        self.init_state[e:e + s] = [float(v) for v in meta["spec_norm_init"].split(",")]
        i, o = self.sess.get_inputs(), self.sess.get_outputs()
        assert (i[0].shape[-2] - 1) * 2 == self.win, (i[0].shape, self.win)
        self.in_spec, self.in_state, self.out_spec, self.out_state = i[0].name, i[1].name, o[0].name, o[1].name

    def stream(self, x):
        x = np.asarray(x, dtype=np.float32)
        W, H = self.win, self.hop
        state = self.init_state.copy()
        nfr = (len(x) - W) // H + 1
        y = np.zeros(len(x) + W, dtype=np.float32)
        times = np.empty(nfr)
        pc = time.perf_counter
        win = self.window
        for k in range(nfr):
            t0 = pc()
            sp = np.fft.rfft(x[k * H:k * H + W] * win)
            ri = np.stack([sp.real, sp.imag], -1).astype(np.float32)[None, None]
            e, state = self.sess.run([self.out_spec, self.out_state], {self.in_spec: ri, self.in_state: state})
            fr = e[0, 0]
            y[k * H:k * H + W] += np.fft.irfft(fr[:, 0] + 1j * fr[:, 1], n=W).astype(np.float32) * win
            times[k] = pc() - t0
        # Host convention (as dpdfnet.StreamEnhancer): the call that completes frame k returns
        # block k, i.e. output index = position + hop. Stream-level lag then equals the latency.
        return np.concatenate([np.zeros(H, np.float32), y[: len(x) - H]]), times


class LiveGTCRN(_Live):
    """GTCRN (MIT, 16 kHz). Streaming ONNX (bundled in the repo) is used for timing and as the
    reference; quality runs use the offline PyTorch model, which is causal and numerically equivalent."""

    sr, hop, win = 16000, 256, 512

    def __init__(self, ckpt="model_trained_on_dns3", use_onnx=True):
        import onnxruntime as ort
        self.name = f"gtcrn_{ckpt.split('_')[-1]}"
        self.ckpt = ckpt
        self.use_onnx = use_onnx and ckpt == 'model_trained_on_dns3'
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(GTCRN_ROOT / "stream" / "onnx_models" / "gtcrn.onnx"), so, providers=["CPUExecutionProvider"])
        self._torch_model = None
        self.window = np.sqrt(np.hanning(self.win + 1)[:-1]).astype(np.float32)  # periodic sqrt-hann

    def _tm(self):
        if self._torch_model is None:
            import torch
            sys.path.insert(0, str(GTCRN_ROOT))
            from gtcrn import GTCRN
            m = GTCRN().eval()
            ck = torch.load(GTCRN_ROOT / "checkpoints" / f"{self.ckpt}.tar", map_location="cpu", weights_only=True)
            m.load_state_dict(ck["model"])
            self._torch_model = m
        return self._torch_model

    def stream(self, x):
        """Frame-by-frame ONNX stream (causal STFT, centre=False)."""
        x = np.asarray(x, dtype=np.float32)
        W, H = self.win, self.hop
        conv = np.zeros((2, 1, 16, 16, 33), np.float32)
        tra = np.zeros((2, 3, 1, 1, 16), np.float32)
        inter = np.zeros((2, 1, 33, 16), np.float32)
        nfr = (len(x) - W) // H + 1
        y = np.zeros(len(x) + W, dtype=np.float32)
        times = np.empty(nfr)
        pc = time.perf_counter
        win = self.window
        for k in range(nfr):
            t0 = pc()
            sp = np.fft.rfft(x[k * H:k * H + W] * win)
            ri = np.stack([sp.real, sp.imag], -1).astype(np.float32)[None, :, None, :]
            e, conv, tra, inter = self.sess.run(None, {"mix": ri, "conv_cache": conv, "tra_cache": tra, "inter_cache": inter})
            fr = e[0, :, 0, :]
            y[k * H:k * H + W] += np.fft.irfft(fr[:, 0] + 1j * fr[:, 1], n=W).astype(np.float32) * win
            times[k] = pc() - t0
        return np.concatenate([np.zeros(H, np.float32), y[: len(x) - H]]), times

    def enhance(self, x48):
        if self.use_onnx:
            return super().enhance(x48)
        import torch
        x16 = resample(x48, SR, 16000)
        with torch.no_grad():
            w = torch.hann_window(512).pow(0.5)
            st = torch.view_as_real(torch.stft(torch.from_numpy(x16), 512, 256, 512, w, return_complex=True))[None]
            o = self._tm()(st)[0]
            y = torch.istft(torch.view_as_complex(o.contiguous()), 512, 256, 512, w).numpy()
        return resample(y, 16000, SR)[: len(x48)]
