"""Model adapters. Every offline adapter maps mono 48 kHz float32 -> mono 48 kHz float32
of the same length, with chunking inside, so the benchmark treats them uniformly.

Third-party code is imported from checkouts under ~/audio-cleaner/research/denoise-src
(see fetch_models.sh). Checkpoints are loaded with torch.load(weights_only=True).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np

from common import (DFN3, DFN3_LL, LIBDF, MODELS, SR, SRC, resample)

CV_ROOT = SRC if (SRC / "clearvoice" / "models").exists() and not (SRC / "ClearerVoice-Studio").exists() else SRC / "ClearerVoice-Studio" / "clearvoice"
RE_ROOT = SRC / "resemble-enhance"
GTCRN_ROOT = SRC / "gtcrn"


def _torch():
    import torch
    return torch


def chunk_apply(fn, x: np.ndarray, window: int, stride: int, progress=None) -> np.ndarray:
    """Run `fn(segment)->segment` (same length) over overlapping windows and keep
    the middle of each window; first/last windows keep their outer edge. The final
    window is aligned to the end of the signal so no zero padding is needed."""
    n = len(x)
    if n <= window:
        return fn(x)
    give = (window - stride) // 2
    starts = list(range(0, n - window + 1, stride))
    if starts[-1] + window < n:
        starts.append(n - window)
    out = np.zeros(n, dtype=np.float32)
    written = 0  # first sample of `out` not yet filled
    for k, s in enumerate(starts):
        y = fn(x[s:s + window])
        lo = 0 if k == 0 else s + give  # absolute start of kept region
        hi = n if k == len(starts) - 1 else s + window - give
        lo = max(lo, written)
        if hi > lo:
            out[lo:hi] = y[lo - s:hi - s]
            written = hi
        if progress:
            progress((k + 1) / len(starts))
    return out


# ------------------------------------------------------------------ baseline
class DFN3Clarity:
    """Production chain: DeepFilterNet3 (libDF) [+ clarity chain]."""

    name = "dfn3"
    sr = SR

    def __init__(self, device="cpu", threads=4, clarity=False):
        sys.path.insert(0, "/home/silver/repos/Audio-Cleaner-Web/server")
        from acw.dsp import clarity as _clarity, dfn
        self._dfn, self._clarity = dfn, _clarity
        self.threads, self.use_clarity = threads, clarity
        self.device = "cpu"

    def __call__(self, x):
        y = self._dfn.denoise_array(x, str(LIBDF), str(DFN3), threads=self.threads)
        if self.use_clarity:
            y = self._clarity.process(x[:, None], y[:, None], SR)[:, 0]
        return y


# ------------------------------------------------------- ClearerVoice models
def _cv_args(name: str):
    import yaml
    cfg = yaml.safe_load(open(CV_ROOT / "clearvoice" / "config" / "inference" / f"{name}.yaml"))
    return types.SimpleNamespace(**cfg)


class MossFormer2SE48k:
    """ClearerVoice MossFormer2_SE_48K (55.3M params, Apache-2.0).

    Window logic is the upstream short-input branch of
    decode_one_audio_mossformer2_se_48k (fbank + deltas -> mask on STFT); long
    inputs are chunked by `chunk_apply` (4 s window, 3 s stride, as upstream).
    """

    name = "mossformer2_se_48k"
    sr = SR

    def __init__(self, device="cpu", threads=4, window_s=4.0, stride_s=3.0):
        torch = _torch()
        sys.path.insert(0, str(CV_ROOT))
        from clearvoice.models.mossformer2_se.mossformer2_se_wrapper import MossFormer2_SE_48K
        self.args = _cv_args("MossFormer2_SE_48K")
        self.device = torch.device(device)
        self.model = MossFormer2_SE_48K(self.args).model
        sd = torch.load(MODELS / "mossformer2_se_48k" / "last_best_checkpoint.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(sd)
        self.model.eval().to(self.device)
        self.window, self.stride = int(window_s * SR), int(stride_s * SR)
        self.one_time = int(20 * SR)
        self.threads = threads

    def _window(self, seg: np.ndarray) -> np.ndarray:
        import torchaudio
        from clearvoice.utils.misc import compute_fbank, istft, stft
        torch, a = _torch(), self.args
        audio = torch.from_numpy(seg.astype(np.float32) * 32768.0)
        fb = compute_fbank(audio.unsqueeze(0), a)
        d1 = torchaudio.functional.compute_deltas(fb.T)
        d2 = torchaudio.functional.compute_deltas(d1)
        feats = torch.cat([fb, d1.T, d2.T], dim=1).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            mask = self.model(feats)[-1]
        spec = stft(audio, a)
        masked = spec * mask.permute(2, 1, 0).float().cpu()
        out = istft(masked[:, :, 0] + 1j * masked[:, :, 1], a, len(audio))
        return (out.numpy() / 32768.0).astype(np.float32)

    def __call__(self, x):
        torch = _torch()
        torch.set_num_threads(self.threads)
        x = x.astype(np.float32)
        if len(x) <= self.one_time:  # upstream: whole-utterance decode up to 20 s
            return self._window(x)
        return chunk_apply(self._window, x, self.window, self.stride)


class MossFormerGANSE16k:
    """ClearerVoice MossFormerGAN_SE_16K (generator only). Input/ output 16 kHz;
    this adapter resamples 48k <-> 16k and so is band-limited to 8 kHz."""

    name = "mossformergan_se_16k"
    sr = 16000

    def __init__(self, device="cpu", threads=4, window_s=10.0):
        torch = _torch()
        sys.path.insert(0, str(CV_ROOT))
        from clearvoice.models.mossformer_gan_se.generator import MossFormerGAN_SE_16K
        self.args = _cv_args("MossFormerGAN_SE_16K")
        # upstream: 10 s one-pass window, 7.5 s stride. 10 s needs ~3 GB VRAM; smaller windows fit 4 GB.
        self.args.decode_window = window_s
        self.args.one_time_decode_length = window_s
        self.device = torch.device(device)
        self.model = MossFormerGAN_SE_16K(self.args).model
        ck = torch.load(MODELS / "mossformergan_se_16k" / "last_best_checkpoint.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(ck["model"])
        self.model.eval().to(self.device)
        self.threads = threads

    def enhance16(self, x16: np.ndarray) -> np.ndarray:
        """16 kHz mono in, 16 kHz out, using the upstream long-input decode (10 s windows, 7.5 s stride)."""
        torch = _torch()
        torch.set_num_threads(self.threads)
        from clearvoice.utils.decode import decode_one_audio_mossformergan_se_16k
        y = decode_one_audio_mossformergan_se_16k(self.model, self.device, x16[None].astype(np.float64), self.args)
        return np.asarray(y, dtype=np.float32)[: len(x16)]

    def __call__(self, x48):
        y16 = self.enhance16(resample(x48, SR, 16000))
        return resample(y16, 16000, SR)[: len(x48)]


# -------------------------------------------------------------- Resemble Enhance
def _stub_deepspeed():
    """resemble_enhance imports deepspeed in its training utilities even for
    inference; provide inert stand-ins so deepspeed need not be installed."""
    if "deepspeed" in sys.modules:
        return
    ds, acc, rt = types.ModuleType("deepspeed"), types.ModuleType("deepspeed.accelerator"), types.ModuleType("deepspeed.runtime")
    eng, ut = types.ModuleType("deepspeed.runtime.engine"), types.ModuleType("deepspeed.runtime.utils")
    eng.DeepSpeedEngine = type("DeepSpeedEngine", (), {})
    ds.DeepSpeedConfig = type("DeepSpeedConfig", (), {})
    acc.get_accelerator = lambda: None
    ut.clip_grad_norm_ = lambda *a, **k: None
    ds.accelerator, ds.runtime = acc, rt
    rt.engine, rt.utils = eng, ut
    sys.modules.update({"deepspeed": ds, "deepspeed.accelerator": acc, "deepspeed.runtime": rt,
                        "deepspeed.runtime.engine": eng, "deepspeed.runtime.utils": ut})


class ResembleEnhance:
    """Resemble Enhance (MIT). mode='denoise' = UNet denoiser only (44.1 kHz,
    non-generative); mode='enhance' = denoiser + latent-CFM enhancer + UnivNet
    vocoder (generative, 44.1 kHz). Uses the project's own chunked `inference()`."""

    sr = SR

    def __init__(self, device="cpu", threads=4, mode="denoise", nfe=32, lambd=1.0, tau=0.5, chunk_seconds=30.0):
        torch = _torch()
        _stub_deepspeed()
        sys.path.insert(0, str(RE_ROOT))
        from resemble_enhance.enhancer.enhancer import Enhancer
        from resemble_enhance.enhancer.hparams import HParams
        # NumPy >= 2.5 refuses float(<1-element array>), which the upstream CFM time mapping does
        # on the result of scipy.optimize.fsolve; hand it a 0-d array instead (no repo edit).
        import scipy.optimize as _so
        if not getattr(_so.fsolve, "_acw_patched", False):
            _orig = _so.fsolve

            def _fsolve0d(*a, **k):
                r = _orig(*a, **k)
                return np.asarray(r).reshape(()) if np.size(r) == 1 and not isinstance(r, tuple) else r
            _fsolve0d._acw_patched = True
            _so.fsolve = _fsolve0d
        run_dir = MODELS / "resemble_enhance"
        hp = HParams.load(run_dir)
        enh = Enhancer(hp)
        ck = torch.load(run_dir / "ds" / "G" / "default" / "mp_rank_00_model_states.pt", map_location="cpu", weights_only=True)
        enh.load_state_dict(ck["module"])
        self.device = torch.device(device)
        self.enh = enh.eval().to(self.device)
        self.mode, self.threads, self.chunk_seconds = mode, threads, chunk_seconds
        self.name = f"resemble_{mode}" + (f"_nfe{nfe}" if mode == "enhance" else "")
        if mode == "enhance":
            self.enh.configurate_(nfe=nfe, solver="midpoint", lambd=lambd, tau=tau)

    def __call__(self, x):
        torch = _torch()
        torch.set_num_threads(self.threads)
        from resemble_enhance.inference import inference
        model = self.enh.denoiser if self.mode == "denoise" else self.enh
        # Upstream chunking (hop = chunk - 1 s overlap, at 44.1 kHz) can leave a final chunk of a few hundred
        # samples, which torch.stft's reflect padding (840) rejects ("Padding size should be less than ... [1, 1, 645]").
        # Append just enough zeros that the last chunk is >= 0.2 s; they are trimmed from the output.
        hop44 = int((self.chunk_seconds - 1.0) * 44100)
        n44 = int(np.ceil(len(x) * 44100 / SR))
        last = n44 - hop44 * ((n44 - 1) // hop44)
        pad48 = int(np.ceil((8820 - last + 8) * SR / 44100)) if last < 8820 else 0
        xp = np.concatenate([x.astype(np.float32), np.zeros(pad48, np.float32)])
        with torch.inference_mode():
            y, sr = inference(model=model, dwav=torch.from_numpy(xp), sr=SR,
                              device=self.device, chunk_seconds=self.chunk_seconds)
        y = resample(y.numpy(), sr, SR)
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        return y[: len(x)]


# ------------------------------------------------------------------ registry
def make(name: str, device: str = "cpu", threads: int = 4):
    """Factory used by the benchmark scripts. Returns an object callable on mono 48 kHz audio."""
    from common import DFN3, DFN3_LL
    import live
    if name == "dfn3":
        return DFN3Clarity(device, threads, clarity=False)
    if name == "mf2_48k":
        return MossFormer2SE48k(device, threads)
    if name.startswith("mfgan16"):  # mfgan16 or mfgan16_w6 (6 s windows; fits the 4 GB card)
        w = float(name.split("_w")[1]) if "_w" in name else 10.0
        return MossFormerGANSE16k(device, threads, window_s=w)
    if name == "resemble_denoise":
        return ResembleEnhance(device, threads, mode="denoise", chunk_seconds=10.0)
    if name.startswith("resemble_enhance"):  # resemble_enhance_nfe16 etc.
        nfe = int(name.split("nfe")[1]) if "nfe" in name else 32
        return ResembleEnhance(device, threads, mode="enhance", nfe=nfe, chunk_seconds=4.0)
    if name == "dfn3_ll":
        return live.LiveDFN(DFN3_LL, "dfn3_ll")
    if name == "dfn3_live":
        return live.LiveDFN(DFN3, "dfn3_live")
    if name in ("dpdfnet2_48khz_hr", "dpdfnet8_48khz_hr"):
        return live.LiveDPDFNet(name, 48000)
    if name in ("dpdfnet2", "dpdfnet4", "baseline"):
        return live.LiveDPDFNet(name, 16000)
    if name == "gtcrn_dns3":
        return live.LiveGTCRN("model_trained_on_dns3")
    if name == "gtcrn_vctk":
        return live.LiveGTCRN("model_trained_on_vctk")
    raise KeyError(name)


# Models whose native output rate is 16 kHz (evaluated "as is" at 16 kHz)
NATIVE_16K = {"mfgan16", "mfgan16_w6", "dpdfnet2", "dpdfnet4", "baseline", "gtcrn_dns3", "gtcrn_vctk"}
LIVE = {"dfn3_ll", "dfn3_live", "dpdfnet2_48khz_hr", "dpdfnet8_48khz_hr", "dpdfnet2", "dpdfnet4", "baseline", "gtcrn_dns3", "gtcrn_vctk"}
