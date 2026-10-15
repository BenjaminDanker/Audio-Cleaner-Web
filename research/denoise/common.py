"""Shared helpers for the speech-enhancement benchmark (paths, audio IO, metrics, timing).

Everything here is for INTERNAL evaluation only (DNSMOS weights and the
VoiceBank/DEMAND test set are not shipped in the product).
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from math import gcd
from pathlib import Path

sys.dont_write_bytecode = True  # never write .pyc into the repo or model checkouts

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HOME = Path.home()
RESEARCH = HOME / "audio-cleaner" / "research"
# After the clean-up only RESEARCH/keep/denoise remains (MossFormerGAN weights, its minimal code, the 200-pair eval
# subset, the 10-minute file, DNSMOS onnx). It is used automatically when the fetch_models.sh trees are gone, or
# always when ACW_DENOISE_KEEP_ONLY=1 (used to prove the kept set is self-sufficient).
KEEP = RESEARCH / "keep" / "denoise"
_KO = os.environ.get("ACW_DENOISE_KEEP_ONLY") == "1" or not (RESEARCH / "denoise-models").exists()
DATA = KEEP / "eval" if _KO else RESEARCH / "denoise-data"
MODELS = KEEP / "models" if _KO else RESEARCH / "denoise-models"
SRC = KEEP / "code" if _KO else RESEARCH / "denoise-src"
PROD_MODELS = HOME / "audio-cleaner" / "models"
LIBDF = HOME / "audio-cleaner" / "bin" / "libdf.so"
DFN3 = PROD_MODELS / "DeepFilterNet3_onnx.tar.gz"
DFN3_LL = PROD_MODELS / "DeepFilterNet3_ll_onnx.tar.gz"
ASSETS = HOME / "audio-cleaner" / "src" / "DeepFilterNet" / "assets"
SAMPLES = RESEARCH / "samples"
OUT = RESEARCH / "denoise-out"
REPO_RESEARCH = Path(__file__).resolve().parent
SR = 48000

VB_CLEAN = DATA / "clean" if _KO else DATA / "vbd48_clean"
VB_NOISY = DATA / "noisy" if _KO else DATA / "vbd48_noisy"
LONG_FILE = DATA / "long_600s.wav"
DNSMOS_ONNX = DATA / "dnsmos" / "sig_bak_ovr.onnx" if _KO else SRC / "dnsmos_dl" / "DNSMOS_DNSMOS_sig_bak_ovr.onnx"


def loadavg() -> str:
    """`uptime` style load averages (1/5/15 min) as a compact string."""
    with open("/proc/loadavg") as f:
        a = f.read().split()
    return "/".join(a[:3])


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def read_mono(path, sr: int = SR) -> np.ndarray:
    x, s = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    return resample(x, s, sr)


def write_wav(path, x: np.ndarray, sr: int = SR) -> None:
    sf.write(str(path), np.clip(x, -1, 1), sr, subtype="PCM_16")


def subset_ids(n: int = 200, seed: int = 1234) -> list[str]:
    """Fixed random subset of the 824 VoiceBank test utterances."""
    f = REPO_RESEARCH / f"subset{n}.json"
    if f.exists():
        return json.loads(f.read_text())
    ids = sorted(p.stem for p in VB_NOISY.glob("*.wav"))
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(ids), size=n, replace=False).tolist())
    out = [ids[i] for i in pick]
    f.write_text(json.dumps(out, indent=0))
    return out


# ----------------------------------------------------------------- metrics
def _align(ref: np.ndarray, deg: np.ndarray):
    n = min(len(ref), len(deg))
    return ref[:n], deg[:n]


def si_sdr(ref: np.ndarray, est: np.ndarray, eps: float = 1e-8) -> float:
    ref, est = _align(ref.astype(np.float64), est.astype(np.float64))
    ref = ref - ref.mean()
    est = est - est.mean()
    a = (est @ ref) / (ref @ ref + eps)
    t = a * ref
    return float(10 * np.log10((t @ t + eps) / ((est - t) @ (est - t) + eps)))


def lsd_hf(clean48: np.ndarray, test48: np.ndarray, f_lo: float = 8000.0, nfft: int = 1024, hop: int = 256) -> float:
    """Log-spectral distance (dB, lower is better) over the band above f_lo, on frames where the clean
    reference is active (top 70 % by energy). Sees what the 16 kHz metrics (PESQ-WB, STOI, DNSMOS) cannot."""
    from scipy.signal import stft as _stft
    c, t = _align(clean48, test48)
    _, _, C = _stft(c, SR, nperseg=nfft, noverlap=nfft - hop)
    _, _, T = _stft(t, SR, nperseg=nfft, noverlap=nfft - hop)
    n = min(C.shape[1], T.shape[1])
    C, T = np.abs(C[:, :n]) ** 2, np.abs(T[:, :n]) ** 2
    k0 = int(f_lo / (SR / nfft))
    e = C.sum(0)
    act = e >= np.percentile(e, 30)
    d = 10 * np.log10(C[k0:, act] + 1e-12) - 10 * np.log10(T[k0:, act] + 1e-12)
    return float(np.mean(np.sqrt(np.mean(d ** 2, axis=0))))


def intrusive_metrics(clean48: np.ndarray, test48: np.ndarray) -> dict:
    """PESQ-WB and STOI/ESTOI on 16 kHz resampled audio, SI-SDR at 48 kHz."""
    from pesq import pesq
    from pystoi import stoi

    c48, t48 = _align(clean48, test48)
    c16, t16 = resample(c48, SR, 16000), resample(t48, SR, 16000)
    out = {"si_sdr": si_sdr(c48, t48), "lsd_hf": lsd_hf(c48, t48)}
    try:
        out["pesq"] = float(pesq(16000, c16.astype(np.float64), t16.astype(np.float64), "wb"))
    except Exception:  # pesq raises on silence / too short output
        out["pesq"] = float("nan")
    out["stoi"] = float(stoi(c16, t16, 16000, extended=False))
    out["estoi"] = float(stoi(c16, t16, 16000, extended=True))
    return out


class DNSMOS:
    """DNSMOS P.835 (non-personalized) following Microsoft's dnsmos_local.py.

    Weights: DNS-Challenge repo DNSMOS/DNSMOS/sig_bak_ovr.onnx. Internal
    evaluation only: no weights licence is stated next to the model files.
    """

    INPUT_LENGTH = 9.01
    P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
    P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
    P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])

    def __init__(self):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(DNSMOS_ONNX), so, providers=["CPUExecutionProvider"])

    def __call__(self, x48: np.ndarray) -> dict:
        fs = 16000
        audio = resample(x48, SR, fs)
        n = int(self.INPUT_LENGTH * fs)
        while len(audio) < n:
            audio = np.append(audio, audio)
        hops = int(np.floor(len(audio) / fs) - self.INPUT_LENGTH) + 1
        sig, bak, ovr = [], [], []
        for i in range(hops):
            seg = audio[i * fs: int((i + self.INPUT_LENGTH) * fs)]
            if len(seg) < n:
                continue
            s, b, o = self.sess.run(None, {"input_1": seg.astype(np.float32)[None]})[0][0]
            sig.append(self.P_SIG(s))
            bak.append(self.P_BAK(b))
            ovr.append(self.P_OVR(o))
        return {"sig": float(np.mean(sig)), "bak": float(np.mean(bak)), "ovrl": float(np.mean(ovr))}


def mix_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float, rng) -> np.ndarray:
    """Add noise (looped / random offset) to `clean` at the given SNR."""
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
    off = int(rng.integers(0, len(noise) - len(clean) + 1))
    nz = noise[off: off + len(clean)]
    ps = float(np.mean(clean ** 2)) + 1e-12
    pn = float(np.mean(nz ** 2)) + 1e-12
    g = np.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    return (clean + g * nz).astype(np.float32)
