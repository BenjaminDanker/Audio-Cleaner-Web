"""Hybrid 48 kHz delivery: MossFormerGAN_SE_16K for 0-8 kHz, DeepFilterNet3 above ~7.5 kHz.

Complementary linear-phase FIR crossover (low-pass h, high-pass = delta - h), so lo + hi reconstructs a flat
response and both branches see the same group delay (compensated with 'same'-mode FFT convolution).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.dont_write_bytecode = True
import numpy as np
from scipy.signal import firwin, oaconvolve
from common import OUT, SR, read_mono, resample, subset_ids, write_wav
import soundfile as sf

FC = 7500.0
NTAPS = 1535  # 32 ms at 48 kHz: ~300 Hz transition (Kaiser beta 8)
_h_lp = firwin(NTAPS, FC, window=("kaiser", 8.0), fs=SR).astype(np.float64)


def merge(gan48: np.ndarray, dfn48: np.ndarray) -> np.ndarray:
    n = min(len(gan48), len(dfn48))
    gan48, dfn48 = gan48[:n].astype(np.float64), dfn48[:n].astype(np.float64)
    lo = oaconvolve(gan48, _h_lp, mode="same")
    hi = dfn48 - oaconvolve(dfn48, _h_lp, mode="same")
    return (lo + hi).astype(np.float32)


def hybrid(gan16: np.ndarray, dfn48: np.ndarray) -> np.ndarray:
    gan48 = resample(gan16, 16000, SR)
    if len(gan48) < len(dfn48):
        gan48 = np.pad(gan48, (0, len(dfn48) - len(gan48)))
    return merge(gan48, dfn48)


if __name__ == "__main__":
    src_gan = sys.argv[1] if len(sys.argv) > 1 else "mfgan16"
    out = OUT / "hybrid_mfgan_dfn3"
    out.mkdir(parents=True, exist_ok=True)
    for i in subset_ids(200):
        g, sg = sf.read(str(OUT / src_gan / f"{i}.wav"), dtype="float32")
        d = read_mono(OUT / "dfn3" / f"{i}.wav")
        write_wav(out / f"{i}.wav", hybrid(g, d), SR)
    print("hybrid written:", len(list(out.glob('*.wav'))))
