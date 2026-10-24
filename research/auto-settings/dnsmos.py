"""DNSMOS P.835 (SIG, BAK, OVRL) with the microsoft/DNS-Challenge sig_bak_ovr.onnx model and its polynomial calibration
(DNSMOS/dnsmos_local.py, MIT). Mean over 1-s hops of 9.01 s segments, as the repo's script."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import librosa
import onnxruntime as ort

MODEL = "/home/silver/audio-cleaner/research/auto-settings/work/DNS-Challenge/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
_sess = None

def _s():
    global _sess
    if _sess is None:
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        _sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    return _sess

def _poly(sig, bak, ovr):
    p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
    p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
    p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
    return p_sig(sig), p_bak(bak), p_ovr(ovr)

def dnsmos16(audio16):
    """audio: float array at 16 kHz"""
    audio = np.asarray(audio16, np.float32)
    L = int(9.01 * 16000)
    while len(audio) < L:
        audio = np.append(audio, audio)
    hops = int(np.floor(len(audio) / 16000) - 9.01) + 1
    raw, pol = [], []
    for i in range(hops):
        seg = audio[i * 16000: int((i + 9.01) * 16000)]
        if len(seg) < L:
            continue
        r = _s().run(None, {"input_1": seg.astype("float32")[None, :]})[0][0]
        raw.append(r); pol.append(_poly(*r))
    raw = np.array(raw); pol = np.array(pol)
    return dict(SIG=float(pol[:, 0].mean()), BAK=float(pol[:, 1].mean()), OVRL=float(pol[:, 2].mean()))
