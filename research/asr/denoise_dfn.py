#!/usr/bin/env python
"""Run DeepFilterNet3 (production dsp module, libdf.so) over eval/noisy5_48k/*.wav -> eval/noisy5_dfn3_48k/*.wav.
Run with the *production* interpreter (read-only use; only stdlib + numpy are used here):
  ~/audio-cleaner/venv/bin/python -I denoise_dfn.py
Then:  ~/audio-cleaner/research/venv/bin/python prep_data.py dfn_out
"""
import sys, wave
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/silver/repos/Audio-Cleaner-Web/server")
from acw.dsp import dfn  # noqa: E402

LIB = "/home/silver/audio-cleaner/bin/libdf.so"
MODEL = "/home/silver/audio-cleaner/models/DeepFilterNet3_onnx.tar.gz"
EVAL = Path.home() / "audio-cleaner/research/data/eval"
src, dst = EVAL / "noisy5_48k", EVAL / "noisy5_dfn3_48k"
dst.mkdir(parents=True, exist_ok=True)
for p in sorted(src.glob("*.wav")):
    with wave.open(str(p)) as w:
        assert w.getframerate() == 48000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    y = dfn.denoise_array(x, LIB, MODEL, threads=2)
    y = np.clip(y, -1.0, 1.0)
    with wave.open(str(dst / p.name), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(48000)
        w.writeframes((y * 32767.0).astype(np.int16).tobytes())
print("denoised", len(list(dst.glob("*.wav"))), "files")
