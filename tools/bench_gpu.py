#!/usr/bin/env python3
"""Time one model on the CPU or the GPU and compare the output with the CPU's (docs/research/gpu-2026-10.md).

    LD_LIBRARY_PATH=<venv-gpu CUDA libs> PYTHONPATH=common ~/audio-cleaner/venv-gpu/bin/python tools/bench_gpu.py \\
        dpdfnet|sidon|parakeet --device cuda --wav noisy.wav --seconds 150 [--save out.npy] [--compare cpu_out.npy]

dpdfnet and sidon take a 48 kHz wav, parakeet a 16 kHz one. It prints wall time, process CPU time, the card's peak
memory over the whole run (nvidia-smi, whole card: nothing else may use it), and the SNR against `--compare`.
Check `nvidia-smi` first: a Parakeet job in the GPU worker holds 3.3 GB."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))


class GpuMonitor:
    """Peak memory.used of the card while the `with` block runs, over its value at the start (MiB)."""

    def __init__(self, every: float = 0.2):
        self.every, self.peak, self.base, self._stop = every, 0, 0, threading.Event()

    @staticmethod
    def used() -> int:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        return int(out.stdout.split()[0]) if out.returncode == 0 and out.stdout.strip() else 0

    def __enter__(self):
        self.base = self.peak = self.used()

        def run():
            while not self._stop.is_set():
                self.peak = max(self.peak, self.used())
                self._stop.wait(self.every)

        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    @property
    def peak_mb(self) -> int:
        return self.peak - self.base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["dpdfnet", "sidon", "parakeet"])
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--seconds", type=float, default=150)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--models", default=str(Path.home() / "audio-cleaner" / "models"))
    ap.add_argument("--save")
    ap.add_argument("--compare")
    a = ap.parse_args()
    import soundfile as sf

    x, sr = sf.read(a.wav, dtype="float32")
    x = np.ascontiguousarray(x[: int(a.seconds * sr)])
    models = Path(a.models)
    kw = {"device": a.device} if a.device != "cpu" else {}
    if a.model == "dpdfnet":
        from acw_common import dpdfnet

        run = lambda y: dpdfnet.denoise_array(y, threads=a.threads, **kw)  # noqa: E731
    elif a.model == "sidon":
        from acw_common import restore

        run = lambda y: restore.restore(y, models / "sidon", threads=a.threads, **kw)  # noqa: E731
    else:
        from acw_common import transcribe

        def run(y):
            segs = transcribe.transcribe_parakeet(y, models / "parakeet-v2", models / "silero-vad", device=a.device, threads=a.threads)
            print(" ".join(s.text for s in segs)[:300])
            return np.zeros(1)
    with GpuMonitor() as mon:
        run(x[: 3 * sr])  # loads the model
        t0, c0 = time.time(), time.process_time()
        y = run(x)
        wall, cpu = time.time() - t0, time.process_time() - c0
    secs = len(x) / sr
    line = (f"{a.model} {a.device}: {secs:.0f} s of audio in {wall:.1f} s = {secs / wall:.2f}x real time, "
            f"{cpu:.1f} CPU-s ({cpu / secs:.3f} per audio second), card peak +{mon.peak_mb} MiB")
    if a.compare and a.model != "parakeet":
        ref = np.load(a.compare)
        d = ref.astype(np.float64) - y
        line += f", max abs diff {np.abs(d).max():.2e}, SNR vs {Path(a.compare).name} {10 * np.log10(np.sum(ref.astype(np.float64) ** 2) / max(np.sum(d ** 2), 1e-30)):.1f} dB"
    print(line)
    if a.save and a.model != "parakeet":
        np.save(a.save, y)


if __name__ == "__main__":
    main()
