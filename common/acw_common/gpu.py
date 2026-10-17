"""Settings every CUDA user in this process shares (the GPU worker's runner: DPDFNet, Sidon, AP-BWE, Parakeet).

onnxruntime-gpu (Maxwell build, docs/research/gpu-2026-10.md) is the only CUDA runtime here; its CUDA libraries come from
the pip wheels in the worker's venv, on LD_LIBRARY_PATH.
"""
from __future__ import annotations

import ctypes
import threading

_lock = threading.Lock()
_done = False

_cudaDeviceScheduleBlockingSync = 4
_cudaErrorSetOnActiveProcess = 708


def prepare_cuda() -> None:
    """Make the host thread sleep while it waits for the card instead of spinning. Must run before the first CUDA
    session of the process (the flag is read when the device's context is made; later calls are harmless no-ops).

    A GPU stage waits for a kernel or a graph a few hundred times a second; by default the waiting thread spins, which
    keeps one CPU core at 100 % for the whole job (DPDFNet on 150 s: 39 CPU-s; asleep: 5.3 CPU-s, wall time +3 %)."""
    global _done
    with _lock:
        if _done:
            return
        try:
            rt = ctypes.CDLL("libcudart.so.12")
        except OSError as e:
            raise RuntimeError(f"CUDA runtime library not found (LD_LIBRARY_PATH): {e}") from e
        rc = rt.cudaSetDeviceFlags(_cudaDeviceScheduleBlockingSync)
        if rc not in (0, _cudaErrorSetOnActiveProcess):
            raise RuntimeError(f"cudaSetDeviceFlags failed: {rc}")
        _done = True
