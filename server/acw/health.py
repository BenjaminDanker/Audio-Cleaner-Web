"""What is actually working, beyond "the web process answers": used by `/api/health?deep=1` (local or admin
only) and by the monitor."""
from __future__ import annotations

import socket
import time

from . import db
from .billing import ledger
from .config import get_settings

# Files the product needs, relative to the models directory / the bin directory.
REQUIRED_MODELS = (
    "DeepFilterNet3_onnx.tar.gz", "captions-en/encoder.int8.onnx", "dpdfnet/dpdfnet8_48khz_hr_batch.onnx",
    "parakeet-v2/encoder-model.onnx", "parakeet-v3/encoder-model.onnx", "silero-vad", "whisper/large-v3-turbo",
    "whisper/small", "whisper/base", "sidon/sidon_fe.onnx",
)
REQUIRED_BINARIES = ("ffmpeg", "ffprobe", "libdf.so")
HEARTBEAT_STALE_S = 180
GPU_MARKER_FRESH_S = 30


def missing_files() -> list[str]:
    s = get_settings()
    out = [f"models/{m}" for m in REQUIRED_MODELS if not (s.models_dir / m).exists()]
    out += [f"bin/{b}" for b in REQUIRED_BINARIES if not (s.bin_dir / b).exists()]
    return out


def stream_worker_ok(timeout: float = 2.0) -> bool:
    """The live engine answers on its socket."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(str(get_settings().stream_worker_socket))
        return True
    except OSError:
        return False
    finally:
        s.close()


def stale_running_jobs(now: int | None = None) -> int:
    now = db.now() if now is None else now
    return db.one("SELECT COUNT(*) AS n FROM jobs WHERE status = 'running' AND heartbeat_at < ?", (now - HEARTBEAT_STALE_S,))["n"]


def gpu_worker_age() -> float | None:
    """Seconds since the GPU worker last said it was idle; None when it has never run here (no GPU worker)."""
    try:
        return time.time() - (get_settings().data_dir / "gpu-worker-idle").stat().st_mtime
    except OSError:
        return None


def deep() -> dict:
    db.one("SELECT 1")
    gpu_age = gpu_worker_age()
    checks = {
        "database": True,
        "streamWorker": stream_worker_ok(),
        "staleRunningJobs": stale_running_jobs(),
        "missingFiles": missing_files(),
        "money": ledger.invariant_problems(),
        # Idle for ages is normal while it works on a job; this only shows the last time it reported in.
        "gpuWorkerSecondsSinceIdle": None if gpu_age is None else round(gpu_age),
    }
    checks["ok"] = bool(
        checks["streamWorker"] and checks["staleRunningJobs"] == 0 and not checks["missingFiles"]
        and not any(checks["money"].values())
    )
    return checks
