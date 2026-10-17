"""The GPU paths of the models (docs/research/gpu-2026-10.md).

Without a usable CUDA, "cuda" must raise, not quietly run on the CPU. The tests that need the card run only with
ACW_TEST_GPU=1 under the GPU venv's onnxruntime-gpu (and its CUDA libraries on LD_LIBRARY_PATH): they take the card
for a few seconds, so never while the GPU worker is busy. They check the output against the CPU path.
Run: ACW_TEST_GPU=1 PYTHONPATH=<venv-gpu site-packages>:common pytest common/tests/test_gpu.py"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acw_common import bwe, dpdfnet, restore  # noqa: E402

MODELS = Path(os.environ.get("ACW_MODELS_DIR", Path.home() / "audio-cleaner" / "models"))
SR = 48_000


def have_card() -> bool:
    if os.environ.get("ACW_TEST_GPU") != "1":
        return False
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    return "CUDAExecutionProvider" in ort.get_available_providers()


CARD = pytest.mark.skipif(not have_card(), reason="needs ACW_TEST_GPU=1 and onnxruntime-gpu")


def speech(seconds, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f0 = 130 + 25 * np.sin(2 * np.pi * 0.8 * t)
    voice = sum(np.sin(k * 2 * np.pi * np.cumsum(f0) / SR) / k for k in range(1, 14))
    gate = np.convolve((np.sin(2 * np.pi * 1.3 * t) > 0.0).astype(np.float64), np.ones(480) / 480, mode="same")
    return (0.2 * voice * gate + 0.02 * rng.standard_normal(len(t))).astype(np.float32)


def snr_db(a, b):
    return 10 * np.log10(np.sum(a.astype(np.float64) ** 2) / max(np.sum((a.astype(np.float64) - b) ** 2), 1e-30))


def test_no_cuda_means_an_error_not_a_slow_cpu_run():
    """Where onnxruntime has no CUDA (or no CUDA libraries are on the path) every GPU entry point raises."""
    if have_card():
        pytest.skip("this machine has a usable card")
    x = speech(3)
    if (MODELS / "ap-bwe" / "8kto48k.onnx").is_file():
        with pytest.raises(RuntimeError, match="CUDA"):
            bwe.extend(x, MODELS / "ap-bwe", model="8k", device="cuda")
    if restore.available(MODELS / "sidon"):
        with pytest.raises(RuntimeError, match="CUDA"):
            restore.restore(x, MODELS / "sidon", device="cuda")
    batch = dpdfnet.default_model_path().with_name(dpdfnet.BATCH_MODEL_FILE)
    if batch.is_file():
        with pytest.raises(RuntimeError, match="CUDA"):
            dpdfnet.denoise_array(x, device="cuda")


def test_unknown_devices_are_refused():
    x = speech(1)
    with pytest.raises(ValueError):
        bwe.extend(x, MODELS / "ap-bwe", model="8k", device="tpu")
    with pytest.raises(ValueError):
        restore.restore(x, MODELS / "sidon", device="tpu")


@CARD
def test_dpdfnet_on_the_card_equals_the_cpu():
    if not dpdfnet.default_model_path().with_name(dpdfnet.BATCH_MODEL_FILE).is_file():
        pytest.skip("batched DPDFNet model not installed")
    x = speech(24)
    kw = dict(threads=4, segment_seconds=10, overlap_seconds=1.5)
    cpu = dpdfnet.denoise_array(x, **kw)
    gpu = dpdfnet.denoise_array(x, device="cuda", **kw)
    assert snr_db(cpu, gpu) > 80, snr_db(cpu, gpu)
    # a second call reuses the CUDA graphs and starts from a clean state
    again = dpdfnet.denoise_array(x, device="cuda", **kw)
    assert np.array_equal(again, gpu)
    # fewer rows than the first call used, in a graph of another size, then the first size again
    short = dpdfnet.denoise_array(x[: 12 * SR], device="cuda", **kw)
    assert snr_db(dpdfnet.denoise_array(x[: 12 * SR], **kw), short) > 80
    assert np.array_equal(dpdfnet.denoise_array(x, device="cuda", **kw), gpu)
    dpdfnet.unload()


@CARD
def test_band_extension_on_the_card_equals_the_cpu():
    if not (MODELS / "ap-bwe" / "8kto48k.onnx").is_file():
        pytest.skip("AP-BWE models not installed")
    import scipy.signal as ss

    x = ss.sosfilt(ss.butter(10, 3800, "low", fs=SR, output="sos"), speech(8)).astype(np.float32)
    cpu = bwe.extend(x, MODELS / "ap-bwe", model="8k")
    gpu = bwe.extend(x, MODELS / "ap-bwe", model="8k", device="cuda")
    assert snr_db(cpu, gpu) > 80, snr_db(cpu, gpu)
    bwe.unload()


@CARD
def test_voice_rebuild_on_the_card_equals_the_cpu():
    if not restore.available(MODELS / "sidon"):
        pytest.skip("Sidon models not installed")
    x = speech(8)
    cpu = restore.restore(x, MODELS / "sidon", threads=2)
    gpu = restore.restore(x, MODELS / "sidon", device="cuda")
    assert snr_db(cpu, gpu) > 70, snr_db(cpu, gpu)
    restore.unload()
