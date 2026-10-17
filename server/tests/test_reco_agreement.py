"""The website's "Recommended for this file" measures audio in the browser (web/src/media/analysis.js, with the noise remover's
WebAssembly build). Its noise figure, bandwidth and channel layout are meant to be the very numbers the server uses to decide how to
clean a file, so that "heavy noise" is only ever recommended where the server would pick the stronger noise remover. This runs the
browser code under Node (web/tests/reco-cli.mjs) and the server functions on the same files and fails when they drift apart.

Files: noisy mixes made here, plus every WAV in $ACW_RECO_FILES when set (48 kHz; the calibration material).
Needs `node` (18 or newer); skipped without it."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from acw_common import bwe, pipeline

from .conftest import LIVE_LIBDF, LIVE_MODELS

ROOT = Path(__file__).resolve().parents[2]
MODEL = LIVE_MODELS / "DeepFilterNet3_onnx.tar.gz"
CLI = ROOT / "web" / "tests" / "reco-cli.mjs"
SR = 48000

pytestmark = [pytest.mark.skipif(shutil.which("node") is None, reason="needs node"),
              pytest.mark.skipif(not (LIVE_LIBDF.is_file() and MODEL.is_file()), reason="needs libdf and the model")]


def _figure(x):
    return pipeline.noise_figure_db(x, LIVE_LIBDF, MODEL)


def _server_heavy(x):
    return pipeline.choose_noise_profile(x, LIVE_LIBDF, MODEL) == "heavy"


def _write(path: Path, x: np.ndarray) -> np.ndarray:
    """16-bit WAV (what both sides then read back, exactly). Returns the samples as float32, shape (n, channels)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[:, None] if x.ndim == 1 else x
    q = np.clip(np.round(x * 32767), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(q.shape[1])
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(q.tobytes())
    return (q.astype(np.float32) / 32768.0)


def _read(path: Path) -> np.ndarray:
    sf = pytest.importorskip("soundfile")
    x, rate = sf.read(path, dtype="float32", always_2d=True)
    assert rate == SR, path
    return x


def _voice(seconds: float, seed: int = 0) -> np.ndarray:
    """Bursts of a harmonic voice with a few formant-ish bumps, words 0.35 s long with pauses, on a slowly changing pitch."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    f0 = 120 + 25 * np.sin(2 * np.pi * 0.3 * t + seed)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    v = sum(np.sin(k * phase) / k ** 0.8 for k in range(1, 14))
    env = np.zeros(n)
    pos = 0
    while pos < n:
        word = int(rng.uniform(0.2, 0.6) * SR)
        env[pos:pos + word] = np.hanning(min(word, n - pos)) if word else 0
        pos += word + int(rng.uniform(0.1, 0.5) * SR)
    return v * env


def _mix(voice: np.ndarray, snr_db: float, kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 100)
    n = len(voice)
    noise = rng.standard_normal(n)
    if kind == "pink":
        f = np.fft.rfftfreq(n, 1 / SR)
        noise = np.fft.irfft((np.fft.rfft(noise)) / np.sqrt(np.maximum(f, 30.0)), n)
    noise *= np.sqrt(np.mean(voice ** 2) / np.mean(noise ** 2)) * 10 ** (-snr_db / 20)
    y = voice + noise
    return y * (0.5 / np.max(np.abs(y)))


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    d = tmp_path_factory.mktemp("reco")
    files: dict[Path, np.ndarray] = {}
    voice = _voice(20, 1)
    for kind in ("white", "pink"):
        for snr in (0, 5, 10, 15, 20, 25, 30, 45):
            files[d / f"{kind}_{snr}.wav"] = _write(d / f"{kind}_{snr}.wav", _mix(voice, snr, kind, snr))
    clean = _mix(voice, 45, "white", 3)
    # a gated recording: digital silence between the words
    gated = clean.copy()
    gated[np.abs(gated) < 0.004] = 0.0
    files[d / "gated.wav"] = _write(d / "gated.wav", gated)
    # phone quality and a mild low-pass
    for fc in (3400, 5500, 7500):
        sos = butter(14, fc, fs=SR, output="sos")
        files[d / f"lp{fc}.wav"] = _write(d / f"lp{fc}.wav", sosfilt(sos, clean))
    # stereo: two different voices, the same voice on both sides, one side silent
    other = _mix(_voice(20, 2), 20, "pink", 9)
    files[d / "stereo.wav"] = _write(d / "stereo.wav", np.stack([clean, other], axis=1))
    files[d / "dual.wav"] = _write(d / "dual.wav", np.stack([clean, clean], axis=1))
    files[d / "oneside.wav"] = _write(d / "oneside.wav", np.stack([clean, np.zeros_like(clean)], axis=1))
    files[d / "silence.wav"] = _write(d / "silence.wav", np.zeros(5 * SR))
    extra = os.environ.get("ACW_RECO_FILES")
    if extra:
        for p in sorted(Path(extra).glob("*.wav")):
            files[p] = _read(p)
    out = subprocess.run(["node", str(CLI), *map(str, files)], capture_output=True, text=True, timeout=600, cwd=ROOT / "web")
    assert out.returncode == 0, out.stderr[-2000:]
    rows = [json.loads(line) for line in out.stdout.splitlines()]
    assert len(rows) == len(files)
    return [(Path(r["file"]), r, files[Path(r["file"])]) for r in rows]


def test_noise_figure_matches_noise_figure_db(measured):
    """The same noise remover (WebAssembly in the browser, the native library on the server) on the same audio."""
    for path, js, x in measured:
        assert js["noiseDb"] == pytest.approx(_figure(x), abs=0.05), path.name


def test_heavy_is_recommended_exactly_where_the_server_picks_it(measured):
    for path, js, x in measured:
        assert (js["noiseDb"] < pipeline.HEAVY_BELOW_DB) == _server_heavy(x), path.name


def test_the_heavy_cut_is_the_same_number():
    src = (ROOT / "web" / "src" / "media" / "analysis.js").read_text()
    assert float(re.search(r"HEAVY_BELOW_DB\s*=\s*([\d.]+)", src).group(1)) == pipeline.HEAVY_BELOW_DB
    silence = float(re.search(r"SILENCE_DB\s*=\s*(-?[\d.]+)", src).group(1))
    assert silence == pipeline.SILENCE_DB


def test_bandwidth_and_the_restore_choice_match(measured):
    for path, js, x in measured:
        py = bwe.effective_bandwidth(x)
        assert js["bandwidthHz"] == pytest.approx(py, abs=1.0), path.name
        assert bwe.pick_model(js["bandwidthHz"]) == bwe.pick_model(py), path.name


def test_channel_layout_matches_the_servers_plan(measured):
    want = {(0,): "mono", (0, 1): "stereo", (0, 0): "dual-mono", (None, 1): "one-side", (0, None): "one-side"}
    for path, js, x in measured:
        plan = tuple(pipeline.channel_plan(x))
        if x.shape[1] == 2 and plan == (0, 0) and not np.any(np.abs(x) > 1e-4):
            assert js["layout"] == "silent"
        else:
            assert js["layout"] == want[plan], path.name


def test_loudness_is_close_to_bs1770(measured):
    pyln = pytest.importorskip("pyloudnorm")
    meter = pyln.Meter(SR)
    for path, js, x in measured:
        if not np.any(np.abs(x) > 1e-3):
            continue
        ref = meter.integrated_loudness(x.astype(np.float64))
        if np.isfinite(ref):
            assert js["loudnessLufs"] == pytest.approx(ref, abs=0.3), path.name


def test_a_long_file_is_judged_from_the_same_sample(tmp_path):
    """Only ~60 s of a long recording are measured, the same six stretches in the browser and on the server, so the figures
    agree and 'heavy' is only recommended where the server picks it."""
    voice = _voice(240, 5)
    for snr in (3, 12, 18, 21, 28, 40):
        x = _mix(voice, snr, "pink", 11)
        p = tmp_path / f"long_{snr}.wav"
        xs = _write(p, x)
        out = subprocess.run(["node", str(CLI), "--sampled", "--reco", str(p)], capture_output=True, text=True, timeout=600, cwd=ROOT / "web")
        assert out.returncode == 0, out.stderr[-2000:]
        js = json.loads(out.stdout.splitlines()[0])
        assert js["segments"] == 6 and 55 <= js["seconds"] <= 61
        py = _figure(xs)
        assert abs(js["noiseDb"] - py) < 0.1, (snr, js["noiseDb"], py)
        if js["reco"] and js["reco"]["heavy"]:
            assert _server_heavy(xs), snr
