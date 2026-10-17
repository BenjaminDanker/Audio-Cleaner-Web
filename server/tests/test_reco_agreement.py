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
from scipy.signal import butter, fftconvolve, sosfilt

from acw_common import advice, bwe, pipeline

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


def _talk(seconds: float, seed: int = 0) -> np.ndarray:
    """Close-up words with pauses whose spectrum reaches the top (a glottal pulse train and hiss, 10 ms ramps): the voice
    the room and the low-pass below are put on, so that echoey and muffled are judged on both sides."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    f0 = 120 * (1 + 0.15 * np.sin(2 * np.pi * 0.4 * np.arange(n) / SR + seed))
    pulses = np.zeros(n)
    pulses[np.nonzero(np.diff(np.floor(np.cumsum(f0) / SR)))[0]] = 1.0
    voiced = sosfilt(butter(2, 900, fs=SR, output="sos"), pulses)
    v = voiced / np.std(voiced) + 0.6 * sosfilt(butter(2, 3000, "high", fs=SR, output="sos"), rng.standard_normal(n))
    env = np.zeros(n)
    pos = int(0.2 * SR)
    while pos + SR // 2 < n:
        w = int(rng.uniform(0.2, 0.5) * SR)
        env[pos:pos + w] = np.minimum(1, np.minimum(np.arange(w), np.arange(w)[::-1]) / (SR / 100))
        pos += w + int(rng.uniform(0.3, 0.8) * SR)
    x = v * env
    return 0.3 * x / np.max(np.abs(x)) + 3e-4 * rng.standard_normal(n)


def _room(x: np.ndarray, rt60: float, seed: int = 0) -> np.ndarray:
    m = int(rt60 * 1.5 * SR)
    h = np.random.default_rng(seed + 50).standard_normal(m) * 10 ** (-3 * np.arange(m) / SR / rt60)
    h[0] = 1.0
    y = fftconvolve(x, h)[:len(x)]
    return 0.3 * y / np.max(np.abs(y)) + 3e-4 * np.random.default_rng(seed).standard_normal(len(y))


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
    # what rebuilding the voice is for (advice.rebuild_reason): a close voice, the same in two rooms, and muffled
    for seed in (0, 1):
        talk = _talk(20, seed)
        files[d / f"close{seed}.wav"] = _write(d / f"close{seed}.wav", talk)
        for rt in (0.3, 0.8):
            files[d / f"room{rt}_{seed}.wav"] = _write(d / f"room{rt}_{seed}.wav", _room(talk, rt, seed))
        for fc in (2000, 3000):
            files[d / f"muffled{fc}_{seed}.wav"] = _write(d / f"muffled{fc}_{seed}.wav", sosfilt(butter(2, fc, fs=SR, output="sos"), talk))
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


def test_the_strength_steps_are_the_same_numbers():
    """The desktop's automatic strength (pipeline.strength_for) picks what the website recommends (recommend.js
    strengthFor) and the OBS plugin suggests (input_advice.cpp strength_for): the same cuts and the same strengths."""
    js = (ROOT / "web" / "src" / "recommend.js").read_text()
    cpp = (ROOT / "obs-plugin" / "core" / "src" / "input_advice.cpp").read_text()
    web_db = {m[1]: float(m[2]) for m in re.finditer(r"id: '(\w+)',[^}]*?\bdb: ([\d.]+)", (ROOT / "web" / "src" / "strengths.js").read_text())}
    belows = [b for b, _ in pipeline.STRENGTH_STEPS]
    assert [float(re.search(rf"\b{name}:\s*([\d.]+)", js).group(1)) for name in ("MAX_BELOW", "STRONG_BELOW", "BALANCED_BELOW")] == belows
    assert [float(re.search(rf"\b{name}\s*=\s*([\d.]+)", cpp).group(1)) for name in ("kMaxBelow", "kStrongBelow", "kBalancedBelow")] == belows
    # what each step gives: the website's strength ids (in strengthFor's order) and the plugin's numbers
    picks = re.search(r"function strengthFor\(noiseDb\) \{(.*?)\n\}", js, re.S).group(1)
    assert [web_db[i] for i in re.findall(r"'(\w+)'", picks)] == [db for _, db in pipeline.STRENGTH_STEPS] + [pipeline.GENTLE_DB]
    plugin = re.search(r"int strength_for\(double noise_db\)\s*\{(.*?)\n\}", cpp, re.S).group(1)
    assert [float(n) for n in re.findall(r"(?:return|\?|:)\s*(\d+)\s*;?", plugin)] == [db for _, db in pipeline.STRENGTH_STEPS] + [pipeline.GENTLE_DB]


def test_the_advice_thresholds_are_the_same_numbers():
    """advice.py decides what the website recommends (recommend.js T, analysis.js) with the same numbers."""
    js = (ROOT / "web" / "src" / "recommend.js").read_text()
    analysis = (ROOT / "web" / "src" / "media" / "analysis.js").read_text()

    def t(name):
        return float(re.search(rf"\b{name}:\s*(-?[\d.]+)", js).group(1))

    assert (t("ECHO_DROP_DB"), t("DULL_TILT_DB"), t("PHONE_HZ")) == (advice.ECHO_DROP_DB, advice.DULL_TILT_DB, advice.PHONE_HZ)
    assert float(re.search(r"SPEECH_BAND_DYN_DB\s*=\s*([\d.]+)", analysis).group(1)) == advice.SPEECH_BAND_DYN_DB
    # the speech band's filters and the 60 ms of dropOf
    assert (re.search(r"BAND_HP = biquad\('high', (\d+)\)", analysis).group(1), re.search(r"BAND_LP = biquad\('low', (\d+)\)", analysis).group(1)) == ("150", "3500")
    assert int(re.search(r"function dropOf\(.*?const W = (\d+);", analysis, re.S).group(1)) == advice._DROP_FRAMES


def test_bandwidth_and_the_restore_choice_match(measured):
    for path, js, x in measured:
        py = bwe.effective_bandwidth(x)
        assert js["bandwidthHz"] == pytest.approx(py, abs=1.0), path.name
        assert bwe.pick_model(js["bandwidthHz"]) == bwe.pick_model(py), path.name


def _js_reason(js: dict) -> str | None:
    """recommend.js recommendForFile's rebuild decision, from the browser's figures (the numbers are checked to be the
    same as advice's in test_the_advice_thresholds_are_the_same_numbers)."""
    if not js["hasSpeech"]:
        return None
    if js["dropDb"] is not None and js["dropDb"] < advice.ECHO_DROP_DB:
        return "echoey"
    if js["bandwidthHz"] > advice.PHONE_HZ and js["tiltDb"] is not None and js["tiltDb"] <= advice.DULL_TILT_DB:
        return "muffled"
    return None


def test_speech_echo_and_muffling_are_judged_the_same(measured):
    """advice.py (what the desktop app rebuilds) against the browser's figures (what the website recommends). Every file
    here is short enough to be measured as one stretch: the browser also counts level drops across the joins of the
    stretches of a long file, Python within each stretch only, so long files can differ by a few drops."""
    reasons = set()
    for path, js, x in measured:
        assert js["hasSpeech"] == advice.has_speech(x), path.name
        py_drop = advice.drop_db(x)
        assert (js["dropDb"] is None) == (py_drop is None), (path.name, js["dropDb"], py_drop)
        if py_drop is not None:
            assert js["dropDb"] == pytest.approx(py_drop, abs=1.0), path.name
        py_tilt = bwe.tilt_db(x)
        assert (js["tiltDb"] is None) == (py_tilt is None), path.name
        if py_tilt is not None:
            assert js["tiltDb"] == pytest.approx(py_tilt, abs=0.5), path.name
        reason = advice.rebuild_reason(x)
        assert reason == _js_reason(js), (path.name, js["dropDb"], js["tiltDb"], js["bandwidthHz"])
        reasons.add(reason)
    assert reasons == {None, "echoey", "muffled"}  # the files cover every answer


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
