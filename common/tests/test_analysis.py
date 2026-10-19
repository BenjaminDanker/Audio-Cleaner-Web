"""analysis.py: the measurements of the website (web/src/media/analysis.js) in Python, which the trained model of "Automatic
best settings" needs to the last digit. Made-up recordings (analysis_signals.py) are measured here and, in fixtures/analysis_js.json,
by the website's own code (make_reco_fixtures.py wrote it; with node and the website's files at hand the same is also run live)."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis_signals as sig  # noqa: E402
from acw_common import analysis, dfn, pipeline, reco_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = json.loads((Path(__file__).with_name("fixtures") / "analysis_js.json").read_text())
HOME = Path.home() / "audio-cleaner"
LIB = Path(os.environ.get("ACW_LIBDF", HOME / "bin" / dfn.LIB_NAME))
MODEL = Path(os.environ.get("ACW_DFN_MODEL", HOME / "models" / dfn.MODEL_FILE))
needs_dfn = pytest.mark.skipif(not (LIB.is_file() and MODEL.is_file()), reason="libDF build/model not available")
CLI = ROOT / "web" / "tests" / "reco-cli.mjs"
needs_js = pytest.mark.skipif(
    shutil.which("node") is None or not CLI.is_file() or not (HOME / "wasm" / "df_bg.wasm").is_file() or not MODEL.is_file(),
    reason="needs node, the website's code and its noise remover files")

# How close each measurement has to be: the port follows the browser step by step, so they agree to rounding (1e-12); these
# only leave room for the made-up recordings coming out of another scipy build a few ulps apart.
TOLERANCE = {"bandwidthHz": 1.0, "silentFraction": 0.002, "speechFraction": 0.005, "clipFraction": 1e-5, "peak": 1e-6}
DB = 0.02
MEASURED = [k for k in analysis.FEATURES if k != "noiseDb"]


@pytest.fixture(scope="module")
def recordings() -> dict[str, np.ndarray]:
    return sig.signals()


def assert_same(py: dict, js: dict, where: str) -> None:
    for k in MEASURED:
        a, b = js[k], py[k]
        if a is None:
            assert b is None or (isinstance(b, float) and math.isnan(b)), (where, k, a, b)
        elif k == "hasSpeech":
            assert bool(a) == bool(b), (where, k, a, b)
        elif isinstance(a, float) and math.isnan(a):
            assert math.isnan(b), (where, k, a, b)
        else:
            assert b is not None and not math.isnan(b), (where, k, a, b)
            assert b == pytest.approx(a, abs=TOLERANCE.get(k, DB)), (where, k, a, b)


@pytest.mark.parametrize("name", sorted(FIXTURE))
def test_the_measurements_are_the_websites(name, recordings):
    """Stored values of the browser's Analyser.result() for the same recordings: all 13 measurements that do not need the
    noise remover, whole short files and the six stretches of a long one."""
    js = FIXTURE[name]["stats"]
    py = analysis.measure_features(recordings[name], js["noiseDb"])
    assert py["seconds"] == pytest.approx(js["seconds"], abs=1e-9)
    assert_same(py, js, name)
    assert py["noiseDb"] == js["noiseDb"]


@pytest.mark.parametrize("name", sorted(FIXTURE))
def test_the_decisions_on_them_are_the_websites(name, recordings):
    js = FIXTURE[name]
    stats = analysis.measure_features(recordings[name], js["stats"]["noiseDb"])
    rec = reco_model.recommend(stats)
    if js["reco"] is None:
        assert rec is None
        return
    assert (rec.strength, rec.rebuild, rec.restore_highs, rec.no_speech) == (
        js["reco"]["strength"], js["reco"]["rebuild"], js["reco"]["restoreHighs"], js["reco"]["noSpeech"]), name
    assert rec.rebuild_reason == js["reco"]["rebuildReason"], name  # why the voice is rebuilt: the same first reason
    assert js["reco"]["rebuildWhy"][:1] == ([rec.rebuild_reason] if rec.rebuild else []), name


def test_the_recordings_cover_what_the_decisions_tell_apart():
    decided = [FIXTURE[n]["reco"] for n in FIXTURE if FIXTURE[n]["reco"]]
    assert {d["rebuild"] for d in decided} == {True, False} and {d["restoreHighs"] for d in decided} == {True, False}
    assert any(d["noSpeech"] for d in decided) and len({d["strength"] for d in decided}) >= 2
    # ... and every reason for rebuilding the voice (the echoey, muffled, phone and too loud recordings of analysis_signals.py), or none
    assert {d["rebuildReason"] for d in decided} == {None, "echoey", "muffled", "phone", "distorted", "unclear"}
    assert any(FIXTURE[n]["stats"]["dropDb"] is None for n in FIXTURE) and any(FIXTURE[n]["stats"]["dropDb"] is not None for n in FIXTURE)
    assert FIXTURE["long"]["stats"]["segments"] == 6 and FIXTURE["close"]["stats"]["segments"] == 1  # the browser's own count
    assert FIXTURE["long"]["stats"]["seconds"] == pytest.approx(60.0, abs=0.01) and FIXTURE["close"]["stats"]["seconds"] == 8.0


@needs_dfn
@pytest.mark.parametrize("name", ["close", "noisy", "stereo", "long"])
def test_the_noise_figure_is_the_websites(name, recordings):
    """The same noise remover (WebAssembly in the browser, the native library here) over the same stretches."""
    py = analysis.measure(recordings[name], LIB, MODEL)
    assert py["noiseDb"] == pytest.approx(FIXTURE[name]["stats"]["noiseDb"], abs=0.1), name
    assert py["noiseDb"] == pytest.approx(pipeline.noise_figure_db(recordings[name], LIB, MODEL), abs=1e-9)


@needs_js
def test_the_measurements_are_the_websites_run_live(recordings):
    """The website's code run now (node) on a few of the recordings, instead of the stored values."""
    names = ["echoey", "phone", "gated", "stereo", "long"]
    with tempfile.TemporaryDirectory() as d:
        paths = {}
        for n in names:
            paths[n] = Path(d) / f"{n}.wav"
            sig.write_wav(paths[n], recordings[n])
        out = subprocess.run(["node", str(CLI), "--sampled", *map(str, paths.values())], capture_output=True, text=True,
                             timeout=900, cwd=ROOT / "web")
    assert out.returncode == 0, out.stderr[-2000:]
    rows = {Path(json.loads(line)["file"]).stem: json.loads(line) for line in out.stdout.splitlines()}
    for n in names:
        assert_same(analysis.measure_features(recordings[n], rows[n]["noiseDb"]), rows[n], n)


# ---- what does not need the website -------------------------------------------------------------------------------

def test_what_is_measured(recordings):
    s = analysis.measure_features(recordings["close"], 30.0)
    assert set(s) == {*analysis.FEATURES, "seconds"} and s["hasSpeech"] is True and s["noiseDb"] == 30.0
    assert s["seconds"] == 8.0 and 0 < s["speechFraction"] < 1 and s["bandwidthHz"] == analysis.RATE / 2
    assert analysis.measure_features(recordings["close"])["noiseDb"] is None  # not measured: the noise remover did not run


def test_nobody_speaking_and_nothing_to_hear(recordings):
    assert analysis.measure_features(recordings["hum"], 40.0)["hasSpeech"] is False
    s = analysis.measure_features(recordings["silence"], 99.0)
    assert s["hasSpeech"] is False and s["silentFraction"] == 1.0 and math.isnan(s["floorDb"]) and s["peak"] == 0.0
    assert s["dropDb"] is None and s["loudnessLufs"] is None and s["bandwidthHz"] == analysis.RATE / 2 and s["tiltDb"] is None
    empty = analysis.measure_features(np.zeros(0, np.float32), None)
    assert empty["seconds"] == 0 and empty["silentFraction"] == 1.0 and empty["clipFraction"] == 0.0


def test_clipping_and_peak(recordings):
    s = analysis.measure_features(recordings["clipped"], 20.0)
    assert 0.01 < s["clipFraction"] < 0.5 and s["peak"] > 0.999
    assert analysis.measure_features(recordings["close"], 20.0)["clipFraction"] == 0.0
    # runs of at least three samples at the top count, shorter ones do not
    x = np.zeros(100, np.float32)
    x[10:12] = 1.0
    x[20:25] = -1.0
    x[40:43] = 0.9995
    assert analysis._clipped(x) == 5 + 3


def test_a_long_recording_is_measured_on_six_stretches_and_read_only_there(tmp_path):
    n = 30 * 60 * analysis.RATE  # half an hour in a (sparse) file: only the stretches are ever read
    big = np.memmap(tmp_path / "long.f32", dtype=np.float32, mode="w+", shape=(n, 2))
    big[:analysis.RATE] = 0.1
    s = analysis.measure_features(big, 25.0)
    assert s["seconds"] == pytest.approx(60.0) and len(pipeline.measure_windows(n)) == 6
    assert s["peak"] <= 0.1 + 1e-6


def test_mono_stereo_and_more_channels_are_the_browsers(recordings):
    mono = recordings["close"]
    a = analysis.measure_features(mono, 1.0)
    b = analysis.measure_features(mono[:, None], 1.0)
    assert a == b
    # two channels that are the same: the mix is the channel, the levels agree (the loudness counts both)
    st = analysis.measure_features(recordings["dual_mono"], 1.0)
    for k in ("floorDb", "levelDb", "peakDb", "bandwidthHz", "tiltDb", "dropDb", "peak"):
        assert st[k] == pytest.approx(a[k]), k
    assert st["loudnessLufs"] == pytest.approx(a["loudnessLufs"] + 10 * math.log10(2), abs=1e-6)
    # more than two channels are mixed down to one, as the browser does
    three = np.stack([mono, mono, mono], axis=1)
    assert analysis.measure_features(three, 1.0)["levelDb"] == pytest.approx(a["levelDb"], abs=1e-3)


def test_percentile_is_numpys():
    rng = np.random.default_rng(0)
    for n in (1, 2, 10, 11, 1000, 4097):
        v = np.sort(rng.normal(size=n))
        for q in (0, 10, 50, 60, 90, 95, 100):
            assert analysis.percentile(v, q) == pytest.approx(float(np.percentile(v, q)), abs=1e-12)
    assert math.isnan(analysis.percentile(np.zeros(0), 50))


def test_loudness_is_bs1770(recordings):
    pyln = pytest.importorskip("pyloudnorm")
    x = recordings["close"].astype(np.float64)
    ref = pyln.Meter(analysis.RATE).integrated_loudness(x)
    assert analysis.measure_features(recordings["close"])["loudnessLufs"] == pytest.approx(ref, abs=0.3)


def test_the_features_are_the_models_in_its_order():
    assert analysis.FEATURES == tuple(reco_model.load()["features"]) and len(analysis.FEATURES) == 14
