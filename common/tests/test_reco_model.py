"""reco_model.py: the trained model of "Automatic best settings" and the decision built on it, against the website's evaluator
(web/src/reco-model.js, web/src/recommend.js): the same trees, the same answers on the same measurements. The website's answers
for a few hundred sets of measurements (real files and made up ones, some of them missing) are in fixtures/reco_model_js.json
(make_reco_fixtures.py); with node and the website's files at hand the same is also run live."""
from __future__ import annotations

import json
import math
import re
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from acw_common import analysis, pipeline, reco_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = json.loads((Path(__file__).with_name("fixtures") / "reco_model_js.json").read_text())
WEB_DATA = ROOT / "web" / "src" / "reco-model-data.js"
needs_web = pytest.mark.skipif(shutil.which("node") is None or not WEB_DATA.is_file(), reason="needs node and the website's code")


def test_the_trees_are_the_websites():
    if not WEB_DATA.is_file():
        pytest.skip("the website's code is not here")
    text = WEB_DATA.read_text()
    web = json.loads(re.search(r"export default (\{.*\});\s*$", text, re.S).group(1))
    assert reco_model.load() == web


def test_the_model_file_is_the_one_this_code_reads():
    m = reco_model.load()
    assert m["version"] == reco_model.VERSION == 2 and tuple(m["features"]) == analysis.FEATURES
    assert m["strength_classes"] == list(reco_model.STRENGTH_CLASSES)
    assert m["noise"]["num_class"] == 4 and len(m["noise"]["trees"]) % 4 == 0 and m["rebuild"]["num_class"] == m["restore"]["num_class"] == 1
    assert reco_model.MODEL_FILE.is_file() and reco_model.MODEL_FILE.parent == Path(analysis.__file__).parent


def test_a_model_that_does_not_fit_is_refused(tmp_path):
    bad = dict(reco_model.load(), version=1)
    path = tmp_path / "m.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="does not match"):
        reco_model.load(path)
    path.write_text(json.dumps(dict(reco_model.load(), features=list(reversed(analysis.FEATURES)))))
    with pytest.raises(ValueError, match="does not match"):
        reco_model.load(path)


def test_the_evaluator_agrees_with_the_websites_on_every_stored_set_of_measurements():
    assert len(FIXTURE) >= 400
    outcomes = {"strength": set(), "rebuild": set(), "restore": set()}
    for i, row in enumerate(FIXTURE):
        m = reco_model.model_decision(row["stats"])
        js = row["model"]
        assert (m.strength, m.rebuild, m.restore_highs) == (js["strength"], js["rebuild"], js["restoreHighs"]), i
        assert list(m.p_strength) == pytest.approx(js["p"]["strength"], abs=1e-9), i
        assert m.p_rebuild == pytest.approx(js["p"]["rebuild"], abs=1e-9) and m.p_restore_highs == pytest.approx(js["p"]["restoreHighs"], abs=1e-9), i
        outcomes["strength"].add(m.strength)
        outcomes["rebuild"].add(m.rebuild)
        outcomes["restore"].add(m.restore_highs)
    # the stored sets reach every answer, so a wrong threshold or a mixed up class cannot hide
    assert outcomes == {"strength": set(reco_model.STRENGTH_CLASSES), "rebuild": {True, False}, "restore": {True, False}}


def test_the_decision_agrees_with_the_websites_on_every_stored_set_of_measurements():
    for i, row in enumerate(FIXTURE):
        s, js = row["stats"], row["reco"]
        rec = reco_model.recommend(s)
        if js is None:
            assert rec is None and (s["seconds"] < 2 or s["noiseDb"] is None), i
            continue
        assert (rec.strength, rec.rebuild, rec.restore_highs, rec.no_speech) == (js["strength"], js["rebuild"], js["restoreHighs"], js["noSpeech"]), i
        assert rec.heavy_remover == (s["noiseDb"] < pipeline.HEAVY_BELOW_DB), i
        assert (rec.rebuild_reason is not None) == rec.rebuild, i


def test_missing_measurements_go_the_way_the_trees_say():
    """None and NaN are the same (the trees send a missing value one way), and hasSpeech counts as 1 or 0."""
    base = dict(next(r["stats"] for r in FIXTURE if r["stats"]["hasSpeech"] and r["stats"]["tiltDb"] is not None and r["stats"]["dropDb"] is not None))
    a = reco_model.model_decision({**base, "tiltDb": None, "dropDb": None, "loudnessLufs": None})
    b = reco_model.model_decision({**base, "tiltDb": math.nan, "dropDb": math.nan, "loudnessLufs": math.nan})
    assert a == b
    v = reco_model.feature_vector({**base, "tiltDb": None})
    assert len(v) == 14 and math.isnan(v[analysis.FEATURES.index("tiltDb")]) and v[analysis.FEATURES.index("hasSpeech")] == 1


# ---- the decision on top of the trees (web/src/recommend.js) ------------------------------------------------------

def stats(**kw) -> dict:
    """A clean, close voice that the trees leave alone."""
    base = next(r["stats"] for r in FIXTURE if r["reco"] and not r["reco"]["rebuild"] and not r["reco"]["restoreHighs"]
                and r["stats"]["hasSpeech"] and r["stats"]["seconds"] >= 2 and r["stats"]["dropDb"] is not None
                and r["stats"]["tiltDb"] is not None and r["stats"]["tiltDb"] > -15 and r["stats"]["dropDb"] > 15
                and r["stats"]["bandwidthHz"] > 7000)
    return {**base, **kw}


def test_echo_and_a_muffled_voice_are_rebuilt_whatever_the_trees_say():
    clean = stats()
    echo = reco_model.recommend({**clean, "dropDb": reco_model.ECHO_DROP_DB - 0.1})
    assert echo.rebuild and echo.rebuild_reason == "echoey"
    assert not reco_model.recommend({**clean, "dropDb": None}).rebuild_reason == "echoey"
    dull = reco_model.recommend({**clean, "dropDb": None, "tiltDb": reco_model.DULL_TILT_DB, "bandwidthHz": 24000.0})
    assert dull.rebuild and dull.rebuild_reason == "muffled"
    # echo is named before muffled
    both = reco_model.recommend({**clean, "dropDb": 3.0, "tiltDb": -30.0, "bandwidthHz": 24000.0})
    assert both.rebuild_reason == "echoey"
    # a phone call is not muffled in this sense (restoring its highs is the fix): only the trees can rebuild it
    phone = {**clean, "dropDb": None, "tiltDb": -40.0, "bandwidthHz": reco_model.PHONE_HZ}
    rec = reco_model.recommend(phone)
    assert rec.rebuild_reason in (None, "unclear")
    assert (rec.rebuild_reason == "unclear") == reco_model.model_decision(phone).rebuild


def test_a_voice_only_the_trees_want_rebuilt_is_called_unclear():
    row = next(r for r in FIXTURE if r["reco"] and r["reco"]["rebuild"] and r["stats"]["hasSpeech"] and r["stats"]["seconds"] >= 2
               and r["stats"]["noiseDb"] is not None and (r["stats"]["dropDb"] is None or r["stats"]["dropDb"] >= reco_model.ECHO_DROP_DB)
               and (r["stats"]["bandwidthHz"] <= reco_model.PHONE_HZ or r["stats"]["tiltDb"] is None
                    or r["stats"]["tiltDb"] > reco_model.DULL_TILT_DB))
    rec = reco_model.recommend(row["stats"])
    assert rec.rebuild and rec.rebuild_reason == "unclear" and reco_model.model_decision(row["stats"]).rebuild


def test_nobody_speaking_gets_the_middle_strength_and_nothing_else():
    rec = reco_model.recommend(stats(hasSpeech=False, noiseDb=5.0))
    assert (rec.strength, rec.rebuild, rec.restore_highs, rec.no_speech, rec.rebuild_reason) == ("balanced", False, False, True, None)
    assert rec.heavy_remover  # the stronger noise remover is still the rule on the noise figure


def test_the_heavy_noise_remover_is_the_rule_not_the_model():
    assert reco_model.recommend(stats(noiseDb=pipeline.HEAVY_BELOW_DB - 0.01)).heavy_remover
    assert not reco_model.recommend(stats(noiseDb=pipeline.HEAVY_BELOW_DB)).heavy_remover
    assert not reco_model.recommend(stats(noiseDb=40.0)).heavy_remover


def test_no_decision_where_the_website_makes_none():
    assert reco_model.recommend(stats(seconds=1.9)) is None and reco_model.recommend(stats(noiseDb=None)) is None
    assert reco_model.recommend({}) is None
    assert reco_model.recommend(stats(seconds=1.9), min_seconds=0.0) is not None  # the desktop app decides every file
    assert reco_model.recommend(stats(seconds=2.0)) is not None


# ---- against the website's code, run now ---------------------------------------------------------------------------

@needs_web
def test_the_evaluator_agrees_with_the_websites_run_live():
    import make_reco_fixtures as fx

    rows = fx.random_stats(1500, seed=11)
    js = fx.js_decide(rows)
    for i, (s, j) in enumerate(zip(rows, js)):
        m = reco_model.model_decision(s)
        assert (m.strength, m.rebuild, m.restore_highs) == (j["model"]["strength"], j["model"]["rebuild"], j["model"]["restoreHighs"]), (i, s)
        r = reco_model.recommend(s)
        if j["reco"] is None:
            assert r is None, i
        else:
            assert (r.strength, r.rebuild, r.restore_highs) == (j["reco"]["strength"], j["reco"]["rebuild"], j["reco"]["restoreHighs"]), (i, s)
