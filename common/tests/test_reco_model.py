"""reco_model.py: the trained model of "Automatic best settings" and the decision built on it, against the website's evaluator
(web/src/reco-model.js, web/src/recommend.js): the same trees, the same answers on the same measurements, and the same reasons
for rebuilding the voice (echoey, muffled, phone, distorted, unclear). The website's answers for a few hundred sets of
measurements (real files, made up ones with some of them missing, and ones on both sides of every threshold) are in
fixtures/reco_model_js.json (make_reco_fixtures.py); with node and the website's files at hand the same is also run live."""
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
    assert m["version"] == reco_model.VERSION == 3 and tuple(m["features"]) == analysis.FEATURES
    assert m["strength_classes"] == list(reco_model.STRENGTH_CLASSES)
    assert m["noise"]["num_class"] == 4 and len(m["noise"]["trees"]) % 4 == 0 and m["rebuild"]["num_class"] == 1 and "restore" not in m
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
    outcomes = {"strength": set(), "rebuild": set()}
    for i, row in enumerate(FIXTURE):
        m = reco_model.model_decision(row["stats"])
        js = row["model"]
        assert (m.strength, m.rebuild) == (js["strength"], js["rebuild"]), i
        assert list(m.p_strength) == pytest.approx(js["p"]["strength"], abs=1e-9), i
        assert m.p_rebuild == pytest.approx(js["p"]["rebuild"], abs=1e-9), i
        outcomes["strength"].add(m.strength)
        outcomes["rebuild"].add(m.rebuild)
    # the stored sets reach every answer, so a wrong threshold or a mixed up class cannot hide
    assert outcomes == {"strength": set(reco_model.STRENGTH_CLASSES), "rebuild": {True, False}}


def test_the_decision_agrees_with_the_websites_on_every_stored_set_of_measurements():
    for i, row in enumerate(FIXTURE):
        s, js = row["stats"], row["reco"]
        rec = reco_model.recommend(s)
        if js is None:
            assert rec is None and (s["seconds"] < 2 or s["noiseDb"] is None), i
            continue
        assert (rec.strength, rec.rebuild, rec.no_speech) == (js["strength"], js["rebuild"], js["noSpeech"]), i
        assert rec.heavy_remover == (s["noiseDb"] < pipeline.HEAVY_BELOW_DB), i
        assert (rec.rebuild_reason is not None) == rec.rebuild, i
        assert rec.rebuild_reason == js["rebuildReason"], (i, s)  # the same first reason as the website says
        assert rec.rebuild_reasons == (tuple(js["rebuildWhy"]) if not rec.no_speech else ()), (i, s)  # and every other one it hears
        assert js["rebuildWhy"][:1] == ([rec.rebuild_reason] if rec.rebuild else []), i


def test_the_reasons_agree_with_the_websites_on_every_stored_set_of_measurements_and_with_the_rules():
    """The website names every cause it hears, the app the first: all five reasons and "none" are in the stored sets, and each
    set's reasons are the ones the rules (written out here with their numbers) call for."""
    reached = set()
    for i, row in enumerate(FIXTURE):
        s, js = row["stats"], row["reco"]
        if js is None or js["noSpeech"]:
            continue
        assert js["rebuildWhy"] == rule_reasons(s), (i, s)
        rec = reco_model.recommend(s)
        assert rec.rebuild_reason == (js["rebuildWhy"][0] if js["rebuildWhy"] else None), (i, s)
        reached.add(rec.rebuild_reason)
        assert rec.rebuild == bool(js["rebuildWhy"]) and js["rebuild"] == rec.rebuild, i
    assert reached == {None, "echoey", "muffled", "phone", "distorted", "unclear"}
    # the website adds up the causes: more than one of them is said where more than one is heard
    assert {tuple(r["reco"]["rebuildWhy"]) for r in FIXTURE if r["reco"]} >= {("echoey", "phone"), ("phone", "distorted"), ("muffled", "distorted")}


def test_the_stored_sets_have_every_threshold_on_both_sides():
    edge = [r for r in FIXTURE if r["kind"] == "edge"]
    assert len(edge) >= 200 and {r["kind"] for r in FIXTURE} == {"real", "made-up", "edge"}
    for key, below, at, above in (("bandwidthHz", 6599.9, 6600.0, 6600.1), ("clipFraction", 0.0, 0.002, 0.0020001), ("dropDb", 10.4999, 10.5, 30.0),
                                  ("tiltDb", -19.0001, -19.0, -18.9999)):
        assert {below, at, above} <= {r["stats"][key] for r in edge}, key
    assert any(r["stats"]["tiltDb"] is None for r in edge) and any(r["stats"]["dropDb"] is None for r in edge)


def test_missing_measurements_go_the_way_the_trees_say():
    """None and NaN are the same (the trees send a missing value one way), and hasSpeech counts as 1 or 0."""
    base = dict(next(r["stats"] for r in FIXTURE if r["stats"]["hasSpeech"] and r["stats"]["tiltDb"] is not None and r["stats"]["dropDb"] is not None))
    a = reco_model.model_decision({**base, "tiltDb": None, "dropDb": None, "loudnessLufs": None})
    b = reco_model.model_decision({**base, "tiltDb": math.nan, "dropDb": math.nan, "loudnessLufs": math.nan})
    assert a == b
    v = reco_model.feature_vector({**base, "tiltDb": None})
    assert len(v) == 14 and math.isnan(v[analysis.FEATURES.index("tiltDb")]) and v[analysis.FEATURES.index("hasSpeech")] == 1


# ---- the decision on top of the trees (web/src/recommend.js) ------------------------------------------------------

def rule_reasons(s: dict) -> list[str]:
    """Why the voice is rebuilt, from the rules written out with their numbers (not read from reco_model): every cause heard, in
    the order the website says them (the app's reason is the first); "unclear" (the trees only) when nothing else is heard."""
    phone = s["bandwidthHz"] <= 6600  # the speech stops at 6600 Hz or lower: a phone call or a very small file
    out = []
    if s["dropDb"] is not None and s["dropDb"] < 10.5:
        out.append("echoey")  # the level falls less than 10.5 dB within 60 ms
    if not phone and s["tiltDb"] is not None and s["tiltDb"] <= -19:
        out.append("muffled")  # a phone call is not "muffled": it is a phone call
    if phone:
        out.append("phone")
    if (s["clipFraction"] or 0) > 0.002:
        out.append("distorted")  # more than 0.2 % of the samples are clipped
    if not out and reco_model.model_decision(s).rebuild:
        out.append("unclear")
    return out


def stats(**kw) -> dict:
    """A clean, close voice that the trees leave alone: nothing to rebuild."""
    base = next(r["stats"] for r in FIXTURE if r["reco"] and not r["reco"]["rebuild"]
                and r["stats"]["hasSpeech"] and r["stats"]["seconds"] >= 2 and r["stats"]["dropDb"] is not None
                and r["stats"]["tiltDb"] is not None and r["stats"]["tiltDb"] > -15 and r["stats"]["dropDb"] > 15
                and r["stats"]["bandwidthHz"] > 7000 and not r["stats"]["clipFraction"])
    return {**base, **kw}


def reason_of(s: dict) -> str | None:
    return reco_model.recommend(s).rebuild_reason


def test_the_base_voice_of_these_tests_is_left_alone():
    clean = reco_model.recommend(stats())
    assert (clean.rebuild, clean.rebuild_reason) == (False, None)


def test_each_cause_is_its_own_reason():
    clean = stats()
    assert reason_of({**clean, "dropDb": reco_model.ECHO_DROP_DB - 0.1}) == "echoey"
    assert reason_of({**clean, "tiltDb": reco_model.DULL_TILT_DB, "bandwidthHz": 24000.0}) == "muffled"
    assert reason_of({**clean, "bandwidthHz": reco_model.PHONE_HZ}) == "phone"
    assert reason_of({**clean, "clipFraction": reco_model.CLIP_FRACTION * 1.5}) == "distorted"
    for kw, reason in (({"dropDb": 3.0}, "echoey"), ({"tiltDb": -30.0}, "muffled"), ({"bandwidthHz": 3800.0}, "phone"), ({"clipFraction": 0.02}, "distorted")):
        rec = reco_model.recommend({**clean, **kw})
        assert rec.rebuild and rec.rebuild_reason == reason, kw


def test_echo_is_named_before_muffled_before_phone_before_distortion():
    clean = stats()
    everything = {**clean, "dropDb": 3.0, "tiltDb": -30.0, "bandwidthHz": 24000.0, "clipFraction": 0.02}
    assert reason_of(everything) == "echoey"
    assert reason_of({**everything, "dropDb": None}) == "muffled"
    assert reason_of({**everything, "dropDb": None, "tiltDb": None}) == "distorted"
    assert reason_of({**everything, "dropDb": None, "tiltDb": -30.0, "bandwidthHz": 3800.0}) == "phone"  # not muffled: it is a phone call
    assert reason_of({**everything, "tiltDb": -30.0, "bandwidthHz": 3800.0}) == "echoey"
    assert reason_of({**clean, "dropDb": None, "bandwidthHz": 3800.0, "clipFraction": 0.02}) == "phone"
    # a phone call is never "muffled", whatever its tilt
    for tilt in (None, -5.0, -19.0, -45.0):
        assert reason_of({**clean, "dropDb": None, "bandwidthHz": 3800.0, "tiltDb": tilt, "clipFraction": 0.0}) == "phone"


def test_a_phone_call_is_rebuilt_not_left_to_the_trees():
    """A phone call is rebuilt every time (it is the only fix for missing high sounds)."""
    for bandwidth in (2800.0, 3800.0, 5600.0, reco_model.PHONE_HZ):
        rec = reco_model.recommend({**stats(), "bandwidthHz": bandwidth})
        assert rec.rebuild and rec.rebuild_reason == "phone", bandwidth
    assert reason_of({**stats(), "bandwidthHz": reco_model.PHONE_HZ + 0.1}) != "phone"


@pytest.mark.parametrize("key, value, reason, rebuilt", [
    ("bandwidthHz", 6600.0, "phone", True), ("bandwidthHz", 6600.1, "phone", False),  # at the number: a phone call
    ("clipFraction", 0.002, "distorted", False), ("clipFraction", 0.0020001, "distorted", True),  # more than the number: distorted
    ("dropDb", 10.5, "echoey", False), ("dropDb", 10.4999, "echoey", True),  # below the number: echoey
    ("tiltDb", -19.0, "muffled", True), ("tiltDb", -18.9999, "muffled", False),  # at the number or lower: muffled
])
def test_every_threshold_is_where_the_website_has_it(key, value, reason, rebuilt):
    s = {**stats(), "bandwidthHz": 24000.0, key: value}
    got = reco_model.recommend(s)
    assert (got.rebuild_reason == reason) == rebuilt, (key, value, got.rebuild_reason)
    assert (reason in rule_reasons(s)) == rebuilt


def test_what_is_not_measured_is_not_a_cause():
    clean = stats()
    for gone in ({"clipFraction": None}, {"clipFraction": math.nan}):
        assert reason_of({**clean, **gone}) != "distorted", gone
    no_clip_key = {k: v for k, v in clean.items() if k != "clipFraction"}
    assert reco_model.recommend(no_clip_key).rebuild_reason != "distorted"
    assert reason_of({**clean, "dropDb": None}) != "echoey" and reason_of({**clean, "tiltDb": None}) != "muffled"


def test_a_voice_only_the_trees_want_rebuilt_is_called_unclear():
    row = next(r for r in FIXTURE if r["reco"] and r["reco"]["rebuild"] and r["stats"]["hasSpeech"] and r["stats"]["seconds"] >= 2
               and r["stats"]["noiseDb"] is not None and reco_model.model_decision(r["stats"]).rebuild
               and not [c for c in rule_reasons(r["stats"]) if c != "unclear"])
    rec = reco_model.recommend(row["stats"])
    assert rec.rebuild and rec.rebuild_reason == "unclear" and reco_model.model_decision(row["stats"]).rebuild
    # any cause heard is named instead
    assert reco_model.recommend({**row["stats"], "clipFraction": 0.05}).rebuild_reason == "distorted"
    assert reco_model.recommend({**row["stats"], "dropDb": 1.0}).rebuild_reason == "echoey"


def test_the_words_of_every_reason_follow_it_sounded():
    """engine.py says "the voice was rebuilt because it sounded <words>" and the history "(it sounded <words>)"."""
    reasons = ("echoey", "muffled", "phone", "distorted", "unclear")
    assert tuple(reco_model.REBUILD_WORDS) == reasons
    assert len(set(reco_model.REBUILD_WORDS.values())) == len(reasons)
    # every reason the recommender gives has its words (the stored sets reach all of them)
    given = {reco_model.recommend(r["stats"]).rebuild_reason for r in FIXTURE if r["reco"]} - {None}
    assert given == set(reasons) and given <= set(reco_model.REBUILD_WORDS)
    for reason, words in reco_model.REBUILD_WORDS.items():
        for sentence in (f"it sounded {words}", f"it sounds {words}"):
            assert re.fullmatch(r"it sound(s|ed) [a-z ]+", sentence), sentence
        assert words == words.strip() and words != "" and not words.endswith((".", ","))
    assert reco_model.REBUILD_WORDS["phone"] == "like a phone call"  # not the bare word: "it sounded phone" is not English


def test_nobody_speaking_gets_the_middle_strength_and_nothing_else():
    rec = reco_model.recommend(stats(hasSpeech=False, noiseDb=5.0))
    assert (rec.strength, rec.rebuild, rec.no_speech, rec.rebuild_reason) == ("balanced", False, True, None)
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
        assert (m.strength, m.rebuild) == (j["model"]["strength"], j["model"]["rebuild"]), (i, s)
        r = reco_model.recommend(s)
        if j["reco"] is None:
            assert r is None, i
        else:
            assert (r.strength, r.rebuild) == (j["reco"]["strength"], j["reco"]["rebuild"]), (i, s)
            assert r.rebuild_reason == j["reco"]["rebuildReason"] and j["reco"]["rebuildWhy"][:1] == ([r.rebuild_reason] if r.rebuild else []), (i, s)
            if not r.no_speech:
                assert j["reco"]["rebuildWhy"] == rule_reasons(s), (i, s)


# ---- what the recommendation means for the settings of a job (shared by the desktop app and the API) -----------------

def rows():
    """Every stored set of measurements the website decides on."""
    return [r["stats"] for r in FIXTURE if r["reco"] is not None]


def test_the_four_strengths_are_the_same_everywhere():
    assert tuple(reco_model.STRENGTH_DB) == reco_model.STRENGTH_CLASSES == tuple(reco_model.STRENGTH_NAMES)
    # the steps of the rule are these strengths, so the rule and the model give values from one list
    assert set(reco_model.STRENGTH_DB.values()) == {db for _, db in pipeline.STRENGTH_STEPS} | {pipeline.GENTLE_DB}
    web = ROOT / "web" / "src" / "strengths.js"
    if web.is_file():
        found = re.findall(r"id: '(\w+)', label: '(\w+)', db: (\d+)", web.read_text())
        assert [(i, float(db)) for i, _, db in found] == list(reco_model.STRENGTH_DB.items())
        assert {i: label for i, label, _ in found} == reco_model.STRENGTH_NAMES


def test_choose_turns_every_recommendation_into_the_settings_the_website_shows():
    seen = set()
    for i, s in enumerate(rows()):
        rec = reco_model.recommend(s)
        c = reco_model.choose(rec)
        assert (c.strength, c.atten_lim_db) == (rec.strength, reco_model.STRENGTH_DB[rec.strength]), i
        assert (c.noise, c.heavy) == (("heavy", True) if rec.heavy_remover else ("light", False)), i
        assert (c.rebuild, c.rebuild_wanted, c.rebuild_reason) == (rec.rebuild, rec.rebuild, rec.rebuild_reason), i
        assert c.leveling is True and c.no_speech == rec.no_speech, i
        assert c.noise_db == rec.noise_db
        seen.add((c.rebuild, c.heavy, c.no_speech))
    assert seen >= {(True, False, False), (True, True, False), (False, True, False), (False, False, False), (False, False, True)}


def test_where_the_voice_cannot_be_rebuilt_the_noise_is_removed_instead_and_it_is_said():
    wanted = [reco_model.recommend(s) for s in rows() if reco_model.recommend(s).rebuild]
    assert len(wanted) > 20
    for rec in wanted:
        c = reco_model.choose(rec, rebuild_available=False, rebuild_unavailable="this file is longer than 90 minutes")
        assert (c.rebuild, c.rebuild_wanted, c.rebuild_reason) == (False, True, rec.rebuild_reason)
        assert (c.strength, c.noise) == (rec.strength, "heavy" if rec.heavy_remover else "light")
        assert any("would fix this, but this file is longer than 90 minutes, so Denoising is used instead" in r for r in c.reasons)
        assert not any("is rebuilt instead" in r for r in c.reasons)
    # nothing is said about a rebuild that was not wanted, available or not
    plain = next(r for r in (reco_model.recommend(s) for s in rows()) if not r.rebuild)
    for avail in (True, False):
        c = reco_model.choose(plain, rebuild_available=avail)
        assert not c.rebuild and not c.rebuild_wanted and c.rebuild_reason is None
        assert not any("Rebuilding" in r for r in c.reasons)


def test_the_reasons_say_what_was_heard_and_what_is_done_in_plain_words():
    forbidden = re.compile(r"(?i)deepfilter|dpdfnet|sidon|ap-bwe|polish|\bmodel\b|\btrees?\b|neural|\bai\b|libdf|onnx")
    kinds = set()
    for i, s in enumerate(rows()):
        rec = reco_model.recommend(s)
        for avail in (True, False):
            c = reco_model.choose(rec, rebuild_available=avail)
            assert c.reasons and all(isinstance(r, str) and r.endswith(".") and r == r.strip() for r in c.reasons), i
            assert not [r for r in c.reasons if forbidden.search(r)], (i, c.reasons)
            assert c.reasons[-1] == "Leveling evens out the voice and sets the loudness."
            text = " ".join(c.reasons)
            if c.no_speech:
                assert "hardly hear anyone speaking" in text and "Balanced" in text
                kinds.add("no speech")
            elif c.rebuild:
                assert "Rebuilding the voice fixes this" in text
                for reason in rec.rebuild_reasons:  # every cause that was heard is said, none that was not
                    assert reco_model._PROBLEMS[reason] in text
                for reason in set(reco_model._PROBLEMS) - set(rec.rebuild_reasons):
                    assert reco_model._PROBLEMS[reason] not in text
                kinds.add("rebuilt")
            else:
                assert reco_model.STRENGTH_NAMES[rec.strength] in text or ("clean and quiet" in text and rec.strength == "gentle")
                assert ("stronger noise remover" in text) == rec.heavy_remover
                kinds.add("denoised")
    assert kinds == {"no speech", "rebuilt", "denoised"}


def test_automatic_is_recommend_then_choose_and_says_nothing_where_the_website_does():
    for s in rows()[:60]:
        assert reco_model.automatic(s) == reco_model.choose(reco_model.recommend(s))
        assert reco_model.automatic(s, rebuild_available=False) == reco_model.choose(reco_model.recommend(s), rebuild_available=False)
    assert reco_model.automatic(stats(seconds=1.9)) is None and reco_model.automatic(stats(noiseDb=None)) is None
    assert reco_model.automatic(stats(seconds=1.9), min_seconds=0.0) is not None


def test_the_automatic_strength_of_the_pipeline_is_the_models_where_the_measurements_are_at_hand():
    """pipeline.automatic_choices(noise_db, stats): the model's strength; without stats only the noise figure is known and the
    rule (strength_for) stays. The remover is the rule either way."""
    differs = 0
    for s in rows():
        if s["noiseDb"] is None:
            continue
        rec = reco_model.recommend(s, min_seconds=0.0)
        remover, db = pipeline.automatic_choices(s["noiseDb"], s)
        assert (remover, db) == ("heavy" if s["noiseDb"] < pipeline.HEAVY_BELOW_DB else "light", reco_model.STRENGTH_DB[rec.strength])
        assert pipeline.automatic_choices(s["noiseDb"]) == (remover, pipeline.strength_for(s["noiseDb"]))
        differs += db != pipeline.strength_for(s["noiseDb"])
    assert differs > 20  # the model really decides differently from the rule on many recordings
    # the figure given is the one used (clean_audio measures the recording's own, then measures the rest with it)
    base = stats()
    assert pipeline.automatic_choices(5.0, {**base, "noiseDb": 99.0})[0] == "heavy"
