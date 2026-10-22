"""Every job is made one way, with a timeline (editing.py): a job asked for with whole-file options gets the timeline they stand for
(`timeline_from_options`: each sound step on the whole file, the subtitles as one part), made again for the file's real length once it is
read (`refit_whole_file`); a transcription alone is a timeline of subtitles; whole-file subtitle choices sent with a timeline become a part
of it; an automatic job gets its timeline once it is decided; and an edit of a job whose video our side made makes the video again.
The runner's stages are replaced (tests/test_automatic_settings.py Stages), so no models are needed."""
from __future__ import annotations

import json
import math

import pytest

from acw import automatic, broker, db, editing, jobs
from acw.billing import ledger
from acw.worker import main as worker

from .job_support import (account, api_key, count, files_of, folder, make_done_job, params_of, rebuilt, roomy_disk, row,  # noqa: F401 - fixtures
                          standard, step)
from .test_automatic_settings import Stages, measurement
from .test_editor_api import edit_job, finish_run, level, noise, quote, save_params, tl_of

pytestmark = pytest.mark.usefixtures("roomy_disk")


def funded(signed_in, micros=5_000_000):
    uid, h = signed_in
    ledger.credit(uid, micros, "grant", "t", "g")
    return uid, h


def held(uid):
    return ledger.balance(uid)["held_micros"]


def asked(kind="denoise", tier="standard", transcribe=None, **options):
    return jobs.validate_options(kind, tier, transcribe, options)


def start(client, h, kind="denoise", seconds=60, filename="talk.wav", tier="standard", transcribe=None, options=None, timeline=None):
    body = {"kind": kind, "filename": filename, "sizeBytes": 3000, "durationSeconds": seconds, "options": options or {}}
    if timeline is not None:
        body["timeline"] = timeline
    elif kind != "transcribe":
        body["denoiseTier"] = tier
    if transcribe:
        body["transcribeTier"] = transcribe
    r = client.post("/api/jobs", json=body, headers=h)
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    assert client.put(f"/api/jobs/{jid}/input", content=bytes(3000), headers=h).status_code == 200
    return jid


def take(worker_id="test"):
    job = worker.claim(worker_id)
    assert job is not None
    worker.process(job, worker._settings_extra())
    return job["id"]


def whole(seconds, *layers, version=1, **extra):
    """A timeline with these layers (names or (name, part) pairs) over the whole file."""
    out = {"version": version, "seconds": float(seconds), "layers": {}}
    for layer in layers:
        name, part = layer if isinstance(layer, tuple) else (layer, {})
        out["layers"][name] = [{"start": 0.0, "end": float(seconds), **part}]
    return out


# ------------------------------------------------------------------------------------- the timeline the options stand for

@pytest.mark.parametrize("kind, tier, transcribe, options, want", [
    ("denoise", "standard", None, {}, whole(90, ("noise", {"strength_db": 100.0, "remover": "auto"}), ("leveling", {"loudness_lufs": -16.0}))),
    ("denoise", "standard", None, {"atten_lim_db": 24, "noise": "heavy", "echo_reference_track": True, "loudness_lufs": -20},
     whole(90, "echo", ("noise", {"strength_db": 24.0, "remover": "heavy"}), ("leveling", {"loudness_lufs": -20.0}))),
    ("denoise", "standard", None, {"leveling": False, "noise": "light"}, whole(90, ("noise", {"strength_db": 100.0, "remover": "light"}))),
    ("denoise", "standard", None, {"atten_lim_db": 5}, whole(90, ("noise", {"strength_db": 6.0, "remover": "auto"}), ("leveling", {"loudness_lufs": -16.0}))),
    ("denoise", "rebuild", None, {"leveling": False}, whole(90, "rebuild")),
    ("denoise", "rebuild", None, {}, whole(90, "rebuild", ("leveling", {"loudness_lufs": -16.0}))),
    ("denoise", "off", None, {}, whole(90, ("leveling", {"loudness_lufs": -16.0}))),
    ("transcribe", None, "local", {"language": "en", "translate": ["es"]},
     whole(90, ("subtitles", {"quality": "standard", "language": "en", "translate": ["es"]}), version=2)),
    ("transcribe", None, "premium", {}, whole(90, ("subtitles", {"quality": "accurate", "language": None, "translate": []}), version=2)),
    ("denoise_transcribe", "standard", "local", {"noise": "light", "translate": ["fr", "de"]},
     whole(90, ("noise", {"strength_db": 100.0, "remover": "light"}), ("leveling", {"loudness_lufs": -16.0}),
           ("subtitles", {"quality": "standard", "language": None, "translate": ["fr", "de"]}), version=2)),
])
def test_whole_file_options_are_the_timeline_they_stand_for(kind, tier, transcribe, options, want):
    raw, rest = editing.timeline_from_options(kind, tier, transcribe, asked(kind, tier, transcribe, **options), 90.0)
    assert raw == want
    assert not set(rest) & set(editing.REPLACED_OPTIONS)  # what the timeline holds is not said twice
    assert {"audio_format", "caption_formats", "output_video", "embed_subtitles", "video_out"} <= set(rest)
    assert ("language" in rest) == (transcribe is None)  # (the subtitles' language and translations are the part's own)
    editing.parse_timeline(raw, 90.0)  # (and it is a timeline the editor takes)


def test_a_transcription_alone_has_no_sound_layers_and_the_kinds_are_all_editable():
    raw, _ = editing.timeline_from_options("transcribe", None, "local", asked("transcribe", None, "local"), 60.0)
    assert list(raw["layers"]) == ["subtitles"]
    for kind in ("denoise", "denoise_transcribe", "transcribe"):
        assert editing.editable_kind({"kind": kind}) is True
    assert editing.editable_kind({"kind": "convert"}) is False
    assert editing.QUALITY_OF_TIER == {"local": "standard", "premium": "accurate"}


def test_the_price_of_options_is_the_price_of_their_timeline_each_step_with_its_fixed_amount():
    micros, lines = editing.options_price("denoise", "standard", None, asked(), 600)
    assert micros == standard(600) == step("light", 600) + step("leveling", 600)
    assert [(x["item"], x["micros"]) for x in lines] == [("denoise", step("light", 600)), ("leveling", step("leveling", 600))]
    # (more than the per-minute prices alone: a run's fixed amount counts, as it does for a step on the website)
    assert micros > 10 * (9_000 + 1_000)
    assert editing.options_price("denoise", "rebuild", None, asked("denoise", "rebuild"), 600)[0] == rebuilt(600)
    assert editing.options_price("denoise", "off", None, asked("denoise", "off"), 600)[0] == step("leveling", 600)
    sub = editing.options_price("transcribe", None, "local", asked("transcribe", None, "local", translate=["es"], language="en"), 600)
    assert sub[0] == 10 * 4_000 + 10 * 4_000 and [x["item"] for x in sub[1]] == ["transcribe", "translate"]
    # the same timeline sent as a timeline costs the same
    raw, _ = editing.timeline_from_options("denoise", "standard", None, asked(), 600.0)
    again = editing.parse_timeline(raw, 600.0)
    assert editing.price_timeline(again, 600.0, editing.step_rates())[0] == micros


def test_the_estimate_of_options_is_the_estimate_of_the_timeline_sent_instead(client, signed_in, gpu_up):
    _, h = signed_in
    cases = [("denoise", "standard", None, {}), ("denoise", "standard", None, {"noise": "heavy", "echo_reference_track": True}),
             ("denoise", "rebuild", None, {}), ("denoise", "off", None, {}), ("denoise_transcribe", "standard", "local", {"translate": ["es"]})]
    for kind, tier, transcribe, options in cases:
        raw, _ = editing.timeline_from_options(kind, tier, transcribe, asked(kind, tier, transcribe, **options), 600.0)
        with_options = client.post("/api/jobs/estimate", json={"kind": kind, "denoiseTier": tier, "transcribeTier": transcribe, "durationSeconds": 600,
                                                               "options": options}, headers=h)
        with_timeline = client.post("/api/jobs/estimate", json={"kind": kind, "durationSeconds": 600, "options": {}, "timeline": raw}, headers=h)
        assert with_options.status_code == with_timeline.status_code == 200, (with_options.text, with_timeline.text)
        a, b = with_options.json(), with_timeline.json()
        assert a["estCostMicros"] == b["estCostMicros"] > 0 and a["lines"] == b["lines"], (kind, tier, options)
        assert sum(x["micros"] for x in a["lines"]) == a["estCostMicros"]


def test_creating_a_job_with_options_makes_a_job_with_the_timeline_they_stand_for(client, signed_in):
    uid, h = funded(signed_in)
    options = {"atten_lim_db": 40, "noise": "light", "echo_reference_track": True}
    est = client.post("/api/jobs/estimate", json={"kind": "denoise", "denoiseTier": "standard", "durationSeconds": 600, "options": options}, headers=h).json()
    jid = start(client, h, seconds=600, options=options)
    p = params_of(jid)
    want, _ = editing.timeline_from_options("denoise", "standard", None, asked(**options), 600.0)
    assert p["edit"] == {"timeline": want, "first_run": True} and p["whole_file"] == {"kind": "denoise", "denoise_tier": "standard", "transcribe_tier": None,
                                                                                      "options": asked(**options)}
    assert (p["transcribe_tier"], p["rates"]) == (None, {})  # (the steps are the timeline's, at the prices pinned now)
    assert p["edit_rates"] == editing.step_rates() and "subtitle_rates" not in p
    assert row(jid)["est_cost_micros"] == est["estCostMicros"] == held(uid) == step("echo", 600) + step("light", 600) + step("leveling", 600)
    assert db.one("SELECT amount_micros FROM holds WHERE ref_id = ?", (jid,))["amount_micros"] == est["estCostMicros"]
    shown = client.get(f"/api/jobs/{jid}").json()
    assert shown["kind"] == "denoise" and shown["label"] == "Denoising + Echo removal + Leveling" and shown["edit"]["hasTimeline"] is True
    assert shown["estCostMicros"] == est["estCostMicros"]


@pytest.mark.xfail(strict=True, reason="PRODUCT BUG: the job's options show the defaults of the choices the timeline replaced (atten_lim_db 100, noise auto, "
                                       "leveling true, echo false, loudness -16), not what was asked for")
def test_the_job_shows_the_options_it_was_asked_for(client, signed_in):
    uid, h = funded(signed_in)
    asked_for = {"atten_lim_db": 24, "noise": "heavy", "leveling": False, "echo_reference_track": True, "loudness_lufs": -20}
    jid = start(client, h, options=asked_for)
    shown = client.get(f"/api/jobs/{jid}").json()["options"]
    assert {k: shown[k] for k in asked_for} == {"atten_lim_db": 24.0, "noise": "heavy", "leveling": False, "echo_reference_track": True, "loudness_lufs": -20.0}


def test_a_job_with_options_is_priced_the_same_as_the_same_job_with_its_timeline_when_it_is_created(client, signed_in):
    uid, h = funded(signed_in)
    by_options = start(client, h, seconds=600)
    raw, _ = editing.timeline_from_options("denoise", "standard", None, asked(), 600.0)
    by_timeline = start(client, h, seconds=600, timeline=raw)
    assert row(by_options)["est_cost_micros"] == row(by_timeline)["est_cost_micros"] == standard(600)


# ------------------------------------------------------------------------------------------ made again for the real length

@pytest.mark.parametrize("claimed, real", [(60, 90.0), (120, 45.0), (60, 60.0)])
def test_the_timeline_is_made_again_for_the_length_the_file_really_has_and_that_is_charged(client, signed_in, monkeypatch, claimed, real):
    uid, h = funded(signed_in)
    Stages(monkeypatch, None, duration=real)
    jid = start(client, h, seconds=claimed)
    assert held(uid) == standard(claimed) and params_of(jid)["edit"]["timeline"]["seconds"] == float(claimed)
    seen = {}
    run = worker.run_stage

    def watch(job, d, stage, timeout):
        if stage == "main":  # the probe is over, the file is priced for what it is
            seen["layers"] = json.loads((d / "spec.json").read_text())["editor"]["timeline"]
            seen["held"] = held(uid)
        return run(job, d, stage, timeout)

    monkeypatch.setattr(worker, "run_stage", watch)
    take()
    done = client.get(f"/api/jobs/{jid}").json()
    assert done["status"] == "done" and done["durationSeconds"] == real
    assert seen["layers"]["seconds"] == real and all(p["end"] == real for name in ("noise", "leveling") for p in seen["layers"]["layers"][name])
    assert seen["held"] == max(standard(real), standard(claimed))  # (the hold grows to the real length; it is let go of at the end if it was more)
    assert done["costMicros"] == standard(real) and held(uid) == 0
    assert ledger.balance(uid)["balance_micros"] == 5_000_000 - standard(real)
    assert params_of(jid)["edit"]["timeline"]["seconds"] == real and params_of(jid)["whole_file"]["options"] == asked()  # (what was asked for stays)


def test_the_whole_file_subtitles_part_is_made_again_too(client, signed_in, monkeypatch):
    uid, h = funded(signed_in)
    Stages(monkeypatch, None, duration=100.0)
    jid = start(client, h, kind="denoise_transcribe", seconds=60, transcribe="local", options={"noise": "light", "language": "en"})
    p = params_of(jid)
    assert p["edit"]["timeline"]["layers"]["subtitles"] == [{"start": 0.0, "end": 60.0, "quality": "standard", "language": "en", "translate": []}]
    assert p["subtitle_rates"] == editing.subtitle_rates() and p["transcribe_tier"] is None
    assert held(uid) == standard(60) + math.ceil(60 * 4_000 / 60)
    take()
    done = client.get(f"/api/jobs/{jid}").json()
    assert done["status"] == "done"
    part = params_of(jid)["edit"]["timeline"]["layers"]["subtitles"][0]
    assert (part["end"], part["quality"], part["language"]) == (100.0, "standard", "en")
    assert done["costMicros"] == standard(100) + math.ceil(100 * 4_000 / 60) and held(uid) == 0


# ----------------------------------------------------------------------------------------------- a transcription alone

def test_a_transcription_alone_is_a_timeline_of_subtitles(client, signed_in, monkeypatch):
    uid, h = funded(signed_in)
    est = client.post("/api/jobs/estimate", json={"kind": "transcribe", "transcribeTier": "local", "durationSeconds": 600,
                                                  "options": {"language": "en", "translate": ["es"]}}, headers=h).json()
    assert [x["item"] for x in est["lines"]] == ["transcribe", "translate"] and est["estCostMicros"] == 10 * 4_000 + 10 * 4_000
    jid = start(client, h, kind="transcribe", seconds=600, transcribe="local", options={"language": "en", "translate": ["es"]})
    p = params_of(jid)
    assert p["edit"]["timeline"] == {"version": 2, "seconds": 600.0, "layers": {"subtitles": [
        {"start": 0.0, "end": 600.0, "quality": "standard", "language": "en", "translate": ["es"]}]}}
    assert (p["denoise_tier"], p["transcribe_tier"], p["rates"]) == (None, None, {}) and p["subtitle_rates"] == editing.subtitle_rates()
    assert row(jid)["kind"] == "transcribe" and row(jid)["est_cost_micros"] == est["estCostMicros"] == held(uid)
    shown = client.get(f"/api/jobs/{jid}").json()
    assert shown["edit"]["hasTimeline"] is True and shown["label"] == "Subtitles"
    # the local recogniser has the card's places; the cloud one runs nowhere here
    assert broker.gpu_stages(dict(row(jid))) == ["transcribe"] and broker.gpu_only(dict(row(jid))) is False
    premium = start(client, h, kind="transcribe", seconds=60, transcribe="premium", filename="b.wav")
    assert params_of(premium)["edit"]["timeline"]["layers"]["subtitles"][0]["quality"] == "accurate" and broker.gpu_stages(dict(row(premium))) == []


def test_a_transcription_alone_is_made_again_for_its_real_length_and_charged(client, signed_in, monkeypatch):
    uid, h = funded(signed_in)
    Stages(monkeypatch, None, duration=90.0)
    jid = start(client, h, kind="transcribe", seconds=30, transcribe="local")
    assert held(uid) == math.ceil(30 * 4_000 / 60)
    take()
    done = client.get(f"/api/jobs/{jid}").json()
    assert done["status"] == "done" and params_of(jid)["edit"]["timeline"]["layers"]["subtitles"][0]["end"] == 90.0
    assert done["costMicros"] == math.ceil(90 * 4_000 / 60) and held(uid) == 0


def test_the_timeline_of_a_finished_transcription_is_read_like_any_job_s(client, signed_in):
    uid, h = funded(signed_in)
    jid = make_done_job(uid, kind="transcribe", denoise_tier=None, transcribe_tier="local", seconds=60.0)
    r = client.get(f"/api/jobs/{jid}/timeline")
    assert r.status_code == 200 and list(r.json()["timeline"]["layers"]) == ["subtitles"] and r.json()["timeline"]["version"] == 2
    assert client.get(f"/api/jobs/{jid}").json()["edit"]["hasTimeline"] is True


def test_a_transcription_job_takes_a_timeline_of_subtitles_only(client, signed_in):
    _, h = signed_in
    body = {"kind": "transcribe", "durationSeconds": 600, "options": {}}
    ok = client.post("/api/jobs/estimate", json={**body, "timeline": whole(600, ("subtitles", {"quality": "accurate", "language": None, "translate": []}), version=2)}, headers=h)
    assert ok.status_code == 200 and [x["item"] for x in ok.json()["lines"]] == ["transcribe_accurate"]
    sound = client.post("/api/jobs/estimate", json={**body, "timeline": whole(600, ("noise", {"strength_db": 100.0, "remover": "light"}))}, headers=h)
    assert sound.status_code == 400 and "A transcribe job only has subtitles" in sound.json()["detail"]
    assert client.post("/api/jobs/estimate", json={**body, "timeline": {"version": 1, "seconds": 600, "layers": {}}}, headers=h).status_code == 400
    assert client.post("/api/jobs/estimate", json={"kind": "convert", "durationSeconds": 60, "options": {"video_out": {"format": "mkv"}}, "timeline": whole(60, "echo")},
                       headers=h).status_code == 400


# ---------------------------------------------------------------------------- whole-file subtitle choices sent with a timeline

def test_whole_file_subtitle_choices_sent_with_a_timeline_become_a_part_of_it():
    sent = {"version": 1, "seconds": 600, "layers": {"noise": [{"start": 10, "end": 20, "strength_db": 40, "remover": "light"}]}}
    raw, rest, tier = editing.subtitles_into_timeline(sent, "local", {"language": "en", "translate": ["es"], "audio_format": "wav"}, 600.0)
    assert raw["version"] == 2 and raw["layers"]["noise"] == sent["layers"]["noise"] and tier is None
    assert raw["layers"]["subtitles"] == [{"start": 0.0, "end": 600, "quality": "standard", "language": "en", "translate": ["es"]}]
    assert rest == {"audio_format": "wav"}
    raw, rest, tier = editing.subtitles_into_timeline(sent, "premium", {}, 600.0)
    assert raw["layers"]["subtitles"][0]["quality"] == "accurate" and raw["layers"]["subtitles"][0]["language"] is None and tier is None
    assert "subtitles" not in sent["layers"]  # (the timeline sent is not changed)
    # nothing to fold: no tier, an unknown one, a timeline with subtitle parts of its own, or something that is not a timeline
    for given, tier_ in ((sent, None), (sent, "off"), (sent, "standard"), ("nonsense", "local"), ({"layers": []}, "local"),
                         ({"version": 2, "seconds": 600, "layers": {"subtitles": [{"start": 0, "end": 5, "quality": "free", "language": None, "translate": []}]}}, "local")):
        assert editing.subtitles_into_timeline(given, tier_, {"language": "en"}, 600.0) == (given, {"language": "en"}, tier_)
    # the length of the part is the timeline's own, else the file's
    raw, _, _ = editing.subtitles_into_timeline({"layers": {}}, "local", {}, 123.0)
    assert raw["layers"]["subtitles"][0]["end"] == 123.0


def test_a_timeline_sent_with_a_transcription_tier_gets_the_subtitles_as_a_part_in_the_estimate_and_the_job(client, signed_in):
    uid, h = funded(signed_in)
    tl = {"version": 1, "seconds": 600, "layers": {"noise": [{"start": 0, "end": 120, "strength_db": 40, "remover": "light"}]}}
    plain = client.post("/api/jobs/estimate", json={"kind": "denoise", "durationSeconds": 600, "options": {}, "timeline": tl}, headers=h).json()
    body = {"kind": "denoise_transcribe", "transcribeTier": "local", "durationSeconds": 600, "options": {"language": "en", "translate": ["es"]}, "timeline": tl}
    est = client.post("/api/jobs/estimate", json=body, headers=h)
    assert est.status_code == 200, est.text
    est = est.json()
    assert est["estCostMicros"] == plain["estCostMicros"] + 10 * 4_000 + 10 * 4_000
    assert [x["item"] for x in est["lines"]] == ["denoise", "transcribe", "translate"] and est["timeline"]["layers"]["subtitles"][0]["translate"] == ["es"]
    r = client.post("/api/jobs", json={**body, "filename": "talk.wav", "sizeBytes": 1000}, headers=h)
    assert r.status_code == 200, r.text
    p = params_of(r.json()["id"])
    assert r.json()["estCostMicros"] == est["estCostMicros"] == held(uid)
    assert p["transcribe_tier"] is None and p["edit"]["timeline"]["layers"]["subtitles"] == [
        {"start": 0.0, "end": 600.0, "quality": "standard", "language": "en", "translate": ["es"]}]
    assert "language" not in p["options"] or p["options"]["language"] is None  # (the part holds them)
    assert p["subtitle_rates"] == editing.subtitle_rates()


def test_a_transcription_alone_may_be_sent_as_an_empty_timeline_with_a_tier(client, signed_in):
    _, h = signed_in
    r = client.post("/api/jobs/estimate", json={"kind": "transcribe", "transcribeTier": "local", "durationSeconds": 600, "options": {},
                                                "timeline": {"version": 1, "seconds": 600, "layers": {}}}, headers=h)
    assert r.status_code == 200, r.text
    assert [x["item"] for x in r.json()["lines"]] == ["transcribe"] and list(r.json()["timeline"]["layers"]) == ["subtitles"]


def test_a_timeline_with_subtitle_parts_of_its_own_refuses_a_tier_besides(client, signed_in):
    _, h = signed_in
    tl = whole(600, ("subtitles", {"quality": "standard", "language": None, "translate": []}), version=2)
    r = client.post("/api/jobs/estimate", json={"kind": "denoise_transcribe", "transcribeTier": "local", "durationSeconds": 600, "options": {}, "timeline": tl}, headers=h)
    assert r.status_code == 400 and "transcribeTier" in r.json()["detail"]


# --------------------------------------------------------------------------------------------------- automatic settings

def test_an_automatic_job_is_priced_and_held_by_the_timeline_it_could_become_and_becomes_it(client, signed_in, monkeypatch, gpu_up):
    uid, h = funded(signed_in)
    stats = measurement(strength="balanced", rebuild=False, heavy=False)
    stages = Stages(monkeypatch, stats, duration=60.0)
    r = client.post("/api/jobs", json={"kind": "denoise_transcribe", "denoiseTier": "standard", "transcribeTier": "local", "filename": "talk.wav",
                                       "sizeBytes": 3000, "durationSeconds": 60, "options": {"automatic": True, "language": "en"}}, headers=h)
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    assert client.put(f"/api/jobs/{jid}/input", content=bytes(3000), headers=h).status_code == 200
    subs = math.ceil(60 * 4_000 / 60)
    assert r.json()["estCostMicros"] == rebuilt(60) + subs == held(uid) and r.json()["maximum"] is True  # (the dearest it could be)
    assert params_of(jid)["automatic"]["edit_rates"] == editing.step_rates()  # pinned now
    take()
    p = params_of(jid)
    done = client.get(f"/api/jobs/{jid}").json()
    assert done["status"] == "done" and p["automatic"] == {"fixed": []} and (p["denoise_tier"], p["transcribe_tier"], p["rates"]) == ("standard", None, {})
    layers = p["edit"]["timeline"]["layers"]
    assert [x["remover"] for x in layers["noise"]] == ["light"] and layers["subtitles"] == [{"start": 0.0, "end": 60.0, "quality": "standard", "language": "en", "translate": []}]
    assert done["costMicros"] == standard(60) + subs and held(uid) == 0 and "main" in stages.stages
    assert p["subtitle_rates"] == editing.subtitle_rates()


def test_an_automatic_job_is_decided_at_the_prices_it_was_made_at(client, signed_in, monkeypatch, gpu_up):
    """The rates are pinned with the job when it is made, not read again when it is decided."""
    uid, h = funded(signed_in)
    stages = Stages(monkeypatch, measurement(strength="balanced", rebuild=False, heavy=False), duration=60.0)
    r = client.post("/api/jobs", json={"kind": "denoise", "denoiseTier": "standard", "filename": "talk.wav", "sizeBytes": 3000, "durationSeconds": 60,
                                       "options": {"automatic": True}}, headers=h)
    jid = r.json()["id"]
    client.put(f"/api/jobs/{jid}/input", content=bytes(3000), headers=h)
    p = params_of(jid)
    old = {**p["automatic"]["edit_rates"], "light": 4_000, "leveling": 500}  # the price list of that day
    p["automatic"]["edit_rates"] = old
    save_params(jid, p)
    take()
    after = params_of(jid)
    assert after["edit_rates"] == old
    assert client.get(f"/api/jobs/{jid}").json()["costMicros"] == step("light", 60, old) + step("leveling", 60, old) < standard(60)


def test_the_most_an_automatic_job_can_cost_includes_the_subtitles_part_and_the_estimate_lists_the_dearest_outcome(client, signed_in, gpu_up):
    _, h = signed_in
    e = client.post("/api/jobs/estimate", json={"kind": "denoise_transcribe", "denoiseTier": "standard", "transcribeTier": "premium", "durationSeconds": 600,
                                                "options": {"automatic": True, "translate": ["es"]}}, headers=h).json()
    assert [x["item"] for x in e["lines"]] == ["rebuild", "leveling", "transcribe_accurate", "translate"]
    assert e["estCostMicros"] == rebuilt(600) + 10 * 15_000 + 10 * 4_000 == sum(x["micros"] for x in e["lines"])
    assert [o["estCostMicros"] for o in e["outcomes"]] == [standard(600) + 150_000 + 40_000, e["estCostMicros"]]


# --------------------------------------------------------------------- an edit makes a video our side made again

VIDEO = {"video_out": {"format": "mp4", "size": 720, "source": {"codec": "h264", "width": 1920, "height": 1080, "fps": 30}}}


def video_job(uid, *, seconds=600.0, options=None, **params):
    """A finished job whose video our side made (the whole video was uploaded): its audio and its video result, and the record an edit needs."""
    jid = edit_job(uid, seconds=seconds, ext=".mp4", name="talk.mp4", options=options if options is not None else VIDEO)
    p = params_of(jid)
    p.update(has_video=True, **jobs.convert_params({**p["options"], **(options if options is not None else VIDEO)}, ".mp4"), **params)
    p["options"] = {**p["options"], **(options if options is not None else VIDEO)}
    save_params(jid, p)
    outputs = [{"file": "talk-cleaned.flac", "kind": "audio", "bytes": 10}, {"file": "talk-cleaned.mp4", "kind": "video", "bytes": 20}]
    (folder(jid) / "talk-cleaned.mp4").write_bytes(b"video")
    db.execute("UPDATE jobs SET outputs_json = ? WHERE id = ?", (json.dumps(outputs), jid))
    return jid


def test_a_quote_of_an_edit_of_a_video_made_on_our_side_adds_the_picture_made_again(client, signed_in):
    uid, h = funded(signed_in)
    jid = video_job(uid)
    p = params_of(jid)
    assert p["picture_encode"] is True and p["has_video"] is True
    tl = tl_of(jid, noise=[noise(0, 600, 24)], leveling=[level(0, 600)])
    q = quote(client, h, jid, tl)
    assert q.status_code == 200, q.text
    lines = q.json()["lines"]
    convert = jobs.convert_cost(p, 600)
    assert convert > 0 and lines[-1]["item"] == "convert" and lines[-1]["micros"] == convert
    assert q.json()["estCostMicros"] == sum(x["micros"] for x in lines) == convert + sum(x["micros"] for x in lines[:-1])
    inner = editing.make_quote(row(jid), params_of(jid), editing.parse_timeline(tl, 600.0))
    assert inner["video"] is True and inner["convertMicros"] == convert
    # an unchanged timeline changes nothing: no video is made
    same = editing.make_quote(row(jid), params_of(jid), editing.saved_timeline(row(jid), params_of(jid), 600.0))
    assert same["unchanged"] is True and same["video"] is False and same["convertMicros"] == 0


def test_processing_an_edit_of_such_a_job_holds_the_conversion_and_sends_the_video_to_be_made(client, signed_in):
    uid, h = funded(signed_in)
    jid = video_job(uid)
    tl = tl_of(jid, noise=[noise(0, 600, 24)])
    price = quote(client, h, jid, tl).json()["estCostMicros"]
    r = client.post(f"/api/jobs/{jid}/process", json={"timeline": tl, "maxCostMicros": price}, headers=h)
    assert r.status_code == 200, r.text
    run = params_of(jid)["run"]
    assert run["video"] is True and run["audio_file"] == "talk-cleaned.flac" and run["convert_micros"] == jobs.convert_cost(params_of(jid), 600) > 0
    assert held(uid) == price == run["hold"]
    spec = editing.spec_for(row(jid), params_of(jid))
    assert spec["video"] is True and spec["audio_file"] == "talk-cleaned.flac" and spec["op"] == "process"


def test_an_export_of_a_cached_change_is_free_for_a_sound_only_job_and_costs_the_picture_for_a_video(client, signed_in):
    """A free change of the sound (everything is in the cache) still has the video made again when our side made one."""
    uid, h = funded(signed_in)
    sound = edit_job(uid, seconds=600.0, rendered="empty")
    video = video_job(uid)
    for jid, want in ((sound, 0), (video, jobs.convert_cost(params_of(video), 600))):
        tl = tl_of(jid, leveling=[level(0, 600, -14)])
        q = editing.make_quote(row(jid), params_of(jid), editing.parse_timeline(tl, 600.0))
        assert q["convertMicros"] == want and q["video"] is (jid == video)
    # (the sound-only job: leveling alone is paid for as a step, no picture in it)
    q = quote(client, h, sound, tl_of(sound, leveling=[level(0, 600, -14)])).json()
    assert [x["item"] for x in q["lines"]] == ["leveling"]


def test_a_copied_picture_is_made_again_for_nothing_and_a_sound_asked_for_as_a_video_gets_its_black_picture_again(client, signed_in):
    uid, h = funded(signed_in)
    copy = video_job(uid, options={"video_out": {"format": "mkv", "size": None}})
    assert params_of(copy)["picture_encode"] is False
    q = editing.make_quote(row(copy), params_of(copy), editing.parse_timeline(tl_of(copy, noise=[noise(0, 600, 24)]), 600.0))
    assert q["video"] is True and q["convertMicros"] == 0  # (the picture is copied into the new file with the new sound)
    black = edit_job(uid, seconds=600.0, options=VIDEO)  # a sound asked for as a video: no picture came with it
    p = params_of(black)
    p.update(has_video=False, **jobs.convert_params({**p["options"], **VIDEO}, ".wav"))
    p["options"] = {**p["options"], **VIDEO}
    save_params(black, p)
    q = editing.make_quote(row(black), params_of(black), editing.parse_timeline(tl_of(black, noise=[noise(0, 600, 24)]), 600.0))
    assert q["video"] is True and q["convertMicros"] == 0 and params_of(black)["picture_encode"] is False


def test_the_edit_of_a_job_whose_audio_was_extracted_makes_no_video(client, signed_in):
    """The browser sent only the sound of a video (the picture never left the device): there is no video of ours to make again."""
    uid, h = funded(signed_in)
    jid = edit_job(uid, seconds=600.0, options={"output_video": True})
    p = params_of(jid)
    assert p.get("has_video") in (None, False) and not p["options"].get("video_out")
    tl = tl_of(jid, noise=[noise(0, 600, 24)])
    q = editing.make_quote(row(jid), params_of(jid), editing.parse_timeline(tl, 600.0))
    assert q["video"] is False and q["convertMicros"] == 0
    r = client.post(f"/api/jobs/{jid}/process", json={"timeline": tl}, headers=h)
    assert r.status_code == 200, r.text
    run = params_of(jid)["run"]
    assert "video" not in run and "audio_file" not in run and "convert_micros" not in run
    spec = editing.spec_for(row(jid), params_of(jid))
    assert spec["video"] is False and spec["audio_file"] is None
    # and without output_video there is no video either, whatever the job was
    audio_only = video_job(uid)
    p = params_of(audio_only)
    p["options"]["output_video"] = False
    save_params(audio_only, p)
    assert editing.make_quote(row(audio_only), params_of(audio_only), editing.parse_timeline(tl_of(audio_only, noise=[noise(0, 600, 24)]), 600.0))["video"] is False


def test_the_run_that_made_the_video_is_charged_for_it_and_replaces_the_video_result(client, signed_in):
    uid, h = funded(signed_in)
    jid = video_job(uid)
    tl = tl_of(jid, noise=[noise(0, 600, 24)])
    price = quote(client, h, jid, tl).json()["estCostMicros"]
    assert client.post(f"/api/jobs/{jid}/process", json={"timeline": tl, "maxCostMicros": price}, headers=h).status_code == 200
    convert = params_of(jid)["run"]["convert_micros"]
    before = ledger.balance(uid)["balance_micros"]
    db.execute("UPDATE jobs SET status = 'running', worker_id = 't', started_at = ? WHERE id = ?", (db.now(), jid))
    d = folder(jid)
    (d / "talk-cleaned.flac.run").write_bytes(b"audio")
    (d / "talk-cleaned.mp4.run").write_bytes(b"new video")
    outputs = [{"file": "talk-cleaned.flac.run", "kind": "audio"}, {"file": "talk-cleaned.mp4.run", "kind": "video"}]
    worker._complete_edit(row(jid), d, {"ok": True, "outputs": outputs, "editor": {"rendered": tl, "billed": {}, "resolved": {}}})
    done = row(jid)
    shown = json.loads(done["outputs_json"])
    assert done["status"] == "done" and sorted((o["file"], o["kind"]) for o in shown) == [("talk-cleaned.flac", "audio"), ("talk-cleaned.mp4", "video")]
    assert (d / "talk-cleaned.mp4").read_bytes() == b"new video" and "talk-cleaned.mp4.run" not in files_of(jid)
    steps_charged = ledger.balance(uid)["balance_micros"]
    assert before - steps_charged >= convert  # (the picture made again is part of what the run is charged)
    assert params_of(jid).get("run") is None and params_of(jid)["edit"]["runs"][-1]["op"] == "process"


def test_a_run_that_made_no_video_is_not_charged_for_one(client, signed_in):
    uid, h = funded(signed_in)
    jid = video_job(uid)
    tl = tl_of(jid, noise=[noise(0, 600, 24)])
    assert client.post(f"/api/jobs/{jid}/process", json={"timeline": tl}, headers=h).status_code == 200
    convert = params_of(jid)["run"]["convert_micros"]
    assert convert > 0 and held(uid) == row(jid)["est_cost_micros"] > convert
    before = ledger.balance(uid)["balance_micros"]
    db.execute("UPDATE jobs SET status = 'running', worker_id = 't', started_at = ? WHERE id = ?", (db.now(), jid))
    d = folder(jid)
    (d / "talk-cleaned.flac").write_bytes(b"audio")
    worker._complete_edit(row(jid), d, {"ok": True, "outputs": [{"file": "talk-cleaned.flac", "kind": "audio"}], "editor": {"rendered": tl, "billed": {}, "resolved": {}}})
    assert row(jid)["status"] == "done"
    # nothing was run (all in the cache) and no picture was made: nothing is charged, and the hold is let go
    assert ledger.balance(uid)["balance_micros"] == before and held(uid) == 0
    assert [o["kind"] for o in json.loads(row(jid)["outputs_json"])].count("video") == 0
