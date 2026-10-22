"""Automatic settings for API jobs (`options.automatic: true`, docs/api.md): the server measures the upload and decides like the
website's "Automatic best settings", with the one shared function (acw_common/reco_model.py).

The measurement is stubbed here (the runner's `analyse` stage returns stored sets of measurements, real ones from the website's
code, whose decisions are known), so every branch of the decision is reached without models: what is held for the maximum, the
hold shrinking to what was decided, the charge being what ran, the voice rebuild going to the card or falling back to Denoising
and saying why, explicit fields winning, the API key's ownership and spending cap, and the decided job being a job made with the timeline
the settings stand for (priced, held and charged like any other: each step's fixed amount of a run included). One test (native pieces
needed) runs the real sandboxed stage on a real file."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from acw import automatic, broker, db, jobs
from acw.billing import ledger, pricing
from acw.worker import main as worker
from acw_common import reco_model

from .job_support import api_key, count, files_of, other_browser, params_of, rebuilt, roomy_disk, row, standard, step  # noqa: F401 - fixtures
from .test_job_echo import NATIVE
from .test_job_pipeline import HOME, SAMPLE, _run_worker_once

pytestmark = pytest.mark.usefixtures("roomy_disk")

FIXTURE = json.loads((Path(__file__).resolve().parents[2] / "common" / "tests" / "fixtures" / "reco_model_js.json").read_text())


def per(micros_per_minute, seconds):
    return math.ceil(micros_per_minute * math.ceil(seconds) / 60)


def measurement(**want) -> dict:
    """A real set of measurements whose decision has these values: any of strength, rebuild, reason (echoey / muffled / phone /
    distorted / unclear / None), heavy (noise figure below the heavy cut)."""
    for r in FIXTURE:
        s, js = r["stats"], r["reco"]
        if not js or js["noSpeech"] or s["noiseDb"] is None:
            continue
        rec = reco_model.recommend(s, min_seconds=0.0)
        got = {"strength": rec.strength, "rebuild": rec.rebuild, "reason": rec.rebuild_reason, "heavy": rec.heavy_remover}
        if all(got[k] == v for k, v in want.items()):
            return dict(s)
    raise LookupError(f"no stored measurements with {want}")


def create(client, h, seconds=60, tier="standard", kind="denoise", **options):
    return client.post("/api/jobs", json={"kind": kind, "denoiseTier": tier, "filename": "talk.wav", "sizeBytes": 3000,
                                          "durationSeconds": seconds, "options": {"automatic": True, **options}}, headers=h)


def estimate(client, h, seconds=600, tier="standard", kind="denoise", expect=200, transcribe=None, **options):
    r = client.post("/api/jobs/estimate", json={"kind": kind, "denoiseTier": tier, "transcribeTier": transcribe, "durationSeconds": seconds,
                                                "options": {"automatic": True, **options}}, headers=h)
    assert r.status_code == expect, r.text
    return r.json()


def start(client, h, uid=None, **kw) -> str:
    """An automatic job, created and uploaded (queued); returns its id."""
    r = create(client, h, **kw)
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    assert client.put(f"/api/jobs/{jid}/input", content=bytes(3000), headers=h).status_code == 200
    return jid


class Stages:
    """The sandboxed runner's stages, replaced: a probe that reports `duration`, a measurement that reports `stats`, and a main
    stage that writes the result. `calls` records (stage, spec, worker) in order."""

    def __init__(self, monkeypatch, stats, duration=60.0, channels=1):
        self.stats, self.duration, self.channels, self.calls = stats, duration, channels, []
        monkeypatch.setattr(worker, "run_stage", self)

    def __call__(self, job, d, stage, timeout):
        spec = json.loads((d / "spec.json").read_text())
        self.calls.append((stage, spec, job.get("worker_id")))
        if stage == "probe":
            return {"ok": True, "duration": self.duration, "channels": self.channels}
        if stage == "analyse":
            return {"ok": True, "stats": self.stats}
        assert stage == "main"
        name = f"{spec['output_stem']}-cleaned.flac"
        (d / name).write_bytes(b"cleaned")
        return {"ok": True, "outputs": [{"file": name, "kind": "audio"}], "notice": None}

    @property
    def stages(self):
        return [c[0] for c in self.calls]

    def spec(self, stage):
        return next(c[1] for c in self.calls if c[0] == stage)


def take(worker_id="test"):
    """A worker takes the next queued job and processes it; returns the job's id."""
    job = worker.claim(worker_id)
    assert job is not None
    worker.process(job, worker._settings_extra())
    return job["id"]


@pytest.fixture
def as_card(monkeypatch):
    """The next worker is the card's: it takes the jobs the CPU workers leave to it."""
    monkeypatch.setattr(worker, "GPU_MODE", True)
    broker.mark_gpu_worker("alive")


def held(uid):
    return ledger.balance(uid)["held_micros"]


# ------------------------------------------------------------------------------------------------ the request

def test_the_option_is_a_boolean_for_noise_removal_with_the_standard_tier(client, signed_in, gpu_up):
    _, h = signed_in
    for bad in ("yes", 1, None, "true"):
        r = client.post("/api/jobs/estimate", json={"kind": "denoise", "denoiseTier": "standard", "durationSeconds": 60,
                                                    "options": {"automatic": bad}}, headers=h)
        assert r.status_code == 400 and "automatic must be true or false" in r.json()["detail"], bad
    for tier in ("rebuild", "off"):
        r = client.post("/api/jobs/estimate", json={"kind": "denoise", "denoiseTier": tier, "durationSeconds": 60,
                                                    "options": {"automatic": True}}, headers=h)
        assert r.status_code == 400 and "need denoiseTier standard" in r.json()["detail"], tier
    r = client.post("/api/jobs/estimate", json={"kind": "transcribe", "transcribeTier": "local", "durationSeconds": 60,
                                                "options": {"automatic": True}}, headers=h)
    assert r.status_code == 400 and "noise removal" in r.json()["detail"]
    assert count("jobs") == 0 and count("holds") == 0
    assert jobs.validate_options("denoise", "standard", None, {})["automatic"] is False  # off unless asked for
    off = client.post("/api/jobs/estimate", json={"kind": "denoise", "denoiseTier": "standard", "durationSeconds": 60,
                                                  "options": {"automatic": False}}, headers=h).json()
    assert "maximum" not in off and "outcomes" not in off  # an ordinary price
    with_subtitles = estimate(client, h, kind="denoise_transcribe", transcribe="local")
    assert [x["item"] for x in with_subtitles["lines"]][-1] == "transcribe"


def test_what_the_request_gives_itself_is_what_wins():
    assert automatic.fixed_fields({}) == []
    assert automatic.fixed_fields({"audio_format": "mp3", "loudness_lufs": -14, "caption_formats": ["srt"]}) == []  # not decided options
    assert automatic.fixed_fields({"noise": "auto"}) == []  # that is the automatic choice
    assert automatic.fixed_fields({"noise": "heavy", "atten_lim_db": 24, "leveling": True}) == ["atten_lim_db", "noise", "leveling"]
    assert automatic.fixed_fields({"restore_highs": True}) == []  # (a step that does not exist any more: not an option)
    opts = jobs.validate_options("denoise", "standard", None, {"automatic": True})
    # a strength or a noise type say "Denoising": no voice rebuild is ever considered then
    for given, blocked in (({}, False), ({"leveling": False}, False), ({"atten_lim_db": 100}, True),
                           ({"noise": "light"}, True), ({"noise": "auto"}, False), ({"restore_highs": True}, False)):
        full = jobs.validate_options("denoise", "standard", None, {"automatic": True, **given})
        fixed = automatic.fixed_fields(given)
        assert (automatic.rebuild_blocker(full, fixed) is not None) == blocked, given
        assert [c["tier"] for c in automatic.candidates("denoise", None, full, fixed, 0)] == (["standard"] if blocked else ["standard", "rebuild"])
    assert automatic.rebuild_blocker({**opts, "echo_reference_track": True}, []) == "voice rebuilding is not done together with echo removal"
    assert automatic.rebuild_blocker(opts, [], seconds=90 * 60) is None
    assert automatic.rebuild_blocker(opts, [], seconds=90 * 60 + 1) == "voice rebuilding takes files up to 90 minutes"
    assert automatic.rebuild_blocker(opts, [], seconds=60, card_up=False) == "voice rebuilding is paused right now"


# --------------------------------------------------------------------------------------- the price is the most it can be

def test_the_estimate_is_the_most_the_decision_can_cost_and_says_so(client, signed_in):
    _, h = signed_in
    e = estimate(client, h, 600)
    cheap = standard(600)
    rebuild = rebuilt(600)  # the dearer outcome: the voice rebuilt, with Leveling
    assert rebuild > cheap
    assert e["estCostMicros"] == rebuild == sum(x["micros"] for x in e["lines"]) and e["maximum"] is True
    assert [x["item"] for x in e["lines"]] == ["rebuild", "leveling"] and e["lines"][0]["label"] == "Voice rebuild" and e["fixed"] == []
    assert [x["micros"] for x in e["lines"]] == [step("rebuild", 600), step("leveling", 600)]  # (the timeline's steps, each with its fixed amount of a run)
    assert e["outcomes"] == [
        {"label": "Denoising + Leveling", "denoiseTier": "standard", "estCostMicros": cheap},
        {"label": "Voice rebuild + Leveling", "denoiseTier": "rebuild", "estCostMicros": rebuild}]
    assert e["maxSeconds"] == 4 * 3600 and e["options"]["automatic"] is True
    # an ordinary estimate says nothing of this
    plain = client.post("/api/jobs/estimate", json={"kind": "denoise", "denoiseTier": "standard", "durationSeconds": 600}, headers=h).json()
    assert plain["estCostMicros"] == cheap and "maximum" not in plain


def test_the_maximum_is_the_dearest_outcome_that_is_possible(client, signed_in):
    """A strength or a noise type given: no rebuild is possible, the maximum is Denoising. Leveling fixed off: the rebuild alone. A
    file over 90 minutes: no rebuild outcome."""
    _, h = signed_in
    e = estimate(client, h, 600, atten_lim_db=24)
    assert [o["denoiseTier"] for o in e["outcomes"]] == ["standard"] and e["estCostMicros"] == standard(600)
    assert e["fixed"] == ["atten_lim_db"] and e["lines"][0]["label"] == "Denoising"
    e = estimate(client, h, 600, noise="heavy")
    assert [o["denoiseTier"] for o in e["outcomes"]] == ["standard"] and e["fixed"] == ["noise"]
    e = estimate(client, h, 600, leveling=False)
    assert e["estCostMicros"] == step("rebuild", 600) and [x["item"] for x in e["lines"]] == ["rebuild"] and e["fixed"] == ["leveling"]
    assert [o["estCostMicros"] for o in e["outcomes"]] == [step("light", 600), step("rebuild", 600)]
    e = estimate(client, h, 600, noise="light", leveling=False)
    assert e["estCostMicros"] == step("light", 600) and e["outcomes"] == [{"label": "Denoising", "denoiseTier": "standard", "estCostMicros": step("light", 600)}]
    long = estimate(client, h, 91 * 60)  # too long to rebuild: only Denoising is possible
    assert [o["denoiseTier"] for o in long["outcomes"]] == ["standard"] and long["estCostMicros"] == standard(91 * 60)
    edge = estimate(client, h, 90 * 60)  # exactly 90 minutes can still be rebuilt
    assert [o["denoiseTier"] for o in edge["outcomes"]] == ["standard", "rebuild"] and edge["estCostMicros"] == rebuilt(90 * 60)
    # an old client's restore_highs is an unknown option: nothing changes
    assert estimate(client, h, 600, restore_highs=True)["estCostMicros"] == rebuilt(600)


def test_subtitles_and_translations_are_part_of_every_outcome(client, signed_in):
    _, h = signed_in
    e = estimate(client, h, 600, kind="denoise_transcribe", transcribe="local", language="en", translate=["es"])
    extra = 10 * 4_000 + 10 * 4_000  # the subtitles are one part over the whole file: Standard transcription and one translation
    assert [x["item"] for x in e["lines"]] == ["rebuild", "leveling", "transcribe", "translate"]
    assert e["estCostMicros"] == rebuilt(600) + extra
    assert [o["estCostMicros"] for o in e["outcomes"]] == [standard(600) + extra, e["estCostMicros"]]


def test_the_hold_is_the_estimate_and_the_job_says_it_is_a_maximum(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    for options in ({}, {"atten_lim_db": 40, "noise": "heavy"}, {"leveling": False}):
        e = estimate(client, h, 95.2, **options)
        made = create(client, h, 95.2, **options)
        assert made.status_code == 200, made.text
        assert made.json()["estCostMicros"] == e["estCostMicros"] > 0 and made.json()["maximum"] is True
        jid = made.json()["id"]
        hold = db.one("SELECT * FROM holds WHERE ref_id = ?", (jid,))
        assert hold["amount_micros"] == e["estCostMicros"] and hold["status"] == "active"
        shown = client.get(f"/api/jobs/{jid}").json()
        assert shown["estCostMicros"] == e["estCostMicros"] and shown["maximum"] is True and shown["chosen"] is None
        assert shown["label"] == "Automatic settings" and shown["options"]["automatic"] is True
        assert params_of(jid)["automatic"]["fixed"] == automatic.fixed_fields(options)
        assert params_of(jid)["denoise_tier"] == "standard"
        client.post(f"/api/jobs/{jid}/cancel", headers=h)
    assert held(uid) == 0


def test_an_account_that_cannot_cover_the_maximum_is_refused_before_anything_is_held(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 300_000, "grant", "t", "g")  # a 10 minute file: 310,000 at most (the voice rebuilt, with Leveling)
    e = estimate(client, h, 600)
    assert e["affordable"] is False and e["availableMicros"] == 300_000
    refused = create(client, h, 600)
    assert refused.status_code == 402 and count("jobs") == 0 and held(uid) == 0
    assert create(client, h, 600, atten_lim_db=40).status_code == 200  # 100,000 at most (Denoising has no rebuild to hold for)


# ------------------------------------------------------------------------- the decision, with the measurement stubbed

CASES = [  # what the measurement says -> the settings of the job
    ({"strength": "gentle", "rebuild": False, "heavy": False}, "standard"),
    ({"strength": "balanced", "rebuild": False, "heavy": False}, "standard"),
    ({"strength": "strong", "rebuild": False, "heavy": True}, "standard"),
    ({"strength": "max", "rebuild": False, "heavy": False}, "standard"),
    ({"strength": "gentle", "rebuild": False, "heavy": True}, "standard"),
    ({"rebuild": True, "reason": "echoey"}, "rebuild"),
    ({"rebuild": True, "reason": "phone"}, "rebuild"),
    ({"rebuild": True, "reason": "muffled"}, "rebuild"),
    ({"rebuild": True, "reason": "distorted"}, "rebuild"),
    ({"rebuild": True, "reason": "unclear"}, "rebuild"),
]


def _case_measurement(want):
    try:
        return measurement(**want)
    except LookupError:  # (not every combination is among the stored sets)
        pytest.skip(f"no stored measurements with {want}")


@pytest.mark.parametrize(("want", "tier"), CASES)
def test_the_job_becomes_the_job_the_model_decided(client, signed_in, monkeypatch, gpu_up, want, tier):
    """probe -> measure -> decide -> main: the main stage gets the timeline the decided settings stand for, the job reports what was
    chosen and why, and is charged for exactly what ran. Any worker measures and decides it; a decided rebuild goes on the card."""
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    stats = _case_measurement(want)
    stages = Stages(monkeypatch, stats)
    jid = start(client, h)
    assert held(uid) == rebuilt(60)  # the maximum (the voice rebuilt, with Leveling), until it is decided
    take("cpu:t")  # (the card does not take a job that is not decided yet)
    if tier == "rebuild":
        assert row(jid)["status"] == "queued" and row(jid)["next_stage"] == "main" and stages.stages == ["probe", "analyse"]
        monkeypatch.setattr(worker, "GPU_MODE", True)  # the rebuild runs on the card only
        take("gpu:t")
    done = client.get(f"/api/jobs/{jid}").json()
    assert done["status"] == "done" and stages.stages == ["probe", "analyse", "main"]
    rec = reco_model.recommend(stats, min_seconds=0.0)
    chosen = done["chosen"]
    assert tier == chosen["denoiseTier"] and "maximum" not in done
    assert row(jid)["tier"] == tier  # (the record's tier names what was decided, a Voice rebuild too)
    assert chosen["rebuild"] is (tier == "rebuild") and chosen["rebuildWanted"] is rec.rebuild and chosen["leveling"] is True
    assert chosen["reasons"] and all(isinstance(r, str) for r in chosen["reasons"]) and chosen["fixed"] == []
    main = stages.spec("main")
    assert main["editor"]["timeline"] == params_of(jid)["edit"]["timeline"] and main["editor"]["op"] == "first"
    if tier == "standard":
        assert (chosen["strength"], chosen["attenLimDb"]) == (rec.strength, reco_model.STRENGTH_DB[rec.strength])
        assert (chosen["noise"], chosen["heavy"]) == (("heavy", True) if rec.heavy_remover else ("light", False))
        assert "restore_highs" not in main["options"] and "restoreHighs" not in chosen
        expect_label, cost = "Denoising + Leveling", standard(60, chosen["noise"])
    else:
        assert (chosen["strength"], chosen["attenLimDb"], chosen["noise"], chosen["heavy"]) == (None, None, None, False)
        expect_label, cost = "Voice rebuild + Leveling", rebuilt(60)
        assert any("Rebuilding the voice fixes this" in r for r in chosen["reasons"])
    assert done["label"] == expect_label and params_of(jid)["rates"] == {}
    # the job is made with the timeline the decided settings stand for: every step on the whole file
    layers = params_of(jid)["edit"]["timeline"]["layers"]
    assert [p["loudness_lufs"] for p in layers["leveling"]] == [-16.0]
    if tier == "standard":
        assert "rebuild" not in layers and [(p["strength_db"], p["remover"]) for p in layers["noise"]] == [(chosen["attenLimDb"], chosen["noise"])]
    else:
        assert "noise" not in layers and [(p["start"], p["end"]) for p in layers["rebuild"]] == [(0.0, 60.0)]
    # charged for what ran, which is also the ledger line and what the account was left with
    assert done["costMicros"] == cost
    line = db.one("SELECT * FROM ledger WHERE ref_id = ?", (jid,))
    assert (-line["amount_micros"], line["description"]) == (cost, f"{expect_label}: talk.wav (1.0 min)")
    assert held(uid) == 0 and ledger.balance(uid)["balance_micros"] == 5_000_000 - cost


def test_the_hold_shrinks_to_the_decision_when_it_is_made(client, signed_in, monkeypatch):
    """The cheapest outcome frees the rest of the held money at once, before the main stage runs."""
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    stats = measurement(strength="gentle", rebuild=False, heavy=False)
    jid = start(client, h, seconds=600)
    before = db.one("SELECT * FROM holds WHERE ref_id = ?", (jid,))["amount_micros"]
    assert before == held(uid) == rebuilt(600) == row(jid)["est_cost_micros"]
    seen = {}

    def run(job, d, stage, timeout):
        if stage == "main":  # the decision has been made and applied by now
            seen["held"], seen["est"] = held(uid), row(jid)["est_cost_micros"]
            seen["hold"] = db.one("SELECT amount_micros FROM holds WHERE ref_id = ?", (jid,))["amount_micros"]
            seen["tier"] = row(jid)["tier"]
            return {"ok": True, "outputs": [], "kept": {}}
        return {"ok": True, "duration": 600.0, "channels": 1} if stage == "probe" else {"ok": True, "stats": stats}

    monkeypatch.setattr(worker, "run_stage", run)
    monkeypatch.setattr(worker, "_complete", lambda *a, **k: None)
    take()
    after = standard(600)
    assert seen == {"held": after, "est": after, "hold": after, "tier": "standard"} and after < before
    shown = client.get(f"/api/jobs/{jid}").json()
    assert shown["estCostMicros"] == after and "maximum" not in shown and shown["chosen"]["denoiseTier"] == "standard"


def test_a_file_longer_than_it_was_said_to_be_is_held_for_its_real_length(client, signed_in, monkeypatch):
    """The probe measures the real length first: the hold grows to the maximum for it (as for any job) before the decision."""
    uid, h = signed_in
    ledger.credit(uid, 20_000_000, "grant", "t", "g")
    stats = measurement(strength="balanced", rebuild=False, heavy=False)
    stages = Stages(monkeypatch, stats, duration=300.0)
    jid = start(client, h, seconds=60)
    assert held(uid) == rebuilt(60)
    seen = []
    real = worker._reprice
    monkeypatch.setattr(worker, "_reprice", lambda job, duration: real(job, duration) or seen.append((duration, held(uid))))
    take()
    done = client.get(f"/api/jobs/{jid}").json()
    assert done["status"] == "done" and done["durationSeconds"] == 300.0
    assert seen == [(300.0, rebuilt(300))]  # held for the maximum at the real length, before the decision
    assert done["costMicros"] == standard(300)
    assert held(uid) == 0


# ---------------------------------------------------------------------------------- the voice rebuild: the card, or not

def test_a_rebuild_decided_on_a_cpu_worker_goes_to_the_card_and_keeps_its_place(client, signed_in, monkeypatch, gpu_up):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    stats = _case_measurement({"rebuild": True, "reason": "echoey"})
    stages = Stages(monkeypatch, stats)
    jid = start(client, h)
    queued_at = row(jid)["queued_at"]
    take("cpu:1")  # a CPU worker: probes, measures, decides on a rebuild, and gives the job up
    mid = row(jid)
    assert stages.stages == ["probe", "analyse"] and mid["status"] == "queued" and mid["worker_id"] is None
    assert (mid["next_stage"], mid["queued_at"], mid["attempts"]) == ("main", queued_at, 0)
    shown = client.get(f"/api/jobs/{jid}").json()
    assert shown["chosen"]["rebuild"] is True and shown["chosen"]["denoiseTier"] == "rebuild" and shown["label"] == "Voice rebuild + Leveling"
    assert shown["queuePosition"] == 1
    assert mid["est_cost_micros"] == rebuilt(60) and held(uid) == mid["est_cost_micros"]  # held for what will be charged
    assert files_of(jid) == {"input.wav"}  # scratch files are gone
    # the CPU workers leave it alone: it is a voice rebuild now (the card runs those only)
    assert worker.claim("cpu:2") is None
    monkeypatch.setattr(worker, "GPU_MODE", True)  # now the card's worker comes for it
    take("gpu:1")
    assert stages.stages == ["probe", "analyse", "main"]  # the card's worker goes straight on: no second probe, no second measurement
    assert "rebuild" in stages.calls[-1][1]["editor"]["timeline"]["layers"]  # (the timeline the decision became)
    done = client.get(f"/api/jobs/{jid}").json()
    assert done["status"] == "done" and done["costMicros"] == rebuilt(60)
    assert held(uid) == 0


def test_the_card_does_not_take_a_job_that_is_not_decided_yet(client, signed_in, monkeypatch, gpu_up):
    """Any worker measures an automatic job first (the card only runs what is decided as a rebuild)."""
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    jid = start(client, h)
    assert broker.gpu_stages(dict(row(jid))) == [] and broker.gpu_only(dict(row(jid))) is False
    monkeypatch.setattr(worker, "GPU_MODE", True)
    assert worker.claim("gpu:1") is None and row(jid)["status"] == "queued"
    monkeypatch.setattr(worker, "GPU_MODE", False)
    assert worker.claim("cpu:1")["id"] == jid


@pytest.mark.parametrize("why", ["too long", "card down", "own denoising", "echo"])
def test_where_a_rebuild_is_wanted_but_cannot_be_done_it_is_denoised_and_says_so(client, signed_in, monkeypatch, why):
    """The recording needs its voice rebuilt, but the file is too long, the card is not up, the request named Denoising settings, or
    it asked for echo removal: Denoising instead, with the strength the model gives, and `chosen` says what was wanted and why not."""
    uid, h = signed_in
    ledger.credit(uid, 20_000_000, "grant", "t", "g")
    stats = _case_measurement({"rebuild": True, "reason": "echoey"})
    rec = reco_model.recommend(stats, min_seconds=0.0)
    options, seconds, duration = {}, 60, 60.0
    if why == "too long":
        seconds, duration = 91 * 60, 91 * 60.0  # real length: over the rebuild's 90 minutes (and said so)
    if why == "own denoising":
        options = {"atten_lim_db": 24}
    if why == "echo":
        options = {"echo_reference_track": True}
    if why != "card down":
        broker.mark_gpu_worker("alive")
    else:
        broker.mark_gpu_worker("stopped")
    stages = Stages(monkeypatch, stats, duration=duration)
    jid = start(client, h, seconds=seconds, **options)
    take("cpu:1")
    done = client.get(f"/api/jobs/{jid}").json()
    assert done["status"] == "done" and stages.stages == ["probe", "analyse", "main"]
    chosen = done["chosen"]
    assert chosen["rebuild"] is False and chosen["rebuildWanted"] is True and done["denoiseTier"] == "standard"
    blocker = {"too long": "voice rebuilding takes files up to 90 minutes", "card down": "voice rebuilding is paused right now",
               "own denoising": "you chose Denoising settings yourself", "echo": "voice rebuilding is not done together with echo removal"}[why]
    assert any(f"Rebuilding the voice would fix this, but {blocker}, so Denoising is used instead." == r for r in chosen["reasons"])
    assert done["notice"] and blocker in done["notice"] and "cleaned with Denoising instead" in done["notice"]
    if why != "own denoising":
        assert chosen["strength"] == rec.strength and chosen["attenLimDb"] == reco_model.STRENGTH_DB[rec.strength]  # the model's call
    layers = stages.spec("main")["editor"]["timeline"]["layers"]
    assert "rebuild" not in layers and [p["strength_db"] for p in layers["noise"]] == [chosen["attenLimDb"]]
    assert done["label"].startswith("Denoising") and done["costMicros"] > 0
    assert held(uid) == 0


def test_the_card_coming_up_between_creating_and_deciding_is_enough(client, signed_in, monkeypatch):
    """The hold covers a rebuild whether or not the card is up when the job is made: it is asked when the decision is made."""
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    broker.mark_gpu_worker("stopped")
    jid = start(client, h)  # Denoising + Leveling would be 10,000 a minute; the rebuild is dearer, and the card may come up
    assert row(jid)["est_cost_micros"] == rebuilt(60) == held(uid)
    broker.mark_gpu_worker("alive")
    stages = Stages(monkeypatch, _case_measurement({"rebuild": True, "reason": "muffled"}))
    take("cpu:1")
    assert row(jid)["status"] == "queued" and row(jid)["next_stage"] == "main" and stages.stages == ["probe", "analyse"]


# ------------------------------------------------------------------------------------------ what the request chose wins

def test_explicit_fields_win_over_the_decision_and_are_reported(client, signed_in, monkeypatch, gpu_up):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    stats = _case_measurement({"strength": "gentle", "rebuild": False, "heavy": True})
    stages = Stages(monkeypatch, stats)
    jid = start(client, h, atten_lim_db=24, noise="light", leveling=False)
    take()
    done = client.get(f"/api/jobs/{jid}").json()
    chosen = done["chosen"]
    assert (chosen["attenLimDb"], chosen["strength"], chosen["noise"], chosen["heavy"], chosen["leveling"]) == (24, "balanced", "light", False, False)
    assert chosen["fixed"] == ["atten_lim_db", "noise", "leveling"]
    assert chosen["reasons"][-1] == "You chose the noise strength, the noise type and Leveling yourself, so your choices are used."
    layers = stages.spec("main")["editor"]["timeline"]["layers"]
    assert [(p["strength_db"], p["remover"]) for p in layers["noise"]] == [(24, "light")] and "leveling" not in layers
    assert done["label"] == "Denoising" and done["costMicros"] == step("light", 60)  # only what was left on: nothing is added
    # a strength that is not one of the four has no name
    other = start(client, h, atten_lim_db=30)
    take()
    assert client.get(f"/api/jobs/{other}").json()["chosen"]["strength"] is None
    assert client.get(f"/api/jobs/{other}").json()["chosen"]["attenLimDb"] == 30


def test_a_field_that_is_not_fixed_is_still_decided(client, signed_in, monkeypatch, gpu_up):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    stats = _case_measurement({"strength": "balanced", "rebuild": False, "heavy": False})
    stages = Stages(monkeypatch, stats)
    jid = start(client, h, leveling=False)  # only Leveling is the request's
    take()
    chosen = client.get(f"/api/jobs/{jid}").json()["chosen"]
    assert chosen["strength"] == "balanced" and chosen["leveling"] is False and chosen["fixed"] == ["leveling"] and chosen["noise"] == "light"
    assert row(jid)["cost_micros"] == step("light", 60) and "leveling" not in params_of(jid)["edit"]["timeline"]["layers"]
    assert "leveling" not in stages.spec("main")["editor"]["timeline"]["layers"]


# ----------------------------------------------------------------------------------------- the stage's answers are checked

def test_the_measurements_the_runner_reports_are_checked():
    good = measurement(strength="gentle")
    assert automatic.clean_stats(good)["noiseDb"] == good["noiseDb"]
    assert automatic.clean_stats({**good, "extra": "x"}).keys() == automatic.clean_stats(good).keys()  # nothing but the numbers it reads
    assert automatic.clean_stats({**good, "tiltDb": float("nan"), "dropDb": "x", "peak": True})["tiltDb"] is None
    assert automatic.clean_stats({**good, "dropDb": "x"})["dropDb"] is None and automatic.clean_stats({**good, "peak": True})["peak"] is None
    assert automatic.clean_stats({**good, "hasSpeech": 1})["hasSpeech"] is True and automatic.clean_stats({**good, "hasSpeech": "yes"})["hasSpeech"] is False
    for bad in (None, "x", [], {**good, "noiseDb": None}, {**good, "noiseDb": float("inf")}, {**good, "seconds": None}):
        assert automatic.clean_stats(bad) is None


def test_a_measurement_that_cannot_be_read_fails_the_job_without_a_charge(client, signed_in, monkeypatch):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    Stages(monkeypatch, "garbage")
    jid = start(client, h)
    take()
    failed = client.get(f"/api/jobs/{jid}").json()
    assert failed["status"] == "failed" and "not charged" in failed["error"]
    assert held(uid) == 0 and ledger.balance(uid)["balance_micros"] == 5_000_000


# ----------------------------------------------------------------------------------------------- a job that is not a hurry

def test_an_interrupted_job_is_decided_once(client, signed_in, monkeypatch, gpu_up):
    """Put back in the queue after it was decided (a deploy): the next worker does not measure it again."""
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    stages = Stages(monkeypatch, _case_measurement({"strength": "balanced", "rebuild": False}))
    jid = start(client, h)
    job = worker.claim("w1")
    calls = []
    real = worker.run_stage

    def run(job_, d, stage, timeout):
        calls.append(stage)
        if stage == "main":
            raise worker.Interrupted()
        return real(job_, d, stage, timeout)

    monkeypatch.setattr(worker, "run_stage", run)
    worker.process(job, worker._settings_extra())
    assert calls == ["probe", "analyse", "main"] and row(jid)["status"] == "queued"
    params = params_of(jid)
    assert "chosen" in params and "candidates" not in params["automatic"]
    calls.clear()
    monkeypatch.setattr(worker, "run_stage", real)
    take("w2")
    assert stages.stages[-2:] == ["probe", "main"] and row(jid)["status"] == "done"  # (a probe again, no measurement)


# ----------------------------------------------------------------------------------- API keys: ownership and spending caps

def test_the_keys_monthly_cap_holds_for_the_maximum(client, signed_in):
    uid, _ = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    _, capped = api_key(uid, cap=500_000)
    e = estimate(client, capped, 600)
    assert e["estCostMicros"] == rebuilt(600) and e["keyLimit"]["monthlyCapMicros"] == 500_000
    _, tight = api_key(uid, cap=200_000)
    refused = create(client, tight, 600)  # the maximum (a rebuild) is over the cap, though Denoising alone (about 102,000) would not be
    assert refused.status_code == 403 and refused.headers["X-Error-Code"] == "key_spending_cap" and count("jobs") == 0
    assert create(client, tight, 600, atten_lim_db=40).status_code == 200  # Denoising only: about 102,000 at most
    assert create(client, capped, 600).status_code == 200  # the most it can cost (322,234) under the cap of 500,000


def test_a_job_belongs_to_its_key_and_its_account(client, signed_in, other_browser, monkeypatch, gpu_up):
    uid, _ = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    key_a, a = api_key(uid)
    _, b = api_key(uid)
    Stages(monkeypatch, _case_measurement({"strength": "balanced", "rebuild": False}))
    jid = start(client, a)
    assert row(jid)["api_key_id"] == key_a
    take()
    other_client, other_uid, other_h = other_browser
    assert other_client.get(f"/api/jobs/{jid}").status_code == 404  # another account sees nothing of it
    assert client.get(f"/api/jobs/{jid}", headers=b).status_code == 200  # the account's other key can read it ...
    r = client.get(f"/api/jobs/{jid}/timeline", headers=b)
    assert r.status_code == 404  # ... but only the key that made it can change it
    assert client.get(f"/api/jobs/{jid}/timeline", headers=a).status_code == 200


# ------------------------------------------------------------------------------------------ changing a decided job

def test_a_decided_job_is_changed_like_any_job_through_its_timeline(client, signed_in, monkeypatch, gpu_up):
    """The settings the decision made are the job's timeline: it is read, and edited, like the timeline of any job."""
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    key, a = api_key(uid)
    stats = _case_measurement({"strength": "balanced", "rebuild": False, "heavy": False})
    Stages(monkeypatch, stats)
    jid = start(client, a)
    take()
    first = client.get(f"/api/jobs/{jid}").json()
    assert first["status"] == "done" and first["chosen"]["strength"] == "balanced" and "redo" not in first and first["edit"]["hasTimeline"] is True
    tl = client.get(f"/api/jobs/{jid}/timeline", headers=a)
    assert tl.status_code == 200, tl.text
    layers = tl.json()["timeline"]["layers"]
    assert [(p["strength_db"], p["remover"]) for p in layers["noise"]] == [(first["chosen"]["attenLimDb"], first["chosen"]["noise"])]
    assert [p["loudness_lufs"] for p in layers["leveling"]] == [-16.0]
    assert first["options"].get("automatic") is None and params_of(jid)["automatic"] == {"fixed": []}  # (decided: only what the request fixed stays)
    # the routes of the old "change and save again" are gone
    for suffix in ("/rerun", "/rerun/estimate"):
        assert client.post(f"/api/jobs/{jid}{suffix}", json={"options": {"atten_lim_db": 12}}, headers=a).status_code in (404, 405)


# ------------------------------------------------------------------------------------------------ the ledger and the rest

def test_reduce_hold_gives_back_what_the_job_will_not_use(client, signed_in):
    uid, _ = signed_in
    ledger.credit(uid, 1_000_000, "grant", "t", "g")
    hold = ledger.create_hold(uid, 600_000, "job", "j1")
    assert ledger.balance(uid) == {"balance_micros": 1_000_000, "held_micros": 600_000, "available_micros": 400_000}
    assert ledger.reduce_hold(hold, 200_000) == 200_000
    assert ledger.balance(uid) == {"balance_micros": 1_000_000, "held_micros": 200_000, "available_micros": 800_000}
    assert db.one("SELECT amount_micros FROM holds WHERE id = ?", (hold,))["amount_micros"] == 200_000
    assert ledger.reduce_hold(hold, 900_000) == 200_000 and held(uid) == 200_000  # never larger
    with pytest.raises(ledger.LedgerError):
        ledger.reduce_hold(hold, 0)
    assert ledger.capture_hold(hold, 150_000) == 150_000 and held(uid) == 0
    with pytest.raises(ledger.LedgerError):
        ledger.reduce_hold(hold, 100_000)  # a closed hold is not changed
    assert ledger.invariant_problems() == {} or not any(ledger.invariant_problems().values())


def test_an_ordinary_job_has_no_automatic_record_and_is_the_timeline_its_options_stand_for(client, signed_in, monkeypatch):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    r = client.post("/api/jobs", json={"kind": "denoise", "denoiseTier": "standard", "filename": "talk.wav", "sizeBytes": 3000,
                                       "durationSeconds": 60}, headers=h)
    jid = r.json()["id"]
    assert "maximum" not in r.json() and "automatic" not in params_of(jid)
    shown = client.get(f"/api/jobs/{jid}").json()
    assert "chosen" not in shown and "maximum" not in shown and shown["label"] == "Denoising + Leveling"
    assert params_of(jid)["edit"]["first_run"] is True and params_of(jid)["whole_file"]["kind"] == "denoise"  # (before it has run)
    assert worker._job_cost(row(jid), params_of(jid), 60) == standard(60) == r.json()["estCostMicros"]


# ------------------------------------------------------------------------------------------ the real stage (native pieces)

@NATIVE
def test_the_real_measurement_stage_decides_and_the_job_is_charged_for_what_ran(client, signed_in):
    """The sandboxed `analyse` stage on a real file, with the real models: the job finishes, `chosen` is filled in from real
    measurements, and the charge is the price of exactly what it says it did. (No card here, so no rebuild.)"""
    import subprocess

    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    sample = Path(SAMPLE)
    if not sample.exists():
        pytest.skip("no sample audio")
    clip = Path(jobs.get_settings().data_dir) / "clip.wav"
    subprocess.run([str(HOME / "bin/ffmpeg"), "-v", "error", "-i", str(sample), "-t", "8", "-y", str(clip)], check=True)
    data = clip.read_bytes()
    r = client.post("/api/jobs", json={"kind": "denoise", "denoiseTier": "standard", "filename": "talk.wav", "sizeBytes": len(data),
                                       "durationSeconds": 8, "options": {"automatic": True, "output_video": False, "audio_format": "wav"}}, headers=h)
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    assert client.put(f"/api/jobs/{jid}/input", content=data, headers=h).status_code == 200
    _run_worker_once("done")
    done = client.get(f"/api/jobs/{jid}").json()
    chosen = done["chosen"]
    assert chosen and chosen["denoiseTier"] == "standard" and chosen["strength"] in reco_model.STRENGTH_DB and chosen["reasons"]
    assert chosen["noise"] in ("light", "heavy")
    layers = params_of(jid)["edit"]["timeline"]["layers"]
    assert [(p["strength_db"], p["remover"]) for p in layers["noise"]] == [(chosen["attenLimDb"], chosen["noise"])]
    seconds = row(jid)["duration_seconds"]
    assert 0 < done["costMicros"] <= standard(seconds, chosen["noise"]) and held(uid) == 0  # (what the runner reports it ran: never more than the timeline)
    assert done["label"] == "Denoising + Leveling"
