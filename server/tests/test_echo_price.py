"""Echo removal is its own price (pricing.py "echo" / "standard"): on the price list, a line of its own in the estimate and the ledger of a job
asked for with options (the timeline they stand for: the echo step on the whole file, with the fixed amount of a run every step has),
pinned with the job, and the same constant as the per-part editor's. (A file with one audio track fails the job: test_echo_one_track.py.)"""
import json
import math

import pytest

from acw import db, edit_prices, editing, jobs
from acw.billing import ledger, pricing
from acw.worker import main as worker_main
from acw_common import editor

SECONDS = 600


def step(name, seconds=SECONDS, rates=None):
    return edit_prices.step_micros((rates or edit_prices.step_rates())[name], math.ceil(seconds), editor.STEP_COSTS[name].fixed_equivalent_seconds)


def body(**options):
    return {"kind": "denoise", "denoiseTier": "standard", "filename": "rec.mkv", "sizeBytes": 1000, "durationSeconds": SECONDS, "options": options}


def est(**kw):
    """The body of an estimate: a job's, without the file."""
    b = {**body(), **kw}
    b.pop("filename"), b.pop("sizeBytes")
    return b


def params_of(jid):
    return json.loads(db.one("SELECT params_json FROM jobs WHERE id = ?", (jid,))["params_json"])


def test_the_price_list_has_the_row_and_the_editor_uses_the_same_constant(client):
    row = next(r for r in client.get("/api/billing/pricing").json()["rates"] if r["product"] == "echo")
    assert row == {"product": "echo", "tier": "standard", "label": "Echo removal", "usdPerMinute": 0.0026, "usdPerHour": 0.156, "available": True}
    assert pricing.rate("echo", "standard").micros_per_minute == 2_600 == edit_prices.step_rates()["echo"]
    assert not hasattr(editing, "ECHO_MICROS_PER_MINUTE")


def test_the_estimate_has_an_echo_line_only_when_it_is_on(client, signed_in):
    uid, h = signed_in
    plain = client.post("/api/jobs/estimate", json=est(), headers=h).json()
    on = client.post("/api/jobs/estimate", json=est(options={"echo_reference_track": True}), headers=h).json()
    assert [x["item"] for x in plain["lines"]] == ["denoise", "leveling"]
    assert [x["item"] for x in on["lines"]] == ["echo", "denoise", "leveling"]  # (the order of the steps: the echo is taken out first)
    echo = on["lines"][0]
    assert echo["label"] == "Echo removal" and echo["usdPerMinute"] == 0.0026 and echo["micros"] == step("echo") == 26_153  # 10 min at 2,600 a minute, and a run's fixed amount
    assert on["estCostMicros"] == plain["estCostMicros"] + step("echo") == sum(x["micros"] for x in on["lines"])
    # every kind of job that cleans: a job that only levels, and one that also writes subtitles; subtitles alone have no echo removal
    off = client.post("/api/jobs/estimate", json=est(denoiseTier="off", options={"echo_reference_track": True}), headers=h).json()
    assert [x["item"] for x in off["lines"]] == ["echo", "leveling"]
    both = client.post("/api/jobs/estimate", json=est(kind="denoise_transcribe", transcribeTier="local", options={"echo_reference_track": True}), headers=h).json()
    assert [x["item"] for x in both["lines"]] == ["echo", "denoise", "leveling", "transcribe"]
    subs = client.post("/api/jobs/estimate", json={"kind": "transcribe", "transcribeTier": "local", "durationSeconds": SECONDS,
                                                   "options": {"echo_reference_track": True}}, headers=h).json()
    assert [x["item"] for x in subs["lines"]] == ["transcribe"]


def test_the_job_pins_the_rate_holds_the_price_and_has_the_echo_step_on_the_whole_file(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 2_000_000, "grant", "t", "g")
    plain = client.post("/api/jobs", json=body(), headers=h).json()
    on = client.post("/api/jobs", json=body(echo_reference_track=True), headers=h).json()
    assert on["estCostMicros"] == plain["estCostMicros"] + step("echo")
    p_on, p_plain = params_of(on["id"]), params_of(plain["id"])
    assert p_on["edit_rates"]["echo"] == 2_600 and p_on["rates"] == {} and p_on["edit"]["first_run"] is True
    assert p_on["edit"]["timeline"]["layers"]["echo"] == [{"start": 0.0, "end": float(SECONDS)}]
    assert "echo" not in p_plain["edit"]["timeline"]["layers"]
    assert p_on["whole_file"]["options"]["echo_reference_track"] is True and p_on["options"]["echo_reference_track"] is True  # (the record keeps what was asked for; the timeline is what is made)
    assert editing.wants_echo(p_on) is True and editing.wants_echo(p_plain) is False


def test_a_job_made_with_a_timeline_is_priced_by_its_parts_not_by_the_option(client, signed_in):
    uid, h = signed_in
    tl = {"version": 1, "seconds": SECONDS, "layers": {"echo": [{"start": 0, "end": 60}]}}
    e = client.post("/api/jobs/estimate", json={"kind": "denoise", "durationSeconds": SECONDS, "timeline": tl}, headers=h).json()
    assert [x["item"] for x in e["lines"]] == ["echo"] and e["lines"][0]["usdPerMinute"] == 0.0026
    assert e["estCostMicros"] == step("echo", 60)


def test_a_job_keeps_the_rates_it_was_made_with_and_is_charged_by_the_second_for_each_step(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 2_000_000, "grant", "t", "g")
    made = client.post("/api/jobs", json=body(echo_reference_track=True), headers=h).json()
    job = dict(db.one("SELECT * FROM jobs WHERE id = ?", (made["id"],)))
    params = params_of(made["id"])
    assert worker_main._job_cost(job, params, SECONDS) == step("echo") + step("light") + step("leveling") == made["estCostMicros"]
    # the price list changes later: this job is charged at the prices it was made with
    cheaper = {**params, "edit_rates": {**params["edit_rates"], "echo": 1_000, "light": 4_000}}
    assert worker_main._job_cost(job, cheaper, SECONDS) == step("echo", rates=cheaper["edit_rates"]) + step("light", rates=cheaper["edit_rates"]) + step("leveling")
    assert worker_main._job_cost(job, cheaper, SECONDS) < made["estCostMicros"]
    # (by the second, rounded up, each step apart; and the fixed amount of a run makes a short file cost more than its seconds)
    short = worker_main._job_cost(job, params, 0.4)
    assert short == step("echo", 1) + step("light", 1) + step("leveling", 1) > 150 + 44 + 17


def test_automatic_settings_hold_for_the_echo_removal_too(client, signed_in):
    uid, h = signed_in
    on = client.post("/api/jobs/estimate", json=est(options={"automatic": True, "echo_reference_track": True}), headers=h).json()
    assert [x["item"] for x in on["lines"]] == ["echo", "denoise", "leveling"]  # (a rebuild is not done with echo removal: Denoising it is)
    assert on["estCostMicros"] == step("echo") + step("light") + step("leveling") == sum(x["micros"] for x in on["lines"])
