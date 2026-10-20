"""Echo removal is its own price (pricing.py "echo" / "standard"): on the price list, a line of its own in the estimate and the ledger of a job
made the old way, pinned with the job (a job made before has none: nothing charged), not charged when the file had no second track, and
the same constant as the per-part editor's."""
import json

import pytest

from acw import db, editing, jobs, rerun
from acw.billing import ledger, pricing
from acw.worker import main as worker_main

from .test_rerun_plan import replan, source

SECONDS = 600


def body(**options):
    return {"kind": "denoise", "denoiseTier": "standard", "filename": "rec.mkv", "sizeBytes": 1000, "durationSeconds": SECONDS, "options": options}


def est(**kw):
    """The body of an estimate: a job's, without the file."""
    b = {**body(), **kw}
    b.pop("filename"), b.pop("sizeBytes")
    return b


def test_the_price_list_has_the_row_and_the_editor_uses_the_same_constant(client):
    row = next(r for r in client.get("/api/billing/pricing").json()["rates"] if r["product"] == "echo")
    assert row == {"product": "echo", "tier": "standard", "label": "Echo removal", "usdPerMinute": 0.0026, "usdPerHour": 0.156, "available": True}
    assert pricing.rate("echo", "standard").micros_per_minute == 2_600 == editing.step_rates()["echo"]
    assert not hasattr(editing, "ECHO_MICROS_PER_MINUTE")


def test_the_estimate_has_an_echo_line_only_when_it_is_on(client, signed_in):
    uid, h = signed_in
    plain = client.post("/api/jobs/estimate", json=est(), headers=h).json()
    on = client.post("/api/jobs/estimate", json=est(options={"echo_reference_track": True}), headers=h).json()
    assert [x["item"] for x in plain["lines"]] == ["denoise", "leveling"]
    assert [x["item"] for x in on["lines"]] == ["denoise", "echo", "leveling"]
    echo = on["lines"][1]
    assert echo["label"] == "Echo removal" and echo["usdPerMinute"] == 0.0026 and echo["micros"] == 26_000  # 10 min at 2,600 a minute
    assert on["estCostMicros"] == plain["estCostMicros"] + 26_000 == sum(x["micros"] for x in on["lines"])
    # every kind of job that cleans: a job that only levels, and one that also writes subtitles; subtitles alone have no echo removal
    off = client.post("/api/jobs/estimate", json=est(denoiseTier="off", options={"echo_reference_track": True}), headers=h).json()
    assert [x["item"] for x in off["lines"]] == ["echo", "leveling"]
    both = client.post("/api/jobs/estimate", json=est(kind="denoise_transcribe", transcribeTier="local", options={"echo_reference_track": True}), headers=h).json()
    assert [x["item"] for x in both["lines"]] == ["denoise", "echo", "leveling", "transcribe"]
    subs = client.post("/api/jobs/estimate", json={"kind": "transcribe", "transcribeTier": "local", "durationSeconds": SECONDS,
                                                   "options": {"echo_reference_track": True}}, headers=h).json()
    assert [x["item"] for x in subs["lines"]] == ["transcribe"]


def test_the_job_pins_the_rate_and_holds_the_price(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 2_000_000, "grant", "t", "g")
    plain = client.post("/api/jobs", json=body(), headers=h).json()
    on = client.post("/api/jobs", json=body(echo_reference_track=True), headers=h).json()
    assert on["estCostMicros"] == plain["estCostMicros"] + 26_000
    rates = json.loads(db.one("SELECT params_json FROM jobs WHERE id = ?", (on["id"],))["params_json"])["rates"]
    assert rates == {"denoise": 9_000, "echo": 2_600, "leveling": 1_000}
    assert "echo" not in json.loads(db.one("SELECT params_json FROM jobs WHERE id = ?", (plain["id"],))["params_json"])["rates"]


def test_a_job_made_with_a_timeline_is_priced_by_its_parts_not_by_the_option(client, signed_in):
    uid, h = signed_in
    tl = {"version": 1, "seconds": SECONDS, "layers": {"echo": [{"start": 0, "end": 60}]}}
    e = client.post("/api/jobs/estimate", json={"kind": "denoise", "durationSeconds": SECONDS, "timeline": tl}, headers=h).json()
    assert [x["item"] for x in e["lines"]] == ["echo"] and e["lines"][0]["usdPerMinute"] == 0.0026


def test_a_job_made_before_the_price_keeps_its_rates_and_one_track_is_not_charged():
    job = {"kind": "denoise"}
    old = {"denoise_tier": "standard", "transcribe_tier": None, "options": {"echo_reference_track": True}, "rates": {"denoise": 9_000, "leveling": 1_000}}
    assert worker_main._job_cost(job, old, 600) == 90_000 + 10_000  # no echo line pinned: nothing for echo
    new = {**old, "rates": {"denoise": 9_000, "echo": 2_600, "leveling": 1_000}}
    assert worker_main._job_cost(job, new, 600) == 90_000 + 26_000 + 10_000
    assert worker_main._job_cost(job, {**new, "echo_skipped": True}, 600) == 90_000 + 10_000
    assert worker_main._job_cost(job, new, 0.4) == 150 + 44 + 17  # by the second, rounded up, each step apart


def test_automatic_settings_hold_for_the_echo_removal_too(client, signed_in):
    uid, h = signed_in
    on = client.post("/api/jobs/estimate", json=est(options={"automatic": True, "echo_reference_track": True}), headers=h).json()
    assert [x["item"] for x in on["lines"]] == ["denoise", "echo", "leveling"]  # (a rebuild is not done with echo removal: Denoising it is)
    assert on["estCostMicros"] == 90_000 + 26_000 + 10_000


def test_a_rerun_pays_echo_removal_when_it_runs_again():
    assert replan(source(), echo_reference_track=True)["paid"] == ["denoise", "echo", "leveling"]
    assert "echo" not in replan(source(), atten_lim_db=24)["paid"]  # (a new strength: from the kept noise removal, echo removal is not run again)
    rates = pricing.rerun_rates("standard", None, ["denoise", "echo", "leveling"])
    assert rates == {"denoise": 9_000, "echo": 2_600, "leveling": 1_000}
    assert pricing.rerun_cost_micros(rates, 600) == 90_000 + 26_000 + 10_000
    assert [x["item"] for x in jobs.price_lines(rates, "standard", None, 600, 0)] == ["denoise", "echo", "leveling"]
    assert rerun.rerun_label(rates, "standard", None).endswith("(Denoising + Echo removal + Leveling)")
    assert "echo" not in pricing.rerun_rates("standard", None, ["denoise", "leveling"])
