"""Restore highs as a paid feature of its own: the price list, estimates and holds, the stored default (off), the voice rebuild
never having it, Leveling and highs on their own ("off" noise tier), what a job pinned before it had a price is charged,
the ledger wording, and (native-guarded) the real worker running it and charging for it."""
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from acw import db, jobs
from acw.billing import ledger, pricing
from acw_common.ffmpeg import FFmpeg

from .rerun_support import clock, complete_like_the_worker, count, folder, params_of, roomy_disk, row, set_env  # noqa: F401 - fixtures
from .test_job_echo import NATIVE
from .test_job_pipeline import HOME, SAMPLE, _run_worker_once

pytestmark = pytest.mark.usefixtures("roomy_disk")

HIGHS = 70_000  # micros per minute: 4.20 USD an hour
SECONDS = 8


def per(micros_per_minute, seconds):
    """The price of a step: per second of audio, a started second counting."""
    return math.ceil(micros_per_minute * math.ceil(seconds) / 60)


def create(client, h, tier="standard", kind="denoise", seconds=60, **options):
    return client.post("/api/jobs", json={"kind": kind, "denoiseTier": tier, "filename": "talk.wav", "sizeBytes": 3000,
                                          "durationSeconds": seconds, "options": options}, headers=h)


def estimate(client, h, tier="standard", seconds=600, **over):
    options = over.pop("options", {})
    r = client.post("/api/jobs/estimate", json={"kind": "denoise", "denoiseTier": tier, "durationSeconds": seconds, "options": options, **over},
                    headers=h)
    assert r.status_code == 200, r.text
    return r.json()


def upload(client, h, jid):
    assert client.put(f"/api/jobs/{jid}/input", content=bytes(3000), headers=h).status_code == 200


# ---------------------------------------------------------------------------------------------------- price list

def test_restore_highs_is_on_the_price_list_at_4_20_an_hour():
    """One more line of the price list, with its own words."""
    rate = pricing.rate("highs", "standard")
    assert (rate.label, rate.micros_per_minute) == ("Restore highs", HIGHS)
    listed = next(r for r in pricing.public_price_list() if r["product"] == "highs")
    assert (listed["label"], listed["usdPerMinute"], listed["usdPerHour"]) == ("Restore highs", 0.07, 4.2)


def test_job_rates_pin_the_highs_price_only_where_it_applies():
    """highs is in the rates of a Denoising or Leveling-only job that has it on; never for a rebuild or a subtitles-only job."""
    assert pricing.job_rates("denoise", "standard", None, 0, True, True) == {"denoise": 9_000, "highs": HIGHS, "leveling": 1_000}
    assert pricing.job_rates("denoise", "standard", None, 0, False, True) == {"denoise": 9_000, "highs": HIGHS}
    assert pricing.job_rates("denoise", "standard", None, 0, True, False) == {"denoise": 9_000, "leveling": 1_000}
    assert pricing.job_rates("denoise", "off", None, 0, False, True) == {"highs": HIGHS}
    assert pricing.job_rates("denoise", "off", None, 0, True, True) == {"highs": HIGHS, "leveling": 1_000}
    assert pricing.job_rates("denoise", "rebuild", None, 0, True, True) == {"denoise": 30_000, "leveling": 1_000}
    assert pricing.job_rates("transcribe", None, "local", 0, True, True) == {"transcribe": 4_000}
    assert list(pricing.job_rates("denoise_transcribe", "standard", "local", 2, True, True)) == [
        "denoise", "highs", "leveling", "transcribe", "translate"]


@pytest.mark.parametrize("seconds", [1, 14, 14.2, 59.9, 600, 3600, 8423.4])
def test_a_job_with_highs_costs_denoising_plus_highs_plus_leveling_per_second(seconds):
    """job_cost_micros with highs == the three per-second prices added (each rounded up on its own)."""
    cost = pricing.job_cost_micros("denoise", "standard", None, seconds, None, 0, True, True)
    assert cost == per(9_000, seconds) + per(HIGHS, seconds) + per(1_000, seconds)
    rates = pricing.job_rates("denoise", "standard", None, 0, True, True)
    assert pricing.job_cost_micros("denoise", "standard", None, seconds, rates, 0, True, True) == cost
    assert pricing.job_cost_micros("denoise", "standard", None, seconds, None, 0, True, False) == cost - per(HIGHS, seconds)
    assert pricing.job_cost_micros("denoise", "off", None, seconds, None, 0, False, True) == per(HIGHS, seconds)
    assert pricing.job_cost_micros("denoise", "rebuild", None, seconds, None, 0, True, True) == per(30_000, seconds) + per(1_000, seconds)
    assert pricing.job_cost_micros("transcribe", None, "local", seconds, None, 0, True, True) == per(4_000, seconds)
    pinned = pricing.job_cost_micros("denoise", "standard", None, seconds, {"denoise": 5, "highs": 6, "leveling": 7}, 0, True, True)
    assert pinned == per(5, seconds) + per(6, seconds) + per(7, seconds)  # a price change never reaches a pinned job


# ------------------------------------------------------------------------------- estimates, holds, defaults

def test_estimate_has_a_highs_line_between_denoising_and_leveling(client, signed_in):
    """Lines in price-list order with the highs line labelled "Restore highs"; none unless it is on."""
    _, h = signed_in
    e = estimate(client, h, options={"restore_highs": True})
    assert [x["item"] for x in e["lines"]] == ["denoise", "highs", "leveling"]
    assert e["lines"][1] == {"item": "highs", "label": "Restore highs", "quantity": 1, "usdPerMinute": 0.07, "micros": 10 * HIGHS}
    assert e["estCostMicros"] == sum(x["micros"] for x in e["lines"]) == 90_000 + 700_000 + 10_000
    assert e["options"]["restore_highs"] is True
    assert [x["item"] for x in estimate(client, h)["lines"]] == ["denoise", "leveling"]  # not asked for: no line
    assert [x["item"] for x in estimate(client, h, options={"restore_highs": False})["lines"]] == ["denoise", "leveling"]
    full = estimate(client, h, kind="denoise_transcribe", transcribeTier="local", options={"restore_highs": True, "translate": ["es"], "language": "en"})
    assert [x["item"] for x in full["lines"]] == ["denoise", "highs", "leveling", "transcribe", "translate"]
    assert full["estCostMicros"] == sum(x["micros"] for x in full["lines"])
    text_only = estimate(client, h, tier=None, kind="transcribe", transcribeTier="local", options={"restore_highs": True})
    assert [x["item"] for x in text_only["lines"]] == ["transcribe"]


def test_estimate_for_leveling_and_highs_on_their_own(client, signed_in):
    """Noise tier off: highs alone, highs with Leveling, Leveling alone."""
    _, h = signed_in
    assert [x["item"] for x in estimate(client, h, "off", options={"restore_highs": True, "leveling": False})["lines"]] == ["highs"]
    both = estimate(client, h, "off", options={"restore_highs": True})
    assert [x["item"] for x in both["lines"]] == ["highs", "leveling"] and both["estCostMicros"] == 710_000
    assert [x["item"] for x in estimate(client, h, "off")["lines"]] == ["leveling"]


def test_no_highs_line_for_the_voice_rebuild_even_if_the_option_is_sent(client, signed_in, gpu_up):
    """The rebuild makes its own highs: the option is stored false and nothing is charged for it."""
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    e = estimate(client, h, "rebuild", options={"restore_highs": True})
    assert [x["item"] for x in e["lines"]] == ["denoise", "leveling"] and e["options"]["restore_highs"] is False
    made = create(client, h, "rebuild", restore_highs=True)
    assert made.status_code == 200 and made.json()["estCostMicros"] == e["estCostMicros"] // 10 == 31_000
    shown = client.get(f"/api/jobs/{made.json()['id']}").json()
    assert shown["options"]["restore_highs"] is False and shown["label"] == "Voice rebuild + Leveling"
    assert params_of(made.json()["id"])["rates"] == {"denoise": 30_000, "leveling": 1_000}


def test_the_hold_is_the_estimate_with_highs(client, signed_in):
    """What the estimate shows is what is held, and the rates are pinned with the job."""
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    for options, tier in (({"restore_highs": True}, "standard"), ({"restore_highs": True, "leveling": False}, "standard"),
                          ({"restore_highs": True}, "off"), ({"restore_highs": True, "leveling": False}, "off")):
        e = estimate(client, h, tier, seconds=95.2, options=options)
        made = create(client, h, tier, seconds=95.2, **options)
        assert made.status_code == 200, made.text
        assert made.json()["estCostMicros"] == e["estCostMicros"] > 0
        hold = db.one("SELECT * FROM holds WHERE ref_id = ?", (made.json()["id"],))
        assert hold["amount_micros"] == e["estCostMicros"] and hold["status"] == "active"
        assert params_of(made.json()["id"])["rates"]["highs"] == HIGHS
        client.post(f"/api/jobs/{made.json()['id']}/cancel", headers=h)  # keeps the active-job limit free
    assert ledger.balance(uid)["held_micros"] == 0


def test_restore_highs_defaults_to_off_in_the_api(client, signed_in, monkeypatch):
    """Created without the option it is stored false and not charged; true is stored and charged; only booleans are taken."""
    set_env(monkeypatch, ACW_MAX_ACTIVE_JOBS=20)
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    assert jobs.validate_options("denoise", "standard", None, {})["restore_highs"] is False
    plain = create(client, h)
    assert plain.status_code == 200
    assert client.get(f"/api/jobs/{plain.json()['id']}").json()["options"]["restore_highs"] is False
    assert "highs" not in params_of(plain.json()["id"])["rates"]
    assert plain.json()["estCostMicros"] == per(9_000, 60) + per(1_000, 60)
    on = create(client, h, restore_highs=True)
    assert client.get(f"/api/jobs/{on.json()['id']}").json()["options"]["restore_highs"] is True
    assert on.json()["estCostMicros"] == plain.json()["estCostMicros"] + per(HIGHS, 60)
    for bad in ("yes", 1, None, "true"):
        assert create(client, h, restore_highs=bad).status_code == 400
    # a subtitles-only job has no sound to restore: the option is accepted and never charged
    text = client.post("/api/jobs", json={"kind": "transcribe", "transcribeTier": "local", "filename": "talk.wav", "sizeBytes": 3000,
                                          "durationSeconds": 60, "options": {"restore_highs": True}}, headers=h)
    assert text.status_code == 200 and text.json()["estCostMicros"] == per(4_000, 60)
    assert params_of(text.json()["id"])["rates"] == {"transcribe": 4_000}


# --------------------------------------------------------------------- the "off" noise tier: highs and Leveling

def test_the_off_tier_needs_leveling_or_highs(client, signed_in):
    """Neither: 400 with the new words. Highs only: works, costs only highs, and is named "Restore highs"."""
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    words = "Choose Denoising, Voice rebuild, Restore highs or Leveling."
    with pytest.raises(jobs.JobError, match="Choose Denoising, Voice rebuild, Restore highs or Leveling."):
        jobs.validate_options("denoise", "off", None, {"leveling": False})
    with pytest.raises(jobs.JobError, match="Choose Denoising, Voice rebuild, Restore highs or Leveling."):
        jobs.validate_options("denoise", "off", None, {"leveling": False, "restore_highs": False})
    refused = create(client, h, "off", leveling=False)
    assert refused.status_code == 400 and words in refused.json()["detail"]
    assert client.post("/api/jobs/estimate", json={"kind": "denoise", "denoiseTier": "off", "durationSeconds": 60,
                                                   "options": {"leveling": False}}, headers=h).status_code == 400
    assert count("jobs") == 0 and count("holds") == 0
    only = create(client, h, "off", leveling=False, restore_highs=True)
    assert only.status_code == 200, only.text
    jid = only.json()["id"]
    assert only.json()["estCostMicros"] == per(HIGHS, 60) == HIGHS
    shown = client.get(f"/api/jobs/{jid}").json()
    assert shown["label"] == "Restore highs" and shown["options"]["restore_highs"] is True and shown["options"]["leveling"] is False
    assert params_of(jid)["rates"] == {"highs": HIGHS}
    both = create(client, h, "off", restore_highs=True)
    assert client.get(f"/api/jobs/{both.json()['id']}").json()["label"] == "Restore highs + Leveling"
    # either is enough for the other options of the form to be taken
    assert jobs.validate_options("denoise", "off", None, {"leveling": False, "restore_highs": True})["restore_highs"] is True
    assert jobs.validate_options("denoise", "off", None, {})["leveling"] is True
    # the other tiers do not need either
    assert jobs.validate_options("denoise", "standard", None, {"leveling": False})["leveling"] is False


def test_job_labels_put_restore_highs_between_the_noise_step_and_leveling():
    """jobs.job_label with the highs flag, and the compatibility rule for jobs stored without it."""
    label = jobs.job_label
    assert label("denoise", "standard", None) == "Denoising"
    assert label("denoise", "standard", None, True) == "Denoising + Leveling"
    assert label("denoise", "standard", None, False, True) == "Denoising + Restore highs"
    assert label("denoise", "standard", None, True, True) == "Denoising + Restore highs + Leveling"
    assert label("denoise", "rebuild", None, True) == "Voice rebuild + Leveling"
    assert label("denoise", "off", None, False, True) == "Restore highs"
    assert label("denoise", "off", None, True, True) == "Restore highs + Leveling"
    assert label("denoise", "off", None, True, False) == "Leveling"
    assert label("denoise", "off", None) == "Leveling"  # an old Leveling-only job stored without any rates
    assert label("denoise_transcribe", "standard", "local", True, True) == "Denoising + Restore highs + Leveling + Subtitles & transcript"
    assert label("transcribe", None, "local", True, True) == "Subtitles & transcript"
    assert jobs.label_of("denoise", {"denoise_tier": "off", "rates": {"highs": HIGHS}}) == "Restore highs"
    assert jobs.label_of("denoise", {"denoise_tier": "standard", "rates": {"denoise": 9_000, "highs": HIGHS, "leveling": 1_000}}) == \
        "Denoising + Restore highs + Leveling"


# ------------------------------------------------------------------------ what a job is charged when it completes

def _charge_job(client, h, uid, seconds=60, tier="standard", **options):
    r = create(client, h, tier, seconds=seconds, **options)
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    upload(client, h, jid)
    return jid


def test_a_job_with_highs_is_charged_the_three_lines_and_the_ledger_names_them(client, signed_in):
    """The charge equals the hold; the ledger line, the job label and the credit history say "Denoising + Restore highs + Leveling"."""
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    jid = _charge_job(client, h, uid, restore_highs=True)
    done = complete_like_the_worker(jid, outputs=["talk-cleaned.flac"])
    assert done["cost_micros"] == done["est_cost_micros"] == 9_000 + HIGHS + 1_000
    line = db.one("SELECT * FROM ledger WHERE ref_id = ?", (jid,))
    assert line["amount_micros"] == -80_000
    assert line["description"] == "Denoising + Restore highs + Leveling: talk.wav (1.0 min)"
    usage = [e for e in client.get("/api/billing/history").json()["ledger"] if e["kind"] == "usage"]
    assert usage[0]["label"] == "Denoising + Restore highs + Leveling"
    assert client.get(f"/api/jobs/{jid}").json()["label"] == "Denoising + Restore highs + Leveling"


def test_a_highs_only_job_is_charged_only_highs(client, signed_in):
    """Noise tier off, Leveling off: the one line, named "Restore highs"."""
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    jid = _charge_job(client, h, uid, tier="off", leveling=False, restore_highs=True)
    done = complete_like_the_worker(jid, outputs=["talk-cleaned.flac"])
    assert done["cost_micros"] == HIGHS
    line = db.one("SELECT * FROM ledger WHERE ref_id = ?", (jid,))
    assert (line["amount_micros"], line["description"]) == (-HIGHS, "Restore highs: talk.wav (1.0 min)")
    usage = [e for e in client.get("/api/billing/history").json()["ledger"] if e["kind"] == "usage"]
    assert usage[0]["label"] == "Restore highs"


def test_a_job_pinned_before_highs_had_a_price_is_never_charged_for_it(client, signed_in):
    """Its stored rates have no highs line: even with restore_highs true in its options the worker charges the rates it has."""
    from acw.worker import main as worker

    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    jid = _charge_job(client, h, uid)  # created with its rates: denoise + leveling
    jobs.update_params(jid, options={**params_of(jid)["options"], "restore_highs": True})  # an older job's default
    assert "highs" not in params_of(jid)["rates"]
    assert worker._job_cost(row(jid), params_of(jid), 60) == 10_000
    done = complete_like_the_worker(jid, outputs=["talk-cleaned.flac"])
    assert done["cost_micros"] == 10_000
    line = db.one("SELECT * FROM ledger WHERE ref_id = ?", (jid,))
    assert (line["amount_micros"], line["description"]) == (-10_000, "Denoising + Leveling: talk.wav (1.0 min)")
    # and the same job with the highs line pinned would be charged for it
    pinned = {**params_of(jid), "rates": {"denoise": 9_000, "highs": HIGHS, "leveling": 1_000}}
    assert worker._job_cost(row(jid), pinned, 60) == 80_000


def test_worker_job_cost_follows_the_pinned_rates_for_every_combination():
    """worker._job_cost: denoise, highs, leveling each only when their rate is pinned."""
    from acw.worker import main as worker

    def cost(kind, d, rates, seconds=100.0, **opts):
        params = {"options": {"restore_highs": True, "leveling": True, **opts}, "denoise_tier": d, "transcribe_tier": None, "rates": rates}
        return worker._job_cost({"kind": kind}, params, seconds)

    assert cost("denoise", "standard", {"denoise": 9_000, "highs": HIGHS, "leveling": 1_000}) == per(9_000, 100) + per(HIGHS, 100) + per(1_000, 100)
    assert cost("denoise", "standard", {"denoise": 9_000, "leveling": 1_000}) == per(9_000, 100) + per(1_000, 100)
    assert cost("denoise", "standard", {"denoise": 9_000}) == per(9_000, 100)
    assert cost("denoise", "off", {"highs": HIGHS}) == per(HIGHS, 100)
    assert cost("denoise", "off", {"leveling": 1_000}) == per(1_000, 100)
    assert cost("denoise", "rebuild", {"denoise": 30_000, "leveling": 1_000}) == per(30_000, 100) + per(1_000, 100)
    assert cost("denoise", "rebuild", {"denoise": 30_000, "highs": HIGHS}) == per(30_000, 100)  # never for a rebuild


def test_the_admin_split_of_a_charge_shows_restore_highs():
    """admin_app._job_parts: a line per price-list item, "Restore highs" between Denoising and Leveling."""
    from acw.admin_app import _job_parts

    params = {"denoise_tier": "standard", "rates": {"denoise": 9_000, "highs": HIGHS, "leveling": 1_000}}
    parts = _job_parts("denoise", params, 80_000, 60)
    assert parts == {"Denoising": 9_000, "Restore highs": HIGHS, "Leveling": 1_000} and list(parts) == ["Denoising", "Restore highs", "Leveling"]
    assert _job_parts("denoise", {"denoise_tier": "off", "rates": {"highs": HIGHS}}, HIGHS, 60) == {"Restore highs": HIGHS}


# ------------------------------------------------------------------------------ the real worker (native)

def _sample(tmp_path_factory):

    out = tmp_path_factory.mktemp("highs-sample") / "talk.wav"
    subprocess.run([str(HOME / "bin/ffmpeg"), "-v", "error", "-i", SAMPLE, "-t", str(SECONDS), "-y", str(out)], check=True)
    return out


@pytest.fixture(scope="module")
def speech(tmp_path_factory):
    """Eight seconds of the full-band speech sample."""
    if not (HOME / "bin/ffmpeg").exists() or not Path(SAMPLE).exists():
        pytest.skip("native binaries or sample audio not available")
    return _sample(tmp_path_factory)


def _decode(path):
    f = FFmpeg(HOME / "bin")
    return f.decode(path, f.probe(path), 48000, 1, SECONDS + 5)[:, 0]


def _worker_job(client, h, data, tier="standard", **options):
    r = client.post("/api/jobs", json={
        "kind": "denoise", "denoiseTier": tier, "filename": "talk.wav", "sizeBytes": len(data), "durationSeconds": SECONDS,
        "options": {"output_video": False, "noise": "light", "audio_format": "wav", "loudness_lufs": -16.0, **options},
    }, headers=h)
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    assert client.put(f"/api/jobs/{jid}/input", content=data, headers=h).status_code == 200
    _run_worker_once("done")
    return jid


@NATIVE
def test_restore_highs_runs_on_a_full_band_recording_and_is_charged_with_its_own_line(client, signed_in, speech):
    """On by request it always runs (even on a recording that has its highs): notice, a different sound, the highs line in the
    charge. Not asked for: no notice, no line."""
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    data = speech.read_bytes()
    with_highs = _worker_job(client, h, data, restore_highs=True)
    without = _worker_job(client, h, data)
    done, plain = client.get(f"/api/jobs/{with_highs}").json(), client.get(f"/api/jobs/{without}").json()
    assert done["notice"] == "Its high frequencies were restored, as you asked." and plain["notice"] is None
    assert done["label"] == "Denoising + Restore highs + Leveling" and plain["label"] == "Denoising + Leveling"
    r = row(with_highs)
    rates = {"denoise": 9_000, "highs": HIGHS, "leveling": 1_000}
    assert done["costMicros"] == pricing.job_cost_micros("denoise", "standard", None, r["duration_seconds"], rates, 0, True, True)
    assert done["costMicros"] == per(9_000, r["duration_seconds"]) + per(HIGHS, r["duration_seconds"]) + per(1_000, r["duration_seconds"])
    assert plain["costMicros"] == pricing.job_cost_micros("denoise", "standard", None, row(without)["duration_seconds"], None, 0, True, False)
    assert done["costMicros"] > plain["costMicros"] + per(HIGHS, SECONDS) - 2
    line = db.one("SELECT * FROM ledger WHERE ref_id = ?", (with_highs,))
    assert line["description"] == "Denoising + Restore highs + Leveling: talk.wav (0.1 min)" and -line["amount_micros"] == done["costMicros"]
    a, b = _decode(folder(with_highs) / "talk-cleaned.wav"), _decode(folder(without) / "talk-cleaned.wav")
    assert abs(len(a) - len(b)) <= 1 and np.max(np.abs(a[:min(len(a), len(b))] - b[:min(len(a), len(b))])) > 1e-4  # the highs step changed the sound
    assert ledger.balance(uid)["held_micros"] == 0


@NATIVE
def test_a_highs_only_job_with_no_noise_removal_finishes_and_charges_only_highs(client, signed_in, speech):
    """Noise tier off, Leveling off, Restore highs on: a result, the notice, and exactly the highs line."""
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    jid = _worker_job(client, h, speech.read_bytes(), tier="off", leveling=False, restore_highs=True)
    done = client.get(f"/api/jobs/{jid}").json()
    assert done["status"] == "done" and done["label"] == "Restore highs"
    assert done["notice"] == "Its high frequencies were restored, as you asked."
    seconds = row(jid)["duration_seconds"]
    assert done["costMicros"] == per(HIGHS, seconds) == pricing.job_cost_micros("denoise", "off", None, seconds, {"highs": HIGHS}, 0, False, True)
    line = db.one("SELECT * FROM ledger WHERE ref_id = ?", (jid,))
    assert line["description"] == "Restore highs: talk.wav (0.1 min)" and -line["amount_micros"] == done["costMicros"]
    out = _decode(folder(jid) / "talk-cleaned.wav")
    original = _decode(speech)
    n = min(len(out), len(original))
    assert abs(len(out) - len(original)) <= SECONDS * 48 and np.max(np.abs(out[:n] - original[:n])) > 1e-4  # changed, but still the same recording
    assert np.sqrt(np.mean(out[:n] ** 2)) > 0.3 * np.sqrt(np.mean(original[:n] ** 2))  # not silence, no noise removal
