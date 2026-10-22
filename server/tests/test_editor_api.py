"""The per-part editor's server side without the models (acw/editing.py): timelines, quotes, prices, process and export runs, the
limits, ownership, retention, suggestions and the preview stretches. Jobs are made by hand with their facts and a block cache of
placeholder files (only the planner's `has` looks at them); the real runner is in test_editor_native.py."""
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from acw import db, editing, jobs
from acw.billing import ledger, pricing
from acw.worker import main as worker
from acw_common import blocks, editor
from acw_common.ffmpeg import FFmpeg
from acw_common.timeline import Timeline

from .job_support import (SR, account, add_job, api_key, clock, count, deadline_in, files_of, folder, make_done_job,  # noqa: F401
                            other_browser, params_of, roomy_disk, row, set_env)  # noqa: F401
from .test_job_pipeline import HOME

pytestmark = pytest.mark.usefixtures("roomy_disk")

HOUR = 3600


# ------------------------------------------------------------------------------------------------------------ helpers

def funded(client, signed_in, micros=50_000_000):
    uid, h = signed_in
    ledger.credit(uid, micros, "grant", "t", "g")
    return uid, h


def save_params(jid: str, params: dict) -> None:
    db.execute("UPDATE jobs SET params_json = ? WHERE id = ?", (json.dumps(params), jid))


def edit_job(uid, *, seconds=600.0, channels=1, api_key_id=None, ref=False, rendered="default", options=None, finished_ago=60, **kw) -> str:
    """A finished job of `seconds` with the editor's record of its recording (facts) and an empty block cache. `rendered`:
    "default" (the timeline the job was made with, on the whole file, as a first run leaves it), "empty" (nothing is done to the sound), a
    timeline JSON, or None (not recorded)."""
    jid = make_done_job(uid, seconds=seconds, channels=channels, api_key_id=api_key_id, finished_ago=finished_ago,
                        options={"noise": "light", **(options or {})}, **kw)
    n = int(round(seconds * SR))
    p = params_of(jid)
    made_with = p["edit"]["timeline"]  # (every job is made with a timeline: the one its options stand for)
    p["edit"] = {"timeline": made_with, "facts": {"n": n, "channels": channels, "plan": list(range(channels)), "spans": {str(c): [[0, n]] for c in range(channels)},
                                                  "rebuild_mid": False, "reference": ref}}
    save_params(jid, p)
    (folder(jid) / "blocks").mkdir(mode=0o700, exist_ok=True)
    if rendered == "default":
        p["edit"]["rendered"] = made_with
        save_params(jid, p)
    elif rendered == "empty":
        p["edit"]["rendered"] = Timeline(n / SR).to_json()
        save_params(jid, p)
    elif rendered is not None:
        p["edit"]["rendered"] = rendered
        save_params(jid, p)
    return jid


def tl_of(jid: str, noise=(), rebuild=(), leveling=(), echo=()) -> dict:
    seconds = editing.file_seconds(row(jid), params_of(jid))
    layers = {k: [dict(p) for p in v] for k, v in (("echo", echo), ("noise", noise), ("rebuild", rebuild), ("leveling", leveling)) if v}
    return {"version": 1, "seconds": seconds, "layers": layers}


def noise(a, b, strength=100, remover="light"):
    return {"start": a, "end": b, "strength_db": strength, "remover": remover}


def level(a, b, lufs=-16):
    return {"start": a, "end": b, "loudness_lufs": lufs}


def part(a, b):
    return {"start": a, "end": b}


def fill_cache(jid: str, tl: dict) -> int:
    """Put a placeholder block in the job's cache for every block the timeline needs; returns how many."""
    job, params = row(jid), params_of(jid)
    t, _ = editing.for_planning(Timeline.from_json(tl, seconds=editing.file_seconds(job, params)), (params["edit"].get("resolved")))
    plan = editor.plan(t, editing.inputs_for(job, params), None)
    cache = blocks.FlacCache(editing.blocks_dir(jid), None)
    for t_ in plan.tasks:
        cache.path(t_.key).write_bytes(b"fLaC")
    return len(plan.tasks)


def quote(client, h, jid, tl=None):
    r = client.post(f"/api/jobs/{jid}/quote", json={} if tl is None else {"timeline": tl}, headers=h)
    return r


def expected(step_seconds: dict[str, int]) -> int:
    """The price of whole seconds per step: rate x (seconds + the step's fixed amount in seconds), each step rounded up."""
    rates = editing.step_rates()
    return sum(editing.step_micros(rates[s], secs, editor.STEP_COSTS[s].fixed_equivalent_seconds) for s, secs in step_seconds.items())


def finish_run(jid: str, *, billed=None, rendered="empty", outputs=("talk-cleaned.flac",), suggestions=None, facts=None) -> dict:
    """The end of a run as the worker does it after the runner reported (`_complete_edit`): marks it running, writes the audio, completes it."""
    db.execute("UPDATE jobs SET status = 'running', worker_id = 't', started_at = ? WHERE id = ? AND status = 'queued'", (db.now(), jid))
    job = row(jid)
    d = folder(jid)
    for name in outputs:
        (d / name).write_bytes(b"audio:" + name.encode())
    (d / "decoded.f32").write_bytes(b"scratch")
    report = {"rendered": rendered, "billed": billed or {}, "resolved": {}}
    if suggestions is not None:
        report["suggestions"] = suggestions
    if facts is not None:
        report["facts"] = facts
    worker._complete_edit(job, d, {"ok": True, "outputs": [{"file": n, "kind": "audio"} for n in outputs], "editor": report})
    return row(jid)


def plan_billed(jid: str, tl: dict) -> dict:
    """What the runner would report having run for the timeline: the planner's billable seconds of the cache misses."""
    job, params = row(jid), params_of(jid)
    t, _ = editing.for_planning(Timeline.from_json(tl, seconds=editing.file_seconds(job, params)), params["edit"].get("resolved"))
    plan = editor.plan(t, editing.inputs_for(job, params), editing.cache_for(jid), None)
    return {s: {"seconds": b["seconds"]} for s, b in editor.billable(plan).items()}


# ----------------------------------------------------------------------------------------------------- the timeline

def test_timelines_are_checked_strictly(client, signed_in):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid)
    bad = {
        "not an object": [1, 2],
        "version": {"version": 3, "seconds": 600, "layers": {}},
        "unknown layer": {"version": 1, "seconds": 600, "layers": {"bass": [part(0, 1)]}},
        "unknown key": {"version": 1, "seconds": 600, "layers": {}, "extra": 1},
        "unknown setting": {"version": 1, "seconds": 600, "layers": {"noise": [{**noise(0, 5), "speed": 2}]}},
        "past the end": {"version": 1, "seconds": 600, "layers": {"noise": [noise(590, 601.5)]}},
        "negative start": {"version": 1, "seconds": 600, "layers": {"noise": [noise(-1, 5)]}},
        "ends before it starts": {"version": 1, "seconds": 600, "layers": {"noise": [noise(5, 5)]}},
        "bad strength": {"version": 1, "seconds": 600, "layers": {"noise": [noise(0, 5, strength=0)]}},
        "bad remover": {"version": 1, "seconds": 600, "layers": {"noise": [noise(0, 5, remover="x")]}},
        "bad loudness": {"version": 1, "seconds": 600, "layers": {"leveling": [level(0, 5, lufs=5)]}},
        "another recording": {"version": 1, "seconds": 300, "layers": {"noise": [noise(0, 5)]}},
        "text for a time": {"version": 1, "seconds": 600, "layers": {"noise": [{"start": "0", "end": 5, "strength_db": 100}]}},
        "layer not a list": {"version": 1, "seconds": 600, "layers": {"noise": {"start": 0}}},
        "too many parts": {"version": 1, "seconds": 600, "layers": {"rebuild": [part(i, i + 0.5) for i in range(0, 501)]}},
    }
    for what, tl in bad.items():
        for r in (client.put(f"/api/jobs/{jid}/timeline", json={"timeline": tl}, headers=h), quote(client, h, jid, tl)):
            assert r.status_code == 400 and r.json()["code"] == "bad_timeline" and r.headers["X-Error-Code"] == "bad_timeline", (what, r.text)
    # 500 parts are fine (the limit is per layer), and a part a hair past the end is cut to the file
    ok = {"version": 1, "seconds": 600, "layers": {"rebuild": [part(i, i + 0.5) for i in range(0, 500)], "noise": [noise(590, 600.3)]}}
    r = client.put(f"/api/jobs/{jid}/timeline", json={"timeline": ok}, headers=h)
    assert r.status_code == 200, r.text
    saved = r.json()["timeline"]
    assert len(saved["layers"]["rebuild"]) == 500 and saved["layers"]["noise"][0]["end"] == 600.0
    # the whole timeline must fit in 200 KB, whatever its parts
    huge = {"version": 1, "seconds": 600, "layers": {"noise": [{**noise(i * 0.01, i * 0.01 + 0.005), "pad": "x" * 200} for i in range(0, 400)]}}
    assert client.put(f"/api/jobs/{jid}/timeline", json={"timeline": huge}, headers=h).status_code == 400
    giant = {"version": 1, "seconds": 600, "layers": {"noise": [], "junk": "x" * 300_000}}
    assert client.put(f"/api/jobs/{jid}/timeline", json={"timeline": giant}, headers=h).status_code == 413  # (past the body limit)


def test_the_timeline_of_a_job_is_saved_returned_and_is_the_default_until_then(client, signed_in):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=120.0, options={"atten_lim_db": 24}, rendered=None)
    r = client.get(f"/api/jobs/{jid}/timeline")
    assert r.status_code == 200, r.text
    body = r.json()
    default = {"version": 1, "seconds": 120.0, "layers": {"noise": [{"start": 0.0, "end": 120.0, "strength_db": 24.0, "remover": "light"}],
                                                          "leveling": [{"start": 0.0, "end": 120.0, "loudness_lufs": -16.0}]}}
    assert body["timeline"] == default and body["rendered"] is None and "custom" not in body  # (nothing was recorded as rendered: everything is made again; every job has a timeline, so there is nothing to say about a saved one)
    assert body["seconds"] == 120.0 and body["channels"] == 1 and body["secondTrack"] is False
    assert body["cached"] == {"echo": [], "noise": [], "light": [], "heavy": [], "rebuild": []} and body["suggestions"] is None
    assert body["preview"] is None and body["run"] is None and body["hardDeadlineAt"] == jobs.hard_deadline(row(jid))
    mine = tl_of(jid, noise=[noise(10, 20.5, 40)], rebuild=[part(30, 40)], leveling=[level(0, 120, -18)])
    put = client.put(f"/api/jobs/{jid}/timeline", json={"timeline": mine}, headers=h)
    assert put.status_code == 200 and put.json()["timeline"] == mine
    again = client.get(f"/api/jobs/{jid}/timeline").json()
    assert again["timeline"] == mine and "custom" not in again and again["rendered"] is None  # (saving a timeline does not change what was rendered)
    # a part painted over another replaces it, neighbours with the same settings are one part
    messy = {"version": 1, "seconds": 120, "layers": {"noise": [noise(0, 10), noise(10, 20), noise(5, 8, 24)]}}
    saved = client.put(f"/api/jobs/{jid}/timeline", json={"timeline": messy}, headers=h).json()["timeline"]["layers"]["noise"]
    assert [(p["start"], p["end"], p["strength_db"]) for p in saved] == [(0, 5, 100), (5, 8, 24), (8, 20, 100)]


# --------------------------------------------------------------------------------------------------------- prices

def test_the_price_is_the_rate_times_the_seconds_asked_rounded_up_plus_the_fixed_amount_and_nothing_for_the_context(client, signed_in):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=600.0, rendered="empty")
    # 30.2 s of noise removal: it touches two 30 s blocks (60 s of audio is run), but only the 31 whole seconds asked are billed
    tl = tl_of(jid, noise=[noise(100, 130.2)])
    q = quote(client, h, jid, tl).json()
    assert q["estCostMicros"] == expected({"light": 31}) and q["free"] is False and q["needsProcess"] is True and q["unchanged"] is False
    assert q["lines"] == [{"item": "denoise", "label": "Denoising", "quantity": 1, "seconds": 31, "fixedSeconds": round(0.27 / 0.116, 3),
                           "usdPerMinute": 0.009, "micros": expected({"light": 31})}]
    assert q["steps"]["light"] == {"blocks": 2, "seconds": 31} and q["asked"] == {"denoise": 30.2}
    # the explicit formula, with the actual numbers: rate x (seconds + fixed / per-second cost)
    assert expected({"light": 31}) == math.ceil(9_000 * (31 + 0.27 / 0.116) / 60)
    # several steps: each is rounded up on its own, Leveling is its own line; no minimum, and a hair of a part is one second
    tl = tl_of(jid, noise=[noise(10, 10.001)], leveling=[level(0, 61.5)], rebuild=[part(200, 205)], echo=[])
    q = quote(client, h, jid, tl).json()
    want = expected({"light": 1, "leveling": 62, "rebuild": 5})
    assert q["estCostMicros"] == want == sum(line["micros"] for line in q["lines"])
    assert [(line["item"], line["seconds"]) for line in q["lines"]] == [("denoise", 1), ("rebuild", 5), ("leveling", 62)]
    # each step's own per-minute rate: Denoising, Voice rebuild, Leveling (and echo removal's, below)
    assert editing.step_rates() == {"echo": 2_600, "light": 9_000, "heavy": 9_000, "rebuild": 30_000, "leveling": 1_000}
    # a timeline without parts costs nothing and does nothing (an empty timeline is not an error for a quote)
    q = quote(client, h, jid, tl_of(jid)).json()
    assert q["estCostMicros"] == 0 and q["free"] is True and q["lines"] == []
    # the whole file: today's price plus the fixed amounts (0.4 %)
    whole = tl_of(jid, noise=[noise(0, 600)], leveling=[level(0, 600)])
    q = quote(client, h, jid, whole).json()
    today = pricing.job_cost_micros("denoise", "standard", None, 600, {"denoise": 9_000, "leveling": 1_000}, 0, True)
    assert today == 100_000 and today < q["estCostMicros"] == expected({"light": 600, "leveling": 600}) < today * 1.03  # (the fixed amounts: 2 % of ten minutes)
    hour = edit_job(uid, seconds=3600.0, rendered="empty")
    q = quote(client, h, hour, tl_of(hour, noise=[noise(0, 3600)], leveling=[level(0, 3600)])).json()
    assert 600_000 < q["estCostMicros"] < 600_000 * 1.005  # ... and 0.4 % of an hour
    # echo removal needs the file's second track
    r = quote(client, h, jid, tl_of(jid, echo=[part(0, 10)]))
    assert r.status_code == 400 and "second audio track" in r.json()["detail"]
    ref = edit_job(uid, ref=True, rendered="empty")
    q = quote(client, h, ref, tl_of(ref, echo=[part(0, 10)])).json()
    assert q["estCostMicros"] == expected({"echo": 10}) and q["lines"][0]["label"] == "Echo removal"
    # the heavy remover is Denoising too, with its own fixed amount; "auto" is decided when it runs: the quote says so
    q = quote(client, h, jid, tl_of(jid, noise=[noise(0, 10, remover="heavy")])).json()
    assert q["estCostMicros"] == expected({"heavy": 10}) and q["approximate"] is False
    q = quote(client, h, jid, tl_of(jid, noise=[noise(0, 10, remover="auto")])).json()
    assert q["approximate"] is True and q["estCostMicros"] == expected({"light": 10})


def test_voice_rebuild_wins_where_it_overlaps_noise_removal_and_only_its_part_is_billed(client, signed_in):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=600.0, rendered="empty")
    q = quote(client, h, jid, tl_of(jid, noise=[noise(0, 100)], rebuild=[part(40, 60)])).json()
    assert q["estCostMicros"] == expected({"light": 80, "rebuild": 20})  # the noise removal is not run under the rebuilt part
    assert q["steps"]["rebuild"]["seconds"] == 20 and q["steps"]["light"]["seconds"] == 80


def test_a_part_longer_than_the_rebuild_limit_is_refused_and_the_whole_file_limit_applies_to_the_file(client, signed_in):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=2 * HOUR, rendered="empty")
    r = quote(client, h, jid, tl_of(jid, rebuild=[part(0, 91 * 60)]))
    assert r.status_code == 400 and r.json()["code"] == "bad_timeline" and "90 minutes" in r.json()["detail"]
    assert quote(client, h, jid, tl_of(jid, rebuild=[part(0, 90 * 60)])).status_code == 200


# ---------------------------------------------------------------------------------------------------- the cache is free

def test_whatever_the_cache_holds_is_free_and_only_the_missing_blocks_are_billed(client, signed_in):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=600.0, rendered="empty")
    first = tl_of(jid, noise=[noise(0, 300)])
    assert fill_cache(jid, first) == 11  # ten 30 s blocks and the one the 0.2 s join margin reaches into
    q = quote(client, h, jid, first).json()
    assert q["estCostMicros"] == 0 and q["free"] is True and q["needsProcess"] is False  # (nothing rendered yet: the render is free)
    # a part inside what is cached: free; a part that reaches 10 s into a block that is not: that block's own part only
    assert quote(client, h, jid, tl_of(jid, noise=[noise(100, 200, 24)])).json()["estCostMicros"] == 0
    q = quote(client, h, jid, tl_of(jid, noise=[noise(320, 340)])).json()
    assert q["steps"]["light"] == {"blocks": 1, "seconds": 10} and q["estCostMicros"] == expected({"light": 10})
    # the other remover is another cache: its blocks are missing
    q = quote(client, h, jid, tl_of(jid, noise=[noise(0, 30, remover="heavy")])).json()
    assert q["steps"]["heavy"] == {"blocks": 2, "seconds": 30}  # (the next block is read for the join, not billed)
    # what the cache holds, per layer, as time ranges
    state = client.get(f"/api/jobs/{jid}/timeline").json()["cached"]
    assert state["light"] == [[0.0, 330.0]] and state["noise"] == [[0.0, 330.0]] and state["heavy"] == [] and state["rebuild"] == []


def test_a_strength_change_alone_is_free_and_an_export_but_leveling_on_changed_audio_is_charged(client, signed_in):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=600.0, rendered="empty")
    base = tl_of(jid, noise=[noise(0, 300, 100)])
    fill_cache(jid, base)
    p = params_of(jid)
    p["edit"]["rendered"] = Timeline.from_json(base, seconds=600.0).to_json()
    save_params(jid, p)
    # the result files hold exactly this: nothing to do
    q = quote(client, h, jid, base).json()
    assert q["unchanged"] is True and q["free"] and q["needsProcess"] is False
    r = client.post(f"/api/jobs/{jid}/export", json={"timeline": base}, headers=h)
    assert r.status_code == 400 and r.json()["code"] == "nothing_to_change"
    # a new strength: no model runs, no price, the export makes the files (and holds nothing)
    softer = tl_of(jid, noise=[noise(0, 300, 24)])
    q = quote(client, h, jid, softer).json()
    assert q["estCostMicros"] == 0 and q["free"] is True and q["unchanged"] is False and q["steps"] == {}
    balance = ledger.balance(uid)
    r = client.post(f"/api/jobs/{jid}/export", json={"timeline": softer}, headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["free"] is True and r.json()["estCostMicros"] == 0 and r.json()["run"]["op"] == "export"
    assert ledger.balance(uid) == balance and row(jid)["hold_id"] is None
    assert row(jid)["status"] == "queued" and row(jid)["next_stage"] == "edit"
    run = params_of(jid)["run"]
    assert run["timeline"]["layers"]["noise"][0]["strength_db"] == 24.0 and run["gpu"] == [] and run["previous"] == Timeline.from_json(base, seconds=600.0).to_json()
    done = finish_run(jid, rendered=softer, billed={})
    assert done["status"] == "done" and done["cost_micros"] in (0, None) and "run" not in params_of(jid)
    assert not count("ledger", "kind = 'usage'")
    assert params_of(jid)["edit"]["rendered"] == Timeline.from_json(softer, seconds=600.0).to_json()
    # with Leveling on, the audio under it changed, so the part is leveled again: that and only that is charged (and it is a process)
    rendered_now = tl_of(jid, noise=[noise(0, 300, 24)], leveling=[level(0, 300)])
    p = params_of(jid)
    p["edit"]["rendered"] = Timeline.from_json(rendered_now, seconds=600.0).to_json()
    save_params(jid, p)
    firmer = tl_of(jid, noise=[noise(0, 300, 40)], leveling=[level(0, 300)])
    q = quote(client, h, jid, firmer).json()
    assert q["estCostMicros"] == expected({"leveling": 300}) and [line["item"] for line in q["lines"]] == ["leveling"] and q["steps"].get("light") is None
    r = client.post(f"/api/jobs/{jid}/export", json={"timeline": firmer}, headers=h)
    assert r.status_code == 409 and r.json()["code"] == "needs_processing" and r.json()["quote"]["estCostMicros"] == q["estCostMicros"]
    assert row(jid)["status"] == "done"
    # leveling that is already as it was (same settings, same audio under it) is not charged again
    again = tl_of(jid, noise=[noise(0, 300, 24)], leveling=[level(0, 300)])
    assert quote(client, h, jid, again).json()["unchanged"] is True
    louder = tl_of(jid, noise=[noise(0, 300, 24)], leveling=[level(0, 300, -12)])
    assert quote(client, h, jid, louder).json()["estCostMicros"] == expected({"leveling": 300})


# --------------------------------------------------------------------------------------------------------- process

def test_process_holds_the_quote_charges_what_ran_in_one_ledger_line_per_run_and_keeps_the_results(client, signed_in):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=600.0, rendered="empty", name="talk.wav")
    tl = tl_of(jid, noise=[noise(100, 130.2)], leveling=[level(100, 130.2)])
    q = quote(client, h, jid, tl).json()
    price = expected({"light": 31, "leveling": 31})
    assert q["estCostMicros"] == price
    held_before = ledger.balance(uid)["held_micros"]
    r = client.post(f"/api/jobs/{jid}/process", json={"timeline": tl, "maxCostMicros": price}, headers=h)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["id"] == jid and out["estCostMicros"] == price == out["holdMicros"] and out["run"]["op"] == "process" and out["free"] is False
    assert ledger.balance(uid)["held_micros"] == held_before + price
    j = row(jid)
    assert j["status"] == "queued" and j["next_stage"] == "edit" and j["est_cost_micros"] == price and j["attempts"] == 0
    shown = client.get(f"/api/jobs/{jid}").json()
    assert shown["status"] == "queued" and shown["label"] == "Edited (Denoising + Leveling)" and shown["edit"]["run"] == "process"
    # the runner's report of what it ran (blocks that were missing) is what is charged, never more than was held
    billed = plan_billed(jid, tl)
    assert {s: b["seconds"] for s, b in billed.items()} == {"light": 31, "leveling": 31}
    done = finish_run(jid, billed=billed, rendered=tl)
    assert done["status"] == "done" and done["cost_micros"] == price and done["est_cost_micros"] == price
    lines = db.all_rows("SELECT * FROM ledger WHERE user_id = ? AND kind = 'usage'", (uid,))
    assert len(lines) == 1 and -lines[0]["amount_micros"] == price and lines[0]["ref_id"] == jid
    assert lines[0]["description"] == "Edited (Denoising + Leveling): talk.wav (0.5 min)"
    assert ledger.balance(uid)["held_micros"] == held_before and ledger.invariant_problems() == {"balance_mismatch": 0, "held_mismatch": 0}
    assert "decoded.f32" not in files_of(jid) and "blocks" in files_of(jid) and params_of(jid)["edit"]["runs"][-1]["op"] == "process"
    assert [o["name"] for o in client.get(f"/api/jobs/{jid}").json()["outputs"]] == ["talk-cleaned.flac"]
    # a second run is another hold, another line, and the job's cost adds up
    tl2 = tl_of(jid, noise=[noise(100, 130.2)], leveling=[level(100, 130.2)], rebuild=[part(300, 305)])
    q2 = quote(client, h, jid, tl2).json()
    assert q2["estCostMicros"] > 0
    r2 = client.post(f"/api/jobs/{jid}/process", json={"timeline": tl2}, headers=h)
    assert r2.status_code == 400 and "paused" in r2.json()["detail"] and row(jid)["status"] == "done"  # (the card is down in this test)


def test_what_ran_less_than_the_quote_is_charged_less_and_never_more(client, signed_in):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=600.0, rendered="empty")
    tl = tl_of(jid, noise=[noise(0, 60)], leveling=[level(0, 60)])
    price = expected({"light": 60, "leveling": 60})
    assert client.post(f"/api/jobs/{jid}/process", json={"timeline": tl}, headers=h).json()["holdMicros"] == price
    done = finish_run(jid, billed={"light": {"seconds": 30}, "leveling": {"seconds": 60}}, rendered=tl)  # (half the blocks were cached meanwhile)
    assert done["cost_micros"] == expected({"light": 30, "leveling": 60}) < price
    assert ledger.balance(uid)["held_micros"] == 0
    # a runner that claims more than was quoted is held to the quote
    p2 = client.post(f"/api/jobs/{jid}/process", json={"timeline": tl_of(jid, noise=[noise(0, 90)], leveling=[level(0, 90)])}, headers=h).json()["holdMicros"]
    done = finish_run(jid, billed={"light": {"seconds": 590}, "leveling": {"seconds": 590}}, rendered=tl)
    assert done["cost_micros"] - expected({"light": 30, "leveling": 60}) == p2


def test_the_price_is_refused_when_it_is_above_the_quoted_maximum_and_money_is_checked(client, signed_in):
    uid, h = funded(client, signed_in, micros=100)
    jid = edit_job(uid, seconds=600.0, rendered="empty")
    tl = tl_of(jid, noise=[noise(0, 60)])
    r = client.post(f"/api/jobs/{jid}/process", json={"timeline": tl, "maxCostMicros": 1}, headers=h)
    assert r.status_code == 409 and r.json()["code"] == "price_changed"
    r = client.post(f"/api/jobs/{jid}/process", json={"timeline": tl}, headers=h)
    assert r.status_code == 402 and "available balance" in r.json()["detail"]
    assert row(jid)["status"] == "done" and ledger.balance(uid)["held_micros"] == 0
    q = quote(client, h, jid, tl).json()
    assert q["affordable"] is False and q["availableMicros"] == 100


def test_a_run_that_fails_or_is_canceled_gives_the_money_back_and_leaves_the_job_as_it_was(client, signed_in):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=600.0, rendered="empty", name="talk.wav")
    (folder(jid) / "blocks" / "keep.flac").write_bytes(b"x")
    tl = tl_of(jid, noise=[noise(0, 60)])
    client.post(f"/api/jobs/{jid}/process", json={"timeline": tl}, headers=h)
    assert ledger.balance(uid)["held_micros"] > 0
    assert client.post(f"/api/jobs/{jid}/cancel", headers=h).status_code == 200  # still queued
    j = row(jid)
    assert j["status"] == "done" and j["hold_id"] and ledger.balance(uid)["held_micros"] == 0 and "run" not in params_of(jid)
    assert params_of(jid)["edit"]["lastRun"]["ok"] is False and params_of(jid)["edit"]["lastRun"]["error"] == "Canceled."
    assert {"input.wav", "blocks"} <= files_of(jid) and (folder(jid) / "blocks" / "keep.flac").exists()
    # a run that dies in the worker: the same
    client.post(f"/api/jobs/{jid}/process", json={"timeline": tl}, headers=h)
    db.execute("UPDATE jobs SET status = 'running', worker_id = 't' WHERE id = ?", (jid,))
    (folder(jid) / "decoded.f32").write_bytes(b"scratch")
    assert jobs.finish_failed(row(jid), "Processing failed; you were not charged.")
    assert row(jid)["status"] == "done" and ledger.balance(uid)["held_micros"] == 0 and (folder(jid) / "blocks" / "keep.flac").exists()
    assert not (folder(jid) / "decoded.f32").exists() and not count("ledger", "kind = 'usage'")
    # a worker that stops puts the run back in the queue with the job's files intact
    client.post(f"/api/jobs/{jid}/process", json={"timeline": tl}, headers=h)
    db.execute("UPDATE jobs SET status = 'running', worker_id = 't', attempts = 1 WHERE id = ?", (jid,))
    jobs.requeue(row(jid))
    assert row(jid)["status"] == "queued" and (folder(jid) / "input.wav").exists() and (folder(jid) / "blocks" / "keep.flac").exists()


def test_a_run_too_close_to_the_six_hour_end_is_refused_and_one_that_finishes_too_late_is_stopped(client, signed_in, clock):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=600.0, rendered="empty")
    tl = tl_of(jid, noise=[noise(0, 60)])
    deadline_in(jid, 200)
    r = client.post(f"/api/jobs/{jid}/process", json={"timeline": tl}, headers=h)
    assert r.status_code == 409 and r.json()["code"] == "job_expiring"
    deadline_in(jid, 3000)
    assert client.post(f"/api/jobs/{jid}/process", json={"timeline": tl}, headers=h).status_code == 200
    clock.advance(2990)  # it only gets its turn when the files are about to go
    done = finish_run(jid, billed=plan_billed(jid, tl), rendered=tl)
    assert done["status"] == "done" and ledger.balance(uid)["held_micros"] == 0 and not count("ledger", "kind = 'usage'")
    assert "weren" in params_of(jid)["edit"]["lastRun"]["error"] and params_of(jid)["edit"]["lastRun"]["ok"] is False


# ------------------------------------------------------------------------------------------------------------ limits

def test_process_runs_count_toward_the_active_job_limit_and_quotes_do_not(client, signed_in):
    uid, h = funded(client, signed_in)
    for _ in range(3):
        add_job(uid, "running")  # three jobs being processed: the account's limit
    jid = edit_job(uid, seconds=600.0, rendered="empty")
    tl = tl_of(jid, noise=[noise(0, 60)])
    for _ in range(5):
        assert quote(client, h, jid, tl).status_code == 200  # quotes, timelines and previews are not jobs
    assert client.put(f"/api/jobs/{jid}/timeline", json={"timeline": tl}, headers=h).status_code == 200
    assert client.get(f"/api/jobs/{jid}/timeline").status_code == 200
    for path, body in (("process", {"timeline": tl}), ("export", {"timeline": tl}), ("suggestions", None)):
        r = client.post(f"/api/jobs/{jid}/{path}", json=body, headers=h) if body is not None else client.post(f"/api/jobs/{jid}/{path}", headers=h)
        assert r.status_code == 429 and r.json()["code"] == "too_many_active_jobs" and r.headers["Retry-After"], (path, r.text)
    assert row(jid)["status"] == "done" and ledger.balance(uid)["held_micros"] == 0
    db.execute("UPDATE jobs SET status = 'done' WHERE status = 'running'")  # one of them is done: a process fits again
    assert client.post(f"/api/jobs/{jid}/process", json={"timeline": tl}, headers=h).status_code == 200
    # the job is busy now: no second run and no new timeline while it is
    r = client.post(f"/api/jobs/{jid}/process", json={"timeline": tl}, headers=h)
    assert r.status_code == 409 and r.json()["code"] == "job_busy"
    assert client.put(f"/api/jobs/{jid}/timeline", json={"timeline": tl}, headers=h).status_code == 409
    assert quote(client, h, jid, tl).status_code == 409


def test_an_api_key_spending_cap_holds_for_a_process(client, signed_in):
    uid, h = funded(client, signed_in)
    key, kh = api_key(uid, cap=1_000)
    jid = edit_job(uid, seconds=600.0, rendered="empty", api_key_id=key)
    tl = tl_of(jid, noise=[noise(0, 60)])
    r = client.post(f"/api/jobs/{jid}/process", json={"timeline": tl}, headers=kh)
    assert r.status_code == 403 and r.headers["X-Error-Code"] == "key_spending_cap"
    assert client.post(f"/api/jobs/{jid}/process", json={"timeline": tl}, headers=h).status_code == 200  # a browser session has no cap


# --------------------------------------------------------------------------------------------------------- ownership

@pytest.mark.parametrize("call", ["get_timeline", "put_timeline", "quote", "process", "export", "suggestions", "get_suggestions", "preview"])
def test_who_may_edit_which_job(client, signed_in, other_browser, call):
    """The account's browser session can edit any job of the account; a key only the jobs it made; anyone else gets a 404."""
    uid, h = funded(client, signed_in)
    other_client, other_uid, other_h = other_browser
    ledger.credit(other_uid, 5_000_000, "grant", "t", "g2")
    k1, kh1 = api_key(uid)
    k2, kh2 = api_key(uid)
    ok1, okh1 = api_key(other_uid)
    by_browser = edit_job(uid, rendered="empty")
    by_k1 = edit_job(uid, rendered="empty", api_key_id=k1)
    tl = tl_of(by_browser, noise=[noise(0, 5)])

    def send(caller, headers, jid):
        tl_ = {**tl}
        if call == "get_timeline":
            return caller.get(f"/api/jobs/{jid}/timeline", headers=headers)
        if call == "put_timeline":
            return caller.put(f"/api/jobs/{jid}/timeline", json={"timeline": tl_}, headers=headers)
        if call in ("quote", "process", "export"):
            return caller.post(f"/api/jobs/{jid}/{call}", json={"timeline": tl_}, headers=headers)
        if call == "suggestions":
            return caller.post(f"/api/jobs/{jid}/suggestions", headers=headers)
        if call == "get_suggestions":
            return caller.get(f"/api/jobs/{jid}/suggestions", headers=headers)
        return caller.get(f"/api/jobs/{jid}/preview/source?start=0&seconds=1", headers=headers)

    # (what comes back for an allowed call differs by call: 200, or 409/404 for the ones that need something that is not there)
    allowed_status = {r.status_code for r in (send(client, h, by_browser),)}
    assert 404 not in allowed_status or call == "preview"
    matrix = [(client, h, by_k1, True), (client, kh1, by_k1, True), (client, kh1, by_browser, False), (client, kh2, by_k1, False),
              (client, kh2, by_browser, False), (other_client, other_h, by_browser, False), (other_client, other_h, by_k1, False),
              (client, okh1, by_browser, False), (client, okh1, by_k1, False)]
    for caller, headers, jid, may in matrix:
        r = send(caller, headers, jid)
        if not may:
            assert r.status_code == 404 and r.json() == {"detail": "Job not found.", "code": "job_not_found"} and r.headers["X-Error-Code"] == "job_not_found", (
                call, headers.get("Authorization", "session")[-6:], jid == by_k1, r.status_code, r.text)
        elif call != "preview":
            assert r.status_code != 404, (call, r.text)
    for jid in ("abc", "0" * 32, "z" * 32):
        assert send(client, h, jid).status_code == 404


def test_a_key_needs_the_jobs_scope_and_a_browser_its_csrf_token(client, signed_in):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid, rendered="empty")
    _, stream_only = api_key(uid, "stream")
    tl = tl_of(jid, noise=[noise(0, 5)])
    for method, path, body in (("post", "quote", {"timeline": tl}), ("put", "timeline", {"timeline": tl}), ("post", "process", {"timeline": tl}),
                               ("post", "export", {"timeline": tl})):
        assert getattr(client, method)(f"/api/jobs/{jid}/{path}", json=body, headers=stream_only).status_code == 403
        assert getattr(client, method)(f"/api/jobs/{jid}/{path}", json=body, headers={"Origin": h["Origin"]}).status_code == 403  # no CSRF token
    client.cookies.clear()
    assert client.get(f"/api/jobs/{jid}/timeline").status_code == 401


def test_jobs_without_sound_or_files_are_not_editable(client, signed_in):
    uid, h = funded(client, signed_in)
    sub = make_done_job(uid, kind="transcribe", denoise_tier=None, transcribe_tier="local")
    r = client.get(f"/api/jobs/{sub}/timeline")  # (subtitles are a part of the timeline: a job that only writes subtitles has one)
    assert r.status_code == 200 and r.json()["timeline"]["version"] == 2 and r.json()["timeline"]["layers"]["subtitles"][0]["quality"] == "standard"
    conversion = make_done_job(uid)
    db.execute("UPDATE jobs SET kind = 'convert' WHERE id = ?", (conversion,))  # (a conversion has no timeline)
    r = client.get(f"/api/jobs/{conversion}/timeline")
    assert r.status_code == 409 and r.json()["code"] == "job_not_editable"
    gone = make_done_job(uid, status="expired", files=False)
    assert client.get(f"/api/jobs/{gone}/timeline").status_code == 409 and client.get(f"/api/jobs/{gone}/timeline").json()["code"] == "job_files_gone"
    running = add_job(uid, "running")
    r = client.post(f"/api/jobs/{running}/quote", json={}, headers=h)
    assert r.status_code == 409 and r.json()["code"] == "job_not_finished"
    assert client.get(f"/api/jobs/{running}/timeline").status_code == 200  # (the timeline of a job in progress can be read)


# ------------------------------------------------------------------------------------------------------ retention

def test_files_go_two_hours_after_the_last_edit_and_never_later_than_six_hours_after_the_upload(client, signed_in, clock):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=600.0, rendered="empty", finished_ago=0)
    up = clock.t - 30  # (make_done_job: created 30 s before it finished)
    db.execute("UPDATE jobs SET uploaded_at = ?, edited_at = ? WHERE id = ?", (up, clock.t, jid))
    assert row(jid)["expires_at"] == clock.t + 7200
    tl = tl_of(jid, noise=[noise(0, 60)])

    def deadline():
        return row(jid)["expires_at"]

    # every edit starts the two hours again: a timeline save, a quote (not a read, not a preview)
    clock.advance(3000)
    client.get(f"/api/jobs/{jid}/timeline")
    assert deadline() == clock.t - 3000 + 7200  # (reading is not an edit)
    assert client.put(f"/api/jobs/{jid}/timeline", json={"timeline": tl}, headers=h).status_code == 200
    assert deadline() == clock.t + 7200 and row(jid)["edited_at"] == clock.t
    clock.advance(4000)
    assert quote(client, h, jid, tl).status_code == 200
    assert deadline() == clock.t + 7200 and client.get(f"/api/jobs/{jid}").json()["expiresAt"] == clock.t + 7200
    # a process starts it again, and so does its end
    clock.advance(1000)
    r = client.post(f"/api/jobs/{jid}/process", json={"timeline": tl}, headers=h)
    assert r.status_code == 200 and r.json()["expiresAt"] == clock.t + 7200 and deadline() == clock.t + 7200
    clock.advance(500)
    finish_run(jid, billed=plan_billed(jid, tl), rendered=tl)
    assert deadline() == clock.t + 7200
    # ... but never later than six hours after the upload
    clock.advance(1900)
    client.put(f"/api/jobs/{jid}/timeline", json={"timeline": tl}, headers=h)
    assert deadline() == clock.t + 7200 and clock.t - up == 3000 + 4000 + 1000 + 500 + 1900 + 30  # (still room: 3 h 6 min after the upload)
    clock.advance(2 * HOUR)
    client.put(f"/api/jobs/{jid}/timeline", json={"timeline": tl}, headers=h)
    assert deadline() == up + 6 * HOUR < clock.t + 7200  # the cap bites: less than two hours are left
    clock.advance(6 * HOUR - (clock.t - up) - 600)  # ten minutes before six hours
    assert quote(client, h, jid, tl).status_code == 200
    assert deadline() == up + 6 * HOUR == clock.t + 600
    jobs.sweep()
    assert row(jid)["status"] == "done" and folder(jid).is_dir()
    clock.advance(600)
    jobs.sweep()
    assert row(jid)["status"] == "expired" and not folder(jid).exists()  # files, cache, results: all together
    assert client.get(f"/api/jobs/{jid}/timeline").json()["code"] == "job_files_gone"
    # the sweep also cuts a deadline that was given by an older rule down to the new ones
    old = edit_job(uid, finished_ago=100)
    db.execute("UPDATE jobs SET expires_at = ?, uploaded_at = ?, edited_at = NULL WHERE id = ?", (clock.t + 5 * HOUR, clock.t - 2 * HOUR, old))
    jobs.sweep()
    assert row(old)["expires_at"] == row(old)["finished_at"] + 7200


def test_the_retention_follows_the_settings(client, signed_in, clock, monkeypatch):
    uid, h = funded(client, signed_in)
    set_env(monkeypatch, ACW_JOB_RETENTION_MINUTES=30, ACW_JOB_MAX_RETENTION_MINUTES=50)
    jid = edit_job(uid, rendered="empty", finished_ago=0)
    db.execute("UPDATE jobs SET uploaded_at = ? WHERE id = ?", (clock.t - 60, jid))
    tl = tl_of(jid, noise=[noise(0, 5)])
    client.put(f"/api/jobs/{jid}/timeline", json={"timeline": tl}, headers=h)
    assert row(jid)["expires_at"] == clock.t + 1800
    clock.advance(1500)
    client.put(f"/api/jobs/{jid}/timeline", json={"timeline": tl}, headers=h)
    assert row(jid)["expires_at"] == clock.t - 1500 - 60 + 3000 and client.get("/api/jobs").json()["limits"]["maxRetentionSeconds"] == 3000


def test_an_edit_gives_that_job_alone_two_more_hours(client, signed_in, clock):
    uid, h = funded(client, signed_in)
    a = make_done_job(uid, finished_ago=0)
    b = make_done_job(uid, finished_ago=0)
    clock.advance(3600)
    jobs.touch(b)
    assert row(b)["expires_at"] == clock.t + 7200 and row(a)["expires_at"] == clock.t - 3600 + 7200  # (jobs are not chained any more)


# --------------------------------------------------------------------------------------------------- suggestions

SUGGESTIONS = {
    "seconds": 600.0, "whole": {"rebuild": False, "remover": "light", "strength": "balanced"},
    "parts": [],
    "suggestions": [
        {"start": 100.0, "end": 160.5, "kind": "part", "rebuild": False, "noise": {"strength_db": 100.0, "remover": "heavy"}, "reasons": ["is noisier"]},
        {"start": 300.0, "end": 301.2, "kind": "clipping", "rebuild": True, "noise": None, "reasons": ["sounds distorted here (clipped)"]},
        {"start": 590.0, "end": 9999.0, "kind": "part", "rebuild": False, "noise": None, "reasons": []},  # (past the end: dropped)
        {"start": 10.0, "end": 20.0, "kind": "weird", "rebuild": False, "noise": None, "reasons": []},  # (unknown kind: dropped)
    ],
}


def test_the_suggestions_of_a_run_are_checked_stored_with_the_job_and_returned(client, signed_in):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=600.0, rendered="empty")
    assert client.get(f"/api/jobs/{jid}/suggestions").json() == {"suggestions": None}
    assert client.get(f"/api/jobs/{jid}/timeline").json()["suggestions"] is None
    # on demand for a job that has none: a free run through the queue
    r = client.post(f"/api/jobs/{jid}/suggestions", headers=h)
    assert r.status_code == 200 and r.json()["run"]["op"] == "suggest" and r.json()["free"] is True and row(jid)["hold_id"] is None
    assert client.get(f"/api/jobs/{jid}").json()["label"] == "Finding suggestions"
    done = finish_run(jid, outputs=(), suggestions=SUGGESTIONS)
    assert done["status"] == "done" and not count("ledger", "kind = 'usage'") and done["cost_micros"] in (0, None)
    got = client.get(f"/api/jobs/{jid}/suggestions").json()["suggestions"]
    assert got["whole"] == {"rebuild": False, "remover": "light", "strength": "balanced"} and got["seconds"] == 600.0
    assert got["suggestions"] == [
        {"start": 100.0, "end": 160.5, "kind": "part", "rebuild": False, "noise": {"strength_db": 100.0, "remover": "heavy"}, "reasons": ["is noisier"]},
        {"start": 300.0, "end": 301.2, "kind": "clipping", "rebuild": True, "noise": None, "reasons": ["sounds distorted here (clipped)"]},
    ]
    assert client.get(f"/api/jobs/{jid}/timeline").json()["suggestions"] == got and client.get(f"/api/jobs/{jid}").json()["edit"]["suggestions"] is True
    assert (folder(jid) / "suggestions.json").exists() and oct((folder(jid) / "suggestions.json").stat().st_mode & 0o777) == "0o600"
    assert "weren" not in str(params_of(jid)["edit"].get("lastRun"))
    # the audio is untouched by a search for suggestions
    # a runner's garbage is not believed
    assert editing.sanitize_suggestions({"whole": {"rebuild": 1, "remover": "x", "strength": "balanced"}, "suggestions": []}, 600) is None
    assert editing.sanitize_suggestions("nothing", 600) is None


def test_the_suggestions_of_a_first_run_are_stored_when_the_job_finishes(client, signed_in):
    """The worker's finishing step: the runner's report (facts, suggestions) is checked, the suggestions go to their file, the facts
    to the job's record, the timeline the options stand for is what the result holds."""
    uid, h = funded(client, signed_in)
    jid = make_done_job(uid, seconds=60.0, options={"noise": "light"}, status="queued", files=True)
    db.execute("UPDATE jobs SET status = 'running', finished_at = NULL, outputs_json = NULL, hold_id = NULL, uploaded_at = ? WHERE id = ?", (db.now() - 90, jid))
    n = 60 * SR
    (folder(jid) / "kept-source.flac").write_bytes(b"fLaC")
    report = {"facts": {"n": n, "channels": 1, "plan": [0], "spans": {"0": [[0, n]]}, "rebuild_mid": False, "reference": True,
                        "source": {"file": "kept-source.flac", "channels": [0], "scale": 1.0, "samples": n}},
              "suggestions": {**SUGGESTIONS, "seconds": 60.0, "suggestions": [{**SUGGESTIONS["suggestions"][0], "start": 10.0, "end": 40.5}]}}
    job = row(jid)
    (folder(jid) / "talk-cleaned.flac").write_bytes(b"audio")
    worker._complete(job, folder(jid), {"ok": True, "outputs": [{"file": "talk-cleaned.flac", "kind": "audio"}], "editor": report})
    p = params_of(jid)
    assert row(jid)["status"] == "done" and p["edit"]["facts"]["n"] == n and p["edit"]["facts"]["source"]["scale"] == 1.0 and p["edit"]["suggestions"] is True
    assert p["edit"]["timeline"]["layers"]["noise"][0]["remover"] == "light" and p["edit"]["facts"]["reference"] is True
    assert {"kept-source.flac", "suggestions.json"} <= files_of(jid)
    assert client.get(f"/api/jobs/{jid}/suggestions").json()["suggestions"]["suggestions"][0]["noise"]["remover"] == "heavy"
    assert row(jid)["expires_at"] == row(jid)["finished_at"] + 7200
    # facts that are not sane are dropped (a copy that is not a file, channels that do not add up)
    f2 = {**report["facts"], "plan": [0, 1]}
    assert editing.sanitize_facts(f2, folder(jid)) is None
    assert "source" not in editing.sanitize_facts({**report["facts"], "source": {**report["facts"]["source"], "file": "../x"}}, folder(jid))


# --------------------------------------------------------------------------------------- a job made with a timeline

def test_creating_a_job_with_a_timeline_checks_it_prices_only_its_parts_and_replaces_the_choices_it_makes(client, signed_in):
    uid, h = funded(client, signed_in)
    tl = {"version": 1, "seconds": 600, "layers": {"noise": [noise(100, 160.5)], "leveling": [level(100, 160.5)]}}
    base = {"kind": "denoise", "filename": "talk.wav", "sizeBytes": 1000, "durationSeconds": 600, "options": {"audio_format": "flac"}}
    # the estimate and the created job agree, and are the parts' price: not the whole file's
    est = client.post("/api/jobs/estimate", json={"kind": "denoise", "durationSeconds": 600, "options": {}, "timeline": tl}, headers=h)
    assert est.status_code == 200, est.text
    want = expected({"light": 61, "leveling": 61})
    assert est.json()["estCostMicros"] == want and est.json()["timeline"] == Timeline.from_json(tl).to_json()
    whole = client.post("/api/jobs/estimate", json={"kind": "denoise", "denoiseTier": "standard", "durationSeconds": 600, "options": {}}, headers=h).json()
    assert want < whole["estCostMicros"] / 5
    r = client.post("/api/jobs", json={**base, "timeline": tl}, headers=h)
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    assert r.json()["estCostMicros"] == want and ledger.balance(uid)["held_micros"] == want
    j = row(jid)
    p = params_of(jid)
    assert j["status"] == "uploading" and j["est_cost_micros"] == want and p["edit"]["first_run"] is True and p["denoise_tier"] == "standard"
    assert p["edit"]["timeline"]["layers"]["noise"][0]["end"] == 160.5 and p["edit_rates"] == editing.step_rates()
    assert client.get(f"/api/jobs/{jid}").json()["label"] == "Voice rebuild + Denoising + Leveling".replace("Voice rebuild + ", "")
    # the subtitles are priced as always, next to the parts
    r = client.post("/api/jobs", json={**base, "kind": "denoise_transcribe", "transcribeTier": "local", "timeline": tl}, headers=h)
    assert r.status_code == 200 and r.json()["estCostMicros"] == want + pricing.job_cost_micros("transcribe", None, "local", 600, {"transcribe": 4_000}, 0)
    # what the timeline says is not said twice
    for body, code in (({"denoiseTier": "standard"}, "timeline_options"), ({"options": {"audio_format": "flac", "noise": "heavy"}}, "timeline_options"),
                       ({"options": {"leveling": False}}, "timeline_options"), ({"options": {"atten_lim_db": 24}}, "timeline_options"),
                       ({"options": {"automatic": True}}, "timeline_options"), ({"options": {"echo_reference_track": True}}, "timeline_options"),
                       ({"kind": "transcribe", "transcribeTier": "local"}, "bad_timeline"),
                       ({"timeline": {"version": 1, "seconds": 600, "layers": {}}}, "bad_timeline"),
                       ({"timeline": {"version": 1, "seconds": 600, "layers": {"noise": [noise(0, 700)]}}}, "bad_timeline"),
                       ({"timeline": {"version": 3, "seconds": 600, "layers": {}}}, "bad_timeline")):
        r = client.post("/api/jobs", json={**base, "timeline": tl, **body}, headers=h)
        assert r.status_code == 400 and r.json()["code"] == code, (body, r.text)
    # without a timeline the options become the one they stand for: the whole file, each step with its fixed amount of a run
    r = client.post("/api/jobs", json={**base, "denoiseTier": "standard"}, headers=h)
    assert r.status_code == 200 and r.json()["estCostMicros"] == whole["estCostMicros"]
    made = params_of(r.json()["id"])
    assert made["edit"]["first_run"] is True and sorted(made["edit"]["timeline"]["layers"]) == ["leveling", "noise"] and made["whole_file"]["denoise_tier"] == "standard"


def test_a_job_made_with_a_timeline_is_charged_for_what_it_ran_and_leaves_its_record(client, signed_in):
    uid, h = funded(client, signed_in)
    tl = {"version": 1, "seconds": 60, "layers": {"noise": [noise(10, 20.2)], "leveling": [level(10, 20.2)]}}
    r = client.post("/api/jobs", json={"kind": "denoise", "filename": "talk.wav", "sizeBytes": 1000, "durationSeconds": 60, "timeline": tl,
                                      "options": {"audio_format": "flac"}}, headers=h)
    jid = r.json()["id"]
    hold = r.json()["estCostMicros"]
    assert hold == expected({"light": 11, "leveling": 11})
    n = 60 * SR
    db.execute("UPDATE jobs SET status = 'running', duration_seconds = 60, uploaded_at = ?, worker_id = 't', next_stage = 'main' WHERE id = ?", (db.now(), jid))
    d = folder(jid)
    (d / "input.wav").write_bytes(b"x")
    (d / "talk-cleaned.flac").write_bytes(b"audio")
    (d / "kept-source.flac").write_bytes(b"fLaC")
    billed = {"light": {"seconds": 11}, "leveling": {"seconds": 11}}
    res = {"ok": True, "outputs": [{"file": "talk-cleaned.flac", "kind": "audio"}],
           "editor": {"billed": billed, "rendered": tl, "resolved": {}, "facts": {"n": n, "channels": 1, "plan": [0], "spans": {"0": [[0, n]]},
                                                                                  "rebuild_mid": False, "reference": False}}}
    worker._complete(row(jid), d, res)
    done = row(jid)
    assert done["status"] == "done" and done["cost_micros"] == hold == expected({"light": 11, "leveling": 11})
    lines = db.all_rows("SELECT * FROM ledger WHERE user_id = ? AND kind = 'usage'", (uid,))
    assert len(lines) == 1 and lines[0]["description"] == "Denoising + Leveling: talk.wav (1.0 min)"
    p = params_of(jid)
    assert "first_run" not in p["edit"] and p["edit"]["timeline"] == Timeline.from_json(tl).to_json() and p["edit"]["rendered"]["layers"]["leveling"]
    # the job is then edited like any other
    q = quote(client, h, jid, tl).json()
    assert q["unchanged"] is True and q["free"] is True


def test_a_leveling_only_job_made_with_a_timeline_is_named_leveling_in_its_charge_line_and_the_billing_list(client, signed_in):
    uid, h = funded(client, signed_in)
    tl = {"version": 1, "seconds": 60, "layers": {"leveling": [level(10, 20.2)]}}
    r = client.post("/api/jobs", json={"kind": "denoise", "filename": "talk.wav", "sizeBytes": 1000, "durationSeconds": 60, "timeline": tl,
                                      "options": {"audio_format": "flac"}}, headers=h)
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    assert client.get(f"/api/jobs/{jid}").json()["label"] == "Leveling"
    hold = r.json()["estCostMicros"]
    assert hold == expected({"leveling": 11})
    n = 60 * SR
    db.execute("UPDATE jobs SET status = 'running', duration_seconds = 60, uploaded_at = ?, worker_id = 't', next_stage = 'main' WHERE id = ?", (db.now(), jid))
    d = folder(jid)
    (d / "input.wav").write_bytes(b"x")
    (d / "talk-cleaned.flac").write_bytes(b"audio")
    (d / "kept-source.flac").write_bytes(b"fLaC")
    res = {"ok": True, "outputs": [{"file": "talk-cleaned.flac", "kind": "audio"}],
           "editor": {"billed": {"leveling": {"seconds": 11}}, "rendered": tl, "resolved": {},
                      "facts": {"n": n, "channels": 1, "plan": [0], "spans": {"0": [[0, n]]}, "rebuild_mid": False, "reference": False}}}
    worker._complete(row(jid), d, res)
    assert row(jid)["status"] == "done" and "first_run" not in params_of(jid)["edit"]  # the first run is over: the label no longer comes from it
    lines = db.all_rows("SELECT * FROM ledger WHERE user_id = ? AND kind = 'usage'", (uid,))
    assert len(lines) == 1 and lines[0]["description"] == "Leveling: talk.wav (1.0 min)"
    line = next(x for x in client.get("/api/billing/history").json()["ledger"] if x["kind"] == "usage")
    assert line["label"] == "Leveling" and line["displayName"] == "talk.wav"
    assert client.get(f"/api/jobs/{jid}").json()["label"] == "Leveling"


def test_the_routes_of_the_old_rerun_are_gone_and_every_job_has_its_timeline(client, signed_in):
    uid, h = funded(client, signed_in)
    plain = make_done_job(uid)
    for suffix in ("/rerun/estimate", "/rerun"):
        assert client.post(f"/api/jobs/{plain}{suffix}", json={"options": {"atten_lim_db": 24}}, headers=h).status_code in (404, 405)
    shown = client.get(f"/api/jobs/{plain}").json()
    assert "redo" not in shown and shown["edit"]["hasTimeline"] is True  # (a job asked for with options has the timeline they stand for)


def test_a_gpu_step_needs_the_card_and_is_routed_to_it(client, signed_in, gpu_up):
    from acw import broker

    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=600.0, rendered="empty")
    tl = tl_of(jid, noise=[noise(0, 30, remover="heavy")], rebuild=[part(100, 120)])
    q = quote(client, h, jid, tl).json()
    assert q["estCostMicros"] == expected({"heavy": 30, "rebuild": 20})
    r = client.post(f"/api/jobs/{jid}/process", json={"timeline": tl}, headers=h)
    assert r.status_code == 200, r.text
    assert params_of(jid)["run"]["gpu"] == ["rebuild", "heavy"] or params_of(jid)["run"]["gpu"] == ["rebuild"]
    assert broker.gpu_only(row(jid)) is True and broker.gpu_stages(row(jid))[0] == "rebuild"
    # CPU-only work (the light remover, Leveling) stays with the CPU workers
    other = edit_job(uid, seconds=600.0, rendered="empty")
    client.post(f"/api/jobs/{other}/process", json={"timeline": tl_of(other, noise=[noise(0, 30)], leveling=[level(0, 30)])}, headers=h)
    assert broker.gpu_stages(row(other)) == [] and broker.gpu_only(row(other)) is False


def test_voice_rebuild_waits_when_the_card_is_down(client, signed_in):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=600.0, rendered="empty")
    tl = tl_of(jid, rebuild=[part(100, 120)])
    r = client.post(f"/api/jobs/{jid}/process", json={"timeline": tl}, headers=h)
    assert r.status_code == 400 and "paused" in r.json()["detail"] and row(jid)["status"] == "done"
    assert quote(client, h, jid, tl).status_code == 200  # a quote does not need the card


# ---------------------------------------------------------------------------------------------- the block cache disk

def test_the_cache_is_counted_in_the_disks_promise_and_deleted_with_the_job(client, signed_in):
    from acw.config import get_settings

    uid, h = funded(client, signed_in)
    # the upload reserve counts the recording's copy and one cached step per hour of audio and channel: 0.4 + 0.35 GB, stereo
    assert editing.cache_bytes(3600, 2, 1) == 2 * (editing.SOURCE_PER_HOUR_PER_CHANNEL + editing.BLOCKS_PER_HOUR_PER_CHANNEL) and editing.cache_bytes(3600, 2, 3) > editing.cache_bytes(3600, 2, 1)
    r = client.post("/api/jobs", json={"kind": "denoise", "denoiseTier": "standard", "filename": "a.wav", "sizeBytes": 1000, "durationSeconds": 3600,
                                      "options": {}}, headers=h)
    assert r.status_code == 200
    from types import SimpleNamespace

    from acw import hostload

    need = jobs.scratch_bytes("denoise", 3600) + jobs.DISK_RESERVE + 1000
    free = need + editing.cache_bytes(3600, 2, 1) - 1
    hostload.get = lambda d: SimpleNamespace(snapshot=lambda: SimpleNamespace(disk_free=free))  # (roomy_disk's patch is undone with the test)
    r = client.post("/api/jobs", json={"kind": "denoise", "denoiseTier": "standard", "filename": "a.wav", "sizeBytes": 1000, "durationSeconds": 3600,
                                      "options": {}}, headers=h)
    assert r.status_code == 507
    # a process promises the room its blocks and scratch take
    jid = edit_job(uid, seconds=3600.0, rendered="empty")
    tl = tl_of(jid, noise=[noise(0, 3600)])
    need = jobs.scratch_bytes("denoise", 3600, 1) + editing.BLOCKS_PER_HOUR_PER_CHANNEL + jobs.DISK_RESERVE
    hostload.get = lambda d: SimpleNamespace(snapshot=lambda: SimpleNamespace(disk_free=need - 10**6))
    r = client.post(f"/api/jobs/{jid}/process", json={"timeline": tl}, headers=h)
    assert r.status_code == 507 and row(jid)["status"] == "done"
    hostload.get = lambda d: SimpleNamespace(snapshot=lambda: SimpleNamespace(disk_free=need + 10**8))
    assert client.post(f"/api/jobs/{jid}/process", json={"timeline": tl}, headers=h).status_code == 200
    db.execute("UPDATE jobs SET status = 'done' WHERE id = ?", (jid,))
    # (the folder of a job, blocks included, goes with it)
    (folder(jid) / "blocks" / "a.flac").write_bytes(b"x")
    assert editing.cache_size(jid) == 1
    assert client.delete(f"/api/jobs/{jid}", headers=h).status_code == 200 and not folder(jid).exists()
    assert get_settings().jobs_dir.exists()


# ------------------------------------------------------------------------- the first run fills the editor's cache

def test_a_whole_file_run_after_echo_removal_fills_the_blocks_under_the_editors_own_identities():
    """pipeline.clean_audio keys every block with one input id; after a whole-file echo removal the editor's identity of a block names the
    echo parts that touch what it read. The runner hands clean_audio a cache that rewrites the keys (worker.editrun.EchoWholeFile), so
    the editor's plan of the job's own timeline finds every block."""
    from acw.worker import editrun
    from acw_common import restore

    n = int(70 * SR)
    inputs = editor.Inputs(np.zeros((n, 1), np.float32), "jabc", reference=np.zeros(n, np.float32), reference_id="r")
    tl = Timeline.whole_file(70.0, echo=True, noise={"strength_db": 24, "remover": "light"}, leveling=False)
    tl2 = Timeline.whole_file(70.0, echo=True, noise=False, rebuild=True, leveling=False)
    cache = blocks.MemoryCache()
    wrapped = editrun.EchoWholeFile(cache, "jabc", n)
    mark = editrun.EchoWholeFile.MARK
    # what clean_audio does for the light remover (blocks.cache_hooks with its one input id) and for Voice rebuild
    for lo, s, e, hi in blocks.grid_windows(0, n, blocks.grid_of("light")):
        wrapped.put(blocks.block_key("light", mark, 0, lo, s, e, hi), np.zeros(e - s, np.float32))
    for b in range(-(-n // restore.block_samples())):
        wrapped.put(blocks.rebuild_key(mark, 0, b, n, restore._WINDOW_SECONDS, restore._OVERLAP_SECONDS), np.zeros(10, np.float32))
    for lo, s, e, hi in blocks.grid_windows(0, n, blocks.grid_of("echo")):
        cache.put(blocks.block_key("echo", "jabc~r", 0, lo, s, e, hi), np.zeros(e - s, np.float32))  # (remove_echo's own, no wrapper)
    assert not editor.plan(tl, inputs, cache).misses
    assert not editor.plan(tl2, inputs, cache).misses
    assert not any(mark in k for k in cache.blocks)
    # and a key that has no mark is left alone
    wrapped.put("plain", np.zeros(3, np.float32))
    assert cache.has("plain") and wrapped.has("plain") and not wrapped.has(blocks.block_key("light", mark, 0, 1, 2, 3, 4))


def test_the_credit_history_names_each_edit_run_for_its_steps_and_minutes(client, signed_in):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=600.0, rendered="empty", name="talk.wav")
    tl = tl_of(jid, noise=[noise(100, 130.2)], leveling=[level(100, 130.2)])
    client.post(f"/api/jobs/{jid}/process", json={"timeline": tl}, headers=h)
    finish_run(jid, billed=plan_billed(jid, tl), rendered=tl)
    line = next(r for r in client.get("/api/billing/history").json()["ledger"] if r["kind"] == "usage")
    assert line["label"] == "Edited (Denoising + Leveling)" and line["minutes"] == 0.5 and line["displayName"] == "talk.wav"
    assert line["product"] == "clean" and line["amount_micros"] == -expected({"light": 31, "leveling": 31})


def test_a_timeline_with_voice_rebuild_needs_the_card_and_goes_to_it(client, signed_in):
    from acw import broker

    uid, h = funded(client, signed_in)
    tl = {"version": 1, "seconds": 600, "layers": {"rebuild": [part(100, 160)], "leveling": [level(100, 160)]}}
    body = {"kind": "denoise", "filename": "talk.wav", "sizeBytes": 1000, "durationSeconds": 600, "options": {}, "timeline": tl}
    for path, send in (("/api/jobs/estimate", {k: v for k, v in body.items() if k not in ("filename", "sizeBytes")}), ("/api/jobs", body)):
        r = client.post(path, json=send, headers=h)
        assert r.status_code == 400 and "paused" in r.json()["detail"], (path, r.text)
    broker.mark_gpu_worker("alive")
    r = client.post("/api/jobs", json=body, headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["estCostMicros"] == expected({"rebuild": 60, "leveling": 60})
    assert broker.gpu_only(row(r.json()["id"])) is True and broker.gpu_stages(row(r.json()["id"])) == ["rebuild"]
    tl["layers"]["rebuild"] = [part(0, 91 * 60)]
    r = client.post("/api/jobs", json={**body, "durationSeconds": 7200, "timeline": {**tl, "seconds": 7200}}, headers=h)
    assert r.status_code == 400 and r.json()["code"] == "bad_timeline" and "90 minutes" in r.json()["detail"]


def test_a_first_run_looks_for_suggestions_up_to_the_configured_length(client, signed_in, monkeypatch):
    uid, h = signed_in
    short = make_done_job(uid, seconds=600.0)
    long = make_done_job(uid, seconds=2 * HOUR)
    spec = lambda jid: editing.spec_for(row(jid), params_of(jid))  # noqa: E731
    assert spec(short)["suggest"] is True and spec(short)["op"] == "first" and spec(short)["source_id"] == "j" + short
    assert spec(long)["suggest"] is False
    set_env(monkeypatch, ACW_SUGGEST_MAX_MINUTES=180)
    assert spec(long)["suggest"] is True
    set_env(monkeypatch, ACW_SUGGEST_MAX_MINUTES=0)
    assert spec(short)["suggest"] is False  # (0: only on request)
    sub = make_done_job(uid, kind="transcribe", denoise_tier=None, transcribe_tier="local")
    assert editing.spec_for(row(sub), params_of(sub)) is not None  # (a job with only subtitles is made with a timeline too: its subtitles part)
    assert editing.spec_for(row(sub), params_of(sub))["timeline"]["layers"]["subtitles"][0]["quality"] == "standard"
