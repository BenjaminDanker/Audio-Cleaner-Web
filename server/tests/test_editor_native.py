"""The per-part editor end to end through the real worker and the sandboxed runner (acw/editing.py).

Needs the native pieces (~/audio-cleaner/bin, models) and the speech sample the other pipeline tests use; skipped otherwise. The
sample is cut to eight seconds (one block of every model grid), so blocks are cached whole while parts are billed by their own seconds."""
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from acw import db, editing, jobs
from acw.billing import ledger
from acw_common import blocks, editor
from acw_common.ffmpeg import FFmpeg
from acw_common.timeline import Timeline

from .rerun_support import SR, clock, files_of, folder, params_of, roomy_disk, row, set_env  # noqa: F401
from .test_job_pipeline import HOME, SAMPLE, _run_worker_once

NATIVE = pytest.mark.skipif(
    not (Path(SAMPLE).exists() and (HOME / "bin/libdf.so").exists() and (HOME / "bin/ffmpeg").exists()),
    reason="native binaries/models or sample audio not available",
)
SECONDS = 8
MODELS = HOME / "models"


@pytest.fixture(scope="module")
def speech(tmp_path_factory):
    out = tmp_path_factory.mktemp("editor-sample") / "talk.wav"
    subprocess.run([str(HOME / "bin/ffmpeg"), "-v", "error", "-i", SAMPLE, "-t", str(SECONDS), "-y", str(out)], check=True)
    return out


def ff() -> FFmpeg:
    return FFmpeg(HOME / "bin")


def decode(path: Path, channels=1) -> np.ndarray:
    f = ff()
    return f.decode(path, f.probe(path), SR, channels, SECONDS + 5)


def tl_json(noise=(), rebuild=(), leveling=(), echo=()) -> dict:
    def layer(parts):
        return [dict(p) for p in parts]
    layers = {k: layer(v) for k, v in (("noise", noise), ("rebuild", rebuild), ("leveling", leveling), ("echo", echo)) if v}
    return {"version": 1, "seconds": SECONDS, "layers": layers}


def noise(a, b, strength=100, remover="light"):
    return {"start": a, "end": b, "strength_db": strength, "remover": remover}


def level(a, b, lufs=-16):
    return {"start": a, "end": b, "loudness_lufs": lufs}


def create(client, h, data, timeline=None, kind="denoise", **options):
    body = {"kind": kind, "filename": "talk.wav", "sizeBytes": len(data), "durationSeconds": SECONDS,
            "options": {"output_video": False, "audio_format": "flac", **options}}
    if timeline is not None:
        body["timeline"] = timeline
    else:
        body.update(denoiseTier="standard", options={**body["options"], "noise": "light", "loudness_lufs": -16.0})
    r = client.post("/api/jobs", json=body, headers=h)
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    assert client.put(f"/api/jobs/{jid}/input", content=data, headers=h).status_code == 200
    return jid


def render_like_the_server(job_id: str, tl: dict, cache=None, *, models: bool = True) -> np.ndarray:
    """editor.render of the timeline on the job's upload (a memory cache unless one is given): what the job's result must hold.
    `models` False: no Runner, so a block the cache lacks raises."""
    from acw_common import dpdfnet

    f = ff()
    src = folder(job_id) / "input.wav"
    x = f.decode(src, f.probe(src), SR, 1, SECONDS + 5)
    inputs = editor.Inputs(x, editing.source_id({"id": job_id, "root_job_id": None}))
    runner = editor.Runner(HOME / "bin/libdf.so", MODELS / "DeepFilterNet3_onnx.tar.gz", MODELS / "dpdfnet" / dpdfnet.MODEL_FILE, MODELS / "sidon") if models else None
    return editor.render(Timeline.from_json(tl, seconds=len(x) / SR), inputs, cache if cache is not None else blocks.MemoryCache(), runner)


def usage_lines(uid):
    return db.all_rows("SELECT * FROM ledger WHERE user_id = ? AND kind = 'usage' ORDER BY id", (uid,))


@NATIVE
def test_a_job_made_with_a_timeline_only_processes_and_charges_its_parts(client, signed_in, speech, roomy_disk):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    data = speech.read_bytes()
    tl = tl_json(noise=[noise(2, 5)], leveling=[level(2, 5)])
    est = client.post("/api/jobs/estimate", json={"kind": "denoise", "durationSeconds": SECONDS, "options": {}, "timeline": tl}, headers=h)
    assert est.status_code == 200, est.text
    jid = create(client, h, data, tl)
    assert row(jid)["est_cost_micros"] == est.json()["estCostMicros"] > 0
    assert _run_worker_once("done") == jid
    done = row(jid)
    rates = editing.step_rates()
    want = editing.step_micros(rates["light"], 3, 0.27 / 0.116) + editing.step_micros(rates["leveling"], 3, 1.28 / 0.013)
    assert done["cost_micros"] == want == est.json()["estCostMicros"]
    assert [-r["amount_micros"] for r in usage_lines(uid)] == [want]
    assert {"input.wav", "talk-cleaned.flac", "kept-source.flac", "blocks", "suggestions.json"} <= files_of(jid)
    # the result is the render of the timeline (24-bit blocks and a 16-bit file: a few 1e-5)
    got = decode(folder(jid) / "talk-cleaned.flac")
    ref = render_like_the_server(jid, tl)
    assert np.max(np.abs(got[:len(ref), 0] - ref[:len(got), 0])) < 1e-4
    # outside its parts nothing was done to it: the first two seconds are the recording
    f = ff()
    x = f.decode(folder(jid) / "input.wav", f.probe(folder(jid) / "input.wav"), SR, 1, SECONDS + 5)
    assert np.max(np.abs(got[: 2 * SR - 4800, 0] - x[: 2 * SR - 4800, 0])) < 1e-4
    # the facts, the cache and the record of what the result holds
    p = params_of(jid)
    assert p["edit"]["facts"]["n"] == len(x) and p["edit"]["rendered"]["layers"]["noise"][0]["remover"] == "light"
    assert p["edit"]["suggestions"] is True and "first_run" not in p["edit"]
    shown = client.get(f"/api/jobs/{jid}/timeline").json()
    assert shown["cached"]["light"] == [[0.0, round(len(x) / SR, 3)]] and shown["custom"] is True and shown["suggestions"]["seconds"] > 0
    assert shown["timeline"]["layers"]["noise"][0]["start"] == 2.0 and shown["preview"]["samples"] == len(x)


def decode_flac(data: bytes, channels: int) -> np.ndarray:
    """The samples of a FLAC stream (a stretch is written to a pipe: its header does not say how long it is), float32 (n, channels)."""
    r = subprocess.run([str(HOME / "bin/ffmpeg"), "-v", "error", "-f", "flac", "-i", "pipe:0", "-f", "f32le", "-ac", str(channels), "-ar", str(SR), "pipe:1"],
                       input=data, capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32).reshape(-1, channels)


def stretch_of(client, jid, layer, start, seconds, tmp_path=None, channels=1, expect=200):
    """GET a preview stretch; returns (its samples divided by its scale, the response)."""
    r = client.get(f"/api/jobs/{jid}/preview/{layer}", params={"start": start, "seconds": seconds})
    assert r.status_code == expect, r.text
    if expect != 200:
        return None, r
    return decode_flac(r.content, channels) / float(r.headers["X-Stretch-Scale"]), r


@NATIVE
def test_a_job_made_as_always_fills_the_cache_and_is_edited_from_it(client, signed_in, speech, roomy_disk, tmp_path):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    data = speech.read_bytes()
    jid = create(client, h, data)  # no timeline: the whole file, per the options, priced and run as before
    assert row(jid)["est_cost_micros"] > 0 and "edit" not in params_of(jid)
    assert _run_worker_once("done") == jid
    first = row(jid)
    assert {"input.wav", "talk-cleaned.flac", "kept-noise.flac", "kept-input.flac", "kept-source.flac", "blocks", "suggestions.json"} == files_of(jid)
    assert first["cost_micros"] == pricing_whole(SECONDS)
    f = ff()
    src = folder(jid) / "input.wav"
    x = f.decode(src, f.probe(src), SR, 1, SECONDS + 5)
    shown = client.get(f"/api/jobs/{jid}/timeline").json()
    assert shown["custom"] is False and shown["cached"]["light"] == [[0.0, round(len(x) / SR, 3)]] and shown["suggestions"] is not None
    assert shown["timeline"]["layers"]["noise"] == [{"start": 0.0, "end": round(len(x) / SR, 3), "strength_db": 100.0, "remover": "light"}]
    # the editor's cache is the one the first run filled: rendering the job's timeline needs no model (no runner is given)
    cache = blocks.FlacCache(editing.blocks_dir(jid), f)
    again = render_like_the_server(jid, shown["timeline"], cache, models=False)  # (no Runner: a missing block would raise)
    assert np.max(np.abs(again[:, 0] - decode(folder(jid) / "talk-cleaned.flac")[:len(again), 0])) < 1e-4
    # the result files hold this timeline: asking for it again changes nothing
    q = client.post(f"/api/jobs/{jid}/quote", json={}, headers=h).json()
    assert q["unchanged"] is True and q["estCostMicros"] == 0
    # a softer strength without Leveling: nothing to make, so a free export of the cache
    soft = tl_json(noise=[noise(0, len(x) / SR, 24)])
    q = client.post(f"/api/jobs/{jid}/quote", json={"timeline": soft}, headers=h).json()
    assert q["estCostMicros"] == 0 and q["needsProcess"] is False and q["steps"] == {}
    spent = ledger.balance(uid)
    r = client.post(f"/api/jobs/{jid}/export", json={"timeline": soft}, headers=h)
    assert r.status_code == 200 and r.json()["free"] is True, r.text
    assert _run_worker_once("done") == jid
    assert ledger.balance(uid) == spent and len(usage_lines(uid)) == 1 and row(jid)["cost_micros"] == first["cost_micros"]
    got = decode(folder(jid) / "talk-cleaned.flac")
    ref = render_like_the_server(jid, soft)
    assert np.max(np.abs(got[:len(ref), 0] - ref[:len(got), 0])) < 1e-4
    mix = 10 ** (-24 / 20)
    cleaned = render_like_the_server(jid, tl_json(noise=[noise(0, len(x) / SR, 100)]))
    assert np.max(np.abs(got[:len(x), 0] - ((1 - mix) * cleaned[:, 0] + mix * x[:, 0]))) < 1e-4  # a true blend of the cached full-strength output
    p = params_of(jid)
    assert "kept-noise.flac" not in files_of(jid) and p["edit"]["runs"][-1]["op"] == "export" and p["kept"] == {}
    assert client.get(f"/api/jobs/{jid}").json()["redo"] == {"available": False}  # the old slider's copies no longer fit: the editor is the way
    # a heavy remover on the last three seconds: only that is made and charged (the light blocks and the recording are cached)
    mixed = tl_json(noise=[noise(0, 5, 24), noise(5, len(x) / SR, 24, "heavy")])
    q = client.post(f"/api/jobs/{jid}/quote", json={"timeline": mixed}, headers=h).json()
    secs = len(x) / SR - 5
    assert q["steps"]["heavy"]["seconds"] == math.ceil(secs - 1e-9) and "light" not in q["steps"]
    want = editing.step_micros(editing.step_rates()["heavy"], math.ceil(secs - 1e-9), 2.9 / 0.386)
    assert q["estCostMicros"] == want > 0 and q["lines"][0]["label"] == "Denoising"
    r = client.post(f"/api/jobs/{jid}/process", json={"timeline": mixed, "maxCostMicros": want}, headers=h)
    assert r.status_code == 200, r.text
    assert _run_worker_once("done") == jid
    lines = usage_lines(uid)
    assert len(lines) == 2 and -lines[1]["amount_micros"] == want and row(jid)["cost_micros"] == first["cost_micros"] + want
    got = decode(folder(jid) / "talk-cleaned.flac")
    ref = render_like_the_server(jid, mixed)
    assert np.max(np.abs(got[:len(ref), 0] - ref[:len(got), 0])) < 1e-4
    shown = client.get(f"/api/jobs/{jid}/timeline").json()
    assert shown["cached"]["heavy"] == [[0.0, round(len(x) / SR, 3)]] and shown["rendered"]["layers"]["noise"][1]["remover"] == "heavy"
    assert client.post(f"/api/jobs/{jid}/quote", json={"timeline": mixed}, headers=h).json()["unchanged"] is True
    # the preview: the recording and the cached full-strength outputs, sample aligned, with the mixing left to the page
    n = len(x)
    src_stretch, resp = stretch_of(client, jid, "source", 1.0, 2.0, tmp_path)
    assert resp.headers["X-Stretch-Start"] == str(SR) and resp.headers["X-Stretch-Samples"] == str(2 * SR) and resp.headers["X-Total-Samples"] == str(n)
    assert np.max(np.abs(src_stretch[:, 0] - x[SR:3 * SR, 0])) < 2e-7
    full_light = render_like_the_server(jid, tl_json(noise=[noise(0, n / SR, 100)]))
    full_heavy = render_like_the_server(jid, tl_json(noise=[noise(0, n / SR, 100, "heavy")]))
    light, resp = stretch_of(client, jid, "noise", 1.0, 2.0, tmp_path)  # (the timeline has the light remover there)
    assert np.max(np.abs(light[:, 0] - full_light[SR:3 * SR, 0])) < 1e-5 and resp.headers["X-Stretch-Missing"] == "[]"
    heavy, _ = stretch_of(client, jid, "noise", 6.0, 2.0, tmp_path)  # (and the heavy one here)
    assert np.max(np.abs(heavy[:, 0] - full_heavy[6 * SR:8 * SR, 0])) < 1e-5
    # one request across a part boundary holds each remover's output where the timeline says (cut short by the end of the recording)
    both, resp = stretch_of(client, jid, "noise", 4.0, 3.0, tmp_path)
    assert np.max(np.abs(both[:SR, 0] - full_light[4 * SR:5 * SR, 0])) < 1e-5 and np.max(np.abs(both[SR:, 0] - full_heavy[5 * SR:5 * SR + len(both) - SR, 0])) < 1e-5
    assert stretch_of(client, jid, "rebuild", 0, 2, tmp_path, expect=404)[1].json()["code"] == "not_cached"
    assert stretch_of(client, jid, "echo", 0, 2, tmp_path, expect=404)[1].json()["code"] == "not_cached"
    assert client.get(f"/api/jobs/{jid}/preview/source", params={"start": 100}).status_code == 400  # past the end
    assert client.get(f"/api/jobs/{jid}/preview/bass", params={"start": 0}).status_code == 404
    # suggestions on demand (a job that has none): a free run through the real runner, the audio and the cache untouched
    (folder(jid) / "suggestions.json").unlink()
    before = (files_of(jid) - {"suggestions.json"}, ledger.balance(uid), decode(folder(jid) / "talk-cleaned.flac"))
    assert client.get(f"/api/jobs/{jid}/suggestions").json() == {"suggestions": None}
    r = client.post(f"/api/jobs/{jid}/suggestions", headers=h)
    assert r.status_code == 200 and r.json()["free"] is True, r.text
    assert _run_worker_once("done") == jid
    got = client.get(f"/api/jobs/{jid}/suggestions").json()["suggestions"]
    assert got is not None and abs(got["seconds"] - n / SR) < 0.01 and got["whole"]["remover"] in ("light", "heavy")
    assert files_of(jid) - {"suggestions.json"} == before[0] and ledger.balance(uid) == before[1]
    assert np.array_equal(decode(folder(jid) / "talk-cleaned.flac"), before[2]) and params_of(jid)["edit"]["lastRun"]["op"] == "suggest"


def pricing_whole(seconds: float) -> int:
    """What a whole-file Denoising + Leveling job costs (the price list, per second rounded up)."""
    from acw.billing import pricing

    return pricing.job_cost_micros("denoise", "standard", None, seconds, {"denoise": 9_000, "leveling": 1_000}, 0, True)


def smooth(n: int, seed: int, amp: float = 0.3) -> np.ndarray:
    """Something that is different at every sample and that a block boundary in the wrong place shows up in."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * (200 + 37 * seed) * t) + 0.05 * rng.standard_normal(n)).astype(np.float32)


@NATIVE
def test_preview_stretches_line_up_sample_for_sample_for_every_layer_and_say_what_is_missing(client, signed_in, tmp_path):
    from acw_common import kept as kept_copies

    from .test_editor_api import edit_job, tl_of

    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    f = ff()
    seconds = 70.0
    n = int(seconds * SR)

    def make(plan, mid=False, ref=False):
        jid = edit_job(uid, seconds=seconds, channels=2, ref=ref, rendered="empty")
        p = params_of(jid)
        p["edit"]["facts"]["plan"], p["edit"]["facts"]["rebuild_mid"] = plan, mid
        p["edit"]["facts"]["spans"] = {str(c): [[0, n]] for c in sorted({c for c in plan if c is not None})}
        x = np.stack([smooth(n, 1), smooth(n, 2)], axis=1)
        p["edit"]["facts"]["source"] = kept_copies.save(f, x, None, folder(jid) / "kept-source")
        db.execute("UPDATE jobs SET params_json = ? WHERE id = ?", (json.dumps(p), jid))
        return jid, x

    def fill(jid, layer, truth, skip=(), remover="light", loud=None):
        """Put the blocks of the layer's whole-file plan in the job's FLAC cache: block = the truth of its channel (not the ones in `skip`)."""
        job, p = row(jid), params_of(jid)
        tl = editing.saved_timeline(job, p, n / SR)
        plan = editor.plan(editing.for_planning(editing.probe_timeline(tl, layer, remover), None)[0], editing.inputs_for(job, p), None)
        cache = blocks.FlacCache(editing.blocks_dir(jid), f)
        steps = [s for s in ("echo", "light", "heavy", "rebuild") if any(t.step == s for t in plan.tasks)]
        put = 0
        for t in plan.tasks:
            if t.step != steps[0] or t.s in skip:
                continue
            y = truth[t.channel if t.channel != "mid" else "mid"][t.s:t.e].copy()
            if loud is not None and t.s == loud:
                y *= 3.0 / max(float(np.max(np.abs(y))), 1e-9)
            cache.put(t.key, y)
            put += 1
        return put

    # the light remover's blocks of a stereo file; the block [30, 60) is not there
    jid, x = make([0, 1])
    truth = {0: smooth(n, 11), 1: smooth(n, 12)}
    assert fill(jid, "noise", truth, skip=(30 * SR,)) == 4  # two channels, blocks [0, 30) and [60, 70)
    y, r = stretch_of(client, jid, "noise", 20.0, 20.0, channels=2)
    want = np.zeros((20 * SR, 2), np.float32)
    want[:10 * SR, 0], want[:10 * SR, 1] = truth[0][20 * SR:30 * SR], truth[1][20 * SR:30 * SR]
    assert np.max(np.abs(y - want)) < 2e-7 and json.loads(r.headers["X-Stretch-Missing"]) == [[10.0, 20.0]]
    assert r.headers["X-Stretch-Layer"] == "noise" and r.headers["X-Total-Samples"] == str(n) and r.headers["X-Stretch-Start"] == str(20 * SR)
    # nothing of a stretch cached: 404 not_cached, with what is missing
    r = client.get(f"/api/jobs/{jid}/preview/noise", params={"start": 35, "seconds": 5})
    assert r.status_code == 404 and r.json()["code"] == "not_cached" and r.json()["missing"] == [[0.0, 5.0]]
    # the recording: exactly the samples (24-bit)
    src, r = stretch_of(client, jid, "source", 33.25, 3.0, channels=2)
    assert np.max(np.abs(src - x[int(33.25 * SR):int(36.25 * SR)])) < 2e-7 and r.headers["X-Stretch-Scale"] == "1.0"
    # a stretch is cut short by the end of the recording, and one past it is refused
    tail, r = stretch_of(client, jid, "source", 69.0, 10.0, channels=2)
    assert len(tail) == SR and r.headers["X-Stretch-Samples"] == str(SR)

    # Voice rebuild on its own windows (14 s steps) that do not line up with anything; one block is louder than a file can hold
    jid2, _ = make([0, 1], mid=False)
    rb = {0: smooth(n, 21), 1: smooth(n, 22)}
    blocks_made = fill(jid2, "rebuild", rb, loud=14 * SR)
    assert blocks_made >= 8
    y, r = stretch_of(client, jid2, "rebuild", 10.0, 30.0, channels=2)
    tol = 2e-7 / float(r.headers["X-Stretch-Scale"])  # (the whole stretch was brought down for the loud block: one 24-bit step counts for more)
    assert np.max(np.abs(y[:4 * SR, 0] - rb[0][10 * SR:14 * SR])) < tol and np.max(np.abs(y[:4 * SR, 1] - rb[1][10 * SR:14 * SR])) < tol
    assert np.max(np.abs(y[18 * SR:, 1] - rb[1][28 * SR:40 * SR])) < tol and float(r.headers["X-Stretch-Scale"]) < 1.0  # the loud block's scale
    loud = y[4 * SR:8 * SR, 0]
    assert 2.9 < float(np.max(np.abs(loud))) <= 3.01  # (the stretch is brought down by `scale` for the file; the page puts it back)
    assert json.loads(r.headers["X-Stretch-Missing"]) == []
    # the mix of a stereo recording whose sides are the same voice is rebuilt once: both channels get it
    jid3, _ = make([0, 1], mid=True)
    mid = {"mid": smooth(n, 31)}
    fill(jid3, "rebuild", mid)
    y, _ = stretch_of(client, jid3, "rebuild", 12.5, 9.0, channels=2)
    assert np.max(np.abs(y[:, 0] - mid["mid"][int(12.5 * SR):int(21.5 * SR)])) < 2e-7 and np.array_equal(y[:, 0], y[:, 1])
    # a file whose sides are the same recording is cleaned once and both channels play it; a silent side stays silent
    jid4, _ = make([0, 0])
    fill(jid4, "noise", {0: smooth(n, 41)})
    y, _ = stretch_of(client, jid4, "noise", 5.0, 4.0, channels=2)
    assert np.array_equal(y[:, 0], y[:, 1]) and np.max(np.abs(y[:, 0] - smooth(n, 41)[5 * SR:9 * SR])) < 2e-7
    jid5, _ = make([0, None])
    fill(jid5, "noise", {0: smooth(n, 51)})
    y, _ = stretch_of(client, jid5, "noise", 5.0, 4.0, channels=2)
    assert np.max(np.abs(y[:, 1])) == 0 and np.max(np.abs(y[:, 0] - smooth(n, 51)[5 * SR:9 * SR])) < 2e-7
    # echo removal: both channels, blocks of 30 s
    jid6, _ = make([0, 1], ref=True)
    fill(jid6, "echo", {0: smooth(n, 61), 1: smooth(n, 62)})
    y, _ = stretch_of(client, jid6, "echo", 28.0, 5.0, channels=2)
    assert np.max(np.abs(y[:, 1] - smooth(n, 62)[28 * SR:33 * SR])) < 2e-7 and np.max(np.abs(y[:, 0] - smooth(n, 61)[28 * SR:33 * SR])) < 2e-7
    # the layer a part's timeline names is the one played: a heavy part is heard from the heavy blocks
    jid7, _ = make([0, 1])
    fill(jid7, "noise", {0: smooth(n, 71), 1: smooth(n, 72)}, remover="light")
    fill(jid7, "noise", {0: smooth(n, 73), 1: smooth(n, 74)}, remover="heavy")
    client.put(f"/api/jobs/{jid7}/timeline", json={"timeline": tl_of(jid7, noise=[{"start": 10, "end": 20, "strength_db": 100, "remover": "heavy"}])}, headers=h)
    y, _ = stretch_of(client, jid7, "noise", 8.0, 4.0, channels=2)
    assert np.max(np.abs(y[:2 * SR, 0] - smooth(n, 71)[8 * SR:10 * SR])) < 2e-7 and np.max(np.abs(y[2 * SR:, 0] - smooth(n, 73)[10 * SR:12 * SR])) < 2e-7
    assert client.get(f"/api/jobs/{jid7}/timeline").json()["cached"]["heavy"] == [[0.0, 70.0]]


@pytest.mark.skipif(not (HOME / "bin/ffmpeg").exists(), reason="FFmpeg not available")
def test_the_flac_cache_hands_back_what_was_stored_to_one_24_bit_step_at_any_level(tmp_path):
    f = ff()
    cache = blocks.FlacCache(tmp_path / "b", f)
    quiet, loud = smooth(30 * SR, 5), smooth(30 * SR, 6) * 20.0  # (the loud one is scaled into a 24-bit file and back)
    short = np.array([0.25, -0.5, 0.125], np.float32)
    cache.put("quiet", quiet)
    cache.put("loud", loud)
    cache.put("short", short)
    assert np.max(np.abs(cache.get("quiet") - quiet)) < 2e-7
    assert np.max(np.abs(cache.get("loud") - loud)) < 2e-6 * 20 and float(np.max(np.abs(loud))) > 1.5
    assert np.max(np.abs(cache.get("short") - short)) < 2e-7 and cache.puts == 3
    assert cache.has("quiet") and not cache.has("nothing") and cache.get("nothing") is None
    assert (cache.path("loud").with_suffix(".scale")).exists() and not (cache.path("quiet").with_suffix(".scale")).exists()
    # a block put again replaces the old one, scale file included
    cache.put("loud", quiet)
    assert not cache.path("loud").with_suffix(".scale").exists() and np.max(np.abs(cache.get("loud") - quiet)) < 2e-7
    # without FFmpeg it can only look (the web process); a link in the place of a block does not count
    looker = blocks.FlacCache(tmp_path / "b", None, create=False)
    assert looker.has("quiet") and not looker.has("nothing")
    with pytest.raises(ValueError):
        looker.get("quiet")
    cache.path("link").symlink_to(cache.path("quiet"))
    assert cache.path("link").exists() and not cache.has("link") and not looker.has("link") and cache.get("link") is None
    assert 0 < cache.bytes_used() < 12 * 1024**2
    # no bigger than 24 bits a sample (and much smaller for speech: 2.6 MB for 30 s, against 5.8 MB of 32-bit floats)
    assert cache.path("quiet").stat().st_size < 0.75 * 4 * len(quiet)
