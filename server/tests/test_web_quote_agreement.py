"""The price of saving an edit of a finished job, worked out by the website (web/src/editor/quote.js quoteLocal, from what GET /api/jobs/{id}/timeline
carries as `quote`) must be the server's (POST /api/jobs/{id}/quote, editing.make_quote) to the micro, line by line, and say the same about `free`, `unchanged`
and `approximate`. Jobs are made by hand with a block cache of placeholder files named as the real cache names them (tests/test_editor_api.py), subtitle
folders and a video made on our side. The website's side runs in node (web/tests/quote-grid.mjs, the site's own code bundled; needs web/node_modules)."""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from acw import edit_store, edit_subtitles, editing, job_convert
from acw.api import billing_routes
from acw_common import blocks
from acw_common.timeline import Timeline

from .job_support import SR, params_of, roomy_disk, row  # noqa: F401 - fixture
from .test_editor_api import edit_job, fill_cache, funded, level, noise, part, save_params, tl_of
from .test_timeline_jobs import video_job

WEB = Path(__file__).resolve().parents[2] / "web"
pytestmark = pytest.mark.usefixtures("roomy_disk")

needs_node = pytest.mark.skipif(shutil.which("node") is None or not (WEB / "node_modules/esbuild").is_dir(),
                                reason="needs node and the website's packages (npm ci in web/)")


# ---------------------------------------------------------------------------------------------------------------------------- the two sides

def server_quote(client, h, jid: str, tl: dict) -> dict:
    r = client.post(f"/api/jobs/{jid}/quote", json={"timeline": tl}, headers=h)
    if r.status_code != 200:
        return {"error": r.json().get("detail", r.text)}
    q = r.json()
    return {"micros": q["estCostMicros"], "lines": q["lines"], "free": q["free"], "unchanged": q["unchanged"], "approximate": q["approximate"]}


def _line(x: dict) -> tuple:
    if x["item"] == "convert":  # (the server's line carries the size factor and the rate; its price is what is compared)
        return ("convert", x["micros"])
    return (x["item"], x["seconds"], round(x["fixedSeconds"], 3), x["micros"])


def compare(client, h, jid: str, cases: dict[str, dict]) -> list:
    """Every case's timeline quoted by the server and by the website (given the job's own timeline answer); the cases that differ."""
    answer = client.get(f"/api/jobs/{jid}/timeline", headers=h)
    assert answer.status_code == 200, answer.text
    a = answer.json()
    fixed = billing_routes.get_pricing()["stepFixedSeconds"]
    run = subprocess.run(["node", str(WEB / "tests/quote-grid.mjs")], capture_output=True, text=True, timeout=180, cwd=WEB,
                         input=json.dumps({"cases": [{"timeline": tl, "seconds": a["seconds"], "quote": a["quote"], "rendered": a["rendered"], "stepFixed": fixed}
                                                     for tl in cases.values()]}))
    assert run.returncode == 0, run.stderr[-1200:]
    site = json.loads(run.stdout)
    wrong = []
    for (name, tl), mine in zip(cases.items(), site):
        theirs = server_quote(client, h, jid, tl)
        if "error" in theirs or "error" in mine:
            if "error" in theirs and "error" in mine:
                continue
            wrong.append((name, {"site": mine}, {"server": theirs}))
            continue
        keys = ("micros", "free", "unchanged", "approximate")
        if (sorted(_line(x) for x in mine["lines"]) != sorted(_line(x) for x in theirs["lines"])
                or any(mine[k] != theirs[k] for k in keys)):
            wrong.append((name, {"site": {k: mine[k] for k in keys}, "lines": sorted(_line(x) for x in mine["lines"])},
                          {"server": {k: theirs[k] for k in keys}, "lines": sorted(_line(x) for x in theirs["lines"])}))
    return wrong


def report(wrong: list, total: int) -> str:
    return f"{len(wrong)} of {total} differ:\n" + "\n".join(map(str, wrong))


# ------------------------------------------------------------------------------------------------------------------------------ the jobs

def sound_job(uid, rendered: dict, *, seconds=600.0, ref=True, resolved=None, **kw) -> str:
    """A finished job (two channels) that was last made as `rendered`, with every block of it in the cache."""
    jid = edit_job(uid, seconds=seconds, channels=2, ref=ref, rendered=rendered, **kw)
    p = params_of(jid)
    p["edit"]["timeline"] = rendered
    if resolved:
        p["edit"]["resolved"] = resolved
    save_params(jid, p)
    fill_cache(jid, rendered)
    return jid


def with_layers(base: dict, **changes) -> dict:
    """`base` with whole layers replaced (a layer given as None is dropped)."""
    layers = {k: [dict(x) for x in v] for k, v in base["layers"].items()}
    for k, v in changes.items():
        if v is None:
            layers.pop(k, None)
        else:
            layers[k] = v
    return {**base, "layers": layers}


def sound_cases(base: dict) -> dict[str, dict]:
    """Edits of `base` (noise light 0-300 and heavy 300-400, Voice rebuild 450-480, echo removal 20-60, Leveling on the whole file)."""
    L = base["layers"]
    n = L["noise"]
    c: dict[str, dict] = {}
    c["unchanged"] = base
    c["the same parts split in two (same settings)"] = with_layers(base, noise=[noise(0, 150, 24), noise(150, 300, 24), n[1]])
    c["nothing at all asked for"] = {**base, "layers": {}}
    c["strength changed on the whole noise part (cached: only Leveling's context changes)"] = with_layers(base, noise=[noise(0, 300, 12), n[1]])
    c["strength changed on half of a part"] = with_layers(base, noise=[noise(0, 150, 12), noise(150, 300, 24), n[1]])
    c["strength changed on everything"] = with_layers(base, noise=[noise(0, 300, 40), noise(300, 400, 40, "heavy")])
    c["Voice rebuild on a new part"] = with_layers(base, rebuild=[*L["rebuild"], part(200, 230)])
    c["Voice rebuild on a new part touching noise removal"] = with_layers(base, rebuild=[*L["rebuild"], part(280, 320)])
    c["Voice rebuild extended"] = with_layers(base, rebuild=[part(450, 500)])
    c["Voice rebuild moved"] = with_layers(base, rebuild=[part(455, 485)])
    c["Voice rebuild removed"] = with_layers(base, rebuild=None)
    c["a part's remover changed to heavy"] = with_layers(base, noise=[noise(0, 100, 24), noise(100, 150, 24, "heavy"), noise(150, 300, 24), n[1]])
    c["a heavy part's remover changed to light"] = with_layers(base, noise=[n[0], noise(300, 400, 24, "light")])
    c["a new noise part"] = with_layers(base, noise=[*n, noise(410, 440, 24), noise(500, 600, 24, "heavy")])
    c["a new noise part with fractional ends"] = with_layers(base, noise=[*n, noise(410.3, 433.7, 24)])
    c["a noise part with fractional ends inside an existing one"] = with_layers(base, noise=[*n, noise(10.3, 33.7, 24)])
    c["a noise part moved by a fraction of a second"] = with_layers(base, noise=[noise(0.4, 300.2, 24), n[1]])
    c["noise removal removed"] = with_layers(base, noise=None)
    c["echo removal added near noise blocks"] = with_layers(base, echo=[*L["echo"], part(100, 140)])
    c["echo removal added on a part of a different place"] = with_layers(base, echo=[*L["echo"], part(500, 560)])
    c["echo removal changed to another range"] = with_layers(base, echo=[part(30, 90)])
    c["echo removal removed"] = with_layers(base, echo=None)
    c["echo removal on the whole file"] = with_layers(base, echo=[part(0, 600)])
    c["Leveling changed on one part"] = with_layers(base, leveling=[level(0, 100), level(100, 200, -23), level(200, 600)])
    c["Leveling target changed on the whole file"] = with_layers(base, leveling=[level(0, 600, -23)])
    c["Leveling removed"] = with_layers(base, leveling=None)
    c["Leveling on a part only"] = with_layers(base, leveling=[level(300, 330)])
    c["Leveling on a part next to a changed noise part"] = with_layers(base, noise=[noise(0, 300, 24), noise(300, 400, 24, "light")], leveling=[level(0, 600)])
    c["several changes at once"] = with_layers(base, noise=[noise(0, 100, 24), noise(100, 200, 40, "heavy"), noise(200, 300, 24), n[1], noise(500, 520.5, 24)],
                                               rebuild=[part(450, 480), part(540, 570.25)], echo=[part(20, 60), part(150, 180)], leveling=[level(0, 250), level(250, 600, -18)])
    return c


def auto(a, b, strength=24):
    return noise(a, b, strength, "auto")


# ----------------------------------------------------------------------------------------------------------------------------- the tests

@needs_node
def test_the_website_quotes_the_edits_of_a_finished_sound_job_as_the_server_does(client, signed_in):
    uid, h = funded(client, signed_in)
    base = {"version": 1, "seconds": 600.0, "layers": {"noise": [noise(0, 300, 24), noise(300, 400, 24, "heavy")], "rebuild": [part(450, 480)],
                                                        "echo": [part(20, 60)], "leveling": [level(0, 600)]}}
    jid = sound_job(uid, base)
    cases = sound_cases(base)
    assert len(cases) >= 25
    wrong = compare(client, h, jid, cases)
    assert not wrong, report(wrong, len(cases))


@needs_node
def test_the_website_quotes_leveling_redone_only_where_its_surroundings_changed_as_the_server_does(client, signed_in):
    uid, h = funded(client, signed_in)
    lv = [level(0, 200), level(200, 400, -18), level(400, 600)]
    base = {"version": 1, "seconds": 600.0, "layers": {"noise": [noise(0, 100, 24), noise(250, 350, 24, "heavy"), noise(500, 600, 24)], "leveling": lv}}
    jid = sound_job(uid, base)
    n = base["layers"]["noise"]
    cases = {
        "unchanged": base,
        "the middle part's loudness changed": with_layers(base, leveling=[lv[0], level(200, 400, -23), lv[2]]),
        "the first part's loudness changed": with_layers(base, leveling=[level(0, 200, -23), lv[1], lv[2]]),
        "a part ends elsewhere": with_layers(base, leveling=[lv[0], level(200, 410, -18), level(410, 600)]),
        "noise strength changed far from the first part": with_layers(base, noise=[n[0], n[1], noise(500, 600, 12)]),
        "noise strength changed inside the middle part": with_layers(base, noise=[n[0], noise(250, 350, 12, "heavy"), n[2]]),
        "noise strength changed within 10 s of a part": with_layers(base, noise=[noise(0, 100, 12), n[1], n[2]]),
        "a new noise part inside the last part": with_layers(base, noise=[*n[:2], noise(450, 520, 24)]),
        "a new noise part at the very start": with_layers(base, noise=[noise(0, 10, 24), *n[1:]]),
        "noise removed from the middle": with_layers(base, noise=[n[0], n[2]]),
        "Voice rebuild added in the middle": with_layers(base, rebuild=[part(210, 240)]),
        "Voice rebuild added at the end": with_layers(base, rebuild=[part(580, 600)]),
        "echo removal added in the first part": with_layers(base, echo=[part(50, 80)]),
        "leveling removed from the middle": with_layers(base, leveling=[lv[0], lv[2]]),
        "leveling on the middle part only": with_layers(base, leveling=[lv[1]]),
        "leveling parts with fractional ends": with_layers(base, leveling=[level(0, 200.4), level(200.4, 400, -18), lv[2]]),
    }
    wrong = compare(client, h, jid, cases)
    assert not wrong, report(wrong, len(cases))


@needs_node
def test_the_website_quotes_a_job_whose_noise_removal_was_left_to_auto_as_the_server_does(client, signed_in):
    uid, h = funded(client, signed_in)
    base = {"version": 1, "seconds": 600.0, "layers": {"noise": [auto(0, 200), noise(300, 400, 24, "light")], "leveling": [level(0, 600)]}}
    # the part (0, 200) was decided as heavy when the job ran, and its blocks are the heavy remover's
    jid = sound_job(uid, base, resolved={edit_store.part_key(0, 200): "heavy"})
    n = base["layers"]["noise"]
    cases = {
        "unchanged (auto decided as heavy)": base,
        "the decided part asked for as heavy": with_layers(base, noise=[noise(0, 200, 24, "heavy"), n[1]]),
        "the decided part asked for as light": with_layers(base, noise=[noise(0, 200, 24, "light"), n[1]]),
        "the decided part's strength changed": with_layers(base, noise=[auto(0, 200, 12), n[1]]),
        "the decided part made a little longer (no longer the part that was decided)": with_layers(base, noise=[auto(0, 201), n[1]]),
        "a new part left to auto (counts as light until it is decided)": with_layers(base, noise=[*n, auto(450, 500)]),
        "every part left to auto": with_layers(base, noise=[auto(0, 200), auto(300, 400)]),
        "auto on the whole file": with_layers(base, noise=[auto(0, 600)]),
    }
    wrong = compare(client, h, jid, cases)
    assert not wrong, report(wrong, len(cases))


def sub(a, b, quality="standard", language="en", translate=()):
    return {"start": a, "end": b, "quality": quality, "language": language, "translate": list(translate)}


def make_subtitle_folder(jid: str, tl: dict, index: int, language: str, translations=()) -> None:
    t = editing.parse_timeline(tl, tl["seconds"])
    d = edit_subtitles.subtitle_dir(jid, t.layers["subtitles"][index])
    d.mkdir(parents=True, exist_ok=True)
    (d / "segments.json").write_text(json.dumps({"language": language, "segments": []}))
    for lang in translations:
        (d / f"segments.{lang}.json").write_text(json.dumps({"language": lang, "segments": []}))


@needs_node
def test_the_website_quotes_subtitle_parts_made_already_new_ones_and_added_translations_as_the_server_does(client, signed_in):
    uid, h = funded(client, signed_in)
    sound = {"noise": [noise(0, 600, 24)], "leveling": [level(0, 600)]}
    base = {"version": 2, "seconds": 600.0, "layers": {**sound, "subtitles": [sub(0, 100, "standard", None), sub(100, 200, "accurate", "en", ["es"])]}}
    jid = sound_job(uid, base)
    make_subtitle_folder(jid, base, 0, "en")  # (the language was found: the part's own language is "auto")
    make_subtitle_folder(jid, base, 1, "en", ["es"])
    p = params_of(jid)
    t = editing.parse_timeline(base, 600.0)
    p["edit"]["browser_rendered"], p["edit"]["subs_rendered"] = edit_subtitles.uploads_fingerprint(jid, t)
    save_params(jid, p)
    S = base["layers"]["subtitles"]
    cases = {
        "unchanged": base,
        "a new Standard part": with_layers(base, subtitles=[*S, sub(300, 350, "standard", "en")]),
        "a new Accurate part with two translations": with_layers(base, subtitles=[*S, sub(300, 400.5, "accurate", "en", ["es", "fr"])]),
        "a translation added to an existing part": with_layers(base, subtitles=[S[0], sub(100, 200, "accurate", "en", ["es", "fr"])]),
        "a translation added to a part with the spoken language found (auto)": with_layers(base, subtitles=[sub(0, 100, "standard", None, ["de"]), S[1]]),
        "the spoken language as the translation (made already, free)": with_layers(base, subtitles=[sub(0, 100, "standard", None, ["en"]), S[1]]),
        "a translation made already asked for again": with_layers(base, subtitles=[S[0], sub(100, 200, "accurate", "en", ["es"])]),
        "a part's language changed": with_layers(base, subtitles=[sub(0, 100, "standard", "es"), S[1]]),
        "a part's quality changed": with_layers(base, subtitles=[sub(0, 100, "accurate", None), S[1]]),
        "a part made a second longer": with_layers(base, subtitles=[sub(0, 101, "standard", None), S[1]]),
        "a part removed": with_layers(base, subtitles=[S[1]]),
        "all subtitles removed": with_layers(base, subtitles=None),
        "a free part added (made by the browser: costs nothing)": with_layers(base, subtitles=[*S, sub(400, 420, "free", "en")]),
        "a part under a second": with_layers(base, subtitles=[*S, sub(500, 500.3, "standard", "en")]),
        "subtitles and sound both changed": with_layers(base, noise=[noise(0, 600, 24, "heavy")], subtitles=[*S, sub(300, 350, "standard", "en", ["es"])]),
        "sound changed, subtitles kept": with_layers(base, noise=[noise(0, 300, 24), noise(300, 600, 24, "heavy")]),
    }
    wrong = compare(client, h, jid, cases)
    assert not wrong, report(wrong, len(cases))


@needs_node
def test_the_website_quotes_a_video_made_on_our_side_with_the_picture_made_again_as_the_server_does(client, signed_in):
    uid, h = funded(client, signed_in)
    jid = video_job(uid)
    p = params_of(jid)
    base = {"version": 1, "seconds": 600.0, "layers": {"noise": [noise(0, 600, 24)], "leveling": [level(0, 600)]}}
    p["edit"]["timeline"] = base
    p["edit"]["rendered"] = base
    save_params(jid, p)
    fill_cache(jid, base)
    assert job_convert.convert_cost(params_of(jid), 600) > 0
    cases = {
        "unchanged": base,
        "noise strength changed (the sound is made again: the video too)": with_layers(base, noise=[noise(0, 600, 12)]),
        "a part's remover changed": with_layers(base, noise=[noise(0, 300, 24), noise(300, 600, 24, "heavy")]),
        "Leveling changed": with_layers(base, leveling=[level(0, 600, -23)]),
        "a new Standard subtitles part (inside the video by default)": {"version": 2, "seconds": 600.0, "layers": {**base["layers"], "subtitles": [sub(0, 60, "standard", "en")]}},
        "subtitles removed from nothing (still unchanged)": {"version": 2, "seconds": 600.0, "layers": base["layers"]},
        "sound and subtitles": {"version": 2, "seconds": 600.0, "layers": {"noise": [noise(0, 600, 24, "heavy")], "leveling": [level(0, 600)], "subtitles": [sub(10, 70, "accurate", "en", ["es"])]}},
        "nothing at all asked for": {**base, "layers": {}},
    }
    wrong = compare(client, h, jid, cases)
    assert not wrong, report(wrong, len(cases))


@needs_node
def test_the_website_quotes_new_subtitles_of_a_video_whose_subtitles_stay_out_of_it_as_the_server_does(client, signed_in):
    uid, h = funded(client, signed_in)
    from .test_timeline_jobs import VIDEO

    jid = video_job(uid, options={**VIDEO, "embed_subtitles": False})
    p = params_of(jid)
    base = {"version": 1, "seconds": 600.0, "layers": {"noise": [noise(0, 600, 24)], "leveling": [level(0, 600)]}}
    p["edit"]["timeline"] = base
    p["edit"]["rendered"] = base
    save_params(jid, p)
    fill_cache(jid, base)
    cases = {
        "unchanged": base,
        "new subtitles (not in the video: no picture to make)": {"version": 2, "seconds": 600.0, "layers": {**base["layers"], "subtitles": [sub(0, 60, "standard", "en")]}},
        "a new sound": with_layers(base, leveling=[level(0, 600, -23)]),
    }
    wrong = compare(client, h, jid, cases)
    assert not wrong, report(wrong, len(cases))


@needs_node
def test_a_job_without_a_second_audio_track_is_quoted_by_both_without_echo_removal_and_with_the_same_words_for_it(client, signed_in):
    uid, h = funded(client, signed_in)
    base = {"version": 1, "seconds": 600.0, "layers": {"noise": [noise(0, 300, 24)], "leveling": [level(0, 600)]}}
    jid = sound_job(uid, base, ref=False)
    cases = {"unchanged": base, "echo removal asked for (the file has no second track)": with_layers(base, echo=[part(10, 50)]),
             "a new noise part": with_layers(base, noise=[noise(0, 300, 24), noise(310, 340, 24)])}
    wrong = compare(client, h, jid, cases)
    assert not wrong, report(wrong, len(cases))


def test_the_timeline_answer_of_a_finished_job_carries_what_the_website_needs_for_its_own_quote(client, signed_in):
    uid, h = funded(client, signed_in)
    base = {"version": 1, "seconds": 600.0, "layers": {"noise": [noise(0, 300, 24), noise(300, 400, 24, "heavy")], "rebuild": [part(450, 480)],
                                                        "echo": [part(20, 60)], "leveling": [level(0, 600)]}}
    jid = sound_job(uid, base, resolved={"1.000-2.000": "light"})
    d = edit_store.blocks_dir(jid)
    on_disk = sorted(x.name[:-5] for x in os.scandir(d) if x.name.endswith(".flac"))
    assert len(on_disk) > 5
    (d / "notes.txt").write_text("not a block")  # (only block files count)
    (d / ("0" * 40 + ".scale")).write_text("1.0")
    q = client.get(f"/api/jobs/{jid}/timeline", headers=h).json()["quote"]
    assert q["cached"] == on_disk
    cache = blocks.FlacCache(d, None, create=False)
    assert q["cached"] == sorted(hashlib.sha1(k.encode()).hexdigest() for k in [t.key for t in editing_plan_tasks(jid, base)]) == sorted(cache.name(t.key) for t in editing_plan_tasks(jid, base))
    f = edit_store.facts_of(row(jid), params_of(jid))
    assert q["facts"] == {"n": f["n"], "channels": 2, "plan": [0, 1], "spans": {"0": [[0, 600 * SR]], "1": [[0, 600 * SR]]}, "rebuildMid": False, "reference": True}
    assert q["sourceId"] == edit_store.source_id(row(jid)) and q["resolved"] == {"1.000-2.000": "light"}
    assert q["rates"] == {"echo": 2600, "light": 9000, "heavy": 9000, "rebuild": 30000, "leveling": 1000}
    assert set(q["subtitleRates"]) == {"transcribe_standard", "transcribe_accurate", "translate"}
    assert q["subtitles"] == {} and q["video"] == {"makes": False, "embedded": True, "convertMicros": 0, "convertLine": None}


def editing_plan_tasks(jid: str, tl: dict):
    from acw_common import editor

    job, params = row(jid), params_of(jid)
    t, _ = editing.for_planning(Timeline.from_json(tl, seconds=edit_store.file_seconds(job, params)), params["edit"].get("resolved"))
    return editor.plan(t, edit_store.inputs_for(job, params), None).tasks


def test_the_timeline_answer_lists_the_subtitle_folders_and_what_the_video_costs_to_make_again(client, signed_in):
    uid, h = funded(client, signed_in)
    jid = video_job(uid)
    base = {"version": 2, "seconds": 600.0, "layers": {"noise": [noise(0, 600, 24)], "subtitles": [sub(0, 100, "standard", None), sub(100, 200, "accurate", "en", ["es"])]}}
    p = params_of(jid)
    p["edit"]["timeline"] = base
    p["edit"]["rendered"] = base
    save_params(jid, p)
    make_subtitle_folder(jid, base, 0, "en")
    make_subtitle_folder(jid, base, 1, "en", ["es", "fr"])
    q = client.get(f"/api/jobs/{jid}/timeline", headers=h).json()["quote"]
    t = editing.parse_timeline(base, 600.0)
    k0, k1 = (edit_subtitles.subtitle_key(x) for x in t.layers["subtitles"])
    assert q["subtitles"] == {k0: {"done": True, "language": "en", "translations": []}, k1: {"done": True, "language": "en", "translations": ["es", "fr"]}}
    assert q["video"]["makes"] is True and q["video"]["embedded"] is True
    assert q["video"]["convertMicros"] == job_convert.convert_cost(params_of(jid), 600) > 0
    assert q["video"]["convertLine"] == job_convert.convert_line(params_of(jid), 600)


def test_the_timeline_answer_of_a_job_that_is_not_done_has_no_quote(client, signed_in):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid)
    from acw import db

    db.execute("UPDATE jobs SET status = 'uploading' WHERE id = ?", (jid,))
    r = client.get(f"/api/jobs/{jid}/timeline", headers=h)
    if r.status_code == 200:
        assert "quote" not in r.json()
