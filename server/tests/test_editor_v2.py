"""Timeline version 2 on the server (acw/editing.py, jobs.py, api/jobs_routes.py): the subtitles layer, the noise parts the browser cleans,
what the browser uploads for them, and the outline of a layer. Jobs are made by hand like in test_editor_api.py; the real worker is in
test_editor_v2_native.py."""
import json
import os
import struct

import pytest

from acw import db, editing, jobs
from acw.billing import ledger, pricing
from acw.worker import main as worker
from acw_common.timeline import Timeline

from .job_support import SR, account, add_job, api_key, clock, files_of, folder, make_done_job, other_browser, params_of, roomy_disk, row  # noqa: F401
from .test_editor_api import edit_job, expected, fill_cache, funded, level, noise, part, quote, save_params, tl_of

pytestmark = pytest.mark.usefixtures("roomy_disk")


def sub(a, b, quality="standard", language="en", translate=()):
    return {"start": a, "end": b, "quality": quality, "language": language, "translate": list(translate)}


def browser(a, b):
    return {"start": a, "end": b, "browser": True}


def v2(seconds, **layers):
    return {"version": 2, "seconds": seconds, "layers": {k: v for k, v in layers.items() if v}}


def flac(samples: int, rate=48000, channels=1, tail=b"") -> bytes:
    """A FLAC file's header (what the web process reads of an upload) for `samples` samples; the frames are not real."""
    info = (rate << 44) | ((channels - 1) << 41) | (23 << 36) | samples
    return b"fLaC" + b"\x80" + (34).to_bytes(3, "big") + struct.pack(">HHII", 4096, 4096, 0, 0)[:0] + b"\x10\x00\x10\x00" + b"\x00" * 6 + info.to_bytes(8, "big") + b"\x00" * 16 + tail


def rates():
    return editing.subtitle_rates()


def part_cost(rate_key, seconds):
    return editing.step_micros(rates()[rate_key], seconds, 0.0)


# ----------------------------------------------------------------------------------------------------- the format

def test_the_new_rates_are_the_price_lists():
    assert rates() == {"transcribe_standard": 4_000, "transcribe_accurate": 15_000, "translate": 4_000}


def test_a_flac_header_is_read_and_nothing_else_is_believed():
    assert editing.flac_header(flac(96_000)) == (48000, 1, 96_000)
    assert editing.flac_header(flac(5, 44100, 2)) == (44100, 2, 5)
    for junk in (b"", b"RIFF" + b"\0" * 60, flac(5)[:30], b"fLaC\x01" + flac(5)[5:]):
        with pytest.raises(ValueError):
            editing.flac_header(junk)


# ------------------------------------------------------------------------------------ estimate, create and the needs

SUBS = [sub(0, 60.4, "standard", "en", ["es", "fr"]), sub(100, 130, "accurate", None, ["de"]), sub(200, 260, "free", "en")]


def test_estimate_and_create_price_each_subtitle_part_and_free_parts_cost_nothing(client, signed_in):
    uid, h = funded(client, signed_in)
    tl = v2(600, subtitles=SUBS)
    est = client.post("/api/jobs/estimate", json={"kind": "denoise_transcribe", "durationSeconds": 600, "options": {}, "timeline": tl}, headers=h)
    assert est.status_code == 200, est.text
    # per part: the quality's rate x the whole seconds rounded up; the translations: the rate x the same seconds for each language
    want = part_cost("transcribe_standard", 61) + 2 * part_cost("translate", 61) + part_cost("transcribe_accurate", 30) + part_cost("translate", 30)
    assert est.json()["estCostMicros"] == want
    lines = {l["item"]: l for l in est.json()["lines"]}
    assert lines["transcribe"]["seconds"] == 61 and lines["transcribe"]["micros"] == part_cost("transcribe_standard", 61)
    assert lines["transcribe_accurate"]["label"] == "Transcription, extra accurate" and lines["transcribe_accurate"]["micros"] == part_cost("transcribe_accurate", 30)
    assert lines["translate"]["quantity"] == 3 and lines["translate"]["micros"] == 2 * part_cost("translate", 61) + part_cost("translate", 30)
    assert sum(l["micros"] for l in est.json()["lines"]) == want
    assert est.json()["timeline"]["version"] == 2
    body = {"kind": "denoise_transcribe", "filename": "talk.wav", "sizeBytes": 1000, "durationSeconds": 600, "options": {"audio_format": "flac"}, "timeline": tl}
    r = client.post("/api/jobs", json=body, headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["estCostMicros"] == want and ledger.balance(uid)["held_micros"] == want
    assert r.json()["needs"] == [{"kind": "cues", "start": 200.0, "end": 260.0}]  # what the browser still has to send
    jid = r.json()["id"]
    p = params_of(jid)
    assert p["transcribe_tier"] is None and p["subtitle_rates"] == rates() and p["edit"]["timeline"]["version"] == 2
    assert client.get(f"/api/jobs/{jid}").json()["needs"] == [{"kind": "cues", "start": 200.0, "end": 260.0}]
    assert client.get(f"/api/jobs/{jid}").json()["label"] == "Subtitles, extra accurate"  # (one of the parts is Accurate)
    # the same details, the same key: the repeat answers with the needs too
    again = client.post("/api/jobs", json=body, headers={**h, "Idempotency-Key": "k1"})
    repeat = client.post("/api/jobs", json=body, headers={**h, "Idempotency-Key": "k1"})
    assert repeat.json()["idempotentReplay"] is True and repeat.json()["needs"] == again.json()["needs"]


def test_a_timeline_without_the_new_things_is_written_as_version_1_and_priced_as_always(client, signed_in):
    uid, h = funded(client, signed_in)
    tl = {"version": 1, "seconds": 600, "layers": {"noise": [noise(100, 160.5)], "leveling": [level(100, 160.5)]}}
    est = client.post("/api/jobs/estimate", json={"kind": "denoise", "durationSeconds": 600, "options": {}, "timeline": tl}, headers=h).json()
    assert est["timeline"] == Timeline.from_json(tl).to_json() and est["timeline"]["version"] == 1
    r = client.post("/api/jobs", json={"kind": "denoise", "filename": "talk.wav", "sizeBytes": 1000, "durationSeconds": 600, "timeline": tl}, headers=h)
    assert r.status_code == 200 and "needs" not in r.json() and "needs" not in client.get(f"/api/jobs/{r.json()['id']}").json()
    assert "subtitle_rates" not in params_of(r.json()["id"])
    # a version 2 document that uses nothing new is written back as version 1
    t2 = Timeline.from_json({**tl, "version": 2})
    assert t2.to_json()["version"] == 1


def test_subtitles_in_a_timeline_replace_the_whole_file_subtitle_options_and_are_checked(client, signed_in):
    uid, h = funded(client, signed_in)
    base = {"kind": "denoise_transcribe", "filename": "talk.wav", "sizeBytes": 1000, "durationSeconds": 600}
    ok = v2(600, subtitles=[sub(0, 60)])
    for extra, code in (({"transcribeTier": "local"}, "timeline_options"), ({"options": {"language": "en"}}, "timeline_options"),
                        ({"options": {"translate": ["es"]}}, "timeline_options")):
        r = client.post("/api/jobs", json={**base, "timeline": ok, **extra}, headers=h)
        assert r.status_code == 400 and r.json()["code"] == code, (extra, r.text)
    # unset language and an empty translate list are not "given"
    r = client.post("/api/jobs", json={**base, "timeline": ok, "options": {"language": None, "translate": []}}, headers=h)
    assert r.status_code == 200, r.text
    bad = {
        "free with a translation": [sub(0, 10, "free", "en", ["es"])],
        "a translation into the part's language": [sub(0, 10, "standard", "en", ["en"])],
        "four translations": [sub(0, 10, "standard", "en", ["es", "fr", "de", "it"])],
        "no quality": [{"start": 0, "end": 10, "language": "en"}],
        "a quality we don't have": [sub(0, 10, "perfect")],
        "an odd language for free": [sub(0, 10, "free", "haw")],
        "a language that is not a code for standard": [sub(0, 10, "standard", "english")],
        "a language that is not a code": [sub(0, 10, "standard", 5)],
        "an unknown setting": [{**sub(0, 10), "speed": 2}],
        "translations that are not codes": [sub(0, 10, "standard", "en", ["xx"])],
    }
    for what, parts in bad.items():
        r = client.post("/api/jobs", json={**base, "timeline": v2(600, subtitles=parts)}, headers=h)
        assert r.status_code == 400 and r.json()["code"] == "bad_timeline", (what, r.text)
    assert "Standard or Accurate" in client.post("/api/jobs", json={**base, "timeline": v2(600, subtitles=bad["free with a translation"])}, headers=h).json()["detail"]
    # Standard and Accurate take the other codes of their engines (free only the site's list); a version 1 document may not have the new things; neighbours with equal settings merge
    r = client.post("/api/jobs/estimate", json={"kind": "denoise", "durationSeconds": 600, "options": {}, "timeline": v2(600, subtitles=[sub(0, 10, "accurate", "haw"), sub(10, 20, "accurate", "haw")])}, headers=h)
    assert r.status_code == 200 and r.json()["timeline"]["layers"]["subtitles"] == [{**sub(0, 20, "accurate", "haw")}]
    r = client.post("/api/jobs/estimate", json={"kind": "denoise", "durationSeconds": 600, "options": {}, "timeline": v2(600, subtitles=[sub(0, 10, "standard", "haw"), sub(10, 20, "standard", "haw")])}, headers=h)
    assert r.status_code == 200 and r.json()["timeline"]["layers"]["subtitles"] == [{**sub(0, 20, "standard", "haw")}]
    r = client.post("/api/jobs/estimate", json={"kind": "denoise", "durationSeconds": 600, "options": {}, "timeline": {**ok, "version": 1}}, headers=h)
    assert r.status_code == 400 and "version 2" in r.json()["detail"]
    r = client.post("/api/jobs/estimate", json={"kind": "denoise", "durationSeconds": 600, "options": {}, "timeline": v2(600, subtitles=[sub(0, 10)]) | {"version": 3}}, headers=h)
    assert r.status_code == 400
    # kinds: a transcribe job is made with a timeline of subtitles, and has no sound steps to put in one
    r = client.post("/api/jobs", json={**base, "kind": "transcribe", "timeline": ok}, headers=h)
    assert r.status_code == 200, r.text
    r = client.post("/api/jobs", json={**base, "kind": "transcribe", "timeline": v2(600, noise=[noise(0, 60)], subtitles=[sub(0, 60)])}, headers=h)
    assert r.status_code == 400 and r.json()["code"] == "bad_timeline" and "only has subtitles" in r.json()["detail"]


def test_an_echo_part_over_a_part_cleaned_in_the_browser_is_refused(client, signed_in):
    uid, h = funded(client, signed_in)
    tl = v2(600, noise=[browser(100, 200)], echo=[part(150, 250)])
    for body in ({"kind": "denoise", "durationSeconds": 600, "options": {}, "timeline": tl},):
        r = client.post("/api/jobs/estimate", json=body, headers=h)
        assert r.status_code == 400 and r.json()["code"] == "bad_timeline"
        assert r.json()["detail"] == "Echo removal can't be used on a part cleaned in the browser"
    ok = v2(600, noise=[browser(100, 200)], echo=[part(200, 250)])  # touching is not overlapping
    assert client.post("/api/jobs/estimate", json={"kind": "denoise", "durationSeconds": 600, "options": {}, "timeline": ok}, headers=h).status_code == 200


def test_browser_noise_parts_cost_nothing_and_the_job_waits_for_them(client, signed_in):
    uid, h = funded(client, signed_in)
    tl = v2(600, noise=[browser(100, 160.5)], leveling=[level(100, 160.5)])
    est = client.post("/api/jobs/estimate", json={"kind": "denoise", "durationSeconds": 600, "options": {}, "timeline": tl}, headers=h).json()
    assert est["estCostMicros"] == expected({"leveling": 61}) and [l["item"] for l in est["lines"]] == ["leveling"]
    assert est["timeline"]["layers"]["noise"] == [{"start": 100.0, "end": 160.5, "browser": True}]  # no strength, no remover
    # the strength the person chose is kept for the page and used for nothing (not the price, not the work); the remover is not kept
    again = client.post("/api/jobs/estimate", json={"kind": "denoise", "durationSeconds": 600, "options": {},
                                                    "timeline": v2(600, noise=[{**browser(100, 160.5), "strength_db": 24, "remover": "heavy"}], leveling=[level(100, 160.5)])}, headers=h)
    assert again.json()["timeline"]["layers"]["noise"] == [{"start": 100.0, "end": 160.5, "browser": True, "strength_db": 24.0}]
    assert again.json()["estCostMicros"] == est["estCostMicros"] and again.json()["lines"] == est["lines"]
    # neighbours with different strengths are separate parts, each of them uploaded on its own
    two = client.post("/api/jobs", json={"kind": "denoise", "filename": "talk.wav", "sizeBytes": 1000, "durationSeconds": 600, "timeline": v2(600, noise=[
        {**browser(10, 20), "strength_db": 12}, {**browser(20, 30), "strength_db": 40}, {**browser(30, 40), "strength_db": 40}])}, headers=h)
    assert two.status_code == 200 and two.json()["needs"] == [{"kind": "noise", "start": 10.0, "end": 20.0}, {"kind": "noise", "start": 20.0, "end": 40.0}]
    assert params_of(two.json()["id"])["edit"]["timeline"]["layers"]["noise"][1] == {"start": 20.0, "end": 40.0, "browser": True, "strength_db": 40.0}
    # a browser part with nothing else: nothing to pay
    only = client.post("/api/jobs/estimate", json={"kind": "denoise", "durationSeconds": 600, "options": {}, "timeline": v2(600, noise=[browser(10, 20)])}, headers=h).json()
    assert only["estCostMicros"] == 0 and only["lines"] == []
    r = client.post("/api/jobs", json={"kind": "denoise", "filename": "talk.wav", "sizeBytes": 1000, "durationSeconds": 600, "timeline": tl}, headers=h)
    assert r.status_code == 200 and r.json()["needs"] == [{"kind": "noise", "start": 100.0, "end": 160.5}]


# ----------------------------------------------------------------------------------------- uploads and the queue

def make_uploading(client, h, timeline, kind="denoise", seconds=600, size=1000):
    r = client.post("/api/jobs", json={"kind": kind, "filename": "talk.wav", "sizeBytes": size, "durationSeconds": seconds, "timeline": timeline,
                                      "options": {"audio_format": "flac"}}, headers=h)
    assert r.status_code == 200, r.text
    return r.json()["id"], r.json().get("needs", [])


def put_source(client, h, jid, size=1000):
    return client.put(f"/api/jobs/{jid}/input", content=b"x" * size, headers=h)


def put_cues(client, h, jid, a, b, cues, **kw):
    return client.put(f"/api/jobs/{jid}/browser/cues", params={"start": a, "end": b}, content=json.dumps(cues), headers=h, **kw)


def put_noise(client, h, jid, a, b, data):
    return client.put(f"/api/jobs/{jid}/browser/noise", params={"start": a, "end": b}, content=data, headers=h)


def test_a_job_is_queued_only_when_the_file_and_everything_the_browser_makes_are_in(client, signed_in):
    uid, h = funded(client, signed_in)
    tl = v2(600, noise=[browser(10, 12)], subtitles=[sub(20, 30, "free", "en"), sub(40, 50, "standard", "en")], leveling=[level(0, 5)])
    jid, needs = make_uploading(client, h, tl)
    assert needs == [{"kind": "noise", "start": 10.0, "end": 12.0}, {"kind": "cues", "start": 20.0, "end": 30.0}]
    # the file first: the job waits (its upload window is restarted), and says what it still needs
    assert put_source(client, h, jid).status_code == 200
    j = row(jid)
    assert j["status"] == "uploading" and j["uploaded_at"] is not None and j["expires_at"] > db.now() + 3000
    assert client.get(f"/api/jobs/{jid}").json()["needs"] == needs
    # the cues: checked, stored with the job
    assert put_cues(client, h, jid, 20, 30, [{"start": 20.5, "end": 22, "text": "Hello  there"}, {"start": 25, "end": 26, "text": "  "}]).status_code == 200
    stored = json.loads((editing.browser_dir(jid) / "cues-20000-30000.json").read_text())
    assert stored == {"language": "en", "cues": [{"start": 20.5, "end": 22.0, "text": "Hello there"}]}
    assert row(jid)["status"] == "uploading" and client.get(f"/api/jobs/{jid}").json()["needs"] == [needs[0]]
    # the cleaned audio is the last thing: now the job is queued like any other
    assert put_noise(client, h, jid, 10, 12, flac(96_000)).status_code == 200
    j = row(jid)
    assert j["status"] == "queued" and j["next_stage"] == "probe" and client.get(f"/api/jobs/{jid}").json()["needs"] == []
    assert files_of(jid) >= {"input.wav", "browser"} and (editing.browser_dir(jid) / "noise-10000-12000.flac").is_file()
    # a queued job takes no more (it may be running on them)
    r = put_noise(client, h, jid, 10, 12, flac(96_000))
    assert r.status_code == 409 and r.json()["code"] == "job_busy"
    # the other order: the pieces first, the file last
    other, needs = make_uploading(client, h, tl)
    assert put_cues(client, h, other, 20, 30, []).status_code == 200 and put_noise(client, h, other, 10, 12, flac(96_000)).status_code == 200
    assert row(other)["status"] == "uploading"
    assert put_source(client, h, other).status_code == 200 and row(other)["status"] == "queued"


def test_a_job_without_such_parts_is_queued_by_its_file_alone_as_always(client, signed_in):
    uid, h = funded(client, signed_in)
    jid, needs = make_uploading(client, h, {"version": 1, "seconds": 600, "layers": {"noise": [noise(0, 30)]}})
    assert needs == [] and put_source(client, h, jid).status_code == 200
    assert row(jid)["status"] == "queued"
    plain = client.post("/api/jobs", json={"kind": "denoise", "denoiseTier": "standard", "filename": "talk.wav", "sizeBytes": 1000, "durationSeconds": 60}, headers=h).json()["id"]
    assert put_source(client, h, plain).status_code == 200 and row(plain)["status"] == "queued"
    assert put_cues(client, h, plain, 0, 10, []).status_code == 409  # (queued already: nothing more is taken)


def test_an_upload_must_match_a_part_of_the_current_timeline_exactly(client, signed_in):
    uid, h = funded(client, signed_in)
    jid, _ = make_uploading(client, h, v2(600, noise=[browser(10, 12)], subtitles=[sub(20, 30, "free", "en"), sub(40, 50, "standard")]))
    assert put_source(client, h, jid).status_code == 200
    cases = [
        ("noise: another range", put_noise(client, h, jid, 10, 12.5, flac(120_000))),
        ("noise: a part that is not a browser one", put_noise(client, h, jid, 40, 50, flac(480_000))),
        ("cues: a standard part", put_cues(client, h, jid, 40, 50, [])),
        ("cues: another range", put_cues(client, h, jid, 20, 31, [])),
        ("noise: not 48 kHz", put_noise(client, h, jid, 10, 12, flac(88_200, rate=44100))),
        ("noise: too short", put_noise(client, h, jid, 10, 12, flac(95_000))),
        ("noise: three channels", put_noise(client, h, jid, 10, 12, flac(96_000, channels=3))),
        ("noise: not FLAC", put_noise(client, h, jid, 10, 12, b"RIFF" + b"\0" * 100)),
        ("cues: not JSON", put_cues(client, h, jid, 20, 30, [])),
    ]
    for what, r in cases[:-1]:
        assert r.status_code == 400, (what, r.text)
    assert client.put(f"/api/jobs/{jid}/browser/cues", params={"start": 20, "end": 30}, content=b"not json", headers=h).status_code == 400
    for body in ({"start": 21, "end": 22, "text": "x"}, {"start": 20, "end": 19, "text": "x"}, {"start": 20, "end": 22, "text": 5}, {"start": 20, "end": 22},
                 {"start": 20, "end": 22, "text": "x", "extra": 1}):
        assert put_cues(client, h, jid, 20, 30, [body]).status_code == (200 if body == {"start": 21, "end": 22, "text": "x"} else 400), body
    assert put_cues(client, h, jid, 20, 30, {"not": "a list"}).status_code == 400
    # a cue a hair past the part is cut to it; far outside is refused
    assert put_cues(client, h, jid, 20, 30, [{"start": 29.8, "end": 30.3, "text": "end"}]).status_code == 200
    assert json.loads((editing.browser_dir(jid) / "cues-20000-30000.json").read_text())["cues"] == [{"start": 29.8, "end": 30.0, "text": "end"}]
    assert put_cues(client, h, jid, 20, 30, [{"start": 31, "end": 32, "text": "late"}]).status_code == 400
    # sizes: the cues are small JSON (the body limit answers 413), the audio can not be more than a FLAC of the part can be
    assert client.put(f"/api/jobs/{jid}/browser/cues", params={"start": 20, "end": 30}, content=b" " * (1024 * 1024 + 10), headers=h).status_code == 413
    big = client.put(f"/api/jobs/{jid}/browser/noise", params={"start": 10, "end": 12}, content=flac(96_000, tail=b"\0" * (2 * 96_000 * 8 + 2_000_000)), headers=h)
    assert big.status_code == 413
    assert not [p for p in editing.browser_dir(jid).iterdir() if p.name.endswith(".tmp")]
    # nobody else's, and nothing unknown
    assert put_cues(client, h, "0" * 32, 20, 30, []).status_code == 404
    assert client.put(f"/api/jobs/{jid}/browser/other", content=b"x", headers=h).status_code == 404


def test_other_accounts_and_keys_and_scopes_are_kept_out(client, signed_in, other_browser):
    uid, h = funded(client, signed_in)
    jid, _ = make_uploading(client, h, v2(600, subtitles=[sub(20, 30, "free", "en")]))
    oc, ouid, oh = other_browser
    assert put_cues(oc, oh, jid, 20, 30, []).status_code == 404
    key_id, kh = api_key(uid, "stream")
    assert client.put(f"/api/jobs/{jid}/browser/cues", params={"start": 20, "end": 30}, content=b"[]", headers=kh).status_code == 403
    key_id, kh = api_key(uid, "jobs")
    assert client.put(f"/api/jobs/{jid}/browser/cues", params={"start": 20, "end": 30}, content=b"[]", headers=kh).status_code == 404  # (a key: its own jobs)


def test_the_browser_uploads_have_their_own_rate_limit(client, signed_in, monkeypatch):
    from acw.security import LIMITS

    uid, h = funded(client, signed_in)
    jid, _ = make_uploading(client, h, v2(600, subtitles=[sub(20, 30, "free", "en")]))
    assert LIMITS["browser_upload"][0] >= 120 and LIMITS["upload"][0] == 30  # (a timeline has many pieces; the file upload keeps its 30 an hour)
    monkeypatch.setitem(LIMITS, "browser_upload", (3, 0.0))
    for _ in range(3):
        assert put_cues(client, h, jid, 20, 30, []).status_code == 200
    r = put_cues(client, h, jid, 20, 30, [])
    assert r.status_code == 429 and "Retry-After" in r.headers


# ----------------------------------------------------------------------------- a finished job: quote, process, needs

def tl_with(jid, **layers):
    seconds = editing.file_seconds(row(jid), params_of(jid))
    return {"version": 2, "seconds": seconds, "layers": {k: v for k, v in layers.items() if v}}


def test_process_answers_409_needs_uploads_until_the_browser_has_sent_its_part(client, signed_in):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=600.0, rendered="empty")
    tl = tl_with(jid, noise=[browser(100, 130)], subtitles=[sub(200, 260, "free", "en"), sub(300, 330, "standard", "en", ["es"])])
    q = quote(client, h, jid, tl)
    assert q.status_code == 200, q.text
    qj = q.json()
    want = part_cost("transcribe_standard", 30) + part_cost("translate", 30)
    assert qj["estCostMicros"] == want and qj["needs"] == [{"kind": "noise", "start": 100.0, "end": 130.0}, {"kind": "cues", "start": 200.0, "end": 260.0}]
    assert qj["steps"]["subtitles"] == {"parts": 1, "seconds": 60} and qj["asked"]["subtitles"] == 30.0
    assert {l["item"] for l in qj["lines"]} == {"transcribe", "translate"} and "denoise" not in {l["item"] for l in qj["lines"]}
    spent = ledger.balance(uid)
    r = client.post(f"/api/jobs/{jid}/process", json={"timeline": tl}, headers=h)
    assert r.status_code == 409 and r.json()["code"] == "needs_uploads" and r.headers["X-Error-Code"] == "needs_uploads", r.text
    assert r.json()["needs"] == qj["needs"] and ledger.balance(uid) == spent and row(jid)["status"] == "done"
    # the refused timeline is the working one now, so the uploads can be matched with it
    assert params_of(jid)["edit"]["timeline"]["layers"]["subtitles"][0]["quality"] == "free"
    assert client.get(f"/api/jobs/{jid}/timeline").json()["needs"] == qj["needs"]
    assert put_cues(client, h, jid, 200, 260, [{"start": 201, "end": 203, "text": "hi"}]).status_code == 200
    r = client.post(f"/api/jobs/{jid}/process", json={"timeline": tl}, headers=h)
    assert r.status_code == 409 and r.json()["needs"] == [{"kind": "noise", "start": 100.0, "end": 130.0}]
    assert put_noise(client, h, jid, 100, 130, flac(30 * SR)).status_code == 200  # (a mono recording)
    r = client.post(f"/api/jobs/{jid}/process", json={"timeline": tl}, headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["estCostMicros"] == want == r.json()["holdMicros"]
    run = params_of(jid)["run"]
    assert run["audio"] is True and [e["quality"] for e in run["subs"]["parts"]] == ["free", "standard"] and run["subs"]["parts"][1]["translate_run"] == ["es"]
    assert row(jid)["status"] == "queued" and ledger.balance(uid)["held_micros"] == want
    assert client.get(f"/api/jobs/{jid}").json()["edit"]["run"] == "process"


def test_an_upload_to_a_finished_job_is_an_edit_and_wrong_channels_are_refused(client, signed_in):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=600.0, channels=2, rendered="empty")
    tl = tl_with(jid, noise=[browser(100, 130)])
    assert client.put(f"/api/jobs/{jid}/timeline", json={"timeline": tl}, headers=h).json()["needs"] == [{"kind": "noise", "start": 100.0, "end": 130.0}]
    r = put_noise(client, h, jid, 100, 130, flac(30 * SR, channels=1))
    assert r.status_code == 400 and "channels" in r.json()["detail"]
    assert put_noise(client, h, jid, 100, 130, flac(30 * SR, channels=2)).status_code == 200
    assert client.get(f"/api/jobs/{jid}/timeline").json()["needs"] == []
    e1 = row(jid)["expires_at"]
    assert e1 == row(jid)["edited_at"] + 7200
    # the job is being processed: nothing more is taken
    assert client.post(f"/api/jobs/{jid}/process", json={}, headers=h).status_code == 200
    assert put_noise(client, h, jid, 100, 130, flac(30 * SR, channels=2)).status_code == 409


def test_a_free_part_reuploaded_for_another_language_is_needed_again(client, signed_in):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=600.0, rendered="empty")
    assert client.put(f"/api/jobs/{jid}/timeline", json={"timeline": tl_with(jid, subtitles=[sub(200, 260, "free", "en")])}, headers=h).status_code == 200
    assert put_cues(client, h, jid, 200, 260, [{"start": 201, "end": 203, "text": "hi"}]).status_code == 200
    assert client.get(f"/api/jobs/{jid}/timeline").json()["needs"] == []
    assert client.put(f"/api/jobs/{jid}/timeline", json={"timeline": tl_with(jid, subtitles=[sub(200, 260, "free", "es")])}, headers=h).json()["needs"] == [
        {"kind": "cues", "start": 200.0, "end": 260.0}]


def test_get_timeline_returns_the_layers_and_what_is_made(client, signed_in):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=600.0, rendered="empty")
    plain = client.get(f"/api/jobs/{jid}/timeline").json()
    assert "needs" not in plain and "subtitles" not in plain["cached"] and plain["timeline"]["version"] == 1
    tl = tl_with(jid, noise=[browser(100, 130)], subtitles=[sub(200, 260, "free"), sub(300, 330, "standard", "en"), sub(400, 410, "accurate", "es")])
    assert client.put(f"/api/jobs/{jid}/timeline", json={"timeline": tl}, headers=h).status_code == 200
    (editing.browser_dir(jid)).mkdir(mode=0o700)
    t = Timeline.from_json(tl)
    editing.store_cues(jid, t.parts("subtitles")[0], [])
    d = editing.subtitle_dir(jid, t.parts("subtitles")[1])
    d.mkdir(parents=True)
    (d / "segments.json").write_text('{"language": "en", "segments": []}')
    shown = client.get(f"/api/jobs/{jid}/timeline").json()
    assert shown["timeline"] == t.to_json() and shown["timeline"]["version"] == 2
    assert shown["timeline"]["layers"]["noise"] == [{"start": 100.0, "end": 130.0, "browser": True}]
    assert shown["cached"]["subtitles"] == [[200.0, 260.0], [300.0, 330.0]]  # the accurate part has nothing yet
    assert shown["needs"] == [{"kind": "noise", "start": 100.0, "end": 130.0}]


# -------------------------------------------------------------------------------------------- what is made and paid

def test_a_run_pays_for_the_subtitle_work_that_is_missing_only(client, signed_in):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=600.0, rendered="empty")
    tl = tl_with(jid, subtitles=[sub(100, 160, "standard", "en", ["es"]), sub(200, 230, "accurate", "fr")])
    t = Timeline.from_json(tl)
    plan = editing.subtitle_plan(jid, t, rates())
    assert plan["micros"] == part_cost("transcribe_standard", 60) + part_cost("translate", 60) + part_cost("transcribe_accurate", 30)
    # the first part is made already, with its Spanish: nothing of it is charged again; a new language is the only cost of that part
    d = editing.subtitle_dir(jid, t.parts("subtitles")[0])
    d.mkdir(parents=True)
    (d / "segments.json").write_text('{"language": "en", "segments": []}')
    (d / "segments.es.json").write_text('{"language": "es", "segments": []}')
    assert editing.subtitle_plan(jid, t, rates())["micros"] == part_cost("transcribe_accurate", 30)
    t2 = Timeline.from_json(tl_with(jid, subtitles=[sub(100, 160, "standard", "en", ["es", "de"]), sub(200, 230, "accurate", "fr")]))
    p2 = editing.subtitle_plan(jid, t2, rates())
    assert p2["parts"][0]["transcribe"] is False and p2["parts"][0]["translate_run"] == ["de"]
    assert p2["micros"] == part_cost("translate", 60) + part_cost("transcribe_accurate", 30)
    # what the same text says in its own language is not translated into it (a language that was detected)
    detect = Timeline.from_json(tl_with(jid, subtitles=[sub(100, 160, "standard", None, ["es"])]))
    d2 = editing.subtitle_dir(jid, detect.parts("subtitles")[0])
    d2.mkdir(parents=True)
    (d2 / "segments.json").write_text('{"language": "es", "segments": []}')
    assert editing.subtitle_plan(jid, detect, rates())["parts"][0]["translate_run"] == []
    # the work is charged as it was done: a translation that was not made is not paid for
    plan = editing.subtitle_plan(None, t2, rates())  # from nothing: both parts, three translations to make
    assert plan["micros"] == part_cost("transcribe_standard", 60) + 2 * part_cost("translate", 60) + part_cost("transcribe_accurate", 30)
    assert editing.subtitle_charge(jid, plan, rates()) == part_cost("transcribe_standard", 60) + part_cost("translate", 60)  # the text and the Spanish exist; German, French and the accurate text do not
    # the pinned rates are the job's: the run is held at them
    r = client.post(f"/api/jobs/{jid}/process", json={"timeline": tl}, headers=h)
    assert r.status_code == 200, r.text
    assert params_of(jid)["run"]["subtitle_rates"] == rates()


def test_a_run_that_only_changes_subtitles_leaves_the_sound_alone_and_replaces_the_caption_files(client, signed_in):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=600.0, rendered="empty")
    d = folder(jid)
    (d / "talk-cleaned.flac").write_bytes(b"audio")
    (d / "talk.srt").write_bytes(b"old")
    (d / "talk-transcript.txt").write_bytes(b"old")
    db.execute("UPDATE jobs SET outputs_json = ? WHERE id = ?", (json.dumps([{"file": "talk-cleaned.flac", "kind": "audio"}, {"file": "talk.srt", "kind": "captions", "lang": "en"},
                                                                          {"file": "talk-transcript.txt", "kind": "transcript", "lang": "en"}]), jid))
    tl = tl_with(jid, subtitles=[sub(100, 160, "free", "en")])
    assert client.put(f"/api/jobs/{jid}/timeline", json={"timeline": tl}, headers=h).status_code == 200
    assert put_cues(client, h, jid, 100, 160, [{"start": 101, "end": 102, "text": "hi"}]).status_code == 200
    q = quote(client, h, jid, tl).json()
    assert q["unchanged"] is False and q["estCostMicros"] == 0 and q["free"] is True and q["needsProcess"] is False
    r = client.post(f"/api/jobs/{jid}/export", json={"timeline": tl}, headers=h)  # nothing to make: an export does it for free
    assert r.status_code == 200, r.text
    run = params_of(jid)["run"]
    assert run["audio"] is False and run["subs"]["parts"][0]["free"] is True
    spec = editing.spec_for(row(jid), params_of(jid))
    assert spec["audio"] is False and spec["subtitles"]["parts"][0]["quality"] == "free" and spec["browser"] == []
    # the runner's answer: new caption files, no audio
    db.execute("UPDATE jobs SET status = 'running', worker_id = 't', started_at = ? WHERE id = ?", (db.now(), jid))
    (d / "talk.srt").write_bytes(b"new")
    (d / "talk-transcript.txt").write_bytes(b"new")
    (d / "talk.vtt").write_bytes(b"new")
    res = {"ok": True, "outputs": [{"file": "talk.srt", "kind": "captions", "lang": "en"}, {"file": "talk.vtt", "kind": "captions", "lang": "en"},
                                  {"file": "talk-transcript.txt", "kind": "transcript", "lang": "en"}], "editor": {"billed": {}}}
    worker._complete_edit(row(jid), d, res)
    done = row(jid)
    assert done["status"] == "done" and {o["file"] for o in json.loads(done["outputs_json"])} == {"talk-cleaned.flac", "talk.srt", "talk.vtt", "talk-transcript.txt"}
    assert (d / "talk-cleaned.flac").read_bytes() == b"audio" and (d / "talk.srt").read_bytes() == b"new" and done["cost_micros"] == 0
    # and the result files now hold exactly this: asking again changes nothing, until the cues are sent again
    p = params_of(jid)
    assert p["edit"]["subs_rendered"] and "browser_rendered" not in p["edit"] and (d / "browser").is_dir()
    assert quote(client, h, jid, tl).json()["unchanged"] is True
    assert client.post(f"/api/jobs/{jid}/process", json={"timeline": tl}, headers=h).json()["code"] == "nothing_to_change"
    assert put_cues(client, h, jid, 100, 160, [{"start": 101, "end": 102, "text": "hello"}]).status_code == 200
    assert quote(client, h, jid, tl).json()["unchanged"] is False
    # taking the subtitles away makes the caption files go too
    plain = tl_with(jid)
    plain["version"] = 1
    q = quote(client, h, jid, plain).json()
    assert q["unchanged"] is False and q["estCostMicros"] == 0


def test_a_first_run_with_subtitles_is_charged_for_what_was_made(client, signed_in):
    uid, h = funded(client, signed_in)
    tl = v2(60, noise=[{"start": 10, "end": 20.2, "strength_db": 100, "remover": "light"}], subtitles=[sub(0, 30, "standard", "en", ["es", "fr"]), sub(40, 50, "free", "en")])
    jid, needs = make_uploading(client, h, tl, kind="denoise_transcribe", seconds=60)
    hold = row(jid)["est_cost_micros"]
    audio = expected({"light": 11})
    subs = part_cost("transcribe_standard", 30) + 2 * part_cost("translate", 30)
    assert hold == audio + subs
    # the job ran: its text and the Spanish exist, the French could not be made
    n = 60 * SR
    db.execute("UPDATE jobs SET status = 'running', duration_seconds = 60, uploaded_at = ?, worker_id = 't', next_stage = 'main' WHERE id = ?", (db.now(), jid))
    d = folder(jid)
    (d / "input.wav").write_bytes(b"x")
    (d / "talk-cleaned.flac").write_bytes(b"audio")
    (d / "talk.srt").write_bytes(b"srt")
    (d / "kept-source.flac").write_bytes(b"fLaC")
    key = editing.subtitle_key(Timeline.from_json(tl).parts("subtitles")[0])
    (d / "subs" / key).mkdir(parents=True)
    (d / "subs" / key / "segments.json").write_text('{"language": "en", "segments": []}')
    (d / "subs" / key / "segments.es.json").write_text('{"language": "es", "segments": []}')
    (d / "subs" / key / "asr_chunks").mkdir()
    (d / "subs" / key / "asr_chunks" / "0000.mp3").write_bytes(b"x")
    editing.store_cues(jid, Timeline.from_json(tl).parts("subtitles")[1], [])
    res = {"ok": True, "outputs": [{"file": "talk-cleaned.flac", "kind": "audio"}, {"file": "talk.srt", "kind": "captions", "lang": "en"}],
           "editor": {"billed": {"light": {"seconds": 11}}, "resolved": {}, "rendered": Timeline.from_json(tl, seconds=60).audio().to_json(),
                      "facts": {"n": n, "channels": 1, "plan": [0], "spans": {"0": [[0, n]]}, "rebuild_mid": False, "reference": False}}}
    worker._complete(row(jid), d, res)
    done = row(jid)
    assert done["status"] == "done" and done["cost_micros"] == audio + part_cost("transcribe_standard", 30) + part_cost("translate", 30) < hold
    assert files_of(jid) >= {"subs", "browser"} and not (d / "subs" / key / "asr_chunks").exists()
    p = params_of(jid)
    assert "first_run" not in p["edit"] and p["edit"]["subs_rendered"] and p["edit"]["rendered"]["layers"].get("subtitles") is None
    assert [l["description"] for l in db.all_rows("SELECT * FROM ledger WHERE user_id = ? AND kind = 'usage'", (uid,))] == [
        "Denoising + Subtitles: talk.wav (1.0 min)"]
    # the job is edited like any other; the subtitles that are there are not made or charged again
    q = quote(client, h, jid, tl).json()
    assert q["unchanged"] is False and q["estCostMicros"] == part_cost("translate", 30)  # (the French was not made: the only thing left to pay for)


# ------------------------------------------------------------------------------------------------------ the outline

def test_peaks_come_from_the_cached_blocks_and_are_kept_per_block(client, signed_in, monkeypatch):
    from acw import stretch

    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=600.0, rendered="empty")
    assert client.get(f"/api/jobs/{jid}/peaks/bass").status_code == 404
    r = client.get(f"/api/jobs/{jid}/peaks/noise")
    assert r.status_code == 404 and r.json()["code"] == "not_cached"
    tl = tl_of(jid, noise=[noise(0, 100)])
    fill_cache(jid, tl)
    client.put(f"/api/jobs/{jid}/timeline", json={"timeline": tl}, headers=h)
    asked = []

    def fake(job_dir, spec, timeout=60.0):
        asked.append(spec)
        return {b["name"]: {"first": b["at"] // 12_000, "p": [round(0.001 * (b["at"] // 12_000 + i), 5) for i in range(-(-1_440_000 // 12_000))]} for b in spec["blocks"]}

    monkeypatch.setattr(stretch, "peaks", fake)
    r = client.get(f"/api/jobs/{jid}/peaks/noise")
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["rate"] == 4 and out["seconds"] == 600.0 and len(out["peaks"]) == 2400 and "pending" not in out
    # the noise removal of the first 100 s is four blocks of the 30 s grid (the last reaches to 120 s); the rest is not cached
    assert out["peaks"][0] == 0.0 and out["peaks"][1] == 0.001 and out["peaks"][479] is not None
    assert out["peaks"][480] is None and out["peaks"][-1] is None
    assert sum(len(a["blocks"]) for a in asked) == 4
    # kept next to the blocks: asking again decodes nothing
    n_asked = len(asked)
    assert client.get(f"/api/jobs/{jid}/peaks/noise").json() == out and len(asked) == n_asked
    assert any(p.suffix == ".peaks" for p in (folder(jid) / "blocks").iterdir())
    # the other layers: nothing of them is cached; the recording's copy is read as one file
    assert client.get(f"/api/jobs/{jid}/peaks/rebuild").status_code == 404 and client.get(f"/api/jobs/{jid}/peaks/echo").status_code == 404
    assert client.get(f"/api/jobs/{jid}/peaks/source").status_code == 404
    p = params_of(jid)
    p["edit"]["facts"]["source"] = {"file": "kept-source.flac", "channels": [0], "scale": 0.5, "samples": p["edit"]["facts"]["n"]}
    save_params(jid, p)
    (folder(jid) / "kept-source.flac").write_bytes(b"fLaC")
    monkeypatch.setattr(stretch, "peaks", lambda d, spec, timeout=60.0: {"first": 0, "p": [0.25] * 2400} if spec.get("file") else fake(d, spec))
    src = client.get(f"/api/jobs/{jid}/peaks/source").json()
    assert src["peaks"] == [0.25] * 2400 and (folder(jid) / "kept-source.peaks").is_file()
    # a block of the plan that has no outline yet when the time runs out is unknown for now, and the answer says so
    for p_ in (folder(jid) / "blocks").glob("*.peaks"):
        p_.unlink()
    monkeypatch.setattr(editing, "PEAKS_BUDGET_SECONDS", -1.0)
    late = client.get(f"/api/jobs/{jid}/peaks/noise")
    assert late.status_code == 404  # (nothing had its outline in the time: nothing to draw yet)


def test_the_outline_has_the_previews_auth_and_limits(client, signed_in, other_browser):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=600.0, rendered="empty")
    oc, ouid, oh = other_browser
    assert oc.get(f"/api/jobs/{jid}/peaks/noise").status_code == 404
    key_id, kh = api_key(uid, "stream")
    assert client.get(f"/api/jobs/{jid}/peaks/noise", headers=kh).status_code == 403
    converted = make_done_job(uid)
    db.execute("UPDATE jobs SET kind = 'convert' WHERE id = ?", (converted,))  # (a conversion has no timeline)
    r = client.get(f"/api/jobs/{converted}/peaks/noise")
    assert r.status_code == 409 and r.json()["code"] == "job_not_editable"
    sub_only = make_done_job(uid, kind="transcribe", transcribe_tier="local")  # (a transcription is made with a timeline too; nothing of it is cached)
    r = client.get(f"/api/jobs/{sub_only}/peaks/noise")
    assert r.status_code == 404 and r.json()["code"] == "not_cached"
    busy = edit_job(uid, seconds=600.0, rendered="empty")
    db.execute("UPDATE jobs SET status = 'running' WHERE id = ?", (busy,))
    assert client.get(f"/api/jobs/{busy}/peaks/noise").status_code == 409


def test_window_peaks_count_from_the_start_of_the_recording():
    import numpy as np

    from acw.worker import peaker

    y = np.zeros(30_000, np.float32)
    y[5] = -0.5
    y[12_001] = 0.25
    y[29_999] = 0.75
    assert peaker.window_peaks(y, 0) == (0, [0.5, 0.25, 0.75])
    first, p = peaker.window_peaks(y, 6_000)  # a block that starts in the middle of a window holds only its own samples of it
    assert first == 0 and p == [0.5, 0.25, 0.75]  # (samples 6,000 to 36,000 of the recording: windows 0, 1 and 2)
    assert peaker.window_peaks(y[:5], 24_000 * 3 + 7) == (6, [0.0])


def test_a_new_strength_on_a_browser_part_changes_nothing_in_the_sound_or_the_price(client, signed_in):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=600.0, rendered="empty")
    tl = tl_with(jid, noise=[{**browser(100, 130), "strength_db": 12}], leveling=[level(100, 130)])
    assert client.put(f"/api/jobs/{jid}/timeline", json={"timeline": tl}, headers=h).status_code == 200
    assert put_noise(client, h, jid, 100, 130, flac(30 * SR)).status_code == 200
    t = Timeline.from_json(tl)
    p = params_of(jid)
    p["edit"]["rendered"] = t.audio().to_json()
    p["edit"]["browser_rendered"] = editing.uploads_fingerprint(jid, t)[0]
    save_params(jid, p)
    assert client.get(f"/api/jobs/{jid}/timeline").json()["timeline"]["layers"]["noise"] == [{"start": 100.0, "end": 130.0, "browser": True, "strength_db": 12.0}]
    q = quote(client, h, jid, tl).json()
    assert q["unchanged"] is True and q["estCostMicros"] == 0
    stronger = tl_with(jid, noise=[{**browser(100, 130), "strength_db": 40}], leveling=[level(100, 130)])
    q = quote(client, h, jid, stronger).json()
    assert q["unchanged"] is True and q["estCostMicros"] == 0 and q["timeline"]["layers"]["noise"][0]["strength_db"] == 40.0
    assert client.post(f"/api/jobs/{jid}/process", json={"timeline": stronger}, headers=h).json()["code"] == "nothing_to_change"
    # saving it keeps the strength, so the page shows it after a reload
    assert client.put(f"/api/jobs/{jid}/timeline", json={"timeline": stronger}, headers=h).json()["timeline"]["layers"]["noise"][0]["strength_db"] == 40.0
    assert client.get(f"/api/jobs/{jid}/timeline").json()["timeline"]["layers"]["noise"][0]["strength_db"] == 40.0


def test_standard_parts_are_made_one_after_the_other_with_the_model_held_for_all_of_them(tmp_path, monkeypatch):
    """editrun.subtitles_prepare: every part of one language gets the same `held` dicts (the model is loaded once, in transcribe), parts of
    the same model are made in a row, and what was held is let go when the run is over."""
    import numpy as np

    from acw.worker import editrun
    from acw_common import transcribe
    from acw_common.captions import Segment

    calls = []

    def parakeet(audio, model, vad, *, device, threads, progress, held):
        calls.append(("parakeet", model.name, id(held), held.setdefault("n", 0)))
        held["n"] += 1
        return [Segment(0.5, 1.0, "hi")]

    def whisper_detect(audio, quick, sure, threads, held=None):
        calls.append(("detect", id(held)))
        return "es" if len([c for c in calls if c[0] == "detect"]) == 1 else "ja"

    monkeypatch.setattr(transcribe, "transcribe_parakeet", parakeet)
    monkeypatch.setattr(transcribe, "detect_language", whisper_detect)
    monkeypatch.setattr(transcribe, "transcribe", lambda audio, path, **kw: calls.append(("whisper", kw["model"])) or ([Segment(0.5, 1.0, "yo")], kw["language"]))
    monkeypatch.setattr(transcribe, "load_model", lambda path, threads: "the whisper model")
    unloaded = []
    monkeypatch.setattr(transcribe, "unload", lambda: unloaded.append(True))
    x = np.zeros((20 * SR, 1), np.float32)
    parts = [{"key": f"k{i}", "start": 2.0 * i, "end": 2.0 * i + 1.5, "quality": "standard", "language": lang, "translate": []}
             for i, lang in enumerate(["en", None, "es", "en", "ja"])]
    spec = {"models_dir": str(tmp_path), "threads": 2, "input": "input.wav", "editor": {"subtitles": {"parts": parts}}}
    probe = type("P", (), {"duration": 20.0})()
    left = editrun.subtitles_prepare(tmp_path, spec, None, probe, x, 100.0, lambda **kw: None)
    assert left is False and unloaded == [True]
    done = sorted(p.name for p in (tmp_path / "subs").iterdir())
    assert done == ["k0", "k1", "k2", "k3", "k4"] and all((tmp_path / "subs" / k / "segments.json").is_file() for k in done)
    names = [c[:2] for c in calls if c[0] != "detect"]
    # the English parts first, one after the other (parakeet-v2), then Spanish (v3), Japanese (Whisper's, loaded through transcribe.load_model),
    # and the part whose language is found out last (here Spanish again: the model that is held is v3, still)
    assert names == [("parakeet", "parakeet-v2")] * 2 + [("parakeet", "parakeet-v3"), ("whisper", "the whisper model"), ("parakeet", "parakeet-v3")]
    held_ids = {c[2] for c in calls if c[0] == "parakeet"}
    assert len(held_ids) == 1 and [c[3] for c in calls if c[0] == "parakeet"] == [0, 1, 2, 3]  # (one dict for the whole run)
    assert len({c[1] for c in calls if c[0] == "detect"}) == 1  # ... and one for the language models


# ------------------------------------------------------------------------- a run never leaves the job half changed

def a_run(client, h, jid, ext="flac"):
    """A job with an old result, and a process of a paid change queued on it (so there is a hold); returns (the timeline, the folder)."""
    d = folder(jid)
    (d / "talk-cleaned.flac").write_bytes(b"OLD AUDIO")
    db.execute("UPDATE jobs SET outputs_json = ? WHERE id = ?", (json.dumps([{"file": "talk-cleaned.flac", "kind": "audio", "bytes": 9}]), jid))
    tl = tl_of(jid, noise=[noise(0, 30)])
    r = client.post(f"/api/jobs/{jid}/process", json={"timeline": tl, **({"audioFormat": ext} if ext != "flac" else {})}, headers=h)
    assert r.status_code == 200, r.text
    db.execute("UPDATE jobs SET status = 'running', worker_id = 't', started_at = ? WHERE id = ?", (db.now(), jid))
    return tl, d


def the_runner_ends(d, tl, ext="flac"):
    """What the sandboxed runner leaves for the server when it is through: the new audio under its temporary name, the scratch, its answer."""
    (d / f"talk-cleaned.{ext}.run").write_bytes(b"NEW AUDIO")
    (d / "decoded.f32").write_bytes(b"scratch")
    billed = {"light": {"seconds": 30}}
    return {"ok": True, "outputs": [{"file": f"talk-cleaned.{ext}.run", "kind": "audio"}],
            "editor": {"billed": billed, "resolved": {}, "rendered": Timeline.from_json(tl, seconds=600).to_json()}}


def test_the_new_audio_takes_the_old_ones_place_only_when_the_run_is_complete(client, signed_in):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=600.0, rendered="empty")
    tl, d = a_run(client, h, jid)
    before = params_of(jid)["edit"].get("rendered")
    res = the_runner_ends(d, tl)
    # while the run runs the old result is what is there, and what the record says
    assert (d / "talk-cleaned.flac").read_bytes() == b"OLD AUDIO" and (d / "talk-cleaned.flac.run").read_bytes() == b"NEW AUDIO"
    worker._complete_edit(row(jid), d, res)
    done = row(jid)
    assert done["status"] == "done" and (d / "talk-cleaned.flac").read_bytes() == b"NEW AUDIO"
    assert [o["file"] for o in json.loads(done["outputs_json"])] == ["talk-cleaned.flac"] and json.loads(done["outputs_json"])[0]["bytes"] == 9
    assert files_of(jid) >= {"talk-cleaned.flac"} and not {n for n in files_of(jid) if n.endswith((".run", ".bak"))} and "decoded.f32" not in files_of(jid)
    assert params_of(jid)["edit"]["rendered"] != before and params_of(jid)["edit"]["rendered"]["layers"]["noise"][0]["end"] == 30.0


def test_a_run_that_fails_leaves_the_old_audio_and_the_record_that_matches_it(client, signed_in):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=600.0, rendered="empty")
    tl, d = a_run(client, h, jid)
    before = params_of(jid)["edit"]["rendered"]
    the_runner_ends(d, tl)  # the runner had written the new audio when the run failed (the web process's part of it, say)
    assert jobs.finish_failed(row(jid), "Extra accurate transcription failed; you were not charged.") is True
    done = row(jid)
    assert done["status"] == "done" and (d / "talk-cleaned.flac").read_bytes() == b"OLD AUDIO"
    assert not {n for n in files_of(jid) if n.endswith(".run")} and "decoded.f32" not in files_of(jid)
    p = params_of(jid)
    assert p["edit"]["rendered"] == before and "run" not in p and p["edit"]["lastRun"]["ok"] is False
    assert [o["file"] for o in json.loads(done["outputs_json"])] == ["talk-cleaned.flac"] and ledger.balance(uid)["held_micros"] == 0
    # the run's own failure (the runner dies, the job is put back): the same
    tl, d = a_run(client, h, jid)
    the_runner_ends(d, tl)
    db.execute("UPDATE jobs SET status = 'running' WHERE id = ?", (jid,))
    jobs.requeue(row(jid))
    assert not {n for n in files_of(jid) if n.endswith(".run")} and (d / "talk-cleaned.flac").read_bytes() == b"OLD AUDIO"


def test_a_run_that_cannot_be_committed_puts_the_old_audio_back(client, signed_in, monkeypatch):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=600.0, rendered="empty")
    tl, d = a_run(client, h, jid)
    res = the_runner_ends(d, tl)
    held = ledger.balance(uid)["held_micros"]

    def broken(*a, **k):
        raise RuntimeError("the ledger is down")

    monkeypatch.setattr(ledger, "capture_hold", broken)
    with pytest.raises(RuntimeError):
        worker._complete_edit(row(jid), d, res)
    assert (d / "talk-cleaned.flac").read_bytes() == b"OLD AUDIO" and not {n for n in files_of(jid) if n.endswith((".run", ".bak"))}
    assert row(jid)["status"] == "running" and "run" in params_of(jid) and ledger.balance(uid)["held_micros"] == held
    # the job was canceled meanwhile: nothing changes either
    monkeypatch.undo()
    res = the_runner_ends(d, tl)
    db.execute("UPDATE jobs SET status = 'canceled' WHERE id = ?", (jid,))
    worker._complete_edit(row(jid), d, res)
    assert (d / "talk-cleaned.flac").read_bytes() == b"OLD AUDIO" and not {n for n in files_of(jid) if n.endswith(".bak")}


def test_a_new_file_type_replaces_the_old_file_only_with_the_commit(client, signed_in, monkeypatch):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=600.0, rendered="empty")
    tl, d = a_run(client, h, jid, ext="wav")
    res = the_runner_ends(d, tl, ext="wav")
    monkeypatch.setattr(ledger, "capture_hold", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    with pytest.raises(RuntimeError):
        worker._complete_edit(row(jid), d, res)
    assert (d / "talk-cleaned.flac").read_bytes() == b"OLD AUDIO" and not (d / "talk-cleaned.wav").exists()  # (the old file was never touched)
    monkeypatch.undo()
    worker._complete_edit(row(jid), d, the_runner_ends(d, tl, ext="wav"))
    assert (d / "talk-cleaned.wav").read_bytes() == b"NEW AUDIO" and not (d / "talk-cleaned.flac").exists()
    assert [o["file"] for o in json.loads(row(jid)["outputs_json"])] == ["talk-cleaned.wav"]


def test_audio_that_was_being_put_in_place_when_the_process_stopped_is_put_back_when_the_run_starts_again(client, signed_in):
    uid, h = funded(client, signed_in)
    jid = edit_job(uid, seconds=600.0, rendered="empty")
    d = folder(jid)
    (d / "talk-cleaned.flac").write_bytes(b"NEW AUDIO")  # the swap had happened, the commit had not
    (d / "talk-cleaned.flac.bak").write_bytes(b"OLD AUDIO")
    (d / "talk-cleaned.flac.run").write_bytes(b"x")
    editing.clear_scratch(d)
    assert (d / "talk-cleaned.flac").read_bytes() == b"OLD AUDIO" and not {n for n in files_of(jid) if n.endswith((".bak", ".run"))}
