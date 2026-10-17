"""Translated subtitles for files: languages checked, text-only requests answered line for line, one set of files per
language, a language that can't be made is left out and not charged."""
import asyncio
import json

import httpx
import pytest

from acw import auth, db, jobs, translate
from acw.billing import ledger, pricing


def fake_model(monkeypatch, answer):
    """The speech partner's text model, answered by `answer(target, lines) -> list | Exception`."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        system = body["messages"][0]["content"]
        target = next(code for code, name in translate.LANGUAGE_NAMES.items() if f"into {name}" in system)
        user = body["messages"][1]["content"]
        calls.append((target, user))
        if body.get("response_format"):
            lines = json.loads(user)["lines"]
            out = answer(target, lines)
            if isinstance(out, Exception):
                return httpx.Response(500, json={"error": {"type": "server_error"}})
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"lines": out})}}]})
        line = user.rsplit("\n", 1)[-1]
        out = answer(target, [line])
        if isinstance(out, Exception):
            return httpx.Response(500, json={"error": {"type": "server_error"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": out[0]}}]})

    real = httpx.AsyncClient

    def client(*a, **k):
        k["transport"] = httpx.MockTransport(handler)
        return real(*a, **k)

    monkeypatch.setattr(translate.httpx, "AsyncClient", client)
    monkeypatch.setattr(translate, "BACKOFF_SECONDS", ())
    monkeypatch.setattr(translate.alerts, "report_openai", lambda *a, **k: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from acw.config import get_settings

    get_settings.cache_clear()
    return calls


def test_languages_are_checked():
    assert translate.check_targets(None) == []
    assert translate.check_targets(["ES", "de", "es"]) == ["es", "de"]
    for bad in (["xx"], ["es", "de", "fr", "it"], "es", [1]):
        with pytest.raises(ValueError):
            translate.check_targets(bad)
    with pytest.raises(ValueError):
        translate.check_targets(["en"], "en")


def test_hungarian_romanian_and_thai_are_offered():
    assert translate.check_targets(["hu", "ro", "th"], "en") == ["hu", "ro", "th"]
    assert translate.LANGUAGE_NAMES["th"] == "Thai"


def test_a_job_asks_for_translations_and_is_priced_per_language():
    uid = auth.get_or_create_user("t@example.com")["id"]
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    base = dict(kind="transcribe", denoise_tier=None, transcribe_tier="local", filename="a.wav", size_bytes=1000,
                claimed_seconds=600.0)
    plain = jobs.create_job(uid, options={}, **base)["estCostMicros"]
    two = jobs.create_job(uid, options={"translate": ["es", "de"]}, **base)["estCostMicros"]
    assert two - plain == 2 * pricing.cost_micros("transcribe_translate", "standard", 600)
    with pytest.raises(jobs.JobError):
        jobs.create_job(uid, options={"translate": ["en"], "language": "en"}, **base)
    with pytest.raises(jobs.JobError):
        jobs.create_job(uid, options={"translate": ["es"]}, **{**base, "kind": "denoise", "denoise_tier": "standard", "transcribe_tier": None})


def _job_with_segments(tmp_path, langs):
    uid = auth.get_or_create_user("t2@example.com")["id"]
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    jid = jobs.create_job(uid, kind="transcribe", denoise_tier=None, transcribe_tier="local", filename="a.wav", size_bytes=1000,
                          claimed_seconds=60.0, options={"translate": langs})["id"]
    d = jobs.job_dir(jid)
    segs = [{"start": 0.0, "end": 2.0, "text": "Hello there", "words": [[0, 1, "Hello"], [1, 2, "there"]]},
            {"start": 2.5, "end": 4.0, "text": "Good game", "words": []}]
    (d / "segments.json").write_text(json.dumps({"language": "en", "segments": segs}))
    db.execute("UPDATE jobs SET status = 'cloud_asr', duration_seconds = 60 WHERE id = ?", (jid,))
    return dict(db.one("SELECT * FROM jobs WHERE id = ?", (jid,))), d


def test_translations_are_made_line_for_line_and_a_failed_language_is_left_out(monkeypatch, tmp_path):
    from acw import app as appmod
    from acw.worker import main as W

    calls = fake_model(monkeypatch, lambda t, lines: RuntimeError() if t == "de" else [f"[{t}] {x}" for x in lines])
    job, d = _job_with_segments(tmp_path, ["es", "de"])
    asyncio.run(appmod._run_cloud_job(job))
    es = json.loads((d / "segments.es.json").read_text())
    assert [s["text"] for s in es["segments"]] == ["[es] Hello there", "[es] Good game"]
    assert es["segments"][0]["start"] == 0.0 and es["segments"][0]["words"] == []  # same timing, no word times
    assert not (d / "segments.de.json").exists()
    row = db.one("SELECT status, next_stage, params_json FROM jobs WHERE id = ?", (job["id"],))
    params = json.loads(row["params_json"])
    assert row["status"] == "queued" and row["next_stage"] == "finalize"
    assert params["translated"] == ["es"] and "German" in params["notice"] and "weren’t charged" in params["notice"]
    assert all("Hello there" in u or "Good game" in u for _, u in calls)  # only the text goes out
    # charged for the one language that was made
    job = dict(db.one("SELECT * FROM jobs WHERE id = ?", (job["id"],)))
    assert W._job_cost(job, params, 60) == pricing.job_cost_micros("transcribe", None, "local", 60, params["rates"], 1)
    # a second run (restart) doesn't pay for Spanish again
    n = len(calls)
    asyncio.run(appmod._run_cloud_job({**job, "status": "cloud_asr"}))
    assert all(t != "es" for t, _ in calls[n:])


def test_a_batch_with_the_wrong_number_of_lines_falls_back_to_one_line_at_a_time(monkeypatch):
    fake_model(monkeypatch, lambda t, lines: ["only one"] if len(lines) > 1 else [f"<{lines[0]}>"])

    async def run():
        async with httpx.AsyncClient() as http:
            return await translate.translate_lines(http, ["a", "b", "c"], "en", "fr")

    assert asyncio.run(run()) == ["<a>", "<b>", "<c>"]


def test_a_broken_answer_is_asked_again_and_then_given_up(monkeypatch):
    answers = iter(["ここ嫌い、笑♀♀♀♀♀♀", "ここマジで嫌いｗ", "x" * 500, "y" * 500])
    calls = fake_model(monkeypatch, lambda t, lines: [next(answers)])

    async def run(text):
        async with httpx.AsyncClient() as http:
            return await translate.translate_line(http, text, "en", "ja")

    assert asyncio.run(run("I hate it here lmao")) == "ここマジで嫌いｗ" and len(calls) == 2
    with pytest.raises(translate.TranslationError):
        asyncio.run(run("short line"))
    assert not translate.looks_broken("wait... what?!", "Warte.... was?!")  # a run the line itself has


def test_the_prompt_keeps_tone_and_gaming_terms():
    p = translate._system_prompt("en", "de")
    assert "into German" in p and "tone" in p and "gaming terms" in p and "switch languages" in p


def test_finalize_writes_one_set_of_files_per_language(tmp_path):
    from acw.worker import runner

    d = tmp_path / "job"
    d.mkdir()
    (d / "main-state.json").write_text(json.dumps({"outputs": [], "cleaned": None, "has_video": False}))
    segs = [{"start": 0.0, "end": 2.0, "text": "Hello there", "words": []}]
    (d / "segments.json").write_text(json.dumps({"language": "en", "segments": segs}))
    (d / "segments.es.json").write_text(json.dumps({"language": "es", "segments": [{**segs[0], "text": "Hola"}]}))
    spec = {"bin_dir": str(tmp_path), "output_stem": "talk", "input_ext": ".wav", "input": "input.wav",
            "options": {"caption_formats": ["srt", "vtt", "txt"], "translate": ["es", "de"]}}
    res = runner.stage_finalize(d, spec)
    by = {o["file"]: o for o in res["outputs"]}
    assert set(by) == {"talk.srt", "talk.vtt", "talk-transcript.txt", "talk.es.srt", "talk.es.vtt", "talk.es-transcript.txt"}
    assert by["talk.srt"]["lang"] == "en" and by["talk.es.vtt"]["lang"] == "es"
    assert "Hola" in (d / "talk.es.srt").read_text(encoding="utf-8")
