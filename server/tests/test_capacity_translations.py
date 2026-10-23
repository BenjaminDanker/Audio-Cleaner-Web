"""Cloud-stage translations get places of their own, and live translation work is bounded (audit M6, M12)."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from acw import app as appmod
from acw import db, translate
from acw.billing import ledger
from acw.stream import live_translate
from tests.conftest import settle_payment
from tests.job_support import part_dir
from tests.test_billing_audit import make_job, user

_n = [0]


def _rich_user():
    _n[0] += 1
    uid = user(f"u{_n[0]}@example.com")
    settle_payment(uid)
    ledger.credit(uid, 100_000_000, "grant", "t", f"g{_n[0]}")
    return uid


def _cloud_job(_uid, tier, translate_to=None, with_segments=False):
    """A transcription job waiting for the web process: its subtitles are one part over the whole file (every job is made with a
    timeline); an Accurate part waits with its speech chunks written, a part with its text is only waiting to be translated."""
    uid = _rich_user()  # a user each: a person has at most three jobs going
    options = {"translate": translate_to} if translate_to else {}
    j = make_job(uid, kind="transcribe", denoise_tier=None, transcribe_tier=tier, options=options)
    db.execute("UPDATE jobs SET status = 'cloud_asr', stage = 'Waiting to transcribe', queued_at = ?, duration_seconds = 60 WHERE id = ?",
               (db.now(), j))
    d = part_dir(j)
    d.mkdir(parents=True)
    if with_segments:
        (d / "segments.json").write_text(json.dumps({"language": "en", "segments": [{"start": 0, "end": 1, "text": "hi"}]}))
    elif tier == "premium":
        (d / "asr_chunks").mkdir()
        (d / "asr_chunks" / "index.json").write_text("{}")
    return j


def test_transcription_and_translation_are_claimed_into_separate_places():
    uid = _rich_user()
    asr = [_cloud_job(uid, "premium") for _ in range(3)]
    tr = [_cloud_job(uid, "local", ["es"], with_segments=True) for _ in range(3)]
    got_asr = appmod._claim_cloud_jobs(appmod.PREMIUM_JOBS_AT_ONCE, "asr")
    assert [j["id"] for j in got_asr] == asr[:2] and all(j["_phase"] == "asr" for j in got_asr)
    got_tr = appmod._claim_cloud_jobs(appmod.TRANSLATION_JOBS_AT_ONCE, "translate")
    assert [j["id"] for j in got_tr] == tr[:2] and all(j["_phase"] == "translate" for j in got_tr)
    stages = {r["id"]: r["stage"] for r in db.all_rows("SELECT id, stage FROM jobs")}
    assert stages[asr[0]] == "Transcribing" and stages[tr[0]] == "Translating" and stages[asr[2]] == "Waiting to transcribe"
    # a claimed job (either phase) is never claimed again while it is being worked on
    assert [j["id"] for j in appmod._claim_cloud_jobs(10)] == [asr[2], tr[2]]
    assert appmod._claim_cloud_jobs(10) == []


def test_a_translation_backlog_does_not_hold_back_transcription():
    uid = _rich_user()
    for _ in range(5):
        _cloud_job(uid, "local", ["es"], with_segments=True)  # oldest: five translations waiting
    asr = _cloud_job(uid, "premium")
    assert [j["id"] for j in appmod._claim_cloud_jobs(appmod.PREMIUM_JOBS_AT_ONCE, "asr")] == [asr]


def test_after_the_transcription_a_job_with_translations_waits_for_a_translation_place(monkeypatch):
    from acw.asr import openai_asr

    uid = _rich_user()
    j = _cloud_job(uid, "premium", ["es"])
    job = appmod._claim_cloud_jobs(2, "asr")[0]

    async def fake_asr(d, *a, **k):
        (d / "segments.json").write_text(json.dumps({"language": "en", "segments": []}))

    monkeypatch.setattr(openai_asr, "run_for_job", fake_asr)
    asyncio.run(appmod._run_cloud_job(job))
    row = db.one("SELECT status, stage FROM jobs WHERE id = ?", (j,))
    assert (row["status"], row["stage"]) == ("cloud_asr", "Waiting to translate")  # not finished, not holding an ASR place
    assert appmod._claim_cloud_jobs(2, "asr") == []
    assert [x["id"] for x in appmod._claim_cloud_jobs(2, "translate")] == [j]


def test_a_job_being_translated_is_not_started_a_second_time():
    uid = _rich_user()
    j = _cloud_job(uid, "local", ["es"], with_segments=True)
    assert len(appmod._claim_cloud_jobs(1)) == 1
    db.execute("UPDATE jobs SET stage = 'Translating' WHERE id = ?", (j,))
    assert appmod._claim_cloud_jobs(5) == []


def test_housekeeping_puts_back_stale_work_in_its_own_phase():
    uid = _rich_user()
    a = _cloud_job(uid, "premium")
    t = _cloud_job(uid, "local", ["es"], with_segments=True)
    db.execute("UPDATE jobs SET stage = 'Transcribing', heartbeat_at = ? WHERE id = ?", (db.now() - 700, a))
    db.execute("UPDATE jobs SET stage = 'Translating', heartbeat_at = ? WHERE id = ?", (db.now() - 700, t))
    appmod._housekeeping()
    assert db.one("SELECT stage FROM jobs WHERE id = ?", (a,))["stage"] == "Waiting to transcribe"
    assert db.one("SELECT stage FROM jobs WHERE id = ?", (t,))["stage"] == "Waiting to translate"


# ---------------------------------------------------------------- the text model: limits

def _counting_transport(delay=0.01, status=200):
    state = {"now": 0, "peak": 0, "calls": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        state["now"] += 1
        state["calls"] += 1
        state["peak"] = max(state["peak"], state["now"])
        await asyncio.sleep(delay)
        state["now"] -= 1
        if status != 200:
            return httpx.Response(status, json={"error": {}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    return httpx.MockTransport(handler), state


def test_requests_to_the_text_model_are_capped_across_sessions_and_jobs(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from acw.config import get_settings

    get_settings.cache_clear()
    transport, state = _counting_transport()
    n = 3 * translate.MAX_IN_FLIGHT

    async def run():
        async with httpx.AsyncClient(transport=transport) as http:
            await asyncio.gather(*(translate.translate_line(http, f"line {i}", "en", "es") for i in range(n)))

    asyncio.run(run())
    assert state["calls"] == n and state["peak"] == translate.MAX_IN_FLIGHT


def test_a_live_line_is_retried_once_quickly_then_skipped(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from acw.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setattr(translate.alerts, "report_openai", lambda *a, **k: None)
    monkeypatch.setattr(translate, "LIVE_BACKOFF_SECONDS", (0.01,))
    transport, state = _counting_transport(delay=0, status=429)

    async def run():
        async with httpx.AsyncClient(transport=transport) as http:
            with pytest.raises(translate.TranslationError):
                await translate.translate_line(http, "hello", "en", "es")

    asyncio.run(run())
    assert state["calls"] == 2  # one retry, not the 3 + 15 + 60 s of a file


def test_the_live_queue_per_language_is_short_and_old_lines_are_skipped(monkeypatch):
    sent = []

    async def line(http, text, source, target, context=None):
        sent.append(text)
        return f"[{target}] {text}"

    monkeypatch.setattr(live_translate.translate, "translate_line", line)
    events = []

    async def run():
        t = live_translate.LiveTranslator(["es"], "en", events.append)
        clock = [1000.0]
        monkeypatch.setattr(live_translate, "time", SimpleNamespace(monotonic=lambda: clock[0]))  # only this module's clock
        for i in range(30):  # a burst: only the newest lines fit
            t.offer({"type": "caption", "id": f"l{i}", "text": f"line {i}", "final": True, "start": float(i), "end": i + 1.0})
        clock[0] += live_translate.MAX_LINE_AGE + 1  # and the answers were so slow that they are all stale now
        t.offer({"type": "caption", "id": "fresh", "text": "fresh line", "final": True, "start": 99.0, "end": 100.0})
        await t.close()
        return t

    t = asyncio.run(run())
    assert sent == ["fresh line"]  # the burst was cut to the newest few, and those had gone stale
    assert [e["of"] for e in events if e["type"] == "translation"] == ["fresh"]
    assert t.total_seconds() == 1.0  # only what was translated is billed
