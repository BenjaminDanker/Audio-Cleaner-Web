"""Live caption translations: what a start may ask for, every final caption translated per language in order,
billed by the speech translated, a language that keeps failing switched off with a warning and not billed."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from starlette.websockets import WebSocketDisconnect

from acw import db
from acw.billing import ledger, pricing
from acw.config import get_settings
from acw.stream import live_translate, routes
from acw.stream import protocol as P

from .conftest import ORIGIN


@pytest.fixture
def worker(monkeypatch):
    """A live worker that says ready, then sends `events` (caption events), then ends when told to."""
    state = SimpleNamespace(events=[])

    async def open_conn(_path):
        ended = asyncio.Event()
        return SimpleNamespace(step=0, ended=ended), SimpleNamespace(ended=ended, close=lambda: None)

    async def send(writer, kind, payload):
        if kind == P.MSG_END:
            writer.ended.set()

    async def read(reader):
        reader.step += 1
        if reader.step == 1:
            return P.MSG_EVENT, P.pack_json({"type": "ready", "latencySamples": 0})
        if reader.step - 2 < len(state.events):
            await asyncio.sleep(0.01)
            return P.MSG_EVENT, P.pack_json(state.events[reader.step - 2])
        await reader.ended.wait()
        return P.MSG_END, b""

    monkeypatch.setattr(routes.asyncio, "open_unix_connection", open_conn)
    monkeypatch.setattr(routes.P, "send_msg", send)
    monkeypatch.setattr(routes.P, "read_msg", read)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()
    return state


def fake_translations(monkeypatch, fail=()):
    calls = []

    async def line(http, text, source, target, context=None):
        calls.append((target, text, list(context or [])))
        if target in fail:
            raise RuntimeError("down")
        return f"[{target}] {text}"

    monkeypatch.setattr(live_translate.translate, "translate_line", line)
    return calls


def session(client, start, seconds=1, wait=0.5):
    got = []
    with client.websocket_connect("/api/stream/ws", headers={"origin": ORIGIN}) as ws:
        ws.send_text(json.dumps({"type": "start", "sampleRate": 48000, **start}))
        got.append(ws.receive_json())
        if got[0]["type"] != "ready":
            return got
        for i in range(int(seconds * 100)):
            ws.send_bytes(P.HEADER.pack(P.VERSION, P.CODEC_S16, 0, i, 0) + bytes(960))
        import time

        time.sleep(wait)  # the captions come and get translated meanwhile
        ws.send_text(json.dumps({"type": "stop"}))
        try:
            while True:
                got.append(ws.receive_json())
        except WebSocketDisconnect:
            pass
    return got


def test_what_a_start_may_ask_for(client, signed_in, worker):
    uid, _ = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    for start, why in (({"denoise": "light", "translate": ["es"]}, "need captions"),
                       ({"captions": "local", "translate": ["en"]}, "same as the spoken"),
                       ({"captions": "local", "translate": ["xx"]}, "unknown"),
                       ({"captions": "local", "translate": ["es", "de", "fr", "it"]}, "at most 3")):
        got = session(client, start)
        assert got[0]["code"] == "bad_config" and why in got[0]["message"], (start, got)


def test_final_captions_are_translated_in_order_and_billed_by_speech(client, signed_in, worker, monkeypatch):
    uid, _ = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    calls = fake_translations(monkeypatch)
    worker.events = [
        {"type": "caption", "id": "l0", "text": "hello", "final": False},
        {"type": "caption", "id": "l0", "text": "hello there", "final": True, "start": 0.0, "end": 2.0},
        {"type": "caption", "id": "l1", "text": "good game", "final": True, "start": 3.0, "end": 4.5},
    ]
    before = ledger.balance(uid)["balance_micros"]
    got = session(client, {"captions": "local", "translate": ["es", "de"]})
    assert got[0]["type"] == "ready" and got[0]["translate"] == ["es", "de"]
    tr = [m for m in got if m["type"] == "translation"]
    assert [(m["lang"], m["of"], m["text"]) for m in tr if m["lang"] == "es"] == [("es", "l0", "[es] hello there"), ("es", "l1", "[es] good game")]
    assert {m["lang"] for m in tr} == {"es", "de"} and tr[0]["start"] == 0.0
    assert all(text != "hello" for _, text, _ in calls)  # partials are never translated
    assert ("es", "good game", ["hello there"]) in calls  # earlier lines go along as context
    row = db.one("SELECT amount_micros, description FROM ledger WHERE ref_id LIKE '%:translate'")
    assert row["amount_micros"] == -pricing.cost_micros("stream_translate", "standard", 2 * 3.5)
    assert "translation" in row["description"]
    assert before - ledger.balance(uid)["balance_micros"] == db.one("SELECT cost_micros FROM stream_sessions")["cost_micros"]


def test_a_language_that_keeps_failing_is_switched_off_and_not_billed(client, signed_in, worker, monkeypatch):
    uid, _ = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    fake_translations(monkeypatch, fail={"de"})
    worker.events = [{"type": "caption", "id": f"l{i}", "text": f"line {i}", "final": True, "start": float(i), "end": i + 0.5}
                     for i in range(4)]
    got = session(client, {"captions": "local", "translate": ["es", "de"]})
    warn = [m for m in got if m.get("code") == "translation_unavailable"]
    assert len(warn) == 1 and warn[0]["lang"] == "de" and "not charged" in warn[0]["message"]
    assert not any(m["type"] == "translation" and m["lang"] == "de" for m in got)
    assert sum(m["type"] == "translation" and m["lang"] == "es" for m in got) == 4
    row = db.one("SELECT amount_micros FROM ledger WHERE ref_id LIKE '%:translate'")
    assert row["amount_micros"] == -pricing.cost_micros("stream_translate", "standard", 4 * 0.5)  # Spanish only


def test_a_session_without_translations_has_no_translation_charge(client, signed_in, worker, monkeypatch):
    uid, _ = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    fake_translations(monkeypatch)
    worker.events = [{"type": "caption", "id": "l0", "text": "hi", "final": True, "start": 0.0, "end": 1.0}]
    got = session(client, {"captions": "local"})
    assert "translate" not in got[0] and not any(m["type"] == "translation" for m in got)
    assert db.one("SELECT COUNT(*) n FROM ledger WHERE ref_id LIKE '%:translate'")["n"] == 0
