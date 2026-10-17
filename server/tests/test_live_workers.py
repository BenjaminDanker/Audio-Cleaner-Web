"""The web side of live sessions (2026-10-16/17 efficiency work): file jobs keep off the cores the live engine uses,
silence frames, returnAudio, captions-only sessions billed for speech, extra-accurate audio in 100 ms pieces, several
spoken languages."""
import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

from acw import broker
from acw.billing import ledger
from acw.config import get_settings
from acw.stream import routes
from acw.stream import protocol as P

from .test_live_trial import fake_worker, run_session  # noqa: F401 - fixture


def test_extra_accurate_audio_goes_in_100_ms_pieces_and_is_committed_at_pauses():
    from acw.stream.openai_live import PremiumCaptions

    sent = []

    class FakeWs:
        async def send(self, m):
            sent.append(json.loads(m))

    p = PremiumCaptions(["en", "es"], lambda ev: None)
    assert p.languages == ["en", "es"]
    p.ws = FakeWs()
    talk = np.full(960, 0.1, np.float32)
    quiet = np.zeros(960, np.float32)

    async def run():
        for _ in range(10):  # 200 ms of talk
            await p.feed(talk)
        for _ in range(40):  # 800 ms quiet: the turn ends after 600 ms
            await p.feed(quiet)

    asyncio.run(run())
    appends = [m for m in sent if m["type"] == "input_audio_buffer.append"]
    assert len(appends) == 8  # 0.8 s sent (talk + the pause until the commit) in 100 ms pieces, not 40 messages
    assert sent[-1]["type"] == "input_audio_buffer.commit"
    assert sum(len(m["audio"]) for m in appends) * 3 // 4 == 40 * 480 * 2  # every 24 kHz sample went out once


def test_a_speaker_may_switch_between_up_to_three_languages(client, signed_in, fake_worker, monkeypatch):
    uid, _ = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()
    made = []

    class FakePremium:
        def __init__(self, languages, *a, **k):
            made.append(languages)
            self.failed = False

        async def connect(self):
            pass

        async def feed(self, y):
            pass

        async def close(self):
            pass

    monkeypatch.setattr("acw.stream.openai_live.PremiumCaptions", FakePremium)
    got = run_session(client, 0.1, {"denoise": None, "captions": "premium", "language": ["en", "ES"], "translate": ["es"]})
    assert got[0]["type"] == "ready" and made == [["en", "es"]]  # a target that is one of the spoken languages is fine
    got = run_session(client, 0.1, {"captions": "premium", "language": ["en", "es", "de", "fr"]})
    assert got[0]["code"] == "bad_config"
    got = run_session(client, 0.1, {"captions": "local", "language": ["en", "es"]})
    assert got[0]["code"] == "bad_config" and "English only" in got[0]["message"]


# --------------------------------------------------------------- silence and audio nobody needs (2026-10-16)

def _session(client, start, frames):
    """Start, send `frames` (bytes), stop; every JSON message and binary frame that came back. Frames are answered in
    order, so the stop comes after the last of them."""
    from starlette.websockets import WebSocketDisconnect

    from .conftest import ORIGIN

    got, binary = [], []
    with client.websocket_connect("/api/stream/ws", headers={"origin": ORIGIN}) as ws:
        ws.send_text(json.dumps({"type": "start", "sampleRate": 48000, **start}))
        got.append(ws.receive_json())
        if got[0]["type"] != "ready":
            return got, binary
        for f in frames:
            ws.send_bytes(f)
        ws.send_text(json.dumps({"type": "stop"}))
        try:
            while True:
                m = ws.receive()
                if m.get("bytes") is not None:
                    binary.append(m["bytes"])
                elif m.get("text"):
                    got.append(json.loads(m["text"]))
                elif m["type"] == "websocket.close":
                    break
        except WebSocketDisconnect:
            pass
    return got, binary


@pytest.fixture
def echo_worker(monkeypatch):
    """A worker that answers every audio frame with itself and records what it was sent."""
    state = SimpleNamespace(sent=[], events=[])

    async def open_conn(_path):
        q = asyncio.Queue()
        return SimpleNamespace(q=q, first=True), SimpleNamespace(q=q, close=lambda: None)

    async def send(writer, kind, payload):
        state.sent.append((kind, payload))
        if kind == P.MSG_AUDIO:
            await writer.q.put((P.MSG_AUDIO, payload))
        elif kind == P.MSG_END:
            for ev in state.events:
                await writer.q.put((P.MSG_EVENT, P.pack_json(ev)))
            await writer.q.put((P.MSG_END, b""))

    async def read(reader):
        if reader.first:
            reader.first = False
            return P.MSG_EVENT, P.pack_json({"type": "ready", "latencySamples": 0, "cores": [0, 1]})
        return await reader.q.get()

    monkeypatch.setattr(routes.asyncio, "open_unix_connection", open_conn)
    monkeypatch.setattr(routes.P, "send_msg", send)
    monkeypatch.setattr(routes.P, "read_msg", read)
    return state


def _pcm(seq):
    return P.HEADER.pack(P.VERSION, P.CODEC_S16, 0, seq, seq * 20_000) + (np.full(960, 3000, "<i2")).tobytes()


def test_silence_frames_are_answered_at_once_and_not_billed(client, signed_in, echo_worker):
    uid, _ = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    frames = [_pcm(0), P.silence_frame(1, 20_000, 960), P.silence_frame(2, 40_000, 960), _pcm(3)]
    got, binary = _session(client, {"denoise": "light"}, frames)
    assert got[0]["silenceFrames"] is True
    silent = [b for b in binary if P.frame_header(b)[0] == P.CODEC_SILENCE]
    assert sorted(P.frame_header(b)[1] for b in silent) == [1, 2] and all(P.silence_samples(b) == 960 for b in silent)
    kinds = [k for k, _ in echo_worker.sent]
    assert kinds.count(P.MSG_AUDIO) == 2 and kinds.count(P.MSG_SILENCE) == 2  # the worker only hears how much
    from acw import db

    assert db.one("SELECT seconds FROM stream_sessions")["seconds"] == pytest.approx(0.04)  # the two real frames only


def test_a_client_that_needs_no_audio_back_gets_none(client, signed_in, echo_worker):
    uid, _ = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    got, binary = _session(client, {"denoise": "light", "returnAudio": False}, [_pcm(0), _pcm(1), P.silence_frame(2, 0, 960)])
    assert got[0]["type"] == "ready" and binary == []
    start = json.loads(next(p for k, p in echo_worker.sent if k == P.MSG_START))
    assert start["return_audio"] is False


def test_standard_captions_only_are_not_billed_for_the_silence_they_skipped(client, signed_in, echo_worker):
    uid, _ = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    echo_worker.events = [{"samples": 960, "type": "_idle"}]  # the engine writes keys in this order
    got, _ = _session(client, {"denoise": None, "captions": "local"}, [_pcm(i) for i in range(3)])
    assert not any(m.get("type") == "_idle" for m in got)  # internal: never reaches the client
    from acw import db

    assert db.one("SELECT seconds FROM stream_sessions")["seconds"] == pytest.approx(0.04)
    # with noise removal the whole time is work: billed as before
    db.execute("DELETE FROM stream_sessions")
    _session(client, {"denoise": "light", "captions": "local"}, [_pcm(i) for i in range(3)])
    assert db.one("SELECT seconds FROM stream_sessions")["seconds"] == pytest.approx(0.06)


def test_the_cores_in_use_are_published_for_file_jobs(client, signed_in, echo_worker, monkeypatch):
    uid, _ = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    seen = []
    monkeypatch.setattr(broker, "publish_live_cores", lambda cores, data_dir=None: seen.append(sorted(cores)))
    _session(client, {"denoise": "light"}, [_pcm(0)])
    assert seen == [[0, 1], []]  # the engine said cores 0 and 1 (echo_worker's ready); free again afterwards


def test_file_jobs_keep_off_live_cores_and_one_spare(tmp_path, monkeypatch):
    monkeypatch.setattr(broker, "physical_cores", lambda: [[i, i + 8] for i in range(8)])
    monkeypatch.setattr(broker, "max_live_cores", lambda: 7)
    broker.publish_live_cores([2, 3], tmp_path)
    assert broker.job_cpus(tmp_path) == {c for i in (1, 4, 5, 6, 7) for c in (i, i + 8)}  # core 0: the spare
    broker.publish_live_cores([], tmp_path)
    assert broker.job_cpus(tmp_path) == broker.all_cpus()
