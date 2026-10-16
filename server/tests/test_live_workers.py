"""Live audio across several worker processes (2026-10-16 efficiency audit): sessions fill the workers in order and
move on when one is full, file jobs keep off the cores in use, and the cheaper per-session pieces (libopus, batched
caption audio, several spoken languages)."""
import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

from acw import cpusplit
from acw.billing import ledger
from acw.config import get_settings
from acw.stream import engine, routes
from acw.stream import protocol as P

from .test_live_trial import run_session


@pytest.fixture
def workers(monkeypatch, tmp_path):
    """Three live workers on their own sockets; `replies[name]` is what each answers to a start."""
    monkeypatch.setenv("ACW_STREAM_SOCKET", str(tmp_path / "stream.sock"))
    get_settings.cache_clear()
    for name in ("stream.sock", "stream-2.sock", "stream-3.sock", "stream-x.sock"):
        (tmp_path / name).touch()
    cores = {"stream.sock": [0, 1], "stream-2.sock": [2, 3], "stream-3.sock": [4, 5]}
    state = SimpleNamespace(replies={n: {"type": "ready", "latencySamples": 0, "cores": c} for n, c in cores.items()},
                            tried=[], started={})

    async def open_conn(path):
        name = path.rsplit("/", 1)[-1]
        state.tried.append(name)
        if state.replies.get(name) is None:
            raise ConnectionRefusedError(name)
        ended = asyncio.Event()
        return SimpleNamespace(name=name, first=True, ended=ended), SimpleNamespace(name=name, ended=ended, close=lambda: None)

    async def send(writer, kind, payload):
        if kind == P.MSG_START:
            state.started[writer.name] = json.loads(payload)
        elif kind == P.MSG_END:
            writer.ended.set()

    async def read(reader):
        if reader.first:
            reader.first = False
            return P.MSG_EVENT, P.pack_json(state.replies[reader.name])
        await reader.ended.wait()
        return P.MSG_END, b""

    monkeypatch.setattr(routes.asyncio, "open_unix_connection", open_conn)
    monkeypatch.setattr(routes.P, "send_msg", send)
    monkeypatch.setattr(routes.P, "read_msg", read)
    return state


def test_workers_are_found_by_their_sockets_in_order(workers):
    assert [p.name for p in routes.worker_sockets()] == ["stream.sock", "stream-2.sock", "stream-3.sock"]


def test_a_full_worker_passes_the_session_to_the_next(client, signed_in, workers):
    uid, _ = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    workers.replies["stream.sock"] = {"type": "error", "code": "full", "available": [{"captions": "premium"}]}
    workers.replies["stream-2.sock"] = None  # restarting
    got = run_session(client, 0.1)
    assert got[0]["type"] == "ready"
    assert workers.tried == ["stream.sock", "stream-2.sock", "stream-3.sock"] and list(workers.started) == ["stream.sock", "stream-3.sock"]


def test_when_every_worker_is_full_the_client_hears_what_still_fits(client, signed_in, workers):
    uid, _ = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    for name in workers.replies:
        workers.replies[name] = {"type": "error", "code": "full", "available": [{"captions": "premium"}]}
    got = run_session(client, 0.1, {"captions": "local"})
    assert got[0]["code"] == "full" and got[0]["available"] == [{"captions": "premium"}]


def test_the_cores_in_use_are_published_for_file_jobs(client, signed_in, workers, monkeypatch):
    uid, _ = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    seen = []
    monkeypatch.setattr(cpusplit, "set_live_units", lambda d, units, per_core, cores=None: seen.append((units, sorted(cores or []))))
    run_session(client, 0.1)
    assert (1.0, [0, 1]) in seen and seen[-1] == (0.0, [])


def test_live_cores_are_split_into_pairs(monkeypatch):
    monkeypatch.setattr(cpusplit, "max_live_cores", lambda: 7)
    monkeypatch.delenv("ACW_STREAM_WORKERS", raising=False)
    assert cpusplit.stream_workers() == 4
    assert [cpusplit.stream_worker_cores(i, 4) for i in range(1, 5)] == [[0, 1], [2, 3], [4, 5], [6]]
    monkeypatch.setenv("ACW_STREAM_WORKERS", "1")
    assert cpusplit.stream_workers() == 1 and cpusplit.stream_worker_cores(1, 1) == list(range(7))


def test_file_jobs_keep_off_busy_workers_and_one_spare_core(tmp_path, monkeypatch):
    monkeypatch.setattr(cpusplit, "physical_cores", lambda: [[i, i + 8] for i in range(8)])
    monkeypatch.setattr(cpusplit, "max_live_cores", lambda: 7)
    cpusplit.set_live_units(tmp_path, 3.0, 4.3, [2, 3])
    assert cpusplit.job_cpus_now(tmp_path) == {c for i in (1, 4, 5, 6, 7) for c in (i, i + 8)}  # core 0: the spare
    cpusplit.set_live_units(tmp_path, 0.0, 4.3, [])
    assert cpusplit.job_cpus_now(tmp_path) == cpusplit.all_cpus()
    (tmp_path / "live-units").write_text("3.0 2.4")  # written by an older web process: by units, as before
    assert cpusplit.job_cpus_now(tmp_path) == cpusplit.job_cpus(3.0, 2.4)


def test_libopus_round_trip_and_bad_packets():
    if engine._libopus() is None:
        pytest.skip("libopus not installed")
    x = (0.3 * np.sin(np.arange(48_000) * 2 * np.pi * 440 / 48_000)).astype(np.float32)
    enc, dec = engine.OpusCodec(96_000), engine.OpusCodec(encoder=False)
    y = np.concatenate([dec.decode(enc.encode(x[i:i + 960])) for i in range(0, len(x), 960)])
    assert len(y) == len(x) and abs(np.sqrt(np.mean(y[9600:] ** 2)) - np.sqrt(np.mean(x ** 2))) < 0.02
    with pytest.raises(ValueError):
        dec.decode(b"\xff\xff\xff")  # not a 20 ms packet
    enc.close()
    dec.close()


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


def test_a_speaker_may_switch_between_up_to_three_languages(client, signed_in, workers, monkeypatch):
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


def test_digital_silence_is_not_run_through_the_live_model():
    calls = []

    class FakeDf:
        delay = 1440

        def process(self, x):
            calls.append(len(x))
            return x.copy()

    chain = engine.LiveChain.__new__(engine.LiveChain)
    engine.LiveChain.__init__(chain, None, None, None)  # no model: then give it the fake one
    chain.df = FakeDf()
    voice = np.full(960, 0.1, np.float32)
    zeros = np.zeros(960, np.float32)
    for _ in range(5):
        chain.process(voice)
    out = [chain.process(zeros) for _ in range(50)]  # 1 s of a noise gate's silence
    assert not np.any(np.concatenate(out[engine.SILENT_SKIP_AFTER // 960:]))  # skipped: silence out
    assert len(calls) == 5 + engine.SILENT_SKIP_AFTER // 960  # the model only saw the first 200 ms of it
    assert np.any(chain.process(voice)) and len(calls) == 5 + engine.SILENT_SKIP_AFTER // 960 + 1  # sound: back at once


# --------------------------------------------------------------- silence and audio nobody needs (2026-10-16)

def _session(client, start, frames, wait=0.3):
    """Start, send `frames` (bytes), stop; every JSON message and binary frame that came back."""
    import time

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
        time.sleep(wait)
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
    echo_worker.events = [{"type": "_idle", "samples": 960}]
    got, _ = _session(client, {"denoise": None, "captions": "local"}, [_pcm(i) for i in range(3)])
    assert not any(m.get("type") == "_idle" for m in got)  # internal: never reaches the client
    from acw import db

    assert db.one("SELECT seconds FROM stream_sessions")["seconds"] == pytest.approx(0.04)
    # with noise removal the whole time is work: billed as before
    db.execute("DELETE FROM stream_sessions")
    _session(client, {"denoise": "light", "captions": "local"}, [_pcm(i) for i in range(3)])
    assert db.one("SELECT seconds FROM stream_sessions")["seconds"] == pytest.approx(0.06)


def test_the_captioner_ends_the_line_then_only_counts_silence_frames(monkeypatch):
    from .test_capacity_limits import _captioner, _talk

    c, events = _captioner(monkeypatch)
    for _ in range(25):
        c.feed(_talk())
    c._take_audio()
    before = c.stream.fed
    c.silence(48_000 * 10)  # 10 s muted
    c._take_audio()
    assert abs((c.stream.fed - before) - engine.GATE_AFTER_S * 16_000) < 10  # trailing silence for the line to end
    assert abs(c._fed - 10.5 * 16_000) < 40  # the clock still follows (the resampler holds back a few samples)
    assert not any(e.get("type") == "_idle" for e in events)  # silence frames were never billed: nothing to give back
