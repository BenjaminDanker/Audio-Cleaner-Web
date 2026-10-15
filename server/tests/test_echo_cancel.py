"""Live echo cancellation end to end: client -> /api/stream/ws -> stream worker -> echo canceller."""
import asyncio
import threading

import numpy as np
import pytest

from acw import cpusplit
from acw.billing import ledger
from acw.stream import engine, worker
from acw.stream import protocol as P
from acw_common.aec import SAMPLE_RATE

from .conftest import login
from .test_api_security import make_key

FRAME = 960
SECONDS = 8


class PassChain:
    """Stands in for the denoiser, so the test measures only what comes before it."""

    latency_samples = 0

    def __init__(self, *_args):
        pass

    def process(self, x):
        return x

    def set_atten_lim(self, db):
        pass

    def close(self):
        pass


class RecordingCaptioner:
    fed: list = []

    def __init__(self, *_args):
        RecordingCaptioner.fed = []

    def feed(self, x):
        RecordingCaptioner.fed.append(x)

    def flush(self):
        pass

    def close(self):
        pass


def room(seconds=SECONDS):
    """(mic, far): far-end noise played by the speakers and heard by the mic 40 ms later, 8 dB down,
    plus the near-end voice (a tone) in the last two seconds only."""
    rng = np.random.default_rng(1)
    n = SAMPLE_RATE * seconds
    far = np.convolve(rng.standard_normal(n), np.ones(8) / 8, mode="same").astype(np.float32) * 0.2
    echo = np.concatenate([np.zeros(1920, np.float32), far[:-1920]]) * 0.4
    near = np.zeros(n, np.float32)
    t = np.arange(2 * SAMPLE_RATE) / SAMPLE_RATE
    near[-2 * SAMPLE_RATE:] = 0.2 * np.sin(2 * np.pi * 330 * t)
    return (echo + near).astype(np.float32), far


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


@pytest.fixture
def live(monkeypatch, tmp_path):
    """A stream worker (denoiser replaced) listening where the web side expects it."""
    monkeypatch.setattr(cpusplit, "units_per_core", lambda: 2.4)
    monkeypatch.setattr(worker, "LiveChain", PassChain)
    monkeypatch.setattr(worker, "StreamingCaptioner", RecordingCaptioner)
    monkeypatch.setattr("acw.stream.routes.PACE_BURST_SECONDS", 3600.0)
    monkeypatch.setenv("ACW_STREAM_SOCKET", str(tmp_path / "s.sock"))
    from acw.config import get_settings

    get_settings.cache_clear()
    sock = get_settings().stream_worker_socket
    loop = asyncio.new_event_loop()
    server = loop.run_until_complete(asyncio.start_unix_server(worker.handle, path=str(sock)))
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield

    async def drain():  # let the sessions of this test finish
        if tasks := {t for t in asyncio.all_tasks() if t is not asyncio.current_task()}:
            await asyncio.wait(tasks, timeout=5)

    asyncio.run_coroutine_threadsafe(drain(), loop).result(10)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(10)
    server.close()
    loop.close()


def stream(client, mic, far, start, codec=P.CODEC_S16, send_reference="before"):
    """Sends mic (and far as the reference, before or after its mic frame, or not at all) frame by frame; returns (ready message, processed audio)."""
    uid, _ = login(client)
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    key = make_key(uid)
    client.cookies.clear()
    enc = engine.OpusCodec(128_000) if codec == P.CODEC_OPUS else None
    ref_enc = engine.OpusCodec(128_000) if enc else None
    dec = engine.OpusCodec() if enc else None
    out = []

    def frame(x, seq, flags=0):
        coder = ref_enc if flags else enc
        body = coder.encode(x) if coder else (x * 32767).astype("<i2").tobytes()
        return P.HEADER.pack(P.VERSION, codec, flags, seq, seq * 20_000) + body

    with client.websocket_connect("/api/stream/ws", headers={"Authorization": f"Bearer {key}"}) as ws:
        ws.send_json({"type": "start", "version": 1, "denoise": "light", "sampleRate": SAMPLE_RATE, **start})
        ready = ws.receive_json()
        assert ready["type"] == "ready", ready
        for seq in range(len(mic) // FRAME):
            a = slice(seq * FRAME, (seq + 1) * FRAME)
            if send_reference == "before":
                ws.send_bytes(frame(far[a], seq, P.FLAG_REFERENCE))
            ws.send_bytes(frame(mic[a], seq))
            if send_reference == "after":
                ws.send_bytes(frame(far[a], seq, P.FLAG_REFERENCE))
            reply = ws.receive_bytes()  # one frame back per mic frame, never one for the reference
            _, rseq, ts = P.frame_header(reply)[:3]
            assert (rseq, ts) == (seq, seq * 20_000)
            out.append(dec.decode(reply[P.HEADER.size:]) if dec else P.decode_frame(reply)[3])
        ws.send_json({"type": "stop"})
    return ready, np.concatenate(out)


@pytest.mark.parametrize("codec, min_db", [(P.CODEC_S16, 20), (P.CODEC_OPUS, 20)])
def test_reference_removes_the_echo_and_keeps_the_voice(live, client, codec, min_db):
    mic, far = room()
    ready, y = stream(client, mic, far, {"echoReference": True, "captions": "local", "language": "en"}, codec)
    assert ready["echoReference"] is True
    assert ready["modelLatencyMs"] == 9.0  # the canceller's own delay is part of what clients budget for
    n = SAMPLE_RATE
    echo_only = slice(5 * n, 6 * n)  # after it has settled, before the voice starts
    removed_db = 20 * np.log10(rms(mic[echo_only]) / rms(y[echo_only]))
    assert removed_db > min_db, removed_db
    voice = slice(-n, None)
    assert 0.8 < rms(y[voice]) / rms(mic[voice]) < 1.2  # the near-end voice is left alone
    captioned = np.concatenate(RecordingCaptioner.fed)  # captions hear the echo-free original too
    assert len(captioned) == len(mic)
    assert rms(mic[echo_only]) / rms(captioned[echo_only]) > 10 ** (min_db / 20)


def test_without_a_reference_the_audio_is_unchanged(live, client):
    mic, far = room(3)
    ready, y = stream(client, mic, far, {}, send_reference=None)
    assert "echoReference" not in ready and ready["modelLatencyMs"] == 0.0
    assert np.max(np.abs(y - mic)) < 1e-3
    # reference frames from a session that never asked for them are ignored, not mixed in
    ready, y = stream(client, mic, far, {}, send_reference="before")
    assert np.max(np.abs(y - mic)) < 1e-3


def test_late_reference_counts_as_silence(live, client):
    mic, far = room(3)
    _, y = stream(client, mic, far, {"echoReference": True}, send_reference="after")
    assert len(y) == len(mic) and 0.9 < rms(y) / rms(mic) < 1.1  # nothing to cancel against, one frame back each


def test_echo_cancellation_adds_capacity_units_but_is_never_refused_alone(monkeypatch):
    plain = cpusplit.session_units({"denoise": "light"})
    assert 0 < cpusplit.session_units({"denoise": "light", "echo_reference": True}) - plain < 0.5
    monkeypatch.setattr(worker, "_monitor", None)
    monkeypatch.setattr(cpusplit, "units_per_core", lambda: 2.4)
    monkeypatch.setattr(worker, "_units", 1000.0)  # live is full
    assert not worker.admit({"denoise": "light", "echo_reference": True})[0]
    assert worker.admit({"denoise": None, "captions": "premium", "echo_reference": True}) == (True, [])
