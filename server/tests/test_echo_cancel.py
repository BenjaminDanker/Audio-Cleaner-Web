"""Live echo cancellation end to end: client -> /api/stream/ws -> the native live engine -> echo canceller."""
import numpy as np
import pytest

from acw.billing import ledger
from acw.stream import protocol as P

from . import opus
from .conftest import login
from .test_api_security import make_key

SAMPLE_RATE = 48_000
FRAME = 960
SECONDS = 8


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
def live(live_engine, monkeypatch):
    monkeypatch.setattr("acw.stream.routes.PACE_BURST_SECONDS", 3600.0)  # the test sends faster than real time
    return live_engine()


def stream(client, mic, far, start, codec=P.CODEC_S16, send_reference="before"):
    """Sends mic (and far as the reference, before or after its mic frame, or not at all) frame by frame; returns (ready message, processed audio)."""
    uid, _ = login(client)
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    key = make_key(uid)
    client.cookies.clear()
    enc = opus.Encoder(128_000) if codec == P.CODEC_OPUS else None
    ref_enc = opus.Encoder(128_000) if enc else None
    dec = opus.Decoder() if enc else None
    out = []
    texts = []

    def frame(x, seq, flags=0):
        coder = ref_enc if flags else enc
        body = coder.encode(x) if coder else (x * 32767).astype("<i2").tobytes()
        return P.HEADER.pack(P.VERSION, codec, flags, seq, seq * 20_000) + body

    with client.websocket_connect("/api/stream/ws", headers={"Authorization": f"Bearer {key}"}) as ws:
        ws.send_json({"type": "start", "version": 1, "denoise": None, "captions": "local", "sampleRate": SAMPLE_RATE, **start})
        ready = ws.receive_json()
        assert ready["type"] == "ready", ready
        for seq in range(len(mic) // FRAME):
            a = slice(seq * FRAME, (seq + 1) * FRAME)
            if send_reference == "before":
                ws.send_bytes(frame(far[a], seq, P.FLAG_REFERENCE))
            ws.send_bytes(frame(mic[a], seq))
            if send_reference == "after":
                ws.send_bytes(frame(far[a], seq, P.FLAG_REFERENCE))
            while True:  # one frame back per mic frame, never one for the reference (captions come as text between)
                m = ws.receive()
                assert m["type"] == "websocket.send", (m, texts)
                if m.get("text"):
                    texts.append(m["text"])
                if m.get("bytes") is not None:
                    reply = m["bytes"]
                    break
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
    assert ready["modelLatencyMs"] == 39.0  # the same in every session: the model's 30 ms and the canceller's 9
    n = SAMPLE_RATE
    echo_only = slice(5 * n, 6 * n)  # after it has settled, before the voice starts
    removed_db = 20 * np.log10(rms(mic[echo_only]) / rms(y[echo_only]))
    assert removed_db > min_db, removed_db
    voice = slice(-n, None)
    assert 0.8 < rms(y[voice]) / rms(mic[voice]) < 1.2  # the near-end voice is left alone


def test_without_a_reference_the_audio_is_unchanged(live, client):
    mic, far = room(3)
    ready, y = stream(client, mic, far, {}, send_reference=None)
    assert "echoReference" not in ready and ready["modelLatencyMs"] == 39.0
    d = 1872  # the session's delay: nothing runs, the audio only waits
    assert np.max(np.abs(y[d:] - mic[:-d])) < 1e-3
    # reference frames from a session that never asked for them are ignored, not mixed in
    ready, y = stream(client, mic, far, {}, send_reference="before")
    assert np.max(np.abs(y[d:] - mic[:-d])) < 1e-3


def test_late_reference_counts_as_silence(live, client):
    mic, far = room(3)
    _, y = stream(client, mic, far, {"echoReference": True}, send_reference="after")
    assert len(y) == len(mic) and 0.9 < rms(y) / rms(mic) < 1.1  # nothing to cancel against, one frame back each
