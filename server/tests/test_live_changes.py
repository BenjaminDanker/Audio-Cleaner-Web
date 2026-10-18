"""Changing a live session's features while it runs (a `config` message): client -> /api/stream/ws -> the native live
engine. The audio keeps flowing with the same delay, and each stretch is billed at the rate it had."""
import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from acw import db
from acw.billing import ledger
from acw.config import get_settings
from acw.stream import protocol as P
from acw.stream.features import Features
from acw.stream.meter import micros as meter_micros

from .conftest import login
from .test_api_security import make_key
from .test_echo_cancel import engine, rms, room  # noqa: F401 - engine: one live engine for this module

SAMPLE_RATE = 48_000
FRAME = 960
LATENCY = 1872  # every session: the model's 1440 samples and the echo canceller's 432


@pytest.fixture
def live(engine, monkeypatch):
    monkeypatch.setattr("acw.stream.routes.PACE_BURST_SECONDS", 3600.0)  # the test sends faster than real time
    return engine


class Live:
    """One session over the test client: frames go out one at a time and their answers are collected."""

    def __init__(self, ws):
        self.ws, self.seq, self.out, self.texts = ws, 0, [], []

    def send(self, mic, far=None):
        for i in range(0, len(mic) - FRAME + 1, FRAME):
            if far is not None:
                self.ws.send_bytes(self._frame(far[i:i + FRAME], P.FLAG_REFERENCE))
            self.ws.send_bytes(self._frame(mic[i:i + FRAME]))
            while True:
                m = self.ws.receive()
                if m.get("text"):
                    self.texts.append(m["text"])
                if m.get("bytes") is not None:
                    _, rseq, ts = P.frame_header(m["bytes"])[:3]
                    assert (rseq, ts) == (self.seq, self.seq * 20_000)
                    self.out.append(P.decode_frame(m["bytes"])[3])
                    break
            self.seq += 1

    def _frame(self, x, flags=0):
        return P.HEADER.pack(P.VERSION, P.CODEC_S16, flags, self.seq, self.seq * 20_000) + (x * 32767).astype("<i2").tobytes()

    def config(self, **changes):
        """Sends a change and returns the answer to it."""
        import json

        self.ws.send_json({"type": "config", "id": f"c{self.seq}", **changes})
        while True:
            m = self.ws.receive_json()
            if m.get("type") == "config":
                assert m["id"] == f"c{self.seq}"
                return m
            self.texts.append(json.dumps(m))

    def audio(self):
        return np.concatenate(self.out)


def session(client, live_client, credit=5_000_000, **start):
    uid, _ = login(client)
    if credit:
        ledger.credit(uid, credit, "grant", "t", "g")
    key = make_key(uid)
    client.cookies.clear()
    ws = live_client.websocket_connect("/api/stream/ws", headers={"Authorization": f"Bearer {key}"})
    w = ws.__enter__()
    w.send_json({"type": "start", "version": 1, "sampleRate": SAMPLE_RATE, **start})
    ready = w.receive_json()
    return uid, ws, Live(w), ready


def end(ws, live):
    """Stop the session and return its row once the server has settled it."""
    from starlette.websockets import WebSocketDisconnect

    live.ws.send_json({"type": "stop"})
    try:
        while live.ws.receive()["type"] != "websocket.close":
            pass
    except WebSocketDisconnect:
        pass
    ws.__exit__(None, None, None)
    return settled()


def settled():
    for _ in range(50):
        row = db.one("SELECT * FROM stream_sessions ORDER BY started_at DESC LIMIT 1")
        if row["status"] != "active":
            break
        time.sleep(0.1)
    return dict(row)


def tone(seconds, f=220.0):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.2 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_features_change_while_live_and_each_stretch_is_billed_at_its_rate(live, client, live_client):
    uid, ws, s, ready = session(client, live_client, denoise="light")
    assert ready["type"] == "ready" and ready["changeable"] is True and ready["modelLatencyMs"] == 39.0
    noise = Features(denoise="light").rate()
    both = Features(denoise="light", captions="local").rate()
    captions = Features(captions="local").rate()
    s.send(tone(1))  # noise removal only
    got = s.config(captions="local", language="en")  # and captions
    assert got["ok"] and got["usdPerMinute"] == both / 1e6 and got["captions"] == "local"
    s.send(tone(1))
    got = s.config(denoise=None)  # captions only from here
    assert got["ok"] and got["denoise"] is None
    s.send(tone(1, 330))
    got = s.config(denoise="light", attenLimDb=12)  # and back, gentler
    assert got["ok"] and got["attenLimDb"] == 12
    s.send(tone(1))
    assert len(s.audio()) == 4 * SAMPLE_RATE  # one answer per frame throughout: no gap, no restart
    row = end(ws, s)
    # a tone is no speech: in the captions-only stretch it counts as silence, which isn't billed
    assert 3 <= row["paid_seconds"] <= 4
    assert row["cost_micros"] <= math.ceil(round(noise / 60 + both / 60 + captions / 60 + both / 60, 6))
    assert row["cost_micros"] >= math.ceil(round(noise / 60 + both * 2 / 60, 6))
    assert row["denoise_tier"] == "light" and row["caption_tier"] == "local"  # what it ended with
    assert ledger.balance(uid)["held_micros"] == 0


def test_echo_removal_can_be_switched_on_while_live(live, client, live_client):
    mic, far = room(6)
    _, ws, s, _ = session(client, live_client, captions="local", language="en")
    s.send(mic[:2 * SAMPLE_RATE])  # no echo removal yet: the echo comes back
    got = s.config(echoReference=True)
    assert got["ok"] and got["echoReference"] is True
    s.send(mic[2 * SAMPLE_RATE:], far[2 * SAMPLE_RATE:])
    end(ws, s)
    y = s.audio()
    before = slice(SAMPLE_RATE, 2 * SAMPLE_RATE)
    after = slice(int(3.5 * SAMPLE_RATE), 4 * SAMPLE_RATE)  # settled, before the voice starts
    assert rms(y[before]) / rms(mic[before]) > 0.9
    assert 20 * np.log10(rms(mic[after]) / rms(y[after])) > 20


def test_a_change_that_can_not_be_made_leaves_the_session_as_it_was(live, client, live_client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _, ws, s, ready = session(client, live_client, denoise="light")
    s.send(tone(1))
    got = s.config(captions="premium")
    assert not got["ok"] and got["code"] == "unavailable" and "Extra-accurate" in got["message"]
    got = s.config(captions="local", language="de")
    assert not got["ok"] and got["code"] == "bad_config"
    got = s.config(attenLimDb=20)
    assert got["ok"] and got["usdPerMinute"] == ready["usdPerMinute"] and got["captions"] is None
    s.send(tone(1))
    row = end(ws, s)
    assert row["cost_micros"] == meter_micros(0, Features(denoise="light").rate(), 2, 2)


def test_a_change_the_balance_can_not_cover_is_refused(live, client, live_client):
    noise = Features(denoise="light").rate()
    # the hold of the noise removal and a little more: not the next minute at the higher rate
    _, ws, s, _ = session(client, live_client, credit=noise * 5 + 1000, denoise="light")
    s.send(tone(2))
    got = s.config(captions="local", language="en")
    assert not got["ok"] and got["code"] == "insufficient_funds"
    s.send(tone(1))
    row = end(ws, s)
    assert row["caption_tier"] is None and row["cost_micros"] == meter_micros(0, noise, 3, 3)


def test_a_trial_session_moves_on_to_credit_when_it_starts_doing_more_than_the_trial_covers(live, client, live_client, monkeypatch):
    from acw.stream import openai_live

    class Captions:  # extra-accurate captions without the network
        failed = False

        def __init__(self, *a, **k):
            pass

        async def connect(self):
            pass

        async def feed(self, pcm):
            return True

        async def close(self):
            pass

    monkeypatch.setattr(openai_live, "PremiumCaptions", Captions)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()
    uid, _ = login(client)
    db.execute("UPDATE users SET trial_live_seconds = 600 WHERE id = ?", (uid,))
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    key = make_key(uid)
    client.cookies.clear()
    with live_client.websocket_connect("/api/stream/ws", headers={"Authorization": f"Bearer {key}"}) as w:
        w.send_json({"type": "start", "version": 1, "sampleRate": SAMPLE_RATE, "denoise": "light"})
        ready = w.receive_json()
        assert ready["trial"] is True
        s = Live(w)
        s.send(tone(1))
        got = s.config(captions="premium", language="de")
        assert got["ok"] and "trial" not in got
        s.send(tone(1))
        row = end(SimpleNamespace(__exit__=lambda *a: None), s)
    paid = Features(denoise="light", captions="premium").rate()
    assert row["trial_seconds"] == 1 and row["cost_micros"] == meter_micros(0, paid, 1, 1)
    assert db.one("SELECT trial_live_seconds FROM users WHERE id = ?", (uid,))[0] == 599


# --- the room test (the plugin's "Test my room": the noise measured at the noise remover)

def voice_in(snr_db, seconds=7):
    """Bursts of a harmonic voice with white noise `snr_db` under it (the voice of common/tests/test_noise_profile.py: the
    noise remover keeps it as a voice; a buzz of many harmonics it would partly take out as noise)."""
    rng = np.random.default_rng(2)
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    v = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((140, 280, 420, 560), 1)) * (np.sin(2 * np.pi * 1.5 * t) > -0.3)
    noise = rng.standard_normal(len(t)) * np.sqrt(np.mean(v ** 2)) * 10 ** (-snr_db / 20)
    return (0.1 * (v + noise) / np.max(np.abs(v + noise))).astype(np.float32)


def room_test(live_):
    """The room_test event among what came back, or None."""
    import json

    for text in live_.texts:
        ev = json.loads(text)
        if ev.get("type") == "room_test":
            return ev
    return None


def test_the_room_test_measures_the_noise_at_the_noise_remover(live, client, live_client):
    _, ws, s, _ = session(client, live_client, denoise="light", attenLimDb=12)  # a gentle strength: still the full-strength figure
    s.ws.send_json({"type": "room_test", "seconds": 5})
    s.send(voice_in(5, 6))  # 5 dB under the voice
    end(ws, s)
    ev = room_test(s)
    assert ev and ev["ok"] and ev["seconds"] == 5, s.texts
    assert abs(ev["noiseDb"] - 5) < 3, ev


def test_the_room_test_needs_noise_removal(live, client, live_client):
    _, ws, s, _ = session(client, live_client, captions="local", language="en")
    s.ws.send_json({"type": "room_test"})
    ev = s.ws.receive_json()
    assert ev == {"type": "room_test", "ok": False, "code": "needs_noise_removal",
                  "message": "The room test listens through noise removal: turn noise removal on first."}
    end(ws, s)
    # switched off before it is over: the same answer, from the engine
    _, ws, s, _ = session(client, live_client, denoise="light", captions="local", language="en")
    s.ws.send_json({"type": "room_test", "seconds": 5})
    s.send(voice_in(20, 1))
    assert s.config(denoise=None)["ok"]
    s.send(voice_in(20, 1))
    end(ws, s)
    assert room_test(s)["code"] == "needs_noise_removal"
