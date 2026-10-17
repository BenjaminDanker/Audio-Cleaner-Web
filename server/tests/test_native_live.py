"""The native live engine (native/live, bin/acw-live) through its socket: the audio it returns (against a reference
built here from the same model library), echo removal, silence, returnAudio, admission, captions. Skipped where it
isn't built (cargo build in native/live) or the models aren't installed."""
import json
import os
import socket
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from acw.stream import protocol as P

from .conftest import LIVE_LIBDF, LIVE_MODELS


AEC_DELAY = 432  # the echo canceller's delay (native/live/src/main.rs)
LATENCY = 1440 + AEC_DELAY


@pytest.fixture
def engine(live_engine):
    return live_engine


def _msg(s, kind, payload):
    s.sendall(kind + struct.pack("<I", len(payload)) + payload)


def _read(s):
    head = b""
    while len(head) < 5:
        chunk = s.recv(5 - len(head))
        if not chunk:
            raise ConnectionError
        head += chunk
    kind, n = head[:1], struct.unpack("<I", head[1:])[0]
    data = b""
    while len(data) < n:
        data += s.recv(n - len(data))
    return kind, data


def _session(sock, cfg, frames, ref=None):
    s = socket.socket(socket.AF_UNIX)
    s.connect(sock)
    s.settimeout(10)
    _msg(s, P.MSG_START, json.dumps(cfg).encode())
    k, p = _read(s)
    ready = json.loads(p)
    if ready.get("type") != "ready":
        s.close()
        return ready, None
    out = []
    for i, f in enumerate(frames):
        if ref is not None:
            _msg(s, P.MSG_REF, P.pack_audio(i, ref[i]))
        if f is None:
            _msg(s, P.MSG_SILENCE, P.pack_silence(i, 960))
            continue
        _msg(s, P.MSG_AUDIO, P.pack_audio(i, f))
        while True:
            k, p = _read(s)
            if k == P.MSG_AUDIO:
                out.append(P.unpack_audio(p)[2])
                break
    _msg(s, P.MSG_END, b"")
    while _read(s)[0] != P.MSG_END:
        pass
    s.close()
    return ready, out


def _speech(seconds=3.0):
    rng = np.random.default_rng(0)
    t = np.arange(int(48_000 * seconds)) / 48_000
    x = 0.2 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 1.5 * t) > 0) + 0.02 * rng.standard_normal(len(t))
    return x.astype(np.float32)


def _reference_chain(x: np.ndarray, atten_db: float) -> np.ndarray:
    """What the engine must return: DeepFilterNet3 (libdf), an 80 Hz high-pass (scipy) and the peak limiter."""
    from scipy.signal import butter, sosfilt

    from acw_common.dfn import DeepFilter

    df = DeepFilter(LIVE_LIBDF, LIVE_MODELS / "DeepFilterNet3_onnx.tar.gz", atten_db)
    y = sosfilt(butter(2, 80, btype="highpass", fs=48_000, output="sos"), df.process(x)).astype(np.float32)
    ceiling, release, gain = np.float32(10 ** (-1 / 20)), np.float32(10 ** (6 / 20 / 100)), np.float32(1.0)
    out = np.empty_like(y)
    for i in range(0, len(y), 480):
        hop = y[i:i + 480]
        peak = float(np.max(np.abs(hop)))
        target = np.float32(min(1.0, ceiling / peak) if peak > 0 else 1.0)
        new = min(target, gain * release, np.float32(1.0))
        ramp = (gain + (new - gain) * np.arange(480, dtype=np.float32) / np.float32(479)) if new < gain else np.full(480, new, np.float32)
        out[i:i + 480] = np.clip(hop * ramp, -ceiling, ceiling)
        gain = new
    return out


def test_the_audio_matches_the_reference_chain(engine):
    x = _speech()
    frames = [x[i:i + 960] for i in range(0, len(x), 960)]
    ready, out = _session(engine(), {"denoise": "light", "captions": None, "atten_lim_db": 24}, frames)
    # Every session has the same delay, whatever runs (features can change mid-session): the model's 1440 samples and
    # the echo canceller's 432, which a session without it spends in a delay line before the model.
    assert ready["latencySamples"] == LATENCY and ready["cores"] == [0]
    y = np.concatenate(out)
    want = _reference_chain(np.concatenate([np.zeros(AEC_DELAY, np.float32), x])[:len(x)], 24)
    assert np.max(np.abs(y - want)) < 1e-6, np.max(np.abs(y - want))


def test_digital_silence_is_answered_with_silence(engine):
    x = _speech(1.0)
    frames = [x[i:i + 960] for i in range(0, len(x), 960)] + [np.zeros(960, np.float32)] * 50
    _, out = _session(engine(), {"denoise": "light"}, frames)
    assert not np.any(np.concatenate(out[-35:]))  # after 0.2 s of it the model isn't run: exact silence


@pytest.mark.skipif(not (LIVE_MODELS / "captions-en/encoder.int8.onnx").is_file(), reason="caption model missing")
def test_standard_captions_keep_the_first_words(engine):
    sample = Path.home() / "audio-cleaner/dev/live_opus.wav"
    if not sample.is_file():
        pytest.skip("speech sample missing")
    raw = subprocess.run([str(Path.home() / "audio-cleaner/bin/ffmpeg"), "-v", "error", "-i", str(sample), "-ac", "1",
                          "-ar", "48000", "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32)
    x = x[:len(x) // 960 * 960]
    s = socket.socket(socket.AF_UNIX)
    s.connect(engine())
    s.settimeout(30)
    _msg(s, P.MSG_START, json.dumps({"denoise": None, "captions": "local"}).encode())
    assert json.loads(_read(s)[1])["type"] == "ready"
    events = []
    for i in range(0, len(x), 960):
        _msg(s, P.MSG_AUDIO, P.pack_audio(i // 960, x[i:i + 960]))
        while True:
            k, p = _read(s)
            if k == P.MSG_AUDIO:
                break
            events.append(json.loads(p))
        if i % (960 * 25) == 0:
            time.sleep(0.1)
    _msg(s, P.MSG_END, b"")
    while True:
        k, p = _read(s)
        if k == P.MSG_END:
            break
        events.append(json.loads(p))
    finals = [e["text"] for e in events if e.get("type") == "caption" and e.get("final")]
    assert finals and finals[0].startswith("We will not be held responsible"), finals
    assert any(e.get("type") == "caption" and not e.get("final") for e in events)  # partial lines while talking


def test_echo_is_removed_and_the_voice_kept(engine):
    rng = np.random.default_rng(1)
    n = 48_000 * 8
    far = (np.convolve(rng.standard_normal(n), np.ones(8) / 8, mode="same") * 0.2).astype(np.float32)
    mic = np.concatenate([np.zeros(1920, np.float32), far[:-1920]]) * 0.4
    mic[-96_000:] += (0.2 * np.sin(2 * np.pi * 330 * np.arange(96_000) / 48_000)).astype(np.float32)
    frames = [mic[i:i + 960] for i in range(0, n, 960)]
    refs = [far[i:i + 960] for i in range(0, n, 960)]
    ready, out = _session(engine(), {"denoise": None, "captions": None, "echo_reference": True}, frames, refs)
    y = np.concatenate(out)
    assert ready["latencySamples"] == LATENCY
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))  # noqa: E731
    echo = slice(5 * 48_000, 6 * 48_000)
    assert 20 * np.log10(rms(mic[echo]) / rms(y[echo])) > 20
    assert 0.8 < rms(y[-48_000:]) / rms(mic[-48_000:]) < 1.2


def test_silence_return_audio_and_full(engine):
    x = _speech(1.0)
    frames = [x[i:i + 960] for i in range(0, len(x), 960)]
    sock = engine(ACW_LIVE_UNITS_PER_CORE="1.0", ACW_MIN_JOB_CORES="999")  # one live core
    _, out = _session(sock, {"denoise": "light", "return_audio": False}, frames[:10] + [None, None] + frames[10:20])
    assert len(out) == 20 and all(len(o) == 0 for o in out)  # answered (for timing), but empty; silence not answered
    # one core x 1 unit: a second session at once doesn't fit
    s = socket.socket(socket.AF_UNIX)
    s.connect(sock)
    _msg(s, P.MSG_START, json.dumps({"denoise": "light"}).encode())
    assert json.loads(_read(s)[1])["type"] == "ready"
    refused, _ = _session(sock, {"denoise": "light", "captions": "local"}, [])
    assert refused["code"] == "full" and refused["available"]
    s.close()
