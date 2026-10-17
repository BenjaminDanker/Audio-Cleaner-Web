"""The native live engine (native/live, bin/acw-live) against the Python worker it replaces: the same protocol and the
same audio. Skipped where it isn't built (cargo build in native/live) or the models aren't installed."""
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

REPO = Path(__file__).resolve().parents[2]
HOME = Path.home() / "audio-cleaner"
BINARY = next((p for p in (REPO / "native/live/target/release/acw-live", HOME / "bin/acw-live") if p.is_file()), None)
MODELS = HOME / "models"
LIBDF = HOME / "bin/libdf.so"

pytestmark = pytest.mark.skipif(
    BINARY is None or not (MODELS / "DeepFilterNet3_onnx.tar.gz").is_file() or not LIBDF.is_file(),
    reason="native engine or models not available here",
)


@pytest.fixture
def engine(tmp_path):
    def start(**env):
        sock = tmp_path / "stream.sock"
        e = {**os.environ, "ACW_ENV": "development", "ACW_DATA_DIR": str(tmp_path), "ACW_STREAM_SOCKET": str(sock),
             "ACW_BIN_DIR": str(LIBDF.parent), "ACW_MODELS_DIR": str(MODELS), "ACW_MIN_JOB_CORES": "1", **env}
        proc = subprocess.Popen([str(BINARY)], env=e, stderr=subprocess.PIPE)
        for _ in range(100):
            if sock.exists():
                break
            time.sleep(0.05)
        procs.append(proc)
        return str(sock)

    procs: list[subprocess.Popen] = []
    yield start
    for p in procs:
        p.terminate()
        p.wait(5)


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


def test_the_same_audio_as_the_python_chain(engine):
    from acw.stream.engine import LiveChain

    x = _speech()
    frames = [x[i:i + 960] for i in range(0, len(x), 960)]
    ready, out = _session(engine(), {"denoise": "light", "captions": None, "atten_lim_db": 24}, frames)
    assert ready["latencySamples"] == 1440 and ready["cores"] == [0]
    chain = LiveChain(LIBDF, MODELS, "light", 24)
    want = np.concatenate([chain.process(f) for f in frames])
    assert np.array_equal(np.concatenate(out), want)  # bit for bit


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
    assert ready["latencySamples"] == 432
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))  # noqa: E731
    echo = slice(5 * 48_000, 6 * 48_000)
    assert 20 * np.log10(rms(mic[echo]) / rms(y[echo])) > 20
    assert 0.8 < rms(y[-48_000:]) / rms(mic[-48_000:]) < 1.2


def test_silence_return_audio_and_full(engine):
    x = _speech(1.0)
    frames = [x[i:i + 960] for i in range(0, len(x), 960)]
    sock = engine(ACW_LIVE_UNITS_PER_CORE="1.0", ACW_MIN_JOB_CORES="999", ACW_LIVE_IGNORE_HOST_LOAD="1")  # one live core
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
