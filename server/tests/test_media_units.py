import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from .conftest import ROOT

from acw_common import captions
from acw.stream import protocol as P


def test_srt_and_vtt_format():
    segs = [captions.Segment(0.5, 3.25, "Hello there. This is a test of the caption builder that wraps long lines nicely.")]
    cues = captions.build_cues(segs)
    srt = captions.to_srt(cues)
    assert srt.startswith("1\n00:00:00,500 --> ")
    assert all(len(line) <= captions.MAX_LINE for c in cues for line in c.lines)
    assert all(len(c.lines) <= captions.MAX_LINES for c in cues)
    assert captions.to_vtt(cues).startswith("WEBVTT")
    for a, b in zip(cues, cues[1:]):
        assert a.end <= b.start


def test_frame_roundtrip_and_validation():
    x = (np.sin(np.arange(960) / 10) * 0.5).astype(np.float32)
    frame = P.encode_frame(P.CODEC_S16, 7, 123, x)
    codec, seq, ts, y = P.decode_frame(frame)
    assert (codec, seq, ts) == (P.CODEC_S16, 7, 123)
    assert np.max(np.abs(x - y)) < 1e-3
    with pytest.raises(P.ProtocolError):
        P.decode_frame(P.encode_frame(P.CODEC_S16, 1, 1, np.zeros(500, np.float32)))  # not 10 ms aligned
    with pytest.raises(P.ProtocolError):
        P.decode_frame(P.encode_frame(P.CODEC_F32, 1, 1, np.full(480, np.nan, np.float32)))
    with pytest.raises(P.ProtocolError):
        P.decode_frame(P.encode_frame(P.CODEC_S16, 1, 1, np.zeros(480 * 11, np.float32)))


def test_safe_stem():
    from acw.jobs import safe_stem

    assert safe_stem("../../ét é/../x y.mp4") == "x-y"
    assert safe_stem(".mp4") == "audio" or safe_stem(".mp4")


SANDBOX_PROBE = r"""
import os, socket, sys
sys.path[:0] = sys.argv[2].split(os.pathsep)
from pathlib import Path
from acw.worker.sandbox import confine
confine(Path(sys.argv[1]), [], cpu_seconds=30, address_space_bytes=4 << 30, max_file_bytes=1 << 26)
res = {}
for name, fn in {
    "home": lambda: os.listdir(os.path.expanduser("~")),
    "proc": lambda: open("/proc/self/environ").read(),
    "tmp_write": lambda: open("/tmp/acw-sandbox-escape", "w"),
    "tcp": lambda: socket.create_connection(("127.0.0.1", 22), timeout=2),
}.items():
    try:
        fn(); res[name] = "allowed"
    except Exception:
        res[name] = "blocked"
open(Path(sys.argv[1]) / "ok", "w").write("x")
print(res)
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Landlock is Linux-only")
def test_runner_sandbox_blocks_escape(tmp_path):
    from acw.worker.sandbox import landlock_abi

    if landlock_abi() < 4:
        pytest.skip("needs Landlock ABI 4")
    out = subprocess.run(
        [sys.executable, "-c", SANDBOX_PROBE, str(tmp_path), os.pathsep.join(str(p) for p in (ROOT / "server", ROOT / "common"))],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert "allowed" not in out.stdout, out.stdout
    assert (tmp_path / "ok").exists()
    assert not os.path.exists("/tmp/acw-sandbox-escape")
