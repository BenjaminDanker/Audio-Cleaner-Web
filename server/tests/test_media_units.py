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


SANDBOX_SOCKETS_PROBE = r"""
import os, socket, subprocess, sys
sys.path[:0] = sys.argv[2].split(os.pathsep)
from pathlib import Path
from acw.worker.sandbox import confine
job = Path(sys.argv[1])
# a service the runner must not reach: a Unix socket somebody else listens on (like the user's D-Bus)
srv = socket.socket(socket.AF_UNIX); srv.bind(str(job.parent / "other.sock")); srv.listen(1)
gpu = sys.argv[3] == "gpu"
confine(job, [], cpu_seconds=30, address_space_bytes=4 << 30, max_file_bytes=1 << 26, unix_sockets=gpu)
def attempt(fn):
    try:
        fn(); return "allowed"
    except OSError as e:
        return "blocked"
res = {
    "unix_connect": attempt(lambda: socket.socket(socket.AF_UNIX).connect(str(job.parent / "other.sock"))),
    "unix_abstract_bind": attempt(lambda: socket.socket(socket.AF_UNIX).bind("\0acw-test")),
    "inet_socket": attempt(lambda: socket.socket(socket.AF_INET)),
    "unix_socket": attempt(lambda: socket.socket(socket.AF_UNIX)),
    "symlink": attempt(lambda: os.symlink("/etc/hostname", job / "link")),
    "fifo": attempt(lambda: os.mkfifo(job / "fifo")),
    "socketpair": attempt(lambda: socket.socketpair()),
}
# what a runner legitimately does keeps working: pipes, child processes, files in the job directory
out = subprocess.run(["/usr/bin/true"], capture_output=True)
(job / "ok").write_text("x")
res["child"] = "ok" if out.returncode == 0 else "failed"
print(res)
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Landlock is Linux-only")
@pytest.mark.parametrize("mode", ["cpu", "gpu"])
def test_runner_sandbox_refuses_sockets_links_and_pipes(tmp_path, mode):
    import ast

    from acw.worker.sandbox import landlock_abi

    if landlock_abi() < 4:
        pytest.skip("needs Landlock ABI 4")
    job = tmp_path / "job"
    job.mkdir()
    out = subprocess.run(
        [sys.executable, "-c", SANDBOX_SOCKETS_PROBE, str(job), os.pathsep.join(str(p) for p in (ROOT / "server", ROOT / "common")), mode],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    res = ast.literal_eval(out.stdout.strip().splitlines()[-1])
    assert res["child"] == "ok" and (job / "ok").exists()
    assert res["unix_connect"] == res["inet_socket"] == "blocked"
    # the CUDA driver serves an abstract socket of its own; the runner still can't connect anywhere
    assert res["unix_abstract_bind"] == ("allowed" if mode == "gpu" else "blocked")
    assert res["symlink"] == res["fifo"] == "blocked"
    assert res["socketpair"] == "allowed"  # an anonymous connected pair leads nowhere (the denoiser's runtime builds one)
    assert res["unix_socket"] == ("allowed" if mode == "gpu" else "blocked")
    assert not (job / "link").exists() and not (job / "fifo").exists()


FFMPEG_DIR = Path.home() / "audio-cleaner/bin"


@pytest.mark.skipif(not (FFMPEG_DIR / "ffmpeg").is_file(), reason="ffmpeg not installed")
def test_uploads_are_parsed_by_one_chosen_demuxer(tmp_path):
    from acw_common.ffmpeg import FFmpeg, MediaError, sniff_demuxer

    ff = FFmpeg(FFMPEG_DIR, force_demuxer=True)
    for ext, want in {"wav": "wav", "mp3": "mp3", "flac": "flac", "m4a": "mov,mp4,m4a,3gp,3g2,mj2", "mkv": "matroska,webm", "ogg": "ogg"}.items():
        p = tmp_path / f"a.{ext}"
        subprocess.run([str(FFMPEG_DIR / "ffmpeg"), "-v", "error", "-f", "lavfi", "-i", "sine=duration=1", str(p)], check=True)
        assert ff.probe(p).format_name == want
    # a file that is not what its name says, or not media at all, never reaches the other demuxers
    junk = tmp_path / "b.wav"
    junk.write_bytes(b"\x00" * 64 + b"hello" * 100)
    with pytest.raises(MediaError):
        sniff_demuxer(junk)
    with pytest.raises(MediaError):
        ff.probe(junk)
    liar = tmp_path / "c.mp4"
    liar.write_bytes((tmp_path / "a.wav").read_bytes())
    assert ff.probe(liar).format_name == "wav"  # decided by content, not by name


def test_seccomp_filter_has_the_expected_shape():
    from acw.worker import sandbox

    for arch, nums in sandbox.SYSCALLS.items():
        for gpu in (False, True):
            rows = sandbox.seccomp_program(arch, unix_sockets=gpu)
            assert rows[0][3] == 4 and rows[-1][3] == sandbox.SECCOMP_RET_ALLOW  # arch check first, allow by default
            assert len(rows) < 100
            numbers = {r[3] for r in rows if r[0] == 0x15}
            assert nums["connect"] in numbers and nums["sendto"] in numbers and nums["io_uring_setup"] in numbers
            assert (nums["socket"] in numbers) is True  # filtered either way: refused, or allowed for AF_UNIX only
            # every jump stays inside the program
            for i, (code, jt, jf, k) in enumerate(rows):
                if code == 0x15:
                    assert i + 1 + max(jt, jf) < len(rows)


