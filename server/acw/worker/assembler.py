"""Puts a stretch together from a job's cached blocks: `python -m acw.worker.assembler <bin dir> <job dir> <json>`.

The preview of the per-part editor plays short stretches of what a step made: the echo-free voice, the noise removal at full
strength, the rebuilt voice (acw/editing.py). Those are cached as one 24-bit FLAC file per block (`blocks/<sha1>.flac`, made by the
sandboxed runner), a block is up to 30 s and the blocks do not line up with the stretch a page asks for, so the stretch is cut from
the blocks it overlaps: each is decoded, put at its place, and the result is written to the standard output as a 24-bit FLAC with one
channel per output channel (a silent one where the job has nothing to make), the samples [first, first + count) of the recording.

    json   {"first": int, "count": int, "channels": [[{"name": <sha1>, "at": int, "len": int, "from": int, "to": int}, ...] | null, ...]}
           each block is `blocks/<name>.flac` (and `<name>.scale`, when it was written scaled down), whose samples start at `at`;
           it is used for the samples [from, to) of the recording (two removers can hold different parts of one stretch)

Like the cutter (cutter.py) this confines itself first (the job's block folder is readable, nothing is writable: a pipe is all it
writes to, no network, hard limits), because the files were made by the runner, which handles untrusted uploads; this program is the
only one that decodes them for the web process. The caller (stretch.py) kills it after a few seconds. The scale a loud stretch had to
be brought down by is the last line on the standard error (`scale=<number>`; the samples are the stretch times it).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # (numpy is imported only after the sandbox is up)
    import numpy as np

SAMPLE_RATE = 48_000
CPU_SECONDS = 30
ADDRESS_SPACE_BYTES = 4 * 1024**3
MAX_FILE_BYTES = 1024**2
NAME = re.compile(r"^[0-9a-f]{40}$")
MAX_COUNT = 30 * SAMPLE_RATE
MAX_BLOCKS = 16
CEILING = 0.98
# acw_common.ffmpeg.AUDIO_FORMATS["flac24"]
FLAC24 = ["-c:a", "flac", "-compression_level", "3", "-sample_fmt", "s32", "-bits_per_raw_sample", "24"]


def decode(ffmpeg: str, path: Path):
    import numpy as np

    r = subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-v", "error", "-protocol_whitelist", "file", "-f", "flac", "-i", str(path), "-f", "f32le",
                        "-ac", "1", "-ar", str(SAMPLE_RATE), "-threads", "1", "pipe:1"], capture_output=True, stdin=subprocess.DEVNULL, timeout=20)
    if r.returncode != 0:
        raise ValueError("a block could not be decoded")
    y = np.frombuffer(r.stdout, dtype=np.float32)
    try:
        scale = float(path.with_suffix(".scale").read_text())
    except (OSError, ValueError):
        scale = 1.0
    return y / np.float32(scale if 0.0 < scale <= 1.0 else 1.0)


def assemble(ffmpeg: str, blocks_dir: Path, first: int, count: int, channels: list) -> "np.ndarray":
    import numpy as np

    out = np.zeros((count, len(channels)), np.float32)
    used = 0
    for c, blocks in enumerate(channels):
        for b in blocks or []:
            name, at, length = str(b["name"]), int(b["at"]), int(b["len"])
            if not NAME.match(name) or length <= 0 or at < 0:
                raise ValueError("bad block")
            used += 1
            if used > MAX_BLOCKS * len(channels):
                raise ValueError("too many blocks")
            a, e = max(first, at, int(b.get("from", 0))), min(first + count, at + length, int(b.get("to", first + count)))
            if e <= a:
                continue
            path = blocks_dir / f"{name}.flac"
            if not path.is_file() or path.is_symlink():
                continue
            y = decode(ffmpeg, path)
            e = min(e, at + len(y))
            if e > a:
                out[a - first:e - first, c] = y[a - at:e - at]
    return out


def main(argv: list[str]) -> int:
    try:
        bin_dir, job_dir, spec = Path(argv[1]), Path(argv[2]), json.loads(argv[3])
        first, count, channels = int(spec["first"]), int(spec["count"]), spec["channels"]
        if not (job_dir.is_absolute() and first >= 0 and 0 < count <= MAX_COUNT and 1 <= len(channels) <= 2):
            raise ValueError("bad arguments")
    except (IndexError, ValueError, KeyError, TypeError):
        sys.stderr.write("usage: assembler <bin dir> <absolute job dir> <json>\n")
        return 2
    from acw.worker import sandbox

    sandbox.apply_landlock(sandbox.default_read_paths([bin_dir, job_dir / "blocks"]), ["/dev/null"])
    sandbox.apply_rlimits(CPU_SECONDS, ADDRESS_SPACE_BYTES, MAX_FILE_BYTES)
    sandbox.apply_seccomp()
    import numpy as np

    try:
        y = assemble(str(bin_dir / "ffmpeg"), job_dir / "blocks", first, count, channels)
    except (ValueError, OSError, subprocess.SubprocessError) as e:
        sys.stderr.write(f"{e}\n")
        return 1
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    scale = 1.0 if peak <= CEILING else CEILING / peak
    raw = np.ascontiguousarray(y * np.float32(scale), dtype=np.float32).tobytes()
    r = subprocess.run([str(bin_dir / "ffmpeg"), "-nostdin", "-hide_banner", "-v", "error", "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", str(y.shape[1]), "-i", "pipe:0",
                        *FLAC24, "-threads", "1", "-f", "flac", "pipe:1"], input=raw, capture_output=True, timeout=20)
    if r.returncode != 0 or not r.stdout.startswith(b"fLaC"):
        sys.stderr.write("the encoder failed\n")
        return 1
    sys.stdout.buffer.write(r.stdout)
    sys.stdout.buffer.flush()
    sys.stderr.write(f"scale={scale!r}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
