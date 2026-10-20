"""The loudness outline of cached blocks and of the recording copy: `python -m acw.worker.peaker <bin dir> <job dir> <json>`.

`GET /api/jobs/{id}/peaks/{layer}` (acw/editing.py) draws what a layer holds as one number per quarter second: the largest absolute
sample. The blocks (`blocks/<sha1>.flac`) and the recording's copy (`kept-source.flac`) were made by the sandboxed runner, which handles
untrusted uploads, so the web process never decodes them; this program does, confined like the assembler (assembler.py): the job's
folder is readable, nothing is writable (a pipe is all it writes to), no network, hard limits. It answers with JSON on the standard
output, and the caller keeps the numbers next to the blocks (`blocks/<sha1>.peaks`) so a block is decoded for them only once.

    {"blocks": [{"name": <sha1>, "at": <first sample of the block>}, ...]}     ->   {"<sha1>": {"first": w, "p": [...]}, ...}
    {"file": "kept-source.flac", "channels": n, "scale": s}                    ->   {"first": 0, "p": [...]}

`p[i]` is the largest absolute sample of window `first + i` (a window is 12,000 samples = 0.25 s of the recording, counted from its
start) among the block's own samples (the first and last window of a block that does not start or end on the grid hold only those),
for the file over all its channels, divided by the scale a loud copy was brought down by.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

SAMPLE_RATE = 48_000
WINDOW = SAMPLE_RATE // 4
CPU_SECONDS = 120
ADDRESS_SPACE_BYTES = 4 * 1024**3
MAX_FILE_BYTES = 1024**2
NAME = re.compile(r"^[0-9a-f]{40}$")
SOURCE = re.compile(r"^kept-source\.flac$")
MAX_BLOCKS = 40


def window_peaks(y, at: int) -> tuple[int, list[float]]:
    """(index of the first window, the peaks of the windows) of the samples `y` (mono) that start at sample `at` of the recording."""
    import numpy as np

    first = at // WINDOW
    lead = at - first * WINDOW
    z = np.abs(np.concatenate([np.zeros(lead, np.float32), np.asarray(y, np.float32)]))
    pad = (-len(z)) % WINDOW
    if pad:
        z = np.concatenate([z, np.zeros(pad, np.float32)])
    return first, [round(float(v), 5) for v in z.reshape(-1, WINDOW).max(axis=1)] if len(z) else []


def blocks_peaks(ffmpeg: str, blocks_dir: Path, blocks: list) -> dict:
    from acw.worker.assembler import decode

    out = {}
    for b in blocks:
        name, at = str(b["name"]), int(b["at"])
        if not NAME.match(name) or at < 0:
            raise ValueError("bad block")
        path = blocks_dir / f"{name}.flac"
        if not path.is_file() or path.is_symlink():
            continue
        first, p = window_peaks(decode(ffmpeg, path), at)
        out[name] = {"first": first, "p": p}
    return out


def source_peaks(ffmpeg: str, path: Path, channels: int, scale: float) -> dict:
    """The outline of the recording's copy, decoded in pieces (a recording of hours is never held whole)."""
    import numpy as np

    args = [ffmpeg, "-nostdin", "-hide_banner", "-v", "error", "-protocol_whitelist", "file", "-f", "flac", "-i", str(path), "-f", "f32le",
            "-ac", str(channels), "-ar", str(SAMPLE_RATE), "-threads", "1", "pipe:1"]
    win_bytes = WINDOW * channels * 4
    out: list[float] = []
    buf = b""
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    try:
        while True:
            chunk = proc.stdout.read(win_bytes * 40)
            if not chunk:
                break
            buf += chunk
            k = len(buf) // win_bytes
            if k:
                a = np.frombuffer(buf[:k * win_bytes], np.float32).reshape(k, WINDOW * channels)
                out += [round(float(v), 5) for v in np.abs(a).max(axis=1) / scale]
                buf = buf[k * win_bytes:]
        if buf:
            a = np.frombuffer(buf[:len(buf) - len(buf) % 4], np.float32)
            if len(a):
                out.append(round(float(np.abs(a).max()) / scale, 5))
    finally:
        proc.stdout.close()
        proc.wait()
    if proc.returncode != 0:
        raise ValueError("the recording could not be decoded")
    return {"first": 0, "p": out}


def main(argv: list[str]) -> int:
    try:
        bin_dir, job_dir, spec = Path(argv[1]), Path(argv[2]), json.loads(argv[3])
        if not job_dir.is_absolute():
            raise ValueError("bad arguments")
        if "blocks" in spec:
            if not isinstance(spec["blocks"], list) or len(spec["blocks"]) > MAX_BLOCKS:
                raise ValueError("too many blocks")
        elif not (SOURCE.match(str(spec["file"])) and 1 <= int(spec["channels"]) <= 2 and 0.0 < float(spec.get("scale", 1.0)) <= 1.0):
            raise ValueError("bad source")
    except (IndexError, ValueError, KeyError, TypeError):
        sys.stderr.write("usage: peaker <bin dir> <absolute job dir> <json>\n")
        return 2
    from acw.worker import sandbox

    sandbox.apply_landlock(sandbox.default_read_paths([bin_dir, job_dir / ("blocks" if "blocks" in spec else "")]), ["/dev/null"])
    sandbox.apply_rlimits(CPU_SECONDS, ADDRESS_SPACE_BYTES, MAX_FILE_BYTES)
    sandbox.apply_seccomp()
    try:
        if "blocks" in spec:
            result = blocks_peaks(str(bin_dir / "ffmpeg"), job_dir / "blocks", spec["blocks"])
        else:
            result = source_peaks(str(bin_dir / "ffmpeg"), job_dir / str(spec["file"]), int(spec["channels"]), float(spec.get("scale", 1.0)))
    except (ValueError, OSError, subprocess.SubprocessError) as e:
        sys.stderr.write(f"{e}\n")
        return 1
    sys.stdout.write(json.dumps(result, separators=(",", ":")))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
