"""Cuts a stretch out of a kept copy: `python -m acw.worker.cutter <bin dir> <kept file> <first sample> <samples>`.

The strength slider of the website (and the API) plays a job's two kept copies (the noise removal at full strength and the
recording, 24-bit FLAC, 48 kHz) in short pieces. This is how a piece is made: FFmpeg opens the file, goes to the second before
`first`, drops the samples up to `first` and writes exactly `samples` samples to the standard output as the same kind of FLAC
(24-bit, so nothing is lost: the stretch decodes to the very values the whole file decodes to at those positions). Two stretches
cut with the same numbers from the two copies line up sample for sample.

The program confines itself first, like the job runner (sandbox.py): the file's folder is readable and nothing is writable
(a pipe is all it writes to), no network, no sockets, hard CPU, memory and size limits, and then becomes FFmpeg (exec), so a damaged
or planted file can not do more than make this one answer fail. The caller (stretch.py) kills it after a few seconds.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SAMPLE_RATE = 48_000
# acw_common.ffmpeg.AUDIO_FORMATS["flac24"] (a test keeps the two equal; numpy is not imported here to keep the start quick)
FLAC24 = ["-c:a", "flac", "-compression_level", "3", "-sample_fmt", "s32", "-bits_per_raw_sample", "24"]
CPU_SECONDS = 20
ADDRESS_SPACE_BYTES = 4 * 1024**3
MAX_FILE_BYTES = 1024**2  # nothing is written to a file at all


def ffmpeg_args(ffmpeg: str, path: str, first: int, count: int) -> list[str]:
    """FFmpeg's command line for `count` samples from sample `first` of the FLAC file `path`, FLAC to the standard output."""
    seconds, skip = divmod(first, SAMPLE_RATE)
    return [
        ffmpeg, "-nostdin", "-hide_banner", "-v", "error", "-protocol_whitelist", "file", "-f", "flac",
        *(["-ss", str(seconds)] if seconds else []),  # a whole second: the sample position is exact
        "-i", path,
        "-af", f"atrim=start_sample={skip}:end_sample={skip + count}",
        *FLAC24, "-threads", "1", "-f", "flac", "pipe:1",
    ]


def main(argv: list[str]) -> int:
    try:
        bin_dir, path, first, count = Path(argv[1]), Path(argv[2]), int(argv[3]), int(argv[4])
        if not (path.is_absolute() and first >= 0 and count > 0):
            raise ValueError("bad arguments")
    except (IndexError, ValueError):
        sys.stderr.write("usage: cutter <bin dir> <absolute kept file> <first sample> <samples>\n")
        return 2
    from acw.worker import sandbox

    sandbox.apply_landlock(sandbox.default_read_paths([bin_dir, path.parent]), ["/dev/null"])
    sandbox.apply_rlimits(CPU_SECONDS, ADDRESS_SPACE_BYTES, MAX_FILE_BYTES)
    sandbox.apply_seccomp()
    args = ffmpeg_args(str(bin_dir / "ffmpeg"), str(path), first, count)
    os.execve(args[0], args, {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
    return 1  # not reached


if __name__ == "__main__":
    sys.exit(main(sys.argv))
