"""Subtitles for the preview: read SRT / WebVTT text (a file next to the result, or the track inside a video) into
timed cues, and find the one to show at a moment."""
from __future__ import annotations

import bisect
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_TIME = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")
_TAG = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


def _seconds(m: re.Match) -> float:
    h, mi, s, ms = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse(text: str) -> list[Cue]:
    """SRT or WebVTT text -> cues in time order. Formatting tags are dropped; lines of one cue stay lines."""
    cues = []
    for block in re.split(r"\r?\n\s*\r?\n", text.replace("﻿", "")):
        lines = [ln.strip() for ln in block.strip().splitlines()]
        for i, line in enumerate(lines):
            if "-->" not in line:
                continue
            left, _, right = line.partition("-->")
            a, b = _TIME.search(left), _TIME.search(right)
            if a and b:
                body = "\n".join(_TAG.sub("", ln) for ln in lines[i + 1:] if ln).strip()
                if body:
                    cues.append(Cue(_seconds(a), _seconds(b), body))
            break
    return sorted(cues, key=lambda c: c.start)


def read_file(path: Path) -> list[Cue]:
    try:
        return parse(path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return []


def read_embedded(ffmpeg: Path, video: Path) -> list[Cue]:
    """The first subtitle track inside `video` (none: an empty list)."""
    try:
        out = subprocess.run(
            [str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe", "-i", str(video),
             "-map", "0:s:0", "-c:s", "srt", "-f", "srt", "pipe:1"],
            capture_output=True, timeout=60, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return parse(out.stdout.decode("utf-8", "replace")) if out.returncode == 0 else []


class Track:
    """Cues with a quick lookup by time."""

    def __init__(self, cues: list[Cue]):
        self.cues = cues
        self._starts = [c.start for c in cues]

    def __bool__(self) -> bool:
        return bool(self.cues)

    def at(self, seconds: float) -> str:
        i = bisect.bisect_right(self._starts, seconds) - 1
        if i >= 0 and self.cues[i].start <= seconds < self.cues[i].end:
            return self.cues[i].text
        return ""
