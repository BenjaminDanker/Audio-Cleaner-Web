"""Caption encoder utilities for SRT/VTT (and placeholder for 608/708 packaging)."""
from __future__ import annotations

import os
from typing import List


def _fmt_time_srt(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_time_vtt(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


class Segment:
    def __init__(self, start: float, end: float, text: str):
        self.start = start
        self.end = end
        self.text = text


def write_srt(segments: List[Segment], path: str) -> str:
    lines = []
    for i, s in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{_fmt_time_srt(s.start)} --> {_fmt_time_srt(s.end)}")
        lines.append(s.text)
        lines.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def write_vtt(segments: List[Segment], path: str) -> str:
    lines = ["WEBVTT", ""]
    for s in segments:
        lines.append(f"{_fmt_time_vtt(s.start)} --> {_fmt_time_vtt(s.end)}")
        lines.append(s.text)
        lines.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# Placeholder for 608/708 packaging — to be implemented in streaming service

def build_cea608_packets(segments: List[Segment], service: int = 1) -> bytes:  # pragma: no cover - stub
    return b""
