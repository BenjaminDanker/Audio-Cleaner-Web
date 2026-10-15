"""Which files the app offers to clean, and how folders are expanded."""
from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".mts", ".m2ts", ".ts", ".wmv", ".flv", ".3gp")
AUDIO_EXTENSIONS = (
    ".wav", ".mp3", ".m4a", ".m4b", ".aac", ".flac", ".ogg", ".oga", ".opus", ".aiff", ".aif", ".aifc", ".caf", ".w64",
    ".wma",
)
MEDIA_EXTENSIONS = frozenset(VIDEO_EXTENSIONS + AUDIO_EXTENSIONS)


def dialog_filter() -> str:
    patterns = " ".join(f"*{e}" for e in (*VIDEO_EXTENSIONS, *AUDIO_EXTENSIONS))
    return f"Video and audio ({patterns});;All files (*)"


def is_media(path: Path) -> bool:
    return path.suffix.lower() in MEDIA_EXTENSIONS


def expand(paths: Iterable[str | os.PathLike]) -> list[Path]:
    """Files stay as they are (the engine decides whether it can read them); folders
    are searched recursively for media files. Hidden and temporary files are skipped."""
    found: list[Path] = []
    seen: set[Path] = set()

    def add(p: Path) -> None:
        key = p.absolute()
        if key not in seen:
            seen.add(key)
            found.append(key)

    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for root, dirs, files in os.walk(p):
                dirs[:] = sorted(d for d in dirs if not d.startswith("."))
                for name in sorted(files):
                    if not name.startswith(".") and is_media(Path(name)):
                        add(Path(root) / name)
        elif p.is_file():
            add(p)
    return found
