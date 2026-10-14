"""Make print() usable in the command-line modes of a windowed Windows build."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

LOG_NAME = "audio-cleaner-cli.log"


def _attach_parent_console() -> bool:
    import ctypes
    from ctypes import wintypes

    attach = ctypes.windll.kernel32.AttachConsole
    attach.argtypes = [wintypes.DWORD]
    attach.restype = wintypes.BOOL
    return bool(attach(0xFFFFFFFF))  # ATTACH_PARENT_PROCESS


def ensure_output() -> None:
    """A PyInstaller windowed exe starts without stdout/stderr. Attach to the parent
    console when there is one; otherwise write to audio-cleaner-cli.log in the temp folder."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    if os.name == "nt" and _attach_parent_console():
        stream = open("CONOUT$", "w", encoding="utf-8", errors="replace", buffering=1)
    else:
        stream = open(Path(tempfile.gettempdir()) / LOG_NAME, "a", encoding="utf-8", errors="replace", buffering=1)
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream
