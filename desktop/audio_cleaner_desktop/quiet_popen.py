"""No console window for programs started by libraries (Windows only).

The app is a windowed program, so every console program it starts gets a new console window of its own unless the
caller passes CREATE_NO_WINDOW. Our own FFmpeg calls do (acw_common/ffmpeg.py); libraries do not: Python's
`platform` module, for one, runs `cmd /c ver` when its WMI query fails, which flashed a window when a clean started.
`install()` makes CREATE_NO_WINDOW the default for every subprocess.Popen (subprocess.run and friends, os.popen and
platform all go through it). It must run before anything else is imported; packaging/launcher.py calls it first.

Left alone: callers that pass their own console flag (CREATE_NEW_CONSOLE, DETACHED_PROCESS or CREATE_NO_WINDOW),
callers that pass creationflags positionally, and Explorer (a window program: "explorer /select," must open one).
"""
from __future__ import annotations

import functools
import os
import subprocess
import sys

CREATE_NEW_CONSOLE = 0x00000010
DETACHED_PROCESS = 0x00000008
CREATE_NO_WINDOW = 0x08000000
_CONSOLE_FLAGS = CREATE_NEW_CONSOLE | DETACHED_PROCESS | CREATE_NO_WINDOW
_CREATIONFLAGS_POSITION = 13  # Popen(args, bufsize, executable, stdin, stdout, stderr, preexec_fn, close_fds, shell,
#                               cwd, env, universal_newlines, startupinfo, creationflags, ...)
_MARK = "_acw_quiet_popen"


def _is_explorer(args: object) -> bool:
    first = args[0] if isinstance(args, (list, tuple)) and args else args
    if not isinstance(first, (str, bytes, os.PathLike)):
        return False
    name = os.fsdecode(first).replace("\\", "/").rsplit("/", 1)[-1].lower()
    return name in ("explorer", "explorer.exe")


def install(force: bool = False) -> bool:
    """Patch subprocess.Popen on Windows (or anywhere with force=True, for tests). Safe to call twice.
    Returns True when the patch was applied by this call."""
    if not force and sys.platform != "win32":
        return False
    original = subprocess.Popen.__init__
    if getattr(original, _MARK, False):
        return False

    @functools.wraps(original)
    def __init__(self, args, *rest, **kwargs):  # noqa: N807
        if len(rest) < _CREATIONFLAGS_POSITION and not _is_explorer(args):
            flags = kwargs.get("creationflags", 0)
            if not flags & _CONSOLE_FLAGS:
                kwargs["creationflags"] = flags | CREATE_NO_WINDOW
        original(self, args, *rest, **kwargs)

    setattr(__init__, _MARK, True)
    subprocess.Popen.__init__ = __init__
    return True


def warm_platform_info() -> None:
    """Ask `platform` once, on a background thread, so the slow WMI query is done before it is needed."""
    import threading

    def work() -> None:
        try:
            import platform

            platform.uname()
        except Exception:  # noqa: BLE001 - only a warm-up
            pass

    threading.Thread(target=work, name="platform-warmup", daemon=True).start()
