"""PyInstaller runtime hook: give faster-whisper a stand-in for PyAV.

faster_whisper.audio does `import av` at import time but only uses it to decode audio files, which Audio Cleaner
never asks it to do (FFmpeg decodes, faster-whisper gets a numpy array). The PyAV wheels ship GPL builds of x264
and x265, which must not be in this LGPL-only bundle, so the spec excludes `av` and this hook installs an empty
module under that name.
"""
import sys
import types


def _unavailable(name):
    # AttributeError (not ImportError) so that hasattr()/inspect probing of the module keeps working.
    raise AttributeError(f"av.{name} is not available in Audio Cleaner: audio is decoded by FFmpeg, not PyAV")


_stub = types.ModuleType("av")
_stub.__getattr__ = _unavailable  # type: ignore[attr-defined]
sys.modules.setdefault("av", _stub)
