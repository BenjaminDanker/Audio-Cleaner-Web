"""Locate the native pieces the app ships: libDF, the DeepFilterNet3 model, FFmpeg.

Frozen (PyInstaller) builds find everything under sys._MEIPASS:

    native/{df.dll|libdf.dylib|libdf.so, ffmpeg[.exe], ffprobe[.exe]}
    models/DeepFilterNet3_onnx.tar.gz
    licenses/

From source the same layout is read from $ACW_DESKTOP_VENDOR (default
desktop/vendor, filled by the packaging scripts). On Linux, missing pieces fall
back to the server layout in ~/audio-cleaner/{bin,models}.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from acw_common import dfn

_EXE = ".exe" if os.name == "nt" else ""
_FFMPEG = f"ffmpeg{_EXE}"
_FFPROBE = f"ffprobe{_EXE}"


def _vendor_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    env = os.environ.get("ACW_DESKTOP_VENDOR")
    return Path(env) if env else Path(__file__).resolve().parents[1] / "vendor"


def _dev_fallback() -> Path | None:
    if getattr(sys, "frozen", False) or not sys.platform.startswith("linux"):
        return None
    return Path.home() / "audio-cleaner"


def _locate(vendor_sub: str, fallback_sub: str, *names: str) -> Path:
    """Directory holding all of `names`: vendor first, then the dev fallback.
    Returns the vendor directory when nothing matches, so errors name it."""
    vendor = _vendor_root() / vendor_sub
    candidates = [vendor]
    fallback = _dev_fallback()
    if fallback is not None:
        candidates.append(fallback / fallback_sub)
    for d in candidates:
        if all((d / n).is_file() for n in names):
            return d
    return vendor


def libdf_path() -> Path:
    return _locate("native", "bin", dfn.LIB_NAME) / dfn.LIB_NAME


def model_path() -> Path:
    return _locate("models", "models", dfn.MODEL_FILE) / dfn.MODEL_FILE


def ffmpeg_dir() -> Path:
    return _locate("native", "bin", _FFMPEG, _FFPROBE)


def licenses_dir() -> Path:
    return _vendor_root() / "licenses"


def check() -> list[str]:
    """File names of required components that are missing (empty when all present)."""
    missing = []
    if not libdf_path().is_file():
        missing.append(dfn.LIB_NAME)
    if not model_path().is_file():
        missing.append(dfn.MODEL_FILE)
    ff = ffmpeg_dir()
    missing += [n for n in (_FFMPEG, _FFPROBE) if not (ff / n).is_file()]
    return missing
