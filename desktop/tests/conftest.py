"""Shared fixtures. desktop/ and common/ go on sys.path; tests that need the bundled native
pieces (libDF, models, FFmpeg) are skipped when `resources.check()` reports something missing."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for sub in ("desktop", "common"):
    path = str(ROOT / sub)
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from audio_cleaner_desktop import resources  # noqa: E402

SPEECH_SAMPLE = Path.home() / "audio-cleaner" / "dev" / "live_opus.wav"
FULL_FFMPEG = Path(os.environ.get("ACW_FULL_FFMPEG", Path.home() / "audio-cleaner" / "bin" / "ffmpeg"))

_MISSING = resources.check()
NATIVE_MARK = "native"


def pytest_collection_modifyitems(config, items):
    if not _MISSING:
        return
    skip = pytest.mark.skip(reason="missing native components: " + ", ".join(_MISSING))
    for item in items:
        if NATIVE_MARK in item.keywords:
            item.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line("markers", f"{NATIVE_MARK}: needs the bundled libDF, models and FFmpeg")


def run_ffmpeg(binary: Path, *args: str) -> None:
    subprocess.run([str(binary), "-v", "error", "-y", *args], check=True, stdin=subprocess.DEVNULL)


def bundled_ffmpeg() -> Path:
    return resources.ffmpeg_dir() / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")


def bundled_ffprobe() -> Path:
    return resources.ffmpeg_dir() / ("ffprobe.exe" if os.name == "nt" else "ffprobe")


def ffprobe_json(path: Path) -> dict:
    import json

    out = subprocess.run(
        [str(bundled_ffprobe()), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        check=True, capture_output=True,
    ).stdout
    return json.loads(out)


@pytest.fixture(scope="session")
def speech_wav(tmp_path_factory) -> Path:
    """About ten seconds of real speech (the dev sample), or noisy tones when that is not available."""
    path = tmp_path_factory.mktemp("media") / "speech.wav"
    if SPEECH_SAMPLE.is_file():
        shutil.copy(SPEECH_SAMPLE, path)
        return path
    from scipy.io import wavfile

    sr = 48000
    t = np.arange(10 * sr) / sr
    rng = np.random.default_rng(1)
    x = 0.2 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 2 * t) > 0) + 0.03 * rng.standard_normal(t.size)
    wavfile.write(path, sr, (x * 32767).astype(np.int16))
    return path


@pytest.fixture(scope="session")
def has_real_speech() -> bool:
    return SPEECH_SAMPLE.is_file()


@pytest.fixture(scope="session")
def mp3_file(speech_wav, tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("media") / "talk.mp3"
    run_ffmpeg(bundled_ffmpeg(), "-i", str(speech_wav), "-t", "4", "-ac", "1", str(path))
    return path


@pytest.fixture(scope="session")
def mp4_file(speech_wav, tmp_path_factory) -> Path:
    """A small video with speech and a creation_time tag; needs a full FFmpeg to make."""
    if not FULL_FFMPEG.is_file():
        pytest.skip("a full FFmpeg is needed to generate test video (set ACW_FULL_FFMPEG)")
    path = tmp_path_factory.mktemp("media") / "clip.mp4"
    run_ffmpeg(
        FULL_FFMPEG, "-f", "lavfi", "-i", "testsrc=size=160x90:rate=10", "-i", str(speech_wav), "-t", "5",
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest",
        "-metadata", "creation_time=2021-03-04T05:06:07Z", str(path),
    )
    return path


def wait_until(predicate, timeout: float = 120.0, interval: float = 0.01) -> bool:
    """Run the Qt event loop until `predicate()` is true. (QTest.qWait holds the GIL, which would
    starve the worker threads that run Python code.)"""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture(scope="session")
def qapp():
    try:
        from PySide6.QtCore import QStandardPaths

        QStandardPaths.setTestModeEnabled(True)
        from audio_cleaner_desktop.gui.app import create_application

        app = create_application(["audio-cleaner-tests"])
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Qt cannot start here: {e}")
    return app
