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

SPEECH_SECONDS = 4  # length of the stand-in used where the dev sample is missing (CI)
SPEECH_SAMPLE = Path.home() / "audio-cleaner" / "dev" / "live_opus.wav"
FULL_FFMPEG = Path(os.environ.get("ACW_FULL_FFMPEG", Path.home() / "audio-cleaner" / "bin" / "ffmpeg"))

_MISSING = resources.check()
_SPEECH = resources.speech_ready()
NATIVE_MARK = "native"
SPEECH_MARK = "speech"


def pytest_collection_modifyitems(config, items):
    native = pytest.mark.skip(reason="missing native components: " + ", ".join(_MISSING))
    speech = pytest.mark.skip(reason="no speech model (prepare_vendor.py --download-whisper puts a test model in vendor/test-models)")
    for item in items:
        if _MISSING and NATIVE_MARK in item.keywords:
            item.add_marker(native)
        elif not _SPEECH and SPEECH_MARK in item.keywords:
            item.add_marker(speech)


def pytest_configure(config):
    config.addinivalue_line("markers", f"{NATIVE_MARK}: needs the bundled libDF, models and FFmpeg")
    config.addinivalue_line("markers", f"{SPEECH_MARK}: also needs a speech model for subtitles")


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
    """About ten seconds of real speech (the dev sample), or a few seconds of noisy tones when that is not available.
    The tones sit in so much noise that the heavy-noise remover is picked, which takes about as long as the
    recording itself: the shorter the stand-in, the faster every test that cleans it."""
    path = tmp_path_factory.mktemp("media") / "speech.wav"
    if SPEECH_SAMPLE.is_file():
        shutil.copy(SPEECH_SAMPLE, path)
        return path
    from scipy.io import wavfile

    sr = 48000
    t = np.arange(SPEECH_SECONDS * sr) / sr
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


def make_video(path: Path, wav: Path, size: tuple[int, int] = (160, 90), seconds: float = 2, fps: int = 10,
               vcodec: str = "mpeg4", acodec: str = "aac", rotation: int = 0) -> Path:
    """A small video of one colour with the sound of `wav`, made by the app's own FFmpeg (its colour filter and its
    MPEG-4 / VP9 encoders: no test video file and no full FFmpeg needed). `rotation` stores it turned, like a phone does."""
    width, height = size
    codec = ["-c:v", vcodec, *(["-b:v", "150k", "-deadline", "realtime", "-cpu-used", "8"] if vcodec == "libvpx-vp9" else [])]
    target = path if not rotation else path.with_name(path.stem + "-straight" + path.suffix)
    run_ffmpeg(
        bundled_ffmpeg(), "-i", str(wav), "-filter_complex", f"color=c=0x3060a0:s={width}x{height}:r={fps}:d={seconds}[v]",
        "-map", "[v]", "-map", "0:a", *codec, "-c:a", acodec, "-t", str(seconds), str(target),
    )
    if rotation:
        run_ffmpeg(bundled_ffmpeg(), "-display_rotation", str(rotation), "-i", str(target), "-c", "copy", str(path))
        target.unlink()
    return path


def stream_facts(path: Path, kind: str) -> list[dict]:
    return [s for s in ffprobe_json(path)["streams"] if s["codec_type"] == kind]


def picture_packets(path: Path) -> list[int]:
    """Sizes of the packets of the first picture stream: equal for two files when the picture was copied, not re-encoded."""
    out = subprocess.run(
        [str(bundled_ffprobe()), "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=size", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return [int(x) for x in out.split()]


def displayed_size(path: Path) -> tuple[int, int]:
    """Width and height of the picture as a player shows it (turned when the file says so)."""
    stream = stream_facts(path, "video")[0]
    turn = int(next((d.get("rotation", 0) for d in stream.get("side_data_list", []) if "rotation" in d), 0))
    width, height = int(stream["width"]), int(stream["height"])
    return (height, width) if abs(turn) % 180 == 90 else (width, height)


@pytest.fixture(scope="session")
def clip_wav(tmp_path_factory) -> Path:
    """Three seconds of a noisy tone: cheap to clean."""
    from scipy.io import wavfile

    path = tmp_path_factory.mktemp("clip") / "clip.wav"
    sr = 48000
    t = np.arange(3 * sr) / sr
    rng = np.random.default_rng(2)
    x = 0.2 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 2 * t) > 0) + 0.03 * rng.standard_normal(t.size)
    wavfile.write(path, sr, (x * 32767).astype(np.int16))
    return path


@pytest.fixture
def quick_clean(monkeypatch):
    """The noise removal is left out (the audio goes through as it is): tests of file types and names do not need it."""
    from acw_common import pipeline

    monkeypatch.setattr(pipeline, "clean_audio", lambda x, *a, **k: np.array(x, dtype=np.float32))


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


@pytest.fixture(autouse=True)
def _no_windows_left_behind():
    """Delete the windows a test opened. They were never deleted, so each test left ~10 more top-level widgets
    (a window, its dialogs and tooltips), and every wait_until after them served all the earlier ones as well: the
    last tests of a long run took ten times as long as they do alone."""
    yield
    widgets = sys.modules.get("PySide6.QtWidgets")
    app = widgets.QApplication.instance() if widgets is not None else None
    if app is None:
        return
    from PySide6.QtCore import QEvent

    app.processEvents()  # timers the test left pending (zero-delay calls) run while the windows still exist
    keep = getattr(sys.modules.get("audio_cleaner_desktop.gui.tips"), "_popup", None)  # the run's one tooltip popup
    for w in app.topLevelWidgets():
        if w is keep:
            continue
        try:
            if hasattr(w, "_closing"):
                w._closing = True  # a main window: no delayed housekeeping after the test has ended
            w.hide()
            w.deleteLater()
        except RuntimeError:  # already deleted
            pass
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)


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


@pytest.fixture(autouse=True)
def _models_in_a_scratch_folder(tmp_path, monkeypatch):
    """No test downloads to, or tidies up in, the real models folder of whoever runs the tests."""
    monkeypatch.setenv("ACW_DESKTOP_MODELS", str(tmp_path / "default-models"))
