"""Headless modes: --cli (clean files) and --selftest (check the bundled native pieces)."""
from __future__ import annotations

import signal
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from acw_common import pipeline
from acw_common.ffmpeg import MediaError

from . import engine, resources, sysinfo


class _ProgressLine:
    """Rewrites one line on a terminal; prints a line every 10% otherwise (logs, CI)."""

    def __init__(self, label: str, tty: bool | None = None):
        self._label = label
        self._tty = sys.stdout.isatty() if tty is None else tty
        self._last = -1

    def __call__(self, fraction: float) -> None:
        pct = int(fraction * 100)
        if self._tty:
            print(f"\r{self._label}  {pct:3d}%", end="", flush=True)
        elif pct // 10 > self._last // 10:
            print(f"{self._label}  {pct}%", flush=True)
        self._last = pct

    def finish(self) -> None:
        if self._tty:
            print("\r" + " " * (len(self._label) + 7) + "\r", end="")


def run(inputs: list[str], out_dir: str | None, options: engine.JobOptions, parallel: int = 1,
        output_name: str | None = None, output_format: str | None = None) -> int:
    """Clean each input. Returns 0 when all succeed, 1 when any fails, 130 on Ctrl+C. `parallel` is the number
    of files cleaned at once (1-4, or 0 for Automatic), never more than the memory allows. `output_name` is the
    file name of the result (one input only); `output_format` the type ("mp3", "mp4", ...) of every result."""
    if output_name is not None and len(inputs) != 1:
        print("--output-name works for one file at a time; use --format to choose the type of several.", file=sys.stderr)
        return 1
    missing = resources.check()
    if missing:
        print("Missing components: " + ", ".join(missing), file=sys.stderr)
        return 1
    if options.subtitles != "off" and not resources.speech_ready():
        print("Subtitles need the speech model. Download it first: open Audio Cleaner, Settings, Subtitles "
              "(or run with --download-models).", file=sys.stderr)
        return 1
    if options.rebuild_voice and resources.voice_model_dir() is None:
        print("Rebuilding the voice needs its model. Download it first: open Audio Cleaner, Settings, Extra cleanup "
              "(or run with --download-models).", file=sys.stderr)
        return 1
    cancel = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: cancel.set())
    sysinfo.lower_priority()
    at_once = min(len(inputs), sysinfo.max_parallel(
        parallel, options.subtitles != "off", rebuild_voice=options.may_rebuild, echo_track=options.echo_track))
    threads = sysinfo.threads_per_job(at_once)
    print_lock = threading.Lock()

    def one(n: int, name: str) -> int:
        """0 done, 1 failed, 130 cancelled."""
        label = f"[{n}/{len(inputs)}] {Path(name).name}"
        line = _ProgressLine(label, tty=False if at_once > 1 else None)
        try:
            result = engine.process(name, options, out_dir=out_dir, progress=line, cancel=cancel, threads=threads,
                                    output_name=output_name, output_format=output_format)
        except engine.Cancelled:
            line.finish()
            print(f"{label}  cancelled", file=sys.stderr)
            return 130
        except (MediaError, OSError) as e:
            line.finish()
            print(f"{label}  failed: {e}", file=sys.stderr)
            return 1
        line.finish()
        notes = [*([f"{result.strength} noise reduction picked for this file"] if result.strength else []),
                 *([result.notice] if result.notice else [])]
        note = f"  ({'; '.join(notes)})" if notes else ""
        with print_lock:
            print(f"{label}  ->  " + ", ".join(str(p) for p in result.paths) + note)
        return 0

    if at_once <= 1:
        failures = 0
        for n, name in enumerate(inputs, 1):
            code = one(n, name)
            if code == 130:
                return 130
            failures += code
        return 1 if failures else 0
    with ThreadPoolExecutor(max_workers=at_once) as pool:
        codes = list(pool.map(lambda item: one(*item), enumerate(inputs, 1)))
    if 130 in codes:
        return 130
    return 1 if 1 in codes else 0


def download_models() -> int:
    """Download (or finish downloading) the models for subtitles, restoring highs and rebuilding the voice,
    with a progress line each."""
    from . import models

    cancel = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: cancel.set())
    worst = 0
    for model in (resources.SPEECH_MODEL, resources.HIGHS_MODEL, resources.VOICE_MODEL):
        if models.is_installed(model):
            print(f"{model.title}: already there ({models.location(model)})")
            continue
        line = _ProgressLine(f"Downloading: {model.title} ({model.download_size / 1e9:.1f} GB)")
        try:
            path = models.download(model, progress=lambda done, total: line(done / total), cancel=cancel)
        except models.Cancelled:
            line.finish()
            print("Stopped. Run it again to continue where it left off.", file=sys.stderr)
            return 130
        except models.DownloadError as e:
            line.finish()
            print(f"{model.title}: {e}", file=sys.stderr)
            worst = 1
        else:
            line.finish()
            print(f"{model.title}: done ({path})")
    if worst == 0:
        models.remove_stale()  # copies of older versions of a model, now that the current ones are there
    return worst


def _check_speech_runtime() -> None:
    """The subtitle engine: faster-whisper, CTranslate2's native library and the Silero VAD model
    (onnxruntime) load, and so does the speech model when it has been downloaded. Catches missing
    shared libraries or data files in a frozen build."""
    import ctranslate2
    from faster_whisper import WhisperModel
    from faster_whisper.vad import get_vad_model

    if "int8" not in ctranslate2.get_supported_compute_types("cpu"):
        raise RuntimeError("CTranslate2 cannot run int8 models on this CPU")
    get_vad_model()
    path = resources.whisper_model_path()
    if path is not None:
        WhisperModel(str(path), device="cpu", compute_type="int8", cpu_threads=1, num_workers=1)


def _check_models() -> list[str]:
    """One second of audio through each model the app runs next to the always-installed ones. The noise
    remover for heavy noise ships with the app; the downloaded ones are only checked where they are installed.
    Returns what was skipped."""
    from acw_common import aec, bwe, restore

    sr = pipeline.SAMPLE_RATE
    t = np.arange(sr) / sr
    rng = np.random.default_rng(1)
    x = (0.2 * np.sin(2 * np.pi * 220 * t) + 0.03 * rng.standard_normal(t.size)).astype(np.float32)
    skipped = []

    y = pipeline.clean_audio(
        x, resources.libdf_path(), resources.model_path(), pipeline.CleanOptions(noise="heavy", polish=False),
        threads=1, heavy_model_path=resources.heavy_noise_model_path(),
    )
    if y.shape != (x.size, 1) or not np.isfinite(y).all():
        raise RuntimeError("the heavy-noise remover returned unexpected audio")

    echo = aec.cancel_echo(x, np.roll(x, 480))
    if echo.shape != x.shape or not np.isfinite(echo).all():
        raise RuntimeError("the echo canceller returned unexpected audio")

    highs = resources.highs_model_dir()
    if highs is not None and bwe.available(highs):
        y = bwe.extend(x, highs, threads=1, model="8k")
        if y.shape != x.shape or not np.isfinite(y).all():
            raise RuntimeError("the model that restores highs returned unexpected audio")
    else:
        skipped.append("highs model")

    voice = resources.voice_model_dir()
    if voice is not None and restore.available(voice):
        y = restore.restore(x, voice, threads=1)
        if not np.isfinite(y).all() or abs(len(y) - len(x)) > sr // 10:
            raise RuntimeError("the model that rebuilds the voice returned unexpected audio")
        restore.unload()
    else:
        skipped.append("voice model")
    return skipped


def _check_gui() -> None:
    """The window, the preview and the About box can be built (a missing Qt plugin or data file shows here)."""
    from PySide6.QtCore import QSettings

    from .gui.about import AboutDialog, notice_files
    from .gui.app import create_application
    from .gui.main_window import MainWindow
    from .gui.player import PreviewPane

    app = create_application(["audio-cleaner-selftest"])
    with tempfile.TemporaryDirectory(prefix="acw-selftest-") as tmp:
        window = MainWindow([], settings=QSettings(str(Path(tmp) / "settings.ini"), QSettings.Format.IniFormat),
                            history_path=Path(tmp) / "history.sqlite3")
        try:
            window.show()
            pane = PreviewPane()
            about = AboutDialog(window)
            app.processEvents()
            if getattr(sys, "frozen", False) and not notice_files():
                raise RuntimeError("the licence notices are missing")
            pane.deleteLater()
            about.deleteLater()
        finally:
            window.queue.shutdown()
            window.history.close()


def _check_update_verification() -> None:
    """The bundled `cryptography` can check an Ed25519 signature the way the updater does: a signed manifest is
    accepted, a tampered one is refused, and the embedded public key is a valid key."""
    import base64

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

    from . import updater

    Ed25519PublicKey.from_public_bytes(base64.b64decode(updater.PUBLIC_KEY, validate=True))  # the built-in key loads
    private = Ed25519PrivateKey.generate()
    public = base64.b64encode(private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()
    data = b'{"version": "1.0.0"}'
    updater.verify_signature(data, private.sign(data), public)
    for bad_data, bad_key in ((data + b" ", public), (data, updater.PUBLIC_KEY)):
        try:
            updater.verify_signature(bad_data, private.sign(data), bad_key)
        except updater.UpdateError:
            continue
        raise RuntimeError("the update signature check accepted a signature it should refuse")


def _check_video_output(ff, sound: Path) -> None:
    """FFmpeg can make the video results: a black picture for a sound as MP4 (the system's H.264 encoder where it works,
    FFmpeg's own MPEG-4 otherwise), WebM (libvpx) and Flash video, resized (the colour source, scaler and encoders are
    there), and re-pack a picture into another container."""
    from acw_common.ffmpeg import VideoOut

    for ending, size in ((".mp4", 480), (".webm", None), (".flv", None)):
        made = ff.make_video(sound, sound.with_name("selftest" + ending), 3.0, video=VideoOut(size))
        info = ff.probe(made)
        if not info.has_video or info.video_height != (size or 720) or abs(info.duration - 3.0) > 1.5:
            raise RuntimeError(f"FFmpeg could not make a {ending} video from a sound")
    report: dict = {}
    mkv = ff.mux_video(sound.with_name("selftest.mp4"), sound, sound.with_name("selftest.mkv"), video=VideoOut(), report=report)
    if not ff.probe(mkv).has_video or report.get("encoder") != "copy":
        raise RuntimeError("FFmpeg could not copy a picture into another container")


def _check_windows_quiet() -> None:
    """Windows: `platform` must be able to ask WMI itself (else it runs `cmd /c ver`), and programs libraries start
    must not get a console window."""
    if sys.platform != "win32":
        return
    import _wmi  # noqa: F401 - bundled on purpose (hidden import in audio_cleaner.spec)
    import subprocess

    if not getattr(subprocess.Popen.__init__, "_acw_quiet_popen", False):
        raise RuntimeError("subprocess.Popen is not set to start programs without a console window")


def selftest(gui: bool = False) -> int:
    """Denoise 3 s of noisy tone and round-trip it through FFmpeg, using the resolved native pieces,
    load the subtitle engine, run a second through the other models that are there and, with `gui`, build the
    window."""
    try:
        missing = resources.check()
        if missing:
            raise RuntimeError("missing components: " + ", ".join(missing))
        sr = pipeline.SAMPLE_RATE
        t = np.arange(3 * sr) / sr
        rng = np.random.default_rng(0)
        x = (0.2 * np.sin(2 * np.pi * 220 * t) + 0.03 * rng.standard_normal(t.size)).astype(np.float32)
        y = pipeline.clean_audio(x, resources.libdf_path(), resources.model_path(), threads=1)
        if y.shape != (x.size, 1) or not np.isfinite(y).all():
            raise RuntimeError("the noise reduction engine returned unexpected audio")
        ff = engine.ffmpeg()
        with tempfile.TemporaryDirectory(prefix="acw-selftest-") as tmp:
            out = ff.write_audio(pipeline.as_output(y), sr, Path(tmp) / "selftest", "flac")
            info = ff.probe(out)
            decoded = ff.decode(out, info, sr, 1, 10)
            _check_video_output(ff, out)
        if abs(info.duration - 3.0) > 0.2 or info.audio_channels != 1 or decoded.size < 2.8 * sr:
            raise RuntimeError(f"FFmpeg round-trip mismatch (duration {info.duration:.2f} s)")
        _check_windows_quiet()
        _check_update_verification()
        _check_speech_runtime()
        skipped = _check_models()
        if gui:
            _check_gui()
    except Exception as e:  # noqa: BLE001 - any failure is a failed selftest
        print(f"selftest failed: {e}")
        return 1
    if skipped:
        print("not checked (not installed): " + ", ".join(skipped))
    print("selftest ok")
    return 0
