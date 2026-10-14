"""Clean one media file end to end. No Qt in here: the GUI worker and the CLI both call process_file().

    probe -> decode -> DeepFilterNet3 (+ polish) -> write audio / mux into the video

The result is written next to the source (or into `out_dir`) as "<name>-cleaned.<ext>"
and never replaces an existing file.
"""
from __future__ import annotations

import logging
import os
import tempfile
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from acw_common import pipeline
from acw_common.ffmpeg import AUDIO_FORMATS, FFmpeg, MediaError, ProbeResult

from . import resources

log = logging.getLogger(__name__)

Cancelled = pipeline.Cancelled

# Maximum noise attenuation per strength preset; "max" is DeepFilterNet's no-limit setting
# and matches the website's default.
STRENGTHS: dict[str, float] = {"gentle": 12.0, "balanced": 24.0, "strong": 40.0, "max": 100.0}
DEFAULT_STRENGTH = "max"
# The bundled LGPL FFmpeg has no MP3 encoder.
OUTPUT_AUDIO_FORMATS = ("wav", "flac", "m4a", "opus")
AUDIO_CHOICES = ("same", *OUTPUT_AUDIO_FORMATS)

MAX_SECONDS = 6 * 3600
# Audio is processed through memory-mapped scratch files (~0.7 GB of disk per
# hour and channel, twice), so long files need free disk space rather than RAM.
LONG_FILE_SECONDS = 3 * 3600
GB_PER_HOUR_PER_CHANNEL = 1.4

_SAME_AS_INPUT = {
    ".wav": "wav", ".flac": "flac", ".aiff": "wav", ".aif": "wav", ".aifc": "wav",
    ".m4a": "m4a", ".m4b": "m4a", ".mp4": "m4a", ".aac": "m4a", ".mp3": "m4a",
    ".opus": "opus", ".ogg": "opus", ".oga": "opus", ".webm": "opus",
}
_VIDEO_CONTAINERS = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}
# A 24-bit WAV header cannot describe more than 4 GiB of samples.
_WAV_MAX_BYTES = 3.9e9

# Share of the progress bar given to each stage.
_DECODE_SHARE, _CLEAN_SHARE = 0.05, 0.85

Progress = Callable[[float], None]


@dataclass(frozen=True)
class JobOptions:
    strength: str = DEFAULT_STRENGTH
    polish: bool = True
    audio_format: str = "same"

    def __post_init__(self) -> None:
        if self.strength not in STRENGTHS:
            raise ValueError(f"unknown strength {self.strength!r}")
        if self.audio_format not in AUDIO_CHOICES:
            raise ValueError(f"unknown audio format {self.audio_format!r}")

    @property
    def atten_lim_db(self) -> float:
        return STRENGTHS[self.strength]


def worker_threads() -> int:
    return max(1, min(4, (os.cpu_count() or 2) - 1))


def ffmpeg() -> FFmpeg:
    return FFmpeg(resources.ffmpeg_dir(), worker_threads())


def probe(path: Path | str, ff: FFmpeg | None = None) -> ProbeResult:
    """Read a file's basic facts. Raises MediaError with a message fit for the user."""
    if not Path(path).is_file():
        raise MediaError("This file was not found.")
    result = (ff or ffmpeg()).probe(Path(path))
    if result.duration > MAX_SECONDS:
        raise MediaError(f"This file is longer than {MAX_SECONDS // 3600} hours, which is the limit.")
    return result


def estimated_scratch_gb(duration: float, channels: int) -> float:
    """Temporary disk space a job needs next to its output."""
    return duration / 3600 * min(channels, pipeline.MAX_CHANNELS) * GB_PER_HOUR_PER_CHANNEL


def output_extension(probe_result: ProbeResult, options: JobOptions) -> str:
    """Extension of the cleaned file: the container of the video, or the chosen audio format."""
    if probe_result.has_video:
        return probe_result.ext if probe_result.ext in _VIDEO_CONTAINERS else ".mkv"
    return AUDIO_FORMATS[_audio_format(probe_result, options)][0]


def _audio_format(probe_result: ProbeResult, options: JobOptions) -> str:
    fmt = options.audio_format
    if fmt == "same":
        fmt = _SAME_AS_INPUT.get(probe_result.ext, "flac")
    channels = min(probe_result.audio_channels, pipeline.MAX_CHANNELS)
    if fmt == "wav" and probe_result.duration * pipeline.SAMPLE_RATE * channels * 3 > _WAV_MAX_BYTES:
        return "flac"
    return fmt


def unique_output(directory: Path, stem: str, ext: str) -> Path:
    """"<stem>-cleaned<ext>", then "<stem>-cleaned (2)<ext>" and so on: the first free name."""
    candidate = directory / f"{stem}-cleaned{ext}"
    n = 2
    while candidate.exists():
        candidate = directory / f"{stem}-cleaned ({n}){ext}"
        n += 1
    return candidate


class _Report:
    """Forwards a never-decreasing fraction of the whole job."""

    def __init__(self, callback: Progress | None):
        self._callback = callback
        self._last = -1.0

    def __call__(self, fraction: float) -> None:
        fraction = min(1.0, max(self._last, fraction))
        if self._callback is not None and fraction != self._last:
            self._callback(fraction)
        self._last = fraction


def _check_cancel(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise Cancelled()


def process_file(
    src: Path | str,
    options: JobOptions | None = None,
    *,
    out_dir: Path | str | None = None,
    progress: Progress | None = None,
    cancel: threading.Event | None = None,
) -> Path:
    """Clean `src` and return the path of the new file.

    `progress` receives a fraction in [0, 1]. Setting `cancel` raises `Cancelled`
    (at the next checkpoint; encoding steps run to completion first). Raises
    MediaError for problems the user can act on. No partial output is left behind,
    and an existing file is never replaced, so the output cannot be the input."""
    src = Path(src)
    options = options or JobOptions()
    cancel = cancel or threading.Event()
    report = _Report(progress)
    ff = ffmpeg()

    info = probe(src, ff)
    dest_dir = Path(out_dir) if out_dir is not None else src.absolute().parent
    ext = output_extension(info, options)
    tmp_out = dest_dir / f".acw-{uuid.uuid4().hex}{ext}"
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        tmp_out.touch(exist_ok=False)
    except OSError as e:
        raise MediaError(f"Cannot write to the folder {dest_dir}: {e.strerror or e}") from e

    try:
        # Decoded and cleaned audio are memory-mapped scratch files next to the
        # output (same disk; never a RAM-backed /tmp), so RAM use stays flat.
        with tempfile.TemporaryDirectory(dir=dest_dir, prefix=".acw-work-") as work:
            _check_cancel(cancel)
            x = pipeline.decode_for_clean(ff, src, info, MAX_SECONDS, work_dir=Path(work))
            report(_DECODE_SHARE)
            y = pipeline.clean_audio(
                x,
                resources.libdf_path(),
                resources.model_path(),
                pipeline.CleanOptions(atten_lim_db=options.atten_lim_db, polish=options.polish),
                threads=worker_threads(),
                progress=lambda f: report(_DECODE_SHARE + _CLEAN_SHARE * f),
                cancel=cancel,
                out=pipeline.output_buffer(x, Path(work)),
            )
            _check_cancel(cancel)
            _write(ff, src, pipeline.as_output(y), info, options, tmp_out)
            # Windows can't delete a file that is still mapped.
            del x, y
        _check_cancel(cancel)
        final = unique_output(dest_dir, src.stem, ext)
        os.replace(tmp_out, final)
    finally:
        tmp_out.unlink(missing_ok=True)
    report(1.0)
    return final


def _write(ff: FFmpeg, src: Path, audio: np.ndarray, info: ProbeResult, options: JobOptions, target: Path) -> None:
    try:
        if info.has_video:
            with tempfile.TemporaryDirectory(prefix="acw-") as tmp:
                cleaned = ff.write_audio(audio, pipeline.SAMPLE_RATE, Path(tmp) / "cleaned", "flac")
                ff.mux_video(src, cleaned, target, keep_metadata=True)
        else:
            ff.write_audio(audio, pipeline.SAMPLE_RATE, target, _audio_format(info, options))
    except RuntimeError as e:
        log.error("encoding failed: %s", e)
        raise MediaError("Saving the cleaned audio failed.") from e
