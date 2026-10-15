"""Clean one media file end to end. No Qt in here: the GUI worker and the CLI both call process().

    probe -> [transcribe the original] -> decode -> DeepFilterNet3 (+ polish) -> write audio / mux into the video

The result is written next to the source (or into `out_dir`) as "<name>-cleaned.<ext>"
and never replaces an existing file. Subtitles ("<name>-cleaned.srt" ...) go next to it,
or into the video as a soft track.
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

from acw_common import captions, pipeline, transcribe
from acw_common.ffmpeg import AUDIO_FORMATS, FFmpeg, MediaError, ProbeResult

from . import resources

log = logging.getLogger(__name__)

Cancelled = pipeline.Cancelled

# Maximum noise attenuation per strength preset; "max" is DeepFilterNet's no-limit setting
# and matches Studio clean's default on the server (same pipeline).
STRENGTHS: dict[str, float] = {"gentle": 12.0, "balanced": 24.0, "strong": 40.0, "max": 100.0}
DEFAULT_STRENGTH = "max"
OUTPUT_AUDIO_FORMATS = ("wav", "flac", "mp3", "m4a", "opus")
AUDIO_CHOICES = ("same", *OUTPUT_AUDIO_FORMATS)

# Loudness targets (LUFS) for the polish step; None keeps the level of the recording.
LOUDNESS_TARGETS: dict[str, float | None] = {"podcast": -16.0, "youtube": -14.0, "broadcast": -23.0, "keep": None}
DEFAULT_LOUDNESS = "podcast"

# "embed" adds a soft track to a video; for audio files it falls back to an .srt file.
SUBTITLE_MODES = ("off", "srt", "vtt", "txt", "embed")
_SUBTITLE_SUFFIX = {"srt": ".srt", "vtt": ".vtt", "txt": ".txt"}
# code -> (name, ISO 639-2 code for the video's language tag). "auto" detects the language.
LANGUAGES: dict[str, tuple[str, str]] = {
    "en": ("English", "eng"), "es": ("Spanish", "spa"), "fr": ("French", "fra"), "de": ("German", "deu"),
    "it": ("Italian", "ita"), "pt": ("Portuguese", "por"), "nl": ("Dutch", "nld"), "pl": ("Polish", "pol"),
    "ru": ("Russian", "rus"), "uk": ("Ukrainian", "ukr"), "tr": ("Turkish", "tur"), "sv": ("Swedish", "swe"),
    "da": ("Danish", "dan"), "fi": ("Finnish", "fin"), "no": ("Norwegian", "nor"), "cs": ("Czech", "ces"),
    "el": ("Greek", "ell"), "hu": ("Hungarian", "hun"), "ro": ("Romanian", "ron"), "he": ("Hebrew", "heb"),
    "ar": ("Arabic", "ara"), "hi": ("Hindi", "hin"), "id": ("Indonesian", "ind"), "vi": ("Vietnamese", "vie"),
    "th": ("Thai", "tha"), "zh": ("Chinese", "zho"), "ja": ("Japanese", "jpn"), "ko": ("Korean", "kor"),
}

MAX_SECONDS = 6 * 3600
# Audio is processed through memory-mapped scratch files (~0.7 GB of disk per
# hour and channel, twice), so long files need free disk space rather than RAM.
LONG_FILE_SECONDS = 3 * 3600
GB_PER_HOUR_PER_CHANNEL = 1.4

_SAME_AS_INPUT = {
    ".wav": "wav", ".flac": "flac", ".aiff": "wav", ".aif": "wav", ".aifc": "wav",
    ".m4a": "m4a", ".m4b": "m4a", ".mp4": "m4a", ".aac": "m4a", ".mp3": "mp3",
    ".opus": "opus", ".ogg": "opus", ".oga": "opus", ".webm": "opus",
}
_VIDEO_CONTAINERS = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}
# A 24-bit WAV header cannot describe more than 4 GiB of samples.
_WAV_MAX_BYTES = 3.9e9

# Speech recognition runs on 16 kHz mono; each piece is transcribed on its own
# so the model's working memory stays flat for long recordings.
_SPEECH_RATE = transcribe.SAMPLE_RATE
_PIECE_SECONDS = 300
_PIECE_SEARCH_SECONDS = 30

Progress = Callable[[float], None]


@dataclass(frozen=True)
class JobOptions:
    strength: str = DEFAULT_STRENGTH
    polish: bool = True
    audio_format: str = "same"
    loudness: str = DEFAULT_LOUDNESS
    subtitles: str = "off"
    language: str = "auto"

    def __post_init__(self) -> None:
        if self.strength not in STRENGTHS:
            raise ValueError(f"unknown strength {self.strength!r}")
        if self.audio_format not in AUDIO_CHOICES:
            raise ValueError(f"unknown audio format {self.audio_format!r}")
        if self.loudness not in LOUDNESS_TARGETS:
            raise ValueError(f"unknown loudness target {self.loudness!r}")
        if self.subtitles not in SUBTITLE_MODES:
            raise ValueError(f"unknown subtitle mode {self.subtitles!r}")
        if self.language != "auto" and self.language not in LANGUAGES:
            raise ValueError(f"unknown language {self.language!r}")

    @property
    def atten_lim_db(self) -> float:
        return STRENGTHS[self.strength]

    @property
    def loudness_lufs(self) -> float | None:
        return LOUDNESS_TARGETS[self.loudness]


@dataclass(frozen=True)
class JobResult:
    output: Path
    duration: float
    has_video: bool
    subtitles: Path | None = None  # the .srt/.vtt/.txt file written next to `output`
    embedded: bool = False  # subtitles went into the video
    language: str | None = None  # language the speech was transcribed in

    @property
    def paths(self) -> list[Path]:
        return [self.output, *([self.subtitles] if self.subtitles else [])]


def worker_threads() -> int:
    return max(1, min(4, (os.cpu_count() or 2) - 1))


def ffmpeg(threads: int | None = None) -> FFmpeg:
    return FFmpeg(resources.ffmpeg_dir(), threads or worker_threads())


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


def unique_output(directory: Path, stem: str, ext: str, companion: str = "") -> Path:
    """"<stem>-cleaned<ext>", then "<stem>-cleaned (2)<ext>" and so on: the first free name.
    A name is also taken when the file with the `companion` extension (a subtitle file) exists."""
    n = 1
    while True:
        candidate = directory / (f"{stem}-cleaned{ext}" if n == 1 else f"{stem}-cleaned ({n}){ext}")
        if not candidate.exists() and not (companion and candidate.with_suffix(companion).exists()):
            return candidate
        n += 1


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

    def span(self, lo: float, hi: float) -> Progress:
        """A callback for a stage that covers [lo, hi] of the whole job."""
        return lambda f: self(lo + (hi - lo) * f)


def _check_cancel(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise Cancelled()


def _piece_bounds(audio: np.ndarray) -> list[tuple[int, int]]:
    """Cut a long recording into pieces of about five minutes, each cut at the
    quietest tenth of a second in the half minute before it."""
    frame = _SPEECH_RATE // 10
    step, search = _PIECE_SECONDS * _SPEECH_RATE, _PIECE_SEARCH_SECONDS * _SPEECH_RATE
    bounds, start = [], 0
    while len(audio) - start > step + search:
        lo = start + step - search
        window = np.asarray(audio[lo:start + step, 0], dtype=np.float32)
        energy = np.square(window[: len(window) // frame * frame]).reshape(-1, frame).sum(axis=1)
        cut = lo + int(np.argmin(energy)) * frame + frame // 2
        bounds.append((start, cut))
        start = cut
    bounds.append((start, len(audio)))
    return bounds


def _shifted(segments: list[captions.Segment], offset: float) -> list[captions.Segment]:
    for seg in segments:
        seg.start += offset
        seg.end += offset
        for w in seg.words:
            w.start += offset
            w.end += offset
    return segments


def _transcribe(
    ff: FFmpeg, src: Path, info: ProbeResult, work: Path, options: JobOptions, threads: int,
    progress: Progress, cancel: threading.Event,
) -> tuple[list[captions.Segment], str]:
    """Transcribe the original audio (more accurate than the cleaned one)."""
    audio = ff.decode_to_file(src, info, _SPEECH_RATE, 1, MAX_SECONDS, work / "speech.f32")
    language = None if options.language == "auto" else options.language
    segments: list[captions.Segment] = []
    model = resources.whisper_model_path()
    try:
        for a, b in _piece_bounds(audio):
            piece = np.ascontiguousarray(audio[a:b, 0])
            done = a / len(audio)
            share = (b - a) / len(audio)
            found, language = transcribe.transcribe(
                piece, model, language=language, threads=threads,
                progress=lambda f, done=done, share=share: progress(done + share * f), cancel=cancel,
            )
            segments += _shifted(found, a / _SPEECH_RATE)
    except Cancelled:
        raise
    except Exception as e:  # noqa: BLE001 - the model and its runtime can fail in many ways
        log.exception("transcription failed")
        raise MediaError("Could not make the subtitles.") from e
    progress(1.0)
    return segments, language or "en"


def _subtitle_text(kind: str, segments: list[captions.Segment]) -> str:
    if kind == "txt":
        return captions.to_text(segments)
    cues = captions.build_cues(segments)
    return captions.to_vtt(cues) if kind == "vtt" else captions.to_srt(cues)


def process_file(
    src: Path | str,
    options: JobOptions | None = None,
    *,
    out_dir: Path | str | None = None,
    progress: Progress | None = None,
    cancel: threading.Event | None = None,
    threads: int | None = None,
) -> Path:
    """Clean `src` and return the path of the new file (see `process`)."""
    return process(src, options, out_dir=out_dir, progress=progress, cancel=cancel, threads=threads).output


def process(
    src: Path | str,
    options: JobOptions | None = None,
    *,
    out_dir: Path | str | None = None,
    progress: Progress | None = None,
    cancel: threading.Event | None = None,
    threads: int | None = None,
) -> JobResult:
    """Clean `src` (and make subtitles when asked) and say what was written.

    `progress` receives a fraction in [0, 1]. Setting `cancel` raises `Cancelled`
    (at the next checkpoint; encoding steps run to completion first). Raises
    MediaError for problems the user can act on. No partial output is left behind,
    and an existing file is never replaced, so the output cannot be the input.
    `threads` limits the CPU threads this job uses."""
    src = Path(src)
    options = options or JobOptions()
    cancel = cancel or threading.Event()
    report = _Report(progress)
    threads = threads or worker_threads()
    ff = ffmpeg(threads)

    info = probe(src, ff)
    dest_dir = Path(out_dir) if out_dir is not None else src.absolute().parent
    ext = output_extension(info, options)
    tmp_out = dest_dir / f".acw-{uuid.uuid4().hex}{ext}"
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        tmp_out.touch(exist_ok=False)
    except OSError as e:
        raise MediaError(f"Cannot write to the folder {dest_dir}: {e.strerror or e}") from e

    want_subs = options.subtitles != "off"
    embed = options.subtitles == "embed" and info.has_video
    sub_kind = "srt" if options.subtitles == "embed" else options.subtitles  # the file format of the text
    # Share of the progress bar given to each stage.
    transcribe_end = 0.45 if want_subs else 0.0
    decode_end = transcribe_end + 0.04
    clean_end = 0.9

    sub_tmp = None
    try:
        # Decoded and cleaned audio are memory-mapped scratch files next to the
        # output (same disk; never a RAM-backed /tmp), so RAM use stays flat.
        with tempfile.TemporaryDirectory(dir=dest_dir, prefix=".acw-work-") as work_name:
            work = Path(work_name)
            _check_cancel(cancel)
            segments, language = [], None
            if want_subs:
                segments, language = _transcribe(
                    ff, src, info, work, options, threads, report.span(0.0, transcribe_end), cancel
                )
                _check_cancel(cancel)
            has_text = any(s.text.strip() for s in segments)
            if has_text and sub_kind != "off":
                sub_tmp = work / f"subtitles{_SUBTITLE_SUFFIX[sub_kind]}"
                sub_tmp.write_text(_subtitle_text(sub_kind, segments), encoding="utf-8")

            x = pipeline.decode_for_clean(ff, src, info, MAX_SECONDS, work_dir=work)
            report(decode_end)
            y = pipeline.clean_audio(
                x,
                resources.libdf_path(),
                resources.model_path(),
                pipeline.CleanOptions(
                    atten_lim_db=options.atten_lim_db, polish=options.polish, loudness_lufs=options.loudness_lufs
                ),
                threads=threads,
                progress=report.span(decode_end, clean_end),
                cancel=cancel,
                out=pipeline.output_buffer(x, work),
            )
            _check_cancel(cancel)
            _write(ff, src, pipeline.as_output(y), info, options, tmp_out, sub_tmp if embed else None, language)
            # Windows can't delete a file that is still mapped.
            del x, y
            _check_cancel(cancel)
            sidecar = sub_tmp is not None and not embed
            final = unique_output(dest_dir, src.stem, ext, _SUBTITLE_SUFFIX[sub_kind] if sidecar else "")
            sub_final = final.with_suffix(_SUBTITLE_SUFFIX[sub_kind]) if sidecar else None
            os.replace(tmp_out, final)
            if sub_final is not None:
                os.replace(sub_tmp, sub_final)
    finally:
        tmp_out.unlink(missing_ok=True)
    report(1.0)
    return JobResult(
        final, info.duration, info.has_video, sub_final, embedded=embed and sub_tmp is not None, language=language
    )


def _write(
    ff: FFmpeg, src: Path, audio: np.ndarray, info: ProbeResult, options: JobOptions, target: Path,
    subtitles: Path | None, language: str | None,
) -> None:
    try:
        if info.has_video:
            with tempfile.TemporaryDirectory(prefix="acw-") as tmp:
                cleaned = ff.write_audio(audio, pipeline.SAMPLE_RATE, Path(tmp) / "cleaned", "flac")
                iso3 = LANGUAGES.get(language or "", ("", ""))[1] or None
                ff.mux_video(src, cleaned, target, subtitles=subtitles, sub_lang=iso3, keep_metadata=True)
        else:
            ff.write_audio(audio, pipeline.SAMPLE_RATE, target, _audio_format(info, options))
    except RuntimeError as e:
        log.error("encoding failed: %s", e)
        raise MediaError("Saving the cleaned audio failed.") from e
