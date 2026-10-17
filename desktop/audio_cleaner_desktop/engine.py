"""Clean one media file end to end. No Qt in here: the GUI worker and the CLI both call process().

    probe -> [transcribe the original] -> decode -> DeepFilterNet3 (+ leveling) -> write audio / mux into the video

The result is written next to the source (or into `out_dir`) as "<name>-cleaned.<ext>" (or under the name and in the
format asked for: its ending says which, see available_endings()) and never replaces an existing file. A video can
become another kind of video, or just its sound; a sound can become a video with a black picture (so it can carry
subtitles). Subtitles ("<name>-cleaned.srt" ...) go next to it, or into the video as a soft track.
"""
from __future__ import annotations

import gc
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from acw_common import advice, captions, pipeline, transcribe
from acw_common.ffmpeg import AUDIO_FORMATS, FFmpeg, MediaError, ProbeResult, VideoOut

from . import models, resources

log = logging.getLogger(__name__)

Cancelled = pipeline.Cancelled

# Maximum noise attenuation per strength preset; "max" is DeepFilterNet's no-limit setting
# and matches Studio clean's default on the server (same pipeline).
STRENGTHS: dict[str, float] = {"gentle": 12.0, "balanced": 24.0, "strong": 40.0, "max": 100.0}
DEFAULT_STRENGTH = "max"
# "auto": each file gets the strength that fits how noisy it is (pipeline.strength_for, measured by the noise remover).
AUTO_STRENGTH = "auto"
STRENGTH_CHOICES = (AUTO_STRENGTH, *STRENGTHS)
OUTPUT_AUDIO_FORMATS = ("wav", "flac", "mp3", "m4a", "opus")
AUDIO_CHOICES = ("same", *OUTPUT_AUDIO_FORMATS)
# Every ending a result can have. Sound: the name in acw_common.ffmpeg.AUDIO_FORMATS; video: the container.
AUDIO_ENDINGS = {".mp3": "mp3", ".wav": "wav", ".flac": "flac", ".m4a": "m4a", ".opus": "opus", ".ogg": "ogg"}
VIDEO_ENDINGS = (".mp4", ".mov", ".mkv", ".flv", ".webm")
ENDING_NAMES = {
    ".mp3": "MP3", ".wav": "WAV", ".flac": "FLAC", ".m4a": "M4A", ".opus": "Opus", ".ogg": "OGG",
    ".mp4": "MP4", ".mov": "MOV", ".mkv": "MKV", ".flv": "FLV", ".webm": "WebM", ".m4v": "M4V",
}
# A picture that WebM takes as it is; any other picture becomes VP9 (libvpx is part of the FFmpeg build).
WEBM_COPY_CODECS = ("vp8", "vp9", "av1")
WEBM_ENCODE = True
# Containers that hold soft subtitles; the others (flv) get an .srt file next to the result instead.
_EMBED_ENDINGS = (".mp4", ".m4v", ".mov", ".mkv", ".webm")
# Video size: the shorter side in pixels ("same" keeps the original).
VIDEO_SIZES = ("same", "2160", "1440", "1080", "720", "480")

# Loudness targets (LUFS) for the leveling step; None keeps the level of the recording.
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

SCRATCH_SAFETY = 1.1  # free space wanted in the scratch folder, as a share more than the estimate
STALE_SECONDS = 24 * 3600  # leftovers of a crashed run older than this are swept away
_WORK_PREFIX = ".acw-work-"
_LEFTOVER_FILE = re.compile(r"^\.acw-[0-9a-f]{32}\.[A-Za-z0-9]{1,5}$")

Progress = Callable[[float], None]


@dataclass(frozen=True)
class JobOptions:
    strength: str = DEFAULT_STRENGTH
    leveling: bool = True
    audio_format: str = "same"
    loudness: str = DEFAULT_LOUDNESS
    subtitles: str = "off"
    language: str = "auto"
    noise: str = "auto"  # pipeline.NOISE_PROFILES: measured per file, or forced light / heavy
    echo_track: bool = False  # remove speaker echo using the file's second audio track (when it has one)
    restore_highs: bool = True  # bring back the highs of phone-quality recordings (needs HIGHS_MODEL)
    rebuild_voice: bool = False  # Sidon instead of noise removal: also removes room echo; slow (VOICE_MODEL)
    # Rebuild the voice of a file that sounds echoey or muffled (advice.rebuild_reason), when VOICE_MODEL is installed.
    auto_rebuild: bool = False
    video_size: str = "same"  # of a video result: "same", or the pixels of the shorter side (VIDEO_SIZES)

    def __post_init__(self) -> None:
        if self.noise not in pipeline.NOISE_PROFILES:
            raise ValueError(f"unknown noise setting {self.noise!r}")
        if self.strength not in STRENGTH_CHOICES:
            raise ValueError(f"unknown strength {self.strength!r}")
        if self.audio_format not in AUDIO_CHOICES:
            raise ValueError(f"unknown audio format {self.audio_format!r}")
        if self.loudness not in LOUDNESS_TARGETS:
            raise ValueError(f"unknown loudness target {self.loudness!r}")
        if self.subtitles not in SUBTITLE_MODES:
            raise ValueError(f"unknown subtitle mode {self.subtitles!r}")
        if self.language != "auto" and self.language not in LANGUAGES:
            raise ValueError(f"unknown language {self.language!r}")
        if self.video_size not in VIDEO_SIZES:
            raise ValueError(f"unknown video size {self.video_size!r}")

    @property
    def atten_lim_db(self) -> float | None:
        """None: the one that fits each file (measured when it is cleaned)."""
        return None if self.strength == AUTO_STRENGTH else STRENGTHS[self.strength]

    @property
    def loudness_lufs(self) -> float | None:
        return LOUDNESS_TARGETS[self.loudness]

    @property
    def may_rebuild(self) -> bool:
        """The voice may be rebuilt (what the memory planning has to allow for)."""
        return self.rebuild_voice or (self.auto_rebuild and resources.voice_model_dir() is not None)

    @property
    def short_side(self) -> int | None:
        """Pixels of the shorter side of a video result; None keeps the size of the original."""
        return None if self.video_size == "same" else int(self.video_size)


@dataclass(frozen=True)
class JobResult:
    output: Path
    duration: float
    has_video: bool
    subtitles: Path | None = None  # the .srt/.vtt/.txt file written next to `output`
    embedded: bool = False  # subtitles went into the video
    language: str | None = None  # language the speech was transcribed in
    notice: str | None = None  # something the user should know about how the file was cleaned
    strength: str | None = None  # the strength picked for this file (STRENGTHS key) when it was set to automatic
    rebuilt: str | None = None  # why the voice was rebuilt automatically ("echoey", "muffled"); None when it wasn't

    @property
    def paths(self) -> list[Path]:
        return [self.output, *([self.subtitles] if self.subtitles else [])]


def worker_threads() -> int:
    return max(1, min(4, (os.cpu_count() or 2) - 1))


def ffmpeg(threads: int | None = None, cancel: threading.Event | None = None) -> FFmpeg:
    """The FFmpeg programs; setting `cancel` stops whatever is running at once (raises Cancelled)."""
    return FFmpeg(resources.ffmpeg_dir(), threads or worker_threads(), cancel=cancel)


def probe(path: Path | str, ff: FFmpeg | None = None) -> ProbeResult:
    """Read a file's basic facts. Raises MediaError with a message fit for the user."""
    if not Path(path).is_file():
        raise MediaError("This file was not found.")
    result = (ff or ffmpeg()).probe(Path(path))
    if result.duration > MAX_SECONDS:
        raise MediaError(f"This file is longer than {MAX_SECONDS // 3600} hours, which is the limit.")
    if result.has_video and result.video_codec == "flv1":
        raise MediaError(
            "This video uses an old Flash video format that can't be put back together with new sound. "
            "Convert it to MP4 first, then clean that."
        )
    return result


def estimated_scratch_gb(duration: float, channels: int) -> float:
    """Temporary disk space a job needs next to its output."""
    return duration / 3600 * min(channels, pipeline.MAX_CHANNELS) * GB_PER_HOUR_PER_CHANNEL


@dataclass(frozen=True)
class Source:
    """What decides which results a file can become."""

    has_video: bool
    ext: str = ""  # of the file, lower case with the dot
    video_codec: str | None = None
    duration: float = 0.0
    channels: int = 2

    @classmethod
    def of(cls, info: object, path: Path | str | None = None) -> "Source":
        """From a probe result (or anything with the same fields: the ones that are missing count as unknown)."""
        ext = getattr(info, "ext", "") or (Path(path).suffix.lower() if path else "")
        return cls(
            bool(getattr(info, "has_video", False)), ext, getattr(info, "video_codec", None),
            float(getattr(info, "duration", 0.0) or 0.0), int(getattr(info, "audio_channels", 2) or 2),
        )


def default_ending(source: Source, options: JobOptions) -> str:
    """Ending of the cleaned file when nothing else is asked for: the container of the video, or the chosen audio format."""
    if source.has_video:
        return source.ext if source.ext in _VIDEO_CONTAINERS else ".mkv"
    return AUDIO_FORMATS[_audio_format(source, options)][0]


def output_extension(probe_result: ProbeResult, options: JobOptions) -> str:
    return default_ending(Source.of(probe_result), options)


def available_endings(source: Source) -> list[str]:
    """The endings this file can become, in the order the format list shows them: the file's own kind first (a video:
    the video types, then the sound types; a sound: the sound types, then the video types)."""
    video = [e for e in VIDEO_ENDINGS if e != ".webm" or WEBM_ENCODE or (source.has_video and source.video_codec in WEBM_COPY_CODECS)]
    if source.has_video and source.ext == ".m4v":
        video.insert(1, ".m4v")
    audio = list(AUDIO_ENDINGS)
    return [*video, *audio] if source.has_video else [*audio, *video]


def split_name(name: str) -> tuple[str, str]:
    """"talk-cleaned.MP3" -> ("talk-cleaned", ".mp3"); a name without an ending gives an empty ending."""
    stem, dot, ending = name.rpartition(".")
    return (stem, "." + ending.lower()) if dot else (name, "")


def default_output_name(path: Path | str, source: Source, options: JobOptions, ending: str | None = None) -> str:
    """"<name>-cleaned.<ending>": what the result is called unless the user types another name."""
    return f"{Path(path).stem}-cleaned{ending or default_ending(source, options)}"


def with_ending(name: str, ending: str) -> str:
    """`name` with another ending: a known one is replaced, anything else (no ending, or something that is only part of
    the name, like "v1.2") gets the new one added."""
    stem, current = split_name(name.strip())
    if current in ENDING_NAMES:
        return stem + ending
    return name.strip() + ending


_BAD_NAME_CHARS = set('<>:"/\\|?*')
_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def check_output_name(name: str, source: Source) -> tuple[str, str]:
    """Is `name` a good name for a result of `source`? Returns (ending, "") when it is, ("", reason) when not; the reason
    is short and fit for the user."""
    name = name.strip()
    if not name:
        return "", "Type a name for the cleaned file."
    if any(c in _BAD_NAME_CHARS or ord(c) < 32 for c in name):
        return "", 'A name can\'t contain any of these: \\ / : * ? " < > |'
    stem, ending = split_name(name)
    if not ending:
        return "", "Add an ending such as .mp3, or pick a type from the list."
    if not stem.strip() or stem.strip() == ".":
        return "", f"Type a name in front of {ending}."
    if name.endswith((".", " ")):
        return "", "A name can't end with a dot or a space."
    if len(name.encode("utf-8")) > 255:
        return "", "That name is too long."
    if stem.split(".")[0].strip().upper() in _RESERVED_NAMES:
        return "", "Windows doesn't allow that name."
    if ending not in available_endings(source):
        options = ", ".join(e for e in available_endings(source)[:6])
        return "", f"This file can't be saved as {ending}. Pick a type from the list ({options} ...)."
    if ending == ".wav" and _wav_too_big(source):
        return "", "This recording is too long for a WAV file. Pick FLAC or another type."
    return ending, ""


def _wav_too_big(source: Source) -> bool:
    channels = min(source.channels, pipeline.MAX_CHANNELS)
    return source.duration * pipeline.SAMPLE_RATE * channels * 3 > _WAV_MAX_BYTES


def _audio_format(source: Source | ProbeResult, options: JobOptions) -> str:
    if not isinstance(source, Source):
        source = Source.of(source)
    fmt = options.audio_format
    if fmt == "same":
        fmt = _SAME_AS_INPUT.get(source.ext, "flac")
    if fmt == "wav" and _wav_too_big(source):
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


def claim_output(directory: Path, stem: str, ext: str, companion: str = "", suffix: str = "-cleaned") -> Path:
    """Like unique_output(), but the name is taken for good: an empty file is created under it (and under the
    companion name), so a job running at the same time can't pick the same name. The caller replaces the
    placeholder with the result, or removes it. With `suffix` "" the first try is `stem` itself (the name the user
    typed) and a taken name becomes "<stem> (2)"."""
    n = 1
    while True:
        candidate = directory / (f"{stem}{suffix}{ext}" if n == 1 else f"{stem}{suffix} ({n}){ext}")
        n += 1
        mine: list[Path] = []
        try:
            for path in (candidate, *([candidate.with_suffix(companion)] if companion else [])):
                os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666))
                mine.append(path)
            return candidate
        except FileExistsError:
            for path in mine:
                path.unlink(missing_ok=True)


def scratch_root() -> Path | None:
    """Where the temporary work files go: the app's own data folder on Windows and macOS, not the folder of the
    result, which is often synced to the cloud or on a slow drive. None means next to the result (Linux, where
    the temporary folder is memory). $ACW_DESKTOP_SCRATCH overrides it."""
    env = os.environ.get("ACW_DESKTOP_SCRATCH")
    if env:
        return Path(env)
    if sys.platform in ("win32", "darwin"):
        return models.app_data_dir() / "scratch"
    return None


def _work_parent(dest_dir: Path, need_gb: float) -> Path:
    """The folder for this job's work folder: the scratch folder when it has the room, else next to the result."""
    root = scratch_root()
    if root is not None:
        try:
            root.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(root).free / 1e9 >= need_gb * SCRATCH_SAFETY:
                return root
        except OSError:
            pass
    return dest_dir


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _remove_quietly(path: Path) -> None:
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    except OSError:
        pass


def sweep_leftovers(folders=(), *, now: float | None = None) -> int:
    """Remove what a crashed or killed run left behind: everything in the scratch folder whose process is gone,
    and ".acw-work-*" folders and ".acw-<id>.<ext>" files older than a day in `folders` (the folders results
    were written to). Never touches anything else. Returns how many items were removed."""
    now = time.time() if now is None else now
    removed = 0
    root = scratch_root()
    if root is not None and root.is_dir():
        for entry in root.iterdir():
            match = re.match(rf"^{re.escape(_WORK_PREFIX)}(\d+)-", entry.name)
            if match and _pid_alive(int(match.group(1))):
                continue  # a job of another Audio Cleaner window or the command line, still running
            if not match and _age(entry, now) < STALE_SECONDS:
                continue
            _remove_quietly(entry)
            removed += 1
    for folder in dict.fromkeys(Path(f) for f in folders):
        try:
            entries = list(folder.iterdir())
        except OSError:
            continue
        for entry in entries:
            if (entry.name.startswith(_WORK_PREFIX) and entry.is_dir()) or _LEFTOVER_FILE.match(entry.name):
                if _age(entry, now) >= STALE_SECONDS:
                    _remove_quietly(entry)
                    removed += 1
    return removed


def _age(path: Path, now: float) -> float:
    try:
        return now - path.stat().st_mtime
    except OSError:
        return 0.0


def release_memory() -> None:
    """Let go of the models kept in memory between files (the speech model, Sidon, AP-BWE, DPDFNet): the
    window calls this when the queue runs dry, so an idle app does not hold gigabytes."""
    from acw_common import bwe, dpdfnet, restore

    transcribe.unload()
    restore.unload()
    bwe.unload()
    dpdfnet.unload()
    gc.collect()


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
    progress: Progress, cancel: threading.Event, echo_free: np.ndarray | None = None,
) -> tuple[list[captions.Segment], str]:
    """Transcribe the original audio (more accurate than the cleaned one), or the echo-free mic track
    when the speaker echo was removed first."""
    if echo_free is not None:
        audio = pipeline.to_speech_16k(echo_free)[:, None]
    else:
        audio = ff.decode_to_file(src, info, _SPEECH_RATE, 1, MAX_SECONDS, work / "speech.f32")
    language = None if options.language == "auto" else options.language
    segments: list[captions.Segment] = []
    model = resources.whisper_model_path()
    if model is None:
        raise MediaError("Subtitles need the speech model, which isn't downloaded yet. Download it in Settings, then try again.")
    try:
        loaded = transcribe.load_model(model, threads)  # once for all the pieces of this file
        for a, b in _piece_bounds(audio):
            piece = np.ascontiguousarray(audio[a:b, 0])
            done = a / len(audio)
            share = (b - a) / len(audio)
            found, language = transcribe.transcribe(
                piece, model, language=language, threads=threads, model=loaded,
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


def _subtitle_text(kind: str, segments: list[captions.Segment], language: str | None = None) -> str:
    if kind == "txt":
        return captions.to_text(segments, language)
    cues = captions.build_cues(segments, language)
    return captions.to_vtt(cues) if kind == "vtt" else captions.to_srt(cues)


def process_file(
    src: Path | str,
    options: JobOptions | None = None,
    *,
    out_dir: Path | str | None = None,
    progress: Progress | None = None,
    cancel: threading.Event | None = None,
    threads: int | None = None,
    output_name: str | None = None,
    output_format: str | None = None,
) -> Path:
    """Clean `src` and return the path of the new file (see `process`)."""
    return process(src, options, out_dir=out_dir, progress=progress, cancel=cancel, threads=threads,
                   output_name=output_name, output_format=output_format).output


def output_name_for(src: Path | str, source: Source, options: JobOptions, output_name: str | None = None,
                    output_format: str | None = None) -> str:
    """The name of the result: `output_name` when given (its ending says the format),
    else "<name>-cleaned" in the format `output_format` ("mp3" or ".mp3"), else in the usual one. MediaError when the
    name or the format is not possible for this file."""
    ending = None
    if output_format:
        ending = "." + output_format.lower().lstrip(".")
    if output_name is None:
        name = default_output_name(src, source, options, ending)
    else:
        name = output_name.strip()
        if ending and split_name(name)[1] != ending:
            raise MediaError(f"The name {name} does not end in {ending}.")
    problem = check_output_name(name, source)[1]
    if problem:
        raise MediaError(problem)
    return name


def process(
    src: Path | str,
    options: JobOptions | None = None,
    *,
    out_dir: Path | str | None = None,
    progress: Progress | None = None,
    cancel: threading.Event | None = None,
    threads: int | None = None,
    output_name: str | None = None,
    output_format: str | None = None,
) -> JobResult:
    """Clean `src` (and make subtitles when asked) and say what was written.

    `output_name` is the file name of the result (its ending, such as .mp3 or .mp4, says what it becomes); without it
    the result is "<name>-cleaned" plus the usual ending, or the one in `output_format`. A taken name gets " (2)" and
    so on.

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
    ff = ffmpeg(threads, cancel)

    info = probe(src, ff)
    dest_dir = Path(out_dir) if out_dir is not None else src.absolute().parent
    out_stem, ext = split_name(output_name_for(src, Source.of(info), options, output_name, output_format))
    tmp_out = dest_dir / f".acw-{uuid.uuid4().hex}{ext}"
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        tmp_out.touch(exist_ok=False)
    except OSError as e:
        raise MediaError(f"Cannot write to the folder {dest_dir}: {e.strerror or e}") from e

    want_subs = options.subtitles != "off"
    embed = options.subtitles == "embed" and ext in _EMBED_ENDINGS
    sub_kind = "srt" if options.subtitles == "embed" else options.subtitles  # the file format of the text
    # Share of the progress bar given to each stage.
    transcribe_end = 0.45 if want_subs else 0.0
    decode_end = transcribe_end + 0.04
    clean_end = 0.9

    sub_tmp = None
    x = y = None
    claimed: list[Path] = []
    try:
        # Decoded and cleaned audio are memory-mapped scratch files, so RAM use stays flat. On Windows and macOS
        # they go to the app's own data folder (the result's folder is often synced to the cloud, and a USB
        # stick or network share would make the work slow); where that is too full, and on Linux (whose
        # temporary folder is memory), next to the result.
        work = Path(tempfile.mkdtemp(
            dir=_work_parent(dest_dir, estimated_scratch_gb(info.duration, info.audio_channels)),
            prefix=f"{_WORK_PREFIX}{os.getpid()}-",
        ))
        try:
            _check_cancel(cancel)
            notice = None
            if options.echo_track:
                if info.audio_tracks >= 2:
                    mic = pipeline.decode_for_clean(ff, src, info, MAX_SECONDS, work_dir=work)
                    x = pipeline.remove_echo(ff, src, info, mic, MAX_SECONDS, work_dir=work, cancel=cancel)
                    mic = None
                else:
                    notice = "only one audio track, so no speaker echo was removed"
            segments, language = [], None
            if want_subs:
                segments, language = _transcribe(
                    ff, src, info, work, options, threads, report.span(0.0, transcribe_end), cancel, echo_free=x
                )
                _check_cancel(cancel)
            has_text = any(s.text.strip() for s in segments)
            if has_text and sub_kind != "off":
                sub_tmp = work / f"subtitles{_SUBTITLE_SUFFIX[sub_kind]}"
                sub_tmp.write_text(_subtitle_text(sub_kind, segments, language), encoding="utf-8")

            if x is None:
                x = pipeline.decode_for_clean(ff, src, info, MAX_SECONDS, work_dir=work)
            report(decode_end)
            used: dict = {}
            voice_dir = resources.voice_model_dir()
            reason = advice.rebuild_reason(x) if options.auto_rebuild and not options.rebuild_voice else None
            _check_cancel(cancel)
            rebuild = (options.rebuild_voice or reason is not None) and voice_dir is not None
            y = pipeline.clean_audio(
                x,
                resources.libdf_path(),
                resources.model_path(),
                pipeline.CleanOptions(
                    atten_lim_db=options.atten_lim_db, leveling=options.leveling, loudness_lufs=options.loudness_lufs,
                    noise=options.noise, restore_highs=options.restore_highs, restore_voice=rebuild,
                ),
                threads=threads,
                progress=report.span(decode_end, clean_end),
                cancel=cancel,
                out=pipeline.output_buffer(x, work),
                heavy_model_path=resources.heavy_noise_model_path(),
                bwe_model_dir=resources.highs_model_dir(),
                restore_model_dir=voice_dir,
                report=used,
            )
            picked = None
            if options.strength == AUTO_STRENGTH and "atten_lim_db" in used:
                picked = next((k for k, db in STRENGTHS.items() if db == used["atten_lim_db"]), None)
            if reason and rebuild:
                notice = "; ".join(filter(None, [notice, f"the voice was rebuilt because it sounded {reason}"]))
            elif reason:
                notice = "; ".join(filter(None, [notice, f"it sounds {reason}: download the model that rebuilds the voice "
                                                 "in Settings and clean it again to fix that"]))
            if options.rebuild_voice and not rebuild:
                notice = "; ".join(filter(None, [notice, "the model that rebuilds the voice is not installed, so it was "
                                                 "cleaned the usual way. Download it in Settings to rebuild voices"]))
            if used.get("highs"):
                notice = "; ".join(filter(None, [notice, "missing highs restored"]))
            elif options.restore_highs and resources.highs_model_dir() is None and _sounds_band_limited(x):
                notice = "; ".join(filter(None, [notice, "sounds like a phone recording. Download the model that "
                                                 "restores highs in Settings to improve it"]))
            _check_cancel(cancel)
            kept = _write(ff, src, pipeline.as_output(y), info, options, tmp_out, sub_tmp if embed else None, language,
                          work=work, progress=report.span(clean_end, 0.99))
            if info.audio_tracks > 1:
                notice = "; ".join(filter(None, [notice, _other_tracks_notice(info.audio_tracks - 1, kept.get("audio_kept", 0))]))
            if kept.get("fallback"):
                notice = "; ".join(filter(None, [notice, "this computer's own video encoder wasn't available, so the "
                                                 "picture was saved in a simpler format and the file is bigger than usual"]))
            # Windows can't delete a file that is still mapped.
            x = y = None
            _check_cancel(cancel)
            sidecar = sub_tmp is not None and not embed
            final = claim_output(dest_dir, out_stem, ext, _SUBTITLE_SUFFIX[sub_kind] if sidecar else "", suffix="")
            sub_final = final.with_suffix(_SUBTITLE_SUFFIX[sub_kind]) if sidecar else None
            claimed = [final, *([sub_final] if sub_final else [])]
            os.replace(tmp_out, final)
            if sub_final is not None:
                os.replace(sub_tmp, sub_final)
            claimed = []
        except BaseException as e:
            # A cancelled or failed job's traceback still holds the memory-mapped scratch arrays
            # (in clean_audio's frames); Windows can't delete a file that is still mapped.
            x = y = None
            traceback.clear_frames(e.__traceback__)
            raise
        finally:
            x = y = None
            # Usually gone at once. On Windows a memory-mapped file can stay locked until the garbage collector lets
            # go of it: then the retries run in the background, so a stopped job is reported (and restartable) at
            # once. What a crash leaves behind is swept at the next start.
            try:
                shutil.rmtree(work)
            except FileNotFoundError:
                pass
            except OSError:
                threading.Thread(target=_remove_work_dir, args=(work,), name="scratch-cleanup", daemon=True).start()
    finally:
        tmp_out.unlink(missing_ok=True)
        for placeholder in claimed:  # a name taken but never filled (the job failed in between)
            try:
                if placeholder.stat().st_size == 0:
                    placeholder.unlink()
            except OSError:
                pass
    report(1.0)
    return JobResult(
        final, info.duration, info.has_video, sub_final, embedded=embed and sub_tmp is not None, language=language,
        notice=notice, strength=picked, rebuilt=reason if rebuild else None,
    )


def _sounds_band_limited(x: np.ndarray) -> bool:
    from acw_common import bwe

    return bwe.pick_model(bwe.effective_bandwidth(x)) is not None  # read block by block, however long x is


def _remove_work_dir(work: Path) -> None:
    """Delete the scratch folder. On Windows a mapping can outlive its last Python reference for a
    moment (until the garbage collector runs), so retry briefly; never mask the job's own error."""
    for attempt in range(5):
        try:
            shutil.rmtree(work)
            return
        except FileNotFoundError:
            return
        except OSError:
            gc.collect()
            time.sleep(0.1 * (attempt + 1))
    shutil.rmtree(work, ignore_errors=True)
    if work.exists():
        log.warning("could not remove the scratch folder %s", work)


def _other_tracks_notice(others: int, kept: int) -> str:
    """Only the first audio track is cleaned. The others come along unchanged in a video, when the file type can hold them."""
    many = "the other one" if others == 1 else f"the other {others}"
    if kept >= others:
        return f"only the first audio track was cleaned; {many} {'was' if others == 1 else 'were'} kept as it was"
    if kept == 0:
        return f"only the first audio track was cleaned; {many} {'wasn' if others == 1 else 'weren'}'t kept in the result"
    return f"only the first audio track was cleaned; {kept} of the other {others} were kept as they were, the rest couldn't be kept in this file type"


def _write(
    ff: FFmpeg, src: Path, audio: np.ndarray, info: ProbeResult, options: JobOptions, target: Path,
    subtitles: Path | None, language: str | None, *, work: Path, progress: Progress | None = None,
) -> dict:
    """Write the result (`target`'s ending says what it is). Returns what the video keeps of the source (see
    FFmpeg.mux_video's report)."""
    kept: dict = {}
    ending = target.suffix.lower()
    try:
        if ending in AUDIO_ENDINGS:  # just the sound, also of a video
            ff.write_audio(audio, pipeline.SAMPLE_RATE, target, AUDIO_ENDINGS[ending])
        else:
            # The cleaned sound is written to the job's work folder first (up to several GB for a long video).
            cleaned = ff.write_audio(audio, pipeline.SAMPLE_RATE, work / "cleaned", "flac")
            iso3 = LANGUAGES.get(language or "", ("", ""))[1] or None
            picture = VideoOut(short_side=options.short_side)
            if info.has_video:
                ff.mux_video(src, cleaned, target, subtitles=subtitles, sub_lang=iso3, keep_metadata=True, report=kept,
                             video=picture, progress=progress)
            else:  # a sound becomes a video with a black picture
                ff.make_video(cleaned, target, len(audio) / pipeline.SAMPLE_RATE, subtitles=subtitles, sub_lang=iso3,
                              report=kept, video=picture, progress=progress)
    except RuntimeError as e:
        log.error("encoding failed: %s", e)
        raise MediaError("Saving the cleaned audio failed.") from e
    return kept
