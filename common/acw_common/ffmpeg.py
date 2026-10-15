"""ffprobe/ffmpeg wrappers, shared by the server's sandboxed job runner and
the desktop app.

Untrusted input is read with an explicit protocol whitelist (file only) so a
crafted playlist/concat file cannot make ffmpeg open other paths or URLs.
"""
from __future__ import annotations

import json
import ntpath
import os
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dfn import Cancelled

SAFE_INPUT = ["-protocol_whitelist", "file,pipe", "-probesize", "50M", "-analyzeduration", "50M"]
# Container formats we accept from users. Anything else is rejected after probing.
ALLOWED_FORMATS = {
    "mov,mp4,m4a,3gp,3g2,mj2", "matroska,webm", "mp3", "wav", "w64", "flac", "ogg", "aac", "aiff", "caf",
    "asf", "avi", "flv", "mpeg", "mpegts", "mxf", "dv", "rm", "au", "amr", "ac3", "eac3", "dts", "wv", "ape",
    "tta", "mpc", "mpc8",
}
VIDEO_CONTAINERS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}

AUDIO_FORMATS = {
    # name: (extension, ffmpeg args)
    "wav": (".wav", ["-c:a", "pcm_s24le"]),
    "flac": (".flac", ["-c:a", "flac", "-compression_level", "8"]),
    "mp3": (".mp3", ["-c:a", "libmp3lame", "-b:a", "256k"]),
    "m4a": (".m4a", ["-c:a", "aac", "-b:a", "256k"]),
    "opus": (".opus", ["-c:a", "libopus", "-b:a", "128k"]),
}


class MediaError(Exception):
    """A user-facing problem with the input file."""


# Container signatures: (offset, bytes) -> FFmpeg demuxer name. With `force_demuxer` only the demuxer chosen
# here ever parses an untrusted file, instead of every demuxer in the build probing its bytes first.
_MAGIC = [
    ((0, b"fLaC"), "flac"), ((0, b"OggS"), "ogg"), ((0, b"\x1aE\xdf\xa3"), "matroska"), ((4, b"ftyp"), "mov"),
    ((0, b"caff"), "caf"), ((0, b".snd"), "au"), ((0, b"wvpk"), "wv"), ((0, b"MAC "), "ape"), ((0, b"TTA1"), "tta"),
    ((0, b"MPCK"), "mpc8"), ((0, b"MP+"), "mpc"), ((0, b"FLV\x01"), "flv"), ((0, b"#!AMR"), "amr"),
    ((0, b"\x30\x26\xb2\x75\x8e\x66\xcf\x11"), "asf"), ((0, b".RMF"), "rm"), ((0, b"\x00\x00\x01\xba"), "mpeg"),
    ((0, b"\x06\x0e\x2b\x34"), "mxf"), ((0, b"riff\x2e\x91\xcf\x11"), "w64"), ((0, b"\x0b\x77"), "ac3"),
    ((0, b"\x7f\xfe\x80\x01"), "dts"),
]
# When the signature isn't found (a file with junk in front, say) the extension picks among the formats whose
# demuxers scan for a sync pattern.
_BY_EXTENSION = {
    ".mp3": "mp3", ".mp2": "mp3", ".mpa": "mp3", ".aac": "aac", ".ac3": "ac3", ".eac3": "eac3", ".dts": "dts",
    ".amr": "amr", ".mpg": "mpeg", ".mpeg": "mpeg", ".vob": "mpeg", ".ts": "mpegts", ".mts": "mpegts",
    ".m2ts": "mpegts", ".dv": "dv", ".tta": "tta", ".mpc": "mpc8", ".wv": "wv", ".ape": "ape",
}


def sniff_demuxer(path: Path) -> str:
    """The FFmpeg demuxer for the file's content (by signature, else by extension). MediaError when it is none we accept."""
    with open(path, "rb") as f:
        head = f.read(4096)
    if head[:4] in (b"RIFF", b"RF64", b"BW64"):
        kind = head[8:12]
        if kind == b"WAVE" or head[:4] != b"RIFF":
            return "wav"
        if kind == b"AVI ":
            return "avi"
    if head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"):
        return "aiff"
    if head[:3] == b"ID3" or (len(head) > 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0):
        return "aac" if head[:3] != b"ID3" and head[1] & 0xF6 == 0xF0 else "mp3"
    if len(head) > 196 and ((head[0] == 0x47 and head[188] == 0x47) or (head[4] == 0x47 and head[196] == 0x47)):
        return "mpegts"
    for (offset, magic), name in _MAGIC:
        if head[offset:offset + len(magic)] == magic:
            return name
    name = _BY_EXTENSION.get(path.suffix.lower())
    if name:
        return name
    raise MediaError("This file could not be read as audio or video.")


@dataclass
class ProbeResult:
    format_name: str
    duration: float
    audio_index: int
    audio_channels: int
    audio_codec: str
    has_video: bool
    video_codec: str | None
    ext: str
    # Stream indices of every audio track, in file order (audio_index is the first).
    audio_indices: tuple[int, ...] = ()
    # Codec names, in the same order as audio_indices; and (stream index, codec name) of each subtitle stream.
    audio_codecs: tuple[str, ...] = ()
    subtitle_streams: tuple[tuple[int, str], ...] = ()

    @property
    def audio_tracks(self) -> int:
        return max(1, len(self.audio_indices))

    def stream_index(self, track: int) -> int:
        """FFmpeg stream index of the `track`-th audio track (0 = the first)."""
        if track == 0:
            return self.audio_index
        if not 0 < track < len(self.audio_indices):
            raise MediaError("This file does not have that audio track.")
        return self.audio_indices[track]


_EXE = ".exe" if os.name == "nt" else ""
# Keep a windowed (no console) desktop app from flashing a console per call.
_NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def path_arg(path: Path | str) -> str:
    """A path as FFmpeg and FFprobe are given it. On Windows an absolute path gets the \\\\?\\ prefix, which lifts
    the 259-character limit of the ordinary path forms (the programs open files through the wide-character API)."""
    s = str(path)
    if sys.platform == "win32" and not s.startswith("\\\\?\\"):
        s = ntpath.abspath(s)  # also turns / into \ : the prefix switches path normalisation off
        s = "\\\\?\\UNC\\" + s[2:] if s.startswith("\\\\") else "\\\\?\\" + s
    return s


class FFmpeg:
    def __init__(self, bin_dir: Path, threads: int = 2, force_demuxer: bool = False, cancel: threading.Event | None = None):
        self.ffmpeg = str(bin_dir / f"ffmpeg{_EXE}")
        self.ffprobe = str(bin_dir / f"ffprobe{_EXE}")
        self.threads = threads
        self.force_demuxer = force_demuxer  # the server: untrusted uploads are parsed by one chosen demuxer only
        # Setting this event stops whatever program is running at once and raises Cancelled (dfn.Cancelled).
        self.cancel = cancel

    def _cancelled(self) -> bool:
        return self.cancel is not None and self.cancel.is_set()

    @contextmanager
    def _guard(self, proc: subprocess.Popen):
        """While `proc` runs, kill it as soon as the cancel event is set (a decode or a rebuild of a long video
        can take minutes)."""
        if self.cancel is None:
            yield
            return
        done = threading.Event()

        def watch() -> None:
            while not done.is_set():
                if self.cancel.wait(0.1):
                    if proc.poll() is None:
                        proc.kill()
                    return

        t = threading.Thread(target=watch, daemon=True)
        t.start()
        try:
            yield
        finally:
            done.set()
            t.join(1)

    def _in(self, path: Path) -> list[str]:
        """Input options for an untrusted file: the protocol whitelist, plus the demuxer when forced."""
        return [*SAFE_INPUT, *(["-f", sniff_demuxer(path)] if self.force_demuxer else [])]

    def _run(self, args: list[str], timeout: float, input: bytes | None = None) -> subprocess.CompletedProcess:
        if self.cancel is None:
            return subprocess.run(
                args, capture_output=True, timeout=timeout, input=input,
                stdin=subprocess.DEVNULL if input is None else None, **_NO_WINDOW,
            )
        proc = subprocess.Popen(
            args, stdin=subprocess.DEVNULL if input is None else subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, **_NO_WINDOW,
        )
        with self._guard(proc):
            try:
                out, err = proc.communicate(input, timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                raise
        if self._cancelled():
            raise Cancelled()
        return subprocess.CompletedProcess(args, proc.returncode, out, err)

    def probe(self, path: Path) -> ProbeResult:
        r = self._run(
            [self.ffprobe, "-v", "error", *self._in(path), "-print_format", "json", "-show_format", "-show_streams", path_arg(path)],
            timeout=60,
        )
        if r.returncode != 0:
            raise MediaError("This file could not be read as audio or video.")
        info = json.loads(r.stdout or b"{}")
        fmt = info.get("format", {})
        format_name = fmt.get("format_name", "")
        if format_name not in ALLOWED_FORMATS:
            raise MediaError(f"Unsupported file format ({format_name or 'unknown'}).")
        streams = info.get("streams", [])
        audio = [s for s in streams if s.get("codec_type") == "audio"]
        video = [
            s for s in streams
            if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")
        ]
        if not audio:
            raise MediaError("This file has no audio track.")
        a = audio[0]
        try:
            duration = float(fmt.get("duration") or a.get("duration") or 0)
        except ValueError:
            duration = 0.0
        if duration <= 0:
            raise MediaError("Could not determine the length of this file.")
        return ProbeResult(
            format_name=format_name,
            duration=duration,
            audio_index=int(a["index"]),
            audio_channels=int(a.get("channels") or 1),
            audio_codec=a.get("codec_name", ""),
            has_video=bool(video),
            video_codec=video[0].get("codec_name") if video else None,
            ext=path.suffix.lower(),
            audio_indices=tuple(int(s["index"]) for s in audio),
            audio_codecs=tuple(s.get("codec_name", "") for s in audio),
            subtitle_streams=tuple(
                (int(s["index"]), s.get("codec_name", "")) for s in streams if s.get("codec_type") == "subtitle"
            ),
        )

    def measure_duration(self, path: Path, probe: ProbeResult, max_seconds: float, track: int = 0) -> float:
        """Length of an audio track as it really decodes, in seconds (at most `max_seconds`).

        The duration in the container header is whatever the file's author wrote (a FLAC or Matroska
        header can say 10 s over hours of audio), so anything billed or limited by length must use
        max(probe.duration, this)."""
        rate = 8000  # only the length matters: decode cheaply, discard the samples
        args = [
            self.ffmpeg, "-nostdin", "-hide_banner", "-v", "error", *self._in(path), "-i", path_arg(path),
            "-map", f"0:{probe.stream_index(track)}", "-vn", "-sn", "-dn", "-t", str(max_seconds),
            "-ac", "1", "-ar", str(rate), "-f", "s16le", "-threads", str(self.threads), "pipe:1",
        ]
        size = 0
        with tempfile.TemporaryFile() as err:
            proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err, **_NO_WINDOW)
            with self._guard(proc):
                try:
                    while chunk := proc.stdout.read(1 << 20):
                        size += len(chunk)
                    proc.wait(timeout=600)
                finally:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait()
                    proc.stdout.close()
        if self._cancelled():
            raise Cancelled()
        if proc.returncode != 0:
            raise MediaError("Decoding the audio failed.")
        return size / 2 / rate

    def decode(
        self, path: Path, probe: ProbeResult, sr: int, channels: int, max_seconds: float, track: int = 0
    ) -> np.ndarray:
        """Decode an audio track (the first by default) to float32 (samples, channels).

        Reads straight into one buffer sized from the probed duration (grown if
        the stream turns out longer), so a long file is held in memory once."""
        args = [
            self.ffmpeg, "-nostdin", "-hide_banner", "-v", "error", *self._in(path), "-i", path_arg(path),
            "-map", f"0:{probe.stream_index(track)}", "-vn", "-sn", "-dn", "-t", str(max_seconds),
            "-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-threads", str(self.threads), "pipe:1",
        ]
        frame = 4 * channels
        buf = bytearray(int(min(probe.duration, max_seconds) * sr + sr) * frame)
        used = 0
        with tempfile.TemporaryFile() as err:
            proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err, **_NO_WINDOW)
            with self._guard(proc):
                try:
                    view = memoryview(buf)
                    while True:
                        if used == len(buf):
                            view.release()
                            buf.extend(bytes(len(buf) // 4 + frame * sr))
                            view = memoryview(buf)
                        got = proc.stdout.readinto(view[used:])
                        if not got:
                            break
                        used += got
                    view.release()
                    proc.wait(timeout=max_seconds + 600)
                finally:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait()
                    proc.stdout.close()
        if self._cancelled():
            raise Cancelled()
        if proc.returncode != 0:
            raise MediaError("Decoding the audio failed.")
        used -= used % frame
        if used < frame * sr // 10:
            raise MediaError("The audio track is empty.")
        del buf[used:]
        return np.frombuffer(buf, dtype=np.float32).reshape(-1, channels)

    def decode_to_file(
        self, path: Path, probe: ProbeResult, sr: int, channels: int, max_seconds: float, raw: Path, track: int = 0
    ) -> np.ndarray:
        """Like decode(), but FFmpeg writes the samples to `raw` and they come back
        memory-mapped (read-only) instead of being held in RAM."""
        args = [
            self.ffmpeg, "-nostdin", "-hide_banner", "-v", "error", *self._in(path), "-i", path_arg(path),
            "-map", f"0:{probe.stream_index(track)}", "-vn", "-sn", "-dn", "-t", str(max_seconds),
            "-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-threads", str(self.threads), "-y", path_arg(raw),
        ]
        if self._run(args, timeout=max_seconds + 600).returncode != 0:
            raise MediaError("Decoding the audio failed.")
        size = raw.stat().st_size // (4 * channels)
        if size < sr // 10:
            raise MediaError("The audio track is empty.")
        return np.memmap(raw, dtype=np.float32, mode="r", shape=(size, channels))

    def write_audio(self, x: np.ndarray, sr: int, out: Path, fmt: str) -> Path:
        """Encode float32 audio to `out` (suffix set by `fmt`), streaming it to
        FFmpeg in blocks so a long (or memory-mapped) input is never copied whole."""
        ext, codec_args = AUDIO_FORMATS[fmt]
        out = out.with_suffix(ext)
        channels = 1 if x.ndim == 1 else x.shape[1]
        args = [
            self.ffmpeg, "-nostdin", "-hide_banner", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", str(channels),
            "-i", "pipe:0", *codec_args, "-threads", str(self.threads), "-y", path_arg(out),
        ]
        block = sr * 10
        with tempfile.TemporaryFile() as err:
            proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err, **_NO_WINDOW)
            with self._guard(proc):
                try:
                    for s in range(0, len(x), block):
                        if self._cancelled():
                            raise Cancelled()
                        proc.stdin.write(np.ascontiguousarray(x[s:s + block], dtype=np.float32).tobytes())
                    proc.stdin.close()
                    proc.wait(timeout=3600)
                except (BrokenPipeError, OSError):
                    proc.kill()
                    proc.wait()
                finally:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait()
            if self._cancelled():
                raise Cancelled()
            if proc.returncode != 0:
                err.seek(0)
                raise RuntimeError(f"encode failed: {err.read()[-500:]!r}")
        return out

    def mux_video(
        self,
        video_src: Path,
        audio: Path | None,
        out: Path,
        subtitles: Path | None = None,
        sub_lang: str | None = None,
        keep_metadata: bool = False,
        report: dict | None = None,
    ) -> Path:
        """Copy the video stream; replace the first audio track with `audio` (or keep the source audio when
        None); optionally add a soft subtitle track.

        The file's other audio tracks and its own subtitles are kept as well (copied; subtitles converted where
        the container needs it) as far as the container can hold them. `report`, when given, is filled with
        {"audio_kept": n, "audio_dropped": n, "subtitles_kept": n, "subtitles_dropped": n}. If FFmpeg refuses the
        combination, the video is rebuilt without those extras rather than failing.

        Metadata and chapters are dropped unless `keep_metadata` (the server
        strips them for privacy; the desktop app keeps e.g. creation time).
        Only the first video stream is kept, which also drops camera data
        tracks such as Sony's "rtmd" that break some players."""
        try:
            info = self.probe(video_src)
        except MediaError:
            info = None
        plan = _plan_extras(info, out.suffix.lower())
        args = self._mux_args(video_src, audio, out, subtitles, sub_lang, keep_metadata, plan)
        r = self._run(args, timeout=7200)
        if r.returncode != 0 and (plan["audio"] or plan["subs"]):
            plan = {**plan, "audio": [], "subs": []}
            r = self._run(self._mux_args(video_src, audio, out, subtitles, sub_lang, keep_metadata, plan), timeout=7200)
        if r.returncode != 0:
            raise MediaError("Could not rebuild the video with the cleaned audio.")
        if report is not None:
            report.update(audio_kept=len(plan["audio"]), audio_dropped=plan["total_audio"] - len(plan["audio"]),
                          subtitles_kept=len(plan["subs"]), subtitles_dropped=plan["total_subs"] - len(plan["subs"]))
        return out

    def _mux_args(self, video_src, audio, out, subtitles, sub_lang, keep_metadata, plan) -> list[str]:
        ext = out.suffix.lower()
        if ext == ".webm":
            acodec, new_sub = ["libopus", "160k"], "webvtt"
        elif ext == ".mkv":
            acodec, new_sub = ["aac", "256k"], "srt"
        else:
            acodec, new_sub = ["aac", "256k"], "mov_text"
        # +genpts: MPEG program streams (.mpg/.vob) carry packets without timestamps,
        # which other containers refuse; harmless for files that have them.
        args = [self.ffmpeg, "-nostdin", "-hide_banner", "-v", "error", *self._in(video_src), "-fflags", "+genpts", "-i", path_arg(video_src)]
        maps = ["-map", "0:v:0"]
        if audio is not None:
            args += ["-i", path_arg(audio)]
            maps += ["-map", "1:a:0", "-c:a", "copy", "-c:a:0", acodec[0], "-b:a:0", acodec[1]]
        else:
            maps += ["-map", "0:a:0", "-c:a", "copy"]
        for index in plan["audio"]:  # the file's other audio tracks, as they were
            maps += ["-map", f"0:{index}"]
        maps += ["-disposition:a", "0", "-disposition:a:0", "default"]
        for index, _ in plan["subs"]:  # its own subtitles
            maps += ["-map", f"0:{index}"]
        kept_subs = len(plan["subs"])
        if plan["subs"] or subtitles is not None:
            # mkv keeps the file's subtitles as they are; mp4/mov/webm get text subtitles converted to their own kind
            maps += ["-c:s", "copy" if ext == ".mkv" else new_sub]
            if ext == ".mkv":  # MP4 text subtitles can't be copied into Matroska as they are
                maps += [x for n, (_, codec) in enumerate(plan["subs"]) if codec == "mov_text" for x in (f"-c:s:{n}", "srt")]
        if subtitles is not None:
            args += ["-i", path_arg(subtitles)]
            maps += ["-map", f"{2 if audio is not None else 1}:s:0"]
            if ext == ".mkv":
                maps += [f"-c:s:{kept_subs}", "srt"]
            if sub_lang:
                maps += [f"-metadata:s:s:{kept_subs}", f"language={sub_lang}"]
        args += maps + ["-c:v", "copy"]
        args += ["-map_metadata", "0"] if keep_metadata else ["-map_metadata", "-1", "-map_chapters", "-1"]
        if ext in (".mp4", ".mov", ".m4v"):
            args += ["-movflags", "+faststart"]
        args += ["-threads", str(self.threads), "-y", path_arg(out)]
        return args


# What each container can hold without re-encoding (extra audio is copied, never re-encoded).
_COPY_AUDIO = {
    ".mkv": None,  # anything
    ".webm": {"opus", "vorbis"},
    ".mp4": {"aac", "ac3", "eac3", "mp3", "mp2", "opus", "flac", "alac"},
    ".m4v": {"aac", "ac3", "eac3", "mp3", "mp2", "opus", "flac", "alac"},
    ".mov": {"aac", "ac3", "eac3", "mp3", "mp2", "alac", "pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_s16be", "pcm_s24be", "pcm_f32le"},
}
_TEXT_SUBS = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text"}


def _plan_extras(info: ProbeResult | None, ext: str) -> dict:
    """Which of the source's other audio tracks and its subtitles go into a container with extension `ext`."""
    if info is None:
        return {"audio": [], "subs": [], "total_audio": 0, "total_subs": 0}
    audio: list[int] = []
    if ext in _COPY_AUDIO:
        allowed = _COPY_AUDIO[ext]
        audio = [idx for idx, codec in zip(info.audio_indices[1:], info.audio_codecs[1:]) if allowed is None or codec in allowed]
    if ext == ".mkv":
        subs = list(info.subtitle_streams)
    elif ext in (".mp4", ".m4v", ".mov", ".webm"):
        subs = [(idx, codec) for idx, codec in info.subtitle_streams if codec in _TEXT_SUBS]
    else:
        subs = []
    return {"audio": audio, "subs": subs, "total_audio": max(0, len(info.audio_indices) - 1),
            "total_subs": len(info.subtitle_streams)}
