"""ffprobe/ffmpeg wrappers, shared by the server's sandboxed job runner and
the desktop app.

Untrusted input is read with an explicit protocol whitelist (file only) so a
crafted playlist/concat file cannot make ffmpeg open other paths or URLs.
"""
from __future__ import annotations

import json
import math
import ntpath
import os
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
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
    # Not for people: the server keeps its intermediate results as 24-bit FLAC (exact to one 24-bit step, 1.2e-7 of full scale, a third to
    # a half of the size of the 32-bit floats they are made of); the level is set by write_audio's `gain`.
    "flac24": (".flac", ["-c:a", "flac", "-compression_level", "3", "-sample_fmt", "s32", "-bits_per_raw_sample", "24"]),
    "ogg": (".ogg", ["-c:a", "libopus", "-b:a", "128k"]),  # Opus in an Ogg file with the usual .ogg ending
}

# Speech segments sent to the transcription service: 16 kHz mono mp3, 64 kbps. Measured with
# the real API on LibriSpeech (clean, and mixed with noise at 5 dB SNR): the word error rate is the same as with the
# ~160 kbps these used to be, at 40% of the bytes.
ASR_CHUNK_EXT = ".mp3"
ASR_CHUNK_ARGS = ["-c:a", "libmp3lame", "-b:a", "64k", "-ac", "1", "-ar", "16000"]
# Chunks per FFmpeg run. One run for all of them is cheaper than one process each, but every chunk's trim filter
# sees all the audio of the run, so the cost grows with the square of the count: 504 chunks (70 min of audio) took
# 83 s of CPU in one run, 17.5 s in runs of 32 (and 34.9 s with a process per chunk).
ASR_CHUNKS_PER_RUN = 32


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
    # The first picture: size as stored, pixel shape (width / height of one pixel), turn in degrees (a phone video
    # may be stored sideways and say "show it turned"), frames per second and bits per second (0 = unknown).
    video_width: int = 0
    video_height: int = 0
    video_sar: float = 1.0
    video_rotation: int = 0
    video_fps: float = 0.0
    video_bitrate: int = 0

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


def _ratio(text: object) -> float:
    """"30000/1001" or "16:9" as a number; 0 when it is not one."""
    try:
        a, b = str(text).replace(":", "/").split("/")
        return float(a) / float(b) if float(b) else 0.0
    except (ValueError, TypeError):
        return 0.0


def _picture_facts(stream: dict) -> dict:
    """The video_* fields of ProbeResult from ffprobe's description of the first picture stream."""
    rotation = 0
    try:
        rotation = int(float((stream.get("tags") or {}).get("rotate", 0)))
    except (ValueError, TypeError):
        pass
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            try:
                rotation = int(round(float(side["rotation"])))
            except (ValueError, TypeError):
                pass
    fps = _ratio(stream.get("avg_frame_rate")) or _ratio(stream.get("r_frame_rate"))
    try:
        bitrate = int(stream.get("bit_rate") or 0)
    except (ValueError, TypeError):
        bitrate = 0
    return {
        "video_width": int(stream.get("width") or 0),
        "video_height": int(stream.get("height") or 0),
        "video_sar": _ratio(stream.get("sample_aspect_ratio")) or 1.0,
        "video_rotation": rotation % 360,
        "video_fps": fps if 0 < fps < 1000 else 0.0,
        "video_bitrate": max(0, bitrate),
    }


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
            **(_picture_facts(video[0]) if video else {}),
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

    def write_audio(self, x: np.ndarray, sr: int, out: Path, fmt: str, gain: float = 1.0) -> Path:
        """Encode float32 audio to `out` (suffix set by `fmt`), streaming it to
        FFmpeg in blocks so a long (or memory-mapped) input is never copied whole. `gain` scales the samples on the way."""
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
                        part = np.ascontiguousarray(x[s:s + block], dtype=np.float32)
                        proc.stdin.write((part * np.float32(gain) if gain != 1.0 else part).tobytes())
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

    def write_audio_chunks(self, x: np.ndarray, sr: int, spans: list[tuple[int, int]], out_dir: Path, names: list[str]) -> list[Path]:
        """Encode the sample ranges `spans` of mono `x` as ASR_CHUNK files `out_dir/<name>.mp3`, ASR_CHUNKS_PER_RUN
        per FFmpeg run (each run reads only the samples of its chunks from one pipe) instead of one process per chunk."""
        if len(names) != len(spans):
            raise ValueError("one name per span")
        outs: list[Path] = []
        for i in range(0, len(spans), ASR_CHUNKS_PER_RUN):
            outs += self._encode_chunk_run(x, sr, spans[i:i + ASR_CHUNKS_PER_RUN], out_dir, names[i:i + ASR_CHUNKS_PER_RUN])
        return outs

    def _encode_chunk_run(self, x: np.ndarray, sr: int, spans: list[tuple[int, int]], out_dir: Path, names: list[str]) -> list[Path]:
        outs = [out_dir / f"{n}{ASR_CHUNK_EXT}" for n in names]
        n = len(spans)
        starts = np.concatenate([[0], np.cumsum([b - a for a, b in spans])]).tolist()
        # asplit + atrim cut the concatenated speech into the chunks; asetpts restarts each chunk at 0.
        graph = [f"[0:a]asplit={n}" + "".join(f"[s{i}]" for i in range(n)) if n > 1 else "[0:a]anull[s0]"]
        graph += [f"[s{i}]atrim=start_sample={starts[i]}:end_sample={starts[i + 1]},asetpts=N/{sr}/TB[o{i}]" for i in range(n)]
        script = out_dir / ".chunks-graph.txt"
        script.write_text(";\n".join(graph), encoding="utf-8")
        args = [self.ffmpeg, "-nostdin", "-hide_banner", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
                "-/filter_complex", path_arg(script)]
        for i, out in enumerate(outs):
            args += ["-map", f"[o{i}]", *ASR_CHUNK_ARGS, "-threads", "1", "-y", path_arg(out)]
        block = sr * 10
        try:
            with tempfile.TemporaryFile() as err:
                proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err, **_NO_WINDOW)
                with self._guard(proc):
                    try:
                        for a, b in spans:
                            for s in range(a, b, block):
                                if self._cancelled():
                                    raise Cancelled()
                                proc.stdin.write(np.ascontiguousarray(x[s:min(s + block, b)], dtype=np.float32).tobytes())
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
                    raise RuntimeError(f"chunk encode failed: {err.read()[-500:]!r}")
        finally:
            script.unlink(missing_ok=True)
        return outs

    def mux_video(
        self,
        video_src: Path,
        audio: Path | None,
        out: Path,
        subtitles: Path | None = None,
        sub_lang: str | None = None,
        keep_metadata: bool = False,
        report: dict | None = None,
        *,
        video: "VideoOut | None" = None,
        progress: Callable[[float], None] | None = None,
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
        tracks such as Sony's "rtmd" that break some players.

        Without `video` the picture is always copied (the server). With it, the picture is copied when the container
        takes it and the size stays, and re-encoded otherwise: with the operating system's H.264 encoder where
        there is one, else FFmpeg's own (see picture_encoders()). `report` then also says {"encoder": name or
        "copy", "fallback": True when the first choice failed}. `progress` gets the finished share (0..1) of
        an encode."""
        try:
            info = self.probe(video_src)
        except MediaError:
            info = None
        plan = _plan_extras(info, out.suffix.lower())
        picture = _picture_plan(info, out.suffix.lower(), video) if video is not None else None
        encoders: list[str | None] = picture["encoders"] if picture else [None]
        duration = info.duration if info is not None else 0.0
        timeout = max(7200.0, duration * 8)
        r = None
        used = 0
        for used, encoder in enumerate(encoders):
            options = {"encoder": encoder, "picture": picture}
            variants = [plan, {**plan, "audio": [], "subs": []}] if (plan["audio"] or plan["subs"]) else [plan]
            for variant in variants:
                args = self._mux_args(video_src, audio, out, subtitles, sub_lang, keep_metadata, variant, **options)
                r = self._run_encode(args, timeout, duration, progress if picture else None)
                if r.returncode == 0:
                    plan = variant
                    break
            if r is not None and r.returncode == 0:
                break
        if r is None or r.returncode != 0:
            raise MediaError(
                "Could not convert the picture of this video. Try another file type, or keep its size."
                if picture else "Could not rebuild the video with the cleaned audio."
            )
        if report is not None:
            report.update(audio_kept=len(plan["audio"]), audio_dropped=plan["total_audio"] - len(plan["audio"]),
                          subtitles_kept=len(plan["subs"]), subtitles_dropped=plan["total_subs"] - len(plan["subs"]))
            if video is not None:
                report.update(encoder=encoders[used] or "copy", fallback=used > 0)
        return out

    def make_video(
        self,
        audio: Path,
        out: Path,
        duration: float,
        subtitles: Path | None = None,
        sub_lang: str | None = None,
        report: dict | None = None,
        *,
        video: "VideoOut | None" = None,
        progress: Callable[[float], None] | None = None,
    ) -> Path:
        """A video file for a sound: a black picture (1280x720, or the size `video` asks for; one frame a second, so
        the file stays small) with `audio` as the sound and, when given, a soft subtitle track. Lets a recording
        carry subtitles. `report` gets {"encoder": name, "fallback": bool} like mux_video()."""
        ext = out.suffix.lower()
        width, height = black_size(video.short_side if video else None)
        timeout = max(7200.0, duration * 8)
        r = None
        used = 0
        encoders = picture_encoders(ext)
        for used, encoder in enumerate(encoders):
            args = self._black_args(audio, out, duration, subtitles, sub_lang, encoder, (width, height))
            r = self._run_encode(args, timeout, duration, progress)
            if r.returncode == 0:
                break
        if r is None or r.returncode != 0:
            raise MediaError("Could not make a video file for this sound.")
        if report is not None:
            report.update(encoder=encoders[used], fallback=used > 0)
        return out

    def _run_encode(self, args: list[str], timeout: float, duration: float,
                    progress: Callable[[float], None] | None) -> subprocess.CompletedProcess:
        """Run an FFmpeg command; with `progress`, FFmpeg's own progress report turns into fractions of `duration`."""
        if progress is None or duration <= 0:
            return self._run(args, timeout=timeout)
        args = [args[0], "-progress", "pipe:1", "-nostats", *args[1:]]
        with tempfile.TemporaryFile() as err:
            proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err, **_NO_WINDOW)
            limit = threading.Timer(timeout, proc.kill)
            limit.daemon = True
            limit.start()
            with self._guard(proc):
                try:
                    for line in proc.stdout:
                        key, _, value = line.decode("ascii", "replace").strip().partition("=")
                        if key in ("out_time_us", "out_time_ms") and value.lstrip("-").isdigit():
                            progress(min(1.0, max(0.0, int(value) / 1e6 / duration)))
                    proc.wait()
                finally:
                    limit.cancel()
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait()
                    proc.stdout.close()
            if self._cancelled():
                raise Cancelled()
            err.seek(0)
            return subprocess.CompletedProcess(args, proc.returncode, b"", err.read())

    def _black_args(self, audio, out, duration, subtitles, sub_lang, encoder, size) -> list[str]:
        ext = out.suffix.lower()
        acodec, new_sub = _container_codecs(ext)
        width, height = size
        args = [self.ffmpeg, "-nostdin", "-hide_banner", "-v", "error", "-i", path_arg(audio)]
        if subtitles is not None:
            args += ["-i", path_arg(subtitles)]
        # The picture comes from a filter (no input device needed): black, one frame a second, a little longer than
        # the sound; -shortest ends the file with the sound.
        args += ["-filter_complex", f"color=c=black:s={width}x{height}:r=1:d={math.ceil(duration) + 2},format=yuv420p[v]",
                 "-map", "[v]", "-map", "0:a:0", "-c:a", acodec[0], "-b:a", acodec[1]]
        if subtitles is not None:
            args += ["-map", "1:s:0", "-c:s", "srt" if ext == ".mkv" else new_sub]
            if sub_lang:
                args += ["-metadata:s:s:0", f"language={sub_lang}"]
        args += _picture_args(encoder, {"fps": 1.0, "bitrate": picture_bitrate(encoder, width, height, 1.0, floor=64_000)})
        args += ["-map_metadata", "-1", "-map_chapters", "-1", "-shortest"]
        if ext in (".mp4", ".mov", ".m4v"):
            args += ["-movflags", "+faststart"]
        args += ["-threads", str(self.threads), "-y", path_arg(out)]
        return args

    def _mux_args(self, video_src, audio, out, subtitles, sub_lang, keep_metadata, plan, encoder=None, picture=None) -> list[str]:
        ext = out.suffix.lower()
        acodec, new_sub = _container_codecs(ext)
        # +genpts: MPEG program streams (.mpg/.vob) carry packets without timestamps,
        # which other containers refuse; harmless for files that have them.
        args = [self.ffmpeg, "-nostdin", "-hide_banner", "-v", "error", *self._in(video_src), "-fflags", "+genpts", "-i", path_arg(video_src)]
        maps = ["-map", "0:v:0"]
        if audio is not None:
            args += ["-i", path_arg(audio)]
            maps += ["-map", "1:a:0", "-c:a", "copy", "-c:a:0", acodec[0], "-b:a:0", acodec[1]]
        else:
            # the file's own sound (none is fine): copied, or encoded as the container's own kind when it can't hold it as it is
            maps += ["-map", "0:a:0?", "-c:a", "copy"]
            if not plan.get("audio0_copy", True):
                maps += ["-c:a:0", acodec[0], "-b:a:0", acodec[1]]
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
        args += maps + (_picture_args(encoder, picture) if encoder else ["-c:v", "copy"])
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
        return {"audio": [], "subs": [], "total_audio": 0, "total_subs": 0, "audio0_copy": True}
    audio: list[int] = []
    # the first audio track, when it is kept as it is (no new sound): can this container hold it?
    first = info.audio_codecs[0] if info.audio_codecs else None
    audio0_copy = first is None or ext not in _COPY_AUDIO or _COPY_AUDIO[ext] is None or first in _COPY_AUDIO[ext]
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
            "total_subs": len(info.subtitle_streams), "audio0_copy": audio0_copy}


def _container_codecs(ext: str) -> tuple[list[str], str]:
    """The audio encoder (name, bitrate) a container gets for the cleaned sound, and the kind of text subtitles it holds."""
    if ext == ".webm":
        return ["libopus", "160k"], "webvtt"
    if ext == ".mkv":
        return ["aac", "256k"], "srt"
    return ["aac", "256k"], "mov_text"


# ---- the picture of a video result -------------------------------------------------------------------------------

@dataclass(frozen=True)
class VideoOut:
    """What the picture of a result should be like. `short_side`: the pixels of the shorter side; the picture is resized
    to it (plain lanczos resizing, same shape, never turned from portrait to landscape or back; bigger than the
    original only stretches it). None keeps the size. `encoders`: the picture encoders to try, best first (None: the
    app's own choice, picture_encoders()); the server names its own (the card's NVENC, x264). `fast`: VP9 at its
    real-time speed (about x264's cost) instead of the slower, smaller setting."""

    short_side: int | None = None
    encoders: tuple[str, ...] | None = None
    fast: bool = False


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def picture_size(width: int, height: int, sar: float, rotation: int, short_side: int) -> tuple[int, int]:
    """The size of a picture of `width` x `height` pixels (pixels `sar` times as wide as tall, shown turned by `rotation`
    degrees) after resizing it to a shorter side of `short_side` pixels. Both numbers are even (video encoders need that)."""
    shown_w, shown_h = width * (sar if sar > 0 else 1.0), float(height)
    if rotation % 180 == 90:
        shown_w, shown_h = shown_h, shown_w
    aspect = shown_w / shown_h if shown_h else 1.0
    if aspect >= 1:
        return _even(short_side * aspect), _even(short_side)
    return _even(short_side), _even(short_side / aspect)


def black_size(short_side: int | None) -> tuple[int, int]:
    """The size of the black picture made for a sound: 1280x720, or 16:9 at the shorter side asked for."""
    return (1280, 720) if short_side is None else picture_size(16, 9, 1.0, 0, short_side)


# What each container takes as the picture without re-encoding (a container not listed takes anything).
_COPY_VIDEO = {
    ".mp4": {"h264", "hevc", "mpeg4", "av1", "vp9", "mpeg2video", "mjpeg"},
    ".m4v": {"h264", "hevc", "mpeg4", "av1", "vp9", "mpeg2video", "mjpeg"},
    ".mov": {"h264", "hevc", "mpeg4", "prores", "mjpeg", "mpeg2video", "dnxhd"},
    ".webm": {"vp8", "vp9", "av1"},
    ".flv": {"h264", "flv1"},
}
# Containers that are the same kind of file as far as the picture goes.
_SAME_KIND = (frozenset({".mp4", ".m4v"}),)
# How many bits a second a picture of one pixel and one frame gets (the efficient encoders need fewer).
_BITS_PER_PIXEL = {"h264_mf": 0.07, "h264_videotoolbox": 0.07, "libx264": 0.07, "h264_nvenc": 0.08, "libvpx-vp9": 0.05, "mpeg4": 0.14,
                   "flv": 0.16}


def picture_encoders(ext: str) -> list[str]:
    """The picture encoders to try for a result with ending `ext`, best first. The app's FFmpeg is LGPL (no x264), so
    H.264 comes from the operating system: Media Foundation on Windows, VideoToolbox on macOS; if there is none, or it
    refuses, FFmpeg's own MPEG-4 (Sorenson H.263 for .flv) does it. WebM gets VP9."""
    if ext == ".webm":
        return ["libvpx-vp9"]
    system = {"win32": ["h264_mf"], "darwin": ["h264_videotoolbox"]}.get(sys.platform, [])
    return [*system, "flv" if ext == ".flv" else "mpeg4"]


def picture_bitrate(encoder: str, width: int, height: int, fps: float, source_bps: int = 0, source_pixels: int = 0,
                    floor: int = 250_000) -> int:
    """Bits a second for a re-encoded picture: a share of the pixels per second that suits the encoder, between `floor`
    and 80 Mbit/s, and not more than half as much again as the original had when the picture does not get bigger."""
    pixels = width * height
    target = pixels * (fps if fps > 0 else 30.0) * _BITS_PER_PIXEL.get(encoder, 0.1)
    if source_bps > 0 and source_pixels and pixels <= source_pixels:
        target = min(target, max(source_bps * 1.5, 500_000))
    return int(min(max(target, floor), 80_000_000))


def _picture_plan(info: ProbeResult | None, ext: str, video: VideoOut) -> dict | None:
    """What to do with the picture of `info` for a result ending in `ext`: None (copy it), or a dict with the encoders to
    try, the size, the frame rate and the bit rate base."""
    if info is None or not info.has_video:
        return None
    width, height = info.video_width, info.video_height
    turned = info.video_rotation % 180 == 90
    shown = (height, width) if turned else (width, height)  # as stored, after turning
    size = None
    if video.short_side and width and height:
        size = picture_size(width, height, info.video_sar, info.video_rotation, video.short_side)
        if size == shown and abs(info.video_sar - 1.0) < 0.01:
            size = None  # already that size
    same_kind = ext == info.ext or any(ext in kind and info.ext in kind for kind in _SAME_KIND)
    fits = same_kind or ext not in _COPY_VIDEO or info.video_codec in _COPY_VIDEO[ext]
    if size is None and fits:
        return None
    if size is None and (shown[0] % 2 or shown[1] % 2):
        size = (shown[0] // 2 * 2, shown[1] // 2 * 2)  # encoders want even numbers
    out = size or shown
    fps = info.video_fps or 30.0
    return {
        "encoders": list(video.encoders) if video.encoders else picture_encoders(ext), "size": size, "fps": fps, "out": out,
        "source_bps": info.video_bitrate, "source_pixels": shown[0] * shown[1], "fast": video.fast,
    }


def needs_picture_encode(info: ProbeResult | None, ext: str, video: VideoOut) -> bool:
    """Is the picture of `info` made again for a result ending in `ext` at `video`'s size (a new size, or a file type that
    can't hold it as it is)? False: it is copied."""
    return _picture_plan(info, ext, video) is not None


def _picture_args(encoder: str, picture: dict) -> list[str]:
    """FFmpeg options that re-encode the picture with `encoder` (see _picture_plan; a dict with just "fps" and "bitrate"
    works too)."""
    fps = picture.get("fps") or 30.0
    bitrate = picture.get("bitrate")
    if bitrate is None:
        width, height = picture["out"]
        bitrate = picture_bitrate(encoder, width, height, fps, picture.get("source_bps", 0), picture.get("source_pixels", 0))
    args: list[str] = []
    if picture.get("size"):
        width, height = picture["size"]
        args += ["-vf", f"scale={width}:{height}:flags=lanczos,setsar=1"]
    args += ["-c:v", encoder, "-b:v", str(bitrate), "-g", str(max(10, round(fps * 2))), "-pix_fmt", "yuv420p"]
    if encoder == "libvpx-vp9":
        args += ["-deadline", "realtime", "-cpu-used", "8"] if picture.get("fast") else ["-deadline", "good", "-cpu-used", "4"]
        args += ["-row-mt", "1"]
    elif encoder == "libx264":
        args += ["-preset", "veryfast"]
    elif encoder == "h264_nvenc":
        args += ["-preset", "p4"]
    elif encoder == "h264_videotoolbox":
        args += ["-allow_sw", "1"]  # an Intel Mac without a hardware encoder
    return args
