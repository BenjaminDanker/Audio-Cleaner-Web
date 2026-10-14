"""ffprobe/ffmpeg wrappers, shared by the server's sandboxed job runner and
the desktop app.

Untrusted input is read with an explicit protocol whitelist (file only) so a
crafted playlist/concat file cannot make ffmpeg open other paths or URLs.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAFE_INPUT = ["-protocol_whitelist", "file,pipe", "-probesize", "50M", "-analyzeduration", "50M"]
# Container formats we accept from users. Anything else is rejected after probing.
ALLOWED_FORMATS = {
    "mov,mp4,m4a,3gp,3g2,mj2", "matroska,webm", "mp3", "wav", "flac", "ogg", "aac", "avi", "mpegts",
    "aiff", "caf", "w64", "asf", "flv",
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


_EXE = ".exe" if os.name == "nt" else ""
# Keep a windowed (no console) desktop app from flashing a console per call.
_NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


class FFmpeg:
    def __init__(self, bin_dir: Path, threads: int = 2):
        self.ffmpeg = str(bin_dir / f"ffmpeg{_EXE}")
        self.ffprobe = str(bin_dir / f"ffprobe{_EXE}")
        self.threads = threads

    def _run(self, args: list[str], timeout: float, input: bytes | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            args, capture_output=True, timeout=timeout, input=input,
            stdin=subprocess.DEVNULL if input is None else None, **_NO_WINDOW,
        )

    def probe(self, path: Path) -> ProbeResult:
        r = self._run(
            [self.ffprobe, "-v", "error", *SAFE_INPUT, "-print_format", "json", "-show_format", "-show_streams", str(path)],
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
        )

    def decode(self, path: Path, probe: ProbeResult, sr: int, channels: int, max_seconds: float) -> np.ndarray:
        """Decode the first audio track to float32 (samples, channels)."""
        args = [
            self.ffmpeg, "-nostdin", "-hide_banner", "-v", "error", *SAFE_INPUT, "-i", str(path),
            "-map", f"0:{probe.audio_index}", "-vn", "-sn", "-dn", "-t", str(max_seconds),
            "-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-threads", str(self.threads), "pipe:1",
        ]
        r = self._run(args, timeout=max_seconds + 600)
        if r.returncode != 0:
            raise MediaError("Decoding the audio failed.")
        x = np.frombuffer(r.stdout, dtype=np.float32)
        if x.size < channels * sr // 10:
            raise MediaError("The audio track is empty.")
        return x.reshape(-1, channels).copy()

    def write_audio(self, x: np.ndarray, sr: int, out: Path, fmt: str) -> Path:
        ext, codec_args = AUDIO_FORMATS[fmt]
        out = out.with_suffix(ext)
        channels = 1 if x.ndim == 1 else x.shape[1]
        args = [
            self.ffmpeg, "-nostdin", "-hide_banner", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", str(channels),
            "-i", "pipe:0", *codec_args, "-threads", str(self.threads), "-y", str(out),
        ]
        p = self._run(args, timeout=3600, input=np.ascontiguousarray(x, dtype=np.float32).tobytes())
        if p.returncode != 0:
            raise RuntimeError(f"encode failed: {p.stderr[-500:]!r}")
        return out

    def mux_video(
        self,
        video_src: Path,
        audio: Path | None,
        out: Path,
        subtitles: Path | None = None,
        sub_lang: str | None = None,
        keep_metadata: bool = False,
    ) -> Path:
        """Copy the video stream; replace the audio with `audio` (or keep the
        source audio when None); optionally add a soft subtitle track.

        Metadata and chapters are dropped unless `keep_metadata` (the server
        strips them for privacy; the desktop app keeps e.g. creation time).
        Only the first video stream is kept, which also drops camera data
        tracks such as Sony's "rtmd" that break some players."""
        ext = out.suffix.lower()
        if ext == ".webm":
            acodec, sub_codec = ["-c:a", "libopus", "-b:a", "160k"], "webvtt"
        elif ext == ".mkv":
            acodec, sub_codec = ["-c:a", "aac", "-b:a", "256k"], "srt"
        else:
            acodec, sub_codec = ["-c:a", "aac", "-b:a", "256k"], "mov_text"
        args = [self.ffmpeg, "-nostdin", "-hide_banner", "-v", "error", *SAFE_INPUT, "-i", str(video_src)]
        maps = ["-map", "0:v:0"]
        if audio is not None:
            args += ["-i", str(audio)]
            maps += ["-map", "1:a:0", *acodec]
        else:
            maps += ["-map", "0:a:0", "-c:a", "copy"]
        if subtitles is not None:
            args += ["-i", str(subtitles)]
            maps += ["-map", f"{2 if audio is not None else 1}:s:0", "-c:s", sub_codec]
            if sub_lang:
                maps += ["-metadata:s:s:0", f"language={sub_lang}"]
        args += maps + ["-c:v", "copy"]
        args += ["-map_metadata", "0"] if keep_metadata else ["-map_metadata", "-1", "-map_chapters", "-1"]
        if ext in (".mp4", ".mov", ".m4v"):
            args += ["-movflags", "+faststart"]
        args += ["-threads", str(self.threads), "-y", str(out)]
        r = self._run(args, timeout=7200)
        if r.returncode != 0:
            raise MediaError("Could not rebuild the video with the cleaned audio.")
        return out
