from __future__ import annotations

import os
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Optional

from ai.base import MediaTask, MediaTaskContext, registry, ProgressCallback
from media_extractor import MediaExtractor, MediaType
from .audio_denoise_dfnet import resolve_models_root  # for SR consistency via DFNet state
from .audio_clarity_pipeline import process_file as run_clarity

logger = logging.getLogger(__name__)


class ClarityTask(MediaTask):
    kind = "audio"

    def __init__(self):
        self._extractor = None
        self._sample_rate = 48000  # default; will be refined after first extraction

    def process(self, input_path: str, ctx: MediaTaskContext, progress_cb: Optional[ProgressCallback] = None) -> str:
        # Extract and run clarity pipeline to wav
        if self._extractor is None:
            # Use DFNet's configured sample rate indirectly by inspecting models root via resolve_models_root
            # MediaExtractor requires a target SR; we set after first extraction
            # Here we'll just initialize with a common rate; extract will resample
            self._extractor = MediaExtractor(self._sample_rate)
        if progress_cb:
            try:
                import asyncio

                async def p(pct: int):
                    await progress_cb(pct)

                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(asyncio.create_task, p(10))
            except Exception:
                pass
        extraction = self._extractor.extract(input_path, ctx.work_dir)
        if progress_cb:
            try:
                import asyncio

                async def p2(pct: int):
                    await progress_cb(pct)

                asyncio.get_event_loop().call_soon_threadsafe(asyncio.create_task, p2(30))
            except Exception:
                pass
        wav_out, sr = run_clarity(extraction.extracted_wav_path, ctx.work_dir, params=ctx.extra or {})
        self._sample_rate = sr
        if extraction.media_type == MediaType.AUDIO:
            return self._finalize_audio(wav_out, extraction.original_extension, ctx)
        return self._finalize_video(wav_out, extraction.source_path, extraction.original_extension, ctx)

    # Finalization mirrors DenoiseDFNetTask but without progress mapping here
    def _finalize_audio(self, enhanced_wav: str, ext: str, ctx: MediaTaskContext) -> str:
        out_name = f"output{ext}" if ext else "output.wav"
        out_path = os.path.join(ctx.work_dir, out_name)
        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        encode_args = self._audio_encode_args(ext)
        if ext == ".wav":
            shutil.copyfile(enhanced_wav, out_path)
        else:
            cmd = [ffmpeg, "-y", "-i", enhanced_wav, *encode_args, out_path]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
            except subprocess.CalledProcessError as e:  # noqa: BLE001
                logger.error("Audio encode failed (%s) stderr=%s", ext, e.stderr)
                return enhanced_wav
        return out_path

    def _finalize_video(self, enhanced_wav: str, source_video: str, ext: str, ctx: MediaTaskContext) -> str:
        out_path = os.path.join(ctx.work_dir, f"output{ext or '.mp4'}")
        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        audio_codec, bitrate = self._video_audio_codec(ext)
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            source_video,
            "-i",
            enhanced_wav,
            "-c:v",
            "copy",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:a",
            audio_codec,
            "-b:a",
            bitrate,
            out_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        except subprocess.CalledProcessError as e:  # noqa: BLE001
            logger.error("Video mux failed (%s) stderr=%s", ext, e.stderr)
            raise RuntimeError("Video mux failed") from e
        return out_path

    @staticmethod
    def _audio_encode_args(ext: str) -> list[str]:
        mapping = {
            ".mp3": ["-c:a", "libmp3lame", "-b:a", "320k"],
            ".m4a": ["-c:a", "aac", "-b:a", "192k"],
            ".aac": ["-c:a", "aac", "-b:a", "192k"],
            ".flac": ["-c:a", "flac"],
            ".ogg": ["-c:a", "libvorbis", "-qscale:a", "5"],
            ".opus": ["-c:a", "libopus", "-b:a", "128k"],
            ".wav": ["-c:a", "pcm_s16le"],
        }
        return mapping.get(ext.lower(), ["-c:a", "aac", "-b:a", "192k"])

    @staticmethod
    def _video_audio_codec(ext: str) -> tuple[str, str]:
        video_map = {
            ".mp4": ("aac", "320k"),
            ".mov": ("aac", "320k"),
            ".mkv": ("aac", "320k"),
            ".avi": ("aac", "320k"),
            ".webm": ("libopus", "160k"),
        }
        return video_map.get(ext.lower(), ("aac", "256k"))


# Register task
registry.register("clarity", lambda: ClarityTask(), overwrite=True)
