import os
import shutil
import subprocess
import logging
from enum import Enum
from pathlib import Path
from typing import Optional

import imageio_ffmpeg  # type: ignore

logger = logging.getLogger(__name__)

FFMPEG_TIMEOUT_S = 300

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
# Keep audio extensions in sync with api/shared/inputValidator.js allowedFileTypes
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}


class MediaType(Enum):
    VIDEO = "video"
    AUDIO = "audio"


class ExtractionResult:
    def __init__(
        self,
        media_type: 'MediaType',
        source_path: str,
        extracted_wav_path: str,
        working_video_path: Optional[str] = None,
        original_extension: Optional[str] = None,
    ):
        self.media_type = media_type
        self.source_path = source_path
        self.extracted_wav_path = extracted_wav_path
        self.working_video_path = working_video_path
        self.original_extension = original_extension or Path(source_path).suffix.lower()


class MediaExtractor:
    """Identify media type and extract / normalize audio to WAV for model input.

    Responsibilities ONLY: detection, (optional) light remux, audio decode -> wav.
    """

    def __init__(self, target_sample_rate: int):
        self.ffmpeg = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
        self.ffprobe = shutil.which("ffprobe")  # optional; fallback to extension logic
        self.target_sample_rate = target_sample_rate

    # ---- Detection ----
    def _probe_streams(self, path: str) -> tuple[bool, bool]:
        if not self.ffprobe:
            # No ffprobe available -> rely on extension fallback
            ext = Path(path).suffix.lower()
            return ext in VIDEO_EXTS, ext in AUDIO_EXTS
        try:
            # Query for a video stream
            v_cmd = [self.ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_type", "-of", "csv=p=0", path]
            a_cmd = [self.ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type", "-of", "csv=p=0", path]
            v_res = subprocess.run(v_cmd, capture_output=True, text=True, timeout=10)
            a_res = subprocess.run(a_cmd, capture_output=True, text=True, timeout=10)
            has_video = v_res.returncode == 0 and v_res.stdout.strip() == "video"
            has_audio = a_res.returncode == 0 and a_res.stdout.strip() == "audio"
            return has_video, has_audio
        except Exception as e:  # noqa: BLE001
            logger.warning("ffprobe detection failed (%s); falling back to extension", e)
            ext = Path(path).suffix.lower()
            return ext in VIDEO_EXTS, ext in AUDIO_EXTS

    def detect_media_type(self, path: str) -> 'MediaType':
        has_video, has_audio = self._probe_streams(path)
        if has_video:
            return MediaType.VIDEO
        # If no video but audio present OR extension says audio -> AUDIO
        return MediaType.AUDIO

    # ---- Core extraction ----
    def extract(self, source_path: str, temp_dir: str) -> ExtractionResult:
        media_type = self.detect_media_type(source_path)
        ext = Path(source_path).suffix.lower()
        # Always normalize to PCM 16-bit mono wav at target sample rate (model expectation)
        out_wav = os.path.join(temp_dir, "model_input.wav")
        cmd = [
            self.ffmpeg,
            "-y",
            "-i",
            source_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(self.target_sample_rate),
            "-ac",
            "1",
            out_wav,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_S)
        except subprocess.CalledProcessError as e:  # noqa: BLE001
            logger.error("Audio extraction failed: %s", e.stderr)
            raise RuntimeError("Audio extraction failed") from e
        if not os.path.exists(out_wav):
            raise RuntimeError("Expected WAV was not produced")
        return ExtractionResult(media_type, source_path, out_wav, original_extension=ext)
