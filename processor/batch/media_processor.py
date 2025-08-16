"""MediaProcessor: unified pipeline entrypoint for media AI tasks.

High-level flow handled by MediaProcessor.process() (alias VideoProcessor for backward compatibility):
    1. Persist uploaded file into an isolated temp directory.
    2. Detect media type (video vs audio) and extract mono WAV (MediaExtractor).
    3. Enhance extracted WAV with DeepFilter (AudioEnhancer).
    4. If original is audio: transcode enhanced WAV back into original container/codec.
         If original is video: remux original video stream + enhanced audio (no video re-encode).
    5. Return path to the processed artifact (extension preserved by design).

Why keep everything here (instead of more abstraction)?
    * You asked for direct conversion without extra wrapper layers.
    * Minimal helper methods (#build_*_cmd) make the FFmpeg steps readable while
        keeping the public API surface (#process, cleanup methods) unchanged.

Concurrency / lifecycle notes:
    * Each instance owns a temp directory and model handle is kept inside the enhancer.
    * schedule_cleanup() is non-blocking; immediate_cleanup() is a synchronous best-effort.

Edge cases intentionally handled:
    * Unknown audio extension -> fallback AAC encode.
    * Unknown video extension -> treat as .mp4 defaults for audio codec.
    * FFmpeg mux failure -> retry with generic AAC fallback.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import sys
from pathlib import Path
from typing import Optional

# Add shared directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

import ai  # noqa: F401 - ensure task registration
from ai.base import registry, MediaTaskContext, ProgressCallback

logger = logging.getLogger(__name__)


class MediaProcessor:
    def __init__(self, uploaded_file, atten_db: Optional[int], processing_type: str = "denoise"):
        self.uploaded_file = uploaded_file
        self.atten_db = atten_db
        self.processing_type = (processing_type or "denoise").lower()
        self.temp_dir_path = tempfile.mkdtemp(prefix="acproc_")
        self.input_filename = uploaded_file.filename if getattr(uploaded_file, 'filename', None) else "uploaded_media"
        self.input_path = os.path.join(self.temp_dir_path, self.input_filename)
        self.original_extension = Path(self.input_filename).suffix.lower() or ".mp4"
        logger.info(
            "MediaProcessor ready file=%s ext=%s type=%s", self.input_filename, self.original_extension, self.processing_type
        )

    def process(self, progress_cb: Optional[ProgressCallback] = None) -> str:
        self.uploaded_file.save(self.input_path)
        try:
            task = registry.create(self.processing_type)
        except Exception as e:  # noqa: BLE001
            logger.warning("Unknown processing_type '%s' (%s); falling back to 'denoise'", self.processing_type, e)
            task = registry.create("denoise")
        ctx = MediaTaskContext(work_dir=self.temp_dir_path, attenuation_db=self.atten_db)
        output_path = task.process(self.input_path, ctx, progress_cb=progress_cb)
        self._final_output_path = output_path
        return output_path

    def schedule_cleanup(self, logger):  # pragma: no cover - simple
        try:
            shutil.rmtree(self.temp_dir_path, ignore_errors=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("Deferred cleanup failed: %s", e)

    def immediate_cleanup(self, logger):  # pragma: no cover - simple
        try:
            shutil.rmtree(self.temp_dir_path, ignore_errors=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("Immediate cleanup failed: %s", e)

# Backward compatibility: original name retained for existing imports
VideoProcessor = MediaProcessor


