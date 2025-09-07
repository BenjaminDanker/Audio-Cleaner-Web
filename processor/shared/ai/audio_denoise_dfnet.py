"""DeepFilterNet3 denoise task (wraps existing AudioEnhancer flow).

Performance notes:
    * Loading DeepFilterNet weights is moderately expensive; we keep a module-level
        singleton of the underlying enhancer + extractor and have the registry
        return the same task instance each time (see registration at bottom).
    * Progress callbacks are best-effort when the task is executed in a worker
        thread (run_in_executor). We attempt to schedule onto the main loop but if
        no loop is accessible, progress updates are silently skipped.
"""
from __future__ import annotations
from typing import Optional, Tuple
import os
import shutil
import subprocess
import logging
from pathlib import Path

from ai.base import MediaTask, MediaTaskContext, registry, ProgressCallback
from df.enhance import enhance, init_df, load_audio, save_audio  # type: ignore
# Note: do NOT import audio_clarity_pipeline at module level to avoid circular imports.

logger = logging.getLogger(__name__)


_ENHANCER = None
_EXTRACTOR = None

# ---- DeepFilterNet model wrapper (merged formerly separate module) ----
DEFAULT_ATTEN_DB = 30
VALID_MIN_DB = -10
VALID_MAX_DB = 80


class DeepFilterNetEnhancer:
    """Wrap DeepFilterNet model lifecycle + enhancement utilities.

    Previously lived in ai.deepfilternet; merged here to keep model + task together.
    """

    def __init__(self, models_root: str):
        self.models_root = models_root
        logger.info("Initializing DeepFilterNet model from %s", models_root)
        self.model, self.df_state, _ = init_df(models_root, post_filter=True)

    def clamp_atten(self, atten_db: Optional[int]) -> Optional[int]:
        if atten_db is None:
            return None
        try:
            val = int(atten_db)
        except (ValueError, TypeError):  # noqa: PERF203
            val = DEFAULT_ATTEN_DB
        if val < VALID_MIN_DB:
            val = VALID_MIN_DB
        if val > VALID_MAX_DB:
            val = VALID_MAX_DB
        return val

    def enhance_file(self, in_wav_path: str, atten_db: Optional[int], out_wav_path: str) -> Tuple[str, int]:
        atten = self.clamp_atten(atten_db)
        audio, _ = load_audio(in_wav_path, sr=self.df_state.sr())
        enhanced = enhance(self.model, self.df_state, audio, atten_lim_db=atten)
        save_audio(out_wav_path, enhanced, sr=self.df_state.sr())
        return out_wav_path, self.df_state.sr()

    @property
    def sample_rate(self) -> int:  # noqa: D401 - simple forwarder
        return self.df_state.sr()


def resolve_models_root(relative: str = "models/DeepFilterNet3") -> str:
    base_dir = Path(__file__).parent.parent.parent  # processor/src/ai -> processor/src -> processor -> /app
    root = getattr(__import__('sys'), "_MEIPASS", str(base_dir))  # type: ignore[attr-defined]
    return os.path.join(root, relative)


def _get_enhancer():
    """Return singleton DeepFilterNet enhancer without requiring media_extractor.

    Used by streaming path to avoid importing batch-only utilities.
    """
    global _ENHANCER  # noqa: PLW0603
    if _ENHANCER is None:
        model_root = resolve_models_root()
        _ENHANCER = DeepFilterNetEnhancer(model_root)
    return _ENHANCER


def _get_enhancer_and_extractor():
    """Return enhancer + extractor, importing media_extractor lazily.

    Keeps module import side-effect free so streaming doesn't need batch/ on path.
    """
    global _ENHANCER, _EXTRACTOR  # noqa: PLW0603
    if _ENHANCER is None:
        _ENHANCER = _get_enhancer()
    if _EXTRACTOR is None:
        try:
            from media_extractor import MediaExtractor  # type: ignore
        except ImportError:
            from processor.batch.media_extractor import MediaExtractor
        _EXTRACTOR = MediaExtractor(_ENHANCER.sample_rate)
    return _ENHANCER, _EXTRACTOR


def _schedule_progress(cb: Optional[ProgressCallback], pct: int):
    if not cb:
        return
    try:  # Best-effort scheduling; tolerate worker thread context.
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(asyncio.create_task, cb(pct))
        except RuntimeError:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(asyncio.create_task, cb(pct))
    except Exception:  # noqa: BLE001
        pass


class DenoiseDFNetTask(MediaTask):
    kind = "audio"

    def __init__(self):
        # Heavy objects obtained lazily when first process() runs.
        self._enhancer = None
        self._extractor = None

    def process(self, input_path: str, ctx: MediaTaskContext, progress_cb: Optional[ProgressCallback] = None) -> str:
        if self._enhancer is None or self._extractor is None:
            self._enhancer, self._extractor = _get_enhancer_and_extractor()
        _schedule_progress(progress_cb, 10)
        # Detect media type first, then run the full clarity pipeline on the ORIGINAL input
        # (clarity_process_file performs its own extraction and returns a processed WAV)
        extraction = self._extractor.extract(input_path, ctx.work_dir)
        _schedule_progress(progress_cb, 30)
        try:
            from .audio_clarity_pipeline import process_file as clarity_process_file
            clarity_wav, _sr = clarity_process_file(input_path, ctx.work_dir, params={"denoise_atten_db": ctx.attenuation_db})
        except Exception:
            # Fallback to simple DFNet enhancement if clarity chain fails for any reason
            enhanced_wav = os.path.join(ctx.work_dir, "enhanced.wav")
            self._enhancer.enhance_file(extraction.extracted_wav_path, ctx.attenuation_db, enhanced_wav)
            clarity_wav = enhanced_wav
        try:
            from media_extractor import MediaType  # type: ignore
        except Exception:
            MediaType = None  # type: ignore
        if MediaType is None or extraction.media_type == MediaType.AUDIO:
            return self._finalize_audio(clarity_wav, extraction.original_extension, ctx, progress_cb)
        return self._finalize_video(clarity_wav, input_path, extraction.original_extension, ctx, progress_cb)

    # ---- Internal finalize helpers ----
    def _finalize_audio(
        self,
        enhanced_wav: str,
        ext: str,
        ctx: MediaTaskContext,
        progress_cb: Optional[ProgressCallback],
    ) -> str:
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
        _schedule_progress(progress_cb, 85)
        return out_path

    def _finalize_video(
        self,
        enhanced_wav: str,
        source_video: str,
        ext: str,
        ctx: MediaTaskContext,
        progress_cb: Optional[ProgressCallback],
    ) -> str:
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
        _schedule_progress(progress_cb, 85)
        return out_path

    # ---- Static codec helpers ----
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


_SINGLETON_DENOISE_TASK = DenoiseDFNetTask()
registry.register("denoise", lambda: _SINGLETON_DENOISE_TASK, overwrite=True)
