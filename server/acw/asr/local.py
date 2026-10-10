"""On-prem transcription with faster-whisper (CTranslate2, int8 on CPU).

Models live under $ACW_MODELS_DIR/whisper/<name> and are fetched ahead of
time by `acw-admin fetch-models`; the sandboxed runner has no network.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..captions import Segment, Word

DEFAULT_MODEL = "large-v3-turbo"


def model_dir(models_root: Path, name: str = DEFAULT_MODEL) -> Path:
    return models_root / "whisper" / name


def transcribe(
    audio_16k: np.ndarray,
    models_root: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    language: str | None = None,
    threads: int = 4,
    progress=None,
) -> tuple[list[Segment], str]:
    """Returns (segments with word timings, detected language)."""
    from faster_whisper import WhisperModel

    path = model_dir(models_root, model_name)
    if not path.exists():
        raise RuntimeError(f"whisper model {model_name} is not installed")
    model = WhisperModel(str(path), device="cpu", compute_type="int8", cpu_threads=threads, num_workers=1)
    total = len(audio_16k) / 16000.0
    segs_iter, info = model.transcribe(
        audio_16k.astype(np.float32, copy=False),
        language=language,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
    )
    out: list[Segment] = []
    for s in segs_iter:
        words = [Word(float(w.start), float(w.end), w.word.strip()) for w in (s.words or [])]
        out.append(Segment(float(s.start), float(s.end), s.text.strip(), words))
        if progress and total > 0:
            progress(min(1.0, s.end / total))
    return out, info.language
