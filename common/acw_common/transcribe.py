"""Local transcription with faster-whisper (CTranslate2, int8 on CPU), shared by
the server's job runner and the desktop app.

Takes a model directory (CTranslate2 format, e.g. Systran/faster-whisper-small);
nothing is downloaded here. Segments come with word timings for captions.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from .captions import Segment, Word
from .dfn import Cancelled

SAMPLE_RATE = 16_000


def transcribe(
    audio_16k: np.ndarray,
    model_path: Path,
    *,
    language: str | None = None,
    threads: int = 4,
    progress=None,
    cancel: threading.Event | None = None,
) -> tuple[list[Segment], str]:
    """Returns (segments with word timings, detected language)."""
    from faster_whisper import WhisperModel

    if not Path(model_path).exists():
        raise RuntimeError(f"transcription model not found: {model_path}")
    model = WhisperModel(str(model_path), device="cpu", compute_type="int8", cpu_threads=threads, num_workers=1)
    total = len(audio_16k) / SAMPLE_RATE
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
    for s in segs_iter:  # decoding happens lazily, segment by segment
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        words = [Word(float(w.start), float(w.end), w.word.strip()) for w in (s.words or [])]
        out.append(Segment(float(s.start), float(s.end), s.text.strip(), words))
        if progress and total > 0:
            progress(min(1.0, s.end / total))
    return out, info.language
