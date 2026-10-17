"""Local transcription, shared by the server's job runner and the desktop app.

Two engines (docs/research/asr.md):
  * Parakeet TDT 0.6B (onnx-asr, ONNX Runtime): ~4.5x faster than Whisper on
    CPU with better long-form accuracy, and fast on a CUDA GPU. v2 = English,
    v3 = 25 European languages.
  * Whisper (faster-whisper, CTranslate2 int8 on CPU): every other language,
    and language detection.
`pick_engine()` routes by language. Models are directories passed in;
nothing is downloaded here. Segments come with word timings for captions.
"""
from __future__ import annotations

import gc
import threading
from pathlib import Path

import numpy as np

from .captions import Segment, Word
from .dfn import Cancelled

SAMPLE_RATE = 16_000

# Languages Parakeet TDT 0.6B v3 transcribes (besides English, which v2 does better).
PARAKEET_V3_LANGUAGES = frozenset(
    "bg hr cs da nl et fi fr de el hu it lv lt mt pl pt ro sk sl es sv ru uk".split()
)


def pick_engine(language: str | None) -> str:
    """"parakeet-v2", "parakeet-v3" or "whisper" for a known language code;
    None (unknown) means: detect first (detect_language), then pick."""
    if language == "en":
        return "parakeet-v2"
    if language in PARAKEET_V3_LANGUAGES:
        return "parakeet-v3"
    return "whisper"


_cache_lock = threading.Lock()
_cached: dict[tuple[str, int, str], object] = {}


def load_model(model_path: Path, threads: int = 4, device: str = "cpu"):
    """The Whisper model in `model_path`, loaded once and kept (keyed by path, threads and device; only one is
    ever kept) so a long recording that is transcribed in pieces reads the 1.6 GB file once. Pass it to
    `transcribe(model=...)`. `unload()` drops it."""
    from faster_whisper import WhisperModel

    if not Path(model_path).exists():
        raise RuntimeError(f"transcription model not found: {model_path}")
    key = (str(Path(model_path).resolve()), int(threads), device)
    with _cache_lock:
        model = _cached.get(key)
        if model is None:
            _cached.clear()
            gc.collect()
            model = _cached[key] = WhisperModel(
                str(model_path), device=device, compute_type="int8", cpu_threads=threads, num_workers=1
            )
        return model


def unload() -> None:
    """Drop the cached Whisper model (about 1.3 GB of RAM)."""
    with _cache_lock:
        _cached.clear()
    gc.collect()


def speech_sample(audio_16k: np.ndarray, seconds: float = 30.0, floor_dbfs: float = -55.0) -> np.ndarray:
    """Up to `seconds` of the recording that isn't (near) silence: half-second pieces louder than `floor_dbfs`, in
    order. A long silent intro would otherwise be all the language detector hears (with 35 s of leading silence a
    German recording was detected as English). The start of the recording if nothing is loud enough."""
    win = SAMPLE_RATE // 2
    want = int(seconds * SAMPLE_RATE)
    keep: list[np.ndarray] = []
    have = 0
    for a in range(0, len(audio_16k) - win + 1, win):
        piece = audio_16k[a:a + win]
        if 20 * np.log10(float(np.sqrt(np.mean(np.square(piece, dtype=np.float64)))) + 1e-12) >= floor_dbfs:
            keep.append(piece)
            have += win
            if have >= want:
                break
    if not keep:
        return audio_16k[:want]
    return np.concatenate(keep)


# A cheap detector is trusted when it is this sure; otherwise the better one is asked (docs/research/heavy-2026-10.md).
LANGUAGE_CONFIDENCE = 0.6


def _detect(audio_16k: np.ndarray, path: Path, threads: int) -> tuple[str, float]:
    from faster_whisper import WhisperModel

    model = WhisperModel(str(path), device="cpu", compute_type="int8", cpu_threads=threads, num_workers=1)
    language, prob, _all = model.detect_language(speech_sample(audio_16k).astype(np.float32, copy=False))
    return language, float(prob)


def detect_language(audio_16k: np.ndarray, quick_model_path: Path, sure_model_path: Path, threads: int = 4) -> str:
    """Spoken language of the first 30 s of speech. A quick Whisper model (base) decides when it is at least
    LANGUAGE_CONFIDENCE sure; when it isn't (short clips, speech under other voices) a bigger one (small) does: the
    bigger model's answers at about a quarter of its CPU (docs/research/heavy-2026-10.md)."""
    language, prob = _detect(audio_16k, quick_model_path, threads)
    if prob < LANGUAGE_CONFIDENCE:
        language, _ = _detect(audio_16k, sure_model_path, threads)
    return language


def _words(tokens, starts, seg_start: float, seg_end: float) -> list[Word]:
    """SentencePiece tokens (a leading space starts a word) with start times ->
    words; a word ends where the next starts (at most 1 s later)."""
    groups: list[list] = []
    for tok, t in zip(tokens, starts):
        if tok[:1] in ("\u2581", " ") or not groups:
            groups.append([tok.lstrip("\u2581 "), float(t)])
        else:
            groups[-1][0] += tok
    out = []
    for i, (w, t) in enumerate(groups):
        nxt = groups[i + 1][1] if i + 1 < len(groups) else seg_end - seg_start
        if w:
            out.append(Word(round(seg_start + t, 3), round(seg_start + min(max(nxt, t + 0.04), t + 1.0), 3), w))
    return out


# CPU settings measured in docs/research/heavy-2026-10.md (150 s of LibriSpeech, WER 5.28 % in every setting):
# the encoder is compute bound and gains little from more than two threads (1: 0.294 CPU-s per audio second but
# twice the wall time, 2: 0.314, 4: 0.37-0.43; the old setting with 4 threads and VAD batches of 8 was 0.44-0.50).
_PARAKEET_CPU_THREADS = 2
# ONNX Runtime copies the 2.4 GB of external weights into private memory and re-packs them for its kernels (2.7 GB of
# anonymous memory per job). With prepacking off the weights are the file's own pages: shared by every concurrent
# job, evictable, 0.27 GB of private memory per job, for 9 % more CPU.
_PARAKEET_NO_PREPACK = True


def transcribe_parakeet(
    audio_16k: np.ndarray,
    model_path: Path,
    vad_path: Path,
    *,
    device: str = "cpu",
    threads: int = 4,
    progress=None,
    cancel: threading.Event | None = None,
) -> list[Segment]:
    """Parakeet TDT with Silero VAD segmentation (<=30 s pieces at pauses) and
    token timestamps. device "cuda" needs onnxruntime-gpu; it falls back to the
    CPU if CUDA can't be used."""
    import onnx_asr
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = min(threads, _PARAKEET_CPU_THREADS) if device == "cpu" else 2
    so.inter_op_num_threads = 1
    if device == "cpu" and _PARAKEET_NO_PREPACK:
        so.add_session_config_entry("session.disable_prepacking", "1")
    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        # HEURISTIC: the default EXHAUSTIVE cuDNN search re-runs for every new
        # input length (~50 s each on a GTX 960 instead of 0.2 s).
        providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"]
    model = onnx_asr.load_model("nemo-conformer-tdt", str(model_path), sess_options=so, providers=providers)
    # The Silero VAD is a tiny network run on 32 ms frames: one thread is all it can use. Left to itself it
    # sizes a thread pool from the CPU count (27 threads in the process at the peak, none of them useful).
    if device == "cpu":
        vad_so = ort.SessionOptions()
        vad_so.intra_op_num_threads = 1
        vad_so.inter_op_num_threads = 1
        vad = onnx_asr.load_vad("silero", path=str(vad_path), sess_options=vad_so, providers=["CPUExecutionProvider"])
    else:  # the GPU path is left exactly as it was
        vad = onnx_asr.load_vad("silero", path=str(vad_path))
    total = len(audio_16k) / SAMPLE_RATE
    pieces = model.with_vad(vad, max_speech_duration_s=30, min_silence_duration_ms=500,
                            batch_size=1 if device == "cpu" else 2).with_timestamps()
    out: list[Segment] = []
    for seg in pieces.recognize(audio_16k.astype(np.float32, copy=False)):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        text = seg.text.strip()
        if text:
            out.append(Segment(float(seg.start), float(seg.end), text,
                               _words(seg.tokens or [], seg.timestamps or [], float(seg.start), float(seg.end))))
        if progress and total > 0:
            progress(min(1.0, seg.end / total))
    return out


def transcribe(
    audio_16k: np.ndarray,
    model_path: Path,
    *,
    language: str | None = None,
    threads: int = 4,
    progress=None,
    cancel: threading.Event | None = None,
    model=None,
) -> tuple[list[Segment], str]:
    """Returns (segments with word timings, detected language). `model` is an already loaded Whisper model
    (`load_model`); without it the model in `model_path` is loaded for this call only."""
    from faster_whisper import WhisperModel

    if not Path(model_path).exists():
        raise RuntimeError(f"transcription model not found: {model_path}")
    if model is None:
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
