"""Per-session live DSP: DeepFilterNet3 + stateful high-pass + peak limiter,
and an utterance-based local caption engine.

Everything here is frame-synchronous: `process()` takes N*480 samples and
returns N*480 samples, delayed by `latency_samples` (model delay).
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt, sosfilt_zi

from ..dsp.dfn import MODEL_DELAY, DeepFilter
from .protocol import HOP, SAMPLE_RATE

log = logging.getLogger("acw.stream.engine")
DENOISE_TIERS = {"light": "DeepFilterNet3_onnx.tar.gz"}


class LiveChain:
    def __init__(self, lib_path: Path, models_dir: Path, tier: str | None, atten_lim_db: float = 100.0):
        self.df = DeepFilter(lib_path, models_dir / DENOISE_TIERS[tier], atten_lim_db) if tier else None
        self.sos = butter(2, 80, btype="highpass", fs=SAMPLE_RATE, output="sos")
        self.zi = sosfilt_zi(self.sos) * 0.0
        self.gain = 1.0
        self.ceiling = 10 ** (-1.0 / 20)
        self.release = 10 ** (6.0 / 20 / 100)  # recover ~6 dB per second at 100 hops/s
        self.latency_samples = MODEL_DELAY if self.df else 0

    def set_atten_lim(self, db: float) -> None:
        if self.df:
            self.df.set_atten_lim(min(max(db, 6.0), 100.0))

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.df is None:
            return x
        y = self.df.process(x)
        y, self.zi = sosfilt(self.sos, y, zi=self.zi)
        y = y.astype(np.float32)
        out = np.empty_like(y)
        for i in range(0, len(y), HOP):
            hop = y[i:i + HOP]
            peak = float(np.max(np.abs(hop))) if len(hop) else 0.0
            target = min(1.0, self.ceiling / peak) if peak > 0 else 1.0
            new = min(target, self.gain * self.release, 1.0)
            ramp = np.linspace(self.gain, new, len(hop), dtype=np.float32) if new < self.gain else np.full(len(hop), new, np.float32)
            out[i:i + HOP] = np.clip(hop * ramp, -self.ceiling, self.ceiling)
            self.gain = new
        return out

    def close(self) -> None:
        if self.df:
            self.df.close()


class _SharedWhisper:
    """One model shared by all sessions in this process."""

    def __init__(self, path: Path, threads: int, workers: int):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(str(path), device="cpu", compute_type="int8", cpu_threads=threads, num_workers=workers)
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="live-asr")


_whisper: _SharedWhisper | None = None
_whisper_lock = threading.Lock()


def shared_whisper(models_dir: Path, name: str, threads: int = 2, workers: int = 2) -> _SharedWhisper:
    global _whisper
    with _whisper_lock:
        if _whisper is None:
            _whisper = _SharedWhisper(models_dir / "whisper" / name, threads, workers)
        return _whisper


class UtteranceCaptioner:
    """Cuts the (denoised) stream into utterances at pauses and transcribes
    each one once. Cheap and robust; latency is roughly the utterance length
    plus ~0.5-1 s. Interim results come from the premium (OpenAI) engine."""

    SPEECH_DB = -45.0
    END_SILENCE_S = 0.5
    MAX_UTTERANCE_S = 7.0
    MIN_UTTERANCE_S = 0.4

    def __init__(self, models_dir: Path, model_name: str, language: str | None, emit):
        self.whisper = shared_whisper(models_dir, model_name)
        self.language = language
        self.emit = emit
        self.buf: list[np.ndarray] = []
        self.buf_start = 0.0
        self.silence = 0.0
        self.voiced = 0.0
        self.t = 0.0
        self.prompt = ""
        self.closed = False
        self.pending: list = []

    def feed(self, y: np.ndarray) -> None:
        dur = len(y) / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(y * y) + 1e-12))
        speaking = 20 * np.log10(rms) > self.SPEECH_DB
        if speaking:
            if not self.buf:
                self.buf_start = self.t
            self.silence = 0.0
            self.voiced += dur
            self.buf.append(y.copy())
        elif self.buf:
            self.silence += dur
            self.buf.append(y.copy())
        self.t += dur
        length = sum(len(b) for b in self.buf) / SAMPLE_RATE
        if self.buf and (self.silence >= self.END_SILENCE_S or length >= self.MAX_UTTERANCE_S):
            audio, start = np.concatenate(self.buf), self.buf_start
            voiced = self.voiced
            self.buf, self.silence, self.voiced = [], 0.0, 0.0
            if voiced >= self.MIN_UTTERANCE_S:
                self._submit(audio, start)

    def _submit(self, audio: np.ndarray, start: float) -> None:
        self.pending = [f for f in self.pending if not f.done()]
        self.pending.append(self.whisper.pool.submit(self._transcribe_safe, audio, start))

    def flush(self, timeout: float = 6.0) -> None:
        """Transcribe whatever is buffered and wait for outstanding work."""
        if self.buf and self.voiced >= self.MIN_UTTERANCE_S:
            self._submit(np.concatenate(self.buf), self.buf_start)
        self.buf, self.silence, self.voiced = [], 0.0, 0.0
        from concurrent.futures import wait

        wait(self.pending, timeout=timeout)

    def _transcribe_safe(self, audio48: np.ndarray, start: float) -> None:
        try:
            self._transcribe(audio48, start)
        except Exception:  # noqa: BLE001
            log.exception("live caption transcription failed")

    def _transcribe(self, audio48: np.ndarray, start: float) -> None:
        if self.closed:
            return
        audio16 = resample_poly(audio48, 1, 3).astype(np.float32)
        segs, info = self.whisper.model.transcribe(
            audio16, language=self.language, beam_size=1, vad_filter=False,
            condition_on_previous_text=False, initial_prompt=self.prompt[-200:] or None,
            without_timestamps=True,
        )
        text = " ".join(s.text.strip() for s in segs).strip()
        if text and not self.closed:
            self.prompt = (self.prompt + " " + text)[-400:]
            self.emit({"type": "caption", "text": text, "final": True, "start": round(start, 3),
                       "end": round(start + len(audio48) / SAMPLE_RATE, 3), "language": info.language})

    def close(self) -> None:
        self.closed = True
