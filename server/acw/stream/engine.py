"""Per-session live DSP: DeepFilterNet3 + stateful high-pass + peak limiter,
and the streaming English caption engine.

Everything here is frame-synchronous: `process()` takes N*480 samples and
returns N*480 samples, delayed by `latency_samples` (model delay).
"""
from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt, sosfilt_zi

from acw_common.dfn import DeepFilter

from .protocol import HOP, SAMPLE_RATE

log = logging.getLogger("acw.stream.engine")
# One live tier (30 ms). A 10 ms low-latency model was tried and dropped (see docs/research/denoise.md).
DENOISE_TIERS = {"light": "DeepFilterNet3_onnx.tar.gz"}


class LiveChain:
    def __init__(self, lib_path: Path, models_dir: Path, tier: str | None, atten_lim_db: float = 100.0):
        self.df = DeepFilter(lib_path, models_dir / DENOISE_TIERS[tier], atten_lim_db) if tier else None
        self.sos = butter(2, 80, btype="highpass", fs=SAMPLE_RATE, output="sos")
        self.zi = sosfilt_zi(self.sos) * 0.0
        self.gain = 1.0
        self.ceiling = 10 ** (-1.0 / 20)
        self.release = 10 ** (6.0 / 20 / 100)  # recover ~6 dB per second at 100 hops/s
        self.latency_samples = self.df.delay if self.df else 0

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


def fit(x: np.ndarray | None, n: int) -> np.ndarray:
    """`x` cut or zero-padded to n samples (None = silence)."""
    if x is None:
        return np.zeros(n, np.float32)
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


class OpusCodec:
    """Per-session Opus decoder (client uplink) and encoder (cleaned audio
    back). Runs only in the sandboxed stream worker: the uplink bitstream is
    untrusted."""

    def __init__(self, bitrate: int = 96_000, encoder: bool = True):
        import av

        self._av = av
        self.dec = av.CodecContext.create("opus", "r")
        self.dec.sample_rate = SAMPLE_RATE
        self.dec.layout = "mono"
        self.pts = 0
        if not encoder:  # e.g. the echo reference: decoded only
            return
        self.enc = av.CodecContext.create("libopus", "w")
        self.enc.sample_rate = SAMPLE_RATE
        self.enc.layout = "mono"
        self.enc.format = "s16"
        self.enc.bit_rate = int(min(max(bitrate, 24_000), 160_000))
        self.enc.options = {"application": "audio", "frame_duration": "20", "vbr": "on"}
        self.enc.open()

    def decode(self, pkt: bytes) -> np.ndarray:
        frames = self.dec.decode(self._av.Packet(pkt))
        if not frames:
            return np.zeros(960, np.float32)
        x = np.concatenate([f.to_ndarray().reshape(-1) for f in frames])
        if x.dtype != np.float32:
            x = x.astype(np.float32) / (32768.0 if x.dtype == np.int16 else 1.0)
        if len(x) != 960:
            raise ValueError("opus packets must be 20 ms")
        return x

    def encode(self, y: np.ndarray) -> bytes:
        pcm = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16).reshape(1, -1)
        frame = self._av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        frame.sample_rate = SAMPLE_RATE
        frame.pts = self.pts
        self.pts += pcm.shape[1]
        return b"".join(bytes(p) for p in self.enc.encode(frame))


class _StreamingRecognizer:
    """One streaming English recognizer (NeMo FastConformer transducer, 480 ms
    chunks, int8, via sherpa-onnx) shared by every session in this process.

    Session threads only queue audio; one decoder thread owns all model calls
    and decodes every session that has a chunk ready in one batch. That keeps
    the cost near 0.2 cores per talking stream (docs/research/asr.md)."""

    INPUT_RATE = 16_000
    TICK_S = 0.05

    def __init__(self, model_dir: Path, threads: int):
        import sherpa_onnx

        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(model_dir / "tokens.txt"),
            encoder=str(model_dir / "encoder.int8.onnx"),
            decoder=str(model_dir / "decoder.int8.onnx"),
            joiner=str(model_dir / "joiner.int8.onnx"),
            num_threads=threads,
            sample_rate=self.INPUT_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            provider="cpu",
            enable_endpoint_detection=True,
            # Long silence with nothing said: reset. Shorter values cut off the
            # first words, which are still in the model's pipeline.
            rule1_min_trailing_silence=5.0,
            rule2_min_trailing_silence=0.7,  # a pause after speech ends the line
            rule3_min_utterance_length=12.0,  # long monologues still get lines
        )
        self._lock = threading.Lock()
        self._captioners: set[StreamingCaptioner] = set()
        threading.Thread(target=self._loop, name="live-captions", daemon=True).start()

    def add(self, c: "StreamingCaptioner") -> None:
        with self._lock:
            self._captioners.add(c)

    def remove(self, c: "StreamingCaptioner") -> None:
        """Returns once the decoder thread no longer touches `c`."""
        with self._lock:
            self._captioners.discard(c)

    def _loop(self) -> None:
        while True:
            time.sleep(self.TICK_S)
            try:
                self._tick()
            except Exception:  # noqa: BLE001 - the shared thread must never end: it serves every session
                log.exception("caption tick failed")

    def _try(self, c: "StreamingCaptioner", what: str, fn, default=None):
        """One captioner's step. A captioner that fails is closed (its session goes on without captions)
        instead of stopping captions for everyone."""
        if c not in self._captioners:
            return default
        try:
            return fn()
        except Exception:  # noqa: BLE001
            log.exception("caption %s failed; closing that session's captions", what)
            c.closed = True
            self._captioners.discard(c)
            return default

    def _tick(self) -> None:
        with self._lock:  # held for the whole tick: add/remove wait for it
            active = list(self._captioners)
            for c in active:
                self._try(c, "audio", c._take_audio)
            ready = [c for c in active if self._try(c, "state", lambda c=c: self.rec.is_ready(c.stream), False)]
            while ready:
                try:
                    self.rec.decode_streams([c.stream for c in ready])
                except Exception:  # noqa: BLE001 - one bad stream must not stop captions for everyone
                    log.exception("caption decode failed")
                    break
                ready = [c for c in ready if self._try(c, "state", lambda c=c: self.rec.is_ready(c.stream), False)]
            for c in active:
                self._try(c, "publish", c._publish)


_recognizer: _StreamingRecognizer | None = None
_recognizer_lock = threading.Lock()


def shared_recognizer(models_dir: Path, threads: int = 2) -> _StreamingRecognizer:
    global _recognizer
    with _recognizer_lock:
        if _recognizer is None:
            _recognizer = _StreamingRecognizer(models_dir / "captions-en", threads)
        return _recognizer


_PRONOUN_I = re.compile(r"\bi(?=$|\s|'(?:m|ve|ll|d)\b)")


def _tidy(text: str) -> str:
    """The model writes lowercase without punctuation: capitalise the line and "I"."""
    text = _PRONOUN_I.sub("I", text.strip())
    return text[:1].upper() + text[1:]


class StreamingCaptioner:
    """Live English captions for one session: a partial line while someone
    talks, a final line at each pause. Fed the session's original (not
    denoised) audio, which every recognizer we tested transcribes better."""

    LANGUAGES = ("en",)

    def __init__(self, models_dir: Path, emit):
        self.engine = shared_recognizer(models_dir)
        self.stream = self.engine.rec.create_stream()
        self.emit = emit
        self._pending: list[np.ndarray] = []
        self._pending_lock = threading.Lock()
        self._fed = 0  # 16 kHz samples given to the recognizer
        self._line_start = 0  # sample where the current line began
        self._last_text = ""
        self._line = 0
        self.closed = False
        self.engine.add(self)

    def feed(self, x48: np.ndarray) -> None:
        with self._pending_lock:
            self._pending.append(x48)

    def _take_audio(self) -> None:  # decoder thread
        with self._pending_lock:
            chunks, self._pending = self._pending, []
        if chunks:
            x = resample_poly(np.concatenate(chunks), 1, 3).astype(np.float32)
            self.stream.accept_waveform(_StreamingRecognizer.INPUT_RATE, x)
            self._fed += len(x)

    def _publish(self, final: bool = False) -> None:  # decoder thread
        rec = self.engine.rec
        text = _tidy(rec.get_result(self.stream))
        endpoint = final or rec.is_endpoint(self.stream)
        rate = _StreamingRecognizer.INPUT_RATE
        if text and (endpoint or text != self._last_text) and not self.closed:
            self.emit({"type": "caption", "id": f"l{self._line}", "text": text, "final": bool(endpoint),
                       "start": round(self._line_start / rate, 3), "end": round(self._fed / rate, 3), "language": "en"})
        self._last_text = text
        if endpoint:
            rec.reset(self.stream)
            self._line_start, self._last_text = self._fed, ""
            if text:
                self._line += 1

    def flush(self, timeout: float = 3.0) -> None:
        """End of stream: decode what is left and emit it as a final line."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._pending_lock:
                if not self._pending:
                    break
            time.sleep(_StreamingRecognizer.TICK_S)
        self.engine.remove(self)
        self._take_audio()
        self.stream.input_finished()
        while self.engine.rec.is_ready(self.stream):
            self.engine.rec.decode_stream(self.stream)
        self._publish(final=True)

    def close(self) -> None:
        self.closed = True
        self.engine.remove(self)
