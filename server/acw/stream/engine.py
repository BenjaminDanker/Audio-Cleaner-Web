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
from scipy.signal import butter, firwin, sosfilt, sosfilt_zi

from acw_common.dfn import DeepFilter

from .protocol import HOP, SAMPLE_RATE

log = logging.getLogger("acw.stream.engine")
# One live tier (30 ms). A 10 ms low-latency model was tried and dropped (see docs/research/denoise.md).
DENOISE_TIERS = {"light": "DeepFilterNet3_onnx.tar.gz"}


# Digital silence (a noise gate before our filter, a muted interface) isn't run through the model: once this many
# samples of it have gone through (more than the model's 30 ms delay, so its buffers hold nothing but silence), the
# output is silence too, until sound comes back.
SILENT_PEAK = 1e-5  # -100 dBFS
SILENT_SKIP_AFTER = SAMPLE_RATE // 5  # 200 ms


class LiveChain:
    def __init__(self, lib_path: Path, models_dir: Path, tier: str | None, atten_lim_db: float = 100.0):
        self.df = DeepFilter(lib_path, models_dir / DENOISE_TIERS[tier], atten_lim_db) if tier else None
        self.sos = butter(2, 80, btype="highpass", fs=SAMPLE_RATE, output="sos")
        self.zi = sosfilt_zi(self.sos) * 0.0
        self.gain = 1.0
        self.ceiling = 10 ** (-1.0 / 20)
        self.release = 10 ** (6.0 / 20 / 100)  # recover ~6 dB per second at 100 hops/s
        self.latency_samples = self.df.delay if self.df else 0
        self.silent_run = 0  # samples of digital silence in a row
        self.skipped = 0  # samples not run through the model (for the stats)

    def set_atten_lim(self, db: float) -> None:
        if self.df:
            self.df.set_atten_lim(min(max(db, 6.0), 100.0))

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.df is None:
            return x
        if float(np.max(np.abs(x))) < SILENT_PEAK:
            self.silent_run += len(x)
            if self.silent_run > SILENT_SKIP_AFTER:
                self.skipped += len(x)
                return np.zeros(len(x), np.float32)
        else:
            self.silent_run = 0
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


_OPUS_APPLICATION_AUDIO = 2049
_OPUS_SET_BITRATE = 4002
_OPUS_SET_VBR = 4006
_opus_lib = None


def _libopus():
    """The system libopus through ctypes (None if it isn't installed). ctypes lets go of the GIL during each call, so
    the sessions of one worker encode and decode in parallel; PyAV holds the GIL for the whole encode (measured
    2026-10-16: four streams in four threads took 3x as long as one)."""
    global _opus_lib
    if _opus_lib is None:
        import ctypes
        import ctypes.util

        try:
            lib = ctypes.CDLL(ctypes.util.find_library("opus") or "libopus.so.0")
        except OSError:
            _opus_lib = False
            return None
        c_int_p = ctypes.POINTER(ctypes.c_int)
        lib.opus_decoder_create.restype = ctypes.c_void_p
        lib.opus_decoder_create.argtypes = [ctypes.c_int32, ctypes.c_int, c_int_p]
        lib.opus_decode_float.restype = ctypes.c_int
        lib.opus_decode_float.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
                                          ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
        lib.opus_decoder_destroy.argtypes = [ctypes.c_void_p]
        lib.opus_encoder_create.restype = ctypes.c_void_p
        lib.opus_encoder_create.argtypes = [ctypes.c_int32, ctypes.c_int, ctypes.c_int, c_int_p]
        lib.opus_encode_float.restype = ctypes.c_int32
        lib.opus_encode_float.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                                          ctypes.c_char_p, ctypes.c_int32]
        lib.opus_encoder_destroy.argtypes = [ctypes.c_void_p]
        lib.opus_encoder_ctl.restype = ctypes.c_int
        _opus_lib = lib
    return _opus_lib or None


class OpusCodec:
    """Per-session Opus decoder (client uplink) and encoder (cleaned audio
    back). Runs only in the sandboxed stream worker: the uplink bitstream is
    untrusted. libopus directly when it is installed, else PyAV."""

    FRAME = 960  # 20 ms at 48 kHz: the only packet size the clients send
    MAX_PACKET = 4000

    def __init__(self, bitrate: int = 96_000, encoder: bool = True):
        bitrate = int(min(max(bitrate, 24_000), 160_000))
        self._lib = _libopus()
        self.dec = self.enc = None
        if self._lib is not None:
            import ctypes

            self._c = ctypes
            err = ctypes.c_int(0)
            self.dec = self._lib.opus_decoder_create(SAMPLE_RATE, 1, ctypes.byref(err))
            if not self.dec or err.value:
                raise RuntimeError("opus decoder")
            self._pcm = np.zeros(self.FRAME, np.float32)
            if encoder:
                self.enc = self._lib.opus_encoder_create(SAMPLE_RATE, 1, _OPUS_APPLICATION_AUDIO, ctypes.byref(err))
                if not self.enc or err.value:
                    raise RuntimeError("opus encoder")
                self._lib.opus_encoder_ctl(ctypes.c_void_p(self.enc), _OPUS_SET_BITRATE, ctypes.c_int32(bitrate))
                self._lib.opus_encoder_ctl(ctypes.c_void_p(self.enc), _OPUS_SET_VBR, ctypes.c_int32(1))
                self._out = ctypes.create_string_buffer(self.MAX_PACKET)
            return
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
        self.enc.bit_rate = bitrate
        self.enc.options = {"application": "audio", "frame_duration": "20", "vbr": "on"}
        self.enc.open()

    def decode(self, pkt: bytes) -> np.ndarray:
        if self._lib is not None:
            fp = self._pcm.ctypes.data_as(self._c.POINTER(self._c.c_float))
            n = self._lib.opus_decode_float(self.dec, pkt, len(pkt), fp, self.FRAME, 0)
            if n != self.FRAME:  # an error (< 0, e.g. a packet longer than 20 ms) or a shorter packet
                raise ValueError("opus packets must be 20 ms")
            return self._pcm.copy()
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
        if self._lib is not None:
            y = np.ascontiguousarray(np.clip(y, -1.0, 1.0), np.float32)
            n = self._lib.opus_encode_float(self.enc, y.ctypes.data_as(self._c.POINTER(self._c.c_float)), len(y),
                                            self._out, self.MAX_PACKET)
            if n < 0:
                raise RuntimeError(f"opus encode failed ({n})")
            return self._out.raw[:n]
        pcm = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16).reshape(1, -1)
        frame = self._av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        frame.sample_rate = SAMPLE_RATE
        frame.pts = self.pts
        self.pts += pcm.shape[1]
        return b"".join(bytes(p) for p in self.enc.encode(frame))

    def close(self) -> None:
        if self._lib is not None:
            if self.dec:
                self._lib.opus_decoder_destroy(self.dec)
            if self.enc:
                self._lib.opus_encoder_destroy(self.enc)
            self.dec = self.enc = None

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


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


class Decimate3:
    """48 kHz -> 16 kHz for the recognizer, one batch at a time, as if the whole stream were converted in one go
    (the same filter as scipy's resample_poly(x, 1, 3)). Converting each batch on its own would put a small click at
    every batch edge, and the edges fall wherever the network timing puts them."""

    HALF = 30  # resample_poly's filter half-length for a factor of 3

    def __init__(self) -> None:
        self.h = firwin(2 * self.HALF + 1, 1 / 3, window=("kaiser", 5.0)).astype(np.float32)
        self.buf = np.zeros(self.HALF, np.float32)  # what resample_poly assumes before the first sample: silence
        self.next = self.HALF  # where in `buf` the next output sample is centred

    def __call__(self, x: np.ndarray) -> np.ndarray:
        buf = np.concatenate([self.buf, np.asarray(x, np.float32)])
        last = len(buf) - 1 - self.HALF  # the last centre whose filter window is complete
        if last < self.next:
            self.buf = buf
            return np.zeros(0, np.float32)
        centres = np.arange(self.next, last + 1, 3)
        valid = np.convolve(buf[centres[0] - self.HALF:centres[-1] + self.HALF + 1], self.h, mode="valid")
        out = valid[::3].astype(np.float32)
        keep_from = centres[-1] + 3 - self.HALF  # the next window starts here
        self.buf = buf[keep_from:]
        self.next = self.HALF
        return out


def _tidy(text: str) -> str:
    """The model writes lowercase without punctuation: capitalise the line and "I"."""
    text = _PRONOUN_I.sub("I", text.strip())
    return text[:1].upper() + text[1:]


# A caption stream is shed, not queued, when the decoder falls behind it (see StreamingCaptioner._take_audio).
MAX_BACKLOG_SAMPLES = 2 * SAMPLE_RATE  # 2 s of 48 kHz audio waiting for the decoder
KEEP_BACKLOG_SAMPLES = SAMPLE_RATE // 2  # what is kept of it: the newest half second
RECOVERED_BACKLOG_SAMPLES = SAMPLE_RATE // 2  # the incident is over once a tick finds less than this waiting
CAPTIONS_BEHIND_MESSAGE = "Captions are running behind because live is very busy, so some words are being skipped. They catch up on their own."


def shed_backlog(chunks: list[np.ndarray], keep: int) -> tuple[list[np.ndarray], int]:
    """Drop the oldest audio, keeping the newest `keep` samples (whole chunks, the oldest one trimmed). Returns (kept, dropped samples)."""
    kept: list[np.ndarray] = []
    have = 0
    for c in reversed(chunks):
        if have >= keep:
            break
        if have + len(c) > keep:
            c = c[len(c) - (keep - have):]
        kept.append(c)
        have += len(c)
    kept.reverse()
    return kept, sum(len(c) for c in chunks) - have


# Silence isn't decoded: after a pause long enough for the line to end, the audio is skipped (only counted, so the
# captions' times stay right) until someone talks again; the last moments before that are kept and decoded with it.
SPEECH_DBFS = -50.0  # quieter than this is silence (of the denoised audio when noise removal is on, else the mic's)
GATE_AFTER_S = 2.0  # quiet this long first, so the line has ended (WER on LibriSpeech with pauses: 4.4% gated vs 4.7-5.2% not)
PREROLL_FRAMES = 15  # 0.3 s of 20 ms frames kept while gated: the start of the next words


class StreamingCaptioner:
    """Live English captions for one session: a partial line while someone
    talks, a final line at each pause. Fed the session's original (not
    denoised) audio, which every recognizer we tested transcribes better."""

    LANGUAGES = ("en",)

    def __init__(self, models_dir: Path, emit):
        self.engine = shared_recognizer(models_dir)
        self.stream = self.engine.rec.create_stream()
        self.emit = emit
        self._pending: list[np.ndarray | int] = []  # audio to decode, and (ints) samples skipped as silence
        self._pending_lock = threading.Lock()
        self._to16k = Decimate3()
        self._fed = 0  # 16 kHz samples given to the recognizer (or skipped as silence)
        self._line_start = 0  # sample where the current line began
        self._last_text = ""
        self._line = 0
        self.closed = False
        self._behind = False  # one "captions_behind" warning per incident
        self._quiet = 0.0  # seconds since the last speech
        self._gated = False
        self._preroll: list[np.ndarray] = []
        self._skipped = 0  # 48 kHz samples skipped since the gate closed (not counting the preroll)
        self.skipped_total = 0
        self.engine.add(self)

    def feed(self, x48: np.ndarray, level_dbfs: float | None = None) -> None:
        """`level_dbfs`: how loud the speech in this frame is (the denoised frame's level when noise removal runs);
        by default the frame's own level."""
        if level_dbfs is None:
            level_dbfs = 20 * np.log10(float(np.sqrt(np.mean(x48 * x48))) + 1e-9)
        if level_dbfs >= SPEECH_DBFS:
            self._quiet = 0.0
            with self._pending_lock:
                if self._gated:
                    self._gated = False
                    self._pending.append(self._skipped)
                    self._pending.extend(self._preroll)
                    self._preroll, self._skipped = [], 0
                self._pending.append(x48)
            return
        self._quiet += len(x48) / SAMPLE_RATE
        if self._quiet <= GATE_AFTER_S:
            with self._pending_lock:
                self._pending.append(x48)
            return
        self._gated = True
        self._preroll.append(x48)
        if len(self._preroll) > PREROLL_FRAMES:
            n = len(self._preroll.pop(0))
            self._skipped += n
            self.skipped_total += n

    def _take_audio(self) -> None:  # decoder thread
        with self._pending_lock:
            items, self._pending = self._pending, []
        if not items:
            return
        total = sum(len(c) for c in items if not isinstance(c, int))
        if total > MAX_BACKLOG_SAMPLES:
            # The decoder is more than ~2 s behind this stream (too many streams, or the box is busy). Queueing the
            # rest would only make every caption later and grow memory: drop the oldest audio, say so once.
            kept, dropped = shed_backlog([c for c in items if not isinstance(c, int)], KEEP_BACKLOG_SAMPLES)
            items = [dropped + sum(c for c in items if isinstance(c, int))] + kept
            if not self._behind:
                self._behind = True
                self.emit({"type": "warning", "code": "captions_behind", "message": CAPTIONS_BEHIND_MESSAGE})
        elif total < RECOVERED_BACKLOG_SAMPLES:
            self._behind = False
        run: list[np.ndarray] = []
        for c in items + [0]:
            if isinstance(c, int):
                if run:
                    x = self._to16k(np.concatenate(run))
                    self.stream.accept_waveform(_StreamingRecognizer.INPUT_RATE, x)
                    self._fed += len(x)
                    run = []
                self._fed += c // 3  # skipped: the captions' times keep following the talk
            else:
                run.append(c)

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
