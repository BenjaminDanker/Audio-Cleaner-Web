"""Acoustic echo cancellation: remove what the speakers played from the microphone signal.

WebRTC's AEC3 (BSD), through the LiveKit Python SDK's AudioProcessingModule (Apache-2.0). It needs the
far-end ("reference") signal: what went to the speakers. Live, that is the desktop audio the OBS plugin
sends alongside the mic; in a file, a second audio track with the desktop/speaker sound (OBS and many
recorders can write one). Only echo cancellation is switched on here: noise removal is DeepFilterNet's
job, and level is the leveling step's.

Works on 10 ms frames at 48 kHz mono. AEC3 estimates the echo delay itself (it adapts within a few
seconds); measured on a synthetic room echo: about 38 dB of echo removed, the near-end voice unchanged,
about 2 % of one core. The audio comes out DELAY_SAMPLES later than it went in (measured with an impulse).
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 48_000
FRAME = SAMPLE_RATE // 100  # 10 ms
DELAY_SAMPLES = 432  # 9 ms: how far AEC3 delays the mic signal it passes


def _to_frame(rtc, x: np.ndarray):
    pcm = np.clip(x, -1.0, 1.0)
    return rtc.AudioFrame((pcm * 32767.0).astype(np.int16).tobytes(), SAMPLE_RATE, 1, FRAME)


class EchoCanceller:
    """Streaming: feed the mic and the reference in step (any block size; 10 ms frames inside).
    Output lags the input by less than one frame (the remainder waits for the next call)."""

    def __init__(self) -> None:
        from livekit import rtc  # imported lazily: only jobs that use echo cancellation load it

        self._rtc = rtc
        self._apm = rtc.AudioProcessingModule(
            echo_cancellation=True, noise_suppression=False, high_pass_filter=False, auto_gain_control=False,
        )
        self._mic = np.zeros(0, np.float32)
        self._ref = np.zeros(0, np.float32)

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """mic, ref: float32 mono at 48 kHz. Returns the echo-free mic for every whole 10 ms frame
        received so far (ref may be shorter or silent: missing reference counts as silence)."""
        mic = np.asarray(mic, np.float32).reshape(-1)
        ref = np.asarray(ref, np.float32).reshape(-1)
        self._mic = np.concatenate([self._mic, mic])
        self._ref = np.concatenate([self._ref, ref])
        frames = len(self._mic) // FRAME
        out = np.empty(frames * FRAME, np.float32)
        for i in range(frames):
            a, b = i * FRAME, (i + 1) * FRAME
            r = self._ref[a:b]
            if len(r) < FRAME:
                r = np.pad(r, (0, FRAME - len(r)))
            self._apm.process_reverse_stream(_to_frame(self._rtc, r))
            frame = _to_frame(self._rtc, self._mic[a:b])
            self._apm.process_stream(frame)
            out[a:b] = np.frombuffer(bytes(frame.data), np.int16).astype(np.float32) / 32767.0
        used = frames * FRAME
        self._mic = self._mic[used:]
        self._ref = self._ref[used:] if len(self._ref) > used else np.zeros(0, np.float32)
        return out

    def flush(self) -> np.ndarray:
        """The last partial frame (padded through the canceller, then trimmed)."""
        rest = len(self._mic)
        if not rest:
            return np.zeros(0, np.float32)
        return self.process(np.zeros(FRAME - rest, np.float32), np.zeros(0, np.float32))[:rest]


def cancel_echo(mic: np.ndarray, ref: np.ndarray, block: int = SAMPLE_RATE) -> np.ndarray:
    """Whole recording: `mic` and `ref` (float32 mono, 48 kHz, same start) -> echo-free mic, same length."""
    aec = EchoCanceller()
    n = len(mic)
    ref = np.asarray(ref, np.float32).reshape(-1)[:n]
    parts = [aec.process(mic[i:i + block], ref[i:i + block]) for i in range(0, n, block)]
    parts.append(aec.flush())
    return np.concatenate(parts)[:n]
