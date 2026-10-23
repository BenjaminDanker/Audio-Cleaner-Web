"""The "Studio clean" chain, shared by the server's job runner and the desktop app:

    decode (48 kHz, mono or stereo) -> [echo removal with a second track] -> noise removal per channel
        -> clarity chain

With restore_voice (off by default) Sidon rebuilds the voice instead of the noise removal:
it also removes room echo and brings back high sounds that are missing (phone-quality recordings), but it is ~30x the CPU of DeepFilterNet3 and on very
echoey recordings it can change a word (docs/research/restore-sidon.md).

Noise removal uses DeepFilterNet3, or DPDFNet-8 for recordings with heavy background noise, where it does
clearly better (DeepFilterNet3 is better on lighter noise). "auto" measures the recording and picks.

Keeping it in one place means a file cleaned on the desktop sounds exactly
like the same file cleaned on the website.

The chain itself is made by editor.render from a timeline (what to do to which part; a whole-file run is the timeline with every
layer on the whole file): every model step runs on the block grid of blocks.py (30 s blocks, each with the context of the recording
around it that the step measured to need), so there is one code path and no seams. This module has what both sides need around it:
the automatic choices, the noise figure, the channel plan, decoding and the output buffers.

Strength (the attenuation limit) is a blend, in the time domain, of the noise remover's full-strength output with the
recording itself (`strength_mix`): the models always run at full strength, so a strength change never runs a model again. This is
the same arithmetic libDF does in its STFT domain and DPDFNet does in the time domain (a * noisy + (1 - a) * enhanced,
a = 10 ** (-dB / 20)); tests/test_blend.py checks it against the models' own limit.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import dfn
from .dfn import Cancelled
from .ffmpeg import FFmpeg, ProbeResult

SAMPLE_RATE = dfn.SAMPLE_RATE
MAX_CHANNELS = 2

Progress = Callable[[float], None]

NOISE_PROFILES = ("auto", "light", "heavy")
# "auto" picks the heavy-noise model below this noise figure (noise_figure_db: dB between the voice and the noise the
# light remover takes out). Calibrated on 552 mixes (12 LibriSpeech clips x 7 kinds of noise x 0-30 dB, SI-SDR of both
# removers against the clean clip): this cut picks the remover that does better on 72 % of the mixes worth cleaning
# (docs/research/recommend-2026-10.md).
HEAVY_BELOW_DB = 19.0
# The gentlest strength (attenuation limit, dB) that removes about as much as the strongest, by noise figure (the rule; where
# the full measurements of the recording are at hand the trained model picks instead, see automatic_choices): Maximum
# below the first figure, Strong below the second, Balanced below the third, Gentle from there (also for a clean
# recording). The same numbers as the website (web/src/recommend.js T) and the OBS plugin (input_advice.cpp), from the
# same calibration (docs/research/recommend-2026-10.md).
STRENGTH_STEPS = ((15.0, 100.0), (20.0, 40.0), (24.0, 24.0))
GENTLE_DB = 12.0


def strength_for(noise_db: float) -> float:
    """The attenuation limit (dB) that fits a recording with this noise figure."""
    return next((db for below, db in STRENGTH_STEPS if noise_db < below), GENTLE_DB)


def automatic_choices(noise_db: float, stats: dict | None = None) -> tuple[str, float]:
    """What "automatic" means for the noise step, in one place: (the noise remover, "light" or "heavy"; the strength in dB) for a
    recording with this noise figure. `clean_audio` (noise "auto", and atten_lim_db None) and `choose_noise_profile` both ask here, for
    the file jobs of the server and for the desktop app, so the rules for automatic choices change by changing this function.

    With `stats`, the full measurements of the recording (analysis.measure_features(x, noise_db)), the strength is the trained model's
    (reco_model.recommend): the same as the website's "Automatic best settings", the desktop app's and the API's. Without them only
    the noise figure is known, and the strength is the rule `strength_for` (calibrated on the same figure, less exact): that is
    what remains for a caller that has no recording to measure. `clean_audio` always has it and measures the rest itself when it has
    to pick a strength. The noise remover is the rule (HEAVY_BELOW_DB) either way."""
    heavy = noise_db < HEAVY_BELOW_DB
    if stats is not None:
        from . import reco_model

        rec = reco_model.recommend({**stats, "noiseDb": noise_db}, min_seconds=0.0)
        if rec is not None:
            return ("heavy" if rec.heavy_remover else "light"), reco_model.STRENGTH_DB[rec.strength]
    return ("heavy" if heavy else "light"), strength_for(noise_db)


@dataclass
class CleanOptions:
    # Maximum noise attenuation in dB; 100 = no limit. Lower keeps more ambience. None: the one that fits the recording
    # (strength_for its noise figure).
    atten_lim_db: float | None = 100.0
    # Run the clarity chain (level match, high-pass, compressor, loudness, limiter).
    leveling: bool = True
    # Remove the noise. False: the recording as it is, only leveled (Leveling on its own).
    denoise: bool = True
    # Loudness target for the clarity chain; None keeps the level.
    loudness_lufs: float | None = -16.0
    # Which noise remover: "light" (DeepFilterNet3), "heavy" (DPDFNet-8) or "auto" (measured per recording).
    noise: str = "auto"
    # Rebuild the voice with Sidon (noise, room echo and missing high sounds in one; slow; the strength setting
    # does not apply). Needs the models (restore_model_dir).
    restore_voice: bool = False
    # Where the models that have a GPU path run: "cpu" or "cuda" (DPDFNet-8, Sidon; DeepFilterNet3 is CPU only).
    # "cuda" never falls back to the CPU: without a usable card it raises.
    device: str = "cpu"


def measure_windows(n: int, total_seconds: float = 60.0, count: int = 6) -> list[tuple[int, int]]:
    """The parts of a recording of `n` samples that are measured: all of a short one, otherwise `count` stretches spread
    evenly over it, `total_seconds` together. The website measures the same ones (web/src/media/analysis.js planWindows)."""
    duration = n / SAMPLE_RATE
    if n <= 0:
        return []
    if duration <= total_seconds * 1.25:
        return [(0, n)]
    length = total_seconds / count
    out = []
    for i in range(count):
        mid = (i + 0.5) * duration / count
        start = min(duration - length, max(0.0, mid - length / 2))
        a = int(round(start * SAMPLE_RATE))
        out.append((a, min(n, a + int(round(length * SAMPLE_RATE)))))
    return out


def noise_figure_db(x: np.ndarray, lib_path: str | Path, model_path: str | Path,
                    cancel: threading.Event | None = None) -> float:
    """dB between the voice and the noise in it, as the noise remover itself hears it: DeepFilterNet3 at full strength
    runs over the measured parts of the recording (measure_windows, mono mix), and what it keeps is compared with what
    it takes out, over all of it: wind, keyboards, chatter and steady hiss alike, under the words and between them.
    99 when there is nothing to hear. The website (web/src/media/analysis.js NoiseFigure, with the same model in the
    browser) and the live engine (native/live NoiseMeter) measure the same way. Setting `cancel` raises `Cancelled`."""
    voice = removed = 0.0
    windows = measure_windows(len(x))
    for a, b in windows:
        seg = np.asarray(x[a:b], dtype=np.float32)
        seg = seg.mean(axis=1, dtype=np.float32) if seg.ndim == 2 else seg
        y = dfn.denoise_array(np.ascontiguousarray(seg), lib_path, model_path, atten_lim_db=100.0, threads=1, cancel=cancel)
        voice += float(np.sum(np.square(y, dtype=np.float64)))
        removed += float(np.sum(np.square(seg.astype(np.float64) - y)))
    if voice + removed <= 0:
        return 99.0
    return float(min(99.0, 10 * np.log10((voice + 1e-12) / (removed + 1e-12))))


# Frame energy (dBFS, 10 ms frames) below which audio is digital silence: a noise gate's output, a muted stretch.
SILENCE_DB = -90.0
SKIP_MIN_GAP_S = 1.5  # silent stretches at least this long are not run through the noise models
SKIP_CONTEXT_S = 0.5  # of the silence on each side that is: the models settle on it (measured: 53.9 dB SNR vs a full run)
# Left and right are the same recording when the difference is this far below the sum (or exactly equal): cleaned
# once. Stricter than the rebuild tier's -20 dB (restore.py).
DUAL_MONO_SIDE_DB = -40.0
DEAD_CHANNEL_PEAK = 1e-4  # a channel that never gets louder than this (-80 dBFS) carries nothing


def channel_plan(x: np.ndarray, block: int = SAMPLE_RATE * 30) -> list[int | None]:
    """Per output channel, the channel whose cleaned result it gets: [0, 1] for real stereo, [0, 0] when both sides
    carry the same recording (a mono mic saved as stereo), [0, None] / [None, 1] when one side is silent (it stays
    silent). Read block by block, so a memory-mapped recording is never loaded at once."""
    if x.ndim != 2 or x.shape[1] != 2:
        return list(range(x.shape[1] if x.ndim == 2 else 1))
    exact = True
    side = mid = 0.0
    peaks = [0.0, 0.0]
    for a in range(0, len(x), block):
        b = np.asarray(x[a:a + block], dtype=np.float64)
        exact = exact and np.array_equal(b[:, 0], b[:, 1])
        side += float(np.sum(np.square(b[:, 0] - b[:, 1])))
        mid += float(np.sum(np.square(b[:, 0] + b[:, 1])))
        peaks = [max(peaks[c], float(np.max(np.abs(b[:, c])))) if len(b) else peaks[c] for c in range(2)]
    dead = [p < DEAD_CHANNEL_PEAK for p in peaks]
    if dead[0] and dead[1]:
        return [0, 0]
    if dead[0] or dead[1]:
        return [None, 1] if dead[0] else [0, None]
    if exact or (mid > 0 and 10 * np.log10(side / mid + 1e-30) < DUAL_MONO_SIDE_DB):
        return [0, 0]
    return [0, 1]


def sound_spans(x: np.ndarray, block: int = SAMPLE_RATE * 60) -> list[tuple[int, int]]:
    """The parts of a mono recording worth running through a noise model: everything except digital-silence
    stretches of SKIP_MIN_GAP_S or more, with SKIP_CONTEXT_S of the silence kept on each side. One span covering
    everything when there is nothing (much) to skip."""
    frame = SAMPLE_RATE // 100
    silent = []
    for a in range(0, len(x), block):
        b = np.asarray(x[a:a + block], dtype=np.float64)
        n = len(b) // frame
        e = 10 * np.log10(np.mean(np.square(b[:n * frame].reshape(n, frame)), axis=1) + 1e-30) if n else np.zeros(0)
        silent.append(e < SILENCE_DB)
    quiet = np.concatenate(silent) if silent else np.zeros(0, bool)
    n_frames = len(quiet)
    min_gap = int(SKIP_MIN_GAP_S * 100)
    ctx = int(SKIP_CONTEXT_S * 100)
    spans: list[tuple[int, int]] = []
    start = 0  # frame where the current span began
    i = 0
    while i < n_frames:
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < n_frames and quiet[j]:
            j += 1
        if j - i >= min_gap:
            if i > 0:  # (a silent start has nothing before it to finish)
                spans.append((start, i + ctx))
            start = j - ctx if j < n_frames else n_frames
        i = j
    if start < n_frames:
        spans.append((start, n_frames))
    spans = [(a * frame, min(len(x), b * frame) if b < n_frames else len(x)) for a, b in spans if b > a]
    if not spans or sum(b - a for a, b in spans) > 0.9 * len(x):
        return [(0, len(x))]
    return spans


def strength_mix(atten_lim_db: float) -> float:
    """The share `a` of the recording that stays in the result for an attenuation limit in dB (0 = nothing is blended:
    100 dB or more, the full-strength output). The same arithmetic as libDF (tract.rs) and DPDFNet."""
    lim = abs(float(atten_lim_db))
    return 0.0 if lim >= 100.0 else min(float(10.0 ** (-lim / 20.0)), 1.0)


def decode_for_clean(
    ff: FFmpeg, src: Path, probe: ProbeResult, max_seconds: float, work_dir: Path | None = None
) -> np.ndarray:
    """Decode the first audio track at SAMPLE_RATE, keeping stereo (downmixing more).

    With `work_dir`, the samples go to a raw file there and come back
    memory-mapped (read-only), so RAM use stays flat for long files."""
    channels = min(probe.audio_channels, MAX_CHANNELS)
    if work_dir is None:
        return ff.decode(src, probe, SAMPLE_RATE, channels, max_seconds)
    return ff.decode_to_file(src, probe, SAMPLE_RATE, channels, max_seconds, Path(work_dir) / "decoded.f32")


def output_buffer(like: np.ndarray, work_dir: Path | None = None) -> np.ndarray:
    """A float32 array shaped like `like`: memory-mapped in `work_dir` when given."""
    if work_dir is None:
        return np.empty(like.shape, dtype=np.float32)
    return np.lib.format.open_memmap(Path(work_dir) / "cleaned.npy", mode="w+", dtype=np.float32, shape=like.shape)


def to_speech_16k(x: np.ndarray) -> np.ndarray:
    """Float32 (samples, channels) at SAMPLE_RATE -> mono at 16 kHz for speech recognition, in blocks
    (with some context at each side, so the joins are clean)."""
    from scipy.signal import resample_poly  # imported here: it takes most of a second to load

    n = len(x)
    out = np.empty(-(-n // 3), dtype=np.float32)
    block, context = SAMPLE_RATE * 30, SAMPLE_RATE  # both multiples of 3
    for a in range(0, n, block):
        lo, hi = max(0, a - context), min(n, a + block + context)
        y = resample_poly(np.asarray(x[lo:hi], dtype=np.float32).mean(axis=1), 1, 3)
        first = (a - lo) // 3
        part = y[first:first + -(-min(block, n - a) // 3)]
        out[a // 3:a // 3 + len(part)] = part
    return out


__all__ = [
    "SAMPLE_RATE", "CleanOptions", "Cancelled", "automatic_choices", "strength_mix",
    "channel_plan", "decode_for_clean", "output_buffer", "as_output", "to_speech_16k",
]


def as_output(y: np.ndarray) -> np.ndarray:
    """Mono results are written as 1-D arrays."""
    return y[:, 0] if y.ndim == 2 and y.shape[1] == 1 else y
