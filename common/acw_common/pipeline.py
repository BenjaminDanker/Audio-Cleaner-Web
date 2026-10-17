"""The "Studio clean" chain, shared by the server's job runner and the desktop app:

    decode (48 kHz, mono or stereo) -> [echo removal with a second track] -> noise removal per channel
        -> [restore missing highs, for band-limited (phone-quality) recordings] -> clarity chain

With restore_voice (off by default) Sidon rebuilds the voice instead of the noise removal and highs steps:
it also removes room echo and restores missing highs, but it is ~30x the CPU of DeepFilterNet3 and on very
echoey recordings it can change a word (docs/research/restore-sidon.md).

Noise removal uses DeepFilterNet3, or DPDFNet-8 for recordings with heavy background noise, where it does
clearly better (DeepFilterNet3 is better on lighter noise). "auto" measures the recording and picks.

Keeping it in one place means a file cleaned on the desktop sounds exactly
like the same file cleaned on the website.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from . import bwe, clarity, dfn, dpdfnet, restore
from .aec import DELAY_SAMPLES as AEC_DELAY_SAMPLES, EchoCanceller
from .dfn import Cancelled
from .ffmpeg import FFmpeg, ProbeResult

SAMPLE_RATE = dfn.SAMPLE_RATE
MAX_CHANNELS = 2
ECHO_BLOCK = SAMPLE_RATE * 30

Progress = Callable[[float], None]

NOISE_PROFILES = ("auto", "light", "heavy")
# "auto" picks the heavy-noise model below this many dB between the speech peaks (90th percentile of
# 10 ms frame energies) and the background (10th percentile); the estimate reads about 9 dB above the
# true signal-to-noise ratio. Measured on 200 VoiceBank-DEMAND clips (SI-SDR against the clean speech):
# DPDFNet-8 was better on 97 % of clips below 14 dB and ~70 % from 14 to 19 dB, even at 19-21 dB, and
# DeepFilterNet3 better above 21 dB (docs/research/denoise-2026-10.md).
HEAVY_BELOW_DB = 19.0


@dataclass
class CleanOptions:
    # Maximum noise attenuation in dB; 100 = no limit. Lower keeps more ambience.
    atten_lim_db: float = 100.0
    # Run the clarity chain (level match, high-pass, compressor, loudness, limiter).
    polish: bool = True
    # Loudness target for the clarity chain; None keeps the level.
    loudness_lufs: float | None = -16.0
    # Which noise remover: "light" (DeepFilterNet3), "heavy" (DPDFNet-8) or "auto" (measured per recording).
    noise: str = "auto"
    # Regenerate the highs of band-limited recordings (phone calls, low-bitrate files) with AP-BWE; full-band
    # recordings are never touched. Needs the models (bwe_model_dir).
    restore_highs: bool = True
    # Rebuild the voice with Sidon (noise, room echo and missing highs in one; slow; the strength setting
    # does not apply). Needs the models (restore_model_dir).
    restore_voice: bool = False
    # Where the models that have a GPU path run: "cpu" or "cuda" (DPDFNet-8, Sidon, AP-BWE; DeepFilterNet3 is CPU only).
    # "cuda" never falls back to the CPU: without a usable card it raises.
    device: str = "cpu"


def noise_estimate_db(x: np.ndarray, block: int = SAMPLE_RATE * 60) -> float:
    """dB between speech peaks and background: the 90th minus the 10th percentile of 10 ms frame energies
    (mono mix). Read block by block, so a memory-mapped recording is never loaded at once."""
    frame = SAMPLE_RATE // 100
    energies = []
    for a in range(0, len(x), block):
        b = x[a:a + block]
        b = b.mean(axis=1) if b.ndim == 2 else b
        n = len(b) // frame
        if n:
            energies.append(10 * np.log10(np.mean(np.square(b[:n * frame].reshape(n, frame), dtype=np.float64), axis=1) + 1e-12))
    if not energies:
        return 99.0
    e = np.concatenate(energies)
    # Digital silence (a noise gate, a muted stretch) says nothing about the noise under the speech: a gated recording
    # with 10 dB SNR measured 98 dB with it counted, and went to the light model.
    e = e[e > SILENCE_DB]
    if len(e) < 10:
        return 99.0
    return float(np.percentile(e, 90) - np.percentile(e, 10))


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


def choose_noise_profile(x: np.ndarray, wanted: str = "auto") -> str:
    """"light" or "heavy" for this recording ("auto" measures it)."""
    if wanted in ("light", "heavy"):
        return wanted
    return "heavy" if noise_estimate_db(x) < HEAVY_BELOW_DB else "light"


def done_end(spans: list[tuple[int, int]], a: int) -> slice:
    """The silent stretch that ends where the span starting at `a` begins."""
    prev = 0
    for s0, s1 in spans:
        if s0 == a:
            return slice(prev, a)
        prev = s1
    return slice(a, a)


def clean_audio(
    x: np.ndarray,
    lib_path: str | Path,
    model_path: str | Path,
    opts: CleanOptions | None = None,
    *,
    threads: int = 2,
    progress: Progress | None = None,
    cancel: threading.Event | None = None,
    out: np.ndarray | None = None,
    heavy_model_path: str | Path | None = None,
    report: dict | None = None,
    bwe_model_dir: str | Path | None = None,
    restore_model_dir: str | Path | None = None,
) -> np.ndarray:
    """Clean float32 audio shaped (samples, channels) at SAMPLE_RATE.

    `progress` gets a fraction in [0, 1]; denoising is ~95% of the work.
    Setting `cancel` aborts with `Cancelled`. The result is written into `out`
    when given (same shape, float32; e.g. a memory-mapped file, so long inputs
    need little RAM), otherwise into a new array. `heavy_model_path` is the
    DPDFNet-8 model (default: dpdfnet.default_model_path()). `bwe_model_dir` holds the AP-BWE models for
    restore_highs (the desktop app downloads them on request; None skips the step). `restore_model_dir` holds
    Sidon's ONNX files, required for restore_voice. `report`, when given,
    gets {"noise": "light"|"heavy"|"rebuilt", "highs": None|"8k".."24k"}."""
    opts = opts or CleanOptions()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    channels = x.shape[1]
    if opts.restore_voice:
        if restore_model_dir is None or not restore.available(restore_model_dir):
            raise FileNotFoundError(f"voice rebuilding needs Sidon's models (not in {restore_model_dir})")
        if report is not None:
            report.update(noise="rebuilt", highs=None)
        y = restore.restore(
            x, restore_model_dir, threads=threads, cancel=cancel, out=out, device=opts.device,
            progress=(lambda f: progress(0.95 * f)) if progress else None,
        )
        y = y[:, None] if y.ndim == 1 else y
        if opts.polish:
            y = clarity.process(x, y, SAMPLE_RATE, clarity.ClarityParams(loudness_lufs=opts.loudness_lufs))
        if progress:
            progress(1.0)
        return y
    profile = choose_noise_profile(x, opts.noise)
    heavy = Path(heavy_model_path) if heavy_model_path else dpdfnet.default_model_path()
    highs = None  # the AP-BWE generator for a band-limited recording, measured on the original
    if opts.restore_highs and bwe_model_dir is not None and bwe.available(bwe_model_dir):
        highs = bwe.pick_model(bwe.effective_bandwidth(x))
    if report is not None:
        report["noise"] = profile
        report["highs"] = highs
    share = 0.8 if highs else 0.95  # of the progress bar for noise removal
    y = np.empty(x.shape, dtype=np.float32) if out is None else (out[:, None] if out.ndim == 1 else out)
    # Work only where it changes something: a channel that duplicates the other is cleaned once, a silent one not at
    # all, and digital silence isn't run through the model (it stays silent).
    plan = channel_plan(x)
    todo = sorted({c for c in plan if c is not None})
    if report is not None and todo != list(range(channels)):
        report["channels_cleaned"] = len(todo)
    for k, ch in enumerate(todo):
        spans = sound_spans(x[:, ch])
        total = sum(b - a for a, b in spans) or 1
        done = 0
        for a, b in spans:
            def prog(f: float, k: int = k, done: int = done, size: int = b - a) -> None:
                if progress:
                    progress(share * (k + (done + f * size) / total) / len(todo))

            if a > 0 or b < len(x):
                y[done_end(spans, a), ch] = 0.0  # the silence before this span
            if profile == "heavy":
                dpdfnet.denoise_array(
                    x[a:b, ch], atten_lim_db=opts.atten_lim_db, threads=threads, progress=prog, cancel=cancel,
                    out=y[a:b, ch], model_path=heavy, device=opts.device,
                )
            else:
                dfn.denoise_array(
                    x[a:b, ch], lib_path, model_path, atten_lim_db=opts.atten_lim_db, threads=threads, progress=prog,
                    cancel=cancel, out=y[a:b, ch],
                )
            done += b - a
        if spans[-1][1] < len(x):
            y[spans[-1][1]:, ch] = 0.0
    if cancel is not None and cancel.is_set():
        raise Cancelled()
    if highs:
        for ch in todo:
            bwe.extend(
                y[:, ch:ch + 1], bwe_model_dir, threads=threads, cancel=cancel, out=y[:, ch:ch + 1], model=highs,
                device=opts.device, progress=(lambda f: progress(share + (0.95 - share) * f)) if progress else None,
            )
    for c, src in enumerate(plan):
        if src is None:
            y[:, c] = 0.0
        elif src != c:
            y[:, c] = y[:, src]  # before the polish: its loudness is measured on both channels, as before
    if opts.polish:
        y = clarity.process(x, y, SAMPLE_RATE, clarity.ClarityParams(loudness_lufs=opts.loudness_lufs))
    if progress:
        progress(1.0)
    return y


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


def remove_echo(
    ff: FFmpeg,
    src: Path,
    probe: ProbeResult,
    mic: np.ndarray,
    max_seconds: float,
    work_dir: Path | None = None,
    *,
    progress: Progress | None = None,
    cancel: threading.Event | None = None,
) -> np.ndarray:
    """Take the speaker echo out of `mic` (the decoded first track, (samples, channels)) using the file's
    second audio track, which carries what the speakers played. Run it before noise removal.

    Works in blocks, so with `work_dir` (reference and result are scratch files there) RAM stays flat for
    long files. Returns an array shaped like `mic`; `mic` itself when the file has no second track."""
    if probe.audio_tracks < 2:
        return mic
    if work_dir is None:
        ref = ff.decode(src, probe, SAMPLE_RATE, 1, max_seconds, track=1)
    else:
        ref = ff.decode_to_file(src, probe, SAMPLE_RATE, 1, max_seconds, Path(work_dir) / "reference.f32", track=1)
    ref = ref[:, 0]  # a shorter reference counts as silence at the end
    n, channels = mic.shape
    out = (
        np.empty(mic.shape, dtype=np.float32) if work_dir is None
        else np.lib.format.open_memmap(Path(work_dir) / "echo-free.npy", mode="w+", dtype=np.float32, shape=mic.shape)
    )
    for ch in range(channels):
        aec = EchoCanceller()
        done = 0
        skip = AEC_DELAY_SAMPLES  # the canceller delays the signal by this much: drop it, so the result lines up in time

        def put(part: np.ndarray) -> None:
            nonlocal done, skip
            if skip:
                k = min(skip, len(part))
                part, skip = part[k:], skip - k
            part = part[:n - done]
            out[done:done + len(part), ch] = part
            done += len(part)

        for a in range(0, n, ECHO_BLOCK):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            put(aec.process(mic[a:a + ECHO_BLOCK, ch], ref[a:a + ECHO_BLOCK]))
            if progress:
                progress((ch + min(a + ECHO_BLOCK, n) / n) / channels)
        put(aec.flush())
        out[done:, ch] = 0.0  # the end is padded with silence: the length is unchanged
    return out


def to_speech_16k(x: np.ndarray) -> np.ndarray:
    """Float32 (samples, channels) at SAMPLE_RATE -> mono at 16 kHz for speech recognition, in blocks
    (with some context at each side, so the joins are clean)."""
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
    "SAMPLE_RATE", "CleanOptions", "Cancelled", "clean_audio", "decode_for_clean", "output_buffer", "as_output",
    "remove_echo", "to_speech_16k",
]


def as_output(y: np.ndarray) -> np.ndarray:
    """Mono results are written as 1-D arrays."""
    return y[:, 0] if y.ndim == 2 and y.shape[1] == 1 else y
