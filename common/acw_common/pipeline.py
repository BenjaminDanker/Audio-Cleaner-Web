"""The "Studio clean" chain, shared by the server's job runner and the desktop app:

    decode (48 kHz, mono or stereo) -> [echo removal with a second track] -> noise removal per channel
        -> [restore missing highs, for band-limited (phone-quality) recordings] -> clarity chain

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

from . import bwe, clarity, dfn, dpdfnet
from .aec import EchoCanceller
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
    return float(np.percentile(e, 90) - np.percentile(e, 10))


def choose_noise_profile(x: np.ndarray, wanted: str = "auto") -> str:
    """"light" or "heavy" for this recording ("auto" measures it)."""
    if wanted in ("light", "heavy"):
        return wanted
    return "heavy" if noise_estimate_db(x) < HEAVY_BELOW_DB else "light"


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
) -> np.ndarray:
    """Clean float32 audio shaped (samples, channels) at SAMPLE_RATE.

    `progress` gets a fraction in [0, 1]; denoising is ~95% of the work.
    Setting `cancel` aborts with `Cancelled`. The result is written into `out`
    when given (same shape, float32; e.g. a memory-mapped file, so long inputs
    need little RAM), otherwise into a new array. `heavy_model_path` is the
    DPDFNet-8 model (default: dpdfnet.default_model_path()); without it heavy noise
    falls back to DeepFilterNet3. `bwe_model_dir` holds the AP-BWE models for restore_highs (skipped
    without them). `report`, when given, gets {"noise": "light"|"heavy", "highs": None|"8k".."24k"}."""
    opts = opts or CleanOptions()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    channels = x.shape[1]
    profile = choose_noise_profile(x, opts.noise)
    heavy = Path(heavy_model_path) if heavy_model_path else dpdfnet.default_model_path()
    if profile == "heavy" and not heavy.is_file():
        profile = "light"
    highs = None  # the AP-BWE generator for a band-limited recording, measured on the original
    if opts.restore_highs and bwe_model_dir is not None and bwe.available(bwe_model_dir):
        highs = bwe.pick_model(bwe.effective_bandwidth(x[:, 0] if channels == 1 else x.mean(axis=1)))
    if report is not None:
        report["noise"] = profile
        report["highs"] = highs
    share = 0.8 if highs else 0.95  # of the progress bar for noise removal
    y = np.empty(x.shape, dtype=np.float32) if out is None else (out[:, None] if out.ndim == 1 else out)
    for ch in range(channels):
        def prog(f: float, ch: int = ch) -> None:
            if progress:
                progress(share * (ch + f) / channels)

        if profile == "heavy":
            dpdfnet.denoise_array(
                x[:, ch], atten_lim_db=opts.atten_lim_db, threads=threads, progress=prog, cancel=cancel, out=y[:, ch],
                model_path=heavy,
            )
        else:
            dfn.denoise_array(
                x[:, ch], lib_path, model_path, atten_lim_db=opts.atten_lim_db, threads=threads, progress=prog,
                cancel=cancel, out=y[:, ch],
            )
    if cancel is not None and cancel.is_set():
        raise Cancelled()
    if highs:
        bwe.extend(
            y, bwe_model_dir, threads=threads, cancel=cancel, out=y, model=highs,
            progress=(lambda f: progress(share + (0.95 - share) * f)) if progress else None,
        )
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
        for a in range(0, n, ECHO_BLOCK):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            got = aec.process(mic[a:a + ECHO_BLOCK, ch], ref[a:a + ECHO_BLOCK])
            out[done:done + len(got), ch] = got
            done += len(got)
            if progress:
                progress((ch + min(a + ECHO_BLOCK, n) / n) / channels)
        rest = aec.flush()
        out[done:done + len(rest), ch] = rest
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
