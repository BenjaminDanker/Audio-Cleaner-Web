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

Every model step runs on the block grid of blocks.py (30 s blocks, each with the context of the recording around it that the
step measured to need), whole file or part: a whole-file run IS all the blocks, so there is one code path, no seams, and a
block cache (`cache` / `input_id` of clean_audio) that a run fills is the one editor.render reads.

Strength (the attenuation limit) is a blend, in the time domain, of the noise remover's full-strength output with the
recording itself (`blend_strength`): the models always run at full strength, so that output can be kept and the
strength and leveling changed afterwards without running a model again (`redo_denoised`,
`redo_rebuilt`, used by the server's "change and save again"). This is the same arithmetic libDF does in its STFT domain
and DPDFNet does in the time domain (a * noisy + (1 - a) * enhanced, a = 10 ** (-dB / 20)); tests/test_blend.py checks it
against the models' own limit.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import blocks, clarity, dfn, dpdfnet, restore
from .aec import echo_block
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


@dataclass
class Measured:
    """What noise_figure_db heard when it measured a whole recording at once: the mono samples and DeepFilterNet3's
    full-strength output for them. Cleaning a channel with exactly those samples at full strength with the light noise
    remover gives that output (one contiguous run), so clean_audio uses it instead of running the model again."""
    mono: np.ndarray
    light: np.ndarray


def noise_figure_db(x: np.ndarray, lib_path: str | Path, model_path: str | Path,
                    cancel: threading.Event | None = None, keep: dict | None = None) -> float:
    """dB between the voice and the noise in it, as the noise remover itself hears it: DeepFilterNet3 at full strength
    runs over the measured parts of the recording (measure_windows, mono mix), and what it keeps is compared with what
    it takes out, over all of it: wind, keyboards, chatter and steady hiss alike, under the words and between them.
    99 when there is nothing to hear. The website (web/src/media/analysis.js NoiseFigure, with the same model in the
    browser) and the live engine (native/live NoiseMeter) measure the same way. Setting `cancel` raises `Cancelled`.
    With `keep`, a measurement of the whole recording at once is kept there as keep["measured"] (Measured)."""
    voice = removed = 0.0
    windows = measure_windows(len(x))
    for a, b in windows:
        seg = np.asarray(x[a:b], dtype=np.float32)
        seg = seg.mean(axis=1, dtype=np.float32) if seg.ndim == 2 else seg
        y = dfn.denoise_array(np.ascontiguousarray(seg), lib_path, model_path, atten_lim_db=100.0, threads=1, cancel=cancel)
        voice += float(np.sum(np.square(y, dtype=np.float64)))
        removed += float(np.sum(np.square(seg.astype(np.float64) - y)))
        if keep is not None and windows == [(0, len(x))]:
            keep["measured"] = Measured(seg, y)
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


def choose_noise_profile(x: np.ndarray, lib_path: str | Path, model_path: str | Path, wanted: str = "auto",
                         cancel: threading.Event | None = None, keep: dict | None = None) -> str:
    """"light" or "heavy" for this recording ("auto" measures it: noise_figure_db; `keep` as there)."""
    if wanted in ("light", "heavy"):
        return wanted
    return automatic_choices(noise_figure_db(x, lib_path, model_path, cancel, keep))[0]


def done_end(spans: list[tuple[int, int]], a: int) -> slice:
    """The silent stretch that ends where the span starting at `a` begins."""
    prev = 0
    for s0, s1 in spans:
        if s0 == a:
            return slice(prev, a)
        prev = s1
    return slice(a, a)


# Called by clean_audio with what a later change of the choices can start from: ("denoised", y, todo) after the noise
# removal at full strength (before strength and leveling; y holds the channels in `todo`, the others are filled in
# from them afterwards), ("rebuilt", y, None) after the voice rebuild (before leveling).
Keep = Callable[[str, np.ndarray, "list[int] | None"], None]
BLEND_BLOCK = SAMPLE_RATE * 30


def strength_mix(atten_lim_db: float) -> float:
    """The share `a` of the recording that stays in the result for an attenuation limit in dB (0 = nothing is blended:
    100 dB or more, the full-strength output). The same arithmetic as libDF (tract.rs) and DPDFNet."""
    lim = abs(float(atten_lim_db))
    return 0.0 if lim >= 100.0 else min(float(10.0 ** (-lim / 20.0)), 1.0)


def blend_strength(y: np.ndarray, x: np.ndarray, atten_lim_db: float, channels: list[int] | None = None) -> np.ndarray:
    """Limit the noise removal to `atten_lim_db`, in place: y = (1 - a) * y + a * x in the given channels (default all),
    where y is the full-strength output and x the recording it was made from (same length and alignment), a from
    strength_mix. Read block by block, so memory-mapped arrays are never loaded whole."""
    a = strength_mix(atten_lim_db)
    if a <= 0.0:
        return y
    for ch in (range(y.shape[1]) if channels is None else channels):
        for s in range(0, len(y), BLEND_BLOCK):
            e = min(len(y), s + BLEND_BLOCK)
            y[s:e, ch] = ((1.0 - a) * y[s:e, ch].astype(np.float64) + a * x[s:e, ch].astype(np.float64)).astype(np.float32)
    return y


def _after_noise(
    x: np.ndarray, y: np.ndarray, plan: list[int | None], todo: list[int], *, atten: float, opts: CleanOptions,
    progress: Progress | None,
) -> np.ndarray:
    """Everything after the noise removal, from y (its full-strength output in the channels `todo`): the strength, the
    channels that copy another, Leveling. clean_audio and redo_denoised both end here."""
    blend_strength(y, x, atten, todo)
    for c, src in enumerate(plan):
        if src is None:
            y[:, c] = 0.0
        elif src != c:
            y[:, c] = y[:, src]  # before the leveling: its loudness is measured on both channels, as before
    if opts.leveling:
        y = clarity.process(x, y, SAMPLE_RATE, clarity.ClarityParams(loudness_lufs=opts.loudness_lufs))
    if progress:
        progress(1.0)
    return y


def redo_denoised(
    x: np.ndarray, full: np.ndarray, plan: list[int | None], opts: CleanOptions, *, progress: Progress | None = None,
    report: dict | None = None,
) -> np.ndarray:
    """The Studio-clean result for new choices (strength, leveling, loudness) from a kept full-strength noise removal: no
    model runs. `full` is shaped like x with the channels `plan` (channel_plan(x), as kept) names filled in, and is changed
    in place into the result."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    atten = opts.atten_lim_db if opts.atten_lim_db is not None else 100.0
    if report is not None:
        report.update(noise="kept", atten_lim_db=atten)
    todo = sorted({c for c in plan if c is not None})
    return _after_noise(x, full, plan, todo, atten=atten, opts=opts, progress=progress)


def redo_rebuilt(
    x: np.ndarray, rebuilt: np.ndarray, opts: CleanOptions, *, progress: Progress | None = None, report: dict | None = None,
) -> np.ndarray:
    """The Voice-rebuild result for new leveling choices from the kept rebuilt voice (changed in place)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if report is not None:
        report.update(noise="rebuilt")
    y = rebuilt[:, None] if rebuilt.ndim == 1 else rebuilt
    if opts.leveling:
        y = clarity.process(x, y, SAMPLE_RATE, clarity.ClarityParams(loudness_lufs=opts.loudness_lufs))
    if progress:
        progress(1.0)
    return y


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
    restore_model_dir: str | Path | None = None,
    measured: Measured | None = None,
    keep: Keep | None = None,
    cache: blocks.BlockCache | None = None,
    input_id: str | None = None,
) -> np.ndarray:
    """Clean float32 audio shaped (samples, channels) at SAMPLE_RATE.

    `cache`, with `input_id` (anything that identifies the samples of `x`; editor.Inputs.source_id when x is the recording
    itself): the model steps use the editor's block cache, so blocks already in it are not run again and the others are
    stored in it (the strength is a blend afterwards: the cache holds full-strength blocks only).

    `progress` gets a fraction in [0, 1]; denoising is ~95% of the work.
    Setting `cancel` aborts with `Cancelled`. The result is written into `out`
    when given (same shape, float32; e.g. a memory-mapped file, so long inputs
    need little RAM), otherwise into a new array. `heavy_model_path` is the
    DPDFNet-8 model (default: dpdfnet.default_model_path()). `restore_model_dir` holds
    Sidon's ONNX files, required for restore_voice. `report`, when given,
    gets {"noise": "light"|"heavy"|"rebuilt"|"none"}, plus (unless
    rebuilt) "atten_lim_db" (the strength used) and, when the noise was measured, "noise_db" (noise_figure_db).
    `measured`: what choose_noise_profile kept (its `keep`), when the caller measured the recording itself. `keep`: see Keep."""
    opts = opts or CleanOptions()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    channels = x.shape[1]
    if cache is not None and input_id is None:
        raise ValueError("a block cache needs input_id")
    if opts.restore_voice:
        if restore_model_dir is None or not restore.available(restore_model_dir):
            raise FileNotFoundError(f"voice rebuilding needs Sidon's models (not in {restore_model_dir})")
        if report is not None:
            report.update(noise="rebuilt")
        y = restore.restore(
            x, restore_model_dir, threads=threads, cancel=cancel, out=out, device=opts.device,
            progress=(lambda f: progress(0.95 * f)) if progress else None,
            hooks=None if cache is None else blocks.cache_hooks(cache, lambda d: blocks.rebuild_key(
                input_id, d[0], d[1], len(x), restore._WINDOW_SECONDS, restore._OVERLAP_SECONDS)),
        )
        y = y[:, None] if y.ndim == 1 else y
        if keep is not None:
            keep("rebuilt", y, None)
        if opts.leveling:
            y = clarity.process(x, y, SAMPLE_RATE, clarity.ClarityParams(loudness_lufs=opts.loudness_lufs))
        if progress:
            progress(1.0)
        return y
    if not opts.denoise:  # no noise removal: the recording as it is, evened out and set to the loudness (Leveling on its own)
        if report is not None:
            report.update(noise="none")
        y = np.empty(x.shape, dtype=np.float32) if out is None else (out[:, None] if out.ndim == 1 else out)
        y[:] = x
        if opts.leveling:
            y = clarity.process(x, y, SAMPLE_RATE, clarity.ClarityParams(loudness_lufs=opts.loudness_lufs))
        if progress:
            progress(1.0)
        return y
    # One measurement serves both automatic choices: which remover, and how strong.
    figure = stats = None
    if opts.noise == "auto" or opts.atten_lim_db is None:
        kept: dict = {}
        figure = noise_figure_db(x, lib_path, model_path, cancel, kept)
        measured = kept.get("measured", measured)
        if opts.atten_lim_db is None:  # the strength is picked by the model, which needs the rest of the measurements too
            from . import analysis

            stats = analysis.measure_features(x, figure)
    auto_profile, auto_strength = automatic_choices(figure, stats) if figure is not None else (None, None)
    profile = opts.noise if opts.noise != "auto" else auto_profile
    atten = opts.atten_lim_db if opts.atten_lim_db is not None else auto_strength
    heavy = Path(heavy_model_path) if heavy_model_path else dpdfnet.default_model_path()
    if report is not None:
        report["noise"] = profile
        report["atten_lim_db"] = atten
        if figure is not None:
            report["noise_db"] = round(figure, 1)
    share = 0.95  # of the progress bar for noise removal
    y = np.empty(x.shape, dtype=np.float32) if out is None else (out[:, None] if out.ndim == 1 else out)
    # Work only where it changes something: a channel that duplicates the other is cleaned once, a silent one not at
    # all, and digital silence isn't run through the model (it stays silent).
    plan = channel_plan(x)
    todo = sorted({c for c in plan if c is not None})
    if report is not None and todo != list(range(channels)):
        report["channels_cleaned"] = len(todo)
    for k, ch in enumerate(todo):
        spans = sound_spans(x[:, ch])
        if (measured is not None and profile == "light" and spans == [(0, len(x))]
                and len(measured.mono) == len(x) and np.array_equal(measured.mono, x[:, ch])):
            y[:, ch] = measured.light  # the measurement already cleaned exactly this at full strength (any strength is a blend of it)
            if cache is not None:  # (the measurement ran on the same grid: these are the blocks a block run makes)
                for lo, s, e, hi in blocks.grid_windows(0, len(x), blocks.grid_of("light")):
                    cache.put(blocks.block_key("light", input_id, ch, lo, s, e, hi), y[s:e, ch])
            if progress:
                progress(share * (k + 1) / len(todo))
            continue
        total = sum(b - a for a, b in spans) or 1
        done = 0
        for a, b in spans:
            def prog(f: float, k: int = k, done: int = done, size: int = b - a) -> None:
                if progress:
                    progress(share * (k + (done + f * size) / total) / len(todo))

            if a > 0 or b < len(x):
                y[done_end(spans, a), ch] = 0.0  # the silence before this span
            extra: dict = dict(grid_offset=a)  # the blocks are the file's grid cells; x[a:b] is a stretch of the file
            if cache is not None:
                step = "heavy" if profile == "heavy" else "light"
                extra["hooks"] = blocks.cache_hooks(cache, lambda d, step=step, ch=ch: blocks.block_key(step, input_id, ch, *d))
            if profile == "heavy":
                dpdfnet.denoise_array(
                    x[a:b, ch], atten_lim_db=100.0, threads=threads, progress=prog, cancel=cancel,
                    out=y[a:b, ch], model_path=heavy, device=opts.device, **extra,
                )
            else:
                dfn.denoise_array(
                    x[a:b, ch], lib_path, model_path, atten_lim_db=100.0, threads=threads, progress=prog,
                    cancel=cancel, out=y[a:b, ch], **extra,
                )
            done += b - a
        if spans[-1][1] < len(x):
            y[spans[-1][1]:, ch] = 0.0
    if cancel is not None and cancel.is_set():
        raise Cancelled()
    if keep is not None:
        keep("denoised", y, todo)
    return _after_noise(x, y, plan, todo, atten=atten, opts=opts, progress=progress)


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
    cache: blocks.BlockCache | None = None,
    input_id: str | None = None,
) -> np.ndarray:
    """Take the speaker echo out of `mic` (the decoded first track, (samples, channels)) using the file's
    second audio track, which carries what the speakers played. Run it before noise removal.

    Works on the echo block grid (blocks.GRID_ECHO: 30 s blocks, each run with a fresh canceller that has the 10 s of the
    recording before it to adapt on; the canceller converges well inside that, so the blocks join without a seam), so with
    `work_dir` (reference and result are scratch files there) RAM stays flat for long files. `cache` / `input_id` (what
    identifies mic and the reference together): the editor's block cache, as in clean_audio. Returns an array shaped
    like `mic`; `mic` itself when the file has no second track."""
    if probe.audio_tracks < 2:
        return mic
    if work_dir is None:
        ref = ff.decode(src, probe, SAMPLE_RATE, 1, max_seconds, track=1)
    else:
        ref = ff.decode_to_file(src, probe, SAMPLE_RATE, 1, max_seconds, Path(work_dir) / "reference.f32", track=1)
    ref = ref[:, 0]  # a shorter reference counts as silence at the end
    if cache is not None and input_id is None:
        raise ValueError("a block cache needs input_id")
    n, channels = mic.shape
    out = (
        np.empty(mic.shape, dtype=np.float32) if work_dir is None
        else np.lib.format.open_memmap(Path(work_dir) / "echo-free.npy", mode="w+", dtype=np.float32, shape=mic.shape)
    )
    wins = blocks.grid_windows(0, n, blocks.grid_of("echo"))
    for ch in range(channels):
        for k, (lo, s, e, hi) in enumerate(wins):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            key = blocks.block_key("echo", input_id, ch, lo, s, e, hi) if cache is not None else None
            y = cache.get(key) if cache is not None else None
            if y is None or len(y) != e - s:
                y = echo_block(mic[:, ch], ref, lo, s, e, hi)
                if cache is not None:
                    cache.put(key, y)
            out[s:e, ch] = y
            if progress:
                progress((ch + (k + 1) / len(wins)) / channels)
    return out


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
    "SAMPLE_RATE", "CleanOptions", "Cancelled", "clean_audio", "automatic_choices", "blend_strength", "strength_mix", "redo_denoised", "redo_rebuilt",
    "channel_plan", "decode_for_clean", "output_buffer", "as_output",
    "remove_echo", "to_speech_16k",
]


def as_output(y: np.ndarray) -> np.ndarray:
    """Mono results are written as 1-D arrays."""
    return y[:, 0] if y.ndim == 2 and y.shape[1] == 1 else y
