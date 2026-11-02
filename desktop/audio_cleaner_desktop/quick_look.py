"""A fast first look at a recording, so the editor can show itself at once whatever the length of the file. No Qt in here.

Opening a file used to decode all of it at 48 kHz (to a scratch file the size of the recording) and read every sample for the waveform and the
facts before the editor showed anything. Everything the screen needs first is much cheaper:

    quick_peaks     the waveform: the sound decoded once at 8 kHz in one channel, streamed (nothing is kept), as the loudest sample of each of
                    a couple of thousand stretches; the stretches that are read so far can be drawn while it goes on
    WindowReader    reads only where it is asked: `x[a:b]` decodes that stretch of the file. The measurement behind "Automatic" reads about a
                    minute of a recording, spread over it (pipeline.measure_windows), so the best settings are known after a few seconds
                    on an hour of sound, with the same stretches (and so the same answer) as decoding all of it
"""
from __future__ import annotations

import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from acw_common import pipeline
from acw_common.ffmpeg import MediaError, ProbeResult, path_arg

from . import engine

SR = pipeline.SAMPLE_RATE
PEAK_RATE = 8000  # the waveform is drawn from the sound at this rate (a bucket of a long file has plenty of samples)
PEAK_BUCKETS = 2000
PARTIAL_EVERY_S = 0.25  # how often the waveform read so far is handed over


def _args(ff, path: Path, info: ProbeResult, track: int, extra_in: list[str], rate: int, channels: int, fmt: str, extra_out: list[str]) -> list[str]:
    return [
        ff.ffmpeg, "-nostdin", "-hide_banner", "-v", "error", *ff._in(path), *extra_in, "-i", path_arg(path),
        "-map", f"0:{info.stream_index(track)}", "-vn", "-sn", "-dn", *extra_out,
        "-ac", str(channels), "-ar", str(rate), "-f", fmt, "-threads", str(ff.threads), "pipe:1",
    ]


def quick_peaks(
    ff, path: Path, info: ProbeResult, *, buckets: int = PEAK_BUCKETS, max_seconds: float = engine.MAX_SECONDS, track: int = 0,
    progress: Callable[[float], None] | None = None, partial: Callable[[list[int], float], None] | None = None,
) -> tuple[list[int], float]:
    """(the waveform as 0..100 per bucket, the seconds that were read). The buckets divide the length the file's header gives; a stream
    that turns out longer or shorter fills fewer or more of them. `partial(peaks, fraction)` is called about four times a second with the
    waveform of what has been read so far (the rest zero). Raises MediaError; Cancelled (the FFmpeg's `cancel` event)."""
    total = max(int(min(info.duration, max_seconds) * PEAK_RATE), 1)
    buckets = max(1, min(buckets, total))
    out = np.zeros(buckets, np.float64)
    args = _args(ff, path, info, track, [], PEAK_RATE, 1, "s16le", ["-t", str(max_seconds)])
    pos = 0
    leftover = b""
    last = 0.0
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err, **_no_window())
        with ff._guard(proc):
            try:
                while True:
                    chunk = proc.stdout.read(1 << 19)
                    if not chunk:
                        break
                    chunk = leftover + chunk
                    usable = len(chunk) - len(chunk) % 2
                    leftover = chunk[usable:]
                    if not usable:
                        continue
                    v = np.abs(np.frombuffer(chunk[:usable], dtype="<i2").astype(np.int32))
                    ids = np.minimum((pos + np.arange(len(v), dtype=np.int64)) * buckets // total, buckets - 1)
                    starts = np.r_[0, np.flatnonzero(np.diff(ids)) + 1]
                    maxima = np.maximum.reduceat(v, starts)
                    out[ids[starts]] = np.maximum(out[ids[starts]], maxima)
                    pos += len(v)
                    now = time.monotonic()
                    if (progress or partial) and now - last >= PARTIAL_EVERY_S:
                        last = now
                        if progress:
                            progress(min(1.0, pos / total))
                        if partial:
                            partial(_scaled(out), min(1.0, pos / total))
                proc.wait(timeout=600)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                proc.stdout.close()
    if ff._cancelled():
        raise pipeline.Cancelled()
    if proc.returncode != 0:
        raise MediaError("Reading the sound of this file failed.")
    if pos < PEAK_RATE // 10:
        raise MediaError("The audio track is empty.")
    if progress:
        progress(1.0)
    return _scaled(out), pos / PEAK_RATE


def _scaled(values: np.ndarray) -> list[int]:
    top = float(values.max()) if len(values) else 0.0
    if top <= 0:
        return [0] * len(values)
    return [int(round(v / top * 100)) for v in values]


def _no_window() -> dict:
    from acw_common.ffmpeg import _NO_WINDOW

    return _NO_WINDOW


def decode_window(ff, path: Path, info: ProbeResult, start: float, length: float, channels: int, track: int = 0) -> np.ndarray:
    """`length` seconds from `start` as float32 (samples, channels) at 48 kHz (what pipeline.decode_for_clean gives there, up to the
    resampler's first few samples)."""
    args = _args(ff, path, info, track, ["-ss", f"{max(0.0, start):.3f}"], SR, channels, "f32le", ["-t", f"{length:.3f}"])
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err, **_no_window())
        with ff._guard(proc):
            try:
                data = proc.stdout.read()
                proc.wait(timeout=600)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                proc.stdout.close()
    if ff._cancelled():
        raise pipeline.Cancelled()
    if proc.returncode != 0:
        raise MediaError("Reading the sound of this file failed.")
    usable = len(data) - len(data) % (4 * channels)
    return np.frombuffer(data[:usable], dtype=np.float32).reshape(-1, channels)


class WindowReader:
    """The recording as an array that is only read where it is looked at: `len(x)` samples of `channels` channels, and `x[a:b]` decodes
    that stretch. Stands in for the decoded recording where only parts of it are measured (engine._decide: analysis.measure)."""

    ndim = 2

    def __init__(self, ff, path: Path, info: ProbeResult, *, track: int = 0):
        self.ff, self.path, self.info, self.track = ff, Path(path), info, track
        self.channels = min(info.audio_channels, pipeline.MAX_CHANNELS)
        self.n = int(min(info.duration, engine.MAX_SECONDS) * SR)
        self.read = 0  # samples decoded so far (what a test looks at)

    def __len__(self) -> int:
        return self.n

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n, self.channels)

    def __getitem__(self, key) -> np.ndarray:
        if not isinstance(key, slice):
            raise TypeError("a WindowReader is read in slices")
        a, b, step = key.indices(self.n)
        if step != 1:
            raise TypeError("a WindowReader is read in plain slices")
        if b <= a:
            return np.zeros((0, self.channels), np.float32)
        x = decode_window(self.ff, self.path, self.info, a / SR, (b - a) / SR, self.channels, self.track)
        if len(x) < b - a:  # the end of a stream that is shorter than its header says: silence, as a padded decode would give
            x = np.vstack([x, np.zeros((b - a - len(x), self.channels), np.float32)])
        self.read += b - a
        return x[: b - a]


def decide(path: Path, info: ProbeResult, options: engine.JobOptions, *, threads: int = 2, cancel: threading.Event | None = None,
           voice_model: bool = False):
    """What "Automatic" settles for this recording (engine._decide), from the stretches of it that are measured only. None when nothing
    is left automatic in `options`, or the recording gave nothing to go by."""
    cancel = cancel or threading.Event()
    if not options.decides:
        return None
    return engine._decide(WindowReader(engine.ffmpeg(threads, cancel), path, info), options, cancel, voice_model)
