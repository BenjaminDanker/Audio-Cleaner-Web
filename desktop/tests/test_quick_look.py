"""What makes a long file or job show itself at once (quick_look.py, the engine's FFmpeg, the live sound's loading): the waveform streamed at a low rate,
the measurement behind Automatic read from a few stretches of the file instead of all of it, the blocks of a job read a few at a time, the Leveling
gains worked out in the background, and an FFmpeg that does not wait 0.1 s after every program it ran."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from acw_common import pipeline
from acw_common.timeline import Timeline
from audio_cleaner_desktop import engine, live_mix, quick_look

from conftest import bundled_ffmpeg, run_ffmpeg

SR = pipeline.SAMPLE_RATE


@pytest.fixture(scope="session")
def tone_file(tmp_path_factory):
    """Three minutes of a quiet tone with one loud burst at two thirds of the way."""
    from scipy.io import wavfile

    path = tmp_path_factory.mktemp("look") / "burst.wav"
    sr, seconds = 16000, 180
    t = np.arange(sr * seconds) / sr
    x = 0.05 * np.sin(2 * np.pi * 220 * t)
    a, b = int(sr * seconds * 2 / 3), int(sr * (seconds * 2 / 3 + 2))
    x[a:b] = 0.8 * np.sin(2 * np.pi * 330 * t[a:b])
    wavfile.write(path, sr, (x * 32767).astype(np.int16))
    return path


@pytest.mark.native
def test_the_waveform_of_a_long_file_is_read_in_a_few_seconds_and_fills_in_as_it_goes(tone_file):
    ff = engine.ffmpeg(2)
    info = engine.probe(tone_file, ff)
    seen: list[float] = []
    began = time.monotonic()
    peaks, seconds = quick_look.quick_peaks(ff, tone_file, info, buckets=600, partial=lambda pk, f: seen.append(f))
    assert time.monotonic() - began < 30  # (about a second on a desktop machine; the margin is for CI runners)
    assert len(peaks) == 600 and max(peaks) == 100 and seconds == pytest.approx(180.0, abs=0.5)
    loudest = int(np.argmax(peaks))
    assert abs(loudest - 400) <= 8  # the burst at two thirds
    assert peaks[100] < 20  # the quiet tone is much lower
    assert all(b >= a for a, b in zip(seen, seen[1:])) and (not seen or seen[-1] <= 1.0)


@pytest.mark.native
def test_reading_the_waveform_can_be_stopped(tone_file):
    cancel = threading.Event()
    ff = engine.ffmpeg(2, cancel)
    info = engine.probe(tone_file, ff)
    cancel.set()
    with pytest.raises(engine.Cancelled):
        quick_look.quick_peaks(ff, tone_file, info)


@pytest.mark.native
def test_a_window_reader_reads_only_where_it_is_asked_and_what_it_gives_is_what_a_full_decode_gives(tone_file):
    ff = engine.ffmpeg(2)
    info = engine.probe(tone_file, ff)
    reader = quick_look.WindowReader(ff, tone_file, info)
    assert len(reader) == pytest.approx(180 * SR, abs=SR) and reader.shape[1] == 1 and reader.ndim == 2
    a = int(100 * SR)
    window = reader[a:a + 2 * SR]
    assert window.shape == (2 * SR, 1) and window.dtype == np.float32 and reader.read == 2 * SR  # (only that stretch was decoded)
    full = pipeline.decode_for_clean(ff, tone_file, info, engine.MAX_SECONDS)
    ref = full[a:a + 2 * SR]
    assert np.sqrt(np.mean(window ** 2)) == pytest.approx(np.sqrt(np.mean(ref ** 2)), rel=0.02)
    assert np.allclose(window[1000:-1000], ref[1000:-1000], atol=5e-3)
    assert reader[5:5].shape == (0, 1)
    with pytest.raises(TypeError):
        reader[3]


@pytest.mark.native
def test_automatic_decides_from_a_few_stretches_the_same_as_from_all_of_the_file(speech_wav, tmp_path):
    """Three minutes of the speech sample, looped: the stretches that are measured (pipeline.measure_windows) are the same, so the decision is."""
    long = tmp_path / "long-speech.wav"
    run_ffmpeg(bundled_ffmpeg(), "-stream_loop", "-1", "-i", str(speech_wav), "-t", "180", str(long))
    ff = engine.ffmpeg(2)
    info = engine.probe(long, ff)
    opts = engine.JobOptions(strength=engine.AUTO_STRENGTH, auto_rebuild=False)
    began = time.monotonic()
    quick = quick_look.decide(long, info, opts)
    quick_took = time.monotonic() - began
    x = pipeline.decode_for_clean(ff, long, info, engine.MAX_SECONDS)
    full = engine._decide(x, opts, threading.Event(), False)
    assert quick is not None and full is not None
    assert (quick.strength, quick.noise, quick.rebuild) == (full.strength, full.noise, full.rebuild)
    assert quick_took < 60
    reader = quick_look.WindowReader(ff, long, info)
    engine._decide(reader, opts, threading.Event(), False)
    assert reader.read <= 2 * 61 * SR  # about a minute of the three, read for each of the two measurements: not the whole recording each time
    assert quick_look.decide(long, info, engine.JobOptions(strength="gentle", noise="light")) is None  # nothing left automatic: nothing to listen for


def test_the_engines_ffmpeg_does_not_wait_for_its_cancel_watcher_after_every_program(monkeypatch):
    """The shared FFmpeg starts a watcher thread per program and joins it, which waits up to 0.1 s for it to look again: 120 blocks of an hour-long job
    took 13 s. This one lets the watcher end by itself; and it still stops a running program when cancelled."""
    import subprocess
    import sys

    from pathlib import Path

    ff = engine._DesktopFFmpeg(Path("no-programs-are-run-here"), 1, cancel=threading.Event())
    quick = subprocess.Popen([sys.executable, "-c", "pass"])
    began = time.monotonic()
    for _ in range(60):
        with ff._guard(quick):
            pass
    assert time.monotonic() - began < 2.0  # (the shared one: 60 x up to 0.1 s, so 3 to 6 s; this takes milliseconds)
    quick.wait()
    slow = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        with ff._guard(slow):
            ff.cancel.set()
            assert slow.wait(5) is not None  # killed by the watcher
    finally:
        if slow.poll() is None:
            slow.kill()
    # with no cancel event there is no watcher at all
    plain = engine._DesktopFFmpeg(Path("no-programs-are-run-here"), 1)
    with plain._guard(quick):
        pass


class SlowCache:
    """A block cache whose reads take a while (each is a program started and a file decoded)."""

    def __init__(self, n, delay):
        self.keys = {}
        self.delay, self.n = delay, n
        self.reading = 0
        self.max_reading = 0
        self.lock = threading.Lock()

    def has(self, key):
        return key in self.keys

    def get(self, key):
        with self.lock:
            self.reading += 1
            self.max_reading = max(self.max_reading, self.reading)
        time.sleep(self.delay)
        with self.lock:
            self.reading -= 1
        return self.keys[key]


def make_plan(seconds=60.0, block=1.0):
    from acw_common import editor

    n = int(seconds * SR)
    src = np.zeros((n, 1), np.float32)
    inputs = editor.Inputs(src, "sid")
    tl = Timeline.whole_file(seconds, noise={"strength_db": 24.0, "remover": "light"}, leveling=False)
    cache = SlowCache(n, 0.05)
    plan = editor.plan(tl, inputs, cache)
    for t in plan.tasks:
        cache.keys[t.key] = np.full(t.e - t.s, 0.1, np.float32)
    return src, plan, cache


def test_the_blocks_of_a_job_are_read_a_few_at_a_time_and_say_how_far_it_is(tmp_path):
    src, plan, cache = make_plan(150.0)
    layers = live_mix.Layers(src, [0], tmp_path)
    steps: list[float] = []
    read = layers.load(cache, plan, progress=steps.append)
    assert read == len(plan.tasks) and cache.max_reading > 1 and cache.max_reading <= live_mix.BLOCK_READERS  # (more than one read at a time: they overlap)
    assert steps and steps[-1] == 1.0 and all(b >= a for a, b in zip(steps, steps[1:]))
    assert layers.have["light"].all() and np.allclose(layers.array("light")[:100, 0], 0.1)
    read_again = layers.load(cache, plan)  # nothing new: nothing is read again
    assert read_again == 0 and layers.have["light"].all()
    layers.close()


def test_loading_the_blocks_can_be_stopped(tmp_path):
    src, plan, cache = make_plan(150.0)
    layers = live_mix.Layers(src, [0], tmp_path)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(engine.Cancelled):
        layers.load(cache, plan, cancel=cancel)
    layers.close()


def test_the_leveling_gains_are_worked_out_in_the_background_and_applied_when_ready(tmp_path, monkeypatch):
    seconds = 10.0
    n = int(seconds * SR)
    rng = np.random.default_rng(3)
    src = (0.05 * rng.standard_normal((n, 1))).astype(np.float32)
    layers = live_mix.Layers(src, [0])
    tl = Timeline.whole_file(seconds, noise=False, leveling={"loudness_lufs": -16.0})
    release = threading.Event()
    from acw_common import clarity

    real = clarity.integrated_loudness

    def slow(*a, **k):
        release.wait(10)
        return real(*a, **k)

    monkeypatch.setattr(clarity, "integrated_loudness", slow)
    began = time.monotonic()
    mix = live_mix.LiveMix(layers, tl, background=True)
    assert time.monotonic() - began < 5.0  # built at once: it can play now, without Leveling, until the gains are there (waiting for them would take 10 s)
    assert mix.level_gains_db() == [] and mix.read(0, 100).shape == (100, 1)
    release.set()
    assert mix.wait_gains(30) and len(mix.level_gains_db()) == 1
    # an edit while they are being worked out is followed: one measurement at a time, and again after
    release.clear()
    mix.set_timeline(tl)
    mix.set_timeline(Timeline.whole_file(seconds, noise=False, leveling={"loudness_lufs": -23.0}))
    release.set()
    assert mix.wait_gains(30) and mix.level_gains_db()[0][2] != 0.0
    mix.close()  # waits for the measurement: nothing reads the memory maps after it
    plain = live_mix.LiveMix(live_mix.Layers(src, [0]), tl)  # the default measures at once (the way every earlier caller works)
    assert len(plain.level_gains_db()) == 1
    plain.close()
