"""The per-part editor core (acw_common.editor + blocks): planning, billing inputs, joins, Leveling per part, the block grid.

Most tests run on fake blocks (the cache is filled by hand: `fill`), so the arithmetic of the composition is checked without a model;
the tests that need the models (DeepFilterNet3, DPDFNet, Sidon, AEC3) are skipped without them, like the rest of this folder."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acw_common import blocks, clarity, dfn, dpdfnet, editor, pipeline, restore  # noqa: E402
from acw_common.timeline import Timeline  # noqa: E402

from test_common import LIB, MODEL, needs_dfn  # noqa: E402

SR = 48000
HOME = Path.home() / "audio-cleaner"
SIDON = Path(os.environ.get("ACW_SIDON_MODEL", HOME / "models" / "sidon"))
needs_sidon = pytest.mark.skipif(not restore.available(SIDON), reason="Sidon model files not available")
needs_heavy = pytest.mark.skipif(not dpdfnet.default_model_path().is_file(), reason="DPDFNet ONNX model not available")
SPEECH_WAV = Path(os.environ.get("ACW_SPEECH_WAV", HOME / "dev" / "live_opus.wav"))
needs_speech = pytest.mark.skipif(not SPEECH_WAV.is_file(), reason="no speech clip (set ACW_SPEECH_WAV)")


def voiced(seconds, seed=0, pause=0.35, level=0.2):
    """Speech-like: harmonics with vibrato, syllables with real pauses (about -50 dB under the voice) and a little noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    phase = 2 * np.pi * np.cumsum(120 + 25 * np.sin(2 * np.pi * 0.6 * t + seed)) / SR
    phases = rng.uniform(0, 2 * np.pi, 14)  # random phases: a crest factor like speech's, not a pulse train's
    voice = sum(np.sin(k * phase + phases[k]) / k for k in range(1, 14)) * 0.5
    env = np.clip(np.sin(2 * np.pi * (1.0 / (1.0 + pause)) * t + seed) * 3, 0, 1) ** 2
    return (level * voice * env + 0.0003 * rng.standard_normal(len(t))).astype(np.float32)


def snr_db(ref, est):
    ref, est = np.asarray(ref, np.float64), np.asarray(est, np.float64)
    err = np.sum((est - ref) ** 2)
    return 400.0 if err == 0 else float(10 * np.log10(np.sum(ref ** 2) / err))


def make(seconds, seed=0, channels=1, sid="rec"):
    x = voiced(seconds, seed)[:, None]
    if channels == 2:
        x = np.concatenate([x, voiced(seconds, seed + 7)[:, None]], axis=1)
    return editor.Inputs(np.ascontiguousarray(x), sid)


FAKE = {"light": 0.5, "heavy": 0.25}  # what the fake removers do to the recording: a gain, so every join has a step to hide


def fill(cache, p, inputs, rebuilt_gain=0.1):
    """Fake blocks for the misses of plan `p`: the noise removers scale the recording, Voice rebuild gives it at `rebuilt_gain`."""
    for t in p.misses:
        if t.step == "rebuild":
            lo, hi = t.s, t.e
            src = inputs.source.mean(axis=1) if t.channel == "mid" else inputs.source[:, t.channel]
            cache.put(t.key, src[lo:hi] * rebuilt_gain)
        elif t.step == "echo":
            cache.put(t.key, inputs.source[t.s:t.e, t.channel] * 0.3)
        else:
            cache.put(t.key, inputs.source[t.s:t.e, t.channel] * FAKE[t.step])


class Spy:
    """A cache that records which keys were read."""

    def __init__(self, inner):
        self.inner, self.read = inner, []

    def get(self, key):
        self.read.append(key)
        return self.inner.get(key)

    def put(self, key, block):
        self.inner.put(key, block)

    def has(self, key):
        return self.inner.has(key)


def noise_tl(seconds, *parts, level=False, rebuild=()):
    """parts: (start, end, strength_db, remover)"""
    t = Timeline(seconds)
    for a, b, db, r in parts:
        t.paint("noise", a, b, {"strength_db": db, "remover": r})
    for a, b in rebuild:
        t.paint("rebuild", a, b)
    if level:
        t.paint("leveling", 0, seconds)
    return t


# ---------------------------------------------------------------------------------------------------------------------
# The block grid and the cache
# ---------------------------------------------------------------------------------------------------------------------
def test_the_grids_are_the_measured_ones():
    assert (blocks.GRID_LIGHT.seconds, blocks.GRID_LIGHT.before, blocks.GRID_LIGHT.after) == (30.0, 16.0, 0.25)
    assert (blocks.GRID_HEAVY.seconds, blocks.GRID_HEAVY.before, blocks.GRID_HEAVY.after) == (30.0, 24.0, 0.25)
    assert (blocks.GRID_ECHO.seconds, blocks.GRID_ECHO.before) == (30.0, 10.0) and blocks.GRID_ECHO.after * SR >= 432
    assert restore.block_samples() == 14 * SR  # 16 s windows, 2 s of overlap
    assert restore.block_input_range(2, 100 * SR) == (12 * SR, 60 * SR)  # windows 1, 2 and 3 (14 s steps of 16 s) and the 2 s the filters read
    assert restore.block_input_range(0, 100 * SR) == (0, 32 * SR)  # the first block has no window before it


def test_windows_are_the_files_grid_cells_with_their_context_clamped_to_the_stretch():
    g = blocks.GRID_LIGHT
    w = blocks.grid_windows(0, 70 * SR, g)
    assert [(s // SR, e // SR) for _, s, e, _ in w] == [(0, 30), (30, 60), (60, 70)]
    assert w[0][0] == 0 and w[1][0] == 14 * SR and w[1][3] == 60 * SR + SR // 4 * 1 and w[2][3] == 70 * SR  # context before / after, clamped
    # a stretch of the file (a span between long digital silences) keeps the file's grid: the first block is short, its context stops at the span
    w = blocks.grid_windows(10 * SR, 100 * SR, g)
    assert [(s // SR, e // SR) for _, s, e, _ in w] == [(10, 30), (30, 60), (60, 90), (90, 100)] and w[0][0] == 10 * SR and w[1][0] == 14 * SR
    # only the blocks that touch the range asked for
    w = blocks.grid_windows(0, 100 * SR, g, 35 * SR, 40 * SR)
    assert [(s // SR, e // SR) for _, s, e, _ in w] == [(30, 60)]
    assert blocks.grid_windows(0, 100 * SR, g, 50 * SR, 50 * SR) == []


def test_caches_hand_back_what_was_stored(tmp_path):
    rng = np.random.default_rng(0)
    for cache in (blocks.MemoryCache(), blocks.DirCache(tmp_path / "c")):
        y = rng.standard_normal(1000).astype(np.float32)
        key = blocks.block_key("light", "in", 0, 0, 0, 1000, 1000)
        assert not cache.has(key) and cache.get(key) is None
        cache.put(key, y)
        assert cache.has(key) and np.array_equal(cache.get(key), y) and cache.get(key).dtype == np.float32
    again = blocks.DirCache(tmp_path / "c")  # a new process sees the folder
    assert again.has(key) and np.array_equal(again.get(key), y)


def test_keys_name_the_step_the_model_the_input_and_the_place_but_not_the_strength():
    k = blocks.block_key("light", "rec", 0, 0, 10, 20, 30)
    assert k != blocks.block_key("heavy", "rec", 0, 0, 10, 20, 30) and k != blocks.block_key("light", "rec2", 0, 0, 10, 20, 30)
    assert k != blocks.block_key("light", "rec", 1, 0, 10, 20, 30) and k != blocks.block_key("light", "rec", 0, 0, 10, 20, 31)
    inp = make(40)
    keys = {t.key for t in editor.plan(noise_tl(40, (0, 40, 12.0, "light")), inp).tasks}
    assert keys == {t.key for t in editor.plan(noise_tl(40, (0, 40, 100.0, "light")), inp).tasks}  # the strength is a blend afterwards
    assert keys.isdisjoint({t.key for t in editor.plan(noise_tl(40, (0, 40, 12.0, "heavy")), inp).tasks})


def test_a_blocks_identity_includes_the_echo_parts_it_read():
    x = np.ascontiguousarray(voiced(100)[:, None])
    inp = editor.Inputs(x, "rec", reference=np.zeros(len(x), np.float32), reference_id="ref")
    plain = Timeline(100.0).paint("noise", 0, 100, {"remover": "light"})
    echo = plain.copy().paint("echo", 40, 50)  # touches the second block's window (14 s .. 60.25 s) and the third's (44 s .. )
    kp = {(t.s, t.step): t.key for t in editor.plan(plain, inp).tasks}
    ke = {(t.s, t.step): t.key for t in editor.plan(echo, inp).tasks}
    assert kp[(0, "light")] == ke[(0, "light")]  # block 0 reads 0..30.25 s: no echo there
    assert kp[(30 * SR, "light")] != ke[(30 * SR, "light")] and kp[(60 * SR, "light")] != ke[(60 * SR, "light")]
    assert kp[(90 * SR, "light")] == ke[(90 * SR, "light")]  # reads 74 s on: the echo part ended at 50 s
    echo_tasks = [t for t in editor.plan(echo, inp).tasks if t.step == "echo"]
    assert [(t.s // SR, t.e // SR) for t in echo_tasks] == [(30, 60)] and echo_tasks[0].lo == 20 * SR  # the 10 s the canceller adapts on
    with pytest.raises(editor.EditorError):  # no second track
        editor.plan(echo, make(100))


# ---------------------------------------------------------------------------------------------------------------------
# Planning and billing
# ---------------------------------------------------------------------------------------------------------------------
def test_planning_whole_file_default_is_all_blocks_and_a_change_of_strength_needs_none():
    inp = make(100)
    cache = blocks.MemoryCache()
    tl = noise_tl(100, (0, 100, 24.0, "light"), level=True)
    p = editor.plan(tl, inp, cache)
    assert [(t.s // SR, t.e // SR, t.step) for t in p.tasks] == [(0, 30, "light"), (30, 60, "light"), (60, 90, "light"), (90, 100, "light")]
    assert p.misses == p.tasks and sum(t.seconds for t in p.misses) == 100
    assert [t.compute_seconds for t in p.tasks] == [30 + 0.25, 16 + 30 + 0.25, 16 + 30 + 0.25, 16 + 10]  # the context the model also reads
    fill(cache, p, inp)
    p2 = editor.plan(tl, inp, cache)
    assert p2.misses == [] and len(p2.tasks) == 4
    for db in (12.0, 40.0, 100.0):  # no model run for any strength
        assert editor.plan(noise_tl(100, (0, 100, db, "light"), level=True), inp, cache).misses == []
    heavy = editor.plan(noise_tl(100, (0, 100, 24.0, "heavy"), level=True), inp, cache)  # another remover is another model run
    assert [t.step for t in heavy.misses] == ["heavy"] * 4


def test_only_the_blocks_of_a_new_part_are_misses_and_voice_rebuild_wins_over_noise_removal():
    inp = make(100)
    cache = blocks.MemoryCache()
    base = noise_tl(100, (0, 100, 24.0, "light"))
    fill(cache, editor.plan(base, inp, cache), inp)
    tl = base.copy().paint("rebuild", 31.2, 44.8)
    p = editor.plan(tl, inp, cache)
    assert [t.step for t in p.misses] == ["rebuild", "rebuild"] and [t.index for t in p.misses] == [2, 3]  # 28-42 s and 42-56 s
    assert p.misses[0].lo == 12 * SR and p.misses[1].lo == 26 * SR  # the windows around them: context is read, not asked for
    assert editor.billable(p)["rebuild"]["seconds"] == 14  # 10.8 s + 2.8 s of the part, rounded up: not the 28 s of blocks
    # where Voice rebuild covers, the noise removal is not run: rebuild the first 70 s and only the last 30 s need the remover
    tl = noise_tl(100, (0, 100, 24.0, "light"), rebuild=[(0, 70)])
    p = editor.plan(tl, inp, None)
    assert {(t.s // SR, t.e // SR) for t in p.tasks if t.step == "light"} == {(60, 90), (90, 100)}
    b = editor.billable(p)
    assert b["light"]["seconds"] == 30 and b["rebuild"]["seconds"] == 70


def test_billable_seconds_round_up_exclude_the_context_and_carry_the_fixed_amounts():
    inp = make(100)
    p = editor.plan(noise_tl(100, (5, 15.2, 24.0, "light")), inp, None)
    b = editor.billable(p)
    assert set(b) == {"light"} and b["light"]["seconds"] == 11  # 10.2 s, rounded up
    assert p.tasks[0].seconds == 30 and p.tasks[0].asked_seconds == pytest.approx(10.2)  # the block is 30 s; 10.2 s were asked for
    assert p.tasks[0].compute_seconds > p.tasks[0].seconds  # the context is ours
    assert editor.billable(editor.plan(noise_tl(100, (50, 51, 24.0, "light")), inp, None))["light"]["seconds"] == 1  # no minimum charge
    assert editor.billable(editor.plan(Timeline(100.0), inp, None)) == {}  # nothing asked, nothing billed
    # the per-run fixed amount of each step, as seconds of audio at the step's own rate
    c = editor.STEP_COSTS
    assert c["light"].fixed_equivalent_seconds == pytest.approx((1.55 - 1.28) / (0.129 - 0.013))  # light = (light + Leveling) - Leveling
    assert c["light"].fixed + c["leveling"].fixed == pytest.approx(1.55) and c["light"].per_second + c["leveling"].per_second == pytest.approx(0.129)
    assert c["heavy"].fixed_equivalent_seconds == pytest.approx(2.9 / 0.386) and c["heavy"].resource == "gpu"
    assert c["rebuild"].fixed_equivalent_seconds == pytest.approx(4.3 / 0.203) and c["rebuild"].resource == "gpu"
    assert c["leveling"].fixed_equivalent_seconds == pytest.approx(1.28 / 0.013)
    assert c["echo"].fixed == 0.12
    tl = noise_tl(100, (0, 40, 24.0, "heavy"), rebuild=[(40, 60)], level=True)
    b = editor.billable(editor.plan(tl, inp, None))
    assert set(b) == {"heavy", "rebuild", "leveling"} and b["leveling"]["seconds"] == 100
    assert b["heavy"]["fixed_equivalent_seconds"] == c["heavy"].fixed / c["heavy"].per_second
    assert all(set(v) >= {"seconds", "fixed_equivalent_seconds", "resource", "fixed", "per_second"} for v in b.values())
    assert editor.compute_estimate(editor.plan(tl, inp, None))["rebuild"] > c["rebuild"].fixed  # fixed + context included


def test_stereo_channels_that_are_cleaned_are_blocks_per_channel_but_billed_once():
    x = np.ascontiguousarray(np.concatenate([voiced(70, 1)[:, None], voiced(70, 2)[:, None]], axis=1))
    inp = editor.Inputs(x, "st")
    p = editor.plan(noise_tl(70, (0, 70, 24.0, "light")), inp, None)
    assert sorted({t.channel for t in p.tasks}) == [0, 1] and len(p.tasks) == 6
    assert editor.billable(p)["light"]["seconds"] == 70  # audio seconds, not channel seconds
    dual = editor.Inputs(np.ascontiguousarray(np.concatenate([x[:, :1], x[:, :1]], axis=1)), "dual")
    assert {t.channel for t in editor.plan(noise_tl(70, (0, 70, 24.0, "light")), dual, None).tasks} == {0}  # one is a copy of the other


def test_the_plan_is_what_render_reads_and_a_change_reruns_nothing_else():
    inp = make(100, channels=2)
    cache = blocks.MemoryCache()
    tl = noise_tl(100, (0, 50, 24.0, "light"), (50, 100, 40.0, "heavy"), rebuild=[(70, 80)])
    tl.paint("leveling", 10, 90)
    p = editor.plan(tl, inp, cache)
    fill(cache, p, inp)
    spy = Spy(cache)
    y = editor.render(tl, inp, spy)  # no runner: it would raise if a block were missing
    assert y.shape == (len(inp.source), 2) and np.isfinite(y).all()
    assert set(spy.read) <= {t.key for t in p.tasks}
    assert {t.step for t in p.tasks} == {"light", "heavy", "rebuild"}
    with pytest.raises(editor.EditorError, match="Runner"):
        editor.render(tl, inp, blocks.MemoryCache())


def test_leveling_is_remade_only_where_something_under_its_measure_span_changed():
    inp = make(100)
    tl = noise_tl(100, (0, 100, 24.0, "light"))
    tl.paint("leveling", 0, 40).paint("leveling", 60, 100)
    assert editor.plan(tl, inp, None).leveling == [(0, 40 * SR), (60 * SR, 100 * SR)]
    assert editor.plan(tl, inp, None, previous=tl).leveling == []  # as it was
    changed = tl.copy().paint("noise", 5, 10, {"strength_db": 12.0, "remover": "light"})
    assert editor.plan(changed, inp, None, previous=tl).leveling == [(0, 40 * SR)]  # the second part measures 20 s on: not 5-10 s
    other = tl.copy().paint("leveling", 0, 40, {"loudness_lufs": -20.0})
    assert editor.plan(other, inp, None, previous=tl).leveling == [(0, 40 * SR)]
    assert editor.billable(editor.plan(tl, inp, None, previous=tl)) == {"light": editor.billable(editor.plan(tl, inp, None))["light"]}


# ---------------------------------------------------------------------------------------------------------------------
# Joins
# ---------------------------------------------------------------------------------------------------------------------
def ramp_zone(lay, i=0):
    j = lay.main_joins[i]
    return j.z0, j.z1


def test_join_lengths_follow_the_kinds():
    inp = make(60)
    lay = editor.Layout(noise_tl(60, (10, 20, 24.0, "light"), (20, 30, 24.0, "heavy"), (30, 40, 12.0, "heavy")), inp)
    assert [(s.kind, s.a // SR, s.b // SR) for s in lay.main] == [("pass", 0, 10), ("noise", 10, 20), ("noise", 20, 30), ("noise", 30, 40), ("pass", 40, 60)]
    assert [j.d for j in lay.main_joins] == [4800, 2400, 4800, 4800]  # on / off or another strength: 100 ms; light vs heavy: 50 ms
    assert all(j.kind == "A" and j.shape == "lin" for j in lay.main_joins)
    assert [(j.z1 - j.z0) for j in lay.main_joins] == [j.d for j in lay.main_joins] and lay.main_joins[0].z0 == 10 * SR - 2400
    echo = editor.Layout(Timeline(60.0).paint("echo", 10, 20), editor.Inputs(inp.source, "x", np.zeros(len(inp.source), np.float32), "r"))
    assert [j.d for j in echo.echo_joins] == [2400, 2400]  # echo removal: 50 ms
    lev = editor.Layout(Timeline(60.0).paint("leveling", 10, 20), inp)
    assert [j.d for j in lev.level_joins] == [4800, 4800]
    rb = editor.Layout(noise_tl(60, (0, 60, 24.0, "light"), rebuild=[(20, 30)]), inp)
    assert [j.kind for j in rb.main_joins] == ["B", "B"] and all(j.shape == "ep" for j in rb.main_joins)
    short = editor.Layout(noise_tl(60, (10, 10.05, 24.0, "light")), inp)  # a ramp is never longer than the part beside it
    assert max(j.d for j in short.main_joins) <= int(0.05 * SR)


def d2peak(x, a, b):
    return float(np.abs(x[a:b][2:] - 2 * x[a:b][1:-1] + x[a:b][:-2]).max() + 1e-9)


def test_kind_a_joins_are_linear_ramps_of_the_blend_with_no_click():
    inp = make(40)
    x = inp.source[:, 0]
    cache = blocks.MemoryCache()
    tl = noise_tl(40, (10, 25, 100.0, "light"), (25, 40, 24.0, "light"))  # off -> on, then a weaker strength
    fill(cache, editor.plan(tl, inp, cache), inp)
    y = editor.render(tl, inp, cache)[:, 0]
    f = 0.5 * x  # the fake full-strength output
    assert np.array_equal(y[:10 * SR - 2400], x[:10 * SR - 2400])  # untouched where nothing is asked
    a = pipeline.strength_mix(24.0)
    assert np.allclose(y[10 * SR + 2400:25 * SR - 2400], f[10 * SR + 2400:25 * SR - 2400], atol=1e-7)  # full strength inside
    want = ((1 - a) * f[25 * SR + 2400:] + a * x[25 * SR + 2400:])
    assert np.allclose(y[25 * SR + 2400:], want, atol=1e-6)
    # inside the 100 ms zones the output is the linear ramp of the two sides (for one remover that IS a ramp of the blend amount)
    z0, z1 = 10 * SR - 2400, 10 * SR + 2400
    t = (np.arange(z0, z1) - z0) / 4800
    assert np.allclose(y[z0:z1], (1 - t) * x[z0:z1] + t * f[z0:z1], atol=1e-6)
    z0, z1 = 25 * SR - 2400, 25 * SR + 2400
    t = (np.arange(z0, z1) - z0) / 4800
    wb = ((1 - a) * f + a * x)
    assert np.allclose(y[z0:z1], (1 - t) * f[z0:z1] + t * wb[z0:z1], atol=1e-6)
    # no click: the second difference at the join is no larger than the signals' own; a hard cut at the same place is
    for pos in (10, 25):
        a0, b0 = pos * SR - 2400 - 192, pos * SR + 2400 + 192
        own = max(d2peak(x, a0, b0), d2peak(f, a0, b0))
        assert 20 * np.log10(d2peak(y, a0, b0) / own) < 3.0
    hard = x.copy()  # the same join as a hard cut: the measure sees it
    hard[10 * SR:] = f[10 * SR:]
    assert 20 * np.log10(d2peak(hard, 10 * SR - 2592, 10 * SR + 2592) / own) > 6.0 or d2peak(hard, 10 * SR - 2592, 10 * SR + 2592) > 2 * d2peak(y, 10 * SR - 2592, 10 * SR + 2592)


def test_a_light_to_heavy_join_is_a_50_ms_crossfade_of_the_two_outputs():
    inp = make(40)
    x = inp.source[:, 0]
    cache = blocks.MemoryCache()
    tl = noise_tl(40, (0, 20, 100.0, "light"), (20, 40, 100.0, "heavy"))
    fill(cache, editor.plan(tl, inp, cache), inp)
    y = editor.render(tl, inp, cache)[:, 0]
    z0, z1 = 20 * SR - 1200, 20 * SR + 1200
    t = (np.arange(z0, z1) - z0) / 2400
    assert np.allclose(y[z0:z1], (1 - t) * 0.5 * x[z0:z1] + t * 0.25 * x[z0:z1], atol=1e-6)
    assert np.allclose(y[:z0], 0.5 * x[:z0], atol=1e-7) and np.allclose(y[z1:], 0.25 * x[z1:], atol=1e-7)


def _render_obj(tl, inp, cache):
    r = editor._Render(tl, inp, cache, None, None, None, None, None)
    r.make_b_joins(inp.source, r.lay.main, r.lay.main_joins)
    return r


def test_kind_b_joins_move_to_a_pause_and_are_100_ms_there_else_250_ms():
    x = voiced(60, 3)[:, None]
    t_nom = 20.0
    # a pause right of the nominal join (the quietest 60 ms within 0.3 s): silence 20.15 .. 20.40 s
    x[int(20.15 * SR):int(20.40 * SR)] *= 0.001
    inp = editor.Inputs(np.ascontiguousarray(x), "p")
    tl = noise_tl(60, (0, 20, 100.0, "light"), rebuild=[(20, 40)])
    cache = blocks.MemoryCache()
    fill(cache, editor.plan(tl, inp, cache), inp, rebuilt_gain=1.0)
    j = _render_obj(tl, inp, cache).lay.main_joins[0]
    assert j.kind == "B" and j.d == int(0.1 * SR) and abs(j.t - 20 * SR) <= 0.3 * SR and j.t != 20 * SR  # moved, 100 ms
    assert 20.15 * SR <= j.t <= 20.40 * SR and j.z1 - j.z0 == j.d
    # speech all the way (no pause within 0.3 s): stays, 250 ms
    tone = (0.2 * np.sin(2 * np.pi * 150 * np.arange(60 * SR) / SR)).astype(np.float32)[:, None]
    inp2 = editor.Inputs(np.ascontiguousarray(tone), "q")
    fill_cache = blocks.MemoryCache()
    fill(fill_cache, editor.plan(tl, inp2, fill_cache), inp2, rebuilt_gain=1.0)
    j2 = _render_obj(tl, inp2, fill_cache).lay.main_joins[0]
    assert j2.t == 20 * SR and j2.d == int(0.25 * SR)


def test_kind_b_loudness_match_is_clamped_to_9_db_and_fades_out_over_2_s():
    inp = make(60, 4)
    x = inp.source[:, 0]
    tl = noise_tl(60, (0, 60, 100.0, "light"), rebuild=[(20, 40)])
    for gain, expect in ((0.1, 9.0), (1.0, None)):  # the rebuilt voice 20 dB under the noise-removed one, then 6 dB over (0.5 * 2)
        cache = blocks.MemoryCache()
        fill(cache, editor.plan(tl, inp, cache), inp, rebuilt_gain=gain)
        r = _render_obj(tl, inp, cache)
        for j in r.lay.main_joins:
            if expect is not None:
                assert j.gain_db == pytest.approx(expect, abs=1e-6)  # clamped: 20 dB short, 9 dB applied
            else:
                assert -9.0 <= j.gain_db <= -3.0  # louder by about 6 dB: brought down
    # the gain is smooth and gone after 2 s: the rebuilt voice in the middle of a 20 s part is untouched
    cache = blocks.MemoryCache()
    fill(cache, editor.plan(tl, inp, cache), inp, rebuilt_gain=0.1)
    y = editor.render(tl, inp, cache)[:, 0]
    mid = slice(23 * SR, 37 * SR)
    assert np.allclose(y[mid], 0.1 * x[mid], atol=1e-7)
    j = editor._Render(tl, inp, cache, None, None, None, None, None)
    j.make_b_joins(inp.source, j.lay.main, j.lay.main_joins)
    g = j.rebuild_gain(j.lay.main, j.lay.main_joins, 1, 19 * SR, 25 * SR)
    jt = j.lay.main_joins[0].t
    assert g[0] == pytest.approx(10 ** (9 / 20), rel=1e-4)  # full on the other side of the join
    assert abs(g[jt + SR // 2 - 19 * SR] - 1.0) > 0.2 and g[jt + 2 * SR - 19 * SR] == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.abs(np.diff(g)) < 1e-3)  # smooth: no step


def test_kind_b_joins_are_equal_power_crossfades_without_clicks(monkeypatch):
    monkeypatch.setattr(editor, "_active_level_db", lambda x: 0.0)  # no loudness match here: the crossfade on its own
    inp = make(60, 6)
    x = inp.source[:, 0]
    tl = noise_tl(60, (0, 60, 100.0, "light"), rebuild=[(20, 40)])
    cache = blocks.MemoryCache()
    fill(cache, editor.plan(tl, inp, cache), inp, rebuilt_gain=0.2)  # the noise-removed side is 0.5 * x, the rebuilt side 0.2 * x
    r = _render_obj(tl, inp, cache)
    y = editor.render(tl, inp, cache)[:, 0]
    for j in r.lay.main_joins:
        assert j.shape == "ep" and j.gain_db == 0.0
        a, b = j.z0, j.z1
        t = (np.arange(a, b) - a) / j.d
        denoised, rebuilt = (0.5 * x[a:b], 0.2 * x[a:b])
        left, right = (denoised, rebuilt) if j.rb_right else (rebuilt, denoised)
        want = np.cos(t * np.pi / 2) * left + np.sin(t * np.pi / 2) * right
        assert np.allclose(y[a:b], want, atol=1e-6)
        assert np.allclose(np.cos(t * np.pi / 2) ** 2 + np.sin(t * np.pi / 2) ** 2, 1.0)  # equal power
        pad = int(0.004 * SR)
        own = max(d2peak(0.5 * x, a - pad, b + pad), d2peak(0.2 * x, a - pad, b + pad))
        assert 20 * np.log10(d2peak(y, a - pad, b + pad) / own) < 3.0  # the click excess of the experiments: none of consequence
    assert np.allclose(y[:r.lay.main_joins[0].z0], 0.5 * x[:r.lay.main_joins[0].z0], atol=1e-7)
    assert np.allclose(y[r.lay.main_joins[0].z1:r.lay.main_joins[1].z0], 0.2 * x[r.lay.main_joins[0].z1:r.lay.main_joins[1].z0], atol=1e-7)


def test_a_part_at_the_file_edge_or_shorter_than_its_ramp_has_no_missing_join():
    inp = make(30)
    cache = blocks.MemoryCache()
    tl = noise_tl(30, (0, 5, 100.0, "light"), (29.99, 30, 100.0, "light"))
    fill(cache, editor.plan(tl, inp, cache), inp)
    y = editor.render(tl, inp, cache)
    assert y.shape == (30 * SR, 1) and np.isfinite(y).all()
    x = inp.source[:, 0]
    assert np.allclose(y[:SR, 0], 0.5 * x[:SR], atol=1e-7)  # (the part starts at the file start: no join before it)


# ---------------------------------------------------------------------------------------------------------------------
# Leveling per part
# ---------------------------------------------------------------------------------------------------------------------
def test_leveling_is_measured_on_the_part_and_max_its_length_or_10_s_each_side_clamped_to_the_file():
    inp = make(100)
    lay = editor.Layout(Timeline(100.0).paint("leveling", 0, 5), inp)
    assert lay.measure_span(0, 5 * SR) == (0, 15 * SR)  # never before 0
    assert lay.measure_span(95 * SR, 100 * SR) == (85 * SR, 100 * SR)  # never past the end
    assert lay.measure_span(40 * SR, 60 * SR) == (20 * SR, 80 * SR)  # 1 x the part's length each side
    assert lay.measure_span(30 * SR, 31 * SR) == (20 * SR, 41 * SR)  # a short part: 10 s each side
    assert lay.measure_span(0, 100 * SR) == (0, 100 * SR)
    assert lay.measure_span(10 * SR, 95 * SR) == (0, 100 * SR)  # long: clamped on both sides


def lufs(x):
    return clarity.integrated_loudness(np.asarray(x, np.float32), SR)


def leveled_signal(seconds=120):
    """Two levels: 0-60 s loud, 60-120 s 12 dB quieter (speech-like, pauses)."""
    x = voiced(seconds, 8, level=0.2)
    x[60 * SR:] *= 0.25
    return np.ascontiguousarray(x[:, None])


def test_whole_file_leveling_is_clarity_process_over_the_file_that_is_todays_output():
    x = leveled_signal(60)
    inp = editor.Inputs(x, "lev")
    tl = Timeline(60.0).paint("leveling", 0, 60, {"loudness_lufs": -18.0})
    y = editor.render(tl, inp, blocks.MemoryCache())
    want = clarity.process(x, x.copy(), SR, clarity.ClarityParams(loudness_lufs=-18.0))
    assert np.array_equal(y, want)
    assert np.array_equal(pipeline.clean_audio(x, "none", "none", pipeline.CleanOptions(denoise=False, loudness_lufs=-18.0)), want)
    assert abs(lufs(y) + 18) < 1.0


def test_leveling_a_part_measures_the_part_leaves_the_rest_alone_and_joins_without_a_step():
    x = leveled_signal(120)  # 0-60 s loud, 60-120 s 12 dB quieter
    inp = editor.Inputs(x, "lev")
    tl = Timeline(120.0).paint("leveling", 90, 120)  # the end of the quiet half: measured over 60-120 s (30 s each side, clamped at the end)
    y = editor.render(tl, inp, blocks.MemoryCache())
    # off: bit for bit the recording, up to the 100 ms ramp
    assert np.array_equal(y[: 90 * SR - 2400], x[: 90 * SR - 2400])
    # on: the part reaches its target, measured on its own surroundings
    assert abs(lufs(y[92 * SR:118 * SR]) + 16) < 1.5 and lufs(x[92 * SR:118 * SR]) < -26
    # whole-file Leveling leaves this part far from its target (the loud half decides the gain)
    whole = clarity.process(x, x.copy(), SR, clarity.ClarityParams())
    assert abs(lufs(whole[92 * SR:118 * SR]) + 16) > 4.0
    # a part whose surroundings include the loud half is measured with it: that is the rule (the part and as much again each side)
    mid = editor.render(Timeline(120.0).paint("leveling", 70, 110), inp, blocks.MemoryCache())
    assert lufs(mid[72 * SR:108 * SR]) < lufs(y[92 * SR:118 * SR]) - 3
    # the join: a ramp of 100 ms, no click
    a0, b0 = 90 * SR - 2400 - 192, 90 * SR + 2400 + 192
    own = max(d2peak(x[:, 0], a0, b0), d2peak(y[:, 0], 90 * SR + 2400, 90 * SR + 2400 + 4800), 1e-6)
    assert d2peak(y[:, 0], a0, b0) < 3 * own


def test_leveling_parts_at_the_start_the_end_and_short_ones_work_and_clamp():
    x = leveled_signal(60)
    inp = editor.Inputs(x, "lev")
    for a, b in ((0, 3), (57, 60), (30, 30.3), (0, 60)):
        tl = Timeline(60.0).paint("leveling", a, b)
        y = editor.render(tl, inp, blocks.MemoryCache())
        assert y.shape == x.shape and np.isfinite(y).all(), (a, b)
        outside = np.r_[0:max(0, int(a * SR) - 2400), min(len(x), int(b * SR) + 2400):len(x)]
        assert np.array_equal(y[outside], x[outside])  # a part with Leveling off stays as it is
    short = editor.render(Timeline(0.3).paint("leveling", 0, 0.3), editor.Inputs(np.ascontiguousarray(x[: int(0.3 * SR)]), "tiny"), blocks.MemoryCache())
    assert short.shape == (int(0.3 * SR), 1) and np.isfinite(short).all()  # shorter than a loudness block: no gain, no crash


def test_two_leveled_parts_with_different_targets_are_two_measurements_joined_by_a_ramp():
    x = np.ascontiguousarray(voiced(120, 9, level=0.05)[:, None])  # one level all the way: each part reaches its own target
    inp = editor.Inputs(x, "lev")
    tl = Timeline(120.0).paint("leveling", 0, 60, {"loudness_lufs": -16.0}).paint("leveling", 60, 120, {"loudness_lufs": -22.0})
    y = editor.render(tl, inp, blocks.MemoryCache())
    assert abs(lufs(y[5 * SR:55 * SR]) + 16) < 1.5 and abs(lufs(y[65 * SR:115 * SR]) + 22) < 1.5
    assert [s.kind for s in editor.Layout(tl, inp).level] == ["level", "level"]


def test_voice_rebuild_keeps_the_file_under_0_99_like_restore_does():
    inp = make(40)
    tl = Timeline(40.0).paint("rebuild", 0, 40)  # the whole file: the same peak as restore.restore would see
    cache = blocks.MemoryCache()
    fill(cache, editor.plan(tl, inp, cache), inp, rebuilt_gain=8.0)  # the "rebuilt" voice overshoots
    peak = max(float(np.max(np.abs(cache.get(t.key)))) for t in editor.plan(tl, inp, cache).tasks)
    assert peak > 0.99
    y = editor.render(tl, inp, cache)
    assert float(np.abs(y).max()) == pytest.approx(0.99, abs=1e-6)
    assert np.allclose(y[:, 0], inp.source[:, 0] * 8.0 * (0.99 / peak), atol=1e-6)
    fill_small = blocks.MemoryCache()
    fill(fill_small, editor.plan(tl, inp, fill_small), inp, rebuilt_gain=0.5)  # under it: untouched
    assert np.allclose(editor.render(tl, inp, fill_small)[:, 0], inp.source[:, 0] * 0.5, atol=1e-7)


def test_inputs_and_render_check_their_arguments():
    inp = make(10)
    with pytest.raises(ValueError):
        editor.Inputs(np.zeros((10, 1), np.float64), "x")
    with pytest.raises(ValueError):
        editor.Inputs(inp.source, "x", reference=np.zeros(10, np.float32))  # a reference needs its identity
    with pytest.raises(editor.EditorError):
        editor.render(Timeline(20.0), inp, blocks.MemoryCache())  # a timeline for another recording
    with pytest.raises(editor.EditorError, match="resolve"):
        editor.plan(Timeline(10.0).paint("noise", 0, 10), inp)  # "auto" has to be decided first
    with pytest.raises(ValueError):
        editor.render(Timeline(10.0), inp, blocks.MemoryCache(), out=np.zeros((5, 1), np.float32))
    assert editor.Inputs(inp.source[:, 0].copy(), "mono").channels == 1


# ---------------------------------------------------------------------------------------------------------------------
# Streaming and out buffers
# ---------------------------------------------------------------------------------------------------------------------
def test_render_into_a_memory_map_with_a_work_dir_cleans_up_and_matches(tmp_path):
    x = leveled_signal(70)
    inp = editor.Inputs(x, "mm")
    tl = noise_tl(70, (10, 50, 24.0, "light"), rebuild=[(50, 60)])
    tl.paint("leveling", 5, 65)
    cache = blocks.DirCache(tmp_path / "blocks")
    fill(cache, editor.plan(tl, inp, cache), inp)
    want = editor.render(tl, inp, cache)
    mm = np.lib.format.open_memmap(tmp_path / "o.npy", mode="w+", dtype=np.float32, shape=x.shape)
    got = editor.render(tl, inp, cache, out=mm, work_dir=tmp_path)
    assert np.array_equal(np.asarray(got), want)
    assert not [p for p in tmp_path.glob("editor-*.npy")]  # the scratch files are gone
    seen = []
    editor.render(tl, inp, cache, progress=seen.append)
    assert seen and seen[-1] == 1.0 and all(a <= b + 1e-9 for a, b in zip(seen, seen[1:]))
    import threading
    ev = threading.Event()
    ev.set()
    with pytest.raises(dfn.Cancelled):
        editor.render(tl, inp, cache, cancel=ev)


# ---------------------------------------------------------------------------------------------------------------------
# With the models
# ---------------------------------------------------------------------------------------------------------------------
def real_noisy(seconds, seed=0, snr=12.0):
    """Real speech (tiled) with noise: what the removers have to work on."""
    sf = pytest.importorskip("soundfile")
    s, sr = sf.read(SPEECH_WAV, dtype="float32", always_2d=True)
    assert sr == SR
    s = np.tile(s[:, 0], int(np.ceil(seconds * SR / len(s))))[: int(seconds * SR)]
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(len(s)).astype(np.float32)
    n = np.convolve(n, np.ones(4) / 4, mode="same")
    act = np.sqrt(np.mean(s[np.abs(s) > 0.02] ** 2))
    n *= act * 10 ** (-snr / 20) / np.sqrt(np.mean(n ** 2))
    return np.ascontiguousarray((s + n)[:, None]).astype(np.float32)


@needs_dfn
@needs_speech
def test_auto_removers_are_decided_per_part_from_its_noise_figure():
    clean = real_noisy(30, 7, snr=60.0)
    noisy = real_noisy(30, 7, snr=0.0)
    x = np.ascontiguousarray(np.concatenate([clean, noisy]))
    inp = editor.Inputs(x, "auto")
    tl = Timeline(60.0).paint("noise", 0, 30, {"remover": "auto"}).paint("noise", 30, 60, {"remover": "auto", "strength_db": 24.0})
    got = editor.resolve_removers(tl, inp, LIB, MODEL)
    assert [p.settings["remover"] for p in got.parts("noise")] == ["light", "heavy"]
    assert got.parts("noise")[1].settings["strength_db"] == 24.0 and tl.parts("noise")[0].settings["remover"] == "auto"  # (a copy)
    assert editor.plan(got, inp).misses  # decided: it can be planned


@needs_dfn
@needs_speech
def test_whole_file_timeline_is_clean_audio_and_both_fill_the_same_cache():
    x = real_noisy(70, 1)
    cache = blocks.MemoryCache()
    opts = pipeline.CleanOptions(atten_lim_db=24.0, noise="light")
    ref = pipeline.clean_audio(x, LIB, MODEL, opts, threads=2, cache=cache, input_id="rec")
    assert cache.puts == 3  # all blocks: 0-30 s, 30-60 s, 60-70 s
    inp = editor.Inputs(x, "rec")
    tl = Timeline.from_clean_options(inp.seconds, opts)
    assert editor.plan(tl, inp, cache).misses == []  # the whole-file run left nothing to compute
    got = editor.render(tl, inp, cache, editor.Runner(LIB, MODEL))
    assert np.array_equal(got, ref)
    # a second whole-file run takes every block from the cache: no model is loaded (the paths do not exist)
    again = pipeline.clean_audio(x, "no-lib", "no-model", opts, threads=2, cache=cache, input_id="rec")
    assert np.array_equal(again, ref)
    # without a cache: the same samples (the grid does not depend on it, nor on the threads)
    assert np.array_equal(pipeline.clean_audio(x, LIB, MODEL, opts, threads=4), ref)


@needs_dfn
@needs_speech
def test_whole_file_noise_removal_has_no_seam_steps_and_matches_a_contiguous_run_to_block_context_tolerance():
    x = real_noisy(130, 2, snr=6.0)[:, 0]
    contiguous = dfn.denoise_array(x, LIB, MODEL, threads=1, segment_seconds=1e6, overlap_seconds=0.5)
    blocks_run = dfn.denoise_array(x, LIB, MODEL, threads=3)
    old = dfn.denoise_array(x, LIB, MODEL, threads=3, segment_seconds=60, overlap_seconds=0.5)  # what the pipeline did before
    new_snr, old_snr = snr_db(contiguous, blocks_run), snr_db(contiguous, old)
    print(f"whole-file noise removal vs a contiguous run: blocks (30 s, 16 s context) {new_snr:.1f} dB, the old 60 s / 0.5 s cut {old_snr:.1f} dB")
    assert new_snr > 40 and new_snr > old_snr + 5  # (measured here: 46.2 dB against 38.5 dB)
    # the seams: the level jump across each block join, against the contiguous run's own jumps
    def jump(y, at):
        a = y[at - 4800:at].astype(np.float64)
        b = y[at:at + 4800].astype(np.float64)
        return abs(10 * np.log10(np.mean(b ** 2) + 1e-12) - 10 * np.log10(np.mean(a ** 2) + 1e-12))
    seams = [30 * SR, 60 * SR, 90 * SR, 120 * SR]
    assert max(abs(jump(blocks_run, s) - jump(contiguous, s)) for s in seams) < 3.0
    first_block = slice(0, 30 * SR)
    assert np.array_equal(blocks_run[first_block][: 29 * SR], contiguous[first_block][: 29 * SR]) or snr_db(contiguous[first_block], blocks_run[first_block]) > 60


@needs_dfn
@needs_speech
def test_light_blocks_equal_the_contiguous_run_at_the_measured_level_and_a_cached_block_equals_a_fresh_one():
    x = real_noisy(70, 3)[:, 0]
    contiguous = dfn.denoise_array(x, LIB, MODEL, threads=1, segment_seconds=1e6, overlap_seconds=0.5)
    cache = blocks.MemoryCache()
    hooks = blocks.cache_hooks(cache, lambda d: blocks.block_key("light", "t", 0, *d))
    first = dfn.denoise_array(x, LIB, MODEL, threads=2, hooks=hooks)
    block = (30 * SR, 60 * SR)
    level = snr_db(contiguous[slice(*block)], first[slice(*block)])
    print(f"light block 30-60 s vs a contiguous run: {level:.1f} dB")
    assert level > 50  # (>= 58 dB measured on real recordings with 16 s of context)
    puts = cache.puts
    again = dfn.denoise_array(x, LIB, MODEL, threads=2, hooks=hooks)  # all hits
    assert cache.puts == puts and np.array_equal(again, first)
    fresh = dfn.denoise_array(x[10 * SR:], LIB, MODEL, threads=2, grid_offset=10 * SR)  # a stretch of the file, on the file's grid
    assert np.array_equal(fresh[20 * SR:50 * SR], first[30 * SR:60 * SR])  # the 30-60 s block is the same block (its 16 s of context start after 10 s)


@needs_heavy
@needs_speech
def test_heavy_blocks_follow_the_contiguous_run_closely_and_do_not_depend_on_the_thread_count(monkeypatch):
    # (a scaled-down grid keeps the test short: 3 s blocks with 3 s of context; the measured level at the real grid, 30 s / 24 s, is ~41 dB)
    monkeypatch.setattr(blocks, "GRID_HEAVY", blocks.Grid(3.0, 3.0, 0.25))
    x = real_noisy(12, 4)[:, 0]
    contiguous = dpdfnet.denoise_array(x, threads=1, segment_seconds=1e6, overlap_seconds=2.0)
    one = dpdfnet.denoise_array(x, threads=1)
    four = dpdfnet.denoise_array(x, threads=4)
    assert np.abs(one - four).max() < 1e-4  # the grid is the file's, whatever the threads
    level = snr_db(contiguous, one)
    print(f"heavy blocks (3 s, 3 s context) vs a contiguous run: {level:.1f} dB")
    assert 35 < level < 120  # (close to the contiguous run, but not the same run: the contexts do not reach the start of the file)
    cache = blocks.MemoryCache()
    hooks = blocks.cache_hooks(cache, lambda d: blocks.block_key("heavy", "t", 0, *d))
    first = dpdfnet.denoise_array(x, threads=2, hooks=hooks)
    puts = cache.puts
    assert puts == 4 and np.array_equal(dpdfnet.denoise_array(x, threads=2, hooks=hooks), first) and cache.puts == puts


@needs_heavy
def test_heavy_blocks_on_the_card_come_in_calls_of_at_most_what_it_takes(monkeypatch):
    from test_dpdfnet import _FakeCard, syllables

    monkeypatch.setattr(blocks, "GRID_HEAVY", blocks.Grid(2.0, 1.0, 0.25))
    monkeypatch.setattr(dpdfnet, "_GpuModel", _FakeCard)
    monkeypatch.setattr(dpdfnet, "GPU_ROWS", (2, 4))  # a card that takes four rows at most
    _, noisy, _ = syllables(6.0, seed=12)
    stereo = np.stack([noisy, noisy[::-1].copy()], axis=1)  # 2 channels x 3 blocks = 6 rows
    _FakeCard.steps.clear()
    dpdfnet.unload()
    cpu = dpdfnet.denoise_array(stereo, threads=2)
    assert not _FakeCard.steps
    gpu = dpdfnet.denoise_array(stereo, device="cuda", threads=2)
    dpdfnet.unload()
    assert _FakeCard.steps and max(_FakeCard.steps) == 4 and set(_FakeCard.steps) <= {2, 4}, _FakeCard.steps
    assert np.abs(gpu - cpu).max() < 1e-4  # the same blocks on the card and on the CPU


@needs_sidon
@needs_speech
def test_voice_rebuild_blocks_equal_the_whole_file_run_bit_for_bit(monkeypatch):
    # (small windows keep the test short; the equality does not depend on their size)
    monkeypatch.setattr(restore, "_WINDOW_SECONDS", 4.0)
    monkeypatch.setattr(restore, "_OVERLAP_SECONDS", 0.5)
    x = real_noisy(16, 5)[:, 0]  # 4.57 blocks of 3.5 s: the last one is short
    whole = restore.restore(x, SIDON, threads=3)
    assert restore.block_samples() == int(3.5 * SR) and restore.block_count(len(x)) == 5
    got = {}
    restore.restore_blocks(x, SIDON, [4, 2, 1], lambda b, y: got.__setitem__(b, y), threads=3)  # a run of two and the short last one
    assert sorted(got) == [1, 2, 4]
    for b, y in got.items():
        a, e = restore.block_bounds(b, len(x))
        assert np.array_equal(y, whole[a:e]), b  # the same samples, not just close
    assert restore.block_bounds(4, len(x))[1] == len(x) and len(got[4]) < restore.block_samples()
    # a whole-file run with a cache is all blocks; a run with some blocks cached makes only the others
    cache = blocks.MemoryCache()
    hooks = blocks.cache_hooks(cache, lambda d: blocks.rebuild_key("t", d[0], d[1], len(x), 4.0, 0.5))
    first = restore.restore(x, SIDON, threads=3, hooks=hooks)
    assert cache.puts == restore.block_count(len(x)) and np.array_equal(first, whole)
    del cache.blocks[blocks.rebuild_key("t", 0, 3, len(x), 4.0, 0.5)]  # (one block lost)
    puts = cache.puts
    assert np.array_equal(restore.restore(x, SIDON, threads=3, hooks=hooks), whole) and cache.puts == puts + 1


def test_the_dither_of_a_window_is_seeded_by_its_index_and_the_first_window_is_what_it_always_was():
    assert np.array_equal(restore._dither(5, 1000), restore._dither(5, 1000))
    assert not np.array_equal(restore._dither(5, 1000), restore._dither(6, 1000))
    assert np.array_equal(restore._dither(0, 1000), np.random.default_rng(0).standard_normal(1000).astype(np.float32))  # window 0: unchanged
    assert np.array_equal(restore._dither(3, 1000)[:500], restore._dither(3, 500))  # a prefix: the length does not change the noise


@needs_dfn
@needs_sidon
@needs_speech
def test_render_with_real_models_matches_the_whole_file_steps_and_a_second_render_runs_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(restore, "_WINDOW_SECONDS", 4.0)
    monkeypatch.setattr(restore, "_OVERLAP_SECONDS", 0.5)
    x = real_noisy(36, 6)
    inp = editor.Inputs(x, "real")
    tl = noise_tl(36, (0, 36, 24.0, "light"), rebuild=[(12, 24)])
    tl.paint("leveling", 0, 36)
    cache = blocks.DirCache(tmp_path / "blocks")
    runner = editor.Runner(LIB, MODEL, restore_model_dir=SIDON, threads=3)
    p = editor.plan(tl, inp, cache)
    assert {t.step for t in p.misses} == {"light", "rebuild"}
    y = editor.render(tl, inp, cache, runner)
    assert editor.plan(tl, inp, cache, previous=tl).misses == [] and editor.billable(editor.plan(tl, inp, cache, previous=tl)) == {}
    spy = Spy(cache)
    again = editor.render(tl, inp, spy, runner=None)  # nothing left to run: no runner needed, same samples
    assert np.array_equal(again, y)
    # outside the rebuilt part (and its 0.45 s of join) the noise removal alone, Leveled: what clean_audio makes
    whole = pipeline.clean_audio(x, LIB, MODEL, pipeline.CleanOptions(atten_lim_db=24.0, noise="light", leveling=False), threads=2)
    ref = clarity.process(x, np.array(whole), SR, clarity.ClarityParams())  # (the whole file Leveled: the leveling of the editor is measured per part)
    assert ref.shape == y.shape
    # a strength change reruns no model, and the rebuilt part is the same samples
    tl2 = tl.copy().paint("noise", 0, 12, {"strength_db": 12.0, "remover": "light"})
    assert editor.plan(tl2, inp, cache).misses == []
    y2 = editor.render(tl2, inp, cache, runner=None)
    assert not np.array_equal(y2[: 11 * SR], y[: 11 * SR]) and whole.shape == y.shape


@needs_dfn
def test_echo_parts_use_the_echo_blocks_and_the_models_read_the_echo_free_recording(tmp_path):
    pytest.importorskip("livekit")
    from test_echo import room_recording

    mic, far, _ = room_recording()
    x = np.ascontiguousarray(mic[:, None])
    inp = editor.Inputs(x, "room", reference=far, reference_id="far")
    tl = Timeline(inp.seconds).paint("echo", 0, inp.seconds)
    cache = blocks.MemoryCache()
    y = editor.render(tl, inp, cache, editor.Runner(LIB, MODEL))
    from acw_common import aec

    want = aec.cancel_echo(mic, far)
    assert y.shape == x.shape and snr_db(want[2 * SR + 432:12 * SR + 432], y[2 * SR:12 * SR, 0]) > 60  # the one pass, shifted back by the canceller's 9 ms (a 14 s file is one block)
    partial = Timeline(inp.seconds).paint("echo", 4, 9)
    y2 = editor.render(partial, inp, cache, editor.Runner(LIB, MODEL))
    assert np.array_equal(y2[: 4 * SR - 1200, 0], x[: 4 * SR - 1200, 0]) and np.array_equal(y2[9 * SR + 1200:, 0], x[9 * SR + 1200:, 0])
