"""Strength is a blend: a noise model run at an attenuation limit of X dB gives (to float precision) the blend
(1 - a) * full + a * recording of its own full-strength output with the recording, a = 10 ** (-X / 20). The pipeline therefore runs
the models at full strength only and blends afterwards (editor.render of the timeline), which lets a job keep the full-strength blocks
and change strength and leveling later without a model run. Every path of the render is checked
against the model's own limit: light (DeepFilterNet3) and heavy (DPDFNet-8), a short recording and one cut into segments, spans around
digital silence, stereo, dual-mono. The SNR of the difference is above 100 dB on all of them (measured 138 dB light, 147 dB heavy)."""
from __future__ import annotations

import numpy as np
import pytest

from acw_common import blocks, dfn, dpdfnet, editor, kept, pipeline
from acw_common.timeline import Timeline
from acw_common.ffmpeg import FFmpeg

from test_common import FF_DIR, LIB, MODEL, needs_dfn, needs_ff
from test_noise_profile import _speechlike

SR = pipeline.SAMPLE_RATE
STRENGTHS = (12.0, 40.0)
needs_heavy = pytest.mark.skipif(not dpdfnet.default_model_path().is_file(), reason="DPDFNet ONNX model not available")


def snr_db(reference: np.ndarray, test: np.ndarray) -> float:
    err = float(np.sum((reference.astype(np.float64) - test.astype(np.float64)) ** 2))
    return 300.0 if err == 0 else 10 * np.log10(float(np.sum(reference.astype(np.float64) ** 2)) / err)


def blend(full: np.ndarray, x: np.ndarray, db: float) -> np.ndarray:
    a = 10 ** (-db / 20)
    return ((1 - a) * full.astype(np.float64) + a * x.astype(np.float64)).astype(np.float32)


def _light(x, db, threads=2):
    return dfn.denoise_array(np.ascontiguousarray(x), LIB, MODEL, atten_lim_db=db, threads=threads)


def _heavy(x, db, threads=2):
    return dpdfnet.denoise_array(np.ascontiguousarray(x), atten_lim_db=db, threads=threads)


# ---------------------------------------------------------------- the arithmetic

def test_strength_mix_is_the_share_of_the_recording_that_stays():
    assert pipeline.strength_mix(100.0) == 0.0 and pipeline.strength_mix(120.0) == 0.0
    assert pipeline.strength_mix(20.0) == pytest.approx(0.1) and pipeline.strength_mix(40.0) == pytest.approx(0.01)
    assert pipeline.strength_mix(0.0) == 1.0 and pipeline.strength_mix(-20.0) == pytest.approx(0.1)  # a limit is a size, not a sign


def test_automatic_choices_are_one_function():
    """With only a noise figure at hand (no recording to measure) the rule decides the strength; with the full measurements the
    model does (tests/test_reco_model.py)."""
    assert pipeline.automatic_choices(5.0) == ("heavy", 100.0)
    assert pipeline.automatic_choices(19.5) == ("light", 40.0)
    assert pipeline.automatic_choices(30.0) == ("light", 12.0)
    assert [pipeline.automatic_choices(f)[1] for f in (10.0, 17.0, 22.0, 40.0)] == [pipeline.strength_for(f) for f in (10.0, 17.0, 22.0, 40.0)]


# ---------------------------------------------------------------- the models' own limit is that blend

@pytest.fixture(scope="module")
def short():
    x = _speechlike(10, 5)
    return x, _light(x, 100.0)


@needs_dfn
@pytest.mark.parametrize("db", STRENGTHS)
def test_light_remover_at_a_limit_is_the_blend_of_its_full_strength_output(short, db):
    x, full = short
    assert snr_db(_light(x, db), blend(full, x, db)) > 100


@needs_dfn
def test_light_remover_cut_into_segments_follows_the_same_rule():
    """More than 61 s, several threads: segments with warm-up context and crossfades (a linear crossfade of blends is the blend)."""
    x = _speechlike(64, 5)
    full = _light(x, 100.0, threads=4)
    assert snr_db(_light(x, 24.0, threads=4), blend(full, x, 24.0)) > 100


@needs_heavy
def test_heavy_remover_at_a_limit_is_the_blend_of_its_full_strength_output():
    x = _speechlike(8, 5)
    full = _heavy(x, 100.0)
    for db in (12.0, 40.0):
        assert snr_db(_heavy(x, db), blend(full, x, db)) > 100


@needs_heavy
def test_heavy_remover_cut_into_segments_follows_the_same_rule():
    x = _speechlike(14, 5)  # with several threads a recording over 12 s is cut into 10 s segments
    full = _heavy(x, 100.0, threads=4)
    assert snr_db(_heavy(x, 24.0, threads=4), blend(full, x, 24.0)) > 100


# ---------------------------------------------------------------- every path of the render

def _clean(x, db, noise, cache=None, runner=None):
    """The whole file at strength `db` with the remover `noise`: the timeline those choices stand for, rendered."""
    opts = pipeline.CleanOptions(atten_lim_db=db, leveling=False, noise=noise)
    inp = editor.Inputs(np.ascontiguousarray(x, dtype=np.float32), "t")
    return editor.render(Timeline.from_clean_options(inp.seconds, opts), inp, cache, runner or editor.Runner(LIB, MODEL, threads=2))


def _per_channel_reference(x, db, remover):
    """What the models give on their own: the model run at the limit on every channel, digital silence left silent."""
    out = np.zeros_like(x)
    for c in range(x.shape[1]):
        for a, b in pipeline.sound_spans(x[:, c]):
            out[a:b, c] = remover(x[a:b, c], db)
    return out


@needs_dfn
def test_light_render_mono_stereo_and_dual_mono_equal_the_model_at_that_limit():
    db = 24.0
    mono = _speechlike(6, 5)
    stereo = np.stack([mono, _speechlike(6, 5, seed=2)], 1)
    dual = np.stack([mono, mono], 1)
    for x in (mono[:, None], stereo, dual):
        got = _clean(x, db, "light")
        assert snr_db(_per_channel_reference(x, db, _light), got) > 100


@needs_dfn
def test_light_render_around_digital_silence_equals_the_model_at_that_limit():
    """Silent stretches are not run through the model; they stay silent (the recording there is zeros, so is its blend)."""
    x = _speechlike(10, 5)
    x = np.concatenate([x[: 4 * SR], np.zeros(3 * SR, np.float32), x[4 * SR:]])[:, None]
    assert pipeline.sound_spans(x[:, 0]) != [(0, len(x))]
    got = _clean(x, 24.0, "light")
    assert snr_db(_per_channel_reference(x, 24.0, _light), got) > 100
    assert not np.any(got[4 * SR + SR: 6 * SR + SR // 2])  # the middle of the gap is exactly silent


@needs_heavy
@needs_dfn
def test_heavy_render_mono_and_stereo_equal_the_model_at_that_limit(tmp_path):
    mono = _speechlike(8, 5)
    stereo = np.stack([mono, _speechlike(8, 5, seed=2)], 1)
    for x in (mono[:, None], stereo):
        got = _clean(x, 24.0, "heavy")
        assert snr_db(_per_channel_reference(x, 24.0, _heavy), got) > 100


# ---------------------------------------------------------------- a new strength from the cached blocks

@needs_dfn
@pytest.mark.parametrize("leveling", (False, True))
def test_a_new_strength_from_the_cached_blocks_is_the_same_as_rendering_again_from_nothing(leveling):
    """The blocks of the first render plus a new strength (and Leveling) give, bit for bit, what a whole new render with that strength gives,
    and no model runs for it."""
    x = np.stack([_speechlike(6, 8), _speechlike(6, 8, seed=2)], 1)
    inp = editor.Inputs(x, "t")
    cache = blocks.MemoryCache()
    runner = editor.Runner(LIB, MODEL, threads=1)

    def render(db, cache=None, runner=runner):
        opts = pipeline.CleanOptions(atten_lim_db=db, leveling=leveling, noise="light")
        return editor.render(Timeline.from_clean_options(inp.seconds, opts), inp, cache, runner)

    first = render(100.0, cache)
    assert first.shape == x.shape and cache.puts > 0
    puts = cache.puts
    for db in (12.0, 40.0):
        redone = render(db, cache, runner=None)  # no runner: any block still to make would fail
        assert np.array_equal(render(db), redone)
    assert cache.puts == puts


@needs_dfn
def test_a_render_with_the_two_sides_the_same_makes_one_channel_and_copies_it():
    mono = _speechlike(5, 8)
    x = np.stack([mono, mono], 1)
    assert pipeline.channel_plan(x) == [0, 0]
    got = _clean(x, 24.0, "light")
    assert np.array_equal(got[:, 0], got[:, 1]) and snr_db(_light(mono, 24.0), got[:, 0]) > 100


# ---------------------------------------------------------------- the 24-bit copies

@needs_ff
def test_a_kept_copy_comes_back_to_one_24_bit_step(tmp_path):
    ff = FFmpeg(FF_DIR)
    rng = np.random.default_rng(1)
    y = (0.1 * rng.standard_normal((SR * 2, 2))).astype(np.float32)
    entry = kept.save(ff, y, None, tmp_path / "kept-noise")
    assert entry["channels"] == [0, 1] and entry["scale"] == 1.0 and entry["samples"] == len(y)
    out = np.zeros_like(y)
    kept.load(ff, entry, tmp_path, out, tmp_path / "scratch.f32")
    assert float(np.max(np.abs(out - y))) < 2e-7
    assert not (tmp_path / "scratch.f32").exists()


def test_the_energy_sums_give_the_loudness_of_every_blend_over_the_kept_channels():
    """E(a) = (1 - a)^2 ef + a^2 ei + 2 a (1 - a) cfi is the energy of (1 - a) * full + a * input, for all kept channels, in blocks."""
    rng = np.random.default_rng(3)
    n = kept.BLOCK * 2 + 12_345  # more than two blocks, so the loop goes round
    full = (0.2 * rng.standard_normal((n, 2))).astype(np.float32)
    heard = (full + 0.1 * rng.standard_normal((n, 2))).astype(np.float32)
    for cols in (None, [0], [1], [0, 1]):
        e = kept.energy(full, heard, cols)
        use = slice(None) if cols is None else cols
        f, x = full[:, use].astype(np.float64), heard[:, use].astype(np.float64)
        assert abs(e["full"] - np.sum(f * f)) <= 1e-9 * np.sum(f * f)
        assert abs(e["input"] - np.sum(x * x)) <= 1e-9 * np.sum(x * x)
        assert abs(e["cross"] - np.sum(f * x)) <= 1e-9 * np.sum(f * x)
        for a in (0.0, 0.063, 0.25, 1.0):
            mixed = (1 - a) * f + a * x
            closed = (1 - a) ** 2 * e["full"] + a * a * e["input"] + 2 * a * (1 - a) * e["cross"]
            assert abs(closed - np.sum(mixed * mixed)) <= 1e-8 * np.sum(mixed * mixed) + 1e-9
    mono = kept.energy(full[:, 0], heard[:, 0])  # one-dimensional arrays are one channel
    assert abs(mono["full"] - kept.energy(full, heard, [0])["full"]) < 1e-9
    short = kept.energy(full, heard[:1000], [0])  # the shorter of the two decides
    assert abs(short["input"] - np.sum(heard[:1000, 0].astype(np.float64) ** 2)) < 1e-9


@needs_ff
def test_a_loud_copy_is_scaled_down_with_the_file_and_restored_and_one_channel_can_be_kept(tmp_path):
    ff = FFmpeg(FF_DIR)
    rng = np.random.default_rng(2)
    y = (1.6 * rng.uniform(-1, 1, (SR, 2))).astype(np.float32)  # louder than full scale: a float result before the limiter
    entry = kept.save(ff, y, [1], tmp_path / "kept-noise")
    assert entry["channels"] == [1] and entry["scale"] < 1.0
    out = np.zeros_like(y)
    kept.load(ff, entry, tmp_path, out, tmp_path / "scratch.f32")
    assert np.all(out[:, 0] == 0) and float(np.max(np.abs(out[:, 1] - y[:, 1]))) < 1e-6
