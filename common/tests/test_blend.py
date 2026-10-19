"""Strength is a blend: a noise model run at an attenuation limit of X dB gives (to float precision) the blend
(1 - a) * full + a * recording of its own full-strength output with the recording, a = 10 ** (-X / 20). The pipeline therefore runs
the models at full strength only and blends afterwards (pipeline.blend_strength), which lets a job keep the full-strength output and
change strength and leveling later without a model run (pipeline.redo_denoised). Every path of clean_audio is checked
against the model's own limit: light (DeepFilterNet3) and heavy (DPDFNet-8), a short recording and one cut into segments, spans around
digital silence, stereo, dual-mono. The SNR of the difference is above 100 dB on all of them (measured 138 dB light, 147 dB heavy)."""
from __future__ import annotations

import numpy as np
import pytest

from acw_common import dfn, dpdfnet, kept, pipeline
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


def test_blend_strength_works_in_place_on_the_chosen_channels_and_does_nothing_at_full_strength():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5000, 2)).astype(np.float32)
    y = rng.standard_normal((5000, 2)).astype(np.float32)
    kept_y = y.copy()
    assert pipeline.blend_strength(y, x, 100.0) is y and np.array_equal(y, kept_y)
    out = pipeline.blend_strength(y, x, 24.0, channels=[1])
    assert out is y and np.array_equal(y[:, 0], kept_y[:, 0])  # channel 0 was not asked for
    assert np.allclose(y[:, 1], blend(kept_y[:, 1], x[:, 1], 24.0), atol=1e-7)


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


# ---------------------------------------------------------------- every path of clean_audio

def _clean(x, db, noise, **kw):
    opts = pipeline.CleanOptions(atten_lim_db=db, leveling=False, noise=noise)
    return pipeline.clean_audio(x, LIB, MODEL, opts, threads=2, **kw)


def _per_channel_reference(x, db, remover):
    """What the pipeline gave before the strength was a blend: the model run at the limit on every channel, digital silence left silent."""
    out = np.zeros_like(x)
    for c in range(x.shape[1]):
        for a, b in pipeline.sound_spans(x[:, c]):
            out[a:b, c] = remover(x[a:b, c], db)
    return out


@needs_dfn
def test_clean_audio_light_mono_stereo_and_dual_mono_equal_the_model_at_that_limit():
    db = 24.0
    mono = _speechlike(6, 5)
    stereo = np.stack([mono, _speechlike(6, 5, seed=2)], 1)
    dual = np.stack([mono, mono], 1)
    for x in (mono[:, None], stereo, dual):
        got = _clean(x, db, "light")
        assert snr_db(_per_channel_reference(x, db, _light), got) > 100


@needs_dfn
def test_clean_audio_light_around_digital_silence_equals_the_model_at_that_limit():
    """Silent stretches are not run through the model; they stay silent (the recording there is zeros, so is its blend)."""
    x = _speechlike(10, 5)
    x = np.concatenate([x[: 4 * SR], np.zeros(3 * SR, np.float32), x[4 * SR:]])[:, None]
    assert pipeline.sound_spans(x[:, 0]) != [(0, len(x))]
    got = _clean(x, 24.0, "light")
    assert snr_db(_per_channel_reference(x, 24.0, _light), got) > 100
    assert not np.any(got[4 * SR + SR: 6 * SR + SR // 2])  # the middle of the gap is exactly silent


@needs_heavy
@needs_dfn
def test_clean_audio_heavy_mono_and_stereo_equal_the_model_at_that_limit(tmp_path):
    mono = _speechlike(8, 5)
    stereo = np.stack([mono, _speechlike(8, 5, seed=2)], 1)
    for x in (mono[:, None], stereo):
        got = _clean(x, 24.0, "heavy")
        assert snr_db(_per_channel_reference(x, 24.0, _heavy), got) > 100


@needs_dfn
def test_a_weaker_strength_needs_no_extra_model_run_for_the_automatic_strength(monkeypatch):
    """atten_lim_db=None: measured once, the strength that fits the file is the blend of the one full-strength run."""
    from acw_common import analysis, reco_model

    x = _speechlike(6, 10)[:, None]
    report: dict = {}
    got = pipeline.clean_audio(x, LIB, MODEL, pipeline.CleanOptions(atten_lim_db=None, leveling=False, noise="light"), threads=1, report=report)
    stats = analysis.measure(x, LIB, MODEL)  # the same measurements, on their own: the strength is what the model says about them
    assert report["atten_lim_db"] == reco_model.STRENGTH_DB[reco_model.recommend(stats, min_seconds=0.0).strength]
    assert snr_db(_per_channel_reference(x, report["atten_lim_db"], _light), got) > 100


# ---------------------------------------------------------------- redoing the later steps from the kept output

@needs_dfn
@pytest.mark.parametrize("leveling", (False, True))
def test_redoing_from_the_kept_full_strength_output_is_the_same_as_cleaning_again(leveling):
    """The kept copy plus a new strength (and Leveling) gives, bit for bit, what a whole new run with that strength gives."""
    x = np.stack([_speechlike(6, 8), _speechlike(6, 8, seed=2)], 1)
    keep: dict = {}

    def hold(name, y, todo):
        keep[name] = (y.copy(), todo)

    first = pipeline.clean_audio(x, LIB, MODEL, pipeline.CleanOptions(atten_lim_db=100.0, leveling=leveling, noise="light"), threads=1, keep=hold)
    full, todo = keep["denoised"]
    assert todo == [0, 1] and first.shape == x.shape
    for db in (12.0, 40.0):
        opts = pipeline.CleanOptions(atten_lim_db=db, leveling=leveling, noise="light")
        again = pipeline.clean_audio(x, LIB, MODEL, opts, threads=1)
        redone = pipeline.redo_denoised(x, full.copy(), pipeline.channel_plan(x), opts)
        assert np.array_equal(again, redone)


@needs_dfn
def test_redoing_with_the_two_sides_the_same_fills_the_copy_after_the_blend():
    mono = _speechlike(5, 8)
    x = np.stack([mono, mono], 1)
    keep: dict = {}
    pipeline.clean_audio(x, LIB, MODEL, pipeline.CleanOptions(leveling=False, noise="light"), threads=1, keep=lambda n, y, t: keep.update({n: (y.copy(), t)}))
    full, todo = keep["denoised"]
    assert todo == [0] and pipeline.channel_plan(x) == [0, 0]
    opts = pipeline.CleanOptions(atten_lim_db=24.0, leveling=False, noise="light")
    redone = pipeline.redo_denoised(x, full.copy(), pipeline.channel_plan(x), opts)
    assert np.array_equal(redone[:, 0], redone[:, 1]) and np.array_equal(redone, pipeline.clean_audio(x, LIB, MODEL, opts, threads=1))


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
