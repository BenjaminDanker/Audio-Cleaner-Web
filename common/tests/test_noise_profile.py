"""Picking the noise remover per recording: light noise -> DeepFilterNet3, heavy noise -> DPDFNet-8, the strength that fits
the recording, and work only where it changes something (gated silence, channels that are the same, device routing)."""
from __future__ import annotations

import numpy as np
import pytest

from acw_common import dpdfnet, pipeline

from test_common import LIB, MODEL, needs_dfn

SR = pipeline.SAMPLE_RATE


def _speechlike(seconds: float, snr_db: float, seed: int = 0) -> np.ndarray:
    """Bursts of a harmonic "voice" (on 60 % of the time) over steady noise at the given SNR."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    voice = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((140, 280, 420, 560), 1))
    voice *= (np.sin(2 * np.pi * 1.5 * t) > -0.3)
    noise = rng.standard_normal(t.size)
    noise *= np.sqrt(np.mean(voice**2) / np.mean(noise**2)) * 10 ** (-snr_db / 20)
    return (0.1 * (voice + noise) / np.max(np.abs(voice + noise))).astype(np.float32)


def _figure(x):
    return pipeline.noise_figure_db(x, LIB, MODEL)


@needs_dfn
def test_the_figure_follows_the_noise_level():
    quiet, loud = _figure(_speechlike(6, 20)), _figure(_speechlike(6, 0))
    assert quiet > loud + 8


@needs_dfn
def test_noise_that_comes_and_goes_counts_too():
    """Clicks, gusts and chatter leave quiet moments between them: the quietest frames say nothing about them."""
    rng = np.random.default_rng(3)
    x = _speechlike(6, 40)
    steady = _figure(x)
    bursts = np.zeros_like(x)
    for at in rng.integers(0, len(x) - SR // 10, 25):  # 25 loud bursts of 50 ms
        bursts[at:at + SR // 20] += (0.05 * rng.standard_normal(SR // 20)).astype(np.float32)
    assert _figure(x + bursts) < steady - 10


@needs_dfn
def test_auto_picks_heavy_only_for_noisy_recordings_and_choices_can_be_forced():
    assert pipeline.choose_noise_profile(_speechlike(6, 30), LIB, MODEL) == "light"
    assert pipeline.choose_noise_profile(_speechlike(6, -2), LIB, MODEL) == "heavy"
    assert pipeline.choose_noise_profile(_speechlike(1, -2), "lib", "model", "light") == "light"  # nothing measured
    assert pipeline.choose_noise_profile(_speechlike(1, 30), "lib", "model", "heavy") == "heavy"


def test_a_long_stereo_recording_is_measured_from_its_mono_mix_in_six_stretches(tmp_path, monkeypatch):
    """Only the six measured stretches are read, from a memory map, and each is the mono mix of its part of the file."""
    x = _speechlike(80, 5)
    windows = pipeline.measure_windows(len(x))
    assert [(b - a) / SR for a, b in windows] == [10.0] * 6
    mm = np.lib.format.open_memmap(tmp_path / "x.npy", mode="w+", dtype=np.float32, shape=(len(x), 2))
    mm[:, 0] = mm[:, 1] = x
    got = []

    def remover(seg, lib, model, **kw):  # records what the remover is given, in place of running it
        got.append(np.array(seg))
        return seg * 0.5

    monkeypatch.setattr(pipeline.dfn, "denoise_array", remover)
    pipeline.noise_figure_db(mm, "lib", "model")
    assert [len(g) for g in got] == [10 * SR] * 6
    for g, (a, b) in zip(got, windows):
        assert np.allclose(g, x[a:b], atol=1e-7)
    assert pipeline.noise_figure_db(np.zeros(SR, np.float32), "lib", "model") == 99.0  # nothing to hear


@needs_dfn
def test_gated_silence_does_not_hide_the_noise():
    rng = np.random.default_rng(0)
    t = np.arange(2 * SR) / SR
    voice = (0.3 * np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 2 * t) > 0)).astype(np.float32)
    noisy = voice + (0.1 * rng.standard_normal(len(t))).astype(np.float32)  # about 10 dB SNR
    gated = np.concatenate([noisy, np.zeros(2 * SR, np.float32)])  # a noise gate's digital silence after it
    assert abs(_figure(gated) - _figure(noisy)) < 1


def test_strength_steps_by_noise_figure():
    assert [pipeline.strength_for(db) for db in (-5.0, 0.0, 14.9)] == [100.0, 100.0, 100.0]
    assert [pipeline.strength_for(db) for db in (15.0, 17.0, 19.99)] == [40.0, 40.0, 40.0]
    assert [pipeline.strength_for(db) for db in (20.0, 22.0, 23.9)] == [24.0, 24.0, 24.0]
    assert [pipeline.strength_for(db) for db in (24.0, 30.0, 99.0)] == [12.0, 12.0, 12.0]  # 99: a silent recording
    belows = [b for b, _ in pipeline.STRENGTH_STEPS]
    assert belows == sorted(belows)  # steps in order, or strength_for would skip one


def _fake_removers(monkeypatch, figure):
    measured, used = [], []
    monkeypatch.setattr(pipeline, "noise_figure_db", lambda x, lib, model, cancel=None: measured.append(1) or figure)
    for mod, name in ((pipeline.dfn, "dfn"), (pipeline.dpdfnet, "dpdfnet")):
        monkeypatch.setattr(mod, "denoise_array", lambda x, *a, out=None, _n=name, **k:
                            used.append((_n, k["atten_lim_db"])) or out.__setitem__(slice(None), x))
    return measured, used


@pytest.mark.parametrize(("figure", "remover", "strength"), [(5.0, "dpdfnet", 100.0), (19.5, "dfn", 40.0), (30.0, "dfn", 12.0)])
def test_automatic_strength_is_measured_once_with_the_remover_choice(monkeypatch, tmp_path, figure, remover, strength):
    measured, used = _fake_removers(monkeypatch, figure)
    report: dict = {}
    pipeline.clean_audio(_speechlike(2, 10), "lib", "model", pipeline.CleanOptions(noise="auto", atten_lim_db=None, leveling=False),
                         heavy_model_path=tmp_path / dpdfnet.MODEL_FILE, report=report)
    assert len(measured) == 1 and used == [(remover, strength)]
    assert report == {"noise": "heavy" if remover == "dpdfnet" else "light", "highs": None,
                      "atten_lim_db": strength, "noise_db": figure}


def test_automatic_strength_with_a_forced_remover_still_measures_once(monkeypatch, tmp_path):
    measured, used = _fake_removers(monkeypatch, 21.234)
    report: dict = {}
    pipeline.clean_audio(_speechlike(2, 10), "lib", "model", pipeline.CleanOptions(noise="heavy", atten_lim_db=None, leveling=False),
                         heavy_model_path=tmp_path / dpdfnet.MODEL_FILE, report=report)
    assert len(measured) == 1 and used == [("dpdfnet", 24.0)]
    assert report["noise"] == "heavy" and report["atten_lim_db"] == 24.0 and report["noise_db"] == 21.2  # rounded


def test_a_chosen_remover_and_strength_measure_nothing(monkeypatch):
    measured, used = _fake_removers(monkeypatch, 5.0)
    report: dict = {}
    pipeline.clean_audio(_speechlike(2, 10), "lib", "model", pipeline.CleanOptions(noise="light", atten_lim_db=40.0, leveling=False),
                         report=report)
    assert measured == [] and used == [("dfn", 40.0)]
    assert report == {"noise": "light", "highs": None, "atten_lim_db": 40.0}  # no noise_db: nothing was measured


def test_every_channel_gets_the_one_strength(monkeypatch):
    """A stereo recording is measured once (its mono mix) and both sides are cleaned with the same strength."""
    measured, used = _fake_removers(monkeypatch, 18.0)
    x = np.stack([_speechlike(2, 10, seed=1), _speechlike(2, 10, seed=2)], 1)
    pipeline.clean_audio(x, "lib", "model", pipeline.CleanOptions(noise="light", atten_lim_db=None, leveling=False))
    assert len(measured) == 1 and used == [("dfn", 40.0), ("dfn", 40.0)]


def test_device_reaches_every_stage_that_has_a_gpu_path(monkeypatch, tmp_path):
    """CleanOptions.device goes to Sidon, AP-BWE and DPDFNet; DeepFilterNet3 has no device."""
    from scipy.signal import butter, sosfilt

    seen = {}
    monkeypatch.setattr(pipeline.dfn, "denoise_array", lambda x, *a, out=None, **k: seen.setdefault("dfn", k) and out.__setitem__(slice(None), x))
    monkeypatch.setattr(pipeline.dpdfnet, "denoise_array", lambda x, *a, out=None, **k: seen.__setitem__("dpdfnet", k["device"]) or out.__setitem__(slice(None), x))
    monkeypatch.setattr(pipeline.bwe, "available", lambda d: True)
    monkeypatch.setattr(pipeline.bwe, "extend", lambda y, d, **k: seen.__setitem__("bwe", k["device"]) or y)
    monkeypatch.setattr(pipeline.restore, "available", lambda d: True)
    monkeypatch.setattr(pipeline.restore, "restore", lambda x, d, **k: seen.__setitem__("sidon", k["device"]) or x.copy())
    heavy = tmp_path / dpdfnet.MODEL_FILE
    heavy.write_bytes(b"x")
    full = _speechlike(4, 30) + (0.01 * np.random.default_rng(1).standard_normal(4 * SR)).astype(np.float32)
    phone = sosfilt(butter(10, 3400, fs=SR, output="sos"), full).astype(np.float32)

    def run(x, opts):
        seen.clear()
        pipeline.clean_audio(x, "lib", "model", opts, heavy_model_path=heavy, bwe_model_dir=tmp_path, restore_model_dir=tmp_path)
        return dict(seen)

    assert run(phone, pipeline.CleanOptions(noise="heavy", leveling=False)) == {"dpdfnet": "cpu", "bwe": "cpu"}
    assert run(phone, pipeline.CleanOptions(noise="heavy", leveling=False, device="cuda")) == {"dpdfnet": "cuda", "bwe": "cuda"}
    assert run(full, pipeline.CleanOptions(restore_voice=True, leveling=False, device="cuda")) == {"sidon": "cuda"}
    assert "device" not in run(full, pipeline.CleanOptions(noise="light", leveling=False, device="cuda")).get("dfn", {})


def test_band_limited_recordings_get_their_highs_restored_full_band_ones_never(monkeypatch, tmp_path):
    from scipy.signal import butter, sosfilt

    monkeypatch.setattr(pipeline.dfn, "denoise_array", lambda x, *a, out=None, **k: out.__setitem__(slice(None), x))
    extended = []
    monkeypatch.setattr(pipeline.bwe, "available", lambda d: True)
    monkeypatch.setattr(pipeline.bwe, "extend", lambda y, d, **k: extended.append(k["model"]) or y)
    full = _speechlike(4, 30) + (0.01 * np.random.default_rng(1).standard_normal(4 * SR)).astype(np.float32)
    phone = sosfilt(butter(10, 3400, fs=SR, output="sos"), full).astype(np.float32)
    for x, want in ((full, None), (phone, "8k")):
        report: dict = {}
        pipeline.clean_audio(x, "lib", "model", pipeline.CleanOptions(noise="light", leveling=False),
                             bwe_model_dir=tmp_path, report=report)
        assert report["highs"] == want
    assert extended == ["8k"]
    report = {}
    pipeline.clean_audio(phone, "lib", "model", pipeline.CleanOptions(noise="light", leveling=False, restore_highs=False),
                         bwe_model_dir=tmp_path, report=report)
    assert report["highs"] is None


def test_restore_voice_replaces_noise_removal_and_highs(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(pipeline.dfn, "denoise_array", lambda x, *a, out=None, **k: calls.append("dfn") or out.__setitem__(slice(None), x))
    monkeypatch.setattr(pipeline.restore, "restore", lambda x, d, **k: calls.append("sidon") or x.copy())
    x = _speechlike(2, 10)
    report: dict = {}
    with pytest.raises(FileNotFoundError):  # asked for and not installed: an error, never a cheaper chain
        pipeline.clean_audio(x, "lib", "model", pipeline.CleanOptions(restore_voice=True, leveling=False),
                             restore_model_dir=tmp_path, report=report)
    monkeypatch.setattr(pipeline.restore, "available", lambda d: True)
    y = pipeline.clean_audio(x, "lib", "model", pipeline.CleanOptions(restore_voice=True, leveling=False),
                             restore_model_dir=tmp_path, report=report)
    assert calls == ["sidon"] and report == {"noise": "rebuilt", "highs": None} and y.shape == (len(x), 1)


def test_channel_plan():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(48_000).astype(np.float32) * 0.1
    b = rng.standard_normal(48_000).astype(np.float32) * 0.1
    assert pipeline.channel_plan(np.stack([a, b], 1)) == [0, 1]
    assert pipeline.channel_plan(np.stack([a, a], 1)) == [0, 0]
    assert pipeline.channel_plan(np.stack([a, a + 0.001 * b], 1)) == [0, 0]  # -40 dB of difference and less
    assert pipeline.channel_plan(np.stack([a, a + 0.1 * b], 1)) == [0, 1]
    assert pipeline.channel_plan(np.stack([a, np.zeros_like(a)], 1)) == [0, None]
    assert pipeline.channel_plan(a[:, None]) == [0]


def test_sound_spans_skip_long_digital_silence_only():
    sr = pipeline.SAMPLE_RATE
    sound = np.full(2 * sr, 0.1, np.float32)
    x = np.concatenate([np.zeros(3 * sr), sound, np.zeros(sr), sound, np.zeros(4 * sr), sound]).astype(np.float32)
    ctx = int(pipeline.SKIP_CONTEXT_S * sr)
    assert pipeline.sound_spans(x) == [(3 * sr - ctx, 8 * sr + ctx), (12 * sr - ctx, len(x))]  # the 1 s pause stays in
    hiss = np.full(4 * sr, 1e-3, np.float32)  # room noise is not silence: it is denoised as before
    assert pipeline.sound_spans(np.concatenate([sound, hiss, sound])) == [(0, 8 * sr)]


def test_language_detection_listens_to_speech_not_a_silent_intro():
    from acw_common import transcribe

    sr = transcribe.SAMPLE_RATE
    speech = np.full(40 * sr, 0.1, np.float32)
    x = np.concatenate([np.zeros(35 * sr, np.float32), speech])
    got = transcribe.speech_sample(x)
    assert len(got) == 30 * sr and np.all(got == 0.1)
    assert len(transcribe.speech_sample(np.zeros(5 * sr, np.float32))) == 5 * sr  # nothing loud: the start, as before
