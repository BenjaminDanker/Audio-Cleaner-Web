"""Picking the noise remover per recording: light noise -> DeepFilterNet3, heavy noise -> DPDFNet-8."""
from __future__ import annotations

import numpy as np

from acw_common import dpdfnet, pipeline

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


def test_the_estimate_follows_the_noise_level():
    quiet, loud = pipeline.noise_estimate_db(_speechlike(10, 20)), pipeline.noise_estimate_db(_speechlike(10, 0))
    assert quiet > loud + 8


def test_auto_picks_heavy_only_for_noisy_recordings_and_choices_can_be_forced():
    assert pipeline.choose_noise_profile(_speechlike(10, 25)) == "light"
    assert pipeline.choose_noise_profile(_speechlike(10, -2)) == "heavy"
    assert pipeline.choose_noise_profile(_speechlike(10, -2), "light") == "light"
    assert pipeline.choose_noise_profile(_speechlike(10, 25), "heavy") == "heavy"


def test_the_estimate_reads_long_memmapped_audio_in_blocks(tmp_path):
    x = _speechlike(130, 5)
    mm = np.lib.format.open_memmap(tmp_path / "x.npy", mode="w+", dtype=np.float32, shape=(len(x), 2))
    mm[:, 0] = mm[:, 1] = x
    assert abs(pipeline.noise_estimate_db(mm) - pipeline.noise_estimate_db(x)) < 1.0


def test_heavy_falls_back_to_deepfilternet_without_its_model(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline.dfn, "denoise_array", lambda x, *a, out=None, **k: calls.append("dfn") or out.__setitem__(slice(None), x))
    monkeypatch.setattr(pipeline.dpdfnet, "denoise_array", lambda x, *a, out=None, **k: calls.append("dpdfnet") or out.__setitem__(slice(None), x))
    report: dict = {}
    pipeline.clean_audio(_speechlike(2, 0), "lib", "model", pipeline.CleanOptions(noise="heavy", polish=False),
                         heavy_model_path=tmp_path / "missing.onnx", report=report)
    assert calls == ["dfn"] and report == {"noise": "light", "highs": None}
    calls.clear()
    present = tmp_path / dpdfnet.MODEL_FILE
    present.write_bytes(b"x")
    pipeline.clean_audio(_speechlike(2, 0), "lib", "model", pipeline.CleanOptions(noise="auto", polish=False),
                         heavy_model_path=present, report=report)
    assert calls == ["dpdfnet"] and report == {"noise": "heavy", "highs": None}



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
        pipeline.clean_audio(x, "lib", "model", pipeline.CleanOptions(noise="light", polish=False),
                             bwe_model_dir=tmp_path, report=report)
        assert report["highs"] == want
    assert extended == ["8k"]
    report = {}
    pipeline.clean_audio(phone, "lib", "model", pipeline.CleanOptions(noise="light", polish=False, restore_highs=False),
                         bwe_model_dir=tmp_path, report=report)
    assert report["highs"] is None


def test_restore_voice_replaces_noise_removal_and_highs_when_its_models_are_there(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(pipeline.dfn, "denoise_array", lambda x, *a, out=None, **k: calls.append("dfn") or out.__setitem__(slice(None), x))
    monkeypatch.setattr(pipeline.restore, "restore", lambda x, d, **k: calls.append("sidon") or x.copy())
    x = _speechlike(2, 10)
    report: dict = {}
    monkeypatch.setattr(pipeline.restore, "available", lambda d: False)
    pipeline.clean_audio(x, "lib", "model", pipeline.CleanOptions(restore_voice=True, noise="light", polish=False),
                         restore_model_dir=tmp_path, report=report)
    assert calls == ["dfn"] and report["noise"] == "light"  # no models: the usual chain
    calls.clear()
    monkeypatch.setattr(pipeline.restore, "available", lambda d: True)
    y = pipeline.clean_audio(x, "lib", "model", pipeline.CleanOptions(restore_voice=True, polish=False),
                             restore_model_dir=tmp_path, report=report)
    assert calls == ["sidon"] and report == {"noise": "rebuilt", "highs": None} and y.shape == (len(x), 1)


# --- work only where it changes something (2026-10-16 efficiency audit)

def test_gated_silence_does_not_hide_the_noise():
    from acw_common import pipeline

    rng = np.random.default_rng(0)
    sr = pipeline.SAMPLE_RATE
    t = np.arange(4 * sr) / sr
    voice = (0.3 * np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 2 * t) > 0)).astype(np.float32)
    noisy = voice + (0.1 * rng.standard_normal(len(t))).astype(np.float32)  # about 10 dB SNR
    gated = np.concatenate([noisy, np.zeros(3 * sr, np.float32)])  # a noise gate's digital silence after it
    assert abs(pipeline.noise_estimate_db(gated) - pipeline.noise_estimate_db(noisy)) < 3


def test_channel_plan():
    from acw_common import pipeline

    rng = np.random.default_rng(1)
    a = rng.standard_normal(48_000).astype(np.float32) * 0.1
    b = rng.standard_normal(48_000).astype(np.float32) * 0.1
    assert pipeline.channel_plan(np.stack([a, b], 1)) == [0, 1]
    assert pipeline.channel_plan(np.stack([a, a], 1)) == [0, 0]
    assert pipeline.channel_plan(np.stack([a, a + 0.001 * b], 1)) == [0, 0]  # -40 dB of difference and less
    assert pipeline.channel_plan(np.stack([a, a + 0.1 * b], 1)) == [0, 1]
    assert pipeline.channel_plan(np.stack([a, np.zeros_like(a)], 1)) == [0, None]
    assert pipeline.channel_plan(np.stack([np.zeros_like(a), b], 1)) == [None, 1]
    assert pipeline.channel_plan(a[:, None]) == [0]


def test_sound_spans_skip_long_digital_silence_only():
    from acw_common import pipeline

    sr = pipeline.SAMPLE_RATE
    sound = np.full(2 * sr, 0.1, np.float32)
    x = np.concatenate([np.zeros(3 * sr), sound, np.zeros(sr), sound, np.zeros(4 * sr), sound]).astype(np.float32)
    ctx = int(pipeline.SKIP_CONTEXT_S * sr)
    assert pipeline.sound_spans(x) == [(3 * sr - ctx, 8 * sr + ctx), (12 * sr - ctx, len(x))]  # the 1 s pause stays in
    assert pipeline.sound_spans(sound) == [(0, len(sound))]
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
