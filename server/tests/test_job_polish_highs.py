"""The `polish` and `restore_highs` job options (the desktop app's "Polish the voice" and "Restore missing highs")."""
import json

import numpy as np
import pytest
import scipy.signal

from acw import jobs
from acw.billing import ledger
from acw_common.ffmpeg import FFmpeg

from .test_job_echo import NATIVE, SR
from .test_job_pipeline import HOME, _run_worker_once

KEYS = ("polish", "restore_highs")


def _opts(**extra):
    return jobs.validate_options("denoise", "standard", None, extra)


def test_both_default_on_and_accept_booleans():
    assert all(_opts()[k] is True for k in KEYS)
    for key in KEYS:
        assert _opts(**{key: False})[key] is False
        assert _opts(**{key: True})[key] is True
    off = _opts(polish=False, restore_highs=False)
    assert (off["polish"], off["restore_highs"]) == (False, False)


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("value", ["yes", 1, 0, None, [True], {"a": 1}, ""])
def test_anything_but_a_boolean_is_refused(key, value):
    with pytest.raises(jobs.JobError, match=f"{key} must be true or false"):
        _opts(**{key: value})


def create(client, h, **options):
    return client.post("/api/jobs", json={
        "kind": "denoise", "denoiseTier": "standard", "filename": "rec.wav", "sizeBytes": 1000, "durationSeconds": 14,
        "options": options,
    }, headers=h)


def test_stored_with_the_job_and_the_price_is_the_same(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 2_000_000, "grant", "t", "g")
    plain = create(client, h)
    off = create(client, h, polish=False, restore_highs=False)
    assert plain.status_code == off.status_code == 200, (plain.text, off.text)
    assert off.json()["estCostMicros"] == plain.json()["estCostMicros"] > 0
    stored = client.get(f"/api/jobs/{plain.json()['id']}").json()["options"]
    assert (stored["polish"], stored["restore_highs"]) == (True, True)
    stored = client.get(f"/api/jobs/{off.json()['id']}").json()["options"]
    assert (stored["polish"], stored["restore_highs"]) == (False, False)
    assert create(client, h, polish="no").status_code == 400
    assert create(client, h, restore_highs=0).status_code == 400


def test_estimate_leaves_out_the_highs_when_they_are_off():
    def job(**opts):
        return {"kind": "denoise", "claimed_seconds": 3600, "duration_seconds": 3600,
                "params_json": json.dumps({"denoise_tier": "standard", "channels": 1, "options": opts})}

    assert jobs.job_seconds(job(restore_highs=False)) < jobs.job_seconds(job(restore_highs=True)) == jobs.job_seconds(job())
    assert jobs.job_seconds(job(polish=False)) == jobs.job_seconds(job())  # the polish step is too small to count


def _phone_call(tmp_path, seconds=14.0):
    """A buzzy voice that stops dead at 3.4 kHz, quiet: it sounds like a phone call."""
    rng = np.random.default_rng(3)
    t = np.arange(int(seconds * SR)) / SR
    phase = 2 * np.pi * np.cumsum(110 * (1 + 0.12 * np.sin(2 * np.pi * 0.7 * t))) / SR
    env = 0.15 + 0.85 * np.clip(np.sin(2 * np.pi * 2.1 * t), 0, None) ** 2
    x = 0.04 * sum(np.sin(k * phase) / np.sqrt(k) for k in range(1, 70)) * env + 0.0005 * rng.standard_normal(len(t))
    x = scipy.signal.sosfiltfilt(scipy.signal.butter(12, 3400, fs=SR, output="sos"), x).astype(np.float32)
    return FFmpeg(HOME / "bin").write_audio(x, SR, tmp_path / "call", "wav").read_bytes()


def _run(client, h, data, **options):
    r = client.post("/api/jobs", json={
        "kind": "denoise", "denoiseTier": "standard", "filename": "rec.wav", "sizeBytes": len(data), "durationSeconds": 14,
        "options": {"output_video": False, "loudness_lufs": -16.0, "noise": "light", **options},
    }, headers=h)
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    assert client.put(f"/api/jobs/{job_id}/input", content=data, headers=h).status_code == 200
    _run_worker_once("done")
    job = client.get(f"/api/jobs/{job_id}").json()
    return job, client.get(f"/api/jobs/{job_id}/outputs/rec-cleaned.wav", headers=h).content


def _level_db(wav, tmp_path):
    path = tmp_path / "level.wav"
    path.write_bytes(wav)
    ff = FFmpeg(HOME / "bin")
    x = ff.decode(path, ff.probe(path), SR, 1, 60)[:, 0]
    return 20 * np.log10(max(float(np.sqrt(np.mean(x**2))), 1e-9))


@NATIVE
def test_the_worker_passes_both_options_on(client, signed_in, tmp_path):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    data = _phone_call(tmp_path)
    job, polished = _run(client, h, data)
    assert "high frequencies were restored" in (job["notice"] or "")
    job, as_is = _run(client, h, data, polish=False, restore_highs=False)
    assert job["notice"] is None  # no highs restored
    # the polish step pulls the voice to the loudness target; without it the level stays where the noise remover left it
    assert abs(_level_db(polished, tmp_path) - _level_db(as_is, tmp_path)) > 1.0
    job, only_polished = _run(client, h, data, restore_highs=False)
    assert job["notice"] is None
    job, only_highs = _run(client, h, data, polish=False)
    assert "high frequencies were restored" in (job["notice"] or "")
    assert abs(_level_db(only_highs, tmp_path) - _level_db(only_polished, tmp_path)) > 1.0
