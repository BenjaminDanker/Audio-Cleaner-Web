"""The `leveling` and `restore_highs` job options (the desktop app's "Leveling" and "Restore missing highs")."""
import json

import numpy as np
import pytest
import scipy.signal

from acw import db, jobs
from acw.billing import ledger, pricing
from acw_common.ffmpeg import FFmpeg

from .test_job_echo import NATIVE, SR
from .test_job_pipeline import HOME, _run_worker_once

KEYS = ("leveling", "restore_highs")


def _opts(**extra):
    return jobs.validate_options("denoise", "standard", None, extra)


def test_both_default_on_and_only_booleans_are_taken():
    assert all(_opts()[k] is True for k in KEYS)
    assert all(_opts(**{k: False})[k] is False for k in KEYS)
    for key in KEYS:
        for value in ("yes", 0, None):
            with pytest.raises(jobs.JobError, match=f"{key} must be true or false"):
                _opts(**{key: value})


def create(client, h, **options):
    return client.post("/api/jobs", json={
        "kind": "denoise", "denoiseTier": "standard", "filename": "rec.wav", "sizeBytes": 1000, "durationSeconds": 14,
        "options": options,
    }, headers=h)


def test_stored_with_the_job_and_leveling_is_its_own_line_of_the_price(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 2_000_000, "grant", "t", "g")
    plain = create(client, h)
    off = create(client, h, leveling=False, restore_highs=False)
    assert plain.status_code == off.status_code == 200, (plain.text, off.text)
    # Leveling is its own line of the price: switching it off takes exactly that line away
    assert off.json()["estCostMicros"] == pricing.cost_micros("denoise", "standard", 14) > 0
    assert plain.json()["estCostMicros"] == off.json()["estCostMicros"] + pricing.cost_micros("leveling", "standard", 14)
    stored = client.get(f"/api/jobs/{plain.json()['id']}").json()["options"]
    assert (stored["leveling"], stored["restore_highs"]) == (True, True)
    stored = client.get(f"/api/jobs/{off.json()['id']}").json()["options"]
    assert (stored["leveling"], stored["restore_highs"]) == (False, False)
    assert create(client, h, leveling="no").status_code == 400


def _phone_call(tmp_path, seconds=14.0):
    """A buzzy voice that stops dead at 3.4 kHz, quiet: it sounds like a phone call."""
    rng = np.random.default_rng(3)
    t = np.arange(int(seconds * SR)) / SR
    phase = 2 * np.pi * np.cumsum(110 * (1 + 0.12 * np.sin(2 * np.pi * 0.7 * t))) / SR
    env = 0.15 + 0.85 * np.clip(np.sin(2 * np.pi * 2.1 * t), 0, None) ** 2
    x = 0.04 * sum(np.sin(k * phase) / np.sqrt(k) for k in range(1, 70)) * env + 0.0005 * rng.standard_normal(len(t))
    x = scipy.signal.sosfiltfilt(scipy.signal.butter(12, 3400, fs=SR, output="sos"), x).astype(np.float32)
    return FFmpeg(HOME / "bin").write_audio(x, SR, tmp_path / "call", "wav").read_bytes()


def _run(client, h, data, tier="standard", **options):
    r = client.post("/api/jobs", json={
        "kind": "denoise", "denoiseTier": tier, "filename": "rec.wav", "sizeBytes": len(data), "durationSeconds": 14,
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
    job, polished = _run(client, h, data)  # both on
    assert "high frequencies were restored" in (job["notice"] or "")
    job, as_is = _run(client, h, data, leveling=False, restore_highs=False)  # both off
    assert job["notice"] is None  # no highs restored
    # the leveling step pulls the voice to the loudness target; without it the level stays where the noise remover left it
    assert abs(_level_db(polished, tmp_path) - _level_db(as_is, tmp_path)) > 1.0


# ---------------------------------------------------------------- Leveling on its own (noise tier "off")

def test_leveling_alone_needs_leveling_and_is_priced_for_it(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 2_000_000, "grant", "t", "g")
    assert jobs.validate_options("denoise", pricing.NO_DENOISE, None, {})["leveling"] is True
    with pytest.raises(jobs.JobError, match="Choose Denoising, Voice rebuild or Leveling."):
        jobs.validate_options("denoise", pricing.NO_DENOISE, None, {"leveling": False})
    refused = client.post("/api/jobs", json={"kind": "denoise", "denoiseTier": "off", "filename": "rec.wav", "sizeBytes": 1000,
                                             "durationSeconds": 14, "options": {"leveling": False}}, headers=h)
    assert refused.status_code == 400 and "Choose Denoising, Voice rebuild or Leveling." in refused.json()["detail"]
    made = client.post("/api/jobs", json={"kind": "denoise", "denoiseTier": "off", "filename": "rec.wav", "sizeBytes": 1000,
                                          "durationSeconds": 14, "options": {}}, headers=h)
    assert made.status_code == 200, made.text
    assert made.json()["estCostMicros"] == pricing.cost_micros("leveling", "standard", 14)
    got = client.get(f"/api/jobs/{made.json()['id']}").json()
    assert got["options"]["leveling"] is True
    stored = json.loads(db.one("SELECT params_json FROM jobs WHERE id = ?", (made.json()["id"],))["params_json"])
    assert stored["rates"] == {"leveling": 1_000}
    e = client.post("/api/jobs/estimate", json={"kind": "denoise", "denoiseTier": "off", "durationSeconds": 600, "options": {}}, headers=h).json()
    assert [x["item"] for x in e["lines"]] == ["leveling"] and e["estCostMicros"] == 10_000


@NATIVE
def test_a_leveling_only_job_levels_without_removing_noise(client, signed_in, tmp_path):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    data = _phone_call(tmp_path)
    job, out = _run(client, h, data, tier="off", restore_highs=False)
    assert job["status"] == "done"
    assert abs(_level_db(out, tmp_path) - _level_db(data, tmp_path)) > 1.0  # the quiet recording is brought to the target
    row = db.one("SELECT cost_micros, duration_seconds FROM jobs WHERE id = ?", (job["id"],))
    assert row["cost_micros"] == pricing.cost_micros("leveling", "standard", row["duration_seconds"])
