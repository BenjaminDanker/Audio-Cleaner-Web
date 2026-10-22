"""The `leveling` job option (the desktop app's "Leveling"), and Leveling on its own; a `restore_highs` option from an older client is no
longer one: it is ignored like any option that is not known."""
import json

import numpy as np
import pytest
import scipy.signal

from acw import db, jobs
from acw.billing import ledger, pricing
from acw_common.ffmpeg import FFmpeg

from .job_support import params_of, standard, step
from .test_job_echo import NATIVE, SR
from .test_job_pipeline import HOME, _run_worker_once

def _opts(**extra):
    return jobs.validate_options("denoise", "standard", None, extra)


def test_leveling_defaults_on_and_only_booleans_are_taken():
    assert _opts()["leveling"] is True
    assert _opts(leveling=True)["leveling"] is True and _opts(leveling=False)["leveling"] is False
    for value in ("yes", 0, None):
        with pytest.raises(jobs.JobError, match="leveling must be true or false"):
            _opts(leveling=value)


def test_restore_highs_is_not_an_option_any_more_and_is_treated_like_any_unknown_option():
    """The step is gone: a request that still sends it is accepted as it would be with any other key the API does not know (options
    are free-form, unknown keys are ignored), and the job neither stores it, prices it nor reports it."""
    assert _opts(restore_highs=True) == _opts() == _opts(restore_highs="yes") == _opts(some_future_option=1)
    assert "restore_highs" not in _opts(restore_highs=True)


def create(client, h, **options):
    return client.post("/api/jobs", json={
        "kind": "denoise", "denoiseTier": "standard", "filename": "rec.wav", "sizeBytes": 1000, "durationSeconds": 14,
        "options": options,
    }, headers=h)


def test_stored_with_the_job_and_leveling_is_its_own_line_of_the_price(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 2_000_000, "grant", "t", "g")
    plain = create(client, h)
    off = create(client, h, leveling=False)
    old_client = create(client, h, restore_highs=True)  # a client that still sends the removed option
    assert plain.status_code == off.status_code == old_client.status_code == 200, (plain.text, off.text, old_client.text)
    # Leveling is its own line of the price: switching it off takes exactly that line away
    assert off.json()["estCostMicros"] == standard(14, leveling=False) > 0
    assert plain.json()["estCostMicros"] == off.json()["estCostMicros"] + step("leveling", 14)
    assert old_client.json()["estCostMicros"] == plain.json()["estCostMicros"]  # nothing is added for it
    # (the job is made with the timeline its options stand for: Leveling is a layer of it, or not)
    layers = lambda r: params_of(r.json()["id"])["edit"]["timeline"]["layers"]  # noqa: E731
    assert "leveling" in layers(plain) and "leveling" not in layers(off) and "noise" in layers(off)
    stored = params_of(old_client.json()["id"])
    assert "leveling" in layers(old_client) and "restore_highs" not in stored["options"] and "restore_highs" not in stored["whole_file"]["options"]
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
        "options": {"output_video": False, "loudness_lufs": -16.0, "noise": "light", "audio_format": "wav", **options},
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
def test_the_worker_passes_the_option_on(client, signed_in, tmp_path):
    uid, h = signed_in
    ledger.credit(uid, 8_000_000, "grant", "t", "g")
    data = _phone_call(tmp_path)
    job, leveled = _run(client, h, data)
    assert job["notice"] is None  # (an old client's restore_highs is ignored: nothing is restored, nothing is said)
    job, as_is = _run(client, h, data, leveling=False, restore_highs=True)
    assert job["notice"] is None
    # the leveling step pulls the voice to the loudness target; without it the level stays where the noise remover left it
    assert abs(_level_db(leveled, tmp_path) - _level_db(as_is, tmp_path)) > 1.0


def test_a_job_stored_with_the_removed_option_runs_and_is_charged_without_it(client, signed_in):
    """Parameters written by an older version (restore_highs in the options, a highs price pinned in the rates) are read past: the job
    is described, specified for the runner and charged by what exists (Denoising and Leveling), and nothing fails."""
    import json as _json

    from acw.worker import main as worker

    from .job_support import complete_like_the_worker, row

    uid, h = signed_in
    ledger.credit(uid, 2_000_000, "grant", "t", "g")
    jid = create(client, h).json()["id"]
    jobs.update_params(jid, options={**params_of(jid)["options"], "restore_highs": True},
                       rates={"denoise": 9_000, "highs": 70_000, "leveling": 1_000})
    assert client.put(f"/api/jobs/{jid}/input", content=bytes(1000), headers=h).status_code == 200
    shown = client.get(f"/api/jobs/{jid}").json()
    assert shown["label"] == "Denoising + Leveling"
    worker._write_spec(row(jid), worker._settings_extra(), 14.0)
    spec = _json.loads((jobs.job_dir(jid) / "spec.json").read_text())
    assert "highs_where_needed" not in spec
    assert set(spec["editor"]["timeline"]["layers"]) == {"noise", "leveling"} and "restore_highs" not in json.dumps(spec["editor"])
    assert worker._job_cost(row(jid), params_of(jid), 14.0) == standard(14)
    done = complete_like_the_worker(jid, outputs=["rec-cleaned.flac"])
    assert done["cost_micros"] == standard(14)  # (Denoising and Leveling: no price for the removed step)
    line = db.one("SELECT description FROM ledger WHERE ref_id = ?", (jid,))["description"]
    assert line.startswith("Denoising + Leveling:")


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
    assert made.json()["estCostMicros"] == step("leveling", 14)
    stored = params_of(made.json()["id"])
    assert set(stored["edit"]["timeline"]["layers"]) == {"leveling"} and stored["edit_rates"]["leveling"] == 1_000  # (the one step, at its pinned price)
    e = client.post("/api/jobs/estimate", json={"kind": "denoise", "denoiseTier": "off", "durationSeconds": 600, "options": {}}, headers=h).json()
    assert [x["item"] for x in e["lines"]] == ["leveling"] and e["estCostMicros"] == step("leveling", 600)


@NATIVE
def test_a_leveling_only_job_levels_without_removing_noise(client, signed_in, tmp_path):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    data = _phone_call(tmp_path)
    job, out = _run(client, h, data, tier="off")
    assert job["status"] == "done"
    assert abs(_level_db(out, tmp_path) - _level_db(data, tmp_path)) > 1.0  # the quiet recording is brought to the target
    row = db.one("SELECT cost_micros, duration_seconds FROM jobs WHERE id = ?", (job["id"],))
    assert row["cost_micros"] == step("leveling", row["duration_seconds"])
