"""The echo_reference_track job option: validation, the runner using a file's second audio track, and a file with only one track: the whole
job fails as soon as it is read, before any work, uncharged (the stubbed worker flow and the re-runs are in test_echo_one_track.py)."""
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from acw import db, jobs
from acw.billing import ledger, pricing
from acw_common.ffmpeg import FFmpeg

from .job_support import params_of, standard, step
from .test_job_pipeline import HOME, _run_worker_once

SR = 48000
SECONDS = 6  # the speaker talks for the first four, the person for the last two
NATIVE = pytest.mark.skipif(
    not ((HOME / "bin/libdf.so").exists() and (HOME / "bin/ffmpeg").exists()), reason="native binaries/models not available"
)


def create(client, h, **options):
    return client.post("/api/jobs", json={
        "kind": "denoise", "denoiseTier": "standard", "filename": "rec.mkv", "sizeBytes": 1000, "durationSeconds": SECONDS,
        "options": options,
    }, headers=h)


def test_the_option_is_taken_as_true_or_false_and_has_its_own_price(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 2_000_000, "grant", "t", "g")
    plain = create(client, h)
    on = create(client, h, echo_reference_track=True)
    assert plain.status_code == on.status_code == 200, (plain.text, on.text)
    assert plain.json()["estCostMicros"] == standard(SECONDS)
    assert on.json()["estCostMicros"] == plain.json()["estCostMicros"] + step("echo", SECONDS) > 0
    # (the job is made with the timeline its options stand for: echo removal is a layer of it, or not)
    assert "echo" not in params_of(plain.json()["id"])["edit"]["timeline"]["layers"]
    assert params_of(on.json()["id"])["edit"]["timeline"]["layers"]["echo"] == [{"start": 0.0, "end": float(SECONDS)}]
    assert create(client, h, echo_reference_track="yes").status_code == 400


def test_echo_removal_needs_scratch_space():
    hour = 3600
    assert jobs.scratch_bytes("denoise", hour) < jobs.scratch_bytes("denoise", hour, echo=True)
    assert jobs.scratch_bytes("transcribe", hour) == 0 < jobs.scratch_bytes("transcribe", hour, echo=True)


def _voice(f0, seed, seconds):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.15 * np.sin(2 * np.pi * 0.9 * t))) / SR
    env = np.clip(np.sin(2 * np.pi * 2.3 * t + seed), 0, None) ** 2
    return (0.25 * sum(np.sin(k * phase) / k for k in range(1, 15)) * env + 0.02 * rng.standard_normal(len(t))).astype(np.float32)


def _recording(tmp_path, two_tracks):
    """Speaker talks for the first four seconds (and the mic hears it back), then the person talks."""
    far, near = _voice(120, 1, SECONDS), _voice(210, 2, SECONDS)
    far[4 * SR:] = 0
    near[:4 * SR] = 0
    ir = np.zeros(SR // 8, np.float32)
    ir[[int(0.04 * SR), int(0.047 * SR), int(0.06 * SR)]] = [0.5, 0.25, -0.12]
    mic = near + np.convolve(far, ir)[:len(far)].astype(np.float32)
    ff = FFmpeg(HOME / "bin")
    args, maps = [], []
    for i, x in enumerate([mic, far] if two_tracks else [mic]):
        args += ["-i", str(ff.write_audio(x, SR, tmp_path / f"t{i}", "wav"))]
        maps += ["-map", f"{i}:a"]
    out = tmp_path / "rec.mkv"
    subprocess.run([str(HOME / "bin/ffmpeg"), "-v", "error", *args, *maps, "-c:a", "flac", "-y", str(out)], check=True)
    return out.read_bytes()


def _clean(client, h, data, **options):
    r = client.post("/api/jobs", json={
        "kind": "denoise", "denoiseTier": "standard", "filename": "rec.mkv", "sizeBytes": len(data), "durationSeconds": SECONDS,
        "options": {"output_video": False, "loudness_lufs": None, "audio_format": "wav", **options},
    }, headers=h)
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    assert client.put(f"/api/jobs/{job_id}/input", content=data, headers=h).status_code == 200
    _run_worker_once("done")
    job = client.get(f"/api/jobs/{job_id}").json()
    out = client.get(f"/api/jobs/{job_id}/outputs/rec-cleaned.wav", headers=h).content
    return job, out


def _speaker_vs_voice_db(wav_bytes, tmp_path):
    """How loud the speaker's echo is compared with the person's voice, in the cleaned audio."""
    path = tmp_path / "out.wav"
    path.write_bytes(wav_bytes)
    ff = FFmpeg(HOME / "bin")
    x = ff.decode(path, ff.probe(path), SR, 1, 60)[:, 0]
    rms = lambda s: 20 * np.log10(max(float(np.sqrt(np.mean(s**2))), 1e-9))  # noqa: E731
    assert abs(len(x) - SECONDS * SR) < SR // 10  # length kept
    return rms(x[2 * SR:4 * SR]) - rms(x[4 * SR + SR // 2:SECONDS * SR])


@NATIVE
def test_second_track_removes_the_echo_and_the_scratch_files_are_gone(client, signed_in, tmp_path):
    uid, h = signed_in
    ledger.credit(uid, 2_000_000, "grant", "t", "g")
    data = _recording(tmp_path, two_tracks=True)
    job, without = _clean(client, h, data)
    assert job["notice"] is None
    job, with_echo_removal = _clean(client, h, data, echo_reference_track=True)
    assert job["notice"] is None
    row = db.one("SELECT cost_micros, duration_seconds FROM jobs WHERE id = ?", (job["id"],))
    assert row["cost_micros"] == standard(row["duration_seconds"]) + step("echo", row["duration_seconds"])
    assert row["cost_micros"] > standard(row["duration_seconds"])
    assert _speaker_vs_voice_db(with_echo_removal, tmp_path) < _speaker_vs_voice_db(without, tmp_path) - 10
    left = {p.name for p in (Path(os.environ["ACW_DATA_DIR"]) / "jobs" / job["id"]).iterdir()}
    # scratch files are gone; the upload and the per-part editor's (block cache, the recording, suggestions) stay
    assert left == {"rec-cleaned.wav", "input.mkv", "blocks", "kept-source.flac", "suggestions.json"}
    # the whole-file run (echo removal, then the noise remover) filled the editor's cache under the editor's own block identities: the job's
    # timeline (echo removal and noise removal on the whole file) needs nothing to be made, and the quote of editing it says so
    import json

    from acw import editing

    row = dict(db.one("SELECT * FROM jobs WHERE id = ?", (job["id"],)))
    params = json.loads(row["params_json"])
    seconds = editing.file_seconds(row, params)
    made_with = editing.saved_timeline(row, params, seconds)
    assert params["edit"]["facts"]["reference"] is True and "echo" in made_with.to_json()["layers"]
    q = editing.make_quote(row, params, made_with)
    assert q["misses"] == 0 and q["unchanged"] is True
    softer = editing.saved_timeline(row, params, seconds)
    softer.paint("noise", 0.0, softer.seconds, {"strength_db": 24.0, "remover": softer.parts("noise")[0].settings["remover"]})
    softer.erase("leveling", 0.0, softer.seconds)
    q = editing.make_quote(row, params, softer)
    assert q["misses"] == 0 and q["micros"] == 0 and q["unchanged"] is False  # a new strength: the cache has every block


def _fail_on_one_track(client, h, uid, data, **body):
    """A one-track file with echo removal asked for (by the option, or by echo parts of a timeline): the job fails at the first stage."""
    before = ledger.balance(uid)
    tier = {} if "timeline" in body else {"denoiseTier": "standard"}  # (a timeline says what is done to each part: no tier)
    r = client.post("/api/jobs", json={"kind": "denoise", "filename": "rec.mkv", "sizeBytes": len(data), "durationSeconds": SECONDS, **tier, **body}, headers=h)
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    assert ledger.balance(uid)["held_micros"] > 0  # (it was held for, like any job)
    assert client.put(f"/api/jobs/{job_id}/input", content=data, headers=h).status_code == 200
    _run_worker_once("failed")
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["error"] == jobs.NO_SECOND_TRACK and not job["costMicros"] and not job["outputs"] and job["notice"] is None
    assert ledger.balance(uid) == before  # nothing held, nothing charged
    assert not (Path(os.environ["ACW_DATA_DIR"]) / "jobs" / job_id).exists()
    return job


def test_the_no_second_track_words_say_what_to_do():
    assert jobs.NO_SECOND_TRACK == ("This file has only one audio track, so there is no desktop sound to remove the echo with. Send it without "
                                    "echo removal (nothing was charged).")


@NATIVE
def test_a_file_with_one_track_fails_the_whole_job_uncharged_when_echo_removal_was_asked_for(client, signed_in, tmp_path):
    uid, h = signed_in
    ledger.credit(uid, 2_000_000, "grant", "t", "g")
    data = _recording(tmp_path, two_tracks=False)
    _fail_on_one_track(client, h, uid, data, options={"output_video": False, "echo_reference_track": True})
    # the same file without echo removal is cleaned as usual, and its single track is remembered
    job, out = _clean(client, h, data)
    assert job["notice"] is None and len(out) > 1000
    assert json.loads(db.one("SELECT params_json FROM jobs WHERE id = ?", (job["id"],))["params_json"])["audio_tracks"] == 1
    row_ = db.one("SELECT cost_micros, duration_seconds FROM jobs WHERE id = ?", (job["id"],))
    assert row_["cost_micros"] == standard(row_["duration_seconds"])


@NATIVE
def test_echo_parts_in_the_timeline_of_a_file_with_one_track_fail_the_job_the_same_way(client, signed_in, tmp_path):
    uid, h = signed_in
    ledger.credit(uid, 2_000_000, "grant", "t", "g")
    data = _recording(tmp_path, two_tracks=False)
    tl = {"version": 1, "seconds": SECONDS, "layers": {"echo": [{"start": 0, "end": SECONDS}], "noise": [{"start": 0, "end": SECONDS, "strength_db": 40, "remover": "light"}]}}
    _fail_on_one_track(client, h, uid, data, options={"output_video": False}, timeline=tl)


@NATIVE
def test_a_second_track_is_remembered_and_echo_removal_goes_on(client, signed_in, tmp_path):
    uid, h = signed_in
    ledger.credit(uid, 2_000_000, "grant", "t", "g")
    job, _ = _clean(client, h, _recording(tmp_path, two_tracks=True), echo_reference_track=True)
    assert job["notice"] is None
    assert json.loads(db.one("SELECT params_json FROM jobs WHERE id = ?", (job["id"],))["params_json"])["audio_tracks"] == 2
