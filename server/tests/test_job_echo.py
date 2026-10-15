"""The echo_reference_track job option: validation, and the runner using a file's second audio track."""
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from acw import jobs
from acw.billing import ledger
from acw_common.ffmpeg import FFmpeg

from .test_job_pipeline import HOME, _run_worker_once

SR = 48000
NATIVE = pytest.mark.skipif(
    not ((HOME / "bin/libdf.so").exists() and (HOME / "bin/ffmpeg").exists()), reason="native binaries/models not available"
)


def create(client, h, **options):
    return client.post("/api/jobs", json={
        "kind": "denoise", "denoiseTier": "standard", "filename": "rec.mkv", "sizeBytes": 1000, "durationSeconds": 14,
        "options": options,
    }, headers=h)


def test_option_defaults_off_validated_and_free_of_charge(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 2_000_000, "grant", "t", "g")
    plain = create(client, h)
    on = create(client, h, echo_reference_track=True)
    assert plain.status_code == on.status_code == 200, (plain.text, on.text)
    assert on.json()["estCostMicros"] == plain.json()["estCostMicros"] > 0
    assert client.get(f"/api/jobs/{plain.json()['id']}").json()["options"]["echo_reference_track"] is False
    assert client.get(f"/api/jobs/{on.json()['id']}").json()["options"]["echo_reference_track"] is True
    assert create(client, h, echo_reference_track="yes").status_code == 400


def test_echo_removal_needs_scratch_space():
    hour = 3600
    assert jobs.scratch_bytes("denoise", hour) < jobs.scratch_bytes("denoise", hour, echo=True)
    assert jobs.scratch_bytes("transcribe", hour) == 0 < jobs.scratch_bytes("transcribe", hour, echo=True)


def _voice(f0, seed, seconds=14.0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.15 * np.sin(2 * np.pi * 0.9 * t))) / SR
    env = np.clip(np.sin(2 * np.pi * 2.3 * t + seed), 0, None) ** 2
    return (0.25 * sum(np.sin(k * phase) / k for k in range(1, 15)) * env + 0.02 * rng.standard_normal(len(t))).astype(np.float32)


def _recording(tmp_path, two_tracks):
    """Speaker talks for 9 s (and the mic hears it back), then the person talks for 5 s."""
    far, near = _voice(120, 1), _voice(210, 2)
    far[9 * SR:] = 0
    near[:9 * SR] = 0
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
        "kind": "denoise", "denoiseTier": "standard", "filename": "rec.mkv", "sizeBytes": len(data), "durationSeconds": 14,
        "options": {"output_video": False, "loudness_lufs": None, **options},
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
    assert abs(len(x) - 14 * SR) < SR // 10  # length kept
    return rms(x[6 * SR:8 * SR]) - rms(x[10 * SR:13 * SR])


@NATIVE
def test_second_track_removes_the_echo_in_the_cleaned_audio(client, signed_in, tmp_path):
    uid, h = signed_in
    ledger.credit(uid, 2_000_000, "grant", "t", "g")
    data = _recording(tmp_path, two_tracks=True)
    job, without = _clean(client, h, data)
    assert job["notice"] is None
    job, with_echo_removal = _clean(client, h, data, echo_reference_track=True)
    assert job["notice"] is None
    assert _speaker_vs_voice_db(with_echo_removal, tmp_path) < _speaker_vs_voice_db(without, tmp_path) - 10
    left = {p.name for p in (Path(os.environ["ACW_DATA_DIR"]) / "jobs" / job["id"]).iterdir()}
    assert left == {"rec-cleaned.wav"}  # scratch files are gone


@NATIVE
def test_one_track_file_is_cleaned_normally_with_a_notice(client, signed_in, tmp_path):
    uid, h = signed_in
    ledger.credit(uid, 2_000_000, "grant", "t", "g")
    job, out = _clean(client, h, _recording(tmp_path, two_tracks=False), echo_reference_track=True)
    assert "only one audio track" in job["notice"]
    assert len(out) > 1000
