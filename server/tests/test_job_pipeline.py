"""End-to-end job pipeline through the real sandboxed runner: one file job per kind, a file that is not media, and the queue.

Needs the native pieces (~/audio-cleaner/bin, models) and a speech sample via
ACW_TEST_AUDIO (not shipped: our sample is CC BY-NC). Skipped otherwise. The sample is cut to six seconds for the jobs here.
"""
import asyncio
import json
import os
import subprocess
from pathlib import Path

import pytest

from acw import db
from acw.billing import ledger

from .conftest import settle_payment

HOME = Path(os.environ.get("ACW_HOME", Path.home() / "audio-cleaner"))
SAMPLE = os.environ.get("ACW_TEST_AUDIO", str(HOME / "src/DeepFilterNet/assets/noisy_snr0.wav"))
pytestmark = pytest.mark.skipif(
    not (Path(SAMPLE).exists() and (HOME / "bin/libdf.so").exists() and (HOME / "bin/ffmpeg").exists()),
    reason="native binaries/models or sample audio not available",
)


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    """The first six seconds of the speech sample."""
    out = tmp_path_factory.mktemp("sample") / "talk.wav"
    subprocess.run([str(HOME / "bin/ffmpeg"), "-v", "error", "-i", SAMPLE, "-t", "6", "-y", str(out)], check=True)
    return out


def _run_worker_once(expect_status):
    from acw.worker import main as worker

    job = worker.claim("test")
    assert job is not None
    worker.process(job, worker._settings_extra())
    row = db.one("SELECT status, error FROM jobs WHERE id = ?", (job["id"],))
    assert row["status"] == expect_status, row["error"]
    return job["id"]


def test_premium_transcription_stage_machine(client, signed_in, monkeypatch, sample):
    from acw.asr import openai_asr
    from acw_common.captions import Segment

    uid, h = signed_in
    ledger.credit(uid, 2_000_000, "grant", "t", "g")
    settle_payment(uid)  # extra accurate needs a first top-up
    data = sample.read_bytes()
    r = client.post("/api/jobs", json={
        "kind": "denoise_transcribe", "denoiseTier": "standard", "transcribeTier": "premium",
        "filename": "talk.wav", "sizeBytes": len(data), "durationSeconds": 6,
        "options": {"audio_format": "mp3", "caption_formats": ["srt", "txt"]},
    }, headers=h)
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    assert client.put(f"/api/jobs/{job_id}/input", content=data, headers=h).status_code == 200

    _run_worker_once("cloud_asr")
    chunk_dir = Path(os.environ["ACW_DATA_DIR"]) / "jobs" / job_id / "asr_chunks"
    index = json.loads((chunk_dir / "index.json").read_text())
    assert index["chunks"] and (chunk_dir / index["chunks"][0]["file"]).stat().st_size > 1000

    async def fake_chunks(_dir, _lang):
        return [Segment(0.5, 3.0, "Hello from the premium engine.", speaker="A")]

    monkeypatch.setattr(openai_asr, "transcribe_chunks", fake_chunks)
    asyncio.run(openai_asr.run_for_job(chunk_dir.parent, None))
    db.execute("UPDATE jobs SET status='queued', next_stage='finalize' WHERE id = ?", (job_id,))

    _run_worker_once("done")
    job = client.get(f"/api/jobs/{job_id}").json()
    names = {o["name"] for o in job["outputs"]}
    assert names == {"talk-cleaned.mp3", "talk.srt", "talk-transcript.txt"}
    assert "Hello from the premium engine." in client.get(f"/api/jobs/{job_id}/outputs/talk.srt").text
    assert job["costMicros"] > 0
    assert ledger.balance(uid)["held_micros"] == 0
    # nothing but the results, the upload and the copies kept for changing the job again is left on disk (all go together, 2 hours later)
    left = {p.name for p in (Path(os.environ["ACW_DATA_DIR"]) / "jobs" / job_id).iterdir()}
    assert left == names | {"input.wav", "kept-noise.flac", "kept-input.flac", "segments.json", "blocks", "kept-source.flac", "suggestions.json"}


def test_garbage_upload_fails_cleanly_and_refunds(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 1_000_000, "grant", "t", "g")
    data = b"this is not a media file\n" * 8000  # random bytes sometimes probe as mp3
    r = client.post("/api/jobs", json={"kind": "denoise", "denoiseTier": "standard", "filename": "x.mp4",
                                       "sizeBytes": len(data), "durationSeconds": 60}, headers=h)
    job_id = r.json()["id"]
    client.put(f"/api/jobs/{job_id}/input", content=data, headers=h)
    _run_worker_once("failed")
    job = client.get(f"/api/jobs/{job_id}").json()
    assert "could not be read" in job["error"]
    assert ledger.balance(uid) == {"balance_micros": 1_000_000, "held_micros": 0, "available_micros": 1_000_000}
    assert not (Path(os.environ["ACW_DATA_DIR"]) / "jobs" / job_id).exists()


def test_queue_order_is_fair_per_user():
    from acw import auth, db, jobs
    from acw.worker.main import claim

    a = auth.get_or_create_user("a@example.com")["id"]
    b = auth.get_or_create_user("b@example.com")["id"]
    now = db.now()

    def add(job_id, user, t, status="queued"):
        db.execute(
            "INSERT INTO jobs(id, user_id, kind, tier, status, params_json, input_name, input_bytes, claimed_seconds,"
            " est_cost_micros, created_at, queued_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (job_id, user, "transcribe", "local", status, '{"options": {}}', "x.wav", 1000, 10, 0, now, now + t),
        )

    add("a-running", a, 0, status="running")
    add("a-second", a, 1)
    add("b-later", b, 2)
    assert jobs.queued_ids() == ["b-later", "a-second"]  # a already has a job running
    assert claim("w1")["id"] == "b-later"


def test_local_transcription_through_the_sandbox(client, signed_in, sample):
    """Auto language -> detected English -> Parakeet v2 in the Landlocked runner."""
    models = Path.home() / "audio-cleaner/models"
    if not all((models / m).is_file() for m in ("parakeet-v2/encoder-model.onnx", "whisper/base/model.bin")):
        pytest.skip("Parakeet or Whisper base not installed")
    uid, h = signed_in
    ledger.credit(uid, 2_000_000, "grant", "t", "g")
    data = sample.read_bytes()
    r = client.post("/api/jobs", json={
        "kind": "transcribe", "transcribeTier": "local", "filename": "talk.wav", "sizeBytes": len(data),
        "durationSeconds": 6, "options": {"caption_formats": ["srt"]},
    }, headers=h)
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    assert client.put(f"/api/jobs/{job_id}/input", content=data, headers=h).status_code == 200
    _run_worker_once("done")
    srt = client.get(f"/api/jobs/{job_id}/outputs/talk.srt", headers=h).text
    assert "held responsible" in srt and "-->" in srt
