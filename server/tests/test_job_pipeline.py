"""End-to-end job pipeline through the real sandboxed runner.

Needs the native pieces (~/audio-cleaner/bin, models) and a speech sample via
ACW_TEST_AUDIO (not shipped: our sample is CC BY-NC). Skipped otherwise.
"""
import asyncio
import json
import os
from pathlib import Path

import pytest

from acw import db
from acw.billing import ledger

from .conftest import login, settle_payment

HOME = Path(os.environ.get("ACW_HOME", Path.home() / "audio-cleaner"))
SAMPLE = os.environ.get("ACW_TEST_AUDIO", str(HOME / "src/DeepFilterNet/assets/noisy_snr0.wav"))
pytestmark = pytest.mark.skipif(
    not (Path(SAMPLE).exists() and (HOME / "bin/libdf.so").exists() and (HOME / "bin/ffmpeg").exists()),
    reason="native binaries/models or sample audio not available",
)


def _run_worker_once(expect_status):
    from acw.worker import main as worker

    job = worker.claim("test")
    assert job is not None
    worker.process(job, worker._settings_extra())
    row = db.one("SELECT status, error FROM jobs WHERE id = ?", (job["id"],))
    assert row["status"] == expect_status, row["error"]
    return job["id"]


def test_premium_transcription_stage_machine(client, signed_in, monkeypatch):
    from acw.asr import openai_asr
    from acw_common.captions import Segment

    uid, h = signed_in
    ledger.credit(uid, 2_000_000, "grant", "t", "g")
    settle_payment(uid)  # extra accurate needs a first top-up
    data = Path(SAMPLE).read_bytes()
    r = client.post("/api/jobs", json={
        "kind": "denoise_transcribe", "denoiseTier": "standard", "transcribeTier": "premium",
        "filename": "talk.wav", "sizeBytes": len(data), "durationSeconds": 10,
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
    # nothing but outputs is left on disk
    left = {p.name for p in (Path(os.environ["ACW_DATA_DIR"]) / "jobs" / job_id).iterdir()}
    assert left == names


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


def test_queue_order_paid_first_then_fair_per_user():
    from acw import auth, db, jobs
    from acw.worker.main import claim

    a = auth.get_or_create_user("a@example.com")["id"]
    b = auth.get_or_create_user("b@example.com")["id"]
    now = db.now()

    def add(job_id, user, free, t, status="queued"):
        db.execute(
            "INSERT INTO jobs(id, user_id, kind, tier, free, status, params_json, input_name, input_bytes, claimed_seconds,"
            " est_cost_micros, created_at, queued_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (job_id, user, "transcribe", "local", free, status, '{"options": {}}', "x.wav", 1000, 10, 0, now, now + t),
        )

    add("free-early", b, 1, 0)
    add("a-running", a, 0, 0, status="running")
    add("a-second", a, 0, 1)
    add("b-paid", b, 0, 2)
    assert jobs.queued_ids() == ["b-paid", "a-second", "free-early"]
    assert jobs.public_job(dict(db.one("SELECT * FROM jobs WHERE id = 'a-second'")))["queuePosition"] == 2
    assert claim("w1")["id"] == "b-paid"


def test_local_transcription_through_the_sandbox(client, signed_in):
    """Auto language -> detected English -> Parakeet v2 in the Landlocked runner."""
    if not (Path.home() / "audio-cleaner/models/parakeet-v2/encoder-model.onnx").is_file():
        pytest.skip("Parakeet model not installed")
    uid, h = signed_in
    ledger.credit(uid, 2_000_000, "grant", "t", "g")
    data = Path(SAMPLE).read_bytes()
    r = client.post("/api/jobs", json={
        "kind": "transcribe", "transcribeTier": "local", "filename": "talk.wav", "sizeBytes": len(data),
        "durationSeconds": 10, "options": {"caption_formats": ["srt"]},
    }, headers=h)
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    assert client.put(f"/api/jobs/{job_id}/input", content=data, headers=h).status_code == 200
    _run_worker_once("done")
    srt = client.get(f"/api/jobs/{job_id}/outputs/talk.srt", headers=h).text
    assert "held responsible" in srt and "-->" in srt
