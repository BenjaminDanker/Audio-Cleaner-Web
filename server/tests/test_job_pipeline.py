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

from .conftest import login

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
    from acw.captions import Segment

    uid, h = signed_in
    ledger.credit(uid, 2_000_000, "grant", "t", "g")
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
    data = os.urandom(200_000)
    r = client.post("/api/jobs", json={"kind": "denoise", "denoiseTier": "standard", "filename": "x.mp4",
                                       "sizeBytes": len(data), "durationSeconds": 60}, headers=h)
    job_id = r.json()["id"]
    client.put(f"/api/jobs/{job_id}/input", content=data, headers=h)
    _run_worker_once("failed")
    job = client.get(f"/api/jobs/{job_id}").json()
    assert "could not be read" in job["error"]
    assert ledger.balance(uid) == {"balance_micros": 1_000_000, "held_micros": 0, "available_micros": 1_000_000}
    assert not (Path(os.environ["ACW_DATA_DIR"]) / "jobs" / job_id).exists()
