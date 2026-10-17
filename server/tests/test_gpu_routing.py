"""Which jobs the GPU worker takes (docs/research/gpu-2026-10.md) and who leaves them to it."""
import json
import os
import time

import pytest

from acw import broker, db, jobs
from acw.billing import ledger
from acw.worker import main as W
from tests.test_billing_audit import make_job, user


def job_row(*, kind="denoise", tier="standard", transcribe=None, noise="auto", echo=False, next_stage=None):
    opts = {"noise": noise, **({"echo_reference_track": True} if echo else {})}
    return {"kind": kind, "next_stage": next_stage,
            "params_json": json.dumps({"denoise_tier": tier, "transcribe_tier": transcribe, "options": opts})}


@pytest.fixture
def models(tmp_path, monkeypatch):
    d = tmp_path / "models"
    monkeypatch.setenv("ACW_MODELS_DIR", str(d))
    from acw.config import get_settings

    get_settings.cache_clear()
    return d


@pytest.mark.parametrize("row, stages, card_only", [
    (job_row(tier="rebuild", noise="light"), ["rebuild"], True),  # the voice rebuild goes to the card only
    (job_row(tier="rebuild", kind="denoise_transcribe", transcribe="premium"), ["rebuild"], True),
    (job_row(noise="heavy"), ["heavy"], False),  # heavy and auto cleaning: the card, or the CPU while it is busy
    (job_row(noise="auto"), ["heavy"], False),
    (job_row(noise="light"), [], False),  # DeepFilterNet3 has no GPU path
    (job_row(tier="off", noise="heavy"), [], False),  # leveling on its own is CPU work
    (job_row(tier="off", kind="denoise_transcribe", transcribe="local"), ["transcribe"], False),  # local subtitles: Parakeet
    (job_row(echo=True), [], False),  # no echo canceller in the GPU venv
    (job_row(next_stage="finalize"), [], False),
    (job_row(kind="transcribe", transcribe="premium"), [], False),  # cloud transcription: nothing to run here
    (job_row(kind="transcribe", transcribe="local"), ["transcribe"], False),
    (job_row(kind="denoise_transcribe", transcribe="local"), ["transcribe", "heavy"], False),
])
def test_which_stages_go_to_the_card(row, stages, card_only):
    assert broker.gpu_stages(row) == stages
    assert broker.gpu_only(row) is card_only


def queued(models, **kw):
    uid = user(f"u{len(kw)}{time.time_ns()}@example.com")
    ledger.credit(uid, 50_000_000, "grant", "t", "g" + uid)
    jid = make_job(uid, **kw)
    jobs.mark_uploaded(jid)
    return jid


def marker(name, age=0.0):
    from acw.config import get_settings

    p = get_settings().data_dir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    t = time.time() - age
    os.utime(p, (t, t))


def test_the_card_runs_the_rebuilds_and_the_cpus_never_do(models, monkeypatch):
    marker("gpu-worker-alive")
    rebuild = queued(models, denoise_tier="rebuild")
    heavy = queued(models, options={"noise": "heavy"})
    assert W.claim("cpu")["id"] == heavy  # work-conserving: the CPU does a heavy clean while the card is busy
    assert W.claim("cpu") is None  # never the rebuild
    marker("gpu-worker-alive", age=300)  # the GPU worker went away: the rebuild still waits for it
    assert W.claim("cpu") is None
    marker("gpu-worker-alive")
    monkeypatch.setattr(W, "GPU_MODE", True)
    assert W.claim("gpu")["id"] == rebuild


def test_cpu_workers_leave_heavy_jobs_to_an_idle_card(models):
    heavy = queued(models, options={"noise": "heavy"})
    plain = queued(models, options={"noise": "light"})
    marker("gpu-worker-idle")
    marker("gpu-worker-alive")
    assert W.claim("cpu")["id"] == plain  # the light job is theirs, the heavy one waits for the card
    assert W.claim("cpu") is None
    assert db.one("SELECT status FROM jobs WHERE id = ?", (heavy,))["status"] == "queued"


def test_no_rebuild_can_be_ordered_while_no_gpu_worker_is_up(models, client, signed_in):
    uid, _ = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    with pytest.raises(jobs.JobError, match="paused"):
        make_job(uid, denoise_tier="rebuild")
    assert client.get("/api/billing/pricing").json()["rebuildAvailable"] is False
    marker("gpu-worker-alive")
    make_job(uid, denoise_tier="rebuild")
    assert client.get("/api/billing/pricing").json()["rebuildAvailable"] is True


def test_queue_estimates_have_a_gpu_lane(models):
    marker("gpu-worker-alive")
    first = queued(models, denoise_tier="rebuild")
    second = queued(models, denoise_tier="rebuild")
    plain = queued(models, options={"noise": "light"})
    jobs.invalidate_queue_cache()
    _, waits = jobs.queue_snapshot()
    rebuild_seconds = jobs.job_seconds(dict(db.one("SELECT * FROM jobs WHERE id = ?", (first,))))
    assert waits[first] == 0 and waits[second] == int(rebuild_seconds)  # rebuilds queue behind each other on the card
    assert waits[plain] == 0  # the CPU workers have nothing in front of it
