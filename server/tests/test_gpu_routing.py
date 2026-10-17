"""Which jobs the GPU worker takes (docs/research/gpu-2026-10.md) and who leaves them to it."""
import json
import os
import time

import pytest

from acw import db, jobs
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
    (d / "dpdfnet").mkdir(parents=True)
    monkeypatch.setenv("ACW_MODELS_DIR", str(d))
    from acw.config import get_settings

    get_settings.cache_clear()
    return d


def install_batch(models):
    from acw_common import dpdfnet

    (models / "dpdfnet" / dpdfnet.BATCH_MODEL_FILE).write_bytes(b"x")


def test_the_voice_rebuild_goes_to_the_card(models):
    assert W.is_gpu_denoise(job_row(tier="rebuild", noise="light"))  # needs no batched DPDFNet file
    assert W.is_gpu_job(job_row(tier="rebuild", kind="denoise_transcribe", transcribe="premium"))


def test_heavy_and_auto_cleaning_need_the_batched_file(models):
    for noise in ("heavy", "auto"):
        assert not W.is_gpu_denoise(job_row(noise=noise))  # file not installed: the CPU workers do it
    install_batch(models)
    for noise in ("heavy", "auto"):
        assert W.is_gpu_denoise(job_row(noise=noise))
    assert not W.is_gpu_denoise(job_row(noise="light"))  # DeepFilterNet3 has no GPU path


def test_what_stays_with_the_cpu_workers(models):
    install_batch(models)
    assert not W.is_gpu_job(job_row(echo=True))  # no echo canceller in the GPU venv
    assert not W.is_gpu_job(job_row(next_stage="finalize"))
    assert not W.is_gpu_job(job_row(kind="transcribe", transcribe="premium"))  # cloud transcription: nothing to run
    assert not W.is_gpu_job(job_row(kind="denoise_transcribe", noise="light", transcribe="premium", echo=True))
    assert W.is_gpu_job(job_row(kind="transcribe", transcribe="local"))  # Parakeet, as before
    assert W.is_local_transcription(job_row(kind="transcribe", transcribe="local"))


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


def test_cpu_workers_take_everything_when_there_is_no_gpu_worker(models):
    install_batch(models)
    rebuild = queued(models, denoise_tier="rebuild")
    assert W.claim("cpu")["id"] == rebuild


def test_cpu_workers_leave_gpu_jobs_to_an_idle_gpu_worker(models):
    install_batch(models)
    heavy = queued(models, options={"noise": "heavy"})
    plain = queued(models, options={"noise": "light"})
    marker("gpu-worker-idle")
    marker("gpu-worker-alive")
    assert W.claim("cpu")["id"] == plain  # the light job is theirs, the heavy one waits for the card
    assert W.claim("cpu") is None
    assert db.one("SELECT status FROM jobs WHERE id = ?", (heavy,))["status"] == "queued"


def test_a_busy_gpu_worker_keeps_only_the_rebuild(models):
    install_batch(models)
    rebuild = queued(models, denoise_tier="rebuild")
    heavy = queued(models, options={"noise": "heavy"})
    marker("gpu-worker-alive")  # up, but not idle: it is running something
    assert W.claim("cpu")["id"] == heavy  # work stays work-conserving: the CPU does a heavy clean while the card is busy
    assert W.claim("cpu") is None
    assert db.one("SELECT status FROM jobs WHERE id = ?", (rebuild,))["status"] == "queued"
    marker("gpu-worker-alive", age=300)  # the GPU worker went away: nobody waits for it
    assert W.claim("cpu")["id"] == rebuild


def test_the_gpu_worker_takes_only_its_jobs(models, monkeypatch):
    install_batch(models)
    light = queued(models, options={"noise": "light"})
    rebuild = queued(models, denoise_tier="rebuild")
    monkeypatch.setattr(W, "GPU_MODE", True)
    got = W.claim("gpu")
    assert got["id"] == rebuild
    assert W.claim("gpu") is None
    assert db.one("SELECT status FROM jobs WHERE id = ?", (light,))["status"] == "queued"


def test_the_gpu_spec_names_the_card_for_cleaning_too():
    spec = W._gpu_spec()
    assert spec["asr_device"] == "cuda" and spec["clean_device"] == "cuda"


def test_heavy_noise_runs_on_the_card_only_with_the_batched_file(models):
    assert W.heavy_device() == "cpu"  # the worker then cleans on its CPU threads: no stage asks the card for what it lacks
    install_batch(models)
    assert W.heavy_device() == "cuda"
