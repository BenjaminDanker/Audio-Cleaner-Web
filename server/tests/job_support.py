"""Shared helpers of the job tests: finished jobs made by hand with their folders (fake outputs, no native binaries) and the parameters a
job made with a timeline has (every job is: editing.timeline_from_options), accounts and API keys, a frozen clock and a roomy disk."""
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from acw import auth, db, jobs
from acw.billing import ledger, pricing
from acw.security import sha256_hex

from .conftest import ORIGIN, login
from .test_api_security import make_key

SR = 48_000
GB = 1024**3
RETENTION = 2 * 3600


# ------------------------------------------------------------------------------------------ accounts and keys

def account(email="a@example.com", credit=50_000_000) -> str:
    """An active account with some credit; returns its id."""
    uid = auth.get_or_create_user(email)["id"]
    if credit:
        ledger.credit(uid, credit, "grant", "t", f"g-{email}")
    return uid


def api_key(uid: str, scopes="jobs", cap=None) -> tuple[str, dict]:
    """A new API key of the account (optionally with a monthly cap in micros); returns (key id, request headers)."""
    key = make_key(uid, scopes)
    key_id = db.one("SELECT id FROM api_keys WHERE key_hash = ?", (sha256_hex(key),))["id"]
    if cap is not None:
        db.execute("UPDATE api_keys SET monthly_cap_micros = ? WHERE id = ?", (cap, key_id))
    return key_id, {"Authorization": f"Bearer {key}"}


@pytest.fixture
def other_browser():
    """A second browser signed in to another account: (client, user id, headers)."""
    from fastapi.testclient import TestClient

    from acw.app import create_app

    with TestClient(create_app(), base_url=ORIGIN) as c:
        uid, h = login(c, "other@example.com")
        yield c, uid, h


# --------------------------------------------------------------------------------------------- clock and disk

class Clock:
    """db.now() frozen at a moment; `advance` moves it."""

    def __init__(self, t: int):
        self.t = t

    def __call__(self) -> int:
        return self.t

    def advance(self, seconds: int) -> None:
        self.t += int(seconds)


@pytest.fixture
def clock(monkeypatch):
    """The database clock stopped at the real time of the test's start (sessions made before it stay valid)."""
    c = Clock(int(time.time()))
    monkeypatch.setattr(db, "now", c)
    return c


@pytest.fixture
def roomy_disk(monkeypatch):
    """The host has plenty of free disk, whatever the machine running the tests has."""
    from acw import hostload

    snap = SimpleNamespace(disk_free=500 * GB)
    monkeypatch.setattr(hostload, "get", lambda d: SimpleNamespace(snapshot=lambda: snap))


def set_env(monkeypatch, **env) -> None:
    """Set ACW_* variables and re-read the settings."""
    from acw.config import get_settings

    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    get_settings.cache_clear()


# ---------------------------------------------------------------------------------------------- what the steps of a timeline cost

def step(name, seconds, rates=None):
    """What one step costs for `seconds` of audio on the whole file: its rate x (the whole seconds + its fixed amount of a run), rounded up
    (every job is made with a timeline, so this is the price of a job asked for with options too)."""
    from acw import editing
    from acw_common import editor

    return editing.step_micros((rates or editing.step_rates())[name], math.ceil(seconds), editor.STEP_COSTS[name].fixed_equivalent_seconds)


def standard(seconds, remover="light", leveling=True, rates=None):
    """Denoising (with the remover said) and Leveling on the whole file: what a job decided or asked for as Denoising costs."""
    return step(remover, seconds, rates) + (step("leveling", seconds, rates) if leveling else 0)


def rebuilt(seconds, leveling=True, rates=None):
    """Voice rebuild and Leveling on the whole file."""
    return step("rebuild", seconds, rates) + (step("leveling", seconds, rates) if leveling else 0)


# ------------------------------------------------------------------------------------------- fake finished jobs

def job_params(kind="denoise", denoise_tier="standard", transcribe_tier=None, options=None, *, seconds=60.0, channels=2, ext=".wav") -> dict:
    """The stored parameters of a finished job made with whole-file options: the timeline they stand for (first run), the prices of
    its parts pinned, what was asked for kept in `whole_file` (as jobs.create_job stores them)."""
    from acw import editing

    d_tier = denoise_tier if kind != "transcribe" else None
    t_tier = transcribe_tier if kind != "denoise" else None
    asked = jobs.validate_options(kind, d_tier, t_tier, options or {})
    raw, rest = editing.timeline_from_options(kind, d_tier, t_tier, asked, seconds)
    tl = editing.parse_timeline(raw, seconds)
    return {
        "options": rest,
        "denoise_tier": "standard" if kind != "transcribe" else None,  # (what editing.prepare_timeline_job says the job stores)
        "transcribe_tier": None,
        "input_ext": ext,
        "display_name": None,
        "channels": channels,
        "rates": {},
        "edit": {"timeline": raw, "first_run": True},
        "edit_rates": editing.step_rates(),
        **({"subtitle_rates": editing.subtitle_rates()} if tl.layers["subtitles"] else {}),
        "whole_file": {"kind": kind, "denoise_tier": d_tier, "transcribe_tier": t_tier, "options": asked},
        **jobs.convert_params(rest, ext),
    }


def queue_row(kind="denoise", denoise_tier="standard", transcribe_tier=None, options=None, *, next_stage=None, seconds=600.0) -> dict:
    """The part of a job's row the broker and the worker read to place it (kind, next stage, parameters), made like a job's: with its
    timeline."""
    return {"kind": kind, "next_stage": next_stage, "params_json": json.dumps(job_params(kind, denoise_tier, transcribe_tier, options, seconds=seconds))}


def file_bytes(job_id: str, name: str) -> bytes:
    """Fake file content that differs per job and file, so a copy can be told from a link."""
    return f"{job_id}:{name}".encode() * 8


def make_done_job(uid, *, kind="denoise", denoise_tier="standard", transcribe_tier=None, seconds=60.0, options=None,
                  channels=2, api_key_id=None, finished_ago=60, expires_in=None, cost=0, name="talk.wav", status="done", files=True,
                  ext=".wav", now=None) -> str:
    """A finished job with its folder: the upload, the audio and caption results. `finished_ago` seconds ago; it expires `expires_in`
    seconds from now (default: when the retention is over)."""
    now = db.now() if now is None else now
    jid = db.new_id()
    has_denoise = kind != "transcribe"
    has_text = kind != "denoise"
    params = job_params(kind, denoise_tier, transcribe_tier, options, seconds=seconds, channels=channels, ext=ext)
    fmt = params["whole_file"]["options"]["audio_format"]
    stem = Path(name).stem
    outputs = []
    if has_denoise:
        outputs.append({"file": f"{stem}-cleaned.{fmt}", "kind": "audio"})
    if has_text:
        outputs += [{"file": f"{stem}.srt", "kind": "captions", "lang": "en"}, {"file": f"{stem}-transcript.txt", "kind": "transcript", "lang": "en"}]
    d = jobs.job_dir(jid)
    names = [f"input{ext}", *(o["file"] for o in outputs)]
    if files:
        d.mkdir(parents=True, mode=0o700)
        for n in dict.fromkeys(names):
            (d / n).write_bytes(file_bytes(jid, n))
    for o in outputs:
        o["bytes"] = len(file_bytes(jid, o["file"]))
    finished = now - finished_ago
    expires = now + expires_in if expires_in is not None else finished + RETENTION
    db.execute(
        "INSERT INTO jobs(id, user_id, kind, tier, status, params_json, next_stage, input_name, input_bytes, received_bytes, claimed_seconds,"
        " duration_seconds, est_cost_micros, cost_micros, progress, stage, outputs_json, created_at, queued_at, started_at, finished_at,"
        " expires_at, api_key_id) VALUES (?,?,?,?,?,?,'probe',?,?,?,?,?,?,?,100,'Done',?,?,?,?,?,?,?)",
        (jid, uid, kind, "", status, json.dumps(params), name, 1000, 1000, seconds, seconds, cost, cost, json.dumps(outputs), finished - 30,
         finished - 30, finished - 20, finished, expires, api_key_id),
    )
    return jid


def add_job(uid, status="queued", *, created_ago=0, finished_ago=None, kind="denoise", api_key_id=None) -> str:
    """A job row in any state, without files (for the lists and the limits)."""
    jid = db.new_id()
    now = db.now()
    params = job_params(kind if kind != "convert" else "denoise")
    finished = None if finished_ago is None else now - finished_ago
    db.execute(
        "INSERT INTO jobs(id, user_id, kind, tier, status, params_json, input_name, input_bytes, claimed_seconds, est_cost_micros,"
        " created_at, queued_at, finished_at, expires_at, api_key_id)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (jid, uid, kind, "", status, json.dumps(params), "talk.wav", 1000, 60.0, 0, now - created_ago, now - created_ago, finished,
         None if finished is None else finished + RETENTION, api_key_id),
    )
    return jid


def deadline_in(job_id: str, seconds: int) -> int:
    """Move the job's upload so that its hard deadline (6 hours after the upload) is `seconds` from now; returns that moment."""
    from acw import jobs as _jobs

    cap = _jobs.get_settings().job_max_retention_seconds
    db.execute("UPDATE jobs SET uploaded_at = ? WHERE id = ?", (db.now() + seconds - cap, job_id))
    return db.now() + seconds


def folder(job_id: str) -> Path:
    return jobs.job_dir(job_id)


def row(job_id: str) -> dict:
    return dict(db.one("SELECT * FROM jobs WHERE id = ?", (job_id,)))


def set_card(others_mb: int, total_mb: int = 4096, age: float = 0.0) -> None:
    """What the GPU worker last wrote about the card's memory (broker.publish_card_state), `age` seconds ago: what other programs hold."""
    from acw import broker

    broker._card_file().write_text(json.dumps({"total_mb": total_mb, "others_mb": others_mb, "at": time.time() - age}))


def part_dir(job_id: str) -> Path:
    """The folder of the only subtitle part of a transcription made with whole-file options (subs/<key>): where its text and speech chunks go."""
    from acw import editing

    raw = params_of(job_id)["edit"]["timeline"]
    (part,) = editing._tl().Timeline.from_json(raw, seconds=raw["seconds"]).parts("subtitles")
    return editing.subtitle_dir(job_id, part)


def params_of(job_id: str) -> dict:
    return json.loads(row(job_id)["params_json"])


def files_of(job_id: str) -> set[str]:
    d = folder(job_id)
    return {p.name for p in d.iterdir()} if d.is_dir() else set()


def count(table: str, where="1=1", args=()) -> int:
    return db.one(f"SELECT COUNT(*) AS n FROM {table} WHERE {where}", args)["n"]


def complete_like_the_worker(job_id: str, outputs=(), scratch=("decoded.f32", "spec.json", "main-state.json")) -> dict:
    """Finish a queued job the way the worker does after the runner reported a result: marks it running, writes the
    result files (and the scratch files the runner leaves behind) the runner would have made, then runs the worker's own
    completion step (files kept or deleted, charge, expiry). `outputs`: file names of audio results. Returns the job's row."""
    from acw.worker import main as worker

    db.execute("UPDATE jobs SET status = 'running', worker_id = 't', started_at = ? WHERE id = ? AND status = 'queued'", (db.now(), job_id))
    job = row(job_id)
    d = folder(job_id)
    for name in (*outputs, *scratch):
        if not (d / name).exists():
            (d / name).write_bytes(b"new:" + name.encode())
    (d / "asr_chunks").mkdir(exist_ok=True)
    (d / "asr_chunks" / "0000.mp3").write_bytes(b"x")
    res = {"ok": True, "outputs": [{"file": n, "kind": "audio"} for n in outputs]}
    worker._complete(job, d, res)
    return row(job_id)
