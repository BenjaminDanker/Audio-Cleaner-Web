"""Shared helpers of the re-run tests ("Change and save again"): finished jobs made by hand with their folders (fake kept
copies and outputs, no native binaries), accounts and API keys, a frozen clock and a roomy disk."""
import json
import os
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


# ------------------------------------------------------------------------------------------- fake finished jobs

def kept_record(seconds=60.0, *, noise=True, input_copy=True, rebuilt=False, channels=2, echo=False, profile="light",
                noise_opt="auto", segments=None) -> dict:
    """The `kept` record a finished job stores (as rerun.collect_kept builds it)."""
    cols = list(range(channels))
    samples = int(seconds * SR)
    base = {"channels": cols, "scale": 1.0, "samples": samples, "echo": echo}
    kept: dict = {}
    if noise:
        kept["noise"] = {"file": "kept-noise.flac", **base, "profile": profile, "noise_opt": noise_opt, "plan": list(cols)}
    if noise and input_copy:
        kept["input"] = {"file": "kept-input.flac", **base}
    if rebuilt:
        kept["rebuilt"] = {"file": "kept-rebuilt.flac", **base}
    if segments is not None:
        kept["segments"] = {"tier": "local", "language": "en", "echo": False, "translations": [], **segments}
    return kept


def job_params(kind="denoise", denoise_tier="standard", transcribe_tier=None, options=None, *, seconds=60.0, kept=None, channels=2,
               ext=".wav") -> dict:
    """The stored parameters of a finished first-run job (real validated options and pinned rates)."""
    d_tier = denoise_tier if kind != "transcribe" else None
    t_tier = transcribe_tier if kind != "denoise" else None
    opts = jobs.validate_options(kind, d_tier, t_tier, options or {})
    leveling = kind != "transcribe" and opts["leveling"]
    highs = kind != "transcribe" and opts["restore_highs"]
    return {
        "options": opts,
        "denoise_tier": d_tier,
        "transcribe_tier": t_tier,
        "input_ext": ext,
        "display_name": None,
        "channels": channels,
        "rates": pricing.job_rates(kind, d_tier, t_tier, len(opts["translate"]), leveling, highs),
        "kept": kept if kept is not None else {},
        "noise_profile": "light",
    }


def file_bytes(job_id: str, name: str) -> bytes:
    """Fake file content that differs per job and file, so a copy can be told from a link."""
    return f"{job_id}:{name}".encode() * 8


def make_done_job(uid, *, kind="denoise", denoise_tier="standard", transcribe_tier=None, seconds=60.0, options=None,
                  noise=True, input_copy=True, rebuilt=False, segments=None, channels=2, api_key_id=None, finished_ago=60,
                  expires_in=None, root=None, source=None, cost=0, name="talk.wav", status="done", files=True, echo=False,
                  noise_opt="auto", profile="light", ext=".wav", now=None) -> str:
    """A finished job with its folder: the upload, the kept copies that `kept` names, the audio and caption results.
    `finished_ago` seconds ago; it expires `expires_in` seconds from now (default: when the retention is over)."""
    now = db.now() if now is None else now
    jid = db.new_id()
    has_denoise = kind != "transcribe"
    has_text = kind != "denoise"
    kept = kept_record(seconds, noise=noise and has_denoise and denoise_tier == "standard", input_copy=input_copy,
                       rebuilt=rebuilt, channels=channels, echo=echo, noise_opt=noise_opt, profile=profile,
                       segments=segments if has_text else None)
    params = job_params(kind, denoise_tier, transcribe_tier, options, seconds=seconds, kept=kept, channels=channels, ext=ext)
    fmt = params["options"]["audio_format"]
    stem = Path(name).stem
    outputs = []
    if has_denoise:
        outputs.append({"file": f"{stem}-cleaned.{fmt}", "kind": "audio"})
    if has_text:
        outputs += [{"file": f"{stem}.srt", "kind": "captions", "lang": "en"}, {"file": f"{stem}-transcript.txt", "kind": "transcript", "lang": "en"}]
    d = jobs.job_dir(jid)
    names = [f"input{ext}", *(o["file"] for o in outputs)]
    names += [e["file"] for k, e in kept.items() if k in ("noise", "rebuilt", "input")]
    if "segments" in kept:
        names += ["segments.json", *(f"segments.{lang}.json" for lang in kept["segments"]["translations"])]
    if files:
        d.mkdir(parents=True, mode=0o700)
        for n in dict.fromkeys(names):
            (d / n).write_bytes(file_bytes(jid, n))
    for o in outputs:
        o["bytes"] = len(file_bytes(jid, o["file"]))
    finished = now - finished_ago
    expires = now + expires_in if expires_in is not None else finished + RETENTION
    tier = "/".join(t for t in (params["denoise_tier"], params["transcribe_tier"]) if t)
    db.execute(
        "INSERT INTO jobs(id, user_id, kind, tier, status, params_json, next_stage, input_name, input_bytes, received_bytes, claimed_seconds,"
        " duration_seconds, est_cost_micros, cost_micros, progress, stage, outputs_json, created_at, queued_at, started_at, finished_at,"
        " expires_at, api_key_id, source_job_id, root_job_id) VALUES (?,?,?,?,?,?,'probe',?,?,?,?,?,?,?,100,'Done',?,?,?,?,?,?,?,?,?)",
        (jid, uid, kind, tier, status, json.dumps(params), name, 1000, 1000, seconds, seconds, cost, cost, json.dumps(outputs), finished - 30,
         finished - 30, finished - 20, finished, expires, api_key_id, source, root),
    )
    return jid


def add_job(uid, status="queued", *, created_ago=0, finished_ago=None, kind="denoise", api_key_id=None, root=None, source=None) -> str:
    """A job row in any state, without files (for the lists and the limits)."""
    jid = db.new_id()
    now = db.now()
    params = {"options": jobs.validate_options("denoise", "standard", None, {}), "denoise_tier": "standard", "transcribe_tier": None,
              "input_ext": ".wav", "display_name": None, "channels": 2, "kept": {}}
    finished = None if finished_ago is None else now - finished_ago
    db.execute(
        "INSERT INTO jobs(id, user_id, kind, tier, status, params_json, input_name, input_bytes, claimed_seconds, est_cost_micros,"
        " created_at, queued_at, finished_at, expires_at, api_key_id, source_job_id, root_job_id)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (jid, uid, kind, "standard", status, json.dumps(params), "talk.wav", 1000, 60.0, 0, now - created_ago, now - created_ago, finished,
         None if finished is None else finished + RETENTION, api_key_id, source, root),
    )
    return jid


def folder(job_id: str) -> Path:
    return jobs.job_dir(job_id)


def row(job_id: str) -> dict:
    return dict(db.one("SELECT * FROM jobs WHERE id = ?", (job_id,)))


def params_of(job_id: str) -> dict:
    return json.loads(row(job_id)["params_json"])


def files_of(job_id: str) -> set[str]:
    d = folder(job_id)
    return {p.name for p in d.iterdir()} if d.is_dir() else set()


def count(table: str, where="1=1", args=()) -> int:
    return db.one(f"SELECT COUNT(*) AS n FROM {table} WHERE {where}", args)["n"]


def same_file(a: Path, b: Path) -> bool:
    """Both names are the same file on disk (a hard link, not a copy)."""
    sa, sb = os.stat(a), os.stat(b)
    return (sa.st_dev, sa.st_ino) == (sb.st_dev, sb.st_ino)


def complete_like_the_worker(job_id: str, outputs=(), kept=None, scratch=("decoded.f32", "spec.json", "main-state.json")) -> dict:
    """Finish a queued job the way the worker does after the runner reported a result: marks it running, writes the
    result files (and the scratch files the runner leaves behind) the runner would have made, then runs the worker's own
    completion step (files kept or deleted, charge, expiry). `outputs`: file names of audio results; `kept`: what the
    runner reports having kept. Returns the job's row."""
    from acw.worker import main as worker

    db.execute("UPDATE jobs SET status = 'running', worker_id = 't', started_at = ? WHERE id = ? AND status = 'queued'", (db.now(), job_id))
    job = row(job_id)
    d = folder(job_id)
    for name in (*outputs, *scratch):
        if not (d / name).exists():
            (d / name).write_bytes(b"new:" + name.encode())
    (d / "asr_chunks").mkdir(exist_ok=True)
    (d / "asr_chunks" / "0000.mp3").write_bytes(b"x")
    res = {"ok": True, "outputs": [{"file": n, "kind": "audio"} for n in outputs], "kept": kept or {}}
    worker._complete(job, d, res)
    return row(job_id)
