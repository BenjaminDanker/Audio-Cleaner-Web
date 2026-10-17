"""The bulk-transfer process (acw/bulk_app.py): only uploads and result downloads, same auth and limits."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from acw import db, jobs
from acw.billing import ledger

from .conftest import ORIGIN, login

JOB = "a" * 32


@pytest.fixture
def bulk():
    from acw.bulk_app import create_bulk_app

    with TestClient(create_bulk_app(), base_url=ORIGIN) as c:
        yield c


def _paths(app):
    return {(m.upper(), path) for path, ops in app.openapi()["paths"].items() for m in ops}


def test_only_the_bulk_routes_exist():
    from acw.bulk_app import create_bulk_app

    assert _paths(create_bulk_app()) == {
        ("GET", "/api/health"),
        ("PUT", "/api/jobs/{job_id}/input"),
        ("GET", "/api/jobs/{job_id}/outputs/{name}"),
        # the release files (acw/api/downloads_routes.py): GET and HEAD each
        *{(m, p) for m in ("GET", "HEAD") for p in (
            "/api/downloads/desktop/{name}", "/api/downloads/obs-plugin/latest/{platform}",
            "/api/downloads/obs-plugin/{name}", "/api/downloads/models/{name}")},
    }


@pytest.mark.parametrize("method,path", [
    ("GET", "/api/me"),
    ("GET", "/api/jobs"),
    ("POST", "/api/jobs"),
    ("GET", f"/api/jobs/{JOB}"),
    ("POST", f"/api/jobs/{JOB}/cancel"),
    ("DELETE", f"/api/jobs/{JOB}"),
    ("POST", "/api/auth/email"),
    ("POST", "/api/stripe/webhook"),
    ("GET", "/api/keys"),
    ("GET", "/api/stream/ws"),
    ("GET", "/"),
    ("GET", "/api/docs"),
    ("GET", "/api/openapi.json"),
])
def test_everything_else_is_404(bulk, method, path):
    r = bulk.request(method, path, headers={"Origin": ORIGIN}, content=b"{}" if method == "POST" else None)
    assert r.status_code == 404, (method, path, r.status_code)


def test_health_answers_without_sign_in(bulk):
    r = bulk.get("/api/health")
    assert r.status_code == 200 and r.json()["ok"] is True
    assert r.headers["cache-control"] == "no-store" and r.headers["x-content-type-options"] == "nosniff"


def test_upload_and_download_need_a_sign_in(bulk):
    assert bulk.put(f"/api/jobs/{JOB}/input", content=b"x" * 10).status_code == 401
    assert bulk.get(f"/api/jobs/{JOB}/outputs/clean.wav").status_code == 401


def test_a_stream_only_api_key_cannot_use_it(client, bulk):
    from .test_api_security import make_key

    uid, _ = login(client)
    key = make_key(uid, "stream")
    client.cookies.clear()
    assert bulk.put(f"/api/jobs/{JOB}/input", content=b"x", headers={"Authorization": f"Bearer {key}"}).status_code == 403


def test_upload_through_the_bulk_process_completes_the_job(client, bulk):
    uid, h = login(client)
    ledger.credit(uid, 1_000_000, "grant", "t", "g")
    r = client.post("/api/jobs", json={"kind": "denoise", "denoiseTier": "standard", "filename": "a.wav",
                                       "sizeBytes": 300, "durationSeconds": 30}, headers=h)
    job_id = r.json()["id"]
    bulk.cookies.update(client.cookies)
    assert bulk.put(f"/api/jobs/{job_id}/input", content=b"x" * 299, headers=h).status_code == 411  # same size rule
    assert bulk.put(f"/api/jobs/{job_id}/input", content=b"x" * 300, headers=h).status_code == 200
    row = db.one("SELECT status, received_bytes FROM jobs WHERE id = ?", (job_id,))
    assert row["status"] == "queued" and row["received_bytes"] == 300
    # a second upload of the same job is refused like on the main process
    assert bulk.put(f"/api/jobs/{job_id}/input", content=b"x" * 300, headers=h).status_code == 409


def test_another_account_cannot_upload_to_the_job(client, bulk):
    uid, h = login(client, "a@example.com")
    ledger.credit(uid, 1_000_000, "grant", "t", "g")
    job_id = client.post("/api/jobs", json={"kind": "denoise", "denoiseTier": "standard", "filename": "a.wav",
                                            "sizeBytes": 100, "durationSeconds": 30}, headers=h).json()["id"]
    client.cookies.clear()
    _, h_b = login(client, "b@example.com")
    bulk.cookies.update(client.cookies)
    assert bulk.put(f"/api/jobs/{job_id}/input", content=b"x" * 100, headers=h_b).status_code == 404


def _done_job(uid, name="clean.wav", data=b"R" * 100_000):
    job_id = db.new_id()
    d = jobs.job_dir(job_id)
    d.mkdir(parents=True)
    (d / name).write_bytes(data)
    db.execute(
        "INSERT INTO jobs(id, user_id, kind, tier, status, params_json, input_name, input_bytes, claimed_seconds, outputs_json, created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (job_id, uid, "denoise", "standard", "done", json.dumps({"options": {}}), "a.wav", 1, 1.0,
         json.dumps([{"file": name, "kind": "audio", "bytes": len(data)}]), db.now()),
    )
    return job_id, data


def test_download_through_the_bulk_process(client, bulk, monkeypatch):
    import acw.bulk_app as b

    uid, _ = login(client)
    job_id, data = _done_job(uid)
    bulk.cookies.update(client.cookies)
    taken = []

    class Counting:
        async def take(self, n):
            taken.append(n)

    monkeypatch.setattr(b, "downloads", Counting())
    r = bulk.get(f"/api/jobs/{job_id}/outputs/clean.wav")
    assert r.status_code == 200 and r.content == data
    assert r.headers["cache-control"] == "private, no-store"
    assert sum(taken) == len(data)  # every byte went through the cap
    assert bulk.get(f"/api/jobs/{job_id}/outputs/other.wav").status_code == 404
    assert bulk.get(f"/api/jobs/{job_id}/outputs/..%2F..%2Fdb").status_code in (404, 422)


def test_download_cap_slows_the_transfer(client, bulk, monkeypatch):
    import acw.bulk_app as b
    from acw.security import DownloadBandwidth

    uid, _ = login(client)
    job_id, data = _done_job(uid, data=b"R" * 1_000_000)
    bulk.cookies.update(client.cookies)
    monkeypatch.setenv("ACW_DOWNLOAD_MBPS_TOTAL", "0.5")
    monkeypatch.setattr(b, "downloads", DownloadBandwidth())
    t0 = time.monotonic()
    r = bulk.get(f"/api/jobs/{job_id}/outputs/clean.wav")
    elapsed = time.monotonic() - t0
    assert r.status_code == 200 and len(r.content) == len(data)
    assert elapsed > 0.8  # 1 MB at 0.5 MB/s, the first second's worth is free


def test_download_cap_zero_means_unlimited(monkeypatch):
    from acw.security import DownloadBandwidth

    monkeypatch.setenv("ACW_DOWNLOAD_MBPS_TOTAL", "0")
    assert DownloadBandwidth().total is None


def test_download_cap_default_is_five_megabytes_per_second_shared_with_uploads(monkeypatch):
    from acw.security import DownloadBandwidth

    monkeypatch.delenv("ACW_DOWNLOAD_MBPS_TOTAL", raising=False)
    monkeypatch.delenv("ACW_UPLOAD_MBPS_TOTAL", raising=False)
    monkeypatch.delenv("ACW_TRANSFER_MBPS_TOTAL", raising=False)
    assert DownloadBandwidth().total.rate == 5e6


def test_the_cap_applies_under_the_base_path(monkeypatch):
    from acw.bulk_app import DownloadThrottleMiddleware

    mw = DownloadThrottleMiddleware(None, base_path="/audio")
    scope = {"type": "http", "method": "GET", "path": f"/audio/api/jobs/{JOB}/outputs/x.wav"}
    assert mw.applies(scope)
    assert not mw.applies({**scope, "path": f"/audio/api/jobs/{JOB}/input"})
    assert not mw.applies({**scope, "method": "PUT"})
    assert not mw.applies({"type": "websocket", "path": scope["path"], "method": "GET"})
