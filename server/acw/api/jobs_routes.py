from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import stat
import time

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .. import auth, db, jobs, keycaps, webhooks
from ..billing import ledger
from ..config import get_settings
from ..security import client_ip, client_net, enforce, uploads

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

MIME = {
    ".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".opus": "audio/ogg",
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".m4v": "video/mp4", ".mkv": "video/x-matroska",
    ".webm": "video/webm", ".srt": "application/x-subrip", ".vtt": "text/vtt", ".txt": "text/plain; charset=utf-8",
    ".json": "application/json",
}


class CreateJob(BaseModel):
    kind: str = Field(max_length=32)
    denoiseTier: str | None = Field(default=None, max_length=32)
    transcribeTier: str | None = Field(default=None, max_length=32)
    filename: str = Field(min_length=1, max_length=255)
    sizeBytes: int = Field(gt=0)
    durationSeconds: float = Field(gt=0)
    free: bool = False
    displayName: str | None = Field(default=None, max_length=300)
    callbackUrl: str | None = Field(default=None, max_length=600)
    options: dict = Field(default_factory=dict)


class EstimateJob(BaseModel):
    kind: str = Field(max_length=32)
    denoiseTier: str | None = Field(default=None, max_length=32)
    transcribeTier: str | None = Field(default=None, max_length=32)
    durationSeconds: float = Field(gt=0)
    options: dict = Field(default_factory=dict)


IDEMPOTENCY_KEY = re.compile(r"^[\x21-\x7e]{1,128}$")  # visible ASCII, no spaces
KEY_CAP_HEADERS = {"X-Error-Code": "key_spending_cap"}


ALMOST_DONE = "This job is almost done and can’t be canceled now."


def _get_owned(job_id: str, principal: auth.Principal) -> dict:
    if len(job_id) != 32:
        raise HTTPException(status_code=404, detail="Job not found.")
    row = db.one("SELECT * FROM jobs WHERE id = ? AND user_id = ?", (job_id, principal.user_id))
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return dict(row)


def _upload_url(job_id: str) -> str:
    return f"api/jobs/{job_id}/input"


@router.post("")
def create(body: CreateJob, request: Request, response: Response, principal: auth.Principal = Depends(auth.require_scope("jobs"))) -> dict:
    # Idempotency-Key: the same key (this account, 24 hours) names the same job. A repeat is answered before anything
    # else, so it uses none of the job creation allowance and holds no money twice.
    idem = None
    raw_key = request.headers.get("idempotency-key")
    if raw_key is not None:
        raw_key = raw_key.strip()
        if not IDEMPOTENCY_KEY.match(raw_key):
            raise HTTPException(status_code=400, detail="Idempotency-Key must be 1 to 128 visible characters.")
        digest = hashlib.sha256(json.dumps(body.model_dump(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        idem = (raw_key, digest)
        try:
            again = jobs.replay_of(principal.user_id, *idem)
        except jobs.JobError as e:
            raise HTTPException(status_code=e.status, detail=str(e))
        if again is not None:
            return _replayed(again, response)
    enforce("job_create", principal.user_id)
    enforce("job_create", f"ip:{client_net(client_ip(request))}")
    auth.accept_terms(principal.user_id, "continued use", client_ip(request))
    if len(json.dumps(body.options)) > 2000:
        raise HTTPException(status_code=400, detail="options too large")
    callback = None
    if body.callbackUrl is not None:
        try:
            callback = webhooks.check_url(body.callbackUrl)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    try:
        created = jobs.create_job(
            principal.user_id,
            kind=body.kind,
            denoise_tier=body.denoiseTier,
            transcribe_tier=body.transcribeTier,
            filename=os.path.basename(body.filename),
            size_bytes=body.sizeBytes,
            claimed_seconds=body.durationSeconds,
            options=body.options,
            free=body.free,
            display_name=body.displayName,
            api_key_id=principal.api_key_id,
            callback_url=callback,
            idempotency=idem,
        )
    except jobs.Replay as r:  # a parallel request with the same key won
        return _replayed({"id": r.job_id, "estCostMicros": r.est_cost_micros}, response)
    except keycaps.KeyCapReached as e:
        raise HTTPException(status_code=403, detail=str(e), headers=KEY_CAP_HEADERS)
    except jobs.JobError as e:
        raise HTTPException(status_code=e.status, detail=str(e), headers={"Retry-After": str(e.retry_after or 60)} if e.status == 429 else None)
    except ValueError as e:  # pricing errors
        raise HTTPException(status_code=400, detail=str(e))
    if callback:
        with db.tx() as c:
            webhooks.ensure_secret(c, principal.user_id)  # so there is a secret to see (and verify with) in the API keys card
    return {**created, "uploadUrl": _upload_url(created["id"])}


def _replayed(found: dict, response: Response) -> dict:
    response.headers["Idempotent-Replayed"] = "true"
    return {**found, "uploadUrl": _upload_url(found["id"]), "idempotentReplay": True}


@router.post("/estimate")
def estimate(body: EstimateJob, principal: auth.Principal = Depends(auth.require_scope("jobs"))) -> dict:
    """The price the website shows for these choices, without creating anything or holding any money."""
    if len(json.dumps(body.options)) > 2000:
        raise HTTPException(status_code=400, detail="options too large")
    try:
        out = jobs.estimate(body.kind, body.denoiseTier, body.transcribeTier, body.durationSeconds, body.options)
    except jobs.JobError as e:
        raise HTTPException(status_code=e.status, detail=str(e))
    except ValueError as e:  # pricing errors
        raise HTTPException(status_code=400, detail=str(e))
    available = ledger.balance(principal.user_id)["available_micros"]
    out["availableMicros"] = available
    out["affordable"] = available >= out["estCostMicros"]
    if principal.api_key_id:
        limit = keycaps.key_limit(principal.api_key_id)
        if limit is not None:
            out["keyLimit"] = limit
    return out


# An upload that sends nothing for this long is abandoned (it would hold a connection and its disk space).
UPLOAD_IDLE_SECONDS = 30
UPLOAD_WRITE_BLOCK = 1024 * 1024  # chunks are collected to this size, then written off the event loop
UPLOAD_CHECK_EVERY = 256 * 1024 * 1024  # re-check free disk space this often while bytes arrive
UPLOAD_KEEPALIVE_EVERY = 60  # seconds between pushing the upload's expiry back while bytes keep coming


def _begin_upload(job_id: str, principal: auth.Principal) -> dict:
    enforce("upload", principal.user_id)
    job = _get_owned(job_id, principal)
    if job["status"] != "uploading":
        raise HTTPException(status_code=409, detail="This job is not waiting for an upload.")
    return job


def _finish_upload(job_id: str, received: int) -> None:
    db.execute("UPDATE jobs SET received_bytes = ? WHERE id = ?", (received, job_id))
    jobs.mark_uploaded(job_id)


def _still_uploading(job_id: str) -> bool:
    """Called while bytes arrive: pushes the expiry back (a slow link may need hours) and says whether the job
    is still waiting for its upload (it was not canceled or swept)."""
    n = db.execute(
        "UPDATE jobs SET expires_at = ? WHERE id = ? AND status = 'uploading'", (db.now() + jobs.UPLOAD_WINDOW_SECONDS, job_id)
    ).rowcount
    return bool(n)


@router.put("/{job_id}/input")
async def upload(job_id: str, request: Request, principal: auth.Principal = Depends(auth.require_scope("jobs"))) -> dict:
    # Every database and disk call below runs off the event loop: it also carries live audio.
    job = await anyio.to_thread.run_sync(_begin_upload, job_id, principal)
    declared = job["input_bytes"]
    length = request.headers.get("content-length")
    if length is None or not length.isdigit() or int(length) != declared:
        raise HTTPException(status_code=411, detail="Content-Length must match the declared file size.")
    path = jobs.input_path(job)
    received = 0
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise HTTPException(status_code=409, detail="Upload already in progress.")
    data_dir = get_settings().data_dir
    user_throttle = uploads.for_user(principal.user_id)
    pending: list[bytes] = []
    pending_n = 0
    last_check = time.monotonic()
    last_keepalive = float("-inf")  # the first bytes move the expiry from "must start soon" to the full upload window
    checked_at = 0
    stream = request.stream().__aiter__()
    try:
        with os.fdopen(fd, "wb") as f:
            while True:
                try:
                    chunk = await asyncio.wait_for(stream.__anext__(), UPLOAD_IDLE_SECONDS)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    raise HTTPException(status_code=408, detail="The upload stalled; please try again.")
                if not chunk:
                    continue
                received += len(chunk)
                if received > declared:
                    raise HTTPException(status_code=413, detail="Upload is larger than declared.")
                await uploads.take(user_throttle, len(chunk))
                pending.append(chunk)
                pending_n += len(chunk)
                if pending_n >= UPLOAD_WRITE_BLOCK:
                    await anyio.to_thread.run_sync(f.write, b"".join(pending))
                    pending, pending_n = [], 0
                if received - checked_at >= UPLOAD_CHECK_EVERY:
                    checked_at = received
                    free = await anyio.to_thread.run_sync(_disk_free, data_dir)
                    if free < jobs.DISK_RESERVE // 2:
                        raise HTTPException(status_code=507, detail="We're short on storage space right now. Please try again a bit later.")
                if time.monotonic() - last_keepalive >= UPLOAD_KEEPALIVE_EVERY:
                    last_keepalive = time.monotonic()
                    if not await anyio.to_thread.run_sync(_still_uploading, job_id):
                        raise HTTPException(status_code=409, detail="This upload was canceled.")
            if pending:
                await anyio.to_thread.run_sync(f.write, b"".join(pending))
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    if received != declared:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Upload was incomplete, try again.")
    await anyio.to_thread.run_sync(_finish_upload, job_id, received)
    return {"ok": True}


def _is_plain_file(path) -> bool:
    """A regular file on the job directory's own device, never a link: outputs are made by the sandboxed runner."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return False
    try:
        st = os.fstat(fd)
        return stat.S_ISREG(st.st_mode) and st.st_dev == os.stat(path.parent).st_dev
    finally:
        os.close(fd)


def _disk_free(path) -> int:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


@router.get("")
def list_jobs(principal: auth.Principal = Depends(auth.require_scope("jobs"))) -> dict:
    rows = db.all_rows(
        "SELECT * FROM jobs WHERE user_id = ? AND status != 'expired' ORDER BY created_at DESC LIMIT 50",
        (principal.user_id,),
    )
    queue = jobs.queue_snapshot() if any(r["status"] == "queued" for r in rows) else ([], {})  # shared for a second across requests
    callbacks = jobs.callbacks_for([r["id"] for r in rows])
    return {"jobs": [jobs.public_job(dict(r), queue, callbacks) for r in rows]}


@router.get("/{job_id}")
def get_job(job_id: str, principal: auth.Principal = Depends(auth.require_scope("jobs"))) -> dict:
    return jobs.public_job(_get_owned(job_id, principal))


@router.post("/{job_id}/cancel")
def cancel(job_id: str, principal: auth.Principal = Depends(auth.require_scope("jobs"))) -> dict:
    job = _get_owned(job_id, principal)
    if job["status"] not in ("uploading", "queued"):
        raise HTTPException(status_code=409, detail="Only jobs that have not started can be canceled.")
    if job["next_stage"] == "finalize":
        # The costly part (extra-accurate transcription) is done; only the packaging is left.
        raise HTTPException(status_code=409, detail=ALMOST_DONE)
    # One transaction moves the status and releases the money; a worker can't claim the job in between.
    # Only a queued job that is still before its last stage: the check is repeated inside that transaction.
    if not jobs.finish_failed(job, "Canceled.", status="canceled", only_from=("uploading", "queued"), not_stage="finalize"):
        raise HTTPException(status_code=409, detail="This job has just started and can’t be canceled now.")
    return {"ok": True}


@router.delete("/{job_id}")
def delete(job_id: str, principal: auth.Principal = Depends(auth.require_scope("jobs"))) -> dict:
    """Delete outputs now instead of waiting for expiry."""
    job = _get_owned(job_id, principal)
    if job["status"] in ("running", "cloud_asr", "queued", "uploading"):
        raise HTTPException(status_code=409, detail="Cancel or wait for the job first.")
    import shutil

    shutil.rmtree(jobs.job_dir(job_id), ignore_errors=True)
    db.execute("UPDATE jobs SET status = 'expired', outputs_json = NULL WHERE id = ?", (job_id,))
    return {"ok": True}


@router.get("/{job_id}/outputs/{name}")
def download(job_id: str, name: str, principal: auth.Principal = Depends(auth.require_scope("jobs"))) -> FileResponse:
    job = _get_owned(job_id, principal)
    if job["status"] != "done":
        raise HTTPException(status_code=404, detail="Not available.")
    outputs = json.loads(job["outputs_json"] or "[]")
    if name not in {o["file"] for o in outputs}:
        raise HTTPException(status_code=404, detail="Not found.")
    path = jobs.job_dir(job_id) / name
    if path.parent != jobs.job_dir(job_id) or not _is_plain_file(path):
        raise HTTPException(status_code=404, detail="Not found.")
    return FileResponse(
        path,
        media_type=MIME.get(path.suffix.lower(), "application/octet-stream"),
        filename=name,
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )
