from __future__ import annotations

import json
import os

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .. import auth, db, jobs
from ..security import client_ip, enforce

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
    options: dict = Field(default_factory=dict)


def _get_owned(job_id: str, principal: auth.Principal) -> dict:
    if len(job_id) != 32:
        raise HTTPException(status_code=404, detail="Job not found.")
    row = db.one("SELECT * FROM jobs WHERE id = ? AND user_id = ?", (job_id, principal.user_id))
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return dict(row)


@router.post("")
def create(body: CreateJob, request: Request, principal: auth.Principal = Depends(auth.require_scope("jobs"))) -> dict:
    enforce("job_create", principal.user_id)
    enforce("job_create", f"ip:{client_ip(request)}")
    auth.accept_terms(principal.user_id, "continued use", client_ip(request))
    if len(json.dumps(body.options)) > 2000:
        raise HTTPException(status_code=400, detail="options too large")
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
        )
    except jobs.JobError as e:
        raise HTTPException(status_code=e.status, detail=str(e))
    except ValueError as e:  # pricing errors
        raise HTTPException(status_code=400, detail=str(e))
    return {**created, "uploadUrl": f"api/jobs/{created['id']}/input"}


@router.put("/{job_id}/input")
async def upload(job_id: str, request: Request, principal: auth.Principal = Depends(auth.require_scope("jobs"))) -> dict:
    enforce("upload", principal.user_id)
    job = _get_owned(job_id, principal)
    if job["status"] != "uploading":
        raise HTTPException(status_code=409, detail="This job is not waiting for an upload.")
    declared = job["input_bytes"]
    length = request.headers.get("content-length")
    if length is None or int(length) != declared:
        raise HTTPException(status_code=411, detail="Content-Length must match the declared file size.")
    path = jobs.input_path(job)
    received = 0
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise HTTPException(status_code=409, detail="Upload already in progress.")
    try:
        with os.fdopen(fd, "wb") as f:
            async for chunk in request.stream():
                received += len(chunk)
                if received > declared:
                    raise HTTPException(status_code=413, detail="Upload is larger than declared.")
                f.write(chunk)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    if received != declared:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Upload was incomplete, try again.")
    db.execute("UPDATE jobs SET received_bytes = ? WHERE id = ?", (received, job_id))
    jobs.mark_uploaded(job_id)
    return {"ok": True}


@router.get("")
def list_jobs(principal: auth.Principal = Depends(auth.require_scope("jobs"))) -> dict:
    rows = db.all_rows(
        "SELECT * FROM jobs WHERE user_id = ? AND status != 'expired' ORDER BY created_at DESC LIMIT 50",
        (principal.user_id,),
    )
    queue = jobs.queued_ids()
    return {"jobs": [jobs.public_job(dict(r), queue) for r in rows]}


@router.get("/{job_id}")
def get_job(job_id: str, principal: auth.Principal = Depends(auth.require_scope("jobs"))) -> dict:
    return jobs.public_job(_get_owned(job_id, principal))


@router.post("/{job_id}/cancel")
def cancel(job_id: str, principal: auth.Principal = Depends(auth.require_scope("jobs"))) -> dict:
    job = _get_owned(job_id, principal)
    if job["status"] not in ("uploading", "queued"):
        raise HTTPException(status_code=409, detail="Only jobs that have not started can be canceled.")
    with db.tx() as c:
        n = c.execute(
            "UPDATE jobs SET status = 'canceling' WHERE id = ? AND status IN ('uploading','queued')", (job_id,)
        ).rowcount
    if n:
        # 'canceling' keeps workers from claiming it; finish_failed releases money.
        jobs.finish_failed(job, "Canceled.", status="canceled")
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
    if path.parent != jobs.job_dir(job_id) or not path.is_file():
        raise HTTPException(status_code=404, detail="Not found.")
    return FileResponse(
        path,
        media_type=MIME.get(path.suffix.lower(), "application/octet-stream"),
        filename=name,
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )
