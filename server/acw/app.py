"""FastAPI application: `uvicorn acw.app:app --host 127.0.0.1 --port 8790`.

The process binds to loopback only. Public traffic reaches it through Caddy on
the VPS and an SSH reverse tunnel, so nothing listens on the home LAN.
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from . import alerts, auth, db, httpstack, jobs, retention
from . import health as health_checks
from .api import auth_routes, billing_routes, contact_routes, jobs_routes, keys_routes
from .asr import openai_asr
from .billing import ledger, stripe_service
from .config import get_settings
from .stream import routes as stream_routes

log = logging.getLogger("acw")


PREMIUM_JOBS_AT_ONCE = 2
_cloud_tasks: set[asyncio.Task] = set()


def _claim_cloud_jobs(limit: int) -> list[dict]:
    claimed = []
    for job in db.all_rows("SELECT * FROM jobs WHERE status = 'cloud_asr' AND stage != 'Transcribing' ORDER BY queued_at LIMIT ?", (limit,)):
        job = dict(job)
        n = db.execute(
            "UPDATE jobs SET stage = 'Transcribing', heartbeat_at = ? WHERE id = ? AND status = 'cloud_asr' AND stage != 'Transcribing'",
            (db.now(), job["id"]),
        ).rowcount
        if n:
            claimed.append(job)
    return claimed


def _cloud_done(job: dict) -> None:
    db.execute(
        "UPDATE jobs SET status = 'queued', next_stage = 'finalize', stage = 'Finishing', progress = 90,"
        " queued_at = ? WHERE id = ? AND status = 'cloud_asr'",
        (db.now(), job["id"]),
    )


def _beat(job_id: str) -> None:
    db.execute("UPDATE jobs SET heartbeat_at = ? WHERE id = ? AND status = 'cloud_asr'", (db.now(), job_id))


async def _run_cloud_job(job: dict) -> None:
    """Extra-accurate transcription of one job. Segments already transcribed are kept on disk, so a restart
    or a failed attempt doesn't pay for them twice."""

    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(60)
            await asyncio.to_thread(_beat, job["id"])

    beat = asyncio.create_task(heartbeat())
    try:
        lang = json.loads(job["params_json"])["options"].get("language")
        await openai_asr.run_for_job(jobs.job_dir(job["id"]), lang, job["duration_seconds"] or job["claimed_seconds"])
        await asyncio.to_thread(_cloud_done, job)
    except Exception:  # noqa: BLE001
        log.exception("cloud ASR failed for %s", job["id"])
        await asyncio.to_thread(jobs.finish_failed, job, "Extra accurate transcription failed; you were not charged.")
    finally:
        beat.cancel()


async def _cloud_asr_loop() -> None:
    while True:
        await asyncio.sleep(3)
        try:
            _cloud_tasks.difference_update({t for t in _cloud_tasks if t.done()})
            free = PREMIUM_JOBS_AT_ONCE - len(_cloud_tasks)
            if free > 0:
                for job in await asyncio.to_thread(_claim_cloud_jobs, free):
                    t = asyncio.create_task(_run_cloud_job(job))
                    _cloud_tasks.add(t)
        except Exception:  # noqa: BLE001
            log.exception("cloud ASR loop failed")


def _housekeeping() -> None:
    steps = (
        auth.purge_expired,
        # Cloud ASR that never finished (process restart mid-request): retry.
        lambda: db.execute(
            "UPDATE jobs SET stage = 'Waiting to transcribe' WHERE status = 'cloud_asr'"
            " AND stage = 'Transcribing' AND heartbeat_at < ?",
            (db.now() - 600,),
        ),
        ledger.reap_stale_holds,  # money reserved by things that are over
        stream_routes.close_orphans,  # live sessions nothing is running any more
        retention.sweep,  # privacy retention (docs/operations.md)
    )
    for step in steps:
        try:
            step()
        except Exception:  # noqa: BLE001
            log.exception("housekeeping step failed")


async def _housekeeping_loop() -> None:
    while True:
        await asyncio.sleep(30)
        try:
            await asyncio.to_thread(_housekeeping)
            if stripe_service.enabled():
                await asyncio.to_thread(stripe_service.reconcile)  # payments whose webhook never came
                for user_id in await asyncio.to_thread(stripe_service.users_needing_reload):
                    await asyncio.to_thread(stripe_service.run_auto_reload, user_id)
                for payment_id in await asyncio.to_thread(stripe_service.pending_tax_transactions):
                    await asyncio.to_thread(stripe_service._record_tax_transaction, payment_id)
        except Exception:  # noqa: BLE001
            log.exception("housekeeping failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    # A previous process died while transcribing: pick those jobs up again right away (finished segments are kept).
    db.execute("UPDATE jobs SET stage = 'Waiting to transcribe' WHERE status = 'cloud_asr' AND stage = 'Transcribing'")
    try:
        await asyncio.to_thread(alerts.web_started)  # mails if the last run ended uncleanly
    except Exception:  # noqa: BLE001
        log.exception("alert start-up check failed")
    tasks = [asyncio.create_task(_cloud_asr_loop()), asyncio.create_task(_housekeeping_loop())]
    await stream_routes.startup()
    try:
        yield
    finally:
        for t in tasks:
            t.cancel()
        await stream_routes.shutdown()
        try:
            alerts.web_stopping()  # clean-stop marker: its absence at next start means a crash
        except Exception:  # noqa: BLE001
            log.exception("could not record clean stop")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Audio Cleaner",
        lifespan=lifespan,
        docs_url=None if settings.production else "/api/docs",
        redoc_url=None,
        openapi_url=None if settings.production else "/api/openapi.json",
        root_path=settings.base_path,
    )

    httpstack.install(app, settings)

    @app.get("/api/health")
    def health(request: Request, deep: int = 0, principal: auth.Principal | None = Depends(auth.optional_principal)) -> dict:
        db.one("SELECT 1")
        if not deep:
            return {"ok": True}
        # Details about the machine: from this machine itself (no proxy header: not via the tunnel) or an admin.
        local = request.client is not None and request.client.host in ("127.0.0.1", "::1") and "x-forwarded-for" not in request.headers
        if not (local or (principal is not None and principal.is_admin and principal.via == "session")):
            return {"ok": True}
        return health_checks.deep()

    for r in (
        auth_routes.router,
        auth_routes.me_router,
        billing_routes.router,
        billing_routes.webhook_router,
        contact_routes.router,
        jobs_routes.router,
        keys_routes.router,
        stream_routes.router,
    ):
        app.include_router(r)

    if settings.web_dist and settings.web_dist.is_dir():
        _mount_spa(app, settings.web_dist)
    return app


def _mount_spa(app: FastAPI, dist: Path) -> None:
    """Dev convenience: serve the built frontend. In production Caddy on the
    VPS serves it so the free browser tier never touches the home link."""
    dist = dist.resolve()
    index = dist / "index.html"

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(status_code=404)
        candidate = (dist / path).resolve()
        if path and candidate.is_file() and dist in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(index, headers={"Cache-Control": "no-cache"})


app = create_app()
