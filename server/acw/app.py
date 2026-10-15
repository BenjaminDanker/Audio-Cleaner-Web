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

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response

from . import alerts, auth, db, jobs
from .api import auth_routes, billing_routes, jobs_routes, keys_routes
from .asr import openai_asr
from .billing import stripe_service
from .config import get_settings
from .security import API_CSP, SECURITY_HEADERS, client_ip, enforce
from .stream import routes as stream_routes

log = logging.getLogger("acw")


async def _cloud_asr_loop() -> None:
    while True:
        await asyncio.sleep(3)
        for job in db.all_rows("SELECT * FROM jobs WHERE status = 'cloud_asr' AND stage != 'cloud asr running'"):
            job = dict(job)
            n = db.execute(
                "UPDATE jobs SET stage = 'cloud asr running', heartbeat_at = ? WHERE id = ? AND status = 'cloud_asr'"
                " AND stage != 'cloud asr running'",
                (db.now(), job["id"]),
            ).rowcount
            if not n:
                continue
            lang = json.loads(job["params_json"])["options"].get("language")
            try:
                await openai_asr.run_for_job(jobs.job_dir(job["id"]), lang)
            except Exception:  # noqa: BLE001
                log.exception("cloud ASR failed for %s", job["id"])
                jobs.finish_failed(job, "Extra accurate transcription failed; you were not charged.")
                continue
            db.execute(
                "UPDATE jobs SET status = 'queued', next_stage = 'finalize', stage = 'finishing', progress = 90,"
                " queued_at = ? WHERE id = ? AND status = 'cloud_asr'",
                (db.now(), job["id"]),
            )


async def _housekeeping_loop() -> None:
    while True:
        await asyncio.sleep(30)
        try:
            auth.purge_expired()
            # Cloud ASR that never finished (process restart mid-request): retry.
            db.execute(
                "UPDATE jobs SET stage = 'transcribing (premium)' WHERE status = 'cloud_asr'"
                " AND stage = 'cloud asr running' AND heartbeat_at < ?",
                (db.now() - 1800,),
            )
            if stripe_service.enabled():
                for user_id in stripe_service.users_needing_reload():
                    await asyncio.to_thread(stripe_service.run_auto_reload, user_id)
                for payment_id in stripe_service.pending_tax_transactions():
                    await asyncio.to_thread(stripe_service._record_tax_transaction, payment_id)
        except Exception:  # noqa: BLE001
            log.exception("housekeeping failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
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

    @app.middleware("http")
    async def security_middleware(request: Request, call_next):
        path = request.url.path
        if path.startswith(settings.base_path + "/api/") and not path.endswith("/stripe/webhook"):
            try:
                enforce("api", client_ip(request))
            except HTTPException as e:
                return JSONResponse({"detail": e.detail}, status_code=e.status_code)
        response: Response = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        if path.startswith(settings.base_path + "/api/"):
            response.headers.setdefault("Content-Security-Policy", API_CSP)
            response.headers.setdefault("Cache-Control", "no-store")
        if settings.production:
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000")
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        # Don't echo submitted values back.
        fields = [".".join(str(p) for p in e.get("loc", [])[1:]) for e in exc.errors()]
        return JSONResponse({"detail": "Invalid request.", "fields": fields}, status_code=422)

    @app.get("/api/health")
    def health() -> dict:
        db.one("SELECT 1")
        return {"ok": True}

    for r in (
        auth_routes.router,
        auth_routes.me_router,
        billing_routes.router,
        billing_routes.webhook_router,
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
