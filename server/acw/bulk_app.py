"""The bulk-transfer web process: `uvicorn acw.bulk_app:app --host 127.0.0.1 --port 8794` (acw-bulk.service).

Uploads of up to 4 GB and result downloads used to share one process and one SSH tunnel with the live audio
WebSocket, so a big transfer could add latency to live sessions (head-of-line blocking in the tunnel's TCP
stream, one event loop, one home uplink). This process serves only
  * PUT  /api/jobs/{id}/input            the upload
  * GET  /api/jobs/{id}/outputs/{name}   a result download
  * GET  /api/downloads/...              the release files (desktop installers, OBS plugin, model archive), mirrored
                                         from the build service by acw/mirror.py and served from our own domain
  * GET  /api/health                     for the monitor and the VPS (answers {"ok": true})
and nothing else (every other path is 404). It reuses the route functions of the main app, so authentication,
body limits, upload bandwidth caps and security headers are the same code. It runs none of the background loops
(housekeeping, cloud transcription, live) and has no live worker. Its own tunnel (acw-tunnel-bulk) carries it
to the VPS, so a transfer queues behind nothing the live session needs.

The main app keeps serving the same routes, so nothing breaks before the VPS sends them here (and if this
process or its tunnel is down, the VPS falls back to the main one).
"""
from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from . import db, httpstack
from .api import downloads_routes, jobs_routes
from .config import get_settings
from .security import downloads

log = logging.getLogger("acw.bulk")

_DOWNLOAD = re.compile(r"^/api/(?:jobs/[0-9a-f]{32}/outputs/|downloads/)")


class DownloadThrottleMiddleware:
    """Result downloads and release downloads share one byte-rate cap (security.downloads). The cap is applied to the response body
    as it is sent, so a slow link or many downloads at once never push more than the cap onto the uplink.
    Zero-copy file sending is switched off for them, otherwise the bytes would not pass through here."""

    def __init__(self, app, base_path: str = ""):
        self.app = app
        self.base_path = base_path.rstrip("/")

    def applies(self, scope) -> bool:
        if scope["type"] != "http" or scope["method"] not in ("GET", "HEAD"):
            return False
        path = scope["path"]
        if self.base_path and path.startswith(self.base_path + "/"):
            path = path[len(self.base_path):]
        return bool(_DOWNLOAD.match(path))

    async def __call__(self, scope, receive, send):
        if not self.applies(scope):
            await self.app(scope, receive, send)
            return
        scope = {**scope, "extensions": {k: v for k, v in scope.get("extensions", {}).items() if k != "http.response.pathsend"}}

        async def throttled_send(message):
            if message["type"] == "http.response.body":
                await downloads.take(len(message.get("body", b"")))
            await send(message)

        await self.app(scope, receive, throttled_send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()  # the same additive schema step every process runs; nothing else is started here
    yield


def create_bulk_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Audio Cleaner bulk transfers",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        root_path=settings.base_path,
    )
    httpstack.install(app, settings)
    app.add_middleware(DownloadThrottleMiddleware, base_path=settings.base_path)

    @app.get("/api/health")
    def health() -> dict:
        db.one("SELECT 1")
        return {"ok": True, "process": "bulk"}

    router = APIRouter(prefix="/api/jobs", tags=["jobs"])
    router.add_api_route("/{job_id}/input", jobs_routes.upload, methods=["PUT"])
    router.add_api_route("/{job_id}/outputs/{name}", jobs_routes.download, methods=["GET"])
    app.include_router(router)
    app.include_router(downloads_routes.router)
    return app


app = create_bulk_app()
