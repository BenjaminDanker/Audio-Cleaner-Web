"""The live web process: `uvicorn acw.live_app:app --host 127.0.0.1 --port 8795` (acw-live.service).

Live audio used to share one event loop (one core) with the whole API: every sign-in, job-list poll and page visit
ran on the loop that relays live sessions' audio frames to the live engine, so heavy API traffic made running
sessions late (admission only turns new sessions away). This process serves only
  * GET /api/stream/ws    the live WebSocket (docs/streaming-protocol.md)
  * GET /api/health       for the monitor and the VPS (answers {"ok": true})
and nothing else (every other path is 404). It reuses the main app's live code (acw.stream.routes): authentication,
billing, admission by this loop's own lag, the live engine connection. Besides its sessions it runs only the sweep
that closes and bills its own sessions whose handler is gone. Its own tunnel (acw-tunnel-live) carries it to the VPS.

The main app keeps serving the WebSocket too, so nothing breaks before the VPS sends it here, and if this process or
its tunnel is down the VPS falls back to the main one. Each session records which process runs it
(stream_sessions.owner), so neither process ever closes the other's sessions.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import db, httpstack
from .config import get_settings
from .stream import routes as stream_routes

log = logging.getLogger("acw.live")


async def _orphan_loop() -> None:
    while True:
        await asyncio.sleep(30)
        try:
            await asyncio.to_thread(stream_routes.close_orphans)
        except Exception:  # noqa: BLE001
            log.exception("closing orphaned live sessions failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()  # the same additive schema step every process runs
    stream_routes.set_role("live")
    await stream_routes.startup()
    task = asyncio.create_task(_orphan_loop())
    try:
        yield
    finally:
        task.cancel()
        await stream_routes.shutdown()


def create_live_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Audio Cleaner live",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        root_path=settings.base_path,
    )
    httpstack.install(app, settings)

    @app.get("/api/health")
    def health() -> dict:
        db.one("SELECT 1")
        return {"ok": True, "process": "live"}

    app.include_router(stream_routes.router)
    return app


app = create_live_app()
