"""Request body limits, enforced while the body is read (a pure ASGI middleware).

Only the upload route takes big bodies. Everything else is small JSON, so a client that declares (or just
streams) more than the limit is answered with 413 before the framework buffers anything, and a client that
stops sending in the middle of a body is dropped after BODY_IDLE_SECONDS, so a stalled request can't hold a
connection slot for ever. WebSockets are not touched (uvicorn's --ws-max-size bounds their messages).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time

DEFAULT_LIMIT = 64 * 1024
WEBHOOK_LIMIT = 1024 * 1024
TIMELINE_LIMIT = 256 * 1024  # a timeline is up to 200 KB of JSON (editing.TIMELINE_MAX_BYTES)
BODY_IDLE_SECONDS = 20

_UPLOAD = re.compile(r"^/api/jobs/[0-9a-f]{32}/input$")
# the routes that take a timeline: creating (or estimating) a job, saving one, quoting, processing, exporting
_TIMELINE = re.compile(r"^/api/jobs(?:/estimate|/[0-9a-f]{32}/(?:timeline|quote|process|export))?$")
_WEBHOOK = "/api/stripe/webhook"


class BodyLimitMiddleware:
    def __init__(self, app, base_path: str = "", default: int = DEFAULT_LIMIT):
        self.app = app
        self.base_path = base_path.rstrip("/")
        self.default = default

    def limit_for(self, path: str) -> int | None:
        """Largest body this path may send; None: not limited here (the route enforces its own size)."""
        if self.base_path and path.startswith(self.base_path + "/"):
            path = path[len(self.base_path):]
        if _UPLOAD.match(path):
            return None
        if path == _WEBHOOK:
            return WEBHOOK_LIMIT
        if _TIMELINE.match(path):
            return TIMELINE_LIMIT
        return self.default

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self.limit_for(scope["path"])
        if limit is None:
            await self.app(scope, receive, send)
            return

        started = False  # a response has begun (ours or the app's)
        refused = False

        async def refuse(status: int, message: str) -> None:
            nonlocal refused, started
            if refused:
                return
            refused = True
            if started:
                return
            started = True
            body = json.dumps({"detail": message}).encode()
            await send({"type": "http.response.start", "status": status, "headers": [
                (b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
                (b"connection", b"close"), (b"cache-control", b"no-store"),
            ]})
            await send({"type": "http.response.body", "body": body})

        def declared() -> int | None:
            for k, v in scope["headers"]:
                if k == b"content-length":
                    try:
                        return int(v)
                    except ValueError:
                        return -1
            return None

        n = declared()
        if n is not None and (n < 0 or n > limit):
            await refuse(413, "Request too large.")
            return

        total = 0
        body_done = False

        async def guarded_receive():
            nonlocal total, body_done
            if refused:
                return {"type": "http.disconnect"}
            if body_done:
                return await receive()
            try:
                message = await asyncio.wait_for(receive(), BODY_IDLE_SECONDS)
            except asyncio.TimeoutError:
                await refuse(408, "The request took too long to arrive.")
                return {"type": "http.disconnect"}
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > limit:
                    await refuse(413, "Request too large.")
                    return {"type": "http.disconnect"}
                if not message.get("more_body", False):
                    body_done = True
            return message

        async def guarded_send(message):
            nonlocal started
            if refused:
                return  # our answer is already on its way; drop what the app says after the disconnect
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        await self.app(scope, guarded_receive, guarded_send)


# ---------------------------------------------------------------- access log

access_log = logging.getLogger("acw.access")


def _setup_access_log() -> None:
    if not access_log.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(message)s"))
        access_log.addHandler(h)
        access_log.setLevel(logging.INFO)
        access_log.propagate = False


def _quiet(method: str, path: str, status: int) -> bool:
    """Requests that succeed and say nothing new: health checks, and the job list a page asks for every 1.5 s while a
    job runs (at 1000 jobs a day that one line would be most of the log). Errors and everything else are logged."""
    if status == 200 and path.endswith("/api/health"):
        return True
    return method == "GET" and status < 400 and path.endswith("/api/jobs")


class AccessLogMiddleware:
    """One line per request: client, method, path, status, milliseconds. The query string is left out on
    purpose: sign-in callbacks carry one-time codes there and payment return links carry ids. Run uvicorn
    with --no-access-log (its own log line includes the query). Successful polling is left out (see `_quiet`)."""

    def __init__(self, app):
        self.app = app
        _setup_access_log()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        t0 = time.monotonic()
        status = 0

        async def logged_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, logged_send)
        finally:
            path = scope["path"]
            if not _quiet(scope["method"], path, status):
                ip = ""
                for k, v in scope["headers"]:
                    if k == b"x-forwarded-for":
                        ip = v.decode("latin-1").split(",")[-1].strip()[:64]
                        break
                if not ip and scope.get("client"):
                    ip = scope["client"][0]
                access_log.info("%s %s %s %s %dms", ip or "-", scope["method"], path[:200], status, (time.monotonic() - t0) * 1000)
