"""The HTTP layers every Audio Cleaner web process shares (the main app and the bulk-transfer app): the
per-address API rate limit, the security headers, the body-size limits and the access log. One definition,
so the two processes can not drift apart."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from .bodylimit import AccessLogMiddleware, BodyLimitMiddleware
from .config import Settings
from .security import API_CSP, SECURITY_HEADERS, client_ip, client_net, enforce


def install(app: FastAPI, settings: Settings) -> None:
    @app.middleware("http")
    async def security_middleware(request: Request, call_next):
        path = request.url.path
        if path.startswith(settings.base_path + "/api/") and not path.endswith("/stripe/webhook"):
            try:
                enforce("api", client_net(client_ip(request)))
            except HTTPException as e:
                return JSONResponse({"detail": e.detail}, status_code=e.status_code, headers=e.headers)
        response: Response = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        if path.startswith(settings.base_path + "/api/"):
            response.headers.setdefault("Content-Security-Policy", API_CSP)
            response.headers.setdefault("Cache-Control", "no-store")
        if settings.production:
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000")
        return response

    # Outermost first: log every request (without query strings), then refuse oversized bodies before anything buffers them.
    app.add_middleware(BodyLimitMiddleware, base_path=settings.base_path)
    app.add_middleware(AccessLogMiddleware)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        # Don't echo submitted values back.
        fields = [".".join(str(p) for p in e.get("loc", [])[1:]) for e in exc.errors()]
        return JSONResponse({"detail": "Invalid request.", "fields": fields}, status_code=422)
