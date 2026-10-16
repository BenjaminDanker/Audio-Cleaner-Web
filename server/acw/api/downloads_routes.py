"""Release downloads under our own domain: `GET|HEAD /api/downloads/...`.

  desktop/update.json, update.json.sig     the signed update information of the desktop app (unchanged bytes)
  desktop/<file>                           an installer the current (or the previous, for a week) manifest lists
  obs-plugin/manifest.json, manifest.json.sig, obs-plugin/<file>
  obs-plugin/latest/<platform>             windows | mac | ubuntu-debian | linux-other: the current package, for the
                                           website's buttons (served directly, never redirected)
  desktop/latest/<platform>                windows | mac-apple-silicon | mac-intel: the current installer, same way
  models/<file>                            a model archive deploy/models.lock lists
  browser/<model>/<file>                   a file of a model that runs in the browser (acw/browser_models.py):
                                           signed-in browsers only, counted like the other free-tool files

Public: no sign-in, a rate limit per network (httpstack: all /api/ calls; here: starting file downloads) and the
download byte-rate cap of the bulk process (acw/bulk_app.py applies it to these paths). The files were verified by
acw/mirror.py before they became visible here; this module serves only names in the state file, opens them without
following links, and re-checks the size. Range requests (one range) and HEAD work, so a download resumes.
The main process serves the same routes: the VPS falls back to it when the bulk process is down.
"""
from __future__ import annotations

import os
import re

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.responses import Response

from .. import auth, browser_models, downloads
from ..security import client_ip, client_net, enforce

router = APIRouter(prefix="/api/downloads", tags=["downloads"])

CHUNK = 256 * 1024
TYPES = {
    ".json": "application/json",
    ".sig": "application/octet-stream",
    ".zip": "application/zip",
    ".deb": "application/vnd.debian.binary-package",
    ".exe": "application/vnd.microsoft.portable-executable",
    ".pkg": "application/octet-stream",
    ".xz": "application/x-xz",
}
IMMUTABLE = "public, max-age=31536000, immutable"
REVALIDATE = "no-cache"


class FileBody(Response):
    """A file (or a byte range of it) read from an open descriptor in chunks, so the process-wide byte-rate cap
    (middleware around `send`) sees every chunk. The descriptor is closed when the response ends or the client goes."""

    def __init__(self, fd: int, start: int, length: int, status: int, headers: dict[str, str], head: bool):
        super().__init__(content=None, status_code=status, headers=headers)
        self.fd, self.start, self.length, self.head = fd, start, length, head

    async def __call__(self, scope, receive, send) -> None:
        try:
            await send({"type": "http.response.start", "status": self.status_code, "headers": self.raw_headers})
            if self.head or self.length <= 0:
                await send({"type": "http.response.body", "body": b"", "more_body": False})
                return
            pos, left = self.start, self.length
            while left > 0:
                chunk = await anyio.to_thread.run_sync(os.pread, self.fd, min(CHUNK, left), pos)
                if not chunk:
                    return  # the file got shorter than it was verified to be: end the connection short
                pos += len(chunk)
                left -= len(chunk)
                await send({"type": "http.response.body", "body": chunk, "more_body": left > 0})
        finally:
            os.close(self.fd)


_RANGE = re.compile(r"bytes=(\d*)-(\d*)")


def parse_range(header: str | None, size: int) -> tuple[int, int] | str | None:
    """(first, last) of a single satisfiable range; None when the header is absent, malformed or asks for several
    ranges (then the whole file is sent, which RFC 9110 allows); "unsatisfiable" for a range outside the file."""
    if not header:
        return None
    m = _RANGE.fullmatch(header.strip())
    if m is None:
        return None
    a, b = m.groups()
    if a == "" and b == "":
        return None
    if a == "":  # the last b bytes
        n = int(b)
        if n == 0:
            return "unsatisfiable"
        return max(0, size - n), size - 1
    first = int(a)
    last = size - 1 if b == "" else min(int(b), size - 1)
    if first >= size:
        return "unsatisfiable"
    if b != "" and int(b) < first:
        return None
    return first, last


def _etag(entry: dict) -> str:
    return f'"{entry["sha256"][:32]}"'


def serve(channel: str, entry: dict, request: Request, *, public_name: str | None = None, opener=None) -> Response:
    """The response for a verified file."""
    head = request.method == "HEAD"
    size = entry["size"]
    name = public_name or entry["name"]
    is_manifest = entry["manifest"]
    rng = None if is_manifest else request.headers.get("range")
    etag = _etag(entry)
    headers = {
        "Content-Type": TYPES.get(os.path.splitext(name)[1].lower(), "application/octet-stream"),
        "Accept-Ranges": "bytes" if not is_manifest else "none",
        "ETag": etag,
        "Cache-Control": REVALIDATE if is_manifest else IMMUTABLE,
        "X-Content-Type-Options": "nosniff",
    }
    if not name.endswith((".json", ".sig")):
        headers["Content-Disposition"] = f'attachment; filename="{name}"'
    inm = request.headers.get("if-none-match")
    if inm and etag in [t.strip().removeprefix("W/") for t in inm.split(",")] and not rng:
        return Response(status_code=304, headers={k: v for k, v in headers.items() if k in ("ETag", "Cache-Control")})
    if_range = request.headers.get("if-range")
    if if_range and if_range.strip() != etag:
        rng = None  # the file changed since the client started: send all of it
    parsed = parse_range(rng, size)
    if parsed == "unsatisfiable":
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}", "Cache-Control": "no-store"})
    try:
        fd = opener(entry) if opener else downloads.open_nofollow(channel, entry)
    except OSError:
        raise HTTPException(status_code=404, detail="Not found.") from None
    if parsed is None:
        status, start, length = 200, 0, size
    else:
        first, last = parsed
        status, start, length = 206, first, last - first + 1
        headers["Content-Range"] = f"bytes {first}-{last}/{size}"
    headers["Content-Length"] = str(length)
    return FileBody(fd, start, length, status, headers, head)


def _limit(request: Request, entry: dict) -> None:
    """Starting a file download costs one token of the per-network budget; asking for the rest of a download in
    progress or for headers costs a tenth (flaky connections resume often), manifests are covered by the general
    API limit only."""
    if entry["manifest"]:
        return
    rng = request.headers.get("range") or ""
    resumed = bool(rng) and not rng.strip().startswith("bytes=0-")
    cost = 0.1 if request.method == "HEAD" or resumed else 1.0
    enforce("download", client_net(client_ip(request)), cost)


def _get(channel: str, name: str, request: Request, *, public_name: str | None = None) -> Response:
    entry = downloads.lookup(channel, name)
    if entry is None:
        raise HTTPException(status_code=404, detail="Not found.")
    _limit(request, entry)
    return serve(channel, entry, request, public_name=public_name)


def _manifest_name(channel: str, public: str) -> str | None:
    for stored, shown in downloads.MANIFESTS.get(channel, ()):
        if shown == public:
            return stored
    return None


def both(path: str):
    """GET and HEAD on one handler, each with its own operation id."""
    def deco(fn):
        return router.head(path)(router.get(path)(fn))
    return deco


def _latest(channel: str, keys: dict[str, str], platform: str, request: Request) -> Response:
    key = keys.get(platform)
    name = downloads.platform_file(channel, key) if key else None
    entry = downloads.lookup(channel, name) if name else None
    if entry is None or entry["manifest"]:
        raise HTTPException(status_code=404, detail="Not found.")
    _limit(request, entry)
    resp = serve(channel, entry, request)
    resp.headers["Cache-Control"] = REVALIDATE  # the same address gets a new file when a new version is published
    return resp


@both("/desktop/latest/{platform}")
def desktop_latest(platform: str, request: Request) -> Response:
    return _latest("desktop", downloads.DESKTOP_PLATFORMS, platform, request)


@both("/desktop/{name}")
def desktop_file(name: str, request: Request) -> Response:
    return _get("desktop", _manifest_name("desktop", name) or name, request)


@both("/obs-plugin/latest/{platform}")
def obs_latest(platform: str, request: Request) -> Response:
    return _latest("obs-plugin", downloads.OBS_PLATFORMS, platform, request)


@both("/obs-plugin/{name}")
def obs_plugin_file(name: str, request: Request) -> Response:
    stored = _manifest_name("obs-plugin", name)
    if stored is None and name in {m[0] for m in downloads.MANIFESTS["obs-plugin"]}:
        raise HTTPException(status_code=404, detail="Not found.")  # the stored names are not public
    return _get("obs-plugin", stored or name, request, public_name=name if stored else None)


@both("/models/{name}")
def model_file(name: str, request: Request) -> Response:
    return _get("models", name, request)


@both("/browser/{model}/{name:path}")
def browser_model_file(model: str, name: str, request: Request, principal: auth.Principal = Depends(auth.require_session)) -> Response:
    entry = browser_models.lookup(model, name)
    if entry is None:
        raise HTTPException(status_code=404, detail="Not found.")
    rng = request.headers.get("range") or ""
    if request.method == "GET" and (not rng or rng.strip().startswith("bytes=0-")):
        enforce("asset_fetch", principal.user_id)  # the same daily allowance as the noise remover's files
    resp = serve("browser", entry, request, opener=browser_models.open_file)
    resp.headers["Cache-Control"] = "private, max-age=31536000, immutable"  # the model id names the version
    if "content-disposition" in resp.headers:
        del resp.headers["content-disposition"]
    return resp
