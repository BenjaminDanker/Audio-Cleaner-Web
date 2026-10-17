"""A stand-in for the GitHub releases API on a local port, for the mirror tests: releases and assets, the redirect to a
second host name for the file bytes (GitHub redirects asset downloads to its storage), Range support, a way to
cut a download short, and a log of every request with the Authorization header it carried."""
from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import subprocess
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = "owner/repo"


class FakeGitHub:
    def __init__(self) -> None:
        self.releases: dict[str, dict] = {}   # tag -> {"draft": bool, "prerelease": bool, "assets": {name: bytes}}
        self.log: list[tuple[str, str | None, str | None]] = []  # (host label, path, Authorization)
        self.ranges: list[str | None] = []
        self.cut_after: int | None = None     # the next blob download stops after this many bytes (once)
        self.status_override: dict[str, int] = {}
        self._ids: dict[tuple[str, str, str], int] = {}
        self._blobs: dict[int, bytes] = {}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):  # quiet
                pass

            def _send(self, code, body=b"", headers=None):
                self.send_response(code)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def do_GET(self):
                host = self.headers.get("Host", "").split(":")[0]
                outer.log.append((host, self.path, self.headers.get("Authorization")))
                forced = outer.status_override.get(self.path)
                if forced:
                    return self._send(forced, b'{"message":"nope"}')
                m = re.fullmatch(rf"/repos/{REPO}/releases(?:/tags/([^/]+))?(?:\?(.*))?", self.path)
                if m:
                    return self._releases(m.group(1), urllib.parse.parse_qs(m.group(2) or ""))
                m = re.fullmatch(rf"/repos/{REPO}/releases/assets/(\d+)", self.path)
                if m:
                    if "octet-stream" not in (self.headers.get("Accept") or ""):
                        return self._send(415)
                    return self._send(302, headers={"Location": f"http://localhost:{self.server.server_port}/blob/{m.group(1)}"})
                m = re.fullmatch(r"/blob/(\d+)", self.path)
                if m:
                    return self._blob(int(m.group(1)))
                self._send(404, b'{"message":"Not Found"}')

            def _releases(self, tag, query):
                def rel(t, r):
                    return {
                        "tag_name": t, "draft": r.get("draft", False), "prerelease": r.get("prerelease", False),
                        "assets": [{"id": outer.asset_id(t, n), "name": n, "size": len(b), "updated_at": r.get("updated", "2026-01-01T00:00:00Z"),
                                    "url": f"{outer.api}/repos/{REPO}/releases/assets/{outer.asset_id(t, n)}"}
                                   for n, b in r["assets"].items()],
                    }
                if tag is None:
                    # GitHub lists newest first; here the release added last is the newest. Paged like GitHub (default 30).
                    per_page = int(query.get("per_page", ["30"])[0])
                    page = int(query.get("page", ["1"])[0])
                    listing = [rel(t, r) for t, r in reversed(list(outer.releases.items()))]
                    chunk = listing[(page - 1) * per_page:page * per_page]
                    return self._send(200, json.dumps(chunk).encode(), {"Content-Type": "application/json"})
                if tag not in outer.releases:
                    return self._send(404, b'{"message":"Not Found"}')
                self._send(200, json.dumps(rel(tag, outer.releases[tag])).encode(), {"Content-Type": "application/json"})

            def _blob(self, aid):
                data = outer._blobs[aid]
                rng = self.headers.get("Range")
                outer.ranges.append(rng)
                start, code, headers = 0, 200, {"Content-Type": "application/octet-stream"}
                if rng:
                    start = int(re.fullmatch(r"bytes=(\d+)-", rng).group(1))
                    code = 206
                    headers["Content-Range"] = f"bytes {start}-{len(data) - 1}/{len(data)}"
                body = data[start:]
                if outer.cut_after is not None and len(body) > outer.cut_after:
                    n, outer.cut_after = outer.cut_after, None
                    self.send_response(code)
                    for k, v in headers.items():
                        self.send_header(k, v)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body[:n])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                self._send(code, body, headers)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.api = f"http://127.0.0.1:{self.server.server_port}"
        # poll_interval: shutdown() waits up to one poll; the default 0.5 s would be paid by every test
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()

    def asset_id(self, tag: str, name: str) -> int:
        data = self.releases[tag]["assets"][name]
        key = (tag, name, hashlib.sha256(data).hexdigest())  # a replaced asset gets a new id, like on GitHub
        if key not in self._ids:
            self._ids[key] = len(self._ids) + 100
        aid = self._ids[key]
        self._blobs[aid] = data
        return aid

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def add(self, tag: str, assets: dict[str, bytes], **kw) -> None:
        self.releases[tag] = {"assets": dict(assets), **kw}


class Signer:
    """An Ed25519 key made with openssl (the tests do not need the `cryptography` package)."""

    def __init__(self, folder: Path):
        self.key = folder / "key.pem"
        subprocess.run(["openssl", "genpkey", "-algorithm", "ed25519", "-out", str(self.key)], check=True, capture_output=True)
        der = subprocess.run(["openssl", "pkey", "-in", str(self.key), "-pubout", "-outform", "DER"], check=True, capture_output=True).stdout
        self.public = base64.b64encode(der[-32:]).decode()
        self.folder = folder

    def sign(self, data: bytes) -> bytes:
        f, o = self.folder / "msg", self.folder / "sig"
        f.write_bytes(data)
        subprocess.run(["openssl", "pkeyutl", "-sign", "-rawin", "-inkey", str(self.key), "-in", str(f), "-out", str(o)], check=True, capture_output=True)
        return o.read_bytes()


def have_openssl() -> bool:
    if shutil.which("openssl") is None:
        return False
    r = subprocess.run(["openssl", "version"], capture_output=True, text=True)
    return r.returncode == 0 and " 3." in r.stdout


def sha(data: bytes, algo: str = "sha256") -> str:
    return hashlib.new(algo, data).hexdigest()
