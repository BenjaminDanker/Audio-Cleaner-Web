#!/usr/bin/env python3
"""Production smoke test: does what users and the apps reach actually work? Run on primary-ram after a deploy
(deploy/install.sh runs it at the end) or any time:

    python3 deploy/smoke.py            one line per check, exit 1 when one fails
    python3 deploy/smoke.py --quiet    only the failures

It changes nothing: GET/HEAD requests, an unauthenticated WebSocket handshake, an unsigned Stripe webhook (refused), and
`ssh vps curl` to the tunnels' VPS ends. Standard library only.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import socket
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SITE = "https://audiocleaner.live"
OLD = "https://mc-multi-world.space/audio"
REPO = Path(__file__).resolve().parents[1]
CERT_DAYS = 14  # a certificate closer to expiry than this means Caddy's renewal is stuck

results: list[tuple[bool, str, str]] = []
checks: list[tuple[str, object]] = []


def check(name: str):
    """Register a check; run_checks() runs them all (importing this file touches nothing, so a test can call one)."""
    def wrap(fn):
        checks.append((name, fn))
        return fn
    return wrap


def run_checks() -> None:
    for name, fn in checks:
        try:
            results.append((True, name, fn() or ""))
        except Exception as e:  # noqa: BLE001 - every failure is a finding
            results.append((False, name, f"{type(e).__name__}: {e}"))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **kw):
        return None


def fetch(url: str, *, method: str = "GET", data: bytes | None = None, headers: dict | None = None,
          follow: bool = True, timeout: float = 15):
    """(status, headers, body) without raising on 4xx/5xx; redirects followed only when `follow`."""
    opener = urllib.request.build_opener(*(() if follow else (_NoRedirect,)))
    req = urllib.request.Request(url, data=data, method=method, headers={"User-Agent": "acw-smoke", **(headers or {})})
    try:
        with opener.open(req, timeout=timeout) as r:
            return r.status, r.headers, r.read() if method != "HEAD" else b""
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def expect(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# ---- the site ---------------------------------------------------------------------------------------------------------
@check("site: page, security headers")
def _site():
    st, h, body = fetch(SITE + "/")
    expect(st == 200, f"status {st}")
    csp = h.get("Content-Security-Policy", "")
    expect("https://huggingface.co" in csp and "frame-ancestors 'none'" in csp, f"CSP: {csp[:120]}")
    expect("max-age=" in h.get("Strict-Transport-Security", ""), "no HSTS")
    expect(h.get("Cache-Control", "").startswith("no-cache"), "index.html must not be cached")
    asset = re.search(rb'src="(/assets/[^"]+\.js)"', body)
    expect(asset is not None, "no script in index.html (built with the wrong base path?)")
    st, h2, _ = fetch(SITE + asset.group(1).decode(), method="HEAD")
    expect(st == 200 and "immutable" in h2.get("Cache-Control", ""), f"{asset.group(1).decode()}: {st}")
    return asset.group(1).decode()


@check("site: an app route serves the app")
def _spa():
    st, _, body = fetch(SITE + "/app")
    expect(st == 200 and b'<div id="root"' in body, f"status {st}")


@check("site: a missing script is a 404, not the app")
def _missing():
    # A tab on an old page asks for scripts that a newer publish removed: a real 404 tells it to reload, the app's
    # index.html (200) would break it quietly (deploy/vps/sites/00-audio-cleaner-snippets.caddy, the acw_spa rewrite).
    for path in ("/assets/acw-smoke-missing.js", "/dfn/acw-smoke-missing.js"):
        st, _, _ = fetch(SITE + path, method="HEAD")
        expect(st == 404, f"{path}: status {st} (expected 404)")


@check("www redirects to the site")
def _www():
    st, h, _ = fetch("https://www.audiocleaner.live/pricing?x=1", follow=False)
    expect(st == 308 and h.get("Location") == SITE + "/pricing?x=1", f"{st} {h.get('Location')}")


@check("free tools' code is public (served without a sign-in)")
def _dfn_public():
    # The free in-browser tools work without an account (Terms, Privacy, web/tests/signedout.mjs), so the glue code they load is
    # plain static files for everybody (docs/security.md, deploy/vps/sites/*.caddy: /dfn/* has no forward_auth). A sign-in here
    # would break them for every signed-out visitor. The check also
    # insists on real JavaScript, not the app's index.html (a missing file served as the page would look like a 200).
    st, h, body = fetch(SITE + "/dfn/df.js")
    expect(st == 200, f"/dfn/df.js: status {st} (the free tools need it without a session)")
    expect("javascript" in h.get("Content-Type", "") and not body.lstrip().startswith(b"<"), f"/dfn/df.js is not JavaScript ({h.get('Content-Type')})")


# ---- the API, by process ----------------------------------------------------------------------------------------------
@check("API (acw-web)")
def _api():
    st, _, body = fetch(SITE + "/api/health")
    expect(st == 200 and json.loads(body).get("ok") is True, f"{st} {body[:80]!r}")
    st, _, body = fetch(SITE + "/api/billing/pricing")
    expect(st == 200 and b"usd" in body.lower(), f"pricing: {st}")


def ws_handshake(url: str) -> int:
    """Status of an unauthenticated WebSocket upgrade (401 means the live route answered and asked for a sign-in)."""
    u = urllib.parse.urlsplit(url)
    conn = http.client.HTTPSConnection(u.hostname, 443, timeout=15, context=ssl.create_default_context())
    conn.request("GET", u.path, headers={
        "Upgrade": "websocket", "Connection": "Upgrade", "Sec-WebSocket-Version": "13",
        "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ==", "Origin": SITE, "User-Agent": "acw-smoke"})
    st = conn.getresponse().status
    conn.close()
    return st


@check("live WebSocket (acw-live)")
def _ws():
    st = ws_handshake(SITE + "/api/stream/ws")
    expect(st == 401, f"handshake answered {st} (expected 401: the route is there and wants a sign-in)")


@check("downloads (acw-bulk)")
def _bulk():
    st, h, body = fetch(SITE + "/api/downloads/obs-plugin/manifest.json")
    expect(st == 200 and json.loads(body).get("version"), f"{st}")
    return "plugin " + json.loads(body)["version"]


@check("Stripe webhook reachable (refuses an unsigned call)")
def _stripe():
    st, _, _ = fetch(SITE + "/api/stripe/webhook", method="POST", data=b"{}", headers={"Content-Type": "application/json"})
    expect(st == 400, f"status {st} (expected 400: signature missing)")


@check("Google sign-in returns to this address")
def _google():
    st, h, _ = fetch(SITE + "/api/auth/oauth/google/start", follow=False)
    loc = h.get("Location", "")
    want = urllib.parse.quote(SITE + "/api/auth/oauth/google/callback", safe="")
    expect(st in (302, 303, 307) and "accounts.google.com" in loc and f"redirect_uri={want}" in loc, f"{st} {loc[:120]}")


# ---- the old address ----------------------------------------------------------------------------------------------------
@check("old address: pages redirect, API still answers")
def _old():
    st, h, _ = fetch(OLD + "/app?x=1", follow=False)
    expect(st == 308 and h.get("Location") == SITE + "/app?x=1", f"page: {st} {h.get('Location')}")
    st, _, body = fetch(OLD + "/api/health")
    expect(st == 200 and json.loads(body).get("ok") is True, f"api: {st}")
    st = ws_handshake(OLD + "/api/stream/ws")
    expect(st == 401, f"old live route: {st}")


# ---- certificates -------------------------------------------------------------------------------------------------------
@check("certificates valid for 14+ days")
def _certs():
    out = []
    for host in ("audiocleaner.live", "www.audiocleaner.live", "mc-multi-world.space"):
        ctx = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=10) as s, ctx.wrap_socket(s, server_hostname=host) as t:
            days = (ssl.cert_time_to_seconds(t.getpeercert()["notAfter"]) - time.time()) / 86400
        expect(days >= CERT_DAYS, f"{host}: {days:.0f} days left")
        out.append(f"{host} {days:.0f}d")
    return ", ".join(out)


# ---- models on Hugging Face (what browsers download) ----------------------------------------------------------------
@check("free tools' models on Hugging Face (size, browser access)")
def _hf():
    src = (REPO / "web/src/media/models.js").read_text()
    ours = re.search(r"const OURS = \{ repo: '([^']+)', revision: '([0-9a-f]{40})' \}", src)
    speech_repo = re.search(r"repo: '(onnx-community/[^']+)'", src).group(1)
    speech_rev = re.search(r"revision: '([0-9a-f]{40})',\s*\n\s*dtype", src).group(1)
    files = [(f"{ours.group(1)}/resolve/{ours.group(2)}/{p}", int(n))
             for p, n in re.findall(r"path: '([^']+)', size: (\d+)", src)]
    files += [(f"{speech_repo}/resolve/{speech_rev}/{p}", int(n))
              for p, n in re.findall(r"'([^']+)': \{ size: (\d+),", src)]
    for path, size in files:
        st, h, _ = fetch("https://huggingface.co/" + path, method="HEAD", headers={"Origin": SITE})
        expect(st == 200, f"{path}: {st}")
        expect(h.get("Access-Control-Allow-Origin") in ("*", SITE), f"{path}: no CORS for the site")
        got = int(h.get("Content-Length") or h.get("X-Linked-Size") or -1)
        expect(got == size, f"{path}: {got} bytes, pinned {size}")
    return f"{len(files)} files"


# ---- the tunnels, from the VPS's side ---------------------------------------------------------------------------------
@check("tunnels: each process answers at the VPS")
def _tunnels():
    out = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "vps",
         "for p in 18790 18794 18795; do printf '%s ' $p; curl -s --max-time 5 127.0.0.1:$p/api/health; echo; done"],
        capture_output=True, text=True, timeout=40, check=False).stdout
    lines = dict(line.split(" ", 1) for line in out.strip().splitlines() if " " in line)
    expect(set(lines) == {"18790", "18794", "18795"}, f"ssh vps: {out.strip()[:120]!r}")
    expect('"ok":true' in lines["18790"].replace(" ", ""), f"18790 (acw-web): {lines['18790']}")
    expect('"process":"bulk"' in lines["18794"].replace(" ", ""), f"18794 (acw-bulk): {lines['18794']}")
    expect('"process":"live"' in lines["18795"].replace(" ", ""), f"18795 (acw-live): {lines['18795']}")


def use_color() -> bool:
    """Same rule as deploy/color.sh: ACW_COLOR=1|0 (install.sh sets it) wins; otherwise a terminal on stdout and no NO_COLOR."""
    forced = os.environ.get("ACW_COLOR")
    if forced in ("0", "1"):
        return forced == "1" and sys.stdout.isatty()
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR") and os.environ.get("TERM", "dumb") != "dumb"


def paint(text: str, code: str, color: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if color else text


def main() -> int:
    quiet = "--quiet" in sys.argv[1:]
    color = use_color()
    run_checks()
    summary = os.environ.get("ACW_SUMMARY")  # deploy/color.sh: the deploy lists every red line again at its end
    for ok, name, detail in results:
        if ok and quiet:
            continue
        line = f"{'ok  ' if ok else 'FAIL'}  {name}{('  (' + detail + ')') if detail else ''}"
        print(paint(line, "32" if ok else "1;31", color))
        if not ok and summary:
            try:
                with open(summary, "a") as f:
                    f.write("R\t" + line.split("\n")[0] + "\n")
            except OSError:
                pass
    failed = sum(not ok for ok, _, _ in results)
    print(paint(f"smoke: {len(results) - failed}/{len(results)} ok" + (f", {failed} FAILED" if failed else ""), "1;31" if failed else "32", color))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
