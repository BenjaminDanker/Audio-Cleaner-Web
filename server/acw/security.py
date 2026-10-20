"""Security helpers: secrets, hashing, rate limiting, request metadata.

Rules followed everywhere:
  * Bearer secrets (sessions, API keys, login links) are 256-bit random values;
    only their SHA-256 is stored, so a database leak does not leak credentials.
  * Comparisons of secrets use hmac.compare_digest.
  * Client IPs come from X-Forwarded-For only because the app binds to
    127.0.0.1 behind exactly one trusted proxy hop (Caddy via the SSH tunnel).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import hmac
import math
import secrets
import threading
import time
import unicodedata
from collections import OrderedDict

import anyio
from fastapi import HTTPException, Request

API_KEY_PREFIX = "acw_"


def random_token(nbytes: int = 32) -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(nbytes)).decode().rstrip("=")


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def canonical_email(email: str) -> str:
    """The mailbox an address really reaches: lower case, NFKC, no +tag (any provider), and no dots at Gmail
    (googlemail.com is Gmail). Used for rate limits and for once-per-person grants."""
    email = unicodedata.normalize("NFKC", email.strip()).lower()
    local, _, domain = email.partition("@")
    local = local.split("+", 1)[0]
    if domain in ("gmail.com", "googlemail.com"):
        local, domain = local.replace(".", ""), "gmail.com"
    return f"{local}@{domain}"


def keyed_hash(purpose: str, value: str) -> str:
    """HMAC-SHA256 of `value` under the server secret, separated by `purpose`, for values that are stored only to
    be matched later (so a copy of the database can't be used to test guesses)."""
    from .config import get_settings

    return hmac.new(get_settings().session_secret, f"{purpose}\0{value}".encode(), hashlib.sha256).hexdigest()


def consteq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


def new_api_key() -> tuple[str, str, str]:
    """Returns (full_key, display_prefix, hash)."""
    body = random_token(32)
    key = f"{API_KEY_PREFIX}{body}"
    return key, key[:12], sha256_hex(key)


def client_ip(request: Request) -> str:
    # Caddy appends the real client address as the last X-Forwarded-For entry;
    # anything before it is client-supplied and untrusted.
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[-1].strip()[:64]
    return request.client.host if request.client else "unknown"


def user_agent(request: Request) -> str:
    return (request.headers.get("user-agent") or "")[:300]


class RateLimiter:
    """In-memory token buckets keyed by string. Single-process app, so this is
    authoritative; it resets on restart, which is acceptable for abuse control.
    Each limit name ("api:", "email_addr:", ...) has its own table, so flooding one kind of key with
    made-up values can't push other limits' buckets out."""

    def __init__(self, max_keys: int = 20_000):
        self._tables: dict[str, OrderedDict[str, tuple[float, float]]] = {}
        self._lock = threading.Lock()
        self._max_keys = max_keys

    def _table(self, key: str) -> "OrderedDict[str, tuple[float, float]]":
        name = key.split(":", 1)[0]
        t = self._tables.get(name)
        if t is None:
            t = self._tables[name] = OrderedDict()
        return t

    def hit(self, key: str, capacity: float, refill_per_sec: float, cost: float = 1.0) -> bool:
        now = time.monotonic()
        with self._lock:
            buckets = self._table(key)
            tokens, last = buckets.get(key, (capacity, now))
            tokens = min(capacity, tokens + (now - last) * refill_per_sec)
            allowed = tokens >= cost
            if allowed:
                tokens -= cost
            buckets[key] = (tokens, now)
            buckets.move_to_end(key)
            while len(buckets) > self._max_keys:
                buckets.popitem(last=False)
            return allowed

    def retry_after(self, key: str, refill_per_sec: float, cost: float = 1.0) -> int:
        """Whole seconds until `key` has `cost` tokens again (at least 1)."""
        with self._lock:
            tokens, _ = self._table(key).get(key, (cost, 0.0))
        return max(1, math.ceil((cost - tokens) / max(refill_per_sec, 1e-9)))

    def reset(self) -> None:
        with self._lock:
            self._tables.clear()


limiter = RateLimiter()

# name -> (capacity, refill per second)
LIMITS = {
    "auth": (10, 10 / 600),            # 10 bursts, refill 10 per 10 min
    "email_link": (6, 6 / 3600),       # 6 per hour per IP (shared addresses: CGNAT, campus, office)
    "email_addr": (3, 3 / 3600),       # 3 per hour per mailbox
    "email_domain": (20, 20 / 3600),   # sign-in mails per hour to one (small) mail domain
    "email_link_wide": (12, 12 / 3600),  # ... per IPv6 /48 (a prefix that size holds 65,536 /64s)
    "email_all": (60, 60 / 3600),      # sign-in mails per hour in total, and per day: a flood of new addresses stops
    "email_all_day": (400, 400 / 86400),  # here, below the mail account's own limits (it also carries the owner's alerts)
    "api": (120, 2.0),                 # general API calls
    "job_create": (20, 20 / 3600),
    "upload": (30, 30 / 3600),
    "edit_plan": (120, 2.0),           # timeline reads and saves, quotes of the per-part editor (a plan costs a little CPU)
    "kept_cut": (120, 2.0),            # stretches of a job's kept copies per account (the strength slider asks for two every ~10 s of playing, more when seeking)
    "checkout": (10, 10 / 3600),
    "asset_fetch": (300, 300 / 86400),  # free-tool files per account per day (api/auth_routes.asset_check)
    "stream_start": (60, 60 / 3600),    # live starts per account per hour (failed starts count too)
    "stream_start_ip": (90, 90 / 3600),  # ... and per network
    "key_create": (10, 10 / 3600),
    "contact_ip": (3, 3 / 3600),       # contact form messages per network per hour (api/contact_routes.py)
    "contact_ip_day": (10, 10 / 86400),
    "contact_email": (3, 3 / 3600),    # ... and per mailbox
    "contact_email_day": (5, 5 / 86400),
    "webhook": (120, 5.0),             # Stripe delivers a handful per payment; this only stops floods
    "ws_handshake": (30, 0.5),         # live WebSocket attempts per IP, before any database work
    "download": (40, 40 / 1800),       # release downloads (installers, plugin, models) started per network; resumes cost a tenth
}


def _wait_text(seconds: int) -> str:
    if seconds < 90:
        return f"{seconds} seconds"
    return f"{math.ceil(seconds / 60)} minutes"


def enforce(name: str, key: str, cost: float = 1.0) -> None:
    capacity, refill = LIMITS[name]
    if not limiter.hit(f"{name}:{key}", capacity, refill, cost):
        wait = limiter.retry_after(f"{name}:{key}", refill, cost)
        raise HTTPException(
            status_code=429, detail=f"Too many requests, try again in {_wait_text(wait)}.", headers={"Retry-After": str(wait)}
        )


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "X-Frame-Options": "DENY",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), geolocation=(), payment=(), usb=()",
}

API_CSP = "default-src 'none'; frame-ancestors 'none'; base-uri 'none'"


class Throttle:
    """Byte-rate limit for streams (uploads): `take` sleeps when the stream runs ahead of `rate` bytes/s, which
    pushes back on the sender through TCP. A burst of one second is free."""

    def __init__(self, rate: float):
        self.rate = max(1.0, rate)
        self.tokens = self.rate
        self.at = time.monotonic()

    async def take(self, n: int) -> None:
        now = time.monotonic()
        self.tokens = min(self.rate, self.tokens + (now - self.at) * self.rate) - n
        self.at = now
        if self.tokens < 0:
            await asyncio.sleep(-self.tokens / self.rate)


TRANSFER_MBPS_DEFAULT = 5.0


def transfer_budget() -> Throttle | None:
    """The one byte-rate budget that uploads and result downloads of a process share (ACW_TRANSFER_MBPS_TOTAL, megabytes
    per second; 0 = none). Every transferred byte crosses the VPS twice (a download goes home -> VPS -> user, an upload
    user -> VPS -> home) on a link of about 9 MB/s in and 11 MB/s out that live audio uses too, so separate caps of 8 and 6
    added up to more than the link. The older ACW_DOWNLOAD_MBPS_TOTAL / ACW_UPLOAD_MBPS_TOTAL are still read when the new
    variable is not set (the smaller one counts)."""
    raw = os.environ.get("ACW_TRANSFER_MBPS_TOTAL")
    if raw is None:
        legacy = [float(v) for v in (os.environ.get("ACW_DOWNLOAD_MBPS_TOTAL"), os.environ.get("ACW_UPLOAD_MBPS_TOTAL")) if v is not None]
        positive = [v for v in legacy if v > 0]
        mbps = min(positive) if positive else (0.0 if legacy else TRANSFER_MBPS_DEFAULT)
    else:
        mbps = float(raw)
    return Throttle(mbps * 1e6) if mbps > 0 else None


class UploadBandwidth:
    """Upload bandwidth caps: the process-wide transfer budget it shares with downloads and one cap per account, so bulk
    uploads can't starve live audio that shares the same tunnel and event loop (ACW_TRANSFER_MBPS_TOTAL /
    ACW_UPLOAD_MBPS_USER, megabytes per second)."""

    def __init__(self, budget: Throttle | None | str = "own") -> None:
        self.total = transfer_budget() if budget == "own" else budget
        self.per_user = float(os.environ.get("ACW_UPLOAD_MBPS_USER", "5")) * 1e6
        self._users: dict[str, Throttle] = {}

    def for_user(self, user_id: str) -> Throttle:
        if len(self._users) > 2000:
            self._users.clear()
        t = self._users.get(user_id)
        if t is None:
            t = self._users[user_id] = Throttle(self.per_user)
        return t

    async def take(self, user: Throttle, n: int) -> None:
        await user.take(n)
        if self.total is not None:
            await self.total.take(n)


class DownloadBandwidth:
    """Result and release downloads of a process draw on the same transfer budget as its uploads (see transfer_budget).
    The home uplink carries live audio replies as well, so transfers must not be able to fill it."""

    def __init__(self, budget: Throttle | None | str = "own") -> None:
        self.total = transfer_budget() if budget == "own" else budget

    async def take(self, n: int) -> None:
        if self.total is not None:
            await self.total.take(n)


_budget = transfer_budget()
uploads = UploadBandwidth(_budget)
downloads = DownloadBandwidth(_budget)  # the same object: uploads and downloads together stay under one rate


def client_net(ip: str, prefix: int = 64) -> str:
    """Rate-limit key for an address: IPv6 addresses count per /64 (one subscriber holds billions of them), or per
    `prefix` (e.g. 48: what one customer or small provider usually holds)."""
    if ":" in ip:
        import ipaddress

        try:
            return str(ipaddress.ip_network(f"{ip}/{prefix}", strict=False))
        except ValueError:
            return ip
    return ip


# Calls that leave the machine (SMTP, Stripe) get their own small budget out of the shared thread pool: when a
# provider hangs, only this many threads wait on it and everything else keeps answering.
OUTBOUND_MAX = 8
_outbound_inflight = 0


async def run_outbound(fn, *args):
    global _outbound_inflight
    if _outbound_inflight >= OUTBOUND_MAX:
        raise HTTPException(status_code=503, detail="We're busy right now. Please try again in a minute.")
    _outbound_inflight += 1  # one event loop thread: no lock needed
    try:
        return await anyio.to_thread.run_sync(fn, *args)
    finally:
        _outbound_inflight -= 1
