"""Security helpers: secrets, hashing, rate limiting, request metadata.

Rules followed everywhere:
  * Bearer secrets (sessions, API keys, login links) are 256-bit random values;
    only their SHA-256 is stored, so a database leak does not leak credentials.
  * Comparisons of secrets use hmac.compare_digest.
  * Client IPs come from X-Forwarded-For only because the app binds to
    127.0.0.1 behind exactly one trusted proxy hop (Caddy via the SSH tunnel).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import threading
import time
from collections import OrderedDict

from fastapi import HTTPException, Request

API_KEY_PREFIX = "acw_"


def random_token(nbytes: int = 32) -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(nbytes)).decode().rstrip("=")


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def consteq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


def sign(key: bytes, message: str) -> str:
    mac = hmac.new(key, message.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).decode().rstrip("=")


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
    authoritative; it resets on restart, which is acceptable for abuse control."""

    def __init__(self, max_keys: int = 50_000):
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._max_keys = max_keys

    def hit(self, key: str, capacity: float, refill_per_sec: float, cost: float = 1.0) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (capacity, now))
            tokens = min(capacity, tokens + (now - last) * refill_per_sec)
            allowed = tokens >= cost
            if allowed:
                tokens -= cost
            self._buckets[key] = (tokens, now)
            self._buckets.move_to_end(key)
            while len(self._buckets) > self._max_keys:
                self._buckets.popitem(last=False)
            return allowed

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()


limiter = RateLimiter()

# name -> (capacity, refill per second)
LIMITS = {
    "auth": (10, 10 / 600),            # 10 bursts, refill 10 per 10 min
    "email_link": (3, 3 / 3600),       # 3 per hour per IP and per address
    "api": (120, 2.0),                 # general API calls
    "job_create": (20, 20 / 3600),
    "upload": (30, 30 / 3600),
    "checkout": (10, 10 / 3600),
    "asset_fetch": (300, 300 / 86400),  # free-tool files per account per day (api/auth_routes.asset_check)
    "stream_start": (20, 20 / 3600),
    "key_create": (10, 10 / 3600),
}


def enforce(name: str, key: str, cost: float = 1.0) -> None:
    capacity, refill = LIMITS[name]
    if not limiter.hit(f"{name}:{key}", capacity, refill, cost):
        raise HTTPException(status_code=429, detail="Too many requests, slow down.")


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "X-Frame-Options": "DENY",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), geolocation=(), payment=(), usb=()",
}

API_CSP = "default-src 'none'; frame-ancestors 'none'; base-uri 'none'"
