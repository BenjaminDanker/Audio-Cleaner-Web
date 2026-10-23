"""Finish notifications for file jobs: `callbackUrl` on POST /api/jobs.

When a job with a callback reaches done / failed / canceled (or expired before the first delivery), the web process
POSTs a small JSON to that address, signed with the account's webhook secret:

    X-Audio-Cleaner-Signature: t=<unix seconds>,v1=<hex HMAC-SHA256 over "<t>.<body>">

Only the web process delivers (the workers and the bulk process have no network by design). It runs `run_loop` from its
lifespan, which looks at the `job_callbacks` table; the job's own code paths (worker, cancel, sweep) don't know about
callbacks, so nothing here can slow a job down or fail one.

The address is untrusted input to a server that sits inside a home network, so delivery is built to be unable to reach
anything but the public internet (docs/security.md, "Job callbacks"):
  * https only, port 443 only, a host name (never an IP address or user:password), a name with a dot;
  * the name is resolved here, every address it resolves to must be public (IPv4 and IPv6: no private, loopback,
    link-local, CGNAT, multicast, reserved, documentation, 6to4/Teredo/NAT64 ranges, and none of this machine's own
    addresses), then the connection goes to that address itself (no second lookup), with the host name only used for the
    TLS check and the Host header;
  * no redirects are followed (any 3xx is a failure), 5 seconds for the whole attempt, 8 KB of response read at most,
    a request body of a few hundred bytes, at most 3 attempts (30 s and 5 min apart), at most 4 deliveries at once;
  * nothing blocks the event loop: DNS goes through the loop's resolver, the sockets are asyncio streams and every
    database call runs in a thread.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import re
import socket
import ssl
import time
from urllib.parse import urlsplit

from . import db
from .config import get_settings
from .security import random_token

log = logging.getLogger("acw.webhooks")

SIGNATURE_HEADER = "X-Audio-Cleaner-Signature"
USER_AGENT = "AudioCleaner-Webhook/1"
MAX_URL_CHARS = 500
ATTEMPT_SECONDS = 5.0  # DNS, connect, TLS, request and the first line of the answer, all together
MAX_ATTEMPTS = 3
RETRY_DELAYS = (30, 300)  # seconds after the 1st and the 2nd failed attempt
CLAIM_SECONDS = 60  # a claimed delivery that never reports back (a restart) is tried again after this
POLL_SECONDS = 3.0
CONCURRENCY = 4
MAX_RESPONSE_LINE = 8192
GIVE_UP_PENDING_SECONDS = 2 * 86400  # a callback whose job never ended
KEEP_RESULT_SECONDS = 7 * 86400
PRUNE_EVERY_SECONDS = 600

# Everything that is not the public internet. `ipaddress` already knows most of it (`is_global`); the list is explicit
# so a change in the library can't quietly open a range, and it adds the ranges that merely wrap another address.
_BLOCKED = [ipaddress.ip_network(n) for n in (
    "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16", "172.16.0.0/12", "192.0.0.0/24",
    "192.0.2.0/24", "192.88.99.0/24", "192.168.0.0/16", "198.18.0.0/15", "198.51.100.0/24", "203.0.113.0/24",
    "224.0.0.0/4", "240.0.0.0/4", "255.255.255.255/32",
    "::/96", "::ffff:0:0/96", "64:ff9b::/96", "64:ff9b:1::/48", "100::/64", "2001::/23", "2001:db8::/32", "2002::/16",
    "3fff::/20", "5f00::/16", "fc00::/7", "fe80::/10", "fec0::/10", "ff00::/8",
)]
_BAD_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".home.arpa", ".localdomain", ".invalid", ".test", ".example", ".onion", ".intranet", ".corp", ".home")
_LABEL = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")


class Refused(Exception):
    """The address may not be called (said to the account owner on the job, in plain words)."""


# ---------------------------------------------------------------- the address

def check_url(raw: str) -> str:
    """The normalised callback address, or ValueError with a message for the person who sent it. Syntax only (no
    network): the name is resolved and its addresses checked again at delivery, when it counts."""
    url = (raw or "").strip()
    if not url:
        raise ValueError("callbackUrl is empty.")
    if len(url) > MAX_URL_CHARS:
        raise ValueError(f"callbackUrl can be at most {MAX_URL_CHARS} characters.")
    if any(ord(ch) <= 32 or ord(ch) > 126 for ch in url):
        raise ValueError("callbackUrl must be a plain ASCII address (encode other characters).")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise ValueError("callbackUrl is not a valid address.") from None
    if parts.scheme.lower() != "https":
        raise ValueError("callbackUrl must start with https://")
    if "@" in parts.netloc or parts.fragment or "#" in url:
        raise ValueError("callbackUrl can't contain a user name, password or #fragment.")
    if port not in (None, 443):
        raise ValueError("callbackUrl must use the standard https port (443).")
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        raise ValueError("callbackUrl needs a host name.")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError("callbackUrl must use a host name, not an IP address.")
    if "." not in host or host.endswith(_BAD_SUFFIXES) or host == "localhost" or not all(_LABEL.match(x) for x in host.split(".")):
        raise ValueError("callbackUrl must be a public host name.")
    if host == (urlsplit(get_settings().public_origin).hostname or "").lower():
        raise ValueError("callbackUrl can't point back at this service.")
    path = parts.path or "/"
    return f"https://{host}{path}" + (f"?{parts.query}" if parts.query else "")


def _unwrap(ip: ipaddress._BaseAddress) -> ipaddress._BaseAddress:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip


def is_public_ip(value: str) -> bool:
    """True only for an address on the public internet that is not one of this machine's own."""
    try:
        ip = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError:
        return False
    ip = _unwrap(ip)
    if not ip.is_global or any(ip in net for net in _BLOCKED if net.version == ip.version):
        return False
    return str(ip) not in own_addresses() and not any(ip in net for net in _extra_blocked())


_own: dict = {"at": -1e9, "addrs": frozenset()}


def own_addresses() -> frozenset[str]:
    """This machine's addresses (its host name, and the source address it would use towards the internet over IPv4 and
    IPv6; connecting a UDP socket sends nothing). Loopback and LAN addresses are in the ranges above anyway; this is
    for a public address bound here."""
    if time.monotonic() - _own["at"] < 60:
        return _own["addrs"]
    found: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            found.add(str(_unwrap(ipaddress.ip_address(info[4][0].split("%", 1)[0]))))
    except (OSError, ValueError):
        pass
    for family, probe in ((socket.AF_INET, "192.0.2.1"), (socket.AF_INET6, "2001:db8::1")):
        try:
            with socket.socket(family, socket.SOCK_DGRAM) as s:
                s.connect((probe, 9))
                found.add(str(ipaddress.ip_address(s.getsockname()[0].split("%", 1)[0])))
        except (OSError, ValueError):
            pass
    _own.update(at=time.monotonic(), addrs=frozenset(found))
    return _own["addrs"]


def _extra_blocked() -> list:
    """ACW_WEBHOOK_DENY: more addresses or networks that may never be called (for example the public address of the home
    connection or of the VPS), comma separated."""
    out = []
    for item in (os.environ.get("ACW_WEBHOOK_DENY") or "").split(","):
        item = item.strip()
        if item:
            try:
                out.append(ipaddress.ip_network(item, strict=False))
            except ValueError:
                log.warning("ACW_WEBHOOK_DENY: ignoring %r", item)
    return out


async def _getaddrinfo(host: str) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(i[4][0] for i in infos))


async def resolve_public(host: str) -> str:
    """The address to connect to. Refuses unless every address the name resolves to is public (a name that answers with
    a public and a private address is a rebinding attempt, not a mixed answer to be picked from)."""
    await asyncio.to_thread(own_addresses)  # fills the cache off the event loop
    try:
        addrs = await _getaddrinfo(host)
    except OSError:
        raise Refused("The address could not be found (DNS).") from None
    if not addrs:
        raise Refused("The address could not be found (DNS).")
    if not all(is_public_ip(a) for a in addrs):
        raise Refused("The address points to a place that can't be called (it is not on the public internet).")
    return addrs[0].split("%", 1)[0]


# ---------------------------------------------------------------- signing

def new_secret() -> str:
    return "whsec_" + random_token(24)


def sign(secret: str, timestamp: int, body: bytes) -> str:
    """The value of X-Audio-Cleaner-Signature."""
    mac = hmac.new(secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256).hexdigest()
    return f"t={timestamp},v1={mac}"


def ensure_secret(c, user_id: str) -> str:
    """The account's signing secret, made on first use. Stored as plain text (it has to be usable to sign); see docs/security.md."""
    row = c.execute("SELECT webhook_secret FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is not None and row["webhook_secret"]:
        return row["webhook_secret"]
    secret = new_secret()
    c.execute("UPDATE users SET webhook_secret = ? WHERE id = ? AND webhook_secret IS NULL", (secret, user_id))
    return c.execute("SELECT webhook_secret FROM users WHERE id = ?", (user_id,)).fetchone()["webhook_secret"]


# ---------------------------------------------------------------- one delivery

async def _open(ip: str, host: str):
    """The connection (a seam for the tests): TLS to `ip` with certificate and name checked against `host`."""
    ctx = ssl.create_default_context()
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return await asyncio.open_connection(ip, 443, ssl=ctx, server_hostname=host, limit=MAX_RESPONSE_LINE)


async def _post(url: str, body: bytes, secret: str) -> int:
    """POST `body`; the HTTP status of the answer. Raises Refused, or an OSError/ssl/Timeout for a network failure."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    ip = await resolve_public(host)
    target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    stamp = int(time.time())
    head = (
        f"POST {target} HTTP/1.1\r\nHost: {host}\r\nUser-Agent: {USER_AGENT}\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n{SIGNATURE_HEADER}: {sign(secret, stamp, body)}\r\n"
        f"Accept: */*\r\nConnection: close\r\n\r\n"
    )
    reader, writer = await _open(ip, host)
    try:
        writer.write(head.encode("ascii") + body)
        await writer.drain()
        line = await reader.readline()
    except ValueError:  # the answer's first line is longer than the read limit
        raise OSError("answer too long") from None
    finally:
        writer.close()
    if len(line) > MAX_RESPONSE_LINE:
        raise OSError("answer too long")
    m = re.match(rb"HTTP/1\.[01] (\d{3})", line)
    if not m:
        raise OSError("not an HTTP answer")
    return int(m.group(1))


async def attempt(url: str, body: bytes, secret: str) -> tuple[int | None, str | None]:
    """One try. (status, None) when the receiver answered 2xx; otherwise (status or None, short reason). The second
    value "refused: ..." marks a failure that retrying can't fix."""
    try:
        status = await asyncio.wait_for(_post(url, body, secret), ATTEMPT_SECONDS)
    except Refused as e:
        return None, f"refused: {e}"
    except asyncio.TimeoutError:
        return None, "timed out"
    except ssl.SSLError:
        return None, "TLS error (certificate or protocol)"
    except (OSError, asyncio.IncompleteReadError):
        return None, "could not connect or read the answer"
    if 200 <= status < 300:
        return status, None
    if 300 <= status < 400:
        return status, f"HTTP {status} (redirects are not followed)"
    return status, f"HTTP {status}"


# ---------------------------------------------------------------- the queue

def payload(job: dict) -> dict:
    """What is sent. Small on purpose: ask the API for anything else."""
    status = job["status"]
    outputs = json.loads(job["outputs_json"]) if job.get("outputs_json") else []
    return {
        "type": f"job.{status}",
        "jobId": job["id"],
        "status": status,
        "error": job["error"],
        "costMicros": job["cost_micros"] or 0,
        "outputs": [{"name": o["file"], "kind": o["kind"], "bytes": o.get("bytes"), **({"lang": o["lang"]} if o.get("lang") else {})} for o in outputs],
        "finishedAt": job["finished_at"],
        "expiresAt": job["expires_at"] if status == "done" else None,
    }


def _claim_due(limit: int) -> list[dict]:
    now = db.now()
    rows = db.all_rows(
        "SELECT c.job_id, c.url, c.attempts FROM job_callbacks c JOIN jobs j ON j.id = c.job_id JOIN users u ON u.id = c.user_id"
        " WHERE c.state = 'pending' AND c.next_attempt_at <= ? AND j.status IN ('done', 'failed', 'canceled', 'expired')"
        " AND u.status = 'active' ORDER BY c.next_attempt_at LIMIT ?",
        (now, limit),
    )
    claimed = []
    for r in rows:
        n = db.execute(
            "UPDATE job_callbacks SET attempts = attempts + 1, next_attempt_at = ?, last_attempt_at = ?"
            " WHERE job_id = ? AND state = 'pending' AND next_attempt_at <= ?",
            (now + CLAIM_SECONDS, now, r["job_id"], now),
        ).rowcount
        if n:
            claimed.append({"job_id": r["job_id"], "url": r["url"], "attempts": r["attempts"] + 1})
    return claimed


def _prepare(job_id: str) -> tuple[bytes, str] | None:
    job = db.one("SELECT * FROM jobs WHERE id = ?", (job_id,))
    if job is None:
        return None
    with db.tx() as c:
        secret = ensure_secret(c, job["user_id"])
    return json.dumps(payload(dict(job)), separators=(",", ":")).encode(), secret


def _record(job_id: str, attempts: int, status: int | None, error: str | None) -> None:
    now = db.now()
    if error is None:
        db.execute(
            "UPDATE job_callbacks SET state = 'delivered', delivered_at = ?, last_status = ?, last_error = NULL WHERE job_id = ?",
            (now, status, job_id),
        )
    elif error.startswith("refused") or attempts >= MAX_ATTEMPTS:
        db.execute(
            "UPDATE job_callbacks SET state = 'failed', last_status = ?, last_error = ? WHERE job_id = ?", (status, error, job_id)
        )
    else:
        db.execute(
            "UPDATE job_callbacks SET next_attempt_at = ?, last_status = ?, last_error = ? WHERE job_id = ?",
            (now + RETRY_DELAYS[min(attempts, len(RETRY_DELAYS)) - 1], status, error, job_id),
        )


async def _deliver(item: dict, gate: asyncio.Semaphore) -> None:
    async with gate:
        try:
            prepared = await asyncio.to_thread(_prepare, item["job_id"])
            if prepared is None:
                await asyncio.to_thread(_record, item["job_id"], MAX_ATTEMPTS, None, "refused: the job is gone")
                return
            status, error = await attempt(item["url"], *prepared)
        except Exception as e:  # noqa: BLE001 - never lose the loop to one bad delivery
            log.exception("callback of job %s", item["job_id"])
            status, error = None, f"internal error ({type(e).__name__})"
        await asyncio.to_thread(_record, item["job_id"], item["attempts"], status, error)


async def deliver_due(limit: int = 20) -> int:
    """Send every callback that is due. Returns how many were tried."""
    items = await asyncio.to_thread(_claim_due, limit)
    if items:
        gate = asyncio.Semaphore(CONCURRENCY)
        await asyncio.gather(*(_deliver(i, gate) for i in items))
    return len(items)


def prune(now: int | None = None) -> None:
    """Old results go; a callback whose job never ended is given up."""
    now = db.now() if now is None else now
    db.execute(
        "UPDATE job_callbacks SET state = 'failed', last_error = 'gave up: the job did not end' WHERE state = 'pending' AND created_at < ?",
        (now - GIVE_UP_PENDING_SECONDS,),
    )
    db.execute(
        "DELETE FROM job_callbacks WHERE state != 'pending' AND COALESCE(last_attempt_at, created_at) < ?", (now - KEEP_RESULT_SECONDS,)
    )


async def run_loop() -> None:
    """Started by the web process only."""
    last_prune = 0.0
    while True:
        await asyncio.sleep(POLL_SECONDS)
        try:
            await deliver_due()
            if time.monotonic() - last_prune > PRUNE_EVERY_SECONDS:
                last_prune = time.monotonic()
                await asyncio.to_thread(prune)
        except Exception:  # noqa: BLE001
            log.exception("callback delivery loop")
