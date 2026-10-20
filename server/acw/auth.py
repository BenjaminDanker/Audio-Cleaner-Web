"""Accounts, browser sessions, API keys and the dependencies routes use.

Browser requests authenticate with an HttpOnly session cookie. Every
state-changing cookie request must also carry X-CSRF-Token (the per-session
token from /api/me) and, when present, a same-site Origin. API clients (the
OBS plugin, scripts) use `Authorization: Bearer acw_...`; bearer requests are
not exposed to CSRF so they skip the token check.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import Depends, HTTPException, Request, Response
from fastapi.websockets import WebSocket

from . import db
from .config import get_settings
from .security import (
    API_KEY_PREFIX,
    canonical_email,
    client_ip,
    consteq,
    keyed_hash,
    random_token,
    sha256_hex,
    user_agent,
)

log = logging.getLogger(__name__)
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
# A browser that has not been seen for this long must sign in again (the session also has a fixed life).
SESSION_IDLE_SECONDS = 14 * 86400


@dataclass
class Principal:
    user_id: str
    email: str
    is_admin: bool
    status: str
    via: Literal["session", "api_key"]
    session_hash: str | None = None
    csrf_token: str | None = None
    api_key_id: str | None = None
    scopes: frozenset[str] = frozenset()


# ---------------------------------------------------------------- users

def _grant_trial(c, user_id: str, email: str) -> None:
    """A new account's free live trial, once per mailbox ever (a deleted account signing up again doesn't get it
    twice: only a keyed hash of the mailbox is kept)."""
    seconds = get_settings().live_trial_seconds
    if seconds <= 0:
        return
    # The key is the mailbox, not the spelling: +tags, Gmail dots and googlemail.com all reach the same inbox.
    key = keyed_hash("welcome", canonical_email(email))
    if c.execute("INSERT OR IGNORE INTO welcome_claims(email_hash, created_at) VALUES (?, ?)", (key, db.now())).rowcount:
        c.execute("UPDATE users SET trial_live_seconds = ? WHERE id = ?", (float(seconds), user_id))


def get_or_create_user(email: str, display_name: str | None = None) -> dict:
    email = email.strip().lower()
    settings = get_settings()
    with db.tx() as c:
        row = c.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if row is None:
            user_id = db.new_id()
            c.execute(
                "INSERT INTO users(id, email, display_name, created_at, is_admin) VALUES (?,?,?,?,?)",
                (user_id, email, display_name, db.now(), 1 if email in settings.admin_emails else 0),
            )
            c.execute(
                "INSERT INTO audit_log(user_id, actor, action, created_at) VALUES (?, 'system', 'user_created', ?)",
                (user_id, db.now()),
            )
            _grant_trial(c, user_id, email)
            row = c.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        elif (email in settings.admin_emails) != bool(row["is_admin"]):
            # The list in the environment is the only source of admin rights: add and remove follow it.
            c.execute("UPDATE users SET is_admin = ? WHERE id = ?", (1 if email in settings.admin_emails else 0, row["id"]))
            row = c.execute("SELECT * FROM users WHERE id = ?", (row["id"],)).fetchone()
    return dict(row)


def link_identity(provider: str, provider_user_id: str, email: str, display_name: str | None) -> dict:
    """Find the account for an external identity, creating or linking by
    verified email. Callers must only pass provider-verified emails."""
    row = db.one(
        "SELECT u.* FROM identities i JOIN users u ON u.id = i.user_id WHERE i.provider = ? AND i.provider_user_id = ?",
        (provider, provider_user_id),
    )
    if row is not None:
        return dict(row)
    user = get_or_create_user(email, display_name)
    db.execute(
        "INSERT OR IGNORE INTO identities(provider, provider_user_id, user_id, email, created_at) VALUES (?,?,?,?,?)",
        (provider, provider_user_id, user["id"], email.lower(), db.now()),
    )
    return user


# ---------------------------------------------------------------- sessions

# The Terms that are live: data/legal/current.json, written by deploy/legal-archive.sh when a deploy sees new text (docs/operations.md,
# "Legal versions"): {"terms": {"version": "2026-10-20", "sha256": "<hash of the text>"}, "privacy": {...}}. Without it (development, the tests) the
# version is "dev" and acceptances carry no hash.
DEV_TOS_VERSION = "dev"
_VERSION_RE = re.compile(r"[0-9A-Za-z._-]{1,32}")
_terms_seen: dict = {"key": None, "value": (DEV_TOS_VERSION, None)}


def _read_terms(f: Path, exists: bool) -> tuple[str, str | None]:
    say = log.warning if get_settings().production else log.info
    if not exists:
        say("%s is missing: the Terms version is %s and acceptances carry no text hash (deploy/legal-archive.sh writes it)", f, DEV_TOS_VERSION)
        return DEV_TOS_VERSION, None
    try:
        terms = json.loads(f.read_text(encoding="utf-8"))["terms"]
        version, digest = str(terms["version"]), str(terms["sha256"])
        if _VERSION_RE.fullmatch(version) and re.fullmatch(r"[0-9a-f]{64}", digest):
            return version, digest
        raise ValueError("not a version and a SHA-256")
    except (OSError, ValueError, KeyError, TypeError) as e:
        log.warning("cannot use %s (%s): the Terms version is %s and acceptances carry no text hash", f, e, DEV_TOS_VERSION)
        return DEV_TOS_VERSION, None


def terms_current() -> tuple[str, str | None]:
    """(version, SHA-256 of the text) of the Terms that are live. Read again whenever the file changes, so a deploy needs no restart."""
    f = get_settings().data_dir / "legal" / "current.json"
    try:
        st = f.stat()
        key = (str(f), st.st_mtime_ns, st.st_size)
    except OSError:
        key = (str(f), None, None)
    if _terms_seen["key"] != key:
        _terms_seen.update(key=key, value=_read_terms(f, key[1] is not None))
    return _terms_seen["value"]


def tos_version() -> str:
    """The version of the Terms users accept; a user whose recorded version differs sees the "We've updated our Terms" notice."""
    return terms_current()[0]


def tos_detail(how: str | None = None) -> str:
    """The audit detail of an acceptance: the version, the hash of the Terms text it names and how it came about,
    e.g. "2026-10-20 sha256:60a6d548... (sign-in)"; just the version (and how) when there is no hash (development)."""
    version, digest = terms_current()
    return version + (f" sha256:{digest}" if digest else "") + (f" ({how})" if how else "")


def accept_terms(user_id: str, how: str, ip: str | None = None) -> None:
    """Record that the user agreed to the current Terms (version, time, IP, how):
    by signing in (the sign-in page says so), or by continuing to use the
    service after the Terms changed. Kept as consent evidence (disputes)."""
    row = db.one("SELECT tos_version FROM users WHERE id = ?", (user_id,))
    version = tos_version()
    if row is None or row["tos_version"] == version:
        return
    db.execute("UPDATE users SET tos_version = ?, tos_accepted_at = ? WHERE id = ?", (version, db.now(), user_id))
    db.audit("user", "tos_accepted", user_id=user_id, detail=tos_detail(how), ip=ip)


def create_session(response: Response, request: Request, user_id: str) -> None:
    from .security import client_ip

    accept_terms(user_id, "sign-in", client_ip(request))
    settings = get_settings()
    token = random_token(32)
    now = db.now()
    db.execute(
        "INSERT INTO sessions(id_hash, user_id, csrf_token, created_at, expires_at, last_seen_at, ip, user_agent)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (
            sha256_hex(token),
            user_id,
            random_token(24),
            now,
            now + settings.session_days * 86400,
            now,
            client_ip(request),
            user_agent(request),
        ),
    )
    db.audit("user", "login", user_id=user_id, ip=client_ip(request))
    response.set_cookie(
        settings.session_cookie_name,
        token,
        max_age=settings.session_days * 86400,
        path=settings.cookie_path,
        secure=settings.production,
        httponly=True,
        samesite="lax",
    )


def clear_session_cookie(response: Response) -> None:
    settings = get_settings()
    response.delete_cookie(
        settings.session_cookie_name,
        path=settings.cookie_path,
        secure=settings.production,
        httponly=True,
        samesite="lax",
    )


def destroy_session(session_hash: str) -> None:
    db.execute("DELETE FROM sessions WHERE id_hash = ?", (session_hash,))


def _principal_from_cookie(token: str | None) -> Principal | None:
    if not token or len(token) > 128:
        return None
    h = sha256_hex(token)
    row = db.one(
        "SELECT s.id_hash, s.csrf_token, s.expires_at, s.last_seen_at, u.id, u.email, u.is_admin, u.status"
        " FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.id_hash = ?",
        (h,),
    )
    now = db.now()
    if row is None or row["expires_at"] <= now or now - row["last_seen_at"] > SESSION_IDLE_SECONDS:
        return None
    if now - row["last_seen_at"] > 300:
        db.execute("UPDATE sessions SET last_seen_at = ? WHERE id_hash = ?", (now, h))
    return Principal(
        user_id=row["id"],
        email=row["email"],
        is_admin=bool(row["is_admin"]),
        status=row["status"],
        via="session",
        session_hash=h,
        csrf_token=row["csrf_token"],
        scopes=frozenset({"*"}),
    )


def _principal_from_api_key(key: str) -> Principal | None:
    if not key.startswith(API_KEY_PREFIX) or len(key) > 100:
        return None
    row = db.one(
        "SELECT k.id AS key_id, k.scopes, k.revoked_at, k.last_used_at, u.id, u.email, u.is_admin, u.status"
        " FROM api_keys k JOIN users u ON u.id = k.user_id WHERE k.key_hash = ?",
        (sha256_hex(key),),
    )
    if row is None or row["revoked_at"] is not None:
        return None
    now = db.now()
    if not row["last_used_at"] or now - row["last_used_at"] > 300:
        db.execute("UPDATE api_keys SET last_used_at = ? WHERE id = ?", (now, row["key_id"]))
    return Principal(
        user_id=row["id"],
        email=row["email"],
        is_admin=False,  # admin actions always need a browser session
        status=row["status"],
        via="api_key",
        api_key_id=row["key_id"],
        scopes=frozenset(row["scopes"].split(",")),
    )


def _bearer(headers) -> str | None:
    auth = headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def _check_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin is None:
        # Browsers send Origin on cross-site POSTs; a missing Origin with a valid
        # CSRF token can only come from a same-origin or non-browser client.
        return
    if origin != get_settings().public_origin:
        raise HTTPException(status_code=403, detail="Cross-origin request refused.")


def optional_principal(request: Request) -> Principal | None:
    bearer = _bearer(request.headers)
    if bearer:
        p = _principal_from_api_key(bearer)
        if p is None:
            raise HTTPException(status_code=401, detail="Invalid API key.")
        return p
    p = _principal_from_cookie(request.cookies.get(get_settings().session_cookie_name))
    if p is not None and request.method not in SAFE_METHODS:
        _check_origin(request)
        sent = request.headers.get("x-csrf-token") or ""
        if not p.csrf_token or not consteq(sent, p.csrf_token):
            raise HTTPException(status_code=403, detail="Missing or invalid CSRF token.")
    return p


def require_user(principal: Principal | None = Depends(optional_principal)) -> Principal:
    if principal is None:
        raise HTTPException(status_code=401, detail="Sign in required.")
    if principal.status != "active":
        raise HTTPException(status_code=403, detail="This account is suspended. Contact support.")
    return principal


def require_session(principal: Principal = Depends(require_user)) -> Principal:
    """Browser-only actions (billing, keys, account changes)."""
    if principal.via != "session":
        raise HTTPException(status_code=403, detail="This action needs a signed-in browser session.")
    return principal


def require_scope(scope: str):
    def dep(principal: Principal = Depends(require_user)) -> Principal:
        if "*" not in principal.scopes and scope not in principal.scopes:
            raise HTTPException(status_code=403, detail=f"API key lacks the '{scope}' scope.")
        return principal

    return dep


def websocket_principal(ws: WebSocket) -> Principal | None:
    """Auth for WebSocket upgrades: API key header, or session cookie with a
    same-origin Origin header (blocks cross-site WebSocket hijacking)."""
    bearer = _bearer(ws.headers)
    if bearer:
        return _principal_from_api_key(bearer)
    p = _principal_from_cookie(ws.cookies.get(get_settings().session_cookie_name))
    if p is None:
        return None
    if ws.headers.get("origin") != get_settings().public_origin:
        return None
    return p


# ---------------------------------------------------------------- login links

LOGIN_LINK_TTL = 15 * 60


def create_login_token(email: str, purpose: str, ip: str | None) -> str:
    token = random_token(32)
    now = db.now()
    db.execute(
        "INSERT INTO login_tokens(token_hash, email, purpose, created_at, expires_at, ip) VALUES (?,?,?,?,?,?)",
        (sha256_hex(token), email.strip().lower(), purpose, now, now + LOGIN_LINK_TTL, ip),
    )
    return token


def consume_login_token(token: str) -> str | None:
    """Single use. Returns the email or None."""
    if not token or len(token) > 128:
        return None
    h = sha256_hex(token)
    with db.tx() as c:
        row = c.execute("SELECT * FROM login_tokens WHERE token_hash = ?", (h,)).fetchone()
        if row is None or row["used_at"] is not None or row["expires_at"] <= db.now():
            return None
        c.execute("UPDATE login_tokens SET used_at = ? WHERE token_hash = ?", (db.now(), h))
        return row["email"]


def purge_expired() -> None:
    now = db.now()
    db.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
    db.execute("DELETE FROM login_tokens WHERE expires_at <= ?", (now - 86400,))
    db.execute("DELETE FROM oauth_states WHERE expires_at <= ?", (now,))
