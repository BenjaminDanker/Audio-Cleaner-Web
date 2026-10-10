"""Sign-in: GitHub OAuth, Google OpenID Connect, email magic links.

OAuth flows use PKCE plus a state value that is both stored server-side and
bound to the browser with a short-lived cookie, so a callback can only finish
in the browser that started it (prevents login CSRF / session fixation).
Accounts are keyed by provider-verified email.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import smtplib
from email.message import EmailMessage
from urllib.parse import urlencode

import anyio
import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from .. import auth, db
from ..billing import ledger
from ..config import get_settings
from ..security import client_ip, consteq, enforce, random_token, sha256_hex
from .turnstile import verify_turnstile

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])

OAUTH_STATE_TTL = 600
OAUTH_COOKIE = "acw_oauth"
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,63}$")

PROVIDERS = {
    "github": {
        "authorize": "https://github.com/login/oauth/authorize",
        "token": "https://github.com/login/oauth/access_token",
        "scope": "read:user user:email",
    },
    "google": {
        "authorize": "https://accounts.google.com/o/oauth2/v2/auth",
        "token": "https://oauth2.googleapis.com/token",
        "scope": "openid email profile",
    },
}


def _client(provider: str) -> tuple[str, str]:
    s = get_settings()
    if provider == "github" and s.github_client_id and s.github_client_secret:
        return s.github_client_id, s.github_client_secret
    if provider == "google" and s.google_client_id and s.google_client_secret:
        return s.google_client_id, s.google_client_secret
    raise HTTPException(status_code=404, detail="Sign-in provider not configured.")


def _redirect_uri(provider: str) -> str:
    return get_settings().url(f"/api/auth/oauth/{provider}/callback")


def _pkce_pair() -> tuple[str, str]:
    verifier = random_token(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    return verifier, challenge


def _require_same_origin(request: Request) -> None:
    if request.headers.get("origin") != get_settings().public_origin:
        raise HTTPException(status_code=403, detail="Cross-origin request refused.")


def _app_redirect(path: str = "/") -> RedirectResponse:
    return RedirectResponse(get_settings().url(path), status_code=303)


@router.get("/providers")
def providers() -> dict:
    s = get_settings()
    return {
        "github": bool(s.github_client_id and s.github_client_secret),
        "google": bool(s.google_client_id and s.google_client_secret),
        "email": bool(s.smtp_host and s.smtp_from),
        "turnstileSiteKey": s.turnstile_site_key,
    }


@router.get("/oauth/{provider}/start")
def oauth_start(provider: str, request: Request) -> RedirectResponse:
    if provider not in PROVIDERS:
        raise HTTPException(status_code=404, detail="Unknown provider.")
    enforce("auth", client_ip(request))
    client_id, _ = _client(provider)
    state = random_token(32)
    verifier, challenge = _pkce_pair()
    nonce = random_token(16) if provider == "google" else None
    now = db.now()
    db.execute(
        "INSERT INTO oauth_states(state_hash, provider, code_verifier, nonce, created_at, expires_at) VALUES (?,?,?,?,?,?)",
        (sha256_hex(state), provider, verifier, nonce, now, now + OAUTH_STATE_TTL),
    )
    params = {
        "client_id": client_id,
        "redirect_uri": _redirect_uri(provider),
        "scope": PROVIDERS[provider]["scope"],
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    if provider == "google":
        params.update({"response_type": "code", "nonce": nonce, "prompt": "select_account"})
    else:
        params["allow_signup"] = "true"
    resp = RedirectResponse(f"{PROVIDERS[provider]['authorize']}?{urlencode(params)}", status_code=303)
    s = get_settings()
    resp.set_cookie(
        OAUTH_COOKIE,
        state,
        max_age=OAUTH_STATE_TTL,
        path=s.cookie_path + "api/auth/oauth/",
        secure=s.production,
        httponly=True,
        samesite="lax",
    )
    return resp


def _jwt_payload(token: str) -> dict:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("malformed id_token")
    pad = "=" * (-len(parts[1]) % 4)
    return json.loads(base64.urlsafe_b64decode(parts[1] + pad))


async def _github_identity(http: httpx.AsyncClient, access_token: str) -> tuple[str, str, str | None]:
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/vnd.github+json"}
    user = (await http.get("https://api.github.com/user", headers=headers)).raise_for_status().json()
    emails = (await http.get("https://api.github.com/user/emails", headers=headers)).raise_for_status().json()
    verified = [e for e in emails if e.get("verified")]
    primary = next((e for e in verified if e.get("primary")), None) or (verified[0] if verified else None)
    if primary is None:
        raise HTTPException(status_code=400, detail="Your GitHub account has no verified email address.")
    return str(user["id"]), primary["email"], user.get("name") or user.get("login")


def _google_identity(token_json: dict, client_id: str, nonce: str | None) -> tuple[str, str, str | None]:
    # The id_token came straight from Google's token endpoint over TLS, so its
    # signature check is not required (OIDC Core 3.1.3.7); claims still are.
    claims = _jwt_payload(token_json.get("id_token", ""))
    if claims.get("iss") not in ("https://accounts.google.com", "accounts.google.com"):
        raise HTTPException(status_code=400, detail="Bad token issuer.")
    if claims.get("aud") != client_id:
        raise HTTPException(status_code=400, detail="Bad token audience.")
    if int(claims.get("exp", 0)) <= db.now():
        raise HTTPException(status_code=400, detail="Token expired.")
    if nonce is None or not consteq(str(claims.get("nonce", "")), nonce):
        raise HTTPException(status_code=400, detail="Bad nonce.")
    if not claims.get("email") or claims.get("email_verified") is not True:
        raise HTTPException(status_code=400, detail="Your Google account email is not verified.")
    return str(claims["sub"]), claims["email"], claims.get("name")


@router.get("/oauth/{provider}/callback")
async def oauth_callback(provider: str, request: Request, code: str = "", state: str = "", error: str = "") -> Response:
    if provider not in PROVIDERS:
        raise HTTPException(status_code=404, detail="Unknown provider.")
    if error:
        return _app_redirect("/login?error=cancelled")
    cookie_state = request.cookies.get(OAUTH_COOKIE) or ""
    if not code or not state or len(state) > 128 or not consteq(cookie_state, state):
        return _app_redirect("/login?error=state")
    with db.tx() as c:
        row = c.execute("SELECT * FROM oauth_states WHERE state_hash = ?", (sha256_hex(state),)).fetchone()
        c.execute("DELETE FROM oauth_states WHERE state_hash = ?", (sha256_hex(state),))
    if row is None or row["provider"] != provider or row["expires_at"] <= db.now():
        return _app_redirect("/login?error=expired")
    client_id, client_secret = _client(provider)
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code[:512],
        "redirect_uri": _redirect_uri(provider),
        "code_verifier": row["code_verifier"],
    }
    if provider == "google":
        data["grant_type"] = "authorization_code"
    try:
        async with httpx.AsyncClient(timeout=15) as http:
            tok = await http.post(PROVIDERS[provider]["token"], data=data, headers={"Accept": "application/json"})
            tok.raise_for_status()
            token_json = tok.json()
            if "error" in token_json:
                raise HTTPException(status_code=400, detail="Sign-in was rejected by the provider.")
            if provider == "github":
                sub, email, name = await _github_identity(http, token_json["access_token"])
            else:
                sub, email, name = _google_identity(token_json, client_id, row["nonce"])
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001
        log.exception("oauth %s exchange failed", provider)
        return _app_redirect("/login?error=provider")
    user = auth.link_identity(provider, sub, email, name)
    if user["status"] != "active":
        return _app_redirect("/login?error=suspended")
    resp = _app_redirect("/app")
    resp.delete_cookie(OAUTH_COOKIE, path=get_settings().cookie_path + "api/auth/oauth/")
    auth.create_session(resp, request, user["id"])
    return resp


class EmailLoginBody(BaseModel):
    email: str = Field(max_length=320)
    turnstileToken: str | None = Field(default=None, max_length=4096)


def _send_mail(to: str, subject: str, body: str) -> None:
    s = get_settings()
    msg = EmailMessage()
    msg["From"] = s.smtp_from
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=20) as smtp:
        smtp.starttls()
        if s.smtp_user:
            smtp.login(s.smtp_user, s.smtp_password or "")
        smtp.send_message(msg)


@router.post("/email")
async def email_login(body: EmailLoginBody, request: Request) -> dict:
    _require_same_origin(request)
    s = get_settings()
    if not (s.smtp_host and s.smtp_from):
        raise HTTPException(status_code=404, detail="Email sign-in is not enabled.")
    ip = client_ip(request)
    enforce("email_link", f"ip:{ip}")
    email = body.email.strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    await verify_turnstile(body.turnstileToken, ip)
    enforce("email_link", f"addr:{email}")
    token = auth.create_login_token(email, "email_login", ip)
    link = s.url(f"/login/verify#token={token}")
    text = (
        "Use this link to sign in to Audio Cleaner. It works once and expires in 15 minutes.\n\n"
        f"{link}\n\nIf you did not ask for this, ignore this email."
    )
    try:
        await anyio.to_thread.run_sync(_send_mail, email, "Your Audio Cleaner sign-in link", text)
    except Exception:  # noqa: BLE001
        log.exception("sending login email failed")
        raise HTTPException(status_code=502, detail="Could not send the email right now.")
    # Same response whether or not the account exists.
    return {"ok": True}


class TokenBody(BaseModel):
    token: str = Field(max_length=128)


@router.post("/email/verify")
def email_verify(body: TokenBody, request: Request, response: Response) -> dict:
    _require_same_origin(request)
    enforce("auth", client_ip(request))
    email = auth.consume_login_token(body.token)
    if email is None:
        raise HTTPException(status_code=400, detail="This sign-in link is invalid or expired.")
    user = auth.get_or_create_user(email)
    if user["status"] != "active":
        raise HTTPException(status_code=403, detail="This account is suspended.")
    auth.create_session(response, request, user["id"])
    return {"ok": True}


@router.post("/logout")
def logout(response: Response, principal: auth.Principal = Depends(auth.require_session)) -> dict:
    if principal.session_hash:
        auth.destroy_session(principal.session_hash)
    auth.clear_session_cookie(response)
    return {"ok": True}


@router.post("/logout-all")
def logout_all(response: Response, principal: auth.Principal = Depends(auth.require_session)) -> dict:
    db.execute("DELETE FROM sessions WHERE user_id = ?", (principal.user_id,))
    auth.clear_session_cookie(response)
    return {"ok": True}


me_router = APIRouter(prefix="/api", tags=["account"])


@me_router.get("/me")
def me(principal: auth.Principal | None = Depends(auth.optional_principal)) -> dict:
    if principal is None:
        return {"signedIn": False}
    user = db.one("SELECT * FROM users WHERE id = ?", (principal.user_id,))
    bal = ledger.balance(principal.user_id)
    return {
        "signedIn": True,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "displayName": user["display_name"],
            "isAdmin": bool(user["is_admin"]),
            "status": user["status"],
            "tosAccepted": user["tos_version"] == auth.TOS_VERSION,
            "createdAt": user["created_at"],
        },
        "csrfToken": principal.csrf_token,
        "balance": bal,
        "autoReload": {
            "enabled": bool(user["auto_reload_enabled"]),
            "thresholdMicros": user["auto_reload_threshold_micros"],
            "amountCents": user["auto_reload_amount_cents"],
            "monthlyCapCents": user["auto_reload_monthly_cap_cents"],
            "hasCard": bool(user["default_payment_method"]),
            "failures": user["auto_reload_failures"],
        },
    }


class TosBody(BaseModel):
    version: str = Field(max_length=32)


@me_router.post("/account/tos")
def accept_tos(body: TosBody, request: Request, principal: auth.Principal = Depends(auth.require_session)) -> dict:
    if body.version != auth.TOS_VERSION:
        raise HTTPException(status_code=409, detail="Terms changed, reload the page.")
    db.execute(
        "UPDATE users SET tos_version = ?, tos_accepted_at = ? WHERE id = ?",
        (auth.TOS_VERSION, db.now(), principal.user_id),
    )
    db.audit("user", "tos_accepted", user_id=principal.user_id, detail=auth.TOS_VERSION, ip=client_ip(request))
    return {"ok": True}
