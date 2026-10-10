"""Cloudflare Turnstile check, used on unauthenticated endpoints that cost us
something (sending email). Disabled when TURNSTILE_SECRET is unset."""
from __future__ import annotations

import httpx
from fastapi import HTTPException

from ..config import get_settings

VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


async def verify_turnstile(token: str | None, ip: str) -> None:
    secret = get_settings().turnstile_secret
    if not secret:
        return
    if not token:
        raise HTTPException(status_code=400, detail="Complete the human check.")
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            r = await http.post(VERIFY_URL, data={"secret": secret, "response": token, "remoteip": ip})
            ok = r.json().get("success") is True
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        raise HTTPException(status_code=400, detail="Human check failed, try again.")
