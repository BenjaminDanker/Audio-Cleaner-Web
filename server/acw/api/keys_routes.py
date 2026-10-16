from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from .. import auth, db, keycaps, webhooks
from ..security import client_ip, enforce, new_api_key

router = APIRouter(prefix="/api/keys", tags=["api-keys"])

SCOPES = {"stream", "jobs"}
MAX_KEYS = 10


class CreateKey(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    scopes: list[str] = Field(default_factory=lambda: ["stream"])
    # Most this key may spend in a calendar month (UTC), in cents; absent or null = no limit of its own.
    monthlyCapCents: int | None = Field(default=None, ge=1, le=keycaps.MAX_CAP_CENTS)


class UpdateKey(BaseModel):
    # null removes the limit. The field must be sent (an empty body changes nothing and is refused).
    monthlyCapCents: int | None = Field(default=None, ge=1, le=keycaps.MAX_CAP_CENTS)


def _key_json(r, used: int, resets_at: int) -> dict:
    cap = r["monthly_cap_micros"]
    return {
        "id": r["id"], "name": r["name"], "prefix": r["prefix"], "scopes": r["scopes"].split(","),
        "createdAt": r["created_at"], "lastUsedAt": r["last_used_at"],
        "monthlyCapCents": None if cap is None else cap // keycaps.MICROS_PER_CENT,
        "usedThisMonthMicros": used,
        "capResetsAt": resets_at,
    }


@router.get("")
def list_keys(principal: auth.Principal = Depends(auth.require_session)) -> dict:
    rows = db.all_rows(
        "SELECT id, name, prefix, scopes, created_at, last_used_at, monthly_cap_micros FROM api_keys"
        " WHERE user_id = ? AND revoked_at IS NULL ORDER BY created_at DESC",
        (principal.user_id,),
    )
    used = keycaps.usage_for([r["id"] for r in rows])
    resets_at = keycaps.month_bounds()[1]
    return {"keys": [_key_json(r, used[r["id"]], resets_at) for r in rows]}


@router.post("")
def create_key(body: CreateKey, request: Request, principal: auth.Principal = Depends(auth.require_session)) -> dict:
    enforce("key_create", principal.user_id)
    scopes = sorted(set(body.scopes))
    if not scopes or not set(scopes) <= SCOPES:
        raise HTTPException(status_code=400, detail="Unknown scope.")
    n = db.one("SELECT COUNT(*) AS n FROM api_keys WHERE user_id = ? AND revoked_at IS NULL", (principal.user_id,))["n"]
    if n >= MAX_KEYS:
        raise HTTPException(status_code=400, detail=f"You can have at most {MAX_KEYS} active keys.")
    key, prefix, key_hash = new_api_key()
    key_id = db.new_id()
    cap = None if body.monthlyCapCents is None else body.monthlyCapCents * keycaps.MICROS_PER_CENT
    db.execute(
        "INSERT INTO api_keys(id, user_id, name, prefix, key_hash, scopes, created_at, monthly_cap_micros) VALUES (?,?,?,?,?,?,?,?)",
        (key_id, principal.user_id, body.name.strip(), prefix, key_hash, ",".join(scopes), db.now(), cap),
    )
    db.audit("user", "api_key_created", user_id=principal.user_id, detail=f"{key_id} {','.join(scopes)}", ip=client_ip(request))
    # The full key is only ever shown here.
    return {"id": key_id, "key": key, "prefix": prefix, "scopes": scopes, "monthlyCapCents": body.monthlyCapCents}


@router.patch("/{key_id}")
def update_key(key_id: str, body: UpdateKey, request: Request, principal: auth.Principal = Depends(auth.require_session)) -> dict:
    """Change a key's monthly spending limit (or remove it with null). Takes effect for the next job or live minute."""
    if "monthlyCapCents" not in body.model_fields_set:
        raise HTTPException(status_code=400, detail="Nothing to change.")
    cap = None if body.monthlyCapCents is None else body.monthlyCapCents * keycaps.MICROS_PER_CENT
    n = db.execute(
        "UPDATE api_keys SET monthly_cap_micros = ? WHERE id = ? AND user_id = ? AND revoked_at IS NULL", (cap, key_id, principal.user_id)
    ).rowcount
    if not n:
        raise HTTPException(status_code=404, detail="Key not found.")
    db.audit("user", "api_key_cap_changed", user_id=principal.user_id, detail=f"{key_id} {body.monthlyCapCents}", ip=client_ip(request))
    r = db.one(
        "SELECT id, name, prefix, scopes, created_at, last_used_at, monthly_cap_micros FROM api_keys WHERE id = ?", (key_id,)
    )
    start, end = keycaps.month_bounds()
    return _key_json(r, keycaps.used_micros(db.conn(), key_id, start), end)


@router.delete("/{key_id}")
def revoke_key(key_id: str, request: Request, principal: auth.Principal = Depends(auth.require_session)) -> dict:
    n = db.execute(
        "UPDATE api_keys SET revoked_at = ? WHERE id = ? AND user_id = ? AND revoked_at IS NULL",
        (db.now(), key_id, principal.user_id),
    ).rowcount
    if not n:
        raise HTTPException(status_code=404, detail="Key not found.")
    db.audit("user", "api_key_revoked", user_id=principal.user_id, detail=key_id, ip=client_ip(request))
    return {"ok": True}


# ---- the secret that signs job callbacks (one per account; browser session only, like the keys)

@router.get("/webhook-secret")
def get_webhook_secret(principal: auth.Principal = Depends(auth.require_session)) -> dict:
    row = db.one("SELECT webhook_secret FROM users WHERE id = ?", (principal.user_id,))
    return {"secret": row["webhook_secret"] if row else None}


@router.post("/webhook-secret")
def rotate_webhook_secret(request: Request, principal: auth.Principal = Depends(auth.require_session)) -> dict:
    """Make the account's callback secret, or replace it with a new one. Callbacks that are still waiting to be
    delivered are signed with the new one."""
    enforce("key_create", principal.user_id)
    secret = webhooks.new_secret()
    db.execute("UPDATE users SET webhook_secret = ? WHERE id = ?", (secret, principal.user_id))
    db.audit("user", "webhook_secret_rotated", user_id=principal.user_id, ip=client_ip(request))
    return {"secret": secret}
