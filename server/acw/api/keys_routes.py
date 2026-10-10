from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from .. import auth, db
from ..security import client_ip, enforce, new_api_key

router = APIRouter(prefix="/api/keys", tags=["api-keys"])

SCOPES = {"stream", "jobs"}
MAX_KEYS = 10


class CreateKey(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    scopes: list[str] = Field(default_factory=lambda: ["stream"])


@router.get("")
def list_keys(principal: auth.Principal = Depends(auth.require_session)) -> dict:
    rows = db.all_rows(
        "SELECT id, name, prefix, scopes, created_at, last_used_at FROM api_keys"
        " WHERE user_id = ? AND revoked_at IS NULL ORDER BY created_at DESC",
        (principal.user_id,),
    )
    return {
        "keys": [
            {"id": r["id"], "name": r["name"], "prefix": r["prefix"], "scopes": r["scopes"].split(","),
             "createdAt": r["created_at"], "lastUsedAt": r["last_used_at"]}
            for r in rows
        ]
    }


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
    db.execute(
        "INSERT INTO api_keys(id, user_id, name, prefix, key_hash, scopes, created_at) VALUES (?,?,?,?,?,?,?)",
        (key_id, principal.user_id, body.name.strip(), prefix, key_hash, ",".join(scopes), db.now()),
    )
    db.audit("user", "api_key_created", user_id=principal.user_id, detail=f"{key_id} {','.join(scopes)}", ip=client_ip(request))
    # The full key is only ever shown here.
    return {"id": key_id, "key": key, "prefix": prefix, "scopes": scopes}


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
