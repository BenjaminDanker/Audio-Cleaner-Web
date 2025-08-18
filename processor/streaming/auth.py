"""Authentication and token verification for streaming service."""
from __future__ import annotations

import os
import hmac
import hashlib
import json
import time
from typing import Union


def _b64url_no_pad(data: bytes) -> str:
    """Base64url encode without padding."""
    import base64
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _b64url_decode(s: str) -> bytes:
    """Base64url decode with padding restoration."""
    import base64
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def verify_session_token(token: str, expected_session: str) -> Union[bool, dict]:
    """Verify HMAC-signed session token.
    
    Token format: base64url(payload).base64url(hmacSHA256(payload, key))
    Payload JSON: {"sid": str, "exp": unix_ts, "mode": "stream", "userId": str}
    
    Args:
        token: The token to verify
        expected_session: Expected session ID
        
    Returns:
        False if invalid, or payload dict if valid
    """
    try:
        signing_key = os.getenv("STREAM_SESSION_SIGNING_KEY", "")
        if not signing_key:
            return False
            
        parts = token.split(".")
        if len(parts) != 2:
            return False
            
        payload_b64, mac_b64 = parts
        
        # Verify HMAC signature
        calc = hmac.new(signing_key.encode(), payload_b64.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(_b64url_no_pad(calc), mac_b64):
            return False
            
        # Decode and validate payload
        payload = json.loads(_b64url_decode(payload_b64).decode())
        
        # Validate session ID
        if str(payload.get("sid")) != str(expected_session):
            return False
            
        # Validate mode
        if str(payload.get("mode")) != "stream":
            return False
            
        # Validate expiration
        exp = int(payload.get("exp", 0))
        if exp <= int(time.time()):
            return False
            
        return payload
        
    except Exception:
        return False
