"""Security utilities and middleware for streaming service."""
from __future__ import annotations

import re
import logging
from typing import Optional, Dict, Any
from fastapi import HTTPException, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field

from auth import verify_session_token


# Configure logging
logger = logging.getLogger(__name__)

# Rate limiter configuration
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["1000/hour"]  # Default global limit
)

# Security bearer scheme
security = HTTPBearer(auto_error=False)

# Session ID validation pattern (alphanumeric, hyphens, underscores only)
SESSION_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,100}$')


class StopStreamRequest(BaseModel):
    """Validated request model for stopping streams."""
    sessionId: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r'^[a-zA-Z0-9_-]+$',
        description="Session ID containing only alphanumeric characters, hyphens, and underscores"
    )


def validate_session_id(session_id: str) -> str:
    """Validate and sanitize session ID.
    
    Args:
        session_id: Session ID to validate
        
    Returns:
        Validated session ID
        
    Raises:
        HTTPException: If session ID is invalid
    """
    if not session_id or not isinstance(session_id, str):
        raise HTTPException(status_code=400, detail="Session ID is required")
        
    if not SESSION_ID_PATTERN.match(session_id):
        raise HTTPException(
            status_code=400, 
            detail="Invalid session ID format. Only alphanumeric characters, hyphens, and underscores allowed."
        )
        
    return session_id


def extract_token_from_request(
    authorization: Optional[HTTPAuthorizationCredentials] = None,
    x_session_token: Optional[str] = None
) -> Optional[str]:
    """Extract authentication token from various sources.
    
    Args:
        authorization: Bearer token from Authorization header
        x_session_token: Token from X-Session-Token header
        
    Returns:
        Extracted token or None
    """
    if authorization and authorization.credentials:
        return authorization.credentials
    elif x_session_token:
        return x_session_token
    
    return None


def verify_session_access(token: str, session_id: str) -> Dict[str, Any]:
    """Verify that the token grants access to the specified session.
    
    Args:
        token: Authentication token
        session_id: Session ID to verify access for
        
    Returns:
        Token payload if valid
        
    Raises:
        HTTPException: If token is invalid or doesn't grant access to session
    """
    if not token:
        raise HTTPException(
            status_code=401, 
            detail="Authentication required. Provide token via Authorization header or X-Session-Token header."
        )
    
    token_payload = verify_session_token(token, expected_session=session_id)
    if not token_payload:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired session token"
        )
        
    return token_payload


def get_client_ip(request: Request) -> str:
    """Get client IP address with proxy support.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Client IP address
    """
    # Check for X-Forwarded-For header (common in load balancers/proxies)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in the chain (original client)
        return forwarded_for.split(",")[0].strip()
    
    # Check for X-Real-IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    
    # Fall back to direct connection IP
    return request.client.host if request.client else "unknown"


def log_security_event(
    event_type: str,
    request: Request,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """Log security-related events for monitoring.
    
    Args:
        event_type: Type of security event
        request: FastAPI request object
        session_id: Associated session ID
        user_id: Associated user ID
        details: Additional event details
    """
    client_ip = get_client_ip(request)
    user_agent = request.headers.get("User-Agent", "unknown")
    
    log_data = {
        "event_type": event_type,
        "client_ip": client_ip,
        "user_agent": user_agent,
        "path": str(request.url.path),
        "method": request.method,
        "session_id": session_id,
        "user_id": user_id,
        **(details or {})
    }
    
    logger.warning(f"Security event: {event_type}", extra=log_data)


def setup_rate_limiting_handler(app):
    """Setup rate limiting exception handler for the FastAPI app.
    
    Args:
        app: FastAPI application instance
    """
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# CORS origins configuration
ALLOWED_ORIGINS = [
    "http://localhost:3000",      # Development frontend
    "http://localhost:4280",      # SWA CLI dev server
    "https://*.azurestaticapps.net",  # Production SWA
    # Add more origins as needed
]


def is_origin_allowed(origin: str) -> bool:
    """Check if an origin is allowed for CORS.
    
    Args:
        origin: Origin header value
        
    Returns:
        True if origin is allowed
    """
    if not origin:
        return False
        
    for allowed in ALLOWED_ORIGINS:
        if allowed == origin:
            return True
        # Handle wildcard patterns
        if "*" in allowed:
            pattern = allowed.replace("*", ".*")
            if re.match(pattern, origin):
                return True
                
    return False


# Connection limits per IP (simple in-memory tracking for development)
# In production, use Redis or similar for distributed rate limiting
connection_counts: Dict[str, int] = {}
MAX_CONNECTIONS_PER_IP = 5


def check_connection_limit(client_ip: str) -> bool:
    """Check if client IP has exceeded connection limits.
    
    Args:
        client_ip: Client IP address
        
    Returns:
        True if within limits
    """
    current_count = connection_counts.get(client_ip, 0)
    return current_count < MAX_CONNECTIONS_PER_IP


def track_connection(client_ip: str, connected: bool) -> None:
    """Track connection count for an IP.
    
    Args:
        client_ip: Client IP address
        connected: True if connecting, False if disconnecting
    """
    if connected:
        connection_counts[client_ip] = connection_counts.get(client_ip, 0) + 1
    else:
        if client_ip in connection_counts:
            connection_counts[client_ip] = max(0, connection_counts[client_ip] - 1)
            if connection_counts[client_ip] == 0:
                del connection_counts[client_ip]


# Audio data size limits (in bytes)
MAX_AUDIO_CHUNK_SIZE = 1024 * 1024  # 1MB per chunk
MAX_AUDIO_RATE = 48000  # Maximum sample rate


def validate_audio_data(audio_data: bytes, expected_sample_rate: int = 16000) -> bool:
    """Validate incoming audio data.
    
    Args:
        audio_data: Raw audio data
        expected_sample_rate: Expected sample rate
        
    Returns:
        True if data is valid
        
    Raises:
        HTTPException: If data is invalid
    """
    if len(audio_data) > MAX_AUDIO_CHUNK_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Audio chunk too large. Maximum size: {MAX_AUDIO_CHUNK_SIZE} bytes"
        )
    
    if expected_sample_rate > MAX_AUDIO_RATE:
        raise HTTPException(
            status_code=400,
            detail=f"Sample rate too high. Maximum: {MAX_AUDIO_RATE} Hz"
        )
    
    # Validate that data length makes sense for float32 audio
    if len(audio_data) % 4 != 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid audio data: length must be multiple of 4 for float32 samples"
        )
    
    return True
