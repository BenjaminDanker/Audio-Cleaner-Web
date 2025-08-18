#!/usr/bin/env python3
"""
Audio Cleaner Streaming Service

FastAPI WebSocket service for real-time audio processing with:
- DFNet denoising and DSP clarity pipeline
- Real-time ASR with Azure AI Speech Services
- Multi-language translation
- Credit-based billing system
- Production-grade security with authentication, rate limiting, and input validation
"""
import os
import sys
import logging
from datetime import datetime, timezone
from typing import Optional
from fastapi import FastAPI, WebSocket, Request, Depends, HTTPException, Header
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

# Add shared directory to Python path
CUR_DIR = os.path.dirname(__file__)
SHARED_DIR = os.path.abspath(os.path.join(CUR_DIR, "..", "shared"))
if SHARED_DIR not in sys.path:
    sys.path.append(SHARED_DIR)

from session import session_manager
from websocket_handler import handle_websocket_connection
from security import (
    StopStreamRequest, 
    limiter, 
    security, 
    extract_token_from_request,
    verify_session_access,
    validate_session_id,
    log_security_event,
    setup_rate_limiting_handler,
    ALLOWED_ORIGINS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Audio Cleaner Streaming Service", 
    version="0.4.0",
    description="Secure real-time audio processing with authentication and rate limiting"
)

# Setup security middleware
setup_rate_limiting_handler(app)

# CORS middleware with security-conscious defaults
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Restrictive - only needed methods
    allow_headers=[
        "Authorization", 
        "Content-Type", 
        "X-Session-Token",
        "Origin",
        "Accept",
        "User-Agent"
    ],
    max_age=600,  # Cache preflight for 10 minutes
)


@app.get("/health")
@limiter.limit("100/minute")  # Reasonable limit for health checks
async def health_check(request: Request):
    """Health check endpoint for container orchestration.
    
    Limited information disclosure - only essential health data.
    """
    return {
        "status": "healthy",
        "service": "audio-cleaner-streaming",
        "timestamp": datetime.now(timezone.utc).isoformat()
        # Version removed to reduce information disclosure
    }


@app.post("/stream/stop")
@limiter.limit("10/minute")  # Strict limit to prevent abuse
async def stop_stream(
    request: Request,
    stop_request: StopStreamRequest,
    authorization: HTTPAuthorizationCredentials = Depends(security),
    x_session_token: Optional[str] = Header(None)
):
    """Stop a streaming session and return statistics.
    
    Requires authentication with the same token used to start the session.
    """
    session_id = stop_request.sessionId
    
    # Extract and verify authentication token
    token = extract_token_from_request(authorization, x_session_token)
    token_payload = verify_session_access(token, session_id)
    
    # Log the stop request for security monitoring
    log_security_event(
        "stream_stop_request",
        request,
        session_id=session_id,
        user_id=token_payload.get("userId"),
        details={"token_valid": True}
    )
    
    # Remove session (this also closes WebSocket if active)
    st = session_manager.remove_session(session_id)
    
    if st and st.ws:
        try:
            await st.ws.close()
        except Exception as e:
            logger.warning(f"Error closing WebSocket for session {session_id}: {e}")
            
    return JSONResponse({
        "sessionId": session_id,
        "processedSeconds": round(st.processed_seconds, 2) if st else 0.0,
        "subtitles": {},  # No transcript persistence by design
        "status": "stopped"
    })


@app.websocket("/stream/{session_id}")
async def stream_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for streaming audio processing.
    
    Requires valid session token for authentication.
    Includes connection limiting and origin validation.
    """
    # Validate session ID format
    try:
        session_id = validate_session_id(session_id)
    except HTTPException:
        await websocket.close(code=4400)  # Bad Request
        return
    
    # Handle the secure WebSocket connection
    await handle_websocket_connection(websocket, session_id)


# Development/standalone mode check
if os.getenv("PROCESSOR_MODE", "stream").lower() not in ("stream", "both"):
    import logging
    logging.warning("processor started with non-stream mode; this container is for streaming only")


if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    host = os.getenv("UVICORN_HOST", "0.0.0.0")
    port = int(os.getenv("UVICORN_PORT", "8000"))
    log_level = os.getenv("UVICORN_LOG_LEVEL", "info")
    
    # Validate required environment variables
    required_vars = [
        "STREAM_SESSION_SIGNING_KEY",
        "AZURE_OPENAI_ENDPOINT", 
        "AZURE_OPENAI_API_KEY",
        "COSMOS_CONNECTION_STRING"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)
    
    print(f"Starting Audio Cleaner Streaming Service on {host}:{port}")
    print(f"Log level: {log_level}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=True
    )
