#!/usr/bin/env python3
"""
Streaming service entrypoint
Runs the FastAPI WebSocket service for real-time audio processing
"""
import os
import sys

# Add shared directory to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = os.path.join(CURRENT_DIR, "..", "shared")
sys.path.insert(0, SHARED_DIR)

if __name__ == "__main__":
    import uvicorn
    
    # Import the FastAPI app
    from app import app
    
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
