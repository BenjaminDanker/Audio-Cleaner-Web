# Streaming Service (Scaffold)

A minimal FastAPI WebSocket service to support low-latency audio streaming.

- Health check: `GET /healthz`
- WebSocket endpoint: `ws://localhost:8000/stream/{sessionId}` (echo scaffold)

## Run locally

```pwsh
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Then from the frontend Streaming tab, start a session and update the WS URL to your local service if needed.
