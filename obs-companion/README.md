# OBS Companion (Audio Cleaner)

Purpose: Start/stop a streaming session using an API key and forward microphone audio to the streaming service over WebSocket. This is a minimal Python reference you can port to a native OBS plugin.

Features:
- API key-only auth
- Creates a session via Azure Function API
- Opens WS to `/stream/{sessionId}` and sends float32 mono PCM chunks
- Receives processed audio and optional caption deltas

Prereqs:
- Python 3.10+
- `pip install -r requirements.txt`
- Set env vars:
  - `API_BASE` e.g. `http://localhost:4280/api` (when running SWA CLI) or your deployed API base
  - `STREAMING_API_KEY` your provisioned key
  - `AUDIO_DEVICE` optional input device name or index

Quick start:

```powershell
# Windows PowerShell
$env:API_BASE = "http://localhost:4280/api"
$env:STREAMING_API_KEY = "<your_key>"
python obs_companion.py --sr 16000 --lang en --device default
```

Porting to OBS:
- Wrap `start_stream()` and `stop_stream()` inside OBS plugin UI callbacks
- Use OBS audio callback to get float32 mono chunks (downmix if needed)
- Send chunks sized ~320ms (5120 samples @ 16kHz) for ~2s buffer on server
- Feed returned processed audio into a new audio source or replace mic
