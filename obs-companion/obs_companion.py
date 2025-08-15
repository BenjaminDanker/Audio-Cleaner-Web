import asyncio
import os
import sys
import json
import time
import argparse
import requests
import numpy as np
import sounddevice as sd
import websockets

API_BASE = os.getenv("API_BASE", "http://localhost:4280/api")
STREAMING_API_KEY = os.getenv("STREAMING_API_KEY", "")

CHUNK_MS = 320  # ~0.32s chunks at 16k -> about 2s server window

def create_session(languages):
    url = f"{API_BASE}/create-stream-session"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": STREAMING_API_KEY,
    }
    resp = requests.post(url, headers=headers, json={"languagesRequested": languages}, timeout=10)
    resp.raise_for_status()
    return resp.json()

async def stream_audio(session_id: str, ws_url: str, sr: int, languages, device=None):
    # Convert relative wsUrl to absolute ws
    if ws_url.startswith("/"):
        # Dev: map /stream/* to local processor service at 127.0.0.1:8000
        host = os.getenv("STREAM_WS_HOST", "ws://127.0.0.1:8000")
        ws_full = f"{host}{ws_url}"
    else:
        ws_full = ws_url

    print(f"Connecting WS: {ws_full}")
    async with websockets.connect(ws_full, extra_headers={"x-api-key": STREAMING_API_KEY}) as ws:
        # Send init
        init = {"type": "init", "sr": sr, "languages": languages}
        await ws.send(json.dumps(init))
        ready = await ws.recv()
        print("Server:", ready)

        # Setup audio stream (mono float32)
        channels = 1
        blocksize = int(sr * (CHUNK_MS / 1000.0))
        frame_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=32)

        def audio_callback(indata, frames, time_info, status):
            x = indata.astype(np.float32)
            if x.ndim == 2 and x.shape[1] > 1:
                x = x.mean(axis=1)
            try:
                frame_q.put_nowait(x.tobytes())
            except asyncio.QueueFull:
                # Drop oldest to keep latency bound
                try:
                    _ = frame_q.get_nowait()
                except Exception:
                    pass
                try:
                    frame_q.put_nowait(x.tobytes())
                except Exception:
                    pass

        async def sender():
            while True:
                chunk = await frame_q.get()
                await ws.send(chunk)

        async def receiver():
            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    # processed audio (bytes). In a full OBS plugin, route to an output buffer
                    pass
                else:
                    # text control/captions
                    # print(msg)
                    pass

        with sd.InputStream(samplerate=sr, channels=channels, dtype='float32', blocksize=blocksize, device=device, callback=audio_callback):
            print("Streaming mic… Press Ctrl+C to stop.")
            send_task = asyncio.create_task(sender())
            recv_task = asyncio.create_task(receiver())
            try:
                await asyncio.gather(send_task, recv_task)
            except KeyboardInterrupt:
                print("Stopping…")
            finally:
                send_task.cancel()
                recv_task.cancel()

    print("WebSocket closed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--lang", action='append', default=['en'])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not STREAMING_API_KEY:
        print("Missing STREAMING_API_KEY env var", file=sys.stderr)
        sys.exit(1)

    data = create_session(args.lang)
    session_id = data.get("sessionId")
    ws_url = data.get("wsUrl")
    print("Session:", session_id, ws_url)
    
    asyncio.run(stream_audio(session_id, ws_url, args.sr, args.lang, device=args.device))

if __name__ == "__main__":
    main()
