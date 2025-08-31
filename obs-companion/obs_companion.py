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

API_BASE = os.getenv("API_BASE", "http://localhost:7071/api")
# SECURITY: Use user-specific API key from environment or config (frontend provides it)
USER_API_KEY = os.getenv("USER_API_KEY", "12345")

# Default to 500ms chunks for better real-time performance
# You can override via --chunk-ms if you want different latency.
CHUNK_MS = int(os.getenv("CHUNK_MS", "500"))  # ms

def create_session(languages):
    url = f"{API_BASE}/create-stream-session"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": USER_API_KEY,  # Use user-specific key
    }
    resp = requests.post(url, headers=headers, json={"languagesRequested": languages}, timeout=10)
    resp.raise_for_status()
    return resp.json()

async def stream_audio(
    session_id: str,
    ws_url: str,
    token: str,
    sr: int,
    languages,
    device=None,
    chunk_ms: int = CHUNK_MS,
    duration_sec: float = 5.0,
    playback: bool = True,
):
    # Convert relative wsUrl to absolute ws
    if ws_url.startswith("/"):
        # Dev: map /stream/* to local processor service at 127.0.0.1:8000
        host = os.getenv("STREAM_WS_HOST", "ws://127.0.0.1:8000")
        ws_full = f"{host}{ws_url}"
    else:
        ws_full = ws_url

    print(f"Connecting WS: {ws_full}")
    # SIMPLIFIED: Only send session token, API key already validated during session creation
    headers = {"x-session-token": token} if not ws_url.startswith("ws://127.0.0.1") else {}
    async with websockets.connect(ws_full, additional_headers=headers) as ws:
        # Send init
        init = {"type": "init", "sr": sr, "languages": languages}
        await ws.send(json.dumps(init))
        ready = await ws.recv()
        print("Server:", ready)

        # Setup audio stream (mono float32)
        channels = 1  # enforce mono to avoid mixing work in callback
        blocksize = max(1, int(sr * (chunk_ms / 1000.0)))
        frame_q = asyncio.Queue(maxsize=128)  # larger queue to reduce drops under load
        capturing_done = asyncio.Event()

        # Buffers to enable A/B playback after capture window
        orig_chunks: list[np.ndarray] = []  # original mic chunks (float32 mono)
        proc_chunks: list[np.ndarray] = []  # processed chunks (float32 mono)

        # Simple client-side metrics (capture->send latency, jitter, drops)
        dropped_frames = 0
        max_q_depth = 0
        frames_sent = 0
        bytes_sent = 0
        last_send_ts = None
        lat_samples = []  # capture->send (ms)
        interval_samples = []  # inter-send spacing (ms)
        stats_last_print = time.perf_counter()

        def audio_callback(indata, frames, time_info, status):
            nonlocal dropped_frames
            # Minimize work in callback: assume mono due to channels=1
            # If device still delivers stereo, quickly downmix.
            x = indata
            if x.ndim == 2 and x.shape[1] > 1:
                x = x[:, 0:1].mean(axis=1)
            else:
                x = x.reshape(-1)
            x = x.astype(np.float32, copy=False)
            # Keep a copy for original playback buffer
            try:
                orig_chunks.append(x.copy())
            except Exception:
                pass
            cap_ts = time.perf_counter()
            try:
                frame_q.put_nowait((cap_ts, x.tobytes()))
            except asyncio.QueueFull:
                # Drop oldest to keep latency bound
                try:
                    _ = frame_q.get_nowait()
                except Exception:
                    pass
                try:
                    frame_q.put_nowait((cap_ts, x.tobytes()))
                except Exception:
                    pass
                dropped_frames += 1

        async def sender():
            nonlocal max_q_depth, frames_sent, bytes_sent, last_send_ts, stats_last_print
            while True:
                # Exit when capture finished and queue drained
                if capturing_done.is_set() and frame_q.empty():
                    break
                cap_ts, chunk = await frame_q.get()
                now = time.perf_counter()
                # Metrics
                lat_ms = (now - cap_ts) * 1000.0
                lat_samples.append(lat_ms)
                if last_send_ts is not None:
                    interval_ms = (now - last_send_ts) * 1000.0
                    interval_samples.append(interval_ms)
                last_send_ts = now
                # Track queue depth
                try:
                    qd = frame_q.qsize()
                    if qd > max_q_depth:
                        max_q_depth = qd
                except Exception:
                    pass

                await ws.send(chunk)
                frames_sent += 1
                bytes_sent += len(chunk)

                # Show chunk sending progress
                chunk_duration = len(chunk) / (4 * sr)  # 4 bytes per float32 sample
                print(f"\rSent chunk {frames_sent}: {chunk_duration:.2f}s", end="", flush=True)

                # Periodic stats print (every ~5s)
                if (now - stats_last_print) >= 5.0:
                    avg_lat = (sum(lat_samples) / len(lat_samples)) if lat_samples else 0.0
                    p95_lat = sorted(lat_samples)[int(0.95 * len(lat_samples))] if len(lat_samples) >= 2 else avg_lat
                    avg_int = (sum(interval_samples) / len(interval_samples)) if interval_samples else 0.0
                    kbps = (bytes_sent * 8) / (now - stats_last_print) / 1000.0
                    print(
                        f"\n[stats] sent={frames_sent} drop={dropped_frames} q_max={max_q_depth} "
                        f"lat_ms(avg/p95)={avg_lat:.1f}/{p95_lat:.1f} send_int_ms~{avg_int:.1f} bitrate~{kbps:.1f} kbps"
                    )
                    # reset windowed stats
                    lat_samples.clear()
                    interval_samples.clear()
                    bytes_sent = 0
                    stats_last_print = now

        async def receiver():
            while True:
                try:
                    msg = await ws.recv()
                    if isinstance(msg, bytes):
                        # processed audio (bytes). Buffer for later playback (float32 mono)
                        try:
                            y = np.frombuffer(msg, dtype=np.float32).copy()
                            proc_chunks.append(y)
                        except Exception:
                            pass
                    else:
                        # text messages ignored for A/B
                        pass
                except Exception:
                    break

        # Explicit start/stop so we can drain after capture
        stream = sd.InputStream(
            samplerate=sr,
            channels=channels,
            dtype='float32',
            blocksize=blocksize,
            device=device,
            callback=audio_callback,
        )
        print("Streaming mic… Press Ctrl+C to stop.")
        stream.start()
        send_task = asyncio.create_task(sender())
        recv_task = asyncio.create_task(receiver())
        stop_task = asyncio.create_task(asyncio.sleep(duration_sec))
        try:
            await stop_task
            print(f"Time window ({duration_sec:.1f}s) reached, stopping capture…")
        except KeyboardInterrupt:
            print("\nStopping…")
        finally:
            # Stop capturing and let sender drain remaining frames
            capturing_done.set()
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

            # Drain remaining capture frames (up to a timeout)
            drain_start = time.perf_counter()
            while not frame_q.empty() and (time.perf_counter() - drain_start) < 2.0:
                await asyncio.sleep(0.02)

            # Allow receiver to collect final processed chunks
            await asyncio.sleep(0.3)

            # Close WS only after draining
            try:
                await ws.close()
            except Exception:
                pass

            # Stop tasks
            for t in (send_task, recv_task):
                if not t.done():
                    t.cancel()
            await asyncio.gather(send_task, recv_task, return_exceptions=True)

        print("\nWebSocket closed.")
        
        # Allow a brief moment for any final chunks to arrive
        await asyncio.sleep(0.1)

        # Playback phase (A/B test)
        try:
            if playback:
                print("\n" + "="*50)
                print("A/B PLAYBACK COMPARISON")
                print("="*50)
                
                if orig_chunks:
                    orig = np.concatenate(orig_chunks)
                    print(f"Playing ORIGINAL ({orig.shape[0]/sr:.2f}s)…")
                    sd.play(orig, sr)
                    sd.wait()
                    print("Original playback complete.")
                else:
                    print("No original audio captured.")

                print("\nStarting processed audio playback in 2 seconds...")
                await asyncio.sleep(2)  # Brief pause between playbacks

                if proc_chunks:
                    proc = np.concatenate(proc_chunks)
                    # Avoid NaNs/Infs just in case
                    proc = np.nan_to_num(proc, nan=0.0, posinf=0.0, neginf=0.0)
                    print(f"Playing PROCESSED ({proc.shape[0]/sr:.2f}s)…")
                    sd.play(proc, sr)
                    sd.wait()
                    print("Processed playback complete.")
                    
                    # Show comparison stats
                    orig_duration = orig.shape[0]/sr if orig_chunks else 0
                    proc_duration = proc.shape[0]/sr
                    print(f"\nComparison Summary:")
                    print(f"  Original duration: {orig_duration:.2f}s")
                    print(f"  Processed duration: {proc_duration:.2f}s")
                    print(f"  Chunk count: {len(proc_chunks)} processed chunks received")
                else:
                    print("No processed audio received from server.")
        except Exception as e:
            print(f"Playback error: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    # Default 16k to preserve current server expectations; override via DEFAULT_SR or --sr if needed
    parser.add_argument("--sr", type=int, default=int(os.getenv("DEFAULT_SR", "16000")))
    parser.add_argument("--lang", action='append', default=['en'])
    parser.add_argument("--device", default=None)
    parser.add_argument("--chunk-ms", type=int, default=CHUNK_MS, help="Capture chunk size in ms (default 1000)")
    parser.add_argument("--duration", type=float, default=float(os.getenv("DURATION_SEC", "5")), help="How long to stream before playback (seconds)")
    parser.add_argument("--no-playback", action="store_true", help="Skip playback phase")
    parser.add_argument("--bypass-api", action="store_true", help="Skip API call and connect directly")
    args = parser.parse_args()

    if args.bypass_api:
        # Direct connection bypass for quick testing
        session_id = "test-session-123"
        ws_url = "ws://127.0.0.1:8000/stream/test-session-123"
        token = "fake-token"
        print("BYPASSING API - Direct connect:", session_id, ws_url)
    else:
        if not USER_API_KEY:
            print("Missing USER_API_KEY env var - please set your personal API key from the frontend", file=sys.stderr)
            sys.exit(1)

        data = create_session(args.lang)
        session_id = data.get("sessionId")
        ws_url = data.get("wsUrl")
        token = data.get("token")  # Get session token from response
        print("Session:", session_id, ws_url)
    
    asyncio.run(
        stream_audio(
            session_id,
            ws_url,
            token,
            args.sr,
            args.lang,
            device=args.device,
            chunk_ms=args.chunk_ms,
            duration_sec=args.duration,
            playback=(not args.no_playback),
        )
    )

if __name__ == "__main__":
    main()
