#!/usr/bin/env python3
"""Reference live-stream client for Audio Cleaner (protocol v1).

Streams a WAV file (or, with --mic, the default microphone via sounddevice)
in real time, writes the cleaned audio to a WAV file and prints captions.
The OBS plugin implements the same protocol; see docs/streaming-protocol.md.

  ACW_API_KEY=acw_... python acw_stream.py --url wss://host/audio/api/stream/ws \
      --in noisy.wav --out cleaned.wav --denoise light --captions local
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import struct
import sys
import time

import numpy as np

HEADER = struct.Struct("<BBHIQ")
SR = 48_000


async def run(args) -> int:
    import soundfile as sf
    from websockets.asyncio.client import connect

    key = os.environ.get("ACW_API_KEY")
    if not key:
        print("set ACW_API_KEY", file=sys.stderr)
        return 2
    x, sr = sf.read(args.input, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        from scipy.signal import resample_poly

        x = resample_poly(x, SR, sr).astype(np.float32)
    frame = SR * args.frame_ms // 1000
    out = np.zeros(len(x) + frame, dtype=np.float32)
    rtts: list[float] = []
    sent_at: dict[int, float] = {}
    async with connect(args.url, additional_headers={"Authorization": f"Bearer {key}"}, max_size=2**20) as ws:
        await ws.send(json.dumps({
            "type": "start", "version": 1, "denoise": args.denoise or None, "captions": args.captions or None,
            "language": args.language, "sampleRate": SR, "client": "acw-reference/0.1",
        }))
        ready = json.loads(await ws.recv())
        print("server:", ready)
        if ready.get("type") != "ready":
            return 1

        async def receiver():
            async for msg in ws:
                if isinstance(msg, bytes):
                    _v, _c, _f, seq, ts = HEADER.unpack_from(msg)
                    y = np.frombuffer(msg, dtype="<i2", offset=HEADER.size).astype(np.float32) / 32768
                    start = seq * frame
                    out[start:start + len(y)] = y[: max(0, len(out) - start)]
                    if seq in sent_at:
                        rtts.append((time.perf_counter() - sent_at.pop(seq)) * 1000)
                else:
                    ev = json.loads(msg)
                    if ev.get("type") == "caption":
                        print(("CAPTION " if ev.get("final") else "partial ") + ev["text"])
                    elif ev.get("type") != "pong":
                        print("event:", ev)

        rx = asyncio.create_task(receiver())
        t0 = time.perf_counter()
        for seq, i in enumerate(range(0, len(x) - frame + 1, frame)):
            # real-time pacing
            due = t0 + seq * args.frame_ms / 1000
            await asyncio.sleep(max(0, due - time.perf_counter()))
            pcm = (np.clip(x[i:i + frame], -1, 1) * 32767).astype("<i2").tobytes()
            sent_at[seq] = time.perf_counter()
            await ws.send(HEADER.pack(1, 1, 0, seq, int(time.time() * 1e6)) + pcm)
        await asyncio.sleep(args.tail)
        await ws.send(json.dumps({"type": "stop"}))
        try:
            await asyncio.wait_for(rx, timeout=10)
        except asyncio.TimeoutError:
            rx.cancel()
    if args.out:
        sf.write(args.out, out[: len(x)], SR)
    if rtts:
        r = np.array(rtts)
        print(f"frames {len(r)}  round trip ms: median {np.median(r):.1f}  p95 {np.percentile(r, 95):.1f}  max {r.max():.1f}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--in", dest="input", required=True)
    ap.add_argument("--out")
    ap.add_argument("--denoise", default="light")
    ap.add_argument("--captions", default="")
    ap.add_argument("--language")
    ap.add_argument("--frame-ms", type=int, default=20)
    ap.add_argument("--tail", type=float, default=1.5, help="seconds to keep listening before sending stop")
    sys.exit(asyncio.run(run(ap.parse_args())))


if __name__ == "__main__":
    main()
