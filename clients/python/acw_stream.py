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
    if args.codec == "opus":
        import av

        args.frame_ms = 20
        enc = av.CodecContext.create("libopus", "w")
        enc.sample_rate, enc.layout, enc.format, enc.bit_rate = SR, "mono", "s16", args.bitrate
        enc.options = {"application": "audio", "frame_duration": "20"}
        enc.open()
        dec = av.CodecContext.create("opus", "r")
        dec.sample_rate, dec.layout = SR, "mono"
    frame = SR * args.frame_ms // 1000
    out = np.zeros(len(x) + frame, dtype=np.float32)
    rtts: list[float] = []
    bytes_up = [0]
    sent_at: dict[int, float] = {}
    async with connect(args.url, additional_headers={"Authorization": f"Bearer {key}"}, max_size=2**20) as ws:
        await ws.send(json.dumps({
            "type": "start", "version": 1, "denoise": args.denoise or None, "captions": args.captions or None,
            "language": args.language, "sampleRate": SR, "client": "acw-reference/0.2",
            "codec": args.codec, "opusBitrate": args.bitrate,
        }))
        ready = json.loads(await ws.recv())
        while ready.get("type") == "queued":  # server busy: wait in line, it starts by itself
            print(f"waiting in line, position {ready.get('position')}")
            ready = json.loads(await ws.recv())
        print("server:", ready)
        if ready.get("type") != "ready":
            return 1

        async def receiver():
            async for msg in ws:
                if isinstance(msg, bytes):
                    _v, codec, _f, seq, ts = HEADER.unpack_from(msg)
                    if codec == 3:
                        frames = dec.decode(av.Packet(msg[HEADER.size:]))
                        y = np.concatenate([f.to_ndarray().reshape(-1) for f in frames]) if frames else np.zeros(frame, np.float32)
                        if y.dtype != np.float32:
                            y = y.astype(np.float32) / 32768
                    else:
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
            pcm16 = (np.clip(x[i:i + frame], -1, 1) * 32767).astype("<i2")
            if args.codec == "opus":
                f = av.AudioFrame.from_ndarray(pcm16.reshape(1, -1), format="s16", layout="mono")
                f.sample_rate, f.pts = SR, i
                payload, codec_id = b"".join(bytes(p) for p in enc.encode(f)), 3
                bytes_up[0] += len(payload)
            else:
                payload, codec_id = pcm16.tobytes(), 1
                bytes_up[0] += len(payload)
            sent_at[seq] = time.perf_counter()
            await ws.send(HEADER.pack(1, codec_id, 0, seq, int(time.time() * 1e6)) + payload)
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
        print(f"frames {len(r)}  round trip ms: median {np.median(r):.1f}  p95 {np.percentile(r, 95):.1f}  max {r.max():.1f}"
              f"  uplink {bytes_up[0] * 8 / (len(x) / SR) / 1000:.0f} kbps")
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
    ap.add_argument("--codec", choices=["pcm_s16le", "opus"], default="pcm_s16le")
    ap.add_argument("--bitrate", type=int, default=64000, help="Opus bitrate (uplink; also requested for the downlink)")
    ap.add_argument("--tail", type=float, default=1.5, help="seconds to keep listening before sending stop")
    sys.exit(asyncio.run(run(ap.parse_args())))


if __name__ == "__main__":
    main()
