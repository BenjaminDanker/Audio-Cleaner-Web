"""Live-stream wire formats.

Client <-> web (WebSocket, docs/streaming-protocol.md):
  text   JSON control messages (start, ready, caption, balance, stop, ...)
  binary 16-byte header + audio payload
         u8  version (1)
         u8  codec   (1 = pcm_s16le, 2 = f32le)
         u16 flags   (reserved, 0)
         u32 seq     (client frame counter, echoed back)
         u64 ts_us   (client timestamp, echoed back for RTT measurement)
  Audio is 48 kHz mono and each frame is a whole number of 10 ms hops
  (480 samples); responses carry the same number of samples per frame.

Web <-> stream worker (Unix socket): u8 type + u32 length + payload.
"""
from __future__ import annotations

import asyncio
import json
import struct

import numpy as np

VERSION = 1
CODEC_S16 = 1
CODEC_F32 = 2
CODECS = {"pcm_s16le": CODEC_S16, "f32le": CODEC_F32}
HEADER = struct.Struct("<BBHIQ")
SAMPLE_RATE = 48_000
HOP = 480
MAX_FRAME_SAMPLES = HOP * 10  # 100 ms


class ProtocolError(ValueError):
    pass


def decode_frame(data: bytes) -> tuple[int, int, int, np.ndarray]:
    if len(data) < HEADER.size:
        raise ProtocolError("frame too short")
    version, codec, _flags, seq, ts = HEADER.unpack_from(data)
    if version != VERSION:
        raise ProtocolError("unsupported frame version")
    payload = memoryview(data)[HEADER.size:]
    if codec == CODEC_S16:
        if len(payload) % 2:
            raise ProtocolError("odd payload")
        x = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    elif codec == CODEC_F32:
        if len(payload) % 4:
            raise ProtocolError("bad payload")
        x = np.frombuffer(payload, dtype="<f4").astype(np.float32)
        if not np.all(np.isfinite(x)):
            raise ProtocolError("non-finite samples")
        np.clip(x, -4.0, 4.0, out=x)
    else:
        raise ProtocolError("unknown codec")
    n = len(x)
    if n == 0 or n % HOP or n > MAX_FRAME_SAMPLES:
        raise ProtocolError("frames must be 10-100 ms, in 10 ms steps, at 48 kHz")
    return codec, seq, ts, x


def encode_frame(codec: int, seq: int, ts: int, y: np.ndarray) -> bytes:
    if codec == CODEC_S16:
        payload = (np.clip(y, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    else:
        payload = y.astype("<f4").tobytes()
    return HEADER.pack(VERSION, codec, 0, seq & 0xFFFFFFFF, ts & 0xFFFFFFFFFFFFFFFF) + payload


# ---- IPC between web and stream worker

MSG_START = b"S"
MSG_AUDIO = b"A"
MSG_CONFIG = b"C"
MSG_EVENT = b"T"
MSG_END = b"E"
_IPC = struct.Struct("<cI")
MAX_IPC = 4 * 1024 * 1024


async def send_msg(writer: asyncio.StreamWriter, kind: bytes, payload: bytes) -> None:
    writer.write(_IPC.pack(kind, len(payload)) + payload)
    await writer.drain()


async def read_msg(reader: asyncio.StreamReader) -> tuple[bytes, bytes]:
    head = await reader.readexactly(_IPC.size)
    kind, n = _IPC.unpack(head)
    if n > MAX_IPC:
        raise ProtocolError("ipc message too large")
    return kind, await reader.readexactly(n)


def pack_audio(seq: int, x: np.ndarray) -> bytes:
    return struct.pack("<I", seq) + x.astype("<f4").tobytes()


def unpack_audio(payload: bytes) -> tuple[int, np.ndarray]:
    (seq,) = struct.unpack_from("<I", payload)
    return seq, np.frombuffer(payload, dtype="<f4", offset=4).copy()


def pack_json(obj: dict) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode()
