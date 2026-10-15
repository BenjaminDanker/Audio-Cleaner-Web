"""Live-stream wire formats.

Client <-> web (WebSocket, docs/streaming-protocol.md):
  text   JSON control messages (start, ready, caption, balance, stop, ...)
  binary 16-byte header + audio payload
         u8  version (1)
         u8  codec   (1 = pcm_s16le, 2 = f32le, 3 = opus: one 20 ms packet)
         u16 flags   (bit 0: echo-reference frame, see below; others 0)
         u32 seq     (client frame counter, echoed back)
         u64 ts_us   (client timestamp, echoed back for RTT measurement)
  Audio is 48 kHz mono and each frame is a whole number of 10 ms hops
  (480 samples); responses carry the same number of samples per frame.
  Echo reference: with "echoReference" in the start message, a client may send
  what its speakers play as extra frames with flags bit 0 set, same codec and
  the same seq as the mic frame they belong to, sent before it. They get no
  reply and are not billed.

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
CODEC_OPUS = 3
CODECS = {"pcm_s16le": CODEC_S16, "f32le": CODEC_F32, "opus": CODEC_OPUS}
OPUS_FRAME = 960  # 20 ms; Opus frames are opaque until the sandboxed worker decodes them
MAX_OPUS_PACKET = 1275
HEADER = struct.Struct("<BBHIQ")
SAMPLE_RATE = 48_000
HOP = 480
MAX_FRAME_SAMPLES = HOP * 10  # 100 ms
FLAG_REFERENCE = 1


class ProtocolError(ValueError):
    pass


def frame_header(data: bytes) -> tuple[int, int, int, int]:
    """(codec, seq, ts_us, flags)"""
    if len(data) < HEADER.size:
        raise ProtocolError("frame too short")
    version, codec, flags, seq, ts = HEADER.unpack_from(data)
    if version != VERSION:
        raise ProtocolError("unsupported frame version")
    return codec, seq, ts, flags


def opus_payload(data: bytes) -> bytes:
    pkt = bytes(memoryview(data)[HEADER.size:])
    if not 1 <= len(pkt) <= MAX_OPUS_PACKET:
        raise ProtocolError("bad opus packet")
    return pkt


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
    if codec == CODEC_OPUS:
        payload = bytes(y)  # already an Opus packet
    elif codec == CODEC_S16:
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
MSG_REF = b"R"  # web -> worker: echo-reference audio for a mic frame (same payload as MSG_AUDIO)
MSG_TAP = b"P"  # worker -> web: the echo-free original (f32, before noise removal) for extra-accurate captions
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


# IPC audio: u32 seq + u8 kind + data; kind 0 = f32 samples, 3 = Opus packet.
IPC_F32 = 0


def pack_audio(seq: int, x: np.ndarray) -> bytes:
    return struct.pack("<IB", seq, IPC_F32) + x.astype("<f4").tobytes()


def pack_opus(seq: int, pkt: bytes) -> bytes:
    return struct.pack("<IB", seq, CODEC_OPUS) + pkt


def unpack_audio(payload: bytes) -> tuple[int, int, np.ndarray | bytes]:
    seq, kind = struct.unpack_from("<IB", payload)
    if kind == CODEC_OPUS:
        return seq, kind, bytes(payload[5:])
    return seq, IPC_F32, np.frombuffer(payload, dtype="<f4", offset=5).copy()


def pack_json(obj: dict) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode()
