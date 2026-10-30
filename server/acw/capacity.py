"""The host's capacity service: one program that hands out memory to everything on this machine.

It runs next to us (Host-Ops `host_capacity.py`, shared with the Minecraft servers), so a file job and a Minecraft server start
can't both count on the same free memory. Audio Cleaner goes first: when a job is refused for memory the service holds back new
Minecraft players and server starts until the job fits. Reached over the abstract unix socket `@host-capacity` (the workers'
sandbox allows unix sockets only). When it doesn't answer, admission falls back to its own memory reserve
(job_resources.MEM_RESERVE), so the service is never needed for jobs to run.
"""
from __future__ import annotations

import json
import os
import socket

SOCKET = "\0" + os.environ.get("ACW_CAPACITY_SOCKET", "host-capacity")
TIMEOUT_S = 1.0


def _post(path: str, body: dict) -> dict | None:
    data = json.dumps(body).encode()
    request = (f"POST {path} HTTP/1.0\r\nHost: localhost\r\nContent-Type: application/json\r\n"
               f"Content-Length: {len(data)}\r\n\r\n").encode() + data
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(TIMEOUT_S)
            s.connect(SOCKET)
            s.sendall(request)
            reply = b""
            while chunk := s.recv(65536):
                reply += chunk
    except OSError:
        return None
    head, _, payload = reply.partition(b"\r\n\r\n")
    if not head.startswith(b"HTTP/") or b" 200 " not in head.split(b"\r\n", 1)[0] + b" ":
        return None
    try:
        out = json.loads(payload)
    except ValueError:
        return None
    return out if isinstance(out, dict) else None


def reserve_job(job_id: str, memory: int) -> tuple[bool, str] | None:
    """Ask for `memory` bytes for job `job_id` (held for a minute while it allocates). (granted, why not), or None when the
    service didn't answer. Asking again for the same job returns the same reservation."""
    reply = _post("/reserve", {"kind": "acw_job", "name": str(job_id), "bytes": int(memory)})
    if reply is None or "granted" not in reply:
        return None
    return bool(reply["granted"]), str(reply.get("reason") or "")
