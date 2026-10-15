"""Live stream endpoint: WebSocket /api/stream/ws (see docs/streaming-protocol.md).

Billing: a rolling hold covers the next few minutes; it is topped up as audio
flows and captured at the exact processed duration when the session ends. A
session stops (after a warning) when the balance cannot cover the next minute.

Pacing: audio must arrive at about real-time speed, otherwise live pricing
could be abused as a cheap batch service.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections import OrderedDict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .. import auth, cpusplit, db
from ..billing import ledger, pricing
from ..config import get_settings
from ..security import limiter
from . import protocol as P

log = logging.getLogger("acw.stream")
router = APIRouter(tags=["stream"])

MAX_SESSIONS_PER_USER = 2
MAX_SESSION_SECONDS = 12 * 3600
START_TIMEOUT = 15
IDLE_TIMEOUT = 30
PACE_BURST_SECONDS = 6.0
PACE_RATE = 1.25  # audio seconds accepted per wall-clock second
LOW_BALANCE_WARN_SECONDS = 300

_live: dict[str, int] = {}  # user_id -> active sessions in this process
_live_units = 0.0  # capacity units of running sessions (file jobs move off the cores they need)


async def startup() -> None:
    await asyncio.to_thread(cpusplit.units_per_core)  # calibrate once, off the event loop
    cpusplit.set_live_units(get_settings().data_dir, 0.0, cpusplit.units_per_core())
    # Sessions left "active" by a previous process were never captured.
    for row in db.all_rows("SELECT id, hold_id, seconds FROM stream_sessions WHERE status = 'active'"):
        if row["hold_id"]:
            try:
                ledger.release_hold(row["hold_id"])
            except Exception:  # noqa: BLE001
                log.exception("releasing stale stream hold")
        db.execute(
            "UPDATE stream_sessions SET status = 'ended', end_reason = 'server_restart', ended_at = ? WHERE id = ?",
            (db.now(), row["id"]),
        )


async def shutdown() -> None:
    return None


def _rate_micros_per_minute(denoise: str | None, captions: str | None) -> int:
    total = 0
    if denoise:
        total += pricing.rate("stream_denoise", denoise).micros_per_minute
    if captions:
        total += pricing.rate("stream_captions", captions).micros_per_minute
    return total


async def _send_json(ws: WebSocket, obj: dict) -> None:
    await ws.send_text(json.dumps(obj, separators=(",", ":")))


class Session:
    def __init__(self, ws: WebSocket, principal: auth.Principal, cfg: dict, rate: int):
        self.ws = ws
        self.p = principal
        self.cfg = cfg
        self.rate = rate  # micros per minute
        self.id = db.new_id()
        self.hold_id: str | None = None
        self.held_seconds = 0.0
        self.seconds = 0.0
        self.frames: OrderedDict[int, tuple[int, int]] = OrderedDict()
        self.pace_tokens = PACE_BURST_SECONDS
        self.pace_last = time.monotonic()
        self.warned = False
        self.next_extend_try = 0.0
        self.end_reason = "client_closed"
        self.reader = self.writer = None
        self.worker_task: asyncio.Task | None = None
        self.premium = None

    def micros_for(self, seconds: float) -> int:
        return math.ceil(self.rate * math.ceil(seconds) / 60)

    def open_hold(self) -> None:
        minutes = pricing.STREAM_HOLD_MINUTES
        self.hold_id = ledger.create_hold(self.p.user_id, self.rate * minutes, "stream", self.id)
        self.held_seconds = minutes * 60.0

    def extend_hold(self) -> bool:
        try:
            ledger.extend_hold(self.hold_id, self.rate)  # one more minute
        except (ledger.InsufficientFunds, ledger.LedgerError):
            return False
        self.held_seconds += 60.0
        return True

    def paced(self, seconds: float) -> bool:
        now = time.monotonic()
        self.pace_tokens = min(PACE_BURST_SECONDS, self.pace_tokens + (now - self.pace_last) * PACE_RATE)
        self.pace_last = now
        if self.pace_tokens < seconds:
            return False
        self.pace_tokens -= seconds
        return True


@router.websocket("/api/stream/ws")
async def stream_ws(ws: WebSocket) -> None:
    principal = auth.websocket_principal(ws)
    if principal is None or principal.status != "active":
        await ws.close(code=1008)
        return
    if "*" not in principal.scopes and "stream" not in principal.scopes:
        await ws.close(code=1008)
        return
    ip = (ws.headers.get("x-forwarded-for") or (ws.client.host if ws.client else "")).split(",")[-1].strip()
    if not limiter.hit(f"stream_start:{principal.user_id}", 20, 20 / 3600) or not limiter.hit(f"stream_start_ip:{ip}", 30, 30 / 3600):
        await ws.close(code=1008)
        return
    await ws.accept()
    if _live.get(principal.user_id, 0) >= MAX_SESSIONS_PER_USER:
        await _send_json(ws, {"type": "error", "code": "too_many_sessions", "message": "Two live sessions at most per account."})
        await ws.close(code=1008)
        return
    _live[principal.user_id] = _live.get(principal.user_id, 0) + 1
    sess: Session | None = None
    global _live_units
    units = 0.0
    try:
        sess = await _start(ws, principal, ip)
        if sess is not None and sess.writer is not None:
            units = cpusplit.session_units(sess.cfg)
            if units:
                _live_units += units
                cpusplit.set_live_units(get_settings().data_dir, _live_units, cpusplit.units_per_core())
            await _run(sess)
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        log.exception("stream session failed")
        if sess:
            sess.end_reason = "server_error"
    finally:
        if units:
            _live_units = max(0.0, _live_units - units)
            cpusplit.set_live_units(get_settings().data_dir, _live_units, cpusplit.units_per_core())
        _live[principal.user_id] -= 1
        if sess is not None:
            await _finish(sess)


def _drop_worker(sess: "Session") -> None:
    """Close the worker connection of a session that won't run."""
    if sess.writer is not None:
        sess.writer.close()
        sess.writer = None


_OPTION_TEXT = {
    ("captions", "premium"): "extra-accurate captions",
}


def _full_message(cfg: dict, options: list[dict]) -> str:
    wanted = []
    if cfg.get("denoise"):
        wanted.append("live noise removal")
    if cfg.get("captions") == "local":
        wanted.append("standard captions")
    msg = f"{' and '.join(wanted).capitalize() or 'Live'} {'is' if len(wanted) < 2 else 'are'} full right now."
    still = [_OPTION_TEXT[k] for o in options for k in o.items() if k in _OPTION_TEXT]
    if still:
        msg += " Still available: " + ", ".join(dict.fromkeys(still)) + "."
    return msg + " Try again in a few minutes."


async def _start(ws: WebSocket, principal: auth.Principal, ip: str) -> Session | None:
    try:
        raw = await asyncio.wait_for(ws.receive_text(), timeout=START_TIMEOUT)
        cfg = json.loads(raw)
    except (asyncio.TimeoutError, ValueError, KeyError):
        await _send_json(ws, {"type": "error", "code": "bad_start", "message": "Send a start message first."})
        await ws.close(code=1008)
        return None
    if not isinstance(cfg, dict) or cfg.get("type") != "start" or len(raw) > 4096:
        await ws.close(code=1008)
        return None
    user = db.one("SELECT tos_version FROM users WHERE id = ?", (principal.user_id,))
    if user["tos_version"] != auth.TOS_VERSION:
        await _send_json(ws, {"type": "error", "code": "tos", "message": "Accept the terms of service on the website first."})
        await ws.close(code=1008)
        return None
    denoise = cfg.get("denoise")
    captions = cfg.get("captions")
    language = cfg.get("language")
    if (language is not None and not (isinstance(language, str) and 2 <= len(language) <= 3 and language.isalpha())) or (
        cfg.get("sampleRate", P.SAMPLE_RATE) != P.SAMPLE_RATE
    ):
        await _send_json(ws, {"type": "error", "code": "bad_config", "message": "48 kHz audio and a 2-3 letter language code."})
        await ws.close(code=1008)
        return None
    if captions == "local" and language not in (None, "en"):
        await _send_json(ws, {"type": "error", "code": "bad_config",
                              "message": "Standard captions are English only. Extra-accurate captions work in other languages."})
        await ws.close(code=1008)
        return None
    try:
        rate = _rate_micros_per_minute(denoise, captions)
    except pricing.PricingError as e:
        await _send_json(ws, {"type": "error", "code": "bad_tier", "message": str(e)})
        await ws.close(code=1008)
        return None
    if rate == 0:
        await _send_json(ws, {"type": "error", "code": "nothing_enabled", "message": "Enable denoise and/or captions."})
        await ws.close(code=1008)
        return None
    if captions == "premium" and not get_settings().openai_api_key:
        await _send_json(ws, {"type": "error", "code": "unavailable", "message": "Premium captions are not available right now."})
        await ws.close(code=1008)
        return None

    sess = Session(ws, principal, {"denoise": denoise, "captions": captions, "language": language,
                                   "atten_lim_db": float(cfg.get("attenLimDb", 100.0))}, rate)
    try:
        sess.open_hold()
    except ledger.InsufficientFunds as e:
        await _send_json(ws, {"type": "error", "code": "insufficient_funds",
                              "message": f"Live processing reserves {pricing.STREAM_HOLD_MINUTES} minutes up front "
                                         f"(${rate * pricing.STREAM_HOLD_MINUTES / 1e6:.2f}); available ${e.available / 1e6:.2f}."})
        await ws.close(code=1008)
        return None
    except ledger.LedgerError:
        await ws.close(code=1008)
        return None
    db.execute(
        "INSERT INTO stream_sessions(id, user_id, api_key_id, denoise_tier, caption_tier, status, hold_id, client_ip, started_at)"
        " VALUES (?,?,?,?,?, 'active', ?,?,?)",
        (sess.id, principal.user_id, principal.api_key_id, denoise, captions, sess.hold_id, ip[:64], db.now()),
    )
    try:
        sess.reader, sess.writer = await asyncio.open_unix_connection(str(get_settings().stream_worker_socket))
        worker_cfg = dict(sess.cfg)
        if captions != "local":
            worker_cfg["captions"] = None
        worker_cfg["tap_pcm"] = captions == "premium"
        worker_cfg["opus_bitrate"] = int(cfg.get("opusBitrate", 96_000)) if str(cfg.get("opusBitrate", "")).isdigit() else 96_000
        await P.send_msg(sess.writer, P.MSG_START, P.pack_json(worker_cfg))
        kind, payload = await asyncio.wait_for(P.read_msg(sess.reader), timeout=20)
        ready = json.loads(payload)
        if kind == P.MSG_EVENT and ready.get("code") == "full":
            sess.end_reason = "full"
            _drop_worker(sess)
            options = ready.get("available") or []
            await _send_json(ws, {"type": "error", "code": "full", "available": options,
                                  "message": _full_message(worker_cfg, options)})
            await ws.close(code=1013)
            return sess
        if kind != P.MSG_EVENT or ready.get("type") != "ready":
            raise RuntimeError(f"worker refused: {ready}")
    except Exception:  # noqa: BLE001
        log.exception("stream worker unavailable")
        sess.end_reason = "worker_unavailable"
        _drop_worker(sess)
        await _send_json(ws, {"type": "error", "code": "unavailable", "message": "Live processing is busy or offline, try again soon."})
        await ws.close(code=1013)
        return sess
    if captions == "premium":
        from .openai_live import PremiumCaptions

        sess.premium = PremiumCaptions(
            language, lambda ev: asyncio.ensure_future(_send_json(ws, ev)), user_hint=principal.user_id
        )
        await sess.premium.connect()
    await _send_json(ws, {
        "type": "ready",
        "sessionId": sess.id,
        "sampleRate": P.SAMPLE_RATE,
        "hopSamples": P.HOP,
        "modelLatencyMs": round(ready.get("latencySamples", 0) * 1000 / P.SAMPLE_RATE, 1),
        "usdPerMinute": rate / 1e6,
        "maxSessionSeconds": MAX_SESSION_SECONDS,
    })
    return sess


async def _run(sess: Session) -> None:
    ws = sess.ws
    if sess.writer is None:
        return

    async def from_worker() -> None:
        while True:
            kind, payload = await P.read_msg(sess.reader)
            if kind == P.MSG_AUDIO:
                seq, akind, y = P.unpack_audio(payload)
                meta = sess.frames.pop(seq, None)
                if meta is None:
                    continue
                codec, ts = meta
                await ws.send_bytes(P.encode_frame(codec, seq, ts, y))
                if sess.premium is not None and akind == P.IPC_F32:
                    await sess.premium.feed(y)
            elif kind == P.MSG_TAP:
                if sess.premium is not None:
                    await sess.premium.feed(P.unpack_audio(payload)[2])
            elif kind == P.MSG_EVENT:
                try:
                    await ws.send_text(payload.decode())
                except Exception:  # noqa: BLE001 - client gone; keep draining
                    pass
            elif kind == P.MSG_END:
                return

    worker_task = asyncio.create_task(from_worker())
    sess.worker_task = worker_task
    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=IDLE_TIMEOUT)
            except asyncio.TimeoutError:
                sess.end_reason = "idle_timeout"
                break
            if msg["type"] == "websocket.disconnect":
                break
            if worker_task.done():
                sess.end_reason = "worker_lost"
                await _send_json(ws, {"type": "error", "code": "worker_lost", "message": "Processing stopped unexpectedly."})
                break
            if msg.get("bytes") is not None:
                try:
                    codec, seq, ts = P.frame_header(msg["bytes"])
                    if codec == P.CODEC_OPUS:
                        # Opaque here; decoded only inside the sandboxed worker.
                        pkt = P.opus_payload(msg["bytes"])
                        dur = P.OPUS_FRAME / P.SAMPLE_RATE
                        ipc = P.pack_opus(seq, pkt)
                    else:
                        codec, seq, ts, x = P.decode_frame(msg["bytes"])
                        dur = len(x) / P.SAMPLE_RATE
                        ipc = P.pack_audio(seq, x)
                except P.ProtocolError as e:
                    await _send_json(ws, {"type": "error", "code": "bad_frame", "message": str(e)})
                    sess.end_reason = "protocol_error"
                    break
                if not sess.paced(dur):
                    await _send_json(ws, {"type": "error", "code": "too_fast", "message": "Audio must be sent in real time."})
                    sess.end_reason = "too_fast"
                    break
                sess.seconds += dur
                if sess.seconds >= MAX_SESSION_SECONDS:
                    sess.end_reason = "max_duration"
                    await _send_json(ws, {"type": "stopped", "reason": "max_duration"})
                    break
                if not await _billing_tick(sess):
                    break
                sess.frames[seq] = (codec, ts)
                while len(sess.frames) > 500:
                    sess.frames.popitem(last=False)
                await P.send_msg(sess.writer, P.MSG_AUDIO, ipc)
            elif msg.get("text") is not None:
                if len(msg["text"]) > 2048:
                    continue
                try:
                    c = json.loads(msg["text"])
                except ValueError:
                    continue
                t = c.get("type") if isinstance(c, dict) else None
                if t == "ping":
                    await _send_json(ws, {"type": "pong", "t": c.get("t"), "seconds": round(sess.seconds, 2)})
                elif t == "stop":
                    sess.end_reason = "client_stop"
                    break
                elif t == "config" and "attenLimDb" in c:
                    await P.send_msg(sess.writer, P.MSG_CONFIG, P.pack_json({"atten_lim_db": float(c["attenLimDb"])}))
    finally:
        # Not cancelled here: _finish lets it drain the worker's last captions.
        pass


async def _billing_tick(sess: Session) -> bool:
    remaining = sess.held_seconds - sess.seconds
    if remaining < 4 * 60 and time.monotonic() >= sess.next_extend_try:
        ok = await asyncio.to_thread(sess.extend_hold)
        if not ok:
            sess.next_extend_try = time.monotonic() + 10
            if not sess.warned:
                sess.warned = True
                bal = ledger.balance(sess.p.user_id)
                await _send_json(sess.ws, {"type": "warning", "code": "low_balance",
                                           "message": "Your balance is running out; live processing will stop soon.",
                                           "secondsLeft": int(remaining), "availableMicros": bal["available_micros"]})
    if remaining <= 0:
        sess.end_reason = "insufficient_funds"
        await _send_json(sess.ws, {"type": "stopped", "reason": "insufficient_funds"})
        return False
    return True


async def _finish(sess: Session) -> None:
    if sess.writer is not None:
        try:
            await P.send_msg(sess.writer, P.MSG_END, b"")
            if sess.worker_task is not None:
                # The reader forwards final captions, then returns on END.
                await asyncio.wait_for(sess.worker_task, timeout=8)
        except Exception:  # noqa: BLE001
            if sess.worker_task is not None:
                sess.worker_task.cancel()
        sess.writer.close()
    if sess.premium is not None:
        await sess.premium.close()
    cost = 0
    if sess.hold_id:
        cost = await asyncio.to_thread(
            ledger.capture_hold, sess.hold_id, sess.micros_for(sess.seconds),
            f"Live session ({sess.seconds / 60:.1f} min)",
        ) if sess.seconds > 0 else 0
        if sess.seconds <= 0:
            await asyncio.to_thread(ledger.release_hold, sess.hold_id)
    db.execute(
        "UPDATE stream_sessions SET status = 'ended', seconds = ?, cost_micros = ?, end_reason = ?, ended_at = ? WHERE id = ?",
        (sess.seconds, cost, sess.end_reason, db.now(), sess.id),
    )
    try:
        await sess.ws.close()
    except Exception:  # noqa: BLE001
        pass
