"""Live stream endpoint: WebSocket /api/stream/ws (see docs/streaming-protocol.md).

Billing: a rolling hold covers the next few minutes; it is topped up as audio
flows and captured at the exact processed duration when the session ends. A
session stops (after a warning) when the balance cannot cover the next minute.

Free trial: a session that is only Standard noise removal and/or Standard captions runs on the account's free
trial seconds (users.trial_live_seconds) while there are any: no hold, nothing charged. When they run out the
session moves on to paid billing if the account has the credit for it, and stops otherwise.

Pacing: audio must arrive at about real-time speed, otherwise live pricing
could be abused as a cheap batch service.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections import OrderedDict, deque

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .. import auth, cpusplit, db, metrics
from ..billing import ledger, pricing, trust
from ..config import get_settings
from ..security import LIMITS, client_net, limiter
from . import protocol as P

log = logging.getLogger("acw.stream")
router = APIRouter(tags=["stream"])

MAX_SESSIONS_PER_USER = trust.MAX_LIVE_SESSIONS  # of which at most one runs on the free trial
MAX_SESSIONS_PER_IP = 6  # across accounts
MAX_SESSION_SECONDS = 12 * 3600
START_TIMEOUT = 15
SILENCE_TIMEOUT = 30  # nothing at all from the client
IDLE_TIMEOUT = 2 * 60  # no microphone audio (pings and echo-reference frames don't count); both clients keep sending frames while muted
PERSIST_EVERY = 30  # seconds between saving the billed time (a restart then bills what was processed)
# A session must keep sending: over the last DUTY_WINDOW seconds at least DUTY_MIN_FRACTION of the wall-clock time
# must have arrived as microphone audio (checked once DUTY_GRACE seconds have passed). Both clients send every
# frame, silent ones too, so a normal session is at 1.0; a link too slow to carry half of the audio is no live
# session anyway. Without this, one tiny frame every two minutes held a place (and its cores) for twelve hours.
DUTY_WINDOW = 60.0
DUTY_GRACE = 30.0
DUTY_MIN_FRACTION = 0.5
# Text messages per connection (ping, config, stop): a token bucket, and a tighter one for `config`, which also
# costs a write to the worker. Real clients send a ping every few seconds and a config when the slider moves.
MSG_RATE = 50.0
MSG_BURST = 100.0
CONFIG_RATE = 5.0
CONFIG_BURST = 5.0
PACE_BURST_SECONDS = 6.0
PACE_RATE = 1.25  # audio seconds accepted per wall-clock second
LOW_BALANCE_WARN_SECONDS = 300
TRIAL_LOW_WARN_SECONDS = 300
TRIAL_BUSY_MESSAGE = "Free trial sessions are paused while live is busy. Try again in a few minutes, or add credit to go live now."
TRIAL_ONE_AT_A_TIME_MESSAGE = "Your free trial covers one live session at a time. Stop the other one first, or add credit to run both."

_live: dict[str, int] = {}  # user_id -> active sessions in this process
_live_ip: dict[str, int] = {}  # client address -> active sessions in this process
_live_ids: set[str] = set()  # ids of the sessions this process is running (the rest "active" in the database are orphans)
_live_units = 0.0  # capacity units of running sessions (file jobs move off the cores they need)


def drain_requested() -> bool:
    """The flag file data/drain exists: new sessions are refused so running ones can end before a restart."""
    return (get_settings().data_dir / "drain").exists()


async def startup() -> None:
    await asyncio.to_thread(cpusplit.units_per_core)  # calibrate once, off the event loop
    cpusplit.set_live_units(get_settings().data_dir, 0.0, cpusplit.units_per_core())
    # Sessions left "active" by a previous process never reached _finish.
    await asyncio.to_thread(close_orphans, 0, "server_restart")


def close_orphans(min_age: int = 120, reason: str = "session_lost") -> int:
    """Bill the time that was saved (at most PERSIST_EVERY seconds behind) and close "active" sessions that
    no live handler in this process owns: left by a restart, or by a handler that died without finishing.
    Sessions started less than `min_age` seconds ago are left alone (their handler may still be setting up)."""
    n = 0
    for row in db.all_rows(
        "SELECT id, hold_id, seconds, denoise_tier, caption_tier, premium_off_seconds, trial_seconds FROM stream_sessions"
        " WHERE status = 'active' AND started_at <= ?",
        (db.now() - min_age,),
    ):
        if row["id"] in _live_ids:
            continue
        cost = 0
        # The free trial seconds were taken off the account when they were saved: only the rest is billed.
        paid = max(0.0, row["seconds"] - row["trial_seconds"])
        if row["hold_id"]:
            try:
                if paid > 0:
                    cost = ledger.capture_hold(
                        row["hold_id"],
                        _split_micros(
                            _rate_micros_per_minute(row["denoise_tier"], row["caption_tier"]),
                            _premium_rate(row["caption_tier"]), paid, row["premium_off_seconds"],
                        ),
                        f"Live session ({paid / 60:.1f} min, cut short)",
                    )
            except Exception:  # noqa: BLE001
                log.exception("billing stale stream session")
            finally:
                ledger.release_hold(row["hold_id"])  # no-op once captured
        db.execute(
            "UPDATE stream_sessions SET status = 'ended', cost_micros = ?, end_reason = ?, ended_at = ? WHERE id = ? AND status = 'active'",
            (cost, reason, db.now(), row["id"]),
        )
        metrics.live_ended(row["id"])
        n += 1
    return n


async def shutdown() -> None:
    return None


def _rate_micros_per_minute(denoise: str | None, captions: str | None) -> int:
    total = 0
    if denoise:
        total += pricing.rate("stream_denoise", denoise).micros_per_minute
    if captions:
        total += pricing.rate("stream_captions", captions).micros_per_minute
    return total


def _atten(value) -> float | None:
    """Noise-strength limit in dB, 6 to 100; None when it isn't a finite number."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return min(max(v, 6.0), 100.0) if math.isfinite(v) else None


def _micros(rate: int, seconds: float) -> int:
    """Price of `seconds` of live audio at `rate` micros per minute (whole seconds, rounded up)."""
    return math.ceil(rate * math.ceil(round(seconds, 6)) / 60)


def _premium_rate(captions: str | None) -> int:
    """Micros per minute of the extra-accurate captions part of a session's rate (0 without them)."""
    return pricing.rate("stream_captions", "premium").micros_per_minute if captions == "premium" else 0


def _split_micros(rate: int, premium_rate: int, seconds: float, premium_off_seconds: float | None) -> int:
    """Price of a session of `seconds`. The extra-accurate captions part (`premium_rate` of `rate`) stops at
    `premium_off_seconds` when they broke down: only what was actually provided is billed."""
    if not premium_rate or premium_off_seconds is None:
        return _micros(rate, seconds)
    return _micros(rate - premium_rate, seconds) + _micros(premium_rate, min(seconds, premium_off_seconds))


class TokenBucket:
    """Messages per second: `rate` sustained, up to `burst` at once."""

    def __init__(self, rate: float, burst: float):
        self.rate, self.burst = rate, burst
        self.tokens, self.at = burst, time.monotonic()

    def take(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.burst, self.tokens + (now - self.at) * self.rate)
        self.at = now
        if self.tokens < 1.0:
            return False
        self.tokens -= 1.0
        return True


class DutyCycle:
    """Microphone audio received against wall-clock time over a sliding window."""

    def __init__(self, now: float):
        self.start = now
        self.marks: deque[tuple[float, float]] = deque([(now, 0.0)])  # (time, audio seconds so far), about one a second

    def starved(self, now: float, audio_seconds: float) -> bool:
        if now - self.marks[-1][0] >= 1.0:
            self.marks.append((now, audio_seconds))
        while len(self.marks) > 1 and self.marks[1][0] <= now - DUTY_WINDOW:
            self.marks.popleft()  # the first mark is the newest one that is at least a window old
        t0, a0 = self.marks[0]
        span = now - t0
        return span >= DUTY_GRACE and (audio_seconds - a0) < DUTY_MIN_FRACTION * span


async def _send_json(ws: WebSocket, obj: dict) -> None:
    await ws.send_text(json.dumps(obj, separators=(",", ":")))


class Session:
    def __init__(self, ws: WebSocket, principal: auth.Principal, cfg: dict, rate: int, trial_budget: float = 0.0):
        self.ws = ws
        self.p = principal
        self.cfg = cfg
        self.rate = rate  # micros per minute
        self.id = db.new_id()
        self.hold_id: str | None = None
        self.held_seconds = 0.0
        self.samples = 0  # of audio processed; counted as integers so the billed seconds are exact
        self.started = time.monotonic()
        self.frames: OrderedDict[int, tuple[int, int, float]] = OrderedDict()  # seq -> (codec, timestamp, sent at)
        self.frame_stats = metrics.FrameStats()
        self.pace = [[PACE_BURST_SECONDS, time.monotonic()] for _ in range(2)]  # mic, echo reference
        self.warned = False
        self.next_extend_try = 0.0
        self.end_reason = "client_closed"
        self.reader = self.writer = None
        self.last_rx = time.monotonic()  # when the client last sent anything
        self.worker_task: asyncio.Task | None = None
        self.premium = None
        # The extra-accurate captions part of the rate, and the audio seconds at which it stopped working (None: still on).
        self.premium_rate = _premium_rate(cfg.get("captions"))
        self.premium_off_seconds: float | None = None
        # Free trial: the trial seconds the account had when this session started (0: not a trial session), how many
        # of the session's seconds are already taken off the account, and whether it has moved on to paid billing.
        self.trial_budget = trial_budget
        self.trial_charged = 0.0
        self.trial_warned = False
        self.switched = False

    @property
    def seconds(self) -> float:
        return self.samples / P.SAMPLE_RATE

    @seconds.setter
    def seconds(self, value: float) -> None:
        self.samples = round(value * P.SAMPLE_RATE)

    @property
    def trial(self) -> bool:
        return self.trial_budget > 0

    @property
    def on_trial(self) -> bool:
        """Running on free trial seconds right now (no hold, nothing billed)."""
        return self.trial_budget > 0 and not self.switched

    @property
    def trial_used(self) -> float:
        """Seconds of this session paid for by the trial: whole seconds, like the paid ones, up to what the account had."""
        return min(float(math.ceil(round(self.seconds, 6))), self.trial_budget) if self.trial_budget else 0.0

    @property
    def paid_seconds(self) -> float:
        return max(0.0, self.seconds - self.trial_used)

    def micros_for(self, seconds: float) -> int:
        return _split_micros(self.rate, self.premium_rate, seconds, self.premium_off_seconds)

    def current_rate(self) -> int:
        """Micros per minute being provided right now (without the captions that broke down)."""
        return self.rate - (self.premium_rate if self.premium_off_seconds is not None else 0)

    def premium_failed(self) -> None:
        """The extra-accurate captions stopped working: they are not billed from this point."""
        if self.premium_off_seconds is None and self.premium_rate:
            self.premium_off_seconds = self.seconds

    def open_hold(self) -> None:
        minutes = pricing.STREAM_HOLD_MINUTES
        self.hold_id = ledger.create_hold(self.p.user_id, self.rate * minutes, "stream", self.id)
        self.held_seconds = minutes * 60.0

    def save_progress(self) -> None:
        """Save the time processed so far (a restart bills it, see close_orphans) and take the trial seconds used
        since the last save off the account, together."""
        used = self.trial_used
        with db.tx() as c:
            c.execute(
                "UPDATE stream_sessions SET seconds = ?, premium_off_seconds = ?, trial_seconds = ?, hold_id = ? WHERE id = ?",
                (self.seconds, self.premium_off_seconds, used, self.hold_id, self.id),
            )
            if used > self.trial_charged:
                c.execute("UPDATE users SET trial_live_seconds = MAX(0, trial_live_seconds - ?) WHERE id = ?",
                          (used - self.trial_charged, self.p.user_id))
        self.trial_charged = max(self.trial_charged, used)

    def switch_to_paid(self) -> bool:
        """The trial seconds are used up: carry on from the account's credit. False if it can't cover it."""
        try:
            self.open_hold()
        except (ledger.InsufficientFunds, ledger.LedgerError):
            return False
        self.switched = True
        self.save_progress()
        return True

    def extend_hold(self) -> bool:
        try:
            ledger.extend_hold(self.hold_id, self.current_rate())  # one more minute
        except (ledger.InsufficientFunds, ledger.LedgerError):
            return False
        self.held_seconds += 60.0
        return True

    def paced(self, seconds: float, reference: bool = False) -> bool:
        bucket = self.pace[reference]
        now = time.monotonic()
        bucket[0] = min(PACE_BURST_SECONDS, bucket[0] + (now - bucket[1]) * PACE_RATE)
        bucket[1] = now
        if bucket[0] < seconds:
            return False
        bucket[0] -= seconds
        return True


# A session that has sent nothing for this long, while its owner asks for another one at the limit, is taken to be a
# dropped connection the server has not noticed yet (a vanished network, a crashed client) and is replaced.
STALE_SESSION_SECONDS = 10.0
_open: dict[str, list["Session"]] = {}  # user_id -> sessions this process is running


async def _deny(ws: WebSocket, status: int, message: str, retry_after: int | None = None) -> None:
    """Refuse the handshake with a real HTTP status (401 sign-in, 403 not allowed, 429 too often) where the
    server supports it; otherwise the plain 403 of a refused upgrade."""
    from fastapi.responses import JSONResponse

    try:
        await ws.send_denial_response(JSONResponse(
            {"detail": message}, status_code=status, headers={"Retry-After": str(retry_after)} if retry_after else None,
        ))
    except Exception:  # noqa: BLE001 - no denial-response support in this server: close the handshake
        await ws.close(code=1008)


async def _refuse(ws: WebSocket, code: str, message: str, retry_after: int | None = None, close: int = 1008) -> None:
    ev = {"type": "error", "code": code, "message": message}
    if retry_after:
        ev["retryAfterSeconds"] = retry_after
    await _send_json(ws, ev)
    await ws.close(code=close)


async def _evict_stale(user_id: str, limit: int | None = None) -> bool:
    """Free a slot for `user_id` by ending their stalest silent session. True if one was ended."""
    limit = MAX_SESSIONS_PER_USER if limit is None else limit
    now = time.monotonic()
    stale = [x for x in _open.get(user_id, []) if now - x.last_rx > STALE_SESSION_SECONDS]
    if not stale:
        return False
    victim = max(stale, key=lambda x: now - x.last_rx)
    victim.end_reason = "replaced"
    try:
        await victim.ws.close(code=1001)
    except Exception:  # noqa: BLE001 - its socket is gone, which is the point
        pass
    for _ in range(30):  # its handler notices, bills what it processed and releases the slot
        if _live.get(user_id, 0) < limit:
            return True
        await asyncio.sleep(0.1)
    return _live.get(user_id, 0) < limit


def _trial_eligible(cfg: dict, denoise, captions) -> bool:
    """The free trial covers Standard live only: noise removal and/or Standard captions. Extra-accurate captions
    and translation are billed from credit."""
    return denoise in (None, "light") and captions in (None, "local") and not cfg.get("translate")


def _trial_seconds_left(user_id: str) -> float:
    row = db.one("SELECT trial_live_seconds FROM users WHERE id = ?", (user_id,))
    return max(0.0, float(row["trial_live_seconds"])) if row else 0.0


async def _trial_slot(user_id: str) -> bool:
    """True when the account has no other free-trial session running (one that has gone silent is replaced,
    like for the session limit). Checked last, with nothing awaited before the new session is registered."""
    def others() -> list["Session"]:
        return [x for x in _open.get(user_id, []) if x.trial]

    now = time.monotonic()
    stale = [x for x in others() if now - x.last_rx > STALE_SESSION_SECONDS]
    if stale:
        victim = max(stale, key=lambda x: now - x.last_rx)
        victim.end_reason = "replaced"
        try:
            await victim.ws.close(code=1001)
        except Exception:  # noqa: BLE001 - its socket is gone, which is the point
            pass
        for _ in range(30):
            if victim not in _open.get(user_id, []):
                break
            await asyncio.sleep(0.1)
    return not others()


@router.websocket("/api/stream/ws")
async def stream_ws(ws: WebSocket) -> None:
    ip = (ws.headers.get("x-forwarded-for") or (ws.client.host if ws.client else "")).split(",")[-1].strip()[:64]
    net = client_net(ip)
    # Per address, before any database work: bogus handshakes must stay cheap.
    if not limiter.hit(f"ws_handshake:{net}", *LIMITS["ws_handshake"]):
        await _deny(ws, 429, "Too many connection attempts. Try again in a minute.", 60)
        return
    principal = await asyncio.to_thread(auth.websocket_principal, ws)  # database lookup: not on the event loop
    if principal is None:
        await _deny(ws, 401, "Sign-in required, or the API key is wrong or revoked.")
        return
    if principal.status != "active":
        await _deny(ws, 403, "This account is suspended.")
        return
    if "*" not in principal.scopes and "stream" not in principal.scopes:
        await _deny(ws, 403, "This API key does not have the 'stream' scope.")
        return
    # Starts are counted whether or not they succeed (a broken client in a loop is the case to stop), so the
    # limit is generous: an hour of one start per minute, with room for a restart storm.
    for key, limit in ((f"stream_start:{principal.user_id}", LIMITS["stream_start"]), (f"stream_start_ip:{net}", LIMITS["stream_start_ip"])):
        if not limiter.hit(key, *limit):
            wait = limiter.retry_after(key, limit[1])
            await _deny(ws, 429, "Too many live sessions started this hour. Try again later.", wait)
            return
    await ws.accept()
    if drain_requested():
        # A deploy is waiting for sessions to end (deploy/drain.sh): no new ones start.
        await _refuse(ws, "restarting", "Live audio is restarting for an update. Please try again in a minute.", 60, 1013)
        return
    if _live.get(principal.user_id, 0) >= MAX_SESSIONS_PER_USER and not await _evict_stale(principal.user_id):
        await _refuse(ws, "too_many_sessions", "Two live sessions at most per account.", 30)
        return
    if _live_ip.get(net, 0) >= MAX_SESSIONS_PER_IP:
        await _refuse(ws, "too_many_sessions", "Too many live sessions from this network.", 30)
        return
    _live[principal.user_id] = _live.get(principal.user_id, 0) + 1
    _live_ip[net] = _live_ip.get(net, 0) + 1
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
        _live_ip[net] = max(0, _live_ip.get(net, 1) - 1)
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
    await asyncio.to_thread(auth.accept_terms, principal.user_id, "continued use", ip)
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
        await _send_json(ws, {"type": "error", "code": "unavailable", "retryAfterSeconds": 300,
                              "message": "Premium captions are not available right now."})
        await ws.close(code=1008)
        return None

    atten = _atten(cfg.get("attenLimDb", 100.0))
    if atten is None or not all(v is None or isinstance(v, str) for v in (denoise, captions)):
        await _send_json(ws, {"type": "error", "code": "bad_config", "message": "Check the noise strength and the options you chose."})
        await ws.close(code=1008)
        return None
    # The free trial pays for Standard live while the account has trial seconds left; anything else is billed.
    trial_left = 0.0
    if _trial_eligible(cfg, denoise, captions):
        trial_left = await asyncio.to_thread(_trial_seconds_left, principal.user_id)
        if trial_left > 0 and not await _trial_slot(principal.user_id):
            await _send_json(ws, {"type": "error", "code": "too_many_sessions", "message": TRIAL_ONE_AT_A_TIME_MESSAGE})
            await ws.close(code=1008)
            return None
    sess = Session(ws, principal, {"denoise": denoise, "captions": captions, "language": language,
                                   "atten_lim_db": atten,
                                   "echo_reference": cfg.get("echoReference") is True}, rate, trial_left)
    # Whatever goes wrong from here (the client vanishing, the worker, the database), _finish gives the hold back.
    _live_ids.add(sess.id)
    _open.setdefault(principal.user_id, []).append(sess)
    try:
        try:
            if not sess.trial:  # a trial session reserves nothing: it is not charged
                await asyncio.to_thread(sess.open_hold)
        except ledger.InsufficientFunds as e:
            _live_ids.discard(sess.id)
            _open.get(principal.user_id, []).remove(sess)
            await _send_json(ws, {"type": "error", "code": "insufficient_funds",
                                  "message": f"Live processing reserves {pricing.STREAM_HOLD_MINUTES} minutes up front "
                                             f"(${rate * pricing.STREAM_HOLD_MINUTES / 1e6:.2f}); available ${e.available / 1e6:.2f}."})
            await ws.close(code=1008)
            return None
        except ledger.LedgerError:
            _live_ids.discard(sess.id)
            _open.get(principal.user_id, []).remove(sess)
            await ws.close(code=1008)
            return None
        await asyncio.to_thread(
            db.execute,
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
            worker_cfg["trial"] = sess.trial
            worker_cfg["opus_bitrate"] = int(cfg.get("opusBitrate", 96_000)) if str(cfg.get("opusBitrate", "")).isdigit() else 96_000
            await P.send_msg(sess.writer, P.MSG_START, P.pack_json(worker_cfg))
            kind, payload = await asyncio.wait_for(P.read_msg(sess.reader), timeout=20)
            ready = json.loads(payload)
            if kind == P.MSG_EVENT and ready.get("code") == "full":
                sess.end_reason = "full"
                _drop_worker(sess)
                options = ready.get("available") or []
                await _send_json(ws, {"type": "error", "code": "full", "available": options, "retryAfterSeconds": 60,
                                      "message": _full_message(worker_cfg, options)})
                await ws.close(code=1013)
                return sess
            if kind == P.MSG_EVENT and ready.get("code") == "trial_busy":
                sess.end_reason = "trial_busy"
                _drop_worker(sess)
                await _send_json(ws, {"type": "error", "code": "trial_busy", "retryAfterSeconds": 120, "message": TRIAL_BUSY_MESSAGE})
                await ws.close(code=1013)
                return sess
            if kind != P.MSG_EVENT or ready.get("type") != "ready":
                raise RuntimeError(f"worker refused: {ready}")
        except Exception:  # noqa: BLE001
            log.exception("stream worker unavailable")
            sess.end_reason = "worker_unavailable"
            _drop_worker(sess)
            await _send_json(ws, {"type": "error", "code": "unavailable", "retryAfterSeconds": 30,
                                  "message": "Live processing is busy or offline, try again soon."})
            await ws.close(code=1013)
            return sess
        if captions == "premium":
            from .openai_live import PremiumCaptions

            sess.premium = PremiumCaptions(
                language, lambda ev: asyncio.ensure_future(_send_json(ws, ev)), user_hint=principal.user_id,
                on_fail=sess.premium_failed,
            )
            await sess.premium.connect()
            if sess.premium.failed and sess.current_rate() == 0:
                # Captions were all this session was for, and they don't work: nothing to bill, nothing to keep open.
                sess.end_reason = "captions_unavailable"
                _drop_worker(sess)
                await _send_json(ws, {"type": "error", "code": "unavailable", "retryAfterSeconds": 60,
                                      "message": "Extra-accurate captions are unavailable right now. Try again in a few minutes."})
                await ws.close(code=1013)
                return sess
        await _send_json(ws, {
            "type": "ready",
            "sessionId": sess.id,
            "sampleRate": P.SAMPLE_RATE,
            "hopSamples": P.HOP,
            "modelLatencyMs": round(ready.get("latencySamples", 0) * 1000 / P.SAMPLE_RATE, 1),
            "usdPerMinute": sess.current_rate() / 1e6,
            "maxSessionSeconds": MAX_SESSION_SECONDS,
            **({"echoReference": True} if sess.cfg["echo_reference"] else {}),
            **({"trial": True, "trialSecondsLeft": int(sess.trial_budget)} if sess.trial else {}),
        })
        return sess
    except BaseException:
        sess.end_reason = "start_failed"
        await _finish(sess)
        raise


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
                codec, ts, sent_at = meta
                sess.frame_stats.answer(time.monotonic() - sent_at)
                await ws.send_bytes(P.encode_frame(codec, seq, ts, y))
            elif kind == P.MSG_TAP:
                # Extra-accurate captions listen to the echo-free original (before noise removal), like every
                # other caption path: recognisers score worse on denoised speech. The worker sends it as a tap.
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
    last_mic = last_persist = time.monotonic()
    duty = DutyCycle(last_mic)
    messages, configs = TokenBucket(MSG_RATE, MSG_BURST), TokenBucket(CONFIG_RATE, CONFIG_BURST)
    try:
        while True:
            now = time.monotonic()
            # Only microphone audio keeps a session alive: pings and echo-reference frames cost nothing
            # to send, yet the session holds a worker slot and live cores.
            idle_left = IDLE_TIMEOUT - (now - last_mic)
            if idle_left <= 0:
                sess.end_reason = "idle_timeout"
                await _send_stopped(ws, "idle_timeout")
                break
            if now - sess.started >= MAX_SESSION_SECONDS:  # wall time too, not just audio received
                sess.end_reason = "max_duration"
                await _send_json(ws, {"type": "stopped", "reason": "max_duration"})
                break
            if duty.starved(now, sess.seconds):
                sess.end_reason = "no_audio"
                await _send_stopped(ws, "no_audio")
                break
            if sess.premium_rate and sess.premium_off_seconds is not None and sess.current_rate() == 0:
                # The captions this session was started for broke down and there is nothing else to provide.
                sess.end_reason = "captions_unavailable"
                await _send_stopped(ws, "captions_unavailable")
                break
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=min(SILENCE_TIMEOUT, idle_left))
            except asyncio.TimeoutError:
                sess.end_reason = "idle_timeout"
                await _send_stopped(ws, "idle_timeout")
                break
            sess.last_rx = time.monotonic()
            if msg["type"] == "websocket.disconnect":
                break
            if time.monotonic() - last_persist >= PERSIST_EVERY:
                last_persist = time.monotonic()
                if not await _persist(sess):
                    sess.end_reason = "account_closed"
                    await _send_json(ws, {"type": "stopped", "reason": "account_closed"})
                    break
            if worker_task.done():
                sess.end_reason = "worker_lost"
                await _send_json(ws, {"type": "error", "code": "worker_lost", "message": "Processing stopped unexpectedly."})
                break
            if msg.get("bytes") is not None:
                try:
                    codec, seq, ts, flags = P.frame_header(msg["bytes"])
                    reference = bool(flags & P.FLAG_REFERENCE)
                    if codec == P.CODEC_OPUS:
                        # Opaque here; decoded only inside the sandboxed worker.
                        pkt = P.opus_payload(msg["bytes"])
                        n = P.OPUS_FRAME
                        ipc = P.pack_opus(seq, pkt)
                    else:
                        codec, seq, ts, x = P.decode_frame(msg["bytes"])
                        n = len(x)
                        ipc = P.pack_audio(seq, x)
                    dur = n / P.SAMPLE_RATE
                except P.ProtocolError as e:
                    await _send_json(ws, {"type": "error", "code": "bad_frame", "message": str(e)})
                    sess.end_reason = "protocol_error"
                    break
                if not sess.paced(dur, reference):
                    await _send_json(ws, {"type": "error", "code": "too_fast", "message": "Audio must be sent in real time."})
                    sess.end_reason = "too_fast"
                    break
                if reference:  # what the speakers played: no reply, no billing; ignored unless the session asked for it
                    if sess.cfg["echo_reference"]:
                        await P.send_msg(sess.writer, P.MSG_REF, ipc)
                    continue
                last_mic = time.monotonic()
                sess.samples += n
                if sess.seconds >= MAX_SESSION_SECONDS:
                    sess.end_reason = "max_duration"
                    await _send_json(ws, {"type": "stopped", "reason": "max_duration"})
                    break
                if not await _billing_tick(sess):
                    break
                sess.frames[seq] = (codec, ts, time.monotonic())
                sess.frame_stats.sent += 1
                while len(sess.frames) > 500:
                    sess.frames.popitem(last=False)
                await P.send_msg(sess.writer, P.MSG_AUDIO, ipc)
            elif msg.get("text") is not None:
                if not messages.take():  # a flood of tiny messages is work on the loop that carries every session
                    sess.end_reason = "rate_limited"
                    await _send_json(ws, {"type": "error", "code": "rate_limited", "message": "Too many messages."})
                    await ws.close(code=1008, reason="too_many_messages")
                    break
                if len(msg["text"]) > 2048:
                    continue
                try:
                    c = json.loads(msg["text"])
                except ValueError:
                    continue
                t = c.get("type") if isinstance(c, dict) else None
                if t == "config" and not configs.take():
                    sess.end_reason = "rate_limited"
                    await _send_json(ws, {"type": "error", "code": "rate_limited", "message": "Too many settings changes."})
                    await ws.close(code=1008, reason="too_many_config_messages")
                    break
                if t == "ping":
                    await _send_json(ws, {"type": "pong", "t": c.get("t"), "seconds": round(sess.seconds, 2)})
                elif t == "stop":
                    sess.end_reason = "client_stop"
                    break
                elif t == "config" and "attenLimDb" in c:
                    atten = _atten(c["attenLimDb"])
                    if atten is not None:
                        await P.send_msg(sess.writer, P.MSG_CONFIG, P.pack_json({"atten_lim_db": atten}))
    finally:
        # Not cancelled here: _finish lets it drain the worker's last captions.
        pass


async def _send_stopped(ws: WebSocket, reason: str) -> None:
    try:
        await _send_json(ws, {"type": "stopped", "reason": reason})
    except Exception:  # noqa: BLE001 - the client may be gone already
        pass


async def _persist(sess: Session) -> bool:
    """Save the time processed so far (a restart bills it, see startup) and check the account is still
    open: suspension and deletion stop a running session. False when the session must end."""

    def work() -> bool:
        sess.save_progress()
        row = db.one("SELECT status FROM users WHERE id = ?", (sess.p.user_id,))
        if row is None or row["status"] != "active":
            return False
        # The key or browser session this stream was started with must still be valid.
        api_key_id, session_hash = getattr(sess.p, "api_key_id", None), getattr(sess.p, "session_hash", None)
        if api_key_id:
            key = db.one("SELECT revoked_at FROM api_keys WHERE id = ?", (api_key_id,))
            if key is None or key["revoked_at"] is not None:
                return False
        elif session_hash:
            ses = db.one("SELECT expires_at FROM sessions WHERE id_hash = ?", (session_hash,))
            if ses is None or ses["expires_at"] <= db.now():
                return False
        return True

    return await asyncio.to_thread(work)


async def _billing_tick(sess: Session) -> bool:
    if sess.on_trial:
        left = sess.trial_budget - sess.seconds
        if left > 0:
            if left <= TRIAL_LOW_WARN_SECONDS and not sess.trial_warned:
                sess.trial_warned = True
                await _send_json(sess.ws, {"type": "warning", "code": "trial_low", "secondsLeft": int(left),
                                           "message": "Your free trial is almost used up."})
            return True
        # Used up: carry on from the account's credit without a gap, or stop if it can't cover it.
        if not await asyncio.to_thread(sess.switch_to_paid):
            sess.end_reason = "trial_over"
            await _send_json(sess.ws, {"type": "stopped", "reason": "trial_over"})
            return False
        await _send_json(sess.ws, {"type": "warning", "code": "trial_used_up_now_paid",
                                   "message": "Your free trial is used up. Live now runs on your credit.",
                                   "usdPerMinute": sess.current_rate() / 1e6})
    remaining = sess.held_seconds - sess.paid_seconds
    if remaining < 4 * 60 and time.monotonic() >= sess.next_extend_try:
        ok = await asyncio.to_thread(sess.extend_hold)
        if not ok:
            sess.next_extend_try = time.monotonic() + 10
            if not sess.warned:
                sess.warned = True
                bal = await asyncio.to_thread(ledger.balance, sess.p.user_id)
                await _send_json(sess.ws, {"type": "warning", "code": "low_balance",
                                           "message": "Your balance is running out; live processing will stop soon.",
                                           "secondsLeft": int(remaining), "availableMicros": bal["available_micros"]})
    if remaining <= 0:
        sess.end_reason = "insufficient_funds"
        await _send_json(sess.ws, {"type": "stopped", "reason": "insufficient_funds"})
        return False
    return True


async def _finish(sess: Session) -> None:
    try:
        await _finish_inner(sess)
    finally:
        _live_ids.discard(sess.id)
        mine = _open.get(sess.p.user_id, [])
        if sess in mine:
            mine.remove(sess)


async def _finish_inner(sess: Session) -> None:
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
        try:
            await sess.premium.close()
        except Exception:  # noqa: BLE001
            log.exception("closing premium captions")
    cost = 0
    try:
        if sess.hold_id and sess.paid_seconds > 0:
            paid = sess.paid_seconds
            cost = await asyncio.to_thread(
                ledger.capture_hold, sess.hold_id, sess.micros_for(paid), f"Live session ({paid / 60:.1f} min)",
            )
    except Exception:  # noqa: BLE001 - e.g. the account was suspended and its holds released
        log.exception("billing stream session %s", sess.id)
    finally:
        if sess.hold_id:
            await asyncio.to_thread(ledger.release_hold, sess.hold_id)  # nothing left reserved, whatever happened
        if sess.trial:
            try:
                await asyncio.to_thread(sess.save_progress)  # the rest of the trial seconds this session used
            except Exception:  # noqa: BLE001
                log.exception("saving the trial seconds of stream session %s", sess.id)
        fs = sess.frame_stats
        await asyncio.to_thread(
            db.execute,
            "UPDATE stream_sessions SET status = 'ended', seconds = ?, trial_seconds = ?, cost_micros = ?, end_reason = ?,"
            " ended_at = ?, frames = ?, late_frames = ?, dropped_frames = ?, premium_off_seconds = ? WHERE id = ?",
            (sess.seconds, sess.trial_used, cost, sess.end_reason, db.now(), fs.sent, fs.late, fs.dropped,
             sess.premium_off_seconds, sess.id),
        )
        await asyncio.to_thread(metrics.live_ended, sess.id, fs, time.monotonic() - sess.started)
    try:
        await sess.ws.close()
    except Exception:  # noqa: BLE001
        pass
