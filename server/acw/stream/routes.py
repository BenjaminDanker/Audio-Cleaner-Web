"""Live stream endpoint: WebSocket /api/stream/ws (see docs/streaming-protocol.md).

Billing: a rolling hold covers the next few minutes; it is topped up as audio
flows and captured at the exact processed duration when the session ends. A
session stops (after a warning) when the balance cannot cover the next minute.
Its features (stream/features.py) can change while it runs (a `config` message): each stretch of audio is billed
at the rate of the features it had (stream/meter.py).

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
import os
import time
from collections import OrderedDict, deque
from dataclasses import replace

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .. import auth, broker, db, keycaps, metrics
from ..billing import ledger, pricing, trust
from ..config import get_settings
from ..security import LIMITS, client_net, limiter
from . import protocol as P
from .features import FeatureError, Features, atten
from .features import parse as parse_features
from .meter import Meter
from .meter import micros as meter_micros

log = logging.getLogger("acw.stream")
router = APIRouter(tags=["stream"])

MAX_SESSIONS_PER_USER = trust.MAX_LIVE_SESSIONS  # of which at most one runs on the free trial
MAX_SESSIONS_PER_IP = 6  # across accounts
# Everything live in this process shares one event loop and one GIL, which relays the audio (and feeds the
# extra-accurate captions' socket) next to the job API. Its cost per session was measured (2026-10-18, compression off:
# about 1.5% of a core per Opus session and 2% per PCM session, extra-accurate captions not measured), but the mix
# of sessions, uploads and API calls varies, so admission looks at the result instead of counting: how late the loop
# runs its timers (LoopLag). New sessions are refused only while it runs late; running sessions are never cut.
# MAX_LIVE_SESSIONS is only a sanity bound below uvicorn's --limit-concurrency.
MAX_LIVE_SESSIONS = int(os.environ.get("ACW_MAX_LIVE_SESSIONS", "300"))
LOOP_LAG_LIMIT = float(os.environ.get("ACW_LOOP_LAG_LIMIT_MS", "25")) / 1000  # p90 timer lateness over 10 s
LIVE_BUSY_MESSAGE = "Live is very busy right now. Please try again in a few minutes."
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
TRIAL_LOW_WARN_SECONDS = 300
TRIAL_BUSY_MESSAGE = "Free trial sessions are paused while live is busy. Try again in a few minutes, or add credit to go live now."
TRIAL_ONE_AT_A_TIME_MESSAGE = "Your free trial covers one live session at a time. Stop the other one first, or add credit to run both."

_live: dict[str, int] = {}  # user_id -> active sessions in this process
_live_ip: dict[str, int] = {}  # client address -> active sessions in this process
_live_ids: set[str] = set()  # ids of the sessions this process is running (the rest "active" in the database are orphans)
_busy_cores: dict[int, int] = {}  # physical core -> sessions the live engine placed there (file jobs keep off it)


def _session_cores(sess: "Session", delta: int) -> None:
    """Count the session on (+1) or off (-1) its core and tell the job workers (resource broker) when that changes."""
    before = {c for c, n in _busy_cores.items() if n > 0}
    for c in sess.worker_cores:
        _busy_cores[c] = max(0, _busy_cores.get(c, 0) + delta)
    after = {c for c, n in _busy_cores.items() if n > 0}
    if after != before:
        broker.publish_live_cores(after)


def drain_requested() -> bool:
    """The flag file data/drain exists: new sessions are refused so running ones can end before a restart."""
    return (get_settings().data_dir / "drain").exists()


async def startup() -> None:
    global _lag_task
    if _lag_task is None or _lag_task.done():
        _lag_task = asyncio.get_running_loop().create_task(_lag.run())
    broker.publish_live_cores([])
    # Sessions left "active" by a previous run of this process never reached _finish.
    await asyncio.to_thread(close_orphans, 0, "server_restart")


def close_orphans(min_age: int = 120, reason: str = "session_lost") -> int:
    """Bill the time that was saved (at most PERSIST_EVERY seconds behind) and close "active" sessions that
    no live handler in this process (the live process, acw.live_app: the only one that runs live sessions) owns: left
    by a restart, or by a handler that died without finishing. Sessions started less than `min_age` seconds ago are
    left alone (their handler may still be setting up)."""
    n = 0
    for row in db.all_rows(
        "SELECT id, hold_id, meter_closed, meter_rate, meter_seconds, paid_seconds FROM stream_sessions"
        " WHERE status = 'active' AND started_at <= ?",
        (db.now() - min_age,),
    ):
        if row["id"] in _live_ids:
            continue
        cost = 0
        if row["hold_id"]:
            try:
                if row["paid_seconds"] > 0:
                    cost = ledger.capture_hold(
                        row["hold_id"],
                        meter_micros(row["meter_closed"], row["meter_rate"] or 0, row["meter_seconds"], row["paid_seconds"]),
                        f"Live session ({row['paid_seconds'] / 60:.1f} min, cut short)",
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
    def __init__(self, ws: WebSocket, principal: auth.Principal, features: Features, trial_budget: float = 0.0,
                 return_audio: bool = True):
        self.ws = ws
        self.p = principal
        self.features = features
        self.return_audio = return_audio
        self.meter = Meter(features.rate())
        self.premium_failed = False  # the extra-accurate captions broke down: not billed from then on
        self.id = db.new_id()
        self.hold_id: str | None = None
        self.held_micros = 0  # reserved by the hold so far
        self.samples = 0  # of audio processed; counted as integers so the billed seconds are exact
        self.trial_samples = 0
        self.arrived = 0  # of microphone audio received, silence frames too (keeps the session alive, not billed)
        self.started = time.monotonic()
        self.frames: OrderedDict[int, tuple[int, int, float]] = OrderedDict()  # seq -> (codec, timestamp, sent at)
        self.frame_stats = metrics.FrameStats()
        self.pace = [[PACE_BURST_SECONDS, time.monotonic()] for _ in range(2)]  # mic, echo reference
        self.warned = False
        self.next_extend_try = 0.0
        self.end_reason = "client_closed"
        self.reader = self.writer = None
        self.worker_cores: list[int] = []  # the physical core of the live engine running this session
        self.last_rx = time.monotonic()  # when the client last sent anything
        self.worker_task: asyncio.Task | None = None
        self.premium = None
        self.translator = None  # LiveTranslator, when translations were asked for
        self.config_waiters: dict[str, asyncio.Future] = {}  # the engine's answers to feature changes, by id
        self._outbox: asyncio.Queue | None = None  # events posted by callbacks that can't wait (post), sent in order by _sender
        self._sender: asyncio.Task | None = None
        # Free trial: the trial seconds the account had when this session started (0: not a trial session), how many
        # of the session's seconds are already taken off the account, and whether it has moved on to paid billing.
        self.trial_budget = trial_budget
        self.trial_charged = 0.0
        self.trial_warned = False
        self.switched = False
        self.capped = False  # the API key's monthly spending cap stopped the next minute from being reserved

    @property
    def seconds(self) -> float:
        return self.samples / P.SAMPLE_RATE

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
        return min(float(math.ceil(round(self.trial_samples / P.SAMPLE_RATE, 6))), self.trial_budget) if self.trial_budget else 0.0

    @property
    def paid_seconds(self) -> float:
        return self.meter.total / P.SAMPLE_RATE

    def count(self, n: int) -> None:
        """`n` samples of audio processed: on the trial while it lasts, then paid."""
        self.samples += n
        if self.on_trial:
            take = min(n, max(0, round(self.trial_budget * P.SAMPLE_RATE) - self.trial_samples))
            self.trial_samples += take
            n -= take
        self.meter.add(n)

    def unbill(self, samples: int) -> None:
        """Audio already counted that turned out to need no work (silence in a captions-only session)."""
        n = self.meter.remove(samples)
        rest = min(max(0, samples) - n, self.trial_samples) if self.on_trial else 0
        self.trial_samples -= rest
        self.samples = max(0, self.samples - n - rest)

    def post(self, ev: dict) -> None:
        """Send `ev` from a callback that can't wait (the caption engines and the translator): in order, after what was posted
        before, by one task the session holds (a task nobody holds can be collected before it ran)."""
        if self._outbox is None:
            self._outbox = asyncio.Queue()
            self._sender = asyncio.get_running_loop().create_task(self._send_posted())
        self._outbox.put_nowait(ev)

    async def _send_posted(self) -> None:
        while True:
            ev = await self._outbox.get()
            try:
                await _send_json(self.ws, ev)
            except Exception:  # noqa: BLE001 - the socket is gone: nothing more can be sent
                return
            finally:
                self._outbox.task_done()  # (sent, or given up: close_outbox waits for this, not for the queue to be empty)

    async def close_outbox(self, wait: float = 2.0) -> None:
        """At the end: let what was posted go out (the last captions), for at most `wait` seconds, then stop the sender."""
        if self._sender is None:
            return
        if not self._sender.done():
            try:  # every posted event sent, the one being sent included (each is marked done after its send)
                await asyncio.wait_for(asyncio.shield(self._outbox.join()), wait)
            except asyncio.TimeoutError:
                pass
        self._sender.cancel()

    def current_rate(self) -> int:
        """Micros per minute being provided right now."""
        return self.meter.rate

    def remaining_seconds(self) -> float:
        """Seconds of audio the hold still covers at the current rate."""
        rate = self.current_rate()
        return math.inf if rate == 0 else (self.held_micros - self.meter.micros()) * 60 / rate

    def failed_premium(self) -> None:
        """The extra-accurate captions stopped working: they are not billed from this point."""
        if not self.premium_failed and self.features.captions == "premium":
            self.premium_failed = True
            self.meter.set_rate(self.meter.rate - self.features.premium_rate())

    def set_features(self, features: Features) -> None:
        """From now on the session does `features`: the audio so far keeps its rate."""
        self.features = features
        self.premium_failed = False
        self.meter.set_rate(features.rate())

    def open_hold(self) -> None:
        minutes = pricing.STREAM_HOLD_MINUTES
        amount = self.current_rate() * minutes
        # A key with a monthly spending cap is checked in the same transaction that reserves the money.
        with db.tx() as c:
            keycaps.check(c, getattr(self.p, "api_key_id", None), amount)
            self.hold_id = ledger.create_hold(self.p.user_id, amount, "stream", self.id, c=c)
        self.held_micros = amount

    def save_progress(self) -> None:
        """Save the time processed so far and the meter (a restart bills it, see close_orphans) and take the trial
        seconds used since the last save off the account, together."""
        used = self.trial_used
        f = self.features
        with db.tx() as c:
            c.execute(
                "UPDATE stream_sessions SET seconds = ?, paid_seconds = ?, trial_seconds = ?, hold_id = ?, meter_closed = ?,"
                " meter_rate = ?, meter_seconds = ?, denoise_tier = ?, caption_tier = ? WHERE id = ?",
                (self.seconds, self.paid_seconds, used, self.hold_id, self.meter.closed, self.meter.rate, self.meter.seconds,
                 f.denoise, f.captions, self.id),
            )
            if used > self.trial_charged:
                c.execute("UPDATE users SET trial_live_seconds = MAX(0, trial_live_seconds - ?) WHERE id = ?",
                          (used - self.trial_charged, self.p.user_id))
        self.trial_charged = max(self.trial_charged, used)

    def switch_to_paid(self) -> bool:
        """The trial seconds are used up, or the session now does more than the trial covers: carry on from the
        account's credit. False if it can't cover it."""
        try:
            self.open_hold()
        except (ledger.InsufficientFunds, ledger.LedgerError):
            return False
        except keycaps.KeyCapReached:
            self.capped = True
            return False
        self.switched = True
        self.save_progress()
        return True

    def extend_hold(self) -> bool:
        try:
            with db.tx() as c:
                keycaps.check(c, getattr(self.p, "api_key_id", None), self.current_rate())  # the key's monthly cap, if it has one
                ledger.extend_hold(self.hold_id, self.current_rate(), c=c)  # one more minute
        except (ledger.InsufficientFunds, ledger.LedgerError):
            return False
        except keycaps.KeyCapReached:
            self.capped = True
            return False
        self.held_micros += self.current_rate()
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


async def _refuse_busy(ws: WebSocket, message: str) -> None:
    """The 'full' answer clients already show (a friendly message and a retry), for the limits of this process."""
    await _send_json(ws, {"type": "error", "code": "full", "available": [], "retryAfterSeconds": 60, "message": message})
    await ws.close(code=1013)


class LoopLag:
    """How late this process's event loop runs a 100 ms timer: the measured cost of everything it does at once."""

    INTERVAL = 0.1

    def __init__(self, window: int = 100):  # the last 10 s
        self.samples: deque[float] = deque(maxlen=window)

    async def run(self) -> None:
        while True:
            t = time.monotonic()
            await asyncio.sleep(self.INTERVAL)
            self.samples.append(max(0.0, time.monotonic() - t - self.INTERVAL))

    def p90(self) -> float:
        if len(self.samples) < 20:  # under 2 s of measurements: one slow moment is not a busy loop
            return 0.0
        xs = sorted(self.samples)
        return xs[int(0.9 * (len(xs) - 1))]


_lag = LoopLag()
_lag_task: asyncio.Task | None = None


def loop_busy() -> bool:
    return _lag.p90() > LOOP_LAG_LIMIT


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
        await _refuse(ws, "too_many_sessions", "Four live sessions at most per account.", 30)
        return
    if _live_ip.get(net, 0) >= MAX_SESSIONS_PER_IP:
        await _refuse(ws, "too_many_sessions", "Too many live sessions from this network.", 30)
        return
    if sum(_live.values()) >= MAX_LIVE_SESSIONS or loop_busy():
        if loop_busy():
            log.warning("live: new session refused, the event loop runs %.0f ms late (p90)", _lag.p90() * 1000)
        await _refuse_busy(ws, LIVE_BUSY_MESSAGE)
        return
    _live[principal.user_id] = _live.get(principal.user_id, 0) + 1
    _live_ip[net] = _live_ip.get(net, 0) + 1
    sess: Session | None = None
    placed = False
    try:
        sess = await _start(ws, principal, ip)
        if sess is not None and sess.writer is not None:
            _session_cores(sess, +1)
            placed = True
            await _run(sess)
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        log.exception("stream session failed")
        if sess:
            sess.end_reason = "server_error"
    finally:
        if placed:
            _session_cores(sess, -1)
        _live[principal.user_id] -= 1
        _live_ip[net] = max(0, _live_ip.get(net, 1) - 1)
        if sess is not None:
            await _finish(sess)


async def _open_worker(sess: "Session", worker_cfg: dict) -> dict:
    """Starts the session in the live engine. Returns its "ready" event or its refusal ("full" with what would still
    fit, "trial_busy"); raises if the engine doesn't answer."""
    reader, writer = await asyncio.open_unix_connection(str(get_settings().stream_worker_socket))
    try:
        await P.send_msg(writer, P.MSG_START, P.pack_json(worker_cfg))
        kind, payload = await asyncio.wait_for(P.read_msg(reader), timeout=20)
        ready = json.loads(payload)
    except BaseException:
        writer.close()
        raise
    if kind == P.MSG_EVENT and ready.get("type") == "ready":
        sess.reader, sess.writer = reader, writer
        sess.worker_cores = [int(c) for c in ready.get("cores") or []]
        return ready
    writer.close()
    if kind == P.MSG_EVENT and ready.get("code") in ("full", "trial_busy"):
        return ready
    raise RuntimeError(f"live engine refused: {ready}")


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
    try:
        if cfg.get("sampleRate", P.SAMPLE_RATE) != P.SAMPLE_RATE:
            raise FeatureError("bad_config", "Send 48 kHz audio.")
        features = parse_features(cfg)
    except FeatureError as e:
        await _send_json(ws, e.event())
        await ws.close(code=1008)
        return None
    # The free trial pays for Standard live while the account has trial seconds left; anything else is billed.
    trial_left = 0.0
    if features.trial_eligible():
        trial_left = await asyncio.to_thread(_trial_seconds_left, principal.user_id)
        if trial_left > 0 and not await _trial_slot(principal.user_id):
            await _send_json(ws, {"type": "error", "code": "too_many_sessions", "message": TRIAL_ONE_AT_A_TIME_MESSAGE})
            await ws.close(code=1008)
            return None
    sess = Session(ws, principal, features, trial_left, return_audio=cfg.get("returnAudio") is not False)
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
                                  "message": _hold_message(sess.current_rate(), e.available)})
            await ws.close(code=1008)
            return None
        except keycaps.KeyCapReached as e:
            _live_ids.discard(sess.id)
            _open.get(principal.user_id, []).remove(sess)
            await _send_json(ws, {"type": "error", "code": "key_spending_cap", "message": str(e)})
            await ws.close(code=1008)
            return None
        except ledger.LedgerError:
            _live_ids.discard(sess.id)
            _open.get(principal.user_id, []).remove(sess)
            await ws.close(code=1008)
            return None
        await asyncio.to_thread(
            db.execute,
            "INSERT INTO stream_sessions(id, user_id, api_key_id, denoise_tier, caption_tier, status, hold_id, client_ip, started_at,"
            " meter_rate) VALUES (?,?,?,?,?, 'active', ?,?,?,?)",
            (sess.id, principal.user_id, principal.api_key_id, features.denoise, features.captions, sess.hold_id, ip[:64],
             db.now(), sess.current_rate()),
        )
        try:
            worker_cfg = features.worker(sess.trial)
            worker_cfg["return_audio"] = sess.return_audio
            # 24-160 kbit/s (docs/streaming-protocol.md); anything else is the default
            rate = int(cfg["opusBitrate"]) if str(cfg.get("opusBitrate", "")).isdigit() else 96_000
            worker_cfg["opus_bitrate"] = rate if 24_000 <= rate <= 160_000 else 96_000
            # What this session's class may take (resource broker): a free-trial session doesn't open a new core
            # while paid file jobs run, and the engine keeps it below 70% of live capacity.
            worker_cfg.update(await asyncio.to_thread(broker.live_options, "live_trial" if sess.trial else "live"))
            ready = await _open_worker(sess, worker_cfg)
            if ready.get("code") == "full":
                sess.end_reason = "full"
                options = ready.get("available") or []
                await _send_json(ws, {"type": "error", "code": "full", "available": options, "retryAfterSeconds": 60,
                                      "message": _full_message(worker_cfg, options)})
                await ws.close(code=1013)
                return sess
            if ready.get("code") == "trial_busy":
                sess.end_reason = "trial_busy"
                await _send_json(ws, {"type": "error", "code": "trial_busy", "retryAfterSeconds": 120, "message": TRIAL_BUSY_MESSAGE})
                await ws.close(code=1013)
                return sess
            if ready.get("type") != "ready":
                raise RuntimeError(f"worker refused: {ready}")
        except Exception:  # noqa: BLE001
            log.exception("stream worker unavailable")
            sess.end_reason = "worker_unavailable"
            _drop_worker(sess)
            await _send_json(ws, {"type": "error", "code": "unavailable", "retryAfterSeconds": 30,
                                  "message": "Live processing is busy or offline, try again soon."})
            await ws.close(code=1013)
            return sess
        _set_translations(sess, features)
        if features.captions == "premium":
            sess.premium = await _open_premium(sess, features)
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
            "silenceFrames": True,
            "changeable": True,  # its features can change while it runs (config)
            **({"echoReference": True} if features.echo_reference else {}),
            **({"translate": list(features.translate)} if features.translate else {}),
            **({"trial": True, "trialSecondsLeft": int(sess.trial_budget)} if sess.trial else {}),
        })
        return sess
    except BaseException:
        sess.end_reason = "start_failed"
        await _finish(sess)
        raise


def _hold_message(rate: int, available: int) -> str:
    return (f"Live processing reserves {pricing.STREAM_HOLD_MINUTES} minutes up front "
            f"(${rate * pricing.STREAM_HOLD_MINUTES / 1e6:.2f}); available ${available / 1e6:.2f}.")


def _event_out(sess: Session):
    """Sends an event to the client and, for captions, to the translator."""
    def emit(ev: dict) -> None:
        sess.post(ev)
        if sess.translator is not None:
            sess.translator.offer(ev)
    return emit


async def _open_premium(sess: Session, features: Features):
    from .openai_live import PremiumCaptions

    premium = PremiumCaptions(list(features.languages), _event_out(sess), user_hint=sess.p.user_id, on_fail=sess.failed_premium)
    await premium.connect()
    return premium


def _set_translations(sess: Session, features: Features) -> None:
    """Translate the captions into `features.translate` from now on (none: whatever runs stops)."""
    if sess.translator is None:
        if not features.translate:
            return
        from .live_translate import LiveTranslator

        sess.translator = LiveTranslator([], None, sess.post)
    sess.translator.set_targets(list(features.translate), features.caption_language)


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
                if sess.return_audio:
                    await ws.send_bytes(P.encode_frame(codec, seq, ts, y))
            elif kind == P.MSG_TAP:
                # Extra-accurate captions listen to the echo-free original (before noise removal), like every
                # other caption path: recognisers score worse on denoised speech. The worker sends it as a tap.
                if sess.premium is not None:
                    tap = P.unpack_audio(payload)[2]
                    if not await sess.premium.feed(tap) and sess.features.denoise is None:
                        sess.unbill(len(tap))  # captions only, and nobody talking: nothing was sent out, nothing billed
            elif kind == P.MSG_EVENT and b'"type":"_' in payload:
                ev = json.loads(payload)  # "_..." events are for us, never for the client
                if ev.get("type") == "_idle" and sess.features.denoise is None:
                    sess.unbill(int(ev.get("samples", 0)))  # standard captions only: the silence the decoder skipped
                elif ev.get("type") == "_config":
                    waiter = sess.config_waiters.pop(str(ev.get("id")), None)
                    if waiter is not None and not waiter.done():
                        waiter.set_result(ev)
            elif kind == P.MSG_EVENT:
                try:
                    await ws.send_text(payload.decode())
                except Exception:  # noqa: BLE001 - client gone; keep draining
                    pass
                if sess.translator is not None:
                    try:
                        sess.translator.offer(json.loads(payload))
                    except ValueError:
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
            if duty.starved(now, sess.arrived / P.SAMPLE_RATE):
                sess.end_reason = "no_audio"
                await _send_stopped(ws, "no_audio")
                break
            if sess.premium_failed and sess.current_rate() == 0:
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
                    if codec == P.CODEC_SILENCE:
                        n = P.silence_samples(msg["bytes"])
                        ipc = None
                    elif codec == P.CODEC_OPUS:
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
                    if sess.features.echo_reference and ipc is not None:  # a silent reference is simply not sent
                        await P.send_msg(sess.writer, P.MSG_REF, ipc)
                    continue
                last_mic = time.monotonic()
                sess.arrived += n
                if ipc is None:
                    # Silence (a muted source, digital silence): answered at once, not billed, no processing. The
                    # worker only hears how much, so the captions' times keep following the talk.
                    if sess.return_audio:
                        await ws.send_bytes(P.silence_frame(seq, ts, n))
                    await P.send_msg(sess.writer, P.MSG_SILENCE, P.pack_silence(seq, n))
                    continue
                sess.count(n)
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
                if t in ("config", "room_test") and not configs.take():
                    sess.end_reason = "rate_limited"
                    await _send_json(ws, {"type": "error", "code": "rate_limited", "message": "Too many settings changes."})
                    await ws.close(code=1008, reason="too_many_config_messages")
                    break
                if t == "ping":
                    await _send_json(ws, {"type": "pong", "t": c.get("t"), "seconds": round(sess.seconds, 2)})
                elif t == "stop":
                    sess.end_reason = "client_stop"
                    break
                elif t == "config" and set(c) - {"type", "attenLimDb"}:  # a feature, or an id to answer to
                    await _reconfigure(sess, c)
                elif t == "config" and "attenLimDb" in c:
                    strength = atten(c["attenLimDb"])
                    if strength is not None:
                        sess.features = replace(sess.features, atten_lim_db=strength)
                        await P.send_msg(sess.writer, P.MSG_CONFIG, P.pack_json({"atten_lim_db": strength}))
                elif t == "room_test":
                    await _room_test(sess, c.get("seconds"))
    finally:
        # Not cancelled here: _finish lets it drain the worker's last captions.
        pass


ROOM_TEST_NEEDS_NOISE_REMOVAL = {"type": "room_test", "ok": False, "code": "needs_noise_removal",
                                 "message": "The room test listens through noise removal: turn noise removal on first."}


async def _room_test(sess, seconds) -> None:
    """The plugin's "Test my room": the engine measures the noise at the noise remover for `seconds` (5-60, 20 by default)
    of processed audio and answers with a room_test event (forwarded as it is)."""
    if sess.features.denoise is None:
        await _send_json(sess.ws, ROOM_TEST_NEEDS_NOISE_REMOVAL)
        return
    n = seconds if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and math.isfinite(seconds) else 20
    await P.send_msg(sess.writer, P.MSG_CONFIG, P.pack_json({"room_test_seconds": int(min(60, max(5, round(n))))}))


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


CONFIG_TIMEOUT = 5.0  # seconds the live engine has to answer a feature change
CHANGE_REFUSED = {
    "full": "Live is full right now, so this change can't be made. Your current settings keep running.",
    "trial_busy": "Free trial sessions can't take on more while live is busy. Your current settings keep running.",
    "unavailable": "That change can't be made right now. Your current settings keep running.",
}


async def _engine_config(sess: Session, f: Features, trial: bool) -> str | None:
    """Tell the live engine the session now does `f`; None when it did, else why not (full, trial_busy, unavailable)."""
    cid = db.new_id()
    waiter = asyncio.get_running_loop().create_future()
    sess.config_waiters[cid] = waiter
    options = await asyncio.to_thread(broker.live_options, "live_trial" if trial else "live")
    await P.send_msg(sess.writer, P.MSG_CONFIG, P.pack_json({
        **f.worker(trial), "id": cid, "may_open_core": options.get("may_open_core", True),
    }))
    try:
        ev = await asyncio.wait_for(waiter, CONFIG_TIMEOUT)
    except asyncio.TimeoutError:
        sess.config_waiters.pop(cid, None)
        return "unavailable"
    if not ev.get("ok"):
        return str(ev.get("code") or "full")
    cores = [int(c) for c in ev.get("cores") or []]
    if cores != sess.worker_cores:  # the engine moved it to a core with room: file jobs keep off that one now
        _session_cores(sess, -1)
        sess.worker_cores = cores
        _session_cores(sess, +1)
    return None


async def _reconfigure(sess: Session, msg: dict) -> None:
    """A `config` message that changes what the session does (stream/features.py). The extra-accurate captions, the
    live engine, the billing and the translations switch together, from the next audio on, or nothing changes and the
    client is told why. The audio so far keeps the rate it had."""
    cid = msg.get("id")
    reply = {"type": "config", **({"id": cid} if isinstance(cid, (str, int)) and len(str(cid)) <= 64 else {})}
    old = sess.features
    trial_before = sess.on_trial

    async def refuse(code: str, message: str, retry: int | None = None) -> None:
        await _send_json(sess.ws, {**reply, "ok": False, "code": code, "message": message,
                                   **({"retryAfterSeconds": retry} if retry else {})})

    try:
        new = parse_features(msg, old)
    except FeatureError as e:
        await refuse(e.code, e.message, e.retry_after)
        return
    if new != old:
        to_paid = trial_before and not new.trial_eligible()  # the trial covers Standard live only
        premium = None
        if new.captions == "premium" and (old.captions != "premium" or new.languages != old.languages or sess.premium_failed):
            premium = await _open_premium(sess, new)
            if premium.failed:
                await premium.close()
                await refuse("unavailable", "Extra-accurate captions are unavailable right now. Try again in a few minutes.", 60)
                return
        why = await _engine_config(sess, new, trial_before and not to_paid)
        if why is not None:
            if premium is not None:
                await premium.close()
            await refuse(why, CHANGE_REFUSED.get(why, CHANGE_REFUSED["unavailable"]), 60)
            return
        sess.set_features(new)
        paid_ok = True
        if to_paid:
            paid_ok = await asyncio.to_thread(sess.switch_to_paid)
        elif not sess.on_trial and sess.remaining_seconds() < 4 * 60:
            paid_ok = await asyncio.to_thread(sess.extend_hold)  # costs more per minute now: the hold must still cover it
        if not paid_ok:
            sess.set_features(old)
            await _engine_config(sess, old, trial_before)
            if premium is not None:
                await premium.close()
            if sess.capped:
                await refuse("key_spending_cap", "This API key's monthly spending limit doesn't leave room for these settings.")
            else:
                bal = await asyncio.to_thread(ledger.balance, sess.p.user_id)
                await refuse("insufficient_funds", _hold_message(new.rate(), bal["available_micros"]))
            return
        if premium is not None or new.captions != "premium":
            if sess.premium is not None:
                await sess.premium.close()
            sess.premium = premium
        _set_translations(sess, new)
        await asyncio.to_thread(sess.save_progress)
    elif new.atten_lim_db != old.atten_lim_db:
        sess.features = new
        await P.send_msg(sess.writer, P.MSG_CONFIG, P.pack_json({"atten_lim_db": new.atten_lim_db}))
    f = sess.features
    await _send_json(sess.ws, {
        **reply, "ok": True, "usdPerMinute": sess.current_rate() / 1e6, "denoise": f.denoise, "captions": f.captions,
        "language": list(f.languages), "translate": list(f.translate), "echoReference": f.echo_reference,
        "attenLimDb": f.atten_lim_db, **({"trial": True} if sess.on_trial else {}),
    })


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
    remaining = sess.remaining_seconds()
    if remaining < 4 * 60 and time.monotonic() >= sess.next_extend_try:
        ok = await asyncio.to_thread(sess.extend_hold)
        if not ok:
            sess.next_extend_try = time.monotonic() + 10
            if sess.capped and not sess.warned:
                sess.warned = True
                await _send_json(sess.ws, {"type": "warning", "code": "key_spending_cap",
                                           "message": "This API key is about to reach its monthly spending limit; live processing will stop soon.",
                                           "secondsLeft": int(max(0, remaining))})
            elif not sess.warned:
                sess.warned = True
                bal = await asyncio.to_thread(ledger.balance, sess.p.user_id)
                await _send_json(sess.ws, {"type": "warning", "code": "low_balance",
                                           "message": "Your balance is running out; live processing will stop soon.",
                                           "secondsLeft": int(max(0, remaining)), "availableMicros": bal["available_micros"]})
    if remaining <= 0:
        sess.end_reason = "key_spending_cap" if sess.capped else "insufficient_funds"
        await _send_json(sess.ws, {"type": "stopped", "reason": sess.end_reason})
        return False
    return True


async def _finish(sess: Session) -> None:
    try:
        await _finish_inner(sess)
    finally:
        # forgotten first: the wait for the last captions below can be cancelled, and must never leave the session counted as live
        _live_ids.discard(sess.id)
        mine = _open.get(sess.p.user_id, [])
        if sess in mine:
            mine.remove(sess)
        await sess.close_outbox()


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
    translated = 0.0
    if sess.translator is not None:
        try:
            await sess.translator.close()
        except Exception:  # noqa: BLE001
            log.exception("closing live translations")
        translated = sess.translator.total_seconds()
    cost = 0
    try:
        if sess.hold_id and sess.paid_seconds > 0:
            paid = sess.paid_seconds
            cost = await asyncio.to_thread(
                ledger.capture_hold, sess.hold_id, sess.meter.micros(), f"Live session ({paid / 60:.1f} min)",
            )
    except Exception:  # noqa: BLE001 - e.g. the account was suspended and its holds released
        log.exception("billing stream session %s", sess.id)
    finally:
        if sess.hold_id:
            await asyncio.to_thread(ledger.release_hold, sess.hold_id)  # nothing left reserved, whatever happened
        if translated > 0:
            # Per second of speech actually translated, all languages together; small, so taken after the session.
            micros = pricing.cost_micros("stream_translate", "standard", translated)
            try:
                cost += await asyncio.to_thread(
                    ledger.debit_available, sess.p.user_id, micros, "usage", "stream", f"{sess.id}:translate",
                    f"Live caption translation ({translated / 60:.1f} min of speech)", allow_partial=True,
                )
            except Exception:  # noqa: BLE001
                log.exception("billing translations of stream session %s", sess.id)
        if sess.trial:
            try:
                await asyncio.to_thread(sess.save_progress)  # the rest of the trial seconds this session used
            except Exception:  # noqa: BLE001
                log.exception("saving the trial seconds of stream session %s", sess.id)
        fs = sess.frame_stats
        await asyncio.to_thread(
            db.execute,
            "UPDATE stream_sessions SET status = 'ended', seconds = ?, paid_seconds = ?, trial_seconds = ?, cost_micros = ?,"
            " end_reason = ?, ended_at = ?, frames = ?, late_frames = ?, dropped_frames = ?, meter_closed = ?, meter_rate = ?,"
            " meter_seconds = ? WHERE id = ?",
            (sess.seconds, sess.paid_seconds, sess.trial_used, cost, sess.end_reason, db.now(), fs.sent, fs.late, fs.dropped,
             sess.meter.closed, sess.meter.rate, sess.meter.seconds, sess.id),
        )
        await asyncio.to_thread(metrics.live_ended, sess.id, fs, time.monotonic() - sess.started)
    try:
        await sess.ws.close()
    except Exception:  # noqa: BLE001
        pass
