"""Live DSP worker: `python -m acw.stream.worker`.

Listens on a Unix socket for sessions forwarded by the web process. After
binding it drops filesystem write access and all TCP access (Landlock), so a
fault in native code here cannot reach the network or the database.
Each session gets its own thread (ordered processing); ctypes releases the
GIL inside the model, so sessions run in parallel across cores.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import hostload
from ..config import get_settings
from . import protocol as P
from .engine import DENOISE_TIERS, LiveChain, OpusCodec, UtteranceCaptioner, caption_backlog_seconds

log = logging.getLogger("acw.stream.worker")
WHISPER_LIVE_MODEL = os.environ.get("ACW_WHISPER_LIVE_MODEL", "small")

# Admission. Live audio is real time, so a session that doesn't fit right now
# waits in line instead of slowing every running stream down. "Fits" is
# measured on the host (acw.hostload): cores that normal-priority work could
# use (niced file jobs yield), minus a reserve for the rest of the machine,
# minus sessions admitted moments ago whose load doesn't show yet.
LIVE_RESERVE_CORES = float(os.environ.get("ACW_LIVE_RESERVE_CORES", "2.0"))
LIVE_MIN_MEMORY = 300 * 1024**2
# Local captions share one model pool; new caption sessions wait while
# utterances already queue for it longer than this.
CAPTION_BACKLOG_S = 2.0
MAX_WAIT_S = 15 * 60
# One thread per session: a sanity bound, far above what the CPU allows.
MAX_SESSIONS = int(os.environ.get("ACW_STREAM_MAX_SESSIONS", "200"))
# Cores per session by denoise tier: starting values from the benchmarks,
# then the running average of what sessions really use.
_cost: dict[str | None, float] = {"light": 0.13, "low_latency": 0.32, None: 0.03}
_recent: list[tuple[float, float]] = []  # (admitted at, estimated cores)
_waiting: list[object] = []
_active = 0
_monitor: hostload.HostLoad | None = None


def _estimate(cfg: dict) -> float:
    return _cost.get(cfg.get("denoise"), _cost["light"]) + 0.02


def _fits(cfg: dict) -> bool:
    if _active >= MAX_SESSIONS:
        return False
    if _monitor is None:
        return True
    snap = _monitor.snapshot()
    now = time.monotonic()
    _recent[:] = [r for r in _recent if now - r[0] < 10]
    pending = sum(c for _, c in _recent)
    if snap.memory_headroom < LIVE_MIN_MEMORY:
        return False
    if snap.cores_for_live - LIVE_RESERVE_CORES - pending < _estimate(cfg):
        return False
    return not (cfg.get("captions") == "local" and caption_backlog_seconds() > CAPTION_BACKLOG_S)


async def _wait_for_room(cfg: dict, writer: asyncio.StreamWriter) -> bool:
    """FIFO waiting line. Tells the client its place while it waits."""
    token = object()
    _waiting.append(token)
    start = time.monotonic()
    told, told_at = None, 0.0
    try:
        while True:
            place = _waiting.index(token) + 1
            if place == 1 and _fits(cfg):
                _recent.append((time.monotonic(), _estimate(cfg)))
                return True
            if time.monotonic() - start > MAX_WAIT_S:
                return False
            if place != told or time.monotonic() - told_at > 10:
                told, told_at = place, time.monotonic()
                await P.send_msg(writer, P.MSG_EVENT, P.pack_json({"type": "queued", "position": place}))
            await asyncio.sleep(1.0)
    finally:
        _waiting.remove(token)


def _learn_cost(tier: str | None, cpu_s: float, audio_s: float) -> None:
    if audio_s >= 10:
        _cost[tier] += 0.1 * (cpu_s / audio_s - _cost[tier])


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    global _active
    settings = get_settings()
    admitted = False
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live-dsp")
    chain = captioner = None
    events: asyncio.Queue = asyncio.Queue(maxsize=200)
    try:
        kind, payload = await P.read_msg(reader)
        if kind != P.MSG_START:
            return
        cfg = json.loads(payload)
        tier = cfg.get("denoise")
        if tier is not None and tier not in DENOISE_TIERS:
            await P.send_msg(writer, P.MSG_EVENT, P.pack_json({"type": "error", "code": "bad_tier"}))
            return
        if not await _wait_for_room(cfg, writer):
            log.warning("live session gave up waiting for room (%d running)", _active)
            await P.send_msg(writer, P.MSG_EVENT, P.pack_json({"type": "error", "code": "capacity"}))
            return
        _active += 1
        admitted = True
        chain = await loop.run_in_executor(
            executor, LiveChain, settings.bin_dir / "libdf.so", settings.models_dir, tier, float(cfg.get("atten_lim_db", 100.0))
        )
        if cfg.get("captions") == "local":
            def emit(ev: dict) -> None:
                loop.call_soon_threadsafe(_offer, events, ev)

            captioner = await loop.run_in_executor(
                executor, UtteranceCaptioner, settings.models_dir, WHISPER_LIVE_MODEL, cfg.get("language"), emit
            )
        await P.send_msg(writer, P.MSG_EVENT, P.pack_json({"type": "ready", "latencySamples": chain.latency_samples}))

        async def pump_events() -> None:
            while True:
                ev = await events.get()
                await P.send_msg(writer, P.MSG_EVENT, P.pack_json(ev))

        pump = asyncio.create_task(pump_events())

        opus: OpusCodec | None = None
        tap_pcm = bool(cfg.get("tap_pcm"))

        cpu_s = audio_s = 0.0

        def work(seq: int, kind: int, data):
            nonlocal opus, cpu_s, audio_s
            t0 = time.thread_time()
            if kind == P.CODEC_OPUS:
                if opus is None:
                    opus = OpusCodec(int(cfg.get("opus_bitrate", 96_000)))
                x = opus.decode(data)
            else:
                x = data
            y = chain.process(x)
            if captioner is not None:
                captioner.feed(y)
            out = P.pack_opus(seq, opus.encode(y)) if kind == P.CODEC_OPUS else P.pack_audio(seq, y)
            tap = P.pack_audio(seq, y) if (tap_pcm and kind == P.CODEC_OPUS) else None
            cpu_s += time.thread_time() - t0
            audio_s += len(y) / P.SAMPLE_RATE
            if audio_s >= 30:
                _learn_cost(tier, cpu_s, audio_s)
                cpu_s = audio_s = 0.0
            return out, tap

        try:
            while True:
                kind, payload = await P.read_msg(reader)
                if kind == P.MSG_AUDIO:
                    seq, akind, data = P.unpack_audio(payload)
                    out, tap = await loop.run_in_executor(executor, work, seq, akind, data)
                    await P.send_msg(writer, P.MSG_AUDIO, out)
                    if tap is not None:
                        await P.send_msg(writer, P.MSG_TAP, tap)
                elif kind == P.MSG_CONFIG:
                    c = json.loads(payload)
                    if "atten_lim_db" in c:
                        chain.set_atten_lim(float(c["atten_lim_db"]))
                elif kind == P.MSG_END:
                    if captioner is not None:
                        await loop.run_in_executor(executor, captioner.flush)
                        await asyncio.sleep(0.05)
                        while not events.empty():
                            await P.send_msg(writer, P.MSG_EVENT, P.pack_json(events.get_nowait()))
                    await P.send_msg(writer, P.MSG_END, b"")
                    break
        finally:
            pump.cancel()
    except (asyncio.IncompleteReadError, ConnectionError):
        pass
    except Exception:  # noqa: BLE001
        log.exception("stream session crashed")
    finally:
        if admitted:
            _active -= 1
        if captioner is not None:
            captioner.close()
        if chain is not None:
            executor.submit(chain.close)
        executor.shutdown(wait=False)
        writer.close()


def _offer(q: asyncio.Queue, ev: dict) -> None:
    try:
        q.put_nowait(ev)
    except asyncio.QueueFull:
        pass


async def serve() -> None:
    global _monitor
    settings = get_settings()
    _monitor = hostload.get(settings.data_dir)  # opens its /proc files before the sandbox closes them off
    sock = settings.stream_worker_socket
    sock.parent.mkdir(parents=True, exist_ok=True)
    sock.unlink(missing_ok=True)
    old = os.umask(0o177)
    server = await asyncio.start_unix_server(handle, path=str(sock))
    os.umask(old)
    from ..worker.sandbox import apply_landlock, default_read_paths

    abi = apply_landlock(default_read_paths([settings.bin_dir, settings.models_dir]), ["/dev/null"], required=False)
    log.info("stream worker listening on %s (landlock abi %s)", sock, abi)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for s in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(s, stop.set)
    async with server:
        await stop.wait()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(serve())


if __name__ == "__main__":
    main()
