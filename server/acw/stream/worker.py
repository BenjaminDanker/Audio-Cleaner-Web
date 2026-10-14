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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import cpusplit, hostload
from ..config import get_settings
from . import protocol as P
from .engine import DENOISE_TIERS, LiveChain, OpusCodec, UtteranceCaptioner

log = logging.getLogger("acw.stream.worker")
WHISPER_LIVE_MODEL = os.environ.get("ACW_WHISPER_LIVE_MODEL", "small")

# Capacity, measured on this machine (docs/operations.md): with live audio on
# its own physical cores (acw.cpusplit), about LIVE_UNITS standard denoise
# sessions keep every frame fast; a low-latency session costs ~3 of them. The
# capacity shrinks with other load measured on those cores (e.g. game servers).
# When full, a new session is refused and told what is still available rather
# than slowing everyone down. Sessions that only send audio to the external
# caption service run no model here and are never limited.
LIVE_UNITS = float(os.environ.get("ACW_LIVE_UNITS", "12"))
UNITS = {"light": 1.0, "low_latency": 3.0}
# Local captions share one CPU model that keeps up with about this many talkers.
MAX_LOCAL_CAPTIONS = int(os.environ.get("ACW_LIVE_MAX_LOCAL_CAPTIONS", "3"))
LIVE_MIN_MEMORY = 300 * 1024**2
MAX_SESSIONS = int(os.environ.get("ACW_STREAM_MAX_SESSIONS", "200"))  # thread sanity bound
_units = 0.0
_local_captions = 0
_active = 0
_monitor: hostload.HostLoad | None = None


def capacity_units() -> float:
    if _monitor is None:
        return LIVE_UNITS
    snap = _monitor.snapshot()
    live = cpusplit.live_cpus()
    other = max(0.0, sum(snap.busy_per_cpu[c] for c in live if c < len(snap.busy_per_cpu)) - snap.own_cores)
    return LIVE_UNITS * max(0.0, 1.0 - other / max(1, cpusplit.live_core_count()))


def admit(cfg: dict) -> tuple[bool, list[dict]]:
    """(admitted, options that would still fit if not)."""
    units = UNITS.get(cfg.get("denoise") or "", 0.0)
    local_captions = cfg.get("captions") == "local"
    if _active >= MAX_SESSIONS or (_monitor is not None and _monitor.snapshot().memory_headroom < LIVE_MIN_MEMORY):
        return False, []
    if not units and not local_captions:
        return True, []
    cap = capacity_units()
    denoise_ok = _units + units <= cap
    captions_ok = not local_captions or _local_captions < MAX_LOCAL_CAPTIONS
    if denoise_ok and captions_ok:
        return True, []
    options: list[dict] = []
    if not denoise_ok and units > UNITS["light"] and _units + UNITS["light"] <= cap:
        options.append({"denoise": "light"})
    if local_captions or not denoise_ok:
        options.append({"captions": "premium"})
    return False, options


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    global _active, _units, _local_captions
    settings = get_settings()
    admitted = False
    units, local_captions = 0.0, False
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
        ok, options = admit(cfg)
        if not ok:
            log.warning("live full: %d sessions, %.1f units in use, %d local caption sessions", _active, _units, _local_captions)
            await P.send_msg(writer, P.MSG_EVENT, P.pack_json({"type": "error", "code": "full", "available": options}))
            return
        units = UNITS.get(cfg.get("denoise") or "", 0.0)
        local_captions = cfg.get("captions") == "local"
        _active += 1
        _units += units
        _local_captions += local_captions
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

        def work(seq: int, kind: int, data):
            nonlocal opus
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
            _units -= units
            _local_captions -= local_captions
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
    # Live audio gets its own physical cores (file jobs move off them while
    # sessions run); threads created from here on inherit this.
    os.sched_setaffinity(0, cpusplit.live_cpus())
    cpusplit.physical_cores()  # read the topology before the sandbox
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
