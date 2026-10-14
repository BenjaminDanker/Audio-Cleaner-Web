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

from ..config import get_settings
from . import protocol as P
from .engine import DENOISE_TIERS, LiveChain, OpusCodec, UtteranceCaptioner

log = logging.getLogger("acw.stream.worker")
MAX_SESSIONS = int(os.environ.get("ACW_STREAM_MAX_SESSIONS", "24"))
WHISPER_LIVE_MODEL = os.environ.get("ACW_WHISPER_LIVE_MODEL", "small")
# Admission control: live audio is real time, so a new session is refused rather
# than letting every running session fall behind. Costs are measured CPU cores
# per session (docs/research/denoise.md, docs/research/asr.md); the budget leaves
# the rest of the 16 threads to file jobs and the other services on the host.
LIVE_CPU_BUDGET = float(os.environ.get("ACW_LIVE_CPU_BUDGET", "7.0"))
DENOISE_COST = {"light": 0.13, "low_latency": 0.32}
# Local captions share one whisper pool (2 workers); each talking stream keeps
# roughly a third of one worker busy, so they also have their own cap.
CAPTIONS_LOCAL_COST = 0.35
MAX_LOCAL_CAPTION_SESSIONS = int(os.environ.get("ACW_LIVE_MAX_LOCAL_CAPTIONS", "6"))
_active = 0
_load = 0.0
_local_captions = 0


def session_cost(cfg: dict) -> float:
    cost = 0.02  # codec + framing
    cost += DENOISE_COST.get(cfg.get("denoise") or "", 0.0)
    if cfg.get("captions") == "local":
        cost += CAPTIONS_LOCAL_COST
    return cost


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    global _active, _load, _local_captions
    settings = get_settings()
    cost = 0.0
    local_captions = False
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live-dsp")
    chain = captioner = None
    events: asyncio.Queue = asyncio.Queue(maxsize=200)
    _active += 1
    try:
        kind, payload = await P.read_msg(reader)
        if kind != P.MSG_START:
            return
        cfg = json.loads(payload)
        local_captions = cfg.get("captions") == "local"
        if (
            _active > MAX_SESSIONS
            or _load + session_cost(cfg) > LIVE_CPU_BUDGET
            or (local_captions and _local_captions >= MAX_LOCAL_CAPTION_SESSIONS)
        ):
            local_captions = False
            log.warning("live capacity reached: %d sessions, %.2f cores", _active, _load)
            await P.send_msg(writer, P.MSG_EVENT, P.pack_json({"type": "error", "code": "capacity"}))
            return
        cost = session_cost(cfg)
        _load += cost
        _local_captions += local_captions
        tier = cfg.get("denoise")
        if tier is not None and tier not in DENOISE_TIERS:
            await P.send_msg(writer, P.MSG_EVENT, P.pack_json({"type": "error", "code": "bad_tier"}))
            return
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
        _active -= 1
        _load = max(0.0, _load - cost)
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
    settings = get_settings()
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
