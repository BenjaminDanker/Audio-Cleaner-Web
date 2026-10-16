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
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from acw_common.aec import DELAY_SAMPLES as AEC_DELAY, EchoCanceller

from .. import cpusplit, hostload
from ..config import get_settings
from . import protocol as P
from .engine import DENOISE_TIERS, LiveChain, OpusCodec, StreamingCaptioner, fit

log = logging.getLogger("acw.stream.worker")

# Live audio comes first. It gets whole physical cores (acw.cpusplit), as many
# as its sessions need, up to all but the few kept for file jobs; capacity is
# that many cores times the calibrated units per core, shrunk by other load
# measured on those cores (e.g. game servers). When live would need more, a
# new session is refused and told what still fits. Sessions that only send
# audio to the external caption service run no model here: never limited.
LIVE_MIN_MEMORY = 300 * 1024**2
MAX_PENDING_REFS = 50  # reference frames kept for mic frames that haven't arrived (1 s)
# Free-trial sessions only start while live is less than this full, so a trial never takes the last places from
# people who pay.
TRIAL_CAPACITY_FRACTION = 0.7
MAX_SESSIONS = int(os.environ.get("ACW_STREAM_MAX_SESSIONS", "500"))  # thread sanity bound
_units = 0.0
_active = 0
_monitor: hostload.HostLoad | None = None


def capacity_units() -> float:
    cap = cpusplit.max_live_cores() * cpusplit.units_per_core()
    if _monitor is None:
        return cap
    snap = _monitor.snapshot()
    live = cpusplit.live_cpus()
    other = max(0.0, sum(snap.busy_per_cpu[c] for c in live if c < len(snap.busy_per_cpu)) - snap.own_cores)
    return cap * max(0.0, 1.0 - other / max(1, cpusplit.max_live_cores()))


def trial_busy(cfg: dict) -> bool:
    """A free-trial session (cfg["trial"]) is turned away while live use is at TRIAL_CAPACITY_FRACTION of its capacity."""
    return bool(cfg.get("trial")) and _units >= TRIAL_CAPACITY_FRACTION * capacity_units()


def admit(cfg: dict) -> tuple[bool, list[dict]]:
    """(admitted, options that would still fit if not)."""
    units = cpusplit.session_units(cfg)
    if _active >= MAX_SESSIONS or (_monitor is not None and _monitor.snapshot().memory_headroom < LIVE_MIN_MEMORY):
        return False, []
    if not cpusplit.session_units({**cfg, "echo_reference": False}):
        return True, []  # only external captions: echo cancellation alone is never worth refusing
    cap = capacity_units()
    if _units + units <= cap:
        return True, []
    options: list[dict] = []
    without_captions = cpusplit.session_units({**cfg, "captions": None})
    if cfg.get("captions") == "local" and _units + without_captions <= cap:
        options.append({"captions": "premium"})
    if cfg.get("captions") != "premium" and not any(o.get("captions") for o in options):
        options.append({"denoise": None, "captions": "premium"})
    return False, options


def _repin() -> None:
    """Keep this process on the live cores its sessions need (threads inherit)."""
    cpus = cpusplit.live_cpus(_units, cpusplit.units_per_core()) or cpusplit.live_cpus()
    # By native thread id: /proc is out of reach once the process is sandboxed.
    for t in threading.enumerate():
        if t.native_id is not None:
            try:
                os.sched_setaffinity(t.native_id, cpus)
            except OSError:
                pass


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    global _active, _units
    settings = get_settings()
    admitted = False
    units = 0.0
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live-dsp")
    chain = captioner = aec = None
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
        if trial_busy(cfg):
            await P.send_msg(writer, P.MSG_EVENT, P.pack_json({"type": "error", "code": "trial_busy"}))
            return
        ok, options = admit(cfg)
        if not ok:
            log.warning("live full: %d sessions, %.1f of %.1f units", _active, _units, capacity_units())
            await P.send_msg(writer, P.MSG_EVENT, P.pack_json({"type": "error", "code": "full", "available": options}))
            return
        units = cpusplit.session_units(cfg)
        _active += 1
        _units += units
        admitted = True
        _repin()
        chain = await loop.run_in_executor(
            executor, LiveChain, settings.bin_dir / "libdf.so", settings.models_dir, tier, float(cfg.get("atten_lim_db", 100.0))
        )
        if cfg.get("captions") == "local":
            def emit(ev: dict) -> None:
                loop.call_soon_threadsafe(_offer, events, ev)

            captioner = await loop.run_in_executor(
                executor, StreamingCaptioner, settings.models_dir, emit
            )
        if cfg.get("echo_reference"):
            aec = await loop.run_in_executor(executor, EchoCanceller)
        latency = chain.latency_samples + (AEC_DELAY if aec else 0)
        await P.send_msg(writer, P.MSG_EVENT, P.pack_json({"type": "ready", "latencySamples": latency}))

        async def pump_events() -> None:
            while True:
                ev = await events.get()
                await P.send_msg(writer, P.MSG_EVENT, P.pack_json(ev))

        pump = asyncio.create_task(pump_events())

        opus: OpusCodec | None = None
        ref_opus: OpusCodec | None = None
        refs: OrderedDict[int, np.ndarray] = OrderedDict()  # echo reference by mic frame seq, until its mic frame comes
        tap_pcm = bool(cfg.get("tap_pcm"))

        def take_reference(seq: int, kind: int, data) -> None:
            nonlocal ref_opus
            if kind == P.CODEC_OPUS:
                if ref_opus is None:
                    ref_opus = OpusCodec(encoder=False)
                data = ref_opus.decode(data)
            refs[seq] = data
            while len(refs) > MAX_PENDING_REFS:
                refs.popitem(last=False)

        def work(seq: int, kind: int, data):
            nonlocal opus
            if kind == P.CODEC_OPUS:
                if opus is None:
                    opus = OpusCodec(int(cfg.get("opus_bitrate", 96_000)))
                x = opus.decode(data)
            else:
                x = data
            if aec is not None:  # a missing (or late) reference counts as silence
                x = aec.process(x, fit(refs.pop(seq, None), len(x)))
            y = chain.process(x)
            if captioner is not None:
                captioner.feed(x)  # the original (echo removed): recognizers do better on it than on denoised audio
            out = P.pack_opus(seq, opus.encode(y)) if kind == P.CODEC_OPUS else P.pack_audio(seq, y)
            tap = P.pack_audio(seq, x) if tap_pcm else None  # extra-accurate captions hear the original, not the denoised audio
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
                elif kind == P.MSG_REF:
                    if aec is not None:
                        seq, akind, data = P.unpack_audio(payload)
                        await loop.run_in_executor(executor, take_reference, seq, akind, data)
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
    except ValueError as e:  # e.g. an Opus packet that isn't 20 ms: that session ends, quietly
        log.warning("stream session ended: %s", e)
    except Exception:  # noqa: BLE001
        log.exception("stream session crashed")
    finally:
        if admitted:
            _active -= 1
            _units = max(0.0, _units - units)
            _repin()
        if captioner is not None:
            executor.submit(captioner.close)  # waits for the recogniser's current tick: not on the event loop
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
    # Read the topology and calibrate before the sandbox closes /sys off.
    cpusplit.physical_cores()
    log.info("live capacity: %d cores x %.2f units", cpusplit.max_live_cores(), cpusplit.units_per_core())
    os.sched_setaffinity(0, cpusplit.live_cpus())
    _monitor = hostload.get(settings.data_dir)  # opens its /proc files before the sandbox closes them off
    sock = settings.stream_worker_socket
    sock.parent.mkdir(parents=True, exist_ok=True)
    sock.unlink(missing_ok=True)
    old = os.umask(0o177)
    server = await asyncio.start_unix_server(handle, path=str(sock))
    os.umask(old)
    from ..worker.sandbox import apply_landlock, default_read_paths

    abi = apply_landlock(default_read_paths([settings.bin_dir, settings.models_dir]), ["/dev/null"], required=settings.production)  # production fails closed
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
