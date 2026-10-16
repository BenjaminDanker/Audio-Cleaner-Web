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
import sys
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

# Live audio comes first. It runs in several worker processes (acw.cpusplit.stream_workers), each pinned to its
# own pair of live cores; the web process fills them in order, so the cores in use stay together and file jobs
# move off them (acw.cpusplit.job_cpus_now). A worker's capacity is its cores times the calibrated units per core,
# shrunk by other load measured on those cores (e.g. game servers). When a worker is full the web process tries the
# next; when all are, a new session is refused and told what still fits. Sessions that only send audio to the
# external caption service run no model here: never limited.
LIVE_MIN_MEMORY = 300 * 1024**2
MAX_PENDING_REFS = 50  # reference frames kept for mic frames that haven't arrived (1 s)
TRIAL_CAPACITY_FRACTION = cpusplit.TRIAL_CAPACITY_FRACTION
MAX_SESSIONS = int(os.environ.get("ACW_STREAM_MAX_SESSIONS", "500"))  # thread sanity bound
# Each worker has one shared caption decoder, which does at most about 13x real time in total however many streams
# share it (2026-10-16, one decoder: 12 talking streams already crept, 24 grew the worker's memory until the kernel
# killed it). Silence isn't decoded (StreamingCaptioner), so with real pauses a worker on two cores kept 10 streams
# within 0.3 s (LibriSpeech with pauses, 30 streams on 3 workers); its own cap keeps a stream that talks all the time
# from pushing the others behind. ACW_STREAM_MAX_CAPTION_STREAMS sets it per worker.
CAPTION_STREAMS_PER_CORE = 5
MAX_CAPTION_STREAMS = int(os.environ.get("ACW_STREAM_MAX_CAPTION_STREAMS", "0"))  # 0: CAPTION_STREAMS_PER_CORE x its cores


def caption_cap() -> int:
    return MAX_CAPTION_STREAMS or CAPTION_STREAMS_PER_CORE * max(1, len(_cores) or cpusplit.max_live_cores())


_units = 0.0
_active = 0
_caption_streams = 0
_monitor: hostload.HostLoad | None = None
_index = 1  # this worker, 1-based
_cores: list[int] = []  # its physical cores (indexes into cpusplit.physical_cores())


def capacity_units() -> float:
    cores = _cores or list(range(cpusplit.max_live_cores()))
    cap = len(cores) * cpusplit.units_per_core()
    if _monitor is None:
        return cap
    snap = _monitor.snapshot()
    live = cpusplit.cpus_of(cores)
    other = max(0.0, sum(snap.busy_per_cpu[c] for c in live if c < len(snap.busy_per_cpu)) - snap.own_cores)
    return cap * max(0.0, 1.0 - other / len(cores))


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
    if cfg.get("captions") == "local" and _caption_streams >= caption_cap():
        # Standard captions are full whatever the units say: keep the rest of the setup if it fits, else extra-accurate alone.
        without_captions = cpusplit.session_units({**cfg, "captions": None})
        if _units + without_captions <= capacity_units():
            return False, [{"captions": "premium"}]
        return False, [{"denoise": None, "captions": "premium"}]
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


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    global _active, _units, _caption_streams
    settings = get_settings()
    admitted = False
    counted_captions = False
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
        if cfg.get("captions") == "local":
            _caption_streams += 1
            counted_captions = True
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
        await P.send_msg(writer, P.MSG_EVENT, P.pack_json({"type": "ready", "latencySamples": latency,
                                                           "worker": _index, "cores": _cores}))

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
                # The original (echo removed): recognizers do better on it than on denoised audio. The denoised level
                # tells speech from background noise, so the decoder skips the silences between.
                level = 20 * np.log10(float(np.sqrt(np.mean(y * y))) + 1e-9) if getattr(chain, "df", None) is not None else None
                captioner.feed(x, level)
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
            if counted_captions:
                _caption_streams = max(0, _caption_streams - 1)
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


def socket_path(index: int):
    """Worker 1 listens on the configured socket (ACW_STREAM_SOCKET), worker n on stream-<n>.sock next to it."""
    base = get_settings().stream_worker_socket
    return base if index == 1 else base.with_name(f"{base.stem}-{index}{base.suffix}")


async def serve(index: int = 1, workers: int = 1) -> None:
    global _monitor, _index, _cores
    settings = get_settings()
    # Read the topology and calibrate before the sandbox closes /sys off.
    cpusplit.physical_cores()
    _index = index
    _cores = cpusplit.stream_worker_cores(index, workers)
    os.sched_setaffinity(0, cpusplit.cpus_of(_cores))  # every thread started from here on inherits it
    log.info("live worker %d of %d: cores %s x %.2f units", index, workers, _cores, cpusplit.units_per_core())
    _monitor = hostload.get(settings.data_dir)  # opens its /proc files before the sandbox closes them off
    sock = socket_path(index)
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


def supervise(workers: int) -> None:
    """Runs `workers` worker processes (python -m acw.stream.worker with ACW_STREAM_WORKER_INDEX), restarts one that
    ends, and stops them all on SIGTERM. One process per pair of cores: a single Python process falls behind at about
    ten sessions (its GIL), however many cores it has."""
    import subprocess
    import time

    cpusplit.units_per_core()  # calibrate once here, before the workers start (in parallel they'd slow each other)
    base = get_settings().stream_worker_socket
    for stale in base.parent.glob(f"{base.stem}-*{base.suffix}"):  # left by an earlier run with more workers
        n = stale.stem.rsplit("-", 1)[-1]
        if not n.isdigit() or int(n) > workers:
            stale.unlink(missing_ok=True)
    stopping = False
    procs: dict[int, subprocess.Popen] = {}

    def start(i: int) -> None:
        env = {**os.environ, "ACW_STREAM_WORKER_INDEX": str(i), "ACW_STREAM_WORKERS": str(workers)}
        procs[i] = subprocess.Popen([sys.executable, "-m", "acw.stream.worker"], env=env)

    def stop(*_a) -> None:
        nonlocal stopping
        stopping = True
        for p in procs.values():
            if p.poll() is None:
                p.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for i in range(1, workers + 1):
        start(i)
    restarts: dict[int, float] = {}
    while not stopping:
        time.sleep(1.0)
        for i, p in list(procs.items()):
            if p.poll() is not None and not stopping:
                # Its sessions ended with it (the clients reconnect, to this or another worker). Not too often.
                if time.monotonic() - restarts.get(i, 0.0) < 2.0:
                    continue
                log.warning("live worker %d exited (%s); starting it again", i, p.returncode)
                restarts[i] = time.monotonic()
                start(i)
    deadline = time.monotonic() + 10
    for p in procs.values():
        try:
            p.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            p.kill()


def main() -> None:
    # Each session's thread needs the GIL back after every native call (model, codec): a shorter switch interval keeps
    # one session's Python work from holding the others' frames for up to the default 5 ms.
    sys.setswitchinterval(0.001)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    index = os.environ.get("ACW_STREAM_WORKER_INDEX", "")
    workers = cpusplit.stream_workers()
    if index.isdigit():
        asyncio.run(serve(int(index), workers))
    elif workers == 1:
        asyncio.run(serve(1, 1))
    else:
        supervise(workers)


if __name__ == "__main__":
    main()
