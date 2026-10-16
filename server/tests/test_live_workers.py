"""Live audio across several worker processes (2026-10-16 efficiency audit): sessions fill the workers in order and
move on when one is full, file jobs keep off the cores in use, and the cheaper per-session pieces (libopus, batched
caption audio, several spoken languages)."""
import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

from acw import cpusplit
from acw.billing import ledger
from acw.config import get_settings
from acw.stream import engine, routes
from acw.stream import protocol as P

from .test_live_trial import run_session


@pytest.fixture
def workers(monkeypatch, tmp_path):
    """Three live workers on their own sockets; `replies[name]` is what each answers to a start."""
    monkeypatch.setenv("ACW_STREAM_SOCKET", str(tmp_path / "stream.sock"))
    get_settings.cache_clear()
    for name in ("stream.sock", "stream-2.sock", "stream-3.sock", "stream-x.sock"):
        (tmp_path / name).touch()
    cores = {"stream.sock": [0, 1], "stream-2.sock": [2, 3], "stream-3.sock": [4, 5]}
    state = SimpleNamespace(replies={n: {"type": "ready", "latencySamples": 0, "cores": c} for n, c in cores.items()},
                            tried=[], started={})

    async def open_conn(path):
        name = path.rsplit("/", 1)[-1]
        state.tried.append(name)
        if state.replies.get(name) is None:
            raise ConnectionRefusedError(name)
        ended = asyncio.Event()
        return SimpleNamespace(name=name, first=True, ended=ended), SimpleNamespace(name=name, ended=ended, close=lambda: None)

    async def send(writer, kind, payload):
        if kind == P.MSG_START:
            state.started[writer.name] = json.loads(payload)
        elif kind == P.MSG_END:
            writer.ended.set()

    async def read(reader):
        if reader.first:
            reader.first = False
            return P.MSG_EVENT, P.pack_json(state.replies[reader.name])
        await reader.ended.wait()
        return P.MSG_END, b""

    monkeypatch.setattr(routes.asyncio, "open_unix_connection", open_conn)
    monkeypatch.setattr(routes.P, "send_msg", send)
    monkeypatch.setattr(routes.P, "read_msg", read)
    return state


def test_workers_are_found_by_their_sockets_in_order(workers):
    assert [p.name for p in routes.worker_sockets()] == ["stream.sock", "stream-2.sock", "stream-3.sock"]


def test_a_full_worker_passes_the_session_to_the_next(client, signed_in, workers):
    uid, _ = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    workers.replies["stream.sock"] = {"type": "error", "code": "full", "available": [{"captions": "premium"}]}
    workers.replies["stream-2.sock"] = None  # restarting
    got = run_session(client, 0.1)
    assert got[0]["type"] == "ready"
    assert workers.tried == ["stream.sock", "stream-2.sock", "stream-3.sock"] and list(workers.started) == ["stream.sock", "stream-3.sock"]


def test_when_every_worker_is_full_the_client_hears_what_still_fits(client, signed_in, workers):
    uid, _ = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    for name in workers.replies:
        workers.replies[name] = {"type": "error", "code": "full", "available": [{"captions": "premium"}]}
    got = run_session(client, 0.1, {"captions": "local"})
    assert got[0]["code"] == "full" and got[0]["available"] == [{"captions": "premium"}]


def test_the_cores_in_use_are_published_for_file_jobs(client, signed_in, workers, monkeypatch):
    uid, _ = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    seen = []
    monkeypatch.setattr(cpusplit, "set_live_units", lambda d, units, per_core, cores=None: seen.append((units, sorted(cores or []))))
    run_session(client, 0.1)
    assert (1.0, [0, 1]) in seen and seen[-1] == (0.0, [])


def test_live_cores_are_split_into_pairs(monkeypatch):
    monkeypatch.setattr(cpusplit, "max_live_cores", lambda: 7)
    monkeypatch.delenv("ACW_STREAM_WORKERS", raising=False)
    assert cpusplit.stream_workers() == 4
    assert [cpusplit.stream_worker_cores(i, 4) for i in range(1, 5)] == [[0, 1], [2, 3], [4, 5], [6]]
    monkeypatch.setenv("ACW_STREAM_WORKERS", "1")
    assert cpusplit.stream_workers() == 1 and cpusplit.stream_worker_cores(1, 1) == list(range(7))


def test_file_jobs_keep_off_busy_workers_and_one_spare_core(tmp_path, monkeypatch):
    monkeypatch.setattr(cpusplit, "physical_cores", lambda: [[i, i + 8] for i in range(8)])
    monkeypatch.setattr(cpusplit, "max_live_cores", lambda: 7)
    cpusplit.set_live_units(tmp_path, 3.0, 4.3, [2, 3])
    assert cpusplit.job_cpus_now(tmp_path) == {c for i in (1, 4, 5, 6, 7) for c in (i, i + 8)}  # core 0: the spare
    cpusplit.set_live_units(tmp_path, 0.0, 4.3, [])
    assert cpusplit.job_cpus_now(tmp_path) == cpusplit.all_cpus()
    (tmp_path / "live-units").write_text("3.0 2.4")  # written by an older web process: by units, as before
    assert cpusplit.job_cpus_now(tmp_path) == cpusplit.job_cpus(3.0, 2.4)


def test_libopus_round_trip_and_bad_packets():
    if engine._libopus() is None:
        pytest.skip("libopus not installed")
    x = (0.3 * np.sin(np.arange(48_000) * 2 * np.pi * 440 / 48_000)).astype(np.float32)
    enc, dec = engine.OpusCodec(96_000), engine.OpusCodec(encoder=False)
    y = np.concatenate([dec.decode(enc.encode(x[i:i + 960])) for i in range(0, len(x), 960)])
    assert len(y) == len(x) and abs(np.sqrt(np.mean(y[9600:] ** 2)) - np.sqrt(np.mean(x ** 2))) < 0.02
    with pytest.raises(ValueError):
        dec.decode(b"\xff\xff\xff")  # not a 20 ms packet
    enc.close()
    dec.close()


def test_extra_accurate_audio_goes_in_100_ms_pieces_and_is_committed_at_pauses():
    from acw.stream.openai_live import PremiumCaptions

    sent = []

    class FakeWs:
        async def send(self, m):
            sent.append(json.loads(m))

    p = PremiumCaptions(["en", "es"], lambda ev: None)
    assert p.languages == ["en", "es"]
    p.ws = FakeWs()
    talk = np.full(960, 0.1, np.float32)
    quiet = np.zeros(960, np.float32)

    async def run():
        for _ in range(10):  # 200 ms of talk
            await p.feed(talk)
        for _ in range(40):  # 800 ms quiet: the turn ends after 600 ms
            await p.feed(quiet)

    asyncio.run(run())
    appends = [m for m in sent if m["type"] == "input_audio_buffer.append"]
    assert len(appends) == 8  # 0.8 s sent (talk + the pause until the commit) in 100 ms pieces, not 40 messages
    assert sent[-1]["type"] == "input_audio_buffer.commit"
    assert sum(len(m["audio"]) for m in appends) * 3 // 4 == 40 * 480 * 2  # every 24 kHz sample went out once


def test_a_speaker_may_switch_between_up_to_three_languages(client, signed_in, workers, monkeypatch):
    uid, _ = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()
    made = []

    class FakePremium:
        def __init__(self, languages, *a, **k):
            made.append(languages)
            self.failed = False

        async def connect(self):
            pass

        async def feed(self, y):
            pass

        async def close(self):
            pass

    monkeypatch.setattr("acw.stream.openai_live.PremiumCaptions", FakePremium)
    got = run_session(client, 0.1, {"denoise": None, "captions": "premium", "language": ["en", "ES"], "translate": ["es"]})
    assert got[0]["type"] == "ready" and made == [["en", "es"]]  # a target that is one of the spoken languages is fine
    got = run_session(client, 0.1, {"captions": "premium", "language": ["en", "es", "de", "fr"]})
    assert got[0]["code"] == "bad_config"
    got = run_session(client, 0.1, {"captions": "local", "language": ["en", "es"]})
    assert got[0]["code"] == "bad_config" and "English only" in got[0]["message"]
