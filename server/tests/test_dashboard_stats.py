"""acw.dashboard_stats: the numbers the web process sends to Netdata for the home dashboard (jobs per feature, live sessions, GPU worker)."""
import asyncio
import json
import logging
import socket
import types

import pytest

from acw import broker, dashboard_stats as ds, db


def add_user():
    uid = db.new_id()
    db.execute("INSERT INTO users(id, email, created_at, status, balance_micros) VALUES (?,?,?,?,?)",
               (uid, f"{uid[:8]}@example.com", db.now(), "active", 0))
    return uid


def add_job(uid, status, kind="denoise", params=None, raw=None):
    db.execute(
        "INSERT INTO jobs(id, user_id, kind, tier, status, params_json, input_name, input_bytes, claimed_seconds, created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (db.new_id(), uid, kind, "standard", status, raw if raw is not None else json.dumps(params if params is not None else {"denoise_tier": "standard"}),
         "a.wav", 1, 1.0, db.now()))


def add_session(uid, status="active", denoise=None, captions=None):
    db.execute("INSERT INTO stream_sessions(id, user_id, denoise_tier, caption_tier, status, started_at) VALUES (?,?,?,?,?,?)",
               (db.new_id(), uid, denoise, captions, status, db.now()))


@pytest.fixture
def gpu(monkeypatch):
    state = {"up": False, "idle": False}
    monkeypatch.setattr(broker, "gpu_worker_up", lambda: state["up"])
    monkeypatch.setattr(broker, "gpu_worker_idle", lambda: state["idle"])
    return state


# ---------------------------------------------------------------------------------------------------------- slug / features

def test_slug_makes_a_metric_name():
    assert ds.slug("Subtitles & transcript, extra accurate") == "subtitles_transcript_extra_accurate"
    assert ds.slug("  Live captions ") == "live_captions"
    assert ds.slug("Voice rebuild") == "voice_rebuild"


def test_features_use_the_websites_labels():
    assert ds.features("denoise", {"denoise_tier": "standard"}) == ["Denoising"]
    assert ds.features("denoise", {"denoise_tier": "rebuild"}) == ["Voice rebuild"]
    assert ds.features("denoise_transcribe", {"denoise_tier": "standard", "transcribe_tier": "premium"}) == [
        "Denoising", "Subtitles & transcript, extra accurate"]
    assert ds.features("transcribe", {"transcribe_tier": "standard"}) == ["Subtitles & transcript"]
    assert ds.features("denoise", {"denoise_tier": "off"}) == ["Leveling"]


def test_a_convert_job_is_video_conversion():
    assert ds.features("convert", {}) == ["Video conversion"]


def test_translation_and_echo_removal_are_added_from_params_or_options():
    p = {"denoise_tier": "standard"}
    assert ds.features("denoise", {**p, "translate": ["de"]}) == ["Denoising", "Translation"]
    assert ds.features("denoise", {**p, "options": {"translate": ["de", "fr"]}}) == ["Denoising", "Translation"]
    assert ds.features("denoise", {**p, "options": {"echo_reference_track": True}}) == ["Denoising", "Echo removal"]
    assert ds.features("denoise", {**p, "echo_reference_track": True, "translate": ["de"]}) == ["Denoising", "Translation", "Echo removal"]
    # empty / false / missing values add nothing
    assert ds.features("denoise", {**p, "translate": [], "echo_reference_track": False}) == ["Denoising"]
    assert ds.features("denoise", {**p, "options": {"translate": [], "echo_reference_track": False}}) == ["Denoising"]
    assert ds.features("denoise", {**p, "options": None}) == ["Denoising"]


def test_features_are_not_repeated_and_keep_their_order():
    # the same feature named by the label and by the options (top-level and nested) is listed once
    out = ds.features("denoise", {"denoise_tier": "standard", "translate": ["de"], "options": {"translate": ["fr"], "echo_reference_track": True}})
    assert out == ["Denoising", "Translation", "Echo removal"]
    assert len(out) == len(set(out))


def test_features_dedupe_a_label_that_repeats(monkeypatch):
    monkeypatch.setattr(ds.job_view, "label_of", lambda kind, params: "Translation + Denoising + Translation")
    assert ds.features("x", {"translate": ["de"], "echo_reference_track": True}) == ["Translation", "Denoising", "Echo removal"]


# ---------------------------------------------------------------------------------------------------------- live_features

@pytest.mark.parametrize("denoise, captions, expected", [
    ("standard", None, ["Live noise removal"]),
    ("premium", None, ["Live noise removal"]),
    (None, "premium", ["Live captions, extra accurate"]),
    (None, "standard", ["Live captions"]),
    ("standard", "premium", ["Live noise removal", "Live captions, extra accurate"]),
    ("standard", "standard", ["Live noise removal", "Live captions"]),
    (None, None, []),
    ("none", "none", []),
    ("off", "off", []),
    ("", "", []),
    ("off", "premium", ["Live captions, extra accurate"]),
    ("standard", "off", ["Live noise removal"]),
])
def test_live_features(denoise, captions, expected):
    assert ds.live_features(denoise, captions) == expected


# ---------------------------------------------------------------------------------------------------------- collect

def test_an_idle_system_sends_every_known_key_at_zero(gpu):
    g = ds.collect()
    for name in ds.FEATURES:
        assert g[f"job_running_{ds.slug(name)}"] == 0
        assert g[f"job_waiting_{ds.slug(name)}"] == 0
    for name in ds.LIVE:
        assert g[f"live_{ds.slug(name)}"] == 0
    assert g["jobs_running"] == g["jobs_waiting"] == g["live_sessions"] == g["gpu_worker"] == 0
    assert len(g) == 2 * len(ds.FEATURES) + len(ds.LIVE) + 4


def test_queued_jobs_wait_and_running_and_cloud_asr_jobs_run(gpu):
    uid = add_user()
    add_job(uid, "queued")
    add_job(uid, "queued")
    add_job(uid, "running")
    add_job(uid, "cloud_asr")
    g = ds.collect()
    assert g["job_waiting_denoising"] == 2
    assert g["job_running_denoising"] == 2
    assert g["jobs_waiting"] == 2 and g["jobs_running"] == 2


def test_other_states_are_not_counted(gpu):
    uid = add_user()
    for status in ("uploading", "done", "failed", "canceled", "expired"):
        add_job(uid, status)
    g = ds.collect()
    assert g["job_waiting_denoising"] == g["job_running_denoising"] == 0
    assert g["jobs_running"] == g["jobs_waiting"] == 0


def test_a_job_with_several_features_counts_once_per_feature(gpu):
    uid = add_user()
    add_job(uid, "running", "denoise_transcribe",
            {"denoise_tier": "standard", "transcribe_tier": "premium", "translate": ["de"], "options": {"echo_reference_track": True}})
    g = ds.collect()
    for key in ("denoising", "subtitles_transcript_extra_accurate", "translation", "echo_removal"):
        assert g[f"job_running_{key}"] == 1, key
    assert g["job_running_leveling"] == 0 and g["job_running_voice_rebuild"] == 0
    assert g["jobs_running"] == 1  # but it is one job in the total


def test_convert_and_other_known_features(gpu):
    uid = add_user()
    add_job(uid, "queued", "convert", {})
    add_job(uid, "running", "transcribe", {"transcribe_tier": "standard"})
    add_job(uid, "running", "denoise", {"denoise_tier": "rebuild"})
    g = ds.collect()
    assert g["job_waiting_video_conversion"] == 1
    assert g["job_running_subtitles_transcript"] == 1
    assert g["job_running_voice_rebuild"] == 1


def test_a_feature_that_is_not_in_the_list_gets_both_keys(gpu, monkeypatch):
    monkeypatch.setattr(ds.job_view, "label_of", lambda kind, params: "Brand new thing")
    uid = add_user()
    add_job(uid, "queued")
    g = ds.collect()
    assert g["job_waiting_brand_new_thing"] == 1
    assert g["job_running_brand_new_thing"] == 0
    # and for a running one the other way round
    add_job(uid, "running")
    add_job(uid, "running")
    g = ds.collect()
    assert g["job_waiting_brand_new_thing"] == 1 and g["job_running_brand_new_thing"] == 2


def test_live_sessions_count_only_active_ones(gpu):
    uid = add_user()
    add_session(uid, denoise="standard")
    add_session(uid, denoise="standard", captions="premium")
    add_session(uid, captions="standard")
    add_session(uid, status="ended", denoise="standard", captions="premium")
    add_session(uid, denoise="off", captions="none")  # active, but with no feature
    g = ds.collect()
    assert g["live_sessions"] == 4
    assert g["live_live_noise_removal"] == 2
    assert g["live_live_captions"] == 1
    assert g["live_live_captions_extra_accurate"] == 1


@pytest.mark.parametrize("up, idle, expected", [(False, False, 0), (False, True, 0), (True, True, 1), (True, False, 2)])
def test_gpu_worker_state(gpu, up, idle, expected):
    gpu.update(up=up, idle=idle)
    assert ds.collect()["gpu_worker"] == expected


def test_a_job_with_broken_params_is_skipped_not_fatal(gpu):
    uid = add_user()
    add_job(uid, "queued", raw="{not json")
    add_job(uid, "running", raw="")
    add_job(uid, "running", raw='{"options": ["x"]}')  # options that are not a mapping
    add_job(uid, "queued")
    g = ds.collect()
    assert g["job_waiting_denoising"] == 1
    assert g["job_running_denoising"] == 0
    assert g["jobs_waiting"] == 2 and g["jobs_running"] == 2  # the totals still count every queued/running row


@pytest.mark.parametrize("raw", ["null", "[]", "5", '"text"'])
def test_params_that_are_json_but_not_an_object_are_skipped(gpu, raw):
    uid = add_user()
    add_job(uid, "queued", raw=raw)
    add_job(uid, "queued")
    assert ds.collect()["job_waiting_denoising"] == 1


# ---------------------------------------------------------------------------------------------------------- target

@pytest.mark.parametrize("raw, expected", [
    (None, None),
    ("", None),
    ("   ", None),
    ("127.0.0.1:8125", ("127.0.0.1", 8125)),
    (" 127.0.0.1:8125 ", ("127.0.0.1", 8125)),
    (":9000", ("127.0.0.1", 9000)),
    ("localhost:8125", ("localhost", 8125)),
    ("nonsense", None),
    ("host:port", None),
    ("127.0.0.1:", None),
])
def test_target(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("ACW_DASHBOARD_STATSD", raising=False)
    else:
        monkeypatch.setenv("ACW_DASHBOARD_STATSD", raw)
    assert ds.target() == expected


def test_a_bad_target_is_logged(monkeypatch, caplog):
    monkeypatch.setenv("ACW_DASHBOARD_STATSD", "nonsense")
    with caplog.at_level(logging.WARNING, logger="acw.dashboard_stats"):
        assert ds.target() is None
    assert "nonsense" in caplog.text


# ---------------------------------------------------------------------------------------------------------- send

def receiver():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    s.settimeout(2)
    return s


def drain(s, n):
    return [s.recv(1024).decode() for _ in range(n)]


def test_send_writes_one_statsd_gauge_line_per_key():
    with receiver() as rx:
        ds.send({"jobs_running": 3, "job_waiting_denoising": 0, "gpu_worker": 2}, rx.getsockname())
        assert drain(rx, 3) == ["acw.jobs_running:3|g", "acw.job_waiting_denoising:0|g", "acw.gpu_worker:2|g"]


def test_send_sends_the_whole_collect(gpu):
    uid = add_user()
    add_job(uid, "running")
    g = ds.collect()
    with receiver() as rx:
        ds.send(g, rx.getsockname())
        lines = drain(rx, len(g))
    assert sorted(lines) == sorted(f"acw.{k}:{v}|g" for k, v in g.items())
    assert "acw.job_running_denoising:1|g" in lines


# ---------------------------------------------------------------------------------------------------------- send_loop

def fake_clock_rounds(monkeypatch, rounds, start=1000.0):
    """Runs send_loop for `rounds` rounds on a fake clock. loop.time() reads the clock, and each asyncio.sleep moves it on
    by the seconds slept, so round k starts at start + (k - 1) while SEND_S is 1. The `rounds`-th sleep cancels the loop.
    Returns (sleeps, clock): the argument of every sleep, and the clock."""
    clock = types.SimpleNamespace(now=start)
    sleeps = []
    real_get_running_loop = asyncio.get_running_loop

    class LoopWithFakeTime:
        def __init__(self, loop):
            self._loop = loop

        def time(self):
            return clock.now

        def __getattr__(self, name):
            return getattr(self._loop, name)

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        clock.now += seconds
        if len(sleeps) >= rounds:
            raise asyncio.CancelledError

    monkeypatch.setattr(ds.asyncio, "get_running_loop", lambda: LoopWithFakeTime(real_get_running_loop()))
    monkeypatch.setattr(ds.asyncio, "sleep", fake_sleep)
    return sleeps, clock


def test_the_loop_returns_at_once_when_off(monkeypatch):
    monkeypatch.delenv("ACW_DASHBOARD_STATSD", raising=False)

    def boom(*a, **k):
        raise AssertionError("must not be used")
    for name in ("collect", "send"):
        monkeypatch.setattr(ds, name, boom)
    monkeypatch.setattr(ds, "socket", types.SimpleNamespace(socket=boom))  # (not the real module: the event loop needs it)
    loop = asyncio.new_event_loop()
    try:
        monkeypatch.setattr(ds.asyncio, "sleep", boom)
        assert loop.run_until_complete(ds.send_loop()) is None
    finally:
        loop.close()




def test_the_loop_sends_a_round_then_sleeps_the_send_interval(monkeypatch, gpu):
    with receiver() as rx:
        host, port = rx.getsockname()
        monkeypatch.setenv("ACW_DASHBOARD_STATSD", f"{host}:{port}")
        sleeps, _ = fake_clock_rounds(monkeypatch, 1)
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(ds.send_loop())
        assert sleeps == [ds.SEND_S]
        n = len(ds.collect())
        lines = drain(rx, n)
    assert "acw.jobs_running:0|g" in lines and "acw.gpu_worker:0|g" in lines


def test_rounds_within_the_interval_resend_the_count_and_it_is_counted_again_after_it(monkeypatch):
    monkeypatch.setenv("ACW_DASHBOARD_STATSD", "127.0.0.1:9")
    counts = iter([{"x": 1}, {"x": 2}])
    collected = []

    def collect():
        collected.append(1)
        return next(counts)
    monkeypatch.setattr(ds, "collect", collect)
    sent = []
    monkeypatch.setattr(ds, "send", lambda g, a: sent.append(g))
    # rounds start at t, t+1, ... t+10: counted at t and at t+10 (INTERVAL_S later), the rounds between resend the count
    sleeps, _ = fake_clock_rounds(monkeypatch, 11)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(ds.send_loop())
    assert sleeps == [ds.SEND_S] * 11
    assert len(collected) == 2
    assert sent == [{"x": 1}] * 10 + [{"x": 2}]


def test_a_failing_count_drops_the_stored_count_until_the_next_good_one(monkeypatch):
    monkeypatch.setenv("ACW_DASHBOARD_STATSD", "127.0.0.1:9")
    outcomes = iter([{"x": 1}, RuntimeError("boom"), {"x": 3}])

    def collect():
        o = next(outcomes)
        if isinstance(o, Exception):
            raise o
        return o
    monkeypatch.setattr(ds, "collect", collect)
    sent = []
    monkeypatch.setattr(ds, "send", lambda g, a: sent.append(g))
    fake_clock_rounds(monkeypatch, 21)  # counts at t, t+10, t+20
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(ds.send_loop())
    # the count of t is sent until t+10; the failing count at t+10 sends nothing until the good one at t+20
    assert sent == [{"x": 1}] * 10 + [{"x": 3}]


def test_a_failing_count_sends_nothing_and_is_logged_once_within_one_interval(monkeypatch, caplog):
    monkeypatch.setenv("ACW_DASHBOARD_STATSD", "127.0.0.1:9")
    runs = []

    def broken():
        runs.append(1)
        raise RuntimeError("database is locked")
    monkeypatch.setattr(ds, "collect", broken)
    sent = []
    monkeypatch.setattr(ds, "send", lambda g, a: sent.append(g))
    sleeps, _ = fake_clock_rounds(monkeypatch, 5)  # one count, then four rounds that do not count
    with caplog.at_level(logging.WARNING, logger="acw.dashboard_stats"):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(ds.send_loop())
    assert sleeps == [ds.SEND_S] * 5  # the loop went on every round
    assert len(runs) == 1 and not sent
    records = [r for r in caplog.records if r.name == "acw.dashboard_stats"]
    assert len(records) == 1 and "database is locked" in caplog.text


def test_a_failing_count_is_logged_once_across_several_intervals(monkeypatch, caplog):
    monkeypatch.setenv("ACW_DASHBOARD_STATSD", "127.0.0.1:9")
    runs = []

    def broken():
        runs.append(1)
        raise RuntimeError("database is locked")
    monkeypatch.setattr(ds, "collect", broken)
    sent = []
    monkeypatch.setattr(ds, "send", lambda g, a: sent.append(g))
    fake_clock_rounds(monkeypatch, 21)  # counts at t, t+10, t+20: all fail
    with caplog.at_level(logging.WARNING, logger="acw.dashboard_stats"):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(ds.send_loop())
    assert len(runs) == 3 and not sent
    records = [r for r in caplog.records if r.name == "acw.dashboard_stats"]
    assert len(records) == 1


def test_the_log_is_armed_again_after_a_good_count(monkeypatch, caplog):
    monkeypatch.setenv("ACW_DASHBOARD_STATSD", "127.0.0.1:9")
    outcomes = iter([RuntimeError("a"), {"x": 1}, RuntimeError("c")])

    def collect():
        o = next(outcomes)
        if isinstance(o, Exception):
            raise o
        return o
    monkeypatch.setattr(ds, "collect", collect)
    monkeypatch.setattr(ds, "send", lambda g, a: None)
    fake_clock_rounds(monkeypatch, 21)  # counts at t (fails), t+10 (good), t+20 (fails)
    with caplog.at_level(logging.WARNING, logger="acw.dashboard_stats"):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(ds.send_loop())
    assert [r.exc_info[1].args[0] for r in caplog.records if r.name == "acw.dashboard_stats"] == ["a", "c"]
