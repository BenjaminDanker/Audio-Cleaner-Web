"""acw.dashboard_stats: the numbers the web process POSTs to the home service (jobs per feature, live sessions, GPU worker, money summary)."""
import asyncio
import json
import logging
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
    ("\n", None),
    ("http://192.168.1.200/api/acw", "http://192.168.1.200/api/acw"),
    ("  http://192.168.1.200/api/acw \n", "http://192.168.1.200/api/acw"),
])
def test_target(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("ACW_DASHBOARD_URL", raising=False)
    else:
        monkeypatch.setenv("ACW_DASHBOARD_URL", raw)
    assert ds.target() == expected


def test_the_old_statsd_variable_is_not_used(monkeypatch):
    monkeypatch.delenv("ACW_DASHBOARD_URL", raising=False)
    monkeypatch.setenv("ACW_DASHBOARD_STATSD", "127.0.0.1:8125")
    assert ds.target() is None


# ---------------------------------------------------------------------------------------------------------- post

class Sink:
    """A local HTTP server on 127.0.0.1 that records every request and answers with `status`."""

    def __init__(self, status=200):
        self.requests = []
        self.status = status
        sink = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                sink.requests.append({"method": self.command, "path": self.path, "type": self.headers.get("Content-Type"), "body": body})
                self.send_response(sink.status)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"ok")

            do_PUT = do_GET = do_POST

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/api/acw"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)


@pytest.fixture(autouse=True)
def no_proxy(monkeypatch):
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")


def test_post_sends_the_body_as_json(no_proxy):
    body = {"live": {"jobs_running": 3, "gpu_worker": 2}, "money": {"month": {"a": 1.5}, "series": [{"t": 1}]}}
    with Sink() as sink:
        assert ds.post(sink.url, body) is None
    assert len(sink.requests) == 1
    r = sink.requests[0]
    assert r["method"] == "POST" and r["path"] == "/api/acw"
    assert r["type"] == "application/json"
    assert json.loads(r["body"]) == body


def test_post_turns_values_json_cannot_hold_into_text(no_proxy):
    with Sink() as sink:
        ds.post(sink.url, {"live": {"x": 1}, "odd": {1, 2}})
    got = json.loads(sink.requests[0]["body"])
    assert got["live"] == {"x": 1} and isinstance(got["odd"], str)


def test_post_raises_when_the_service_answers_with_an_error(no_proxy):
    import urllib.error
    with Sink(status=500) as sink:
        with pytest.raises(urllib.error.HTTPError):
            ds.post(sink.url, {"live": {}})
    assert len(sink.requests) == 1


def test_post_raises_when_nothing_listens(no_proxy):
    import urllib.error
    with Sink() as sink:
        url = sink.url
    with pytest.raises(urllib.error.URLError):
        ds.post(url, {"live": {}})


# ---------------------------------------------------------------------------------------------------------- money

def local_ts(year, month, day, hour=12):
    return int(time.mktime((year, month, day, hour, 0, 0, 0, 0, -1)))


def add_payment(uid, status, cents, at):
    db.execute("INSERT INTO payments(id, user_id, kind, status, amount_cents, tax_cents, created_at, updated_at)"
               " VALUES (?,?,?,?,?,?,?,?)", (db.new_id(), uid, "checkout", status, cents, 0, at, at))


def add_ledger(uid, kind, micros, at):
    db.execute("INSERT INTO ledger(user_id, amount_micros, kind, balance_after_micros, created_at) VALUES (?,?,?,?,?)",
               (uid, micros, kind, 0, at))


def freeze(monkeypatch, now):
    """Only dashboard_stats sees the fixed time (its `time` name is replaced, not the time module)."""
    monkeypatch.setattr(ds, "time", types.SimpleNamespace(time=lambda: now + 0.4, localtime=time.localtime, mktime=time.mktime))


def test_money_has_this_month_last_month_and_the_series(monkeypatch):
    now = local_ts(2026, 10, 20)
    freeze(monkeypatch, now)
    m = ds.money()
    assert set(m) == {"month", "prev", "series"}
    this_start, prev_start = local_ts(2026, 10, 1, 0), local_ts(2026, 9, 1, 0)
    assert m["month"]["range"] == {"start": this_start, "end": now}
    assert m["prev"]["range"] == {"start": prev_start, "end": this_start}
    for key in ("month", "prev"):
        assert {"money", "users", "jobs", "live", "revenueByProduct"} <= set(m[key])
    pts = m["series"]
    assert isinstance(pts, list) and pts
    assert all({"t", "topupsUsd", "usageUsd", "refundsUsd", "signups", "jobsDone"} <= set(p) for p in pts)
    assert pts[-1]["t"] < now and pts[0]["t"] <= now - 30 * 86400 < pts[0]["t"] + (pts[1]["t"] - pts[0]["t"])
    assert [p["t"] for p in pts] == sorted(p["t"] for p in pts)
    json.dumps(m)  # the whole summary can be posted


def test_money_empty_database_is_all_zero(monkeypatch):
    freeze(monkeypatch, local_ts(2026, 10, 20))
    m = ds.money()
    assert m["month"]["money"]["topupsUsd"] == m["prev"]["money"]["topupsUsd"] == 0
    assert m["month"]["money"]["payments"] == 0
    assert all(p["topupsUsd"] == 0 and p["usageUsd"] == 0 for p in m["series"])


def test_this_months_topups_are_in_month_and_not_in_prev(monkeypatch):
    now = local_ts(2026, 10, 20)
    freeze(monkeypatch, now)
    uid = add_user()
    add_payment(uid, "succeeded", 1000, local_ts(2026, 10, 5))
    add_payment(uid, "succeeded", 2500, local_ts(2026, 10, 19))
    add_payment(uid, "succeeded", 400, local_ts(2026, 9, 28))  # last month
    add_payment(uid, "succeeded", 9900, local_ts(2026, 8, 31))  # two months ago: in neither
    add_payment(uid, "failed", 7777, local_ts(2026, 10, 6))  # not paid
    add_ledger(uid, "usage", -3_000_000, local_ts(2026, 10, 7))
    add_ledger(uid, "usage", -1_000_000, local_ts(2026, 9, 10))
    m = ds.money()
    assert (m["month"]["money"]["topupsUsd"], m["month"]["money"]["payments"], m["month"]["money"]["usageUsd"]) == (35.0, 2, 3.0)
    assert (m["prev"]["money"]["topupsUsd"], m["prev"]["money"]["payments"], m["prev"]["money"]["usageUsd"]) == (4.0, 1, 1.0)
    assert m["month"]["money"]["failedPayments"] == 1


def test_the_month_starts_at_local_midnight_on_the_first_and_ends_now(monkeypatch):
    now = local_ts(2026, 10, 20)
    freeze(monkeypatch, now)
    uid = add_user()
    first = local_ts(2026, 10, 1, 0)
    add_payment(uid, "succeeded", 100, first)  # exactly the start: this month
    add_payment(uid, "succeeded", 200, first - 1)  # a second earlier: last month
    add_payment(uid, "succeeded", 300, now)  # now is exclusive
    m = ds.money()
    assert (m["month"]["money"]["topupsUsd"], m["prev"]["money"]["topupsUsd"]) == (1.0, 2.0)


def test_in_january_last_month_is_december_of_the_year_before(monkeypatch):
    now = local_ts(2027, 1, 10)
    freeze(monkeypatch, now)
    uid = add_user()
    add_payment(uid, "succeeded", 500, local_ts(2026, 12, 31, 23))
    add_payment(uid, "succeeded", 100, local_ts(2027, 1, 2))
    add_payment(uid, "succeeded", 900, local_ts(2026, 11, 30))
    m = ds.money()
    assert m["prev"]["range"] == {"start": local_ts(2026, 12, 1, 0), "end": local_ts(2027, 1, 1, 0)}
    assert (m["month"]["money"]["topupsUsd"], m["prev"]["money"]["topupsUsd"]) == (1.0, 5.0)


def test_the_series_covers_the_last_30_days_in_buckets(monkeypatch):
    now = local_ts(2026, 10, 20)
    freeze(monkeypatch, now)
    uid = add_user()
    t1, t2 = now - 2 * 86400, now - 29 * 86400
    add_payment(uid, "succeeded", 1200, t1)
    add_payment(uid, "succeeded", 800, t1 + 60)  # same bucket
    add_payment(uid, "succeeded", 5000, now - 31 * 86400)  # older than 30 days: not in the series
    add_ledger(uid, "usage", -2_000_000, t2)
    m = ds.money()
    pts = m["series"]
    step = pts[1]["t"] - pts[0]["t"]
    by = {p["t"]: p for p in pts}
    assert by[(t1 // step) * step]["topupsUsd"] == 20.0
    assert by[(t2 // step) * step]["usageUsd"] == 2.0
    assert sum(p["topupsUsd"] for p in pts) == 20.0
    assert all(b["t"] - a["t"] == step for a, b in zip(pts, pts[1:]))


# ---------------------------------------------------------------------------------------------------------- send_loop

def fake_clock_rounds(monkeypatch, rounds, start=1000.0):
    """Runs send_loop on a fake clock. loop.time() reads the clock and each asyncio.sleep moves it on by the seconds slept,
    so round k (0-based) starts at start + k * INTERVAL_S. The `rounds`-th sleep cancels the loop, so `rounds` rounds run.
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


def run_loop():
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(ds.send_loop())


class Recorder:
    """Replaces collect, money and post: counts the calls and records each posted (url, body). `post_fails` and
    `money_fails` are sets of 0-based call numbers that raise."""

    def __init__(self, monkeypatch, post_fails=(), money_fails=(), collect_fails=()):
        self.posts, self.collects, self.moneys = [], 0, 0
        self.post_fails, self.money_fails, self.collect_fails = set(post_fails), set(money_fails), set(collect_fails)
        monkeypatch.setattr(ds, "collect", self.collect)
        monkeypatch.setattr(ds, "money", self.money)
        monkeypatch.setattr(ds, "post", self.post)

    def collect(self):
        n, self.collects = self.collects, self.collects + 1
        if n in self.collect_fails:
            raise RuntimeError(f"collect {n} broke")
        return {"n": n}

    def money(self):
        n, self.moneys = self.moneys, self.moneys + 1
        if n in self.money_fails:
            raise RuntimeError(f"money {n} broke")
        return {"m": n}

    def post(self, url, body):
        n = len(self.posts)
        self.posts.append((url, body))
        if n in self.post_fails:
            raise OSError(f"post {n} broke")


URL = "http://192.168.1.200/api/acw"


def test_the_loop_returns_at_once_when_the_url_is_unset(monkeypatch):
    for value in (None, "", "  "):
        if value is None:
            monkeypatch.delenv("ACW_DASHBOARD_URL", raising=False)
        else:
            monkeypatch.setenv("ACW_DASHBOARD_URL", value)
        rec = Recorder(monkeypatch)

        async def boom(*a, **k):
            raise AssertionError("must not sleep")
        monkeypatch.setattr(ds.asyncio, "sleep", boom)
        assert asyncio.run(ds.send_loop()) is None
        assert rec.collects == rec.moneys == 0 and rec.posts == []


def test_the_first_round_sends_live_and_money_then_sleeps_the_interval(monkeypatch):
    monkeypatch.setenv("ACW_DASHBOARD_URL", URL)
    rec = Recorder(monkeypatch)
    sleeps, _ = fake_clock_rounds(monkeypatch, 1)
    run_loop()
    assert sleeps == [ds.INTERVAL_S] == [10]
    assert rec.posts == [(URL, {"live": {"n": 0}, "money": {"m": 0}})]


def test_later_rounds_within_the_money_interval_send_only_live(monkeypatch):
    monkeypatch.setenv("ACW_DASHBOARD_URL", URL)
    rec = Recorder(monkeypatch)
    rounds = ds.MONEY_S // ds.INTERVAL_S  # rounds 0 .. rounds-1 start less than MONEY_S after the first
    sleeps, _ = fake_clock_rounds(monkeypatch, rounds)
    run_loop()
    assert sleeps == [ds.INTERVAL_S] * rounds
    assert len(rec.posts) == rounds
    assert rec.posts[0][1] == {"live": {"n": 0}, "money": {"m": 0}}
    assert all(set(body) == {"live"} for _, body in rec.posts[1:])
    assert [body["live"]["n"] for _, body in rec.posts] == list(range(rounds))
    assert rec.moneys == 1 and rec.collects == rounds


def test_money_comes_again_after_money_s_and_then_every_money_s(monkeypatch):
    monkeypatch.setenv("ACW_DASHBOARD_URL", URL)
    rec = Recorder(monkeypatch)
    per = ds.MONEY_S // ds.INTERVAL_S
    rounds = 2 * per + 1  # rounds 0, per, 2 * per carry money
    fake_clock_rounds(monkeypatch, rounds)
    run_loop()
    with_money = [i for i, (_, body) in enumerate(rec.posts) if "money" in body]
    assert with_money == [0, per, 2 * per]
    assert [rec.posts[i][1]["money"] for i in with_money] == [{"m": 0}, {"m": 1}, {"m": 2}]
    assert rec.moneys == 3 and len(rec.posts) == rounds


def test_money_is_timed_by_the_loop_clock_not_the_round_count(monkeypatch):
    """A slow round (the clock jumps) brings the money summary earlier in rounds."""
    monkeypatch.setenv("ACW_DASHBOARD_URL", URL)
    Recorder(monkeypatch)
    sleeps, clock = fake_clock_rounds(monkeypatch, 4)
    # round 0 normal, round 1 slow, so round 2 is due, round 3 not
    posts_seen = []

    def post(url, body):
        posts_seen.append(body)
        if len(posts_seen) == 2:
            clock.now += ds.MONEY_S
    monkeypatch.setattr(ds, "post", post)
    run_loop()
    assert ["money" in b for b in posts_seen] == [True, False, True, False]


def test_a_failing_post_is_logged_once_and_the_loop_goes_on(monkeypatch, caplog):
    monkeypatch.setenv("ACW_DASHBOARD_URL", URL)
    rec = Recorder(monkeypatch, post_fails=range(100))
    sleeps, _ = fake_clock_rounds(monkeypatch, 6)
    with caplog.at_level(logging.WARNING, logger="acw.dashboard_stats"):
        run_loop()
    assert sleeps == [ds.INTERVAL_S] * 6  # it slept after each failure
    assert len(rec.posts) == 6 and rec.collects == 6
    records = [r for r in caplog.records if r.name == "acw.dashboard_stats"]
    assert len(records) == 1 and "post 0 broke" in caplog.text


def test_the_log_is_armed_again_after_a_good_round(monkeypatch, caplog):
    monkeypatch.setenv("ACW_DASHBOARD_URL", URL)
    # posts 0, 1 fail; 2 succeeds; 3 and 4 fail again
    Recorder(monkeypatch, post_fails={0, 1, 3, 4})
    fake_clock_rounds(monkeypatch, 6)
    with caplog.at_level(logging.WARNING, logger="acw.dashboard_stats"):
        run_loop()
    msgs = [r.exc_info[1].args[0] for r in caplog.records if r.name == "acw.dashboard_stats"]
    assert msgs == ["post 0 broke", "post 3 broke"]


def test_a_failing_count_is_logged_once_and_sends_nothing_that_round(monkeypatch, caplog):
    monkeypatch.setenv("ACW_DASHBOARD_URL", URL)
    rec = Recorder(monkeypatch, collect_fails={0, 1})
    fake_clock_rounds(monkeypatch, 4)
    with caplog.at_level(logging.WARNING, logger="acw.dashboard_stats"):
        run_loop()
    assert [b["live"]["n"] for _, b in rec.posts] == [2, 3]
    assert len([r for r in caplog.records if r.name == "acw.dashboard_stats"]) == 1
    # the money summary was not tried in the rounds whose count failed, so it comes with the first good round
    assert "money" in rec.posts[0][1] and "money" not in rec.posts[1][1]


def test_a_failing_money_summary_is_logged_and_retried_next_round(monkeypatch, caplog):
    monkeypatch.setenv("ACW_DASHBOARD_URL", URL)
    rec = Recorder(monkeypatch, money_fails={0, 1})
    fake_clock_rounds(monkeypatch, 5)
    with caplog.at_level(logging.WARNING, logger="acw.dashboard_stats"):
        run_loop()
    assert rec.moneys == 3  # tried in rounds 0, 1 (failed) and 2 (good), then not again for MONEY_S
    assert [("money" in b) for _, b in rec.posts] == [True, False, False]  # nothing was posted in the two failed rounds...
    assert [b["live"]["n"] for _, b in rec.posts] == [2, 3, 4]
    assert len([r for r in caplog.records if r.name == "acw.dashboard_stats"]) == 1  # ...and the two failures logged once


def test_the_real_collect_and_money_are_posted_end_to_end(monkeypatch, gpu, no_proxy):
    freeze_now = local_ts(2026, 10, 20)
    freeze(monkeypatch, freeze_now)
    uid = add_user()
    add_job(uid, "running")
    add_payment(uid, "succeeded", 1000, local_ts(2026, 10, 5))
    with Sink() as sink:
        monkeypatch.setenv("ACW_DASHBOARD_URL", sink.url)
        fake_clock_rounds(monkeypatch, 2)
        run_loop()
    assert len(sink.requests) == 2
    first, second = (json.loads(r["body"]) for r in sink.requests)
    assert first["live"]["job_running_denoising"] == 1 and first["live"]["jobs_running"] == 1
    assert first["money"]["month"]["money"]["topupsUsd"] == 10.0 and first["money"]["series"]
    assert set(second) == {"live"}
