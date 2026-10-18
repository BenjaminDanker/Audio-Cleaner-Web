"""Live admission in the web process (audit M7): a sanity bound, and the event loop's measured lateness."""
from types import SimpleNamespace

from acw.billing import ledger
from acw.config import get_settings
from acw.stream import routes

from .test_live_trial import fake_worker, run_session  # noqa: F401 - fixture


def test_all_live_sessions_together_are_limited(client, live_client, signed_in, fake_worker, monkeypatch):
    uid, _ = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    monkeypatch.setattr(routes, "MAX_LIVE_SESSIONS", 30)
    monkeypatch.setitem(routes._live, "others", 29)
    assert run_session(live_client, 0.1)[0]["type"] == "ready"  # the 30th still fits
    monkeypatch.setitem(routes._live, "others", 30)
    got = run_session(live_client, 0.1)
    assert got[0]["type"] == "error" and got[0]["code"] == "full" and got[0]["available"] == []
    assert got[0]["retryAfterSeconds"] == 60 and "try again" in got[0]["message"].lower()
    assert len(fake_worker.started) == 1  # the refused one never reached the worker


def test_extra_accurate_sessions_are_not_counted_separately(client, live_client, signed_in, fake_worker, monkeypatch):
    """They use the speech partner's computers, not ours: only the relay's measured load can turn them away."""
    uid, _ = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()
    others = [SimpleNamespace(cfg={"captions": "premium"}, last_rx=1e18) for _ in range(40)]
    monkeypatch.setitem(routes._open, "others", others)
    monkeypatch.setitem(routes._live, "others", 40)
    got = run_session(live_client, 0.1, {"denoise": None, "captions": "premium"})
    assert got[0].get("code") != "full"


def test_a_late_event_loop_refuses_new_sessions_only(client, live_client, signed_in, fake_worker, monkeypatch):
    uid, _ = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    lag = routes.LoopLag()
    lag.samples.extend([0.002] * 95 + [0.2] * 5)  # a few late timers: not busy
    monkeypatch.setattr(routes, "_lag", lag)
    assert not routes.loop_busy() and run_session(live_client, 0.1)[0]["type"] == "ready"
    lag.samples.extend([0.06] * 30)  # late for seconds: busy
    assert routes.loop_busy()
    got = run_session(live_client, 0.1)
    assert got[0]["code"] == "full" and got[0]["available"] == [] and "try again" in got[0]["message"].lower()
    assert len(fake_worker.started) == 1
