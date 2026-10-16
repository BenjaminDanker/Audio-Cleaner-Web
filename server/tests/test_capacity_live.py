"""Global limits on live sessions in the web process (audit M7)."""
from types import SimpleNamespace

from acw.billing import ledger
from acw.config import get_settings
from acw.stream import routes

from .test_live_trial import fake_worker, run_session  # noqa: F401 - fixture


def test_all_live_sessions_together_are_limited(client, signed_in, fake_worker, monkeypatch):
    uid, _ = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    monkeypatch.setattr(routes, "MAX_LIVE_SESSIONS", 30)
    monkeypatch.setitem(routes._live, "others", 29)
    assert run_session(client, 0.1)[0]["type"] == "ready"  # the 30th still fits
    monkeypatch.setitem(routes._live, "others", 30)
    got = run_session(client, 0.1)
    assert got[0]["type"] == "error" and got[0]["code"] == "full" and got[0]["available"] == []
    assert got[0]["retryAfterSeconds"] == 60 and "try again" in got[0]["message"].lower()
    assert len(fake_worker.started) == 1  # the refused one never reached the worker


def test_extra_accurate_sessions_have_their_own_limit(client, signed_in, fake_worker, monkeypatch):
    uid, _ = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()
    monkeypatch.setattr(routes, "MAX_PREMIUM_SESSIONS", 2)
    others = [SimpleNamespace(cfg={"captions": "premium"}), SimpleNamespace(cfg={"captions": "local"})]
    monkeypatch.setitem(routes._open, "others", others)
    assert routes.premium_sessions() == 1
    # one extra-accurate session is running: a second still fits, a standard one is never affected
    monkeypatch.setitem(routes._open, "others", others + [SimpleNamespace(cfg={"captions": "premium"})])
    assert routes.premium_sessions() == 2
    got = run_session(client, 0.1, {"denoise": None, "captions": "premium"})
    assert got[0]["code"] == "full" and "extra-accurate" in got[0]["message"].lower()
    assert fake_worker.started == []
    assert run_session(client, 0.1, {"denoise": "light"})[0]["type"] == "ready"
