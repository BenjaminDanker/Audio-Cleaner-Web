import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "server"), str(ROOT / "common")]

ORIGIN = "http://testserver"


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("ACW_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ACW_PUBLIC_ORIGIN", ORIGIN)
    monkeypatch.delenv("ACW_BASE_PATH", raising=False)
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test_secret")
    monkeypatch.setenv("ACW_WELCOME_CREDIT_MICROS", "0")  # tests start from an empty balance; test_welcome_credit.py turns it on
    from acw import db
    from acw.config import get_settings
    from acw.security import limiter

    get_settings.cache_clear()
    db.reset_connection()
    limiter.reset()
    db.init_db()
    yield
    db.reset_connection()
    get_settings.cache_clear()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from acw.app import create_app

    with TestClient(create_app(), base_url=ORIGIN) as c:
        yield c


def login(client, email="user@example.com", accept_tos=True):
    from acw import auth

    tok = auth.create_login_token(email, "admin_login", "test")
    r = client.post("/api/auth/email/verify", json={"token": tok}, headers={"Origin": ORIGIN})
    assert r.status_code == 200, r.text
    me = client.get("/api/me").json()
    h = {"X-CSRF-Token": me["csrfToken"], "Origin": ORIGIN}
    if accept_tos:
        assert client.post("/api/account/tos", json={"version": auth.TOS_VERSION}, headers=h).status_code == 200
    return me["user"]["id"], h


@pytest.fixture
def signed_in(client):
    return login(client)


def settle_payment(user_id, cents=1000):
    """A top-up that went through: the account counts as paid (extra-accurate choices, two live sessions, large top-ups)."""
    from acw import db

    now = db.now()
    db.execute(
        "INSERT INTO payments(id, user_id, kind, status, amount_cents, created_at, updated_at)"
        " VALUES (?, ?, 'checkout', 'succeeded', ?, ?, ?)", (db.new_id(), user_id, cents, now, now))
