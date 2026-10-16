"""Every new account gets one hour of live once per email address, ever; and the session cleanup runs."""
from acw import auth, db
from acw.billing import ledger, pricing
from acw.config import get_settings

from .conftest import login


def _on(monkeypatch, value=None):
    if value is None:
        monkeypatch.delenv("ACW_WELCOME_CREDIT_MICROS", raising=False)
    else:
        monkeypatch.setenv("ACW_WELCOME_CREDIT_MICROS", str(value))
    get_settings.cache_clear()


def test_a_new_account_gets_one_hour_of_live(monkeypatch):
    _on(monkeypatch)
    hour = 60 * (pricing.rate("stream_denoise", "light").micros_per_minute + pricing.rate("stream_captions", "local").micros_per_minute)
    uid = auth.get_or_create_user("new@example.com")["id"]
    assert ledger.balance(uid)["balance_micros"] == hour == pricing.welcome_credit_micros()
    row = db.one("SELECT kind, ref_type, description FROM ledger WHERE user_id = ?", (uid,))
    assert row["kind"] == "grant" and row["ref_type"] == "welcome" and "Welcome" in row["description"]
    auth.get_or_create_user("NEW@example.com")  # signing in again: nothing more
    assert ledger.balance(uid)["balance_micros"] == hour
    assert db.one("SELECT COUNT(*) n FROM welcome_claims WHERE email_hash LIKE '%new%'")["n"] == 0  # only a hash


def test_deleting_the_account_and_coming_back_does_not_give_it_twice(client, monkeypatch):
    _on(monkeypatch)
    uid, h = login(client, "again@example.com")
    assert ledger.balance(uid)["balance_micros"] > 0
    assert client.post("/api/account/delete", json={"confirm": "again@example.com"}, headers=h).status_code == 200
    uid2, _ = login(client, "again@example.com")
    assert uid2 != uid and ledger.balance(uid2)["balance_micros"] == 0


def test_it_can_be_set_or_switched_off(monkeypatch):
    _on(monkeypatch, 250_000)
    assert ledger.balance(auth.get_or_create_user("a@example.com")["id"])["balance_micros"] == 250_000
    _on(monkeypatch, 0)
    assert ledger.balance(auth.get_or_create_user("b@example.com")["id"])["balance_micros"] == 0


def test_the_session_cleanup_runs():
    db.execute("INSERT INTO deleted_emails(email_hash, day, free_seconds) VALUES ('x', '2026-01-01', 1)")
    auth.purge_expired()
    assert db.one("SELECT COUNT(*) n FROM deleted_emails")["n"] == 0


def test_a_refund_never_pays_out_given_credit(monkeypatch):
    from types import SimpleNamespace

    from acw.billing import stripe_service

    _on(monkeypatch, 600_000)  # $0.60 welcome credit
    uid = auth.get_or_create_user("refund@example.com")["id"]
    now = db.now()
    db.execute(
        "INSERT INTO payments(id, user_id, kind, status, amount_cents, refunded_cents, tax_cents, stripe_charge, created_at, updated_at)"
        " VALUES ('pay_w', ?, 'checkout', 'succeeded', 500, 0, 0, 'ch_w', ?, ?)", (uid, now, now))
    ledger.credit(uid, 500 * 10_000, "topup", "payment", "pay_w")  # $5 bought
    refunds = []

    class Fake:
        class v1:
            class charges:
                @staticmethod
                def retrieve(_id, params=None):
                    return SimpleNamespace(balance_transaction=SimpleNamespace(fee=45, currency="usd"))

            class refunds:
                @staticmethod
                def create(params=None, options=None):
                    refunds.append(params["amount"])
                    return SimpleNamespace(id="re_w")

    monkeypatch.setattr(stripe_service, "client", lambda: Fake)
    monkeypatch.setattr(stripe_service, "_reverse_tax", lambda *a: None)
    ledger.debit_available(uid, 4_000_000, "usage", "job", "j", "used $4")  # $1.60 left: $1 bought + $0.60 given
    assert stripe_service.refund_unused(uid, "pay_w") == 100 - 45  # only the $1 that was bought (less the card fee)
    assert refunds == [55]
    assert ledger.balance(uid)["balance_micros"] == 600_000  # the welcome credit stays to be used
