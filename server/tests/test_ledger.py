import threading

import pytest

from acw import auth, db
from acw.billing import ledger, pricing


def user(email="a@example.com"):
    return auth.get_or_create_user(email)["id"]


def test_credit_is_idempotent_per_ref():
    u = user()
    assert ledger.credit(u, 1_000_000, "topup", "payment", "p1")
    assert not ledger.credit(u, 1_000_000, "topup", "payment", "p1")
    assert ledger.balance(u)["balance_micros"] == 1_000_000


def test_hold_capture_and_release():
    u = user()
    ledger.credit(u, 1_000_000, "grant", "t", "g")
    h = ledger.create_hold(u, 600_000, "job", "j1")
    assert ledger.balance(u)["available_micros"] == 400_000
    with pytest.raises(ledger.InsufficientFunds):
        ledger.create_hold(u, 500_000, "job", "j2")
    assert ledger.capture_hold(h, 250_000) == 250_000
    b = ledger.balance(u)
    assert b == {"balance_micros": 750_000, "held_micros": 0, "available_micros": 750_000}
    with pytest.raises(ledger.LedgerError):
        ledger.capture_hold(h, 1)  # already captured
    h2 = ledger.create_hold(u, 100_000, "job", "j3")
    ledger.release_hold(h2)
    ledger.release_hold(h2)  # second release is a no-op
    assert ledger.balance(u)["available_micros"] == 750_000


def test_capture_overage_takes_only_what_is_available():
    u = user()
    ledger.credit(u, 100_000, "grant", "t", "g")
    h = ledger.create_hold(u, 50_000, "job", "j")
    assert ledger.capture_hold(h, 500_000) == 100_000
    assert ledger.balance(u)["balance_micros"] == 0


def test_extend_hold_respects_available():
    u = user()
    ledger.credit(u, 100_000, "grant", "t", "g")
    h = ledger.create_hold(u, 60_000, "stream", "s")
    ledger.extend_hold(h, 40_000)
    with pytest.raises(ledger.InsufficientFunds):
        ledger.extend_hold(h, 1)


def test_concurrent_holds_never_overdraw():
    u = user()
    ledger.credit(u, 1_000_000, "grant", "t", "g")
    ok, errors = [], []

    def worker(i):
        db.reset_connection()
        try:
            ledger.create_hold(u, 100_000, "job", f"j{i}")
            ok.append(i)
        except ledger.InsufficientFunds:
            errors.append(i)
        finally:
            db.reset_connection()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(ok) == 10 and len(errors) == 20
    assert ledger.balance(u)["held_micros"] == 1_000_000


def test_balance_constraints_hold_in_sql():
    u = user()
    with pytest.raises(Exception):
        db.execute("UPDATE users SET balance_micros = -1 WHERE id = ?", (u,))
    with pytest.raises(Exception):
        db.execute("UPDATE users SET held_micros = 5 WHERE id = ?", (u,))


def test_suspended_account_cannot_hold():
    u = user()
    ledger.credit(u, 100_000, "grant", "t", "g")
    db.execute("UPDATE users SET status = 'suspended' WHERE id = ?", (u,))
    with pytest.raises(ledger.LedgerError):
        ledger.create_hold(u, 10_000, "job", "j")


def test_pricing_rounds_up_to_whole_seconds():
    assert pricing.cost_micros("denoise", "standard", 60) == 10_000
    assert pricing.cost_micros("denoise", "standard", 0.2) == 167  # 1 s
    assert pricing.job_cost_micros("denoise", "standard", None, 1) == 167  # per second, no minimum
    with pytest.raises(pricing.PricingError):
        pricing.rate("stream_denoise", "high")  # no such tier
