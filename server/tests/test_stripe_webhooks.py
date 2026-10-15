import hashlib
import hmac
import json
import time
from types import SimpleNamespace

import pytest

from acw import auth, db
from acw.billing import ledger, stripe_service

SECRET = "whsec_test_secret"


def signed(payload: dict) -> tuple[bytes, str]:
    body = json.dumps(payload).encode()
    t = int(time.time())
    sig = hmac.new(SECRET.encode(), f"{t}.".encode() + body, hashlib.sha256).hexdigest()
    return body, f"t={t},v1={sig}"


def event(etype: str, obj: dict, eid: str = "evt_1") -> dict:
    return {"id": eid, "object": "event", "type": etype, "api_version": "2026-09-30.endive",
            "created": int(time.time()), "data": {"object": obj}, "livemode": False, "pending_webhooks": 1,
            "request": {"id": None, "idempotency_key": None}}


@pytest.fixture
def user_with_pending_payment():
    uid = auth.get_or_create_user("payer@example.com")["id"]
    db.execute(
        "INSERT INTO payments(id,user_id,kind,status,amount_cents,stripe_checkout_session,created_at,updated_at)"
        " VALUES ('pay1', ?, 'checkout', 'pending', 2000, 'cs_1', ?, ?)",
        (uid, db.now(), db.now()),
    )
    return uid


@pytest.fixture
def fake_pi(monkeypatch):
    """Fakes the Stripe network calls: the payment intent, the charge (card country) and refunds."""
    pi = SimpleNamespace(payment_method="pm_1", latest_charge="ch_1", setup_future_usage="off_session")
    state = SimpleNamespace(country="US", refunds=[])

    def charge(_id, params=None):
        return SimpleNamespace(payment_method_details=SimpleNamespace(card=SimpleNamespace(country=state.country)))

    def refund(params=None, options=None):
        state.refunds.append((params, options))
        return SimpleNamespace(id=f"re_{len(state.refunds)}")

    real_client = stripe_service.client

    def client():
        c = real_client()
        c.v1.payment_intents = SimpleNamespace(retrieve=lambda _id: pi)  # only the network calls are faked
        c.v1.charges = SimpleNamespace(retrieve=charge)
        c.v1.refunds = SimpleNamespace(create=refund)
        return c

    monkeypatch.setattr(stripe_service, "client", client)
    return state


def checkout_completed(uid, eid="evt_1"):
    return event("checkout.session.completed", {
        "id": "cs_1", "object": "checkout.session", "mode": "payment", "payment_status": "paid",
        "amount_total": 2000, "payment_intent": "pi_1",
        "metadata": {"acw_user_id": uid, "acw_payment_id": "pay1"},
    }, eid)


def test_bad_signature_rejected(client, user_with_pending_payment):
    body, _ = signed(checkout_completed(user_with_pending_payment))
    r = client.post("/api/stripe/webhook", content=body, headers={"Stripe-Signature": "t=1,v1=deadbeef"})
    assert r.status_code == 400
    assert ledger.balance(user_with_pending_payment)["balance_micros"] == 0


def test_checkout_credits_once_and_saves_card(client, user_with_pending_payment, fake_pi):
    uid = user_with_pending_payment
    for eid in ("evt_1", "evt_1", "evt_2"):  # replay + a duplicate event with a new id
        body, sig = signed(checkout_completed(uid, eid))
        assert client.post("/api/stripe/webhook", content=body, headers={"Stripe-Signature": sig}).status_code == 200
    assert ledger.balance(uid)["balance_micros"] == 20_000_000
    u = db.one("SELECT default_payment_method FROM users WHERE id = ?", (uid,))
    assert u["default_payment_method"] == "pm_1"
    assert db.one("SELECT status, stripe_charge FROM payments WHERE id='pay1'")["stripe_charge"] == "ch_1"


def test_dispute_suspends_and_zeroes(client, user_with_pending_payment, fake_pi):
    uid = user_with_pending_payment
    body, sig = signed(checkout_completed(uid))
    client.post("/api/stripe/webhook", content=body, headers={"Stripe-Signature": sig})
    ledger.create_hold(uid, 1_000_000, "job", "j")
    body, sig = signed(event("charge.dispute.created", {"id": "dp_1", "object": "dispute", "charge": "ch_1",
                                                          "reason": "fraudulent"}, "evt_d"))
    assert client.post("/api/stripe/webhook", content=body, headers={"Stripe-Signature": sig}).status_code == 200
    u = db.one("SELECT status, balance_micros, held_micros FROM users WHERE id = ?", (uid,))
    assert (u["status"], u["balance_micros"], u["held_micros"]) == ("suspended", 0, 0)
    assert db.one("SELECT COUNT(*) n FROM sessions WHERE user_id = ?", (uid,))["n"] == 0


def test_dashboard_refund_debits_balance(client, user_with_pending_payment, fake_pi):
    uid = user_with_pending_payment
    body, sig = signed(checkout_completed(uid))
    client.post("/api/stripe/webhook", content=body, headers={"Stripe-Signature": sig})
    charge = {"id": "ch_1", "object": "charge", "payment_intent": "pi_1", "amount_refunded": 500,
              "refunds": {"object": "list", "data": [{"id": "re_1", "object": "refund", "amount": 500, "status": "succeeded"}]}}
    for eid in ("evt_r1", "evt_r2"):
        body, sig = signed(event("charge.refunded", charge, eid))
        client.post("/api/stripe/webhook", content=body, headers={"Stripe-Signature": sig})
    assert ledger.balance(uid)["balance_micros"] == 15_000_000


def test_reload_failure_disables_after_three(monkeypatch):
    uid = auth.get_or_create_user("r@example.com")["id"]
    for i in range(3):
        db.execute("INSERT INTO payments(id,user_id,kind,status,amount_cents,created_at,updated_at) VALUES (?,?,'auto_reload','pending',1000,0,0)", (f"p{i}", uid))
        db.execute("UPDATE users SET auto_reload_enabled = 1, auto_reload_in_flight = 1 WHERE id = ?", (uid,))
        stripe_service._record_reload_failure(f"p{i}", uid, "card_declined")
    u = db.one("SELECT auto_reload_enabled, auto_reload_failures, auto_reload_in_flight FROM users WHERE id = ?", (uid,))
    assert (u["auto_reload_enabled"], u["auto_reload_failures"], u["auto_reload_in_flight"]) == (0, 3, 0)


def test_checkout_for_other_session_is_not_credited(client, user_with_pending_payment, fake_pi):
    uid = user_with_pending_payment
    ev = checkout_completed(uid)
    ev["data"]["object"]["id"] = "cs_someone_else"
    ev["data"]["object"]["amount_total"] = 99_999_99  # amounts from events are never trusted
    body, sig = signed(ev)
    assert client.post("/api/stripe/webhook", content=body, headers={"Stripe-Signature": sig}).status_code == 200
    assert ledger.balance(uid)["balance_micros"] == 0


def test_tax_split_on_refunds(client, user_with_pending_payment, fake_pi):
    """With tax on top, refunds only ever remove the pre-tax credit share."""
    uid = user_with_pending_payment
    ev = checkout_completed(uid)  # $20 credit + $4 VAT
    ev["data"]["object"]["total_details"] = {"amount_tax": 400}
    body, sig = signed(ev)
    client.post("/api/stripe/webhook", content=body, headers={"Stripe-Signature": sig})
    assert ledger.balance(uid)["balance_micros"] == 20_000_000
    charge = {"id": "ch_1", "object": "charge", "payment_intent": "pi_1", "amount_refunded": 1200,
              "refunds": {"object": "list", "data": [{"id": "re_9", "object": "refund", "amount": 1200, "status": "succeeded"}]}}
    body, sig = signed(event("charge.refunded", charge, "evt_tx"))
    client.post("/api/stripe/webhook", content=body, headers={"Stripe-Signature": sig})
    assert ledger.balance(uid)["balance_micros"] == 10_000_000  # $12 gross = $10 credit + $2 tax
    assert stripe_service.tax_part(1000, 2000, 400) == 200
    assert stripe_service.credit_part(2400, 2000, 400) == 2000


def test_self_service_refund_keeps_the_card_fee_once(monkeypatch):
    from types import SimpleNamespace

    from acw import auth, db
    from acw.billing import ledger, stripe_service

    uid = auth.get_or_create_user("fee@example.com")["id"]
    now = db.now()
    db.execute(
        "INSERT INTO payments(id, user_id, kind, status, amount_cents, refunded_cents, tax_cents, stripe_charge, created_at, updated_at)"
        " VALUES ('pay_fee', ?, 'checkout', 'succeeded', 1000, 0, 0, 'ch_fee', ?, ?)",
        (uid, now, now),
    )
    ledger.credit(uid, 1000 * 10_000, "topup", "payment", "pay_fee")
    refunds = []

    class Fake:
        class v1:
            class charges:
                @staticmethod
                def retrieve(_id, params=None):
                    return SimpleNamespace(balance_transaction=SimpleNamespace(fee=59, currency="usd"))

            class refunds:
                @staticmethod
                def create(params=None, options=None):
                    refunds.append(params["amount"])
                    return SimpleNamespace(id=f"re_{len(refunds)}")

    monkeypatch.setattr(stripe_service, "client", lambda: Fake)
    monkeypatch.setattr(stripe_service, "_reverse_tax", lambda *a: None)
    ledger.create_hold(uid, 600 * 10_000, "job", "j")  # $6 still spendable is held -> $4 unused
    assert stripe_service.refund_unused(uid, "pay_fee") == 400 - 59
    assert refunds == [341]
    assert ledger.balance(uid)["balance_micros"] == 600 * 10_000  # all $4 of unused credit is gone


def _post(client, ev):
    body, sig = signed(ev)
    return client.post("/api/stripe/webhook", content=body, headers={"Stripe-Signature": sig})


@pytest.mark.parametrize("country", ["DE", "gb", "CH", "NO", "FR"])
def test_checkout_by_blocked_country_card_is_refunded_not_credited(client, user_with_pending_payment, fake_pi, country):
    uid = user_with_pending_payment
    fake_pi.country = country
    db.execute("UPDATE users SET auto_reload_enabled = 1 WHERE id = ?", (uid,))
    for eid in ("evt_1", "evt_dup"):  # replayed delivery refunds only once (same idempotency key)
        assert _post(client, checkout_completed(uid, eid)).status_code == 200
    assert ledger.balance(uid)["balance_micros"] == 0
    assert {o["idempotency_key"] for _, o in fake_pi.refunds} == {"blocked-country-pay1"}
    assert fake_pi.refunds[0][0]["charge"] == "ch_1" and "amount" not in fake_pi.refunds[0][0]  # full refund
    assert db.one("SELECT status FROM payments WHERE id='pay1'")["status"] == "refused_region"
    assert db.one("SELECT auto_reload_enabled FROM users WHERE id = ?", (uid,))["auto_reload_enabled"] == 0
    assert db.one("SELECT COUNT(*) n FROM audit_log WHERE user_id = ? AND action = 'payment_refused_region'", (uid,))["n"] >= 1


def test_us_card_is_credited_and_countries_are_configurable(client, user_with_pending_payment, fake_pi, monkeypatch):
    uid = user_with_pending_payment
    fake_pi.country = "DE"
    monkeypatch.setenv("ACW_BLOCKED_CARD_COUNTRIES", "none")  # guard switched off
    assert _post(client, checkout_completed(uid)).status_code == 200
    assert ledger.balance(uid)["balance_micros"] == 20_000_000 and not fake_pi.refunds
    assert stripe_service.blocked_card_countries() == frozenset()
    monkeypatch.delenv("ACW_BLOCKED_CARD_COUNTRIES")
    assert {"DE", "GB", "CH", "IS", "LI", "NO", "FR"} <= stripe_service.blocked_card_countries()
    assert "US" not in stripe_service.blocked_card_countries() and len(stripe_service.BLOCKED_CARD_COUNTRIES) == 30 + 2
    monkeypatch.setenv("ACW_BLOCKED_CARD_COUNTRIES", "br, us")
    assert stripe_service.blocked_card_countries() == {"BR", "US"}


def test_refund_of_a_blocked_payment_does_not_touch_other_credit(client, user_with_pending_payment, fake_pi):
    uid = user_with_pending_payment
    ledger.credit(uid, 7_000_000, "grant", "t", "g")
    fake_pi.country = "DE"
    assert _post(client, checkout_completed(uid)).status_code == 200
    charge = {"id": "ch_1", "object": "charge", "payment_intent": "pi_1", "amount_refunded": 2000,
              "refunds": {"object": "list", "data": [{"id": "re_1", "object": "refund", "amount": 2000, "status": "succeeded"}]}}
    assert _post(client, event("charge.refunded", charge, "evt_rf")).status_code == 200
    assert ledger.balance(uid)["balance_micros"] == 7_000_000


def test_auto_reload_by_blocked_country_card_is_refunded_and_disabled(user_with_pending_payment, fake_pi, monkeypatch):
    uid = user_with_pending_payment
    fake_pi.country = "IE"
    db.execute("DELETE FROM payments")
    db.execute(
        "UPDATE users SET auto_reload_enabled = 1, auto_reload_threshold_micros = 5000000, auto_reload_amount_cents = 2000,"
        " auto_reload_monthly_cap_cents = 100000, default_payment_method = 'pm_1', stripe_customer_id = 'cus_1' WHERE id = ?",
        (uid,),
    )
    real = stripe_service.client

    def client():
        c = real()
        c.v1.payment_intents = SimpleNamespace(
            create=lambda params=None, options=None: SimpleNamespace(id="pi_r", status="succeeded", latest_charge="ch_1"))
        return c

    monkeypatch.setattr(stripe_service, "client", client)
    assert stripe_service.run_auto_reload(uid) == "refused_region"
    assert ledger.balance(uid)["balance_micros"] == 0 and len(fake_pi.refunds) == 1
    u = db.one("SELECT auto_reload_enabled, auto_reload_in_flight FROM users WHERE id = ?", (uid,))
    assert (u["auto_reload_enabled"], u["auto_reload_in_flight"]) == (0, 0)
    # The payment_intent.succeeded webhook for the same payment changes nothing.
    pid = db.one("SELECT id FROM payments")["id"]
    stripe_service._handle_pi_succeeded({"id": "pi_r", "currency": "usd", "latest_charge": "ch_1",
                                         "metadata": {"acw_kind": "auto_reload", "acw_payment_id": pid, "acw_user_id": uid}})
    assert ledger.balance(uid)["balance_micros"] == 0 and len(fake_pi.refunds) == 1
