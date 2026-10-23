"""Auto-reload (stripe_service.run_auto_reload) with the Stripe client stood in for: when it does nothing (skip, cap), and what is left behind
by each way a charge can go wrong: the tax calculation failing, the card refusing, the network failing, the charge step failing, a card from a
country we don't serve. (The happy path and the reconcile of reloads left in flight are in test_stripe_webhooks.py and test_billing_audit.py.)"""
from types import SimpleNamespace

import pytest
import stripe

from acw import db
from acw.billing import ledger, pricing, stripe_service
from acw.config import get_settings

from .job_support import account

AMOUNT_CENTS = 2000


def reload_user(**cols) -> str:
    """An account with auto-reload on: under its threshold (no credit), a saved card, $20 a reload, $100 a month."""
    uid = account("reload@example.com", credit=0)
    values = {"auto_reload_enabled": 1, "auto_reload_threshold_micros": 5_000_000, "auto_reload_amount_cents": AMOUNT_CENTS,
              "auto_reload_monthly_cap_cents": 10_000, "default_payment_method": "pm_1", "stripe_customer_id": "cus_1", **cols}
    db.execute(f"UPDATE users SET {', '.join(f'{k} = ?' for k in values)} WHERE id = ?", (*values.values(), uid))
    return uid


class Stripe:
    """The parts of the Stripe client a reload uses; what each does is set by the test (a value is returned, an exception is raised)."""

    def __init__(self, monkeypatch, *, pi=None, tax=None, create=None, country="US"):
        self.created, self.taxed, self.cancels, self.captures = [], [], [], []
        self.pi = pi if pi is not None else SimpleNamespace(id="pi_r", status="requires_capture", latest_charge="ch_1")
        self.tax, self.create_error = tax, create
        fake = self

        def calculate(params):
            fake.taxed.append(params)
            if isinstance(fake.tax, Exception):
                raise fake.tax
            return fake.tax or SimpleNamespace(id="taxcalc_1", amount_total=params["line_items"][0]["amount"], tax_amount_exclusive=0)

        def create(params=None, options=None):
            fake.created.append((params, options))
            if isinstance(fake.create_error, Exception):
                raise fake.create_error
            return fake.pi

        def cancel(_id, params=None, options=None):
            fake.cancels.append((_id, options))
            return SimpleNamespace(id=_id, status="canceled")

        def capture(_id, params=None, options=None):
            fake.captures.append((_id, options))
            return SimpleNamespace(id=_id, status="succeeded", latest_charge="ch_1")

        charges = SimpleNamespace(retrieve=lambda _id, params=None: SimpleNamespace(
            payment_method_details=SimpleNamespace(card=SimpleNamespace(country=country))))
        client = SimpleNamespace(v1=SimpleNamespace(tax=SimpleNamespace(calculations=SimpleNamespace(create=calculate)), charges=charges,
                                                    payment_intents=SimpleNamespace(create=create, cancel=cancel, capture=capture)))
        monkeypatch.setattr(stripe_service, "client", lambda: client)


def user_row(uid):
    return dict(db.one("SELECT * FROM users WHERE id = ?", (uid,)))


def payments():
    return [dict(r) for r in db.all_rows("SELECT * FROM payments ORDER BY created_at")]


def audit(action):
    return [r["detail"] for r in db.all_rows("SELECT detail FROM audit_log WHERE action = ?", (action,))]


def nothing_started(uid, fake):
    assert payments() == [] and fake.created == [] and fake.taxed == [] and user_row(uid)["auto_reload_in_flight"] == 0


# --------------------------------------------------------------------------------------------------------------------- skip and cap

@pytest.mark.parametrize("why, cols", [
    ("switched off", {"auto_reload_enabled": 0}),
    ("one already on its way", {"auto_reload_in_flight": 1}),
    ("no saved card", {"default_payment_method": None}),
    ("a suspended account", {"status": "suspended"}),
    ("a threshold the reload would not lift the balance over", {"auto_reload_threshold_micros": AMOUNT_CENTS * pricing.MICROS_PER_CENT}),
    ("a threshold over the amount", {"auto_reload_threshold_micros": AMOUNT_CENTS * pricing.MICROS_PER_CENT + 1}),
])
def test_nothing_is_charged_when_auto_reload_does_not_apply(monkeypatch, why, cols):
    uid = reload_user(**cols)
    fake = Stripe(monkeypatch)
    before = user_row(uid)
    assert stripe_service.run_auto_reload(uid) == "skip", why
    assert payments() == [] and fake.created == [] and user_row(uid) == before


def test_an_unknown_account_and_a_balance_over_the_threshold_are_skipped(monkeypatch):
    fake = Stripe(monkeypatch)
    assert stripe_service.run_auto_reload("nobody") == "skip"
    uid = reload_user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")  # exactly the threshold: not under it
    assert stripe_service.run_auto_reload(uid) == "skip"
    db.execute("UPDATE users SET held_micros = 1 WHERE id = ?", (uid,))  # what is held is not available: now it is under
    assert stripe_service.run_auto_reload(uid) != "skip" and len(fake.created) == 1


def test_the_monthly_cap_the_smallest_amount_and_the_top_up_limit_stop_a_reload_before_anything_is_started(monkeypatch):
    fake = Stripe(monkeypatch)
    month = stripe_service._month()
    # this month's reloads plus this one would pass the cap
    uid = reload_user(auto_reload_month=month, auto_reload_month_cents=8_100)
    assert stripe_service.run_auto_reload(uid) == "cap"
    nothing_started(uid, fake)
    # (a month that has passed is not counted)
    db.execute("UPDATE users SET auto_reload_month = '2020-01' WHERE id = ?", (uid,))
    assert stripe_service.run_auto_reload(uid) != "cap" and len(fake.created) == 1
    # exactly the cap is allowed
    Stripe(monkeypatch, create=ConnectionError("down"))
    other = account("second@example.com", credit=0)
    db.execute("UPDATE users SET auto_reload_enabled=1, auto_reload_threshold_micros=5000000, auto_reload_amount_cents=2000,"
               " auto_reload_monthly_cap_cents=10000, default_payment_method='pm_2', stripe_customer_id='cus_2', auto_reload_month=?,"
               " auto_reload_month_cents=8000 WHERE id = ?", (month, other))
    assert stripe_service.run_auto_reload(other) != "cap"
    # an amount under the least a top-up can be
    small = account("small@example.com", credit=0)
    db.execute("UPDATE users SET auto_reload_enabled=1, auto_reload_threshold_micros=1000, auto_reload_amount_cents=?,"
               " default_payment_method='pm_3', stripe_customer_id='cus_3' WHERE id = ?", (pricing.TOPUP_MIN_CENTS - 1, small))
    count = len(payments())
    assert stripe_service.run_auto_reload(small) == "cap" and len(payments()) == count
    # an amount over what a first top-up may be
    first = account("first@example.com", credit=0)
    db.execute("UPDATE users SET auto_reload_enabled=1, auto_reload_threshold_micros=1000, auto_reload_amount_cents=?,"
               " auto_reload_monthly_cap_cents=100000, default_payment_method='pm_4', stripe_customer_id='cus_4' WHERE id = ?",
               (pricing.TOPUP_MAX_CENTS, first))
    assert stripe_service.run_auto_reload(first) == "cap"


def test_a_started_reload_is_one_pending_payment_with_the_flag_and_the_months_total_set_first(monkeypatch):
    uid = reload_user(auto_reload_month="2020-01", auto_reload_month_cents=999)
    fake = Stripe(monkeypatch, create=ConnectionError("down"))
    assert stripe_service.run_auto_reload(uid) == "error"
    (p,) = payments()
    assert (p["kind"], p["status"], p["amount_cents"], p["user_id"]) == ("auto_reload", "pending", AMOUNT_CENTS, uid)
    u = user_row(uid)
    assert (u["auto_reload_in_flight"], u["auto_reload_month"], u["auto_reload_month_cents"]) == (1, stripe_service._month(), AMOUNT_CENTS)
    params, options = fake.created[0]
    assert params["amount"] == AMOUNT_CENTS and params["off_session"] is True and params["confirm"] is True and params["capture_method"] == "manual"
    assert params["customer"] == "cus_1" and params["payment_method"] == "pm_1"
    assert params["metadata"] == {"acw_user_id": uid, "acw_payment_id": p["id"], "acw_kind": "auto_reload"}
    assert options == {"idempotency_key": f"reload-{p['id']}"}
    assert stripe_service.run_auto_reload(uid) == "skip"  # (one is on its way)


# --------------------------------------------------------------------------------------------------------------------- the tax

@pytest.fixture
def tax_on(monkeypatch):
    monkeypatch.setenv("STRIPE_AUTOMATIC_TAX", "1")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_a_failing_tax_calculation_fails_the_reload_before_any_charge_and_clears_the_flag(monkeypatch, tax_on):
    uid = reload_user()
    fake = Stripe(monkeypatch, tax=RuntimeError("customer_tax_location_invalid"))
    assert stripe_service.run_auto_reload(uid) == "failed:tax_location"
    assert len(fake.taxed) == 1 and fake.created == []  # (no charge was tried)
    (p,) = payments()
    assert (p["status"], p["failure_code"]) == ("failed", "tax_location")
    u = user_row(uid)
    assert u["auto_reload_in_flight"] == 0 and u["auto_reload_failures"] == 1 and u["auto_reload_enabled"] == 1
    assert audit("auto_reload_failed") == ["tax_location"]


def test_the_tax_is_asked_for_the_amount_and_the_card_is_charged_what_it_says(monkeypatch, tax_on):
    uid = reload_user()
    taxed = SimpleNamespace(id="taxcalc_9", amount_total=2150, tax_amount_exclusive=150)
    fake = Stripe(monkeypatch, tax=taxed, create=ConnectionError("down"))
    assert stripe_service.run_auto_reload(uid) == "error"
    (p,) = payments()
    asked = fake.taxed[0]
    assert asked["customer"] == "cus_1" and asked["line_items"][0]["amount"] == AMOUNT_CENTS and asked["line_items"][0]["reference"] == p["id"]
    assert fake.created[0][0]["amount"] == 2150  # (the total with the tax)
    assert (p["tax_cents"], p["tax_calculation"]) == (150, "taxcalc_9")
    assert user_row(uid)["auto_reload_in_flight"] == 1


def test_without_automatic_tax_nothing_is_asked_about_it(monkeypatch):
    uid = reload_user()
    fake = Stripe(monkeypatch, create=ConnectionError("down"))
    stripe_service.run_auto_reload(uid)
    assert fake.taxed == [] and fake.created[0][0]["amount"] == AMOUNT_CENTS


# --------------------------------------------------------------------------------------------------------------------- the charge fails

@pytest.mark.parametrize("code", ["card_declined", "expired_card", "insufficient_funds", None])
def test_a_card_that_refuses_fails_the_reload_with_its_code_and_frees_the_flag(monkeypatch, code):
    uid = reload_user()
    Stripe(monkeypatch, create=stripe.CardError("Your card was declined.", "payment_method", code))
    want = code or "card_error"
    assert stripe_service.run_auto_reload(uid) == f"failed:{want}"
    (p,) = payments()
    assert (p["status"], p["failure_code"]) == ("failed", want)
    u = user_row(uid)
    assert (u["auto_reload_in_flight"], u["auto_reload_failures"]) == (0, 1)
    assert audit("auto_reload_failed") == [want]
    assert ledger.balance(uid)["balance_micros"] == 0


def test_three_failures_in_a_row_switch_auto_reload_off(monkeypatch):
    uid = reload_user()
    Stripe(monkeypatch, create=stripe.CardError("declined", "payment_method", "card_declined"))
    for n in (1, 2, 3):
        assert stripe_service.run_auto_reload(uid) == "failed:card_declined"
        u = user_row(uid)
        assert (u["auto_reload_failures"], u["auto_reload_enabled"]) == (n, 0 if n == 3 else 1)
    assert stripe_service.run_auto_reload(uid) == "skip"


@pytest.mark.parametrize("failure", [ConnectionError("network"), TimeoutError(), stripe.APIConnectionError("no route"), RuntimeError("boom")])
def test_a_request_that_may_have_gone_through_leaves_the_reload_in_flight_for_reconcile(monkeypatch, failure):
    uid = reload_user()
    Stripe(monkeypatch, create=failure)
    assert stripe_service.run_auto_reload(uid) == "error"
    (p,) = payments()
    assert p["status"] == "pending" and p["stripe_payment_intent"] is None
    u = user_row(uid)
    assert (u["auto_reload_in_flight"], u["auto_reload_failures"]) == (1, 0)  # (not a failure yet: it may have been charged)
    assert audit("auto_reload_failed") == []


def test_a_charge_step_that_fails_after_the_authorization_is_left_for_the_webhook(monkeypatch):
    uid = reload_user()
    Stripe(monkeypatch)

    def settle(pi, payment_id, user_id):
        raise RuntimeError("stripe down")

    monkeypatch.setattr(stripe_service, "_settle_authorized", settle)
    assert stripe_service.run_auto_reload(uid) == "error"
    (p,) = payments()
    assert p["status"] == "pending" and p["stripe_payment_intent"] == "pi_r"  # (the intent is known: the webhook retries the charge)
    assert user_row(uid)["auto_reload_in_flight"] == 1 and ledger.balance(uid)["balance_micros"] == 0


def test_crediting_that_fails_is_left_for_the_webhook_too(monkeypatch):
    uid = reload_user()
    Stripe(monkeypatch, pi=SimpleNamespace(id="pi_r", status="requires_capture", latest_charge="ch_1"))
    monkeypatch.setattr(stripe_service, "_settle_authorized", lambda pi, payment_id, user_id: SimpleNamespace(id="pi_r", status="succeeded", latest_charge="ch_1"))

    def credit(*a, **k):
        raise RuntimeError("country lookup failed")

    monkeypatch.setattr(stripe_service, "_credit_payment", credit)
    assert stripe_service.run_auto_reload(uid) == "error"
    assert user_row(uid)["auto_reload_in_flight"] == 1 and ledger.balance(uid)["balance_micros"] == 0


def test_a_card_from_a_country_we_dont_serve_is_never_charged_and_auto_reload_is_switched_off(monkeypatch):
    uid = reload_user()
    fake = Stripe(monkeypatch, country="DE")
    assert stripe_service.run_auto_reload(uid) == "refused_region"
    (p,) = payments()
    assert (p["status"], p["failure_code"]) == ("refused_region", "card_country_DE")
    assert fake.captures == [] and fake.cancels == [("pi_r", {"idempotency_key": f"blocked-country-cancel-{p['id']}"})]
    u = user_row(uid)
    assert (u["auto_reload_enabled"], u["auto_reload_in_flight"]) == (0, 0) and ledger.balance(uid)["balance_micros"] == 0
    assert audit("payment_refused_region")


def test_a_reload_by_a_card_we_serve_is_captured_and_credited_and_frees_the_flag(monkeypatch):
    uid = reload_user()
    fake = Stripe(monkeypatch, country="US")
    monkeypatch.setattr(stripe_service, "_record_tax_transaction", lambda _pid: None)
    assert stripe_service.run_auto_reload(uid) == "succeeded"
    (p,) = payments()
    assert p["status"] == "succeeded" and p["stripe_payment_intent"] == "pi_r" and fake.cancels == [] and len(fake.captures) == 1
    assert ledger.balance(uid)["balance_micros"] == AMOUNT_CENTS * pricing.MICROS_PER_CENT
    u = user_row(uid)
    assert (u["auto_reload_in_flight"], u["auto_reload_failures"]) == (0, 0)
    assert stripe_service.run_auto_reload(uid) == "skip"  # (the balance is over the threshold now)


def test_a_reload_that_is_not_done_yet_says_how_far_it_got(monkeypatch):
    uid = reload_user()
    Stripe(monkeypatch)
    monkeypatch.setattr(stripe_service, "_settle_authorized", lambda pi, payment_id, user_id: SimpleNamespace(id="pi_r", status="processing", latest_charge=None))
    assert stripe_service.run_auto_reload(uid) == "processing"
    assert user_row(uid)["auto_reload_in_flight"] == 1 and ledger.balance(uid)["balance_micros"] == 0
