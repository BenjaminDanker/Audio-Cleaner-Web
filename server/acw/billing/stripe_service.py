"""Stripe: top-ups via Checkout, saved-card auto-reload, refunds, disputes.

Money only enters the ledger from verified webhooks (or the synchronous result
of an off-session PaymentIntent we created), and every credit is keyed by a
Stripe object id so replays are harmless. See docs/payments.md for the policy.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import stripe

from .. import db
from ..config import _env, get_settings
from . import ledger
from .pricing import MICROS_PER_CENT, TOPUP_MAX_CENTS, TOPUP_MIN_CENTS

log = logging.getLogger(__name__)

AUTO_RELOAD_MAX_FAILURES = 3
EFW_AUTO_REFUND_MAX_CENTS = 5_000

# Not offered in the EU/EEA, UK or Switzerland (no VAT / consumer-law handling). Payments by
# cards issued there are refunded in full instead of credited. Override with
# ACW_BLOCKED_CARD_COUNTRIES (comma-separated ISO codes; "none" disables the guard).
BLOCKED_CARD_COUNTRIES = (
    "AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE"  # EU
    " IS LI NO"  # rest of the EEA
    " GB CH"
).split()


def blocked_card_countries() -> frozenset[str]:
    raw = _env("ACW_BLOCKED_CARD_COUNTRIES")
    if raw is None:
        return frozenset(BLOCKED_CARD_COUNTRIES)
    return frozenset(c.strip().upper() for c in raw.replace(" ", ",").split(",") if c.strip() and c.strip().lower() != "none")


def _card_country(charge_id: str) -> str | None:
    """Issuing country of the card behind a charge (None for non-card payments)."""
    charge = client().v1.charges.retrieve(charge_id)
    details = getattr(charge, "payment_method_details", None)
    card = getattr(details, "card", None) if details is not None else None
    country = getattr(card, "country", None) if card is not None else None
    return country.upper() if country else None


def _refund_blocked_payment(payment_id: str, user_id: str, charge_id: str, country: str) -> None:
    """Refund in full (idempotent), record it, and stop auto-reload for the user."""
    client().v1.refunds.create(
        params={"charge": charge_id, "reason": "requested_by_customer",
                "metadata": {"acw_payment_id": payment_id, "acw_reason": "blocked_card_country"}},
        options={"idempotency_key": f"blocked-country-{payment_id}"},
    )
    with db.tx() as c:
        c.execute(
            "UPDATE payments SET status = 'refused_region', stripe_charge = ?, failure_code = ?, updated_at = ?"
            " WHERE id = ? AND status != 'succeeded'",
            (charge_id, f"card_country_{country}", db.now(), payment_id),
        )
        c.execute("UPDATE users SET auto_reload_enabled = 0, auto_reload_in_flight = 0 WHERE id = ?", (user_id,))
        c.execute(
            "INSERT INTO audit_log(user_id, actor, action, detail, created_at) VALUES (?, 'stripe', 'payment_refused_region', ?, ?)",
            (user_id, f"{payment_id}:{charge_id}:{country}", db.now()),
        )
    log.warning("payment %s refunded: card issued in blocked country %s", payment_id, country)


class BillingUnavailable(Exception):
    pass


def client() -> stripe.StripeClient:
    key = get_settings().stripe_secret_key
    if not key:
        raise BillingUnavailable("Stripe is not configured")
    return stripe.StripeClient(key, max_network_retries=2)


def enabled() -> bool:
    return bool(get_settings().stripe_secret_key)


def ensure_customer(user_id: str) -> str:
    user = db.one("SELECT id, email, stripe_customer_id FROM users WHERE id = ?", (user_id,))
    if user["stripe_customer_id"]:
        return user["stripe_customer_id"]
    cust = client().v1.customers.create(
        params={"email": user["email"], "metadata": {"acw_user_id": user_id}},
        options={"idempotency_key": f"customer-{user_id}"},
    )
    db.execute(
        "UPDATE users SET stripe_customer_id = ? WHERE id = ? AND stripe_customer_id IS NULL",
        (cust.id, user_id),
    )
    return db.one("SELECT stripe_customer_id FROM users WHERE id = ?", (user_id,))["stripe_customer_id"]


def create_checkout(user_id: str, amount_cents: int, save_card: bool) -> str:
    if not (TOPUP_MIN_CENTS <= amount_cents <= TOPUP_MAX_CENTS):
        raise ValueError("amount out of range")
    settings = get_settings()
    customer = ensure_customer(user_id)
    payment_id = db.new_id()
    now = db.now()
    db.execute(
        "INSERT INTO payments(id, user_id, kind, status, amount_cents, created_at, updated_at)"
        " VALUES (?,?,'checkout','pending',?,?,?)",
        (payment_id, user_id, amount_cents, now, now),
    )
    pi_data: dict = {
        "metadata": {"acw_user_id": user_id, "acw_payment_id": payment_id, "acw_kind": "checkout"},
        "statement_descriptor_suffix": "CREDITS",
    }
    if save_card:
        pi_data["setup_future_usage"] = "off_session"
    tax_params: dict = {}
    if settings.stripe_automatic_tax:
        # Tax is added on top of the credit amount; the address (and an
        # optional VAT ID for business reverse charge) is saved on the
        # customer so off-session reloads can be taxed the same way.
        tax_params = {
            "automatic_tax": {"enabled": True},
            "billing_address_collection": "required",
            "customer_update": {"address": "auto", "name": "auto"},
            "tax_id_collection": {"enabled": True},
        }
    session = client().v1.checkout.sessions.create(
        params={
            **tax_params,
            "mode": "payment",
            "customer": customer,
            "client_reference_id": user_id,
            "payment_method_types": ["card"],
            "line_items": [
                {
                    "quantity": 1,
                    "price_data": {
                        "currency": "usd",
                        "unit_amount": amount_cents,
                        "tax_behavior": "exclusive",
                        "product_data": {
                            "name": "Audio Cleaner prepaid credit",
                            "description": "Usage credit for audio processing. Credit is not a gift card and has no cash value.",
                            "tax_code": settings.stripe_tax_code,
                        },
                    },
                }
            ],
            "payment_intent_data": pi_data,
            # 3DS on the first, on-session purchase shifts fraud liability in
            # most cases and makes later off-session reloads less likely to fail.
            "payment_method_options": {"card": {"request_three_d_secure": "any"}},
            "metadata": {"acw_user_id": user_id, "acw_payment_id": payment_id},
            # Shown above the Pay button: paying is agreeing to the Terms.
            "custom_text": {"submit": {"message": f"By paying you agree to the [Terms of Service]({settings.url('/terms')})."}},
            "success_url": settings.url("/app/billing?checkout=success"),
            "cancel_url": settings.url("/app/billing?checkout=cancel"),
            "expires_at": now + 60 * 60,
        },
        options={"idempotency_key": f"checkout-{payment_id}"},
    )
    db.execute(
        "UPDATE payments SET stripe_checkout_session = ?, updated_at = ? WHERE id = ?",
        (session.id, db.now(), payment_id),
    )
    return session.url


def billing_portal_url(user_id: str) -> str:
    customer = ensure_customer(user_id)
    portal = client().v1.billing_portal.sessions.create(
        params={"customer": customer, "return_url": get_settings().url("/app/billing")}
    )
    return portal.url


# ------------------------------------------------------------------ crediting

def _credit_payment(
    payment_id: str,
    user_id: str,
    *,
    checkout_session: str | None,
    pi_id: str | None,
    charge_id: str | None,
    pm_id: str | None,
    save_pm: bool,
) -> None:
    """Credit the USD amount *we* priced, once, if the Stripe object matches
    our payment row. Never trust amounts from the event: with Adaptive Pricing
    they can be in the customer's currency. Cards issued in blocked countries
    (see BLOCKED_CARD_COUNTRIES) are refunded instead of credited."""
    prior = db.one("SELECT status FROM payments WHERE id = ? AND user_id = ?", (payment_id, user_id))
    if prior is not None and prior["status"] in ("succeeded", "refused_region"):
        return
    if prior is not None and charge_id:
        country = _card_country(charge_id)  # a lookup failure raises: the webhook is retried
        if country in blocked_card_countries():
            _refund_blocked_payment(payment_id, user_id, charge_id, country)
            return
    with db.tx() as c:
        p = c.execute("SELECT * FROM payments WHERE id = ?", (payment_id,)).fetchone()
        if p is None or p["user_id"] != user_id:
            log.error("payment %s does not belong to user %s", payment_id, user_id)
            return
        if p["status"] == "succeeded":
            return
        if checkout_session is not None and p["stripe_checkout_session"] != checkout_session:
            log.error("payment %s: checkout session mismatch", payment_id)
            return
        if checkout_session is None and p["stripe_payment_intent"] not in (None, pi_id):
            log.error("payment %s: payment intent mismatch", payment_id)
            return
        amount_cents = int(p["amount_cents"])
        ledger.credit(
            user_id,
            amount_cents * MICROS_PER_CENT,
            "topup",
            "payment",
            payment_id,
            f"Top-up ${amount_cents / 100:.2f}" + (" (auto-reload)" if p["kind"] == "auto_reload" else ""),
            c=c,
        )
        c.execute(
            "UPDATE payments SET status = 'succeeded', stripe_payment_intent = COALESCE(stripe_payment_intent, ?),"
            " stripe_charge = ?, updated_at = ? WHERE id = ?",
            (pi_id, charge_id, db.now(), payment_id),
        )
        if p["kind"] == "auto_reload":
            c.execute(
                "UPDATE users SET auto_reload_in_flight = 0, auto_reload_failures = 0 WHERE id = ?",
                (user_id,),
            )
        if save_pm and pm_id:
            c.execute("UPDATE users SET default_payment_method = ? WHERE id = ?", (pm_id, user_id))


def _handle_checkout_completed(session: dict) -> None:
    if session.get("mode") != "payment" or session.get("payment_status") != "paid":
        return
    meta = session.get("metadata") or {}
    payment_id, user_id = meta.get("acw_payment_id"), meta.get("acw_user_id")
    if not payment_id or not user_id:
        return
    pi_id = session.get("payment_intent")
    pm_id = charge_id = None
    save_pm = False
    if pi_id:
        pi = client().v1.payment_intents.retrieve(pi_id)
        pm_id = pi.payment_method if isinstance(pi.payment_method, str) else getattr(pi.payment_method, "id", None)
        charge_id = pi.latest_charge if isinstance(pi.latest_charge, str) else getattr(pi.latest_charge, "id", None)
        save_pm = pi.setup_future_usage == "off_session"
    tax = int((session.get("total_details") or {}).get("amount_tax") or 0)
    db.execute("UPDATE payments SET tax_cents = ? WHERE id = ? AND status = 'pending'", (tax, payment_id))
    _credit_payment(
        payment_id, user_id, checkout_session=session.get("id"), pi_id=pi_id, charge_id=charge_id, pm_id=pm_id, save_pm=save_pm
    )


def _handle_pi_succeeded(pi: dict) -> None:
    meta = pi.get("metadata") or {}
    if meta.get("acw_kind") != "auto_reload":
        return  # checkout payments are credited from checkout.session.completed
    if pi.get("currency") != "usd":
        log.error("auto-reload %s in unexpected currency %s", pi.get("id"), pi.get("currency"))
        return
    _credit_payment(
        meta["acw_payment_id"], meta["acw_user_id"], checkout_session=None, pi_id=pi["id"],
        charge_id=pi.get("latest_charge"), pm_id=None, save_pm=False,
    )
    _record_tax_transaction(meta["acw_payment_id"])


def _record_reload_failure(payment_id: str, user_id: str, code: str | None) -> None:
    with db.tx() as c:
        c.execute(
            "UPDATE payments SET status = 'failed', failure_code = ?, updated_at = ? WHERE id = ? AND status = 'pending'",
            (code, db.now(), payment_id),
        )
        c.execute(
            "UPDATE users SET auto_reload_in_flight = 0, auto_reload_failures = auto_reload_failures + 1,"
            " auto_reload_enabled = CASE WHEN auto_reload_failures + 1 >= ? THEN 0 ELSE auto_reload_enabled END"
            " WHERE id = ?",
            (AUTO_RELOAD_MAX_FAILURES, user_id),
        )
        c.execute(
            "INSERT INTO audit_log(user_id, actor, action, detail, created_at) VALUES (?, 'stripe', 'auto_reload_failed', ?, ?)",
            (user_id, code, db.now()),
        )


def _handle_pi_failed(pi: dict) -> None:
    meta = pi.get("metadata") or {}
    if meta.get("acw_kind") != "auto_reload":
        return
    err = pi.get("last_payment_error") or {}
    _record_reload_failure(meta["acw_payment_id"], meta["acw_user_id"], err.get("decline_code") or err.get("code"))


def _user_for_charge(charge: dict) -> tuple[str | None, str | None]:
    pi_id = charge.get("payment_intent")
    row = db.one("SELECT id, user_id FROM payments WHERE stripe_payment_intent = ? OR stripe_charge = ?", (pi_id, charge.get("id")))
    if row is None:
        return None, None
    return row["user_id"], row["id"]


def _handle_charge_refunded(charge: dict) -> None:
    user_id, payment_id = _user_for_charge(charge)
    if user_id is None:
        return
    refunds = (charge.get("refunds") or {}).get("data") or []
    if not refunds:
        refunds = [r for r in client().v1.refunds.list(params={"charge": charge["id"], "limit": 100}).data]
    p = db.one("SELECT amount_cents, tax_cents, status FROM payments WHERE id = ?", (payment_id,))
    if p["status"] == "refused_region":
        return  # our own full refund of a payment that was never credited
    for r in refunds:
        r = r if isinstance(r, dict) else r.to_dict()
        if r.get("status") not in ("succeeded", "pending"):
            continue
        credit_cents = credit_part(int(r["amount"]), p["amount_cents"], p["tax_cents"])
        ledger.debit_available(
            user_id,
            credit_cents * MICROS_PER_CENT,
            "refund",
            "stripe_refund",
            r["id"],
            f"Refund ${credit_cents / 100:.2f} credit",
            allow_partial=True,
        )
    refunded = credit_part(int(charge.get("amount_refunded") or 0), p["amount_cents"], p["tax_cents"])
    db.execute(
        "UPDATE payments SET refunded_cents = MAX(refunded_cents, ?),"
        " status = CASE WHEN MAX(refunded_cents, ?) >= amount_cents THEN 'refunded' ELSE status END,"
        " updated_at = ? WHERE id = ?",
        (refunded, refunded, db.now(), payment_id),
    )


def suspend_and_zero(user_id: str, reason: str, ref_id: str) -> None:
    """Disputes: stop all spending and remove remaining credit."""
    with db.tx() as c:
        c.execute(
            "UPDATE users SET status = 'suspended', auto_reload_enabled = 0 WHERE id = ? AND status != 'deleted'",
            (user_id,),
        )
        ledger.release_all_holds(user_id, c)
        row = c.execute("SELECT balance_micros FROM users WHERE id = ?", (user_id,)).fetchone()
        if row["balance_micros"] > 0:
            ledger.debit_available(user_id, row["balance_micros"], "dispute", "stripe_dispute", ref_id, reason, allow_partial=True, c=c)
        c.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        c.execute(
            "INSERT INTO audit_log(user_id, actor, action, detail, created_at) VALUES (?, 'stripe', 'suspended', ?, ?)",
            (user_id, reason, db.now()),
        )


def _handle_dispute_created(dispute: dict) -> None:
    charge_id = dispute.get("charge")
    row = db.one("SELECT id, user_id FROM payments WHERE stripe_charge = ?", (charge_id,))
    if row is None:
        log.error("dispute %s for unknown charge %s", dispute.get("id"), charge_id)
        return
    db.execute("UPDATE payments SET status = 'disputed', updated_at = ? WHERE id = ?", (db.now(), row["id"]))
    suspend_and_zero(row["user_id"], f"dispute {dispute.get('reason')}", dispute["id"])


def _handle_early_fraud_warning(efw: dict) -> None:
    charge_id = efw.get("charge")
    row = db.one("SELECT id, user_id, amount_cents, refunded_cents FROM payments WHERE stripe_charge = ?", (charge_id,))
    if row is None:
        return
    suspend_and_zero(row["user_id"], "early fraud warning", efw["id"])
    remaining = row["amount_cents"] - row["refunded_cents"]
    if 0 < remaining <= EFW_AUTO_REFUND_MAX_CENTS:
        # Refunding before a dispute avoids the dispute fee and VAMP count.
        client().v1.refunds.create(
            params={"charge": charge_id, "reason": "fraudulent", "metadata": {"acw_reason": "efw"}},
            options={"idempotency_key": f"efw-refund-{efw['id']}"},
        )


def _handle_pm_detached(pm: dict) -> None:
    db.execute("UPDATE users SET default_payment_method = NULL WHERE default_payment_method = ?", (pm.get("id"),))


HANDLERS = {
    "checkout.session.completed": _handle_checkout_completed,
    "checkout.session.async_payment_succeeded": _handle_checkout_completed,
    "payment_intent.succeeded": _handle_pi_succeeded,
    "payment_intent.payment_failed": _handle_pi_failed,
    "charge.refunded": _handle_charge_refunded,
    "charge.dispute.created": _handle_dispute_created,
    "radar.early_fraud_warning.created": _handle_early_fraud_warning,
    "payment_method.detached": _handle_pm_detached,
}


def handle_webhook(payload: bytes, sig_header: str | None) -> str:
    secret = get_settings().stripe_webhook_secret
    if not secret:
        raise BillingUnavailable("webhook secret not configured")
    event = client().construct_event(payload, sig_header, secret)  # raises on bad signature
    handler = HANDLERS.get(event.type)
    seen = db.one("SELECT processed_at FROM stripe_events WHERE id = ?", (event.id,))
    if seen is not None and seen["processed_at"] is not None:
        return "duplicate"
    db.execute(
        "INSERT INTO stripe_events(id, type, received_at) VALUES (?,?,?) ON CONFLICT(id) DO NOTHING",
        (event.id, event.type, db.now()),
    )
    if handler is not None:
        handler(event.data.object.to_dict())
    db.execute("UPDATE stripe_events SET processed_at = ? WHERE id = ?", (db.now(), event.id))
    return "ok" if handler else "ignored"


# ------------------------------------------------------------------ auto-reload

def _month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def users_needing_reload() -> list[str]:
    rows = db.all_rows(
        "SELECT id FROM users WHERE status = 'active' AND auto_reload_enabled = 1 AND auto_reload_in_flight = 0"
        " AND default_payment_method IS NOT NULL AND stripe_customer_id IS NOT NULL"
        " AND auto_reload_failures < ? AND balance_micros - held_micros < auto_reload_threshold_micros",
        (AUTO_RELOAD_MAX_FAILURES,),
    )
    return [r["id"] for r in rows]


def run_auto_reload(user_id: str) -> str:
    """Charge the saved card once if the user is under their threshold."""
    month = _month()
    with db.tx() as c:
        u = c.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if (
            u is None
            or u["status"] != "active"
            or not u["auto_reload_enabled"]
            or u["auto_reload_in_flight"]
            or not u["default_payment_method"]
            or u["balance_micros"] - u["held_micros"] >= u["auto_reload_threshold_micros"]
        ):
            return "skip"
        spent = u["auto_reload_month_cents"] if u["auto_reload_month"] == month else 0
        amount = int(u["auto_reload_amount_cents"])
        if amount < TOPUP_MIN_CENTS or spent + amount > u["auto_reload_monthly_cap_cents"]:
            return "cap"
        payment_id = db.new_id()
        now = db.now()
        c.execute(
            "INSERT INTO payments(id, user_id, kind, status, amount_cents, created_at, updated_at)"
            " VALUES (?,?,'auto_reload','pending',?,?,?)",
            (payment_id, user_id, amount, now, now),
        )
        c.execute(
            "UPDATE users SET auto_reload_in_flight = 1, auto_reload_month = ?, auto_reload_month_cents = ? WHERE id = ?",
            (month, spent + amount, user_id),
        )
        customer, pm = u["stripe_customer_id"], u["default_payment_method"]
    total, tax, calc_id = amount, 0, None
    if get_settings().stripe_automatic_tax:
        try:
            calc = client().v1.tax.calculations.create(params={
                "currency": "usd",
                "customer": customer,
                "line_items": [{"amount": amount, "reference": payment_id, "quantity": 1,
                                "tax_code": get_settings().stripe_tax_code, "tax_behavior": "exclusive"}],
            })
        except Exception as e:  # noqa: BLE001 - typically no saved address
            log.warning("tax calculation for reload %s failed: %s", payment_id, e)
            _record_reload_failure(payment_id, user_id, "tax_location")
            return "failed:tax_location"
        total, tax, calc_id = int(calc.amount_total), int(calc.tax_amount_exclusive), calc.id
        db.execute("UPDATE payments SET tax_cents = ?, tax_calculation = ? WHERE id = ?", (tax, calc_id, payment_id))
    try:
        pi = client().v1.payment_intents.create(
            params={
                "amount": total,
                "currency": "usd",
                "customer": customer,
                "payment_method": pm,
                "off_session": True,
                "confirm": True,
                "description": "Audio Cleaner auto-reload",
                "statement_descriptor_suffix": "RELOAD",
                "metadata": {"acw_user_id": user_id, "acw_payment_id": payment_id, "acw_kind": "auto_reload"},
            },
            options={"idempotency_key": f"reload-{payment_id}"},
        )
    except stripe.CardError as e:
        code = getattr(e, "code", None) or "card_error"
        _record_reload_failure(payment_id, user_id, code)
        return f"failed:{code}"
    except Exception:  # noqa: BLE001 - network etc.; leave pending for the webhook/sweeper
        log.exception("auto-reload request for %s failed", user_id)
        _record_reload_failure(payment_id, user_id, "api_error")
        return "error"
    db.execute("UPDATE payments SET stripe_payment_intent = ?, updated_at = ? WHERE id = ?", (pi.id, db.now(), payment_id))
    if pi.status == "succeeded":
        charge = pi.latest_charge if isinstance(pi.latest_charge, str) else getattr(pi.latest_charge, "id", None)
        try:
            _credit_payment(payment_id, user_id, checkout_session=None, pi_id=pi.id, charge_id=charge, pm_id=None, save_pm=False)
        except Exception:  # noqa: BLE001 - e.g. country lookup failed; the webhook retries the check
            log.exception("crediting auto-reload %s failed; left for the webhook", payment_id)
            return "error"
        if db.one("SELECT status FROM payments WHERE id = ?", (payment_id,))["status"] != "succeeded":
            return "refused_region"
        _record_tax_transaction(payment_id)
        return "succeeded"
    return pi.status


def refund_unused(user_id: str, payment_id: str, max_age_days: int = 14) -> int:
    """Self-serve refund of unused credit from one recent payment. Returns cents refunded."""
    p = db.one("SELECT * FROM payments WHERE id = ? AND user_id = ?", (payment_id, user_id))
    if p is None or p["status"] != "succeeded" or not p["stripe_charge"]:
        raise ValueError("This payment can't be refunded.")
    if db.now() - p["created_at"] > max_age_days * 86400:
        raise ValueError(f"Payments older than {max_age_days} days can't be refunded online; contact support.")
    bal = ledger.balance(user_id)
    refundable = min(p["amount_cents"] - p["refunded_cents"], bal["available_micros"] // MICROS_PER_CENT)
    if refundable <= 0:
        raise ValueError("No unused credit to refund from this payment.")
    # Stripe keeps its processing fee when we refund, so it comes out of the
    # first refund of a payment (stated in the Terms and on the Billing page).
    fee = card_fee_cents(p) if p["refunded_cents"] == 0 else 0
    if refundable <= fee:
        raise ValueError(f"The unused credit (${refundable / 100:.2f}) doesn't cover the card fee (${fee / 100:.2f}).")
    to_card = refundable - fee
    gross = to_card + tax_part(to_card, p["amount_cents"], p["tax_cents"])
    hold_id = ledger.create_hold(user_id, refundable * MICROS_PER_CENT, "refund", payment_id)
    try:
        refund = client().v1.refunds.create(
            params={"charge": p["stripe_charge"], "amount": gross, "reason": "requested_by_customer",
                    "metadata": {"acw_user_id": user_id, "acw_payment_id": payment_id}},
            options={"idempotency_key": f"refund-{payment_id}-{refundable}-{p['refunded_cents']}"},
        )
    except Exception:
        ledger.release_hold(hold_id)
        raise
    _reverse_tax(p, refund.id, gross)
    ledger.capture_hold(
        hold_id, refundable * MICROS_PER_CENT,
        f"Refund ${to_card / 100:.2f}" + (f" (card fee ${fee / 100:.2f} kept)" if fee else ""),
        kind="refund", ref_type="stripe_refund", ref_id=refund.id,
    )
    db.execute(
        "UPDATE payments SET refunded_cents = refunded_cents + ?, updated_at = ? WHERE id = ?",
        (refundable, db.now(), payment_id),
    )
    db.audit("user", "refund", user_id=user_id, detail=f"{payment_id}:{refundable}:fee={fee}")
    return to_card


def card_fee_cents(p) -> int:
    """What Stripe charged us for this payment (its balance transaction), else
    the standard US card rate (2.9% + 30c)."""
    try:
        charge = client().v1.charges.retrieve(p["stripe_charge"], params={"expand": ["balance_transaction"]})
        bt = charge.balance_transaction
        if bt is not None and getattr(bt, "currency", "usd") == "usd":
            return int(bt.fee)
    except Exception:  # noqa: BLE001
        log.warning("could not read the Stripe fee for %s; using the standard rate", p["id"])
    return round((p["amount_cents"] + p["tax_cents"]) * 0.029) + 30


# ------------------------------------------------------------------ tax helpers

def tax_part(credit_cents: int, amount_cents: int, tax_cents: int) -> int:
    """Tax that belongs to `credit_cents` of a payment (proportional)."""
    if amount_cents <= 0 or tax_cents <= 0:
        return 0
    return round(credit_cents * tax_cents / amount_cents)


def credit_part(gross_cents: int, amount_cents: int, tax_cents: int) -> int:
    """Credit (pre-tax) share of a gross refund amount."""
    total = amount_cents + tax_cents
    if total <= 0:
        return gross_cents
    return min(amount_cents, round(gross_cents * amount_cents / total))


def _record_tax_transaction(payment_id: str) -> None:
    """Off-session reloads use the Tax Calculation API, so we must turn the
    calculation into a transaction ourselves (Checkout does this itself)."""
    p = db.one("SELECT tax_calculation, tax_transaction, status FROM payments WHERE id = ?", (payment_id,))
    if p is None or not p["tax_calculation"] or p["tax_transaction"] or p["status"] != "succeeded":
        return
    try:
        tx = client().v1.tax.transactions.create_from_calculation(
            params={"calculation": p["tax_calculation"], "reference": payment_id},
            options={"idempotency_key": f"taxtx-{payment_id}"},
        )
    except Exception:  # noqa: BLE001 - retried by the housekeeping loop
        log.exception("recording tax transaction for %s failed", payment_id)
        return
    db.execute("UPDATE payments SET tax_transaction = ? WHERE id = ?", (tx.id, payment_id))


def pending_tax_transactions() -> list[str]:
    rows = db.all_rows(
        "SELECT id FROM payments WHERE status = 'succeeded' AND tax_calculation IS NOT NULL AND tax_transaction IS NULL"
    )
    return [r["id"] for r in rows]


def _reverse_tax(p, refund_id: str, gross_cents: int) -> None:
    if not p["tax_transaction"] or not p["tax_cents"]:
        return  # Checkout payments: Stripe tracks refunds itself
    try:
        client().v1.tax.transactions.create_reversal(
            params={"mode": "partial", "original_transaction": p["tax_transaction"],
                    "reference": f"refund-{refund_id}", "flat_amount": -gross_cents},
            options={"idempotency_key": f"taxrev-{refund_id}"},
        )
    except Exception:  # noqa: BLE001
        log.exception("tax reversal for refund %s failed; reverse it in the Stripe dashboard", refund_id)
