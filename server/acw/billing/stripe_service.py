"""Stripe: top-ups via Checkout, saved-card auto-reload, refunds, disputes.

Money only enters the ledger from verified webhooks (or the synchronous result
of an off-session PaymentIntent we created), and every credit is keyed by a
Stripe object id so replays are harmless. See docs/payments.md for the policy.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone

import stripe

from .. import db
from ..config import _env, get_settings
from . import ledger, trust
from .pricing import MICROS_PER_CENT, TOPUP_MAX_CENTS, TOPUP_MIN_CENTS

log = logging.getLogger(__name__)

AUTO_RELOAD_MAX_FAILURES = 3
EFW_AUTO_REFUND_MAX_CENTS = 5_000

# Not offered in the EU/EEA, UK or Switzerland (no VAT / consumer-law handling). Every payment is
# only authorized at first (capture_method=manual); a card issued there has its authorization
# cancelled, so it is never charged and no card fee is paid. Override with
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


def _get(obj, key: str):
    """A field of a Stripe object or of an event's plain dict."""
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _id_of(ref) -> str | None:
    """An id from an expandable field (either the id itself or the expanded object)."""
    return ref if isinstance(ref, str) or ref is None else _get(ref, "id")


def _settle_authorized(pi, payment_id: str, user_id: str):
    """Payments are authorized first and charged only if the card was issued in a country we
    serve: capture the authorization, or cancel it for a blocked country (the card is never
    charged). Returns the PaymentIntent as it is now, or None when it was refused. Idempotent;
    a failed Stripe call raises so the webhook is retried (an authorization lasts 7 days)."""
    if _get(pi, "status") != "requires_capture":
        return pi
    row = db.one("SELECT user_id, status FROM payments WHERE id = ?", (payment_id,))
    if row is None or row["user_id"] != user_id:
        log.error("authorization %s: payment %s does not belong to user %s", _get(pi, "id"), payment_id, user_id)
        return pi
    if row["status"] == "refused_region":
        return None
    charge_id = _id_of(_get(pi, "latest_charge"))
    country = _card_country(charge_id) if charge_id else None
    if country is None or country in blocked_card_countries():
        # Not a card (no issuing country to check), or a card from a country we don't serve: never charged.
        country = country or "unknown"
        client().v1.payment_intents.cancel(
            _get(pi, "id"), params={"cancellation_reason": "requested_by_customer"},
            options={"idempotency_key": f"blocked-country-cancel-{payment_id}"},
        )
        _mark_refused(payment_id, user_id, charge_id, country)
        log.warning("payment %s not charged: card issued in blocked country %s", payment_id, country)
        return None
    return client().v1.payment_intents.capture(_get(pi, "id"), options={"idempotency_key": f"capture-{payment_id}"})


def _mark_refused(payment_id: str, user_id: str, charge_id: str | None, country: str) -> None:
    """Record a payment refused for its card's country, and stop auto-reload for the user."""
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


def _refund_blocked_payment(payment_id: str, user_id: str, charge_id: str, country: str) -> None:
    """Backstop for a blocked-country payment that was charged anyway (e.g. captured before
    authorize-first existed): refund in full (idempotent) and record it."""
    client().v1.refunds.create(
        params={"charge": charge_id, "reason": "requested_by_customer",
                "metadata": {"acw_payment_id": payment_id, "acw_reason": "blocked_card_country"}},
        options={"idempotency_key": f"blocked-country-{payment_id}"},
    )
    _mark_refused(payment_id, user_id, charge_id, country)
    log.warning("payment %s refunded: card issued in blocked country %s", payment_id, country)


class BillingUnavailable(Exception):
    pass


def client() -> stripe.StripeClient:
    key = get_settings().stripe_secret_key
    if not key:
        raise BillingUnavailable("Stripe is not configured")
    # 25 s per attempt (the library's default is 80): a slow Stripe must not tie up our threads for minutes.
    return stripe.StripeClient(key, max_network_retries=2, http_client=stripe.RequestsClient(timeout=25))


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
    limit = trust.topup_max_cents(user_id)
    if amount_cents > limit:
        raise ValueError(trust.topup_limit_message(limit))
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
    # Authorize only; the webhook charges it once the card's issuing country is checked.
    pi_data["capture_method"] = "manual"
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
            # Stripe no longer takes payment_method_types here: the methods come from a payment method configuration
            # (ours offers cards only, see config). Without one, the account's default applies, and a payment that
            # is not a card is refused before it is charged (_settle_authorized).
            **({"payment_method_configuration": settings.stripe_payment_method_configuration}
               if settings.stripe_payment_method_configuration else {}),
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
            # Let Stripe and the bank decide when to ask for 3DS. Forcing it
            # ("any") makes cards whose bank has no 3DS challenge fail outright.
            "payment_method_options": {"card": {"request_three_d_secure": "automatic"}},
            "metadata": {"acw_user_id": user_id, "acw_payment_id": payment_id},
            # Shown above the Pay button: paying is agreeing to the Terms.
            "custom_text": {"submit": {"message": f"By paying you agree to the [Terms of Service]({settings.url('/terms')})."}},
            "success_url": settings.url(f"/app/billing?checkout=success&p={payment_id}"),
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
    settings = get_settings()
    params = {"customer": customer, "return_url": settings.url("/app/billing")}
    if settings.stripe_portal_configuration:  # one made with the API/CLI (docs/payments.md)
        params["configuration"] = settings.stripe_portal_configuration
    portal = client().v1.billing_portal.sessions.create(params=params)
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
    # An authorized (not yet charged) payment completes the session as "unpaid".
    if session.get("mode") != "payment" or session.get("payment_status") not in ("paid", "unpaid"):
        return
    meta = session.get("metadata") or {}
    payment_id, user_id = meta.get("acw_payment_id"), meta.get("acw_user_id")
    if not payment_id or not user_id:
        return
    p = db.one("SELECT stripe_checkout_session FROM payments WHERE id = ? AND user_id = ?", (payment_id, user_id))
    if p is None or p["stripe_checkout_session"] != session.get("id"):
        log.error("payment %s: checkout session mismatch", payment_id)
        return
    pi_id = session.get("payment_intent")
    pm_id = charge_id = None
    save_pm = False
    if not pi_id:
        return
    pi = _settle_authorized(client().v1.payment_intents.retrieve(pi_id), payment_id, user_id)
    if pi is None or _get(pi, "status") != "succeeded":
        return  # refused for its country, or not charged (yet)
    pm_id = _id_of(_get(pi, "payment_method"))
    charge_id = _id_of(_get(pi, "latest_charge"))
    save_pm = _get(pi, "setup_future_usage") == "off_session"
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


def _handle_pi_capturable(pi: dict) -> None:
    """An authorization is waiting: charge it or cancel it (also covers a reload whose own request
    could not finish). Checkout payments are then credited from checkout.session.completed,
    reloads from payment_intent.succeeded."""
    meta = pi.get("metadata") or {}
    if meta.get("acw_kind") not in ("checkout", "auto_reload") or not meta.get("acw_payment_id"):
        return
    _settle_authorized(pi, meta["acw_payment_id"], meta["acw_user_id"])


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
        seen = db.one("SELECT 1 FROM ledger WHERE kind = 'refund' AND ref_type = 'stripe_refund' AND ref_id = ?", (r["id"],))
        taken = ledger.debit_available(
            user_id,
            credit_cents * MICROS_PER_CENT,
            "refund",
            "stripe_refund",
            r["id"],
            f"Refund ${credit_cents / 100:.2f} credit",
            allow_partial=True,
        )
        # A refund the person asked for here is booked in full by refund_unused (its hold is what keeps
        # this debit from finding the money); anything else that the balance couldn't cover is lost to us.
        if not seen and taken < credit_cents * MICROS_PER_CENT and not (r.get("metadata") or {}).get("acw_user_id"):
            _audit_once("refund_unrecovered", user_id, f"{payment_id}:{r['id']}:{credit_cents * MICROS_PER_CENT - taken} micros")
    refunded = credit_part(int(charge.get("amount_refunded") or 0), p["amount_cents"], p["tax_cents"])
    db.execute(
        "UPDATE payments SET refunded_cents = MAX(refunded_cents, ?),"
        " status = CASE WHEN MAX(refunded_cents, ?) >= amount_cents THEN 'refunded' ELSE status END,"
        " updated_at = ? WHERE id = ?",
        (refunded, refunded, db.now(), payment_id),
    )


def _audit_once(action: str, user_id: str, detail: str) -> None:
    """Audit entry for money we could not get back (a webhook may be delivered again)."""
    if db.one("SELECT 1 FROM audit_log WHERE action = ? AND user_id = ? AND detail = ?", (action, user_id, detail)) is None:
        db.audit("stripe", action, user_id=user_id, detail=detail)
        log.warning("%s for %s: %s", action, user_id, detail)


def suspend_and_zero(user_id: str, reason: str, ref_id: str) -> int:
    """Disputes: stop all spending and remove remaining credit. Returns the micros taken back."""
    taken = 0
    with db.tx() as c:
        c.execute(
            "UPDATE users SET status = 'suspended', auto_reload_enabled = 0 WHERE id = ? AND status != 'deleted'",
            (user_id,),
        )
        ledger.release_all_holds(user_id, c)
        row = c.execute("SELECT balance_micros FROM users WHERE id = ?", (user_id,)).fetchone()
        if row["balance_micros"] > 0:
            taken = ledger.debit_available(user_id, row["balance_micros"], "dispute", "stripe_dispute", ref_id, reason, allow_partial=True, c=c)
        c.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        c.execute(
            "INSERT INTO audit_log(user_id, actor, action, detail, created_at) VALUES (?, 'stripe', 'suspended', ?, ?)",
            (user_id, reason, db.now()),
        )
    return taken


def _note_unrecovered(action: str, payment, ref_id: str, taken_micros: int) -> None:
    """The credit of a disputed payment that was already spent can't be taken back: record the shortfall."""
    owed = (payment["amount_cents"] - payment["refunded_cents"]) * MICROS_PER_CENT
    if taken_micros < owed:
        _audit_once(action, payment["user_id"], f"{payment['id']}:{ref_id}:{owed - taken_micros} micros")


def _handle_dispute_created(dispute: dict) -> None:
    charge_id = dispute.get("charge")
    row = db.one("SELECT id, user_id, amount_cents, refunded_cents FROM payments WHERE stripe_charge = ?", (charge_id,))
    if row is None:
        log.error("dispute %s for unknown charge %s", dispute.get("id"), charge_id)
        return
    db.execute("UPDATE payments SET status = 'disputed', updated_at = ? WHERE id = ?", (db.now(), row["id"]))
    taken = suspend_and_zero(row["user_id"], f"dispute {dispute.get('reason')}", dispute["id"])
    _note_unrecovered("dispute_unrecovered", row, dispute["id"], taken)


def _handle_early_fraud_warning(efw: dict) -> None:
    charge_id = efw.get("charge")
    row = db.one("SELECT id, user_id, amount_cents, refunded_cents FROM payments WHERE stripe_charge = ?", (charge_id,))
    if row is None:
        return
    taken = suspend_and_zero(row["user_id"], "early fraud warning", efw["id"])
    _note_unrecovered("efw_unrecovered", row, efw["id"], taken)
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
    "payment_intent.amount_capturable_updated": _handle_pi_capturable,
    "payment_intent.succeeded": _handle_pi_succeeded,
    "payment_intent.payment_failed": _handle_pi_failed,
    "charge.refunded": _handle_charge_refunded,
    "charge.dispute.created": _handle_dispute_created,
    "radar.early_fraud_warning.created": _handle_early_fraud_warning,
    "payment_method.detached": _handle_pm_detached,
}


_webhook_lock = threading.Lock()


def handle_webhook(payload: bytes, sig_header: str | None) -> str:
    secret = get_settings().stripe_webhook_secret
    if not secret:
        raise BillingUnavailable("webhook secret not configured")
    event = client().construct_event(payload, sig_header, secret)  # raises on bad signature
    handler = HANDLERS.get(event.type)
    with _webhook_lock:  # one delivery at a time, so a duplicate arriving together with the original waits and sees it done
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
        " AND auto_reload_failures < ? AND balance_micros - held_micros < auto_reload_threshold_micros"
        " AND auto_reload_threshold_micros < auto_reload_amount_cents * ?",
        (AUTO_RELOAD_MAX_FAILURES, MICROS_PER_CENT),
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
            # a reload that doesn't lift the balance over the threshold would charge the card again and again
            or u["auto_reload_threshold_micros"] >= u["auto_reload_amount_cents"] * MICROS_PER_CENT
        ):
            return "skip"
        spent = u["auto_reload_month_cents"] if u["auto_reload_month"] == month else 0
        amount = int(u["auto_reload_amount_cents"])
        if amount < TOPUP_MIN_CENTS or spent + amount > u["auto_reload_monthly_cap_cents"] or amount > trust.topup_max_cents(user_id):
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
                "capture_method": "manual",  # charged by _settle_authorized after the country check
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
    except Exception:  # noqa: BLE001 - network etc.: the PaymentIntent may exist (the request may even have
        # gone through), so the reload stays in flight until reconcile() finds out what happened to it
        log.exception("auto-reload request for %s failed", user_id)
        return "error"
    db.execute("UPDATE payments SET stripe_payment_intent = ?, updated_at = ? WHERE id = ?", (pi.id, db.now(), payment_id))
    try:
        pi = _settle_authorized(pi, payment_id, user_id)
    except Exception:  # noqa: BLE001 - the amount_capturable_updated webhook retries it
        log.exception("charging auto-reload %s failed; left for the webhook", payment_id)
        return "error"
    if pi is None:
        return "refused_region"
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
    # Given credit (grants) counts as spent last: a refund never pays it out as money.
    bought = max(0, bal["available_micros"] - ledger.granted_micros(user_id))
    refundable = min(p["amount_cents"] - p["refunded_cents"], bought // MICROS_PER_CENT)
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
        _reverse_tax(p, refund.id, gross)
        _book_refund(
            user_id, payment_id, hold_id, refund.id, refundable, p["refunded_cents"],
            f"Refund ${to_card / 100:.2f}" + (f" (card fee ${fee / 100:.2f} kept)" if fee else ""),
        )
    finally:
        ledger.release_hold(hold_id)  # a no-op once booked; frees the money if anything above failed
    db.audit("user", "refund", user_id=user_id, detail=f"{payment_id}:{refundable}:fee={fee}")
    return to_card


def _book_refund(
    user_id: str, payment_id: str, hold_id: str, refund_id: str, refundable_cents: int, previous_refunded: int, description: str
) -> None:
    """Take `refundable_cents` of credit for a refund Stripe has made, once, in one transaction with
    closing the hold and counting it on the payment. The charge.refunded webhook may have booked the
    part that went to the card already (and a double click comes through here twice): only what is
    still missing is debited, so the fee part is never lost and nothing is taken twice."""
    want = refundable_cents * MICROS_PER_CENT
    with db.tx() as c:
        done = c.execute(
            "SELECT COUNT(*) AS n, COALESCE(-SUM(amount_micros), 0) AS micros FROM ledger"
            " WHERE kind = 'refund' AND ref_type IN ('stripe_refund', 'stripe_refund_fee') AND ref_id = ?",
            (refund_id,),
        ).fetchone()
        ledger.release_hold(hold_id, c=c)
        if want > done["micros"]:
            ledger.debit_available(
                user_id, want - done["micros"], "refund", "stripe_refund_fee" if done["n"] else "stripe_refund", refund_id,
                description, allow_partial=True, c=c,
            )
        c.execute(
            "UPDATE payments SET refunded_cents = MAX(refunded_cents, ?), updated_at = ? WHERE id = ?",
            (previous_refunded + refundable_cents, db.now(), payment_id),
        )


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


# ------------------------------------------------------------------ reconciliation

# Webhooks are the normal way a payment reaches us; when one is lost (or keeps failing) these sweep up.
PENDING_RECHECK_SECONDS = 10 * 60
RELOAD_IN_FLIGHT_SECONDS = 30 * 60
UNSETTLED_CANCEL_SECONDS = 24 * 3600


def _to_dict(obj) -> dict:
    return obj if isinstance(obj, dict) else obj.to_dict()


def _fail_payment(payment_id: str, code: str) -> None:
    db.execute(
        "UPDATE payments SET status = 'failed', failure_code = ?, updated_at = ? WHERE id = ? AND status = 'pending'",
        (code, db.now(), payment_id),
    )


def _reload_payment_intent(p) -> dict | None:
    """The PaymentIntent of an auto-reload (its id may not have been saved if our request failed midway)."""
    if p["stripe_payment_intent"]:
        return _to_dict(client().v1.payment_intents.retrieve(p["stripe_payment_intent"]))
    found = client().v1.payment_intents.search(params={"query": f"metadata['acw_payment_id']:'{p['id']}'"}).data
    if not found:
        return None
    pi = _to_dict(found[0])
    db.execute(
        "UPDATE payments SET stripe_payment_intent = ? WHERE id = ? AND stripe_payment_intent IS NULL", (pi["id"], p["id"])
    )
    return pi


def _reconcile_reload(p, age: int) -> None:
    pi = _reload_payment_intent(p)
    if pi is None:
        if age >= RELOAD_IN_FLIGHT_SECONDS:  # the request never reached Stripe
            _record_reload_failure(p["id"], p["user_id"], "no_payment_intent")
        return
    if pi.get("status") == "requires_capture":
        _handle_pi_capturable(pi)
        if db.one("SELECT status FROM payments WHERE id = ?", (p["id"],))["status"] != "pending":
            return  # refused for its card's country
        pi = _to_dict(client().v1.payment_intents.retrieve(pi["id"]))
    status = pi.get("status")
    if status == "succeeded":
        _handle_pi_succeeded(pi)
    elif status in ("canceled", "requires_payment_method"):
        err = pi.get("last_payment_error") or {}
        _record_reload_failure(p["id"], p["user_id"], err.get("decline_code") or err.get("code") or status)
    elif status != "processing" and age >= RELOAD_IN_FLIGHT_SECONDS:
        client().v1.payment_intents.cancel(
            pi["id"], params={"cancellation_reason": "abandoned"}, options={"idempotency_key": f"stale-reload-cancel-{p['id']}"}
        )
        _record_reload_failure(p["id"], p["user_id"], "timeout")


def _reconcile_checkout(p, age: int) -> None:
    if not p["stripe_checkout_session"]:
        if age >= 3600:  # the Checkout session was never created
            _fail_payment(p["id"], "no_session")
        return
    session = _to_dict(client().v1.checkout.sessions.retrieve(p["stripe_checkout_session"]))
    if session.get("status") == "expired":
        _fail_payment(p["id"], "expired")
    elif session.get("status") == "complete":
        _handle_checkout_completed(session)  # the same handler as the webhook; safe to repeat


def _give_up_unsettled(p) -> None:
    """A payment that could not be settled for a day: cancel the card authorization so the customer's
    money is released now instead of when it lapses. A payment that was charged but not credited is left
    alone and recorded for the operator."""
    pi_id = p["stripe_payment_intent"]
    if not pi_id and p["stripe_checkout_session"]:
        pi_id = _id_of(_get(client().v1.checkout.sessions.retrieve(p["stripe_checkout_session"]), "payment_intent"))
    if pi_id:
        pi = client().v1.payment_intents.retrieve(pi_id)
        if _get(pi, "status") == "succeeded":
            _audit_once("payment_uncredited", p["user_id"], f"{p['id']}:{pi_id}")
            return
        if _get(pi, "status") == "requires_capture":
            client().v1.payment_intents.cancel(
                pi_id, params={"cancellation_reason": "abandoned"}, options={"idempotency_key": f"unsettled-cancel-{p['id']}"}
            )
    if p["kind"] == "auto_reload":
        _record_reload_failure(p["id"], p["user_id"], "unsettled")
    else:
        _fail_payment(p["id"], "unsettled")
    log.error("payment %s could not be settled in a day; authorization cancelled", p["id"])


def reconcile() -> int:
    """Look at payments still pending after PENDING_RECHECK_SECONDS in Stripe and run the webhook handlers
    on what is there; give up on authorizations that can't be settled after a day; expire auto-reload
    flags nothing is waiting for. Returns how many payments it looked at."""
    now = db.now()
    rows = db.all_rows(
        "SELECT * FROM payments WHERE status = 'pending' AND updated_at < ?", (now - PENDING_RECHECK_SECONDS,)
    )
    for row in rows:
        p = dict(row)
        age = now - p["created_at"]
        # The next look is in PENDING_RECHECK_SECONDS, not on the next tick.
        db.execute("UPDATE payments SET updated_at = ? WHERE id = ? AND status = 'pending'", (now, p["id"]))
        try:
            (_reconcile_reload if p["kind"] == "auto_reload" else _reconcile_checkout)(p, age)
        except Exception:  # noqa: BLE001
            log.exception("reconciling payment %s failed", p["id"])
        try:
            if age >= UNSETTLED_CANCEL_SECONDS and db.one("SELECT status FROM payments WHERE id = ?", (p["id"],))["status"] == "pending":
                _give_up_unsettled(p)
        except Exception:  # noqa: BLE001
            log.exception("cancelling unsettled payment %s failed", p["id"])
    db.execute(
        "UPDATE users SET auto_reload_in_flight = 0 WHERE auto_reload_in_flight = 1 AND NOT EXISTS ("
        " SELECT 1 FROM payments p WHERE p.user_id = users.id AND p.kind = 'auto_reload' AND p.status = 'pending')"
    )
    return len(rows)
