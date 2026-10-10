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
from ..config import get_settings
from . import ledger
from .pricing import MICROS_PER_CENT, TOPUP_MAX_CENTS, TOPUP_MIN_CENTS

log = logging.getLogger(__name__)

AUTO_RELOAD_MAX_FAILURES = 3
EFW_AUTO_REFUND_MAX_CENTS = 5_000


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
    session = client().v1.checkout.sessions.create(
        params={
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
                        "product_data": {
                            "name": "Audio Cleaner prepaid credit",
                            "description": "Usage credit for audio processing. Credit is not a gift card and has no cash value.",
                        },
                    },
                }
            ],
            "payment_intent_data": pi_data,
            # 3DS on the first, on-session purchase shifts fraud liability in
            # most cases and makes later off-session reloads less likely to fail.
            "payment_method_options": {"card": {"request_three_d_secure": "any"}},
            "metadata": {"acw_user_id": user_id, "acw_payment_id": payment_id},
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

def _credit_payment(payment_id: str, user_id: str, amount_cents: int, pi_id: str | None, charge_id: str | None, pm_id: str | None, save_pm: bool) -> None:
    with db.tx() as c:
        p = c.execute("SELECT * FROM payments WHERE id = ?", (payment_id,)).fetchone()
        if p is None or p["user_id"] != user_id:
            log.error("payment %s does not belong to user %s", payment_id, user_id)
            return
        if p["status"] == "succeeded":
            return
        if amount_cents != p["amount_cents"]:
            log.warning("payment %s amount mismatch: expected %s got %s", payment_id, p["amount_cents"], amount_cents)
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
    _credit_payment(payment_id, user_id, int(session.get("amount_total") or 0), pi_id, charge_id, pm_id, save_pm)


def _handle_pi_succeeded(pi: dict) -> None:
    meta = pi.get("metadata") or {}
    if meta.get("acw_kind") != "auto_reload":
        return  # checkout payments are credited from checkout.session.completed
    charge = pi.get("latest_charge")
    _credit_payment(meta["acw_payment_id"], meta["acw_user_id"], int(pi["amount_received"]), pi["id"], charge, None, False)


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
    for r in refunds:
        r = r if isinstance(r, dict) else r.to_dict()
        if r.get("status") not in ("succeeded", "pending"):
            continue
        ledger.debit_available(
            user_id,
            int(r["amount"]) * MICROS_PER_CENT,
            "refund",
            "stripe_refund",
            r["id"],
            f"Refund ${int(r['amount']) / 100:.2f}",
            allow_partial=True,
        )
    db.execute(
        "UPDATE payments SET refunded_cents = ?, status = CASE WHEN ? >= amount_cents THEN 'refunded' ELSE status END,"
        " updated_at = ? WHERE id = ?",
        (int(charge.get("amount_refunded") or 0), int(charge.get("amount_refunded") or 0), db.now(), payment_id),
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
    try:
        pi = client().v1.payment_intents.create(
            params={
                "amount": amount,
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
        _credit_payment(payment_id, user_id, int(pi.amount_received), pi.id, charge, None, False)
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
    hold_id = ledger.create_hold(user_id, refundable * MICROS_PER_CENT, "refund", payment_id)
    try:
        refund = client().v1.refunds.create(
            params={"charge": p["stripe_charge"], "amount": refundable, "reason": "requested_by_customer",
                    "metadata": {"acw_user_id": user_id, "acw_payment_id": payment_id}},
            options={"idempotency_key": f"refund-{payment_id}-{refundable}-{p['refunded_cents']}"},
        )
    except Exception:
        ledger.release_hold(hold_id)
        raise
    ledger.capture_hold(
        hold_id, refundable * MICROS_PER_CENT, f"Refund ${refundable / 100:.2f}",
        kind="refund", ref_type="stripe_refund", ref_id=refund.id,
    )
    db.execute(
        "UPDATE payments SET refunded_cents = refunded_cents + ?, updated_at = ? WHERE id = ?",
        (refundable, db.now(), payment_id),
    )
    db.audit("user", "refund", user_id=user_id, detail=f"{payment_id}:{refundable}")
    return refundable
