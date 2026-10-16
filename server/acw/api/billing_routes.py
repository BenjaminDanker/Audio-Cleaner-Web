from __future__ import annotations

import logging

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from .. import auth, db
from ..billing import ledger, pricing, stripe_service, trust
from ..billing.pricing import MICROS_PER_CENT
from ..config import get_settings
from ..security import client_ip, client_net, enforce, run_outbound

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/billing", tags=["billing"])


def _require_tos(principal: auth.Principal) -> None:
    """Signing in means agreeing; using the service after a Terms update does too (recorded)."""
    auth.accept_terms(principal.user_id, "continued use")


def _require_stripe() -> None:
    if not stripe_service.enabled():
        raise HTTPException(status_code=503, detail="Payments are not available yet.")


@router.get("/pricing")
def get_pricing() -> dict:
    return {
        "rates": pricing.public_price_list(),
        "topup": {
            "minCents": pricing.TOPUP_MIN_CENTS,
            "maxCents": pricing.TOPUP_MAX_CENTS,
            "presetsCents": pricing.TOPUP_PRESETS_CENTS,
        },
        "maxJobSeconds": pricing.MAX_JOB_SECONDS,
        "maxUploadBytes": get_settings().max_upload_bytes,
        "maxRebuildSeconds": pricing.MAX_REBUILD_SECONDS,
        "paymentsEnabled": stripe_service.enabled(),
        "taxAddedAtCheckout": get_settings().stripe_automatic_tax,
    }


class CheckoutBody(BaseModel):
    amountCents: int = Field(ge=pricing.TOPUP_MIN_CENTS, le=pricing.TOPUP_MAX_CENTS)
    saveCard: bool = False


@router.post("/checkout")
async def checkout(body: CheckoutBody, request: Request, principal: auth.Principal = Depends(auth.require_session)) -> dict:
    _require_stripe()
    _require_tos(principal)
    enforce("checkout", principal.user_id)
    enforce("checkout", f"ip:{client_net(client_ip(request))}")
    limit = await anyio.to_thread.run_sync(trust.topup_max_cents, principal.user_id)
    if body.amountCents > limit:
        raise HTTPException(status_code=400, detail=trust.topup_limit_message(limit))
    url = await run_outbound(stripe_service.create_checkout, principal.user_id, body.amountCents, body.saveCard)
    db.audit("user", "checkout_started", user_id=principal.user_id, detail=str(body.amountCents), ip=client_ip(request))
    return {"url": url}


@router.post("/portal")
async def portal(principal: auth.Principal = Depends(auth.require_session)) -> dict:
    _require_stripe()
    enforce("checkout", principal.user_id)
    return {"url": await run_outbound(stripe_service.billing_portal_url, principal.user_id)}


class AutoReloadBody(BaseModel):
    enabled: bool
    thresholdCents: int = Field(ge=0, le=pricing.TOPUP_MAX_CENTS)
    amountCents: int = Field(ge=pricing.TOPUP_MIN_CENTS, le=pricing.TOPUP_MAX_CENTS)
    monthlyCapCents: int = Field(ge=pricing.TOPUP_MIN_CENTS, le=200_000)
    consent: bool = False


@router.post("/auto-reload")
def set_auto_reload(body: AutoReloadBody, request: Request, principal: auth.Principal = Depends(auth.require_session)) -> dict:
    _require_tos(principal)
    user = db.one("SELECT default_payment_method FROM users WHERE id = ?", (principal.user_id,))
    if body.enabled:
        limit = trust.topup_max_cents(principal.user_id)
        if body.amountCents > limit:
            raise HTTPException(status_code=400, detail=trust.topup_limit_message(limit))
        if not body.consent:
            raise HTTPException(status_code=400, detail="Auto-reload needs your explicit authorization.")
        if not user["default_payment_method"]:
            raise HTTPException(status_code=400, detail="Save a card during a top-up first.")
        if body.thresholdCents >= body.amountCents:
            raise HTTPException(status_code=400, detail="The balance that triggers a reload must be lower than the reload amount.")
        if body.monthlyCapCents < body.amountCents:
            raise HTTPException(status_code=400, detail="Monthly cap must be at least one reload.")
    db.execute(
        "UPDATE users SET auto_reload_enabled = ?, auto_reload_threshold_micros = ?, auto_reload_amount_cents = ?,"
        " auto_reload_monthly_cap_cents = ?, auto_reload_failures = 0,"
        " auto_reload_consent_at = CASE WHEN ? THEN ? ELSE auto_reload_consent_at END WHERE id = ?",
        (
            1 if body.enabled else 0,
            body.thresholdCents * MICROS_PER_CENT,
            body.amountCents,
            body.monthlyCapCents,
            1 if body.enabled else 0,
            db.now(),
            principal.user_id,
        ),
    )
    # The consent record is dispute evidence: what was authorized, when, from where.
    db.audit(
        "user",
        "auto_reload_" + ("enabled" if body.enabled else "disabled"),
        user_id=principal.user_id,
        detail=f"threshold={body.thresholdCents} amount={body.amountCents} cap={body.monthlyCapCents}",
        ip=client_ip(request),
    )
    return {"ok": True}


@router.get("/history")
def history(principal: auth.Principal = Depends(auth.require_session)) -> dict:
    payments = db.all_rows(
        "SELECT id, kind, status, amount_cents, refunded_cents, created_at FROM payments"
        " WHERE user_id = ? AND status != 'pending' ORDER BY created_at DESC LIMIT 100",
        (principal.user_id,),
    )
    entries, more = ledger.history_page(principal.user_id)
    return {
        "balance": ledger.balance(principal.user_id),
        "ledger": entries,
        "ledgerHasMore": more,
        "payments": [dict(p) for p in payments],
    }


class RefundBody(BaseModel):
    paymentId: str = Field(max_length=64)


@router.post("/refund")
async def refund(body: RefundBody, request: Request, principal: auth.Principal = Depends(auth.require_session)) -> dict:
    _require_stripe()
    enforce("checkout", principal.user_id)
    try:
        cents = await run_outbound(stripe_service.refund_unused, principal.user_id, body.paymentId)
    except (ValueError, ledger.InsufficientFunds) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"refundedCents": cents}


webhook_router = APIRouter(tags=["billing"])


@webhook_router.post("/api/stripe/webhook")
async def stripe_webhook(request: Request) -> dict:
    enforce("webhook", client_net(client_ip(request)))
    payload = await request.body()  # BodyLimitMiddleware has already capped it at 1 MB while reading
    if len(payload) > 1_000_000:
        raise HTTPException(status_code=413, detail="too large")
    try:
        result = await anyio.to_thread.run_sync(
            stripe_service.handle_webhook, payload, request.headers.get("stripe-signature")
        )
    except stripe_service.BillingUnavailable:
        raise HTTPException(status_code=503, detail="not configured")
    except Exception as e:  # signature failures and handler errors -> Stripe retries
        if type(e).__name__ == "SignatureVerificationError" or isinstance(e, ValueError):
            raise HTTPException(status_code=400, detail="bad signature")
        log.exception("stripe webhook handler failed")
        raise HTTPException(status_code=500, detail="handler error")
    return {"received": result}
