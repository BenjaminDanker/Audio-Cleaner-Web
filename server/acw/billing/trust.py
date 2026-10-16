"""What an account may do before and after its first payment.

New accounts get a free live trial (see acw.auth and acw.stream.routes) but no free credit, so everything that
costs real money to run is paid for from the start. What is left to limit is how much one top-up may be: a first
top-up is small, and once a payment has gone through the usual limit applies. See docs/security.md.
"""
from __future__ import annotations

from .. import db
from . import pricing

# Until a first payment has gone through, one top-up can be at most this much.
FIRST_TOPUP_MAX_CENTS = 5_000
# Live sessions at once per account (a free-trial session counts too; there is at most one of those).
MAX_LIVE_SESSIONS = 2


def has_settled_payment(user_id: str) -> bool:
    """A top-up that went through and has not been paid back in full."""
    row = db.one(
        "SELECT 1 FROM payments WHERE user_id = ? AND status = 'succeeded' AND refunded_cents < amount_cents LIMIT 1", (user_id,)
    )
    return row is not None


def topup_max_cents(user_id: str) -> int:
    """The most this account may add in one top-up right now."""
    return pricing.TOPUP_MAX_CENTS if has_settled_payment(user_id) else FIRST_TOPUP_MAX_CENTS


def topup_limit_message(max_cents: int) -> str:
    return (
        f"Your first top-up can be up to ${max_cents // 100}. "
        f"Once a top-up has gone through, you can add up to ${pricing.TOPUP_MAX_CENTS // 100} at a time."
    )
