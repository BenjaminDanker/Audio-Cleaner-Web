"""What an account may do before it has paid anything.

A new account gets the welcome credit for free, so anything that costs real money to run (the extra-accurate
captions and transcripts) and anything that holds shared capacity waits for a first settled payment, and the
first top-ups of a very new account stay small. See docs/security.md.
"""
from __future__ import annotations

from .. import db
from . import pricing

# Until the first payment has gone through, an account younger than this can add at most NEW_ACCOUNT_TOPUP_MAX_CENTS at a time.
NEW_ACCOUNT_SECONDS = 72 * 3600
NEW_ACCOUNT_TOPUP_MAX_CENTS = 5_000
# Live sessions at once: normal accounts and accounts that have not paid yet.
MAX_LIVE_SESSIONS = 2
MAX_LIVE_SESSIONS_UNPAID = 1

PREMIUM_NEEDS_TOPUP = (
    "Extra accurate needs a top-up first: add credit once on the Billing page and it unlocks. "
    "The standard choice works with your welcome credit."
)


def has_settled_payment(user_id: str) -> bool:
    """A top-up that went through and has not been paid back in full."""
    row = db.one(
        "SELECT 1 FROM payments WHERE user_id = ? AND status = 'succeeded' AND refunded_cents < amount_cents LIMIT 1", (user_id,)
    )
    return row is not None


def topup_max_cents(user_id: str) -> int:
    """The most this account may add in one top-up right now."""
    if has_settled_payment(user_id):
        return pricing.TOPUP_MAX_CENTS
    user = db.one("SELECT created_at FROM users WHERE id = ?", (user_id,))
    if user is not None and db.now() - user["created_at"] < NEW_ACCOUNT_SECONDS:
        return NEW_ACCOUNT_TOPUP_MAX_CENTS
    return pricing.TOPUP_MAX_CENTS


def topup_limit_message(max_cents: int) -> str:
    return (
        f"Until your first top-up has gone through (or your account is {NEW_ACCOUNT_SECONDS // 3600 // 24} days old), "
        f"you can add up to ${max_cents // 100} at a time."
    )


def max_live_sessions(user_id: str) -> int:
    return MAX_LIVE_SESSIONS if has_settled_payment(user_id) else MAX_LIVE_SESSIONS_UNPAID
