"""The contact form: `POST /api/contact` mails a visitor's message to the owner.

The owner's address (ACW_CONTACT_EMAIL, else the alert address ACW_ALERT_EMAIL) is never sent to a browser, and
there is no way to read it from the API. The message itself is not stored: it goes into the mail and nowhere
else (the audit log only records that a message was sent, with its topic, never its text).

Abuse limits, all of them before any mail is sent:
  * a hidden "website" field that people never see (a filled one is answered with the normal success, silently dropped)
  * per network (IPv6 per /64): 3 per hour and 10 per day; per mailbox: 3 per hour and 5 per day
  * the whole site: CONTACT_DAILY_CAP (100) per rolling day, counted from the audit log so a restart doesn't reset it
  * same-origin requests only, and the usual CSRF token when signed in; bodies are limited to 64 KB elsewhere
  * mail goes out through the outbound thread budget (`run_outbound`)
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from .. import alerts, auth, db
from ..security import client_ip, client_net, enforce, run_outbound
from .auth_routes import _require_same_origin, canonical_email, clean_email

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["contact"])

CONTACT_DAILY_CAP = 100
MESSAGE_MIN = 10
MESSAGE_MAX = 5000
TOPICS = {
    "question": "Question",
    "billing": "Billing/refund",
    "job": "Problem with a job",
    "privacy": "Privacy/data request",
    "other": "Other",
}
_JOB_ID = re.compile(r"^[0-9a-f]{32}$")
# Left-to-right / right-to-left marks and overrides: they can make a subject line or a name read differently than it is.
_BIDI = set("؜‎‏‪‫‬‭‮⁦⁧⁨⁩")


def one_line(text: str, limit: int) -> str:
    """Single line for a header or a name: control, format and line-break characters become spaces; runs of space collapse."""
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        out.append(" " if cat in ("Cc", "Cf", "Zl", "Zp", "Cs", "Co") or ch in _BIDI else ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()[:limit]


def clean_message(text: str) -> str:
    """Body text: line breaks and tabs stay, every other control character and the bidi overrides go."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", "\n").replace(" ", "\n")
    kept = []
    for ch in text:
        if ch in "\n\t":
            kept.append(ch)
        elif unicodedata.category(ch) in ("Cc", "Cs", "Co") or ch in _BIDI:
            continue
        else:
            kept.append(ch)
    return re.sub(r"\n{4,}", "\n\n\n", "".join(kept)).strip()


class ContactBody(BaseModel):
    name: str | None = Field(default=None, max_length=200)
    email: str = Field(max_length=320)
    topic: Literal["question", "billing", "job", "privacy", "other"]
    message: str = Field(max_length=MESSAGE_MAX)
    job: str | None = Field(default=None, max_length=64)
    website: str | None = Field(default=None, max_length=500)  # honeypot


def available() -> bool:
    return alerts.contact_configured()


def _sent_today() -> tuple[int, int]:
    """(messages sent in the last 24 hours, seconds until the cap frees up again if it is reached)."""
    now = db.now()
    since = now - 86400
    n = db.one("SELECT COUNT(*) n FROM audit_log WHERE action = 'contact_sent' AND created_at > ?", (since,))["n"]
    wait = 0
    if n >= CONTACT_DAILY_CAP:
        row = db.one(
            "SELECT created_at FROM audit_log WHERE action = 'contact_sent' AND created_at > ?"
            " ORDER BY id DESC LIMIT 1 OFFSET ?", (since, CONTACT_DAILY_CAP - 1),
        )
        wait = max(60, (row["created_at"] + 86400 - now) if row else 3600)
    return n, wait


def _job_line(job_id: str | None, principal: auth.Principal | None) -> str | None:
    if not job_id:
        return None
    job_id = job_id.strip().lower()
    if not _JOB_ID.match(job_id):
        raise HTTPException(status_code=400, detail="That job number doesn’t look right. It’s the long code from the job’s page.")
    row = db.one("SELECT user_id, kind, status, created_at, error FROM jobs WHERE id = ?", (job_id,))
    if row is None:
        return f"{job_id} (no such job)"
    mine = "" if principal is None else (", this account’s" if row["user_id"] == principal.user_id else ", another account’s")
    err = f", error: {one_line(row['error'], 160)}" if row["error"] else ""
    return f"{job_id} ({row['kind']}, {row['status']}{mine}, created {_utc(row['created_at'])}{err})"


def _utc(ts: int) -> str:
    import time

    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(ts))


def build_mail(body: ContactBody, email: str, message: str, principal: auth.Principal | None, job: str | None) -> tuple[str, str]:
    """(subject, plain-text body). Everything that reaches a header has been through `one_line`."""
    label = TOPICS[body.topic]
    first = one_line(message, 60)
    if len(message) > 60 and " " in first:
        first = first.rsplit(" ", 1)[0] + "…"
    subject = f"{alerts.SUBJECT_PREFIX} {label}: {first}"[:200]
    name = one_line(body.name or "", 100)
    lines = [
        f"From: {name + ' ' if name else ''}<{email}>",
        f"Topic: {label}",
    ]
    if principal is not None:
        lines.append(f"Signed in as: {principal.email} (account {principal.user_id}, {principal.status})")
    else:
        lines.append("Signed in: no")
    if job:
        lines.append(f"Job: {job}")
    lines += ["", message, "", "--", f"Sent from the contact form, {_utc(db.now())}. Reply to this mail to answer the sender.", ""]
    return subject, "\n".join(lines)


@router.post("/contact")
async def contact(body: ContactBody, request: Request, principal: auth.Principal | None = Depends(auth.optional_principal)) -> dict:
    _require_same_origin(request)
    if not available():
        raise HTTPException(status_code=404, detail="The contact form isn’t available right now.")
    email = clean_email(body.email)
    if email is None:
        raise HTTPException(status_code=400, detail="Enter a valid email address so we can reply.")
    message = clean_message(body.message)
    if len(message) < MESSAGE_MIN:
        raise HTTPException(status_code=400, detail=f"Please write a little more (at least {MESSAGE_MIN} characters).")
    if len(message) > MESSAGE_MAX:
        raise HTTPException(status_code=400, detail=f"That’s a bit long. Please keep it under {MESSAGE_MAX} characters.")
    job = _job_line(body.job, principal)
    ip = client_ip(request)
    net = client_net(ip)
    enforce("contact_ip", net)
    enforce("contact_ip_day", net)
    if body.website:  # a human never sees this field
        log.info("contact form: honeypot filled, dropped")
        return {"ok": True}
    canon = canonical_email(email)
    enforce("contact_email", canon)
    enforce("contact_email_day", canon)
    sent, wait = _sent_today()
    if sent >= CONTACT_DAILY_CAP:
        raise HTTPException(
            status_code=429,
            detail="We’ve had a lot of messages today. Please try again tomorrow.",
            headers={"Retry-After": str(wait)},
        )
    subject, text = build_mail(body, email, message, principal, job)
    try:
        await run_outbound(alerts.send_contact_mail, subject, text, email)
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001
        log.exception("sending contact mail failed")
        # 503, not 502: the website reads a 502 as "the whole site is down" and would hide the form with the message in it.
        raise HTTPException(status_code=503, detail="We couldn’t send your message just now. Please try again in a little while.")
    db.audit(
        "user" if principal else "visitor",
        "contact_sent",
        user_id=principal.user_id if principal else None,
        detail=body.topic,
        ip=ip,
    )
    return {"ok": True}

