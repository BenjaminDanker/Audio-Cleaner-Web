"""Email alerts for the operator, with de-duplication and "resolved" mails.

    ACW_ALERT_EMAIL=you@example.com     who hears about problems (comma separated)
    SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASSWORD / SMTP_FROM   the mail account (STARTTLS on 587)

Until both are set nothing is sent: every alert is logged at WARNING/ERROR so
it shows up in `journalctl --user -u acw-web -u acw-monitor`, and it is sent
as soon as mail is configured (the alert stays "pending" in the meantime).

De-duplication (table `alert_state`, one row per alert *kind*):
  * the first time a kind fires it is mailed;
  * while it stays active it is mailed again after 1 hour, then every 6 hours;
  * when the condition clears, one "resolved" mail goes out (only if the alert
    itself was mailed) and the kind is re-armed, so a new occurrence mails at once.
`event=True` alerts (something happened, nothing to resolve) are simply limited
to one mail per kind per hour.

Mail bodies carry counts and ids only: no user emails, file names, transcripts
or secrets (everything is run through `scrub` as a last line of defence).

CLI:  python -m acw.alerts test | status | unit-failed UNIT
"""
from __future__ import annotations

import logging
import os
import re
import smtplib
import socket
import ssl
import subprocess
import sys
import threading
import time
from collections import deque
from email.message import EmailMessage

from . import db
from .config import get_settings

log = logging.getLogger("acw.alerts")

SUBJECT_PREFIX = "[Audio Cleaner]"
FIRST_REPEAT_S = 3600
LATER_REPEAT_S = 6 * 3600

_SECRET_RES = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\b(?:whsec|rk|pk)_(?:live|test)_[A-Za-z0-9]+"),
    re.compile(r"(?i)(password|secret|token|api[_-]?key)\s*[=:]\s*\S+"),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),  # email addresses (users')
]


def scrub(text: str) -> str:
    for rx in _SECRET_RES:
        text = rx.sub("[redacted]", text)
    return text


def recipients() -> list[str]:
    raw = os.environ.get("ACW_ALERT_EMAIL", "")
    return [a for a in re.split(r"[,\s;]+", raw) if "@" in a]


def configured() -> bool:
    s = get_settings()
    return bool(recipients() and s.smtp_host and (s.smtp_from or s.smtp_user))


def _smtp_send(subject: str, body: str) -> None:
    """Deliver one mail through the configured SMTP account. Raises on failure."""
    s = get_settings()
    msg = EmailMessage()
    msg["Subject"] = f"{SUBJECT_PREFIX} {subject}".replace("\n", " ")[:200]
    msg["From"] = s.smtp_from or s.smtp_user
    msg["To"] = ", ".join(recipients())
    msg.set_content(body)
    ctx = ssl.create_default_context()
    if s.smtp_port == 465:
        smtp = smtplib.SMTP_SSL(s.smtp_host, s.smtp_port, timeout=20, context=ctx)
    else:
        smtp = smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=20)
    with smtp:
        if s.smtp_port != 465:
            smtp.starttls(context=ctx)
        if s.smtp_user:
            smtp.login(s.smtp_user, s.smtp_password or "")
        smtp.send_message(msg)


# Indirection so tests can capture mail without a server.
_deliver = _smtp_send


def _footer(kind: str) -> str:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    return f"\n--\nhost {socket.gethostname()} · {stamp} · alert kind: {kind}\nRunbook: docs/operations.md#alerts\n"


def _try_send(kind: str, subject: str, body: str) -> bool:
    """Never raises. False when mail isn't configured or the send failed."""
    subject, body = scrub(subject), scrub(body)
    if not configured():
        log.error("ALERT NOT SENT (email not configured: set ACW_ALERT_EMAIL and SMTP_*): %s | %s", subject, body.replace("\n", " | "))
        return False
    try:
        _deliver(subject, body + _footer(kind))
        log.warning("alert mailed: %s", subject)
        return True
    except Exception as e:  # noqa: BLE001
        log.error("ALERT MAIL FAILED (%s): %s | %s", type(e).__name__, subject, body.replace("\n", " | "))
        return False


def send_alert(kind: str, subject: str, body: str, *, event: bool = False) -> bool:
    """Mail an alert unless the same `kind` was mailed recently. True if a mail was sent."""
    now = db.now()
    with db.tx() as c:
        row = c.execute("SELECT * FROM alert_state WHERE kind = ?", (kind,)).fetchone()
        was_active = bool(row and row["active"]) and not event
        orig_sent = row["last_sent_at"] if row else 0
        orig_count = row["send_count"] if row else 0
        if event:
            suppress = bool(row) and now - orig_sent < FIRST_REPEAT_S
            count = orig_count
        elif was_active:
            interval = FIRST_REPEAT_S if orig_count < 2 else LATER_REPEAT_S
            suppress = orig_sent > 0 and now - orig_sent < interval
            count = orig_count
        else:  # a new episode: re-armed
            suppress, orig_sent, count = False, 0, 0
        if suppress:
            c.execute("UPDATE alert_state SET last_seen_at = ?, suppressed = suppressed + 1, subject = ? WHERE kind = ?",
                      (now, subject[:200], kind))
            return False
        c.execute(
            "INSERT INTO alert_state(kind, active, first_at, last_sent_at, last_seen_at, send_count, subject)"
            " VALUES (?,?,?,?,?,?,?)"
            " ON CONFLICT(kind) DO UPDATE SET active = excluded.active,"
            " first_at = CASE WHEN ? THEN first_at ELSE excluded.first_at END,"
            " last_sent_at = excluded.last_sent_at, last_seen_at = excluded.last_seen_at,"
            " send_count = excluded.send_count, subject = excluded.subject, clear_streak = 0",
            (kind, 0 if event else 1, now, now, now, count + 1, subject[:200], 1 if was_active else 0),
        )
    ok = _try_send(kind, subject, body)
    if not ok:  # keep it pending so the next call retries
        db.execute("UPDATE alert_state SET last_sent_at = ?, send_count = ? WHERE kind = ?", (orig_sent, orig_count, kind))
    return ok


def resolve_alert(kind: str, subject: str | None = None, body: str | None = None) -> bool:
    """Mark `kind` cleared; mail "resolved" if the alert itself had been mailed."""
    with db.tx() as c:
        row = c.execute("SELECT * FROM alert_state WHERE kind = ?", (kind,)).fetchone()
        if row is None or not row["active"]:
            return False
        c.execute("UPDATE alert_state SET active = 0, streak = 0, clear_streak = 0 WHERE kind = ?", (kind,))
        notify = row["last_sent_at"] > 0
        started = row["first_at"]
        was = row["subject"] or kind
    if not notify:
        return False
    lasted = f"It was active for about {max(1, (db.now() - started) // 60)} minutes." if started else ""
    _try_send(
        kind,
        subject or f"Resolved: {was}",
        (body + "\n" if body else "") + f"The condition has cleared. {lasted}\n",
    )
    return True


def observe(kind: str, bad: bool, subject: str, body: str, *, confirm: int = 1, clear: int = 2) -> None:
    """Feed one check result: alert after `confirm` consecutive bad runs, resolve after `clear` good ones."""
    with db.tx() as c:
        row = c.execute("SELECT streak, clear_streak, active FROM alert_state WHERE kind = ?", (kind,)).fetchone()
        if row is None and not bad:
            return
        streak = (row["streak"] if row else 0) + 1 if bad else 0
        clear_streak = 0 if bad else (row["clear_streak"] if row else 0) + 1
        c.execute(
            "INSERT INTO alert_state(kind, streak, clear_streak, subject) VALUES (?,?,?,?)"
            " ON CONFLICT(kind) DO UPDATE SET streak = excluded.streak, clear_streak = excluded.clear_streak",
            (kind, streak, clear_streak, subject[:200]),
        )
        active = bool(row and row["active"])
    if bad:
        if streak >= confirm:
            send_alert(kind, subject, body)
    elif active and clear_streak >= clear:
        resolve_alert(kind, f"Resolved: {subject}", None)


def expire_stale(prefix: str, max_age_s: int) -> None:
    """Resolve active kinds starting with `prefix` that haven't been reported for `max_age_s`."""
    cutoff = db.now() - max_age_s
    rows = db.all_rows(
        "SELECT kind, subject FROM alert_state WHERE active = 1 AND kind LIKE ? AND COALESCE(last_seen_at, 0) < ?",
        (prefix + "%", cutoff),
    )
    for r in rows:
        resolve_alert(r["kind"], f"Resolved: {r['subject']}", f"No new reports for {max_age_s // 60} minutes.")


def send_alert_bg(kind: str, subject: str, body: str, *, event: bool = False) -> None:
    """Fire-and-forget from request/stream paths: never blocks and never raises."""
    if not configured():
        log.error("ALERT NOT SENT (email not configured: set ACW_ALERT_EMAIL and SMTP_*): %s | %s", scrub(subject), scrub(body).replace("\n", " | "))
        return

    def run() -> None:
        try:
            send_alert(kind, subject, body, event=event)
        except Exception:  # noqa: BLE001
            log.exception("alert failed: %s", subject)

    threading.Thread(target=run, name="alert", daemon=True).start()


# --- OpenAI problems -------------------------------------------------------

_BILLING_CODES = {"insufficient_quota", "billing_hard_limit_reached", "billing_not_active", "access_terminated",
                  "account_deactivated", "billing_limit_reached"}
_SAFE = re.compile(r"[^A-Za-z0-9_.-]")
_5xx: deque[float] = deque(maxlen=20)
_5XX_COUNT, _5XX_WINDOW_S = 3, 300


def _clean(v: object) -> str:
    return _SAFE.sub("", str(v or ""))[:60]


def _error_fields(body: object) -> tuple[str, str]:
    """(type, code) from an OpenAI error body (bytes/str/dict); never the message text."""
    import json

    try:
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8", "replace")
        if isinstance(body, str):
            body = json.loads(body)
        err = body.get("error", body) if isinstance(body, dict) else {}
        if isinstance(err, dict):
            return _clean(err.get("type")), _clean(err.get("code"))
    except Exception:  # noqa: BLE001
        pass
    return "", ""


def report_openai(source: str, status: int | None, body: object = None, *, err_type: str = "", err_code: str = "") -> None:
    """Call when OpenAI answered with an error. Decides if it's worth an email
    (auth, quota/billing, repeated 5xx) and sends it in the background.
    Only the HTTP status and OpenAI's error type/code are ever included."""
    try:
        t, code = _error_fields(body) if body is not None else ("", "")
        t, code = t or _clean(err_type), code or _clean(err_code)
        quota = code in _BILLING_CODES or t in _BILLING_CODES or "billing" in code or "quota" in code
        detail = f"HTTP status: {status if status is not None else 'n/a'}\nError type: {t or 'n/a'}\nError code: {code or 'n/a'}\nWhere: {source}\n"
        if quota:
            send_alert_bg("openai:quota", "OpenAI quota or billing problem",
                          "OpenAI is refusing requests because of the account's quota or billing.\n"
                          "Premium transcription and premium captions will fail until this is fixed "
                          "(check platform.openai.com billing and usage limits).\n\n" + detail)
        elif status in (401, 403):
            send_alert_bg("openai:auth", "OpenAI rejected our API key",
                          "OpenAI answered 401/403: the API key is invalid, revoked, or lacks permission.\n"
                          "Premium transcription and premium captions will fail until OPENAI_API_KEY is fixed.\n\n" + detail)
        elif status is not None and status >= 500:
            now = time.monotonic()
            _5xx.append(now)
            if sum(1 for t0 in _5xx if now - t0 <= _5XX_WINDOW_S) >= _5XX_COUNT:
                send_alert_bg("openai:5xx", "OpenAI is returning server errors",
                              f"{_5XX_COUNT}+ OpenAI server errors in the last {_5XX_WINDOW_S // 60} minutes. "
                              "This is usually OpenAI's outage (status.openai.com); premium features may fail.\n\n" + detail)
    except Exception:  # noqa: BLE001
        log.exception("report_openai failed")


# --- web process start / stop ---------------------------------------------

def web_started() -> bool:
    """Called at acw-web start-up. Mails if the previous run never recorded a clean stop.
    Returns True when the previous run ended uncleanly."""
    prev = db.one("SELECT value FROM meta WHERE key = 'web.run_state'")
    unclean = bool(prev and prev["value"].startswith("running"))
    since = prev["value"].split(":", 1)[1] if unclean and ":" in prev["value"] else ""
    db.execute(
        "INSERT INTO meta(key, value) VALUES('web.run_state', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (f"running:{db.now()}",),
    )
    if not configured():
        log.warning("alerts are NOT configured (ACW_ALERT_EMAIL / SMTP_*): problems will only appear in this journal")
    if unclean:
        when = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(int(since))) if since.isdigit() else "unknown"
        log.error("acw-web is starting after an unclean stop (previous run started %s)", when)
        send_alert_bg(
            "web_unclean_stop", "acw-web is back up after an unclean stop",
            f"acw-web has just started and the previous run did not shut down cleanly (it crashed, was killed, "
            f"ran out of memory, or the machine lost power). The previous run started {when}.\n"
            "Check: journalctl --user -u acw-web --since '-1h'  and  dmesg | grep -i oom\n",
            event=True,
        )
    return unclean


def web_stopping() -> None:
    db.execute(
        "INSERT INTO meta(key, value) VALUES('web.run_state', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (f"stopped:{db.now()}",),
    )


# --- systemd OnFailure hook ------------------------------------------------

_UNIT_RE = re.compile(r"^acw-[A-Za-z0-9@._-]{1,60}$")


def unit_failed(unit: str) -> bool:
    if not _UNIT_RE.match(unit):
        log.error("unit-failed: refusing odd unit name")
        return False
    props = ""
    try:
        props = subprocess.run(
            ["systemctl", "--user", "show", unit, "-p", "ActiveState", "-p", "SubState", "-p", "Result",
             "-p", "ExecMainStatus", "-p", "NRestarts", "-p", "ActiveEnterTimestamp"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return send_alert(
        f"service:{unit}", f"{unit} failed",
        f"systemd reports that {unit} entered the failed state (it crashed or kept failing to start).\n\n{props}\n\n"
        f"Check: journalctl --user -u {unit} -n 80 --no-pager\n"
        "You will get a second mail when the service is healthy again.\n",
    )


def _status() -> None:
    rows = db.all_rows("SELECT * FROM alert_state ORDER BY active DESC, kind")
    print(f"alerts configured: {configured()}  recipients: {', '.join(recipients()) or '-'}")
    for r in rows:
        print(f"{'ACTIVE ' if r['active'] else 'ok     '} {r['kind']:<40} mails={r['send_count']} suppressed={r['suppressed']} {r['subject'] or ''}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    argv = sys.argv[1:] if argv is None else argv
    db.init_db()
    if argv[:1] == ["unit-failed"] and len(argv) == 2:
        unit_failed(argv[1])
        return 0
    if argv[:1] == ["test"]:
        ok = _try_send("test", "Test alert", "If you can read this, alert emails work.\n")
        print("sent" if ok else "NOT sent (see log above)")
        return 0 if ok else 1
    if argv[:1] == ["status"]:
        _status()
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
