"""Privacy retention, as the Privacy Policy promises it.

  * network addresses of live sessions and of the security log: removed after 90 days
  * file names of jobs: removed after 12 months
  * payment and ledger records: kept 7 years (tax rules); nothing here touches them

`sweep()` is cheap and idempotent; the web process runs it from its housekeeping loop (at most hourly).
"""
from __future__ import annotations

import time

from . import db

IP_DAYS = 90
FILENAME_DAYS = 365
_last = 0.0


def run(now: int | None = None) -> dict[str, int]:
    """Apply the limits now. Returns how many rows each rule changed."""
    now = db.now() if now is None else now
    ip_cut = now - IP_DAYS * 86400
    name_cut = now - FILENAME_DAYS * 86400
    done = {
        "stream_ips": db.execute(
            "UPDATE stream_sessions SET client_ip = NULL WHERE client_ip IS NOT NULL AND started_at < ?", (ip_cut,)
        ).rowcount,
        "audit_ips": db.execute(
            "UPDATE audit_log SET ip = NULL WHERE ip IS NOT NULL AND created_at < ?", (ip_cut,)
        ).rowcount,
        "session_ips": db.execute(
            "UPDATE sessions SET ip = NULL, user_agent = NULL WHERE (ip IS NOT NULL OR user_agent IS NOT NULL) AND created_at < ?", (ip_cut,)
        ).rowcount,
        "job_names": db.execute(
            "UPDATE jobs SET input_name = '', params_json = json_remove(params_json, '$.display_name')"
            " WHERE input_name != '' AND created_at < ?", (name_cut,)
        ).rowcount,
    }
    return done


def sweep() -> None:
    global _last
    if time.monotonic() - _last < 3600:
        return
    _last = time.monotonic()
    run()
