"""SQLite storage.

One database file, WAL mode, short transactions. Money changes always run in
`BEGIN IMMEDIATE` (see `tx`) so balance checks and ledger writes are serialized
across the web process and the workers.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import get_settings

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    display_name TEXT,
    status TEXT NOT NULL DEFAULT 'active',          -- active | suspended | deleted
    is_admin INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    tos_version TEXT,
    tos_accepted_at INTEGER,
    stripe_customer_id TEXT UNIQUE,
    default_payment_method TEXT,
    balance_micros INTEGER NOT NULL DEFAULT 0,      -- 1 USD = 1_000_000
    held_micros INTEGER NOT NULL DEFAULT 0,
    auto_reload_enabled INTEGER NOT NULL DEFAULT 0,
    auto_reload_threshold_micros INTEGER NOT NULL DEFAULT 0,
    auto_reload_amount_cents INTEGER NOT NULL DEFAULT 0,
    auto_reload_consent_at INTEGER,
    auto_reload_failures INTEGER NOT NULL DEFAULT 0,
    auto_reload_in_flight INTEGER NOT NULL DEFAULT 0,
    auto_reload_month TEXT,
    auto_reload_month_cents INTEGER NOT NULL DEFAULT 0,
    auto_reload_monthly_cap_cents INTEGER NOT NULL DEFAULT 10000,
    CHECK (balance_micros >= 0),
    CHECK (held_micros >= 0),
    CHECK (held_micros <= balance_micros)
);

CREATE TABLE IF NOT EXISTS identities (
    provider TEXT NOT NULL,
    provider_user_id TEXT NOT NULL,
    user_id TEXT NOT NULL REFERENCES users(id),
    email TEXT,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (provider, provider_user_id)
);

CREATE TABLE IF NOT EXISTS sessions (
    id_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    csrf_token TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    last_seen_at INTEGER NOT NULL,
    ip TEXT,
    user_agent TEXT
);
CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id);

CREATE TABLE IF NOT EXISTS login_tokens (
    token_hash TEXT PRIMARY KEY,
    email TEXT NOT NULL,
    purpose TEXT NOT NULL,                          -- email_login | admin_login
    created_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    used_at INTEGER,
    ip TEXT
);

CREATE TABLE IF NOT EXISTS oauth_states (
    state_hash TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    code_verifier TEXT NOT NULL,
    nonce TEXT,
    created_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    name TEXT NOT NULL,
    prefix TEXT NOT NULL,
    key_hash TEXT NOT NULL UNIQUE,
    scopes TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    last_used_at INTEGER,
    revoked_at INTEGER
);
CREATE INDEX IF NOT EXISTS api_keys_user ON api_keys(user_id);

-- Append-only money movements. balance_micros on users is the running sum.
CREATE TABLE IF NOT EXISTS ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL REFERENCES users(id),
    amount_micros INTEGER NOT NULL,
    kind TEXT NOT NULL,                             -- topup | usage | refund | dispute | adjustment | grant
    ref_type TEXT,
    ref_id TEXT,
    description TEXT,
    balance_after_micros INTEGER NOT NULL,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ledger_user ON ledger(user_id, id);
CREATE UNIQUE INDEX IF NOT EXISTS ledger_ref_unique ON ledger(kind, ref_type, ref_id)
    WHERE ref_id IS NOT NULL AND kind IN ('topup', 'refund', 'dispute');

CREATE TABLE IF NOT EXISTS holds (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    amount_micros INTEGER NOT NULL,
    ref_type TEXT NOT NULL,
    ref_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',          -- active | captured | released
    created_at INTEGER NOT NULL,
    closed_at INTEGER
);
CREATE INDEX IF NOT EXISTS holds_user_active ON holds(user_id) WHERE status = 'active';

CREATE TABLE IF NOT EXISTS payments (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    kind TEXT NOT NULL,                             -- checkout | auto_reload
    status TEXT NOT NULL,                           -- pending | succeeded | failed | refunded | disputed
    amount_cents INTEGER NOT NULL,
    currency TEXT NOT NULL DEFAULT 'usd',
    stripe_checkout_session TEXT UNIQUE,
    stripe_payment_intent TEXT UNIQUE,
    stripe_charge TEXT,
    refunded_cents INTEGER NOT NULL DEFAULT 0,          -- credit cents refunded (pre-tax)
    tax_cents INTEGER NOT NULL DEFAULT 0,               -- tax charged on top of amount_cents
    tax_calculation TEXT,
    tax_transaction TEXT,
    failure_code TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS payments_user ON payments(user_id, created_at);

CREATE TABLE IF NOT EXISTS stripe_events (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    received_at INTEGER NOT NULL,
    processed_at INTEGER
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    kind TEXT NOT NULL,                             -- denoise | transcribe | denoise_transcribe
    tier TEXT NOT NULL,
    free INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,                           -- uploading | queued | running | cloud_asr | done | failed | canceled | expired
    params_json TEXT NOT NULL,
    next_stage TEXT NOT NULL DEFAULT 'probe',       -- probe | finalize
    input_name TEXT NOT NULL,
    input_bytes INTEGER NOT NULL,
    received_bytes INTEGER NOT NULL DEFAULT 0,
    claimed_seconds REAL NOT NULL,
    duration_seconds REAL,
    hold_id TEXT,
    est_cost_micros INTEGER NOT NULL DEFAULT 0,
    cost_micros INTEGER,
    progress INTEGER NOT NULL DEFAULT 0,
    stage TEXT,
    error TEXT,
    outputs_json TEXT,
    worker_id TEXT,
    heartbeat_at INTEGER,
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    queued_at INTEGER,
    started_at INTEGER,
    finished_at INTEGER,
    expires_at INTEGER
);
CREATE INDEX IF NOT EXISTS jobs_user ON jobs(user_id, created_at);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, queued_at);
-- the queue order counts a user's running jobs per queued row, and the job list polls it (16.7 ms -> 0.2 ms with 30,000 old jobs)
CREATE INDEX IF NOT EXISTS jobs_user_status ON jobs(user_id, status);

CREATE TABLE IF NOT EXISTS free_usage (
    user_id TEXT NOT NULL REFERENCES users(id),
    day TEXT NOT NULL,
    seconds REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, day)
);

CREATE TABLE IF NOT EXISTS stream_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    api_key_id TEXT,
    denoise_tier TEXT,
    caption_tier TEXT,
    status TEXT NOT NULL,                           -- active | ended
    hold_id TEXT,
    seconds REAL NOT NULL DEFAULT 0,
    cost_micros INTEGER NOT NULL DEFAULT 0,
    client_ip TEXT,
    end_reason TEXT,
    started_at INTEGER NOT NULL,
    ended_at INTEGER
);
CREATE INDEX IF NOT EXISTS stream_sessions_user ON stream_sessions(user_id, started_at);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    detail TEXT,
    ip TEXT,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS audit_user ON audit_log(user_id, id);

-- No longer written (it held a deleted account's free subtitle minutes for a day); emptied by auth.purge_expired.
CREATE TABLE IF NOT EXISTS deleted_emails (
    email_hash TEXT PRIMARY KEY,
    day TEXT NOT NULL,
    free_seconds REAL NOT NULL
);
"""

_local = threading.local()


def now() -> int:
    return int(time.time())


def new_id() -> str:
    return uuid.uuid4().hex


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")  # a payment acknowledged to Stripe survives a power cut
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    # Owner only, whatever the directory's mode (the WAL and shared-memory files appear with the first write).
    for suffix in ("", "-wal", "-shm"):
        try:
            os.chmod(str(path) + suffix, 0o600)
        except OSError:
            pass
    return conn


def conn() -> sqlite3.Connection:
    """Per-thread connection to the configured database."""
    path = get_settings().db_path
    c = getattr(_local, "conn", None)
    if c is None or getattr(_local, "path", None) != path:
        c = _connect(path)
        _local.conn = c
        _local.path = path
    return c


def reset_connection() -> None:
    c = getattr(_local, "conn", None)
    if c is not None:
        c.close()
    _local.conn = None
    _local.path = None


@contextmanager
def tx(immediate: bool = True) -> Iterator[sqlite3.Connection]:
    c = conn()
    c.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
    try:
        yield c
    except BaseException:
        c.execute("ROLLBACK")
        raise
    else:
        c.execute("COMMIT")


# Columns added after the first deployment: (table, column, DDL).
MIGRATIONS = [
    ("payments", "tax_cents", "ALTER TABLE payments ADD COLUMN tax_cents INTEGER NOT NULL DEFAULT 0"),
    ("payments", "tax_calculation", "ALTER TABLE payments ADD COLUMN tax_calculation TEXT"),
    ("payments", "tax_transaction", "ALTER TABLE payments ADD COLUMN tax_transaction TEXT"),
    # Metrics (acw.metrics, acw.stats): how long a job waited for its first worker, and how the live frames fared.
    ("jobs", "wait_seconds", "ALTER TABLE jobs ADD COLUMN wait_seconds INTEGER"),
    ("stream_sessions", "frames", "ALTER TABLE stream_sessions ADD COLUMN frames INTEGER NOT NULL DEFAULT 0"),
    ("stream_sessions", "late_frames", "ALTER TABLE stream_sessions ADD COLUMN late_frames INTEGER NOT NULL DEFAULT 0"),
    ("stream_sessions", "dropped_frames", "ALTER TABLE stream_sessions ADD COLUMN dropped_frames INTEGER NOT NULL DEFAULT 0"),
    # Seconds of audio processed when the extra-accurate captions of a session broke down: billed without them from there.
    ("stream_sessions", "premium_off_seconds", "ALTER TABLE stream_sessions ADD COLUMN premium_off_seconds REAL"),
    # The free live trial: a one-way hash of each mailbox that has had it, so deleting the account and signing up
    # again doesn't give it twice. Kept for good (a hash, no account link).
    (
        "welcome_claims",
        "email_hash",
        "CREATE TABLE IF NOT EXISTS welcome_claims (email_hash TEXT PRIMARY KEY, created_at INTEGER NOT NULL)",
    ),
    # The free live trial: seconds of Standard live an account has left, and how many seconds of each session came from it.
    ("users", "trial_live_seconds", "ALTER TABLE users ADD COLUMN trial_live_seconds REAL NOT NULL DEFAULT 0"),
    ("stream_sessions", "trial_seconds", "ALTER TABLE stream_sessions ADD COLUMN trial_seconds REAL NOT NULL DEFAULT 0"),
    # Email alerts (acw.alerts): one row per alert kind, for de-duplication and "resolved" mails.
    (
        "alert_state",
        "kind",
        "CREATE TABLE IF NOT EXISTS alert_state ("
        " kind TEXT PRIMARY KEY, active INTEGER NOT NULL DEFAULT 0, streak INTEGER NOT NULL DEFAULT 0,"
        " clear_streak INTEGER NOT NULL DEFAULT 0, first_at INTEGER, last_sent_at INTEGER NOT NULL DEFAULT 0,"
        " last_seen_at INTEGER, send_count INTEGER NOT NULL DEFAULT 0, suppressed INTEGER NOT NULL DEFAULT 0,"
        " subject TEXT)",
    ),
]


def init_db() -> None:
    c = conn()
    c.executescript(SCHEMA)
    for table, column, ddl in MIGRATIONS:
        cols = {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            c.execute(ddl)
    c.execute(
        "INSERT INTO meta(key, value) VALUES('schema_version', ?) ON CONFLICT(key) DO NOTHING",
        (str(SCHEMA_VERSION),),
    )


def one(sql: str, params: tuple | dict = ()) -> sqlite3.Row | None:
    return conn().execute(sql, params).fetchone()


def all_rows(sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
    return conn().execute(sql, params).fetchall()


def execute(sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
    return conn().execute(sql, params)


def audit(actor: str, action: str, user_id: str | None = None, detail: str | None = None, ip: str | None = None) -> None:
    execute(
        "INSERT INTO audit_log(user_id, actor, action, detail, ip, created_at) VALUES (?,?,?,?,?,?)",
        (user_id, actor, action, detail, ip, now()),
    )
