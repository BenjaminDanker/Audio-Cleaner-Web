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
    status TEXT NOT NULL,                           -- uploading | queued | running | cloud_asr | done | failed | canceled | expired
    params_json TEXT NOT NULL,
    next_stage TEXT NOT NULL DEFAULT 'probe',       -- probe | main (after an automatic job was handed to the card) | finalize
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
    # A live session's billing (stream/meter.py): its paid seconds, and the meter (micros of the stretches before the
    # last change of its features, the rate since, and the paid seconds at that rate). A restart bills from these.
    ("stream_sessions", "paid_seconds", "ALTER TABLE stream_sessions ADD COLUMN paid_seconds REAL NOT NULL DEFAULT 0"),
    ("stream_sessions", "meter_closed", "ALTER TABLE stream_sessions ADD COLUMN meter_closed REAL NOT NULL DEFAULT 0"),
    ("stream_sessions", "meter_rate", "ALTER TABLE stream_sessions ADD COLUMN meter_rate INTEGER"),
    ("stream_sessions", "meter_seconds", "ALTER TABLE stream_sessions ADD COLUMN meter_seconds REAL NOT NULL DEFAULT 0"),
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
    # Public API: which API key created a job (spending caps, usage per key), a monthly spending cap per key
    # (micros; NULL = none), the account's secret for signing job callbacks, and the two small tables below.
    ("jobs", "api_key_id", "ALTER TABLE jobs ADD COLUMN api_key_id TEXT"),
    ("api_keys", "monthly_cap_micros", "ALTER TABLE api_keys ADD COLUMN monthly_cap_micros INTEGER"),
    ("users", "webhook_secret", "ALTER TABLE users ADD COLUMN webhook_secret TEXT"),
    # Idempotency-Key header of POST /api/jobs: the same key (per account, for 24 hours) names the same job.
    # No foreign keys: rows are removed by age (jobs.sweep) and with the account.
    (
        "idempotency_keys",
        "key",
        "CREATE TABLE IF NOT EXISTS idempotency_keys ("
        " user_id TEXT NOT NULL, key TEXT NOT NULL, body_hash TEXT NOT NULL, job_id TEXT NOT NULL,"
        " created_at INTEGER NOT NULL, PRIMARY KEY (user_id, key))",
    ),
    # (no longer written: the re-runs of "change and save again", a job of its own made from another, are gone; a job is changed by editing
    # its timeline. The columns stay so older databases open as they are.)
    ("jobs", "source_job_id", "ALTER TABLE jobs ADD COLUMN source_job_id TEXT"),
    ("jobs", "root_job_id", "ALTER TABLE jobs ADD COLUMN root_job_id TEXT"),
    # The retention clock of a job's files (acw.editing): they go 2 hours after the last edit (`edited_at`: a run finishing, a
    # timeline save, a quote, a process or an export) and at the latest 6 hours after the upload arrived (`uploaded_at`).
    # `expires_at` is the moment that results in.
    ("jobs", "uploaded_at", "ALTER TABLE jobs ADD COLUMN uploaded_at INTEGER"),
    ("jobs", "edited_at", "ALTER TABLE jobs ADD COLUMN edited_at INTEGER"),
    # callbackUrl of a job: delivered by the web process (acw.webhooks) when the job ends. state: pending | delivered | failed.
    (
        "job_callbacks",
        "job_id",
        "CREATE TABLE IF NOT EXISTS job_callbacks ("
        " job_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, url TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending',"
        " attempts INTEGER NOT NULL DEFAULT 0, next_attempt_at INTEGER NOT NULL DEFAULT 0, last_status INTEGER,"
        " last_error TEXT, last_attempt_at INTEGER, delivered_at INTEGER, created_at INTEGER NOT NULL)",
    ),
]

# Columns that are gone: dropped from databases made before (jobs.free: free server-side subtitle minutes;
# stream_sessions.premium_off_seconds: replaced by the meter, see _meter_running_sessions).
DROPPED_COLUMNS = [("jobs", "free"), ("stream_sessions", "premium_off_seconds"), ("stream_sessions", "owner")]


def _meter_running_sessions(c) -> None:
    """Live sessions still marked active from before the meter (an unclean stop of the old version): their meter from
    the old columns, so the restart bills them as before. Runs once: the old column is dropped right after."""
    if "premium_off_seconds" not in {r["name"] for r in c.execute("PRAGMA table_info(stream_sessions)")}:
        return
    from .billing import pricing

    def rate(product: str, tier: str | None) -> int:
        return pricing.rate(product, tier).micros_per_minute if tier else 0

    for r in c.execute("SELECT id, seconds, trial_seconds, denoise_tier, caption_tier, premium_off_seconds FROM stream_sessions"
                       " WHERE status = 'active' AND meter_rate IS NULL").fetchall():
        full = rate("stream_denoise", r["denoise_tier"]) + rate("stream_captions", r["caption_tier"])
        premium = rate("stream_captions", r["caption_tier"]) if r["caption_tier"] == "premium" else 0
        paid = max(0.0, r["seconds"] - r["trial_seconds"])
        before = paid if r["premium_off_seconds"] is None or not premium else min(paid, r["premium_off_seconds"])
        c.execute("UPDATE stream_sessions SET paid_seconds = ?, meter_closed = ?, meter_rate = ?, meter_seconds = ? WHERE id = ?",
                  (paid, full * before / 60, full - premium if before < paid else full, paid - before, r["id"]))

# Indexes on the columns above (a database made before them has no such column when SCHEMA runs, so they come after).
POST_MIGRATION_SQL = [
    # The free server-side subtitle minutes are gone (free subtitles are made in the browser).
    "DROP TABLE IF EXISTS free_usage",
    "DROP TABLE IF EXISTS deleted_emails",
    "CREATE INDEX IF NOT EXISTS jobs_root ON jobs(root_job_id) WHERE root_job_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS jobs_api_key ON jobs(api_key_id, finished_at) WHERE api_key_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS stream_sessions_api_key ON stream_sessions(api_key_id, ended_at) WHERE api_key_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS job_callbacks_due ON job_callbacks(next_attempt_at) WHERE state = 'pending'",
    # The sweeps every 30 seconds (stream.routes.close_orphans, auth.purge_expired) read only these rows.
    "CREATE INDEX IF NOT EXISTS stream_sessions_active ON stream_sessions(started_at) WHERE status = 'active'",
    "CREATE INDEX IF NOT EXISTS sessions_expiry ON sessions(expires_at)",
    "CREATE INDEX IF NOT EXISTS login_tokens_expiry ON login_tokens(expires_at)",
    "CREATE INDEX IF NOT EXISTS oauth_states_expiry ON oauth_states(expires_at)",
]


def init_db() -> None:
    c = conn()
    c.executescript(SCHEMA)
    for table, column, ddl in MIGRATIONS:
        cols = {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            c.execute(ddl)
    _meter_running_sessions(c)
    for table, column in DROPPED_COLUMNS:
        if column in {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}:
            c.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
    for sql in POST_MIGRATION_SQL:
        c.execute(sql)
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
