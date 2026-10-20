"""Runtime configuration, read once from the environment.

Secrets live in an EnvironmentFile (deploy/acw.env.example lists every key);
nothing secret has a default here. Paths default to the on-prem layout under
~/audio-cleaner so a dev checkout and the deployed service share binaries and
models without copying them around.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    return int(raw) if raw is not None else default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_list(name: str) -> list[str]:
    raw = _env(name, "") or ""
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    # Where the app is served. PUBLIC_ORIGIN has no trailing slash; BASE_PATH is
    # "" for a dedicated (sub)domain or e.g. "/audio" when sharing a host.
    public_origin: str
    base_path: str
    environment: str  # "production" or "development"

    home: Path
    data_dir: Path
    jobs_dir: Path
    db_path: Path
    bin_dir: Path
    models_dir: Path
    web_dist: Path | None

    session_secret: bytes
    session_days: int

    google_client_id: str | None
    google_client_secret: str | None

    smtp_host: str | None
    smtp_port: int
    smtp_user: str | None
    smtp_password: str | None
    smtp_from: str | None

    stripe_secret_key: str | None
    stripe_webhook_secret: str | None
    stripe_portal_configuration: str | None  # bpc_...; unset: the portal settings saved in the dashboard
    stripe_statement_descriptor: str
    stripe_automatic_tax: bool
    stripe_tax_code: str
    stripe_payment_method_configuration: str | None
    live_trial_seconds: int  # free Standard live every new account gets once per mailbox (0: no trial)

    openai_api_key: str | None

    turnstile_site_key: str | None
    turnstile_secret: str | None

    admin_emails: list[str] = field(default_factory=list)

    max_upload_bytes: int = 4 * 1024**3
    workers: int = 3  # file-job worker units (ACW_WORKERS), used for queue estimates
    # How long a file job's upload, its cached blocks, its intermediate copies and its results are kept together, counted from
    # its last edit (finishing a run, a timeline save, a quote, a process or an export; ACW_JOB_RETENTION_MINUTES), and never
    # later than `job_max_retention_seconds` after the upload (ACW_JOB_MAX_RETENTION_MINUTES). The sweeps run every 30 to 60
    # seconds, so the deletion is accurate to about a minute.
    job_retention_seconds: int = 2 * 3600
    job_max_retention_seconds: int = 6 * 3600
    # A first run looks for suggestions (acw_common.suggest: about 0.22 s per second of audio) for recordings up to this long
    # (ACW_SUGGEST_MAX_MINUTES); a longer one gets them when they are asked for (POST /api/jobs/{id}/suggestions).
    suggest_max_seconds: int = 3600
    # File jobs an account may have at once (uploading, waiting, running; re-runs count). ACW_MAX_ACTIVE_JOBS.
    max_active_jobs: int = 3
    stream_worker_socket: Path | None = None

    @property
    def production(self) -> bool:
        return self.environment == "production"

    @property
    def cookie_path(self) -> str:
        return (self.base_path or "") + "/"

    @property
    def session_cookie_name(self) -> str:
        # On a host of its own (no base path) the __Host- prefix applies: Secure, Path=/, no Domain, and no
        # sibling subdomain can set or replace it. Sharing a host under a path (today: /audio) needs
        # __Secure-, because __Host- would force Path=/ and send the cookie to the other apps there.
        if not self.production:
            return "acw_session"
        return "__Host-acw_session" if not self.base_path else "__Secure-acw_session"

    def url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.public_origin}{self.base_path}{path}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    home = Path(_env("ACW_HOME", str(Path.home() / "audio-cleaner"))).expanduser()
    data_dir = Path(_env("ACW_DATA_DIR", str(home / "data")))
    environment = _env("ACW_ENV", "development")
    secret = _env("ACW_SESSION_SECRET")
    if not secret:
        if environment == "production":
            raise RuntimeError("ACW_SESSION_SECRET must be set in production")
        secret = "dev-only-insecure-secret-change-me"
    if environment == "production" and len(secret) < 32:
        raise RuntimeError("ACW_SESSION_SECRET must be at least 32 characters")
    base_path = (_env("ACW_BASE_PATH", "") or "").rstrip("/")
    if base_path and not base_path.startswith("/"):
        base_path = "/" + base_path
    web_dist = _env("ACW_WEB_DIST")
    return Settings(
        public_origin=(_env("ACW_PUBLIC_ORIGIN", "http://127.0.0.1:8790") or "").rstrip("/"),
        base_path=base_path,
        environment=environment,
        home=home,
        data_dir=data_dir,
        jobs_dir=Path(_env("ACW_JOBS_DIR", str(data_dir / "jobs"))),
        db_path=Path(_env("ACW_DB_PATH", str(data_dir / "acw.sqlite3"))),
        bin_dir=Path(_env("ACW_BIN_DIR", str(home / "bin"))),
        models_dir=Path(_env("ACW_MODELS_DIR", str(home / "models"))),
        web_dist=Path(web_dist) if web_dist else None,
        session_secret=secret.encode(),
        session_days=_env_int("ACW_SESSION_DAYS", 30),
        google_client_id=_env("GOOGLE_CLIENT_ID"),
        google_client_secret=_env("GOOGLE_CLIENT_SECRET"),
        smtp_host=_env("SMTP_HOST"),
        smtp_port=_env_int("SMTP_PORT", 587),
        smtp_user=_env("SMTP_USER"),
        smtp_password=_env("SMTP_PASSWORD"),
        smtp_from=_env("SMTP_FROM"),
        stripe_secret_key=_env("STRIPE_SECRET_KEY"),
        stripe_webhook_secret=_env("STRIPE_WEBHOOK_SECRET"),
        stripe_portal_configuration=_env("STRIPE_PORTAL_CONFIGURATION"),
        stripe_statement_descriptor=_env("STRIPE_STATEMENT_DESCRIPTOR", "AUDIO CLEANER") or "AUDIO CLEANER",
        stripe_automatic_tax=_env_bool("STRIPE_AUTOMATIC_TAX", False),
        # Electronically supplied services; change if your tax advisor says otherwise.
        stripe_tax_code=_env("STRIPE_TAX_CODE", "txcd_10000000") or "txcd_10000000",
        # A payment method configuration with cards only (incl. Apple Pay / Google Pay): payments are authorized,
        # then the card's issuing country is checked before they are charged, which needs a card.
        stripe_payment_method_configuration=_env("STRIPE_PAYMENT_METHOD_CONFIGURATION", "") or None,
        # Free Standard live (noise removal and/or Standard captions) every new account gets once per mailbox, ever.
        live_trial_seconds=max(0, _env_int("ACW_LIVE_TRIAL_SECONDS", 7200)),
        openai_api_key=_env("OPENAI_API_KEY"),
        turnstile_site_key=_env("TURNSTILE_SITE_KEY"),
        turnstile_secret=_env("TURNSTILE_SECRET"),
        admin_emails=[e.lower() for e in _env_list("ACW_ADMIN_EMAILS")],
        max_upload_bytes=_env_int("ACW_MAX_UPLOAD_BYTES", 4 * 1024**3),
        workers=_env_int("ACW_WORKERS", 3),
        job_retention_seconds=60 * max(1, _env_int("ACW_JOB_RETENTION_MINUTES", 120)),
        job_max_retention_seconds=60 * max(1, _env_int("ACW_JOB_MAX_RETENTION_MINUTES", 360)),
        suggest_max_seconds=60 * max(0, _env_int("ACW_SUGGEST_MAX_MINUTES", 60)),
        max_active_jobs=max(1, _env_int("ACW_MAX_ACTIVE_JOBS", 3)),
        stream_worker_socket=Path(_env("ACW_STREAM_SOCKET", str(data_dir / "stream.sock"))),
    )
