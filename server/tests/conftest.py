import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "server"), str(ROOT / "common")]

ORIGIN = "http://testserver"


_SCHEMA: list[Path] = []  # the database schema, built once per run (see _fresh_database)


def _build_schema(path: Path) -> None:
    from acw import db
    from acw.config import get_settings

    saved = os.environ.get("ACW_DB_PATH")
    os.environ["ACW_DB_PATH"] = str(path)
    get_settings.cache_clear()
    db.reset_connection()
    try:
        db.init_db()
        db.conn().execute("PRAGMA wal_checkpoint(TRUNCATE)")  # the schema must be in the main file, not in the WAL
    finally:
        db.reset_connection()
        if saved is None:
            os.environ.pop("ACW_DB_PATH", None)
        else:
            os.environ["ACW_DB_PATH"] = saved
        get_settings.cache_clear()


def _fresh_database(db_file: Path, factory) -> None:
    """A database file with the current schema. Building the schema takes ~0.15 s, so it is built once per run and
    copied into each test's data directory (init_db then only confirms it)."""
    if not _SCHEMA or not _SCHEMA[0].is_file():
        path = factory.mktemp("schema") / "acw.sqlite3"
        _build_schema(path)
        _SCHEMA[:] = [path]
    db_file.parent.mkdir(parents=True, exist_ok=True)
    if not db_file.exists():
        shutil.copyfile(_SCHEMA[0], db_file)


@pytest.fixture(autouse=True)
def env(tmp_path, tmp_path_factory, monkeypatch):
    monkeypatch.setenv("ACW_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ACW_PUBLIC_ORIGIN", ORIGIN)
    monkeypatch.delenv("ACW_BASE_PATH", raising=False)
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test_secret")
    monkeypatch.setenv("ACW_LIVE_TRIAL_SECONDS", "0")  # tests start without the free live trial; test_live_trial.py turns it on
    from acw import db
    from acw.config import get_settings
    from acw.security import limiter

    from acw.stream import routes

    get_settings.cache_clear()
    db.reset_connection()
    limiter.reset()
    routes._lag.samples.clear()  # how late the event loop ran in earlier tests says nothing about this one
    _fresh_database(get_settings().db_path, tmp_path_factory)
    db.init_db()
    yield
    db.reset_connection()
    get_settings.cache_clear()


@pytest.fixture
def gpu_up():
    """The GPU worker is up (the voice rebuild can be ordered: it runs only on the card)."""
    from acw import broker

    broker.mark_gpu_worker("alive")


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from acw.app import create_app

    with TestClient(create_app(), base_url=ORIGIN) as c:
        yield c


def login(client, email="user@example.com", accept_tos=True):
    from acw import auth

    tok = auth.create_login_token(email, "admin_login", "test")
    r = client.post("/api/auth/email/verify", json={"token": tok}, headers={"Origin": ORIGIN})
    assert r.status_code == 200, r.text
    me = client.get("/api/me").json()
    h = {"X-CSRF-Token": me["csrfToken"], "Origin": ORIGIN}
    if accept_tos:
        assert client.post("/api/account/tos", json={"version": auth.TOS_VERSION}, headers=h).status_code == 200
    return me["user"]["id"], h


@pytest.fixture
def signed_in(client):
    return login(client)


def settle_payment(user_id, cents=1000):
    """A top-up that went through: the account counts as paid (large top-ups)."""
    from acw import db

    now = db.now()
    db.execute(
        "INSERT INTO payments(id, user_id, kind, status, amount_cents, created_at, updated_at)"
        " VALUES (?, ?, 'checkout', 'succeeded', ?, ?, ?)", (db.new_id(), user_id, cents, now, now))


# ------------------------------------------------------------------------------------- the native live engine

REPO_ROOT = Path(__file__).resolve().parents[2]
_HOME = Path.home() / "audio-cleaner"
LIVE_ENGINE = next((p for p in (REPO_ROOT / "native/live/target/release/acw-live", _HOME / "bin/acw-live") if p.is_file()), None)
LIVE_MODELS = _HOME / "models"


def _shared_libdf() -> Path:
    """The live engine needs a libdf that shares the model between sessions (df_shares_model): the installed one, or
    the build output of native/deepfilter/build.sh until it is installed."""
    import ctypes

    for p in (_HOME / "bin/libdf.so", _HOME / "src/deepfilter-build/out/libdf.so"):
        try:
            if p.is_file() and hasattr(ctypes.CDLL(str(p)), "df_shares_model"):
                return p
        except OSError:
            continue
    return _HOME / "bin/libdf.so"


LIVE_LIBDF = _shared_libdf()


def live_engine_available() -> bool:
    return LIVE_ENGINE is not None and (LIVE_MODELS / "DeepFilterNet3_onnx.tar.gz").is_file() and LIVE_LIBDF.is_file()


@pytest.fixture
def live_engine(tmp_path, monkeypatch):
    """Starts the native live engine (native/live; skipped where it isn't built or the models aren't installed) on
    its own socket and points the web side at it. Call it with extra environment: live_engine(ACW_...="...")."""
    import subprocess
    import time

    if not live_engine_available():
        pytest.skip("native live engine or models not available here")
    procs = []

    def start(**env):
        sock = tmp_path / f"stream{len(procs)}.sock"
        e = {**os.environ, "ACW_ENV": "development", "ACW_DATA_DIR": str(tmp_path), "ACW_STREAM_SOCKET": str(sock),
             "ACW_BIN_DIR": str(LIVE_LIBDF.parent), "ACW_MODELS_DIR": str(LIVE_MODELS), "ACW_MIN_JOB_CORES": "1",
             "ACW_LIVE_IGNORE_HOST_LOAD": "1", **env}
        procs.append(subprocess.Popen([str(LIVE_ENGINE)], env=e, stderr=subprocess.DEVNULL))
        for _ in range(200):
            if sock.exists():
                break
            time.sleep(0.05)
        monkeypatch.setenv("ACW_STREAM_SOCKET", str(sock))
        from acw.config import get_settings

        get_settings.cache_clear()
        return str(sock)

    yield start
    for p in procs:
        p.terminate()
        p.wait(10)
    from acw.config import get_settings

    get_settings.cache_clear()  # the next test must not talk to this (stopped) engine
