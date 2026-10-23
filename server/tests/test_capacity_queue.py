"""The job list poll (audit H2): an index for the user and status lookups, and one queue computation shared by requests."""
from acw import auth, db, jobs
from acw.config import get_settings


def _add_job(job_id, user, status, t=0, started=None):
    now = db.now()
    db.execute(
        "INSERT INTO jobs(id, user_id, kind, tier, status, params_json, input_name, input_bytes, claimed_seconds,"
        " est_cost_micros, created_at, queued_at, started_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (job_id, user, "transcribe", "local", status, '{"options": {}, "transcribe_tier": "local"}', "x.wav", 1000, 600, 0,
         now, now + t, started),
    )


def test_user_and_status_lookups_use_an_index_and_a_database_made_before_it_gets_one():
    db.execute("DROP INDEX jobs_user_status")  # a database from before the index
    db.init_db()  # what the app does at start
    assert [r["name"] for r in db.all_rows("PRAGMA index_info(jobs_user_status)")] == ["user_id", "status"]
    plan = " ".join(str(tuple(r)) for r in db.all_rows(
        "EXPLAIN QUERY PLAN SELECT COUNT(*) FROM jobs r WHERE r.user_id = 'u' AND r.status IN ('running','cloud_asr')"))
    assert "jobs_user_status" in plan


def test_the_sweep_indexes_exist_on_a_new_database_and_on_an_old_one():
    want = {"stream_sessions_active", "sessions_expiry", "login_tokens_expiry", "oauth_states_expiry"}
    have = lambda: {r["name"] for r in db.all_rows("SELECT name FROM sqlite_master WHERE type = 'index'")}  # noqa: E731
    assert want <= have()  # a new database
    for name in want:
        db.execute(f"DROP INDEX {name}")
    db.init_db()  # a database from before them
    assert want <= have()


def test_init_db_drops_the_session_owner_column_of_an_old_database():
    db.execute("ALTER TABLE stream_sessions ADD COLUMN owner TEXT")  # the column the live process used to record
    db.init_db()
    assert "owner" not in {r["name"] for r in db.all_rows("PRAGMA table_info(stream_sessions)")}


def test_init_db_drops_the_chain_columns_and_their_index_of_an_old_database_and_keeps_the_jobs():
    user = auth.get_or_create_user("old@example.com")["id"]
    _add_job("old1", user, "done")
    db.execute("ALTER TABLE jobs ADD COLUMN source_job_id TEXT")  # what the re-runs of an older version left
    db.execute("ALTER TABLE jobs ADD COLUMN root_job_id TEXT")
    db.execute("CREATE INDEX jobs_root ON jobs(root_job_id) WHERE root_job_id IS NOT NULL")
    db.execute("UPDATE jobs SET source_job_id = 'src', root_job_id = 'root' WHERE id = 'old1'")
    before = {k: v for k, v in dict(db.one("SELECT * FROM jobs WHERE id = 'old1'")).items() if k not in ("source_job_id", "root_job_id")}
    db.init_db()
    assert not {"source_job_id", "root_job_id"} & {r["name"] for r in db.all_rows("PRAGMA table_info(jobs)")}
    assert "jobs_root" not in {r["name"] for r in db.all_rows("SELECT name FROM sqlite_master WHERE type = 'index'")}
    assert dict(db.one("SELECT * FROM jobs WHERE id = 'old1'")) == before


def test_the_queue_is_computed_once_a_second_and_matches_the_slow_estimates(monkeypatch):
    a = auth.get_or_create_user("a@example.com")["id"]
    b = auth.get_or_create_user("b@example.com")["id"]
    _add_job("run1", a, "running", started=db.now() - 30)
    _add_job("q1", a, "queued", 1)
    _add_job("q2", b, "queued", 2)
    _add_job("q3", b, "queued", 3)
    jobs.invalidate_queue_cache()
    ids, waits = jobs.queue_snapshot()
    assert ids == jobs.queued_ids() and len(ids) == 3
    work = jobs._seconds_left(dict(db.one("SELECT * FROM jobs WHERE id = 'run1'")), db.now())
    for jid in ids:
        assert waits[jid] == int(work / get_settings().workers), jid
        work += jobs._seconds_left(dict(db.one("SELECT * FROM jobs WHERE id = ?", (jid,))), db.now())

    def no_query(*a, **k):
        raise AssertionError("queried")

    with monkeypatch.context() as m:  # a second call within the window does not touch the database at all
        m.setattr(db, "all_rows", no_query)
        assert jobs.queue_snapshot() == (ids, waits)
    _add_job("q4", b, "queued", 4)
    assert "q4" not in jobs.queue_snapshot()[0]  # a change by another process shows after the window
    jobs.invalidate_queue_cache()  # a change made by this process drops the cache at once
    assert "q4" in jobs.queue_snapshot()[0]
