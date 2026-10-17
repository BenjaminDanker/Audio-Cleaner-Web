"""The job list poll (audit H2): an index for the queue order and one queue computation shared by all requests."""
from acw import auth, db, jobs


def _add_job(job_id, user, status, t=0, started=None):
    now = db.now()
    db.execute(
        "INSERT INTO jobs(id, user_id, kind, tier, status, params_json, input_name, input_bytes, claimed_seconds,"
        " est_cost_micros, created_at, queued_at, started_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (job_id, user, "transcribe", "local", status, '{"options": {}, "transcribe_tier": "local"}', "x.wav", 1000, 600, 0,
         now, now + t, started),
    )


def test_jobs_have_a_user_and_status_index():
    names = {r["name"] for r in db.all_rows("PRAGMA index_list(jobs)")}
    assert "jobs_user_status" in names
    assert [r["name"] for r in db.all_rows("PRAGMA index_info(jobs_user_status)")] == ["user_id", "status"]
    plan = " ".join(str(tuple(r)) for r in db.all_rows(
        "EXPLAIN QUERY PLAN SELECT COUNT(*) FROM jobs r WHERE r.user_id = 'u' AND r.status IN ('running','cloud_asr')"))
    assert "jobs_user_status" in plan


def test_an_existing_database_gets_the_index_when_the_app_starts(tmp_path):
    db.execute("DROP INDEX jobs_user_status")
    assert "jobs_user_status" not in {r["name"] for r in db.all_rows("PRAGMA index_list(jobs)")}
    db.init_db()
    assert "jobs_user_status" in {r["name"] for r in db.all_rows("PRAGMA index_list(jobs)")}


def test_the_queue_snapshot_matches_the_slow_estimates_and_is_shared_for_a_second(monkeypatch):
    a = auth.get_or_create_user("a@example.com")["id"]
    b = auth.get_or_create_user("b@example.com")["id"]
    _add_job("run1", a, "running", started=db.now() - 30)
    _add_job("q1", a, "queued", 1)
    _add_job("q2", b, "queued", 2)
    _add_job("q3", b, "queued", 3)
    jobs.invalidate_queue_cache()
    ids, waits = jobs.queue_snapshot()
    assert ids == jobs.queued_ids() and len(ids) == 3
    for i, jid in enumerate(ids):
        assert waits[jid] == jobs.queue_wait_seconds(ids[:i]), jid
    # a second call within the window does not touch the database at all
    def no_query(*a, **k):
        raise AssertionError("queried")

    with monkeypatch.context() as m:
        m.setattr(db, "all_rows", no_query)
        assert jobs.queue_snapshot() == (ids, waits)
        jobs.public_job(dict(db.one("SELECT * FROM jobs WHERE id = 'q2'")))
    # a change made by another process shows after the window; one made by this process drops the cache at once
    _add_job("q4", b, "queued", 4)
    assert "q4" not in jobs.queue_snapshot()[0]
    jobs.invalidate_queue_cache()
    assert "q4" in jobs.queue_snapshot()[0]
    monkeypatch.setattr(jobs, "QUEUE_CACHE_SECONDS", 0.0)
    _add_job("q5", b, "queued", 5)
    assert "q5" in jobs.queue_snapshot()[0]


def test_the_cache_does_not_leak_between_databases(monkeypatch, tmp_path):
    a = auth.get_or_create_user("a@example.com")["id"]
    _add_job("q1", a, "queued", 1)
    jobs.invalidate_queue_cache()
    assert jobs.queue_snapshot()[0] == ["q1"]
    monkeypatch.setenv("ACW_DB_PATH", str(tmp_path / "other.sqlite3"))
    from acw.config import get_settings

    get_settings.cache_clear()
    db.reset_connection()
    db.init_db()
    assert jobs.queue_snapshot()[0] == []
