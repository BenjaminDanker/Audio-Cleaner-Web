"""Metrics (OBS-01): structured log lines, `acw.admin stats`, and the monitor's Prometheus file."""
import json
import logging
import re

import pytest

from acw import auth, db, jobs, metrics, monitor, stats
from acw.billing import ledger


def user(email="a@example.com"):
    uid = auth.get_or_create_user(email)["id"]
    ledger.credit(uid, 50_000_000, "grant", "t", f"g-{email}")
    return uid


def make_job(uid, **kw):
    args = dict(kind="denoise", denoise_tier="standard", transcribe_tier=None, filename="a.wav", size_bytes=1000,
                claimed_seconds=600.0, options={})
    return jobs.create_job(uid, **{**args, **kw})["id"]


def lines(caplog_text_records):
    return [json.loads(r.getMessage()) for r in caplog_text_records if r.name == "acw.metrics"]


@pytest.fixture
def metric_lines():
    """Captures the acw.metrics lines (the logger does not propagate, so attach a handler directly)."""
    records: list[logging.LogRecord] = []

    class H(logging.Handler):
        def emit(self, record):
            records.append(record)

    h = H()
    lg = logging.getLogger("acw.metrics")
    metrics._setup()
    lg.addHandler(h)
    yield lambda: lines(records)
    lg.removeHandler(h)


# ------------------------------------------------------------------ log lines

def test_a_finished_job_writes_one_json_line(metric_lines):
    from acw.worker import main as W

    uid = user()
    jid = make_job(uid)
    now = db.now()
    db.execute("UPDATE jobs SET status = 'queued', queued_at = ? WHERE id = ?", (now - 100, jid))
    job = W.claim("w1")
    assert job["id"] == jid
    wait = db.one("SELECT wait_seconds FROM jobs WHERE id = ?", (jid,))["wait_seconds"]
    assert wait in (100, 101)  # the wait is recorded when the first worker takes the job
    jobs.update_params(jid, noise_profile="heavy")
    job = dict(db.one("SELECT * FROM jobs WHERE id = ?", (jid,)))
    d = jobs.job_dir(jid)
    (d / "out.wav").write_bytes(b"x")
    db.execute("UPDATE jobs SET duration_seconds = 600 WHERE id = ?", (jid,))
    job["duration_seconds"] = 600
    W._complete(job, d, {"outputs": [{"file": "out.wav", "kind": "audio"}]})
    [line] = [x for x in metric_lines() if x["event"] == "job_done"]
    assert line["job"] == jid and line["tier"] == "standard" and line["kind"] == "denoise"
    assert line["audio_s"] == 600.0 and line["queue_wait_s"] == wait and line["noise"] == "heavy"
    assert line["cost_micros"] > 0 and line["run_s"] >= 0 and line["attempts"] == 1
    assert "user" not in json.dumps(line) and "@" not in json.dumps(line)


def test_a_failed_job_and_a_cancel_each_write_one_line(metric_lines):
    uid = user()
    jid = make_job(uid)
    job = dict(db.one("SELECT * FROM jobs WHERE id = ?", (jid,)))
    jobs.finish_failed(job, "Processing failed; you were not charged.")
    jobs.finish_failed(job, "Processing failed; you were not charged.")  # safe to call twice, logged once
    jid2 = make_job(uid)
    jobs.finish_failed(dict(db.one("SELECT * FROM jobs WHERE id = ?", (jid2,))), "Canceled.", status="canceled")
    found = [x for x in metric_lines() if x["event"] == "job_failed"]
    assert [x["status"] for x in found] == ["failed", "canceled"]
    assert found[0]["error"].startswith("Processing failed") and found[0]["noise"] == "unknown"


def test_a_live_session_writes_one_line_when_it_ends(metric_lines):
    from acw.stream import routes

    uid = user()
    sid = db.new_id()
    db.execute("INSERT INTO stream_sessions(id, user_id, denoise_tier, caption_tier, status, seconds, started_at)"
               " VALUES (?,?,?,?, 'active', 125.5, ?)", (sid, uid, "light", "local", db.now() - 200))
    assert routes.close_orphans(min_age=0, reason="server_restart") == 1
    [line] = [x for x in metric_lines() if x["event"] == "live_session_end"]
    assert line["session"] == sid and line["end_reason"] == "server_restart" and line["denoise"] == "light"
    assert line["audio_s"] == 125.5 and line["frames"] == 0


# ------------------------------------------------------------------ stats

def seed(now):
    """Four finished standard jobs (waits 10/20/30/300 s), one failed, one rebuild, one free, a live day, a top-up."""
    uid = user()
    for wait, run in ((10, 60), (20, 70), (30, 80), (300, 90)):
        jid = make_job(uid)
        db.execute(
            "UPDATE jobs SET status = 'done', duration_seconds = 600, wait_seconds = ?, queued_at = ?, started_at = ?, finished_at = ?, created_at = ?,"
            " cost_micros = 6000 WHERE id = ?", (wait, now - 500, now - 400, now - 400 + run, now - 600, jid))
    bad = make_job(uid)
    db.execute("UPDATE jobs SET status = 'failed', error = 'Processing failed; you were not charged.', started_at = ?, finished_at = ?,"
               " created_at = ? WHERE id = ?", (now - 300, now - 200, now - 600, bad))
    bad2 = make_job(uid)
    db.execute("UPDATE jobs SET status = 'failed', error = 'Processing failed; you were not charged.', created_at = ? WHERE id = ?", (now - 600, bad2))
    queued = make_job(uid, kind="transcribe", denoise_tier=None, transcribe_tier="local", claimed_seconds=120.0)
    db.execute("UPDATE jobs SET status = 'queued', created_at = ?, tier = 'local' WHERE id = ?", (now - 100, queued))
    old = make_job(uid)  # outside a 1-day window
    db.execute("UPDATE jobs SET status = 'done', created_at = ?, started_at = ?, finished_at = ? WHERE id = ?",
               (now - 5 * 86400, now - 5 * 86400, now - 5 * 86400 + 50, old))
    for reason, secs, late in (("client_stop", 600.0, 3), ("client_stop", 300.0, 0), ("idle_timeout", 60.0, 1), ("full", 0.0, 0)):
        db.execute(
            "INSERT INTO stream_sessions(id, user_id, denoise_tier, caption_tier, status, seconds, end_reason, started_at, ended_at, frames, late_frames, dropped_frames)"
            " VALUES (?,?,?,?, 'ended', ?,?,?,?, ?,?, 0)",
            (db.new_id(), uid, "light" if secs else None, None, secs, reason, now - 3600, now - 3000, int(secs * 50), late))
    ledger.credit(uid, 1, "adjustment", "t", "x")  # a non-usage entry must not count as usage
    db.execute("INSERT INTO ledger(user_id, amount_micros, kind, ref_type, ref_id, description, balance_after_micros, created_at)"
               " VALUES (?, -24000, 'usage', 'job', 'j', 'x', 0, ?)", (uid, now - 100))
    db.execute("INSERT INTO ledger(user_id, amount_micros, kind, ref_type, ref_id, description, balance_after_micros, created_at)"
               " VALUES (?, -6000, 'usage', 'stream', 's', 'x', 0, ?)", (uid, now - 100))
    db.execute("INSERT INTO payments(id, user_id, kind, status, amount_cents, created_at, updated_at) VALUES ('p1', ?, 'checkout', 'succeeded', 1000, ?, ?)",
               (uid, now - 100, now - 100))
    return uid


def test_stats_report_numbers():
    now = db.now()
    seed(now)
    r = stats.report(1, now)
    std = r["jobs"]["standard"]
    assert std["status"] == {"done": 4, "failed": 2}
    assert std["wait_p50"] == 20 and std["wait_p95"] == 300
    assert std["run_p50"] == 70 and std["run_p95"] == 90
    assert std["rtf"] == pytest.approx((60 + 70 + 80 + 90) / 2400)
    assert std["audio_hours_done"] == pytest.approx(2400 / 3600)
    assert r["jobs"]["local"]["status"] == {"queued": 1}
    assert r["in_progress"] == {"queued": 1}
    assert r["failure_ratio"] == pytest.approx(2 / 6) and r["top_errors"] == [("Processing failed; you were not charged.", 2)]
    live = r["live"]
    assert live["sessions"] == 4 and live["minutes"] == pytest.approx(16.0)
    assert live["end_reasons"] == {"client_stop": 2, "idle_timeout": 1, "full": 1}
    assert live["frames"] == 48000 and live["late_frames"] == 4 and live["dropped_frames"] == 0
    assert r["usage_usd"] == {"job": pytest.approx(0.024), "stream": pytest.approx(0.006)}
    assert r["topups_usd"] == 10.0
    # a longer window includes the older job
    assert stats.report(30, now)["jobs"]["standard"]["status"]["done"] == 5


def test_stats_text_report():
    now = db.now()
    seed(now)
    text = stats.render(stats.report(1, now))
    for needle in ("File jobs", "standard", "failure ratio: 33.3% (2 of 6 finished)", "Processing failed; you were not charged.",
                   "Live sessions", "end reasons: client_stop 2, idle_timeout 1, full 1", "late (>100 ms) 4",
                   "usage charged: $0.03", "top-ups paid: $10.00", "RTF"):
        assert needle in text, needle


def test_stats_on_an_empty_database():
    empty = stats.render(stats.report(30))
    assert "none" in empty and "failure ratio: -" in empty and "frames: not recorded" in empty


def test_stats_only_reads():
    seed(db.now())
    before = db.one("SELECT COUNT(*) AS n FROM jobs")["n"]
    conn = db.conn()
    changes = conn.total_changes
    stats.render(stats.report(30))
    assert conn.total_changes == changes and db.one("SELECT COUNT(*) AS n FROM jobs")["n"] == before


# ------------------------------------------------------------------ the Prometheus file

def parse_prom(text):
    samples = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r'([a-z_0-9]+)(\{[^}]*\})? (-?[0-9.e+]+)', line)
        assert m, f"bad sample line: {line!r}"
        samples[m.group(1) + (m.group(2) or "")] = float(m.group(3))
    return samples


def test_metrics_file_content_and_format(tmp_path):
    now = db.now()
    seed(now)
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "acw-20260101.sqlite3.gz").write_bytes(b"x")
    findings = [monitor.Finding("disk_low", False, "x"), monitor.Finding("queue_long", True, "y")]
    states = {"acw-web.service": {"ActiveState": "active"}, "acw-bulk.service": {"ActiveState": "failed"}}
    path = monitor.write_metrics(findings, states, now, path=tmp_path / "metrics.prom", backup_dir=tmp_path / "b")
    text = path.read_text()
    s = parse_prom(text)
    assert s["acw_queue_length"] == 1
    assert s['acw_jobs_finished_24h{status="done"}'] == 4 and s['acw_jobs_finished_24h{status="failed"}'] == 1  # the 5-day-old job is out
    assert s["acw_queue_wait_p95_seconds_24h"] == 300
    assert s['acw_jobs_finished_total{status="done"}'] == 5
    assert s['acw_live_sessions_ended_total{end_reason="client_stop"}'] == 2
    assert s["acw_live_late_frames_total"] == 4 and s["acw_live_sessions"] == 0
    assert s["acw_usage_micros_total"] == 30000
    assert s['acw_unit_active{unit="acw-web.service"}'] == 1 and s['acw_unit_active{unit="acw-bulk.service"}'] == 0
    assert s['acw_check_problem{kind="queue_long"}'] == 1 and s['acw_check_problem{kind="disk_low"}'] == 0
    assert s["acw_disk_free_bytes"] > 0 and 0 <= s["acw_backup_age_seconds"] < 120
    assert s["acw_metrics_written_timestamp_seconds"] == now
    assert "# TYPE acw_queue_length gauge" in text and "# TYPE acw_jobs_finished_total counter" in text
    assert text.count("# HELP acw_jobs_finished_24h") == 1  # one header per metric
    assert not list(tmp_path.glob("*.tmp*"))  # nothing left over from the atomic write


def test_the_monitor_run_writes_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "unit_states", lambda units: {"acw-web.service": {"ActiveState": "active", "NRestarts": "0"}})
    monkeypatch.setattr(monitor, "expected_units", lambda: ["acw-web.service"])
    monkeypatch.setattr(monitor, "BACKUP_DIR", tmp_path / "nobackup")
    findings = monitor.run(ctx=lambda: "host", http=lambda url: (True, "ok"))
    assert findings
    text = (monitor.metrics_path()).read_text()
    assert 'acw_unit_active{unit="acw-web.service"} 1' in text and "acw_queue_length 0" in text


def test_label_values_are_escaped():
    p = metrics.Prom()
    p.add("acw_x", "gauge", "x", 1, kind='a"b\\c\nd')
    assert 'kind="a\\"b\\\\c\\nd"' in p.text()


# ------------------------------------------------------------------ the bulk process check

@pytest.mark.parametrize(("check", "unit", "port", "kind"), [
    (monitor.check_bulk, "acw-bulk.service", 8794, "health_bulk"),
    (monitor.check_live, "acw-live.service", 8795, "health_live"),
])
def test_bulk_and_live_health_are_checked_only_where_the_unit_is_enabled(tmp_path, check, unit, port, kind):
    calls = []

    def ok(url):
        calls.append(url)
        return False, "URLError"

    assert check(ok, tmp_path) == []  # not enabled: nothing to check
    wants = tmp_path / "default.target.wants"
    wants.mkdir()
    (wants / unit).write_text("")
    [f] = check(ok, tmp_path)
    assert f.kind == kind and f.bad and calls == [f"http://127.0.0.1:{port}/api/health"]
    assert not check(lambda u: (True, "ok"), tmp_path)[0].bad
