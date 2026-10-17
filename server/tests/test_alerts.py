import json
import time

import pytest

from acw import alerts, db, jobs, monitor
from acw.config import get_settings


@pytest.fixture
def mail(monkeypatch):
    """Alerts configured; captures what would be mailed."""
    sent = []
    monkeypatch.setenv("ACW_ALERT_EMAIL", "owner@example.com")
    monkeypatch.setenv("SMTP_HOST", "smtp.invalid")
    monkeypatch.setenv("SMTP_FROM", "acw@example.com")
    get_settings.cache_clear()
    monkeypatch.setattr(alerts, "_deliver", lambda subject, body: sent.append((subject, body)))
    return sent


def _age(kind, seconds):
    db.execute("UPDATE alert_state SET last_sent_at = last_sent_at - ?, first_at = first_at - ? WHERE kind = ?", (seconds, seconds, kind))


# --- sending: de-duplication, resolution, retries, what may be mailed ----------------------------------------------

def test_dedup_within_the_hour_then_reminder(mail):
    assert alerts.send_alert("disk_low", "Disk low", "5 GB free") is True
    assert alerts.send_alert("disk_low", "Disk low", "5 GB free") is False
    assert alerts.send_alert("other", "Something else", "x") is True  # kinds are independent
    assert len(mail) == 2
    _age("disk_low", 3601)
    assert alerts.send_alert("disk_low", "Disk low", "still") is True
    assert len(mail) == 3
    _age("disk_low", 3601)  # after two mails the repeat interval grows to 6 h
    assert alerts.send_alert("disk_low", "Disk low", "still") is False
    _age("disk_low", 6 * 3600)
    assert alerts.send_alert("disk_low", "Disk low", "still") is True
    assert mail[0][0] == "Disk low"  # prefix is added by the SMTP layer, not stored


def test_resolve_sends_one_mail_and_rearms(mail):
    alerts.send_alert("queue_long", "Queue long", "body")
    assert alerts.resolve_alert("queue_long", "Resolved: Queue long") is True
    assert alerts.resolve_alert("queue_long") is False  # already resolved: nothing more
    assert [m[0] for m in mail] == ["Queue long", "Resolved: Queue long"]
    assert alerts.send_alert("queue_long", "Queue long", "again") is True  # re-fires at once
    assert len(mail) == 3


def test_event_alerts_limited_to_one_per_hour(mail):
    assert alerts.send_alert("web_unclean_stop", "back up", "b", event=True)
    assert not alerts.send_alert("web_unclean_stop", "back up", "b", event=True)
    _age("web_unclean_stop", 3601)
    assert alerts.send_alert("web_unclean_stop", "back up", "b", event=True)


def test_observe_confirms_and_clears(mail):
    alerts.observe("health_local", True, "API down", "b", confirm=2)
    assert mail == []
    alerts.observe("health_local", True, "API down", "b", confirm=2)
    assert len(mail) == 1
    alerts.observe("health_local", False, "API down", "b", clear=2)
    assert len(mail) == 1  # one good run isn't enough
    alerts.observe("health_local", False, "API down", "b", clear=2)
    assert len(mail) == 2 and mail[1][0].startswith("Resolved")
    alerts.observe("health_local", False, "API down", "b")  # nothing active: silent
    assert len(mail) == 2


def test_unconfigured_smtp_never_crashes_and_the_alert_is_mailed_once_configured(monkeypatch, caplog):
    monkeypatch.delenv("ACW_ALERT_EMAIL", raising=False)
    monkeypatch.delenv("SMTP_HOST", raising=False)
    get_settings.cache_clear()
    assert not alerts.configured()
    with caplog.at_level("ERROR"):
        assert alerts.send_alert("disk_low", "Disk low", "b") is False
        alerts.send_alert_bg("x", "s", "b")
        alerts.report_openai("t", 401, b'{"error": {"type": "invalid_request_error", "code": "invalid_api_key"}}')
    assert "ALERT NOT SENT" in caplog.text
    row = db.one("SELECT active, last_sent_at FROM alert_state WHERE kind = 'disk_low'")
    assert row["active"] == 1 and row["last_sent_at"] == 0
    assert alerts.resolve_alert("disk_low") is False  # never mailed, so no "resolved" mail
    # once mail is configured the same condition is mailed (not swallowed by de-dup)
    monkeypatch.setenv("ACW_ALERT_EMAIL", "o@example.com")
    monkeypatch.setenv("SMTP_HOST", "smtp.invalid")
    monkeypatch.setenv("SMTP_FROM", "a@example.com")
    get_settings.cache_clear()
    sent = []
    monkeypatch.setattr(alerts, "_deliver", lambda s, b: sent.append(s))
    assert alerts.send_alert("disk_low", "Disk low", "b") is True
    assert sent == ["Disk low"]


def test_smtp_failure_is_retried_not_lost(mail, monkeypatch):
    def boom(subject, body):
        raise OSError("connection refused")

    monkeypatch.setattr(alerts, "_deliver", boom)
    assert alerts.send_alert("disk_low", "Disk low", "b") is False
    monkeypatch.setattr(alerts, "_deliver", lambda s, b: mail.append((s, b)))
    assert alerts.send_alert("disk_low", "Disk low", "b") is True


def test_mail_is_scrubbed(mail):
    alerts.send_alert("k", "s", "key sk-abcdefghijklmnop user bob@example.com Authorization: Bearer abcdefghijkl1234 password=hunter2")
    body = mail[0][1]
    assert "sk-abc" not in body and "bob@example.com" not in body and "hunter2" not in body and "abcdefghijkl1234" not in body


def test_openai_reports(mail, monkeypatch):
    class Inline:  # run the background thread inline
        def __init__(self, target, **kw):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(alerts.threading, "Thread", Inline)
    alerts.report_openai("file transcription", 429, json.dumps({"error": {"type": "insufficient_quota", "code": "insufficient_quota",
                                                                          "message": "secret request text"}}).encode())
    alerts.report_openai("file transcription", 429, b'{"error": {"type": "rate_limit", "code": "rate_limit_exceeded"}}')  # plain rate limit: ignored
    alerts.report_openai("live captions", 401, b"not json")
    assert [m[0] for m in mail] == ["OpenAI quota or billing problem", "OpenAI rejected our API key"]
    assert "secret request text" not in mail[0][1] and "insufficient_quota" in mail[0][1]
    for _ in range(3):
        alerts.report_openai("live captions", 503, None)
    assert mail[-1][0] == "OpenAI is returning server errors"
    alerts._5xx.clear()
    db.execute("UPDATE alert_state SET last_seen_at = ? WHERE kind LIKE 'openai:%'", (db.now() - 3600,))
    alerts.expire_stale("openai:", 1800)
    assert sum(m[0].startswith("Resolved") for m in mail) == 3


def test_web_unclean_stop_marker(mail, monkeypatch):
    monkeypatch.setattr(alerts.threading, "Thread", type("T", (), {"__init__": lambda s, target, **k: setattr(s, "t", target), "start": lambda s: s.t()}))
    assert alerts.web_started() is False  # first ever start
    alerts.web_stopping()
    assert alerts.web_started() is False  # clean stop
    assert alerts.web_started() is True  # previous run never recorded a stop
    assert [m[0] for m in mail] == ["acw-web is back up after an unclean stop"]


def test_unit_failed_rejects_odd_names(mail):
    assert alerts.unit_failed("../../etc/passwd") is False
    assert alerts.unit_failed("acw-web.service") is True
    assert mail[0][0] == "acw-web.service failed"


# --- monitor checks with fake data -----------------------------------------------------------------------------------

def _user():
    uid = db.new_id()
    db.execute("INSERT INTO users(id, email, created_at) VALUES (?,?,?)", (uid, f"{uid}@example.com", db.now()))
    return uid


def _job(uid, status, *, queued_ago=0, started_ago=None, progress=0, seconds=600.0):
    jid = db.new_id()
    now = db.now()
    params = {"options": {}, "denoise_tier": "studio", "transcribe_tier": None, "input_ext": ".wav"}
    db.execute(
        "INSERT INTO jobs(id, user_id, kind, tier, status, params_json, input_name, input_bytes, claimed_seconds,"
        " progress, created_at, queued_at, started_at, heartbeat_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (jid, uid, "denoise", "studio", status, json.dumps(params), "x.wav", 1, seconds, progress, now,
         now - queued_ago, None if started_ago is None else now - started_ago, now),
    )
    return jid


def test_queue_check(monkeypatch):
    uid = _user()
    ctx = lambda: "CPU: 16 cores, 9.0 busy\n  Everything else on the machine: 8.0 cores (busiest: java 7.5)"  # noqa: E731
    assert not monitor.check_queue(ctx)[0].bad  # empty queue
    _job(uid, "queued", queued_ago=60)
    assert not monitor.check_queue(ctx)[0].bad  # short wait
    head = _job(uid, "queued", queued_ago=21 * 60)
    f = monitor.check_queue(ctx)[0]
    assert f.bad and "java" in f.body and "21 minutes" in f.body
    db.execute("DELETE FROM jobs WHERE id = ?", (head,))
    # long estimated wait: a 20 h running job and a queued one behind it
    _job(uid, "running", started_ago=10, progress=1, seconds=20 * 3600)
    _job(uid, "queued", queued_ago=5 * 60)
    monkeypatch.setattr(jobs, "queue_snapshot", lambda: (["x"], {"x": 50 * 60}))
    assert monitor.check_queue(ctx)[0].bad


def test_slow_job_check():
    uid = _user()
    ctx = lambda: "ctx"  # noqa: E731
    _job(uid, "running", started_ago=15 * 60, progress=80, seconds=600)  # expect ~70 s, projected 19 min -> slow
    f = monitor.check_slow_jobs(ctx)[0]
    assert f.bad and "ctx" in f.body
    db.execute("DELETE FROM jobs")
    _job(uid, "running", started_ago=5 * 60, progress=10, seconds=600)  # under the 10 min floor
    assert not monitor.check_slow_jobs(ctx)[0].bad


def test_resource_checks(tmp_path, monkeypatch):
    now = 10_000
    gb = 1024**3
    ctx = lambda: "ctx"  # noqa: E731
    monkeypatch.setattr(monitor.shutil, "disk_usage", lambda p: type("U", (), {"free": 10 * gb})())
    mem = {"MemAvailable": 1 * gb, "MemTotal": 31 * gb, "SwapTotal": 8 * gb, "SwapFree": 6 * gb}
    f, hist = monitor.check_resources(tmp_path, mem, [[now - 1200, 1 * gb]], now, ctx)
    by = {x.kind: x for x in f}
    assert by["disk_low"].bad and by["mem_low"].bad and by["swap_growing"].bad  # 1 GB -> 2 GB in 20 min
    monkeypatch.setattr(monitor.shutil, "disk_usage", lambda p: type("U", (), {"free": 100 * gb})())
    ok = {"MemAvailable": 20 * gb, "MemTotal": 31 * gb, "SwapTotal": 8 * gb, "SwapFree": 6 * gb}
    f, _ = monitor.check_resources(tmp_path, ok, [[now - 1200, int(1.9 * gb)]], now, ctx)
    assert not any(x.bad for x in f)  # swap is used but not growing


def test_service_and_restart_checks():
    states = {
        "acw-web.service": {"ActiveState": "active", "SubState": "running", "Result": "success", "NRestarts": "0"},
        "acw-stream.service": {"ActiveState": "activating", "SubState": "auto-restart", "Result": "exit-code", "NRestarts": "3"},
    }
    prev = {"acw-stream.service": {"n": 1, "events": [900]}}
    f, hist = monitor.check_services(states, prev, 1000)
    by = {x.kind: x for x in f}
    assert not by["service:acw-web.service"].bad and by["service:acw-stream.service"].bad
    assert by["restarts:acw-stream.service"].bad and len(hist["acw-stream.service"]["events"]) == 3
    assert not by["restarts:acw-web.service"].bad
    f, _ = monitor.check_services(states, hist, 1000 + 3600)  # events age out of the window
    assert not {x.kind: x for x in f}["restarts:acw-stream.service"].bad


def test_health_check_says_which_path_failed(monkeypatch):
    monkeypatch.setenv("ACW_PUBLIC_ORIGIN", "https://example.org")
    monkeypatch.setenv("ACW_BASE_PATH", "/audio")
    get_settings.cache_clear()
    calls = []

    def fake(url):
        calls.append(url)
        return ("127.0.0.1" in url, "timeout")

    f = {x.kind: x for x in monitor.check_health(fake)}
    assert not f["health_local"].bad and f["health_public"].bad
    assert calls == ["http://127.0.0.1:8790/audio/api/health", "https://example.org/audio/api/health"]
    f = {x.kind: x for x in monitor.check_health(lambda u: (False, "refused"))}
    assert f["health_local"].bad and "health_public" not in f  # public is only judged when local works


def test_backup_check(tmp_path):
    assert monitor.check_backup(tmp_path / "missing")[0].bad
    snap = tmp_path / "acw-20261014-042251.sqlite3.gz"
    snap.write_bytes(b"x")
    assert not monitor.check_backup(tmp_path)[0].bad
    assert monitor.check_backup(tmp_path, now=time.time() + 37 * 3600)[0].bad


def test_run_end_to_end_mails_once_and_resolves(mail, monkeypatch, tmp_path):
    monkeypatch.setattr(monitor, "UNIT_DIR", tmp_path / "units")  # not this computer's units (the mirror, the bulk process)
    monkeypatch.setattr(monitor, "expected_units", lambda: [])
    monkeypatch.setattr(monitor, "check_backup", lambda: [])
    monkeypatch.setattr(monitor, "check_installation", lambda now: [])  # no models, workers or sockets in a test run
    gb = 1024**3
    state = {"free": 5 * gb}
    monkeypatch.setattr(monitor.shutil, "disk_usage", lambda p: type("U", (), {"free": state["free"]})())
    monkeypatch.setattr(monitor, "meminfo", lambda: {"MemAvailable": 20 * gb, "MemTotal": 31 * gb, "SwapTotal": 0, "SwapFree": 0})
    ok = lambda url: (True, "ok")  # noqa: E731
    monitor.run(ctx=lambda: "ctx", http=ok)
    monitor.run(ctx=lambda: "ctx", http=ok)
    assert [m[0] for m in mail] == ["Disk space is low (5.0 GB free)"]
    state["free"] = 100 * gb
    monitor.run(ctx=lambda: "ctx", http=ok)
    assert len(mail) == 1
    monitor.run(ctx=lambda: "ctx", http=ok)
    assert mail[-1][0].startswith("Resolved: Disk space is low")


def test_check_failure_does_not_stop_the_others(mail, monkeypatch):
    monkeypatch.setattr(monitor, "expected_units", lambda: (_ for _ in ()).throw(RuntimeError("systemctl missing")))
    monkeypatch.setattr(monitor, "check_backup", lambda: [monitor.Finding("backup_stale", False, "b")])
    findings = monitor.run(ctx=lambda: "ctx", http=lambda u: (True, "ok"))
    assert any(f.kind == "queue_long" for f in findings) and any(f.kind == "disk_low" for f in findings)


# --- the machine samples the monitor records for the dashboard -------------------------------------------------------

def _quiet_run(monkeypatch, tmp_path):
    """What test_run_end_to_end_mails_once_and_resolves sets up: no units, models or sockets of this computer."""
    monkeypatch.setattr(monitor, "UNIT_DIR", tmp_path / "units")
    monkeypatch.setattr(monitor, "expected_units", lambda: [])
    monkeypatch.setattr(monitor, "check_backup", lambda: [])
    monkeypatch.setattr(monitor, "check_installation", lambda now: [])
    gb = 1024**3
    monkeypatch.setattr(monitor.shutil, "disk_usage", lambda p: type("U", (), {"free": 100 * gb})())
    monkeypatch.setattr(monitor, "meminfo", lambda: {"MemAvailable": 20 * gb, "MemTotal": 31 * gb, "SwapTotal": 0, "SwapFree": 0})


def _snap():
    return {
        "cpu": {"cores": 8, "busy": 2.0}, "memory": {"available": 1, "sliceHeadroom": 2}, "disk": {"free": 3},
        "live": {"sessions": {"live": 1}, "engine": {"units": 1.0, "capacityUnits": 4.0}},
        "gpu": {"card": {"down": "none"}}, "jobs": {"running": [], "queued": 2},
    }


def _samples_in_db():
    import sqlite3

    from acw import samples

    if not samples.path().exists():
        return []
    c = sqlite3.connect(samples.path())
    try:
        return c.execute("SELECT ts, cpu_busy, jobs_queued FROM samples").fetchall()
    finally:
        c.close()


def test_run_records_one_sample_per_pass(mail, monkeypatch, tmp_path):
    from acw import broker

    _quiet_run(monkeypatch, tmp_path)
    monkeypatch.setattr(broker, "snapshot", _snap)
    findings = monitor.run(ctx=lambda: "ctx", http=lambda u: (True, "ok"))
    rows = _samples_in_db()
    assert len(rows) == 1 and rows[0][1:] == (2.0, 2)  # one row per pass: cpu busy and jobs queued
    assert not any("sample" in f.kind for f in findings)
    assert mail == []


@pytest.mark.parametrize("what", ["snapshot", "record"])
def test_a_failing_sample_never_becomes_a_finding_or_stops_the_other_checks(mail, monkeypatch, tmp_path, caplog, what):
    from acw import broker, samples

    _quiet_run(monkeypatch, tmp_path)
    monkeypatch.setattr(broker, "snapshot", _snap)

    def boom(*a, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(broker if what == "snapshot" else samples, what, boom)
    with caplog.at_level("ERROR"):
        findings = monitor.run(ctx=lambda: "ctx", http=lambda u: (True, "ok"))
    kinds = {f.kind for f in findings}
    assert not any("sample" in k for k in kinds)
    assert not any(f.bad for f in findings)
    assert {"queue_long", "disk_low"} <= kinds  # the checks around it still ran
    assert any("check samples failed" in r.getMessage() for r in caplog.records)
    assert mail == []
    assert db.one("SELECT 1 FROM alert_state WHERE kind LIKE '%sample%'") is None


# --- timer-started services: a clean run resolves the alert its failure opened ----------------------------------------

def test_check_timer_services():
    states = {
        "acw-mirror.service": {"ActiveState": "inactive", "SubState": "dead", "Result": "success"},
        "acw-backup.service": {"ActiveState": "failed", "SubState": "failed", "Result": "exit-code"},
        "acw-monitor.service": {"ActiveState": "activating", "SubState": "start", "Result": "success"},
    }
    f = monitor.check_timer_services(states)
    assert [x.kind for x in f] == ["service:acw-backup.service", "service:acw-mirror.service", "service:acw-monitor.service"]
    by = {x.kind: x for x in f}
    assert by["service:acw-backup.service"].bad
    assert not by["service:acw-mirror.service"].bad and not by["service:acw-monitor.service"].bad  # still running = fine
    bad = by["service:acw-backup.service"]
    assert bad.subject == "acw-backup.service failed" and "exit-code" in bad.body and "journalctl --user -u acw-backup.service" in bad.body
    assert (bad.confirm, bad.clear) == (1, 2)


@pytest.mark.parametrize("st,bad", [
    ({"ActiveState": "failed", "Result": "exit-code"}, True),
    ({"ActiveState": "failed", "Result": "success"}, True),  # failed state wins
    ({"ActiveState": "inactive", "Result": "oom-kill"}, True),
    ({"ActiveState": "inactive", "Result": "success"}, False),
    ({"ActiveState": "inactive"}, False),  # no Result at all: ok
    ({"ActiveState": "active", "Result": "success"}, False),
])
def test_check_timer_services_results(st, bad):
    [f] = monitor.check_timer_services({"acw-mirror.service": st})
    assert f.kind == "service:acw-mirror.service" and f.bad is bad


def _timer_units(monkeypatch, tmp_path, present, states):
    """UNIT_DIR with the unit files `present`; systemd answers from `states` (and records which units were asked)."""
    units = tmp_path / "units"
    units.mkdir(exist_ok=True)
    for u in present:
        (units / u).write_text("[Service]\n")
    asked = []

    def fake(names):
        asked.append(list(names))
        return {n: states[n] for n in names if n in states}

    monkeypatch.setattr(monitor, "unit_states", fake)
    return asked


def test_run_includes_timer_services_findings(mail, monkeypatch, tmp_path):
    _quiet_run(monkeypatch, tmp_path)
    asked = _timer_units(monkeypatch, tmp_path, ["acw-mirror.service", "acw-backup.service"], {
        "acw-mirror.service": {"ActiveState": "inactive", "Result": "success"},
        "acw-backup.service": {"ActiveState": "failed", "Result": "exit-code"},
        "acw-monitor.service": {"ActiveState": "failed", "Result": "exit-code"},
    })
    findings = {f.kind: f for f in monitor.run(ctx=lambda: "ctx", http=lambda u: (True, "ok"))}
    assert ["acw-mirror.service", "acw-backup.service"] in asked  # only units that are installed are asked about
    assert not any("acw-monitor.service" in a for a in asked if a)
    assert not findings["service:acw-mirror.service"].bad and findings["service:acw-backup.service"].bad
    assert "service:acw-monitor.service" not in findings
    assert mail[-1][0] == "acw-backup.service failed"


def test_timer_service_alert_resolves_after_two_clean_runs_in_a_row(mail, monkeypatch, tmp_path):
    """OnFailure runs `acw.alerts unit-failed UNIT`; the monitor then resolves that alert after two clean runs in a row."""
    _quiet_run(monkeypatch, tmp_path)
    monkeypatch.setattr(alerts.subprocess, "run", lambda *a, **kw: type("R", (), {"stdout": "ActiveState=failed"})())
    failed = {"ActiveState": "failed", "Result": "exit-code"}
    clean = {"ActiveState": "inactive", "Result": "success"}
    state = {"acw-mirror.service": failed}
    _timer_units(monkeypatch, tmp_path, ["acw-mirror.service"], state)
    row = lambda: db.one("SELECT active FROM alert_state WHERE kind = 'service:acw-mirror.service'")["active"]  # noqa: E731
    ok = lambda url: (True, "ok")  # noqa: E731

    assert alerts.main(["unit-failed", "acw-mirror.service"]) == 0
    assert [m[0] for m in mail] == ["acw-mirror.service failed"] and row() == 1

    seen = []
    for st in (failed, clean, failed, clean, clean, clean):  # still failed; one clean run; a failure resets the count;
        state["acw-mirror.service"] = st                       # one clean run; the second in a row resolves it; then quiet
        monitor.run(ctx=lambda: "ctx", http=ok)
        seen.append((len(mail), row()))
    assert seen == [(1, 1), (1, 1), (1, 1), (1, 1), (2, 0), (2, 0)]
    assert [m[0] for m in mail] == ["acw-mirror.service failed", "Resolved: acw-mirror.service failed"]
