"""Capacity audit: sweeping from two places at once (M1), OOM kills that must not loop (M3), stage time limits that follow
the progress made (M2)."""
import json
import subprocess
import sys
import textwrap
import threading
from types import SimpleNamespace

import pytest

from acw import db, jobs
from acw.billing import ledger
from tests.test_billing_audit import make_job, running_job, user


# ---------------------------------------------------------------- M1: sweeps from the web and the worker at once

def test_sweeping_from_several_threads_at_once_is_safe():
    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    jid = make_job(uid)
    hold = db.one("SELECT hold_id FROM jobs WHERE id = ?", (jid,))["hold_id"]
    db.execute("UPDATE jobs SET expires_at = ? WHERE id = ?", (db.now() - 1, jid))  # an upload nobody finished
    stale = make_job(uid)
    db.execute("UPDATE jobs SET status = 'running', attempts = 1, heartbeat_at = ? WHERE id = ?", (db.now() - 500, stale))
    errors = []

    def go():
        try:
            jobs.sweep()
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=go) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert db.one("SELECT status FROM jobs WHERE id = ?", (jid,))["status"] == "canceled"
    assert db.one("SELECT status FROM holds WHERE id = ?", (hold,))["status"] != "active"  # released once, not twice
    assert db.one("SELECT status, attempts FROM jobs WHERE id = ?", (stale,))["status"] == "queued"
    assert ledger.balance(uid)["held_micros"] > 0  # the stale job's hold is still reserved: it will run again


# ---------------------------------------------------------------- M3: OOM kill and interruption loops

def _stage_with(monkeypatch, d, returncode, oom):
    """run_stage with a runner that dies without a result (returncode) and a kernel OOM counter that reads `oom`."""
    from acw.worker import main as W

    (d / "spec.json").write_text("{}")
    monkeypatch.setattr(W, "oom_kills", lambda it=iter(oom): next(it))
    monkeypatch.setattr(W.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(
        stdout=iter(()), stderr=iter(()), poll=lambda: 0, wait=lambda: 0, returncode=returncode, pid=1))
    return W


@pytest.mark.parametrize("returncode, oom, outcome", [
    (-9, [3, 4], "oom"),  # killed while the OOM counter moved: a plain memory failure, not charged
    (-9, [3, 3], "interrupted"),  # killed with no OOM kill: an interruption, the job goes back in the queue
    (1, [3, 9], "failed"),  # exited with an error, not killed: a plain failure
])
def test_a_runner_that_dies_is_an_oom_failure_an_interruption_or_a_plain_failure(monkeypatch, returncode, oom, outcome):
    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    job, d = running_job(uid)
    W = _stage_with(monkeypatch, d, returncode, oom)
    if outcome != "failed":
        W._stop.set()  # systemd is stopping the worker: a kill then must not read as a deploy unless the OOM counter moved
    try:
        if outcome == "interrupted":
            with pytest.raises(W.Interrupted):
                W.run_stage(job, d, "main", 10)
        else:
            with pytest.raises(W.StageFailed) as e:
                W.run_stage(job, d, "main", 10)
            assert (e.value.user_message == W.OOM_MESSAGE) == (outcome == "oom")
            if outcome == "oom":
                assert "not charged" in W.OOM_MESSAGE and "memory" in W.OOM_MESSAGE
    finally:
        W._stop.clear()


def test_oom_kills_reads_the_cgroup_counter(monkeypatch, tmp_path):
    from acw.worker import main as W

    cg = tmp_path / "sys" / "user.slice" / "w.service"
    cg.mkdir(parents=True)
    (cg / "memory.events").write_text("low 0\nhigh 0\nmax 12\noom 2\noom_kill 2\n")
    real_read = W.Path.read_text

    def fake_read(self, *a, **kw):
        if str(self) == "/proc/self/cgroup":
            return "0::/user.slice/w.service\n"
        return real_read(self, *a, **kw)

    monkeypatch.setattr(W.Path, "read_text", fake_read)
    monkeypatch.setattr(W, "Path", lambda *p: tmp_path / "sys" if p == ("/sys/fs/cgroup",) else __import__("pathlib").Path(*p))
    assert W.oom_kills() == 2


def test_a_job_that_keeps_getting_interrupted_is_failed_not_requeued_forever(monkeypatch):
    from acw.worker import main as W

    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    jid = make_job(uid)
    jobs.mark_uploaded(jid)

    def stopped(*a, **kw):
        raise W.Interrupted()

    monkeypatch.setattr(W, "run_stage", stopped)
    monkeypatch.setattr(W, "MAX_INTERRUPTIONS", 3)
    for n in (1, 2):  # a deploy or two: the job goes back, no attempt used
        W.process(W.claim("w"), {})
        row = db.one("SELECT status, attempts, params_json FROM jobs WHERE id = ?", (jid,))
        assert (row["status"], row["attempts"], json.loads(row["params_json"])["interruptions"]) == ("queued", 0, n)
    W.process(W.claim("w"), {})  # the third time it is given up
    row = db.one("SELECT status, error, hold_id FROM jobs WHERE id = ?", (jid,))
    assert row["status"] == "failed" and row["error"] == W.INTERRUPTED_MESSAGE and "not charged" in row["error"]
    assert ledger.balance(uid)["held_micros"] == 0


# ---------------------------------------------------------------- M2: limits that follow the progress

def test_a_stage_gets_time_while_it_progresses_up_to_a_ceiling_and_not_when_stuck_or_finished():
    base = 23_400.0  # a 90 minute voice rebuild
    # 70 percent done after the whole limit: it needs about 10,000 s more, up to the ceiling
    new = jobs.stretched_limit(base, base, 5, 70, 60)
    assert new is not None and base + 9_000 < new <= base * jobs.STRETCH_LIMIT
    assert jobs.stretched_limit(base, base, 5, 99, 60) < base * 1.3  # almost finished: a little more is enough
    assert jobs.stretched_limit(base, base, 5, 6, 60) == base * jobs.STRETCH_LIMIT  # however slowly: never beyond the ceiling
    assert jobs.stretched_limit(base, base * 3, 5, 6, 60) is None  # and once there, it is over
    base = 10_000.0
    assert jobs.stretched_limit(base, base, 5, 60, jobs.STALL_SECONDS + 1) is None  # stopped moving: stuck
    assert jobs.stretched_limit(base, base, 5, 5, 0) is None  # no progress at all
    assert jobs.stretched_limit(base, base, 5, 100, 0) is None  # finished (its result is what is awaited)


FAKE_RUNNER = textwrap.dedent("""
    import json, sys, time
    d, stage = sys.argv[1], sys.argv[2]
    plan = json.loads(sys.argv[3])
    t0 = time.time()
    for at, progress in plan["steps"]:
        time.sleep(max(0, at - (time.time() - t0)))
        print(json.dumps({"stage": "Cleaning", "progress": progress}), flush=True)
    time.sleep(max(0, plan["finish"] - (time.time() - t0)))
    open(d + "/result-" + stage + ".json", "w").write(json.dumps({"ok": True, "outputs": []}))
""")


def _run_main(monkeypatch, plan, timeout):
    """run_stage('main') against a stand-in runner that reports progress on a schedule."""
    from acw.worker import main as W

    monkeypatch.setattr(jobs, "MIN_STRETCH_SECONDS", 0.5)  # seconds-long stages in this test
    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    job, d = running_job(uid)
    (d / "spec.json").write_text("{}")
    real_popen = subprocess.Popen
    monkeypatch.setattr(W.subprocess, "Popen", lambda args, **kw: real_popen(
        [sys.executable, "-c", FAKE_RUNNER, str(d), args[-1], json.dumps(plan)], **kw))
    return W.run_stage(job, d, "main", timeout)


def test_a_main_stage_that_runs_past_its_limit_but_keeps_progressing_is_not_killed(monkeypatch):
    # limit 2 s, finishes at 3.5 s, progress moves all along: stretched up to 3 x 2 s
    res = _run_main(monkeypatch, {"steps": [[0.1, 5], [1.0, 30], [2.0, 55], [3.0, 90]], "finish": 3.5}, 2)
    assert res["ok"]


def test_a_main_stage_whose_progress_stalls_is_killed_at_the_limit(monkeypatch):
    from acw.worker import main as W

    monkeypatch.setattr(jobs, "STALL_SECONDS", 0.5)
    with pytest.raises(W.StageFailed) as e:
        _run_main(monkeypatch, {"steps": [[0.1, 5], [0.5, 30]], "finish": 30}, 2)
    assert "took too long" in e.value.user_message


def test_the_ceiling_still_stops_a_stage_that_never_finishes(monkeypatch):
    from acw.worker import main as W

    with pytest.raises(W.StageFailed) as e:  # keeps reporting tiny steps forever: stopped at 3 x the limit
        _run_main(monkeypatch, {"steps": [[0.1 + i * 0.3, 5 + i] for i in range(40)], "finish": 60}, 1)
    assert "took too long" in e.value.user_message
