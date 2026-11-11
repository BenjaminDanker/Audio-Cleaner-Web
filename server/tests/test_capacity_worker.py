"""Capacity audit: sweeping from two places at once (M1), OOM kills that must not loop (M3), stage time limits that follow
the progress made (M2)."""
import json
import subprocess
import sys
import textwrap
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from acw import db, job_resources, jobs
from acw.config import get_settings
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
    W.started_env = []
    monkeypatch.setattr(W.subprocess, "Popen", lambda *a, **kw: W.started_env.append(kw["env"]) or SimpleNamespace(
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
            assert W.started_env[0]["OPENBLAS_NUM_THREADS"] == W.started_env[0]["MKL_NUM_THREADS"] == "1"  # numpy's own pool
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


def _interrupted(W, jid, progress, next_stage="probe", run_id=None):
    """The job is claimed, its stage got as far as `progress` and the worker stops: (status, attempts, params) afterwards."""
    if run_id:
        jobs.update_params(jid, run={"id": run_id, "op": "process", "timeline": None})
    db.execute("UPDATE jobs SET next_stage = ?, duration_seconds = claimed_seconds WHERE id = ?", (next_stage, jid))
    claimed = W.claim("w")
    assert claimed["id"] == jid
    db.execute("UPDATE jobs SET progress = ? WHERE id = ?", (progress, jid))  # (what the stage had reported)
    W.process(claimed, {})
    row = db.one("SELECT status, attempts, params_json FROM jobs WHERE id = ?", (jid,))
    return row["status"], row["attempts"], json.loads(row["params_json"])


def test_the_interruption_count_only_goes_up_for_a_stop_at_the_same_place_and_starts_again_otherwise(monkeypatch):
    from acw.worker import main as W

    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    jid = make_job(uid)
    jobs.mark_uploaded(jid)

    def stopped(*a, **kw):
        raise W.Interrupted()

    monkeypatch.setattr(W, "run_stage", stopped)
    monkeypatch.setattr(W, "MAX_INTERRUPTIONS", 3)

    def stop(*place):
        status, attempts, params = _interrupted(W, jid, *place)
        assert (status, attempts) == ("queued", 0)
        return params["interruptions"], params["interrupted_at"]

    assert stop(40) == (1, [None, "probe", 40])
    assert stop(40) == (2, [None, "probe", 40])  # the same place again: counted
    assert stop(60) == (1, [None, "probe", 60])  # further than before: the count starts again
    assert stop(60, "main") == (1, [None, "main", 60])  # another stage
    assert stop(60, "main", "r1") == (1, ["r1", "main", 60])  # another run of the job
    assert stop(60, "main") == (2, ["r1", "main", 60])  # the same run, stage and place
    _, _, params = _interrupted(W, jid, 20, "main")  # not further than the one before: the third in a row, given up
    assert "run" not in params and params["edit"]["lastRun"]["error"] == W.INTERRUPTED_MESSAGE[:300]  # (a run: the job is as it was)
    assert db.one("SELECT status FROM jobs WHERE id = ?", (jid,))["status"] == "done"


def test_stops_a_little_earlier_and_a_little_later_in_turn_still_add_up_because_the_furthest_stop_is_kept(monkeypatch):
    from acw.worker import main as W

    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    jid = make_job(uid)
    jobs.mark_uploaded(jid)

    def stopped(*a, **kw):
        raise W.Interrupted()

    monkeypatch.setattr(W, "run_stage", stopped)
    monkeypatch.setattr(W, "MAX_INTERRUPTIONS", 3)  # (the default limit)

    def stop(progress):
        status, attempts, params = _interrupted(W, jid, progress)
        assert (status, attempts) == ("queued", 0)
        return params["interruptions"], params["interrupted_at"]

    assert stop(40) == (1, [None, "probe", 40])
    assert stop(41) == (1, [None, "probe", 41])  # a little further: the count starts again
    assert stop(40) == (2, [None, "probe", 41])  # a little earlier: counted, and the furthest stop stays the place
    status, _, params = _interrupted(W, jid, 41)  # the same place as the furthest: the third in a row, given up
    assert status == "failed"
    assert db.one("SELECT error FROM jobs WHERE id = ?", (jid,))["error"] == W.INTERRUPTED_MESSAGE


@pytest.mark.parametrize("stage", ["main", "edit"])
def test_a_job_that_waited_too_long_is_stopped_before_its_stage_and_one_with_room_or_a_restart_behind_it_is_not(monkeypatch, stage):
    from acw.worker import main as W

    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    ran = []

    def stage_runs(job, d, name, timeout):
        ran.append(name)
        raise W.StageFailed("the stage ran")

    monkeypatch.setattr(W, "run_stage", stage_runs)
    cap = get_settings().job_max_retention_seconds
    # (what the job's work and the delivery need before its files go, and how much is there: -10 s is too little, 600 s is plenty)
    for room, interruptions, runs in ((-10, 0, False), (600, 0, True), (-10, 1, True)):
        jid = make_job(uid)
        jobs.mark_uploaded(jid)
        if stage == "edit":  # an edit run of a finished recording
            jobs.update_params(jid, run={"id": "r1", "op": "process", "timeline": None, "est_seconds": 100})
        if interruptions:
            jobs.update_params(jid, interruptions=interruptions)  # it has part of the work behind it: not judged by the whole
        db.execute("UPDATE jobs SET next_stage = ?, duration_seconds = claimed_seconds WHERE id = ?", (stage, jid))
        needed = job_resources.job_seconds(dict(db.one("SELECT * FROM jobs WHERE id = ?", (jid,)))) + jobs.MIN_DELIVERY_SECONDS
        db.execute("UPDATE jobs SET uploaded_at = ? WHERE id = ?", (db.now() + needed + room - cap, jid))
        ran.clear()
        W.process(W.claim("w"), {})
        row = db.one("SELECT status, error, params_json FROM jobs WHERE id = ?", (jid,))
        assert (ran == [stage]) == runs, (room, interruptions, ran)
        params = json.loads(row["params_json"])
        error = params["edit"]["lastRun"]["error"] if stage == "edit" else row["error"]  # (an edit run that ends leaves the job done)
        assert error == ("the stage ran" if runs else jobs.TOO_LATE[:300])
        assert row["status"] == ("done" if stage == "edit" else "failed") and ("run" not in params or stage != "edit")
    assert ledger.balance(uid)["held_micros"] == 0


def test_a_busy_database_does_not_stop_a_stage_from_keeping_its_result(monkeypatch, caplog):
    import sqlite3

    from acw.worker import main as W

    uid = user()
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    job, d = running_job(uid)
    (d / "spec.json").write_text("{}")
    lines = [json.dumps({"progress": 50, "stage": "Denoising"}).encode(), json.dumps({"progress": 100}).encode()]

    def runner(*a, **kw):  # (the runner writes its result file as it ends)
        (d / "result-main.json").write_text(json.dumps({"ok": True, "outputs": []}))
        return SimpleNamespace(stdout=iter(lines), stderr=iter(()), poll=lambda: 0, wait=lambda: 0, returncode=0, pid=1)

    writes = []

    def busy(sql, params=()):
        writes.append(sql)
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(W.subprocess, "Popen", runner)
    monkeypatch.setattr(W.db, "execute", busy)
    with caplog.at_level("WARNING", logger="acw.worker"):
        assert W.run_stage(job, d, "main", 10)["ok"] is True  # its progress could not be written, its result is kept
    assert len(writes) == 2 and caplog.text.count("could not write its progress") == 2


def test_a_worker_whose_claim_meets_a_busy_database_asks_again(monkeypatch):
    import sqlite3

    from acw.worker import main as W

    asked = []

    class Stop:  # (a stop flag that does not wait)
        def is_set(self):
            return len(asked) >= 2

        def wait(self, seconds):
            return None

    def claim(worker_id, load=None):
        asked.append(1)
        if len(asked) == 1:
            raise sqlite3.OperationalError("database is locked")
        return None

    monkeypatch.setattr(W, "_stop", Stop())
    monkeypatch.setattr(W, "claim", claim)
    monkeypatch.setattr(W.signal, "signal", lambda *a: None)
    monkeypatch.setattr(W.hostload, "get", lambda d: None)
    monkeypatch.setattr(W.jobs, "sweep", lambda: None)
    W.main()  # (it returns when the flag says stop; an error from claim would have ended it with the exception)
    assert len(asked) == 2


# ---------------------------------------------------------------- M2: limits that follow the progress

def test_a_stage_gets_time_while_it_progresses_up_to_a_ceiling_and_not_when_stuck_or_finished():
    base = 23_400.0  # a 90 minute voice rebuild
    # 70 percent done after the whole limit: it needs about 10,000 s more, up to the ceiling
    new = job_resources.stretched_limit(base, base, 5, 70, 60)
    assert new is not None and base + 9_000 < new <= base * job_resources.STRETCH_LIMIT
    assert job_resources.stretched_limit(base, base, 5, 99, 60) < base * 1.3  # almost finished: a little more is enough
    assert job_resources.stretched_limit(base, base, 5, 6, 60) == base * job_resources.STRETCH_LIMIT  # however slowly: never beyond the ceiling
    assert job_resources.stretched_limit(base, base * 3, 5, 6, 60) is None  # and once there, it is over
    base = 10_000.0
    assert job_resources.stretched_limit(base, base, 5, 60, job_resources.STALL_SECONDS + 1) is None  # stopped moving: stuck
    assert job_resources.stretched_limit(base, base, 5, 5, 0) is None  # no progress at all
    assert job_resources.stretched_limit(base, base, 5, 100, 0) is None  # finished (its result is what is awaited)


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

    monkeypatch.setattr(job_resources, "MIN_STRETCH_SECONDS", 0.5)  # seconds-long stages in this test
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

    monkeypatch.setattr(job_resources, "STALL_SECONDS", 0.5)
    with pytest.raises(W.StageFailed) as e:
        _run_main(monkeypatch, {"steps": [[0.1, 5], [0.5, 30]], "finish": 30}, 2)
    assert "took too long" in e.value.user_message


def test_the_ceiling_still_stops_a_stage_that_never_finishes(monkeypatch):
    from acw.worker import main as W

    with pytest.raises(W.StageFailed) as e:  # keeps reporting tiny steps forever: stopped at 3 x the limit
        _run_main(monkeypatch, {"steps": [[0.1 + i * 0.3, 5 + i] for i in range(40)], "finish": 60}, 1)
    assert "took too long" in e.value.user_message


def test_a_job_that_transcribes_and_cleans_decodes_the_file_once(tmp_path, monkeypatch):
    """The 48 kHz decode made for cleaning also gives the recogniser its 16 kHz audio (no second decode of the file)."""
    from acw import edit_subtitles, editing, jobs
    from acw.worker import editrun, runner
    from acw_common import pipeline
    from acw_common.timeline import Timeline

    class Stop(Exception):
        pass

    def stop(*a, **k):
        raise Stop

    calls = []
    x = np.zeros((48_000, 1), np.float32)
    ff = SimpleNamespace(probe=lambda src: SimpleNamespace(audio_tracks=1, duration=1.0), decode=lambda *a, **k: calls.append("decode 16 kHz"))
    monkeypatch.setattr(runner, "_ff", lambda spec: ff)
    monkeypatch.setattr(pipeline, "decode_for_clean", lambda *a, **k: calls.append("decode 48 kHz") or x)
    monkeypatch.setattr(pipeline, "to_speech_16k", lambda y: calls.append("to 16 kHz") or y[:, 0])
    monkeypatch.setattr(runner, "_write_asr_chunks", lambda *a, **k: calls.append("transcribe"))
    monkeypatch.setattr(editrun, "render", stop)  # the cleaning starts: enough
    options = jobs.validate_options("denoise_transcribe", "standard", "premium", {})
    raw, _rest = editing.timeline_from_options("denoise_transcribe", "standard", "premium", options, 1.0)
    spec = {"kind": "denoise_transcribe", "transcribe_tier": "premium", "options": {}, "output_stem": "t", "input": "t.wav",
            "max_seconds": 600, "models_dir": str(tmp_path / "models"),
            "editor": {"timeline": raw, "source_id": "s", "subtitles": edit_subtitles.subtitle_spec(Timeline.from_json(raw, seconds=1.0))}}
    with pytest.raises(Stop):
        runner.stage_main(tmp_path, spec)
    assert calls == ["decode 48 kHz", "to 16 kHz", "transcribe"]
