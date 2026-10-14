"""Queue admission decisions from measured host load (acw.hostload)."""
import json

from acw import hostload, jobs
from acw.worker.main import admission

GB = jobs.GB


def snap(idle=12.0, busy_nice=0.0, mem=20 * GB, disk=100 * GB, cores=16):
    return hostload.Snapshot(cores=cores, busy_cores=cores - idle, nice_cores=busy_nice, mem_available=mem,
                             slice_headroom=None, disk_free=disk)


def job(kind="denoise", transcribe=None, seconds=3600, started=None):
    return {"kind": kind, "params_json": json.dumps({"transcribe_tier": transcribe}), "claimed_seconds": seconds,
            "duration_seconds": None, "next_stage": "probe", "started_at": started}


def test_starts_with_threads_from_idle_cores():
    ok, _, threads = admission(job(), snap(idle=12), [])
    assert ok and threads == 6
    ok, _, threads = admission(job(), snap(idle=3), [])
    assert ok and threads == 2


def test_cpu_only_matters_when_other_jobs_run():
    assert admission(job(), snap(idle=0.5), [])[0]  # nothing running: the queue always moves
    ok, why, _ = admission(job(), snap(idle=0.5), [job(started=0)])
    assert not ok and why.startswith("cpu")


def test_waits_for_memory_and_disk():
    ok, why, _ = admission(job(transcribe="local"), snap(mem=5 * GB), [])
    assert not ok and why.startswith("memory")  # 1.5 + 2.5 GB + 2 GB reserve > 5 GB
    ok, why, _ = admission(job(seconds=4 * 3600), snap(disk=15 * GB), [])
    assert not ok and why.startswith("disk")  # 4 h stereo scratch ~11 GB + 10 GB reserve


def test_recently_started_jobs_count_against_memory():
    import time

    fresh = job(transcribe="local", started=int(time.time()))
    assert admission(job(), snap(mem=7 * GB), [])[0]
    assert not admission(job(), snap(mem=7 * GB), [fresh])[0]


def test_live_counts_niced_work_as_available():
    s = snap(idle=1.0, busy_nice=10.0)  # file jobs (niced) keep the box busy
    assert s.cores_for_live == 11.0 and s.idle_cores == 1.0


def test_live_refuses_with_what_still_fits(monkeypatch):
    from acw.stream import worker

    monkeypatch.setattr(worker, "_monitor", None)
    monkeypatch.setattr(worker, "LIVE_UNITS", 13.0)
    monkeypatch.setattr(worker, "_units", 11.0)
    monkeypatch.setattr(worker, "_local_captions", 3)
    assert worker.admit({"denoise": "light"}) == (True, [])
    ok, options = worker.admit({"denoise": "low_latency"})
    assert not ok and {"denoise": "light"} in options and {"captions": "premium"} in options
    ok, options = worker.admit({"denoise": None, "captions": "local"})
    assert not ok and options == [{"captions": "premium"}]
    # External captions only: no model runs here, never limited.
    monkeypatch.setattr(worker, "_units", 100.0)
    assert worker.admit({"denoise": None, "captions": "premium"}) == (True, [])


def test_live_capacity_shrinks_with_other_load_on_live_cores(monkeypatch):
    from acw import cpusplit
    from acw.stream import worker

    live = sorted(cpusplit.live_cpus())
    per = [0.0] * 16
    for c in live[: len(live) // 2]:
        per[c] = 1.0  # e.g. a game server busy on half the live CPUs

    class Monitor:
        def snapshot(self):
            return hostload.Snapshot(cores=16, busy_cores=8, nice_cores=0, mem_available=20 * GB, slice_headroom=None,
                                     disk_free=100 * GB, busy_per_cpu=tuple(per), own_cores=0.0)

    monkeypatch.setattr(worker, "_monitor", Monitor())
    assert worker.capacity_units() < worker.LIVE_UNITS * 0.5


def test_cpu_split_partitions_physical_cores():
    from acw import cpusplit

    live, jobs_ = cpusplit.live_cpus(), cpusplit.job_cpus()
    assert live and jobs_ and not (live & jobs_)
    assert live | jobs_ == cpusplit.all_cpus()
    for core in cpusplit.physical_cores():  # hyper-thread siblings stay together
        assert set(core) <= live or set(core) <= jobs_


def test_live_marker_round_trip(tmp_path):
    from acw import cpusplit

    assert not cpusplit.live_active(tmp_path)
    cpusplit.set_live_sessions(tmp_path, 2)
    assert cpusplit.live_active(tmp_path)
    cpusplit.set_live_sessions(tmp_path, 0)
    assert not cpusplit.live_active(tmp_path)
