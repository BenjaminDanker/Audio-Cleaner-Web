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
    assert not ok and why.startswith("memory")  # 1 + 3.2 GB + 6 GB reserve > 5 GB
    ok, why, _ = admission(job(seconds=4 * 3600), snap(disk=15 * GB), [])
    assert not ok and why.startswith("disk")  # 4 h stereo scratch ~11 GB + 10 GB reserve


def test_a_job_leaves_the_minecraft_reopen_threshold_free():
    # the memory controller of the game servers reopens at 6 GiB available: a job must not take the host below it
    assert jobs.MEM_RESERVE == 6 * GB
    assert admission(job(), snap(mem=7.1 * GB), [])[0]  # 1 GB job, 6.1 GB left
    ok, why, _ = admission(job(), snap(mem=6.9 * GB), [])
    assert not ok and why.startswith("memory")


def test_the_slice_limit_keeps_its_own_smaller_reserve():
    need = jobs.job_needs(job(transcribe="local"))[0]
    s = hostload.Snapshot(cores=16, busy_cores=4, nice_cores=0, mem_available=30 * GB, slice_headroom=need + jobs.SLICE_RESERVE + 1,
                          disk_free=100 * GB)
    assert admission(job(transcribe="local"), s, [])[0]  # the slice keeps 2 GB, the host side has 30 GB
    s = hostload.Snapshot(cores=16, busy_cores=4, nice_cores=0, mem_available=30 * GB, slice_headroom=need + jobs.SLICE_RESERVE - 1,
                          disk_free=100 * GB)
    assert not admission(job(transcribe="local"), s, [])[0]


def test_recently_started_jobs_count_against_memory():
    import time

    fresh = job(transcribe="local", started=int(time.time()))
    assert admission(job(), snap(mem=8 * GB), [])[0]
    assert not admission(job(), snap(mem=8 * GB), [fresh])[0]


def test_live_counts_niced_work_as_available():
    s = snap(idle=1.0, busy_nice=10.0)  # file jobs (niced) keep the box busy
    assert s.cores_for_live == 11.0 and s.idle_cores == 1.0


def test_live_refuses_with_what_still_fits(monkeypatch):
    from acw import cpusplit
    from acw.stream import worker

    monkeypatch.setattr(worker, "_monitor", None)
    monkeypatch.setattr(cpusplit, "units_per_core", lambda: 2.4)
    cap = worker.capacity_units()
    assert cap == cpusplit.max_live_cores() * 2.4
    monkeypatch.setattr(worker, "_units", cap - 1.5)
    assert worker.admit({"denoise": "light"}) == (True, [])
    monkeypatch.setattr(worker, "_units", cap - 0.5)
    ok, options = worker.admit({"denoise": "light"})
    assert not ok and options == [{"denoise": None, "captions": "premium"}]
    monkeypatch.setattr(worker, "_units", cap - 1.5)
    ok, options = worker.admit({"denoise": "light", "captions": "local"})
    assert not ok and {"captions": "premium"} in options  # same denoise, external captions fit
    # External captions only: no model runs here, never limited.
    monkeypatch.setattr(worker, "_units", 1000.0)
    assert worker.admit({"denoise": None, "captions": "premium"}) == (True, [])


def test_live_capacity_shrinks_with_other_load_on_live_cores(monkeypatch):
    from acw import cpusplit
    from acw.stream import worker

    monkeypatch.setattr(cpusplit, "units_per_core", lambda: 2.4)
    live = sorted(cpusplit.live_cpus())
    per = [0.0] * (max(cpusplit.all_cpus()) + 1)
    for c in live[: len(live) // 2]:
        per[c] = 1.0  # e.g. a game server busy on half the live CPUs

    class Monitor:
        def snapshot(self):
            return hostload.Snapshot(cores=len(per), busy_cores=8, nice_cores=0, mem_available=20 * GB,
                                     slice_headroom=None, disk_free=100 * GB, busy_per_cpu=tuple(per), own_cores=0.0)

    monkeypatch.setattr(worker, "_monitor", Monitor())
    assert worker.capacity_units() < cpusplit.max_live_cores() * 2.4 * 0.5


def test_live_cores_grow_with_load_and_jobs_keep_the_rest(monkeypatch):
    from acw import cpusplit

    # an 8-core / 16-thread CPU, whatever machine runs the test (CI runners have 2 cores)
    monkeypatch.setattr(cpusplit, "physical_cores", lambda: [[i, i + 8] for i in range(8)])
    phys = len(cpusplit.physical_cores())
    assert cpusplit.live_cores_for(0, 2.4) == 0
    assert cpusplit.live_cores_for(1, 2.4) == 2  # one needed + one spare
    assert cpusplit.live_cores_for(1000, 2.4) == phys - cpusplit.min_job_cores()
    for units in (0.5, 5, 12, 1000):
        live, jobs_ = cpusplit.live_cpus(units, 2.4), cpusplit.job_cpus(units, 2.4)
        assert live and jobs_ and not (live & jobs_) and live | jobs_ == cpusplit.all_cpus()
        for core in cpusplit.physical_cores():  # hyper-thread siblings stay together
            assert set(core) <= live or set(core) <= jobs_


def test_live_marker_round_trip(tmp_path):
    from acw import cpusplit

    assert cpusplit.live_units(tmp_path) == (0.0, cpusplit.REFERENCE_UNITS_PER_CORE)
    cpusplit.set_live_units(tmp_path, 4.0, 2.1)
    assert cpusplit.live_units(tmp_path) == (4.0, 2.1)


def test_job_threads_follow_idle_job_cores():
    per = [1.0] * 4 + [0.0] * 12
    s = snap(idle=12)
    s = hostload.Snapshot(**{**s.__dict__, "busy_per_cpu": tuple(per)})
    ok, _, threads = admission(job(), s, [], cpus={0, 1, 2, 3, 4, 5})  # 4 of 6 job CPUs busy
    assert ok and threads == 2
