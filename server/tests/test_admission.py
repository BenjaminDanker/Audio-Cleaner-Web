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


def test_job_threads_follow_idle_job_cores():
    per = [1.0] * 4 + [0.0] * 12
    s = snap(idle=12)
    s = hostload.Snapshot(**{**s.__dict__, "busy_per_cpu": tuple(per)})
    ok, _, threads = admission(job(), s, [], cpus={0, 1, 2, 3, 4, 5})  # 4 of 6 job CPUs busy
    assert ok and threads == 2
