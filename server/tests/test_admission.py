"""Queue admission from measured host load (acw.hostload): what a job waits for, and what it leaves free for the rest."""
import json
import time

import pytest

from acw import hostload, job_resources
from acw.worker.main import admission

GB = job_resources.GB


def snap(idle=12.0, busy_nice=0.0, mem=20 * GB, disk=100 * GB, cores=16):
    return hostload.Snapshot(cores=cores, busy_cores=cores - idle, nice_cores=busy_nice, mem_available=mem,
                             slice_headroom=None, disk_free=disk)


def job(kind="denoise", transcribe=None, seconds=3600, started=None):
    return {"kind": kind, "params_json": json.dumps({"transcribe_tier": transcribe}), "claimed_seconds": seconds,
            "duration_seconds": None, "next_stage": "probe", "started_at": started}


def test_a_job_starts_on_idle_cores_and_waits_for_cpu_only_when_others_run():
    ok, _, threads = admission(job(), snap(idle=12), [])
    assert ok and threads == 6
    assert admission(job(), snap(idle=0.5), [])[0]  # nothing running: the queue always moves
    ok, why, _ = admission(job(), snap(idle=0.5), [job(started=0)])
    assert not ok and why.startswith("cpu")


@pytest.mark.parametrize("j, s, why", [
    (job(transcribe="local"), snap(mem=5 * GB), "memory"),  # 1 + 3.2 GB + 6 GB reserve > 5 GB
    (job(seconds=4 * 3600), snap(disk=15 * GB), "disk"),  # 4 h stereo scratch ~11 GB + 10 GB reserve
    (job(), snap(mem=6.9 * GB), "memory"),  # the game servers reopen at 6 GiB available: the job must leave that free
])
def test_a_job_waits_for_memory_or_disk(j, s, why):
    ok, reason, _ = admission(j, s, [])
    assert not ok and reason.startswith(why)


def test_a_job_that_leaves_the_reserve_free_starts():
    assert admission(job(), snap(mem=7.1 * GB), [])[0]  # 1 GB job, 6.1 GB left


def test_the_slice_limit_keeps_its_own_smaller_reserve():
    need = job_resources.job_needs(job(transcribe="local"))[0]
    s = hostload.Snapshot(cores=16, busy_cores=4, nice_cores=0, mem_available=30 * GB, slice_headroom=need + job_resources.SLICE_RESERVE + 1,
                          disk_free=100 * GB)
    assert admission(job(transcribe="local"), s, [])[0]
    s = hostload.Snapshot(cores=16, busy_cores=4, nice_cores=0, mem_available=30 * GB, slice_headroom=need + job_resources.SLICE_RESERVE - 1,
                          disk_free=100 * GB)
    assert not admission(job(transcribe="local"), s, [])[0]


def test_jobs_started_a_moment_ago_count_against_memory_before_they_show_in_it():
    fresh = job(transcribe="local", started=int(time.time()))
    assert admission(job(), snap(mem=8 * GB), [])[0]
    assert not admission(job(), snap(mem=8 * GB), [fresh])[0]


def test_job_threads_follow_the_idle_cores_of_the_job_cpus():
    per = [1.0] * 4 + [0.0] * 12  # 4 of the 6 job CPUs are busy
    s = snap(idle=12)
    s = hostload.Snapshot(**{**s.__dict__, "busy_per_cpu": tuple(per)})
    ok, _, threads = admission(job(), s, [], cpus={0, 1, 2, 3, 4, 5})
    assert ok and threads == 2
