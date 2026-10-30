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


# --- the host capacity service (acw.capacity) decides memory, asked last ---------------------------------------------

from acw import capacity  # noqa: E402


@pytest.fixture
def asked(monkeypatch):
    """Patches capacity.reserve_job; `answer[0]` is what it replies, `calls` what it was asked."""
    calls, answer = [], [None]

    def reserve_job(job_id, memory):
        calls.append((job_id, memory))
        return answer[0]

    monkeypatch.setattr(capacity, "reserve_job", reserve_job)
    return calls, answer


def with_id(j, job_id="j1"):
    return {**j, "id": job_id}


def test_the_service_granting_memory_admits_even_when_the_local_rule_would_refuse(asked):
    calls, answer = asked
    answer[0] = (True, "")
    ok, why, threads = admission(with_id(job()), snap(mem=1 * GB), [])
    assert ok and why == "" and threads == 6
    assert len(calls) == 1


def test_the_service_refusing_passes_its_reason_through(asked):
    calls, answer = asked
    answer[0] = (False, "game servers hold 40 GB")
    assert admission(with_id(job()), snap(mem=100 * GB), []) == (False, "memory: game servers hold 40 GB", 0)


def test_no_answer_uses_the_local_reserve_both_ways(asked):
    calls, answer = asked
    answer[0] = None
    assert not admission(with_id(job()), snap(mem=6.9 * GB), [])[0]
    assert admission(with_id(job()), snap(mem=7.1 * GB), [])[0]
    assert len(calls) == 2


def test_the_local_rule_still_counts_recent_jobs_when_the_service_is_silent(asked):
    fresh = job(transcribe="local", started=int(time.time()))
    assert admission(with_id(job()), snap(mem=8 * GB), [])[0]
    ok, why, _ = admission(with_id(job()), snap(mem=8 * GB), [fresh])
    assert not ok and why.startswith("memory")


def test_the_service_is_asked_for_this_jobs_own_memory_only(asked):
    calls, answer = asked
    answer[0] = (True, "")
    fresh = job(transcribe="local", started=int(time.time()))  # a recent job: counts locally, not in the request
    assert admission(with_id(job(transcribe="local"), "abc"), snap(), [fresh])[0]
    assert calls == [("abc", job_resources.job_needs(job(transcribe="local"))[0])]


def test_a_finalize_stage_asks_for_its_small_need(asked):
    calls, answer = asked
    answer[0] = (True, "")
    j = {**job(), "next_stage": "finalize"}
    assert admission(with_id(j, "f"), snap(), [])[0]
    assert calls == [("f", job_resources.job_needs(j)[0])]


def test_a_job_without_an_id_never_asks(asked):
    calls, answer = asked
    answer[0] = (False, "never")
    assert admission(job(), snap(), [])[0]  # (the local rule decides)
    assert calls == []


@pytest.mark.parametrize("j, s, why", [
    (job(seconds=4 * 3600), snap(disk=15 * GB), "disk"),
    (job(), snap(idle=0.5), "cpu"),
    (job(), hostload.Snapshot(cores=16, busy_cores=4, nice_cores=0, mem_available=30 * GB, slice_headroom=1 * GB, disk_free=100 * GB), "memory"),
])
def test_a_job_refused_for_disk_cpu_or_the_slice_reserves_nothing(asked, j, s, why):
    calls, answer = asked
    answer[0] = (True, "")
    running = [job(started=0)]  # (the cpu check only applies while others run)
    ok, reason, _ = admission(with_id(j), s, running)
    assert not ok and reason.startswith(why)
    assert calls == []


def test_cpu_is_not_checked_with_nothing_running_so_the_service_is_asked(asked):
    calls, answer = asked
    answer[0] = (True, "")
    assert admission(with_id(job()), snap(idle=0.5), [])[0]
    assert len(calls) == 1


def test_the_gpu_path_asks_with_the_reduced_own_memory(asked, monkeypatch):
    from acw import broker

    calls, answer = asked
    answer[0] = (True, "")
    monkeypatch.setattr(broker, "card_room", lambda stages, usage: (True, ""))
    j = with_id(job(transcribe="local"), "g")
    assert admission(j, snap(), [], gpu=True)[0]
    assert calls == [("g", min(job_resources.job_needs(j)[0], job_resources.JOB_MEM + job_resources.GB))]
