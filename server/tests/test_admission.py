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


def test_live_waiting_line(monkeypatch):
    from acw.stream import worker

    class Monitor:
        def __init__(self, s):
            self.s = s

        def snapshot(self):
            return self.s

    monkeypatch.setattr(worker, "_recent", [])
    monkeypatch.setattr(worker, "_monitor", Monitor(snap(idle=2.1)))
    assert not worker._fits({"denoise": "light"})  # 2 cores reserved for the host
    worker._monitor.s = snap(idle=2.5)
    assert worker._fits({"denoise": "light"})
    worker._recent.append((__import__("time").monotonic(), 0.4))
    assert not worker._fits({"denoise": "light"})  # just admitted, load not visible yet
