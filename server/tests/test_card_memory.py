"""The card's memory (broker.py): the GPU worker measures what other programs hold of it (the owner's model training shares the card)
and writes it every few seconds; a card job is admitted only when its stages fit (GPU_WORK card_mb plus a margin), waiting for room is
not idle (so the CPU workers take what they can), and the voice rebuild can only be ordered while it would fit.

Nothing here asks the real card (it may be in use): `broker._nvidia` is stubbed (the conftest makes it fail, "no card", for every test) and
the owner of a process is read from /proc for real pids only."""
import json
import os
import subprocess
import time
from types import SimpleNamespace

import pytest

from acw import broker, db, jobs
from acw.hostload import Snapshot
from acw.worker import main as worker

from .job_support import params_of, row, set_card
from .test_editor_api import edit_job, tl_of
from .test_timeline_jobs import cancel, funded, start, whole

TOTAL = 4096
SMALL = 250  # encode
HEAVY, REBUILD, TRANSCRIBE = 360, 2600, 3300  # GPU_WORK card_mb


def fake_nvidia(monkeypatch, total=TOTAL, used=0, apps=(), calls=None):
    """The answers of nvidia-smi: the card's total and used memory, and the compute programs (pid, MB)."""
    def nvidia(query, kind="gpu"):
        if calls is not None:
            calls.append((query, kind))
        if (query, kind) == ("memory.total,memory.used", "gpu"):
            return [[str(total), str(used)]]
        assert (query, kind) == ("pid,used_memory", "compute-apps"), (query, kind)
        return [[str(pid), str(mb)] for pid, mb in apps]

    monkeypatch.setattr(broker, "_nvidia", nvidia)


# ---------------------------------------------------------------------------------------------------------------- reading the card

def test_nvidia_smi_is_asked_for_the_totals_and_the_compute_programs(monkeypatch):
    monkeypatch.setattr(broker, "_nvidia", broker._nvidia.real)
    asked = []

    def run(cmd, **kw):
        asked.append((cmd, kw["timeout"], kw["check"]))
        return SimpleNamespace(stdout="4096, 1500\n" if "--query-gpu=memory.total,memory.used" in cmd[1] else " 123 , 800 \n456, 20\n\n")

    monkeypatch.setattr(broker.subprocess, "run", run)
    assert broker._nvidia("memory.total,memory.used") == [["4096", "1500"]]
    assert broker._nvidia("pid,used_memory", "compute-apps") == [["123", "800"], ["456", "20"]]
    assert [a[0] for a in asked] == [["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                                     ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]]
    assert all(timeout == 5 and check is True for _, timeout, check in asked)


@pytest.mark.skipif(os.getuid() == 0 or os.stat("/proc/1").st_uid == os.getuid(), reason="needs a process of another user (init) to be listed")
def test_the_memory_of_other_users_programs_and_of_everything_not_ours_is_what_others_hold(monkeypatch):
    calls = []
    # 4096 MB card, 2000 used: we hold 900 (a listed process of ours), init's 500 is another user's, 600 is not listed at all
    fake_nvidia(monkeypatch, used=2000, apps=[(os.getpid(), 900), (1, 500)], calls=calls)
    before = time.time()
    state = broker.card_usage()
    assert state["total_mb"] == TOTAL and state["others_mb"] == 1100 and before <= state["at"] <= time.time()
    assert calls == [("memory.total,memory.used", "gpu"), ("pid,used_memory", "compute-apps")]


def test_a_program_that_is_not_in_proc_a_row_that_cannot_be_read_and_an_empty_list_never_make_more_free_than_there_is(monkeypatch):
    gone = 4_000_000_000  # (no such process: whose it is cannot be told, so it counts as someone else's)
    fake_nvidia(monkeypatch, used=1500, apps=[(gone, 300)])
    assert broker.card_usage()["others_mb"] == 1500
    monkeypatch.setattr(broker, "_nvidia", lambda q, kind="gpu": [["4096", "1500"]] if kind == "gpu" else [["not a pid", "5"], ["12"], [], ["x", "y"]])
    assert broker.card_usage()["others_mb"] == 1500  # (rows that can't be read are skipped, they do not lower what others hold)
    fake_nvidia(monkeypatch, used=1500, apps=[])
    assert broker.card_usage()["others_mb"] == 1500
    fake_nvidia(monkeypatch, used=100, apps=[(os.getpid(), 900)])  # (listed ours more than the card says used: never below zero)
    assert broker.card_usage()["others_mb"] == 0


@pytest.mark.parametrize("failure", [FileNotFoundError("nvidia-smi"), subprocess.CalledProcessError(9, "nvidia-smi"),
                                     subprocess.TimeoutExpired("nvidia-smi", 5), ValueError("x"), IndexError()])
def test_without_a_card_or_with_an_answer_that_cannot_be_read_nothing_is_known(monkeypatch, failure):
    def nvidia(*a, **k):
        raise failure

    monkeypatch.setattr(broker, "_nvidia", nvidia)
    assert broker.card_usage() is None
    monkeypatch.setattr(broker, "_nvidia", lambda q, kind="gpu": [["lots", "some"]])
    assert broker.card_usage() is None  # (unparseable numbers)
    monkeypatch.setattr(broker, "_nvidia", lambda q, kind="gpu": [])
    assert broker.card_usage() is None  # (no card listed)
    broker.publish_card_state()  # (nothing to write: no error, no file)
    assert not broker._card_file().exists()


# ------------------------------------------------------------------------------------------------------------------ the state file

def test_the_gpu_worker_writes_the_state_atomically_and_it_is_read_back_while_fresh(monkeypatch):
    assert broker.card_state() is None  # (nothing written yet)
    fake_nvidia(monkeypatch, used=1200)
    broker.publish_card_state()
    path = broker._card_file()
    assert path == broker.get_settings().data_dir / "gpu-card.json" and path.is_file() and not path.with_suffix(".tmp").exists()
    state = broker.card_state()
    assert state["total_mb"] == TOTAL and state["others_mb"] == 1200
    fake_nvidia(monkeypatch, used=300)  # the next write replaces it
    broker.publish_card_state()
    assert broker.card_state()["others_mb"] == 300 and json.loads(path.read_text())["others_mb"] == 300


def test_a_state_older_than_thirty_seconds_or_not_readable_is_no_state():
    assert broker.CARD_STATE_FRESH_S == 30
    set_card(1000, age=29)
    assert broker.card_state()["others_mb"] == 1000
    set_card(1000, age=31)
    assert broker.card_state() is None  # (the worker is gone, or cannot see the card)
    path = broker._card_file()
    for text in ("", "not json", "[]", "{}", '{"total_mb": 4096, "others_mb": 1}', '{"at": "yesterday"}', "null"):
        path.write_text(text)
        assert broker.card_state() is None, text
    path.unlink()
    assert broker.card_state() is None


# --------------------------------------------------------------------------------------------------------- what fits in what is left

def test_what_a_job_needs_of_the_card_is_the_most_of_its_stages_because_they_run_one_after_another():
    assert broker.CARD_MARGIN_MB == 300
    need = {s: broker.GPU_WORK[s].card_mb for s in broker.GPU_WORK}
    assert need == {"transcribe": TRANSCRIBE, "heavy": HEAVY, "rebuild": REBUILD, "encode": SMALL}
    assert broker.card_need_mb(["heavy", "encode"]) == HEAVY and broker.card_need_mb(["rebuild", "heavy"]) == REBUILD
    assert broker.card_need_mb(["transcribe", "rebuild"]) == TRANSCRIBE and broker.card_need_mb([]) == 0 and broker.card_need_mb(["other"]) == 0


@pytest.mark.parametrize("others, fits", [
    (0, {"rebuild": True, "transcribe": True, "heavy": True, "encode": True}),  # 3796 free
    (496, {"rebuild": True, "transcribe": True, "heavy": True, "encode": True}),  # 3300 free: exactly Parakeet
    (497, {"rebuild": True, "transcribe": False, "heavy": True, "encode": True}),
    (1196, {"rebuild": True, "transcribe": False, "heavy": True, "encode": True}),  # 2600 free: exactly the rebuild
    (1197, {"rebuild": False, "transcribe": False, "heavy": True, "encode": True}),
    (3436, {"rebuild": False, "transcribe": False, "heavy": True, "encode": True}),  # 360 free: exactly the heavy remover
    (3437, {"rebuild": False, "transcribe": False, "heavy": False, "encode": True}),
    (3546, {"rebuild": False, "transcribe": False, "heavy": False, "encode": True}),  # 250 free: exactly the encoder
    (3547, {"rebuild": False, "transcribe": False, "heavy": False, "encode": False}),
    (TOTAL, {"rebuild": False, "transcribe": False, "heavy": False, "encode": False}),
    (TOTAL + 500, {"rebuild": False, "transcribe": False, "heavy": False, "encode": False}),  # (more held than there is: no negative room)
])
def test_a_stage_fits_when_the_card_less_what_others_hold_less_the_margin_has_its_memory_free(others, fits):
    state = {"total_mb": TOTAL, "others_mb": others, "at": time.time()}
    assert {s: broker.card_room([s], state)[0] for s in fits} == fits
    assert broker.card_room(["rebuild", "heavy"], state)[0] is fits["rebuild"]  # (the most of them)


def test_the_reason_a_job_does_not_fit_says_what_is_free_who_holds_the_rest_and_what_the_job_needs():
    ok, why = broker.card_room(["rebuild"], {"total_mb": TOTAL, "others_mb": 3000, "at": time.time()})
    assert ok is False and why == "card: 796 MB free (other programs hold 3000 of 4096 MB), the job needs 2600 MB"
    ok, why = broker.card_room(["heavy"], {"total_mb": TOTAL, "others_mb": 4000, "at": time.time()})
    assert ok is False and why.startswith("card: 0 MB free (other programs hold 4000 of 4096 MB)")
    assert broker.card_room(["heavy"], {"total_mb": TOTAL, "others_mb": 0, "at": time.time()}) == (True, "")


def test_room_is_taken_as_there_is_when_it_cannot_be_told_or_nothing_is_needed(monkeypatch):
    full = {"total_mb": TOTAL, "others_mb": TOTAL, "at": time.time()}
    assert broker.card_room(["rebuild"], None) == (True, "")  # (no state file either)
    assert broker.card_room([], full) == (True, "") and broker.card_room(["other"], full) == (True, "")  # (no card stage: no need)
    set_card(TOTAL)  # without a given state the worker's file is read
    assert broker.card_room(["rebuild"])[0] is False and broker.card_room(["heavy"])[0] is False
    set_card(TOTAL, age=60)  # (stale)
    assert broker.card_room(["rebuild"]) == (True, "")


def test_the_rebuild_can_be_ordered_when_the_worker_is_up_and_the_card_has_room_for_it(monkeypatch):
    assert broker.rebuild_available() is False  # (no worker)
    broker.mark_gpu_worker("alive")
    assert broker.rebuild_available() is True  # (no state at all: as before)
    set_card(1196)
    assert broker.rebuild_available() is True
    set_card(1197)
    assert broker.rebuild_available() is False  # (the worker is up but the card is full)
    set_card(1197, age=31)
    assert broker.rebuild_available() is True  # (a state that is not fresh says nothing)
    set_card(0)
    broker.mark_gpu_worker("stopped")
    assert broker.rebuild_available() is False  # (room, but no worker)


def test_the_gpu_worker_itself_measures_the_card_now_instead_of_reading_its_file(monkeypatch):
    broker.mark_gpu_worker("alive")
    set_card(0)  # (the file says there is room, an old measure)
    fake_nvidia(monkeypatch, used=3500)
    assert broker.rebuild_available() is True and broker.rebuild_available(live=True) is False
    set_card(3500)
    fake_nvidia(monkeypatch, used=100)
    assert broker.rebuild_available() is False and broker.rebuild_available(live=True) is True
    monkeypatch.setattr(broker, "_nvidia", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))  # (cannot measure: the file decides)
    assert broker.rebuild_available(live=True) is False


# ------------------------------------------------------------------------------------------------------------------------ admission

class Load:
    def snapshot(self):
        return Snapshot(cores=8, busy_cores=0.0, nice_cores=0.0, mem_available=20 * 1024**3, slice_headroom=None, disk_free=100 * 1024**3)


def test_the_card_job_is_refused_with_the_reason_when_its_stages_do_not_fit_and_admitted_when_they_do(client, signed_in, gpu_up, monkeypatch):
    uid, h = funded(signed_in, 50_000_000)
    heavy = row(start(client, h, options={"noise": "heavy"}))
    both = row(start(client, h, tier="rebuild"))
    light = row(start(client, h, options={"noise": "light"}))
    cancel(light["id"])
    snap = Load().snapshot()
    # 3000 MB held by others (measured now): 796 free, the heavy remover (360) fits, the rebuild (2600) does not
    fake_nvidia(monkeypatch, used=3000)
    ok, why, threads = broker.admit_job(heavy, snap, [], gpu=True)
    assert ok is True and why == "" and threads >= 2
    ok, why, threads = broker.admit_job(both, snap, [], gpu=True)
    assert (ok, threads) == (False, 0) and why == "card: 796 MB free (other programs hold 3000 of 4096 MB), the job needs 2600 MB"
    fake_nvidia(monkeypatch, used=3900)
    ok, why, _ = broker.admit_job(heavy, snap, [], gpu=True)
    assert ok is False and why.startswith("card: 0 MB free")
    fake_nvidia(monkeypatch, used=0)
    assert broker.admit_job(both, snap, [], gpu=True)[0] is True
    # a CPU worker does not look at the card, and a job with no card stage is admitted by a full card too (nothing of it is on the card)
    fake_nvidia(monkeypatch, used=3900)
    assert broker.admit_job(heavy, snap, [], gpu=False)[0] is True
    assert broker.gpu_stages(light) == [] and broker.admit_job(light, snap, [], gpu=True)[0] is True
    # when the card cannot be measured nothing is refused
    monkeypatch.setattr(broker, "_nvidia", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    assert broker.admit_job(both, snap, [], gpu=True)[0] is True


def test_the_gpu_worker_that_finds_no_room_says_so_and_the_job_stays_queued(client, signed_in, gpu_up, monkeypatch):
    uid, h = funded(signed_in, 50_000_000)
    jid = start(client, h, options={"noise": "heavy"})
    monkeypatch.setattr(worker, "GPU_MODE", True)
    fake_nvidia(monkeypatch, used=3900)
    assert worker.claim("gpu:t", Load()) is None and worker._card_wait is True and row(jid)["status"] == "queued"
    fake_nvidia(monkeypatch, used=1000)
    assert worker.claim("gpu:t", Load())["id"] == jid and worker._card_wait is False  # (room again)
    # a wait for something else (nothing queued; memory of the host) is not a wait for the card
    assert worker.claim("gpu:t", Load()) is None and worker._card_wait is False


def test_the_gpu_workers_loop_is_busy_not_idle_while_it_waits_for_room_and_a_cpu_worker_takes_the_heavy_job_the_rebuild_keeps_waiting(
        client, signed_in, gpu_up, monkeypatch):
    uid, h = funded(signed_in, 50_000_000)
    rebuild = start(client, h, tier="rebuild")  # (first in the queue)
    heavy = start(client, h, options={"noise": "heavy"})
    db.execute("UPDATE jobs SET queued_at = queued_at - 10 WHERE id = ?", (rebuild,))
    fake_nvidia(monkeypatch, used=3500)  # others hold 3500: 296 MB free, neither fits
    monkeypatch.setattr(worker, "GPU_MODE", True)
    assert worker.claim("gpu:t", Load()) is None and worker._card_wait is True
    # the loop: it publishes the card, marks itself busy for a wait for room (idle when there is nothing at all), and marks itself stopped
    waits = iter([True, False])
    seen = []

    def fake_claim(worker_id, load=None):
        worker._card_wait = next(waits)
        return None

    def stop_wait(seconds):
        seen.append({"idle": broker.gpu_worker_idle(), "up": broker.gpu_worker_up(), "card": broker.card_state()})
        if len(seen) == 2:
            worker._stop.set()

    with monkeypatch.context() as m:
        m.setattr(worker, "claim", fake_claim)
        m.setattr(worker._stop, "wait", stop_wait)
        m.setattr(worker.signal, "signal", lambda *a: None)
        m.setattr(worker.hostload, "get", lambda d: None)
        try:
            worker.main()
        finally:
            worker._stop.clear()
    assert seen[0]["idle"] is False and seen[0]["up"] is True  # waiting for room: busy
    assert seen[0]["card"]["others_mb"] == 3500 and seen[0]["card"]["total_mb"] == TOTAL  # (published by the loop)
    assert seen[1]["idle"] is True and seen[1]["up"] is True  # nothing to wait for: idle
    assert broker.gpu_worker_up() is False  # (and stopped when it ended)
    # what the CPU workers see: a card that is idle leaves its jobs to it; a busy one does not
    monkeypatch.setattr(worker, "GPU_MODE", False)
    broker.mark_gpu_worker("alive")
    broker.mark_gpu_worker("idle")
    assert worker.claim("cpu:t") is None
    broker.mark_gpu_worker("busy")  # (what the loop did)
    cpu_job = worker.claim("cpu:t")
    assert cpu_job["id"] == heavy  # the heavy remover is the CPU workers' to take now
    assert worker.claim("cpu:t") is None and row(rebuild)["status"] == "queued"  # the rebuild runs only on the card: it keeps waiting
    # the card has room again: the GPU worker takes the rebuild
    fake_nvidia(monkeypatch, used=0)
    monkeypatch.setattr(worker, "GPU_MODE", True)
    assert worker.claim("gpu:t", Load())["id"] == rebuild


# ------------------------------------------------------------------------------------------------ ordering a rebuild, the price list

def rebuild_paused(r):
    return r.status_code == 400 and r.json()["detail"] == jobs.REBUILD_PAUSED and "paused" in r.json()["detail"]


def test_a_rebuild_cannot_be_created_estimated_or_started_while_the_card_is_full_and_the_price_list_says_so(client, signed_in, gpu_up):
    uid, h = funded(signed_in, 50_000_000)
    ask = {"kind": "denoise", "durationSeconds": 60, "options": {}}  # (an estimate)
    make = {**ask, "filename": "talk.wav", "sizeBytes": 3000}  # (a job)
    bodies = {"/api/jobs/estimate": ask, "/api/jobs": make}
    assert client.get("/api/billing/pricing").json()["rebuildAvailable"] is True
    set_card(3500)
    assert client.get("/api/billing/pricing").json()["rebuildAvailable"] is False
    for path, body in bodies.items():
        assert rebuild_paused(client.post(path, json={**body, "denoiseTier": "rebuild"}, headers=h)), path
        assert rebuild_paused(client.post(path, json={**body, "timeline": whole(60, "rebuild")}, headers=h)), path  # (a rebuild part of a timeline)
        assert rebuild_paused(client.post(path, json={**body, "kind": "denoise_transcribe", "denoiseTier": "rebuild", "transcribeTier": "local"}, headers=h))
    with pytest.raises(jobs.JobError, match="paused"):
        jobs.create_job(uid, kind="denoise", denoise_tier="rebuild", transcribe_tier=None, filename="a.wav", size_bytes=1000, claimed_seconds=60.0, options={})
    assert db.one("SELECT COUNT(*) AS n FROM jobs")["n"] == 0
    # everything else is still possible: Denoising (heavy included) and a transcription
    assert client.post("/api/jobs/estimate", json={**ask, "denoiseTier": "standard", "options": {"noise": "heavy"}}, headers=h).status_code == 200
    assert client.post("/api/jobs/estimate", json={"kind": "transcribe", "durationSeconds": 60, "transcribeTier": "local", "options": {}},
                       headers=h).status_code == 200
    # an edit of a finished job that would rebuild
    finished = edit_job(uid, seconds=600.0, rendered="empty")
    tl = tl_of(finished, rebuild=[{"start": 100, "end": 120}])
    r = client.post(f"/api/jobs/{finished}/process", json={"timeline": tl}, headers=h)
    assert rebuild_paused(r) and row(finished)["status"] == "done" and "run" not in params_of(finished)
    # room again (others hold 1000 MB: 2796 free): all of it works
    set_card(1000)
    assert client.get("/api/billing/pricing").json()["rebuildAvailable"] is True
    assert client.post("/api/jobs/estimate", json={**ask, "denoiseTier": "rebuild"}, headers=h).status_code == 200
    assert client.post(f"/api/jobs/{finished}/process", json={"timeline": tl}, headers=h).status_code == 200
    made = client.post("/api/jobs", json={**make, "timeline": whole(60, "rebuild")}, headers=h)
    assert made.status_code == 200, made.text
    # a state the worker has not refreshed says nothing
    set_card(TOTAL, age=45)
    assert client.get("/api/billing/pricing").json()["rebuildAvailable"] is True


def test_the_price_list_says_unavailable_without_a_worker_whatever_the_card_has(client):
    set_card(0)
    assert client.get("/api/billing/pricing").json()["rebuildAvailable"] is False
    broker.mark_gpu_worker("alive")
    assert client.get("/api/billing/pricing").json()["rebuildAvailable"] is True


def test_the_snapshot_shows_what_others_hold_and_whether_the_rebuild_can_be_ordered(gpu_up):
    assert broker.snapshot()["gpu"]["othersMb"] is None and broker.snapshot()["gpu"]["rebuildAvailable"] is True
    set_card(3500)
    shown = broker.snapshot()["gpu"]
    assert shown["othersMb"] == 3500 and shown["rebuildAvailable"] is False


# ------------------------------------------------------------------------------------------------------------- no card state at all

def test_with_no_card_state_everything_behaves_as_before(client, signed_in, gpu_up, monkeypatch):
    uid, h = funded(signed_in, 50_000_000)
    assert broker.card_usage() is None and broker.card_state() is None  # (no card, no file)
    assert broker.card_room(["transcribe", "rebuild", "heavy", "encode"]) == (True, "")
    assert broker.rebuild_available() is True and broker.rebuild_available(live=True) is True
    assert client.get("/api/billing/pricing").json()["rebuildAvailable"] is True
    rebuild = start(client, h, tier="rebuild")
    heavy = start(client, h, options={"noise": "heavy"})
    db.execute("UPDATE jobs SET queued_at = queued_at - 10 WHERE id = ?", (rebuild,))
    assert broker.admit_job(row(rebuild), Load().snapshot(), [], gpu=True)[0] is True
    monkeypatch.setattr(worker, "GPU_MODE", True)
    assert worker.claim("gpu:t", Load())["id"] == rebuild and worker._card_wait is False
    assert worker.claim("gpu:t", Load())["id"] == heavy
    broker.mark_gpu_worker("stopped")
    assert broker.rebuild_available() is False  # (and without a worker it is paused, as before)
