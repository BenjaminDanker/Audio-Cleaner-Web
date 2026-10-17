"""The resource broker: one ranked table of who gets the machine, and the whole picture in one place."""
import json

from acw import broker, db
from acw.stream import protocol as P


def test_the_ranking_is_data_not_code():
    assert broker.outranks("live", "job") and broker.outranks("job", "live_trial")
    assert [c.name for c in sorted(broker.CLASSES.values(), key=lambda c: c.rank)] == ["live", "job", "live_trial"]


def test_a_free_trial_session_does_not_open_a_core_while_paid_jobs_run(client, signed_in):
    assert broker.live_options("live_trial") == {} and broker.live_options("live") == {}
    uid, _ = signed_in
    from acw import jobs
    from acw.billing import ledger

    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    jid = jobs.create_job(uid, kind="denoise", denoise_tier="standard", transcribe_tier=None, filename="a.wav",
                          size_bytes=1000, claimed_seconds=600.0, options={})["id"]
    db.execute("UPDATE jobs SET status = 'running' WHERE id = ?", (jid,))
    assert broker.live_options("live_trial") == {"may_open_core": False}
    assert broker.live_options("live") == {}  # paid live outranks jobs: it takes the room it needs


def test_the_whole_picture_in_one_place(client, monkeypatch):
    monkeypatch.setattr(broker, "live_engine_status", lambda timeout=2.0: {
        "engine": "native", "sessions": 3, "units": 3.0, "capacityUnits": 45.5, "captionStreams": 1, "captionCap": 35})
    sn = broker.snapshot()
    assert set(sn) >= {"classes", "cpu", "memory", "disk", "live", "jobs"}
    text = broker.format_snapshot(sn)
    assert "3 sessions, 3.0 of 45.5 units" in text and "Priority: live > job > live_trial" in text
    json.dumps(sn)  # the admin command's --json


def test_engines_answer_a_status_question():
    assert P.MSG_QUERY == b"Q"
