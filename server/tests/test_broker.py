"""The resource broker: one ranked table of who gets the machine, and the whole picture in one place."""
import json

from acw import broker, db


def test_paid_live_outranks_jobs_and_free_trial_sessions_come_last(client, signed_in):
    assert [c.name for c in sorted(broker.CLASSES.values(), key=lambda c: c.rank)] == ["live", "job", "live_trial"]
    assert broker.live_options("live_trial") == {} and broker.live_options("live") == {}
    uid, _ = signed_in
    from acw import jobs
    from acw.billing import ledger

    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    jid = jobs.create_job(uid, kind="denoise", denoise_tier="standard", transcribe_tier=None, filename="a.wav",
                          size_bytes=1000, claimed_seconds=600.0, options={})["id"]
    db.execute("UPDATE jobs SET status = 'running' WHERE id = ?", (jid,))
    # a free trial session does not open a core while paid jobs run; paid live takes the room it needs
    assert broker.live_options("live_trial") == {"may_open_core": False}
    assert broker.live_options("live") == {}


def test_the_whole_picture_is_one_json_document(client, monkeypatch):
    monkeypatch.setattr(broker, "live_engine_status", lambda timeout=2.0: {
        "engine": "native", "sessions": 3, "units": 3.0, "capacityUnits": 45.5, "captionStreams": 1, "captionCap": 35})
    sn = broker.snapshot()
    assert set(sn) >= {"classes", "cpu", "memory", "disk", "live", "jobs"}
    json.dumps(sn)  # the admin command's --json
