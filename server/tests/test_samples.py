"""Machine samples for the dashboard's graphs (acw/samples.py)."""
import sqlite3

import pytest

from acw import samples

GB = 1024**3


def snapshot(**over):
    sn = {
        "cpu": {"cores": 16, "busy": 3.5, "niced": 1.0, "perCpu": []},
        "memory": {"available": 20 * GB, "sliceHeadroom": 8 * GB, "reserve": GB},
        "disk": {"free": 500 * GB, "reserve": GB},
        "live": {"sessions": {"live": 3, "live_trial": 2}, "liveCores": [1, 2],
                 "engine": {"sessions": 5, "units": 4.5, "capacityUnits": 12.0, "captionStreams": 1, "captionCap": 4}},
        "gpu": {"worker": "idle", "card": {"name": "RTX", "memoryUsedMb": 1200, "memoryTotalMb": 12288, "busyPercent": 37}},
        "jobs": {"running": [{"id": "a"}, {"id": "b"}], "queued": 7, "jobCpus": [3, 4]},
    }
    sn.update(over)
    return sn


def test_from_snapshot_reads_each_figure():
    s = samples.from_snapshot(snapshot())
    assert s == {
        "cpu_busy": 3.5, "cpu_cores": 16, "mem_available": 20 * GB, "slice_headroom": 8 * GB, "disk_free": 500 * GB,
        "gpu_busy": 37, "gpu_mem_used": 1200, "gpu_mem_total": 12288,
        "jobs_queued": 7, "jobs_running": 2, "live_sessions": 5, "live_trial": 2, "live_units": 4.5, "live_capacity": 12.0,
    }
    assert set(s) == set(samples.FIELDS)


def test_missing_gpu_or_live_engine_becomes_null_not_an_error():
    for gpu in ({"worker": "down", "card": {"down": "nvidia-smi missing"}}, {"worker": "down", "card": None}, {"worker": "down"}, None):
        s = samples.from_snapshot(snapshot(gpu=gpu))
        assert s["gpu_busy"] is None and s["gpu_mem_used"] is None and s["gpu_mem_total"] is None
        assert s["cpu_busy"] == 3.5 and s["jobs_queued"] == 7  # the rest is untouched
    s = samples.from_snapshot(snapshot(live={"sessions": {}, "liveCores": [], "engine": {"down": "no socket"}}))
    assert s["live_units"] is None and s["live_capacity"] is None and s["live_sessions"] == 0
    s = samples.from_snapshot(snapshot(live=None))
    assert s["live_sessions"] == 0 and s["live_units"] is None


def test_from_snapshot_of_the_real_broker():
    """The shape broker.snapshot() makes today is the shape from_snapshot reads."""
    from acw import broker

    s = samples.from_snapshot(broker.snapshot())
    assert set(s) == set(samples.FIELDS)
    assert s["cpu_cores"] >= 1


def test_record_creates_the_database_and_stores_the_sample_in_its_own_file(tmp_path):
    p = tmp_path / "deep" / "s.sqlite3"
    p.parent.mkdir()
    samples.record(samples.from_snapshot(snapshot()), now=1000, p=p)
    assert p.exists()
    c = sqlite3.connect(p)
    try:
        rows = c.execute("SELECT * FROM samples").fetchall()
        cols = [d[1] for d in c.execute("PRAGMA table_info(samples)")]
    finally:
        c.close()
    assert cols == ["ts", *samples.FIELDS]
    assert len(rows) == 1 and rows[0][0] == 1000 and rows[0][1] == 3.5

    # extra keys are ignored, missing ones are stored as NULL
    samples.record({"cpu_busy": 2.0, "bogus": 9, "gpu_busy": None}, now=10, p=p)
    row = samples.series(0, 100, 100, p)[0]
    assert row["cpu_busy"] == 2.0 and row["gpu_busy"] is None and row["disk_free"] is None and "bogus" not in row


def test_the_default_database_is_the_samples_file_not_the_main_one():
    samples.record({"cpu_busy": 1.0}, now=50)
    assert samples.path().exists()
    assert samples.series(0, 100, 100)[0]["cpu_busy"] == 1.0
    from acw import db

    assert db.one("SELECT name FROM sqlite_master WHERE name = 'samples'") is None  # not in the main database


def test_record_prunes_what_is_older_than_the_retention(tmp_path):
    p = tmp_path / "s.sqlite3"
    now = 2 * samples.KEEP_S
    # old rows are written by "earlier" records (each record prunes relative to its own time, so they survive until later)
    samples.record({"cpu_busy": 1.0}, now=now - samples.KEEP_S - 1, p=p)  # one second too old
    samples.record({"cpu_busy": 2.0}, now=now - samples.KEEP_S, p=p)  # exactly at the limit: kept
    samples.record({"cpu_busy": 3.0}, now=now - 10, p=p)
    samples.record({"cpu_busy": 4.0}, now=now, p=p)
    c = sqlite3.connect(p)
    try:
        ts = [r[0] for r in c.execute("SELECT ts FROM samples ORDER BY ts")]
    finally:
        c.close()
    assert ts == [now - samples.KEEP_S, now - 10, now]


def test_series_averages_per_bucket_and_buckets_line_up(tmp_path):
    p = tmp_path / "s.sqlite3"
    for ts, busy, disk in ((100, 1.0, 10), (110, 3.0, None), (190, 8.0, 30), (200, 10.0, 40), (299, 20.0, 50), (300, 99.0, 1), (50, 99.0, 1)):
        samples.record({"cpu_busy": busy, "disk_free": disk}, now=ts, p=p)
    rows = samples.series(100, 300, 100, p)  # start inclusive, end exclusive
    assert [r["t"] for r in rows] == [100, 200]
    assert rows[0]["cpu_busy"] == pytest.approx(4.0)  # (1 + 3 + 8) / 3
    assert rows[0]["disk_free"] == pytest.approx(20.0)  # missing values are left out of the average: (10 + 30) / 2
    assert rows[1]["cpu_busy"] == pytest.approx(15.0) and rows[1]["disk_free"] == pytest.approx(45.0)
    assert rows[0]["gpu_busy"] is None and set(rows[0]) == {"t", *samples.FIELDS}
    # a bucket is aligned to its size: 07:00 falls in the 06:00 bucket
    other = tmp_path / "aligned.sqlite3"
    samples.record({"cpu_busy": 1.0}, now=7 * 3600 + 5, p=other)
    assert [r["t"] for r in samples.series(0, 86400, 6 * 3600, other)] == [6 * 3600]


def test_reading_never_creates_or_changes_the_database(tmp_path):
    missing = tmp_path / "nothing.sqlite3"
    assert samples.series(0, 100, 10, missing) == [] and not missing.exists()
    assert samples.series(0, 100, 10) == []  # the default path, in an empty data directory
    p = tmp_path / "s.sqlite3"
    samples.record({"cpu_busy": 1.0}, now=10, p=p)
    c = samples._connect(p, readonly=True)
    try:
        with pytest.raises(sqlite3.OperationalError):
            c.execute("DELETE FROM samples")
    finally:
        c.close()
