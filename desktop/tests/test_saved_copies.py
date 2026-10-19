"""The saved copies of a job (copies.py): the folder layout, what makes a copy count, the fingerprint of the original, the disk they
take, and that deleting a job removes its copies and nothing else (never the results or the original)."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pytest

from acw_common import captions
from audio_cleaner_desktop import copies
from audio_cleaner_desktop.gui.history import DONE, FAILED, Entry, HistoryStore


def make_job(root, original, *, with_noise=True):
    folder = copies.new_job_dir(root)
    job = copies.JobCopies.create(folder, original, copies.fingerprint(original), {"duration": 3.0, "has_video": False})
    if with_noise:
        (folder / "noise.flac").write_bytes(b"x" * 1000)
        job.put_audio(copies.NOISE, {"file": "noise.flac", "channels": [0], "scale": 1.0, "samples": 48000}, plan=[0, 0], profile="light")
    return job


@pytest.fixture
def original(tmp_path):
    path = tmp_path / "media" / "talk.wav"
    path.parent.mkdir()
    path.write_bytes(os.urandom(300_000))
    return path


def test_a_job_folder_has_a_name_this_app_gives_and_its_manifest_is_read_back(tmp_path, original):
    root = tmp_path / "jobs"
    job = make_job(root, original)
    assert copies.is_job_name(job.dir.name) and job.dir.parent == root
    again = copies.JobCopies.load(job.dir)
    assert again is not None and again.original_path == original and again.has(copies.NOISE) and again.kinds() == [copies.NOISE]
    assert again.entry(copies.NOISE)["plan"] == [0, 0] and again.sizes() == [(copies.NOISE, 1000)]


def test_a_copy_only_counts_when_its_file_is_there_with_the_size_it_was_written_with(tmp_path, original):
    job = make_job(tmp_path / "jobs", original)
    assert job.has(copies.NOISE)
    (job.dir / "noise.flac").write_bytes(b"x" * 999)  # half-written, or changed
    assert not job.has(copies.NOISE) and not job
    (job.dir / "noise.flac").write_bytes(b"x" * 1000)
    assert job.has(copies.NOISE)
    (job.dir / "noise.flac").unlink()  # removed by hand
    assert not job.has(copies.NOISE) and job.entry(copies.NOISE) is None
    # a manifest entry that points outside its folder is never used
    job.data["copies"][copies.NOISE] = {"file": "../job.json", "channels": [0], "scale": 1.0, "samples": 5, "bytes": 1}
    assert job.entry(copies.NOISE) is None
    job.data["copies"][copies.NOISE] = {"file": "noise.flac", "channels": [7], "scale": 1.0, "samples": 5, "bytes": 1}
    assert job.entry(copies.NOISE) is None


def test_a_missing_unreadable_or_foreign_manifest_is_no_job(tmp_path):
    folder = copies.new_job_dir(tmp_path / "jobs")
    assert copies.JobCopies.load(folder) is None
    (folder / copies.MANIFEST).write_text("{not json")
    assert copies.JobCopies.load(folder) is None
    (folder / copies.MANIFEST).write_text(json.dumps({"version": 99, "original": {}}))
    assert copies.JobCopies.load(folder) is None


def test_the_transcript_round_trips_and_counts_as_a_copy(tmp_path, original):
    job = make_job(tmp_path / "jobs", original, with_noise=False)
    assert not job and job.transcript() is None
    seg = captions.Segment(0.0, 1.5, "hello there", [captions.Word(0.0, 0.7, "hello"), captions.Word(0.7, 1.5, "there")])
    job.put_transcript([seg], "en", echo=False)
    got = copies.JobCopies.load(job.dir).transcript()
    assert got is not None and got[1] == "en" and got[0][0].text == "hello there" and len(got[0][0].words) == 2
    assert job.kinds() == [copies.TRANSCRIPT]


def test_the_fingerprint_notices_a_moved_a_changed_and_a_touched_original(tmp_path, original):
    fp = copies.fingerprint(original)
    assert copies.check_original(original, fp).ok
    # touched, or copied back: another modified time, same bytes -> still the same file
    os.utime(original, ns=(time.time_ns(), fp["mtime_ns"] + 10**9))
    assert copies.check_original(original, fp).ok
    # changed in the middle (the length is the same): found by the parts
    data = bytearray(original.read_bytes())
    for i in range(0, len(data), 997):
        data[i] ^= 0xFF
    original.write_bytes(bytes(data))
    changed = copies.check_original(original, fp)
    assert changed.state == "changed" and "changed" in changed.text
    # a different length
    original.write_bytes(b"short")
    assert copies.check_original(original, fp).state == "changed"
    # moved
    moved = tmp_path / "elsewhere.wav"
    original.rename(moved)
    missing = copies.check_original(original, fp)
    assert missing.state == "missing" and str(original) in missing.text
    assert copies.check_original(original, None).state == "changed"  # a job that never recorded one can't vouch for the file


def test_pointing_at_the_moved_original_is_accepted_only_for_the_same_file(tmp_path, original):
    job = make_job(tmp_path / "jobs", original)
    moved = tmp_path / "moved.wav"
    original.rename(moved)
    assert job.check_original().state == "missing"
    other = tmp_path / "other.wav"
    other.write_bytes(os.urandom(300_000))
    assert job.relocate(other).state == "changed" and job.original_path == original
    assert job.relocate(moved).ok and job.original_path == moved
    assert copies.JobCopies.load(job.dir).original_path == moved and job.check_original().ok


def test_disk_used_counts_the_whole_folder(tmp_path, original):
    job = make_job(tmp_path / "jobs", original)
    manifest = (job.dir / copies.MANIFEST).stat().st_size
    assert job.bytes() == 1000 + manifest == copies.scan_bytes(job.dir)
    (job.dir / "segments.json").write_bytes(b"y" * 50)
    assert job.bytes() == 1050 + manifest


def test_deleting_a_job_folder_removes_only_that_folder(tmp_path, original):
    root = tmp_path / "jobs"
    keep, drop = make_job(root, original), make_job(root, original)
    result = tmp_path / "media" / "talk-cleaned.wav"
    result.write_bytes(b"result")
    assert copies.delete_job_dir(root, drop.dir.name) and not drop.dir.exists()
    assert keep.dir.exists() and original.exists() and result.read_bytes() == b"result"
    assert copies.delete_job_dir(root, drop.dir.name)  # already gone: fine
    # only names this app gives are ever removed
    assert not copies.delete_job_dir(root, "..") and not copies.delete_job_dir(root, "media") and not copies.delete_job_dir(tmp_path, "jobs")
    assert root.exists() and (tmp_path / "media").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="links need privileges on Windows")
def test_a_link_in_the_jobs_folder_is_removed_as_a_link_and_never_followed(tmp_path, original):
    root = tmp_path / "jobs"
    root.mkdir()
    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "keep.txt").write_text("keep")
    name = "a" * 32
    (root / name).symlink_to(precious, target_is_directory=True)
    assert copies.delete_job_dir(root, name)
    assert not os.path.lexists(root / name) and (precious / "keep.txt").read_text() == "keep"
    # a link inside a job folder is not followed either
    job = make_job(root, original)
    (job.dir / "link").symlink_to(precious, target_is_directory=True)
    assert copies.delete_job_dir(root, job.dir.name) and (precious / "keep.txt").exists()


def test_orphan_folders_are_swept_after_a_day_but_never_ones_in_use(tmp_path, original):
    root = tmp_path / "jobs"
    listed, orphan, fresh = make_job(root, original), make_job(root, original), make_job(root, original)
    (root / ".acw-work-1-abc").mkdir()
    old = time.time() - 2 * 24 * 3600
    for job in (listed, orphan):
        for p in (job.dir, *job.dir.iterdir()):
            os.utime(p, (old, old))
    assert copies.sweep_orphans(root, {listed.dir.name}) == 1
    assert listed.dir.exists() and fresh.dir.exists() and not orphan.dir.exists() and (root / ".acw-work-1-abc").exists()


# ---------------------------------------------------------------------------------------------------- the list and its copies

def entry(n, **kw):
    return Entry(name=f"f{n}.wav", source=f"/a/f{n}.wav", duration=5.0 + n, has_video=False, settings={"strength": "gentle"},
                 seconds_taken=1.0, outputs=[f"/a/f{n}-cleaned.wav"], status=DONE, finished_at=1000.0 + n, **kw)


def test_the_list_adds_up_the_disk_and_reconciles_with_the_folders(tmp_path, original):
    store = HistoryStore(tmp_path / "h.db")
    root = store.jobs_root
    assert root == tmp_path / "jobs"
    a, b = make_job(root, original), make_job(root, original)
    store.add(entry(1, job_dir=a.dir.name, copies_bytes=a.bytes()))
    id_b = store.add(entry(2, job_dir=b.dir.name, copies_bytes=1))  # a stale count
    store.add(entry(3))  # an old entry: no copies
    assert store.copies_total().jobs == 2
    (b.dir / "extra").write_bytes(b"z" * 500)
    gone = make_job(root, original)
    id_gone = store.add(entry(4, job_dir=gone.dir.name, copies_bytes=5))
    copies.delete_job_dir(root, gone.dir.name)  # removed by hand
    store.reconcile()
    assert store.get(id_gone).job_dir == "" and store.get(id_gone).copies_bytes == 0
    assert store.get(id_b).copies_bytes == b.bytes()
    total = store.copies_total()
    assert total.jobs == 2 and total.bytes == a.bytes() + b.bytes()
    store.close()


def test_deleting_jobs_removes_their_copies_and_entries_but_never_the_results_or_the_original(tmp_path, original):
    store = HistoryStore(tmp_path / "h.db")
    result = tmp_path / "media" / "talk-cleaned.wav"
    result.write_bytes(b"result")
    job = make_job(store.jobs_root, original)
    done = store.add(Entry("talk.wav", str(original), 3.0, False, {}, 1.0, [str(result)], DONE, job_dir=job.dir.name, copies_bytes=job.bytes()))
    old = store.add(entry(9))  # made before the copies existed
    failed = store.add(Entry("bad.wav", "/a/bad.wav", 0.0, False, {}, 1.0, [], FAILED, "could not be read"))
    assert store.delete([done]) == [] and store.get(done) is None
    assert not job.dir.exists() and original.exists() and result.read_bytes() == b"result"
    assert store.delete([old, failed]) == [] and store.count() == 0
    # delete all, with copies of jobs that were never listed
    one, two = make_job(store.jobs_root, original), make_job(store.jobs_root, original)
    store.add(Entry("a.wav", str(original), 3.0, False, {}, 1.0, [str(result)], DONE, job_dir=one.dir.name))
    assert store.clear() == []
    assert not one.dir.exists() and not two.dir.exists() and store.count() == 0 and original.exists() and result.exists()
    store.close()


def test_a_job_whose_copies_cannot_be_removed_stays_in_the_list(tmp_path, original, monkeypatch):
    store = HistoryStore(tmp_path / "h.db")
    job = make_job(store.jobs_root, original)
    eid = store.add(Entry("talk.wav", str(original), 3.0, False, {}, 1.0, [], DONE, job_dir=job.dir.name))
    monkeypatch.setattr(copies, "delete_job_dir", lambda root, name: False)
    assert store.delete([eid]) == [eid] and store.get(eid) is not None and job.dir.exists()
    store.close()


def test_an_old_history_file_gets_the_new_columns_and_keeps_its_entries(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE history (id INTEGER PRIMARY KEY AUTOINCREMENT, finished_at REAL NOT NULL, name TEXT NOT NULL, "
               "source TEXT NOT NULL, duration REAL NOT NULL, has_video INTEGER NOT NULL, settings TEXT NOT NULL, "
               "seconds_taken REAL NOT NULL, outputs TEXT NOT NULL, status TEXT NOT NULL, message TEXT NOT NULL)")
    db.execute("INSERT INTO history (finished_at, name, source, duration, has_video, settings, seconds_taken, outputs, status, message) "
               "VALUES (5, 'old.wav', '/a/old.wav', 9, 0, ?, 3, '[\"/a/old-cleaned.wav\"]', 'done', '')",
               (json.dumps({"strength": "gentle", "leveling": True, "loudness": "podcast"}),))
    db.commit()
    db.close()
    store = HistoryStore(path)
    [old] = store.search()
    assert old.name == "old.wav" and old.job_dir == "" and old.copies_bytes == 0 and "Gentle" in old.summary
    assert [e.name for e in store.search("gentle")] == ["old.wav"]  # found by what it was made with, in words
    assert store.delete([old.id]) == [] and store.count() == 0
    store.close()


def test_search_sort_and_paging_in_the_store(tmp_path):
    store = HistoryStore(tmp_path / "h.db")
    for n in range(1, 8):
        store.add(entry(n))
    store.add(Entry("Zebra.mp4", "/videos/zebra.mp4", 100.0, True, {"strength": "max", "rebuild_voice": True}, 9.0, ["/videos/zebra-cleaned.mp4"],
                    DONE, finished_at=500.0, job_dir="b" * 32, copies_bytes=7_000_000))
    store.add(Entry("broken.wav", "/x/broken.wav", 0.0, False, {}, 1.0, [], FAILED, "This file could not be read.", finished_at=2000.0))
    assert store.count() == 9
    assert [e.name for e in store.search(limit=3)] == ["broken.wav", "f7.wav", "f6.wav"]  # newest first
    assert [e.name for e in store.search(limit=3, offset=3)] == ["f5.wav", "f4.wav", "f3.wav"]
    assert store.search(limit=3, offset=7)[-1].name == "Zebra.mp4" and len(store.search(limit=3, offset=7)) == 2
    assert [e.name for e in store.search(sort="oldest", limit=2)] == ["Zebra.mp4", "f1.wav"]
    assert store.search(sort="name", limit=2)[0].name == "broken.wav" and store.search(sort="name")[-1].name == "Zebra.mp4"
    assert store.search(sort="disk", limit=1)[0].name == "Zebra.mp4" and store.search(sort="length", limit=1)[0].name == "Zebra.mp4"
    # the search reads the name, the original's path, the results, the message, the settings (in words and as stored) and the status
    assert [e.name for e in store.search("zebra")] == ["Zebra.mp4"]
    assert [e.name for e in store.search("videos")] == ["Zebra.mp4"]
    assert [e.name for e in store.search("could not be read")] == ["broken.wav"]
    assert [e.name for e in store.search("rebuilt")] == ["Zebra.mp4"]
    assert [e.name for e in store.search("failed")] == ["broken.wav"]
    assert [e.name for e in store.search("gentle f3")] == ["f3.wav"]  # every word has to match
    assert store.count("gentle") == 7 and store.count("nothing like this") == 0
    assert store.count("100%") == 0 and store.count("f_") == 0  # wildcards are plain characters
    store.close()


def test_the_list_has_no_cap(tmp_path):
    store = HistoryStore(tmp_path / "h.db")
    for n in range(650):
        store.add(entry(n))
    assert len(store.search()) == 650 and store.count() == 650
    assert len(store.search("f", limit=100, offset=600)) == 50
    store.close()
