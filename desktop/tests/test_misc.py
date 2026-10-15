from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings

from audio_cleaner_desktop import sysinfo
from audio_cleaner_desktop.gui import media
from audio_cleaner_desktop.gui.history import DONE, FAILED, Entry, HistoryStore
from audio_cleaner_desktop.gui.history_view import describe_settings, format_hours, format_spent
from audio_cleaner_desktop.gui.preferences import Preferences


def test_auto_parallelism_follows_cores_and_memory():
    assert sysinfo.max_parallel(0, False, cores=2, ram_gb=32) == 1
    assert sysinfo.max_parallel(0, False, cores=6, ram_gb=32) == 2
    assert sysinfo.max_parallel(0, False, cores=32, ram_gb=64) == 4
    assert sysinfo.max_parallel(3, False, cores=2, ram_gb=64) == 3
    assert sysinfo.max_parallel(4, False, cores=8, ram_gb=3) == 1
    assert sysinfo.max_parallel(4, True, cores=8, ram_gb=8) == 1  # 8 * 0.6 // 2.5 = 1
    assert sysinfo.max_parallel(4, False, cores=8, ram_gb=8) == 3  # 8 * 0.6 // 1.5 = 3


def test_threads_are_split_between_jobs():
    assert sysinfo.threads_per_job(1) >= 2
    assert sysinfo.threads_per_job(64) == 2


def test_folders_are_searched_recursively(tmp_path):
    (tmp_path / "sub" / "deeper").mkdir(parents=True)
    (tmp_path / ".hidden").mkdir()
    for name in ("a.mp4", "sub/b.WAV", "sub/deeper/c.m4a", "sub/notes.txt", ".hidden/d.mp3", ".e.mp3"):
        (tmp_path / name).write_bytes(b"x")
    single = tmp_path / "odd.xyz"
    single.write_bytes(b"x")
    found = media.expand([tmp_path, single, tmp_path / "a.mp4", tmp_path / "missing.wav"])
    assert [p.relative_to(tmp_path).as_posix() for p in found] == ["a.mp4", "sub/b.WAV", "sub/deeper/c.m4a", "odd.xyz"]


def test_preferences_round_trip_and_ignore_bad_values(tmp_path):
    store = QSettings(str(tmp_path / "p.ini"), QSettings.Format.IniFormat)
    prefs = Preferences(store)
    assert (prefs.strength, prefs.polish, prefs.loudness, prefs.subtitles, prefs.parallel) == ("max", True, "podcast", "off", 0)
    prefs.strength, prefs.polish, prefs.subtitles, prefs.parallel = "gentle", False, "embed", 3
    prefs.output_mode, prefs.output_folder = "folder", str(tmp_path)
    store.sync()
    again = Preferences(QSettings(str(tmp_path / "p.ini"), QSettings.Format.IniFormat))
    assert (again.strength, again.polish, again.subtitles, again.parallel) == ("gentle", False, "embed", 3)
    assert again.output_dir() == Path(tmp_path)
    assert again.job_options().subtitles == "embed" and again.job_options().polish is False
    store.setValue("strength", "bogus")
    store.setValue("parallel", "99")
    assert (again.strength, again.parallel) == ("max", 0)


def test_history_store(tmp_path):
    store = HistoryStore(tmp_path / "h.db")
    store.add(Entry("talk.wav", "/a/talk.wav", 3600.0, False, {"strength": "max"}, 120.0, ["/a/talk-cleaned.wav"], DONE, finished_at=100))
    store.add(Entry("clip.mp4", "/a/clip.mp4", 1800.0, True, {}, 60.0, ["/a/clip-cleaned.mp4"], DONE, finished_at=200))
    store.add(Entry("bad_100%.wav", "/a/bad.wav", 0.0, False, {}, 1.0, [], FAILED, "could not be read", finished_at=300))
    totals = store.totals()
    assert (totals.files, totals.audio_seconds, totals.video_seconds, totals.seconds_spent) == (2, 3600.0, 1800.0, 120.0 + 60.0)
    assert [e.name for e in store.search()] == ["bad_100%.wav", "clip.mp4", "talk.wav"]
    assert [e.name for e in store.search("CLIP")] == ["clip.mp4"]
    assert [e.name for e in store.search("100%")] == ["bad_100%.wav"]
    assert store.search("talk")[0].outputs == ["/a/talk-cleaned.wav"]
    store.delete([store.search("talk")[0].id])
    assert store.totals().files == 1
    store.close()
    reopened = HistoryStore(tmp_path / "h.db")  # persists
    assert len(reopened.search()) == 2
    reopened.clear()
    assert reopened.search() == [] and reopened.totals().files == 0


def test_history_wording():
    assert format_hours(0) == "0 min" and format_hours(1800) == "30 min" and format_hours(5400) == "1.5 h"
    assert format_spent(30) == "< 1 min" and format_spent(125 * 60) == "2 h 05 min"
    text = describe_settings({"strength": "gentle", "polish": True, "loudness": "youtube", "subtitles": "srt"}, False)
    assert text == "Gentle noise reduction, polished (YouTube), subtitles: srt file"
    assert "polished" not in describe_settings({"strength": "max", "polish": False}, True)
