from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QSettings

from audio_cleaner_desktop import sysinfo
from audio_cleaner_desktop.gui import media
from audio_cleaner_desktop.gui.history import DONE, FAILED, Entry, HistoryStore
from audio_cleaner_desktop.gui.history_view import describe_settings
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


def test_numpys_own_thread_pool_is_one_thread_unless_set_before_numpy_loads():
    desktop = Path(__file__).resolve().parents[1]
    code = ("import os, sys; sys.path[:0] = sys.argv[1:3]; import audio_cleaner_desktop.__main__; "
            "print(*(os.environ[n] for n in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')))")
    names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    run = lambda **extra: subprocess.run(  # noqa: E731
        [sys.executable, "-I", "-c", code, str(desktop), str(desktop.parent / "common")], capture_output=True, text=True,
        env={**{k: v for k, v in os.environ.items() if k not in names}, **extra}, check=True).stdout.split()
    assert run() == ["1", "1", "1"]
    assert run(OMP_NUM_THREADS="4") == ["1", "4", "1"]  # setdefault: an explicit choice stays


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
    assert (prefs.strength, prefs.leveling, prefs.loudness, prefs.subtitles, prefs.parallel) == ("max", True, "podcast", "off", 0)
    prefs.strength, prefs.leveling, prefs.subtitles, prefs.parallel = "gentle", False, "embed", 3
    prefs.output_mode, prefs.output_folder = "folder", str(tmp_path)
    store.sync()
    again = Preferences(QSettings(str(tmp_path / "p.ini"), QSettings.Format.IniFormat))
    assert (again.strength, again.leveling, again.subtitles, again.parallel) == ("gentle", False, "embed", 3)
    assert again.output_dir() == Path(tmp_path)
    assert again.job_options().subtitles == "embed" and again.job_options().leveling is False
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


def test_automatic_best_settings_are_on_by_default_and_keep_the_manual_choices(tmp_path):
    store = QSettings(str(tmp_path / "p.ini"), QSettings.Format.IniFormat)
    prefs = Preferences(store)
    assert prefs.auto_settings is True
    o = prefs.job_options()
    assert (o.strength, o.noise, o.atten_lim_db) == ("auto", "auto", None)
    assert (o.restore_highs, o.rebuild_voice, o.auto_rebuild) == (True, False, True)
    # chosen by hand: kept, but not used while it is on
    prefs.strength, prefs.noise, prefs.restore_highs, prefs.rebuild_voice = "gentle", "heavy", False, True
    o = prefs.job_options()
    assert (o.strength, o.noise, o.restore_highs, o.rebuild_voice, o.auto_rebuild) == ("auto", "auto", True, False, True)
    prefs.auto_settings = False
    store.sync()
    again = Preferences(QSettings(str(tmp_path / "p.ini"), QSettings.Format.IniFormat))
    assert again.auto_settings is False
    o = again.job_options()
    assert (o.strength, o.noise, o.atten_lim_db) == ("gentle", "heavy", 12.0)
    assert (o.restore_highs, o.rebuild_voice, o.auto_rebuild) == (False, True, False)
    again.auto_settings = True
    assert (again.job_options().strength, again.job_options().noise) == ("auto", "auto")


def test_history_says_which_strength_was_picked():
    auto = {"strength": "auto", "noise": "auto", "leveling": False}
    assert describe_settings({**auto, "picked_strength": "balanced"}, False) == "Balanced noise reduction (picked for this file)"
    assert describe_settings({**auto, "picked_strength": "max"}, False) == "Maximum noise reduction (picked for this file)"
    assert describe_settings(auto, False) == "noise reduction picked for the file"  # e.g. a run that failed
    assert describe_settings({**auto, "picked_strength": "bogus"}, False) == "noise reduction picked for the file"
    # a strength chosen by hand is what is shown, whatever else is stored
    assert describe_settings({"strength": "gentle", "leveling": False, "picked_strength": "max"}, False) == "Gentle noise reduction"
    assert describe_settings({**auto, "rebuild_voice": True, "picked_strength": "max"}, False) == "voice rebuilt"
    assert describe_settings({**auto, "auto_rebuild": True, "picked_rebuild": "echoey"}, False) == "voice rebuilt (it sounded echoey)"
    assert describe_settings({**auto, "auto_rebuild": True, "picked_rebuild": "muffled", "leveling": True, "loudness": "youtube"},
                             False) == "voice rebuilt (it sounded muffled), leveled (YouTube)"
    assert describe_settings({**auto, "auto_rebuild": True, "picked_rebuild": "unclear"}, False) == "voice rebuilt (it sounded unclear)"


def test_history_keeps_the_automatic_choices():
    from audio_cleaner_desktop import engine
    from audio_cleaner_desktop.gui.history_view import settings_dict

    s = settings_dict(engine.JobOptions(strength="auto", auto_rebuild=True))
    assert (s["strength"], s["noise"], s["auto_rebuild"], s["rebuild_voice"]) == ("auto", "auto", True, False)
    assert settings_dict(engine.JobOptions())["auto_rebuild"] is False
    assert settings_dict(engine.JobOptions(auto_highs=True))["auto_highs"] is True and settings_dict(engine.JobOptions())["auto_highs"] is False
