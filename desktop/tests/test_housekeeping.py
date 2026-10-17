"""Scratch folders and leftovers, claimed output names, cancelling ffmpeg, power and priority, single instance,
Show in folder, the lazy history and the stop/Esc behaviour of the window."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from acw_common.ffmpeg import FFmpeg, MediaError
from audio_cleaner_desktop import engine, models, sysinfo
from audio_cleaner_desktop.gui import history_view, main_window, shell, single_instance, tips
from audio_cleaner_desktop.gui.history import DONE, Entry, HistoryStore
from audio_cleaner_desktop.gui.main_window import MainWindow
from audio_cleaner_desktop.gui.queue_view import RUNNING

from conftest import wait_until


# -- scratch folder and leftovers ---------------------------------------------------------------------------

def test_scratch_is_the_app_data_folder_on_windows_and_mac_only(monkeypatch, tmp_path):
    monkeypatch.delenv("ACW_DESKTOP_SCRATCH", raising=False)
    monkeypatch.setattr(engine.sys, "platform", "linux")
    assert engine.scratch_root() is None
    for platform in ("win32", "darwin"):
        monkeypatch.setattr(engine.sys, "platform", platform)
        assert engine.scratch_root() == models.app_data_dir() / "scratch"
    monkeypatch.setenv("ACW_DESKTOP_SCRATCH", str(tmp_path / "s"))
    assert engine.scratch_root() == tmp_path / "s"


def test_the_work_folder_falls_back_to_the_result_folder_when_scratch_is_full(monkeypatch, tmp_path):
    monkeypatch.setenv("ACW_DESKTOP_SCRATCH", str(tmp_path / "scratch"))
    out = tmp_path / "out"
    out.mkdir()
    assert engine._work_parent(out, 0.5) == tmp_path / "scratch"
    monkeypatch.setattr(engine.shutil, "disk_usage", lambda p: SimpleNamespace(free=10**6))
    assert engine._work_parent(out, 0.5) == out


def test_sweep_removes_old_leftovers_and_nothing_else(monkeypatch, tmp_path):
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("ACW_DESKTOP_SCRATCH", str(scratch))
    gone, alive = scratch / ".acw-work-2147483646-abc", scratch / f".acw-work-{os.getpid()}-xyz"
    for d in (gone, alive):
        d.mkdir(parents=True)
        (d / "decoded.f32").write_bytes(b"x")
    old_junk = scratch / "junk"
    old_junk.write_bytes(b"x")
    folder = tmp_path / "videos"
    folder.mkdir()
    old_dir, new_dir = folder / ".acw-work-abcd", folder / ".acw-work-new1"
    old_tmp, new_tmp = folder / f".acw-{'a' * 32}.mp4", folder / f".acw-{'b' * 32}.mp4"
    mine = folder / "holiday.mp4"
    for d in (old_dir, new_dir):
        d.mkdir()
    for f in (old_tmp, new_tmp, mine):
        f.write_bytes(b"x")
    day = 2 * 24 * 3600
    for p in (old_dir, old_tmp, old_junk, mine):
        os.utime(p, (time.time() - day, time.time() - day))
    removed = engine.sweep_leftovers([folder])
    assert not gone.exists() and alive.exists()  # the pid of a running process is left alone
    assert not old_junk.exists() and not old_dir.exists() and not old_tmp.exists()
    assert new_dir.exists() and new_tmp.exists() and mine.exists()  # too new to be a crash's, and not ours
    assert removed == 4


def test_a_job_writes_its_work_files_to_the_scratch_folder(speech_wav, tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    from audio_cleaner_desktop import resources

    if resources.check():
        pytest.skip("native components missing")
    monkeypatch.setenv("ACW_DESKTOP_SCRATCH", str(tmp_path / "scratch"))
    seen = []
    real = engine.shutil.rmtree
    monkeypatch.setattr(engine.shutil, "rmtree", lambda work, *a, **k: seen.append(Path(work).parent) or real(work, *a, **k))
    out = tmp_path / "out"
    result = engine.process(speech_wav, engine.JobOptions(audio_format="flac"), out_dir=out)
    assert seen == [tmp_path / "scratch"]
    assert sorted(p.name for p in out.iterdir()) == [result.output.name]
    assert list((tmp_path / "scratch").iterdir()) == []


# -- output names ---------------------------------------------------------------------------------------------

def test_claimed_names_are_never_handed_out_twice(tmp_path):
    names = []
    lock = threading.Lock()

    def claim():
        p = engine.claim_output(tmp_path, "a", ".wav", ".srt")
        with lock:
            names.append(p.name)

    threads = [threading.Thread(target=claim) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(names)) == 12 and "a-cleaned.wav" in names and "a-cleaned (2).wav" in names
    assert (tmp_path / "a-cleaned.srt").exists()  # the companion name is taken as well


# -- cancelling ffmpeg ---------------------------------------------------------------------------------------

def test_cancel_kills_a_running_ffmpeg_program_at_once():
    cancel = threading.Event()
    ff = FFmpeg(Path("."), 1, cancel=cancel)
    threading.Timer(0.5, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(engine.Cancelled):
        ff._run([sys.executable, "-c", "import time; time.sleep(60)"], timeout=120)
    assert time.monotonic() - started < 10


def test_programs_run_normally_when_not_cancelled():
    ff = FFmpeg(Path("."), 1, cancel=threading.Event())
    r = ff._run([sys.executable, "-c", "import sys; sys.stdout.write('hi'); sys.exit(3)"], timeout=60)
    assert r.returncode == 3 and r.stdout == b"hi"
    with pytest.raises(subprocess.TimeoutExpired):
        ff._run([sys.executable, "-c", "import time; time.sleep(60)"], timeout=0.5)


@pytest.mark.skipif(os.name == "nt", reason="uses a shell script as a stand-in for ffmpeg")
def test_encoding_stops_when_cancelled(tmp_path):
    import numpy as np

    fake = tmp_path / "ffmpeg"
    fake.write_text("#!/bin/sh\nexec sleep 60\n")  # never reads its input: the writer blocks once the pipe is full
    fake.chmod(0o755)
    cancel = threading.Event()
    ff = FFmpeg(tmp_path, 1, cancel=cancel)
    threading.Timer(0.5, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(engine.Cancelled):
        ff.write_audio(np.zeros((48000 * 60, 1), np.float32), 48000, tmp_path / "out", "wav")
    assert time.monotonic() - started < 10


def test_decoding_stops_when_cancelled(tmp_path):
    cancel = threading.Event()
    ff = FFmpeg(Path("."), 1, cancel=cancel)
    cancel.set()
    with pytest.raises(engine.Cancelled):
        ff._run([sys.executable, "-c", "pass"], timeout=60)


def test_long_windows_paths_get_the_extended_prefix(monkeypatch):
    from acw_common import ffmpeg as ffmod

    monkeypatch.setattr(ffmod.sys, "platform", "win32")
    assert ffmod.path_arg("C:\\Users\\me\\a.wav") == "\\\\?\\C:\\Users\\me\\a.wav"
    assert ffmod.path_arg("\\\\?\\C:\\x.wav") == "\\\\?\\C:\\x.wav"
    assert ffmod.path_arg("\\\\server\\share\\a.wav") == "\\\\?\\UNC\\server\\share\\a.wav"
    monkeypatch.setattr(ffmod.sys, "platform", "linux")
    assert ffmod.path_arg("/tmp/a.wav") == "/tmp/a.wav"  # a string, so the expectation holds on every OS
    assert ffmod.path_arg(Path("a.wav")) == str(Path("a.wav"))


# -- messages -------------------------------------------------------------------------------------------------

def test_old_flash_video_is_refused_with_a_plain_message(tmp_path, monkeypatch):
    f = tmp_path / "old.flv"
    f.write_bytes(b"x")
    ff = SimpleNamespace(probe=lambda p: SimpleNamespace(duration=10.0, has_video=True, video_codec="flv1"))
    with pytest.raises(MediaError, match="old Flash video format"):
        engine.probe(f, ff)


# -- power, priority, memory -----------------------------------------------------------------------------------

def test_rebuilding_the_voice_and_echo_removal_count_in_the_memory_plan():
    assert sysinfo.job_memory_gb() == 1.5 and sysinfo.job_memory_gb(subtitles=True) == 2.5
    assert sysinfo.job_memory_gb(rebuild_voice=True) == 2.6
    assert sysinfo.max_parallel(4, False, cores=8, ram_gb=8, rebuild_voice=True) == 1  # 4.8 // 2.6
    assert sysinfo.max_parallel(4, False, cores=8, ram_gb=16, rebuild_voice=True) == 3  # 9.6 // 2.6
    assert sysinfo.max_parallel(4, False, cores=8, ram_gb=8, echo_track=True) == 2  # 4.8 // 1.8


def test_one_core_stays_free():
    cpus = os.cpu_count() or 2
    assert sysinfo.threads_per_job(1) == max(2, cpus - 1)


def test_the_mac_asks_sysctl_for_its_memory(monkeypatch):
    monkeypatch.setattr(sysinfo.sys, "platform", "darwin")
    monkeypatch.setattr(sysinfo.os, "name", "posix")
    monkeypatch.setattr(
        sysinfo.subprocess, "run", lambda cmd, **k: SimpleNamespace(stdout=str(16 * 2**30)) if cmd[-1] == "hw.memsize" else None
    )
    sysinfo.total_ram_gb.cache_clear()
    try:
        assert sysinfo.total_ram_gb() == 16.0
    finally:
        sysinfo.total_ram_gb.cache_clear()


def test_the_mac_stays_awake_with_caffeinate_for_as_long_as_files_are_cleaned(monkeypatch):
    calls = []

    class Proc:
        def terminate(self):
            calls.append("terminate")

        def wait(self, timeout=None):
            calls.append("wait")

    monkeypatch.setattr(sysinfo.os, "name", "posix")
    monkeypatch.setattr(sysinfo.sys, "platform", "darwin")
    monkeypatch.setattr(sysinfo.subprocess, "Popen", lambda cmd, **k: calls.append(cmd) or Proc())
    awake = sysinfo.SleepInhibitor()
    awake.acquire()
    awake.acquire()  # twice is once
    assert calls == [["caffeinate", "-i", "-w", str(os.getpid())]] and awake.held
    awake.release()
    awake.release()
    assert calls[1:] == ["terminate", "wait"] and not awake.held


def test_lowering_the_priority_asks_for_a_nicer_process(monkeypatch):
    asked = []
    monkeypatch.setattr(sysinfo.os, "name", "posix")
    monkeypatch.setattr(sysinfo.os, "nice", lambda n: asked.append(n), raising=False)
    sysinfo.lower_priority()
    assert asked == [5]


# -- Show in folder, single instance ---------------------------------------------------------------------------

def test_windows_selects_the_file_with_two_arguments_even_with_spaces_and_commas():
    path = Path("C:/My Videos/a, b-cleaned.wav")
    command = shell.reveal_command(path, "win32")
    assert command[:2] == ["explorer", "/select,"] and len(command) == 3
    assert command[2].replace("\\", "/") == "C:/My Videos/a, b-cleaned.wav"
    mac_path = Path("/Users/me/a.wav")
    assert shell.reveal_command(mac_path, "darwin") == ["open", "-R", str(mac_path)]
    assert shell.reveal_command(path, "linux") is None


_SECOND_START = """
import json, sys
from PySide6.QtCore import QCoreApplication
from audio_cleaner_desktop.gui import single_instance
app = QCoreApplication([])
sys.exit(0 if single_instance.forward(sys.argv[1], json.loads(sys.argv[2])) else 1)
"""


def _second_start(name, files):
    """What starting the app again does, as the process it really is (the files travel as ASCII JSON in the
    argument, so the command line of any OS passes them on unharmed)."""
    import json

    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    return subprocess.Popen([sys.executable, "-c", _SECOND_START, name, json.dumps(files)], env=env,
                            stdin=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def test_a_second_start_hands_its_files_to_the_first(qapp, tmp_path):
    name = f"audio-cleaner-test-{os.getpid()}"
    assert not single_instance.forward(name, ["x"], timeout_ms=200)  # nobody is listening: this one is the first
    server = single_instance.InstanceServer(name, qapp)
    got = []
    server.received.connect(got.append)
    assert server.listen()
    try:
        files = [str(tmp_path / "a b.wav"), str(tmp_path / "ü録.mp4")]
        starts = [_second_start(name, files), _second_start(name, [])]  # the second: no files, just come to the front
        # each one leaves once the window (this thread, so keep the event loop running) has answered
        assert wait_until(lambda: len(got) == 2 and all(start.poll() is not None for start in starts), timeout=60)
        for start in starts:
            assert start.returncode == 0, start.stderr.read()
        assert sorted(got, key=len) == [[], files]
    finally:
        server.close()
        server.deleteLater()
        wait_until(lambda: False, timeout=0.2)  # let the sockets go before the next test


@pytest.mark.skipif(sys.platform == "win32", reason="a Windows named pipe can drop what a sender wrote before it hung up "
                    "while the receiver was busy: the sender waits for the answer for that reason")
def test_a_second_start_that_does_not_wait_for_the_answer_still_gets_through(qapp, tmp_path):
    """An older program (or a sender that gave up waiting) hangs up right after writing."""
    name = f"audio-cleaner-test-nowait-{os.getpid()}"
    server = single_instance.InstanceServer(name, qapp)
    got = []
    server.received.connect(got.append)
    assert server.listen()
    try:
        files = [str(tmp_path / "a.wav")]
        assert single_instance.forward(name, files, ack_ms=0)
        assert wait_until(lambda: len(got) == 1, timeout=10)
        assert got == [files]
    finally:
        server.close()
        server.deleteLater()
        wait_until(lambda: False, timeout=0.2)


# -- window ---------------------------------------------------------------------------------------------------

def _entry(n: int) -> Entry:
    return Entry(name=f"f{n}.wav", source=f"/a/f{n}.wav", duration=5.0, has_video=False, settings={}, seconds_taken=1.0,
                 outputs=[f"/a/f{n}-cleaned.wav"], status=DONE, finished_at=1000.0 + n)


def test_history_draws_fifty_rows_at_first_and_more_on_request(qapp, tmp_path):
    store = HistoryStore(tmp_path / "h.db")
    for n in range(120):
        store.add(_entry(n))
    page = history_view.HistoryPage(store)
    try:
        rows = lambda: page.findChildren(history_view._HistoryRow)  # noqa: E731
        assert len(rows()) == 50
        page._show_more()
        assert len(rows()) == 100
        page._show_more()
        assert len(rows()) == 120
        assert page._stats["files"].value.text() == "120"
        page._search.setText("f11")
        page._search_changed()
        assert len(rows()) == 11 and page._shown == 50  # f11 and f110..f119
        assert store.recent_folders() == [Path("/a")]
    finally:
        store.close()


@pytest.fixture
def busy_window(qapp, tmp_path, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def runner(path, options, *, cancel, progress, **_):
        started.set()
        while not cancel.wait(0.01):
            if release.is_set():
                return engine.JobResult(path, 5.0, False)
        time.sleep(0.3)  # a step that finishes before it notices
        raise engine.Cancelled()

    monkeypatch.setattr(main_window.resources, "check", lambda: [])
    monkeypatch.setattr(engine, "probe", lambda path, ff=None: SimpleNamespace(duration=5.0, has_video=False, audio_channels=1))
    w = MainWindow([], settings=QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat),
                   history_path=tmp_path / "h.db", runner=runner)
    w.show()
    (tmp_path / "a.wav").write_bytes(b"x")
    w.add_paths([tmp_path / "a.wav"])
    assert wait_until(lambda: next(iter(w._items.values())).probed, timeout=10)
    w.start()
    assert wait_until(started.is_set, timeout=10)
    yield w
    release.set()
    w.queue.shutdown()
    w.history.close()
    w._pool.waitForDone()


def test_stopping_says_so_at_once(busy_window, monkeypatch):
    w = busy_window
    item = next(iter(w._items.values()))
    assert item.state == RUNNING and w.primary.text() == "Cancel all"
    monkeypatch.setattr(w, "_confirm", lambda *_: True)
    w.request_cancel_all()
    row = w.view.row(item.id)
    assert item.stopping and row._status.text() == "Stopping..." and not row._stop.isEnabled()
    assert w.primary.text() == "Stopping..." and not w.primary.isEnabled()
    assert wait_until(lambda: not w.queue.busy, timeout=10)
    assert not item.stopping and w.primary.text() != "Stopping..."


def test_esc_does_not_stop_the_cleaning_while_a_help_text_is_up_or_on_the_history_tab(busy_window, monkeypatch):
    w = busy_window
    asked = []
    monkeypatch.setattr(w, "_confirm", lambda title, text: asked.append(title) or False)
    monkeypatch.setattr(tips, "is_showing", lambda: True)
    w._escape()
    monkeypatch.setattr(tips, "is_showing", lambda: False)
    w.tabs.setCurrentIndex(1)
    w._escape()
    assert asked == []
    w.tabs.setCurrentIndex(0)
    w._escape()
    assert asked == ["Stop cleaning?"]


def test_history_is_drawn_only_when_its_tab_is_open(busy_window):
    w = busy_window
    drawn = []
    real = w.history_page.refresh
    w.history_page.refresh = lambda: drawn.append(1) or real()
    item = next(iter(w._items.values()))
    w._record(item, DONE, "", [])
    assert drawn == [] and w._history_stale
    w.tabs.setCurrentIndex(1)
    assert drawn == [1] and not w._history_stale
    w._record(item, DONE, "", [])
    assert drawn == [1, 1]  # visible: drawn right away


def test_quitting_does_not_wait_for_files_that_will_not_stop_and_never_touches_a_closed_history(qapp, tmp_path, monkeypatch):
    exits = []
    monkeypatch.setattr(main_window.os, "_exit", lambda code: exits.append(code))
    monkeypatch.setattr(main_window, "QUIT_WAIT_MS", 200)
    started, hold = threading.Event(), threading.Event()

    def runner(path, options, *, cancel, progress, **_):
        started.set()
        hold.wait(5)  # a step that does not look at the cancel flag
        raise engine.Cancelled()

    monkeypatch.setattr(main_window.resources, "check", lambda: [])
    monkeypatch.setattr(engine, "probe", lambda path, ff=None: SimpleNamespace(duration=5.0, has_video=False, audio_channels=1))
    w = MainWindow([], settings=QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat),
                   history_path=tmp_path / "h.db", runner=runner)
    (tmp_path / "a.wav").write_bytes(b"x")
    w.add_paths([tmp_path / "a.wav"])
    assert wait_until(lambda: next(iter(w._items.values())).probed, timeout=10)
    w.start()
    assert wait_until(started.is_set, timeout=10)
    monkeypatch.setattr(w, "_confirm", lambda *_: True)
    began = time.monotonic()
    w.close()
    assert time.monotonic() - began < 4 and exits == [0]
    hold.set()
    assert wait_until(lambda: not w.queue._running or True, timeout=1)
    QApplication.processEvents()


def test_files_the_system_opens_reach_the_window(qapp, tmp_path, monkeypatch):
    from PySide6.QtCore import QEvent, QUrl
    from PySide6.QtGui import QFileOpenEvent

    got = []
    qapp.fileOpened.connect(got.append)
    try:
        qapp.sendEvent(qapp, QFileOpenEvent(QUrl.fromLocalFile(str(tmp_path / "a.mp4"))))
    finally:
        qapp.fileOpened.disconnect(got.append)
    assert got == [str(tmp_path / "a.mp4")]  # in the platform's own form (QUrl gives C:/Users/... on Windows)
    assert QEvent.Type.FileOpen


def test_the_second_start_brings_the_window_forward_with_its_files(qapp, tmp_path):
    from audio_cleaner_desktop.gui import app as app_module

    w = MainWindow([], settings=QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat), history_path=tmp_path / "h.db")
    try:
        added = []
        w.add_paths = lambda paths: added.append(list(paths)) or len(added)
        app_module.bring_forward(w, [])
        app_module.bring_forward(w, [str(tmp_path / "x.wav")])
        assert added == [[str(tmp_path / "x.wav")]]
    finally:
        w.queue.shutdown()
        w.history.close()
        w._pool.waitForDone()


def test_about_names_the_downloaded_models_and_their_licences(qapp):
    from PySide6.QtWidgets import QLabel

    from audio_cleaner_desktop import resources
    from audio_cleaner_desktop.gui.about import AboutDialog

    dialog = AboutDialog()
    text = "\n".join(label.text() for label in dialog.findChildren(QLabel))
    for model in (resources.SPEECH_MODEL, resources.HIGHS_MODEL, resources.VOICE_MODEL):
        assert model.title in text and model.licence in text
    assert "Sidon" in text and "AP-BWE" in text


def test_about_points_to_the_licence_texts_that_came_with_the_models(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QLabel

    from audio_cleaner_desktop.gui.about import AboutDialog

    monkeypatch.setenv("ACW_DESKTOP_MODELS", str(tmp_path / "models"))
    def text() -> str:
        dialog = AboutDialog()  # kept alive while its labels are read
        return "\n".join(label.text() for label in dialog.findChildren(QLabel))

    assert str(tmp_path) not in text()  # nothing downloaded yet
    (tmp_path / "models" / "LICENSES").mkdir(parents=True)
    assert str(tmp_path / "models" / "LICENSES") in text()


WEBSITE_JOBS = Path(__file__).resolve().parents[2] / "web" / "src" / "jobs.js"


@pytest.mark.skipif(not WEBSITE_JOBS.is_file(), reason="the website's source is not part of this copy of the repository")
def test_the_settings_use_the_websites_wording():
    from audio_cleaner_desktop.gui.settings_panel import LOUDNESS_LABELS

    assert LOUDNESS_LABELS == {
        "podcast": "Podcast (−16 LUFS)", "youtube": "YouTube (−14 LUFS)", "broadcast": "Broadcast (−23 LUFS)",
        "keep": "Keep original loudness",
    }
    web = WEBSITE_JOBS.read_text(encoding="utf-8")
    for label in LOUDNESS_LABELS.values():
        assert label in web


def test_a_wrapping_check_box_toggles_by_clicking_its_text(qapp):
    from audio_cleaner_desktop.gui.widgets import WrapCheck

    box = WrapCheck("Remove speaker echo using the second audio track (your desktop sound)")
    box.resize(200, 10)
    box.show()
    qapp.processEvents()
    assert box.heightForWidth(200) > box.heightForWidth(600) and box.text().startswith("Remove speaker echo")
    box.click()
    assert box.isChecked()


def test_releasing_memory_drops_the_cached_models(monkeypatch):
    from acw_common import bwe, dpdfnet, restore, transcribe

    bwe._sessions[("x", 1)] = object()
    dpdfnet._models["x"] = ()
    restore._backends[("x",)] = object()
    transcribe._cached[("x", 1, "cpu")] = object()
    engine.release_memory()
    assert not (bwe._sessions or dpdfnet._models or restore._backends or transcribe._cached)
