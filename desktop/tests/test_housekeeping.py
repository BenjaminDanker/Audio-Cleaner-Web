"""Scratch folders and leftovers, claimed output names, cancelling ffmpeg, memory, single instance,
Show in folder, stopping and quitting the window, opening files and the About text."""
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
from audio_cleaner_desktop.gui import main_window, shell, single_instance
from audio_cleaner_desktop.gui.main_window import MainWindow
from audio_cleaner_desktop.gui.items import RUNNING

import studio_helpers as sh
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


DEAD_PID = 2147483646  # (no process has it)


@pytest.fixture
def notes(monkeypatch, tmp_path):
    """A folder for the notes of unfinished temporary files (never the real app data folder), and a way to write one."""
    folder = tmp_path / "notes"
    folder.mkdir()
    monkeypatch.setattr(engine, "_unfinished_dir", lambda: folder)
    monkeypatch.delenv("ACW_DESKTOP_SCRATCH", raising=False)

    def write(target, pid=DEAD_PID, text=None):
        note = folder / f"{pid}-{os.urandom(16).hex()}"
        note.write_text(str(target) if text is None else text, encoding="utf-8")
        return note

    write.folder = folder
    return write


def test_a_note_of_a_dead_run_removes_its_temporary_file_in_a_folder_the_sweep_is_not_given(notes, tmp_path):
    elsewhere = tmp_path / "somewhere" / "else"
    elsewhere.mkdir(parents=True)
    leftover = elsewhere / f".acw-{'0123456789abcdef' * 2}.mp4"
    leftover.write_bytes(b"half a result")
    mine = elsewhere / "holiday.mp4"
    mine.write_bytes(b"x")
    note = notes(leftover)
    assert engine.sweep_leftovers() == 1  # (no folders passed, and the file is brand new: the note alone names it)
    assert not leftover.exists() and not note.exists() and mine.exists()
    assert engine.sweep_leftovers() == 0  # nothing more to do the second time


def test_a_note_of_a_dead_run_removes_its_work_folder_with_what_is_in_it(notes, tmp_path):
    work = tmp_path / "out" / ".acw-work-123-x"
    (work / "deeper").mkdir(parents=True)
    (work / "decoded.f32").write_bytes(b"x" * 100)
    (work / "deeper" / "blocks.bin").write_bytes(b"y")
    note = notes(work)
    assert engine.sweep_leftovers() == 1
    assert not work.exists() and not note.exists() and work.parent.is_dir()  # (the folder it was made in stays)


def test_a_note_of_a_run_that_is_still_going_removes_nothing(notes, tmp_path):
    leftover = tmp_path / f".acw-{'a' * 32}.wav"
    leftover.write_bytes(b"being written")
    work = tmp_path / ".acw-work-1-y"
    work.mkdir()
    n1, n2 = notes(leftover, pid=os.getpid()), notes(work, pid=os.getpid())
    assert engine.sweep_leftovers([tmp_path]) == 0  # (and the age rule does not touch them either: they are new)
    assert leftover.exists() and work.is_dir() and n1.exists() and n2.exists()


@pytest.mark.parametrize("name", [
    "holiday.mp4",                       # something of the person's
    f".acw-{'a' * 31}.mp4",              # one character short of the id this app makes
    f".acw-{'A' * 32}.mp4",              # the id is lower case hex
    f".acw-{'a' * 32}",                  # no ending
    f".acw-{'a' * 32}.mp4.bak",          # something after the ending
    f"x.acw-{'a' * 32}.mp4",             # not at the start
    "notes.txt",
])
def test_a_note_naming_another_file_removes_the_note_and_leaves_the_file(notes, tmp_path, name):
    target = tmp_path / name
    target.write_bytes(b"the person's own file")
    note = notes(target)
    assert engine.sweep_leftovers() == 0
    assert target.read_bytes() == b"the person's own file" and not note.exists()


def test_a_note_naming_a_plain_file_called_like_a_work_folder_leaves_it(notes, tmp_path):
    target = tmp_path / ".acw-work-5-notafolder"  # (only a folder is a work folder)
    target.write_bytes(b"x")
    note = notes(target)
    assert engine.sweep_leftovers() == 0 and target.exists() and not note.exists()


def test_a_note_naming_a_link_removes_the_link_not_what_it_points_at(notes, tmp_path):
    if os.name == "nt":
        pytest.skip("links need rights on Windows")
    real = tmp_path / "holiday.mp4"
    real.write_bytes(b"mine")
    folder = tmp_path / "photos"
    folder.mkdir()
    (folder / "pic.jpg").write_bytes(b"p")
    link_file = tmp_path / f".acw-{'c' * 32}.mp4"
    link_file.symlink_to(real)
    link_dir = tmp_path / ".acw-work-9-z"
    link_dir.symlink_to(folder, target_is_directory=True)
    n1, n2 = notes(link_file), notes(link_dir)
    engine.sweep_leftovers()
    assert not link_file.is_symlink() and not link_dir.is_symlink() and not n1.exists() and not n2.exists()
    assert real.read_bytes() == b"mine" and (folder / "pic.jpg").read_bytes() == b"p"


def test_a_note_whose_file_is_already_gone_is_just_removed(notes, tmp_path):
    note = notes(tmp_path / f".acw-{'d' * 32}.mp4")
    assert engine.sweep_leftovers() == 0 and not note.exists()


def test_an_unreadable_or_empty_note_does_not_raise_and_does_not_delete_anything(notes, tmp_path):
    keep = tmp_path / "keep"
    keep.mkdir()
    (keep / "a.txt").write_bytes(b"a")
    bad = [
        notes(None, text=""),  # empty
        notes(None, text="   \n"),  # blank
        notes(None, text="a\x00b"),  # an embedded NUL
        notes(None, text=str(keep)),  # a folder that is not a work folder
        notes(None, text=str(tmp_path / ("x" * 300) / "y")),  # a name too long for the file system
    ]
    garbage = notes(None, text="")
    garbage.write_bytes(b"\xff\xfe\x00\xc3\x28")  # not text
    (notes.folder / f"{DEAD_PID}-isadir").mkdir()  # not a file at all
    (notes.folder / "no-pid-in-the-name").write_text(str(tmp_path / "nothing"), encoding="utf-8")
    locked = notes(None, text=str(tmp_path / "nothing"))
    locked.chmod(0)  # unreadable
    try:
        assert engine.sweep_leftovers([tmp_path]) == 0
    finally:
        if locked.exists():
            locked.chmod(0o600)
    assert (keep / "a.txt").read_bytes() == b"a"
    assert not any(n.exists() for n in [*bad, garbage])  # (every note of a dead run goes, whatever it holds)


def test_a_missing_notes_folder_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "_unfinished_dir", lambda: tmp_path / "never-made")
    monkeypatch.delenv("ACW_DESKTOP_SCRATCH", raising=False)
    assert engine.sweep_leftovers() == 0


def test_notes_are_written_for_the_run_and_forgotten(monkeypatch, tmp_path):
    folder = tmp_path / "n" / "deeper"
    monkeypatch.setattr(engine, "_unfinished_dir", lambda: folder)  # (made when the first note is written)
    target = tmp_path / f".acw-{'e' * 32}.wav"
    note = engine.note_unfinished(target)
    assert note is not None and note.parent == folder
    pid, _, rest = note.name.partition("-")
    assert pid == str(os.getpid()) and len(rest) == 32 and int(rest, 16) >= 0
    assert note.read_text(encoding="utf-8") == str(target)
    other = engine.note_unfinished(target)
    assert other != note  # (one per file: two runs of the same process do not share)
    monkeypatch.chdir(tmp_path)
    relative = engine.note_unfinished(Path("rel.wav"))
    assert Path(relative.read_text(encoding="utf-8")) == tmp_path / "rel.wav"  # (an absolute path: the sweep runs from anywhere)
    for n in (note, other, relative):
        engine.forget_unfinished(n)
        assert not n.exists()
    engine.forget_unfinished(note)  # (twice is fine)
    engine.forget_unfinished(None)  # (and no note at all, when it could not be written)


def test_a_note_that_cannot_be_written_is_not_an_error(monkeypatch, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_bytes(b"x")
    monkeypatch.setattr(engine, "_unfinished_dir", lambda: blocker / "notes")  # (a folder cannot be made under a file)
    assert engine.note_unfinished(tmp_path / "a.wav") is None
    engine.forget_unfinished(None)


def test_a_run_writes_down_its_temporary_file_and_folder_while_it_works_and_leaves_no_note(speech_wav, tmp_path, monkeypatch, quick_clean, notes):
    pytest.importorskip("numpy")
    from audio_cleaner_desktop import resources

    if resources.check():
        pytest.skip("native components missing")
    out = tmp_path / "out"
    seen = []
    real = engine.note_unfinished

    def record(path):
        note = real(path)
        # what the sweep would be told: the kind of name it accepts, and the thing is there to remove
        seen.append((Path(path).name, Path(path).parent, Path(path).is_dir(), Path(path).exists(), note is not None and note.is_file()))
        return note

    monkeypatch.setattr(engine, "note_unfinished", record)
    monkeypatch.setattr(engine, "scratch_root", lambda: None)  # (no scratch folder, as on Linux: the work folder is made beside the result)
    result = engine.process(speech_wav, engine.JobOptions(audio_format="flac"), out_dir=out)
    files = [x for x in seen if not x[2]]
    folders = [x for x in seen if x[2]]
    assert len(files) == 1 and len(folders) == 1
    assert engine._LEFTOVER_FILE.match(files[0][0]) and files[0][1] == out and files[0][3] and files[0][4]
    assert folders[0][0].startswith(f".acw-work-{os.getpid()}-") and folders[0][1] == out and folders[0][4]
    assert sorted(p.name for p in out.iterdir()) == [result.output.name]
    assert list(notes.folder.iterdir()) == []


def test_a_stopped_or_failed_run_leaves_no_note(speech_wav, tmp_path, monkeypatch, quick_clean, notes):
    from audio_cleaner_desktop import editor_job, resources

    if resources.check():
        pytest.skip("native components missing")
    out = tmp_path / "out"
    written = []
    real = engine.note_unfinished
    monkeypatch.setattr(engine, "note_unfinished", lambda path: written.append(Path(path).name) or real(path))
    monkeypatch.setattr(engine, "scratch_root", lambda: None)  # (as on Linux: the work folder is beside the result, so it is noted too)
    for error in (engine.Cancelled(), RuntimeError("the render broke")):  # (stopped, and failed: both in the middle of the work)
        def render(*a, **k):
            raise error

        monkeypatch.setattr(editor_job, "render_timeline", render)
        before = len(written)
        with pytest.raises(type(error)):
            engine.process(speech_wav, engine.JobOptions(audio_format="flac"), out_dir=out)
        assert len(written) == before + 2  # (the temporary file and the work folder were both written down ...)
        assert list(notes.folder.iterdir()) == [] and list(out.iterdir()) == []  # (... and are gone, with their notes)


def test_a_job_writes_its_work_files_to_the_scratch_folder(speech_wav, tmp_path, monkeypatch, quick_clean):
    pytest.importorskip("numpy")
    from audio_cleaner_desktop import resources

    if resources.check():
        pytest.skip("native components missing")
    monkeypatch.setenv("ACW_DESKTOP_SCRATCH", str(tmp_path / "scratch"))
    noted = []
    real_note = engine.note_unfinished
    monkeypatch.setattr(engine, "note_unfinished", lambda path: noted.append(Path(path)) or real_note(path))
    seen = []
    real = engine.shutil.rmtree

    def record(work, *args, **kwargs):
        # rmtree is process-wide: a cleanup thread left by an earlier test can remove its own folder meanwhile
        if Path(work).resolve().is_relative_to(tmp_path.resolve()):
            seen.append(Path(work).parent)
        return real(work, *args, **kwargs)

    monkeypatch.setattr(engine.shutil, "rmtree", record)
    out = tmp_path / "out"
    result = engine.process(speech_wav, engine.JobOptions(audio_format="flac"), out_dir=out)
    assert seen == [tmp_path / "scratch"]
    assert [p.parent for p in noted] == [out]  # (only the temporary result file is noted: a leftover in the scratch folder is swept by age)
    assert sorted(p.name for p in out.iterdir()) == [result.output.name]
    assert list((tmp_path / "scratch").iterdir()) == []
    assert list(engine._unfinished_dir().iterdir()) == []  # (a normal run leaves no note behind)


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

@pytest.fixture
def busy_window(qapp, tmp_path, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def runner(path, options, *, cancel, progress, **_):
        started.set()
        for _ in range(3000):  # (until it is stopped or let go, at most 30 s: a failed test must not leave it spinning)
            if cancel.wait(0.01):
                break
            if release.is_set():
                return engine.JobResult(path, 5.0, False)
        time.sleep(0.3)  # a step that finishes before it notices
        raise engine.Cancelled()

    monkeypatch.setattr(main_window.resources, "check", lambda: [])
    w = MainWindow([], settings=QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat),
                   history_path=tmp_path / "h.db", runner=runner)
    w.fakes = sh.Fakes(5.0).install(w)
    w.show()
    (tmp_path / "a.wav").write_bytes(b"x")
    w.add_paths([tmp_path / "a.wav"])
    assert wait_until(lambda: next(iter(w._items.values())).probed, timeout=10)
    w.start()
    assert wait_until(started.is_set, timeout=10)
    yield w
    release.set()
    w.queue.shutdown()
    sh.shut(w)


def test_stopping_says_so_at_once(busy_window, monkeypatch):
    w = busy_window
    item = next(iter(w._items.values()))
    assert item.state == RUNNING
    monkeypatch.setattr(w, "_confirm", lambda *_: True)
    w.request_cancel_all()
    assert item.stopping  # at once, before the step has noticed
    assert wait_until(lambda: not w.queue.busy, timeout=10)
    assert not item.stopping


def test_quitting_does_not_wait_for_files_that_will_not_stop_and_never_touches_a_closed_history(qapp, tmp_path, monkeypatch):
    exits = []
    monkeypatch.setattr(main_window.os, "_exit", lambda code: exits.append(code))
    monkeypatch.setattr(main_window, "QUIT_WAIT_MS", 200)
    started, hold = threading.Event(), threading.Event()

    def runner(path, options, *, cancel, progress, **_):
        started.set()
        hold.wait(30)  # a step that does not look at the cancel flag (released by the test below; closing must not wait for it)
        raise engine.Cancelled()

    monkeypatch.setattr(main_window.resources, "check", lambda: [])
    w = MainWindow([], settings=QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat),
                   history_path=tmp_path / "h.db", runner=runner)
    w.fakes = sh.Fakes(5.0).install(w)
    (tmp_path / "a.wav").write_bytes(b"x")
    w.add_paths([tmp_path / "a.wav"])
    assert wait_until(lambda: next(iter(w._items.values())).probed, timeout=10)
    w.start()
    assert wait_until(started.is_set, timeout=10)
    monkeypatch.setattr(w, "_confirm", lambda *_: True)
    began = time.monotonic()
    w.close()
    assert time.monotonic() - began < 10 and exits == [0]
    hold.set()
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


def test_about_names_the_downloaded_models_and_where_their_licence_texts_are(qapp, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QLabel

    from audio_cleaner_desktop import models, resources
    from audio_cleaner_desktop.gui.about import AboutDialog

    def texts():
        dialog = AboutDialog()
        return "\n".join(label.text() for label in dialog.findChildren(QLabel))

    text = texts()
    for model in (resources.SPEECH_MODEL, resources.VOICE_MODEL):
        assert model.title in text and model.licence in text
    assert "licence texts that came with the models" not in text  # nothing downloaded yet
    # an installed model has its own LICENSES folder (also the installer's copy), the old shared one still counts
    (tmp_path / "voice" / "LICENSES").mkdir(parents=True)
    monkeypatch.setattr(models, "location", lambda m: tmp_path / "voice" if m is resources.VOICE_MODEL else None)
    (models.models_dir() / "LICENSES").mkdir(parents=True)
    text = texts()
    assert str(tmp_path / "voice" / "LICENSES") in text and str(models.models_dir() / "LICENSES") in text
