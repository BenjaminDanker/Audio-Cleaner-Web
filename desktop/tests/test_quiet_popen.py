"""subprocess.Popen starts programs without a console window on Windows (tested here on any platform: the real
Popen is replaced by a recorder, so nothing is started)."""
from __future__ import annotations

import subprocess
import sys

import pytest

from audio_cleaner_desktop import quiet_popen as q


@pytest.fixture
def popen(monkeypatch):
    """subprocess.Popen.__init__ replaced by a recorder, then patched by install(force=True)."""
    calls = []

    def recorder(self, args, *rest, **kwargs):
        calls.append((args, rest, kwargs))
        self.args, self.returncode = args, 0

    monkeypatch.setattr(subprocess.Popen, "__init__", recorder)
    assert q.install(force=True)
    return calls


def test_no_window_is_the_default(popen):
    subprocess.Popen(["cmd", "/c", "ver"])
    assert popen[-1][2]["creationflags"] == q.CREATE_NO_WINDOW
    subprocess.Popen("ver", shell=True, creationflags=0x200)  # CREATE_NEW_PROCESS_GROUP is kept, the flag is added
    assert popen[-1][2]["creationflags"] == 0x200 | q.CREATE_NO_WINDOW


@pytest.mark.parametrize("flag", [q.CREATE_NEW_CONSOLE, q.DETACHED_PROCESS, q.CREATE_NO_WINDOW, q.DETACHED_PROCESS | 0x200])
def test_a_console_flag_of_the_caller_is_left_alone(popen, flag):
    subprocess.Popen(["x"], creationflags=flag)
    assert popen[-1][2]["creationflags"] == flag


def test_positional_creationflags_and_explorer_are_left_alone(popen):
    subprocess.Popen(["x"], -1, None, None, None, None, None, True, False, None, None, False, None, 0x10)
    args, rest, kwargs = popen[-1]
    assert "creationflags" not in kwargs and rest[12] == 0x10
    for name in (["explorer", "/select,", "C:\\a b\\c.wav"], "C:\\Windows\\explorer.exe", ["C:/Windows/Explorer.EXE"]):
        subprocess.Popen(name)
        assert "creationflags" not in popen[-1][2]


def test_run_and_check_output_go_through_it(popen, monkeypatch):
    """subprocess.run and check_output (platform uses it for `ver`) both build a Popen."""
    monkeypatch.setattr(subprocess.Popen, "__enter__", lambda self: self, raising=False)
    monkeypatch.setattr(subprocess.Popen, "__exit__", lambda *a: None, raising=False)
    monkeypatch.setattr(subprocess.Popen, "communicate", lambda self, *a, **k: (b"", b""), raising=False)
    monkeypatch.setattr(subprocess.Popen, "poll", lambda self: 0, raising=False)
    monkeypatch.setattr(subprocess.Popen, "wait", lambda self, *a, **k: 0, raising=False)
    subprocess.run(["a"])
    subprocess.check_output("ver", shell=True, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
    assert [c[2]["creationflags"] for c in popen] == [q.CREATE_NO_WINDOW] * 2


def test_install_only_on_windows_and_only_once(monkeypatch):
    original = subprocess.Popen.__init__
    while getattr(original, q._MARK, False):  # on Windows the app import has installed it already: take it off
        original = original.__wrapped__
    monkeypatch.setattr(subprocess.Popen, "__init__", original)  # whatever was there is put back at the end
    monkeypatch.setattr(sys, "platform", "linux")
    assert q.install() is False and subprocess.Popen.__init__ is original
    monkeypatch.setattr(sys, "platform", "darwin")
    assert q.install() is False and subprocess.Popen.__init__ is original
    monkeypatch.setattr(sys, "platform", "win32")
    assert q.install() is True and subprocess.Popen.__init__ is not original
    patched = subprocess.Popen.__init__
    assert q.install() is False and subprocess.Popen.__init__ is patched  # not wrapped twice


def test_the_launcher_installs_it_before_anything_else():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "packaging" / "launcher.py").read_text(encoding="utf-8")
    code = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    assert code[0] == "from audio_cleaner_desktop import quiet_popen"
    assert src.index("quiet_popen.install()") < src.index("import multiprocessing")
