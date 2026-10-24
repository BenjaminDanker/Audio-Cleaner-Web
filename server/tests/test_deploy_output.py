"""The deploy's output: colour only on a terminal (and never with NO_COLOR), the end-of-deploy summary of everything red or
yellow, the Node preflight, and the smoke test's check of the free tools' code (deploy/color.sh, lib.sh, smoke.py)."""
from __future__ import annotations

import importlib.util
import os
import pty
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "deploy"
pytestmark = pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")


def run_piped(script: str, env=None):
    e = {k: v for k, v in os.environ.items() if k not in ("ACW_COLOR", "NO_COLOR", "ACW_SUMMARY")}
    e.update(env or {})
    return subprocess.run(["bash", "-c", script], env=e, capture_output=True, text=True)


def run_on_terminal(script: str, env=None) -> str:
    """The output of the script with stdout and stderr on a pseudo terminal."""
    e = {k: v for k, v in os.environ.items() if k not in ("ACW_COLOR", "NO_COLOR", "ACW_SUMMARY")}
    e.update({"TERM": "xterm", **(env or {})})
    pid, fd = pty.fork()
    if pid == 0:
        os.execvpe("bash", ["bash", "-c", script], e)
    out = b""
    while True:
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            break
        if not chunk:
            break
        out += chunk
    os.waitpid(pid, 0)
    return out.decode()


SAY = f'. "{DEPLOY}/color.sh"; say_ok good; say_warn careful; say_fail broken; echo plain'


def test_no_colour_when_piped():
    r = run_piped(SAY)
    assert "\033" not in r.stdout + r.stderr
    assert "good" in r.stdout and "careful" in r.stderr and "broken" in r.stderr


def test_colours_on_a_terminal_and_not_with_no_color():
    out = run_on_terminal(SAY)
    assert "\033[32mgood" in out and "\033[33mcareful" in out and "\033[1;31mbroken" in out
    assert "\033[0mplain" not in out and "plain" in out          # plain stays plain
    assert "\033" not in run_on_terminal(SAY, {"NO_COLOR": "1"})
    assert "\033" not in run_on_terminal(SAY, {"ACW_COLOR": "0"})
    assert "\033[" in run_on_terminal(SAY, {"NO_COLOR": ""})     # an empty NO_COLOR does not count


def test_summary_lists_red_and_yellow_again(tmp_path):
    summary = tmp_path / "s"
    r = run_piped(f'{{ {SAY}; }} >/dev/null 2>&1; . "{DEPLOY}/color.sh"; say_warn "second look" 2>/dev/null; acw_print_summary "$ACW_SUMMARY" 1',
                  {"ACW_SUMMARY": str(summary)})
    assert "1 to fix (red), 2 to look at (yellow)" in r.stdout
    assert "RED    broken" in r.stdout and "YELLOW careful" in r.stdout and "YELLOW second look" in r.stdout
    assert "good" not in r.stdout
    clean = run_piped(f'. "{DEPLOY}/color.sh"; acw_print_summary /nonexistent 0')
    assert "nothing needs attention" in clean.stdout


def _fake_node(tmp_path: Path, major: int) -> Path:
    d = tmp_path / f"node{major}"
    d.mkdir()
    node = d / "node"
    node.write_text(f'#!/bin/sh\ncase "$1" in -p) echo {major} ;; --version) echo v{major}.1.0 ;; esac\n')
    node.chmod(node.stat().st_mode | stat.S_IEXEC)
    return d


def test_require_node_names_the_node_it_found_and_the_fix(tmp_path):
    lib = f'ACW_SERVICE_USER="" . "{DEPLOY}/lib.sh"; require_node'
    old = run_piped(lib, {"PATH": f"{_fake_node(tmp_path, 18)}:{os.environ['PATH']}"})
    assert old.returncode == 1
    assert "Node v18.1.0" in old.stderr and "node18/node" in old.stderr and "Node 20 or newer" in old.stderr
    assert "sudo apt-get install -y nodejs" in old.stderr and "nodesource" in old.stderr
    assert run_piped(lib, {"PATH": f"{_fake_node(tmp_path, 22)}:{os.environ['PATH']}"}).returncode == 0


def test_install_checks_node_before_it_touches_the_server_and_does_not_swallow_a_failed_publish():
    text = (DEPLOY / "install.sh").read_text()
    assert text.index("require_node") < text.index('STAGE="$(stage_for_acw')
    assert "if ! bash \"$REPO/deploy/publish-web.sh\"" in text           # `bash publish-web.sh` alone is ignored inside a || list
    assert "the website was NOT published" in text


def _smoke():
    spec = importlib.util.spec_from_file_location("acw_smoke", DEPLOY / "smoke.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # importing runs no check (nothing touches the network)
    return mod


class _Headers(dict):
    def get(self, k, d=None):
        return super().get(k, d)


@pytest.mark.parametrize("status,ctype,body,ok", [
    (200, "text/javascript; charset=utf-8", b"export default 1", True),   # the free tools work without an account
    (401, "application/json", b"{}", False),                              # a sign-in here would break them
    (200, "text/html; charset=utf-8", b"<!doctype html>", False),         # the page instead of the file
])
def test_smoke_free_tools_code_must_be_public_javascript(monkeypatch, status, ctype, body, ok):
    smoke = _smoke()
    monkeypatch.setattr(smoke, "fetch", lambda url, **kw: (status, _Headers({"Content-Type": ctype}), body))
    names = [n for n, _ in smoke.checks]
    assert not any("needs a sign-in" in n for n in names) and any("public" in n for n in names)
    fn = next(f for n, f in smoke.checks if "public" in n)
    if ok:
        fn()
    else:
        with pytest.raises(AssertionError):
            fn()
