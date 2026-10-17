"""The desktop requirement locks: every package pinned with the hashes of all its files, and the script that
regenerates them. (No network: PyPI is replaced by a stand-in.)"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

PACKAGING = Path(__file__).resolve().parents[1] / "packaging"
if str(PACKAGING) not in sys.path:
    sys.path.insert(0, str(PACKAGING))

import hash_lock  # noqa: E402

LOCKS = hash_lock.DEFAULT_LOCKS


def _fake_hashes(name: str, version: str) -> list[str]:
    import hashlib

    return sorted(hashlib.sha256(f"{name}=={version}#{n}".encode()).hexdigest() for n in range(3))


def test_render_pins_every_requirement_with_all_hashes_keeping_markers_and_comments():
    text = (
        "# header\n"
        "anyio==4.15.1\n"
        'onnxruntime==1.31.0; sys_platform != "darwin" or platform_machine != "x86_64"\n'
        "\n"
        "# Intel Macs\n"
        'onnxruntime==1.23.2; sys_platform == "darwin" and platform_machine == "x86_64"\n'
        "livekit==1.1.20  # echo cancellation\n"
    )
    out = hash_lock.render(text, _fake_hashes)
    lines = out.splitlines()
    assert lines[0] == "# header" and "# Intel Macs" in lines and "# echo cancellation" in lines
    assert lines.index("# echo cancellation") + 1 == lines.index("livekit==1.1.20 \\")  # a comment cannot follow a `\`
    assert 'onnxruntime==1.31.0; sys_platform != "darwin" or platform_machine != "x86_64" \\' in lines
    assert 'onnxruntime==1.23.2; sys_platform == "darwin" and platform_machine == "x86_64" \\' in lines
    i = lines.index("anyio==4.15.1 \\")
    assert lines[i + 1 : i + 4] == [f"    --hash=sha256:{h}" + (" \\" if n < 2 else "") for n, h in enumerate(_fake_hashes("anyio", "4.15.1"))]
    assert hash_lock.render(out, _fake_hashes) == out  # idempotent: old hashes are replaced, not added to


def test_a_line_that_is_not_a_pin_is_refused():
    with pytest.raises(hash_lock.LockError, match="cannot read"):
        hash_lock.render("numpy>=2\n", _fake_hashes)
    with pytest.raises(hash_lock.LockError, match="cannot read"):
        hash_lock.render("pkg @ https://example.org/pkg.whl\n", _fake_hashes)


@pytest.mark.parametrize("lock", LOCKS, ids=lambda p: p.name)
def test_every_package_in_the_locks_has_hashes_and_the_files_are_unchanged_by_the_script(lock):
    text = lock.read_text(encoding="utf-8")
    entries = [req for _, req in hash_lock.parse(text) if req]
    assert entries
    have: dict[tuple[str, str], list[str]] = {}
    for block in re.split(r"\n(?=[^\s#])", text):
        m = re.match(r"([A-Za-z0-9._-]+)==(\S+?);?\s", block + " ")
        if m:
            have[(m[1], m[2])] = sorted(re.findall(r"--hash=sha256:([0-9a-f]{64})", block))
    for name, version, _marker, _comment in entries:
        assert len(have[(name, version)]) >= 1, f"{name}=={version} has no hash"
    # rewriting with the hashes already in the file gives the same file (so --check only fails when PyPI differs)
    assert hash_lock.render(text, lambda n, v: have[(n, v)]) == text
    assert "--require-hashes" in (LOCKS[0].read_text(encoding="utf-8"))


def test_the_build_and_the_ci_install_with_require_hashes():
    root = Path(__file__).resolve().parents[2]
    workflow = root / ".github" / "workflows" / "desktop.yml"  # not in the public copy, which builds nothing
    for path in (*([workflow] if workflow.is_file() else []), PACKAGING / "build_macos.sh", PACKAGING / "build_windows.ps1"):
        text = path.read_text(encoding="utf-8")
        assert "--require-hashes" in text and "requirements-build.lock" in text, path.name
        assert not re.search(r"pyinstaller==|pillow==|pytest==", text), path.name  # all pinned in the lock
