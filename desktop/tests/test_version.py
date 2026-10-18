"""CI build versions: MAJOR.MINOR from the app, the run number last; a tag builds exactly its version. With --write a
build is marked as a release build and names the repository its releases are in, so it downloads from its own release."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packaging"))
import version  # noqa: E402


def test_branch_builds_use_the_run_number_and_local_builds_the_source():
    assert version.build_version(2, 0, 0, {"GITHUB_REF_TYPE": "branch", "GITHUB_RUN_NUMBER": "12"}) == "2.0.12"
    assert version.build_version(2, 1, 0, {}) == "2.1.0"


def test_a_tag_builds_exactly_its_version_and_must_match_major_minor():
    env = {"GITHUB_REF_TYPE": "tag", "GITHUB_REF_NAME": "desktop-v2.0.40", "GITHUB_RUN_NUMBER": "99"}
    assert version.build_version(2, 0, 0, env) == "2.0.40"
    with pytest.raises(SystemExit):
        version.build_version(2, 1, 0, env)
    with pytest.raises(SystemExit):
        version.build_version(2, 0, 0, {**env, "GITHUB_REF_NAME": "desktop-v2.0"})


def test_the_app_holds_a_plain_three_part_version():
    assert version.source_version(version.INIT.read_text(encoding="utf-8"))[2] == 0


# ---- --write: a release build, and where its releases are ---------------------------------------------------------------
_CI_VARS = ("GITHUB_REF_TYPE", "GITHUB_REF_NAME", "GITHUB_RUN_NUMBER", "ACW_RELEASES_REPO")


def _values(text: str) -> dict:
    found: dict = {}
    exec(compile(text, "__init__.py", "exec"), found)  # the app's __init__.py is three plain assignments
    return {k: found[k] for k in ("__version__", "__releases__")}


@pytest.fixture
def init(tmp_path, monkeypatch):
    """A copy of the app's real __init__.py for version.py to write into, and no CI variables."""
    path = tmp_path / "__init__.py"
    path.write_text(version.INIT.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(version, "INIT", path)
    for name in _CI_VARS:
        monkeypatch.delenv(name, raising=False)
    return path


def test_write_marks_a_release_build_and_names_the_repository_its_releases_are_in(init, monkeypatch, capsys):
    before = init.read_text(encoding="utf-8")
    major, minor, _ = version.source_version(before)
    monkeypatch.setenv("GITHUB_RUN_NUMBER", "41")
    monkeypatch.setenv("ACW_RELEASES_REPO", "someone-else/Trial.Repo_2")
    assert version.main(["--write"]) == 0
    assert capsys.readouterr().out == f"{major}.{minor}.41\n"
    after = init.read_text(encoding="utf-8")
    assert _values(after) == {"__version__": f"{major}.{minor}.41", "__releases__": "someone-else/Trial.Repo_2"}
    changed = [(a, b) for a, b in zip(before.splitlines(), after.splitlines()) if a != b]
    assert len(changed) == 2 and len(before.splitlines()) == len(after.splitlines())  # nothing else is touched
    # writing again (a rerun of the step) changes nothing more
    assert version.main(["--write"]) == 0
    assert init.read_text(encoding="utf-8") == after


def test_write_without_a_repository_keeps_the_default(init, monkeypatch):
    before = _values(init.read_text(encoding="utf-8"))
    monkeypatch.setenv("ACW_RELEASES_REPO", "")  # an unset repository variable reads as empty
    assert version.main(["--write"]) == 0
    after = _values(init.read_text(encoding="utf-8"))
    assert after == before


@pytest.mark.parametrize("bad", [
    "owner", "owner/", "/name", "a/b/c", "owner/name;rm -rf ~", 'owner/na"me', "owner/name\nx = 1", " owner/name",
    "https://github.com/owner/name", "owner/name/", "own er/name", "owner/name\\", "ow_ner/name",
])
def test_write_refuses_a_repository_that_is_not_owner_slash_name(init, monkeypatch, bad):
    before = init.read_bytes()
    monkeypatch.setenv("GITHUB_RUN_NUMBER", "41")
    monkeypatch.setenv("ACW_RELEASES_REPO", bad)
    with pytest.raises(SystemExit, match="is not owner/name"):
        version.main(["--write"])
    assert init.read_bytes() == before  # nothing is written, not even the version


def test_printing_the_version_writes_nothing(init, monkeypatch, capsys):
    before = init.read_bytes()
    monkeypatch.setenv("GITHUB_RUN_NUMBER", "7")
    monkeypatch.setenv("ACW_RELEASES_REPO", "not a repository")  # only --write reads it
    assert version.main([]) == 0
    assert capsys.readouterr().out.strip().endswith(".7") and init.read_bytes() == before


_PROBE = """
import json, sys
sys.path[:0] = sys.argv[1:3]
from audio_cleaner_desktop import __releases__, __version__, config, updater
import installer_models
print(json.dumps({
    "releases": __releases__, "version": __version__, "manifest": updater.MANIFEST_URL,
    "windows": installer_models.render(), "mac": installer_models.render_mac(),
}))
"""


def _probe(desktop: Path) -> dict:
    r = subprocess.run([sys.executable, "-I", "-c", _PROBE, str(desktop), str(desktop / "packaging")],
                       capture_output=True, text=True, env={k: v for k, v in os.environ.items() if k not in _CI_VARS})
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_a_release_build_updates_from_its_repository_and_its_installers_download_models_elsewhere(tmp_path):
    source = Path(__file__).resolve().parents[1]
    desktop = tmp_path / "desktop"
    shutil.copytree(source / "audio_cleaner_desktop", desktop / "audio_cleaner_desktop", ignore=shutil.ignore_patterns("__pycache__"))
    (desktop / "packaging").mkdir()
    for name in ("version.py", "installer_models.py"):
        shutil.copy2(source / "packaging" / name, desktop / "packaging" / name)

    env = {k: v for k, v in os.environ.items() if k not in _CI_VARS}
    env.update(GITHUB_RUN_NUMBER="41", ACW_RELEASES_REPO="someone/Trial-Repo")
    r = subprocess.run([sys.executable, "-I", str(desktop / "packaging" / "version.py"), "--write"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    v = r.stdout.strip()
    assert v.endswith(".41")

    built = _probe(desktop)
    assert built["releases"] == "someone/Trial-Repo" and built["version"] == v
    # updates: the newest release's signed update.json (its installer then comes from that version's release)
    assert built["manifest"] == "https://github.com/someone/Trial-Repo/releases/latest/download/update.json"
    for text in (built["windows"], built["mac"]):  # the models do not depend on where the releases are
        urls = {u.rstrip(":.,") for u in re.findall(r"https://[^\s'\"]+", text)}
        assert urls and all(u.startswith("https://huggingface.co/") for u in urls), urls
        assert "Trial-Repo" not in text and "github.com" not in text
