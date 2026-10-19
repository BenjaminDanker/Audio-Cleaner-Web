"""The trees of "Automatic best settings" (common/acw_common/reco_model.json) have to be in every build of the app:
PyInstaller does not collect the data files of a package by itself, so the spec lists it, `pip install ../common` takes it
as package data, and the copy of the code that goes to the public repository (tools/public-desktop/sync.py) has to carry it
and its new code without tripping the leak scan."""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest

from acw_common import analysis, reco_model

ROOT = Path(__file__).resolve().parents[2]
SPEC = (ROOT / "desktop" / "packaging" / "audio_cleaner.spec").read_text(encoding="utf-8")
SYNC = ROOT / "tools" / "public-desktop" / "sync.py"
needs_sync = pytest.mark.skipif(not SYNC.is_file(), reason="the sync script is only in the private repository")


def test_the_spec_puts_the_model_next_to_the_code_that_reads_it():
    """PyInstaller's datas: (source file, folder in the app). reco_model.py finds its file next to itself, and in the
    app that is <app folder>/acw_common/."""
    match = re.search(r'\(str\(ROOT / "common" / "acw_common" / "reco_model\.json"\), "acw_common"\)', SPEC)
    assert match, "the spec does not ship common/acw_common/reco_model.json into acw_common/"
    assert reco_model.MODEL_FILE.name == "reco_model.json" and reco_model.MODEL_FILE.parent.name == "acw_common"
    assert reco_model.MODEL_FILE.is_file() and reco_model.MODEL_FILE.parent == Path(analysis.__file__).parent
    datas = SPEC[SPEC.index("datas = ["):SPEC.index("hiddenimports = [")]
    assert "reco_model.json" in datas  # in the list that goes into the build, not a comment elsewhere


def test_a_package_install_of_common_takes_the_model_too():
    pyproject = (ROOT / "common" / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'\[tool\.setuptools\.package-data\]\s*\nacw_common = \[[^\]]*"reco_model\.json"', pyproject)


def test_the_model_in_the_build_is_complete_and_loads():
    model = json.loads(reco_model.MODEL_FILE.read_text(encoding="utf-8"))
    assert reco_model.load() == model and 400 < len(model["noise"]["trees"]) and len(model["rebuild"]["trees"]) > 50
    assert reco_model.MODEL_FILE.stat().st_size > 100_000  # not an empty or cut-off file


def test_the_selftest_checks_the_model_and_the_measurements():
    from audio_cleaner_desktop import cli

    cli._check_recommender()  # raises when the trees are missing or decide nothing


@needs_sync
def test_the_public_copy_takes_the_model_and_the_new_code(tmp_path):
    spec = importlib.util.spec_from_file_location("sync_public_desktop", SYNC)
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)
    assert "common" in sync.ALLOWLIST and "desktop" in sync.ALLOWLIST  # all of common/, so the file and the module go
    export = tmp_path / "export"
    ignore = shutil.ignore_patterns("__pycache__", "vendor", "node_modules")
    shutil.copytree(ROOT / "common", export / "common", ignore=ignore)
    for part in ("audio_cleaner_desktop", "tests", "packaging"):
        shutil.copytree(ROOT / "desktop" / part, export / "desktop" / part, ignore=ignore)
    assert (export / "common" / "acw_common" / "reco_model.json").is_file()
    assert sync.leak_scan(export) == []  # nothing the public repository must not have, in the files added for this
