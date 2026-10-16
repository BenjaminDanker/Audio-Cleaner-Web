"""CI build versions: MAJOR.MINOR from the app, the run number last; a tag builds exactly its version."""
from __future__ import annotations

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
