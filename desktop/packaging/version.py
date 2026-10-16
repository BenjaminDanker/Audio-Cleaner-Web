"""The version a CI build of the desktop app gets, and (with --write) putting it into the app.

audio_cleaner_desktop/__init__.py holds MAJOR.MINOR.0; bump MAJOR.MINOR there by hand for a bigger change. A build
of a branch is MAJOR.MINOR.<CI run number> (2.0.11, 2.0.12, ...), so every build has its own, always increasing
version (Windows' installed-apps list and the Mac installer compare them). A pushed tag desktop-vX.Y.Z builds exactly
X.Y.Z; X.Y must match the app's MAJOR.MINOR.

    python desktop/packaging/version.py            print the version (reads GITHUB_REF_TYPE, GITHUB_REF_NAME,
                                                   GITHUB_RUN_NUMBER; without them: the version in __init__.py)
    python desktop/packaging/version.py --write    also write it into __init__.py (CI does this before building)
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

INIT = Path(__file__).resolve().parents[1] / "audio_cleaner_desktop" / "__init__.py"
_LINE = re.compile(r'^__version__ = "(\d+)\.(\d+)\.(\d+)"$', re.M)


def source_version(text: str) -> tuple[int, int, int]:
    m = _LINE.search(text)
    if m is None:
        raise SystemExit(f'{INIT}: no line __version__ = "X.Y.Z"')
    return int(m[1]), int(m[2]), int(m[3])


def build_version(major: int, minor: int, patch: int, env: dict[str, str]) -> str:
    if env.get("GITHUB_REF_TYPE") == "tag":
        tag = env.get("GITHUB_REF_NAME", "")
        m = re.fullmatch(r"desktop-v(\d+)\.(\d+)\.(\d+)", tag)
        if m is None:
            raise SystemExit(f"tag {tag!r} is not desktop-vX.Y.Z")
        if (int(m[1]), int(m[2])) != (major, minor):
            raise SystemExit(f"tag {tag} does not match the app's version {major}.{minor} (audio_cleaner_desktop/__init__.py)")
        return f"{m[1]}.{m[2]}.{m[3]}"
    run = env.get("GITHUB_RUN_NUMBER")
    return f"{major}.{minor}.{int(run)}" if run else f"{major}.{minor}.{patch}"


def main(argv: list[str]) -> int:
    text = INIT.read_text(encoding="utf-8")
    version = build_version(*source_version(text), dict(os.environ))
    if "--write" in argv:
        INIT.write_text(_LINE.sub(f'__version__ = "{version}"', text, count=1), encoding="utf-8")
    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
