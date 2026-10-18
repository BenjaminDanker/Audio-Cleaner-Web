"""Guard: nothing in the repository may name a repository file that does not exist (any more).

A workflow allowlist once kept naming a deleted script and CI failed only after a push. This scans the places that name
repo paths (workflows, the public-desktop sync lists, deploy scripts and units, docs, the READMEs) for paths that start
with a top-level folder and checks each one exists here, as a file or a directory. Paths that only exist after a build
are listed in GENERATED, with the reason.
"""
from __future__ import annotations

import glob
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".github" / "workflows").is_dir() and (p / "server").is_dir())

TOP = ("server", "web", "desktop", "common", "deploy", "native", "obs-plugin", "tools", "clients", "docs", ".github")

SCANNED = [
    ".github/workflows/*.yml", "deploy/*.sh", "deploy/vps/*", "deploy/host/*", "deploy/systemd/*.service",
    "docs/*.md", "README.md", "desktop/README.md", "obs-plugin/README.md", "tools/public-desktop/template/README.md",
]

# Paths that exist only after a build, a download or inside CI (a path matches when it starts with one of these).
GENERATED = (
    "desktop/vendor",            # filled by desktop/packaging/prepare_vendor.py (libDF, FFmpeg, models); gitignored
    "web/dist",                  # `npm run build` output
    "web/node_modules",          # npm ci
    "web/public/dfn",            # glue code copied from the production host's build (server.yml makes stand-ins)
    "obs-plugin/build",          # cmake build trees: build/, build_x64/, build_macos/ ...
    "obs-plugin/.deps",          # prebuilt OBS dependencies the obs-plugin workflow downloads and caches
    "obs-plugin/release",        # cmake install / packaging output
    "build/",                    # build tree of the workflow's own working directory
    "native/live/target",        # cargo build output
    "native/deepfilter/target",  # cargo build output
    "native/deepfilter/libDF",   # unpacked / cloned by native/deepfilter/build.sh
    "native/deepfilter/tract",   # source tree unpacked by build.sh, patched with tract-0.23.patch
)

# Real stale references found when this guard was written: (file, path). TODO: fix each reference, then delete the entry.
KNOWN_STALE: set[tuple[str, str]] = set()

# Words that look like a path but are prose: (file, text). Not stale references.
PROSE = {
    ("deploy/systemd/acw-web.service", "web/API"),  # "Audio Cleaner web/API" (the web and API service)
    ("docs/payments.md", "docs/pricing"),           # the URL path of Stripe's pricing docs
}

# A repo path: starts at a word boundary (not after "/", "~", "$", "." ...: so ~/audio-cleaner/server/x and ../x are runtime or
# relative paths, not repo paths), runs until whitespace or a quoting character.
PATH_RE = re.compile(r"(?<![\w./~$@{}<>-])((?:%s)/[^\s`'\"()\[\]<>,;|=]*)" % "|".join(re.escape(t) for t in TOP))


def _normalise(raw: str) -> str | None:
    """The path a raw match stands for, or None when it is a placeholder / not a path."""
    p = raw
    p = re.sub(r"[#?].*$", "", p)                  # anchors and query strings
    p = re.sub(r":\d+(?::\d+)?(?:-\d+)?$", "", p)  # file.py:123, file.py:12:3
    p = p.rstrip(".:!\\")  # trailing sentence punctuation
    if re.search(r"[<>{}$]|\.\.\.|…", p) or "%" in p:
        # a placeholder: only the part before it can be checked
        p = re.split(r"[<>{}$%]|\.\.\.|…", p)[0]
        p = p.rsplit("/", 1)[0] if not p.endswith("/") else p
    return p.strip("/") or None


def _exists(p: str) -> bool:
    if "*" in p or "?" in p:
        return bool(glob.glob(str(ROOT / p), recursive=True))
    return (ROOT / p).exists()


def _generated(p: str) -> bool:
    return p.startswith(GENERATED)


def _check(file: str, text: str, found: list[tuple[str, int, str]]) -> None:
    for n, line in enumerate(text.splitlines(), 1):
        for m in PATH_RE.finditer(line):
            p = _normalise(m.group(1))
            if p and "/" in p and not _generated(p) and not _exists(p) and (file, p) not in KNOWN_STALE | PROSE:
                found.append((file, n, p))


def _scanned_files() -> list[Path]:
    files: list[Path] = []
    for pattern in SCANNED:
        files += [Path(f) for f in sorted(glob.glob(str(ROOT / pattern))) if Path(f).is_file()]
    return files


def test_scanned_files_are_found():
    names = {f.relative_to(ROOT).as_posix() for f in _scanned_files()}
    for must in ("README.md", ".github/workflows/server.yml", "deploy/install.sh", "docs/desktop.md",
                 "tools/public-desktop/template/README.md"):
        assert must in names, f"{must} not scanned: has the layout moved?"


def test_files_name_only_existing_repo_paths():
    found: list[tuple[str, int, str]] = []
    for f in _scanned_files():
        _check(f.relative_to(ROOT).as_posix(), f.read_text(encoding="utf-8", errors="replace"), found)
    assert not found, "names files that do not exist in this repository:\n" + "\n".join(
        f"  {f}:{n}: {p}" for f, n, p in found)


def test_public_desktop_sync_lists_name_existing_files():
    spec = importlib.util.spec_from_file_location("public_desktop_sync", ROOT / "tools/public-desktop/sync.py")
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)
    template = ROOT / sync.TEMPLATE
    exported = sync.PLACEMENT.values()  # REQUIRED names these as they are in the public repository, not here
    missing = [f"ALLOWLIST: {p}" for p in sync.ALLOWLIST if not (ROOT / p).exists()]
    missing += [f"PLACEMENT: {sync.TEMPLATE}/{name}" for name in sync.PLACEMENT if not (template / name).is_file()]
    missing += [f"REQUIRED: {p}" for p in sync.REQUIRED if p not in exported and not (ROOT / p).is_file()]
    # REQUIRED files must also come from an ALLOWLIST root (or the template), or the export could never contain them
    missing += [f"REQUIRED outside ALLOWLIST: {p}" for p in sync.REQUIRED
                if p not in exported and not any(p == a or p.startswith(a + "/") for a in sync.ALLOWLIST)]
    # ABSENT names files of the public repository that must not be there; they say nothing about this tree.
    assert not missing, "tools/public-desktop/sync.py names files that are not in this repository:\n  " + "\n  ".join(missing)
