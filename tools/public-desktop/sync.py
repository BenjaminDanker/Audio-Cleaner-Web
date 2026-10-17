#!/usr/bin/env python3
"""Copy the desktop app's code, and only that, from a commit of this repository into a checkout of the public repository.

    python3 tools/public-desktop/sync.py --target PATH_TO_PUBLIC_CHECKOUT [--rev REV] [--dry-run | --no-commit]

What happens (docs/desktop.md, "Public copy of the desktop code", has the owner's side of it):

1. The files listed in ALLOWLIST, plus the files in tools/public-desktop/template/ (README, LICENSE, .gitignore,
   and .gitattributes, put where PLACEMENT says), are exported from the commit REV (default HEAD) of
   THIS repository with `git archive`, so what is copied is exactly what is committed (never the working tree).
2. The export is checked (REQUIRED files are there, nothing matches a LEAK pattern, no key-like file names). Any hit
   stops the sync before the target is touched; fix the file in this repository or, if the hit is harmless, add an
   exception to LEAK_EXCEPTIONS with the reason.
3. Every tracked file of the target is replaced by the export, so a file deleted here is deleted there. The target keeps
   nothing of its own (its history and tags stay, the files do not); .git is never touched except by the commit.
   The public repository builds nothing itself (its installers are the releases that this repository's release job
   copies there), so the export has no workflow; a `.github/workflows/desktop.yml` left in the target from the time
   it did is removed (ABSENT).
4. Changes are committed on the target's current branch as "Sync from upstream <short sha>". No change, no commit
   (running it twice in a row, or on a commit that touches nothing public, does nothing). The commit message says
   nothing about where the code comes from beyond that. This script never pushes; the workflow does.

The target must be a clean git checkout of another repository. Only the Python standard library and git are needed.
"""
from __future__ import annotations

import argparse
import fnmatch
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
TEMPLATE = "tools/public-desktop/template"

# Copied as they are (paths relative to the repository root). common/ is all of it: the desktop app imports bwe, captions,
# clarity, dfn, dpdfnet, ffmpeg, pipeline, restore and transcribe, and those import aec and gpu; its tests are part of the
# desktop build's test run. native/deepfilter/ is the libDF build (build.sh, Cargo.lock, tract-0.23.patch) that
# packaging/prepare_vendor.py and the build workflow use; its readme and tests come along. desktop/ holds the app, the
# packaging scripts, the licence texts and EULA the installers ship (desktop/packaging/licenses, EULA.txt) and its tests.
ALLOWLIST = [
    "desktop",
    "common",
    "native/deepfilter",
]

# Files that exist only in the public repository, written from TEMPLATE: template file -> where it goes.
# The public repository's own README and LICENSE are replaced by these (the old VideoDenoiser files stay in its history
# and under its v1.0.0 / v1.0.1 tags).
PLACEMENT = {
    "README.md": "README.md",
    "LICENSE": "LICENSE",
    ".gitignore": ".gitignore",
    ".gitattributes": ".gitattributes",
}

# Paths that must not exist in the public repository: it builds nothing there, so the build workflow it once had is gone
# (a tracked copy is dropped with the rest of the old tree; the check and the explicit removal make sure of it).
ABSENT = [".github/workflows/desktop.yml"]

# The sync is refused when one of these is missing from the export.
REQUIRED = [
    "desktop/pyproject.toml", "desktop/requirements.lock", "desktop/packaging/audio_cleaner.spec",
    "desktop/packaging/EULA.txt", "desktop/packaging/licenses/LGPL-3.0.txt", "desktop/packaging/fetch_dpdfnet.py",
    "common/pyproject.toml", "common/acw_common/pipeline.py", "native/deepfilter/build.sh", "native/deepfilter/Cargo.lock",
    "native/deepfilter/tract-0.23.patch", *PLACEMENT.values(),
]

PRIVATE_PATH = r"(?<![\w.-])(?:deploy|server|web|obs-plugin|terraform|docs)/[A-Za-z0-9_.-]+"
# Text that must not reach the public repository. (regular expression, why)
LEAK_PATTERNS = [
    (r"Audio-Cleaner-Web", "the name of the private repository"),
    (r"BenjaminDanker", "the owner's GitHub account name (nothing that ships names it)"),
    (r"216\.250\.115\.225", "the server's IP address"),
    (r"silver@|/home/silver|/Users/silver", "a login name or home directory of ours"),
    (r"primary-ram", "an internal host name"),
    (r"audio-cleaner/secrets|\.ssh/|(?<!\w)id_(?:rsa|ed25519|ecdsa)", "a path to keys"),
    (r"UPDATE_SIGNING_KEY|\bsecrets\.[A-Za-z_]+", "a CI secret"),
    (r"BEGIN [A-Z ]*PRIVATE KEY", "a private key"),
    (r"\bgh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}", "a GitHub token"),
    (r"\bhf_[A-Za-z0-9]{30,}", "a Hugging Face token"),
    (r"\bAKIA[0-9A-Z]{16}\b|sk_live_[A-Za-z0-9]+|sk-ant-[A-Za-z0-9_-]+", "a cloud, Stripe or API key"),
    (PRIVATE_PATH, "a path in the private repository that is not copied"),
]
# Known, harmless hits: file glob -> patterns (from LEAK_PATTERNS) that may match there.
LEAK_EXCEPTIONS = {
    # These tests assert that the shipped files do NOT name the account / the repository (they do not name it themselves).
    "desktop/tests/test_model_updates.py": [r"BenjaminDanker"],
    # The test of our own release job (skipped when the workflow has no release job, as in the public copy) reads the
    # secret's NAME; a name is not a secret.
    "desktop/tests/test_updater.py": [r"UPDATE_SIGNING_KEY|\bsecrets\.[A-Za-z_]+"],
    # Comments, docstrings and tests of the code that point at the project's other folders (deploy/, docs/research/, ...):
    # a reader of the public copy just does not find them. The rule is for the files written for the public repository.
    "desktop/**": [PRIVATE_PATH],
    "common/**": [PRIVATE_PATH],
    "native/deepfilter/**": [PRIVATE_PATH],
}
# Not scanned for text (binary or generated): the hashes in the lock files would only cause false hits.
SKIP_TEXT = ("*.png", "*.ico", "*.icns", "*.zip", "*.gz", "*.lock", "Cargo.lock")
KEY_FILE_NAMES = (".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "*.crt", "id_rsa*", "id_ed25519*", "*.sqlite3", "secrets")
FALLBACK_IDENTITY = ("Audio Cleaner sync", "noreply@users.noreply.github.com")


def run(cmd: list[str], cwd: Path | None = None, check: bool = True, env: dict | None = None) -> str:
    done = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    if check and done.returncode != 0:
        raise SystemExit(f"sync.py: {' '.join(cmd[:4])} failed ({done.returncode}): {done.stderr.strip() or done.stdout.strip()}")
    return done.stdout


def export(rev: str, paths: list[str], dest: Path) -> None:
    """`git archive rev -- paths` unpacked into dest (modes kept, so scripts stay executable)."""
    tar = subprocess.run(["git", "-C", str(SOURCE), "archive", "--format=tar", rev, "--", *paths], capture_output=True)
    if tar.returncode != 0:
        raise SystemExit(f"sync.py: git archive failed: {tar.stderr.decode(errors='replace').strip()}")
    with tarfile.open(fileobj=io.BytesIO(tar.stdout)) as t:
        t.extractall(dest, filter="data")


def build_export(rev: str, dest: Path) -> None:
    export(rev, ALLOWLIST + [f"{TEMPLATE}/{name}" for name in PLACEMENT], dest)
    for name, where in PLACEMENT.items():
        src = dest / TEMPLATE / name
        target = dest / where
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, target)
    shutil.rmtree(dest / "tools")


def files_of(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() or p.is_symlink())


def leak_scan(root: Path) -> list[str]:
    """Every problem found in the export, as 'path:line: what' (empty: clean)."""
    problems = []
    compiled = [(re.compile(pattern, re.I if "BEGIN" not in pattern else 0), pattern, why) for pattern, why in LEAK_PATTERNS]
    for path in files_of(root):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            problems.append(f"{rel}: is a symbolic link")
            continue
        if any(fnmatch.fnmatch(path.name, pat) for pat in KEY_FILE_NAMES):
            problems.append(f"{rel}: the file name looks like a key or a database")
            continue
        if any(fnmatch.fnmatch(path.name, pat) for pat in SKIP_TEXT):
            continue
        allowed = [pat for glob, pats in LEAK_EXCEPTIONS.items() if _glob(rel, glob) for pat in pats]
        text = path.read_bytes().decode("utf-8", errors="ignore")
        for number, line in enumerate(text.splitlines(), 1):
            for regex, pattern, why in compiled:
                if pattern in allowed or not regex.search(line):
                    continue
                problems.append(f"{rel}:{number}: {why}: {line.strip()[:100]}")
    return problems


def _glob(rel: str, pattern: str) -> bool:
    return fnmatch.fnmatch(rel, pattern.replace("**", "*"))  # fnmatch's * also crosses folders


def check_export(root: Path) -> list[str]:
    problems = [f"{name} is missing from the export" for name in REQUIRED if not (root / name).is_file()]
    problems += leak_scan(root)
    problems += [f"{name} must not be in the export (the public repository builds nothing)" for name in ABSENT if (root / name).exists()]
    return problems


def tracked(target: Path) -> list[str]:
    return [p for p in run(["git", "ls-files", "-z"], target).split("\0") if p]


def replace_tree(target: Path, staged: Path) -> None:
    for rel in tracked(target):
        (target / rel).unlink(missing_ok=True)
    for directory in sorted((p for p in target.rglob("*") if p.is_dir() and ".git" not in p.relative_to(target).parts),
                            key=lambda p: len(p.parts), reverse=True):
        try:
            directory.rmdir()  # only the ones that are empty now
        except OSError:
            pass
    shutil.copytree(staged, target, dirs_exist_ok=True)
    for rel in ABSENT:
        (target / rel).unlink(missing_ok=True)
    run(["git", "add", "-A"], target)


def identity(target: Path, name: str | None, email: str | None) -> tuple[str, str]:
    configured_name = run(["git", "config", "user.name"], target, check=False).strip()
    configured_email = run(["git", "config", "user.email"], target, check=False).strip()
    return (name or configured_name or FALLBACK_IDENTITY[0], email or configured_email or FALLBACK_IDENTITY[1])


def summary(target: Path) -> str:
    status = run(["git", "diff", "--cached", "--name-status", "--no-renames"], target).splitlines()
    counts = {"A": 0, "M": 0, "D": 0}
    top: dict[str, int] = {}
    for line in status:
        kind, _, path = line.partition("\t")
        counts[kind[0]] = counts.get(kind[0], 0) + 1
        key = path.split("/")[0] if "/" in path else "(root)"
        top[key] = top.get(key, 0) + 1
    where = ", ".join(f"{k}: {v}" for k, v in sorted(top.items()))
    return f"{counts['A']} added, {counts['M']} changed, {counts['D']} deleted ({where})"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Copy the desktop app's code into a checkout of the public repository.")
    ap.add_argument("--target", required=True, type=Path, help="a clean checkout of the public repository")
    ap.add_argument("--rev", default="HEAD", help="the commit of this repository to copy (default: HEAD)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="say what would change, leave the target as it was")
    mode.add_argument("--no-commit", action="store_true", help="stage the change in the target but do not commit")
    ap.add_argument("--git-name", help="author and committer name (default: the target's git config)")
    ap.add_argument("--git-email", help="author and committer e-mail (default: the target's git config)")
    ap.add_argument("--export-only", type=Path, metavar="DIR", help="only write the export to the (new, empty) DIR and scan it")
    a = ap.parse_args(argv)

    sha = run(["git", "-C", str(SOURCE), "rev-parse", "--verify", f"{a.rev}^{{commit}}"]).strip()
    short = sha[:7]

    with tempfile.TemporaryDirectory(prefix="public-desktop-") as tmp:
        staged = Path(a.export_only) if a.export_only else Path(tmp) / "export"
        if a.export_only and any(staged.glob("*")):
            raise SystemExit(f"sync.py: {staged} is not empty")
        staged.mkdir(parents=True, exist_ok=True)
        build_export(sha, staged)
        problems = check_export(staged)
        n = len(files_of(staged))
        if problems:
            print(f"sync.py: the export of {short} is not fit to publish ({len(problems)} problem(s)); nothing was changed:", file=sys.stderr)
            for p in problems:
                print(f"  {p}", file=sys.stderr)
            return 1
        print(f"export of {short}: {n} files, {sum(p.stat().st_size for p in files_of(staged)) / 1e6:.1f} MB, scan clean")
        if a.export_only:
            return 0

        target = a.target.resolve()
        if run(["git", "-C", str(target), "rev-parse", "--show-toplevel"], check=False).strip() == "" or not (target / ".git").exists():
            raise SystemExit(f"sync.py: {target} is not the top of a git checkout")
        if Path(run(["git", "rev-parse", "--show-toplevel"], target).strip()).resolve() == SOURCE or (target / "tools" / "public-desktop" / "sync.py").exists():
            raise SystemExit("sync.py: the target is the private repository itself; give the checkout of the public one")
        if run(["git", "status", "--porcelain"], target).strip():
            raise SystemExit(f"sync.py: {target} has uncommitted changes; use a fresh checkout")
        if run(["git", "rev-parse", "--verify", "-q", "HEAD"], target, check=False).strip() == "":
            raise SystemExit(f"sync.py: {target} has no commit yet")

        branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], target).strip()
        replace_tree(target, staged)
        if run(["git", "diff", "--cached", "--name-only"], target).strip() == "":
            print(f"{target.name} ({branch}) already matches {short}: nothing to commit")
            return 0
        print(f"{target.name} ({branch}): {summary(target)}")
        if a.dry_run:
            run(["git", "reset", "-q", "--hard", "HEAD"], target)
            run(["git", "clean", "-q", "-fd"], target)
            print("dry run: the target was restored")
            return 0
        if a.no_commit:
            print("staged, not committed")
            return 0
        name, email = identity(target, a.git_name, a.git_email)
        env = {**os.environ, "GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": email, "GIT_COMMITTER_NAME": name, "GIT_COMMITTER_EMAIL": email}
        run(["git", "-c", "commit.gpgsign=false", "commit", "-q", "--no-verify", "-m", f"Sync from upstream {short}"], target, env=env)
        print(f"committed {run(['git', 'rev-parse', '--short', 'HEAD'], target).strip()} on {branch}: Sync from upstream {short}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
