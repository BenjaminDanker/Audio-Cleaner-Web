#!/usr/bin/env python3
"""Keep the --hash pins of the desktop requirement locks in step with PyPI.

    python hash_lock.py                       rewrite desktop/requirements.lock and desktop/requirements-build.lock
    python hash_lock.py LOCK [LOCK...]        rewrite just these files
    python hash_lock.py --check [LOCK...]     change nothing; exit 1 when a file differs from what PyPI has now

Each lock lists every package as `name==version`, optionally with an environment marker (`; sys_platform == ...`).
For every one of them this asks https://pypi.org/pypi/<name>/<version>/json and writes `--hash=sha256:...` for EVERY
file PyPI has for exactly that version (all wheels and the sdist), so the same lock installs on Windows, macOS arm64 and
x86_64 and Linux with `pip install --require-hashes -r LOCK`. To bump a package, change its version in the lock and
run this script (or add a new `name==version` line, including every dependency pip would otherwise pick by itself:
--require-hashes refuses anything that is not listed). Comment lines and blank lines are kept; a comment after a
requirement moves to the line above it, because a `#` cannot follow a line continuation.

A package that is not on PyPI (a local wheel, a URL) cannot be handled here: the script stops and names it. Put it in
a separate file with its own hash line (`pkg @ https://... --hash=sha256:...`) and install that file in the same pip
command as the locks.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

DESKTOP = Path(__file__).resolve().parents[1]
DEFAULT_LOCKS = (DESKTOP / "requirements.lock", DESKTOP / "requirements-build.lock")
PYPI = "https://pypi.org/pypi/{name}/{version}/json"
_REQUIREMENT = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*(?P<version>[^\s;#]+)\s*(?P<marker>;[^#]*)?(?P<comment>#.*)?$")

Fetch = Callable[[str, str], list[str]]  # (name, version) -> sha256 of every file PyPI has


class LockError(Exception):
    pass


def pypi_hashes(name: str, version: str) -> list[str]:
    """The sha256 of every file (wheels and sdist) PyPI lists for exactly name==version, sorted."""
    url = PYPI.format(name=name, version=version)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "audio-cleaner-hash-lock"}), timeout=60) as r:
                data = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise LockError(f"{name}=={version} is not on PyPI ({url}); it cannot be pinned with hashes by this script") from e
            if attempt == 3:
                raise LockError(f"{url}: HTTP {e.code}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == 3:
                raise LockError(f"{url}: {e}") from e
        time.sleep(2 * (attempt + 1))
    found = sorted({f["digests"]["sha256"] for f in data.get("urls", [])})
    if not found:
        raise LockError(f"{name}=={version} has no files on PyPI")
    return found


def parse(text: str) -> list[tuple[str, tuple[str, str, str, str] | None]]:
    """Lines of a lock as ("keep", None) for comments and blanks (the text is the first item) or
    (line, (name, version, marker, comment)) for a requirement. Existing --hash lines are dropped."""
    out: list[tuple[str, tuple[str, str, str, str] | None]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            out.append((raw, None))
            i += 1
            continue
        joined = stripped
        while joined.endswith("\\") and i + 1 < len(lines):  # a requirement with its old --hash lines
            i += 1
            joined = joined[:-1].rstrip() + " " + lines[i].strip()
        i += 1
        joined = re.sub(r"\s+--hash=\S+", "", joined).rstrip("\\ ").strip()
        m = _REQUIREMENT.match(joined)
        if not m:
            raise LockError(f"cannot read this line (only `name==version [; marker]` is supported): {joined}")
        out.append((joined, (m["name"], m["version"], (m["marker"] or "").strip(), (m["comment"] or "").strip())))
    return out


def render(text: str, fetch: Fetch = pypi_hashes) -> str:
    """`text` (a lock) with fresh --hash pins on every requirement."""
    out: list[str] = []
    for line, req in parse(text):
        if req is None:
            out.append(line)
            continue
        name, version, marker, comment = req
        if comment:
            out.append(comment)
        spec = f"{name}=={version}" + (f"; {marker.lstrip('; ').strip()}" if marker else "")
        hashes = fetch(name, version)
        out.append(spec + " \\")
        out.extend(f"    --hash=sha256:{h}" + (" \\" if n < len(hashes) - 1 else "") for n, h in enumerate(hashes))
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> int:
    check = "--check" in argv
    paths = [Path(a) for a in argv if a != "--check"] or list(DEFAULT_LOCKS)
    status = 0
    for path in paths:
        try:
            old = path.read_text(encoding="utf-8")
            new = render(old)
        except (LockError, OSError) as e:
            print(f"{path}: {e}", file=sys.stderr)
            return 2
        if new == old:
            print(f"{path}: up to date")
        elif check:
            print(f"{path}: differs from PyPI (run hash_lock.py to rewrite it)", file=sys.stderr)
            status = 1
        else:
            path.write_text(new, encoding="utf-8")
            print(f"{path}: written")
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
