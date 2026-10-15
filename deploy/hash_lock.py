#!/usr/bin/env python3
"""Pin every package of a requirements lock to the sha256 of every file PyPI lists for that exact version
(SEC-26), so `pip install --require-hashes -r FILE` installs nothing that is not byte for byte what was locked.

  python3 deploy/hash_lock.py server/requirements.lock             rewrite the file with --hash lines (fetches PyPI)
  python3 deploy/hash_lock.py --check server/requirements.lock     fetch PyPI again, exit 1 if a hash list differs
  python3 deploy/hash_lock.py --verify server/requirements.lock    offline: every requirement is pinned with == and hashed

To change a package: edit its `name==version` line (leave or delete its hashes), run the first command, review the
diff. Comments above a requirement are kept (a comment cannot follow the `\\` of a continued line). All files PyPI
lists for the version are included (wheels for every platform, and the sdist), so one lock serves Linux, macOS and
Windows; pip only ever picks the file that fits the machine. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REQ = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[A-Za-z0-9,._-]+\])?==([A-Za-z0-9._+!-]+)\s*(?:;[^#]*)?")
HASH = re.compile(r"--hash=sha256:([0-9a-f]{64})")


def parse(text: str) -> list[dict]:
    """Items in file order: {"comment": str} for comment/blank lines, {"name", "extras", "version", "hashes", "marker"}."""
    items: list[dict] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            items.append({"comment": line})
            i += 1
            continue
        joined = line
        while joined.rstrip().endswith("\\") and i + 1 < len(lines):
            i += 1
            joined = joined.rstrip()[:-1] + " " + lines[i].strip()
        i += 1
        trailing = ""
        if " #" in joined:
            joined, _, trailing = joined.partition(" #")
            trailing = trailing.strip()
        body = joined.strip().rstrip("\\").strip()
        m = REQ.match(body)
        if not m:
            raise SystemExit(f"not a pinned requirement (name==version): {line!r}")
        marker = ""
        if ";" in body:
            marker = body.split(";", 1)[1].split("--hash", 1)[0].strip()
        items.append({
            "name": m.group(1), "extras": m.group(2) or "", "version": m.group(3), "marker": marker,
            "hashes": sorted(set(HASH.findall(body))), "comment": trailing,
        })
    return items


def render(items: list[dict]) -> str:
    out: list[str] = []
    for it in items:
        if "name" not in it:
            out.append(it["comment"])
            continue
        if it.get("comment"):
            out.append(f"# {it['name']}: {it['comment']}")
        head = f"{it['name']}{it['extras']}=={it['version']}" + (f" ; {it['marker']}" if it["marker"] else "")
        out.append(head + "".join(f" \\\n    --hash=sha256:{h}" for h in it["hashes"]))
    return "\n".join(out) + "\n"


def pypi_hashes(name: str, version: str, retries: int = 4) -> list[str]:
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "audio-cleaner-hash-lock"})
            with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - fixed https host
                data = json.load(r)
            hashes = sorted({f["digests"]["sha256"] for f in data["urls"]})
            if not hashes:
                raise SystemExit(f"PyPI lists no files for {name}=={version}")
            return hashes
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise SystemExit(f"PyPI has no {name}=={version} ({url})") from e
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = e
        time.sleep(2 * (attempt + 1))
    raise SystemExit(f"could not reach PyPI for {name}=={version}: {err}")


def fetch_all(items: list[dict]) -> dict[tuple[str, str], list[str]]:
    wanted = sorted({(it["name"], it["version"]) for it in items if "name" in it})
    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(zip(wanted, pool.map(lambda nv: pypi_hashes(*nv), wanted)))


def verify(items: list[dict]) -> list[str]:
    problems = []
    seen = set()
    for it in items:
        if "name" not in it:
            continue
        key = re.sub(r"[-_.]+", "-", it["name"]).lower()
        if key in seen:
            problems.append(f"{it['name']} is listed twice")
        seen.add(key)
        if not it["hashes"]:
            problems.append(f"{it['name']}=={it['version']} has no hash")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+", type=Path)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="compare with PyPI, change nothing, exit 1 on a difference")
    mode.add_argument("--verify", action="store_true", help="offline: every requirement is pinned and has hashes")
    a = ap.parse_args(argv)
    status = 0
    for path in a.files:
        items = parse(path.read_text())
        if a.verify:
            problems = verify(items)
            for p in problems:
                print(f"{path}: {p}", file=sys.stderr)
            status |= 1 if problems else 0
            continue
        found = fetch_all(items)
        if a.check:
            for it in items:
                if "name" in it and it["hashes"] != found[(it["name"], it["version"])]:
                    print(f"{path}: {it['name']}=={it['version']}: hashes differ from PyPI's list now", file=sys.stderr)
                    status = 1
            continue
        for it in items:
            if "name" in it:
                it["hashes"] = found[(it["name"], it["version"])]
        path.write_text(render(items))
        print(f"{path}: {sum('name' in i for i in items)} packages, {sum(len(i['hashes']) for i in items if 'name' in i)} hashes")
    return status


if __name__ == "__main__":
    sys.exit(main())
