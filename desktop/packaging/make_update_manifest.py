#!/usr/bin/env python3
"""The update manifest of a desktop release (what the app's updater reads), and a check of the signing key.

    python make_update_manifest.py manifest --version 2.0.15 --dir release [--out release/update.json] [--notes-url URL]
        writes update.json: {"files": {"macos-arm64": {"name", "sha256", "size"}, "macos-x86_64": {...},
        "windows-x86_64": {...}}, "published": ISO time, "version": ...} with the keys sorted (and "notes_url" when
        --notes-url is given: apps from before the download server was introduced insist on one, the current app
        ignores it, and none is wanted in a file our server publishes). The
        three installers (AudioCleaner-<version>-Windows.exe, -Mac-Apple-silicon.pkg, -Mac-Intel.pkg) must be in --dir.
    python make_update_manifest.py check-key KEY.pem
        exit 0 when the private key in KEY.pem is the one whose public half is embedded in the app
        (audio_cleaner_desktop/updater.py PUBLIC_KEY), 1 when it is another key. Needs `openssl`.

CI signs update.json afterwards with
    openssl pkeyutl -sign -rawin -inkey KEY.pem -in update.json -out update.json.sig
(a raw 64-byte Ed25519 signature over the exact bytes of the file), see .github/workflows/desktop.yml and docs/desktop.md.
Nothing here ever reads or prints the key itself: `check-key` only asks openssl for the public half.
"""
from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

UPDATER = Path(__file__).resolve().parents[1] / "audio_cleaner_desktop" / "updater.py"
# platform key -> installer name for a version
INSTALLERS = {
    "windows-x86_64": "AudioCleaner-{version}-Windows.exe",
    "macos-arm64": "AudioCleaner-{version}-Mac-Apple-silicon.pkg",
    "macos-x86_64": "AudioCleaner-{version}-Mac-Intel.pkg",
}
_VERSION = re.compile(r"\d+(?:\.\d+){1,3}")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(version: str, folder: Path, published: str | None = None, notes_url: str | None = None) -> bytes:
    """The bytes of update.json (these exact bytes are what gets signed)."""
    if _VERSION.fullmatch(version) is None:
        raise SystemExit(f"{version!r} is not a version number")
    files = {}
    for key, pattern in INSTALLERS.items():
        path = folder / pattern.format(version=version)
        if not path.is_file():
            raise SystemExit(f"{path} is missing")
        files[key] = {"name": path.name, "size": path.stat().st_size, "sha256": sha256_of(path)}
    when = published or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = {
        "version": version,
        "published": when,
        "files": files,
    }
    if notes_url:
        manifest["notes_url"] = notes_url
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def app_public_key() -> str:
    """PUBLIC_KEY of the app, read from the source (no import: the release job has no Python packages)."""
    m = re.search(r'^PUBLIC_KEY = "([A-Za-z0-9+/=]{44})"$', UPDATER.read_text(encoding="utf-8"), re.M)
    if m is None:
        raise SystemExit(f"no PUBLIC_KEY in {UPDATER}")
    return m[1]


def public_key_of(pem: Path) -> str:
    """The raw Ed25519 public key (base64) that belongs to the private key in `pem`, asked from openssl (the DER
    form of an Ed25519 public key ends with the 32 raw bytes)."""
    out = subprocess.run(["openssl", "pkey", "-in", str(pem), "-pubout", "-outform", "DER"],
                         check=True, capture_output=True).stdout
    if len(out) != 44:
        raise SystemExit("the key is not an Ed25519 key")
    return base64.b64encode(out[-32:]).decode()


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--version", required=True)
    m.add_argument("--notes-url", default=None)
    m.add_argument("--dir", required=True, type=Path)
    m.add_argument("--out", type=Path)
    c = sub.add_parser("check-key")
    c.add_argument("pem", type=Path)
    args = p.parse_args(argv)
    if args.cmd == "manifest":
        data = build_manifest(args.version, args.dir, notes_url=args.notes_url)
        out = args.out or args.dir / "update.json"
        out.write_bytes(data)
        print(f"wrote {out}")
        return 0
    if public_key_of(args.pem) != app_public_key():
        print("the signing key is not the one the app trusts (PUBLIC_KEY in updater.py)", file=sys.stderr)
        return 1
    print("the signing key matches the app")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
