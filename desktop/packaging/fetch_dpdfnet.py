#!/usr/bin/env python3
"""Get the heavy-noise model (DPDFNet-8, Apache-2.0) that the app bundles, from the models archive on our download
server (resources.MODELS_ARCHIVE; the archive holds it next to the AP-BWE and Sidon models).

    python fetch_dpdfnet.py [DEST_DIR]        # default: build/dpdfnet

Only the one member (15 MB) is read: the server answers HTTP range requests, so the zip's table of contents and that
member are fetched without the rest of the 0.9 GB archive. A server that ignores ranges gets a full download instead.
The result is checked against the pinned SHA-256 (prepare_vendor.DPDFNET_SHA256); a file that is already right is left
alone.
"""
from __future__ import annotations

import hashlib
import io
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[1] / "common"), str(HERE)]

from audio_cleaner_desktop import resources  # noqa: E402
from prepare_vendor import DPDFNET_FILE, DPDFNET_SHA256  # noqa: E402

MEMBER = "dpdfnet/" + DPDFNET_FILE
CHUNK = 1 << 20  # bytes asked for per request
USER_AGENT = "audio-cleaner-build"

Fetch = Callable[[int, int], bytes]  # (first byte, last byte inclusive) -> those bytes


class RangeFile(io.RawIOBase):
    """A read-only, seekable view of a remote file, read with `fetch(first, last)`; wrap it in io.BufferedReader."""

    def __init__(self, fetch: Fetch, size: int) -> None:
        self._fetch, self._size, self._pos = fetch, size, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def readinto(self, buffer) -> int:
        want = min(len(buffer), self._size - self._pos)
        if want <= 0:
            return 0
        data = self._fetch(self._pos, self._pos + want - 1)
        if len(data) != want:
            raise OSError(f"asked for {want} bytes at {self._pos} and got {len(data)}")
        buffer[:want] = data
        self._pos += want
        return want


def _http_fetch(url: str) -> Fetch:
    def fetch(first: int, last: int) -> bytes:
        request = urllib.request.Request(url, headers={"Range": f"bytes={first}-{last}", "User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 206:
                raise OSError(f"{url}: the server answered {response.status} to a range request")
            return response.read()
    return fetch


def _range_size(url: str) -> int | None:
    """The size of the file when the server does ranges (one 1-byte request), else None."""
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206:
                return None
            total = response.headers.get("Content-Range", "").rpartition("/")[2]
            return int(total) if total.isdigit() else None
    except urllib.error.HTTPError:
        return None


def extract_member(archive: io.RawIOBase, dest: Path) -> str:
    """Write MEMBER of the zip to `dest` and return its SHA-256."""
    digest = hashlib.sha256()
    with zipfile.ZipFile(io.BufferedReader(archive, CHUNK)) as z, z.open(MEMBER) as src, open(dest, "wb") as out:
        while block := src.read(CHUNK):
            digest.update(block)
            out.write(block)
    return digest.hexdigest()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def fetch(dest_dir: Path, url: str | None = None) -> Path:
    url = url or resources.MODELS_ARCHIVE.url
    target = dest_dir / DPDFNET_FILE
    if target.is_file() and sha256_of(target) == DPDFNET_SHA256:
        return target
    dest_dir.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    size = _range_size(url)
    try:
        if size is not None:
            digest = extract_member(RangeFile(_http_fetch(url), size), partial)
        else:
            print("the server does not do range requests: downloading the whole archive", file=sys.stderr)
            with tempfile.TemporaryDirectory() as tmp:
                whole = Path(tmp) / "models.zip"
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=120) as response, open(whole, "wb") as out:
                    while block := response.read(CHUNK):
                        out.write(block)
                with open(whole, "rb", buffering=0) as raw:
                    digest = extract_member(raw, partial)
        if digest != DPDFNET_SHA256:
            raise SystemExit(f"{url}: {MEMBER} is not the expected model (sha256 {digest}, expected {DPDFNET_SHA256})")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def main(argv: list[str]) -> int:
    dest = Path(argv[0]) if argv else Path("build") / "dpdfnet"
    try:
        path = fetch(dest)
    except (OSError, KeyError, zipfile.BadZipFile) as e:
        print(f"fetch_dpdfnet.py: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"{path} ({path.stat().st_size} bytes, sha256 ok)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
