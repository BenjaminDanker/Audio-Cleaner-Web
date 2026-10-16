#!/usr/bin/env python3
"""Check that every download of the model catalog (resources.py) exists and has the size the catalog says: the
speech model's files on Hugging Face and the one models archive (AudioCleaner-models-v1.zip) on our download server
(config.MODELS_URL; the server mirrors the release asset, so while the mirror is not live this check fails on its own).

    python check_model_urls.py

For each URL: HEAD, following redirects by hand, and compare the size (Hugging Face reports it as X-Linked-Size
on the first answer, our server and the CDNs as Content-Length at the end) with the catalog. Exits 1, listing every
download that is missing or differs: a release asset that was not uploaded makes the installers' downloads fail, and
this is what finds out. (The SHA-256 is checked by the installers and the app when they download.)
"""
from __future__ import annotations

import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2] / "common")]

from audio_cleaner_desktop import resources  # noqa: E402

Head = Callable[[str], tuple[int, dict[str, str]]]  # url -> (status, lower-case headers)
MAX_HOPS = 6


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def http_head(url: str) -> tuple[int, dict[str, str]]:
    opener = urllib.request.build_opener(_NoRedirect)
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "audio-cleaner-catalog-check"})
    try:
        with opener.open(request, timeout=60) as response:
            return response.status, {k.lower(): v for k, v in response.headers.items()}
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}


def remote_size(url: str, head: Head = http_head) -> tuple[int | None, str]:
    """(size, "") of the file behind `url`, or (None, why not)."""
    linked = None
    for _ in range(MAX_HOPS):
        status, headers = head(url)
        if "x-linked-size" in headers:
            linked = int(headers["x-linked-size"])
        if status in (301, 302, 303, 307, 308) and headers.get("location"):
            url = urllib.parse.urljoin(url, headers["location"])
            continue
        if status == 200:
            if linked is not None:
                return linked, ""
            if "content-length" in headers:
                return int(headers["content-length"]), ""
            return None, "the server did not say how big the file is"
        return None, f"HTTP {status}"
    return None, "too many redirects"


def problems(model_files, head: Head = http_head) -> list[str]:
    """One line for each (url, size) pair that is missing or differs."""
    found = []
    for url, size in model_files:
        got, why = remote_size(url, head)
        if got is None:
            found.append(f"{url}: {why}")
        elif got != size:
            found.append(f"{url}: {got} bytes there, {size} in the catalog")
    return found


def catalog_files(catalog=(resources.SPEECH_MODEL, resources.HIGHS_MODEL, resources.VOICE_MODEL)):
    """(url, size) of everything that is downloaded: individual files, and each archive once."""
    found: dict[str, int] = {}
    for m in catalog:
        if m.archive is not None:
            found[m.archive.url] = m.archive.size
        else:
            found.update({f.url: f.size for f in m.files})
    return list(found.items())


def main() -> int:
    files = catalog_files()
    bad = problems(files)
    for line in bad:
        print(f"PROBLEM {line}")
    print(f"{len(files) - len(bad)} of {len(files)} model downloads are as the catalog says")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
