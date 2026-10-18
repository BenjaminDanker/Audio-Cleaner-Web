#!/usr/bin/env python3
"""CI guard: every model file the app downloads can be found at its pinned address with its pinned size (HEAD requests,
nothing is downloaded): the speech model's files, and our own models and licence texts in the Hugging Face repository
(config.HF_MODELS_REPO at config.HF_MODELS_REVISION).

    python check_model_urls.py                       exit 1 and one line per file that is missing or has another size
"""
from __future__ import annotations

import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2] / "common")]

from audio_cleaner_desktop import resources  # noqa: E402


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def http_head(url: str) -> tuple[int, dict[str, str]]:
    opener = urllib.request.build_opener(_NoRedirect)
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "audio-cleaner-ci"})
    try:
        with opener.open(request, timeout=60) as response:
            return response.status, {k.lower(): v for k, v in response.headers.items()}
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}


def remote_size(url: str, head=http_head) -> tuple[int | None, str]:
    """(size, "") of the file behind `url`, or (None, why not). Redirects are followed by hand; Hugging Face reports
    the size as X-Linked-Size on its first answer, other hosts as Content-Length at the end."""
    linked = None
    for _ in range(6):
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


def pinned_files():
    """The files the app downloads, each address once."""
    files = [*resources.SPEECH_MODEL.files,
             *(f for m in (resources.HIGHS_MODEL, resources.VOICE_MODEL) for f in (*m.files, *m.licences))]
    return list({f.url: f for f in files}.values())


def check(head=http_head) -> list[str]:
    """What can't be found (empty when everything is there)."""
    problems = []
    for f in pinned_files():
        got, why = remote_size(f.url, head)
        if got != f.size:
            problems.append(f"{f.url}: {why or f'{got} bytes there, {f.size} pinned'}")
    return problems


def main(argv: list[str]) -> int:
    if argv:
        print(__doc__, file=sys.stderr)
        return 2
    found = check()
    for line in found:
        print(line, file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
