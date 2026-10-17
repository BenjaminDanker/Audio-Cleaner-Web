#!/usr/bin/env python3
"""Gather the models archive that goes into a desktop release, next to the installers (the app downloads it from its
own release: config.MODELS_URL). It is resources.MODELS_ARCHIVE, carried over from the newest release that already has
it: first the public repository's releases, then this repository's (where a new archive is uploaded once, see
docs/desktop.md). It is checked against the size and SHA-256 pinned in resources.py. (The speech model for subtitles
is not in the releases: the app downloads it from Hugging Face, pinned to one commit.)

    python release_models.py OUT_DIR                 download them into OUT_DIR (prints the file names)
    python release_models.py --check                 only check that every one can be found (no download)

Environment: PUBLIC_REPO (the public repository, owner/name) with PUBLIC_TOKEN to read it, and GITHUB_REPOSITORY
(this one) with OWN_TOKEN (a token is needed for a private repository; each token only for its own repository).
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2] / "common")]

from audio_cleaner_desktop import models, resources  # noqa: E402

API = "https://api.github.com"


def _token(repo: str) -> str | None:
    if repo == os.environ.get("PUBLIC_REPO"):
        return os.environ.get("PUBLIC_TOKEN") or None
    if repo == os.environ.get("GITHUB_REPOSITORY"):
        return os.environ.get("OWN_TOKEN") or None
    return None


def _api(path: str, repo: str, accept: str = "application/vnd.github+json"):
    headers = {"Accept": accept, "User-Agent": "audio-cleaner-release"}
    if _token(repo):
        headers["Authorization"] = f"Bearer {_token(repo)}"
    with urllib.request.urlopen(urllib.request.Request(API + path, headers=headers), timeout=60) as r:
        return json.load(r)


def _releases(repo: str) -> list[dict]:
    """Newest first (up to 500)."""
    out: list[dict] = []
    for page in range(1, 6):
        try:
            batch = _api(f"/repos/{repo}/releases?per_page=100&page={page}", repo)
        except urllib.error.HTTPError as e:
            if e.code == 404 and page == 1:
                return []  # no such repository (or no access): nothing to carry over from it
            raise
        out += batch
        if len(batch) < 100:
            break
    return out


def find_archive() -> tuple[str, dict]:
    """(repository, asset) of the newest release that has the pinned models archive, with the pinned size."""
    a = resources.MODELS_ARCHIVE
    for repo in filter(None, (os.environ.get("PUBLIC_REPO"), os.environ.get("GITHUB_REPOSITORY"))):
        for rel in _releases(repo):
            for asset in rel.get("assets") or []:
                if asset.get("name") == a.name and asset.get("size") == a.size and asset.get("state", "uploaded") == "uploaded":
                    return repo, asset
    raise SystemExit(f"{a.name} ({a.size} bytes) is in no release of {os.environ.get('PUBLIC_REPO')} or "
                     f"{os.environ.get('GITHUB_REPOSITORY')}: upload it once (docs/desktop.md, \"New models\")")


class _AssetRedirects(urllib.request.HTTPRedirectHandler):
    """The asset API answers with a redirect to a signed download address on another host: https only, and the token
    is not sent there."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlparse(newurl).scheme != "https":
            raise urllib.error.HTTPError(newurl, code, "redirect away from https", headers, fp)
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(newurl).netloc != urllib.parse.urlparse(req.full_url).netloc:
            new.remove_header("Authorization")
        return new


def _download_asset(repo: str, asset: dict, dest: Path) -> None:
    a = resources.MODELS_ARCHIVE
    headers = {"Accept": "application/octet-stream", "User-Agent": "audio-cleaner-release"}
    if _token(repo):
        headers["Authorization"] = f"Bearer {_token(repo)}"
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=models._ssl_context()), _AssetRedirects())
    request = urllib.request.Request(f"{API}/repos/{repo}/releases/assets/{asset['id']}", headers=headers)
    part = dest.with_name(dest.name + ".part")
    with opener.open(request, timeout=60) as r, open(part, "wb") as out:
        while chunk := r.read(1 << 20):
            out.write(chunk)
    if part.stat().st_size != a.size or models._sha256(part, None) != a.sha256:
        part.unlink()
        raise SystemExit(f"{a.name} from {repo} is not the pinned file (size or SHA-256 differ)")
    part.replace(dest)


def gather(out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    repo, asset = find_archive()
    _download_asset(repo, asset, out / resources.MODELS_ARCHIVE.name)
    return [resources.MODELS_ARCHIVE.name]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def http_head(url: str) -> tuple[int, dict[str, str]]:
    opener = urllib.request.build_opener(_NoRedirect)
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "audio-cleaner-release"})
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


def check() -> list[str]:
    """What can't be found (empty when everything is there)."""
    problems = []
    for f in resources.SPEECH_MODEL.files:  # downloaded by the app from Hugging Face: the pinned files must be there
        got, why = remote_size(f.url)
        if got != f.size:
            problems.append(f"{f.url}: {why or f'{got} bytes there, {f.size} pinned'}")
    try:
        find_archive()
    except SystemExit as e:
        problems.append(str(e))
    return problems


def main(argv: list[str]) -> int:
    if argv == ["--check"]:
        found = check()
        for line in found:
            print(line, file=sys.stderr)
        return 1 if found else 0
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    for name in gather(Path(argv[0])):
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
