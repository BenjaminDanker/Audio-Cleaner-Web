"""The release download routes (acw/api/downloads_routes.py), on the main app and on the bulk process. Only the OBS
plugin is served: the desktop app gets its updates and models from its own GitHub releases."""
import hashlib
import os
import time

import pytest
from fastapi.testclient import TestClient

from acw import downloads

from .conftest import ORIGIN

BASE = "/api/downloads"
DAY = 86400
ZIP = bytes(range(256)) * 400  # 102400 bytes: the Windows package, big enough for the range tests
PLUGIN = {"windows-x64": ("win.zip", ZIP), "macos": ("mac.tar.xz", b"M" * 4000),
          "ubuntu-debian": ("plugin.deb", b"D" * 3000), "linux-other": ("lin.tar.xz", b"L" * 2000)}
WIN = f"{BASE}/obs-plugin/win.zip"
OBS_MANIFEST = b'{"files":{},"version":"0.1.60"}'
OLD_ZIP = b"old-zip" * 100


def put(channel, directory, name, data):
    d = downloads.channel_dir(channel) / directory
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(data)
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "dir": directory}


@pytest.fixture
def tree():
    now = int(time.time())
    obs = {name: put("obs-plugin", "0.1.60", name, data) for name, data in PLUGIN.values()}
    obs["obs-plugin-manifest.json"] = put("obs-plugin", "0.1.60", "obs-plugin-manifest.json", OBS_MANIFEST)
    obs["obs-plugin-manifest.json.sig"] = put("obs-plugin", "0.1.60", "obs-plugin-manifest.json.sig", b"S" * 64)
    old = {"old-win.zip": put("obs-plugin", "0.1.59", "old-win.zip", OLD_ZIP),
           "obs-plugin-manifest.json": put("obs-plugin", "0.1.59", "obs-plugin-manifest.json", b'{"version":"0.1.59"}')}
    downloads.write_state("obs-plugin", {
        "current": {"version": "0.1.60", "published_at": now, "files": obs, "platforms": {k: v[0] for k, v in PLUGIN.items()}},
        "previous": [{"version": "0.1.59", "files": old, "retired_at": now - 2 * DAY}], "seen": None})
    # What an older mirror left on disk: the desktop and models channels, never served again.
    for channel, name in (("desktop", "AudioCleaner-2.0.15-Windows.exe"), ("desktop", "update.json"), ("models", "AudioCleaner-models-v2.zip")):
        d = downloads.root() / channel / "old"
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(b"left behind")
        files = {name: {"size": 11, "sha256": hashlib.sha256(b"left behind").hexdigest(), "dir": "old"}}
        downloads.write_json_atomic(downloads.root() / channel / "state.json",
                                    {"current": {"version": "2.0.15", "files": files, "platforms": {}}, "previous": [], "seen": None})


@pytest.fixture
def bulk(tree):
    from acw.bulk_app import create_bulk_app

    with TestClient(create_bulk_app(), base_url=ORIGIN) as c:
        yield c


@pytest.fixture
def main(tree):
    from acw.app import create_app

    with TestClient(create_app(), base_url=ORIGIN) as c:
        yield c


@pytest.fixture(params=["main", "bulk"])
def c(request, main, bulk):
    """Only for what both processes serve the same way."""
    return main if request.param == "main" else bulk


# ---- what is served ----------------------------------------------------------------------------------------------------

def test_manifests_are_served_unchanged_and_revalidated(c):
    r = c.get(f"{BASE}/obs-plugin/manifest.json")
    assert r.status_code == 200 and r.content == OBS_MANIFEST
    assert r.headers["content-type"] == "application/json" and r.headers["cache-control"] == "no-cache"
    assert "content-disposition" not in r.headers and r.headers["accept-ranges"] == "none"
    assert "set-cookie" not in r.headers  # no sign-in needed, no cookies set
    r = c.get(f"{BASE}/obs-plugin/manifest.json.sig")
    assert r.status_code == 200 and r.content == b"S" * 64 and r.headers["content-type"] == "application/octet-stream"
    # a manifest ignores Range (the client needs all of it)
    assert c.get(f"{BASE}/obs-plugin/manifest.json", headers={"Range": "bytes=2-5"}).content == OBS_MANIFEST


def test_file_headers_body_and_head(c):
    r = c.get(WIN)
    assert r.status_code == 200 and r.content == ZIP
    h = r.headers
    assert h["content-length"] == str(len(ZIP)) and h["accept-ranges"] == "bytes"
    assert h["cache-control"] == "public, max-age=31536000, immutable"
    assert h["content-disposition"] == 'attachment; filename="win.zip"'
    assert h["etag"].startswith('"') and h["x-content-type-options"] == "nosniff"
    assert h["content-type"] == "application/zip" and "set-cookie" not in h
    assert c.get(f"{BASE}/obs-plugin/plugin.deb").headers["content-type"] == "application/vnd.debian.binary-package"
    head = c.head(WIN)
    assert head.status_code == 200 and head.content == b"" and head.headers["content-length"] == str(len(ZIP))


@pytest.mark.parametrize("rng,status,first,last", [
    ("bytes=0-9", 206, 0, 9),
    ("bytes=-50", 206, len(ZIP) - 50, len(ZIP) - 1),
    ("bytes=102000-999999", 206, 102000, len(ZIP) - 1),
])
def test_range_requests(c, rng, status, first, last):
    r = c.get(WIN, headers={"Range": rng})
    assert r.status_code == status
    assert r.content == ZIP[first:last + 1]
    assert r.headers["content-range"] == f"bytes {first}-{last}/{len(ZIP)}"


def test_unsatisfiable_range_is_416(c):
    for rng in (f"bytes={len(ZIP)}-", "bytes=-0"):
        r = c.get(WIN, headers={"Range": rng})
        assert r.status_code == 416 and r.headers["content-range"] == f"bytes */{len(ZIP)}", rng


def test_if_range_and_etag(c):
    etag = c.head(WIN).headers["etag"]
    assert c.get(WIN, headers={"Range": "bytes=0-9", "If-Range": etag}).status_code == 206
    r = c.get(WIN, headers={"Range": "bytes=0-9", "If-Range": '"other"'})
    assert r.status_code == 200 and r.content == ZIP  # the file changed: the whole of it
    r = c.get(WIN, headers={"If-None-Match": etag})
    assert r.status_code == 304 and r.content == b""
    m = f"{BASE}/obs-plugin/manifest.json"
    assert c.get(m, headers={"If-None-Match": c.head(m).headers["etag"]}).status_code == 304


# ---- what is not -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "/", "/obs-plugin/", "/obs-plugin/nothing.zip", "/obs-plugin/state.json", "/obs-plugin/.staging-0.1.60",
    "/obs-plugin/obs-plugin-manifest.json",  # the stored name is not public: it is manifest.json
    "/obs-plugin/..%2Fdesktop%2Fold%2Fupdate.json", "/obs-plugin/%2e%2e%2fstate.json", "/obs-plugin/..", "/obs-plugin/%00",
    "/obs-plugin/0.1.60/win.zip", "/obs-plugin/latest/freebsd", "/obs-plugin/latest/windows-setup", "/obs-plugin/latest/win.zip",
    "/obs-plugin/update.json", "/other/x",
    # the desktop app's and the models' channels are gone: its updates and models come from its GitHub releases
    "/desktop/update.json", "/desktop/AudioCleaner-2.0.15-Windows.exe", "/models/AudioCleaner-models-v2.zip",
])
def test_unknown_and_unsafe_names_are_404(main, path):
    r = main.get(BASE + path, follow_redirects=False)
    assert r.status_code in (404, 405, 307, 308)
    if r.status_code in (307, 308):  # Starlette's own trailing-slash redirect, to a path on this site
        assert r.headers["location"].startswith(("/", ORIGIN)) and main.get(BASE + path, follow_redirects=True).status_code == 404
    assert b"root" not in r.content and b"secret" not in r.content and b"left behind" not in r.content


def test_the_desktop_and_models_files_stay_gone_even_on_head(main):
    # Even with an older mirror's files and state still on disk, nothing of them is served, by GET or HEAD.
    for path in ("/desktop/update.json", "/models/AudioCleaner-models-v2.zip"):
        assert main.get(BASE + path).status_code == 404 and main.head(BASE + path).status_code == 404


def test_previous_version_is_served_for_a_week_and_latest_is_always_the_current_one(main):
    assert main.get(f"{BASE}/obs-plugin/old-win.zip").content == OLD_ZIP
    assert main.get(f"{BASE}/obs-plugin/manifest.json").content == OBS_MANIFEST  # always the current one
    st = downloads.load_state("obs-plugin")
    data = b"new windows build"
    st["current"]["files"]["win2.zip"] = put("obs-plugin", "0.1.61", "win2.zip", data)
    st["current"]["platforms"]["windows-x64"] = "win2.zip"
    st["previous"][0]["retired_at"] = int(time.time()) - 8 * DAY
    downloads.write_state("obs-plugin", st)
    assert main.get(f"{BASE}/obs-plugin/old-win.zip").status_code == 404
    assert main.get(f"{BASE}/obs-plugin/latest/windows").content == data
    assert main.get(f"{BASE}/obs-plugin/latest/mac").content == PLUGIN["macos"][1]


def test_a_file_that_is_not_in_the_state_is_not_served_even_if_it_is_on_disk(main):
    (downloads.channel_dir("obs-plugin") / "0.1.60" / "stray.zip").write_bytes(b"stray")
    assert main.get(f"{BASE}/obs-plugin/stray.zip").status_code == 404


def test_symlinks_and_changed_files_are_refused(main, tmp_path):
    path = downloads.channel_dir("obs-plugin") / "0.1.60" / "plugin.deb"
    secret = tmp_path / "secret.txt"
    secret.write_bytes(PLUGIN["ubuntu-debian"][1])  # same size on purpose
    path.unlink()
    os.symlink(secret, path)
    assert main.get(f"{BASE}/obs-plugin/plugin.deb").status_code == 404
    path.unlink()
    path.write_bytes(b"too short")
    assert main.get(f"{BASE}/obs-plugin/plugin.deb").status_code == 404
    path.unlink()
    assert main.get(f"{BASE}/obs-plugin/plugin.deb").status_code == 404
    # a version folder that is a symlink
    d = downloads.channel_dir("obs-plugin") / "0.1.60"
    moved = tmp_path / "elsewhere"
    d.rename(moved)
    os.symlink(moved, d)
    assert main.get(WIN).status_code == 404
    assert main.get(f"{BASE}/obs-plugin/latest/windows").status_code == 404


def test_a_damaged_state_file_serves_nothing(main):
    (downloads.channel_dir("obs-plugin") / "state.json").write_text("{not json")
    assert main.get(WIN).status_code == 404
    assert main.get(f"{BASE}/obs-plugin/latest/windows").status_code == 404
    st = {"current": {"files": {"win.zip": {"dir": "../x", "size": 1, "sha256": "a" * 64}}, "platforms": {"windows-x64": "win.zip"}}}
    downloads.write_state("obs-plugin", st)
    assert main.get(WIN).status_code == 404
    assert main.get(f"{BASE}/obs-plugin/latest/windows").status_code == 404


# ---- the plugin's latest links -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("platform,key", [("windows", "windows-x64"), ("mac", "macos"), ("ubuntu-debian", "ubuntu-debian"), ("linux-other", "linux-other")])
def test_latest_plugin_links_serve_the_file_directly(main, platform, key):
    name, data = PLUGIN[key]
    r = main.get(f"{BASE}/obs-plugin/latest/{platform}", follow_redirects=False)
    assert r.status_code == 200 and r.content == data
    assert r.headers["content-disposition"] == f'attachment; filename="{name}"'
    assert r.headers["cache-control"] == "no-cache" and "location" not in r.headers
    assert main.head(f"{BASE}/obs-plugin/latest/{platform}").headers["content-length"] == str(len(data))
    assert main.get(f"{BASE}/obs-plugin/latest/{platform}", headers={"Range": "bytes=1-3"}).content == data[1:4]


# ---- rate limit and byte cap -----------------------------------------------------------------------------------------------

def test_starting_downloads_is_rate_limited_per_network(main):
    codes = [main.get(WIN, headers={"X-Forwarded-For": "203.0.113.9"}).status_code for _ in range(42)]
    assert codes[:40] == [200] * 40
    assert codes[40] == 429
    r = main.get(WIN, headers={"X-Forwarded-For": "203.0.113.9"})
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert main.get(WIN, headers={"X-Forwarded-For": "203.0.113.10"}).status_code == 200  # another network is not affected
    # the latest links count the same, manifests are only under the general API limit
    assert main.get(f"{BASE}/obs-plugin/latest/windows", headers={"X-Forwarded-For": "203.0.113.9"}).status_code == 429
    assert main.get(f"{BASE}/obs-plugin/manifest.json", headers={"X-Forwarded-For": "203.0.113.9"}).status_code == 200


def test_resuming_and_head_cost_a_tenth(main):
    h = {"X-Forwarded-For": "203.0.113.20"}
    for _ in range(39):
        assert main.get(WIN, headers=h).status_code == 200
    for _ in range(9):  # 9 * 0.1 + the remaining token still fits
        assert main.get(WIN, headers={**h, "Range": "bytes=100-"}).status_code == 206
    assert main.head(WIN, headers=h).status_code == 200


def test_the_bulk_process_passes_every_byte_through_the_cap(bulk, monkeypatch):
    import acw.bulk_app as b

    taken = []

    class Counting:
        async def take(self, n):
            taken.append(n)

    monkeypatch.setattr(b, "downloads", Counting())
    r = bulk.get(WIN)
    assert r.status_code == 200 and sum(taken) == len(ZIP)
    taken.clear()
    r = bulk.get(f"{BASE}/obs-plugin/latest/windows", headers={"Range": "bytes=10-109"})
    assert sum(taken) == 100
    taken.clear()
    bulk.get(f"{BASE}/obs-plugin/manifest.json")
    assert sum(taken) == len(OBS_MANIFEST)


def test_security_headers_and_methods(main):
    r = main.get(WIN)
    assert r.headers["x-frame-options"] == "DENY"
    assert main.post(f"{BASE}/obs-plugin/manifest.json").status_code == 405
    assert main.delete(WIN).status_code == 405
    assert main.put(WIN, content=b"x").status_code in (401, 403, 404, 405)
