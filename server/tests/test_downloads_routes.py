"""The release download routes (acw/api/downloads_routes.py), on the main app and on the bulk process."""
import hashlib
import os
import time

import pytest
from fastapi.testclient import TestClient

from acw import downloads

from .conftest import ORIGIN

BASE = "/api/downloads"
DAY = 86400
EXE = bytes(range(256)) * 400  # 102400 bytes
PKG = b"pkg-bytes " * 1000
MODEL = b"model " * 5000
PLUGIN = {"windows-x64": ("win.zip", b"W" * 5000), "macos": ("mac.tar.xz", b"M" * 4000),
          "ubuntu-debian": ("plugin.deb", b"D" * 3000), "linux-other": ("lin.tar.xz", b"L" * 2000)}
MANIFEST = b'{"version":"2.0.15"}\n'
OBS_MANIFEST = b'{"files":{},"version":"0.1.60"}'


def put(channel, directory, name, data):
    d = downloads.channel_dir(channel) / directory
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(data)
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "dir": directory}


@pytest.fixture
def tree():
    now = int(time.time())
    desktop = {
        "AudioCleaner-2.0.15-Windows.exe": put("desktop", "2.0.15", "AudioCleaner-2.0.15-Windows.exe", EXE),
        "AudioCleaner-2.0.15-Mac-Intel.pkg": put("desktop", "2.0.15", "AudioCleaner-2.0.15-Mac-Intel.pkg", PKG),
        "update.json": put("desktop", "2.0.15", "update.json", MANIFEST),
        "update.json.sig": put("desktop", "2.0.15", "update.json.sig", b"S" * 64),
    }
    old = {"AudioCleaner-2.0.14-Windows.exe": put("desktop", "2.0.14", "AudioCleaner-2.0.14-Windows.exe", b"old-exe" * 100),
           "update.json": put("desktop", "2.0.14", "update.json", b'{"version":"2.0.14"}\n')}
    downloads.write_state("desktop", {
        "current": {"version": "2.0.15", "published_at": now, "files": desktop,
                    "platforms": {"windows-x86_64": "AudioCleaner-2.0.15-Windows.exe", "macos-x86_64": "AudioCleaner-2.0.15-Mac-Intel.pkg"}},
        "previous": [{"version": "2.0.14", "files": old, "retired_at": now - 2 * DAY}], "seen": None})
    obs = {name: put("obs-plugin", "0.1.60", name, data) for name, data in PLUGIN.values()}
    obs["obs-plugin-manifest.json"] = put("obs-plugin", "0.1.60", "obs-plugin-manifest.json", OBS_MANIFEST)
    obs["obs-plugin-manifest.json.sig"] = put("obs-plugin", "0.1.60", "obs-plugin-manifest.json.sig", b"S" * 64)
    downloads.write_state("obs-plugin", {
        "current": {"version": "0.1.60", "published_at": now, "files": obs, "platforms": {k: v[0] for k, v in PLUGIN.items()}},
        "previous": [], "seen": None})
    downloads.write_state("models", {
        "current": {"version": "models-v1", "published_at": now, "platforms": {},
                    "files": {"AudioCleaner-models-v1.zip": put("models", "models-v1", "AudioCleaner-models-v1.zip", MODEL)}},
        "previous": [], "seen": None})


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
    return main if request.param == "main" else bulk


# ---- what is served ----------------------------------------------------------------------------------------------------

def test_manifests_are_served_unchanged_and_revalidated(c):
    r = c.get(f"{BASE}/desktop/update.json")
    assert r.status_code == 200 and r.content == MANIFEST
    assert r.headers["content-type"] == "application/json" and r.headers["cache-control"] == "no-cache"
    assert "content-disposition" not in r.headers and r.headers["accept-ranges"] == "none"
    assert c.get(f"{BASE}/desktop/update.json.sig").content == b"S" * 64
    r = c.get(f"{BASE}/obs-plugin/manifest.json")
    assert r.status_code == 200 and r.content == OBS_MANIFEST
    assert c.get(f"{BASE}/obs-plugin/manifest.json.sig").status_code == 200
    # a manifest ignores Range (the client needs all of it)
    r = c.get(f"{BASE}/desktop/update.json", headers={"Range": "bytes=2-5"})
    assert r.status_code == 200 and r.content == MANIFEST


def test_file_headers_and_body(c):
    r = c.get(f"{BASE}/desktop/AudioCleaner-2.0.15-Windows.exe")
    assert r.status_code == 200 and r.content == EXE
    h = r.headers
    assert h["content-length"] == str(len(EXE)) and h["accept-ranges"] == "bytes"
    assert h["cache-control"] == "public, max-age=31536000, immutable"
    assert h["content-disposition"] == 'attachment; filename="AudioCleaner-2.0.15-Windows.exe"'
    assert h["etag"].startswith('"') and h["x-content-type-options"] == "nosniff"
    assert h["content-type"] == "application/vnd.microsoft.portable-executable"
    assert c.get(f"{BASE}/models/AudioCleaner-models-v1.zip").content == MODEL
    assert c.get(f"{BASE}/models/AudioCleaner-models-v1.zip").headers["content-type"] == "application/zip"


def test_no_sign_in_needed_and_no_cookies_set(c):
    r = c.get(f"{BASE}/desktop/update.json")
    assert r.status_code == 200 and "set-cookie" not in r.headers


def test_head_has_the_headers_and_no_body(c):
    r = c.head(f"{BASE}/desktop/AudioCleaner-2.0.15-Windows.exe")
    assert r.status_code == 200 and r.content == b""
    assert r.headers["content-length"] == str(len(EXE)) and r.headers["accept-ranges"] == "bytes"
    r = c.head(f"{BASE}/models/AudioCleaner-models-v1.zip")
    assert r.headers["content-length"] == str(len(MODEL))
    assert c.head(f"{BASE}/desktop/nope.exe").status_code == 404


@pytest.mark.parametrize("rng,status,first,last", [
    ("bytes=0-9", 206, 0, 9),
    ("bytes=100-", 206, 100, len(EXE) - 1),
    ("bytes=-50", 206, len(EXE) - 50, len(EXE) - 1),
    ("bytes=102000-999999", 206, 102000, len(EXE) - 1),
    ("bytes=0-0", 206, 0, 0),
])
def test_range_requests(c, rng, status, first, last):
    r = c.get(f"{BASE}/desktop/AudioCleaner-2.0.15-Windows.exe", headers={"Range": rng})
    assert r.status_code == status
    assert r.content == EXE[first:last + 1]
    assert r.headers["content-range"] == f"bytes {first}-{last}/{len(EXE)}"
    assert r.headers["content-length"] == str(last - first + 1)


def test_unsatisfiable_range_is_416(c):
    for rng in (f"bytes={len(EXE)}-", "bytes=999999-", "bytes=-0"):
        r = c.get(f"{BASE}/desktop/AudioCleaner-2.0.15-Windows.exe", headers={"Range": rng})
        assert r.status_code == 416 and r.headers["content-range"] == f"bytes */{len(EXE)}", rng


def test_odd_ranges_send_the_whole_file(c):
    for rng in ("bytes=0-1,5-9", "items=0-5", "bytes=9-3", "bytes=abc", "bytes=-"):
        r = c.get(f"{BASE}/desktop/AudioCleaner-2.0.15-Windows.exe", headers={"Range": rng})
        assert r.status_code == 200 and r.content == EXE, rng


def test_if_range_and_etag(c):
    url = f"{BASE}/desktop/AudioCleaner-2.0.15-Windows.exe"
    etag = c.head(url).headers["etag"]
    assert c.get(url, headers={"Range": "bytes=0-9", "If-Range": etag}).status_code == 206
    r = c.get(url, headers={"Range": "bytes=0-9", "If-Range": '"other"'})
    assert r.status_code == 200 and r.content == EXE
    r = c.get(url, headers={"If-None-Match": etag})
    assert r.status_code == 304 and r.content == b""
    assert c.get(f"{BASE}/desktop/update.json", headers={"If-None-Match": c.head(f"{BASE}/desktop/update.json").headers["etag"]}).status_code == 304


def test_a_resumed_download_reassembles(c):
    url = f"{BASE}/models/AudioCleaner-models-v1.zip"
    a = c.get(url, headers={"Range": "bytes=0-9999"}).content
    b = c.get(url, headers={"Range": "bytes=10000-"}).content
    assert a + b == MODEL


# ---- what is not -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "/desktop/nope.exe", "/desktop/", "/desktop", "/", "", "/models/other.zip", "/obs-plugin/nothing.zip",
    "/desktop/AudioCleaner-2.0.15-Windows.exe.part", "/desktop/state.json", "/desktop/.staging-2.0.15",
    "/obs-plugin/obs-plugin-manifest.json",  # the stored name is not public: it is manifest.json
    "/obs-plugin/obs-plugin-manifest.json.sig",
    "/desktop/..%2Fobs-plugin%2F0.1.60%2Fwin.zip", "/desktop/%2e%2e%2fstate.json", "/desktop/..", "/desktop/.",
    "/desktop/%00", "/models/AudioCleaner-models-v1.zip%00",
    "/desktop/a/b", "/models/models-v1/AudioCleaner-models-v1.zip", "/desktop/2.0.15/update.json",
    "/obs-plugin/latest", "/obs-plugin/latest/", "/obs-plugin/latest/win.zip", "/obs-plugin/latest/freebsd",
    "/obs-plugin/latest/windows-x64", "/desktop/latest", "/desktop/latest/linux", "/desktop/latest/windows-x86_64", "/other/x",
])
def test_unknown_and_unsafe_names_are_404(c, path):
    r = c.get(BASE + path, follow_redirects=False)
    assert r.status_code in (404, 405, 307, 308)
    if r.status_code in (307, 308):  # Starlette's own trailing-slash redirect, to a path on this site
        assert r.headers["location"].startswith(("/", ORIGIN)) and c.get(BASE + path, follow_redirects=True).status_code == 404
    assert b"root" not in r.content and b"secret" not in r.content


def test_a_file_of_another_channel_is_not_served_under_this_one(c):
    assert c.get(f"{BASE}/desktop/win.zip").status_code == 404
    assert c.get(f"{BASE}/models/win.zip").status_code == 404
    assert c.get(f"{BASE}/obs-plugin/AudioCleaner-2.0.15-Windows.exe").status_code == 404


def test_the_previous_version_is_served_for_a_week_but_not_its_manifest(c):
    assert c.get(f"{BASE}/desktop/AudioCleaner-2.0.14-Windows.exe").content == b"old-exe" * 100
    assert c.get(f"{BASE}/desktop/update.json").content == MANIFEST  # always the current one
    st = downloads.load_state("desktop")
    st["previous"][0]["retired_at"] = int(time.time()) - 8 * DAY
    downloads.write_state("desktop", st)
    assert c.get(f"{BASE}/desktop/AudioCleaner-2.0.14-Windows.exe").status_code == 404


def test_a_file_that_is_not_in_the_state_is_not_served_even_if_it_is_on_disk(c):
    (downloads.channel_dir("desktop") / "2.0.15" / "stray.exe").write_bytes(b"stray")
    assert c.get(f"{BASE}/desktop/stray.exe").status_code == 404


def test_a_symlink_or_a_changed_file_is_refused(c, tmp_path):
    path = downloads.channel_dir("desktop") / "2.0.15" / "AudioCleaner-2.0.15-Mac-Intel.pkg"
    secret = tmp_path / "secret.txt"
    secret.write_bytes(PKG)  # same size on purpose
    path.unlink()
    os.symlink(secret, path)
    assert c.get(f"{BASE}/desktop/AudioCleaner-2.0.15-Mac-Intel.pkg").status_code == 404
    path.unlink()
    path.write_bytes(b"too short")
    assert c.get(f"{BASE}/desktop/AudioCleaner-2.0.15-Mac-Intel.pkg").status_code == 404
    path.unlink()
    assert c.get(f"{BASE}/desktop/AudioCleaner-2.0.15-Mac-Intel.pkg").status_code == 404


def test_a_symlinked_version_folder_is_refused(c, tmp_path):
    d = downloads.channel_dir("models") / "models-v1"
    moved = tmp_path / "elsewhere"
    d.rename(moved)
    os.symlink(moved, d)
    assert c.get(f"{BASE}/models/AudioCleaner-models-v1.zip").status_code == 404


def test_nothing_published_yet_is_404(tmp_path):
    from acw.app import create_app

    with TestClient(create_app(), base_url=ORIGIN) as c:
        assert c.get(f"{BASE}/desktop/update.json").status_code == 404
        assert c.get(f"{BASE}/obs-plugin/latest/windows").status_code == 404
        assert c.get(f"{BASE}/models/AudioCleaner-models-v1.zip").status_code == 404


def test_a_damaged_state_file_serves_nothing(c):
    (downloads.channel_dir("models") / "state.json").write_text("{not json")
    assert c.get(f"{BASE}/models/AudioCleaner-models-v1.zip").status_code == 404
    st = {"current": {"files": {"AudioCleaner-models-v1.zip": {"dir": "../x", "size": 1, "sha256": "a" * 64}}}}
    downloads.write_state("models", st)
    assert c.get(f"{BASE}/models/AudioCleaner-models-v1.zip").status_code == 404


# ---- the plugin's latest links -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("platform,key", [("windows", "windows-x64"), ("mac", "macos"), ("ubuntu-debian", "ubuntu-debian"), ("linux-other", "linux-other")])
def test_latest_plugin_links_serve_the_file_directly(c, platform, key):
    name, data = PLUGIN[key]
    r = c.get(f"{BASE}/obs-plugin/latest/{platform}", follow_redirects=False)
    assert r.status_code == 200 and r.content == data
    assert r.headers["content-disposition"] == f'attachment; filename="{name}"'
    assert r.headers["cache-control"] == "no-cache" and "location" not in r.headers
    assert c.head(f"{BASE}/obs-plugin/latest/{platform}").headers["content-length"] == str(len(data))
    assert c.get(f"{BASE}/obs-plugin/latest/{platform}", headers={"Range": "bytes=1-3"}).content == data[1:4]
    assert c.get(f"{BASE}/obs-plugin/{name}").content == data  # and by its own name


def test_latest_desktop_links(c):
    r = c.get(f"{BASE}/desktop/latest/windows", follow_redirects=False)
    assert r.status_code == 200 and r.content == EXE and "location" not in r.headers
    assert r.headers["content-disposition"] == 'attachment; filename="AudioCleaner-2.0.15-Windows.exe"'
    assert c.get(f"{BASE}/desktop/latest/mac-intel").content == PKG
    assert c.get(f"{BASE}/desktop/latest/mac-apple-silicon").status_code == 404  # not in this fake manifest
    assert c.head(f"{BASE}/desktop/latest/windows").headers["content-length"] == str(len(EXE))


def test_latest_follows_the_current_version(c):
    st = downloads.load_state("obs-plugin")
    data = b"new windows build"
    st["current"]["files"]["win2.zip"] = put("obs-plugin", "0.1.61", "win2.zip", data)
    st["current"]["platforms"]["windows-x64"] = "win2.zip"
    downloads.write_state("obs-plugin", st)
    assert c.get(f"{BASE}/obs-plugin/latest/windows").content == data
    assert c.get(f"{BASE}/obs-plugin/latest/mac").content == PLUGIN["macos"][1]


# ---- rate limit and byte cap -----------------------------------------------------------------------------------------------

def test_starting_downloads_is_rate_limited_per_network(c):
    url = f"{BASE}/models/AudioCleaner-models-v1.zip"
    codes = [c.get(url, headers={"X-Forwarded-For": "203.0.113.9"}).status_code for _ in range(42)]
    assert codes[:40] == [200] * 40
    assert codes[40] == 429
    r = c.get(url, headers={"X-Forwarded-For": "203.0.113.9"})
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert c.get(url, headers={"X-Forwarded-For": "203.0.113.10"}).status_code == 200  # another network is not affected
    # manifests are only under the general API limit, and HEAD / resumes are cheap
    assert c.get(f"{BASE}/desktop/update.json", headers={"X-Forwarded-For": "203.0.113.9"}).status_code == 200


def test_resuming_and_head_cost_a_tenth(c):
    url = f"{BASE}/models/AudioCleaner-models-v1.zip"
    h = {"X-Forwarded-For": "203.0.113.20"}
    for _ in range(39):
        assert c.get(url, headers=h).status_code == 200
    for _ in range(9):  # 9 * 0.1 + the remaining token still fits
        assert c.get(url, headers={**h, "Range": "bytes=100-"}).status_code == 206
    assert c.head(url, headers=h).status_code == 200


def test_the_bulk_process_passes_every_byte_through_the_cap(bulk, monkeypatch):
    import acw.bulk_app as b

    taken = []

    class Counting:
        async def take(self, n):
            taken.append(n)

    monkeypatch.setattr(b, "downloads", Counting())
    r = bulk.get(f"{BASE}/models/AudioCleaner-models-v1.zip")
    assert r.status_code == 200 and sum(taken) == len(MODEL)
    taken.clear()
    r = bulk.get(f"{BASE}/desktop/AudioCleaner-2.0.15-Windows.exe", headers={"Range": "bytes=10-109"})
    assert sum(taken) == 100
    taken.clear()
    bulk.get(f"{BASE}/desktop/update.json")
    assert sum(taken) == len(MANIFEST)


def test_the_cap_matches_the_downloads_path_under_the_base_path():
    from acw.bulk_app import DownloadThrottleMiddleware

    mw = DownloadThrottleMiddleware(None, base_path="/audio")
    scope = {"type": "http", "method": "GET", "path": "/audio/api/downloads/models/x.zip"}
    assert mw.applies(scope) and mw.applies({**scope, "method": "HEAD"})
    assert not mw.applies({**scope, "path": "/audio/api/downloads"})
    assert not mw.applies({**scope, "path": "/audio/api/health"})


def test_the_download_cap_slows_a_release_download(bulk, monkeypatch):
    import acw.bulk_app as b
    from acw.security import DownloadBandwidth

    monkeypatch.setenv("ACW_DOWNLOAD_MBPS_TOTAL", "0.05")
    monkeypatch.setattr(b, "downloads", DownloadBandwidth())
    t0 = time.monotonic()
    r = bulk.get(f"{BASE}/models/AudioCleaner-models-v1.zip")  # 30000 bytes at 50000 B/s: free (one second burst)
    assert r.status_code == 200
    t1 = time.monotonic()
    r = bulk.get(f"{BASE}/desktop/AudioCleaner-2.0.15-Windows.exe")  # 102400 bytes: the burst is spent, ~1.6 s
    assert r.status_code == 200 and time.monotonic() - t1 > 0.8
    assert t1 - t0 < 0.8


def test_security_headers_and_methods(c):
    r = c.get(f"{BASE}/desktop/AudioCleaner-2.0.15-Windows.exe")
    assert r.headers["x-frame-options"] == "DENY"
    assert c.post(f"{BASE}/desktop/update.json").status_code == 405
    assert c.delete(f"{BASE}/models/AudioCleaner-models-v1.zip").status_code == 405
    assert c.put(f"{BASE}/models/AudioCleaner-models-v1.zip", content=b"x").status_code in (401, 403, 404, 405)
