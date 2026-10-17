"""The release mirror (acw/mirror.py) against a fake GitHub on a local port. One channel: the OBS plugin (the desktop
app gets its updates and models from its own GitHub releases, not from our server)."""
import json
import time

import httpx
import pytest

from acw import downloads, mirror

from .fake_github import REPO, FakeGitHub, Signer, have_openssl, sha

pytestmark = pytest.mark.skipif(not have_openssl(), reason="needs openssl 3")

DAY = 86400
PLATFORMS = (("windows-x64", "zip"), ("macos", "tar.xz"), ("ubuntu-debian", "deb"), ("linux-other", "tar.xz"))
WIN = "audio-cleaner-obs-plugin-{v}-windows-x64.zip"


@pytest.fixture
def gh():
    g = FakeGitHub()
    yield g
    g.close()


@pytest.fixture
def signer(tmp_path):
    d = tmp_path / "keys"
    d.mkdir()
    return Signer(d)


@pytest.fixture
def cfg(gh, signer, tmp_path, monkeypatch):
    monkeypatch.setenv("ACW_GITHUB_API", gh.api)
    monkeypatch.setenv("ACW_GITHUB_REPO", REPO)
    monkeypatch.setenv("ACW_UPDATE_PUBLIC_KEY", signer.public)
    monkeypatch.setenv("ACW_GITHUB_TOKEN_FILE", str(tmp_path / "no-token"))
    return mirror.Config.from_env()


def plugin_bytes(key, version):
    return f"plugin {key} {version} ".encode() * 300


def obs_release(gh, signer, version, *, with_sha512=True, big=False, tag=None, tamper=None, size_lie=None, bad_sig=False, **kw):
    """One release per version, obs-plugin-v<version>, as CI publishes it (`tag` overrides, e.g. for a mismatch).
    `tamper`: other bytes for the Windows package; `size_lie`: a wrong size for it in the manifest; `bad_sig`: a
    signature over other bytes than the manifest's."""
    files, assets = {}, {}
    for key, ext in PLATFORMS:
        name = f"audio-cleaner-obs-plugin-{version}-{key}.{ext}"
        data = plugin_bytes(key, version) + (b"z" * (51 * 1024 * 1024) if big and key == "macos" else b"")
        assets[name] = data
        files[key] = {"name": name, "size": len(data), "sha256": sha(data), **({"sha512": sha(data, "sha512")} if with_sha512 else {})}
    if size_lie:
        files["windows-x64"]["size"] = size_lie
    if tamper:
        assets[files["windows-x64"]["name"]] = tamper
    manifest = json.dumps({"version": version, "files": files}, sort_keys=True, separators=(",", ":")).encode()
    assets["obs-plugin-manifest.json"] = manifest
    assets["obs-plugin-manifest.json.sig"] = signer.sign(manifest + (b"x" if bad_sig else b""))
    gh.add(tag or f"obs-plugin-v{version}", assets, **kw)


def run(cfg, **kw):
    return mirror.run_once(cfg, **kw)


def served(channel="obs-plugin"):
    st = downloads.load_state(channel)
    return (st or {}).get("current")


def blob_downloads(gh):
    return [p for _, p, _ in gh.log if p.startswith("/blob/")]


# ---- one channel: the OBS plugin ---------------------------------------------------------------------------------------

def test_only_the_obs_plugin_is_mirrored(cfg, gh, signer):
    # desktop-v* and models-v* releases are not the mirror's business any more, however they look.
    manifest = b'{"version":"2.0.15","files":{}}'
    gh.add("desktop-v2.0.15", {"update.json": manifest, "update.json.sig": signer.sign(manifest),
                               "AudioCleaner-2.0.15-Windows.exe": b"x" * 10})
    gh.add("models-v2", {"AudioCleaner-models-v2.zip": b"m" * 100})
    obs_release(gh, signer, "0.1.60")
    assert run(cfg) == {"obs-plugin": "published 0.1.60"}
    assert downloads.CHANNELS == ("obs-plugin",) and [c for c, _ in mirror.CHANNEL_SYNC] == ["obs-plugin"]
    assert set(mirror.SIZE_CAP) == {"obs-plugin"} and set(downloads.MANIFESTS) == {"obs-plugin"}
    assert sorted(p.name for p in cfg.root.iterdir() if p.is_dir()) == ["obs-plugin"]
    assert set(mirror.load_status()["channels"]) == {"obs-plugin"}
    assert len(blob_downloads(gh)) == 6  # the plugin's manifest, its signature and four packages: nothing else
    for gone in ("desktop", "models"):
        with pytest.raises(ValueError):
            downloads.channel_dir(gone)
    for gone in ("sync_desktop", "sync_models", "parse_lock"):
        assert not hasattr(mirror, gone)
    assert not hasattr(cfg, "lock_path")


def test_the_admin_command_offers_only_the_plugin_channel(monkeypatch):
    from acw import admin

    seen = {}

    def fake_main(**kw):
        seen.update(kw)
        return 0

    monkeypatch.setattr(mirror, "main", fake_main)
    for gone in ("desktop", "models"):
        with pytest.raises(SystemExit) as e:
            admin.main(["mirror", "--channel", gone])
        assert e.value.code == 2  # argparse: not one of the choices
    assert seen == {}
    with pytest.raises(SystemExit) as e:
        admin.main(["mirror", "--channel", "obs-plugin"])
    assert e.value.code == 0 and seen["only"] == "obs-plugin"


# ---- good releases ---------------------------------------------------------------------------------------------------

def test_good_obs_release_is_mirrored_and_published(cfg, gh, signer):
    obs_release(gh, signer, "0.1.60")
    assert run(cfg)["obs-plugin"] == "published 0.1.60"
    cur = served()
    assert cur["version"] == "0.1.60"
    d = cfg.root / "obs-plugin" / "0.1.60"
    names = {f"audio-cleaner-obs-plugin-0.1.60-{k}.{e}" for k, e in PLATFORMS}
    assert {p.name for p in d.iterdir()} == {"obs-plugin-manifest.json", "obs-plugin-manifest.json.sig", *names}
    assets = gh.releases["obs-plugin-v0.1.60"]["assets"]
    assert (d / "obs-plugin-manifest.json").read_bytes() == assets["obs-plugin-manifest.json"]  # the signed bytes, unchanged
    assert (d / "obs-plugin-manifest.json.sig").read_bytes() == assets["obs-plugin-manifest.json.sig"]
    for name, entry in cur["files"].items():
        assert sha((d / name).read_bytes()) == entry["sha256"]
    assert cur["platforms"] == {k: f"audio-cleaner-obs-plugin-0.1.60-{k}.{e}" for k, e in PLATFORMS}
    # nothing is left behind and a second run changes nothing (and downloads nothing)
    assert not [p for p in (cfg.root / "obs-plugin").iterdir() if p.name.startswith((".staging", ".old"))]
    n = len(blob_downloads(gh))
    assert run(cfg)["obs-plugin"] == "0.1.60 is up to date"
    assert len(blob_downloads(gh)) == n
    assert mirror.load_status()["channels"]["obs-plugin"]["error"] is None
    assert mirror.verify_served() == []
    assert mirror.describe()[0].startswith("obs-plugin  serving 0.1.60 (6 files)")


def test_good_obs_release_with_sha512(cfg, gh, signer):
    obs_release(gh, signer, "0.1.60")
    assert run(cfg)["obs-plugin"] == "published 0.1.60"
    cur = served()
    assert cur["platforms"]["windows-x64"] == "audio-cleaner-obs-plugin-0.1.60-windows-x64.zip"
    assert "sha512" in cur["files"]["audio-cleaner-obs-plugin-0.1.60-windows-x64.zip"]
    assert "obs-plugin-manifest.json.sig" in cur["files"]


def test_sha512_is_optional_but_checked_when_given(cfg, gh, signer):
    obs_release(gh, signer, "0.1.60", with_sha512=False)
    assert run(cfg)["obs-plugin"] == "published 0.1.60"
    assert "sha512" not in served()["files"][WIN.format(v="0.1.60")]
    files, assets = {}, {}
    for key, ext in PLATFORMS:
        name = f"audio-cleaner-obs-plugin-0.1.61-{key}.{ext}"
        data = plugin_bytes(key, "0.1.61")
        assets[name] = data
        files[key] = {"name": name, "size": len(data), "sha256": sha(data), "sha512": sha(b"other bytes", "sha512")}
    manifest = json.dumps({"version": "0.1.61", "files": files}).encode()
    gh.add("obs-plugin-v0.1.61", {**assets, "obs-plugin-manifest.json": manifest, "obs-plugin-manifest.json.sig": signer.sign(manifest)})
    assert run(cfg)["obs-plugin"].startswith("FAILED")
    assert served()["version"] == "0.1.60"


def test_obs_release_without_a_manifest_yet_is_not_an_error(cfg, gh):
    gh.add("obs-plugin-v0.1.1", {"audio-cleaner-obs-plugin-0.1.1-windows-x64.zip": b"z" * 100})
    res = run(cfg)
    assert res["obs-plugin"] == "no plugin release with a signed manifest yet"
    assert mirror.load_status()["channels"]["obs-plugin"]["last_ok"]
    assert served() is None


def test_newest_versioned_plugin_release_wins(cfg, gh, signer):
    obs_release(gh, signer, "0.1.60")
    obs_release(gh, signer, "0.1.9")     # numerically, not as text: 0.1.60 is newer than 0.1.9
    obs_release(gh, signer, "0.1.61")
    gh.add("obs-plugin-latest", {"audio-cleaner-obs-plugin-0.1.99-windows-x64.zip": b"z" * 100}, prerelease=True)  # the old rolling one
    gh.add("obs-plugin-v0.1.70-rc1", {"obs-plugin-manifest.json": b"{}", "obs-plugin-manifest.json.sig": b"0" * 64})  # not a tag we follow
    assert run(cfg)["obs-plugin"] == "published 0.1.61"
    assert served()["version"] == "0.1.61"
    assert (cfg.root / "obs-plugin" / "0.1.61" / "obs-plugin-manifest.json").is_file()
    assert not (cfg.root / "obs-plugin" / "0.1.60").exists()  # only the newest is mirrored
    assert run(cfg)["obs-plugin"] == "0.1.61 is up to date"
    obs_release(gh, signer, "0.1.62")
    assert run(cfg)["obs-plugin"] == "published 0.1.62"


def test_newer_plugin_release_without_a_manifest_falls_back_to_the_older_signed_one(cfg, gh, signer):
    obs_release(gh, signer, "0.1.60")
    gh.add("obs-plugin-v0.1.61", {"audio-cleaner-obs-plugin-0.1.61-windows-x64.zip": b"z" * 100})  # manifest not uploaded yet
    obs_release(gh, signer, "0.1.62")
    del gh.releases["obs-plugin-v0.1.62"]["assets"]["obs-plugin-manifest.json.sig"]  # a manifest without its signature
    assert run(cfg)["obs-plugin"] == "published 0.1.60"
    assert served()["version"] == "0.1.60"


def test_plugin_tag_and_manifest_version_mismatch_is_refused(cfg, gh, signer):
    obs_release(gh, signer, "0.1.60")
    run(cfg)
    obs_release(gh, signer, "0.1.59", tag="obs-plugin-v0.1.61")  # a signed but older manifest under a newer tag
    res = run(cfg)
    assert res["obs-plugin"].startswith("FAILED")
    assert "the manifest says version 0.1.59 but the release is 0.1.61" in res["obs-plugin"]
    assert served()["version"] == "0.1.60"
    assert not (cfg.root / "obs-plugin" / "0.1.59").exists() and not (cfg.root / "obs-plugin" / "0.1.61").exists()


def test_draft_plugin_releases_are_ignored(cfg, gh, signer):
    obs_release(gh, signer, "0.1.60")
    obs_release(gh, signer, "0.1.61", draft=True)
    assert run(cfg)["obs-plugin"] == "published 0.1.60"
    gh.releases.clear()
    obs_release(gh, signer, "0.1.61", draft=True)
    assert run(cfg)["obs-plugin"] == "no plugin release with a signed manifest yet"
    assert served()["version"] == "0.1.60"


def test_no_plugin_release_keeps_the_served_files(cfg, gh, signer):
    obs_release(gh, signer, "0.1.60")
    run(cfg)
    gh.releases.clear()
    gh.add("obs-plugin-latest", {"audio-cleaner-obs-plugin-0.1.99-windows-x64.zip": b"z" * 100}, prerelease=True)
    assert run(cfg)["obs-plugin"] == "no plugin release with a signed manifest yet"
    assert served()["version"] == "0.1.60"
    d = cfg.root / "obs-plugin" / "0.1.60"
    assert (d / "obs-plugin-manifest.json").is_file() and (d / "audio-cleaner-obs-plugin-0.1.60-windows-x64.zip").is_file()
    assert mirror.verify_served() == []


# ---- bad releases: nothing is published ------------------------------------------------------------------------------

def test_bad_signature_publishes_nothing(cfg, gh, signer):
    obs_release(gh, signer, "0.1.60", bad_sig=True)
    res = run(cfg)
    assert res["obs-plugin"].startswith("FAILED") and "not signed" in res["obs-plugin"]
    assert served() is None
    assert not (cfg.root / "obs-plugin" / "0.1.60").exists()
    assert mirror.load_status()["channels"]["obs-plugin"]["error"]
    assert len(blob_downloads(gh)) == 2  # the manifest and its signature, no package


def test_signature_from_another_key_is_refused(cfg, gh, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    obs_release(gh, Signer(other), "0.1.60")
    assert "not signed" in run(cfg)["obs-plugin"]
    assert served() is None


def test_wrong_hash_publishes_nothing_and_throws_the_download_away(cfg, gh, signer):
    obs_release(gh, signer, "0.1.60", tamper=b"X" * len(plugin_bytes("windows-x64", "0.1.60")))  # same size, other bytes
    res = run(cfg)
    assert "does not match its checksum" in res["obs-plugin"]
    assert served() is None and not (cfg.root / "obs-plugin" / "0.1.60").exists()
    staging = cfg.root / "obs-plugin" / ".staging-0.1.60"
    assert not any(p.name.endswith(".part") for p in staging.iterdir())
    assert not (staging / WIN.format(v="0.1.60")).exists()


def test_size_that_differs_from_the_manifest_is_refused(cfg, gh, signer):
    obs_release(gh, signer, "0.1.60", size_lie=1234)
    assert "the manifest says" in run(cfg)["obs-plugin"]
    assert served() is None


def test_oversized_plugin_is_refused_before_any_download(cfg, gh, signer):
    obs_release(gh, signer, "0.1.60", size_lie=mirror.SIZE_CAP["obs-plugin"] + 1)
    assert "larger than the 50 MB allowed for obs-plugin" in run(cfg)["obs-plugin"]
    assert len(blob_downloads(gh)) == 2  # the manifest and its signature, no package


def test_oversized_plugin_is_refused(cfg, gh, signer):
    obs_release(gh, signer, "0.1.60", big=True)
    assert "larger than the 50 MB" in run(cfg)["obs-plugin"]
    assert served() is None


def test_missing_asset_and_missing_disk_space(cfg, gh, signer):
    obs_release(gh, signer, "0.1.60")
    del gh.releases["obs-plugin-v0.1.60"]["assets"]["audio-cleaner-obs-plugin-0.1.60-linux-other.tar.xz"]
    assert "no file" in run(cfg)["obs-plugin"]
    gh.releases.clear()
    obs_release(gh, signer, "0.1.60")
    cfg.min_free = 10**15
    assert "not enough disk space" in run(cfg)["obs-plugin"]
    assert served() is None


@pytest.mark.parametrize("bad", ["../x.zip", "a/b.zip", ".hidden", "x" * 200, "obs-plugin-manifest.json"])
def test_manifest_with_odd_names_is_refused(cfg, gh, signer, bad):
    manifest = json.dumps({"version": "0.1.61", "files": {"windows-x64": {"name": bad, "size": 5, "sha256": "a" * 64}}}).encode()
    gh.add("obs-plugin-v0.1.61", {"obs-plugin-manifest.json": manifest, "obs-plugin-manifest.json.sig": signer.sign(manifest),
                                  bad.replace("/", "_"): b"12345"})
    assert run(cfg)["obs-plugin"].startswith("FAILED"), bad
    assert served() is None


@pytest.mark.parametrize("body", [
    b"not json", b"[]", b'{"version":"0.1.61","files":{}}', b'{"version":"latest","files":{"windows-x64":{}}}',
    b'{"version":"0.1.61","files":{"Windows X64":{"name":"a.zip","size":5,"sha256":"' + b"a" * 64 + b'"}}}',
    b'{"version":"0.1.61","files":{"windows-x64":{"name":"a.zip","size":true,"sha256":"' + b"a" * 64 + b'"}}}',
    b'{"version":"0.1.61","files":{"windows-x64":{"name":"a.zip","size":5,"sha256":"xyz"}}}',
])
def test_signed_but_malformed_manifests_are_refused(cfg, gh, signer, body):
    gh.add("obs-plugin-v0.1.61", {"obs-plugin-manifest.json": body, "obs-plugin-manifest.json.sig": signer.sign(body)})
    assert run(cfg)["obs-plugin"].startswith("FAILED")
    assert served() is None


# ---- versions: no downgrade, previous kept for a week, then pruned ---------------------------------------------------

def test_previous_version_is_kept_for_seven_days_then_deleted(cfg, gh, signer):
    t0 = time.time()
    obs_release(gh, signer, "0.1.60")
    run(cfg, now=t0)
    obs_release(gh, signer, "0.1.61")
    assert run(cfg, now=t0 + 600)["obs-plugin"] == "published 0.1.61"
    assert served()["version"] == "0.1.61"
    old = WIN.format(v="0.1.60")
    assert (cfg.root / "obs-plugin" / "0.1.60" / old).is_file()
    assert downloads.lookup("obs-plugin", old, now=t0 + 3 * DAY)["version"] == "0.1.60"  # still served
    # the old manifest is never served as a file of its own: always the current one
    m = downloads.lookup("obs-plugin", "obs-plugin-manifest.json", now=t0 + 600)
    assert m["version"] == "0.1.61" and m["dir"] == "0.1.61"
    assert downloads.platform_file("obs-plugin", "windows-x64") == WIN.format(v="0.1.61")
    run(cfg, now=t0 + 3 * DAY)
    assert (cfg.root / "obs-plugin" / "0.1.60" / old).is_file()
    run(cfg, now=t0 + 8 * DAY)
    assert not (cfg.root / "obs-plugin" / "0.1.60").exists()
    assert (cfg.root / "obs-plugin" / "0.1.61" / "obs-plugin-manifest.json").is_file()
    assert downloads.lookup("obs-plugin", old, now=t0 + 8 * DAY) is None
    assert downloads.load_state("obs-plugin")["previous"] == []


def test_plugin_release_older_than_served_is_ignored(cfg, gh, signer):
    obs_release(gh, signer, "0.1.61")
    run(cfg)
    gh.releases.clear()  # 0.1.61's release was deleted on GitHub; the newest left is older than what is served
    obs_release(gh, signer, "0.1.60")
    assert "older" in run(cfg)["obs-plugin"]
    assert served()["version"] == "0.1.61"
    obs_release(gh, signer, "0.1.62")
    assert run(cfg)["obs-plugin"] == "published 0.1.62"


def test_force_fetches_the_served_version_again(cfg, gh, signer):
    obs_release(gh, signer, "0.1.60")
    run(cfg)
    n = len(blob_downloads(gh))
    assert run(cfg, force=True)["obs-plugin"] == "published 0.1.60"
    assert len(blob_downloads(gh)) == 2 * n
    assert mirror.verify_served() == []


def test_failed_new_release_keeps_serving_the_old_one(cfg, gh, signer):
    obs_release(gh, signer, "0.1.60")
    run(cfg)
    obs_release(gh, signer, "0.1.61", bad_sig=True)
    assert run(cfg)["obs-plugin"].startswith("FAILED")
    assert served()["version"] == "0.1.60"
    assert (cfg.root / "obs-plugin" / "0.1.60" / "obs-plugin-manifest.json").is_file()


def test_a_repair_of_a_damaged_folder(cfg, gh, signer):
    obs_release(gh, signer, "0.1.60")
    run(cfg)
    victim = cfg.root / "obs-plugin" / "0.1.60" / "audio-cleaner-obs-plugin-0.1.60-linux-other.tar.xz"
    victim.write_bytes(b"short")
    assert run(cfg)["obs-plugin"] == "published 0.1.60"
    assert victim.stat().st_size == served()["files"][victim.name]["size"]
    assert mirror.verify_served() == []


def test_verify_finds_a_changed_file(cfg, gh, signer):
    obs_release(gh, signer, "0.1.60")
    run(cfg)
    f = cfg.root / "obs-plugin" / "0.1.60" / WIN.format(v="0.1.60")
    data = bytearray(f.read_bytes())
    data[0] ^= 1
    f.write_bytes(bytes(data))
    assert mirror.verify_served() and "does not match" in mirror.verify_served()[0]


# ---- GitHub access: token, redirects, resume -------------------------------------------------------------------------

def test_token_is_sent_to_the_api_only(cfg, gh, signer, tmp_path, monkeypatch):
    obs_release(gh, signer, "0.1.60")
    tok = tmp_path / "token"
    tok.write_text("ghp_fake_token_value\n")
    monkeypatch.setenv("ACW_GITHUB_TOKEN_FILE", str(tok))
    cfg2 = mirror.Config.from_env()
    run(cfg2)
    api_calls = [a for host, p, a in gh.log if host == "127.0.0.1"]
    blob_calls = [a for host, p, a in gh.log if host == "localhost"]
    assert api_calls and all(a == "Bearer ghp_fake_token_value" for a in api_calls)
    assert blob_calls and all(a is None for a in blob_calls)  # the redirect to the storage host carries no token
    assert "ghp_fake" not in json.dumps(mirror.load_status())


def test_anonymous_without_a_token_file(cfg, gh, signer):
    obs_release(gh, signer, "0.1.60")
    run(cfg)
    assert gh.log and all(a is None for _, _, a in gh.log)


def test_interrupted_download_resumes(cfg, gh, signer):
    obs_release(gh, signer, "0.1.60")
    gh.cut_after = 3000  # the first package is cut short (the manifest and its signature are smaller)
    res = run(cfg)
    assert res["obs-plugin"].startswith("FAILED") and "resumes next run" in res["obs-plugin"]
    assert served() is None
    parts = list((cfg.root / "obs-plugin" / ".staging-0.1.60").glob("*.part"))
    assert parts and parts[0].stat().st_size == 3000
    gh.ranges.clear()
    assert run(cfg)["obs-plugin"] == "published 0.1.60"
    assert any(r == "bytes=3000-" for r in gh.ranges)
    assert mirror.verify_served() == []


def test_release_service_down_or_rate_limited(cfg, gh):
    gh.status_override[f"/repos/{REPO}/releases?per_page=100&page=1"] = 403
    res = run(cfg)
    assert "refused" in res["obs-plugin"]
    assert mirror.load_status()["channels"]["obs-plugin"]["error"]


def _paged_client(cfg, pages: list[int]):
    """A GitHub client on an httpx MockTransport whose release listing has pages of these sizes (then empty pages).
    The release obs-plugin-v0.0.1 is the last one listed (the oldest), so it sits on the last non-empty page."""
    total = sum(pages)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/repos/{REPO}/releases"
        page, per_page = int(request.url.params["page"]), int(request.url.params["per_page"])
        calls.append((page, per_page))
        start = sum(pages[:page - 1])
        size = pages[page - 1] if page <= len(pages) else 0
        rels = [{"tag_name": f"obs-plugin-v0.1.{total - i}", "draft": False, "assets": []} for i in range(start, start + size)]
        if rels and start + size == total:
            rels[-1]["tag_name"] = "obs-plugin-v0.0.1"
        return httpx.Response(200, json=rels)

    return mirror.GitHub(cfg, client=httpx.Client(transport=httpx.MockTransport(handler))), calls


def test_release_listing_reads_every_page(cfg):
    gh, calls = _paged_client(cfg, [100, 30])
    rels = gh.releases()
    assert len(rels) == 130
    assert calls == [(1, 100), (2, 100)]  # the short second page is the last one asked for
    assert rels[-1]["tag_name"] == "obs-plugin-v0.0.1"  # the oldest release, on page 2, is still seen
    assert len({r["tag_name"] for r in rels}) == 130


def test_release_listing_stops_at_an_empty_page_and_after_five(cfg):
    gh, calls = _paged_client(cfg, [100])
    assert len(gh.releases()) == 100
    assert calls == [(1, 100), (2, 100)]
    gh, calls = _paged_client(cfg, [100] * 7)
    assert len(gh.releases()) == 500
    assert [p for p, _ in calls] == [1, 2, 3, 4, 5]


def test_plugin_release_behind_a_page_of_other_releases_is_found(cfg, gh, signer):
    obs_release(gh, signer, "0.1.60")
    for n in range(120):  # newer than the plugin release (e.g. desktop builds of the same repository): page 2 of 100
        gh.add(f"desktop-v2.0.{n}", {f"AudioCleaner-2.0.{n}-Windows.exe": b"z"})
    assert run(cfg)["obs-plugin"] == "published 0.1.60"
    pages = [p for _, p, _ in gh.log if p.startswith(f"/repos/{REPO}/releases?")]
    assert f"/repos/{REPO}/releases?per_page=100&page=2" in pages
    assert f"/repos/{REPO}/releases?per_page=100&page=3" not in pages
    assert len(blob_downloads(gh)) == 6  # nothing of the desktop releases


# ---- monitor ---------------------------------------------------------------------------------------------------------

def test_stale_checks(cfg, gh, signer):
    assert mirror.stale_problems() == ["the mirror has never run"]
    t0 = time.time()
    obs_release(gh, signer, "0.1.60")
    run(cfg, now=t0)
    assert mirror.stale_problems(t0 + 600) == []
    assert any("no successful mirror run" in p for p in mirror.stale_problems(t0 + 2 * 3600 + 5))
    # a newer release that cannot be published: after an hour the monitor hears about it
    obs_release(gh, signer, "0.1.61", bad_sig=True)
    run(cfg, now=t0 + 600)
    assert mirror.stale_problems(t0 + 1200) == []
    problems = mirror.stale_problems(t0 + 600 + 3600 + 5)
    assert len(problems) == 1 and "0.1.61" in problems[0] and "0.1.60 is still served" in problems[0]
    # served again: fine
    gh.releases.pop("obs-plugin-v0.1.61")
    obs_release(gh, signer, "0.1.61")
    run(cfg, now=t0 + 5000)
    assert mirror.stale_problems(t0 + 5100) == []


def test_channels_an_older_mirror_served_do_not_make_it_look_stale(cfg, gh, signer):
    # A server that ran the mirror before it dropped the desktop and models channels has them in mirror-status.json
    # (and their old folders on disk). They are never run again, so they must not count as "no successful run".
    t0 = time.time()
    old = {"last_run": int(t0 - DAY), "last_ok": int(t0 - DAY), "result": "published 2.0.15", "error": None}
    cfg.root.mkdir(parents=True, exist_ok=True)
    downloads.write_json_atomic(mirror.status_path(cfg), {"last_run": int(t0 - DAY), "channels": {"desktop": old, "models": old}})
    (cfg.root / "desktop" / "2.0.15").mkdir(parents=True)
    (cfg.root / "desktop" / "state.json").write_text(json.dumps({"current": None, "previous": [], "seen": {"version": "2.0.16", "since": 1}}))
    obs_release(gh, signer, "0.1.60")
    assert run(cfg, now=t0) == {"obs-plugin": "published 0.1.60"}
    assert mirror.stale_problems(t0 + 600) == []
    assert [line.split()[0] for line in mirror.describe()] == ["obs-plugin"]
    assert mirror.verify_served() == []


def test_monitor_check_only_where_the_timer_is_enabled(tmp_path, cfg):
    from acw import monitor

    assert monitor.check_mirror(unit_dir=tmp_path) == []
    (tmp_path / "timers.target.wants").mkdir()
    (tmp_path / "timers.target.wants" / "acw-mirror.timer").write_text("")
    f = monitor.check_mirror(unit_dir=tmp_path)
    assert f[0].kind == "mirror_stale" and f[0].bad and "never run" in f[0].body


def test_second_run_while_one_is_running_exits(cfg, gh, signer):
    import fcntl

    cfg.root.mkdir(parents=True, exist_ok=True)
    with open(cfg.root / ".mirror.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        assert run(cfg) == {"-": "another mirror run is in progress"}


def test_prune_never_deletes_without_a_state_file(cfg, gh, signer):
    obs_release(gh, signer, "0.1.60")
    run(cfg)
    (cfg.root / "obs-plugin" / "state.json").write_text("not json")
    downloads._cache.clear()
    mirror.prune("obs-plugin")
    assert (cfg.root / "obs-plugin" / "0.1.60" / "obs-plugin-manifest.json").is_file()


def test_signature_check_with_and_without_the_cryptography_package(signer, tmp_path):
    data = b"hello manifest"
    sig = signer.sign(data)
    assert mirror.verify_signature(signer.public, data, sig)
    assert not mirror.verify_signature(signer.public, data + b"!", sig)
    assert not mirror.verify_signature(signer.public, data, b"\0" * 64)
    assert not mirror.verify_signature(signer.public, data, sig[:-1])
    assert not mirror.verify_signature("not base64!", data, sig)
    # the key the apps carry
    assert not mirror.verify_signature(mirror.PUBLIC_KEY, data, sig)
