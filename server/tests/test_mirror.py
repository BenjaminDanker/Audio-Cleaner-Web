"""The release mirror (acw/mirror.py) against a fake GitHub on a local port."""
import json
import os
import time
from pathlib import Path

import pytest

from acw import downloads, mirror

from .fake_github import REPO, FakeGitHub, Signer, have_openssl, sha

pytestmark = pytest.mark.skipif(not have_openssl(), reason="needs openssl 3")

DAY = 86400
INSTALLERS = {
    "windows-x86_64": "AudioCleaner-{v}-Windows.exe",
    "macos-arm64": "AudioCleaner-{v}-Mac-Apple-silicon.pkg",
    "macos-x86_64": "AudioCleaner-{v}-Mac-Intel.pkg",
}


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
    lock = tmp_path / "models.lock"
    lock.write_text("# test\narchive  -  " + "0" * 64 + " 1 btbn-ffmpeg https://example.com/ffmpeg.tar.xz\n")
    monkeypatch.setenv("ACW_GITHUB_API", gh.api)
    monkeypatch.setenv("ACW_GITHUB_REPO", REPO)
    monkeypatch.setenv("ACW_UPDATE_PUBLIC_KEY", signer.public)
    monkeypatch.setenv("ACW_MODELS_LOCK", str(lock))
    monkeypatch.setenv("ACW_GITHUB_TOKEN_FILE", str(tmp_path / "no-token"))
    return mirror.Config.from_env()


def desktop_release(gh, signer, version, *, tamper=None, size_lie=None, bad_sig=False, installer_bytes=None):
    files, assets = {}, {}
    for key, pat in INSTALLERS.items():
        name = pat.format(v=version)
        data = (installer_bytes or {}).get(key) or (f"installer {key} {version} ".encode() * 5000)
        assets[name] = data
        files[key] = {"name": name, "size": len(data), "sha256": sha(data)}
    if size_lie:
        files["windows-x86_64"]["size"] = size_lie
    if tamper:
        assets[files["windows-x86_64"]["name"]] = tamper
    manifest = (json.dumps({"version": version, "published": "2026-10-16T00:00:00Z", "notes_url": "x", "files": files}, indent=2, sort_keys=True) + "\n").encode()
    sig = signer.sign(manifest + (b"x" if bad_sig else b""))
    assets["update.json"], assets["update.json.sig"] = manifest, sig
    gh.add(f"desktop-v{version}", assets)


def obs_release(gh, signer, version, *, with_sha512=True, big=False):
    files, assets = {}, {}
    for key, ext in (("windows-x64", "zip"), ("macos", "tar.xz"), ("ubuntu-debian", "deb"), ("linux-other", "tar.xz")):
        name = f"audio-cleaner-obs-plugin-{version}-{key}.{ext}"
        data = (f"plugin {key} {version} ".encode() * 300) + (b"z" * (51 * 1024 * 1024) if big and key == "macos" else b"")
        assets[name] = data
        files[key] = {"name": name, "size": len(data), "sha256": sha(data), **({"sha512": sha(data, "sha512")} if with_sha512 else {})}
    manifest = json.dumps({"version": version, "files": files}, sort_keys=True, separators=(",", ":")).encode()
    assets["obs-plugin-manifest.json"], assets["obs-plugin-manifest.json.sig"] = manifest, signer.sign(manifest)
    gh.add("obs-plugin-latest", assets, prerelease=True)


MODELS = b"models archive " * 20000


def models_lock(cfg, data=MODELS, tag="models-v1", name="AudioCleaner-models-v1.zip", extra=""):
    cfg.lock_path.write_text(
        f"archive  -  {sha(data)} {len(data)} {tag}  https://github.com/{REPO}/releases/download/{tag}/{name}\n"
        f"archive  -  {'1' * 64} 99 btbn-ffmpeg https://github.com/BtbN/FFmpeg-Builds/releases/download/x/ffmpeg.tar.xz\n" + extra)


def run(cfg, **kw):
    return mirror.run_once(cfg, **kw)


def served(channel):
    st = downloads.load_state(channel)
    return (st or {}).get("current")


# ---- good releases ---------------------------------------------------------------------------------------------------

def test_good_desktop_release_is_mirrored_and_published(cfg, gh, signer):
    desktop_release(gh, signer, "2.0.15")
    res = run(cfg)
    assert res["desktop"] == "published 2.0.15"
    cur = served("desktop")
    assert cur["version"] == "2.0.15"
    d = cfg.root / "desktop" / "2.0.15"
    assert {p.name for p in d.iterdir()} == {"update.json", "update.json.sig", *(p.format(v="2.0.15") for p in INSTALLERS.values())}
    assert (d / "update.json").read_bytes() == gh.releases["desktop-v2.0.15"]["assets"]["update.json"]  # the signed bytes, unchanged
    for name, entry in cur["files"].items():
        assert sha((d / name).read_bytes()) == entry["sha256"]
    # nothing is left behind and a second run changes nothing
    assert not [p for p in (cfg.root / "desktop").iterdir() if p.name.startswith((".staging", ".old"))]
    assert run(cfg)["desktop"] == "2.0.15 is up to date"
    assert mirror.load_status()["channels"]["desktop"]["error"] is None


def test_good_obs_release_with_sha512(cfg, gh, signer):
    obs_release(gh, signer, "0.1.60")
    assert run(cfg)["obs-plugin"] == "published 0.1.60"
    cur = served("obs-plugin")
    assert cur["platforms"]["windows-x64"] == "audio-cleaner-obs-plugin-0.1.60-windows-x64.zip"
    assert "sha512" in cur["files"]["audio-cleaner-obs-plugin-0.1.60-windows-x64.zip"]
    assert "obs-plugin-manifest.json.sig" in cur["files"]


def test_obs_release_without_a_manifest_yet_is_not_an_error(cfg, gh):
    gh.add("obs-plugin-latest", {"audio-cleaner-obs-plugin-0.1.1-windows-x64.zip": b"z" * 100}, prerelease=True)
    res = run(cfg)
    assert "no signed manifest yet" in res["obs-plugin"]
    assert mirror.load_status()["channels"]["obs-plugin"]["last_ok"]


def test_models_archive_from_the_lock(cfg, gh):
    models_lock(cfg)
    gh.add("models-v1", {"AudioCleaner-models-v1.zip": MODELS})
    assert run(cfg)["models"] == "published AudioCleaner-models-v1.zip"
    assert (cfg.root / "models" / "models-v1" / "AudioCleaner-models-v1.zip").read_bytes() == MODELS
    assert served("models")["files"]["AudioCleaner-models-v1.zip"]["dir"] == "models-v1"
    assert run(cfg)["models"] == "models are up to date"


def test_parse_lock_takes_only_models_releases(cfg):
    models_lock(cfg)
    assert mirror.parse_lock(cfg.lock_path) == [
        {"tag": "models-v1", "name": "AudioCleaner-models-v1.zip", "sha256": sha(MODELS), "size": len(MODELS)}]
    real = Path(__file__).resolve().parents[2] / "deploy" / "models.lock"
    found = mirror.parse_lock(real)
    assert [f["name"] for f in found] == ["AudioCleaner-models-v1.zip", "AudioCleaner-models-v2.zip"]
    assert found[0]["size"] == 908538132 and found[0]["sha256"] == "5592be637b0be833d92b1476ffebfb07c2ac6920d9b24b3378ae6d318ebd81ce"


# ---- bad releases: nothing is published ------------------------------------------------------------------------------

def test_bad_signature_publishes_nothing(cfg, gh, signer):
    desktop_release(gh, signer, "2.0.15", bad_sig=True)
    res = run(cfg)
    assert res["desktop"].startswith("FAILED") and "not signed" in res["desktop"]
    assert served("desktop") is None
    assert not (cfg.root / "desktop" / "2.0.15").exists()
    assert mirror.load_status()["channels"]["desktop"]["error"]


def test_signature_from_another_key_is_refused(cfg, gh, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    desktop_release(gh, Signer(other), "2.0.15")
    assert "not signed" in run(cfg)["desktop"]
    assert served("desktop") is None


def test_wrong_hash_publishes_nothing_and_throws_the_download_away(cfg, gh, signer):
    desktop_release(gh, signer, "2.0.15", tamper=b"X" * len(b"installer windows-x86_64 2.0.15 " * 5000))
    res = run(cfg)
    assert "does not match its checksum" in res["desktop"]
    assert served("desktop") is None and not (cfg.root / "desktop" / "2.0.15").exists()
    staging = cfg.root / "desktop" / ".staging-2.0.15"
    assert not any(p.name.endswith(".part") for p in staging.iterdir())


def test_size_that_differs_from_the_manifest_is_refused(cfg, gh, signer):
    desktop_release(gh, signer, "2.0.15", size_lie=1234)
    assert "the manifest says" in run(cfg)["desktop"]
    assert served("desktop") is None


def test_oversized_installer_is_refused_before_any_download(cfg, gh, signer):
    big = mirror.SIZE_CAP["desktop"] + 1
    desktop_release(gh, signer, "2.0.15", size_lie=big)
    res = run(cfg)
    assert "larger than" in res["desktop"]
    assert len([p for _, p, _ in gh.log if "/blob/" in p]) == 2  # the manifest and its signature, no installer


def test_oversized_plugin_is_refused(cfg, gh, signer):
    obs_release(gh, signer, "0.1.60", big=True)
    assert "larger than the 50 MB" in run(cfg)["obs-plugin"]
    assert served("obs-plugin") is None


def test_models_archive_with_the_wrong_hash_is_refused(cfg, gh):
    models_lock(cfg)
    gh.add("models-v1", {"AudioCleaner-models-v1.zip": b"M" * len(MODELS)})
    assert "checksum" in run(cfg)["models"]
    assert served("models") is None
    assert not (cfg.root / "models" / "models-v1" / "AudioCleaner-models-v1.zip").exists()


def test_models_archive_larger_than_the_cap_is_refused(cfg, gh):
    cfg.lock_path.write_text(f"archive - {'a' * 64} {3 * 1024**3} models-v1 https://x/models-v1/Big.zip\n")
    assert "larger than" in run(cfg)["models"]


def test_missing_asset_and_missing_disk_space(cfg, gh, signer):
    desktop_release(gh, signer, "2.0.15")
    del gh.releases["desktop-v2.0.15"]["assets"]["AudioCleaner-2.0.15-Mac-Intel.pkg"]
    assert "no file" in run(cfg)["desktop"]
    gh.releases.clear()
    desktop_release(gh, signer, "2.0.15")
    cfg.min_free = 10**15
    assert "not enough disk space" in run(cfg)["desktop"]
    assert served("desktop") is None


def test_manifest_with_odd_names_is_refused(cfg, gh, signer):
    for bad in ("../x.exe", "a/b.exe", ".hidden", "x" * 200):
        manifest = json.dumps({"version": "2.0.16", "files": {"windows-x86_64": {"name": bad, "size": 5, "sha256": "a" * 64}}}).encode()
        gh.releases.clear()
        gh.add("desktop-v2.0.16", {"update.json": manifest, "update.json.sig": signer.sign(manifest), bad.replace("/", "_"): b"12345"})
        assert run(cfg)["desktop"].startswith("FAILED"), bad
        assert served("desktop") is None


# ---- versions: no downgrade, previous kept for a week, then pruned ---------------------------------------------------

def test_previous_version_is_kept_for_seven_days_then_deleted(cfg, gh, signer):
    t0 = time.time()
    desktop_release(gh, signer, "2.0.15")
    run(cfg, now=t0)
    desktop_release(gh, signer, "2.0.16")
    assert run(cfg, now=t0 + 600)["desktop"] == "published 2.0.16"
    assert served("desktop")["version"] == "2.0.16"
    old = "AudioCleaner-2.0.15-Windows.exe"
    assert (cfg.root / "desktop" / "2.0.15" / old).is_file()
    assert downloads.lookup("desktop", old, now=t0 + 3 * DAY)["version"] == "2.0.15"  # still served
    assert downloads.lookup("desktop", "update.json", now=t0 + 600)["version"] == "2.0.16"
    # the old manifest is never served as a file of its own
    assert downloads.lookup("desktop", "update.json", now=t0 + 600)["dir"] == "2.0.16"
    run(cfg, now=t0 + 3 * DAY)
    assert (cfg.root / "desktop" / "2.0.15" / old).is_file()
    run(cfg, now=t0 + 8 * DAY)
    assert not (cfg.root / "desktop" / "2.0.15").exists()
    assert (cfg.root / "desktop" / "2.0.16" / "update.json").is_file()
    assert downloads.lookup("desktop", old, now=t0 + 8 * DAY) is None
    assert downloads.load_state("desktop")["previous"] == []


def test_older_release_is_ignored(cfg, gh, signer):
    desktop_release(gh, signer, "2.0.16")
    run(cfg)
    gh.releases.clear()
    desktop_release(gh, signer, "2.0.15")
    assert "older" in run(cfg)["desktop"]
    assert served("desktop")["version"] == "2.0.16"


def test_plugin_manifest_older_than_served_is_ignored(cfg, gh, signer):
    obs_release(gh, signer, "0.1.61")
    run(cfg)
    obs_release(gh, signer, "0.1.60")
    assert "older" in run(cfg)["obs-plugin"]
    assert served("obs-plugin")["version"] == "0.1.61"
    obs_release(gh, signer, "0.1.62")
    assert run(cfg)["obs-plugin"] == "published 0.1.62"


def test_failed_new_release_keeps_serving_the_old_one(cfg, gh, signer):
    desktop_release(gh, signer, "2.0.15")
    run(cfg)
    desktop_release(gh, signer, "2.0.16", bad_sig=True)
    assert run(cfg)["desktop"].startswith("FAILED")
    assert served("desktop")["version"] == "2.0.15"
    assert (cfg.root / "desktop" / "2.0.15" / "update.json").is_file()


def test_release_without_update_information_falls_back_to_an_older_one(cfg, gh, signer):
    desktop_release(gh, signer, "2.0.15")
    gh.add("desktop-v2.0.16", {"AudioCleaner-2.0.16-Windows.exe": b"x" * 10})
    gh.add("desktop-v2.0.16-build.3", {"update.json": b"{}", "update.json.sig": b"0" * 64})  # not a release tag we follow
    assert run(cfg)["desktop"] == "published 2.0.15"


def test_a_repair_of_a_damaged_folder(cfg, gh, signer):
    desktop_release(gh, signer, "2.0.15")
    run(cfg)
    victim = cfg.root / "desktop" / "2.0.15" / "AudioCleaner-2.0.15-Mac-Intel.pkg"
    victim.write_bytes(b"short")
    assert run(cfg)["desktop"] == "published 2.0.15"
    assert victim.stat().st_size == served("desktop")["files"][victim.name]["size"]
    assert mirror.verify_served() == []


def test_verify_finds_a_changed_file(cfg, gh, signer):
    desktop_release(gh, signer, "2.0.15")
    run(cfg)
    f = cfg.root / "desktop" / "2.0.15" / "AudioCleaner-2.0.15-Windows.exe"
    data = bytearray(f.read_bytes())
    data[0] ^= 1
    f.write_bytes(bytes(data))
    assert mirror.verify_served() and "does not match" in mirror.verify_served()[0]


# ---- GitHub access: token, redirects, resume -------------------------------------------------------------------------

def test_token_is_sent_to_the_api_only(cfg, gh, signer, tmp_path, monkeypatch):
    desktop_release(gh, signer, "2.0.15")
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
    desktop_release(gh, signer, "2.0.15")
    run(cfg)
    assert gh.log and all(a is None for _, _, a in gh.log)


def test_interrupted_download_resumes(cfg, gh, signer):
    desktop_release(gh, signer, "2.0.15")
    gh.cut_after = 50_000
    res = run(cfg)
    assert res["desktop"].startswith("FAILED") and "resumes next run" in res["desktop"]
    assert served("desktop") is None
    parts = list((cfg.root / "desktop" / ".staging-2.0.15").glob("*.part"))
    assert parts and parts[0].stat().st_size == 50_000
    gh.ranges.clear()
    assert run(cfg)["desktop"] == "published 2.0.15"
    assert any(r == "bytes=50000-" for r in gh.ranges)
    assert mirror.verify_served() == []


def test_release_service_down_or_rate_limited(cfg, gh):
    gh.status_override[f"/repos/{REPO}/releases?per_page=50"] = 403
    res = run(cfg)
    assert "refused" in res["desktop"]
    assert mirror.load_status()["channels"]["desktop"]["error"]


# ---- monitor ---------------------------------------------------------------------------------------------------------

def test_stale_checks(cfg, gh, signer):
    assert mirror.stale_problems() == ["the mirror has never run"]
    t0 = time.time()
    desktop_release(gh, signer, "2.0.15")
    obs_release(gh, signer, "0.1.60")
    models_lock(cfg)
    gh.add("models-v1", {"AudioCleaner-models-v1.zip": MODELS})
    run(cfg, now=t0)
    assert mirror.stale_problems(t0 + 600) == []
    assert any("no successful mirror run" in p for p in mirror.stale_problems(t0 + 2 * 3600 + 5))
    # a newer release that cannot be published: after an hour the monitor hears about it
    desktop_release(gh, signer, "2.0.16", bad_sig=True)
    run(cfg, now=t0 + 600)
    assert mirror.stale_problems(t0 + 1200) == []
    problems = mirror.stale_problems(t0 + 600 + 3600 + 5)
    assert len(problems) == 1 and "2.0.16" in problems[0] and "2.0.15 is still served" in problems[0]
    # served again: fine
    gh.releases.pop("desktop-v2.0.16")
    desktop_release(gh, signer, "2.0.16")
    run(cfg, now=t0 + 5000)
    assert mirror.stale_problems(t0 + 5100) == []


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
    desktop_release(gh, signer, "2.0.15")
    run(cfg)
    (cfg.root / "desktop" / "state.json").write_text("not json")
    downloads._cache.clear()
    mirror.prune("desktop")
    assert (cfg.root / "desktop" / "2.0.15" / "update.json").is_file()


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
