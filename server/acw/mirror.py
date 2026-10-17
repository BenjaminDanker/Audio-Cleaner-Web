"""The release mirror: `python -m acw.admin mirror` (a oneshot run every 10 minutes by acw-mirror.timer).

GitHub Releases is where CI puts the build output; users and the apps never talk to it. This job copies the
newest release files to primary-ram, checks them, and only then lets acw/api/downloads_routes.py serve them under
our own domain (the bulk process, so a download never queues in front of live audio). Nothing is stored on the VPS.

Three channels (acw/downloads.py describes the layout):
  desktop     the newest `desktop-v<version>` release: update.json + update.json.sig and the installers it lists
  obs-plugin  the newest `obs-plugin-v<version>` release that has obs-plugin-manifest.json + .sig, and the packages it lists
  models      every `models-v*` archive that deploy/models.lock lists (size and sha256 come from the lock)

Before anything is published:
  * the Ed25519 signature of update.json / obs-plugin-manifest.json is checked against the public key built into
    the apps (raw signature over the exact bytes of the file),
  * every file's size and sha256 (and sha512 when the manifest gives it) are checked, and the size cap of its channel
  * the version must not be older than the one already served (no downgrades).
A failed check publishes nothing; the previous version keeps being served and the monitor mails (mirror_stale).
Publishing is: write the files into a staging folder, move it to <channel>/<version>, then replace state.json (the
allow-list the routes read) atomically, last. The previous version's files stay for 7 days (downloads in progress),
then they are deleted. Interrupted downloads resume from the .part file in the staging folder.

GitHub is read anonymously unless the file ACW_GITHUB_TOKEN_FILE exists (a fine-grained token with Contents: read on
that one repository; needed once the repository is private). The token is only sent to the API address and is
never logged. The repository name is ACW_GITHUB_REPO (one place: DEFAULT_REPO below).
"""
from __future__ import annotations

import base64
import contextlib
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

from . import downloads
from .config import get_settings

log = logging.getLogger("acw.mirror")

DEFAULT_REPO = "BenjaminDanker/Audio-Cleaner-Web"
DEFAULT_API = "https://api.github.com"
# The public half of the key CI signs manifests with (raw Ed25519, base64); the same one is built into the apps.
PUBLIC_KEY = "lkLV2nf15eOuttdcqpieY2QJFyPsF4WOIwidisbp6Hw="

MB = 1024**2
GB = 1024**3
SIZE_CAP = {"desktop": 1 * GB, "obs-plugin": 50 * MB, "models": 2 * GB}
MANIFEST_MAX = 1 * MB
STALE_RUN_S = 2 * 3600       # no successful run for this long: the monitor mails
STALE_RELEASE_S = 3600       # a newer release has been on GitHub for this long without being served: the monitor mails
KEEP_PREVIOUS_S = downloads.KEEP_PREVIOUS_S
STALE_STAGING_S = 2 * 86400
_DESKTOP_TAG = re.compile(r"desktop-v(\d+(?:\.\d+){1,3})")
_OBS_TAG = re.compile(r"obs-plugin-v(\d+\.\d+\.\d+)")
_MODELS_TAG = re.compile(r"models-v\d+[A-Za-z0-9._-]*")
_HEX256 = re.compile(r"[0-9a-f]{64}")
_HEX512 = re.compile(r"[0-9a-f]{128}")
_PLATFORM_KEY = re.compile(r"[a-z0-9][a-z0-9_-]{0,30}")


class MirrorError(Exception):
    """Something is wrong with the release or the download; nothing was published."""


@dataclass
class Config:
    repo: str
    api: str
    token: str | None
    public_key: str
    lock_path: Path | None
    min_free: int
    root: Path

    @classmethod
    def from_env(cls) -> "Config":
        s = get_settings()
        token_file = Path(os.environ.get("ACW_GITHUB_TOKEN_FILE") or (s.home / "secrets" / "github-releases-read.token")).expanduser()
        token = None
        try:
            token = token_file.read_text("utf-8").strip() or None
        except FileNotFoundError:
            pass
        except OSError as e:
            log.warning("cannot read the GitHub token file (%s): reading anonymously", type(e).__name__)
        lock = os.environ.get("ACW_MODELS_LOCK")
        candidates = [Path(lock)] if lock else [s.home / "app" / "models.lock", Path(__file__).resolve().parents[2] / "deploy" / "models.lock"]
        return cls(
            repo=os.environ.get("ACW_GITHUB_REPO") or DEFAULT_REPO,
            api=(os.environ.get("ACW_GITHUB_API") or DEFAULT_API).rstrip("/"),
            token=token,
            public_key=os.environ.get("ACW_UPDATE_PUBLIC_KEY") or PUBLIC_KEY,
            lock_path=next((p for p in candidates if p.is_file()), None),
            min_free=int(float(os.environ.get("ACW_MIRROR_MIN_FREE_GB", "5")) * GB),
            root=downloads.root(),
        )


# ---- signatures ----------------------------------------------------------------------------------------------------

_ED25519_SPKI_PREFIX = bytes.fromhex("302a300506032b6570032100")


def verify_signature(public_key_b64: str, data: bytes, signature: bytes) -> bool:
    """True when `signature` (64 raw bytes) is the Ed25519 signature of exactly `data`. Uses the `cryptography`
    package when it is installed, else the openssl program (3.0 or newer: `pkeyutl -verify -rawin`)."""
    try:
        raw = base64.b64decode(public_key_b64, validate=True)
    except ValueError:
        return False
    if len(raw) != 32 or len(signature) != 64:
        return False
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        pass
    else:
        try:
            Ed25519PublicKey.from_public_bytes(raw).verify(signature, data)
            return True
        except InvalidSignature:
            return False
    with tempfile.TemporaryDirectory(prefix="acw-sig-") as d:
        t = Path(d)
        (t / "key.der").write_bytes(_ED25519_SPKI_PREFIX + raw)
        (t / "data").write_bytes(data)
        (t / "sig").write_bytes(signature)
        try:
            r = subprocess.run(
                ["openssl", "pkeyutl", "-verify", "-rawin", "-pubin", "-keyform", "DER", "-inkey", str(t / "key.der"),
                 "-in", str(t / "data"), "-sigfile", str(t / "sig")],
                capture_output=True, timeout=30, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            raise MirrorError(f"cannot check signatures: openssl failed to run ({type(e).__name__})") from e
        return r.returncode == 0


# ---- GitHub --------------------------------------------------------------------------------------------------------

class GitHub:
    def __init__(self, cfg: Config, client: httpx.Client | None = None):
        self.cfg = cfg
        headers = {"User-Agent": "acw-mirror", "X-GitHub-Api-Version": "2022-11-28"}
        if cfg.token:
            headers["Authorization"] = f"Bearer {cfg.token}"  # httpx drops it when a redirect leaves this address
        self.client = client or httpx.Client(
            headers=headers, timeout=httpx.Timeout(30.0, read=90.0), follow_redirects=True,
            limits=httpx.Limits(max_connections=2),
        )

    def close(self) -> None:
        self.client.close()

    def _get_json(self, path: str, params: dict | None = None, *, missing_ok: bool = False):
        try:
            r = self.client.get(self.cfg.api + path, params=params, headers={"Accept": "application/vnd.github+json"})
        except httpx.HTTPError as e:
            raise MirrorError(f"cannot reach the release service ({type(e).__name__})") from e
        if r.status_code == 404 and missing_ok:
            return None
        if r.status_code in (401, 403, 404):
            hint = " (rate limit: put a token in the token file)" if r.status_code == 403 and "rate limit" in r.text.lower() else ""
            raise MirrorError(f"the release service refused {path}: HTTP {r.status_code}{hint}")
        if r.status_code != 200:
            raise MirrorError(f"the release service answered {path} with HTTP {r.status_code}")
        try:
            return r.json()
        except ValueError as e:
            raise MirrorError(f"the release service answered {path} with something that is not JSON") from e

    def releases(self, pages: int = 5) -> list[dict]:
        """Newest first, up to `pages` x 100: the apps each get a release per build, so one page could hide the
        newest of the other channel, or the older signed ones to fall back to."""
        out: list[dict] = []
        for page in range(1, pages + 1):
            data = self._get_json(f"/repos/{self.cfg.repo}/releases", {"per_page": 100, "page": page})
            batch = [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []
            out += batch
            if len(batch) < 100:
                break
        return out

    def release_by_tag(self, tag: str) -> dict | None:
        data = self._get_json(f"/repos/{self.cfg.repo}/releases/tags/{tag}", missing_ok=True)
        return data if isinstance(data, dict) else None

    def _asset_url(self, asset: dict) -> str:
        url = asset.get("url")
        if not isinstance(url, str) or not url.startswith(self.cfg.api + "/"):
            raise MirrorError("a release asset has an unexpected address")
        return url

    def fetch_small(self, asset: dict, limit: int) -> bytes:
        """The whole asset in memory (manifests and signatures)."""
        if not isinstance(asset.get("size"), int) or asset["size"] > limit:
            raise MirrorError(f"{asset.get('name')} is larger than {limit} bytes")
        out = bytearray()
        try:
            with self.client.stream("GET", self._asset_url(asset), headers={"Accept": "application/octet-stream", "Accept-Encoding": "identity"}) as r:
                if r.status_code != 200:
                    raise MirrorError(f"downloading {asset.get('name')} failed: HTTP {r.status_code}")
                for chunk in r.iter_bytes(65536):
                    out += chunk
                    if len(out) > limit:
                        raise MirrorError(f"{asset.get('name')} is larger than {limit} bytes")
        except httpx.HTTPError as e:
            raise MirrorError(f"downloading {asset.get('name')} failed ({type(e).__name__})") from e
        return bytes(out)

    def download(self, asset: dict, part: Path, size: int) -> None:
        """Stream the asset into `part`, resuming a partial file. Raises MirrorError when it ends up too big or short."""
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink()
            have = 0
        if have == size:
            return
        headers = {"Accept": "application/octet-stream", "Accept-Encoding": "identity"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with self.client.stream("GET", self._asset_url(asset), headers=headers) as r:
                if r.status_code == 206 and have and re.match(rf"bytes {have}-", r.headers.get("content-range", "")):
                    mode = "ab"
                elif r.status_code == 200:
                    mode, have = "wb", 0
                elif r.status_code == 416:
                    part.unlink(missing_ok=True)
                    raise MirrorError(f"the partial download of {asset.get('name')} was not usable; it will start again")
                else:
                    raise MirrorError(f"downloading {asset.get('name')} failed: HTTP {r.status_code}")
                written = have
                with open(part, mode) as f:
                    for chunk in r.iter_bytes():  # as they arrive, so an interruption keeps what was received
                        written += len(chunk)
                        if written > size:
                            f.close()
                            part.unlink(missing_ok=True)
                            raise MirrorError(f"{asset.get('name')} is larger than the size it should have")
                        f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
        except httpx.HTTPError as e:
            raise MirrorError(f"downloading {asset.get('name')} was interrupted ({type(e).__name__}); it resumes next run") from e
        if part.stat().st_size != size:
            raise MirrorError(f"{asset.get('name')} ended early ({part.stat().st_size} of {size} bytes); it resumes next run")


# ---- files ---------------------------------------------------------------------------------------------------------

def hash_file(path: Path) -> tuple[str, str]:
    h256, h512 = hashlib.sha256(), hashlib.sha512()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h256.update(chunk)
            h512.update(chunk)
    return h256.hexdigest(), h512.hexdigest()


def _assets(release: dict) -> dict[str, dict]:
    return {a["name"]: a for a in release.get("assets") or [] if isinstance(a, dict) and isinstance(a.get("name"), str)}


def check_disk(cfg: Config, need: int) -> None:
    cfg.root.mkdir(parents=True, exist_ok=True, mode=0o700)
    free = shutil.disk_usage(cfg.root).free
    if free < need + cfg.min_free:
        raise MirrorError(
            f"not enough disk space: {need / MB:.0f} MB to download, {free / MB:.0f} MB free, "
            f"{cfg.min_free / MB:.0f} MB must stay free")


def fetch_verified(cfg: Config, gh: GitHub, asset: dict, staging: Path, name: str, size: int, sha256: str, sha512: str | None) -> Path:
    """The asset as a verified file `staging/name`. A file already there from an earlier run is hashed again (never
    trusted); a download in progress (`name.part`) is resumed. Raises MirrorError if the bytes are not the expected ones."""
    final = staging / name
    if final.exists():
        if final.stat().st_size == size and _digests_match(final, sha256, sha512):
            return final
        final.unlink()
    part = staging / (name + ".part")
    have = part.stat().st_size if part.exists() else 0
    check_disk(cfg, max(0, size - have))
    gh.download(asset, part, size)
    if not _digests_match(part, sha256, sha512):
        part.unlink(missing_ok=True)
        raise MirrorError(f"{name} does not match its checksum; it was thrown away")
    os.replace(part, final)
    return final


def _digests_match(path: Path, sha256: str, sha512: str | None) -> bool:
    d256, d512 = hash_file(path)
    return d256 == sha256 and (sha512 is None or d512 == sha512)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_file(path: Path, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


# ---- manifests -----------------------------------------------------------------------------------------------------

def parse_manifest(channel: str, data: bytes) -> tuple[str, dict[str, dict]]:
    """(version, {platform key: {"name", "size", "sha256", "sha512"?}}) of a signed manifest; MirrorError if the
    contents are not acceptable (this is called after the signature was checked)."""
    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise MirrorError("the manifest is not readable") from e
    if not isinstance(raw, dict) or not isinstance(raw.get("files"), dict) or not raw["files"]:
        raise MirrorError("the manifest has no files")
    version = raw.get("version")
    if downloads.vtuple(version) is None:
        raise MirrorError("the manifest has no valid version")
    files: dict[str, dict] = {}
    names: set[str] = set()
    for key, e in raw["files"].items():
        if not isinstance(key, str) or _PLATFORM_KEY.fullmatch(key) is None:
            raise MirrorError("the manifest has an unexpected platform name")
        if not isinstance(e, dict):
            raise MirrorError(f"the manifest entry for {key} is not valid")
        name, size, sha256, sha512 = e.get("name"), e.get("size"), e.get("sha256"), e.get("sha512")
        if not isinstance(name, str) or downloads.NAME_RE.fullmatch(name) is None or ".." in name:
            raise MirrorError(f"the manifest names an unexpected file for {key}")
        if name in names or name in {m[0] for m in downloads.MANIFESTS.get(channel, ())}:
            raise MirrorError(f"the manifest lists {name} twice or names a manifest file")
        names.add(name)
        if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= SIZE_CAP[channel]:
            raise MirrorError(f"{name} is larger than the {SIZE_CAP[channel] // MB} MB allowed for {channel}, or has no size")
        if not isinstance(sha256, str) or _HEX256.fullmatch(sha256) is None:
            raise MirrorError(f"the manifest has no valid sha256 for {name}")
        if sha512 is not None and (not isinstance(sha512, str) or _HEX512.fullmatch(sha512) is None):
            raise MirrorError(f"the manifest has an invalid sha512 for {name}")
        files[key] = {"name": name, "size": size, "sha256": sha256, **({"sha512": sha512} if sha512 else {})}
    return str(version), files


# ---- publishing ----------------------------------------------------------------------------------------------------

def _state_or_empty(channel: str) -> dict:
    s = downloads.load_state(channel)
    return dict(s) if isinstance(s, dict) else {"current": None, "previous": [], "seen": None}


def _mark_seen(channel: str, version: str | None, now: float) -> None:
    """Remember since when a release newer than the served one exists (the monitor alerts after an hour)."""
    st = _state_or_empty(channel)
    seen = st.get("seen")
    if version is None:
        if seen is not None:
            st["seen"] = None
            downloads.write_state(channel, st)
        return
    if isinstance(seen, dict) and seen.get("version") == version:
        return
    st["seen"] = {"version": version, "since": int(now)}
    downloads.write_state(channel, st)


def _publish(channel: str, new_current: dict, now: float) -> None:
    """Replace state.json (last step of publishing): the old current moves to `previous` if any file of it is no
    longer served, entries older than a week are forgotten, then everything unreferenced is deleted."""
    st = _state_or_empty(channel)
    old = st.get("current")
    previous = [p for p in st.get("previous") or [] if isinstance(p, dict) and now - float(p.get("retired_at") or 0) <= KEEP_PREVIOUS_S]
    if isinstance(old, dict) and isinstance(old.get("files"), dict):
        dropped = {n: e for n, e in old["files"].items() if new_current["files"].get(n) != e}
        if dropped:
            previous.insert(0, {"version": old.get("version"), "files": dropped, "retired_at": int(now)})
    st.update(current=new_current, previous=previous, seen=None)
    downloads.write_state(channel, st)
    prune(channel, now)


def prune(channel: str, now: float | None = None) -> list[str]:
    """Delete everything in the channel folder that state.json does not reference (old versions after their week,
    leftovers of failed publishes, abandoned staging folders). Returns what was removed."""
    now = time.time() if now is None else now
    base = downloads.channel_dir(channel)
    st = downloads.load_state(channel) or {}
    expired = [p for p in st.get("previous") or [] if isinstance(p, dict) and now - float(p.get("retired_at") or 0) > KEEP_PREVIOUS_S]
    if expired:
        st = {**st, "previous": [p for p in st["previous"] if p not in expired]}
        downloads.write_state(channel, st)
    keep: set[tuple[str, str]] = set()
    for group in [st.get("current")] + list(st.get("previous") or []):
        if isinstance(group, dict) and isinstance(group.get("files"), dict):
            if group is not st.get("current") and now - float(group.get("retired_at") or 0) > KEEP_PREVIOUS_S:
                continue
            keep |= {(e["dir"], n) for n, e in group["files"].items() if isinstance(e, dict) and isinstance(e.get("dir"), str)}
    removed: list[str] = []
    have_state = downloads.load_state(channel) is not None  # without a readable state nothing but staging is touched
    try:
        entries = sorted(base.iterdir())
    except OSError:
        return removed
    for d in entries:
        if d.name == "state.json" or d.name.startswith("state.json.tmp"):
            continue
        if d.is_symlink() or not d.is_dir():
            continue  # not ours; leave it
        if d.name.startswith(".staging-"):
            with contextlib.suppress(OSError):
                if now - d.stat().st_mtime > STALE_STAGING_S:
                    shutil.rmtree(d, ignore_errors=True)
                    removed.append(d.name)
            continue
        if not have_state:
            continue
        if d.name.startswith(".old-"):
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d.name)
            continue
        for f in sorted(d.iterdir()):
            if (d.name, f.name) not in keep and not f.is_dir():
                with contextlib.suppress(OSError):
                    f.unlink()
                    removed.append(f"{d.name}/{f.name}")
        with contextlib.suppress(OSError):
            d.rmdir()  # only if now empty
            removed.append(d.name)
    return removed


def _publish_version_dir(base: Path, staging: Path, version: str) -> str:
    """Move a finished staging folder to <channel>/<version>. A folder already there (a repair of the same version) is
    moved aside first and removed by prune."""
    final = base / version
    if final.exists():
        os.rename(final, base / f".old-{version}-{int(time.time())}")
    os.rename(staging, final)
    _fsync_dir(base)
    return version


# ---- channels ------------------------------------------------------------------------------------------------------

@dataclass
class Ctx:
    cfg: Config
    gh: GitHub
    now: float
    force: bool = False


def _sync_signed(ctx: Ctx, channel: str, release: dict, hint: str | None, manifest_name: str, sig_name: str) -> str:
    """Mirror one release of a signed channel. `hint` is the version in the release tag."""
    cfg, gh = ctx.cfg, ctx.gh
    base = downloads.channel_dir(channel)
    st = _state_or_empty(channel)
    cur = st.get("current") if isinstance(st.get("current"), dict) else None
    cur_v = downloads.vtuple(cur.get("version")) if cur else None
    assets = _assets(release)
    m_asset, s_asset = assets[manifest_name], assets[sig_name]
    src = f"{m_asset.get('id')}:{m_asset.get('updated_at')}:{s_asset.get('id')}:{s_asset.get('updated_at')}"
    intact = bool(cur) and downloads.served_intact(channel, cur.get("files") or {})

    if not ctx.force and cur and intact:
        hv = downloads.vtuple(hint) if hint else None
        if hv is not None and cur_v is not None and hv < cur_v:
            _mark_seen(channel, None, ctx.now)
            return f"GitHub's newest is {hint}, older than the {cur['version']} that is served; ignored"
        if cur.get("src") == src:
            _mark_seen(channel, None, ctx.now)
            return f"{cur['version']} is up to date"
    if hint is not None:
        _mark_seen(channel, hint, ctx.now)  # a newer release exists; the monitor mails if it is not served within an hour

    manifest = gh.fetch_small(m_asset, MANIFEST_MAX)
    sig = gh.fetch_small(s_asset, 4096)
    try:
        if not verify_signature(cfg.public_key, manifest, sig):
            raise MirrorError(f"{manifest_name} is not signed with the key the apps trust; nothing was published")
        version, entries = parse_manifest(channel, manifest)
    except MirrorError:
        _mark_seen(channel, hint or "unverified", ctx.now)
        raise
    if hint is not None and version != hint:
        raise MirrorError(f"the manifest says version {version} but the release is {hint}; nothing was published")
    nv = downloads.vtuple(version)
    if cur_v is not None and nv is not None and not ctx.force:
        if nv < cur_v:
            _mark_seen(channel, None, ctx.now)
            return f"{version} is older than the {cur['version']} that is served; ignored"
        if nv == cur_v and intact:
            st["current"] = {**cur, "src": src}
            st["seen"] = None
            downloads.write_state(channel, st)
            return f"{version} is up to date"
    _mark_seen(channel, version, ctx.now)

    for key, e in entries.items():
        a = assets.get(e["name"])
        if a is None:
            raise MirrorError(f"the release has no file {e['name']} that the manifest lists")
        if a.get("size") != e["size"]:
            raise MirrorError(f"{e['name']} on the release is {a.get('size')} bytes, the manifest says {e['size']}")

    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = base / f".staging-{version}"
    staging.mkdir(exist_ok=True, mode=0o700)
    for e in entries.values():
        fetch_verified(cfg, gh, assets[e["name"]], staging, e["name"], e["size"], e["sha256"], e.get("sha512"))
    _write_file(staging / manifest_name, manifest)
    _write_file(staging / sig_name, sig)
    _fsync_dir(staging)
    for f in staging.iterdir():
        if f.name not in {e["name"] for e in entries.values()} | {manifest_name, sig_name}:
            f.unlink()  # a leftover (.part of something no longer listed)
    _publish_version_dir(base, staging, version)

    files = {e["name"]: {"size": e["size"], "sha256": e["sha256"], "dir": version, **({"sha512": e["sha512"]} if e.get("sha512") else {})}
             for e in entries.values()}
    for n, data in ((manifest_name, manifest), (sig_name, sig)):
        files[n] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "dir": version}
    new_current = {
        "version": version, "published_at": int(ctx.now), "src": src, "files": files,
        "platforms": {k: e["name"] for k, e in entries.items()},
    }
    _publish(channel, new_current, ctx.now)
    log.info("%s: published %s (%d files)", channel, version, len(entries))
    return f"published {version}"


def sync_desktop(ctx: Ctx) -> str:
    cands = []
    for rel in ctx.gh.releases():
        m = _DESKTOP_TAG.fullmatch(str(rel.get("tag_name") or ""))
        if m and not rel.get("draft"):
            cands.append((downloads.vtuple(m.group(1)), m.group(1), rel))
    for _, hint, rel in sorted(cands, key=lambda c: c[0], reverse=True):
        assets = _assets(rel)
        if "update.json" in assets and "update.json.sig" in assets:
            return _sync_signed(ctx, "desktop", rel, hint, "update.json", "update.json.sig")
        log.info("desktop: release %s has no signed update information yet; looking at older ones", hint)
    return "no desktop release with update information yet"


def sync_obs(ctx: Ctx) -> str:
    cands = []
    for rel in ctx.gh.releases():
        m = _OBS_TAG.fullmatch(str(rel.get("tag_name") or ""))
        if m and not rel.get("draft"):
            cands.append((downloads.vtuple(m.group(1)), m.group(1), rel))
    for _, hint, rel in sorted(cands, key=lambda c: c[0], reverse=True):
        assets = _assets(rel)
        if "obs-plugin-manifest.json" in assets and "obs-plugin-manifest.json.sig" in assets:
            return _sync_signed(ctx, "obs-plugin", rel, hint, "obs-plugin-manifest.json", "obs-plugin-manifest.json.sig")
        log.info("obs-plugin: release %s has no signed manifest yet; looking at older ones", hint)
    return "no plugin release with a signed manifest yet"


def parse_lock(path: Path) -> list[dict]:
    """The `archive` lines of deploy/models.lock that come from a models-v* release:
    [{"tag", "name", "sha256", "size"}]. Columns: kind flags sha256 size NAME URL."""
    out = []
    for line in path.read_text("utf-8").splitlines():
        if not line.startswith("archive"):
            continue
        parts = line.split(None, 5)
        if len(parts) != 6:
            continue
        _, _, sha, size, tag, url = parts
        name = unquote(urlparse(url.strip()).path.rsplit("/", 1)[-1])
        if _MODELS_TAG.fullmatch(tag) and _HEX256.fullmatch(sha) and size.isdigit() and downloads.NAME_RE.fullmatch(name):
            out.append({"tag": tag, "name": name, "sha256": sha, "size": int(size)})
    return out


def sync_models(ctx: Ctx) -> str:
    cfg, gh = ctx.cfg, ctx.gh
    if cfg.lock_path is None:
        raise MirrorError("deploy/models.lock was not found (ACW_MODELS_LOCK)")
    wanted = parse_lock(cfg.lock_path)
    if not wanted:
        return "deploy/models.lock lists no models release"
    channel = "models"
    base = downloads.channel_dir(channel)
    st = _state_or_empty(channel)
    cur = st.get("current") if isinstance(st.get("current"), dict) else None
    cur_files = (cur or {}).get("files") or {}
    files: dict[str, dict] = {}
    releases: dict[str, dict | None] = {}
    fetched = []
    for w in wanted:
        if w["size"] > SIZE_CAP[channel]:
            raise MirrorError(f"{w['name']} is larger than the {SIZE_CAP[channel] // MB} MB allowed for models")
        entry = {"size": w["size"], "sha256": w["sha256"], "dir": w["tag"]}
        if not ctx.force and cur_files.get(w["name"]) == entry and downloads.served_intact(channel, {w["name"]: entry}):
            files[w["name"]] = entry
            continue
        _mark_seen(channel, w["tag"], ctx.now)
        if w["tag"] not in releases:
            releases[w["tag"]] = gh.release_by_tag(w["tag"])
        rel = releases[w["tag"]]
        asset = _assets(rel).get(w["name"]) if rel else None
        if asset is None:
            raise MirrorError(f"release {w['tag']} has no file {w['name']}")
        if asset.get("size") != w["size"]:
            raise MirrorError(f"{w['name']} on the release is {asset.get('size')} bytes, deploy/models.lock says {w['size']}")
        base.mkdir(parents=True, exist_ok=True, mode=0o700)
        staging = base / f".staging-{w['tag']}"
        staging.mkdir(exist_ok=True, mode=0o700)
        done = fetch_verified(cfg, gh, asset, staging, w["name"], w["size"], w["sha256"], None)
        (base / w["tag"]).mkdir(exist_ok=True, mode=0o700)
        os.replace(done, base / w["tag"] / w["name"])
        _fsync_dir(base / w["tag"])
        with contextlib.suppress(OSError):
            staging.rmdir()
        files[w["name"]] = entry
        fetched.append(w["name"])
    if files == cur_files and cur:
        _mark_seen(channel, None, ctx.now)
        return "models are up to date"
    top = max((w["tag"] for w in wanted), key=lambda t: [int(x) if x.isdigit() else 0 for x in re.split(r"\D+", t)])
    _publish(channel, {"version": top, "published_at": int(ctx.now), "files": files, "platforms": {}}, ctx.now)
    log.info("models: now serving %s", ", ".join(sorted(files)))
    return f"published {', '.join(fetched) or 'a changed list'}"


CHANNEL_SYNC = (("desktop", sync_desktop), ("obs-plugin", sync_obs), ("models", sync_models))


# ---- the run -------------------------------------------------------------------------------------------------------

def status_path(cfg: Config | None = None) -> Path:
    return (cfg.root if cfg else downloads.root()) / "mirror-status.json"


def load_status(cfg: Config | None = None) -> dict:
    try:
        data = json.loads(status_path(cfg).read_text("utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def run_once(cfg: Config | None = None, *, gh: GitHub | None = None, now: float | None = None, force: bool = False,
             only: str | None = None) -> dict[str, str]:
    """One pass over the channels. Returns {channel: outcome text}; a failing channel does not stop the others.
    Only one pass runs at a time (a second one exits at once)."""
    cfg = cfg or Config.from_env()
    now = time.time() if now is None else now
    cfg.root.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = open(cfg.root / ".mirror.lock", "w")
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return {"-": "another mirror run is in progress"}
        own = gh is None
        gh = gh or GitHub(cfg)
        ctx = Ctx(cfg, gh, now, force)
        status = load_status(cfg)
        chans = status.setdefault("channels", {})
        results: dict[str, str] = {}
        try:
            for channel, fn in CHANNEL_SYNC:
                if only and channel != only:
                    continue
                c = chans.setdefault(channel, {})
                c["last_run"] = int(now)
                try:
                    results[channel] = fn(ctx)
                    c.update(last_ok=int(now), result=results[channel], error=None)
                    log.info("%s: %s", channel, results[channel])
                except MirrorError as e:
                    results[channel] = f"FAILED: {e}"
                    c.update(error=str(e)[:500], last_error_at=int(now))
                    log.error("%s: %s", channel, e)
                except Exception as e:  # noqa: BLE001
                    log.exception("%s: unexpected error", channel)
                    results[channel] = f"FAILED: unexpected error ({type(e).__name__})"
                    c.update(error=f"unexpected error ({type(e).__name__})", last_error_at=int(now))
                with contextlib.suppress(Exception):
                    prune(channel, now)
        finally:
            status["last_run"] = int(now)
            downloads.write_json_atomic(status_path(cfg), status)
            if own:
                gh.close()
        return results
    finally:
        lock.close()


def verify_served(cfg: Config | None = None) -> list[str]:
    """Hash every file that is being served and compare with state.json. Returns the problems (empty: all good)."""
    cfg = cfg or Config.from_env()
    problems = []
    for channel in downloads.CHANNELS:
        st = downloads.load_state(channel)
        for group in ([st.get("current")] if st else []) + list((st or {}).get("previous") or []):
            for name, e in ((group or {}).get("files") or {}).items():
                p = downloads.channel_dir(channel) / e["dir"] / name
                try:
                    if p.stat().st_size != e["size"] or hash_file(p)[0] != e["sha256"]:
                        problems.append(f"{channel}/{e['dir']}/{name}: does not match its recorded checksum")
                except OSError:
                    problems.append(f"{channel}/{e['dir']}/{name}: missing")
    return problems


# ---- monitor -------------------------------------------------------------------------------------------------------

def stale_problems(now: float | None = None) -> list[str]:
    """Why the mirror is not doing its job (empty: fine): a channel without a successful run for 2 h, or a release
    newer than the served one for more than an hour."""
    now = time.time() if now is None else now
    chans = load_status().get("channels") or {}
    if not chans:
        return ["the mirror has never run"]
    out = []
    for channel, c in sorted(chans.items()):
        last_ok = c.get("last_ok")
        if not isinstance(last_ok, (int, float)) or now - last_ok > STALE_RUN_S:
            err = f" Last error: {c['error']}" if c.get("error") else ""
            ago = "never" if not isinstance(last_ok, (int, float)) else f"{(now - last_ok) / 3600:.1f} hours ago"
            out.append(f"{channel}: no successful mirror run (last success {ago}).{err}")
    for channel in downloads.CHANNELS:
        st = downloads.load_state(channel)
        seen = (st or {}).get("seen")
        if isinstance(seen, dict) and now - float(seen.get("since") or now) > STALE_RELEASE_S:
            served = ((st or {}).get("current") or {}).get("version", "nothing")
            err = (chans.get(channel) or {}).get("error")
            out.append(f"{channel}: release {seen.get('version')} has been on the release service for "
                       f"{(now - float(seen['since'])) / 3600:.1f} hours but {served} is still served."
                       + (f" Last error: {err}" if err else ""))
    return out


def describe() -> list[str]:
    """What is served, one line per channel (acw.admin mirror --status)."""
    lines = []
    status = load_status()
    for channel in downloads.CHANNELS:
        st = downloads.load_state(channel)
        cur = (st or {}).get("current")
        c = (status.get("channels") or {}).get(channel) or {}
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(c["last_ok"])) if c.get("last_ok") else "never"
        served = f"serving {cur['version']} ({len(cur['files'])} files)" if cur else "nothing served yet"
        prev = ", ".join(str(p.get("version")) for p in (st or {}).get("previous") or [])
        err = f"; last error: {c['error']}" if c.get("error") else ""
        lines.append(f"{channel:11s} {served}; last ok {when}{'; kept ' + prev if prev else ''}{err}")
    return lines


def main(force: bool = False, verify: bool = False, status: bool = False, only: str | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # httpx logs every request, including GitHub's short-lived signed download addresses: not worth keeping.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if status:
        print("\n".join(describe()))
        return 0
    if verify:
        problems = verify_served()
        print("\n".join(problems) if problems else "all served files match their checksums")
        return 1 if problems else 0
    results = run_once(force=force, only=only)
    for channel, text in results.items():
        print(f"{channel}: {text}")
    return 1 if any(t.startswith("FAILED") for t in results.values()) else 0
