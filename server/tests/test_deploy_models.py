"""deploy/models.lock and deploy/fetch-models.sh (DEP-01): the lock is well formed, and the script fetches, verifies,
repairs and checks without touching what is right. Downloads are file:// URLs, so nothing leaves the machine."""
import hashlib
import os
import re
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

DEPLOY = Path(__file__).resolve().parents[2] / "deploy"
SCRIPT = DEPLOY / "fetch-models.sh"
pytestmark = pytest.mark.skipif(not (shutil.which("curl") and shutil.which("bash")), reason="needs curl and bash")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_lock(path: Path):
    entries = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        kind, flags, digest, size, target, rest = line.split(None, 5)
        entries.append(dict(line=n, kind=kind, flags=flags, sha=digest, size=int(size), target=target, rest=rest))
    return entries


# ------------------------------------------------------------------ the real lock

def test_lock_is_well_formed_and_pinned():
    entries = parse_lock(DEPLOY / "models.lock")
    assert len(entries) > 40
    archives = {e["target"] for e in entries if e["kind"] == "archive"}
    seen = set()
    for e in entries:
        assert e["kind"] in ("file", "archive", "member", "local"), e
        assert e["flags"] in ("-", "optional"), e
        assert re.fullmatch(r"[0-9a-f]{64}", e["sha"]) and e["size"] > 0, e
        if e["kind"] != "archive":
            assert e["target"] not in seen, f"duplicate target {e['target']}"
            seen.add(e["target"])
            assert not e["target"].startswith("/") and ".." not in e["target"]
            assert e["target"].split("/")[0] in ("models", "bin", "wasm")
        if e["kind"] == "file":
            assert e["rest"].startswith("https://"), e
            assert not re.search(r"/(main|master|latest)/|-latest-", e["rest"]), f"unpinned URL: {e['rest']}"
            if "huggingface.co" in e["rest"]:
                assert re.search(r"/resolve/[0-9a-f]{40}/", e["rest"]), f"Hugging Face URL without a commit id: {e['rest']}"
            if "github.com" in e["rest"]:
                assert re.search(r"/raw/[0-9a-f]{40}/", e["rest"]), e["rest"]
        if e["kind"] == "archive":
            assert e["rest"].startswith("https://") and "latest" not in e["rest"], e
        if e["kind"] == "member":
            assert e["rest"].split(":", 1)[0] in archives, f"member of an archive the lock does not define: {e['rest']}"


def test_lock_covers_what_the_product_needs():
    from acw import health

    targets = {e["target"] for e in parse_lock(DEPLOY / "models.lock")}
    for m in health.REQUIRED_MODELS:
        assert any(t == f"models/{m}" or t.startswith(f"models/{m}/") for t in targets), f"{m} is not in models.lock"
    for b in health.REQUIRED_BINARIES:
        assert f"bin/{b}" in targets


# ------------------------------------------------------------------ pinned package hashes (SEC-26)

def _hash_lock():
    import importlib.util

    spec = importlib.util.spec_from_file_location("hash_lock", DEPLOY / "hash_lock.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_both_locks_pin_every_package_with_hashes_and_an_unpinned_requirement_is_refused():
    hl = _hash_lock()
    for lock in (DEPLOY.parent / "server" / "requirements.lock", DEPLOY / "requirements-gpu.lock"):
        items = hl.parse(lock.read_text())
        assert len([i for i in items if "name" in i]) > 40
        assert hl.verify(items) == [], lock
        assert all(len(h) == 64 for i in items if "name" in i for h in i["hashes"])
    with pytest.raises(SystemExit):
        hl.parse("foo>=1.0\n")


def test_gpu_lock_agrees_with_the_server_lock():
    hl = _hash_lock()
    norm = lambda n: n.lower().replace("_", "-")  # noqa: E731
    server = {norm(i["name"]): i for i in hl.parse((DEPLOY.parent / "server" / "requirements.lock").read_text()) if "name" in i}
    gpu = {norm(i["name"]): i for i in hl.parse((DEPLOY / "requirements-gpu.lock").read_text()) if "name" in i}
    for name, it in gpu.items():
        if name in server:
            assert it["version"] == server[name]["version"] and it["hashes"] == server[name]["hashes"], name
    assert "onnxruntime-gpu" in gpu and "onnxruntime" not in gpu and "fastapi" not in gpu


def test_installers_require_hashes():
    assert "--require-hashes" in (DEPLOY / "install.sh").read_text()
    assert "--require-hashes" in (DEPLOY / "setup-gpu.sh").read_text()
    assert "requirements-gpu.lock" in (DEPLOY / "setup-gpu.sh").read_text()
    assert "--require-hashes" in (DEPLOY.parent / ".github" / "workflows" / "server.yml").read_text()


# ------------------------------------------------------------------ the script on a toy lock

class Toy:
    """A source directory with files and two archives, a lock describing them, and an empty home."""

    def __init__(self, tmp: Path):
        self.src = tmp / "src"
        self.home = tmp / "home"
        self.src.mkdir()
        self.home.mkdir()
        self.data = {
            "a.bin": b"alpha" * 1000,
            "b.txt": b"bravo\n",
            "c.opt": b"charlie" * 50,
            "zipped.onnx": b"Z" * 3000,
            "tarred": b"#!/bin/sh\necho hi\n",
        }
        for name in ("a.bin", "b.txt", "c.opt"):
            (self.src / name).write_bytes(self.data[name])
        with zipfile.ZipFile(self.src / "pack.zip", "w") as z:
            z.writestr("dir/zipped.onnx", self.data["zipped.onnx"])
        (self.src / "tree" / "x-1.0" / "bin").mkdir(parents=True)
        (self.src / "tree" / "x-1.0" / "bin" / "tarred").write_bytes(self.data["tarred"])
        with tarfile.open(self.src / "pack.tar.xz", "w:xz") as t:
            t.add(self.src / "tree" / "x-1.0", arcname="x-1.0")
        self.local = b"built here"
        self.lock = tmp / "models.lock"
        zp, tp = (self.src / "pack.zip").read_bytes(), (self.src / "pack.tar.xz").read_bytes()
        rows = [
            f"archive - {sha(zp)} {len(zp)} zipp file://{self.src}/pack.zip",
            f"archive - {sha(tp)} {len(tp)} tarp file://{self.src}/pack.tar.xz",
            f"file - {sha(self.data['a.bin'])} {len(self.data['a.bin'])} models/a/a.bin file://{self.src}/a.bin",
            f"file - {sha(self.data['b.txt'])} {len(self.data['b.txt'])} models/b.txt file://{self.src}/b.txt",
            f"file optional {sha(self.data['c.opt'])} {len(self.data['c.opt'])} models/c.opt file://{self.src}/c.opt",
            f"member - {sha(self.data['zipped.onnx'])} {len(self.data['zipped.onnx'])} models/z/zipped.onnx zipp:dir/zipped.onnx",
            f"member - {sha(self.data['tarred'])} {len(self.data['tarred'])} bin/tarred tarp:x-1.0/bin/tarred",
            f"local - {sha(self.local)} {len(self.local)} bin/built.so made here, cannot be downloaded",
        ]
        self.lock.write_text("# toy\n" + "\n".join(rows) + "\n")

    def run(self, *args, env=None, check=False):
        e = {**os.environ, "ACW_HOME": str(self.home), **(env or {})}
        r = subprocess.run(["bash", str(SCRIPT), "--lock", str(self.lock), *args], capture_output=True, text=True, env=e)
        if check:
            assert r.returncode == 0, r.stdout + r.stderr
        return r

    def build(self):
        (self.home / "bin").mkdir(exist_ok=True)
        (self.home / "bin" / "built.so").write_bytes(self.local)


@pytest.fixture
def toy(tmp_path):
    return Toy(tmp_path)


def test_fetches_everything_required_verifies_and_a_second_run_changes_nothing(toy):
    toy.build()
    r = toy.run(check=True)
    assert "1 ok, 4 fetched" in r.stdout
    assert (toy.home / "models/a/a.bin").read_bytes() == toy.data["a.bin"]
    assert (toy.home / "models/z/zipped.onnx").read_bytes() == toy.data["zipped.onnx"]  # from the zip
    assert (toy.home / "bin/tarred").read_bytes() == toy.data["tarred"]  # from the tar.xz
    assert os.access(toy.home / "bin/tarred", os.X_OK) and not os.access(toy.home / "models/b.txt", os.X_OK)
    assert not (toy.home / "models/c.opt").exists()  # optional: not fetched by default
    assert not list(toy.home.rglob("*.part*"))
    before = {p: p.stat().st_mtime_ns for p in toy.home.rglob("*") if p.is_file()}
    r = toy.run(check=True)
    assert "fetched" not in r.stdout and "5 ok" in r.stdout
    assert {p: p.stat().st_mtime_ns for p in toy.home.rglob("*") if p.is_file()} == before  # never touches a correct file


def test_check_reports_and_changes_nothing(toy):
    toy.build()
    r = toy.run("--check")
    assert r.returncode == 1 and "MISSING  models/a/a.bin" in r.stderr and "bin/tarred" in r.stderr
    assert sorted(p.name for p in toy.home.rglob("*") if p.is_file()) == ["built.so"]  # nothing downloaded
    toy.run(check=True)
    ok = toy.run("--check")
    assert ok.returncode == 0 and "models: 5 ok, 1 optional absent" in ok.stdout


def test_a_wrong_file_is_found_and_repaired(toy):
    toy.build()
    toy.run(check=True)
    (toy.home / "models/b.txt").write_bytes(b"tampered")
    (toy.home / "models/z/zipped.onnx").write_bytes(b"Z" * 2999)  # right content, wrong size
    r = toy.run("--check")
    assert r.returncode == 1 and r.stderr.count("WRONG") == 2
    toy.run(check=True)
    assert (toy.home / "models/b.txt").read_bytes() == toy.data["b.txt"]
    assert (toy.home / "models/z/zipped.onnx").read_bytes() == toy.data["zipped.onnx"]
    assert toy.run("--check").returncode == 0


def test_downloads_and_archives_with_the_wrong_hash_are_refused_and_the_rest_still_fetched(toy):
    toy.build()
    (toy.src / "a.bin").write_bytes(b"not what the lock says")
    (toy.src / "pack.zip").write_bytes(b"PK corrupted")
    r = toy.run()
    assert r.returncode == 1 and "FAILED   models/a/a.bin" in r.stderr and "archive pack.zip" in r.stderr
    assert not (toy.home / "models/a/a.bin").exists() and not (toy.home / "models/z/zipped.onnx").exists()
    assert not list(toy.home.rglob("*.part*"))
    assert (toy.home / "models/b.txt").exists() and (toy.home / "bin/tarred").exists()  # the others were fine


def test_optional_files_are_never_fetched_but_a_present_one_must_be_right(toy):
    toy.build()
    toy.run(check=True)
    assert not (toy.home / "models/c.opt").exists()  # never fetched, and its absence is not an error
    r = toy.run("--check", "-v")
    assert r.returncode == 0 and "absent   models/c.opt (optional)" in r.stdout
    assert "not needed to run; -v names them" in toy.run("--check").stdout
    assert toy.run("--with-optional").returncode != 0 and not (toy.home / "models/c.opt").exists()  # the flag is gone
    (toy.home / "models/c.opt").write_bytes(b"x")  # a present optional file must be right
    r = toy.run("--check")
    assert r.returncode == 1 and "WRONG" in r.stderr
    (toy.home / "models/c.opt").write_bytes(toy.data["c.opt"])
    assert toy.run("--check").returncode == 0


def test_local_files_are_checked_not_fetched(toy):
    toy.run()  # no bin/built.so
    r = toy.run("--check")
    assert r.returncode == 1 and "MISSING  bin/built.so: cannot be downloaded" in r.stderr
    toy.build()
    assert toy.run("--check").returncode == 0
    (toy.home / "bin/built.so").write_bytes(b"rebuilt")  # a rebuild differs: a warning, not an error
    r = toy.run("--check")
    assert r.returncode == 0 and "changed  bin/built.so" in r.stderr and "1 local files changed" in r.stdout


def test_only_and_dest_dir(toy, tmp_path):
    toy.run("--only", "models/b.txt", "--dest-dir", str(tmp_path / "elsewhere"), check=True)
    assert (tmp_path / "elsewhere/b.txt").read_bytes() == toy.data["b.txt"]
    assert not (toy.home / "models/b.txt").exists() and not (toy.home / "models/a/a.bin").exists()


def test_local_archive_copy_is_used_without_downloading(toy, tmp_path):
    toy.build()
    spare = tmp_path / "spare"
    spare.mkdir()
    shutil.copy(toy.src / "pack.zip", spare / "pack.zip")
    (toy.src / "pack.zip").unlink()  # the "download" would now fail
    r = toy.run("--only", "models/z/*", "--archive", str(spare / "pack.zip"), check=True)
    assert "fetching archive" not in r.stdout and (toy.home / "models/z/zipped.onnx").exists()
    (toy.home / "models/z/zipped.onnx").unlink()
    toy.run("--only", "models/z/*", env={"ACW_ARCHIVE_DIR": str(spare)}, check=True)
    assert (toy.home / "models/z/zipped.onnx").exists()
