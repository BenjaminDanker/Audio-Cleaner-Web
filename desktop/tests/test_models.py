"""Downloading speech models: pinned files, resume, verification, nothing half-done is ever used."""
from __future__ import annotations

import hashlib
import http.server
import random
import threading

import pytest

from audio_cleaner_desktop import models


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    files: dict[str, bytes] = {}
    requests: list[str] = []

    def do_GET(self):  # noqa: N802
        body = self.files.get(self.path)
        self.requests.append(f"{self.path} {self.headers.get('Range', '')}".strip())
        if body is None:
            self.send_error(404)
            return
        start = 0
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            start = int(rng[6:].split("-")[0])
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(body) - start))
        self.end_headers()
        self.wfile.write(body[start:])

    def log_message(self, *_args):
        pass


@pytest.fixture
def server(monkeypatch, tmp_path):
    _RangeHandler.files, _RangeHandler.requests = {}, []
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    monkeypatch.setenv("ACW_DESKTOP_MODELS", str(tmp_path / "models"))
    # the local test server is plain http (the real catalog is https only)
    monkeypatch.setattr(models, "_ssl_context", lambda: None)
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _model(base: str, payloads: dict[str, bytes], sha_override: dict[str, str] | None = None) -> models.SpeechModel:
    for name, data in payloads.items():
        _RangeHandler.files[f"/{name}"] = data
    files = tuple(
        models.ModelFile(name, f"{base}/{name}", len(data), (sha_override or {}).get(name, hashlib.sha256(data).hexdigest()))
        for name, data in payloads.items()
    )
    return models.SpeechModel("test-model", "Test", "for tests", "MIT", files)


def test_download_verifies_and_installs(server):
    m = _model(server, {"a.bin": b"a" * 3_000_000, "sub/b.json": b"{}"})
    seen = []
    assert not models.is_installed(m) and models.missing_bytes(m) == m.size
    path = models.download(m, progress=lambda done, total: seen.append((done, total)))
    assert models.is_installed(m) and path == models.model_dir(m)
    assert (path / "a.bin").read_bytes() == b"a" * 3_000_000 and (path / "sub" / "b.json").read_bytes() == b"{}"
    assert seen[-1] == (m.size, m.size) and all(t == m.size for _, t in seen)
    assert not (models.models_dir() / "test-model.partial").exists()
    assert models.download(m) == path and len(_RangeHandler.requests) == 2  # already there: nothing fetched


def test_an_interrupted_download_resumes(server):
    data = bytes(range(256)) * 20_000
    m = _model(server, {"big.bin": data})
    part = models.models_dir() / "test-model.partial"
    part.mkdir(parents=True)
    (part / ".pin").write_text(models.pin_token(m))  # a partial download of this very model
    (part / "big.bin.part").write_bytes(data[:1_000_000])
    assert models.missing_bytes(m) == len(data) - 1_000_000
    models.download(m)
    assert models.is_installed(m) and (models.model_dir(m) / "big.bin").read_bytes() == data
    assert _RangeHandler.requests == ["/big.bin bytes=1000000-"]


def test_a_file_that_does_not_match_is_never_used(server):
    m = _model(server, {"model.bin": b"tampered"}, sha_override={"model.bin": "0" * 64})
    with pytest.raises(models.DownloadError, match="did not download correctly"):
        models.download(m)
    assert not models.is_installed(m)
    assert not (models.models_dir() / "test-model.partial" / "model.bin.part").exists()  # starts clean next time


def test_cancel_and_network_errors(server):
    m = _model(server, {"x.bin": b"x" * 5_000_000})
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(models.Cancelled):
        models.download(m, cancel=cancel)
    assert not models.is_installed(m)
    gone = models.SpeechModel("gone", "Gone", "", "MIT", (models.ModelFile("g.bin", f"{server}/missing", 10, "0" * 64),))
    with pytest.raises(models.DownloadError, match="Try again"):
        models.download(gone)


def test_not_enough_disk_space_is_said_up_front(server, monkeypatch):
    m = _model(server, {"x.bin": b"x" * 1000})
    monkeypatch.setattr(models.shutil, "disk_usage", lambda _p: type("U", (), {"free": 10})())
    with pytest.raises(models.DownloadError, match="Not enough free disk space"):
        models.download(m)


def test_an_older_catalog_entry_does_not_count_as_installed(server):
    m = _model(server, {"a.bin": b"one"})
    models.download(m)
    newer = models.SpeechModel(m.id, m.title, m.description, m.licence,
                               (models.ModelFile("a.bin", m.files[0].url, 3, hashlib.sha256(b"two").hexdigest()),))
    assert models.is_installed(m) and not models.is_installed(newer)


def _frozen_installer_dir(monkeypatch, tmp_path):
    """Pretend setup installed models in tmp_path/app/models/<id>."""
    monkeypatch.setattr(models, "installer_dir", lambda model: tmp_path / "app" / "models" / model.id)


def test_the_installers_copy_counts_with_its_marker_and_without_one_for_older_installs(server, monkeypatch, tmp_path):
    m = _model(server, {"a.bin": b"one", "sub/b.json": b"{}"})
    _frozen_installer_dir(monkeypatch, tmp_path)
    d = models.installer_dir(m)
    (d / "sub").mkdir(parents=True)
    (d / "a.bin").write_bytes(b"one")
    (d / "sub" / "b.json").write_bytes(b"{}")
    assert models.location(m) == d  # an install from before the marker existed: sizes are all there is
    (d / ".complete").write_text("\n".join(f.sha256 for f in m.files), encoding="utf-8")
    assert models.location(m) == d
    (d / ".complete").write_text("0" * 64, encoding="utf-8")  # made for another version of the model
    assert models.location(m) is None


def test_files_setup_installed_are_reused_instead_of_downloaded_again(server, monkeypatch, tmp_path):
    m = _model(server, {"a.bin": b"a" * 100_000, "b.bin": b"b" * 50_000})
    _frozen_installer_dir(monkeypatch, tmp_path)
    d = models.installer_dir(m)
    d.mkdir(parents=True)
    (d / "a.bin").write_bytes(b"a" * 100_000)  # setup got this far
    assert models.location(m) is None and models.missing_bytes(m) == 50_000
    _RangeHandler.requests.clear()
    path = models.download(m)
    assert _RangeHandler.requests == ["/b.bin"] and (path / "a.bin").read_bytes() == b"a" * 100_000
    assert models.location(m) == path


def test_a_damaged_installer_copy_is_not_reused(server, monkeypatch, tmp_path):
    m = _model(server, {"a.bin": b"a" * 1000, "b.bin": b"b"})
    _frozen_installer_dir(monkeypatch, tmp_path)
    d = models.installer_dir(m)
    d.mkdir(parents=True)
    (d / "a.bin").write_bytes(b"x" * 1000)  # right size, wrong content
    _RangeHandler.requests.clear()
    models.download(m)
    assert sorted(_RangeHandler.requests) == ["/a.bin", "/b.bin"] and (models.model_dir(m) / "a.bin").read_bytes() == b"a" * 1000


def test_a_redirect_to_plain_http_is_refused():
    handler = models._HttpsRedirects()
    request = models.urllib.request.Request("https://example.org/model.bin")
    with pytest.raises(models.DownloadError, match="not secure"):
        handler.redirect_request(request, None, 302, "Found", {}, "http://evil.example/model.bin")
    ok = handler.redirect_request(request, None, 302, "Found", {}, "https://cdn.example.org/model.bin")
    assert ok is not None and ok.full_url == "https://cdn.example.org/model.bin"


# --- models that come together in one zip archive ---------------------------------------------------------------

def _zip_bytes(members: dict[str, bytes]) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


def _archive_models(base: str, members: dict[str, bytes] | None = None, *, tamper: str = ""):
    """Two small models and a LICENSES file that share one archive served by the local test server."""
    rnd = random.Random(1)
    members = members or {
        "README.txt": b"readme",
        "LICENSES/A-MIT.txt": b"licence text",
        "one/a.onnx": rnd.randbytes(1_000_000),  # incompressible, so the archive is as big as its members
        "one/b.onnx": rnd.randbytes(300_000),
        "two/c.onnx": rnd.randbytes(200_000),
    }
    blob = _zip_bytes(members)
    _RangeHandler.files["/models.zip"] = blob

    def entry(path: str, member: str) -> models.ModelFile:
        sha = hashlib.sha256(members[member]).hexdigest()
        return models.ModelFile(path, "", len(members[member]), "0" * 64 if tamper == member else sha, member=member)

    archive = models.ModelArchive("models.zip", f"{base}/models.zip", len(blob), hashlib.sha256(blob).hexdigest(),
                                  shared=(entry("LICENSES/A-MIT.txt", "LICENSES/A-MIT.txt"),))
    one = models.SpeechModel("arch-one", "One", "", "MIT", (entry("a.onnx", "one/a.onnx"), entry("b.onnx", "one/b.onnx")), archive)
    two = models.SpeechModel("arch-two", "Two", "", "MIT", (entry("c.onnx", "two/c.onnx"),), archive)
    return archive, one, two, members


@pytest.fixture
def archive_catalog(monkeypatch):
    monkeypatch.setattr(models, "CATALOG", {})


def test_one_archive_installs_every_model_that_shares_it(server, archive_catalog):
    archive, one, two, members = _archive_models(server)
    models.register(one, two)
    seen = []
    assert models.missing_bytes(one) == archive.size and models.download_bytes([one, two]) == archive.size
    assert one.download_size == archive.size and one.size == len(members["one/a.onnx"]) + len(members["one/b.onnx"])
    path = models.download(one, progress=lambda d, t: seen.append((d, t)))
    assert path == models.model_dir(one) and models.is_installed(one) and models.is_installed(two)  # both from one download
    assert (path / "b.onnx").read_bytes() == members["one/b.onnx"]
    assert (models.model_dir(two) / "c.onnx").read_bytes() == members["two/c.onnx"]
    assert (models.models_dir() / "LICENSES" / "A-MIT.txt").read_bytes() == b"licence text"
    assert models.licences_dir() == models.models_dir() / "LICENSES"
    assert all(t == archive.size for _, t in seen) and seen[-1] == (archive.size, archive.size)
    assert _RangeHandler.requests == ["/models.zip"]
    leftovers = sorted(p.name for p in models.models_dir().iterdir())
    assert leftovers == ["LICENSES", "arch-one", "arch-two"]  # archive and every .partial folder are gone
    assert models.download(two) == models.model_dir(two) and _RangeHandler.requests == ["/models.zip"]


def test_an_interrupted_archive_download_resumes_and_is_verified(server, archive_catalog):
    archive, one, two, members = _archive_models(server)
    models.register(one, two)
    blob = _RangeHandler.files["/models.zip"]
    part = models.models_dir() / "models.partial"
    part.mkdir(parents=True)
    (part / ".pin").write_text(archive.sha256)
    (part / "models.zip.part").write_bytes(blob[:100_000])
    assert models.missing_bytes(two) == archive.size - 100_000
    models.download(two)
    assert _RangeHandler.requests == ["/models.zip bytes=100000-"]
    assert models.is_installed(one) and models.is_installed(two)


def test_a_damaged_archive_is_rejected(server, archive_catalog):
    archive, one, two, _ = _archive_models(server)
    bad = models.ModelArchive(archive.name, archive.url, archive.size, "0" * 64, archive.shared)
    one = models.SpeechModel(one.id, one.title, "", "MIT", one.files, bad)
    with pytest.raises(models.DownloadError, match="did not download correctly"):
        models.download(one)
    assert not models.is_installed(one) and not (models.models_dir() / "models.partial" / "models.zip.part").exists()


def test_a_tampered_member_is_never_installed(server, archive_catalog):
    archive, one, two, _ = _archive_models(server, tamper="one/b.onnx")  # the catalog's hash differs from the file's
    models.register(one, two)
    with pytest.raises(models.DownloadError, match="not what it should be"):
        models.download(two)
    assert not models.is_installed(one) and not models.is_installed(two)
    root = models.models_dir()
    assert not (root / "arch-one.partial").exists() and not (root / "arch-two.partial").exists()
    assert not (root / "models.partial" / "models.zip").exists()  # fetched again next time


def test_cancelling_while_unpacking_keeps_the_archive_and_nothing_half_done(server, archive_catalog, monkeypatch):
    archive, one, two, members = _archive_models(server)
    models.register(one, two)
    cancel = threading.Event()
    real = models._unpack_member
    calls = []

    def unpack_then_cancel(zf, f, dest, c):
        calls.append(f.path)
        if len(calls) == 2:
            cancel.set()
        return real(zf, f, dest, c)

    monkeypatch.setattr(models, "_unpack_member", unpack_then_cancel)
    with pytest.raises(models.Cancelled):
        models.download(one, cancel=cancel)
    root = models.models_dir()
    assert not models.is_installed(one) and not models.is_installed(two)
    assert not (root / "arch-one.partial").exists() and not (root / "arch-two.partial").exists()
    assert (root / "models.partial" / "models.zip").is_file() and models.missing_bytes(one) == 0  # nothing to download again
    _RangeHandler.requests.clear()
    monkeypatch.setattr(models, "_unpack_member", real)
    models.download(one)
    assert _RangeHandler.requests == [] and models.is_installed(one) and models.is_installed(two)
    assert not (root / "models.partial").exists()


def test_the_disk_check_counts_the_archive_and_what_is_unpacked(server, archive_catalog, monkeypatch):
    archive, one, two, members = _archive_models(server)
    models.register(one, two)
    asked = []
    monkeypatch.setattr(models, "_check_disk", lambda root, need: asked.append(need))
    models.download(two)
    assert asked == [archive.size + sum(len(members[k]) for k in ("one/a.onnx", "one/b.onnx", "two/c.onnx"))
                     + len(members["LICENSES/A-MIT.txt"])]
    monkeypatch.undo()
    models.remove(two)
    models.remove(one)
    monkeypatch.setattr(models.shutil, "disk_usage", lambda _p: type("U", (), {"free": 10})())
    with pytest.raises(models.DownloadError, match="Not enough free disk space"):
        models.download(two)


def test_files_setup_installed_seed_an_archive_model_and_skip_the_download(server, archive_catalog, monkeypatch, tmp_path):
    archive, one, two, members = _archive_models(server)
    models.register(one, two)
    _frozen_installer_dir(monkeypatch, tmp_path)
    for m in (one, two):  # setup unpacked everything it got, but its marker is from another version of the catalog
        d = models.installer_dir(m)
        d.mkdir(parents=True)
        for f in m.files:
            (d / f.path).write_bytes(members[f.member])
        (d / ".complete").write_text("0" * 64, encoding="utf-8")
    assert models.location(one) is None and models.missing_bytes(one) == 0
    path = models.download(one)
    assert _RangeHandler.requests == [] and (models.model_dir(two) / "c.onnx").read_bytes() == members["two/c.onnx"]
    assert (path / "a.onnx").read_bytes() == members["one/a.onnx"] and models.is_installed(two)


def test_a_damaged_installer_copy_makes_an_archive_model_download(server, archive_catalog, monkeypatch, tmp_path):
    archive, one, two, members = _archive_models(server)
    models.register(one, two)
    _frozen_installer_dir(monkeypatch, tmp_path)
    d = models.installer_dir(one)
    d.mkdir(parents=True)
    (d / "a.onnx").write_bytes(b"x" * len(members["one/a.onnx"]))  # right size, wrong content
    (d / "b.onnx").write_bytes(members["one/b.onnx"])
    (d / ".complete").write_text("0" * 64, encoding="utf-8")
    models.download(two)
    assert _RangeHandler.requests == ["/models.zip"]
    assert (models.model_dir(one) / "a.onnx").read_bytes() == members["one/a.onnx"] and models.is_installed(two)


def test_two_downloads_of_one_archive_fetch_it_once(server, archive_catalog):
    archive, one, two, _ = _archive_models(server)
    models.register(one, two)
    results = []
    threads = [threading.Thread(target=lambda m=m: results.append(models.download(m))) for m in (one, two)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert len(results) == 2 and _RangeHandler.requests == ["/models.zip"]
