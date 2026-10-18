"""Downloading speech models: pinned files, resume, verification, nothing half-done is ever used."""
from __future__ import annotations

import hashlib
import http.server
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


# --- licence texts that come with a model (our highs and voice models) -----------------------------------------------

def _model_with_licences(base, payloads, licences):
    m = _model(base, payloads)
    for name, data in licences.items():
        _RangeHandler.files[f"/{name}"] = data
    lic = tuple(models.ModelFile(f"LICENSES/{name}", f"{base}/{name}", len(d), hashlib.sha256(d).hexdigest())
                for name, d in licences.items())
    return models.SpeechModel(m.id, m.title, m.description, m.licence, m.files, lic)


def test_a_model_downloads_its_own_files_and_licences_only(server):
    m = _model_with_licences(server, {"a.onnx": b"a" * 2_000_000}, {"MIT.txt": b"licence text"})
    models.register(m)
    other = _model(server, {"other.onnx": b"o" * 1000})  # a model nobody asked for: nothing of it is fetched
    path = models.download(m)
    assert (path / "a.onnx").is_file() and (path / "LICENSES" / "MIT.txt").read_bytes() == b"licence text"
    assert sorted(r.split()[0] for r in _RangeHandler.requests) == ["/MIT.txt", "/a.onnx"] and not models.is_installed(other)
    assert models.licence_dirs() == [path / "LICENSES"]  # what the About box shows
    # the licences are not what makes the model complete: the marker names the model's files only
    assert (path / ".complete").read_text().split() == [f.sha256 for f in m.files]
    models.remove(m)
    assert not path.exists() and models.licence_dirs() == []


def test_a_copy_from_the_old_models_archive_still_counts_and_is_not_downloaded_again(server, monkeypatch, tmp_path):
    m = _model_with_licences(server, {"a.onnx": b"a" * 1000}, {"MIT.txt": b"licence text"})
    models.register(m)
    old = models.model_dir(m)  # the archive's unpacking: the model's files and its marker, the licences in <models>/LICENSES
    old.mkdir(parents=True)
    (old / "a.onnx").write_bytes(b"a" * 1000)
    (old / ".complete").write_text(m.files[0].sha256, encoding="utf-8")
    (models.models_dir() / "LICENSES").mkdir()
    (models.models_dir() / "LICENSES" / "MIT.txt").write_text("licence text")
    assert models.is_installed(m) and models.missing_bytes(m) == 0 and not models.needs_update(m)
    assert models.download(m) == old and _RangeHandler.requests == []
    assert models.licence_dirs() == [models.models_dir() / "LICENSES"]  # About still finds them
