"""Downloading speech models: pinned files, resume, verification, nothing half-done is ever used."""
from __future__ import annotations

import hashlib
import http.server
import threading

import pytest

from audio_cleaner_desktop import models, resources


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    files: dict[str, bytes] = {}
    requests: list[str] = []
    cut: dict[str, int] = {}  # path -> the connection ends after this many bytes of the body (its length was announced in full)

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
        self.wfile.write(body[start:start + self.cut[self.path]] if self.path in self.cut else body[start:])

    def log_message(self, *_args):
        pass


@pytest.fixture
def server(monkeypatch, tmp_path):
    _RangeHandler.files, _RangeHandler.requests, _RangeHandler.cut = {}, [], {}
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


def test_a_connection_that_ends_early_without_an_error_keeps_the_part_and_the_next_try_continues_from_it(server):
    data = bytes(range(256)) * 20_000
    m = _model(server, {"big.bin": data})
    _RangeHandler.cut["/big.bin"] = 1_500_000  # the server goes away half way (http.client reads that as the end of the body)
    with pytest.raises(models.DownloadError, match="stopped before the end"):
        models.download(m)
    part = models.models_dir() / "test-model.partial" / "big.bin.part"
    assert part.read_bytes() == data[:1_500_000] and not models.is_installed(m)  # what arrived is kept
    _RangeHandler.cut.clear()
    models.download(m)  # again: from where it was
    assert _RangeHandler.requests == ["/big.bin", "/big.bin bytes=1500000-"]
    assert models.is_installed(m) and (models.model_dir(m) / "big.bin").read_bytes() == data and not part.exists()


class _Opener:
    """What `_fetch` opens its request with: answers with `body` (a response whose reads fail with `error` when the body is used up) or raises."""

    def __init__(self, body=b"", error=None, raises=None):
        self.body, self.error, self.raises, self.requests = body, error, raises, []

    def open(self, request, timeout=None):
        self.requests.append(request.get_header("Range"))
        if self.raises is not None:
            raise self.raises
        opener = self

        class Response:
            status = 206 if request.get_header("Range") else 200

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self, n):
                chunk, opener.body = opener.body[:n], opener.body[n:]
                if not chunk and opener.error is not None:
                    raise opener.error
                return chunk

        return Response()


def test_a_body_cut_short_or_a_range_the_server_cannot_serve_is_a_download_error_and_only_the_second_loses_the_part(tmp_path):
    import http.client
    import urllib.error

    data = b"0123456789" * 100
    f = models.ModelFile("a.bin", "https://example.invalid/a.bin", len(data), hashlib.sha256(data).hexdigest())
    dest = tmp_path / "a.bin"
    part = tmp_path / "a.bin.part"
    # a body that stops with an IncompleteRead: the error is ours to say, what arrived stays
    with pytest.raises(models.DownloadError, match="The download stopped"):
        models._fetch(f, dest, 0, f.size, None, None, _Opener(data[:400], error=http.client.IncompleteRead(b"", 600)))
    assert part.read_bytes() == data[:400] and not dest.exists()
    # the next try asks for the rest with a Range header and finishes the file
    again = _Opener(data[400:])
    models._fetch(f, dest, 0, f.size, None, None, again)
    assert again.requests == ["bytes=400-"] and dest.read_bytes() == data and not part.exists()
    # 416: the server has no more of the file than we hold, so this part can't be finished and goes
    dest.unlink()
    part.write_bytes(data[:400])
    gone = urllib.error.HTTPError(f.url, 416, "Range Not Satisfiable", {}, None)
    with pytest.raises(models.DownloadError, match="The download stopped"):
        models._fetch(f, dest, 0, f.size, None, None, _Opener(raises=gone))
    assert not part.exists() and not dest.exists()
    # any other HTTP error keeps the part
    part.write_bytes(data[:400])
    with pytest.raises(models.DownloadError, match="The download stopped"):
        models._fetch(f, dest, 0, f.size, None, None, _Opener(raises=urllib.error.HTTPError(f.url, 503, "Unavailable", {}, None)))
    assert part.read_bytes() == data[:400]


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


# --- licence texts that come with a model (our voice model) -----------------------------------------------

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


def test_the_noise_models_are_found_where_they_were_downloaded_and_check_no_longer_lists_them(monkeypatch, tmp_path):
    """model_path()/heavy_noise_model_path() point into the downloaded noise-removal model (the app's or the installer's copy);
    from source they also find vendor/models; with neither they name where the download will put them."""
    import dataclasses

    from acw_common import dfn, dpdfnet

    names = (dfn.MODEL_FILE, dpdfnet.MODEL_FILE)
    data = {n: n.encode() for n in names}
    fake = dataclasses.replace(resources.NOISE_MODEL, files=tuple(
        models.ModelFile(n, "", len(b), hashlib.sha256(b).hexdigest()) for n, b in data.items()))
    monkeypatch.setattr(resources, "NOISE_MODEL", fake)
    monkeypatch.setenv("ACW_DESKTOP_MODELS", str(tmp_path / "models"))
    monkeypatch.setenv("ACW_DESKTOP_VENDOR", str(tmp_path / "vendor"))
    monkeypatch.setattr(resources, "_dev_fallback", lambda: None)
    target = models.model_dir(fake)
    assert not resources.noise_ready() and resources.model_path() == target / dfn.MODEL_FILE  # names the place to download to
    assert not any(n in resources.check() for n in names)  # not a shipped component any more
    (tmp_path / "vendor" / "models").mkdir(parents=True)
    for n, b in data.items():
        (tmp_path / "vendor" / "models" / n).write_bytes(b)
    assert resources.noise_ready() and resources.model_path().parent == tmp_path / "vendor" / "models"  # run from source
    target.mkdir(parents=True)
    for n, b in data.items():
        (target / n).write_bytes(b)
    (target / ".complete").write_text("\n".join(f.sha256 for f in fake.files))
    assert resources.model_path() == target / dfn.MODEL_FILE and resources.heavy_noise_model_path() == target / dpdfnet.MODEL_FILE
