"""packaging/release_models.py: the models every release carries (the speech model's pinned files, the models archive
carried over from the newest release that has it), gathered with every file checked, and the --check of the
model-urls job. No real network: a fake GitHub API and a fake storage host on local ports, over TLS with a throwaway
certificate (the asset download refuses anything but https), and fakes for HEAD."""
from __future__ import annotations

import datetime
import hashlib
import http.server
import ipaddress
import json
import os
import shutil
import ssl
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from audio_cleaner_desktop import config, models, resources

PACKAGING = Path(__file__).resolve().parents[1] / "packaging"
if str(PACKAGING) not in sys.path:
    sys.path.insert(0, str(PACKAGING))

import release_models  # noqa: E402

PUBLIC, OWN = "pub-owner/App", "own-owner/Build"
ENV = ("PUBLIC_REPO", "GITHUB_REPOSITORY", "PUBLIC_TOKEN", "OWN_TOKEN")


@pytest.fixture(autouse=True)
def env(monkeypatch):
    """The two repositories and their tokens, as CI sets them (nothing from the real environment leaks in)."""
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PUBLIC_REPO", PUBLIC)
    monkeypatch.setenv("GITHUB_REPOSITORY", OWN)
    monkeypatch.setenv("PUBLIC_TOKEN", "tok-public")
    monkeypatch.setenv("OWN_TOKEN", "tok-own")


# ---- a fake GitHub API (find_archive): pages of releases, no network ----------------------------------------------------
def _asset(aid: int, name: str | None = None, size: int | None = None, state: str = "uploaded") -> dict:
    a = resources.MODELS_ARCHIVE
    return {"id": aid, "name": name or a.name, "size": a.size if size is None else size, "state": state}


def _release(tag: str, *assets: dict) -> dict:
    return {"tag_name": tag, "draft": False, "assets": list(assets)}


def _filler(n: int, start: int = 0) -> list[dict]:
    return [_release(f"v9.{start + i}", _asset(9000 + start + i, name="AudioCleaner-9-Windows.exe", size=5)) for i in range(n)]


class FakeApi:
    """Stands in for urllib.request.urlopen in release_models._api: {repo: [page 1, page 2, ...]} or {repo: HTTP code}."""

    def __init__(self, repos: dict):
        self.repos = repos
        self.calls: list[tuple[str, int, str | None]] = []  # (repo, page, Authorization)

    def __call__(self, request, timeout=None):
        url = request.full_url
        assert url.startswith(release_models.API + "/repos/") and request.get_method() == "GET"
        path, _, query = url[len(release_models.API) + len("/repos/"):].partition("?")
        repo = path.removesuffix("/releases")
        params = dict(p.split("=") for p in query.split("&"))
        assert params["per_page"] == "100"
        page = int(params["page"])
        self.calls.append((repo, page, request.get_header("Authorization")))
        pages = self.repos.get(repo, 404)
        if isinstance(pages, int):
            raise urllib.error.HTTPError(url, pages, "nope", {}, None)
        import io

        return io.BytesIO(json.dumps(pages[page - 1] if page <= len(pages) else []).encode())

    def pages(self, repo: str) -> list[int]:
        return [p for r, p, _ in self.calls if r == repo]


@pytest.fixture
def api(monkeypatch):
    def install(repos: dict) -> FakeApi:
        fake = FakeApi(repos)
        monkeypatch.setattr(release_models.urllib.request, "urlopen", fake)
        return fake

    return install


def test_the_archive_is_taken_from_the_public_repository_first(api):
    fake = api({PUBLIC: [[_release("v2.0.41", _asset(1))]], OWN: [[_release("models-v2", _asset(2))]]})
    assert release_models.find_archive() == (PUBLIC, _asset(1))
    assert fake.pages(PUBLIC) == [1] and fake.pages(OWN) == []  # this repository is not even asked


def test_the_newest_release_with_the_pinned_file_wins(api):
    a = resources.MODELS_ARCHIVE
    page = [
        _release("v2.0.45"),  # newest first, as GitHub lists them: no archive
        _release("v2.0.44", _asset(11, size=a.size - 1)),  # the same name, another size: not the pinned file
        _release("v2.0.43", _asset(12, state="starter")),  # an upload that never finished
        _release("v2.0.42", _asset(13, name="AudioCleaner-models-v1.zip")),  # an older archive
        _release("v2.0.41", _asset(14, name="whisper-model.bin", size=7), _asset(15)),
        _release("v2.0.40", _asset(16)),  # older, also has it
    ]
    api({PUBLIC: [page]})
    assert release_models.find_archive() == (PUBLIC, _asset(15))
    no_state = _asset(17)
    del no_state["state"]  # an asset listed without a state counts as uploaded
    api({PUBLIC: [[_release("v2.0.46", no_state), *page]]})
    assert release_models.find_archive()[1]["id"] == 17


def test_up_to_five_pages_of_the_public_repository_then_this_one(api):
    pages = [_filler(100, 100 * i) for i in range(5)] + [[_release("v1.0.0", _asset(1))]]  # page 6 is never read
    fake = api({PUBLIC: pages, OWN: [[*_filler(29), _release("models-v2", _asset(2))]]})
    assert release_models.find_archive() == (OWN, _asset(2))
    assert fake.pages(PUBLIC) == [1, 2, 3, 4, 5]
    assert fake.pages(OWN) == [1]  # a short page is the last one


def test_the_archive_on_a_later_page_is_found(api):
    fake = api({PUBLIC: [_filler(100), _filler(100, 100), [_release("v2.0.1", _asset(5))]]})
    assert release_models.find_archive() == (PUBLIC, _asset(5))
    assert fake.pages(PUBLIC) == [1, 2, 3] and fake.pages(OWN) == []


def test_a_short_page_ends_the_listing(api):
    fake = api({PUBLIC: [_filler(100), _filler(3, 100), [_release("v2.0.1", _asset(5))]], OWN: [[]]})
    with pytest.raises(SystemExit):
        release_models.find_archive()
    assert fake.pages(PUBLIC) == [1, 2] and fake.pages(OWN) == [1]


def test_a_repository_that_is_not_there_is_skipped_but_other_errors_stop(api):
    fake = api({PUBLIC: 404, OWN: [[_release("models-v2", _asset(2))]]})
    assert release_models.find_archive() == (OWN, _asset(2))
    assert fake.pages(PUBLIC) == [1]
    for code in (401, 403, 500):  # a wrong token or a rate limit must not quietly look elsewhere
        api({PUBLIC: code, OWN: [[_release("models-v2", _asset(2))]]})
        with pytest.raises(urllib.error.HTTPError):
            release_models.find_archive()


def test_an_error_on_a_later_page_stops(api, monkeypatch):
    fake = FakeApi({PUBLIC: [_filler(100)]})

    def flaky(request, timeout=None):
        if "page=2" in request.full_url:
            raise urllib.error.HTTPError(request.full_url, 404, "gone", {}, None)
        return fake(request, timeout)

    monkeypatch.setattr(release_models.urllib.request, "urlopen", flaky)
    with pytest.raises(urllib.error.HTTPError):
        release_models.find_archive()


def test_nowhere_says_what_to_do(api):
    api({PUBLIC: [[]], OWN: [_filler(3)]})
    with pytest.raises(SystemExit) as e:
        release_models.find_archive()
    a = resources.MODELS_ARCHIVE
    text = str(e.value)
    assert a.name in text and str(a.size) in text and PUBLIC in text and OWN in text and "upload it once" in text


def test_each_token_goes_only_to_its_own_repository(api):
    fake = api({PUBLIC: [_filler(100), _filler(1, 100)], OWN: [[_release("models-v2", _asset(2))]]})
    release_models.find_archive()
    assert {(r, auth) for r, _, auth in fake.calls} == {(PUBLIC, "Bearer tok-public"), (OWN, "Bearer tok-own")}


def test_without_tokens_or_without_a_public_repository(api, monkeypatch):
    monkeypatch.delenv("PUBLIC_TOKEN")
    monkeypatch.setenv("OWN_TOKEN", "")
    fake = api({PUBLIC: [[]], OWN: [[_release("models-v2", _asset(2))]]})
    release_models.find_archive()
    assert [auth for *_, auth in fake.calls] == [None, None]  # anonymous (a public repository needs no token)
    monkeypatch.delenv("PUBLIC_REPO")
    monkeypatch.setenv("OWN_TOKEN", "tok-own")
    fake = api({OWN: [[_release("models-v2", _asset(2))]]})
    assert release_models.find_archive() == (OWN, _asset(2))
    assert fake.calls == [(OWN, 1, "Bearer tok-own")]
    assert release_models._token("someone/else") is None  # no token for a repository it is not for


def test_the_same_repository_named_twice_gets_the_public_token(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", PUBLIC)
    assert release_models._token(PUBLIC) == "tok-public"


# ---- the redirect of the asset download -----------------------------------------------------------------------------------
def _request(url: str = "https://api.github.com/repos/o/r/releases/assets/1") -> urllib.request.Request:
    return urllib.request.Request(url, headers={"Authorization": "Bearer secret", "Accept": "application/octet-stream"})


def test_the_token_is_not_sent_to_another_host():
    new = release_models._AssetRedirects().redirect_request(
        _request(), None, 302, "Found", {}, "https://objects.githubusercontent.com/x?sig=1")
    assert new.full_url == "https://objects.githubusercontent.com/x?sig=1"
    assert not new.has_header("Authorization") and "secret" not in str(new.header_items())
    assert new.get_header("Accept") == "application/octet-stream"


def test_the_token_stays_on_the_same_host():
    new = release_models._AssetRedirects().redirect_request(
        _request(), None, 301, "Moved", {}, "https://api.github.com/repositories/5/releases/assets/1")
    assert new.get_header("Authorization") == "Bearer secret"
    other_port = release_models._AssetRedirects().redirect_request(
        _request(), None, 302, "Found", {}, "https://api.github.com:8443/x")
    assert not other_port.has_header("Authorization")  # another port is another server


@pytest.mark.parametrize("target", ["http://objects.githubusercontent.com/x", "http://api.github.com/x", "ftp://x/y", "file:///etc/passwd"])
def test_a_redirect_away_from_https_is_refused(target):
    with pytest.raises(urllib.error.HTTPError, match="redirect away from https"):
        release_models._AssetRedirects().redirect_request(_request(), None, 302, "Found", {}, target)


# ---- gather: the real downloads, against fake hosts over TLS ------------------------------------------------------------
def _certificate(folder: Path) -> tuple[Path, Path]:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
        .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), True)
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = folder / "cert.pem", folder / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    return cert_path, key_path


class _Host(http.server.BaseHTTPRequestHandler):
    """routes: path (with query) -> (status, headers, body); log: (host label, path, Authorization, Accept)."""

    label = ""
    routes: dict[str, tuple[int, dict, bytes]] = {}
    log: list[tuple[str, str, str | None, str | None]] = []

    def do_GET(self):  # noqa: N802
        self.log.append((self.label, self.path, self.headers.get("Authorization"), self.headers.get("Accept")))
        status, headers, body = self.routes.get(self.path, (404, {}, b"not found"))
        rng = self.headers.get("Range")
        if status == 200 and rng and rng.startswith("bytes="):
            start = int(rng[6:].split("-")[0])
            status, headers = 206, {**headers, "Content-Range": f"bytes {start}-{len(body) - 1}/{len(body)}"}
            body = body[start:]
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@pytest.fixture
def hosts(tmp_path, monkeypatch):
    """Two https hosts: "api" on 127.0.0.1 (GitHub's API and Hugging Face) and "storage" on localhost (where GitHub
    redirects asset downloads). The client trusts only the throwaway certificate."""
    cert, key = _certificate(tmp_path)
    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.load_cert_chain(cert, key)
    routes: dict[str, tuple[int, dict, bytes]] = {}
    log: list = []
    servers, urls = [], {}
    for label, host in (("api", "127.0.0.1"), ("storage", "localhost")):
        handler = type(f"Host_{label}", (_Host,), {"label": label, "routes": routes, "log": log})
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        httpd.socket = server_ctx.wrap_socket(httpd.socket, server_side=True)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        urls[label] = f"https://{host}:{httpd.server_address[1]}"
    client_ctx = ssl.create_default_context(cafile=str(cert))
    monkeypatch.setattr(models, "_ssl_context", lambda: client_ctx)
    monkeypatch.setattr(release_models, "API", urls["api"])
    monkeypatch.setattr(release_models, "WHISPER_SOURCE", urls["api"] + "/hf/resolve/abc/")
    # urlopen in _api has no context argument: give it one that trusts the certificate, as the system store would
    real_urlopen = urllib.request.urlopen
    monkeypatch.setattr(release_models.urllib.request, "urlopen", lambda req, timeout=None: real_urlopen(req, timeout=timeout, context=client_ctx))
    yield {"routes": routes, "log": log, **urls}
    for httpd in servers:
        httpd.shutdown()
        httpd.server_close()


SPEECH = {"model.bin": b"weights " * 5000, "config.json": b'{"x": 1}'}
ARCHIVE = b"PK zip of the models " * 3000


@pytest.fixture
def catalog(monkeypatch):
    """A small speech model and models archive in place of the real (2.5 GB) ones."""
    files = tuple(models.ModelFile(n, config.MODELS_URL + "whisper-" + n, len(d), hashlib.sha256(d).hexdigest()) for n, d in SPEECH.items())
    monkeypatch.setattr(resources, "SPEECH_MODEL", models.SpeechModel("whisper-test", "t", "t", "MIT", files))
    archive = models.ModelArchive("AudioCleaner-models-v2.zip", config.MODELS_URL + "AudioCleaner-models-v2.zip",
                                  len(ARCHIVE), hashlib.sha256(ARCHIVE).hexdigest())
    monkeypatch.setattr(resources, "MODELS_ARCHIVE", archive)
    return archive


def _publish(hosts, *, archive: bytes = ARCHIVE, speech: dict | None = None, storage: str | None = None) -> None:
    r = hosts["routes"]
    for name, data in (speech or SPEECH).items():
        r[f"/hf/resolve/abc/{name}"] = (200, {}, data)
    listing = [_release("v2.0.41"), _release("v2.0.40", _asset(77, size=len(ARCHIVE)))]
    r[f"/repos/{PUBLIC}/releases?per_page=100&page=1"] = (200, {"Content-Type": "application/json"}, json.dumps(listing).encode())
    r[f"/repos/{PUBLIC}/releases/assets/77"] = (302, {"Location": (storage or hosts["storage"]) + "/blob/77?sig=1"}, b"")
    r["/blob/77?sig=1"] = (200, {}, archive)


def test_gather_downloads_and_checks_every_file(hosts, catalog, tmp_path):
    _publish(hosts)
    out = tmp_path / "release" / "models"
    names = release_models.gather(out)
    assert names == ["whisper-model.bin", "whisper-config.json", "AudioCleaner-models-v2.zip"]
    assert sorted(p.name for p in out.iterdir()) == sorted(names)  # nothing else (no .part left)
    for n, d in SPEECH.items():
        assert (out / f"whisper-{n}").read_bytes() == d
    assert (out / "AudioCleaner-models-v2.zip").read_bytes() == ARCHIVE
    log = hosts["log"]
    asset = [e for e in log if e[1].endswith("/releases/assets/77")]
    assert asset == [("api", f"/repos/{PUBLIC}/releases/assets/77", "Bearer tok-public", "application/octet-stream")]
    blob = [e for e in log if e[1].startswith("/blob/")]
    assert len(blob) == 1 and blob[0][0] == "storage" and blob[0][2] is None  # the token never reaches the storage host
    assert all(e[2] is None for e in log if e[1].startswith("/hf/"))  # nor Hugging Face
    assert [e[2] for e in log if "/releases?" in e[1]] == ["Bearer tok-public"]


def test_an_archive_that_is_not_the_pinned_one_is_refused(hosts, catalog, tmp_path):
    _publish(hosts, archive=b"X" * len(ARCHIVE))  # the right size, other bytes
    out = tmp_path / "models"
    with pytest.raises(SystemExit, match="is not the pinned file"):
        release_models.gather(out)
    assert not (out / "AudioCleaner-models-v2.zip").exists() and not list(out.glob("*.part"))


def test_an_archive_of_another_size_is_refused(hosts, catalog, tmp_path):
    _publish(hosts, archive=ARCHIVE + b"!")  # the listing says the pinned size, the download is longer
    out = tmp_path / "models"
    with pytest.raises(SystemExit, match="is not the pinned file"):
        release_models.gather(out)
    assert not (out / "AudioCleaner-models-v2.zip").exists() and not list(out.glob("*.part"))


def test_a_speech_model_file_that_is_not_the_pinned_one_is_refused(hosts, catalog, tmp_path):
    _publish(hosts, speech={**SPEECH, "model.bin": b"W" * len(SPEECH["model.bin"])})
    out = tmp_path / "models"
    with pytest.raises(models.DownloadError):
        release_models.gather(out)
    assert not (out / "whisper-model.bin").exists() and not list(out.glob("*.part"))
    assert not [e for e in hosts["log"] if "/releases" in e[1]]  # it stopped before the archive


def test_a_redirect_to_plain_http_is_refused(hosts, catalog, tmp_path):
    _publish(hosts, storage=hosts["storage"].replace("https://", "http://"))
    out = tmp_path / "models"
    with pytest.raises(urllib.error.HTTPError, match="redirect away from https"):
        release_models.gather(out)
    assert not (out / "AudioCleaner-models-v2.zip").exists() and not list(out.glob("*.part"))
    assert not [e for e in hosts["log"] if e[1].startswith("/blob/")]


def test_the_main_command_prints_the_files(hosts, catalog, tmp_path, capsys):
    _publish(hosts)
    assert release_models.main([str(tmp_path / "m")]) == 0
    assert capsys.readouterr().out.split() == ["whisper-model.bin", "whisper-config.json", "AudioCleaner-models-v2.zip"]


# ---- what the release carries is what the app downloads ------------------------------------------------------------------
def test_the_speech_model_files_are_the_pinned_ones_under_the_names_the_app_asks_for():
    assert release_models.WHISPER_SOURCE.startswith("https://huggingface.co/")
    commit = release_models.WHISPER_SOURCE.rstrip("/").rsplit("/", 1)[1]
    assert len(commit) == 40 and int(commit, 16) >= 0  # a pinned commit, not a branch
    found = release_models.whisper_files()
    assert [(n, size, sha) for n, _, size, sha in found] == [
        ("whisper-" + f.path, f.size, f.sha256) for f in resources.SPEECH_MODEL.files]
    assert [url for _, url, _, _ in found] == [release_models.WHISPER_SOURCE + f.path for f in resources.SPEECH_MODEL.files]
    # each name in the release is the last part of the address the app and the installers download it from
    assert [n for n, *_ in found] == [f.url.rsplit("/", 1)[1] for f in resources.SPEECH_MODEL.files]
    assert all(f.size < 2 * 1024**3 for f in resources.SPEECH_MODEL.files)  # GitHub's limit per release file
    assert resources.MODELS_ARCHIVE.url.rsplit("/", 1)[1] == resources.MODELS_ARCHIVE.name
    assert resources.MODELS_ARCHIVE.size < 2 * 1024**3


# ---- --check: HEAD sizes of the speech model's files, and find_archive ---------------------------------------------------
def _queue_head(answers: dict[str, list[tuple[int, dict]]]):
    queue = {k: list(v) for k, v in answers.items()}

    def head(url):
        return queue[url].pop(0)

    return head


def test_remote_size_follows_redirects_by_hand():
    head = _queue_head({
        "https://a/ok": [(200, {"content-length": "10"})],
        "https://a/hf": [(302, {"x-linked-size": "20", "location": "https://cdn/x"})],
        "https://cdn/x": [(200, {"content-length": "999"})],  # Hugging Face's own size wins over the CDN's answer
        "https://a/gone": [(404, {})],
        "https://a/small": [(302, {"location": "/next"})],
        "https://a/next": [(200, {"content-length": "5"})],
        "https://a/nosize": [(200, {})],
        "https://a/loop": [(302, {"location": "https://a/loop"})] * 10,
    })
    assert release_models.remote_size("https://a/ok", head) == (10, "")
    assert release_models.remote_size("https://a/hf", head) == (20, "")
    assert release_models.remote_size("https://a/gone", head) == (None, "HTTP 404")
    assert release_models.remote_size("https://a/small", head) == (5, "")  # a relative Location
    assert release_models.remote_size("https://a/nosize", head) == (None, "the server did not say how big the file is")
    assert release_models.remote_size("https://a/loop", head) == (None, "too many redirects")


@pytest.fixture
def heads(monkeypatch):
    """Every pinned speech model file answers HEAD with its pinned size (change `answers` to break one)."""
    answers = {url: (200, {"content-length": str(size)}) for _, url, size, _ in release_models.whisper_files()}
    seen = []

    def head(url):
        seen.append(url)
        return answers.get(url, (404, {}))

    real = release_models.remote_size
    monkeypatch.setattr(release_models, "remote_size", lambda url: real(url, head))
    return answers, seen


def test_check_is_quiet_when_everything_can_be_found(heads, api, capsys):
    answers, seen = heads
    api({PUBLIC: [[_release("v2.0.40", _asset(1))]]})
    assert release_models.check() == []
    assert sorted(seen) == sorted(answers)  # every file of the speech model, from its pinned source
    assert release_models.main(["--check"]) == 0
    assert capsys.readouterr().err == ""


def test_check_names_every_problem(heads, api, capsys):
    answers, _ = heads
    first, second = list(answers)[:2]
    answers[first] = (200, {"content-length": "5"})
    answers[second] = (404, {})
    api({PUBLIC: [[]], OWN: [[]]})
    found = release_models.check()
    assert len(found) == 3
    assert found[0].startswith("whisper-") and first in found[0] and "5 bytes there" in found[0] and "pinned" in found[0]
    assert second in found[1] and "HTTP 404" in found[1]
    assert resources.MODELS_ARCHIVE.name in found[2] and "upload it once" in found[2]
    assert release_models.main(["--check"]) == 1
    assert capsys.readouterr().err.count("\n") == 3


@pytest.mark.parametrize("argv", [[], ["a", "b"], ["--check", "x"]])
def test_usage(argv, capsys):
    assert release_models.main(argv) == 2
    assert "release_models.py OUT_DIR" in capsys.readouterr().err


def test_the_release_job_s_checkout_is_enough_and_needs_only_the_standard_library(tmp_path):
    # The release job checks out desktop/audio_cleaner_desktop/*.py (not its subfolders) and the packaging scripts, and
    # runs them with a bare python3: no numpy, no Qt, no cryptography.
    source = PACKAGING.parent
    app = tmp_path / "desktop" / "audio_cleaner_desktop"
    app.mkdir(parents=True)
    for f in (source / "audio_cleaner_desktop").glob("*.py"):
        shutil.copy2(f, app / f.name)
    (tmp_path / "desktop" / "packaging").mkdir()
    shutil.copy2(PACKAGING / "release_models.py", tmp_path / "desktop" / "packaging" / "release_models.py")
    code = ("import sys\n"
            "for name in ('numpy', 'PySide6', 'cryptography', 'acw_common', 'truststore'):\n"
            "    sys.modules[name] = None\n"
            "sys.argv = [sys.argv[1]]\n"
            "import os\n"
            "sys.path.insert(0, os.path.dirname(sys.argv[0]))\n"
            "import release_models\n"
            "print(len(release_models.whisper_files()), release_models.resources.MODELS_ARCHIVE.name)\n"
            "sys.exit(release_models.main([]))\n")
    script = tmp_path / "desktop" / "packaging" / "release_models.py"
    env = {k: v for k, v in os.environ.items() if k not in ENV and not k.startswith("PYTHON")}
    r = subprocess.run([sys.executable, "-I", "-c", code, str(script)], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert r.returncode == 2, r.stderr  # the usage text: everything imported
    assert r.stdout.split() == [str(len(resources.SPEECH_MODEL.files)), resources.MODELS_ARCHIVE.name]
