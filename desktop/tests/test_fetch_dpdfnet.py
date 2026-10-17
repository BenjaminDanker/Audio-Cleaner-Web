"""packaging/fetch_dpdfnet.py: the heavy-noise model is read out of the models archive with a few range requests, and
only a file with the pinned SHA-256 is kept. No network: a function that serves a zip from memory stands in."""
from __future__ import annotations

import hashlib
import io
import random
import sys
import zipfile
from pathlib import Path

import pytest

PACKAGING = Path(__file__).resolve().parents[1] / "packaging"
if str(PACKAGING) not in sys.path:
    sys.path.insert(0, str(PACKAGING))

import fetch_dpdfnet  # noqa: E402


def make_archive(model: bytes) -> bytes:
    rnd = random.Random(7)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        z.writestr("README.txt", "hi")
        z.writestr("ap-bwe/8kto48k.onnx", rnd.randbytes(3_000_000))  # the big members that must not be downloaded
        z.writestr(fetch_dpdfnet.MEMBER, model)
        z.writestr("sidon/sidon_fe.onnx", rnd.randbytes(3_000_000))
    return buf.getvalue()


@pytest.fixture
def served(monkeypatch):
    """(model bytes, archive bytes, list of the (first, last) ranges asked for), with the pin set to the model."""
    model = random.Random(1).randbytes(200_000)
    archive = make_archive(model)
    asked: list[tuple[int, int]] = []

    def http_fetch(url):
        def fetch(first, last):
            asked.append((first, last))
            return archive[first:last + 1]
        return fetch

    monkeypatch.setattr(fetch_dpdfnet, "DPDFNET_SHA256", hashlib.sha256(model).hexdigest())
    monkeypatch.setattr(fetch_dpdfnet, "_range_size", lambda url: len(archive))
    monkeypatch.setattr(fetch_dpdfnet, "_http_fetch", http_fetch)
    return model, archive, asked


def test_only_the_model_is_read_from_the_archive(tmp_path, served):
    model, archive, asked = served
    path = fetch_dpdfnet.fetch(tmp_path / "out", "https://example.invalid/models.zip")
    assert path == tmp_path / "out" / fetch_dpdfnet.DPDFNET_FILE and path.read_bytes() == model
    assert sum(last - first + 1 for first, last in asked) < len(archive) // 4  # the two 3 MB members stay unread
    assert not list((tmp_path / "out").glob("*.partial"))


def test_a_file_that_is_already_right_is_not_fetched_again(tmp_path, served, monkeypatch):
    model, _, asked = served
    (tmp_path / fetch_dpdfnet.DPDFNET_FILE).write_bytes(model)
    monkeypatch.setattr(fetch_dpdfnet, "_range_size", lambda url: pytest.fail("no request was expected"))
    assert fetch_dpdfnet.fetch(tmp_path, "https://example.invalid/models.zip").read_bytes() == model
    assert asked == []


def test_a_wrong_model_is_refused_and_leaves_nothing_behind(tmp_path, served, monkeypatch):
    monkeypatch.setattr(fetch_dpdfnet, "DPDFNET_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="not the expected model"):
        fetch_dpdfnet.fetch(tmp_path, "https://example.invalid/models.zip")
    assert list(tmp_path.iterdir()) == []


def test_a_server_without_ranges_gets_the_whole_archive(tmp_path, served, monkeypatch):
    model, archive, _ = served
    monkeypatch.setattr(fetch_dpdfnet, "_range_size", lambda url: None)

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    monkeypatch.setattr(fetch_dpdfnet.urllib.request, "urlopen", lambda request, timeout=None: Response(archive))
    assert fetch_dpdfnet.fetch(tmp_path, "https://example.invalid/models.zip").read_bytes() == model


def test_main_reports_a_missing_member_without_a_traceback(tmp_path, monkeypatch, capsys):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("README.txt", "hi")
    archive = buf.getvalue()
    monkeypatch.setattr(fetch_dpdfnet, "_range_size", lambda url: len(archive))
    monkeypatch.setattr(fetch_dpdfnet, "_http_fetch", lambda url: (lambda first, last: archive[first:last + 1]))
    assert fetch_dpdfnet.main([str(tmp_path)]) == 1
    assert "fetch_dpdfnet.py" in capsys.readouterr().err
