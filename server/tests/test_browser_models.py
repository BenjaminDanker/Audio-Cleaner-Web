"""The browser model files: listed the same way in deploy/models.lock, served only to signed-in browsers, only the
listed files, with ranges and a long private cache."""
from pathlib import Path

import pytest

from acw import browser_models
from acw.config import get_settings

LOCK = Path(__file__).resolve().parents[2] / "deploy" / "models.lock"


def test_pins_match_models_lock():
    lock = {}
    for line in LOCK.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == "file":
            lock[parts[4]] = (parts[2], int(parts[3]))
    for folder, files in browser_models.MODELS.values():
        for name, pin in files.items():
            assert lock[f"models/{folder}/{name}"] == pin, name


@pytest.fixture(autouse=True)
def own_models_dir(tmp_path, monkeypatch):
    """Never the real models folder: these tests write stand-in files."""
    monkeypatch.setenv("ACW_MODELS_DIR", str(tmp_path / "models"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _install(model="whisper-base-1846881", corrupt=None):
    folder, files = browser_models.MODELS[model]
    base = get_settings().models_dir / folder
    data = {}
    for name, (_, size) in files.items():
        p = base / name
        p.parent.mkdir(parents=True, exist_ok=True)
        body = (b"x" * size) if name != corrupt else b"short"
        p.write_bytes(body)
        data[name] = body
    return data


def test_served_to_signed_in_browsers_only(client, signed_in, monkeypatch):
    _, h = signed_in
    data = _install()
    url = "/api/downloads/browser/whisper-base-1846881/onnx/encoder_model_quantized.onnx"
    r = client.get(url, headers={"Range": "bytes=10-19"})
    assert r.status_code == 206 and r.content == data["onnx/encoder_model_quantized.onnx"][10:20]
    r = client.get("/api/downloads/browser/whisper-base-1846881/config.json")
    assert r.status_code == 200 and r.headers["cache-control"].startswith("private") and "content-disposition" not in r.headers
    assert client.get("/api/downloads/browser/whisper-base-1846881/quantize_config.json").status_code == 404  # not listed
    assert client.get("/api/downloads/browser/whisper-base-1846881/../../acw.env").status_code == 404
    assert client.get("/api/downloads/browser/other/config.json").status_code == 404
    client.cookies.clear()
    assert client.get("/api/downloads/browser/whisper-base-1846881/config.json").status_code == 401


def test_a_wrong_size_file_is_not_served(client, signed_in):
    _install(corrupt="tokenizer.json")
    assert client.get("/api/downloads/browser/whisper-base-1846881/tokenizer.json").status_code == 404
    assert not browser_models.installed("whisper-base-1846881")
