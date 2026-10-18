"""packaging/fetch_dpdfnet.py: the heavy-noise model is downloaded from its pinned Hugging Face address and only a file
with the pinned size and SHA-256 is kept. A local server stands in for Hugging Face."""
from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import pytest

from audio_cleaner_desktop import config
from test_models import _RangeHandler, server  # noqa: F401  (the local download server fixture)

PACKAGING = Path(__file__).resolve().parents[1] / "packaging"
if str(PACKAGING) not in sys.path:
    sys.path.insert(0, str(PACKAGING))

import fetch_dpdfnet  # noqa: E402


@pytest.fixture
def served(server, monkeypatch):  # noqa: F811
    """The model's bytes, served at `url`, with the pins set to them."""
    model = random.Random(1).randbytes(200_000)
    _RangeHandler.files["/dpdfnet"] = model
    monkeypatch.setattr(fetch_dpdfnet, "DPDFNET_SHA256", hashlib.sha256(model).hexdigest())
    monkeypatch.setattr(fetch_dpdfnet, "DPDFNET_SIZE", len(model))
    return model, server + "/dpdfnet"


def test_the_model_is_downloaded_once_and_checked(tmp_path, served):
    model, url = served
    path = fetch_dpdfnet.fetch(tmp_path / "out", url)
    assert path == tmp_path / "out" / fetch_dpdfnet.DPDFNET_FILE and path.read_bytes() == model
    assert sorted(p.name for p in path.parent.iterdir()) == [fetch_dpdfnet.DPDFNET_FILE]  # no .part left
    assert fetch_dpdfnet.fetch(tmp_path / "out", url + "-gone") == path  # already right: no request is made


def test_a_wrong_model_is_refused_and_leaves_nothing_behind(tmp_path, served, monkeypatch):
    _, url = served
    monkeypatch.setattr(fetch_dpdfnet, "DPDFNET_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="did not download correctly"):
        fetch_dpdfnet.fetch(tmp_path, url)
    assert list(tmp_path.iterdir()) == []


def test_the_default_address_is_the_pinned_one_and_a_pending_revision_says_so(tmp_path, monkeypatch):
    assert fetch_dpdfnet.url() == config.hf_models_url("dpdfnet/dpdfnet8_48khz_hr_batch.onnx")
    assert (fetch_dpdfnet.DPDFNET_SIZE, fetch_dpdfnet.DPDFNET_SHA256[:8]) == (14777093, "b50417b0")
    monkeypatch.setattr(config, "HF_MODELS_REVISION", "PENDING-UPLOAD")
    with pytest.raises(SystemExit, match="PENDING-UPLOAD"):
        fetch_dpdfnet.fetch(tmp_path)
