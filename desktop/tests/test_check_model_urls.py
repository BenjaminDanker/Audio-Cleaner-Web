"""packaging/check_model_urls.py (the CI guard) and how the catalog pins every model file. No network: HEAD is faked."""
from __future__ import annotations

import re
import sys
from pathlib import Path

from audio_cleaner_desktop import config, resources

PACKAGING = Path(__file__).resolve().parents[1] / "packaging"
if str(PACKAGING) not in sys.path:
    sys.path.insert(0, str(PACKAGING))

import check_model_urls  # noqa: E402

OWN = (resources.NOISE_MODEL, resources.VOICE_MODEL)
HF_BASE = f"https://huggingface.co/{config.HF_MODELS_REPO}/resolve/{config.HF_MODELS_REVISION}/"


def test_every_downloaded_file_is_pinned_by_revision_size_and_sha256():
    assert re.fullmatch(r"[0-9a-f]{40}", config.HF_MODELS_REVISION)  # a commit, never a branch
    for m in OWN:
        for f in (*m.files, *m.licences):
            assert f.url.startswith(HF_BASE) and f.size > 0 and re.fullmatch(r"[0-9a-f]{64}", f.sha256), f
        assert all(f.path.startswith("LICENSES/") for f in m.licences) and m.licences
    # the repository layout the owner uploads
    assert {f.url.removeprefix(HF_BASE) for f in resources.NOISE_MODEL.files} == {
        "deepfilternet3/DeepFilterNet3_onnx.tar.gz", "dpdfnet/dpdfnet8_48khz_hr_batch.onnx"}
    assert {f.url.removeprefix(HF_BASE) for f in resources.VOICE_MODEL.files} == {"sidon/sidon_fe.onnx", "sidon/sidon_dec.onnx"}
    assert config.hf_models_url("x/y") == HF_BASE + "x/y"


def _head(answers):
    seen = []

    def head(url):
        seen.append(url)
        return answers.get(url, (404, {}))

    return head, seen


def test_check_reports_every_missing_or_resized_file(monkeypatch):
    monkeypatch.setattr(config, "HF_MODELS_REVISION", "a" * 40)
    files = [f for m in OWN for f in (*m.files, *m.licences)] + list(resources.SPEECH_MODEL.files)
    answers = {f.url: (200, {"content-length": str(f.size)}) for f in files}
    head, seen = _head(answers)
    assert check_model_urls.check(head) == [] and sorted(seen) == sorted(f.url for f in check_model_urls.pinned_files())
    assert all(f.url in seen for m in OWN for f in (*m.files, *m.licences))  # ours are checked once the revision is known
    first, second = check_model_urls.pinned_files()[0], check_model_urls.pinned_files()[-1]
    answers[first.url] = (200, {"content-length": "5"})
    answers[second.url] = (404, {})
    found = check_model_urls.check(head)
    assert len(found) == 2 and "5 bytes there" in found[0] and "HTTP 404" in found[1]


def test_remote_size_follows_redirects_and_prefers_hugging_faces_linked_size():
    answers = {
        "https://a/hf": [(302, {"x-linked-size": "20", "location": "https://cdn/x"})],
        "https://cdn/x": [(200, {"content-length": "999"})],
        "https://a/rel": [(302, {"location": "/next"})],
        "https://a/next": [(200, {"content-length": "5"})],
        "https://a/gone": [(404, {})],
    }
    head = lambda url: answers[url].pop(0)  # noqa: E731
    assert check_model_urls.remote_size("https://a/hf", head) == (20, "")
    assert check_model_urls.remote_size("https://a/rel", head) == (5, "")
    assert check_model_urls.remote_size("https://a/gone", head) == (None, "HTTP 404")
