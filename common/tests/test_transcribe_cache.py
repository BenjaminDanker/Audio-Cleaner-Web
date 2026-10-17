"""The Whisper model is loaded once for all the pieces of a recording, and can be dropped again; the language is
detected with the small model unless it is unsure."""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from acw_common import transcribe


@pytest.fixture
def fake_whisper(monkeypatch):
    loads = []

    class Model:
        def __init__(self, path, **kwargs):
            loads.append((path, kwargs))

        def transcribe(self, audio, **kwargs):
            word = types.SimpleNamespace(start=0.0, end=0.5, word=" hi")
            seg = types.SimpleNamespace(start=0.0, end=0.5, text=" hi", words=[word])
            return iter([seg]), types.SimpleNamespace(language="en")

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=Model))
    transcribe.unload()
    yield loads
    transcribe.unload()


def test_one_load_serves_every_piece_and_a_call_without_a_model_loads_its_own(tmp_path, fake_whisper):
    model = transcribe.load_model(tmp_path, threads=2)
    assert transcribe.load_model(tmp_path, threads=2) is model and len(fake_whisper) == 1
    audio = np.zeros(16000, np.float32)
    for _ in range(3):
        segs, language = transcribe.transcribe(audio, tmp_path, threads=2, model=model)
        assert segs[0].text == "hi" and language == "en"
    assert len(fake_whisper) == 1
    transcribe.transcribe(audio, tmp_path, threads=2)  # the server calls it like this: one load per call
    assert len(fake_whisper) == 2
    transcribe.unload()
    assert transcribe.load_model(tmp_path, threads=3) is not model and len(fake_whisper) == 3


def test_language_detection_asks_the_bigger_model_only_when_unsure(tmp_path, monkeypatch):
    asked = []

    class Model:
        def __init__(self, path, **kwargs):
            self.path = path

        def detect_language(self, audio):
            asked.append(self.path)
            lang, prob = {"sure": ("de", 0.9), "unsure": ("nl", 0.4), "big": ("fr", 0.95)}[Path(self.path).name]
            return lang, prob, None

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=Model))
    audio = np.ones(16000, np.float32) * 0.1
    assert transcribe.detect_language(audio, tmp_path / "sure", tmp_path / "big") == "de"
    assert [Path(p).name for p in asked] == ["sure"]
    asked.clear()
    assert transcribe.detect_language(audio, tmp_path / "unsure", tmp_path / "big") == "fr"
    assert [Path(p).name for p in asked] == ["unsure", "big"]
