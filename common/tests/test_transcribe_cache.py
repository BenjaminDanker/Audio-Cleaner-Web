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


@pytest.fixture
def fake_parakeet(monkeypatch):
    """onnx_asr and onnxruntime without the models: which models were loaded, and the stretches that were recognised."""
    loads, heard = [], []

    class Piece:
        start, end, text, tokens, timestamps = 0.0, 1.0, "hi", ["▁hi"], [0.0]

    class Pieces:
        def with_timestamps(self):
            return self

        def recognize(self, audio):
            heard.append(len(audio))
            return [Piece()]

    class Model:
        def __init__(self, path):
            self.path = path
            self.asr = types.SimpleNamespace(_encoder=types.SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"]))

        def with_vad(self, vad, **kwargs):
            return Pieces()

    fake = types.SimpleNamespace(load_model=lambda kind, path, **kw: (loads.append(("model", path)), Model(path))[1],
                                 load_vad=lambda kind, path, **kw: (loads.append(("vad", path)), object())[1])
    monkeypatch.setitem(sys.modules, "onnx_asr", fake)

    class Options:
        def add_session_config_entry(self, *a):
            pass

    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(SessionOptions=Options))
    return loads, heard


def test_parakeet_is_loaded_once_for_the_stretches_of_a_caller_that_holds_it(tmp_path, fake_parakeet):
    loads, heard = fake_parakeet
    audio = np.zeros(16000, np.float32)
    held: dict = {}
    for n in (1, 2, 3):
        assert transcribe.transcribe_parakeet(audio[: n * 4000], tmp_path / "v2", tmp_path / "vad", threads=2, held=held)[0].text == "hi"
    assert loads == [("model", str(tmp_path / "v2")), ("vad", str(tmp_path / "vad"))] and heard == [4000, 8000, 12000]
    # another model replaces the held one (one is held at a time); asking for the first again loads it again
    transcribe.transcribe_parakeet(audio, tmp_path / "v3", tmp_path / "vad", threads=2, held=held)
    assert [l for l in loads if l[0] == "model"] == [("model", str(tmp_path / "v2")), ("model", str(tmp_path / "v3"))] and held["key"][0] == str(tmp_path / "v3")
    held.clear()
    transcribe.transcribe_parakeet(audio, tmp_path / "v3", tmp_path / "vad", threads=2, held=held)
    assert len([l for l in loads if l[0] == "model"]) == 3
    # without `held` (the server's whole-file call, the desktop app) every call loads its own, as always
    n = len(loads)
    transcribe.transcribe_parakeet(audio, tmp_path / "v3", tmp_path / "vad", threads=2)
    transcribe.transcribe_parakeet(audio, tmp_path / "v3", tmp_path / "vad", threads=2)
    assert len(loads) == n + 4


def test_the_language_models_are_loaded_once_for_the_stretches_of_a_caller_that_holds_them(tmp_path, monkeypatch):
    loads = []

    class Model:
        def __init__(self, path, **kwargs):
            loads.append(Path(path).name)

        def detect_language(self, audio):
            return "de", 0.9, None

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=Model))
    audio = np.ones(16000, np.float32) * 0.1
    held: dict = {}
    assert [transcribe.detect_language(audio, tmp_path / "base", tmp_path / "small", 2, held=held) for _ in range(3)] == ["de"] * 3
    assert loads == ["base"]
    transcribe.detect_language(audio, tmp_path / "base", tmp_path / "small", 2)
    transcribe.detect_language(audio, tmp_path / "base", tmp_path / "small", 2)
    assert loads == ["base", "base", "base"]
