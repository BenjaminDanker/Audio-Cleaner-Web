"""Automatic suggestions per part: change points, one decision per part, reasons in plain words, clipping spots."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acw_common import editor, suggest  # noqa: E402
from acw_common.timeline import Timeline  # noqa: E402

from test_common import LIB, MODEL, needs_dfn  # noqa: E402
from test_editor import SPEECH_WAV, needs_speech  # noqa: E402

SR = 48000


def speech(seconds):
    sf = pytest.importorskip("soundfile")
    s, sr = sf.read(SPEECH_WAV, dtype="float32", always_2d=True)
    assert sr == SR
    return np.tile(s[:, 0], int(np.ceil(seconds * SR / len(s))))[: int(seconds * SR)].copy()


def active(x):
    return float(np.sqrt(np.mean(x[np.abs(x) > 0.02] ** 2)))


def file_with_known_parts():
    """120 s: 0-40 s clean, 40-80 s the same speech in loud low-pitched noise (about 2 dB under the voice), 80-120 s as heard on a phone
    (300-3300 Hz)."""
    rng = np.random.default_rng(1)
    a = speech(40)
    a *= 0.1 / active(a)
    b = speech(40)
    b *= 0.1 / active(b)
    n = signal.lfilter([1], [1, -0.9], rng.standard_normal(len(b))).astype(np.float32)
    b += n / np.sqrt(np.mean(n ** 2)) * active(b) * 10 ** (-2 / 20)
    c = speech(40)
    c = signal.sosfilt(signal.butter(6, [300, 3300], btype="band", fs=SR, output="sos"), c).astype(np.float32)
    c *= 0.1 / active(c)
    return np.ascontiguousarray(np.concatenate([a, b, c]).astype(np.float32)[:, None])


def overlap(s, a, b):
    return max(0.0, min(s.end, b) - max(s.start, a))


@needs_dfn
@needs_speech
def test_a_file_with_known_parts_gets_suggestions_where_it_changes():
    x = file_with_known_parts()
    r = suggest.suggest(x, LIB, MODEL, threads=4, clipping=False)
    cuts = [a for a, _, _ in r.parts[1:]]
    for truth in (40.0, 80.0):  # the changes are found, to the second or so
        assert min(abs(c - truth) for c in cuts) <= 3.0, (truth, cuts)
    assert all(b - a >= suggest.MIN_PART_S - 1e-9 for a, b, _ in r.parts)  # no part shorter than 20 s
    sug = r.suggestions
    # clean speech needs no rebuild, the phone-like part does (and says why), the noisy part is told apart by its noise
    assert not [s for s in sug if s.rebuild and s.end <= 38.0]
    phone = [s for s in sug if s.rebuild and s.start >= 75.0 and any("phone" in w for w in s.reasons)]
    assert sum(s.end - s.start for s in phone) >= 15.0, [(s.start, s.end, s.reasons) for s in sug]
    noisy = [s for s in sug if "is noisier" in s.reasons and not s.rebuild]
    assert noisy and sum(overlap(s, 40.0, 80.0) for s in noisy) >= 30.0
    assert all(s.noise["remover"] == "heavy" for s in noisy)  # (noise nearly as loud as the voice: the stronger remover)
    assert all(s.reasons and all(isinstance(w, str) for w in s.reasons) for s in sug)  # every suggestion says why
    json.dumps(r.to_json())  # JSON-able
    # the cost is a few CPU-seconds per audio-minute, not a model run per second
    assert r.cpu_seconds / (len(x) / SR) < 1.0
    # applied to the whole-file timeline the suggestions are ordinary parts: valid, and they put Voice rebuild on the phone-like part
    base = suggest.whole_timeline(120.0, r.whole)
    tl = suggest.apply(base, sug)
    assert Timeline.from_json(tl.to_json()) == tl
    assert any(p.start >= 75.0 for p in tl.parts("rebuild")) and not any(p.end <= 38.0 for p in tl.parts("rebuild"))
    # a suggestion only where the part's decision differs from the whole file's
    assert all(d.key != r.whole.key for a, b, d in r.parts if any(s.kind == "part" and s.start == a for s in sug))
    assert not [s for s in sug if s.kind == "part" and any(d.key == r.whole.key for a, b, d in r.parts if a == s.start)]


@needs_speech
def test_clipping_spots_are_short_bursts_with_context():
    x = speech(60)
    x *= 0.6 / np.abs(x).max()
    bursts = [(12.0, 12.6), (30.0, 30.4), (47.0, 47.5)]
    for a, b in bursts:  # recorded too loud for 0.4-0.6 s: the peaks pile up at the top
        i, j = int(a * SR), int(b * SR)
        x[i:j] = np.clip(x[i:j] * 8, -0.6, 0.6)
    spots = suggest.clipping_spots(x[:, None])
    assert len(spots) == 3, spots
    for (a, b), (s, e) in zip(bursts, spots):
        assert s <= a - 0.3 and e >= b + 0.3 and e - s < (b - a) + 1.5  # the clipped samples with half a second of context each side
    clean = speech(60)
    assert suggest.clipping_spots(clean[:, None] * (0.6 / np.abs(clean).max())) == []  # untouched speech has none
    assert suggest.clipping_spots(np.zeros((SR, 1), np.float32)) == []


def test_reasons_are_plain_words():
    D = suggest.Decision
    whole = D(False, "light", "gentle", 30.0, -30.0)
    assert suggest.reasons_for(D(True, "light", "gentle", 30.0, -30.0, ("distorted",)), whole) == ["sounds distorted"]
    assert suggest.reasons_for(D(False, "heavy", "strong", 12.0, -30.0), whole) == ["needs the stronger noise remover", "is noisier"]
    assert "is quiet" in suggest.reasons_for(D(False, "light", "gentle", 30.0, -42.0), whole)
    assert "is cleaner" not in suggest.reasons_for(D(False, "light", "balanced", 40.0, -30.0), whole)  # a stronger setting is never "cleaner"
    assert "has noise that needs a stronger setting" in suggest.reasons_for(D(False, "light", "balanced", 40.0, -30.0), whole)
    assert suggest.reasons_for(D(False, "light", "gentle", 40.0, -30.0), D(False, "light", "balanced", 30.0, -30.0)) == ["is cleaner"]
    assert "sounds like a phone call" in suggest.reasons_for(D(True, "light", "gentle", 30.0, -30.0, ("phone", "muffled")), whole)
    assert suggest.reasons_for(D(False, "light", "gentle", 30.0, -30.0), D(True, "light", "gentle", 30.0, -30.0)) == ["sounds fine here, no rebuild needed"]


def test_suggestions_apply_as_editable_parts():
    base = Timeline.whole_file(100.0, rebuild=True, noise={"strength_db": 100.0, "remover": "heavy"})
    sg = [suggest.Suggestion(20, 40, "part", False, {"strength_db": 24.0, "remover": "light"}, ["is cleaner"]),
          suggest.Suggestion(60, 62, "clipping", True, None, ["sounds distorted here (clipped)"])]
    tl = suggest.apply(base, sg)
    assert [(p.start, p.end) for p in tl.parts("rebuild")] == [(0, 20), (40, 100)]  # no rebuild where the suggestion says it is not needed
    assert [(p.start, p.end, p.settings["strength_db"]) for p in tl.parts("noise")] == [(0, 20, 100.0), (20, 40, 24.0), (40, 100, 100.0)]
    assert base.parts("rebuild")[0].end == 100  # (the original is not changed)


def test_the_pause_finder_moves_a_join_into_a_pause_only():
    n = 6 * SR
    t = np.arange(n) / SR
    voice = (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float64)
    voice[int(3.15 * SR):int(3.40 * SR)] *= 0.001  # a pause 0.15 .. 0.40 s after the nominal join
    pos, found = editor.find_pause(voice, 0, 3 * SR, int(0.3 * SR))
    assert found and 3.15 * SR <= pos <= 3.40 * SR
    pos, found = editor.find_pause(0.2 * np.sin(2 * np.pi * 200 * t), 0, 3 * SR, int(0.3 * SR))  # speech all the way: stays
    assert not found and pos == 3 * SR
    voice2 = voice.copy()
    voice2[int(3.15 * SR):int(3.40 * SR)] = voice[int(1.0 * SR):int(1.25 * SR)]  # no pause now; the one that remains is out of reach
    voice2[int(1.0 * SR):int(1.25 * SR)] *= 0.001
    pos, found = editor.find_pause(voice2, 0, 3 * SR, int(0.3 * SR))
    assert not found and pos == 3 * SR
