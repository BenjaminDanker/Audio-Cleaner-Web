"""The editor's timeline format: layers, parts, painting, validation, the JSON."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acw_common import pipeline  # noqa: E402
from acw_common.timeline import LAYERS, VERSION, Timeline, TimelineError  # noqa: E402


def test_whole_file_defaults_are_every_asked_layer_on_the_whole_file():
    t = Timeline.whole_file(120.0)
    assert [(n, p.start, p.end) for n, p in t] == [("noise", 0.0, 120.0), ("leveling", 0.0, 120.0)]
    assert t.parts("noise")[0].settings == {"strength_db": 100.0, "remover": "auto"}
    assert t.parts("leveling")[0].settings == {"loudness_lufs": -16.0}
    full = Timeline.whole_file(10.0, echo=True, noise={"strength_db": 24, "remover": "heavy"}, rebuild=True, leveling={"loudness_lufs": None})
    assert [n for n, _ in full] == ["echo", "noise", "rebuild", "leveling"]
    assert full.parts("leveling")[0].settings == {"loudness_lufs": None}
    assert list(Timeline.whole_file(10.0, noise=False, leveling=False)) == []


def test_from_clean_options_matches_what_clean_audio_does():
    o = pipeline.CleanOptions(atten_lim_db=40.0, noise="heavy", loudness_lufs=-18.0)
    t = Timeline.from_clean_options(60.0, o, echo=True)
    assert t.parts("echo") and t.parts("noise")[0].settings == {"strength_db": 40.0, "remover": "heavy"}
    assert t.parts("leveling")[0].settings == {"loudness_lufs": -18.0} and not t.parts("rebuild")
    r = Timeline.from_clean_options(60.0, pipeline.CleanOptions(restore_voice=True, leveling=False))
    assert [n for n, _ in r] == ["rebuild"]  # the noise removal does not run when the voice is rebuilt
    assert [n for n, _ in Timeline.from_clean_options(60.0, pipeline.CleanOptions(denoise=False))] == ["leveling"]
    with pytest.raises(TimelineError):
        Timeline.from_clean_options(60.0, pipeline.CleanOptions(atten_lim_db=None))


def test_a_part_painted_over_old_ones_replaces_them_and_never_overlaps():
    t = Timeline(100.0)
    t.paint("noise", 0, 100, {"strength_db": 100, "remover": "light"})
    t.paint("noise", 20, 30, {"strength_db": 12, "remover": "light"})  # inside: the old part is split
    assert [(p.start, p.end, p.settings["strength_db"]) for p in t.parts("noise")] == [(0, 20, 100.0), (20, 30, 12.0), (30, 100, 100.0)]
    t.paint("noise", 25, 60, {"strength_db": 40, "remover": "light"})  # reaches over two of them
    assert [(p.start, p.end, p.settings["strength_db"]) for p in t.parts("noise")] == [(0, 20, 100.0), (20, 25, 12.0), (25, 60, 40.0), (60, 100, 100.0)]
    t.paint("noise", 0, 100, {"strength_db": 40, "remover": "light"})  # over everything: one part
    assert [(p.start, p.end) for p in t.parts("noise")] == [(0, 100)]
    for name in LAYERS:
        ps = t.parts(name)
        assert all(a.end <= b.start for a, b in zip(ps, ps[1:]))
    t.erase("noise", 10, 90)
    assert [(p.start, p.end) for p in t.parts("noise")] == [(0, 10), (90, 100)]


def test_neighbours_with_the_same_settings_are_one_part():
    t = Timeline(60.0)
    t.paint("rebuild", 0, 10).paint("rebuild", 10, 20).paint("rebuild", 30, 40)
    assert [(p.start, p.end) for p in t.parts("rebuild")] == [(0, 20), (30, 40)]


def test_times_are_clamped_to_the_file_and_rounded_to_the_millisecond():
    t = Timeline(30.0)
    t.paint("rebuild", -5, 12.00049)
    t.paint("rebuild", 25, 99)
    t.paint("rebuild", 31, 40)  # entirely past the end
    t.paint("rebuild", 12.0004, 12.0004)  # empty
    assert [(p.start, p.end) for p in t.parts("rebuild")] == [(0.0, 12.0), (25.0, 30.0)]
    assert t.samples("rebuild", 30 * 48000 - 100)[-1][1] == 30 * 48000 - 100  # and to the samples of a shorter recording


def test_json_round_trip_and_version():
    t = Timeline.whole_file(100.0, echo=True, rebuild=False)
    t.paint("rebuild", 10.5, 20)
    t.paint("noise", 30, 40, {"strength": "strong", "remover": "heavy"})  # a name for the strength
    doc = t.to_json()
    assert doc["version"] == VERSION == 1 and doc["seconds"] == 100.0
    assert set(doc["layers"]) == {"echo", "noise", "rebuild", "leveling"}
    assert Timeline.from_json(doc) == t == Timeline.from_json(json.dumps(doc))
    assert Timeline.from_json(t.dumps()) == t
    assert doc["layers"]["noise"][1] == {"start": 30.0, "end": 40.0, "strength_db": 40.0, "remover": "heavy"}
    assert "echo" not in Timeline(10.0).to_json()["layers"]  # an empty layer is left out
    # the recording's real length wins and everything is clamped to it
    shorter = Timeline.from_json(doc, seconds=15.0)
    assert shorter.seconds == 15.0 and all(p.end <= 15.0 for _, p in shorter)


def test_in_json_the_later_part_wins():
    doc = {"version": 1, "seconds": 50, "layers": {"noise": [{"start": 0, "end": 50, "strength_db": 100, "remover": "light"},
                                                             {"start": 10, "end": 20, "strength_db": 12, "remover": "light"}]}}
    t = Timeline.from_json(doc)
    assert [(p.start, p.end) for p in t.parts("noise")] == [(0, 10), (10, 20), (20, 50)]


@pytest.mark.parametrize("doc", [
    [],  # not an object
    {"seconds": 10},  # no version
    {"version": 2, "seconds": 10},  # a version this code does not know
    {"version": True, "seconds": 10},
    {"version": 1},  # no length
    {"version": 1, "seconds": "ten"},
    {"version": 1, "seconds": 10, "extra": 1},
    {"version": 1, "seconds": 10, "layers": []},
    {"version": 1, "seconds": 10, "layers": {"music": []}},  # a layer for later
    {"version": 1, "seconds": 10, "layers": {"noise": {}}},
    {"version": 1, "seconds": 10, "layers": {"noise": [{"start": 0}]}},
    {"version": 1, "seconds": 10, "layers": {"noise": [{"start": 5, "end": 1}]}},  # ends before it starts
    {"version": 1, "seconds": 10, "layers": {"noise": [{"start": 0, "end": 5, "strength_db": 0}]}},
    {"version": 1, "seconds": 10, "layers": {"noise": [{"start": 0, "end": 5, "strength_db": 101}]}},
    {"version": 1, "seconds": 10, "layers": {"noise": [{"start": 0, "end": 5, "strength_db": "huge"}]}},
    {"version": 1, "seconds": 10, "layers": {"noise": [{"start": 0, "end": 5, "remover": "medium"}]}},
    {"version": 1, "seconds": 10, "layers": {"noise": [{"start": 0, "end": 5, "colour": "red"}]}},
    {"version": 1, "seconds": 10, "layers": {"rebuild": [{"start": 0, "end": 5, "strength_db": 12}]}},
    {"version": 1, "seconds": 10, "layers": {"leveling": [{"start": 0, "end": 5, "loudness_lufs": -90}]}},
    {"version": 1, "seconds": 10, "layers": {"leveling": [{"start": 0, "end": float("nan")}]}},
    {"version": 1, "seconds": 10, "layers": {"leveling": [{"start": True, "end": 5}]}},
])
def test_invalid_timelines_are_refused(doc):
    with pytest.raises(TimelineError):
        Timeline.from_json(doc)
    assert issubclass(TimelineError, ValueError)


def test_garbage_json_and_too_many_parts_are_refused():
    with pytest.raises(TimelineError):
        Timeline.from_json("{not json")
    parts = [{"start": i, "end": i + 0.5} for i in range(600)]
    with pytest.raises(TimelineError):
        Timeline.from_json({"version": 1, "seconds": 700, "layers": {"rebuild": parts}})
    t = Timeline(10.0)
    with pytest.raises(TimelineError):
        t.paint("music", 0, 1)


def test_the_copy_is_independent_and_defaults_fill_in():
    t = Timeline(40.0)
    t.paint("noise", 0, 40)
    c = t.copy()
    c.paint("noise", 5, 10, {"strength_db": 12})
    assert len(t.parts("noise")) == 1 and len(c.parts("noise")) == 3
    assert t.parts("noise")[0].settings == {"strength_db": 100.0, "remover": "auto"}
    assert t.covers("noise", 0) and not t.covers("noise", 40.0) and not t.covers("rebuild", 1)
