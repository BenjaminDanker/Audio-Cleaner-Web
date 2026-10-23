"""The editor's timeline format: layers, parts, painting, validation, the JSON."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acw_common import pipeline  # noqa: E402
from acw_common.timeline import LANGUAGES, LAYERS, VERSION, VERSION_2, Timeline, TimelineError  # noqa: E402


def test_whole_file_defaults_are_every_asked_layer_on_the_whole_file():
    t = Timeline.whole_file(120.0)
    assert [(n, p.start, p.end) for n, p in t] == [("noise", 0.0, 120.0), ("leveling", 0.0, 120.0)]
    assert t.parts("noise")[0].settings == {"strength_db": 100.0, "remover": "auto"}
    assert t.parts("leveling")[0].settings == {"loudness_lufs": -16.0}
    full = Timeline.whole_file(10.0, echo=True, noise={"strength_db": 24, "remover": "heavy"}, rebuild=True, leveling={"loudness_lufs": None})
    assert [n for n, _ in full] == ["echo", "noise", "rebuild", "leveling"]
    assert full.parts("leveling")[0].settings == {"loudness_lufs": None}
    assert list(Timeline.whole_file(10.0, noise=False, leveling=False)) == []


def test_from_clean_options_is_the_timeline_the_whole_file_choices_stand_for():
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
    {"version": 3, "seconds": 10},  # a version this code does not know
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


# ---------------------------------------------------------------------------------------------------- version 2

def sub(a, b, quality="standard", language="en", translate=()):
    return {"start": a, "end": b, "quality": quality, "language": language, "translate": list(translate)}


def part_settings(quality="standard", language="en", translate=()):
    return {"quality": quality, "language": language, "translate": list(translate)}


V2 = {"version": 2, "seconds": 100, "layers": {
    "noise": [{"start": 0, "end": 10, "browser": True}, {"start": 10, "end": 40, "strength_db": 24, "remover": "light"}],
    "subtitles": [sub(0, 50, "standard", "en", ["es", "fr"]), sub(60, 70, "free", None), sub(80, 90, "accurate", "haw")]}}


def test_version_1_is_written_exactly_as_before_and_version_2_only_for_something_new():
    v1 = {"version": 1, "seconds": 50.0, "layers": {"noise": [{"start": 0.0, "end": 50.0, "strength_db": 100.0, "remover": "light"}],
                                                   "leveling": [{"start": 0.0, "end": 50.0, "loudness_lufs": -16.0}]}}
    assert Timeline.from_json(v1).to_json() == v1 and VERSION == 1 and VERSION_2 == 2
    assert Timeline.whole_file(60.0).to_json()["version"] == 1 and list(Timeline.whole_file(60.0).to_json()["layers"]) == ["noise", "leveling"]
    assert Timeline(10.0).layers["subtitles"] == []
    assert Timeline.from_json({**v1, "version": 2}).to_json() == v1  # (version 2 that says nothing new is version 1 again)
    assert Timeline.from_json(V2).to_json()["version"] == 2
    t = Timeline(100.0).paint("subtitles", 0, 5, part_settings()).erase("subtitles", 0, 5)
    assert t.to_json()["version"] == 1 and "subtitles" not in t.to_json()["layers"]
    assert Timeline(100.0).paint("noise", 0, 5, {"browser": True}).to_json()["version"] == 2
    # a version 1 document may not use the new things, whatever the reader is
    for layers in ({"subtitles": [sub(0, 5)]}, {"noise": [{"start": 0, "end": 5, "browser": True}]}):
        with pytest.raises(TimelineError, match="version 2"):
            Timeline.from_json({"version": 1, "seconds": 10, "layers": layers})
    # the audio layers alone, and the json an old build can read
    assert Timeline.from_json(V2).audio().layers["subtitles"] == [] and Timeline.from_json(V2).audio().layers["noise"] == Timeline.from_json(V2).layers["noise"]


def test_the_subtitles_layer_is_checked_and_its_parts_merge_like_the_others():
    t = Timeline.from_json(V2)
    assert t.to_json() == {"version": 2, "seconds": 100.0, "layers": {
        "noise": [{"start": 0.0, "end": 10.0, "browser": True}, {"start": 10.0, "end": 40.0, "strength_db": 24.0, "remover": "light"}],
        "subtitles": [{"start": 0.0, "end": 50.0, "quality": "standard", "language": "en", "translate": ["es", "fr"]},
                      {"start": 60.0, "end": 70.0, "quality": "free", "language": None, "translate": []},
                      {"start": 80.0, "end": 90.0, "quality": "accurate", "language": "haw", "translate": []}]}}
    assert [p.start for p in t.free_parts()] == [60.0] and [p.end for p in t.browser_parts()] == [10.0]
    t.paint("subtitles", 50, 60, part_settings("standard", "en", ["es", "fr"]))  # same settings as its neighbour: one part
    assert [(p.start, p.end) for p in t.parts("subtitles")][:2] == [(0.0, 60.0), (60.0, 70.0)]
    t.paint("subtitles", 20, 30, part_settings("accurate", "de"))  # painted over: the old part is split
    assert [(p.start, p.end, p.settings["quality"]) for p in t.parts("subtitles")][:3] == [(0.0, 20.0, "standard"), (20.0, 30.0, "accurate"), (30.0, 60.0, "standard")]
    c = t.copy()
    c.layers["subtitles"][0].settings["translate"].append("it")
    assert t.layers["subtitles"][0].settings["translate"] == ["es", "fr"]  # (a copy shares nothing)
    # codes: the site's list for free (the browser's recogniser), any 2-3 letter code for standard and accurate; a language is detected when null or ""
    assert set(LANGUAGES) == {"en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "uk", "tr", "ar", "hi", "ja", "ko", "zh", "sv", "da", "fi", "no", "cs", "el", "he", "id", "vi", "hu", "ro", "th"}
    assert Timeline(10.0).paint("subtitles", 0, 5, {"quality": "standard", "language": ""}).parts("subtitles")[0].settings["language"] is None
    assert Timeline(10.0).paint("subtitles", 0, 5, {"quality": "free", "language": " EN "}).parts("subtitles")[0].settings["language"] == "en"
    assert Timeline(10.0).paint("subtitles", 0, 5, {"quality": "standard", "translate": ["ES", "es", "fr"]}).parts("subtitles")[0].settings["translate"] == ["es", "fr"]
    bad = [
        {"quality": "perfect"}, {}, {"quality": None}, {"quality": "free", "translate": ["es"]},
        {"quality": "standard", "language": "en", "translate": ["en"]}, {"quality": "standard", "translate": ["es", "fr", "de", "it"]},
        {"quality": "free", "language": "haw"}, {"quality": "free", "language": "xx"}, {"quality": "accurate", "language": "english"},
        {"quality": "standard", "language": "english"}, {"quality": "standard", "language": "e"}, {"quality": "standard", "language": "en1"},
        {"quality": "standard", "language": "en-us"}, {"quality": "standard", "language": 5}, {"quality": "standard", "translate": "es"}, {"quality": "standard", "translate": ["haw"]},
        {"quality": "standard", "speed": 2},
    ]
    for settings in bad:
        with pytest.raises(TimelineError):
            Timeline(10.0).paint("subtitles", 0, 5, settings)
    # standard and accurate subtitles take any 2-3 letter code (the server's recognisers know more than the site's list), written as lower case
    for quality in ("standard", "accurate"):
        for code, kept in (("haw", "haw"), ("HAW", "haw"), (" yo ", "yo"), ("ln", "ln"), ("en", "en"), ("zh", "zh")):
            assert Timeline(10.0).paint("subtitles", 0, 5, {"quality": quality, "language": code}).parts("subtitles")[0].settings["language"] == kept
    with pytest.raises(TimelineError, match="unknown language 'haw' for free subtitles"):
        Timeline(10.0).paint("subtitles", 0, 5, {"quality": "free", "language": "haw"})
    with pytest.raises(TimelineError, match="unknown translation language"):  # (a translation target stays in the site's list)
        Timeline(10.0).paint("subtitles", 0, 5, {"quality": "accurate", "translate": ["haw"]})
    with pytest.raises(TimelineError, match="Standard or Accurate"):
        Timeline(10.0).paint("subtitles", 0, 5, {"quality": "free", "translate": ["es"]})


def test_a_noise_part_the_browser_cleans_keeps_the_strength_it_was_sent_but_has_no_remover():
    t = Timeline(100.0).paint("noise", 10, 20, {"browser": True, "strength_db": 12, "remover": "heavy"})
    assert t.parts("noise")[0].settings == {"browser": True, "strength_db": 12.0}  # (kept for the page; the remover is not)
    assert Timeline(100.0).paint("noise", 10, 20, {"browser": True, "strength": "strong"}).parts("noise")[0].settings == {"browser": True, "strength_db": 40.0}
    assert Timeline(100.0).paint("noise", 10, 20, {"browser": True, "strength_db": None}).parts("noise")[0].settings == {"browser": True}
    assert Timeline(100.0).paint("noise", 10, 20, {"browser": True}).to_json()["layers"]["noise"] == [{"start": 10.0, "end": 20.0, "browser": True}]
    assert t.to_json()["layers"]["noise"] == [{"start": 10.0, "end": 20.0, "browser": True, "strength_db": 12.0}] and t.to_json()["version"] == 2
    assert Timeline(100.0).paint("noise", 0, 5, {"browser": False}).parts("noise")[0].settings == {"strength_db": 100.0, "remover": "auto"}
    for bad in ({"browser": "yes"}, {"browser": True, "strength_db": 0}, {"browser": True, "strength_db": 101}, {"browser": True, "strength_db": "huge"}):
        with pytest.raises(TimelineError):
            Timeline(100.0).paint("noise", 0, 5, bad)
    with pytest.raises(TimelineError):
        Timeline(100.0).paint("rebuild", 0, 5, {"browser": True})
    # neighbours with the same settings merge, those with different strengths (or one with none) stay separate parts
    t.paint("noise", 20, 30, {"browser": True, "strength_db": 12})
    assert [(p.start, p.end) for p in t.parts("noise")] == [(10.0, 30.0)]
    t.paint("noise", 30, 40, {"browser": True, "strength_db": 40})
    t.paint("noise", 40, 50, {"browser": True})
    assert [(p.start, p.end, p.settings.get("strength_db")) for p in t.parts("noise")] == [(10.0, 30.0, 12.0), (30.0, 40.0, 40.0), (40.0, 50.0, None)]
    # what the sound depends on does not: the strength is gone, so the three are one part there
    assert [(p.start, p.end, p.settings) for p in t.audio().parts("noise")] == [(10.0, 50.0, {"browser": True})]
    t.paint("noise", 15, 18, {"strength_db": 40, "remover": "light"})  # a part painted over it ends it
    assert [p.settings.get("browser", False) for p in t.parts("noise")] == [True, False, True, True, True]
    doc = {"version": 2, "seconds": 100, "layers": {"noise": [{"start": 10, "end": 20, "browser": True}], "echo": [{"start": 15, "end": 25}]}}
    with pytest.raises(TimelineError) as e:
        Timeline.from_json(doc)
    assert str(e.value) == "Echo removal can't be used on a part cleaned in the browser"
    doc["layers"]["echo"] = [{"start": 20, "end": 25}, {"start": 5, "end": 10}]  # touching is not overlapping
    assert Timeline.from_json(doc).parts("echo")
    # the only place it is checked is where a timeline is read: building one by hand with the desktop's `paint` stays possible
    Timeline(100.0).paint("noise", 10, 20, {"browser": True}).paint("echo", 15, 25)
    with pytest.raises(TimelineError):
        Timeline(100.0).paint("noise", 10, 20, {"browser": True}).paint("echo", 15, 25).check()
