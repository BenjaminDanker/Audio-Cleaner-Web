"""Subtitles per part, the pure steps (acw_common.subtitles): padding, cutting a recogniser's segments to the part, the browser's cues, and
putting the parts together as the job's segment files."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acw_common import captions, subtitles as st  # noqa: E402
from acw_common.captions import Segment, Word  # noqa: E402


def seg(a, b, text, words=True):
    toks = text.split()
    step = (b - a) / len(toks)
    return Segment(a, b, text, [Word(a + i * step, a + (i + 1) * step, t) for i, t in enumerate(toks)] if words else [])


def test_a_part_is_given_a_little_sound_around_it_inside_the_file():
    assert st.padded(10, 20, 100) == (8.0, 22.0)
    assert st.padded(1, 99, 100) == (0.0, 100.0)
    assert st.padded(10, 20, 100, pad=0.5) == (9.5, 20.5)


def test_segments_are_cut_to_the_part_and_moved_to_file_time():
    # a recogniser that was given [8, 22) timed everything from its own start; the part is [10, 20)
    heard = [seg(0, 1.5, "tail of the last"), seg(1.5, 4, "words inside the part"), seg(5, 8, "the middle of it"), seg(12.5, 13.5, "heard next door")]
    moved = st.shift(heard, 8.0)
    assert moved[1].start == 9.5 and moved[1].words[0].start == 9.5 and heard[1].start == 1.5  # (the originals stay)
    cut = st.clip_segments(moved, 10, 20)
    assert [(s.start, s.end, s.text) for s in cut] == [(10.0, 12.0, "inside the part"), (13.0, 16.0, "the middle of it")]
    assert all(10 <= w.start < w.end <= 20 for s in cut for w in s.words) and cut[0].words[0].start == 10.125


def test_a_segment_across_the_edge_keeps_the_words_inside_and_the_text_follows():
    s = seg(8.0, 12.0, "one two three four")  # words at 8-9, 9-10, 10-11, 11-12
    out = st.clip_segments([s], 9.5, 20)  # middle (10) is inside; "one" (8.5) is not, "two" (9.5) is
    assert len(out) == 1 and out[0].text == "two three four" and [w.text for w in out[0].words] == ["two", "three", "four"]
    assert out[0].start == 9.5 and out[0].end == 12.0 and out[0].words[0].start == 9.5
    assert st.clip_segments([seg(0, 4, "a b c d")], 3.5, 9) == []  # middle (2) is outside
    no_words = st.clip_segments([seg(8.0, 11.0, "no word times", words=False)], 9, 20)
    assert [(s.start, s.end, s.text) for s in no_words] == [(9.0, 11.0, "no word times")]
    # a language written without spaces
    zh = Segment(0, 4, "你好世界", [Word(0, 1, "你"), Word(1, 2, "好"), Word(2, 3, "世"), Word(3, 4, "界")])
    assert st.clip_segments([zh], 1.5, 5)[0].text == "好世界"


def test_cues_from_a_browser_are_checked_and_cut_to_the_part():
    ok = st.check_cues([{"start": 11, "end": 12.5, "text": "  hello   there "}, {"start": 10.2, "end": 11, "text": "first"}, {"start": 15, "end": 16, "text": " "}], 10, 20)
    assert ok == [{"start": 10.2, "end": 11.0, "text": "first"}, {"start": 11.0, "end": 12.5, "text": "hello there"}]
    assert st.check_cues([{"start": 9.8, "end": 10.4, "text": "a hair early"}, {"start": 19.5, "end": 20.4, "text": "a hair late"}], 10, 20) == [
        {"start": 10.0, "end": 10.4, "text": "a hair early"}, {"start": 19.5, "end": 20.0, "text": "a hair late"}]
    for bad in ("text", {"a": 1}, [1], [{"start": 11, "end": 12}], [{"start": "11", "end": 12, "text": "x"}], [{"start": 11, "end": 12, "text": 5}], [{"start": True, "end": 12, "text": "x"}],
                [{"start": 12, "end": 11, "text": "x"}], [{"start": 5, "end": 6, "text": "x"}], [{"start": 25, "end": 26, "text": "x"}], [{"start": 11, "end": 12, "text": "x", "lang": "en"}],
                [{"start": float("nan"), "end": 12, "text": "x"}], [{"start": 11, "end": 12, "text": "x"}] * (st.MAX_CUES + 1)):
        with pytest.raises(ValueError):
            st.check_cues(bad, 10, 20)
    assert len(st.check_cues([{"start": 11, "end": 12, "text": "x" * 5000}], 10, 20)[0]["text"]) == st.MAX_CUE_CHARS
    got = st.segments_from_cues(ok, 10, 20)
    assert [(s.start, s.end, s.text, s.words) for s in got] == [(10.2, 11.0, "first", []), (11.0, 12.5, "hello there", [])]


def test_the_parts_are_put_together_in_time_order_with_each_translation_of_the_parts_that_asked_for_it():
    a = {"start": 0, "end": 10, "language": "en", "engine": "standard", "segments": [seg(1, 3, "hello there"), seg(5, 8, "how are you")],
         "translations": {"es": [seg(1, 3, "hola", words=False), seg(5, 8, "como estas", words=False)], "fr": [seg(1, 3, "salut", words=False)]}}
    b = {"start": 20, "end": 30, "language": "en", "engine": "browser", "segments": [seg(21, 22, "free part")], "translations": {}}
    c = {"start": 10, "end": 20, "language": "de", "engine": "premium", "segments": [seg(11, 12, "guten tag")], "translations": {"es": [seg(11, 12, "buenos dias", words=False)]}}
    main, tr = st.combine([b, a, c])  # (in any order)
    assert [s["text"] for s in main["segments"]] == ["hello there", "how are you", "guten tag", "free part"]
    assert main["language"] == "en" and main["engine"] == "parts"
    assert set(tr) == {"es", "fr"} and [s["text"] for s in tr["es"]["segments"]] == ["hola", "como estas", "buenos dias"] and [s["text"] for s in tr["fr"]["segments"]] == ["salut"]
    assert tr["es"]["language"] == "es" and tr["es"]["engine"] == "translation" and tr["es"]["source_language"] == "en" and all(s["words"] == [] for s in tr["es"]["segments"])
    # the files the caption writer reads
    assert captions.segments_from_json(main["segments"])[0].words and captions.build_cues(captions.segments_from_json(tr["es"]["segments"]), "es")
    # one engine for all of them keeps its name; no text at all is nothing
    assert st.combine([a])[0]["engine"] == "standard"
    assert st.combine([{"start": 0, "end": 5, "language": None, "segments": [], "translations": {}}]) == (None, {})
    assert st.dominant_language([{"start": 0, "end": 5, "language": "es"}, {"start": 5, "end": 20, "language": "en"}, {"start": 20, "end": 30, "language": None}]) == "en"
    assert st.dominant_language([{"start": 0, "end": 5, "language": None}]) is None
