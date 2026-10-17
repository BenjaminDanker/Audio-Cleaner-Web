"""Captions for languages written without spaces, and that other languages are unchanged."""
import pytest

from acw_common import captions
from acw_common.captions import Segment, Word

CHINESE = "今天天气很好我们去公园散步吧然后一起吃午饭"


def chars(text, step=0.3):
    return [Word(i * step, (i + 1) * step, ch) for i, ch in enumerate(text)]


@pytest.mark.parametrize("language", ["zh", None])  # given, or guessed from the text
def test_chinese_characters_are_joined_without_spaces_and_lines_break_by_width(language):
    cues = captions.build_cues([Segment(0, 6.5, CHINESE, chars(CHINESE))], language)
    text = "".join("".join(c.lines) for c in cues)
    assert text == CHINESE and " " not in text
    assert all(len(line) * 2 <= captions.MAX_LINE for c in cues for line in c.lines)  # full-width characters
    assert all(len(c.lines) <= captions.MAX_LINES for c in cues)


def test_thai_without_word_timings_is_wrapped_by_characters():
    thai = "สวัสดีครับวันนี้อากาศดีมากเราไปเดินเล่นที่สวนสาธารณะกันเถอะ"
    cues = captions.build_cues([Segment(0, 5, thai, [])], "th")
    assert "".join("".join(c.lines) for c in cues) == thai
    assert all(len(line) <= captions.MAX_LINE for c in cues for line in c.lines)


def test_latin_words_inside_chinese_keep_their_space():
    words = [Word(0, 0.3, "我"), Word(0.3, 0.6, "用"), Word(0.6, 0.9, "Audio"), Word(0.9, 1.2, "Cleaner")]
    cue = captions.build_cues([Segment(0, 1.2, "我用Audio Cleaner", words)], "zh")[0]
    assert cue.lines == ["我用Audio Cleaner"]


def test_the_text_transcript_joins_chinese_segments_without_spaces():
    segs = [Segment(0, 2, "你好。", []), Segment(2, 4, "再见。", [])]
    assert captions.to_text(segs, "zh") == "你好。再见。\n"
    assert captions.to_text([Segment(0, 2, "Hello.", []), Segment(2, 4, "Bye.", [])]) == "Hello. Bye.\n"


def test_other_languages_are_unchanged():
    words = [Word(i * 0.3, i * 0.3 + 0.3, w) for i, w in enumerate("Hello there how are you doing today my friend".split())]
    seg = Segment(0, 3, "Hello there how are you doing today my friend", words)
    for language in (None, "en", "ko"):
        cues = captions.build_cues([seg], language)
        assert [c.lines for c in cues] == [["Hello there how are you doing today my", "friend"]]
