"""The same settings for many files, and Start all: what a file's settings are in words and for other files (doc.FileSettings, Doc.settings,
Doc.take_settings), the dialog that offers them (share_settings.py), the start row's link and "Start all N" button, and the window giving the
ticked files the settings and starting everything that waits, each file with its own choices. The file facts, the waveform, what Automatic picks and
the run are stood in for (studio_helpers.py), so nothing is decoded or cleaned. A last test takes screenshots for a look (only when ACW_SHOTS_DIR is
set; it asserts nothing about how it looks)."""
from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QPushButton

from acw_common.timeline import Timeline
from audio_cleaner_desktop import engine, resources
from audio_cleaner_desktop.gui import main_window
from audio_cleaner_desktop.gui.doc import FileSettings, _most_of
from audio_cleaner_desktop.gui.items import CANCELLED, DONE, RUNNING, WAITING
from audio_cleaner_desktop.gui.share_settings import Choice, ShareSettingsDialog

import studio_helpers as sh
from conftest import wait_until
from studio_helpers import settle

STRONG_HEAVY = {"strength_db": 40.0, "remover": "heavy"}
GENTLE_LIGHT = {"strength_db": 12.0, "remover": "light"}
PODCAST = {"loudness_lufs": -16.0}


def layers(**on):
    """Every layer off but the ones given (what FileSettings.layers says: None is off, an empty dict is a layer without settings)."""
    return {"echo": None, "noise": None, "rebuild": None, "leveling": None, **on}


def explicit(**on) -> FileSettings:
    return FileSettings(layers(**on), "off", "auto", "same")


AUTOMATIC = FileSettings(None, "off", "auto", "same")


# ---------------------------------------------------------------------------------------------------- what the settings say

def test_automatic_is_said_as_picked_for_each_file():
    text = AUTOMATIC.describe(False)
    assert text.startswith("noise reduction picked for the file") and "no noise reduction" not in text


def test_automatic_says_what_it_does_besides_what_it_picks_from_the_remembered_choices():
    assert AUTOMATIC.describe(False) == "noise reduction picked for the file"  # (no leveling and no echo remembered: nothing more to say)
    assert FileSettings(None, "off", "auto", "same", auto_leveling="podcast").describe(False) == "noise reduction picked for the file, leveled (podcast)"
    assert FileSettings(None, "off", "auto", "same", auto_leveling="youtube").describe(False) == "noise reduction picked for the file, leveled (YouTube)"
    both = FileSettings(None, "srt", "auto", "same", auto_leveling="broadcast", auto_echo=True).describe(False)
    assert "speaker echo removed" in both and "leveled (broadcast)" in both and "subtitles: srt file" in both
    assert "speaker echo removed" not in FileSettings(None, "off", "auto", "same", auto_leveling="youtube").describe(False)


def test_explicit_noise_strength_remover_and_leveling_are_said_in_words():
    text = explicit(noise=STRONG_HEAVY, leveling=PODCAST).describe(False)
    assert text == "Strong noise reduction, for heavy noise, leveled (podcast)"
    assert explicit(noise={"strength_db": 24.0, "remover": "auto"}).describe(False) == "Balanced noise reduction"  # (an automatic remover and no leveling: not said)
    assert explicit(noise={"strength_db": 12.0, "remover": "light"}, leveling={"loudness_lufs": -14.0}).describe(True) \
        == "Gentle noise reduction, for light noise, leveled (YouTube)"
    assert "speaker echo removed" in explicit(noise=GENTLE_LIGHT, echo={}).describe(False)


def test_no_noise_and_no_rebuild_says_no_noise_reduction():
    assert explicit(leveling=PODCAST).describe(False) == "no noise reduction, leveled (podcast)"
    assert explicit().describe(False) == "no noise reduction"
    assert "no noise reduction" not in explicit(noise=GENTLE_LIGHT).describe(False)
    assert "no noise reduction" not in explicit(rebuild={}).describe(False)


def test_rebuild_is_said_as_the_voice_rebuilt():
    text = explicit(rebuild={}, leveling=PODCAST).describe(False)
    assert text == "voice rebuilt, leveled (podcast)"


def test_subtitles_video_size_and_the_ending_are_said_last():
    s = FileSettings(layers(noise=GENTLE_LIGHT), "srt", "de", "720", ending=".mp3")
    text = s.describe(False)
    assert text.startswith("Gentle noise reduction") and "subtitles: srt file" in text and "video 720p" in text and text.endswith(", saved as .mp3")
    assert "saved as" not in FileSettings(layers(noise=GENTLE_LIGHT), "off", "auto", "same").describe(False)  # (the usual type: nothing to say)


def test_most_of_is_the_settings_that_cover_the_most_time_and_a_layer_on_less_than_half_is_off():
    tl = Timeline(100.0)
    tl.paint("noise", 0, 100, STRONG_HEAVY)
    tl.paint("noise", 80, 100, GENTLE_LIGHT)  # (20 s of 100)
    tl.paint("rebuild", 0, 40)  # (less than half)
    tl.paint("echo", 0, 100)
    tl.paint("leveling", 0, 30, {"loudness_lufs": -20.0})
    tl.paint("leveling", 50, 80, {"loudness_lufs": -20.0})  # (60 s at -20 in two parts, with a gap that is off, and 20 s at -16 between)
    tl.paint("leveling", 30, 50, PODCAST)
    assert _most_of(tl, "noise") == STRONG_HEAVY
    assert _most_of(tl, "rebuild") is None
    assert _most_of(tl, "echo") == {} and _most_of(tl, "echo") is not None  # (on, and a layer without settings)
    assert _most_of(tl, "leveling") == {"loudness_lufs": -20.0}  # (the two parts with the same settings add up)
    tl.paint("rebuild", 0, 50)  # exactly half counts as on
    assert _most_of(tl, "rebuild") == {}
    tl.erase("rebuild", 0, 100)
    tl.paint("rebuild", 0, 49.9)
    assert _most_of(tl, "rebuild") is None
    assert _most_of(Timeline(100.0), "noise") is None  # (no parts at all)


# ---------------------------------------------------------------------------------------------------- the window

@pytest.fixture
def w(window, runner):
    """The window with the runs held: a started file waits in the runner, so what is queued can be looked at; released at the end. One file at a
    time, so the rest stand in line."""
    runner.gate = threading.Event()
    window.prefs.parallel = 1
    window._settings_changed()
    yield window
    runner.gate.set()
    wait_until(lambda: not window.queue.busy, 10)


def put(w, tmp_path, *names, wait=True) -> list:
    """Add files (the first is in the editor) and wait until they are read; returns their docs."""
    paths = []
    for name in names:
        path = tmp_path / name
        path.write_bytes(b"x" * 2000)
        paths.append(path)
    assert w.add_paths(paths) == len(paths)
    items = [next(i for i in w._items.values() if i.path.name == p.name) for p in paths]
    if wait:
        assert wait_until(lambda: all(i.doc.model is not None for i in items), 10)
        sh.ready(w)
        settle(w)
    return [i.doc for i in items]


def read_each_as(w, **facts_by_name) -> None:
    """Files named here are read as another kind of file (tracks=2, video=True, ext=".mp4")."""
    base = w.doc_env.probe

    def probe(path, ff=None):
        return sh.info(w.fakes.seconds, **facts_by_name[path.name]) if path.name in facts_by_name else base(path, ff)

    w.doc_env.probe = probe


def hold_reading(w, name: str) -> threading.Event:
    """The named file is not read until the event is set (at most 10 s)."""
    gate, base = threading.Event(), w.doc_env.probe

    def probe(path, ff=None):
        if path.name == name:
            gate.wait(10)
        return base(path, ff)

    w.doc_env.probe = probe
    return gate


def whole(doc, layer: str) -> list[dict]:
    """The parts of a layer of the draft, as {start, end, **settings}."""
    return [p.to_json() for p in doc.model.draft.layers[layer]]


def accept_with(monkeypatch, *, untick=(), seen=None):
    """The share dialog is answered at once: the files in `untick` are unticked, then it is accepted. `seen` gets what it showed."""
    def exec_(self):
        if seen is not None:
            seen.update(boxes=dict(self.boxes), ok_at_first=self.ok.text(), texts=[lab.text() for lab in self.findChildren(QLabel)])
        for item_id in untick:
            self.boxes[item_id].setChecked(False)
        if seen is not None:
            seen.update(ok_after=self.ok.text(), enabled=self.ok.isEnabled())
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(ShareSettingsDialog, "exec", exec_)


# ---- Doc.settings

def test_a_file_has_settings_to_give_only_once_it_is_read(w, tmp_path):
    gate = hold_reading(w, "slow.wav")
    try:
        [a, slow] = put(w, tmp_path, "a.wav", "slow.wav", wait=False)
        assert wait_until(lambda: a.model is not None, 10)
        assert a.settings() is not None and slow.model is None and slow.settings() is None
    finally:
        gate.set()
    assert wait_until(lambda: slow.model is not None, 10)
    assert slow.settings() is not None


def test_a_file_on_automatic_gives_automatic_until_something_was_changed_by_hand(w, tmp_path):
    [a, b] = put(w, tmp_path, "a.wav", "b.wav")
    assert a.auto and a.applied and a.untouched_auto
    assert a.settings().layers is None and not a.settings().had_parts  # untouched
    assert b.auto and not b.applied and b.settings().layers is None  # (not opened yet: not listened to, so not applied yet)
    a.model.set_strength("gentle", 0, a.seconds)  # changed by hand: Automatic's choices are not what it has now
    s = a.settings()
    assert not a.untouched_auto and s.layers is not None and s.layers["noise"]["strength_db"] == 12.0
    assert s.layers["leveling"] == PODCAST and s.layers["rebuild"] is None and s.layers["echo"] is None and not s.had_parts


def test_a_file_with_parts_gives_what_most_of_it_has_and_says_it_had_parts(w, tmp_path):
    [a] = put(w, tmp_path, "a.wav")
    a.set_auto(False)
    m, n = a.model, a.seconds  # 11 s
    m.set_strength("gentle", 0, n)
    m.set_strength("strong", 0, 3)  # 3 s strong, 8 s gentle
    m.set_layer("leveling", False, 0, 7)  # leveling on 4 s of 11: less than half, so off
    s = a.settings()
    assert s.had_parts and s.layers["noise"] == {"strength_db": 12.0, "remover": "auto"} and s.layers["leveling"] is None
    assert "Gentle noise reduction" in s.describe(False) and "leveled" not in s.describe(False)


def test_the_ending_of_the_result_is_a_setting_only_when_it_is_not_the_usual_one(w, tmp_path):
    [a, b] = put(w, tmp_path, "a.wav", "b.wav")
    assert a.settings().ending is None
    a.set_out_name("a-cleaned.mp3", custom=True)
    assert a.settings().ending == ".mp3" and a.settings().describe(False).endswith(", saved as .mp3")
    a.set_out_name("another name.wav", custom=True)  # (a name of its own with the usual type)
    assert a.settings().ending is None


# ---- Doc.take_settings

def test_explicit_settings_make_a_whole_file_timeline_and_the_file_keeps_its_own_name(w, tmp_path):
    [a, b] = put(w, tmp_path, "a.wav", "b.wav")
    assert b.auto and not b.model.can_undo
    b.take_settings(explicit(noise=STRONG_HEAVY, leveling=PODCAST))
    assert not b.auto and not b.applied and not b.untouched_auto
    assert whole(b, "noise") == [{"start": 0.0, "end": b.seconds, **STRONG_HEAVY}] and whole(b, "leveling") == [{"start": 0.0, "end": b.seconds, **PODCAST}]
    assert whole(b, "echo") == [] and whole(b, "rebuild") == []
    assert b.out_name == "b-cleaned.wav" and not b.out_custom and b.item.out_name == "b-cleaned.wav" and a.out_name == "a-cleaned.wav"
    options, timeline = b.run_plan()
    assert timeline is not None and timeline["layers"]["noise"][0]["strength_db"] == 40.0 and timeline["layers"]["noise"][0]["remover"] == "heavy"
    assert [p["loudness_lufs"] for p in timeline["layers"]["leveling"]] == [-16.0] and "rebuild" not in timeline["layers"]
    assert b.model.loudness() == -16.0 and a.auto  # (nobody else changed)


def test_explicit_settings_with_a_loudness_that_keeps_the_level_and_no_leveling(w, tmp_path):
    [a, b, c] = put(w, tmp_path, "a.wav", "b.wav", "c.wav")
    b.take_settings(explicit(noise=GENTLE_LIGHT, leveling={"loudness_lufs": -23.0}))
    assert b.model.loudness() == -23.0 and whole(b, "leveling")[0]["loudness_lufs"] == -23.0
    c.take_settings(explicit(noise=GENTLE_LIGHT))  # leveling off
    assert whole(c, "leveling") == [] and whole(c, "noise")[0]["strength_db"] == 12.0
    c.take_settings(explicit(rebuild={}, leveling=PODCAST))  # (a rebuilt voice instead of noise reduction)
    assert whole(c, "noise") == [] and whole(c, "rebuild") == [{"start": 0.0, "end": c.seconds}]


def test_automatic_settings_give_a_file_with_hand_changes_back_to_automatic(w, tmp_path):
    [a, b, c] = put(w, tmp_path, "a.wav", "b.wav", "c.wav")
    b.model.set_strength("gentle", 0, 5)  # hand changes on Automatic
    b.model.set_layer("leveling", False, 0, b.seconds)
    c.set_auto(False)  # c is off Automatic altogether (this also changes what the next files start with, not these)
    for doc in (b, c):
        assert not doc.untouched_auto and doc.run_plan()[1] is not None
        doc.take_settings(FileSettings(None, "srt", "de", "same"))
        assert doc.auto and doc.untouched_auto and not doc.applied and doc.auto_base is None and not doc.model.can_undo
        options, timeline = doc.run_plan()
        assert timeline is None and options.strength == engine.AUTO_STRENGTH and options.subtitles == "srt" and options.language == "de"
        assert (doc.subtitles, doc.language) == ("srt", "de")
    assert whole(b, "leveling") == [{"start": 0.0, "end": b.seconds, **PODCAST}]  # (the draft is the whole-file baseline again: the changes are gone)
    assert whole(b, "noise") != [] and len(whole(b, "noise")) == 1


def test_a_file_on_automatic_that_was_listened_to_listens_again_when_it_is_opened_after_taking_automatic(w, tmp_path):
    [a, b] = put(w, tmp_path, "a.wav", "b.wav")
    w.open_item(b.item.id)
    assert wait_until(lambda: b.applied and not b.listening, 10)
    b.model.set_strength("gentle", 0, 5)
    assert b.auto_custom
    w.open_item(a.item.id)
    settle(w)
    b.take_settings(AUTOMATIC)
    assert not b.applied and b.auto_base is None
    w.open_item(b.item.id)
    assert wait_until(lambda: b.applied and not b.listening, 10)
    assert b.untouched_auto and b.auto_base is not None and b.run_plan()[1] is None  # (decided afresh for this file)


def test_echo_is_dropped_for_a_file_with_one_track_and_kept_for_two(w, tmp_path):
    read_each_as(w, **{"two.wav": {"tracks": 2}})
    [a, one, two] = put(w, tmp_path, "a.wav", "one.wav", "two.wav")
    for doc in (one, two):
        doc.take_settings(explicit(noise=GENTLE_LIGHT, echo={}, leveling=PODCAST))
    assert one.facts.tracks == 1 and whole(one, "echo") == [] and len(whole(one, "noise")) == 1
    assert two.facts.tracks == 2 and whole(two, "echo") == [{"start": 0.0, "end": two.seconds}]


def test_the_ending_is_taken_only_when_the_name_is_not_custom_and_the_file_can_be_saved_as_it(w, tmp_path):
    [a, plain, custom, other] = put(w, tmp_path, "a.wav", "plain.wav", "custom.wav", "other.wav")
    custom.set_out_name("my own name.wav")
    s = FileSettings(layers(noise=GENTLE_LIGHT), "off", "auto", "same", ".mp3")
    plain.take_settings(s)
    assert plain.out_name == "plain-cleaned.mp3" and plain.item.out_name == "plain-cleaned.mp3"
    custom.take_settings(s)
    assert custom.out_name == "my own name.wav" and custom.out_custom
    other.take_settings(FileSettings(layers(noise=GENTLE_LIGHT), "off", "auto", "same", ".m4v"))  # (a type only an .m4v video can be saved as)
    assert other.out_name == "other-cleaned.wav"
    # the usual type again: nothing to change
    plain.take_settings(explicit(noise=GENTLE_LIGHT))
    assert plain.out_name == "plain-cleaned.mp3"  # (a file keeps what it has when the others say nothing about the type)


def test_embedded_subtitles_become_an_srt_file_when_the_result_cannot_hold_them(w, tmp_path):
    read_each_as(w, **{"clip.mp4": {"video": True, "ext": ".mp4"}})
    [a, audio, ending, clip] = put(w, tmp_path, "a.wav", "audio.wav", "ending.wav", "clip.mp4")
    embed = FileSettings(layers(noise=GENTLE_LIGHT), "embed", "auto", "same")
    audio.take_settings(embed)  # (a sound file's result is a sound file)
    assert audio.subtitles == "srt"
    ending.take_settings(FileSettings(layers(noise=GENTLE_LIGHT), "embed", "auto", "same", ".mp3"))
    assert ending.subtitles == "srt" and ending.out_name.endswith(".mp3")
    clip.take_settings(embed)  # a video can hold them
    assert clip.subtitles == "embed" and clip.out_name == "clip-cleaned.mp4"
    clip.take_settings(FileSettings(layers(noise=GENTLE_LIGHT), "embed", "auto", "same", ".flv"))  # (a type that holds none)
    assert clip.out_name == "clip-cleaned.flv" and clip.subtitles == "srt"


def test_a_file_still_being_read_takes_the_settings_when_it_is_read(w, tmp_path):
    gate = hold_reading(w, "slow.wav")
    try:
        [a, slow] = put(w, tmp_path, "a.wav", "slow.wav", wait=False)
        assert wait_until(lambda: a.model is not None, 10)
        slow.take_settings(FileSettings(layers(noise=STRONG_HEAVY, leveling=PODCAST), "vtt", "fr", "same", ".mp3"))
        assert slow.model is None and slow._given is not None and slow.subtitles != "vtt"  # (kept for when it is read)
    finally:
        gate.set()
    assert wait_until(lambda: slow.model is not None, 10)
    settle(w)
    assert slow._given is None and not slow.auto and (slow.subtitles, slow.language) == ("vtt", "fr")
    assert whole(slow, "noise") == [{"start": 0.0, "end": slow.seconds, **STRONG_HEAVY}] and slow.out_name == "slow-cleaned.mp3"


def test_a_job_and_a_file_already_in_line_are_left_alone(w, tmp_path):
    [a, b, c] = put(w, tmp_path, "a.wav", "b.wav", "c.wav")
    assert w.start_item(b.item) and b.stage == "processing"
    before = (b.model.draft.copy(), b.subtitles, b.auto, b.out_name)
    b.take_settings(FileSettings(layers(noise=STRONG_HEAVY), "srt", "de", "same", ".mp3"))
    assert (b.model.draft, b.subtitles, b.auto, b.out_name) == before and b._given is None
    folder = sh.make_folder(tmp_path)
    job, eid = sh.add_job(w.history, folder)
    w.doc_env.open_session = sh.open_session
    w.open_entry(eid)
    doc = w.doc
    assert wait_until(lambda: doc.model is not None, 10)
    assert doc.is_job
    draft, name = doc.model.draft.copy(), doc.out_name
    doc.take_settings(explicit(noise=STRONG_HEAVY))
    assert doc.model.draft == draft and doc.out_name == name and not doc.model.can_undo and doc._given is None


def test_undo_after_explicit_settings_brings_the_old_draft_back(w, tmp_path):
    [a, b] = put(w, tmp_path, "a.wav", "b.wav")
    b.set_auto(False)
    b.model.set_strength("balanced", 0, b.seconds)
    old = b.model.draft.copy()
    b.take_settings(explicit(noise=STRONG_HEAVY, leveling=PODCAST))
    assert b.model.draft != old and b.model.can_undo
    b.model.undo_edit()
    assert b.model.draft == old and b.model.strength_of(0, b.seconds) == "balanced"


def test_automatic_settings_follow_the_leveling_loudness_and_echo_choices_of_the_app(w, tmp_path, monkeypatch):
    [a, b] = put(w, tmp_path, "a.wav", "b.wav")
    p = w.prefs
    s = a.settings()
    assert s.layers is None and (s.auto_leveling, s.auto_echo) == (p.loudness, False)  # (the defaults: leveling on, no echo track)
    assert s.describe(False) == f"noise reduction picked for the file, leveled ({p.loudness})"
    p.leveling = False
    s = a.settings()
    assert s.auto_leveling is None and "leveled" not in s.describe(False)
    p.leveling, p.loudness, p.echo_track = True, "youtube", True
    s = a.settings()
    assert (s.auto_leveling, s.auto_echo) == ("youtube", True)
    assert s.describe(False) == "noise reduction picked for the file, speaker echo removed, leveled (YouTube)"
    seen: dict = {}
    accept_with(monkeypatch, seen=seen)
    w.share_settings()  # (the dialog says the same words)
    assert any("leveled (YouTube)" in t and "speaker echo removed" in t for t in seen["texts"])
    assert b.auto and b.untouched_auto


PREF_NAMES = ("auto_settings", "strength", "leveling", "loudness", "noise", "echo_track", "rebuild_voice", "subtitles", "language", "video_size", "audio_format")


def prefs_now(w) -> dict:
    return {name: getattr(w.prefs, name) for name in PREF_NAMES}


def test_giving_echo_to_a_one_track_file_leaves_the_remembered_echo_choice_on(w, tmp_path, monkeypatch):
    read_each_as(w, **{"a.wav": {"tracks": 2}, "two.wav": {"tracks": 2}})
    [a, two, one] = put(w, tmp_path, "a.wav", "two.wav", "one.wav")  # (the one track file is given the settings last)
    a.set_auto(False)
    a.model.set_layer("echo", True, 0, a.seconds)
    assert w.prefs.echo_track is True and a.settings().layers["echo"] == {}
    accept_with(monkeypatch)
    w.share_settings()
    assert whole(two, "echo") == [{"start": 0.0, "end": two.seconds}] and whole(one, "echo") == []
    assert w.prefs.echo_track is True  # (a file without a second track turned it off before)
    # the other way: the choice is off, and a file with two tracks that is given echo does not turn it on
    w.prefs.echo_track = False
    two.take_settings(a.settings())
    assert whole(two, "echo") != [] and w.prefs.echo_track is False


def test_giving_settings_changes_none_of_the_choices_new_files_start_with_also_for_a_file_still_being_read(w, tmp_path, monkeypatch):
    read_each_as(w, **{"a.wav": {"tracks": 2}})
    gate = hold_reading(w, "slow.wav")
    try:
        [a, b, slow] = put(w, tmp_path, "a.wav", "b.wav", "slow.wav", wait=False)
        assert wait_until(lambda: a.model is not None and b.model is not None, 10)
        sh.ready(w)
        a.set_auto(False)
        m, n = a.model, a.seconds
        m.set_strength("gentle", 0, n)
        m.set_remover("light", 0, n)
        m.set_layer("echo", True, 0, n)
        m.set_loudness(-14.0)
        a.set_subtitles("srt")
        p = w.prefs  # (what the next file starts with is something else than what a has)
        p.strength, p.noise, p.leveling, p.loudness, p.echo_track, p.rebuild_voice = "max", "heavy", False, "broadcast", False, False
        p.language, p.video_size, p.audio_format = "fr", "720", "mp3"
        before = prefs_now(w)
        assert before["subtitles"] == "srt" and before["auto_settings"] is False
        accept_with(monkeypatch)
        w.share_settings()
        assert slow.model is None and slow._given is not None  # (it takes them when it is read)
        assert prefs_now(w) == before
        assert not b.auto and whole(b, "noise") == [{"start": 0.0, "end": b.seconds, **GENTLE_LIGHT}]
    finally:
        gate.set()
    assert wait_until(lambda: slow.model is not None, 10)
    settle(w)
    assert prefs_now(w) == before  # (not when it took them either)
    for doc in (b, slow):
        assert not doc.auto and whole(doc, "noise") == [{"start": 0.0, "end": doc.seconds, **GENTLE_LIGHT}] and doc.subtitles == "srt"
        assert whole(doc, "leveling") == [{"start": 0.0, "end": doc.seconds, "loudness_lufs": -14.0}] and whole(doc, "echo") == []  # (one track)


def test_giving_automatic_changes_none_of_the_choices_new_files_start_with(w, tmp_path):
    [a, b] = put(w, tmp_path, "a.wav", "b.wav")
    b.set_auto(False)
    b.model.set_strength("balanced", 0, b.seconds)
    before = prefs_now(w)
    assert before["auto_settings"] is False
    b.take_settings(FileSettings(None, "vtt", "de", "same"))
    assert b.auto and b.untouched_auto and prefs_now(w) == before


# ---- the window: which files take a file's settings

def test_the_files_that_can_take_the_settings_are_the_waiting_ones_that_are_read_and_not_in_line(w, tmp_path):
    gate = hold_reading(w, "slow.wav")
    try:
        [a, b, cancelled, queued, bad, slow] = put(w, tmp_path, "a.wav", "b.wav", "cancelled.wav", "queued.wav", "bad.wav", "slow.wav", wait=False)
        assert wait_until(lambda: all(d.model is not None for d in (a, b, cancelled, queued, bad)), 10)
        sh.ready(w)
        assert w.start_item(queued.item)
        cancelled.item.state = CANCELLED
        bad.error = "This file could not be read."
        assert {i.id for i in w._others(a)} == {b.item.id, cancelled.item.id, slow.item.id}  # (the one still being read can be given them too)
        assert w._others(None) == []
        assert w._waiting(a) == (4, 3)  # waiting: a, b, bad and slow (not in line yet); the cancelled one is not waiting
        assert {i.id for i in w._others(b)} == {a.item.id, cancelled.item.id, slow.item.id}
        assert w._waiting(None) == (4, 0)
    finally:
        gate.set()


# ---- the start row

def test_with_several_files_the_start_row_has_the_share_link_and_start_all_with_their_number(w, tmp_path):
    [a] = put(w, tmp_path, "a.wav")
    start = w.studio.start
    assert not start.share.isVisibleTo(w) and not start.all_button.isVisibleTo(w)  # one file: nothing to share it with, no "all"
    [b] = put(w, tmp_path, "b.wav")
    settle(w)
    assert start.share.isVisibleTo(w) and start.all_button.isVisibleTo(w)
    assert start.all_button.text() == "Start all 2" and start.share.text() == "Use these settings for other files…"
    [c] = put(w, tmp_path, "c.wav")
    settle(w)
    assert start.all_button.text() == "Start all 3" and start.button.text().startswith("Start") and start.button.isEnabled()
    assert "3 files" in start.all_button.toolTip() and "one after the other" in start.all_button.toolTip()
    # a file that is in line is not waiting any more: one file is not "all", and with nobody to give them to there is no link
    assert w.start_item(b.item) and w.start_item(c.item)
    settle(w)
    assert not start.all_button.isVisibleTo(w) and not start.share.isVisibleTo(w)
    w.open_item(b.item.id)
    settle(w)
    assert not start.share.isVisibleTo(w) and not start.all_button.isVisibleTo(w)  # (b itself is in line: nothing to give, nothing waiting but a)


def test_the_share_link_waits_for_the_file_to_be_read_and_goes_while_it_runs(w, tmp_path):
    gate = hold_reading(w, "a.wav")
    try:
        [a, b] = put(w, tmp_path, "a.wav", "b.wav", wait=False)
        assert wait_until(lambda: b.model is not None, 10)
        settle(w)
        assert w.doc is a and a.model is None and not w.studio.start.share.isVisibleTo(w)  # (the file in the editor is still being read)
        assert w.studio.start.all_button.isVisibleTo(w) and w.studio.start.all_button.text() == "Start all 2"  # (it counts the file being read)
    finally:
        gate.set()
    assert wait_until(lambda: a.model is not None, 10)
    sh.ready(w)
    settle(w)
    assert w.studio.start.share.isVisibleTo(w)
    assert w.start_item(a.item)
    settle(w)
    assert not w.studio.start.share.isVisibleTo(w)  # (a is in line)


def test_the_buttons_of_the_start_row_reach_the_window(w, tmp_path, monkeypatch):
    [a, b, c] = put(w, tmp_path, "a.wav", "b.wav", "c.wav")
    asked = []
    monkeypatch.setattr(ShareSettingsDialog, "exec", lambda self: (asked.append(sorted(self.boxes)), QDialog.DialogCode.Rejected)[1])
    w.studio.start.share.click()
    assert asked == [sorted([b.item.id, c.item.id])]  # (the others: not the file the link is in)
    assert [d.auto for d in (a, b, c)] == [True, True, True]  # (cancelled: nobody was given anything)
    w.studio.start.all_button.click()
    assert wait_until(lambda: {a.item.id, b.item.id, c.item.id} <= w._queued, 10)


def test_the_share_link_is_short_in_a_narrow_start_row_and_long_otherwise(qapp):
    from audio_cleaner_desktop.gui import start_bar

    long, short = "Use these settings for other files…", "Settings for others…"
    bar = start_bar.StartBar()
    try:
        bar.show()
        for width, text in ((start_bar.NARROW - 1, short), (start_bar.NARROW, long), (start_bar.NARROW + 300, long), (start_bar.NARROW - 100, short),
                            (start_bar.NARROW + 1, long)):
            bar.resize(width, 66)
            assert wait_until(lambda: bar.width() == width and bar.share.text() == text, 5), (width, bar.width(), bar.share.text())
        assert start_bar.NARROW == 820
        bar.set_many(True, 3)
        assert bar.share.isVisibleTo(bar) and bar.share.toolTip()  # (the tip says what it does in both)
    finally:
        bar.deleteLater()


def test_the_window_shows_the_short_link_when_the_start_row_is_narrow(w, tmp_path):
    from audio_cleaner_desktop.gui import start_bar

    [a, b] = put(w, tmp_path, "a.wav", "b.wav")
    start = w.studio.start
    for width in (1440, 900, 760, 900, 640):
        sh.resized(w, width, 2300 if width < 1000 else 900)
        settle(w)
        # the platform may not shrink the window to the width asked for, so the expected link comes from the row's measured width
        short = start.width() < start_bar.NARROW
        assert wait_until(lambda: start.share.text() == ("Settings for others…" if short else "Use these settings for other files…"), 5), (width, start.width(), start.share.text())
        assert start.share.isVisibleTo(w)
        if width == 1440:
            assert start.share.text() == "Use these settings for other files…"  # (the widest row: the long link)


# ---- giving the ticked files the settings

def test_share_settings_gives_the_ticked_files_the_settings_and_says_so(w, tmp_path, monkeypatch):
    [a, b, c, d] = put(w, tmp_path, "a.wav", "b.wav", "c.wav", "d.wav")
    a.set_auto(False)
    a.model.set_strength("gentle", 0, a.seconds)
    a.model.set_layer("leveling", False, 0, a.seconds)
    a.set_subtitles("srt")
    a.set_language("de")
    seen: dict = {}
    accept_with(monkeypatch, untick=[c.item.id], seen=seen)
    w.share_settings()
    assert sorted(seen["boxes"]) == [b.item.id, c.item.id, d.item.id] and seen["ok_at_first"] == "Use for 3 files" and seen["ok_after"] == "Use for 2 files"
    summary = " ".join(seen["texts"])
    assert "a.wav" in summary and "Gentle noise reduction" in summary and "subtitles: srt file" in summary
    for doc in (b, d):  # ticked: they have what a has
        assert not doc.auto and whole(doc, "noise") == [{"start": 0.0, "end": doc.seconds, "strength_db": 12.0, "remover": "auto"}]
        assert whole(doc, "leveling") == [] and (doc.subtitles, doc.language) == ("srt", "de")
        assert doc.out_name == f"{doc.name[0]}-cleaned.wav"  # (each keeps its own name)
    assert c.auto and c.untouched_auto and c.run_plan()[1] is None  # unticked: still Automatic
    assert (c.subtitles, c.language) == ("off", "auto")  # (and what it was added with)
    assert w.banner_files.isVisibleTo(w)
    text = w.banner_files._label.text()
    assert "2 files have the settings of a.wav now" in text and "Start all" in text and "all 4" in text and "Open one to change it on its own" in text


def test_the_banner_says_one_file_and_does_not_offer_start_all_when_only_two_files_are_there(w, tmp_path, monkeypatch):
    [a, b] = put(w, tmp_path, "a.wav", "b.wav")
    a.set_auto(False)
    a.model.set_strength("strong", 0, a.seconds)
    accept_with(monkeypatch)
    w.share_settings()
    text = w.banner_files._label.text()
    assert "1 file has the settings of a.wav now" in text and "Start all 2" not in text and "Press Start all to clean all 2" in text
    assert w.banner_files.property("kind") == "info"


def test_a_file_changed_by_hand_after_getting_the_settings_keeps_its_own_change(w, tmp_path, monkeypatch):
    [a, b, c] = put(w, tmp_path, "a.wav", "b.wav", "c.wav")
    a.set_auto(False)
    a.model.set_strength("gentle", 0, a.seconds)
    accept_with(monkeypatch)
    w.share_settings()
    assert whole(b, "noise")[0]["strength_db"] == 12.0 and whole(c, "noise")[0]["strength_db"] == 12.0
    w.open_item(b.item.id)
    sh.ready(w)
    b.model.set_strength("max", 0, b.seconds)  # b is changed on its own
    settle(w)
    assert whole(b, "noise")[0]["strength_db"] == 100.0 and whole(c, "noise")[0]["strength_db"] == 12.0 and whole(a, "noise")[0]["strength_db"] == 12.0
    w.start_all()
    assert {a.item.id, b.item.id, c.item.id} <= w._queued
    strengths = {i.path.name: i.timeline["layers"]["noise"][0]["strength_db"] for i in (a.item, b.item, c.item)}
    assert strengths == {"a.wav": 12.0, "b.wav": 100.0, "c.wav": 12.0}


def test_nothing_is_shared_when_the_dialog_is_cancelled_or_there_is_no_one_to_give_to(w, tmp_path, monkeypatch):
    [a] = put(w, tmp_path, "a.wav")
    monkeypatch.setattr(ShareSettingsDialog, "exec", lambda self: pytest.fail("there is nobody to give them to: no dialog"))
    w.share_settings()
    assert not w.banner_files.isVisibleTo(w)
    [b] = put(w, tmp_path, "b.wav")
    w.open_item(a.item.id)
    monkeypatch.setattr(ShareSettingsDialog, "exec", lambda self: QDialog.DialogCode.Rejected)
    a.set_auto(False)
    a.model.set_strength("gentle", 0, a.seconds)
    w.share_settings()
    assert b.auto and b.untouched_auto and not w.banner_files.isVisibleTo(w)


def test_the_dialog_counts_the_ticks_and_has_all_and_none(qapp):
    files = [Choice(i, f"file {i}.wav", "0:11  ·  2.0 KB") for i in (1, 2, 3)]
    dlg = ShareSettingsDialog("source.wav", "Balanced noise reduction", files, had_parts=True)
    try:
        assert dlg.chosen() == [1, 2, 3] and dlg.ok.text() == "Use for 3 files" and dlg.ok.isEnabled()
        labels = " ".join(label.text() for label in dlg.findChildren(QLabel))
        assert "source.wav" in labels and "Balanced noise reduction" in labels and "parts you changed" in labels
        dlg.boxes[2].setChecked(False)
        assert dlg.chosen() == [1, 3] and dlg.ok.text() == "Use for 2 files"
        dlg.boxes[1].setChecked(False)
        assert dlg.ok.text() == "Use for 1 file" and dlg.ok.isEnabled()
        buttons = {b.text(): b for b in dlg.findChildren(QPushButton)}
        buttons["None"].click()
        assert dlg.chosen() == [] and not dlg.ok.isEnabled() and dlg.ok.text() == "Use for the ticked files"
        buttons["All"].click()
        assert dlg.chosen() == [1, 2, 3] and dlg.ok.isEnabled() and dlg.ok.text() == "Use for 3 files"
        plain = ShareSettingsDialog("source.wav", "x", files)  # (no parts: no word about them)
        try:
            assert not any("parts you changed" in label.text() for label in plain.findChildren(QLabel))
        finally:
            plain.deleteLater()
    finally:
        dlg.deleteLater()


def test_a_finished_job_gives_the_settings_it_was_made_with(w, tmp_path, monkeypatch):
    [a, b] = put(w, tmp_path, "a.wav", "b.wav")
    folder = sh.make_folder(tmp_path)
    job, eid = sh.add_job(w.history, folder)
    w.doc_env.open_session = sh.open_session
    w.open_entry(eid)
    doc = w.doc
    assert wait_until(lambda: doc.is_job and doc.model is not None, 10)
    settle(w)
    s = doc.settings()
    assert s.layers is not None and s.layers["noise"] == {"strength_db": 24.0, "remover": "light"} and s.layers["rebuild"] is None
    assert s.layers["leveling"] == PODCAST
    assert w.studio.start.share.isVisibleTo(w)  # (a job can give its settings to the files that wait)
    accept_with(monkeypatch)
    w.share_settings()
    for doc in (a, b):
        assert not doc.auto and whole(doc, "noise") == [{"start": 0.0, "end": doc.seconds, "strength_db": 24.0, "remover": "light"}]
    assert "2 files have the settings of talk.wav now" in w.banner_files._label.text()


# ---------------------------------------------------------------------------------------------------- Start all

def release(runner, w, items) -> None:
    runner.gate.set()
    assert wait_until(lambda: all(i.state == DONE for i in items) and not w.queue.busy, 20)


def test_start_all_queues_every_waiting_file_with_its_own_options_and_timeline(w, runner, tmp_path):
    [a, b, c, d] = put(w, tmp_path, "a.wav", "b.wav", "c.wav", "d.wav")
    a.set_auto(False)
    a.model.set_strength("gentle", 0, a.seconds)  # a: its own by hand
    # b: Automatic as it is.  c: given other settings.  d: Automatic with a change by hand.
    c.take_settings(explicit(noise=STRONG_HEAVY))
    d.model.set_strength("max", 0, d.seconds)
    w.start_all()
    items = [a.item, b.item, c.item, d.item]
    assert all(i.id in w._queued and i.options is not None for i in items)
    for doc in (a, b, c, d):  # each has exactly what its own screen says it runs with
        assert (doc.item.options, doc.item.timeline) == doc.run_plan()
    assert b.item.timeline is None and b.item.options.strength == engine.AUTO_STRENGTH and b.item.options.auto_rebuild
    assert a.item.timeline["layers"]["noise"][0]["strength_db"] == 12.0
    assert c.item.timeline["layers"]["noise"][0] == {"start": 0.0, "end": c.seconds, "strength_db": 40.0, "remover": "heavy"} and "leveling" not in c.item.timeline["layers"]
    assert d.item.timeline["layers"]["noise"][0]["strength_db"] == 100.0
    assert all(doc.stage == "processing" for doc in (a, b, c, d))
    settle(w)
    assert not w.studio.start.all_button.isVisibleTo(w)  # (nothing waits any more)
    release(runner, w, items)
    got = {call["output_name"]: call for call in runner.calls}
    assert set(got) == {"a-cleaned.wav", "b-cleaned.wav", "c-cleaned.wav", "d-cleaned.wav"} and len(runner.calls) == 4
    assert got["b-cleaned.wav"]["timeline"] is None and got["b-cleaned.wav"]["options"].strength == engine.AUTO_STRENGTH
    assert got["c-cleaned.wav"]["timeline"]["layers"]["noise"][0]["remover"] == "heavy"
    assert got["a-cleaned.wav"]["timeline"]["layers"]["noise"][0]["strength_db"] == 12.0
    assert not w.banner_files.isVisibleTo(w)  # (nothing was held back)


def test_start_all_leaves_out_what_is_running_in_line_or_finished_and_goes_in_the_order_of_the_list(w, runner, tmp_path):
    [a, b, c] = put(w, tmp_path, "a.wav", "b.wav", "c.wav")
    assert w.start_item(b.item)
    assert wait_until(lambda: b.item.state == RUNNING, 10)
    keep = (b.item.options, b.item.timeline, b.item.keep_dir)
    w.start_all()
    assert {a.item.id, c.item.id} <= w._queued and (b.item.options, b.item.timeline, b.item.keep_dir) == keep  # b was not queued a second time
    assert w.queue.pending_ids() == [a.item.id, c.item.id]
    release(runner, w, [a.item, b.item, c.item])
    assert [call["output_name"] for call in runner.calls] == ["b-cleaned.wav", "a-cleaned.wav", "c-cleaned.wav"]


def test_start_all_starts_a_file_that_is_still_being_read_once_it_is_read(w, runner, tmp_path):
    gate = hold_reading(w, "slow.wav")
    try:
        [a, b, slow] = put(w, tmp_path, "a.wav", "b.wav", "slow.wav", wait=False)
        assert wait_until(lambda: a.model is not None and b.model is not None, 10)
        sh.ready(w)
        slow.take_settings(explicit(noise=STRONG_HEAVY, leveling=PODCAST))  # (given before it was read)
        w.start_all()
        assert {a.item.id, b.item.id} <= w._queued
        assert slow.item.start_when_read and slow.item.id not in w._queued and slow.model is None and slow.item.state == WAITING
        assert not w.banner_files.isVisibleTo(w)
    finally:
        gate.set()
    assert wait_until(lambda: slow.item.id in w._queued, 10)
    assert not slow.item.start_when_read and slow.item.state in (WAITING, RUNNING)
    assert slow.item.timeline["layers"]["noise"][0]["strength_db"] == 40.0 and (slow.item.options, slow.item.timeline) == slow.run_plan()  # its own, given and read
    assert w.queue.pending_ids() == [b.item.id, slow.item.id]
    release(runner, w, [a.item, b.item, slow.item])
    assert not w.banner_files.isVisibleTo(w)


def test_a_file_that_cannot_start_is_not_started_and_is_named_with_the_first_reason(w, runner, tmp_path, monkeypatch):
    monkeypatch.setattr(resources, "speech_ready", lambda: False)  # (no speech model: subtitles cannot be made)
    [a, b, c] = put(w, tmp_path, "a.wav", "b.wav", "c.wav")
    b.set_subtitles("srt")
    w.start_all()
    assert {a.item.id, c.item.id} <= w._queued and b.item.id not in w._queued and b.item.state == WAITING and b.stage == "ready"
    assert w.banner_files.isVisibleTo(w) and w.banner_files.property("kind") == "warn"
    text = w.banner_files._label.text()
    assert text.startswith("b.wav was not started: Subtitles need the speech model") and text.endswith("Open it to see more.")
    # the file shows why when it is opened, and can go once it is fixed
    w.open_item(b.item.id)
    settle(w)
    assert not w.studio.start.button.isEnabled() and any("speech model" in t for t, _ in w.studio.note_lines(b))
    b.set_subtitles("off")
    settle(w)
    assert w.studio.start.button.isEnabled()
    release(runner, w, [a.item, c.item])
    assert b.item.state == WAITING  # (it was left alone while the others ran)


def test_several_files_that_cannot_start_are_counted_and_the_first_is_named(w, tmp_path, monkeypatch):
    monkeypatch.setattr(resources, "speech_ready", lambda: False)
    [a, b, c, d] = put(w, tmp_path, "a.wav", "b.wav", "c.wav", "d.wav")
    c.set_subtitles("vtt")
    d.set_subtitles("txt")
    w.start_all()
    assert {a.item.id, b.item.id} <= w._queued and c.item.id not in w._queued and d.item.id not in w._queued
    text = w.banner_files._label.text()
    assert text.startswith("2 files were not started. c.wav: Subtitles need the speech model") and text.endswith("Open them to see more.")


def test_a_file_read_after_start_all_that_cannot_start_is_named_when_it_is_read(w, tmp_path, monkeypatch):
    monkeypatch.setattr(resources, "speech_ready", lambda: False)
    gate = hold_reading(w, "slow.wav")
    try:
        [a, slow] = put(w, tmp_path, "a.wav", "slow.wav", wait=False)
        assert wait_until(lambda: a.model is not None, 10)
        sh.ready(w)
        slow.take_settings(FileSettings(layers(noise=GENTLE_LIGHT), "srt", "auto", "same"))
        w.start_all()
        assert a.item.id in w._queued and slow.item.start_when_read and not w.banner_files.isVisibleTo(w)
    finally:
        gate.set()
    assert wait_until(lambda: w.banner_files.isVisibleTo(w), 10)
    assert w.banner_files._label.text().startswith("slow.wav was not started: Subtitles need the speech model")
    assert slow.item.id not in w._queued and not slow.item.start_when_read and slow.item.state == WAITING


def test_a_file_that_stopped_waiting_before_it_was_read_is_not_started_when_it_is_read(w, tmp_path):
    gate = hold_reading(w, "slow.wav")
    try:
        [a, slow] = put(w, tmp_path, "a.wav", "slow.wav", wait=False)
        assert wait_until(lambda: a.model is not None, 10)
        w.start_all()
        assert slow.item.start_when_read
        slow.item.state = CANCELLED  # (taken out of the list of what waits in the meantime)
    finally:
        gate.set()
    assert wait_until(lambda: slow.model is not None, 10)
    settle(w)
    assert slow.item.id not in w._queued and not slow.item.start_when_read


def test_start_all_with_nothing_waiting_does_nothing(w, tmp_path):
    [a] = put(w, tmp_path, "a.wav")
    assert w.start_item(a.item)
    w.start_all()
    assert w.queue.pending_ids() == [] and not w.banner_files.isVisibleTo(w)


# ---------------------------------------------------------------------------------------------------- screenshots (for a look; not collected without ACW_SHOTS_DIR)

SHOTS = os.environ.get("ACW_SHOTS_DIR")


@pytest.mark.skipif(not SHOTS, reason="set ACW_SHOTS_DIR to take the screenshots")
def test_screenshots_of_the_start_row_and_the_share_dialog(w, runner, tmp_path, monkeypatch, qapp):
    """ACW_SHOTS_ONLY="4,8" takes only the screenshots whose names start with those numbers."""
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QScrollArea

    out = Path(SHOTS)
    out.mkdir(parents=True, exist_ok=True)
    only = {x.strip() for x in os.environ.get("ACW_SHOTS_ONLY", "").split(",") if x.strip()}
    names = ["interview with a long name for the guest.wav", "second talk.wav", "third talk.mp3", "lecture part 1.wav", "lecture part 2.wav", "podcast ep 12.wav"]
    docs = put(w, tmp_path, *names)
    docs[0].set_auto(False)
    docs[0].model.set_strength("strong", 0, docs[0].seconds)
    w.open_item(docs[0].item.id)
    settle(w)
    report: list[str] = []
    start = w.studio.start

    def wanted(tag: str) -> bool:
        return not only or tag.split("-")[0] in only

    def look(tag: str) -> None:
        QApplication.processEvents()
        if not wanted(tag):
            return

        def bars_now():
            return [(sa.horizontalScrollBar().isVisible(), sa.horizontalScrollBar().maximum()) for sa in w.findChildren(QScrollArea)]

        right_after = bars_now()
        wait_until(lambda: not any(shown for shown, _ in bars_now()), 3)  # (the page takes a moment to follow a big change of width)
        w.grab().save(str(out / f"{tag}-window.png"))
        start.grab().save(str(out / f"{tag}-start-row.png"))
        parts = {"share": start.share, "all": start.all_button, "details": start.details, "files": start.files, "start": start.button}
        report.append(f"{tag}: window {w.width()}x{w.height()}, start row {start.width()}x{start.height()} (needs at least {start.minimumSizeHint().width()}), "
                      f"horizontal scrollbars (visible, overflow px) right after the resize {right_after}, settled {bars_now()}; " + ", ".join(
                          f"{k} '{getattr(v, 'text', lambda: '')()}' x={v.mapTo(start, v.rect().topLeft()).x()} w={v.width()} (wants {v.sizeHint().width()})"
                          for k, v in parts.items() if v.isVisibleTo(start)))

    # three files waiting: the start row with the link, "Start all 3" and Start; at the normal size and narrow (a narrow window stacks the page, so
    # it is also taken tall enough to see the row in its place)
    for doc in docs[3:]:
        w.remove_item(doc.item.id)
    settle(w)
    for tag, (width, height) in {"1-three-files-1440": (1440, 900), "2-three-files-900": (900, 2300), "3-three-files-760": (760, 2300), "4-three-files-640": (640, 2300)}.items():
        sh.resized(w, width, height, until=lambda: start.all_button.isVisibleTo(w))
        settle(w)
        look(tag)
    sh.resized(w, 1440, 900)
    # a finished file with the others waiting, at the narrowest window
    runner.gate.set()
    assert w.start_item(docs[0].item)
    assert wait_until(lambda: docs[0].item.state == DONE, 20)
    sh.resized(w, 640, 2300)
    w.open_item(docs[0].item.id)
    settle(w)
    look("8-finished-file-others-waiting-640")
    sh.resized(w, 1440, 900)
    # the dialog with five files waiting (light, and dark with one file unticked)
    assert w.add_paths([tmp_path / n for n in names[3:]]) == 3
    assert wait_until(lambda: len(w._items) == 6 and all(i.doc.model is not None for i in w._items.values()), 10)
    w.open_item(docs[1].item.id)
    settle(w)

    def grab_dialog(tag: str, untick: int | None = None):
        def exec_(self):
            if untick is not None:
                self.boxes[sorted(self.boxes)[untick]].setChecked(False)
            self.show()
            QApplication.processEvents()
            if wanted(tag):
                self.grab().save(str(out / f"{tag}.png"))
                report.append(f"{tag}: dialog {self.width()}x{self.height()}, ok button {self.ok.width()} wide (wants {self.ok.sizeHint().width()}), "
                              f"{len(self.boxes)} files, ok has the focus: {self.ok.hasFocus()}")
            return QDialog.DialogCode.Rejected

        monkeypatch.setattr(ShareSettingsDialog, "exec", exec_)

    grab_dialog("5-share-dialog-five-files")
    w.share_settings()
    old = qapp.palette()
    dark = QPalette()
    for role, color in ((QPalette.ColorRole.Window, "#121826"), (QPalette.ColorRole.WindowText, "#e7ecf5"), (QPalette.ColorRole.Base, "#151b26"),
                        (QPalette.ColorRole.AlternateBase, "#1c2434"), (QPalette.ColorRole.Text, "#e7ecf5"), (QPalette.ColorRole.Button, "#1c2434"),
                        (QPalette.ColorRole.ButtonText, "#e7ecf5"), (QPalette.ColorRole.ToolTipBase, "#1c2434"), (QPalette.ColorRole.ToolTipText, "#e7ecf5"),
                        (QPalette.ColorRole.PlaceholderText, "#a0abc0")):
        dark.setColor(role, QColor(color))
    try:
        qapp.setPalette(dark)
        for _ in range(5):
            QApplication.processEvents()
        grab_dialog("6-share-dialog-five-files-dark-one-unticked", untick=1)
        w.share_settings()
        look("7-dark-1440")
    finally:
        qapp.setPalette(old)
        for _ in range(5):
            QApplication.processEvents()
    (out / "report.txt").write_text("\n".join(report) + "\n")
