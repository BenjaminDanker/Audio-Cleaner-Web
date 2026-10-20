"""What the per-part editor shows and changes (editor_model.py), the live sound it plays (live_mix.py: per-part strengths, the Hear switches,
equal-power edges of the rebuilt voice, Leveling heard by loudness; played through a stand-in sound card), the strip under the player and
its mouse, and the panel's three steps. No model runs: the layers of the live mix are laid out by hand."""
from __future__ import annotations

import math

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest

from acw_common import clarity, pipeline
from acw_common.timeline import Timeline
from audio_cleaner_desktop import editor_job, editor_model, editor_session, live_mix
from audio_cleaner_desktop.editor_model import EditModel, Suggestion
from audio_cleaner_desktop.gui import part_editor, wave_strip

from fake_media import FakeSink, namespace, sink_factory

SR = 48_000


def tl_of(seconds=60.0, **layers):
    t = Timeline(seconds)
    for name, parts in layers.items():
        for p in parts:
            t.paint(name, p[0], p[1], p[2] if len(p) > 2 else None)
    return t


def model_of(seconds=60.0, **kw):
    done = Timeline.whole_file(seconds, noise={"strength_db": 24.0, "remover": "light"}, leveling=False)
    return EditModel(seconds, done, done, **kw)


# ---------------------------------------------------------------------------------------------------- the model

def test_a_strength_for_a_part_keeps_the_remover_it_has_and_off_takes_the_noise_removal_out_of_it():
    m = model_of()
    m.draft.paint("noise", 10, 20, {"strength_db": 24.0, "remover": "heavy"})
    m.select(5, 25)
    m.set_strength("strong")
    assert [(p.start, p.end, p.settings["strength_db"], p.settings["remover"]) for p in m.draft.layers["noise"]] == [
        (0.0, 5.0, 24.0, "light"), (5.0, 10.0, 40.0, "light"), (10.0, 20.0, 40.0, "heavy"), (20.0, 25.0, 40.0, "light"), (25.0, 60.0, 24.0, "light")]
    assert m.strength_of(5, 25) == "strong" and m.strength_of(0, 12) == "mixed" and m.remover_of(5, 25) == "mixed"
    m.set_strength("off", 30, 40)
    assert m.strength_of(30, 40) == "off" and m.remover_of(30, 40) is None and m.switch_of("noise", 28, 42) is None
    m.set_strength("gentle", 28, 42)  # new parts where there were none are automatic
    assert m.remover_of(30, 40) == "auto" and m.strength_of(28, 42) == "gentle"
    m.set_remover("heavy", 28, 42)
    assert m.remover_of(28, 42) == "heavy"
    with pytest.raises(ValueError):
        m.clear_selection()
        m.set_strength("max")  # nothing selected


def test_voice_rebuild_wins_where_both_are_on_and_the_model_says_so():
    m = model_of()
    assert not m.rebuild_wins(10, 20)
    m.set_layer("rebuild", True, 10, 20)
    assert m.rebuild_wins(5, 15) and m.rebuild_wins(10, 20) and not m.rebuild_wins(30, 40)
    assert m.switch_of("rebuild", 10, 20) is True and m.switch_of("rebuild", 5, 15) is None and m.switch_of("rebuild", 30, 40) is False
    m.set_layer("rebuild", False, 12, 14)
    assert [(p.start, p.end) for p in m.draft.layers["rebuild"]] == [(10.0, 12.0), (14.0, 20.0)]


def test_leveling_has_one_loudness_for_the_whole_job_and_echo_removal_needs_the_second_track():
    m = model_of()
    assert m.loudness() == -16.0 and m.switch_of("leveling", 0, 60) is False
    m.set_layer("leveling", True, 0, 30)
    m.set_layer("leveling", True, 40, 50)
    m.set_loudness(-14.0)
    assert {p.settings["loudness_lufs"] for p in m.draft.layers["leveling"]} == {-14.0} and m.loudness() == -14.0
    m.set_loudness(None)  # keep the level
    assert m.draft.layers["leveling"][0].settings["loudness_lufs"] is None
    m.set_layer("echo", True, 0, 10)
    assert not m.draft.layers["echo"]  # (no second track)
    m2 = model_of(echo_available=True)
    m2.set_layer("echo", True, 0, 10)
    assert [(p.start, p.end) for p in m2.draft.layers["echo"]] == [(0.0, 10.0)]


def test_the_selection_has_handles_steppers_the_playhead_and_a_minimum_length():
    m = model_of()
    m.select(30, 20)
    assert m.selection == (20.0, 30.0)
    m.nudge("s", -1)
    m.nudge("e", 2)
    assert m.selection == (19.0, 32.0)
    m.use_playhead("s", 25.0)
    m.use_playhead("e", 26.5)
    assert m.selection == (25.0, 26.5)
    m.use_playhead("e", 25.2)  # never shorter than a second
    assert m.selection == (25.0, 26.0)
    m.around(2.0)
    assert m.selection[0] == 0.0 and m.selection[1] == 7.0
    m.whole()
    assert m.selection == (0.0, 60.0)
    seen = []
    m.on_change(lambda: seen.append(1))
    m.select(1, 9)
    assert seen


def test_undo_changes_puts_a_part_back_as_the_saved_result_has_it_and_the_draft_knows_when_it_differs():
    m = model_of()
    assert not m.dirty
    m.set_strength("max", 10, 20)
    m.set_layer("rebuild", True, 15, 25)
    assert m.dirty and [r["pending"] for r in m.runs("noise")] == [False, True, False]
    m.undo_changes(0, 60)
    assert not m.dirty and m.draft == m.done


def sugg(**kw):
    base = {"id": "s0", "start": 41.0, "end": 44.0, "kind": "clipping", "rebuild": True, "noise": None, "reasons": ["sounds distorted"]}
    return {**base, **kw}


def test_suggestions_are_applied_undone_dismissed_and_shown_with_words():
    m = model_of()
    m.set_suggestions([sugg(), sugg(id="s1", start=10.0, end=20.0, kind="part", rebuild=False,
                                    noise={"strength_db": 40.0, "remover": "heavy"}, reasons=["is noisier"])])
    a, b = m.suggestions
    assert (a.what, a.ask, a.action) == ("sounds distorted", "Rebuild just this part?", "Rebuild this part")
    assert (b.ask, b.action) == ("Strong here?", "Use Strong here") and len(m.open_suggestions()) == 2
    m.apply("s0")
    assert a.state == "applied" and m.switch_of("rebuild", 41, 44) is True and m.dirty
    m.apply("s1")
    assert m.strength_of(10, 20) == "strong" and m.remover_of(10, 20) == "heavy"
    m.set_strength("max", 15, 18)  # other edits inside the part are undone with it
    m.undo("s1")
    assert b.state == "open" and m.strength_of(10, 20) == "balanced" and m.remover_of(10, 20) == "light"
    m.undo("s0")
    assert a.state == "open" and not m.draft.layers["rebuild"]
    m.dismiss("s0")
    assert a.state == "dismissed" and [s.id for s in m.open_suggestions()] == ["s1"]
    m.show("s1")
    assert m.selection == (10.0, 20.0) and m.focus == "s1"
    # they come and go as JSON (the job keeps them), with their states
    again = EditModel(60.0, m.draft, m.done)
    again.set_suggestions(m.suggestions_json())
    assert [(s.id, s.state) for s in again.suggestions] == [("s0", "dismissed"), ("s1", "open")]
    assert Suggestion.from_json(m.suggestions_json()[1]).noise == {"strength_db": 40.0, "remover": "heavy"}


def test_applying_a_part_suggestion_that_does_not_rebuild_takes_the_rebuild_off_there():
    m = model_of()
    m.set_layer("rebuild", True, 0, 60)
    m.set_suggestions([sugg(id="p", start=10.0, end=20.0, kind="part", rebuild=False, noise={"strength_db": 12.0, "remover": "light"}, reasons=[])])
    m.apply("p")
    assert m.switch_of("rebuild", 10, 20) is False and m.switch_of("rebuild", 0, 10) is True and m.strength_of(10, 20) == "gentle"
    assert m.suggestion("p").what == "may sound better with other settings"


def test_plan_mode_chooses_parts_and_only_those_are_processed():
    base = Timeline.whole_file(100.0, noise={"strength_db": 100.0, "remover": "auto"}, leveling={"loudness_lufs": -16.0})
    m = EditModel(100.0, base, None, plan_mode=True)
    assert m.effective() == base and m.processed_seconds() == 100.0
    m.select(10, 30)
    m.choose()
    m.select(25, 40)
    m.choose()
    m.select(70, 80)
    m.choose()
    assert m.chosen == [(10.0, 40.0), (70.0, 80.0)] and m.only
    eff = m.effective()
    assert [(p.start, p.end) for p in eff.layers["noise"]] == [(10.0, 40.0), (70.0, 80.0)] and [(p.start, p.end) for p in eff.layers["leveling"]] == [(10.0, 40.0), (70.0, 80.0)]
    assert m.processed_seconds() == 40.0
    m.set_only(False)
    assert m.effective() == base
    m.set_only(True)
    m.unchoose(0)
    m.unchoose(0)
    assert not m.only and m.chosen == []
    # the settings of a part go with it
    m.select(10, 20)
    m.set_strength("gentle")
    m.choose()
    assert editor_session.only_parts(m.draft, [(10, 20)]).layers["noise"][0].settings["strength_db"] == 12.0


def test_the_bar_says_what_will_happen_in_words():
    m = model_of()
    quote = lambda **k: editor_job.Quote(**{"seconds": 40.0, "steps": {"rebuild": 2}, "misses": 2, **k})  # noqa: E731
    assert part_editor.bar_text(m, None)[0] == "No changes yet" and part_editor.bar_text(m, None)[2] is False
    m.set_strength("max", 10, 20)
    assert part_editor.bar_text(m, editor_job.Quote(misses=0))[:1] == ("Changes: nothing to process",)
    head, detail, can = part_editor.bar_text(m, quote())
    assert head == "Process changes · about 40 s" and can and "Voice rebuild" in detail
    plan = EditModel(100.0, Timeline.whole_file(100.0), None, plan_mode=True)
    assert part_editor.bar_text(plan, quote(seconds=180.0))[0] == "Whole file · about 3 min"
    plan.set_only(True)
    assert part_editor.bar_text(plan, None)[0] == "Nothing will be processed"
    plan.select(10, 20)
    plan.choose()
    head, detail, can = part_editor.bar_text(plan, quote())
    assert head == "Chosen parts · about 40 s" and "10 s of 1:40" in detail and can


# ---------------------------------------------------------------------------------------------------- the live sound

def tone(n, level, seed):
    rng = np.random.default_rng(seed)
    return (level * rng.standard_normal((n, 1))).astype(np.float32)


@pytest.fixture
def layers():
    """Ten seconds, one channel: the recording, the light remover's full-strength sound (a quarter of it), the heavy one's (a tenth) and a
    rebuilt voice (louder and different), each laid out as if blocks had been read for them."""
    n = 10 * SR
    lay = live_mix.Layers(tone(n, 0.1, 1), [0])
    for step, gain in (("light", 0.25), ("heavy", 0.1)):
        lay.array(step)[:] = lay.src * gain
        lay.mark(step, 0, n)
    lay.array("rebuild")[:] = tone(n, 0.2, 9)
    lay.mark("rebuild", 0, n)
    lay.array("echo")[:] = lay.src * 0.5
    lay.mark("echo", 0, n)
    return lay


def whole(noise=100.0, remover="light", level=False, **kw):
    return Timeline.whole_file(10.0, noise={"strength_db": noise, "remover": remover} if noise else False, leveling=level, **kw)


def test_each_part_is_heard_at_its_own_strength_and_with_its_own_remover(layers):
    tl = whole(100.0)
    tl.paint("noise", 4.0, 6.0, {"strength_db": 12.0, "remover": "heavy"})
    mix = live_mix.LiveMix(layers, tl)
    src, full = layers.src, layers.arrays["light"]
    a = SR  # 1 s: full strength light remover
    assert np.allclose(mix.read(a, 100), full[a:a + 100], atol=1e-6)
    share = pipeline.strength_mix(12.0)
    b = 5 * SR  # 5 s: a blend of the heavy remover's sound with the recording at 12 dB
    expected = (1 - share) * layers.arrays["heavy"][b:b + 100] + share * src[b:b + 100]
    assert np.allclose(mix.read(b, 100), expected, atol=1e-6)
    # the join is a ramp of 100 ms across the boundary, halfway at the boundary itself
    j = 4 * SR
    left, right = full[j], (1 - share) * layers.arrays["heavy"][j] + share * src[j]
    assert mix.read(j, 1)[0, 0] == pytest.approx(0.5 * left[0] + 0.5 * right[0], abs=1e-6)
    assert np.allclose(mix.read(j - 4000, 5), full[j - 4000:j - 3995], atol=1e-6)  # 83 ms before: outside the ramp
    tl.erase("noise", 4.0, 6.0)  # Off: the recording as it is
    mix.set_timeline(tl)
    assert np.allclose(mix.read(b, 100), src[b:b + 100], atol=1e-6)


def test_the_hear_switches_take_a_layer_out_of_the_sound_without_touching_the_timeline(layers):
    tl = whole(100.0)
    tl.paint("rebuild", 6.0, 8.0)
    tl.paint("echo", 0.0, 2.0)
    mix = live_mix.LiveMix(layers, tl)
    src, full, reb, ech = layers.src, layers.arrays["light"], layers.arrays["rebuild"], layers.arrays["echo"]
    assert mix.hear == {"echo": True, "noise": True, "rebuild": True, "leveling": True}
    at = lambda s: mix.read(int(s * SR), 50)  # noqa: E731
    assert np.allclose(at(7.0), reb[7 * SR:7 * SR + 50], atol=1e-6)  # the rebuilt voice, where it is on
    assert np.allclose(at(1.0), full[SR:SR + 50], atol=1e-6)  # (the models' blocks are laid out as they were made: the noise layer is theirs)
    mix.set_hear("noise", False)
    assert np.allclose(at(1.0), ech[SR:SR + 50], atol=1e-6) and np.allclose(at(4.0), src[4 * SR:4 * SR + 50], atol=1e-6)
    mix.set_hear("echo", False)
    assert np.allclose(at(1.0), src[SR:SR + 50], atol=1e-6)
    mix.set_hear("rebuild", False)
    assert np.allclose(at(7.0), src[7 * SR:7 * SR + 50], atol=1e-6)
    assert mix.timeline == tl and [p.start for p in tl.layers["rebuild"]] == [6.0]
    mix.set_hear("noise", True)
    assert np.allclose(at(4.0), full[4 * SR:4 * SR + 50], atol=1e-6)
    with pytest.raises(ValueError):
        mix.set_hear("music", True)


def test_voice_rebuild_has_equal_power_edges_of_a_quarter_second_and_wins_over_the_noise_removal(layers):
    tl = whole(100.0)
    tl.paint("rebuild", 4.0, 6.0)
    mix = live_mix.LiveMix(layers, tl)
    reb, noise = layers.arrays["rebuild"], layers.arrays["light"]
    e = 4 * SR
    mid = mix.read(e, 1)[0, 0]
    assert mid == pytest.approx(math.cos(math.pi / 4) * noise[e, 0] + math.sin(math.pi / 4) * reb[e, 0], abs=1e-6)
    assert mix.read(e - 7000, 1)[0, 0] == pytest.approx(noise[e - 7000, 0], abs=1e-6)  # 146 ms before: outside the edge
    assert mix.read(e + 7000, 1)[0, 0] == pytest.approx(reb[e + 7000, 0], abs=1e-6)  # (the noise removal under it is gone)
    # the power of two uncorrelated signals across the edge stays level (cos^2 + sin^2 = 1)
    w = np.arange(e - 6000, e + 6000)
    th = np.clip((w - e) / (0.25 * SR) + 0.5, 0, 1) * (math.pi / 2)
    assert np.allclose(np.cos(th) ** 2 + np.sin(th) ** 2, 1.0)
    assert mix.missing("rebuild") == []


def test_a_part_without_its_blocks_is_not_heard_until_it_is_processed_and_says_so(layers):
    layers.have["heavy"][:] = False
    layers.have["heavy"][:5 * (SR // live_mix.CELL)] = True
    tl = whole(100.0, "light")
    tl.paint("noise", 7.0, 9.0, {"strength_db": 100.0, "remover": "heavy"})
    mix = live_mix.LiveMix(layers, tl)
    assert mix.missing("noise") == [(7.0, 9.0)]
    layers.have["heavy"][:] = False
    mix.set_timeline(tl)
    # no heavy blocks: the light remover's sound stands in until they are made
    assert np.allclose(mix.read(8 * SR, 20), layers.arrays["light"][8 * SR:8 * SR + 20], atol=1e-6)
    layers.have["light"][:] = False
    assert np.allclose(mix.read(8 * SR, 20), layers.src[8 * SR:8 * SR + 20], atol=1e-6)  # nothing: the recording


def test_leveling_is_heard_by_loudness_and_matches_the_target_in_each_leveled_part(layers):
    tl = whole(100.0, level=False)
    tl.paint("leveling", 0.0, 4.0, {"loudness_lufs": -28.0})
    tl.paint("leveling", 6.0, 10.0, {"loudness_lufs": -24.0})
    mix = live_mix.LiveMix(layers, tl)
    plain = np.asarray(layers.arrays["light"])  # what is heard without Leveling
    for a, b, target in ((0, 4, -28.0), (6, 10, -24.0)):
        heard = mix.read(a * SR + SR // 2, (b - a) * SR - SR)
        lufs = clarity.integrated_loudness(heard, SR)
        assert lufs == pytest.approx(target, abs=1.0)
    assert np.allclose(mix.read(5 * SR, 100), plain[5 * SR:5 * SR + 100], atol=1e-6)  # the part between is not leveled
    gains = mix.level_gains_db()
    assert [round(a) for a, _, _ in gains] == [0, 6] and all(abs(g) <= 12.0 for *_, g in gains)
    mix.set_hear("leveling", False)
    assert mix.level_gains_db() == [] and np.allclose(mix.read(SR, 100), plain[SR:SR + 100], atol=1e-6)


def test_the_live_mix_is_what_the_player_reads_and_the_original_plays_the_recording(layers, qapp):
    tl = whole(100.0)
    mix = live_mix.LiveMix(layers, tl)
    assert (mix.frames, mix.channels, mix.seconds) == (10 * SR, 1, 10.0)
    assert len(mix.read(10 * SR - 10, 100)) == 10 and len(mix.read(10 * SR, 5)) == 0
    pcm = np.frombuffer(mix.read_pcm16(0, 1000), dtype="<i2")
    assert pcm.shape == (1000,) and np.abs(pcm).max() <= 32767
    mix.set_original(True)
    assert mix.original and np.array_equal(mix.read(100, 50), layers.src[100:150])
    mix.set_strength(24.0)  # (the player's slider call: every part has its own strength here)
    mix.exact = True
    from audio_cleaner_desktop.gui import live_audio

    FakeSink.made.clear()
    player = live_audio.LivePlayer(namespace(), sink_factory=sink_factory)
    player.load(mix)
    mix.set_original(False)
    player.play()
    sink = FakeSink.made[-1]
    first = sink.pull(2400)
    tl2 = whole(100.0)
    tl2.paint("noise", 0.0, 10.0, {"strength_db": 12.0, "remover": "light"})
    mix.set_timeline(tl2)  # the very next piece of sound follows the timeline
    second = sink.pull(2400)
    a = np.frombuffer(first, dtype="<i2").astype(np.float64) / 32767
    b = np.frombuffer(second, dtype="<i2").astype(np.float64) / 32767
    assert np.allclose(a, mix.read(0, 2400)[:, 0] * 0 + np.clip(layers.arrays["light"][:2400, 0], -1, 1), atol=2e-4)
    share = pipeline.strength_mix(12.0)
    assert np.allclose(b, (1 - share) * layers.arrays["light"][2400:4800, 0] + share * layers.src[2400:4800, 0], atol=2e-4)
    player.stop()


# ---------------------------------------------------------------------------------------------------- the strip and the panel

@pytest.fixture
def strip(qapp):
    m = model_of(60.0)
    m.set_suggestions([sugg()])
    s = wave_strip.WaveStrip(m, [10, 50, 90, 40] * 100)
    s.resize(600, s.height())
    s.show()
    return s


def press(strip_, x, y, release_at=None):
    QTest.mousePress(strip_, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(int(x), int(y)))
    return strip_


def test_a_click_on_the_wave_moves_the_playhead_and_a_drag_chooses_a_part(strip):
    seen, picked = [], []
    strip.seekRequested.connect(seen.append)
    strip.selectionDragged.connect(lambda a, b: (picked.append((a, b)), strip.model.select(a, b)))
    y = wave_strip.RULER_H + 20
    QTest.mouseClick(strip, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(300, y))
    assert seen and seen[-1] == pytest.approx(strip.time_at(300), abs=0.01) and not picked
    press(strip, 100, y)
    QTest.mouseMove(strip, QPoint(104, y))  # less than a few pixels is still a click
    QTest.mouseRelease(strip, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(104, y))
    assert not picked
    press(strip, 100, y)
    QTest.mouseMove(strip, QPoint(160, y))
    QTest.mouseMove(strip, QPoint(220, y))
    QTest.mouseRelease(strip, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(220, y))
    a, b = strip.model.selection
    assert a == pytest.approx(strip.time_at(100), abs=0.5) and b == pytest.approx(strip.time_at(220), abs=0.5) and len(picked) >= 2
    # a handle is grabbed and dragged
    press(strip, strip.x_of(b), y)
    QTest.mouseMove(strip, QPoint(int(strip.x_of(b)) + 80, y))
    QTest.mouseRelease(strip, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(int(strip.x_of(b)) + 80, y))
    assert strip.model.selection[0] == a and strip.model.selection[1] > b + 5


def test_a_pin_in_the_ruler_selects_its_suggestion_and_the_rows_show_every_layer(strip):
    pins = []
    strip.pinClicked.connect(pins.append)
    x = strip.x_of(42.5)
    QTest.mouseClick(strip, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(int(x), 8))
    assert pins == ["s0"] and strip.pin_at(x, 8) == "s0" and strip.pin_at(x, 40) is None
    small = strip.height()
    strip.set_rows(True)
    assert strip.height() > small and strip.row_at(wave_strip.RULER_H + wave_strip.WAVE_H + 5) == "noise"
    assert strip.row_at(10) is None and strip.row_at(wave_strip.RULER_H + wave_strip.WAVE_H + 3 + 3 * wave_strip.ROW_H + 5) is None
    assert strip.row_at(wave_strip.RULER_H + wave_strip.WAVE_H + 3 + 2 * wave_strip.ROW_H + 5) == "leveling"  # (no echo row)
    seen = []
    strip.selectionDragged.connect(lambda a, b: seen.append((a, b)))
    y = wave_strip.RULER_H + wave_strip.WAVE_H + 12
    press(strip, 100, y)
    QTest.mouseMove(strip, QPoint(260, y))
    QTest.mouseRelease(strip, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(260, y))
    assert seen  # dragging across a row chooses a part
    strip.grab()  # paints without trouble in both views
    strip.set_rows(False)
    strip.grab()


@pytest.fixture
def editor(qapp):
    m = model_of(60.0, echo_available=False)
    m.set_suggestions([sugg()])
    e = part_editor.PartEditor(m, [20, 60, 100, 40] * 100)
    e.resize(1000, 700)
    e.show()
    return e


def test_the_panel_goes_moment_part_change_and_edits_the_selected_part(editor, qapp):
    m = editor.model
    assert editor.view == "home" and not editor._chips["echo"].isVisible() and editor._chips["noise"].isVisible()
    editor.set_playhead(20.0)
    editor.goto("range")  # (no part yet: around the playhead)
    assert editor.view == "range" and m.selection == (15.0, 25.0)
    assert editor._time_labels["s"].text() == "0:15" and editor._time_labels["e"].text() == "0:25"
    editor.goto("change")
    assert editor._strength.value() == "balanced" and editor._remover.currentData() == "light" and not editor._wins.isVisible()
    editor._strength.set_value("strong")
    assert m.strength_of(15, 25) == "strong" and m.strength_of(0, 60) == "mixed"
    editor._remover.setCurrentIndex(editor._remover.findData("heavy"))
    editor._remover.activated.emit(editor._remover.currentIndex())
    assert m.remover_of(15, 25) == "heavy"
    editor._rebuild.setChecked(True)
    assert m.switch_of("rebuild", 15, 25) is True and editor._wins.isVisible()  # "Voice rebuild wins where both are on"
    assert "Rebuild the voice wins" in editor._wins.text()
    editor._leveling.setChecked(True)
    assert m.switch_of("leveling", 15, 25) is True
    editor._loudness.setCurrentIndex(editor._loudness.findData(-14.0))
    editor._loudness.activated.emit(editor._loudness.currentIndex())
    assert m.loudness() == -14.0
    editor._strength.set_value("off")
    assert m.strength_of(15, 25) == "off" and not editor._remover.isEnabled()
    editor._undo_btn.click()  # back to the saved result
    assert not m.dirty
    editor._which.setChecked(True)  # the note on which one to pick
    assert editor._which_text.isVisible() and "keeps your voice exactly as recorded" in editor._which_text.text()
    assert "makes the voice again from scratch" in editor._which_text.text()


def test_a_part_that_is_half_rebuilt_shows_a_mixed_state_and_the_steppers_work(editor):
    m = editor.model
    m.set_layer("rebuild", True, 10, 20)
    m.select(15, 30)
    editor.goto("change")
    assert editor._rebuild.checkState() == Qt.CheckState.PartiallyChecked and editor._wins.isVisible()
    editor.goto("range")
    editor._time_labels["s"].parent().findChildren(type(editor.save_button))[0].click()  # the first "−" (Start one second earlier)
    assert m.selection == (14.0, 30.0)
    editor.set_playhead(22.0)
    for b in editor.findChildren(type(editor.save_button)):
        if b.text() == "Around playhead":
            b.click()
    assert m.selection == (17.0, 27.0)


def test_the_hear_chips_tell_the_owner_and_show_all_layers_switches_the_strip(editor):
    heard = []
    editor.hearChanged.connect(lambda layer, on: heard.append((layer, on)))
    editor._chips["noise"].click()
    editor._chips["leveling"].click()
    assert heard == [("noise", False), ("leveling", False)] and editor.strip.hear["noise"] is False
    editor._rows.click()
    assert editor.strip.rows and editor._rows.text() == "Show fewer layers"
    editor._rows.click()
    assert not editor.strip.rows


def test_the_moment_page_shows_suggestion_cards_with_apply_show_me_dismiss_and_undo(editor, qapp):
    m = editor.model
    seeks, kept = [], []
    editor.seekRequested.connect(seeks.append)
    editor.suggestionsChanged.connect(lambda: kept.append(1))
    def buttons():
        qapp.processEvents()  # (a widget added to a window that is showing is shown a moment later)
        return {b.text(): b for b in editor.findChildren(type(editor.save_button)) if b.isVisible()}
    assert "Rebuild this part" in buttons() and "Show me" in buttons() and "Dismiss" in buttons()
    buttons()["Rebuild this part"].click()
    assert m.suggestions[0].state == "applied" and kept and "Undo" in buttons() and "Rebuild this part" not in buttons()
    buttons()["Undo"].click()
    assert m.suggestions[0].state == "open"
    buttons()["Show me"].click()
    assert editor.view == "change" and m.selection == (41.0, 44.0) and seeks[-1] == 41.0
    editor.goto("home")
    buttons()["Dismiss"].click()
    assert m.suggestions[0].state == "dismissed" and "Dismiss" not in buttons() and len(kept) == 3
    editor.set_finding(0.4)
    assert "Stop" in buttons()
    editor.set_finding(None, "", True)
    assert "Find moments to fix" in buttons()
    asked = []
    editor.findRequested.connect(lambda: asked.append(1))
    buttons()["Find moments to fix"].click()
    assert asked


def test_the_bar_shows_the_quote_and_saving_is_a_busy_state_with_stop(editor):
    editor.show_quote(None)
    assert editor.bar_title.text() == "No changes yet" and not editor.save_button.isEnabled()
    editor.model.set_strength("max", 10, 20)
    editor.show_quote(editor_job.Quote(seconds=41.0, steps={"heavy": 1}, misses=1))
    assert editor.bar_title.text() == "Process changes · about 40 s" and editor.save_button.isEnabled()
    saved, stopped = [], []
    editor.saveRequested.connect(lambda: saved.append(1))
    editor.stopRequested.connect(lambda: stopped.append(1))
    editor.save_button.click()
    editor.set_busy(True, 0.25)
    assert saved and editor.progress.isVisible() and editor.progress.value() == 250 and editor.stop_button.isVisible()
    assert not editor.save_button.isEnabled() and not editor.panel.isEnabled() and editor.save_button.text() == "Saving..."
    editor.stop_button.click()
    assert stopped
    editor.set_busy(False)
    assert not editor.progress.isVisible() and editor.panel.isEnabled() and editor.save_button.text() == "Save"
    editor.show_quote(editor_job.Quote())
    assert editor.bar_title.text() == "Changes: nothing to process"


def test_plan_mode_has_no_hear_chips_and_offers_whole_file_or_only_parts(qapp):
    base = Timeline.whole_file(100.0)
    m = EditModel(100.0, base, None, plan_mode=True)
    e = part_editor.PartEditor(m, [50] * 200)
    e.resize(1000, 700)
    e.show()
    assert not e.chips_row.isVisible() and not e.save_button.isVisible()
    texts = [b.text() for b in e.findChildren(type(e.save_button)) if b.isVisible()]
    assert "Whole file" in texts and "Only parts" in texts and "Find moments to fix" in texts
    m.select(10, 20)
    e.goto("range")
    only = next(b for b in e.findChildren(type(e.save_button)) if b.text() == "Only this part")
    only.click()
    assert m.only and m.chosen == [(10.0, 20.0)] and e.view == "home"
    assert any(b.text() == "Remove" for b in e.findChildren(type(e.save_button)))
    e.show_quote(editor_job.Quote(seconds=12.0, misses=1, steps={"light": 1}))
    assert e.bar_title.text().startswith("Chosen parts")
