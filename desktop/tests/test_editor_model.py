"""What the editor screen shows and changes (editor_model.py: the draft, the part that is chosen, Undo and Redo, the suggestions) and the live sound it
plays (live_mix.py: per-part strengths, the Hear switches, equal-power edges of the rebuilt voice, Leveling heard by loudness; played through a
stand-in sound card). No model runs: the layers of the live mix are laid out by hand."""
from __future__ import annotations

import math

import numpy as np
import pytest

from acw_common import clarity, pipeline
from acw_common.timeline import Timeline
from audio_cleaner_desktop import live_mix
from audio_cleaner_desktop.editor_model import EditModel, Suggestion

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


def whole_model(seconds=100.0, **kw):
    return EditModel(seconds, Timeline.whole_file(seconds, noise={"strength_db": 24.0, "remover": "light"}, leveling={"loudness_lufs": -16.0}), None, **kw)


def test_leaving_the_rest_as_it_is_takes_every_layer_off_everything_outside_the_part():
    m = whole_model()
    base = m.draft.copy()
    assert m.effective() == base and m.effective() is not m.draft
    m.select(10, 30)
    m.set_strength("gentle")
    m.leave_the_rest()
    assert [(p.start, p.end, p.settings["strength_db"]) for p in m.draft.layers["noise"]] == [(10.0, 30.0, 12.0)]
    assert [(p.start, p.end) for p in m.draft.layers["leveling"]] == [(10.0, 30.0)]
    m.undo_edit()  # one step brings the rest back
    assert [(p.start, p.end) for p in m.draft.layers["noise"]] == [(0.0, 10.0), (10.0, 30.0), (30.0, 100.0)]
    m.undo_edit()
    assert m.draft == base


def test_undo_and_redo_step_through_every_edit_and_a_new_edit_forgets_the_redo():
    m = whole_model()
    start = m.draft.copy()
    assert not m.can_undo and not m.can_redo
    m.select(10, 20)  # (choosing a part is not an edit)
    assert not m.can_undo
    m.set_strength("gentle")
    m.set_layer("rebuild", True)
    m.set_loudness(-14.0)
    assert m.can_undo and not m.can_redo
    after = m.draft.copy()
    m.undo_edit()
    m.undo_edit()
    m.undo_edit()
    assert m.draft == start and not m.can_undo and m.can_redo and m.loudness() == -16.0
    m.redo_edit()
    m.redo_edit()
    m.redo_edit()
    assert m.draft == after and m.loudness() == -14.0 and not m.can_redo
    m.undo_edit()
    m.set_strength("strong")  # a new edit: what could have been redone is gone
    assert not m.can_redo
    m.clear_history()
    assert not m.can_undo


def test_suggestions_applied_and_dismissed_can_be_undone_with_the_draft():
    m = whole_model()
    m.set_suggestions([sugg(id="a", start=10.0, end=20.0), sugg(id="b", start=30.0, end=40.0, rebuild=False, noise={"strength_db": 12.0, "remover": "light"})])
    m.apply("a")
    m.dismiss("b")
    assert m.suggestion("a").state == "applied" and m.suggestion("b").state == "dismissed" and m.switch_of("rebuild", 10, 20) is True
    m.undo_edit()
    assert m.suggestion("b").state == "open"
    m.undo_edit()
    assert m.suggestion("a").state == "open" and m.switch_of("rebuild", 10, 20) is False
    m.redo_edit()
    assert m.suggestion("a").state == "applied"


def test_the_voice_is_off_noise_removal_or_voice_rebuild_for_a_part_and_the_sections_say_so():
    m = whole_model()
    assert m.voice_mode_of(0, 100) == "noise" and m.level_of(0, 100) == "on"
    m.set_voice("rebuild", 20, 30)
    assert m.voice_mode_of(20, 30) == "rebuild" and m.voice_mode_of(0, 100) == "mixed" and m.switch_of("noise", 20, 30) is False
    m.set_voice("off", 40, 50)
    m.set_voice("noise", 40, 45)  # Noise removal where it was off starts as the default (Max); where it was on it keeps what it had
    assert m.strength_of(40, 45) == "max"
    m.set_strength("gentle", 0, 10)
    m.set_voice("noise", 0, 10)
    assert m.strength_of(0, 10) == "gentle"
    rows = [(round(s["start"]), round(s["end"]), s["mode"], s["label"]) for s in m.voice_sections()]
    assert rows == [(0, 10, "noise", "Gentle · Light"), (10, 20, "noise", "Balanced · Light"), (20, 30, "rebuild", "Rebuild"),
                    (30, 40, "noise", "Balanced · Light"), (40, 45, "noise", "Max"), (45, 50, "off", ""), (50, 100, "noise", "Balanced · Light")]
    m.set_level(-14.0, True, 10, 20)  # a loudness changes only the range it is chosen for; the rest keeps its own
    assert m.loudness() == -14.0
    assert [(round(p.start), round(p.end), p.settings["loudness_lufs"]) for p in m.draft.layers["leveling"]] == [(0, 10, -16.0), (10, 20, -14.0), (20, 100, -16.0)]
    m.set_level(None, False, 60, 70)
    assert m.level_of(60, 70) == "off" and m.level_of(0, 100) == "mixed"
    assert [(round(s["start"]), round(s["end"]), s["on"]) for s in m.sections("leveling")] == [
        (0, 10, True), (10, 20, True), (20, 60, True), (60, 70, False), (70, 100, True)]


def test_leveling_is_chosen_per_part_and_loudness_of_says_one_value_or_mixed():
    m = whole_model()
    m.set_level(-14.0, True, 10, 20)
    m.set_level(-23.0, True, 40, 50)
    assert [(round(p.start), round(p.end), p.settings["loudness_lufs"]) for p in m.draft.layers["leveling"]] == [
        (0, 10, -16.0), (10, 20, -14.0), (20, 40, -16.0), (40, 50, -23.0), (50, 100, -16.0)]
    # one loudness in the range: that value
    assert m.loudness_of(10, 20) == -14.0 and m.loudness_of(12, 18) == -14.0 and m.loudness_of(40, 50) == -23.0 and m.loudness_of(20, 40) == -16.0
    # parts that only touch the range do not count
    assert m.loudness_of(20, 30) == -16.0 and m.loudness_of(0, 10) == -16.0
    # different loudnesses in the range: mixed
    assert m.loudness_of(5, 15) == "mixed" and m.loudness_of(0, 100) == "mixed" and m.loudness_of(15, 45) == "mixed"
    # none of the range is leveled: the loudness that was chosen last
    m.set_level(None, False, 60, 70)
    assert m.loudness_of(60, 70) == -23.0 and m.loudness_of(62, 68) == m.loudness()
    # "keep the level" (None) is a value too, and differs from a number
    m.set_level(None, True, 80, 90)
    assert m.loudness_of(80, 90) is None and m.loudness_of(75, 85) == "mixed"


def test_loudness_of_without_any_leveled_part_falls_back_to_the_one_the_model_started_with():
    m = model_of(100.0, loudness=-14.0)  # leveling off everywhere
    assert m.loudness_of(0, 100) == -14.0 and m.level_of(0, 100) == "off"
    assert whole_model().loudness_of(0, 100) == -16.0


def test_leveling_turned_on_later_starts_with_the_loudness_chosen_last_and_set_loudness_still_levels_every_part():
    m = whole_model()
    m.set_level(None, False, 30, 40)
    m.set_level(-14.0, True, 10, 20)
    m.set_layer("leveling", True, 30, 40)  # on again elsewhere: the last loudness chosen
    assert m.level_of(30, 40) == "on" and m.loudness_of(30, 40) == -14.0 and m.loudness_of(10, 20) == -14.0 and m.loudness_of(50, 60) == -16.0
    m.set_level(-23.0, True, 60, 70)
    m.set_level(None, False, 80, 90)
    m.set_layer("leveling", True, 80, 90)
    assert m.loudness_of(80, 90) == -23.0 and m.loudness_of(10, 20) == -14.0
    m.set_loudness(-16.0)  # (unchanged) the target of the job: every leveled part gets it
    assert m.loudness_of(0, 100) == -16.0 and all(p.settings["loudness_lufs"] == -16.0 for p in m.draft.layers["leveling"])


def test_undo_and_redo_of_a_loudness_for_a_part_put_back_the_part_and_the_remembered_loudness():
    m = whole_model()
    start = m.draft.copy()
    m.set_level(-14.0, True, 10, 20)
    after_first = m.draft.copy()
    m.set_level(-23.0, True, 15, 30)
    after = m.draft.copy()
    assert m.loudness() == -23.0 and m.loudness_of(10, 30) == "mixed" and m.loudness_of(15, 30) == -23.0
    m.undo_edit()
    assert m.draft == after_first and m.loudness() == -14.0 and m.loudness_of(10, 30) == "mixed" and m.loudness_of(10, 20) == -14.0
    m.undo_edit()
    assert m.draft == start and m.loudness() == -16.0 and m.loudness_of(0, 100) == -16.0
    m.redo_edit()
    m.redo_edit()
    assert m.draft == after and m.loudness() == -23.0 and m.loudness_of(10, 15) == -14.0 and m.loudness_of(20, 30) == -23.0
    m.undo_edit()
    m.set_level(None, False, 10, 20)  # Off for a part is one edit that undoes
    assert m.level_of(10, 20) == "off"
    m.undo_edit()
    assert m.level_of(10, 20) == "on" and m.loudness_of(10, 20) == -14.0


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
