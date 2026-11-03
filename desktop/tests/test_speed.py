"""How long a run takes on this computer (speed.py): the first guess from the cost of each step and the threads a run gets, the correction learned from real
runs, and the file that keeps it between runs. Nothing here runs a model: the runs are the {step: [wall seconds, seconds of audio]} a run measured."""
from __future__ import annotations

import json
import os

import pytest

from audio_cleaner_desktop import editor_job, speed, sysinfo


def test_the_first_guess_is_the_steps_cost_spread_over_the_threads_a_run_gets():
    fixed, per = speed.CPU_COSTS["light"]
    assert speed.guess("light", 100.0, 4) == pytest.approx((fixed + per * 100) / (4 * speed.THREAD_EFFICIENCY))
    assert speed.guess("light", 100.0, 8) < speed.guess("light", 100.0, 4) < speed.guess("light", 100.0, 2)
    assert speed.guess("light", 100.0, 1) == pytest.approx(fixed + per * 100)  # (one thread is a whole core: never more than one)
    assert speed.guess("light", 100.0, 0) == speed.guess("light", 100.0, 1)
    assert speed.guess("speech", 60.0, 4) == speed.SPEECH_COST * 60.0  # the speech is written down by its own model, not spread over the threads
    assert speed.guess("heavy", 0.0, 4) == 0.0 and speed.guess("heavy", -3.0, 4) == 0.0  # nothing to do costs nothing, not even the fixed amount
    assert set(speed.CPU_COSTS) == {"echo", "light", "heavy", "rebuild", "leveling"}


def test_the_threads_of_a_run_are_the_ones_a_file_gets_when_it_runs_by_itself():
    assert speed.run_threads() == sysinfo.threads_per_job(1)
    assert speed.cost("heavy", 600.0) == pytest.approx(speed.guess("heavy", 600.0, sysinfo.threads_per_job(1)))
    assert speed.cost("heavy", 600.0, 2) == pytest.approx(speed.guess("heavy", 600.0, 2))  # (threads given: those)
    assert speed.cost("heavy", 600.0, 2) > speed.cost("heavy", 600.0, 8)


def test_the_old_names_still_point_to_the_costs_in_speed_py():
    assert editor_job.CPU_COSTS is speed.CPU_COSTS and editor_job.THREAD_EFFICIENCY == speed.THREAD_EFFICIENCY


def test_a_measured_run_changes_the_next_cost_and_the_next_run_averages_with_it():
    guess = speed.guess("heavy", 600.0, 4)
    assert guess > speed.LEARN_FROM_SECONDS
    assert speed.cost("heavy", 600.0, 4) == pytest.approx(guess)  # (nothing learned yet)
    speed.learn({"heavy": [guess * 2, 600.0]}, 4)  # this computer took twice as long as the guess
    assert speed.cost("heavy", 600.0, 4) == pytest.approx(guess * 2)
    assert speed.cost("light", 600.0, 4) == pytest.approx(speed.guess("light", 600.0, 4))  # (only the step that was measured)
    assert speed.cost("heavy", 1200.0, 8) == pytest.approx(speed.guess("heavy", 1200.0, 8) * 2)  # (the ratio holds for other lengths and threads)
    speed.learn({"heavy": [guess * 0.5, 600.0]}, 4)  # then it was twice as fast: half the old correction, half the new
    assert speed.cost("heavy", 600.0, 4) == pytest.approx(guess * (0.5 * 2 + 0.5 * 0.5))
    speed.learn({"speech": [600.0, 600.0], "rebuild": [0.5, 1.0]}, 4)  # (the speech is a step too, and a tiny rebuild is not counted)
    assert speed.cost("speech", 600.0) == pytest.approx(600.0)  # (the guess was 300 s: twice as long)
    assert speed.cost("rebuild", 1.0, 4) == pytest.approx(speed.guess("rebuild", 1.0, 4))


def test_a_tiny_run_or_a_nonsense_one_does_not_count():
    small = speed.guess("light", 10.0, 4)
    assert small < speed.LEARN_FROM_SECONDS  # (a few seconds are mostly loading the model)
    speed.learn({"light": [small * 20, 10.0]}, 4)
    assert speed._ratios == {} and speed.cost("light", 10.0, 4) == pytest.approx(small)
    big = speed.guess("light", 3000.0, 4)
    speed.learn({"light": [0.0, 3000.0], "heavy": [-5.0, 3000.0], "bogus": [500.0, 3000.0], "echo": [10.0, 0.0]}, 4)
    assert speed._ratios == {}  # no time taken, a negative one, a step nobody knows, no audio made
    assert speed.cost("light", 3000.0, 4) == pytest.approx(big)
    speed.learn({}, 4)
    assert speed._ratios == {}


def test_a_ratio_stays_between_a_hundredth_and_a_hundred():
    guess = speed.guess("light", 3000.0, 4)
    speed.learn({"light": [guess * 5000, 3000.0]}, 4)  # (a computer that was asleep for an hour in the middle of the run)
    assert speed._ratios["light"] == 100.0
    speed.learn({"heavy": [0.001, 3000.0]}, 4)
    assert speed._ratios["heavy"] == 0.01


def test_the_learned_speeds_are_written_to_the_file_and_read_back(tmp_path):
    path = tmp_path / "data" / "speed.json"
    path.parent.mkdir()
    speed.use_file(path)  # (no file yet: nothing learned, nothing written)
    assert not path.exists()
    guess = speed.guess("heavy", 600.0, 4)
    speed.learn({"heavy": [guess * 3, 600.0], "light": [1.0, 5.0]}, 4)
    assert json.loads(path.read_text(encoding="utf-8")) == {"heavy": pytest.approx(3.0)}  # (the tiny light run is not in it)
    speed._ratios.clear()  # the app is closed and opened again
    speed.use_file(path)
    assert speed.cost("heavy", 600.0, 4) == pytest.approx(guess * 3)
    speed.learn({"heavy": [guess * 3, 600.0]}, 4)
    assert json.loads(path.read_text(encoding="utf-8")) == {"heavy": pytest.approx(3.0)}


def test_a_run_that_learns_nothing_does_not_write_the_file(tmp_path):
    path = tmp_path / "speed.json"
    speed.use_file(path)
    speed.learn({"light": [1.0, 5.0]}, 4)
    assert not path.exists()


@pytest.mark.parametrize("content", ["", "not json at all", "[1, 2, 3]", "null", '{"heavy": "slow"}', '{"heavy": [2]}', "\x00\x01\x02"])
def test_a_file_that_cannot_be_read_is_ignored(tmp_path, content):
    path = tmp_path / "speed.json"
    path.write_text(content, encoding="utf-8", errors="ignore")
    speed.use_file(path)  # (no exception: the guesses are as if there were no file)
    assert speed._ratios == {} and speed.cost("heavy", 600.0, 4) == pytest.approx(speed.guess("heavy", 600.0, 4))
    guess = speed.guess("heavy", 600.0, 4)
    speed.learn({"heavy": [guess * 2, 600.0]}, 4)  # and the next run that learns something replaces it with a good one
    assert json.loads(path.read_text(encoding="utf-8")) == {"heavy": pytest.approx(2.0)}


def test_only_good_entries_of_the_file_count(tmp_path):
    path = tmp_path / "speed.json"
    path.write_text(json.dumps({"light": 1e9, "heavy": 2.0, "rebuild": 0.0, "bogus": 3.0, "speech": 1.5, "echo": -1}), encoding="utf-8")
    speed.use_file(path)
    assert speed._ratios == {"heavy": 2.0, "speech": 1.5}  # (out of range or unknown: left out)


def test_a_file_that_cannot_be_written_does_not_stop_a_run(tmp_path):
    speed.use_file(tmp_path / "no" / "such" / "folder" / "speed.json")
    guess = speed.guess("heavy", 600.0, 4)
    speed.learn({"heavy": [guess * 2, 600.0]}, 4)  # (kept until the app closes)
    assert speed.cost("heavy", 600.0, 4) == pytest.approx(guess * 2)


def test_the_time_of_a_plan_is_its_steps_costs_added_up_on_the_threads_of_a_run():
    class Plan:
        def summary(self):
            return {"light": {"blocks": 2, "compute_seconds": 70.0}, "heavy": {"blocks": 0, "compute_seconds": 30.0},
                    "leveling": {"blocks": 1, "compute_seconds": 120.0}}

    want = speed.cost("light", 70.0) + speed.cost("leveling", 120.0)  # (a step with no block to make adds nothing)
    assert editor_job.estimate_seconds(Plan()) == pytest.approx(want)
    assert editor_job.estimate_seconds(Plan(), 2) == pytest.approx(speed.cost("light", 70.0, 2) + speed.cost("leveling", 120.0, 2))
    speed.learn({"light": [speed.guess("light", 600.0, 4) * 4, 600.0]}, 4)
    assert editor_job.estimate_seconds(Plan(), 4) == pytest.approx(speed.cost("light", 70.0, 4) + speed.cost("leveling", 120.0, 4))
    assert speed.cost("light", 70.0, 4) == pytest.approx(speed.guess("light", 70.0, 4) * 4)  # (what was learned is in it)


def test_a_run_by_itself_gets_every_thread_but_one_of_the_processor():
    assert speed.run_threads() == max(2, (os.cpu_count() or 2) - 1)
