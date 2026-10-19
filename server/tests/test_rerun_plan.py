"""Re-runs, the parts that need no database or binaries: what a change of choices runs again and what it costs
(rerun.plan_rerun, pricing.rerun_rates / rerun_cost_micros), the checks of the kept copies a runner reports, the
expiry arithmetic, the active-job limit and the sizes of the kept copies."""
import json
import os

import pytest

from acw import db, jobs, rerun
from acw.billing import pricing

from .rerun_support import account, add_job, job_params, kept_record, set_env

AUDIO = {"file": "talk-cleaned.flac", "kind": "audio"}
ALL_FILES = {"input.wav", "kept-noise.flac", "kept-input.flac", "kept-rebuilt.flac", "segments.json", "segments.fr.json",
             "talk-cleaned.flac"}


def source(kind="denoise", d="standard", t=None, *, kept=None, with_noise=True, rebuilt=False, segments=None, **options):
    """Stored parameters of a finished job: Denoising with Leveling by default, with the kept copies a runner leaves."""
    k = kept if kept is not None else kept_record(60.0, noise=with_noise and d == "standard", rebuilt=rebuilt,
                                                  segments=segments if t else None)
    return job_params(kind, d, t, options, kept=k)


def replan(src, denoise_tier=None, transcribe_tier=None, files=ALL_FILES, audio=AUDIO, **changes):
    """The plan for a re-run of `src` with these changes, built the way rerun._prepare does."""
    _, d_tier, t_tier = rerun._tiers(src, denoise_tier, transcribe_tier)
    kind = rerun._tiers(src, denoise_tier, transcribe_tier)[0]
    new = jobs.validate_options(kind, d_tier, t_tier, rerun.merged_options(src, changes))
    return rerun.plan_rerun(src, audio, d_tier, t_tier, new, set(files))


PLAIN = dict()                       # Denoising + Leveling, kept noise removal and recording
FREE = dict(leveling=False)          # the same without Leveling: every cheap change is free
TRANS = dict(kind="denoise_transcribe", t="local", segments={"language": "en"}, language="en")  # with subtitles


# ---------------------------------------------------------------------------------------------- the plan matrix

@pytest.mark.parametrize("src_kw, change, clean, paid, work", [
    pytest.param(PLAIN, {}, "reuse", [], False, id="same-choices-nothing-to-do"),
    pytest.param(PLAIN, {"atten_lim_db": 24}, "denoised", ["leveling"], True, id="strength-with-leveling-on-pays-leveling"),
    pytest.param(FREE, {"atten_lim_db": 24}, "denoised", [], True, id="strength-only-is-free"),
    pytest.param(FREE, {"leveling": True}, "denoised", ["leveling"], True, id="leveling-on-pays-leveling"),
    pytest.param(PLAIN, {"leveling": False}, "denoised", [], True, id="leveling-off-is-free"),
    pytest.param(PLAIN, {"loudness_lufs": -20.0}, "denoised", ["leveling"], True, id="loudness-with-leveling-pays-leveling"),
    pytest.param(FREE, {"loudness_lufs": -20.0}, "reuse", [], False, id="loudness-means-nothing-without-leveling"),
    pytest.param(PLAIN, {"noise": "light"}, "denoised", ["leveling"], True, id="noise-kind-auto-chose-is-the-kept-one"),
    pytest.param(PLAIN, {"noise": "heavy"}, "full", ["denoise", "leveling"], True, id="other-noise-kind-runs-noise-removal"),
    pytest.param(FREE, {"noise": "heavy"}, "full", ["denoise"], True, id="other-noise-kind-without-leveling"),
    pytest.param(PLAIN, {"echo_reference_track": True}, "full", ["denoise", "leveling"], True, id="echo-on-runs-noise-removal"),
    pytest.param(FREE, {"echo_reference_track": True}, "full", ["denoise"], True, id="echo-on-without-leveling"),
    pytest.param(PLAIN, {"audio_format": "wav"}, "reuse", [], True, id="file-type-only-is-free-work"),
])
def test_what_runs_again_for_a_change_of_choices(src_kw, change, clean, paid, work):
    """Each row: the steps a re-run runs (clean mode), the paid ones, and whether there is anything to do."""
    plan = replan(source(**src_kw), **change)
    assert (plan["clean"], plan["paid"], plan["work"]) == (clean, paid, work)
    assert plan["transcribe"] is False and plan["translate_new"] == []
    assert plan["reuse_audio"] == (AUDIO["file"] if clean == "reuse" else None)


def test_voice_rebuild_on_runs_the_rebuild_from_the_upload_at_its_price():
    """Rebuild on from a Denoising job has no kept rebuilt voice: it runs in full and pays the rebuild (and Leveling when on)."""
    plan = replan(source(), "rebuild")
    assert (plan["clean"], plan["paid"], plan["work"]) == ("full", ["denoise", "leveling"], True)
    assert replan(source(**FREE), "rebuild")["paid"] == ["denoise"]
    rates = pricing.rerun_rates("rebuild", None, plan["paid"])
    assert rates == {"denoise": pricing.rate("denoise", "rebuild").micros_per_minute, "leveling": 1_000}


def test_rebuild_with_a_kept_rebuilt_voice_only_pays_leveling():
    """From a Voice rebuild job: same choices nothing, other Leveling choices run from the kept rebuilt voice."""
    rebuilt = source(d="rebuild", rebuilt=True)
    assert (replan(rebuilt)["clean"], replan(rebuilt)["work"]) == ("reuse", False)
    plan = replan(rebuilt, loudness_lufs=-20.0)
    assert (plan["clean"], plan["paid"], plan["work"]) == ("rebuilt", ["leveling"], True)
    plan = replan(rebuilt, leveling=False)
    assert (plan["clean"], plan["paid"]) == ("rebuilt", [])
    # no kept rebuilt voice (its file is gone) runs the rebuild again
    assert replan(rebuilt, files=ALL_FILES - {"kept-rebuilt.flac"}, loudness_lufs=-20.0)["clean"] == "full"
    # a rebuilt voice kept without echo removal does not fit a re-run with echo removal on
    assert replan(rebuilt, echo_reference_track=True)["clean"] == "full"


def test_rebuild_off_uses_the_kept_noise_removal_or_runs_it_charged():
    """Back to Denoising: free from a kept noise removal, charged when the job never had one."""
    from_rebuild_with_noise = source(d="rebuild", kept=kept_record(60.0, noise=True, rebuilt=True), leveling=False)
    plan = replan(from_rebuild_with_noise, "standard")
    assert (plan["clean"], plan["paid"], plan["work"]) == ("denoised", [], True)
    plan = replan(source(d="rebuild", rebuilt=True), "standard")  # a rebuild job keeps no noise removal
    assert (plan["clean"], plan["paid"]) == ("full", ["denoise", "leveling"])


def test_a_job_that_inherited_the_rebuilt_voice_goes_back_to_rebuild_from_it():
    """A Denoising re-run of a rebuild carries the rebuilt voice: rebuild on again only pays Leveling."""
    carried = source(kept=kept_record(60.0, noise=True, rebuilt=True))
    plan = replan(carried, "rebuild")
    assert (plan["clean"], plan["paid"]) == ("rebuilt", ["leveling"])


def test_leveling_on_its_own_is_always_paid_when_it_changes():
    """Noise tier off: same choices reuse the result; another loudness runs Leveling again from the upload (charged)."""
    off = source(d="off")
    assert replan(off)["clean"] == "reuse" and replan(off)["work"] is False
    plan = replan(off, loudness_lufs=-20.0)
    assert (plan["clean"], plan["paid"], plan["work"]) == ("full", ["leveling"], True)
    assert replan(off, audio_format="wav")["work"] is True and replan(off, audio_format="wav")["paid"] == []
    # from a Denoising job, Leveling only means no noise removal: Leveling from the upload
    plan = replan(source(), "off")
    assert (plan["clean"], plan["paid"]) == ("full", ["leveling"])
    assert plan["carry"] == ["noise", "input"]  # the kept noise removal stays for a later change back


def test_the_noise_tier_off_reuses_a_result_with_the_same_leveling_and_pays_for_leveling_when_it_runs():
    """Leveling on its own: reuse needs the same Leveling choice; Leveling is paid when it runs again; an old option that is no longer
    one (restore_highs in the options of a job from before) changes nothing."""
    leveling_only = source(d="off")
    assert replan(leveling_only)["clean"] == "reuse"
    assert replan(leveling_only, restore_highs=True)["clean"] == "reuse"  # not an option any more: ignored
    old_job = source(d="off", restore_highs=True)
    assert "restore_highs" not in old_job["options"] and replan(old_job)["clean"] == "reuse"
    stored_before = {**leveling_only, "options": {**leveling_only["options"], "restore_highs": True}}  # a job stored by an older version
    assert replan(stored_before)["clean"] == "reuse"  # (its extra key is just not read)


def test_a_missing_file_changes_what_can_be_reused():
    """The plan only trusts files that are in the folder."""
    gone_noise = replan(source(), files=ALL_FILES - {"kept-noise.flac"}, atten_lim_db=24)
    assert (gone_noise["clean"], gone_noise["paid"]) == ("full", ["denoise", "leveling"])
    no_audio = replan(source(), audio=None, audio_format="wav")  # nothing to reuse: made from the kept noise removal
    assert (no_audio["clean"], no_audio["paid"]) == ("denoised", ["leveling"])
    gone_audio = replan(source(), files=ALL_FILES - {"talk-cleaned.flac"}, audio_format="wav")
    assert gone_audio["clean"] == "denoised" and gone_audio["reuse_audio"] is None


def test_which_kept_copies_a_rerun_carries():
    """Copies that still fit the echo choice stay; the ones the run makes again are not carried."""
    assert replan(source(), atten_lim_db=24)["carry"] == ["noise", "input"]
    assert replan(source(), noise="heavy")["carry"] == []  # noise removal runs again and writes its own
    assert replan(source(), echo_reference_track=True)["carry"] == []  # kept without echo removal: does not fit
    with_rebuilt = source(kept=kept_record(60.0, noise=True, rebuilt=True))
    assert replan(with_rebuilt, noise="heavy")["carry"] == ["rebuilt"]
    assert replan(source(), "rebuild")["carry"] == ["noise", "input"]  # the rebuild is made again, the noise removal stays
    echoed = source(kept=kept_record(60.0, noise=True, echo=True), echo_reference_track=True)
    assert replan(echoed, atten_lim_db=24)["carry"] == ["noise", "input"]


# --------------------------------------------------------------------------------------------- subtitles

def test_subtitles_unchanged_run_nothing_and_a_new_tier_or_language_transcribes_again_charged():
    """The transcript is reused while its tier, language and echo choice match; otherwise it is made again (paid)."""
    src = source(**TRANS)
    plan = replan(src)
    assert (plan["clean"], plan["transcribe"], plan["paid"], plan["work"]) == ("reuse", False, [], False)
    assert plan["reuse_segments"] is True
    for kw in ({"transcribe_tier": "premium"}, {"language": "de"}):
        plan = replan(src, **kw)
        assert (plan["clean"], plan["transcribe"], plan["paid"], plan["work"]) == ("reuse", True, ["transcribe"], True)
        assert plan["reuse_segments"] is False
    assert replan(src, files=ALL_FILES - {"segments.json"})["transcribe"] is True  # no transcript file


def test_only_the_new_translation_languages_are_made_and_paid():
    """Languages already translated are kept; the new ones are the "translate" step."""
    kept = kept_record(60.0, segments={"language": "en", "translations": ["fr"]})
    src = source(kind="denoise_transcribe", t="local", kept=kept, language="en", translate=["fr"])
    plan = replan(src, translate=["fr", "es"])
    assert (plan["transcribe"], plan["translate_new"], plan["paid"], plan["work"]) == (False, ["es"], ["translate"], True)
    same = replan(src, translate=["fr"])
    assert (same["translate_new"], same["paid"], same["work"]) == ([], [], False)
    dropped = replan(src, translate=[])
    assert (dropped["translate_new"], dropped["work"]) == ([], False)
    no_seg_translation = source(**TRANS)
    plan = replan(no_seg_translation, translate=["fr", "es"])
    assert (plan["translate_new"], plan["paid"]) == (["fr", "es"], ["translate"])
    # a transcript made again translates every language asked for
    plan = replan(no_seg_translation, transcribe_tier="premium", translate=["fr"])
    assert (plan["transcribe"], plan["translate_new"], plan["paid"]) == (True, ["fr"], ["transcribe", "translate"])


def test_caption_files_only_are_free_work_from_the_kept_transcript():
    """Another set of subtitle files is written again from the kept transcript at no charge; without one it transcribes."""
    src = source(**TRANS)
    plan = replan(src, caption_formats=["srt"])
    assert (plan["work"], plan["paid"], plan["transcribe"], plan["reuse_segments"]) == (True, [], False, True)
    same = replan(src, caption_formats=["txt", "vtt", "srt"])  # the same set in another order
    assert same["work"] is False
    no_transcript = replan(src, files=ALL_FILES - {"segments.json"}, caption_formats=["srt"])
    assert (no_transcript["transcribe"], no_transcript["paid"]) == (True, ["transcribe"])


def test_subtitles_with_echo_removal_changes_start_over_from_the_upload():
    """A transcript kept without echo removal does not fit echo removal on: noise removal and transcription both run."""
    plan = replan(source(**TRANS), echo_reference_track=True)
    assert (plan["clean"], plan["transcribe"], plan["paid"]) == ("full", True, ["denoise", "leveling", "transcribe"])


def test_adding_and_dropping_subtitles():
    """Adding subtitles to a Denoising job transcribes; "off" drops them, which alone gives nothing to do."""
    plan = replan(source(), transcribe_tier="local")
    assert (plan["clean"], plan["transcribe"], plan["paid"], plan["work"]) == ("reuse", True, ["transcribe"], True)
    assert rerun._tiers(source(), None, "local")[0] == "denoise_transcribe"
    dropped = replan(source(**TRANS), transcribe_tier="off")
    assert dropped["work"] is False and dropped["transcribe"] is False and dropped["paid"] == []


def test_a_transcript_only_job_changed_to_clean_too():
    """Noise removal added to a subtitles-only job runs from the upload (charged) and reuses the transcript."""
    transcript_only = source(kind="transcribe", d=None, t="local", with_noise=False, segments={"language": "en"}, language="en")
    plan = replan(transcript_only, "standard")
    assert (plan["clean"], plan["paid"], plan["transcribe"], plan["reuse_segments"]) == ("full", ["denoise", "leveling"], False, True)
    again = replan(transcript_only, transcribe_tier="premium")  # no noise removal either way
    assert (again["clean"], again["transcribe"], again["paid"]) == ("none", True, ["transcribe"])
    assert rerun._tiers(transcript_only, None, None) == ("transcribe", None, "local")


def test_choices_that_leave_no_work_to_do_are_refused_by_the_tier_step():
    """No noise removal and no subtitles at all is nothing to change (400, code nothing_to_change)."""
    transcript_only = source(kind="transcribe", d=None, t="local", with_noise=False)
    with pytest.raises(jobs.JobError) as e:
        rerun._tiers(transcript_only, None, "off")
    assert e.value.code == rerun.NOTHING_TO_CHANGE and e.value.status == 400


def test_tiers_are_what_is_asked_for_else_what_the_job_had():
    """The kind, noise tier and subtitle tier of a re-run."""
    src = source(**TRANS)
    assert rerun._tiers(src, None, None) == ("denoise_transcribe", "standard", "local")
    assert rerun._tiers(src, "rebuild", None) == ("denoise_transcribe", "rebuild", "local")
    assert rerun._tiers(src, None, "premium") == ("denoise_transcribe", "standard", "premium")
    assert rerun._tiers(src, None, "off") == ("denoise", "standard", None)
    assert rerun._tiers(source(d="off"), None, None) == ("denoise", "off", None)


def test_the_changes_are_laid_over_the_choices_the_job_had():
    """merged_options keeps what is not mentioned."""
    src = source(atten_lim_db=24, noise="light")
    assert rerun.merged_options(src, {"atten_lim_db": 50})["noise"] == "light"
    assert rerun.merged_options(src, None) == src["options"]
    assert rerun.merged_options(src, {"atten_lim_db": 50})["atten_lim_db"] == 50


# ------------------------------------------------------------------------------------------------ pricing

def test_rerun_rates_only_include_the_steps_that_run_at_todays_prices():
    """Rates per minute of the paid steps; steps that are not paid, or the off tier, have none."""
    r = pricing.rerun_rates
    assert r("standard", None, []) == {}
    assert r("standard", None, ["leveling"]) == {"leveling": 1_000}
    assert r("standard", None, ["denoise", "leveling"]) == {"denoise": 9_000, "leveling": 1_000}
    assert r("standard", None, ["highs"]) == {}  # (a step that does not exist any more prices nothing)
    assert r("rebuild", None, ["denoise"]) == {"denoise": 30_000}
    assert r("off", None, ["denoise", "leveling"]) == {"leveling": 1_000}  # no noise removal to price
    assert r("standard", "local", ["transcribe"]) == {"transcribe": 4_000}
    assert r("standard", "premium", ["transcribe", "translate"]) == {"transcribe": 15_000, "translate": 4_000}
    assert r(None, "local", ["translate"]) == {"translate": 4_000}
    assert r("standard", None, ["transcribe"]) == {}  # no subtitle tier: nothing to price
    assert pricing.RERUN_ITEMS == ("denoise", "leveling", "transcribe", "translate")


def test_rerun_cost_is_per_second_rounded_up_per_step_and_per_language():
    """Arithmetic of the price: each step rounds up on its own, translations multiply by the languages made."""
    cost = pricing.rerun_cost_micros
    assert cost({}, 100) == 0
    assert cost({"leveling": 1_000}, 60) == 1_000
    assert cost({"leveling": 1_000}, 14) == 234  # 1000 * 14 / 60 = 233.3 -> 234
    assert cost({"leveling": 1_000}, 14.2) == cost({"leveling": 1_000}, 15) == 250  # a started second counts
    assert cost({"leveling": 1_000}, 0) == 0 and cost({"leveling": 1_000}, -5) == 0
    assert cost({"denoise": 9_000, "leveling": 1_000}, 600) == 90_000 + 10_000
    # each step is rounded up separately, then added
    assert cost({"denoise": 9_000, "leveling": 1_000}, 1) == 150 + 17
    assert cost({"denoise": 9_000, "leveling": 1_000}, 15) == 2_250 + 250
    assert cost({"highs": 70_000}, 60) == 0  # (an old pinned line for a step that does not exist any more is not charged)
    assert cost({"translate": 4_000}, 60, 2) == 8_000 and cost({"translate": 4_000}, 60, 0) == 0
    assert cost({"translate": 4_000}, 30, 3) == 3 * 2_000
    assert cost({"transcribe": 15_000, "translate": 4_000}, 90, 2) == 22_500 + 2 * 6_000
    assert cost({"leveling": 1_000}, 60, 5) == 1_000  # languages only count for the translate step


def test_a_rerun_that_runs_everything_costs_what_the_same_first_job_costs():
    """Paying for every step again is the price of a new job with the same choices."""
    rates = pricing.rerun_rates("standard", "local", list(pricing.RERUN_ITEMS))
    first = pricing.job_rates("denoise_transcribe", "standard", "local", 2, True)
    assert rates == first and set(rates) == set(pricing.RERUN_ITEMS)
    assert pricing.rerun_cost_micros(rates, 123.4, 2) == pricing.job_cost_micros("denoise_transcribe", "standard", "local", 123.4, first, 2, True)


# ---------------------------------------------------------------------------------------- labels

def test_the_label_names_the_steps_that_ran_again():
    """The name on the credit history line and in the job list for a re-run."""
    lab = rerun.rerun_label
    assert lab({}, "standard", None) == "Changed and saved again"
    assert lab({"leveling": 1_000}, "standard", None) == "Changed and saved again (Leveling)"
    assert lab({"denoise": 9_000, "leveling": 1_000}, "standard", None) == "Changed and saved again (Denoising + Leveling)"
    assert lab({"denoise": 30_000}, "rebuild", None) == "Changed and saved again (Voice rebuild)"
    assert lab({"leveling": 1_000}, "off", None) == "Changed and saved again (Leveling)"
    assert lab({"transcribe": 4_000}, "standard", "local") == "Changed and saved again (Subtitles & transcript)"
    assert lab({"transcribe": 15_000}, None, "premium") == "Changed and saved again (Subtitles & transcript, extra accurate)"
    assert lab({"translate": 4_000}, None, "local") == "Changed and saved again (Translated subtitles)"
    assert lab({"transcribe": 4_000, "translate": 4_000}, None, "local") == "Changed and saved again (Subtitles & transcript)"
    assert lab({"denoise": 9_000, "leveling": 1_000, "transcribe": 4_000}, "standard", "local") == \
        "Changed and saved again (Denoising + Leveling + Subtitles & transcript)"


def test_label_of_uses_the_rerun_words_only_for_a_rerun():
    """jobs.label_of: a first job is named by its price-list words, a re-run by what it did again."""
    first = job_params("denoise", "standard", None, {})
    assert jobs.label_of("denoise", first) == "Denoising + Leveling"
    again = {**first, "rates": {"leveling": 1_000}, "rerun": {"clean": "denoised"}}
    assert jobs.label_of("denoise", again) == "Changed and saved again (Leveling)"
    free = {**first, "rates": {}, "rerun": {"clean": "denoised"}}
    assert jobs.label_of("denoise", free) == "Changed and saved again"


# ------------------------------------------------------------------------------- the kept-copies checks

def _kept_files(d, *names):
    d.mkdir(exist_ok=True)
    for n in names:
        (d / n).write_bytes(b"x" * 20)
    return d


def _good(**over):
    return {**kept_record(10.0)["noise"], **over}


def test_sanitize_keeps_known_good_entries_that_exist_as_regular_files(tmp_path):
    """A well-formed record with its files in the folder comes back whole; unknown keys are dropped."""
    d = _kept_files(tmp_path / "j", "kept-noise.flac", "kept-input.flac", "kept-rebuilt.flac")
    raw = {**kept_record(10.0, rebuilt=True), "evil": {"file": "x"}, "segments": {"tier": "local"}}
    got = rerun.sanitize_kept(raw, d)
    assert set(got) == {"noise", "input", "rebuilt"}
    assert got["noise"] == {"file": "kept-noise.flac", "channels": [0, 1], "scale": 1.0, "samples": 480_000, "echo": False,
                            "profile": "light", "noise_opt": "auto", "plan": [0, 1]}
    assert got["input"]["file"] == "kept-input.flac" and "profile" not in got["input"]
    assert rerun.sanitize_kept(raw, d) == got  # idempotent: a clean record stays clean


def test_sanitize_drops_what_is_not_a_dict_and_values_of_the_wrong_type(tmp_path):
    """Bad shapes and numbers drop their entry (only that entry); the rest of the record survives."""
    d = _kept_files(tmp_path / "j", "kept-noise.flac", "kept-input.flac", "kept-rebuilt.flac")
    for raw in (None, [], "x", 5, [kept_record(10.0)]):
        assert rerun.sanitize_kept(raw, d) == {}
    bad_noise = [
        _good(channels=[0, "a"]), _good(channels=[2]), _good(channels=[0, 1, 0]), _good(channels=[]),
        _good(scale="loud"), _good(scale=0), _good(scale=-1), _good(scale=1.5), _good(samples=0), _good(samples="many"),
        _good(profile="medium"), _good(noise_opt="max"), _good(plan=[0, 1, 1]), _good(plan=[]), _good(plan=[0, 2]),
        {k: v for k, v in _good().items() if k != "plan"}, {k: v for k, v in _good().items() if k != "profile"},
        {k: v for k, v in _good().items() if k != "samples"}, _good(file="kept-input.flac"), _good(file="../kept-noise.flac"), _good(file=None),
    ]
    for entry in bad_noise:
        got = rerun.sanitize_kept({"noise": entry, "input": kept_record(10.0)["input"]}, d)
        assert "noise" not in got, entry
        assert "input" in got, entry  # a bad entry never takes the others down
    assert "noise" in rerun.sanitize_kept({"noise": _good(plan=[None, 1])}, d)  # a silent channel is a None in the plan
    assert rerun.sanitize_kept({"noise": _good(echo="yes")}, d)["noise"]["echo"] is True  # coerced to a bool
    assert rerun.sanitize_kept({"noise": _good(scale=0.5)}, d)["noise"]["scale"] == 0.5


def test_sanitize_needs_the_right_name_and_a_regular_file(tmp_path):
    """The runner is sandboxed: a link, a directory, a missing file or another name never counts."""
    d = tmp_path / "j"
    d.mkdir()
    (tmp_path / "secret").write_bytes(b"s")
    os.symlink(tmp_path / "secret", d / "kept-noise.flac")  # a link planted by the runner
    (d / "kept-input.flac").mkdir()  # a directory
    (d / "kept-rebuilt.flac").write_bytes(b"x")
    raw = kept_record(10.0, rebuilt=True)
    assert set(rerun.sanitize_kept(raw, d)) == {"rebuilt"}
    # a hard link is a regular file and counts: that is how a re-run carries its source's copies
    os.unlink(d / "kept-noise.flac")
    os.link(d / "kept-rebuilt.flac", d / "kept-noise.flac")
    assert "noise" in rerun.sanitize_kept(raw, d)
    # an entry naming another entry's file is refused even though that file exists
    swapped = {"rebuilt": {**raw["rebuilt"], "file": "kept-noise.flac"}}
    assert rerun.sanitize_kept(swapped, d) == {}


def test_collect_kept_gathers_what_the_folder_holds_and_drops_an_unmatched_recording(tmp_path):
    """The record of a finished job: reported copies that exist, the transcript's tier/language and its translations."""
    d = _kept_files(tmp_path / "j", "kept-noise.flac", "kept-input.flac", "segments.json", "segments.fr.json", "segments.es.json")
    params = {"options": {"language": "en", "echo_reference_track": False}, "transcribe_tier": "local"}
    kept = rerun.collect_kept(params, d, kept_record(10.0))
    assert set(kept) == {"noise", "input", "segments"}
    assert kept["segments"] == {"tier": "local", "language": "en", "echo": False, "translations": ["es", "fr"]}
    # the recording copy only goes with the noise removal it was kept with: a different length or channels drops it
    mismatch = kept_record(10.0)
    mismatch["input"]["samples"] += 1
    assert "input" not in rerun.collect_kept({"options": {}}, d, mismatch)
    mismatch = kept_record(10.0)
    mismatch["input"]["echo"] = True
    assert "input" not in rerun.collect_kept({"options": {}}, d, mismatch)
    only_input = {"input": kept_record(10.0)["input"]}
    assert rerun.collect_kept({"options": {}}, d, only_input) == {}
    assert rerun.collect_kept({"options": {}}, d, "junk") == {}  # a runner that reported nonsense
    # a transcript without a known tier is not recorded
    assert "segments" not in rerun.collect_kept({"options": {}, "transcribe_tier": None}, d, None)


def test_collect_kept_of_a_rerun_inherits_what_it_linked_and_keeps_the_old_transcript_facts(tmp_path):
    """A re-run that did not transcribe keeps the transcript's old record; one that did describes the new one."""
    d = _kept_files(tmp_path / "j", "kept-noise.flac", "kept-input.flac", "segments.json")
    old = kept_record(10.0, segments={"tier": "premium", "language": "de", "echo": True, "translations": []})
    base = {"options": {"language": "en"}, "transcribe_tier": "local", "kept": old}
    reused = rerun.collect_kept({**base, "rerun": {"transcribe": False}}, d, None)
    assert set(reused) == {"noise", "input", "segments"}
    assert reused["segments"] == {"tier": "premium", "language": "de", "echo": True, "translations": []}
    again = rerun.collect_kept({**base, "rerun": {"transcribe": True}}, d, None)
    assert again["segments"] == {"tier": "local", "language": "en", "echo": False, "translations": []}
    # inherited entries that are not files of the folder any more are dropped
    (d / "kept-noise.flac").unlink()
    assert set(rerun.collect_kept({**base, "rerun": {"transcribe": False}}, d, None)) == {"segments"}  # and the recording goes with it


def test_kept_files_names_the_files_a_record_stands_for():
    """The names kept in the folder: copies, the transcript and one file per translation."""
    assert rerun.kept_files({}) == set()
    kept = kept_record(10.0, rebuilt=True, segments={"translations": ["fr", "es"]})
    assert rerun.kept_files(kept) == {"kept-noise.flac", "kept-input.flac", "kept-rebuilt.flac", "segments.json", "segments.fr.json",
                                      "segments.es.json"}
    assert rerun.kept_files({"input": kept["input"]}) == {"kept-input.flac"}


def test_translations_in_lists_only_two_letter_regular_files(tmp_path):
    """segments.<xx>.json files, not links and not other names."""
    d = tmp_path / "j"
    d.mkdir()
    for n in ("segments.json", "segments.fr.json", "segments.de.json", "segments.fra.json", "segments.FR.json", "segments.f.json", "other.fr.json"):
        (d / n).write_bytes(b"{}")
    os.symlink(d / "segments.fr.json", d / "segments.es.json")
    assert rerun.translations_in(d) == ["de", "fr"]


# ------------------------------------------------------------------------------ expiry arithmetic

def test_completion_expiry_is_two_hours_for_a_first_job_and_never_later_than_the_source_for_a_rerun(monkeypatch):
    """expires = now + retention; a re-run: at most the source's deadline."""
    now = 1_000_000
    assert rerun.completion_expiry({}, {}, now) == now + 7200
    far = {"rerun": {"deadline": now + 5 * 3600}}
    assert rerun.completion_expiry({}, far, now) == now + 7200  # never longer than the retention either
    soon = {"rerun": {"deadline": now + 1800}}
    assert rerun.completion_expiry({}, soon, now) == now + 1800  # ends with the job it came from
    assert rerun.completion_expiry({}, {"rerun": {"deadline": now + 119}}, now) == now + 119  # never later, however little is left
    assert rerun.too_late({"rerun": {"deadline": now + 60}}, now) and rerun.too_late({"rerun": {"deadline": now - 10}}, now)
    assert not rerun.too_late({"rerun": {"deadline": now + 61}}, now) and not rerun.too_late({}, now)
    assert rerun.completion_expiry({}, {"rerun": {"deadline": None}}, now) == now + 7200
    set_env(monkeypatch, ACW_JOB_RETENTION_MINUTES=30)
    assert rerun.completion_expiry({}, {}, now) == now + 1800
    assert rerun.completion_expiry({}, soon, now) == now + 1800


def test_too_late_is_true_in_the_last_minute_and_after_the_deadline():
    """A re-run that is only done within MIN_DELIVERY_SECONDS of its source's deadline (or after it) is stopped, not kept."""
    now = 1_000_000
    late = rerun.too_late
    assert late({"rerun": {"deadline": now - 1}}, now) is True
    assert late({"rerun": {"deadline": now + rerun.MIN_DELIVERY_SECONDS}}, now) is True
    assert late({"rerun": {"deadline": now + rerun.MIN_DELIVERY_SECONDS + 1}}, now) is False
    assert late({"rerun": {"deadline": None}}, now) is False
    assert late({"rerun": {}}, now) is False
    assert late({}, now) is False  # a first job is never late


# ----------------------------------------------------------------------------------------- the limits

def test_the_retention_and_the_job_limit_are_set_by_the_environment(monkeypatch):
    """Defaults 120 minutes and 3 jobs; ACW_JOB_RETENTION_MINUTES and ACW_MAX_ACTIVE_JOBS change them (never below 1)."""
    from acw.config import get_settings

    assert (get_settings().job_retention_seconds, get_settings().max_active_jobs) == (7200, 3)
    set_env(monkeypatch, ACW_JOB_RETENTION_MINUTES=45, ACW_MAX_ACTIVE_JOBS=5)
    assert (get_settings().job_retention_seconds, get_settings().max_active_jobs) == (2700, 5)
    set_env(monkeypatch, ACW_JOB_RETENTION_MINUTES=0, ACW_MAX_ACTIVE_JOBS=0)
    assert (get_settings().job_retention_seconds, get_settings().max_active_jobs) == (60, 1)


def _check(uid):
    with db.tx() as c:
        jobs.check_active_limit(c, uid)


def test_the_active_limit_message_code_and_retry_after():
    """Three jobs in progress: the fourth is refused with the exact words, 429, the code and Retry-After seconds."""
    uid = account()
    for status in ("uploading", "queued", "running"):
        _check(uid)  # fewer than three: fine
        add_job(uid, status)
    with pytest.raises(jobs.JobError) as e:
        _check(uid)
    assert str(e.value) == "You have 3 files being processed. Start this one when one of them is done."
    assert (e.value.status, e.value.code, e.value.retry_after) == (429, "too_many_active_jobs", 60)
    assert jobs.TOO_MANY_JOBS == "too_many_active_jobs"


def test_the_active_limit_counts_the_real_number_and_the_right_states(monkeypatch):
    """cloud_asr counts; finished, failed, canceled, expired and other accounts' jobs do not; the limit follows the variable."""
    set_env(monkeypatch, ACW_MAX_ACTIVE_JOBS=2)
    uid, other = account(), account("b@example.com")
    for status in ("done", "failed", "canceled", "expired"):
        add_job(uid, status, finished_ago=5)
    add_job(other, "queued")  # another account's job is its own business
    add_job(uid, "cloud_asr")
    _check(uid)  # one in progress of two
    add_job(uid, "queued")
    with pytest.raises(jobs.JobError) as e:
        _check(uid)
    assert str(e.value) == "You have 2 files being processed. Start this one when one of them is done."
    _check(other)
    set_env(monkeypatch, ACW_MAX_ACTIVE_JOBS=3)
    _check(uid)  # a larger limit lets it through again


def test_the_active_limit_message_is_singular_for_one_file(monkeypatch):
    """With a limit of 1 the message says "1 file"."""
    set_env(monkeypatch, ACW_MAX_ACTIVE_JOBS=1)
    uid = account()
    add_job(uid, "queued")
    with pytest.raises(jobs.JobError) as e:
        _check(uid)
    assert str(e.value) == "You have 1 file being processed. Start this one when one of them is done."
    assert jobs.too_many_jobs_message(1) == "You have 1 file being processed. Start this one when one of them is done."
    assert jobs.too_many_jobs_message(3).startswith("You have 3 files being processed.")


# ------------------------------------------------------------------------------- sizes of the copies

def test_preview_fits_counts_channel_seconds_up_to_twenty_minutes():
    """The strength slider holds two buffers in memory: up to 1200 channel-seconds."""
    assert jobs.PREVIEW_MAX_CHANNEL_SECONDS == 1200
    assert jobs.preview_fits(1200, 1) and not jobs.preview_fits(1200.5, 1)
    assert jobs.preview_fits(600, 2) and not jobs.preview_fits(601, 2)
    assert jobs.preview_fits(600) and not jobs.preview_fits(601)  # no channel count given: counted as two
    assert jobs.preview_fits(600, 5) and not jobs.preview_fits(601, 5)  # more than two channels count as two
    assert jobs.preview_fits(600, 0) and not jobs.preview_fits(601, 0)  # 0 channels is unknown: counted as two
    assert jobs.preview_fits(0.5, 1)


def test_kept_bytes_is_the_disk_the_copies_take_by_kind_tier_and_length():
    """None for jobs without noise removal; Denoising keeps two copies while the slider can take it, a rebuild one."""
    per = jobs.KEPT_PER_HOUR_PER_CHANNEL
    assert jobs.kept_bytes("transcribe", 3600) == 0
    assert jobs.kept_bytes("denoise", 3600, 2, "off") == 0
    assert jobs.kept_bytes("denoise_transcribe", 3600, 2, "off") == 0
    assert jobs.kept_bytes("denoise", 600, 2, "standard") == int(600 / 3600 * 2 * per * 2)  # short: noise removal + recording
    assert jobs.kept_bytes("denoise", 3600, 2, "standard") == int(3600 / 3600 * 2 * per)  # long: noise removal only
    assert jobs.kept_bytes("denoise", 600, 1, "standard") == int(600 / 3600 * 1 * per * 2)
    assert jobs.kept_bytes("denoise", 600, 2, "rebuild") == int(600 / 3600 * 2 * per)  # the rebuilt voice only
    assert jobs.kept_bytes("denoise_transcribe", 600, 2, "standard") == jobs.kept_bytes("denoise", 600, 2, "standard")
    assert jobs.kept_bytes("denoise", 600, 2, "standard") == 2 * jobs.kept_bytes("denoise", 600, 2, "rebuild")


# --------------------------------------------------------------- what a re-run needs from the worker side

def _job_row(params, kind="denoise", seconds=60.0):
    return {"id": "x" * 32, "kind": kind, "params_json": json.dumps(params), "duration_seconds": seconds, "claimed_seconds": seconds,
            "next_stage": "probe"}


def _rerun_params(clean, transcribe=False, d="standard", t=None, **options):
    base = job_params("denoise_transcribe" if t else "denoise", d, t, options)
    return {**base, "rerun": {"clean": clean, "transcribe": transcribe}}


def test_a_rerun_is_timed_and_sized_for_the_work_it_does():
    """job_seconds / job_needs: a blend is quick and needs no scratch disk, a full re-run is as long as a first run."""
    full = jobs.job_seconds(_job_row(job_params("denoise", "standard", None, {"noise": "light"})))
    same_as_full = jobs.job_seconds(_job_row(_rerun_params("full", noise="light")))
    blend = jobs.job_seconds(_job_row(_rerun_params("denoised", noise="light")))
    rebuilt = jobs.job_seconds(_job_row(_rerun_params("rebuilt", d="rebuild")))
    reuse = jobs.job_seconds(_job_row(_rerun_params("reuse")))
    assert same_as_full == full
    assert reuse == jobs.MODEL_LOAD_SECONDS < rebuilt <= blend < full
    assert blend == jobs.MODEL_LOAD_SECONDS + jobs.SPEED["leveling"] * 60 * 2  # Leveling only
    leveling_alone = jobs.job_seconds(_job_row(job_params("denoise", "off", None, {})))
    assert leveling_alone == jobs.MODEL_LOAD_SECONDS + jobs.SPEED["leveling"] * 60 * 2
    assert jobs.job_seconds(_job_row(_rerun_params("reuse", transcribe=True, t="local"), "denoise_transcribe")) == \
        jobs.MODEL_LOAD_SECONDS + jobs.SPEED["transcribe_local"] * 60
    assert jobs.job_seconds(_job_row(_rerun_params("reuse", transcribe=False, t="local"), "denoise_transcribe")) == jobs.MODEL_LOAD_SECONDS
    assert jobs.job_needs(_job_row(_rerun_params("reuse")))[1] == 0
    assert jobs.job_needs(_job_row(_rerun_params("full", noise="light")))[1] > 0
    mem_reuse = jobs.job_needs(_job_row(_rerun_params("reuse", t="local"), "denoise_transcribe"))[0]
    mem_transcribing = jobs.job_needs(_job_row(_rerun_params("reuse", transcribe=True, t="local"), "denoise_transcribe"))[0]
    assert mem_transcribing == mem_reuse + jobs.LOCAL_ASR_MEM  # the speech model's memory only when it runs


def test_a_blend_rerun_is_cpu_work_and_a_full_rebuild_is_gpu_work():
    """broker.gpu_stages: a re-run only runs what it says, so only a full rebuild / re-transcription goes to the card."""
    from acw import broker

    def stages(params, kind="denoise"):
        return broker.gpu_stages(_job_row(params, kind))

    assert stages(_rerun_params("denoised")) == []
    assert stages(_rerun_params("rebuilt", d="rebuild")) == []
    assert stages(_rerun_params("reuse")) == []
    assert stages(_rerun_params("full", d="rebuild")) == ["rebuild"]
    assert broker.gpu_only(_job_row(_rerun_params("full", d="rebuild")))
    assert not broker.gpu_only(_job_row(_rerun_params("rebuilt", d="rebuild")))
    assert stages(_rerun_params("full")) == ["heavy"]
    assert stages(_rerun_params("full", noise="light")) == []
    assert stages(_rerun_params("reuse", transcribe=True, t="local"), "denoise_transcribe") == ["transcribe"]
    assert stages(_rerun_params("reuse", transcribe=False, t="local"), "denoise_transcribe") == []


def test_the_charge_of_a_rerun_is_its_pinned_rates_and_only_the_new_languages_made():
    """worker._job_cost for a re-run: the steps it ran again, translations only for new languages that were made."""
    from acw.worker import main as worker

    params = _rerun_params("reuse", transcribe=False, t="local", language="en", translate=["fr", "es"])
    params["rates"] = {"translate": 4_000}
    params["rerun"] = {"clean": "reuse", "transcribe": False, "translate_new": ["es"]}
    row = _job_row(params, "denoise_transcribe")
    assert worker._job_cost(row, params, 60) == 4_000  # "fr" was made by the source: not charged
    params["translated"] = ["fr", "es"]
    assert worker._job_cost(row, params, 60) == 4_000
    params["translated"] = ["fr"]  # the new language failed: not charged
    assert worker._job_cost(row, params, 60) == 0
    params["rates"], params["rerun"] = {"leveling": 1_000}, {"clean": "denoised", "transcribe": False, "translate_new": []}
    assert worker._job_cost(row, params, 14) == 234
    params["rates"] = {}
    assert worker._job_cost(row, params, 600) == 0  # a free re-run costs nothing, however long
