"""Change and save again of an OLD job (redo.py: one made before the per-part editor, with the whole copies noise.flac / rebuilt.flac /
input.flac; `copies.KEEP_BLOCKS` is off here so the engine writes them as it did; jobs made now keep blocks, test_editor_job.py). With the real
engine and the real noise removal on a short clip: the saved copies a job keeps, the
strength blend equals a fresh run at that strength, later steps rerun without a model, the steps that do run a model (Voice rebuild
on, noise removal in a job that has none), subtitles from the saved transcript, naming and replacing, and every way the original or the
copies can be unusable."""
from __future__ import annotations

import errno
import threading
from pathlib import Path

import numpy as np
import pytest

from acw_common import captions, dfn, dpdfnet, kept, pipeline, restore
from acw_common.ffmpeg import MediaError
from audio_cleaner_desktop import blend, copies, engine, redo, resources

SR = pipeline.SAMPLE_RATE


@pytest.fixture(autouse=True)
def old_style_copies(monkeypatch):
    monkeypatch.setattr(copies, "KEEP_BLOCKS", False)


def load(path) -> np.ndarray:
    ff = engine.ffmpeg()
    return ff.decode(path, ff.probe(path), SR, 1, 600)[:, 0]


def snr(ref: np.ndarray, test: np.ndarray) -> float:
    n = min(len(ref), len(test))
    err = float(np.sum((ref[:n].astype(np.float64) - test[:n]) ** 2))
    return 300.0 if err == 0 else 10 * np.log10(float(np.sum(ref[:n].astype(np.float64) ** 2)) / err)


def first_run(clip, tmp_path, **opts):
    """Clean the clip with copies kept; returns (result, the job's copies)."""
    opts = {"noise": "light", "leveling": False, "audio_format": "wav", **opts}
    keep = copies.new_job_dir(tmp_path / "jobs")
    result = engine.process(clip, engine.JobOptions(**opts), out_dir=tmp_path / "out", keep_dir=keep)
    return result, copies.JobCopies.load(keep)


@pytest.fixture
def clip(clip_wav, tmp_path):
    """The clip copied where tests may change it."""
    path = tmp_path / "media" / "talk.wav"
    path.parent.mkdir()
    path.write_bytes(clip_wav.read_bytes())
    return path


def no_models(monkeypatch):
    """Any model run fails the test: what is measured is that none ran."""
    def boom(*a, **k):
        raise AssertionError("a model ran")

    for mod in (dfn, dpdfnet):
        monkeypatch.setattr(mod, "denoise_array", boom)
    monkeypatch.setattr(restore, "restore", boom)
    monkeypatch.setattr(pipeline, "clean_audio", boom)


@pytest.mark.native
def test_a_cleaned_file_keeps_the_noise_removal_and_what_it_needs_to_be_changed(clip, tmp_path):
    result, job = first_run(clip, tmp_path, strength="strong")
    assert result.kept_dir == job.dir and result.output.is_file()
    assert job.kinds() == [copies.NOISE] and job.has(copies.NOISE) and not job.has(copies.REBUILT)
    entry = job.entry(copies.NOISE)
    assert entry["plan"] == [0] and entry["profile"] == "light" and entry["samples"] == job.samples > 0
    assert job.original_path == clip.absolute() and job.check_original().ok
    assert job.made["atten_lim_db"] == 40.0 and job.made["strength"] == "strong" and job.made["rebuilt"] is False
    assert job.saves[0]["files"] == [str(result.output)] and result.output.parent != job.dir
    assert job.media["has_video"] is False and job.media["duration"] == pytest.approx(3.0, abs=0.1)
    # only the copies are in the folder: neither the original nor the result was copied
    assert sorted(p.name for p in job.dir.iterdir()) == [copies.MANIFEST, "noise.flac"]
    assert job.bytes() == sum(p.stat().st_size for p in job.dir.iterdir())
    # about a third to half of the 32-bit float it is made of (24-bit FLAC)
    assert job.sizes()[0][1] < job.samples * 4


@pytest.mark.native
@pytest.mark.parametrize("leveling", [False, True])
def test_the_strength_blend_equals_a_fresh_run_at_that_strength(clip, tmp_path, leveling, monkeypatch):
    _, job = first_run(clip, tmp_path, strength="max", leveling=leveling)
    fresh = engine.process(clip, engine.JobOptions(noise="light", strength="balanced", leveling=leveling, audio_format="wav"),
                           out_dir=tmp_path / "fresh", output_name="fresh.wav")
    no_models(monkeypatch)  # the change below runs none
    changed = redo.run(job, redo.Wanted(atten_lim_db=24.0, leveling=leveling, output_name="changed.wav"))
    a, b = load(fresh.output), load(changed.output)
    assert len(a) == len(b)
    if not leveling:
        assert float(np.max(np.abs(a - b))) < 1e-5  # float error: the saved copy is 24-bit
    assert snr(a, b) > 80
    assert changed.output.name == "changed.wav" and changed.output.parent == Path(job.saves[0]["files"][0]).parent  # next to the first result


@pytest.mark.native
def test_a_change_between_the_names_and_in_between_is_the_blend_too(clip, tmp_path, monkeypatch):
    _, job = first_run(clip, tmp_path, strength="max")
    db = blend.db_at(1.5)  # between Balanced and Strong
    x = load(clip)
    ff = engine.ffmpeg()
    full = np.zeros((len(x), 1), dtype=np.float32)
    kept.load(ff, job.entry(copies.NOISE), job.dir, full, tmp_path / "scratch.f32")
    no_models(monkeypatch)
    changed = redo.run(job, redo.Wanted(atten_lim_db=db, leveling=False, output_name="between.wav"))
    expected = pipeline.blend_strength(full.copy(), x[:, None], db)[:, 0]
    assert float(np.max(np.abs(load(changed.output) - expected[: len(load(changed.output))]))) < 1e-5
    assert job.made["atten_lim_db"] == pytest.approx(db) and job.made["strength"] is None


@pytest.mark.native
def test_leveling_loudness_type_and_name_change_without_a_model_and_each_save_is_a_new_file(clip, tmp_path, monkeypatch):
    first, job = first_run(clip, tmp_path, strength="gentle", leveling=True)
    no_models(monkeypatch)
    out = redo.run(job, redo.Wanted(atten_lim_db=12.0, leveling=True, loudness="youtube", output_name="talk-cleaned (2).mp3"))
    assert out.output.suffix == ".mp3" and out.output != first.output and first.output.is_file()
    again = redo.run(job, redo.Wanted(atten_lim_db=12.0, leveling=False, output_name="talk-cleaned.wav"))  # the first one's name is taken
    assert again.output.name == "talk-cleaned (2).wav" and first.output.is_file()
    assert len(job.saves) == 3 and job.saves[-1]["files"] == [str(again.output)]
    assert job.made["leveling"] is False and job.made["loudness"] == "podcast"
    assert copies.JobCopies.load(job.dir).made["loudness"] == "podcast"  # written, not just in memory


@pytest.mark.native
def test_saving_with_the_name_of_an_earlier_result_replaces_it_only_when_asked_and_through_the_bin(clip, tmp_path, monkeypatch):
    first, job = first_run(clip, tmp_path, strength="max")
    old_bytes = first.output.read_bytes()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    trashed = []

    def trash(path):
        trashed.append(path.name)
        path.rename(bin_dir / path.name)
        return True

    # not asked to replace: the new file takes another name, nothing moves
    plain = redo.run(job, redo.Wanted(atten_lim_db=12.0, leveling=False, output_name=first.output.name), trash=trash)
    assert plain.output.name == "talk-cleaned (2).wav" and first.output.read_bytes() == old_bytes and not trashed
    # asked: the earlier one goes to the bin and the new one gets its name
    replaced = redo.run(job, redo.Wanted(atten_lim_db=12.0, leveling=False, output_name=first.output.name),
                        replace=first.output, trash=trash)
    assert replaced.output == first.output and trashed == ["talk-cleaned.wav"] and first.output.read_bytes() != old_bytes
    assert (bin_dir / "talk-cleaned.wav").read_bytes() == old_bytes and plain.output.is_file()
    # a bin that does not work: nothing is lost, the new file keeps its own name
    kept_name = redo.run(job, redo.Wanted(atten_lim_db=40.0, leveling=False, output_name=first.output.name),
                         replace=first.output, trash=lambda p: False)
    assert kept_name.output != first.output and first.output.is_file() and kept_name.output.is_file()


@pytest.mark.native
def test_a_moved_or_changed_original_makes_every_change_unavailable_with_the_reason(clip, tmp_path):
    _, job = first_run(clip, tmp_path, strength="max")
    wanted = redo.Wanted(atten_lim_db=24.0, output_name="x.wav")
    check = job.check_original()
    assert redo.plan(job, wanted, check, voice_model=True, speech_model=True).ok
    # changed (same length, other sound)
    data = bytearray(clip.read_bytes())
    for i in range(2000, len(data), 331):
        data[i] ^= 0x55
    clip.write_bytes(bytes(data))
    changed = job.check_original()
    assert changed.state == "changed"
    p = redo.plan(job, wanted, changed)
    assert not p.ok and "changed" in p.problems[0] and "original" in p.problems[0]
    with pytest.raises(MediaError, match="changed"):
        redo.run(job, wanted)
    with pytest.raises(MediaError, match="changed"):
        redo.prepare_stream(job)
    # moved
    moved = clip.with_name("moved.wav")
    clip.rename(moved)
    assert job.check_original().state == "missing" and "not at" in redo.plan(job, wanted, job.check_original()).problems[0]
    with pytest.raises(MediaError, match="not at"):
        redo.run(job, wanted)
    assert not list((tmp_path / "out").glob(".acw-*")) and sorted(p.name for p in (tmp_path / "out").iterdir()) == ["talk-cleaned.wav"]


@pytest.mark.native
def test_an_original_of_another_length_is_refused_even_if_it_slips_past_the_fingerprint(clip, tmp_path):
    _, job = first_run(clip, tmp_path, strength="max")
    job.data["samples"] = job.samples + 480
    with pytest.raises(MediaError, match="not the same length"):
        redo.run(job, redo.Wanted(output_name="x.wav"))


@pytest.mark.native
def test_the_live_stream_is_the_saved_blend(clip, tmp_path):
    _, job = first_run(clip, tmp_path, strength="max", leveling=False)
    prepared = redo.prepare_stream(job, scratch_parent=tmp_path / "scratch", exact=True)
    try:
        stream = prepared.stream
        assert stream.frames == job.samples and stream.channels == 1 and stream.exact
        saved = redo.run(job, redo.Wanted(atten_lim_db=24.0, leveling=False, output_name="b.wav"))
        stream.set_strength(24.0)
        heard = np.concatenate([stream.read(s, 1000)[:, 0] for s in range(0, stream.frames, 1000)])
        assert float(np.max(np.abs(heard - load(saved.output)[: len(heard)]))) < 1e-5  # what you hear is what you save
        stream.set_strength(100.0)
        full = stream.read(0, stream.frames)[:, 0]
        assert not np.allclose(heard, full)
    finally:
        prepared.close()
    assert not prepared.work.exists()


@pytest.mark.native
def test_a_stopped_change_leaves_nothing_behind_and_the_copies_stay(clip, tmp_path):
    first, job = first_run(clip, tmp_path, strength="max")
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(engine.Cancelled):
        redo.run(job, redo.Wanted(atten_lim_db=24.0, output_name="never.wav"), cancel=cancel)
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [first.output.name]
    assert job.has(copies.NOISE) and len(job.saves) == 1


# ---------------------------------------------------------------------------------------------------- Voice rebuild

@pytest.fixture
def fake_voice(monkeypatch, tmp_path):
    """The voice rebuild stands in as a plain gain (the real model is not part of the test setup)."""
    calls = []
    voice = tmp_path / "voice-model"
    voice.mkdir()
    monkeypatch.setattr(resources, "voice_model_dir", lambda: voice)
    monkeypatch.setattr(restore, "available", lambda d: True)

    def fake(x, model_dir, **kw):
        calls.append(len(x))
        out = kw.get("out")
        y = (np.asarray(x, dtype=np.float32) * 0.5)
        if out is not None:
            out[:] = y if out.ndim == y.ndim else y[:, 0]
            return out
        return y

    monkeypatch.setattr(restore, "restore", fake)
    return calls


@pytest.mark.native
def test_voice_rebuild_on_runs_that_step_once_and_keeps_its_result_for_the_next_change(clip, tmp_path, fake_voice, monkeypatch):
    _, job = first_run(clip, tmp_path, strength="max")
    assert not job.has(copies.REBUILT)
    wanted = redo.Wanted(rebuild_voice=True, leveling=False, output_name="rebuilt.wav")
    p = redo.plan(job, wanted, job.check_original(), voice_model=True)
    assert p.sound == redo.SOUND_REBUILD and p.slow
    out = redo.run(job, wanted)
    assert fake_voice == [job.samples] and job.has(copies.REBUILT) and job.has(copies.NOISE)  # the noise removal is kept too
    assert job.made["rebuilt"] is True and job.made["atten_lim_db"] is None
    assert float(np.max(np.abs(load(out.output) - 0.5 * load(clip)[: len(load(out.output))]))) < 1e-4
    # changing Leveling of the rebuilt voice does not rebuild it again
    no_models(monkeypatch)
    again = redo.Wanted(rebuild_voice=True, leveling=True, loudness="broadcast", output_name="rebuilt-2.wav")
    assert redo.plan(job, again, job.check_original()).sound == redo.SOUND_LEVEL_REBUILT
    redo.run(job, again)
    assert fake_voice == [job.samples]
    # Voice rebuild off again: back to the saved noise removal, no model
    off = redo.run(job, redo.Wanted(atten_lim_db=24.0, leveling=False, output_name="plain.wav"))
    assert job.made["rebuilt"] is False and off.output.is_file() and fake_voice == [job.samples]


@pytest.mark.native
def test_voice_rebuild_needs_its_model_and_says_so(clip, tmp_path, monkeypatch):
    _, job = first_run(clip, tmp_path, strength="max")
    monkeypatch.setattr(resources, "voice_model_dir", lambda: None)
    wanted = redo.Wanted(rebuild_voice=True, output_name="r.wav")
    p = redo.plan(job, wanted, job.check_original(), voice_model=False)
    assert not p.ok and "rebuilds the voice" in p.problems[0]
    with pytest.raises(MediaError, match="rebuilds the voice"):
        redo.run(job, wanted)


@pytest.mark.native
def test_voice_rebuild_off_in_a_job_that_rebuilt_runs_the_noise_removal_and_keeps_it(clip, tmp_path, fake_voice):
    result, job = first_run(clip, tmp_path, rebuild_voice=True)
    assert job.has(copies.REBUILT) and not job.has(copies.NOISE) and job.made["rebuilt"] is True
    # nothing to slide: a rebuilt job has no noise removal to blend
    assert not redo.can_slide(job)
    wanted = redo.Wanted(atten_lim_db=24.0, leveling=False, output_name="plain.wav")
    p = redo.plan(job, wanted, job.check_original())
    assert p.sound == redo.SOUND_NOISE and p.slow
    out = redo.run(job, wanted)
    assert job.has(copies.NOISE) and job.has(copies.REBUILT) and redo.can_slide(job)  # now it can be changed cheaply
    fresh = engine.process(clip, engine.JobOptions(noise="light", strength="balanced", leveling=False, audio_format="wav"),
                           out_dir=tmp_path / "fresh", output_name="fresh.wav")
    assert snr(load(fresh.output), load(out.output)) > 80


# ---------------------------------------------------------------------------------------------------- subtitles

SEGMENTS = [captions.Segment(0.0, 1.2, "hello there", [captions.Word(0.0, 0.6, "hello"), captions.Word(0.6, 1.2, "there")]),
            captions.Segment(1.4, 2.6, "good morning", [captions.Word(1.4, 2.0, "good"), captions.Word(2.0, 2.6, "morning")])]


@pytest.mark.native
def test_subtitles_come_from_the_saved_transcript_without_listening_again(clip, tmp_path, monkeypatch):
    _, job = first_run(clip, tmp_path, strength="max")
    job.put_transcript(SEGMENTS, "en")
    monkeypatch.setattr(engine, "_transcribe", lambda *a, **k: pytest.fail("listened again"))
    monkeypatch.setattr(resources, "speech_ready", lambda: False)  # not even the speech model is needed
    wanted = redo.Wanted(atten_lim_db=24.0, leveling=False, subtitles="srt", output_name="with-subs.wav")
    p = redo.plan(job, wanted, job.check_original(), speech_model=False)
    assert p.ok and not p.transcribe
    out = redo.run(job, wanted)
    assert out.subtitles == out.output.with_suffix(".srt") and "hello there" in out.subtitles.read_text()
    assert job.made["subtitles"] == "srt" and job.saves[-1]["files"] == [str(out.output), str(out.subtitles)]
    # a language other than the saved one is listened to again, which needs the speech model
    other = redo.plan(job, redo.Wanted(subtitles="srt", language="de", output_name="x.wav"), job.check_original(), speech_model=False)
    assert other.transcribe and not other.ok and "speech model" in other.problems[0]


@pytest.mark.native
def test_subtitles_a_job_never_made_listen_to_the_original_and_keep_the_transcript(clip, tmp_path, monkeypatch):
    _, job = first_run(clip, tmp_path, strength="max")
    assert not job.has(copies.TRANSCRIPT)
    heard = []

    def fake(ff, src, info, work, options, threads, progress, cancel, echo_free=None):
        heard.append(src)
        progress(1.0)
        return list(SEGMENTS), "en"

    monkeypatch.setattr(engine, "_transcribe", fake)
    monkeypatch.setattr(resources, "speech_ready", lambda: True)
    out = redo.run(job, redo.Wanted(atten_lim_db=24.0, leveling=False, subtitles="vtt", output_name="s.wav"))
    assert heard == [clip.absolute()] and out.subtitles.suffix == ".vtt" and "good morning" in out.subtitles.read_text()
    assert job.has(copies.TRANSCRIPT) and job.made["language"] == "en"
    redo.run(job, redo.Wanted(atten_lim_db=24.0, leveling=False, subtitles="txt", output_name="t.wav"))  # the next kind needs no listening
    assert len(heard) == 1


@pytest.mark.native
def test_subtitle_files_alone_need_no_original_and_no_model(clip, tmp_path):
    _, job = first_run(clip, tmp_path, strength="max")
    job.put_transcript(SEGMENTS, "en")
    clip.unlink()  # the original is gone
    assert job.check_original().state == "missing"
    folder = tmp_path / "out"
    srt = redo.save_subtitles(job, "srt", folder, "talk-cleaned")
    again = redo.save_subtitles(job, "srt", folder, "talk-cleaned")
    txt = redo.save_subtitles(job, "txt", folder, "talk-cleaned")
    assert (srt.name, again.name, txt.name) == ("talk-cleaned.srt", "talk-cleaned (2).srt", "talk-cleaned.txt")
    assert "hello there" in srt.read_text() and "-->" in srt.read_text() and "hello there" in txt.read_text()
    with pytest.raises(ValueError):
        redo.save_subtitles(job, "doc", folder, "x")


@pytest.mark.native
def test_subtitles_in_a_job_with_no_transcript_cannot_be_saved_alone(clip, tmp_path):
    _, job = first_run(clip, tmp_path, strength="max")
    with pytest.raises(MediaError, match="no saved transcript"):
        redo.save_subtitles(job, "srt", tmp_path / "out", "x")


@pytest.mark.native
def test_a_job_that_made_subtitles_keeps_the_transcript_with_its_first_run(clip, tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "_transcribe", lambda ff, src, info, work, options, threads, progress, cancel, echo_free=None: (list(SEGMENTS), "en"))
    monkeypatch.setattr(resources, "speech_ready", lambda: True)
    result, job = first_run(clip, tmp_path, strength="max", subtitles="srt")
    assert result.subtitles is not None and job.has(copies.TRANSCRIPT) and job.transcript()[1] == "en"
    assert job.made["subtitles"] == "srt" and job.made["language"] == "en"
    assert job.saves[0]["files"] == [str(result.output), str(result.subtitles)]


# ---------------------------------------------------------------------------------------------------- failed, stopped, no room

@pytest.mark.native
def test_a_job_that_fails_after_its_noise_removal_keeps_the_copy_and_can_be_saved_again(clip, tmp_path, monkeypatch):
    keep = copies.new_job_dir(tmp_path / "jobs")
    real = engine._write

    def fail(*a, **k):
        raise MediaError("Saving the cleaned audio failed.")

    monkeypatch.setattr(engine, "_write", fail)
    with pytest.raises(MediaError, match="Saving"):
        engine.process(clip, engine.JobOptions(noise="light", strength="max", leveling=False, audio_format="wav"),
                       out_dir=tmp_path / "out", keep_dir=keep)
    job = copies.JobCopies.load(keep)
    assert job is not None and job.has(copies.NOISE) and job.saves == []
    assert not [p for p in (tmp_path / "out").iterdir() if p.suffix == ".wav"]  # no partial result
    monkeypatch.setattr(engine, "_write", real)  # the disk has room again
    out = redo.run(job, redo.Wanted(atten_lim_db=40.0, leveling=False, output_name="late.wav"))
    assert out.output.is_file() and len(job.saves) == 1


@pytest.mark.native
def test_a_job_that_fails_before_any_copy_leaves_no_folder(clip, tmp_path, monkeypatch):
    keep = copies.new_job_dir(tmp_path / "jobs")

    def fail(*a, **k):
        raise MediaError("Decoding the audio failed.")

    monkeypatch.setattr(pipeline, "decode_for_clean", fail)
    with pytest.raises(MediaError):
        engine.process(clip, engine.JobOptions(), out_dir=tmp_path / "out", keep_dir=keep)
    assert not keep.exists()


@pytest.mark.native
def test_a_stopped_job_keeps_nothing(clip, tmp_path):
    keep = copies.new_job_dir(tmp_path / "jobs")
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(engine.Cancelled):
        engine.process(clip, engine.JobOptions(), out_dir=tmp_path / "out", keep_dir=keep, cancel=cancel)
    assert not keep.exists() and not list((tmp_path / "out").glob("*.wav"))


@pytest.mark.native
def test_a_copy_that_cannot_be_written_never_fails_the_job_and_the_notice_says_so(clip, tmp_path, monkeypatch):
    def full(*a, **k):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(kept, "save", full)
    keep = copies.new_job_dir(tmp_path / "jobs")
    result = engine.process(clip, engine.JobOptions(noise="light", strength="max", audio_format="wav"), out_dir=tmp_path / "out", keep_dir=keep)
    assert result.output.is_file() and result.kept_dir is None and not keep.exists()
    assert "can't be changed and saved again" in result.notice and "disk is full" in result.notice


@pytest.mark.native
def test_without_a_folder_for_copies_the_engine_behaves_as_before(clip, tmp_path):
    result = engine.process(clip, engine.JobOptions(noise="light", strength="max", audio_format="wav"), out_dir=tmp_path / "out")
    assert result.kept_dir is None and result.output.is_file() and not (tmp_path / "jobs").exists()


def test_the_next_name_is_free_and_next_to_the_last_result(tmp_path):
    folder = tmp_path / "results"
    folder.mkdir()
    original = tmp_path / "talk.wav"
    original.write_bytes(b"x")
    job = copies.JobCopies.create(copies.new_job_dir(tmp_path / "jobs"), original, copies.fingerprint(original), {})
    source = engine.Source(False, ".wav", None, 3.0, 1)
    assert redo.suggest_name(job, source) == "talk-cleaned.wav"
    (folder / "talk-cleaned.mp3").write_bytes(b"1")
    job.add_save([folder / "talk-cleaned.mp3"], {})
    assert redo.suggest_name(job, source) == "talk-cleaned (2).mp3" and redo.result_dir(job) == folder
    (folder / "talk-cleaned (2).mp3").write_bytes(b"2")
    job.add_save([folder / "talk-cleaned (2).mp3"], {})
    assert redo.suggest_name(job, source) == "talk-cleaned (3).mp3"
