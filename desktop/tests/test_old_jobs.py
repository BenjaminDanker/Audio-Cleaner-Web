"""Jobs made before the block cache (whole saved copies noise.flac / rebuilt.flac / input.flac, built here with the old pieces: kept.save +
JobCopies.put_audio): editor_job.make_ready turns them into blocks the first time the job is opened or changed, a job that has blocks is left
alone, one whose copies can't become blocks starts an empty cache, and Change and save again (redo.run / redo.plan) gets the job ready first.
Also the command-line self-test's way of cleaning (cli._clean)."""
from __future__ import annotations

import numpy as np
import pytest

from acw_common import editor, pipeline
from acw_common.timeline import Timeline
from audio_cleaner_desktop import cli, copies, editor_job, editor_session, engine, redo, resources

from old_jobs import SR, old_job, render_whole
from test_redo import first_run, load, no_models, snr, timeline

pytestmark = pytest.mark.native


@pytest.fixture
def clip(clip_wav, tmp_path):
    path = tmp_path / "media" / "talk.wav"
    path.parent.mkdir()
    path.write_bytes(clip_wav.read_bytes())
    return path


def inputs_of(x, job, reference=None):
    return editor.Inputs(x, job.source_id, reference=reference, reference_id="r" if reference is not None else None)


def block_names(job):
    return sorted(p.name for p in (job.dir / copies.BLOCKS_DIR).glob("*.flac"))


# ---------------------------------------------------------------------------------------------------- make_ready

def test_a_job_that_has_blocks_is_left_alone(clip, tmp_path):
    _, job = first_run(clip, tmp_path)
    ff = engine.ffmpeg()
    x = pipeline.decode_for_clean(ff, clip, ff.probe(clip), 600)
    before = (block_names(job), (job.dir / copies.MANIFEST).read_text(), job.editor.get("timeline"))
    work = tmp_path / "work"
    work.mkdir()
    editor_job.make_ready(job, ff, inputs_of(x, job), work)
    assert (block_names(job), (job.dir / copies.MANIFEST).read_text(), job.editor.get("timeline")) == before and list(work.iterdir()) == []


def test_an_old_jobs_noise_copy_becomes_blocks_and_the_copy_is_removed(clip, tmp_path, monkeypatch):
    job, x, full = old_job(clip, tmp_path)
    assert job.has(copies.NOISE) and not job.has_blocks and editor_job.needs_migration(job)
    ff = engine.ffmpeg()
    work = tmp_path / "work"
    work.mkdir()
    editor_job.make_ready(job, ff, inputs_of(x, job), work)
    assert job.has_blocks and not job.has(copies.NOISE) and not (job.dir / "noise.flac").exists() and not editor_job.needs_migration(job)
    assert copies.JobCopies.load(job.dir).kinds() == [copies.BLOCKS]  # written, not just in memory
    assert block_names(job) and list(work.iterdir()) == []  # (and the scratch files are gone)
    # the blocks are the copy's samples: the job's whole-file timeline needs nothing, and it renders the copy at any strength with no model
    tl = Timeline.whole_file(3.0, noise={"strength_db": 100.0, "remover": "light"}, leveling=False)
    assert editor.plan(tl, inputs_of(x, job), job.block_cache(ff)).misses == []
    no_models(monkeypatch)
    got = editor.render(tl, inputs_of(x, job), job.block_cache(ff))
    assert float(np.max(np.abs(got - full))) < 2e-7 * 4  # (one 24-bit step)
    softer = editor.render(Timeline.from_json(timeline(24.0), seconds=3.0), inputs_of(x, job), job.block_cache(ff))
    a = pipeline.strength_mix(24.0)
    assert float(np.max(np.abs(softer - ((1 - a) * full.astype(np.float64) + a * x).astype(np.float32)))) < 1e-5


def test_an_old_jobs_rebuilt_copy_becomes_blocks_too(clip, tmp_path, monkeypatch):
    ff = engine.ffmpeg()
    x = pipeline.decode_for_clean(ff, clip, ff.probe(clip), 600)
    rebuilt = (x * 0.5).astype(np.float32)  # (what the rebuilt voice was: made up, it only has to come back as it was)
    job, _, _ = old_job(clip, tmp_path, noise=False, rebuilt=rebuilt)
    assert job.has(copies.REBUILT) and editor_job.needs_migration(job)
    work = tmp_path / "work"
    work.mkdir()
    editor_job.make_ready(job, ff, inputs_of(x, job), work)
    assert job.has_blocks and not job.has(copies.REBUILT) and not (job.dir / "rebuilt.flac").exists()
    tl = Timeline.whole_file(3.0, noise=False, rebuild=True, leveling=False)
    assert editor.plan(tl, inputs_of(x, job), job.block_cache(ff)).misses == []
    no_models(monkeypatch)
    assert float(np.max(np.abs(editor.render(tl, inputs_of(x, job), job.block_cache(ff)) - rebuilt))) < 1e-5


def test_a_job_whose_copies_cannot_become_blocks_starts_an_empty_cache(clip, tmp_path):
    """An echo job whose recording without the echo (input.flac) was not kept: blocks made from the copy could not be found again."""
    job, x, full = old_job(clip, tmp_path, made={"echo": True})
    job.data["echo"] = True
    job.save()
    assert job.has(copies.NOISE) and not job.has(copies.INPUT) and editor_job.needs_migration(job)
    ff = engine.ffmpeg()
    work = tmp_path / "work"
    work.mkdir()
    editor_job.make_ready(job, ff, inputs_of(x, job, reference=x[:, 0].copy()), work)
    assert not job.has(copies.NOISE) and not (job.dir / "noise.flac").exists()  # the old copies are dropped
    assert job.echo is True and block_names(job) == [] and job.data["copies"][copies.BLOCKS]["source_id"] == job.source_id  # (an empty cache, kept)
    assert copies.BLOCKS in copies.JobCopies.load(job.dir).data["copies"] and not editor_job.needs_migration(job)
    # what the job's timeline needs is made from the original
    tl = Timeline.whole_file(3.0, echo=True, noise={"strength_db": 100.0, "remover": "light"}, leveling=False)
    assert editor.plan(tl, inputs_of(x, job, reference=x[:, 0].copy()), job.block_cache(ff)).misses


def test_a_job_with_no_copies_at_all_gets_an_empty_cache(clip, tmp_path):
    job, x, _ = old_job(clip, tmp_path, noise=False)
    assert not job.has_blocks and not editor_job.needs_migration(job)
    ff = engine.ffmpeg()
    work = tmp_path / "work"
    work.mkdir()
    editor_job.make_ready(job, ff, inputs_of(x, job), work)
    assert copies.BLOCKS in job.data["copies"] and job.samples == len(x) and job.echo is False


def test_the_session_opens_an_old_job_ready(clip, tmp_path):
    job, x, full = old_job(clip, tmp_path)
    session = editor_session.Session.open(job, scratch_parent=tmp_path / "scratch")
    try:
        assert job.has_blocks and not job.has(copies.NOISE)
        assert session.mix is not None and session.mix.frames == len(x) and abs(session.mix.read(0, 4800)).max() > 0
        assert session.quote(editor_job.baseline_timeline(job)).nothing
    finally:
        session.close()


# ---------------------------------------------------------------------------------------------------- redo on an old job

def test_the_plan_of_an_old_job_says_its_copies_get_ready(clip, tmp_path):
    job, x, _ = old_job(clip, tmp_path)
    wanted = redo.Wanted(timeline=timeline(24.0), output_name="x.wav")
    p = redo.plan(job, wanted, job.check_original())
    assert p.ok and p.slow and p.steps[0].startswith("Get this job's saved copies ready")
    # a job that keeps blocks says what changed instead
    _, fresh = first_run(clip, tmp_path / "fresh")
    q = redo.plan(fresh, wanted, fresh.check_original())
    assert "saved copies ready" not in " ".join(q.steps)


def test_changing_an_old_job_gets_it_ready_and_then_renders_with_no_model(clip, tmp_path, monkeypatch):
    job, x, full = old_job(clip, tmp_path)
    no_models(monkeypatch)  # the copy is the noise removal: nothing runs a model
    out = redo.run(job, redo.Wanted(timeline=timeline(24.0), output_name="migrated.wav"))
    a = pipeline.strength_mix(24.0)
    expected = ((1 - a) * full[:, 0].astype(np.float64) + a * x[:, 0]).astype(np.float32)
    heard = load(out.output)
    assert float(np.max(np.abs(heard - expected[: len(heard)]))) < 1e-5
    assert job.has_blocks and not job.has(copies.NOISE) and job.made["atten_lim_db"] == 24.0 and len(job.saves) == 1
    assert copies.JobCopies.load(job.dir).has_blocks


def test_changing_an_echo_job_without_its_input_copy_starts_an_empty_cache_and_still_gives_a_result(clip, tmp_path):
    job, x, full = old_job(clip, tmp_path, made={"echo": True})
    job.data["echo"] = True
    job.save()
    out = redo.run(job, redo.Wanted(timeline=timeline(24.0), output_name="again.wav"))
    assert out.output.is_file() and job.has_blocks and not job.has(copies.NOISE) and block_names(job)  # made from the original
    fresh = engine.process(clip, engine.JobOptions(noise="light", strength="balanced", leveling=False, audio_format="wav"),
                           out_dir=tmp_path / "fresh", output_name="fresh.wav")
    assert snr(load(fresh.output), load(out.output)) > 80


def test_a_changed_original_stops_an_old_job_before_anything_is_touched(clip, tmp_path):
    job, x, _ = old_job(clip, tmp_path)
    clip.write_bytes(clip.read_bytes()[:-3000])
    with pytest.raises(Exception, match="changed|different"):
        redo.run(job, redo.Wanted(timeline=timeline(24.0), output_name="x.wav"))
    assert job.has(copies.NOISE) and not job.has_blocks  # nothing was migrated


# ---------------------------------------------------------------------------------------------------- the command line

@pytest.mark.skipif(not resources.noise_ready(), reason="the noise removal models are not installed")
def test_the_self_tests_clean_is_the_editors_render_of_the_whole_file_timeline():
    rng = np.random.default_rng(5)
    t = np.arange(SR) / SR
    x = (0.2 * np.sin(2 * np.pi * 220 * t) + 0.03 * rng.standard_normal(t.size)).astype(np.float32)
    y = cli._clean(x, "light", leveling=False)
    assert y.shape == (x.size, 1) and y.dtype == np.float32 and np.isfinite(y).all()
    assert np.array_equal(y, render_whole(x[:, None].copy(), noise={"strength_db": 100.0, "remover": "light"}, leveling=False))
    leveled = cli._clean(x, "light")  # (Leveling is on unless asked not to be)
    assert leveled.shape == y.shape and not np.array_equal(leveled, y)
    if resources.heavy_noise_model_path().is_file():
        assert cli._clean(x, "heavy", leveling=False).shape == (x.size, 1)
