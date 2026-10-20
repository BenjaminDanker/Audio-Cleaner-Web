"""The per-part editor of a desktop job, with the real engine and the real noise removal on short clips: a normal run fills the job's block
cache, an edit starts from it (no model runs for a new strength), the edited sound equals the core's own render, a run with a timeline
processes only its parts, the CLI takes a timeline file, old jobs (saved copies) are turned into blocks, and deleting a job never touches
the results or the original. The Voice rebuild model is not installed here: that step is stood in for where it matters."""
from __future__ import annotations

import json

import numpy as np
import pytest

from acw_common import blocks, dfn, dpdfnet, editor, kept, pipeline, restore
from acw_common.timeline import Timeline
from audio_cleaner_desktop import __main__ as app_main
from audio_cleaner_desktop import cli, copies, editor_job, editor_session, engine, redo, resources

from conftest import bundled_ffmpeg, run_ffmpeg

SR = pipeline.SAMPLE_RATE


def noisy(seconds, seed=2):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    return (0.2 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 2 * t) > 0) + 0.03 * rng.standard_normal(t.size)).astype(np.float32)


@pytest.fixture
def clip(clip_wav, tmp_path):
    path = tmp_path / "media" / "talk.wav"
    path.parent.mkdir()
    path.write_bytes(clip_wav.read_bytes())
    return path


@pytest.fixture
def long_clip(tmp_path):
    """Eight seconds, long enough for several blocks when the grid is made small."""
    from scipy.io import wavfile

    path = tmp_path / "media" / "long.wav"
    path.parent.mkdir()
    wavfile.write(path, SR, (noisy(8) * 32767).astype(np.int16))
    return path


def first_run(clip, tmp_path, **opts):
    opts = {"noise": "light", "leveling": False, "audio_format": "wav", "strength": "max", **opts}
    keep = copies.new_job_dir(tmp_path / "jobs")
    result = engine.process(clip, engine.JobOptions(**opts), out_dir=tmp_path / "out", keep_dir=keep)
    return result, copies.JobCopies.load(keep)


def no_models(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a model ran")

    for mod in (dfn, dpdfnet):
        monkeypatch.setattr(mod, "denoise_array", boom)
        monkeypatch.setattr(mod, "run_windows", boom)
    monkeypatch.setattr(restore, "restore", boom)
    monkeypatch.setattr(restore, "restore_blocks", boom)
    monkeypatch.setattr(pipeline, "clean_audio", boom)


def decode(path):
    ff = engine.ffmpeg()
    return ff.decode(path, ff.probe(path), SR, 1, 600)[:, 0]


def block_count(job):
    return len(list((job.dir / copies.BLOCKS_DIR).glob("*.flac")))


@pytest.mark.native
def test_a_normal_run_fills_the_block_cache_instead_of_keeping_whole_copies(clip, tmp_path):
    _, job = first_run(clip, tmp_path)
    assert job is not None and job.has_blocks and job.kinds() == [copies.BLOCKS]
    assert not (job.dir / "noise.flac").exists() and not job.has(copies.NOISE) and block_count(job) >= 1
    ed = job.editor
    assert ed["facts"]["n"] == job.samples == 3 * SR and ed["facts"]["plan"] == [0] and len(ed["peaks"]) > 100
    assert job.source_id == copies.source_id_of(copies.fingerprint(clip)) and dict(job.sizes())[copies.BLOCKS] > 0
    # the whole-file timeline the job was made with needs nothing more
    tl = editor_job.baseline_timeline(job)
    assert tl.layers["noise"][0].settings == {"strength_db": 100.0, "remover": "light"} and not tl.layers["rebuild"]
    q = editor_job.quote_for(job, tl)
    assert q is not None and q.nothing and q.steps == {}


@pytest.mark.native
def test_an_edit_reuses_the_cache_so_a_new_strength_runs_no_model_and_equals_the_cores_render(clip, tmp_path, monkeypatch):
    _, job = first_run(clip, tmp_path)
    # the cores own render of the same timeline from nothing (the models run)
    ff = engine.ffmpeg()
    tl = Timeline.whole_file(3.0, noise={"strength_db": 24.0, "remover": "light"}, leveling={"loudness_lufs": -16.0})
    x = pipeline.decode_for_clean(ff, clip, ff.probe(clip), 600)
    fresh = editor.render(tl, editor.Inputs(x, "other"), blocks.MemoryCache(), editor_job.runner(2))
    no_models(monkeypatch)
    wanted = redo.Wanted(atten_lim_db=24.0, leveling=True, output_name="again.wav")
    p = redo.plan(job, wanted, job.check_original(), voice_model=False)
    assert p.sound == redo.SOUND_EDITOR and not p.slow and p.ok
    out = redo.run(job, wanted)
    got = np.asarray(decode(out.output))
    assert job.made["atten_lim_db"] == 24.0 and job.made["strength"] == "balanced" and job.editor["timeline"]["layers"]["noise"][0]["strength_db"] == 24.0
    assert abs(len(got) - len(fresh)) < 2 and float(np.max(np.abs(got[:len(fresh)] - fresh[:len(got), 0]))) < 2e-4
    # a later look at the job: the result is the draft's start now, and nothing is left to do
    q = editor_job.quote_for(job, editor_job.current_timeline(job))
    assert q.nothing


@pytest.mark.native
def test_a_run_with_a_timeline_processes_only_its_parts_and_leaves_the_rest_as_it_was(long_clip, tmp_path, monkeypatch):
    monkeypatch.setattr(blocks, "GRID_LIGHT", blocks.Grid(1.0, 0.5, 0.1))
    tl = Timeline(8.0)
    tl.paint("noise", 3.0, 5.0, {"strength_db": 100.0, "remover": "light"})
    keep = copies.new_job_dir(tmp_path / "jobs")
    result = engine.process(long_clip, engine.JobOptions(audio_format="wav"), out_dir=tmp_path / "out", keep_dir=keep, timeline=tl.to_json())
    job = copies.JobCopies.load(keep)
    assert job.has_blocks and 1 <= block_count(job) <= 5 < 8  # not the whole file's eight blocks
    assert job.made["parts"] is True and job.editor["timeline"]["layers"]["noise"][0]["start"] == 3.0
    before, after = decode(long_clip), decode(result.output)
    n = min(len(before), len(after))
    assert float(np.max(np.abs(before[: int(2.7 * SR)] - after[: int(2.7 * SR)]))) < 1e-3  # untouched
    assert float(np.max(np.abs(before[int(5.3 * SR): n] - after[int(5.3 * SR): n]))) < 1e-3
    assert float(np.sqrt(np.mean((before[int(3.5 * SR): int(4.5 * SR)] - after[int(3.5 * SR): int(4.5 * SR)]) ** 2))) > 0.005  # cleaned
    # the editor can start from what was made: the same timeline needs nothing more
    assert editor_job.quote_for(job, Timeline.from_json(job.editor["timeline"], seconds=8.0)).nothing


@pytest.mark.native
def test_the_cli_takes_a_timeline_file(clip, tmp_path, capsys):
    tl = Timeline(3.0)
    tl.paint("noise", 0.0, 3.0, {"strength_db": 24.0, "remover": "light"})
    path = tmp_path / "parts.json"
    path.write_text(json.dumps(tl.to_json()))
    assert app_main.main(["--cli", "--timeline", str(path), "-o", str(tmp_path / "out"), "--audio-format", "wav", str(clip)]) == 0
    assert (tmp_path / "out" / "talk-cleaned.wav").is_file()
    # a timeline is for one recording; a bad file is refused before anything runs
    other = clip.with_name("other.wav")
    other.write_bytes(clip.read_bytes())
    assert app_main.main(["--cli", "--timeline", str(path), str(clip), str(other)]) == 1
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"version": 1, "seconds": 3, "layers": {"noise": [{"start": 0, "end": 1, "strength_db": 0}]}}))
    with pytest.raises(SystemExit):
        app_main.main(["--cli", "--timeline", str(bad), str(clip)])
    assert "not valid" in capsys.readouterr().err
    notjson = tmp_path / "x.json"
    notjson.write_text("{nope")
    assert cli.load_timeline(str(notjson))[0] is None and cli.load_timeline(str(tmp_path / "missing.json"))[0] is None
    assert cli.load_timeline(str(path))[0] is not None


def two_tracks(wav, out):
    run_ffmpeg(bundled_ffmpeg(), "-i", str(wav), "-i", str(wav), "-map", "0:a", "-map", "1:a", "-c:a", "flac", str(out))
    return out


@pytest.mark.native
def test_a_whole_file_run_with_echo_removal_fills_the_blocks_the_editor_plans_against(clip, tmp_path):
    try:
        two = two_tracks(clip, tmp_path / "media" / "two.mkv")
    except Exception as e:  # noqa: BLE001 - this FFmpeg cannot make the test file
        pytest.skip(f"cannot make a two-track file: {e}")
    _, job = first_run(two, tmp_path, echo_track=True)
    assert job.echo and job.has_blocks and job.editor["facts"]["reference"] is True
    tl = editor_job.baseline_timeline(job)
    assert tl.layers["echo"] and editor_job.quote_for(job, tl).nothing  # echo blocks, and the noise blocks keyed by the echo parts
    no = Timeline.from_json({**tl.to_json(), "layers": {k: v for k, v in tl.to_json()["layers"].items() if k != "echo"}}, seconds=3.0)
    assert not editor_job.quote_for(job, no).nothing  # without echo removal the noise blocks are other blocks (they read other audio)


@pytest.mark.native
def test_old_jobs_with_saved_copies_open_and_are_turned_into_blocks(clip, tmp_path, monkeypatch):
    ff = engine.ffmpeg()
    x = pipeline.decode_for_clean(ff, clip, ff.probe(clip), 600)
    made = {}
    pipeline.clean_audio(x, resources.libdf_path(), resources.model_path(), pipeline.CleanOptions(noise="light", leveling=False),
                         keep=lambda name, y, cols: made.update(y=np.array(y), cols=cols))
    folder = copies.new_job_dir(tmp_path / "jobs")
    job = copies.JobCopies.create(folder, clip, copies.fingerprint(clip), engine.media_facts(ff.probe(clip)))
    job.data["samples"] = len(x)
    job.put_audio(copies.NOISE, kept.save(ff, made["y"], made["cols"], folder / "noise"), plan=[0], profile="light", echo=False)
    job.made.update({"atten_lim_db": 100.0, "rebuilt": False, "leveling": False, "loudness": "podcast", "noise": "light", "echo": False})
    job.save()
    assert not job.has_blocks and editor_session.can_edit(job) and editor_job.needs_migration(job)
    session = editor_session.Session.open(job, scratch_parent=tmp_path / "scratch")
    try:
        assert job.has_blocks and not job.has(copies.NOISE) and not (folder / "noise.flac").exists()  # the copy became blocks
        assert editor_job.quote_for(job, editor_job.baseline_timeline(job)).nothing
        assert session.mix is not None and session.mix.frames == len(x) and abs(session.mix.read(0, 4800)).max() > 0
    finally:
        session.close()
    no_models(monkeypatch)
    out = redo.run(job, redo.Wanted(atten_lim_db=24.0, leveling=False, output_name="migrated.wav"))
    expected = pipeline.blend_strength(made["y"].copy(), x, 24.0)  # what the old copy gives at this strength
    assert float(np.max(np.abs(decode(out.output) - expected[:, 0][:len(decode(out.output))]))) < 1e-3
    # a job with nothing saved, or an original that changed, cannot be opened for editing
    bare = copies.JobCopies.create(copies.new_job_dir(tmp_path / "jobs"), clip, copies.fingerprint(clip), {})
    assert not editor_session.can_edit(bare) and not editor_session.can_edit(None)


@pytest.mark.native
def test_a_block_that_cannot_be_written_never_fails_the_job_and_the_notice_says_so(clip, tmp_path, monkeypatch):
    import errno

    def full(self, key, block):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(blocks.FlacCache, "put", full)
    keep = copies.new_job_dir(tmp_path / "jobs")
    result = engine.process(clip, engine.JobOptions(noise="light", audio_format="wav"), out_dir=tmp_path / "out", keep_dir=keep)
    assert result.output.is_file() and result.kept_dir is None and not keep.exists()  # nothing was kept, so no folder
    assert "can't be changed and saved again later" in result.notice and "the disk is full" in result.notice


@pytest.mark.native
def test_deleting_a_job_removes_its_blocks_and_never_the_results_or_the_original(clip, tmp_path):
    result, job = first_run(clip, tmp_path)
    original, output = clip, result.output
    assert block_count(job) >= 1
    assert copies.delete_job_dir(job.dir.parent, job.dir.name)
    assert not job.dir.exists() and original.is_file() and output.is_file()
    assert copies.JobCopies.load(job.dir) is None


@pytest.mark.native
def test_the_session_quotes_and_hears_a_timeline_and_a_changed_original_is_refused(clip, tmp_path):
    _, job = first_run(clip, tmp_path)
    session = editor_session.Session.open(job, scratch_parent=tmp_path / "scratch")
    try:
        tl = editor_job.baseline_timeline(job)
        assert session.quote(tl).nothing
        tl.paint("noise", 1.0, 2.0, {"strength_db": 100.0, "remover": "heavy"})
        q = session.quote(tl)
        assert not q.nothing and q.steps == {"heavy": 1} and q.seconds > 0 and editor_job.say_time(q.seconds).startswith("about ")
        session.refresh(tl)
        assert session.mix.missing("noise") == [(1.0, 2.0)]  # the heavy remover has not run on it yet
        tl.erase("noise", 1.0, 2.0)
        tl.paint("noise", 1.0, 2.0, {"strength_db": 12.0, "remover": "light"})
        session.refresh(tl)
        assert session.mix.missing("noise") == []
    finally:
        session.close()
    clip.write_bytes(clip.read_bytes()[:-5000])
    with pytest.raises(Exception, match="changed|different"):
        editor_session.Session.open(job, scratch_parent=tmp_path / "scratch")


@pytest.mark.native
def test_voice_rebuild_on_one_part_makes_only_that_parts_blocks_and_wins_over_the_noise_removal(clip, tmp_path, monkeypatch):
    _, job = first_run(clip, tmp_path)
    before = block_count(job)
    asked = []

    def fake_blocks(src, model_dir, indices, put, **kw):
        asked.append(list(indices))
        n = len(src)
        for b in indices:
            s, e = restore.block_bounds(b, n)
            put(b, np.asarray(src[s:e], dtype=np.float32) * 0.5)

    monkeypatch.setattr(resources, "voice_model_dir", lambda: tmp_path)
    monkeypatch.setattr(restore, "available", lambda d: True)
    monkeypatch.setattr(restore, "restore_blocks", fake_blocks)
    tl = editor_job.baseline_timeline(job)
    tl.paint("rebuild", 1.0, 2.0)
    wanted = redo.Wanted(timeline=tl.to_json(), leveling=False, output_name="rebuilt-part.wav")
    p = redo.plan(job, wanted, job.check_original(), voice_model=True)
    assert p.sound == redo.SOUND_EDITOR and p.slow and "Process what changed" in p.steps[0]
    assert editor_job.quote_for(job, tl).steps == {"rebuild": 1}
    out = redo.run(job, wanted)
    assert asked == [[0]] and block_count(job) == before + 1  # one block for the part, no noise block again
    assert job.made["rebuilt"] is False and job.editor["timeline"]["layers"]["rebuild"][0]["start"] == 1.0
    plain = redo.run(job, redo.Wanted(timeline=editor_job.baseline_timeline(job).to_json(), leveling=False, output_name="plain.wav"))
    heard, ref = decode(out.output), decode(plain.output)
    n = min(len(heard), len(ref))
    assert float(np.max(np.abs(heard[: int(0.6 * SR)] - ref[: int(0.6 * SR)]))) < 1e-3  # outside the part and its join: the same
    assert float(np.max(np.abs(heard[int(2.4 * SR): n] - ref[int(2.4 * SR): n]))) < 1e-3
    mid = slice(int(1.2 * SR), int(1.8 * SR))
    assert abs(float(np.sqrt(np.mean(heard[mid] ** 2))) - float(np.sqrt(np.mean(ref[mid] ** 2)))) > 0.3 * float(np.sqrt(np.mean(ref[mid] ** 2)))  # the rebuilt voice
    # without the model, a part that needs it cannot be saved (and the plan says so); one whose blocks are there already can
    monkeypatch.setattr(resources, "voice_model_dir", lambda: None)
    assert redo.plan(job, redo.Wanted(timeline=tl.to_json(), output_name="x.wav"), job.check_original(), voice_model=False).ok
    (tmp_path / "again").mkdir()
    _, fresh = first_run(clip, tmp_path / "again")
    p = redo.plan(fresh, redo.Wanted(timeline=tl.to_json(), output_name="x.wav"), fresh.check_original(), voice_model=False)
    assert not p.ok and "model that rebuilds the voice is not installed" in p.problems[0]


def test_say_time_rounds_to_what_is_worth_saying():
    say = editor_job.say_time
    assert say(0.2) == "about 1 s" and say(4.4) == "about 4 s" and say(41) == "about 40 s" and say(100) == "about 2 min"
    assert say(23 * 60) == "about 25 min" and say(3 * 3600) == "about 3.0 h"
