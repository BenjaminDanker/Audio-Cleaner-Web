"""A disk that fills up: the scratch arrays (pipeline.scratch_array) take their room when they are made, so a full disk is an OSError the run
can report ("The disk is full.") and not the end of the program, and a run that fails because FFmpeg could not write is reported the same way
when the disk really has no room left (engine.disk_full_error). The server's way of making the renderer's arrays is unchanged."""
from __future__ import annotations

import errno
import os
import shutil
from collections import namedtuple

import numpy as np
import pytest

from acw_common import blocks, editor, pipeline
from acw_common.ffmpeg import MediaError
from acw_common.timeline import Timeline
from audio_cleaner_desktop import copies, editor_job, editor_session, engine, live_mix, redo

SR = pipeline.SAMPLE_RATE
SHAPE = (500_000, 2)  # four megabytes: more than one piece of the zero-writing loop, and the last piece is short
DATA_BYTES = SHAPE[0] * SHAPE[1] * 4

needs_room = pytest.mark.skipif(os.name == "nt", reason="Windows takes the room when the file gets its length")
has_blocks = hasattr(os.stat_result, "st_blocks")


# ---------------------------------------------------------------------------------------------------- helpers

def raising(code):
    def fallocate(fd, offset, length):
        raise OSError(code, os.strerror(code))

    return fallocate


def no_room(monkeypatch, code=errno.ENOSPC):
    """posix_fallocate (and so a file system that has no room) says `code`. (raising=False: macOS has no such call, the stand-in adds it.)"""
    monkeypatch.setattr(os, "posix_fallocate", raising(code), raising=False)


def no_fallocate(monkeypatch):
    """The way a Mac makes its room: there is no posix_fallocate."""
    monkeypatch.delattr(os, "posix_fallocate", raising=False)


def taken(path) -> int:
    return os.stat(path).st_blocks * 512


def assert_scratch(m, path, *, check_blocks=True):
    """`m` is float32 zeros of SHAPE in `path`, the file is a .npy whose header is whole, what is written to the map reads back, and (where the
    file system tells) the room for the data is really taken."""
    assert isinstance(m, np.ndarray) and m.dtype == np.float32 and m.shape == SHAPE
    assert path.read_bytes()[:6] == b"\x93NUMPY"
    assert not m.any()  # (a float32 whose bits are all zero is zero)
    if has_blocks and check_blocks:
        assert taken(path) >= DATA_BYTES
    assert path.stat().st_size == m.offset + DATA_BYTES  # (not a byte more: the zeros went in from the data offset on)
    m[0, 0], m[-1, 1] = 0.5, -0.25
    m.flush()
    back = np.load(path, mmap_mode="r")
    assert back.shape == SHAPE and back.dtype == np.float32 and back[0, 0] == 0.5 and back[-1, 1] == -0.25
    assert not back[1:-1].any() and back[0, 1] == 0 and back[-1, 0] == 0
    del back


def whole_clip(seconds=1.0):
    x = (0.1 * np.random.default_rng(3).standard_normal((int(seconds * SR), 1))).astype(np.float32)
    return editor.Inputs(x, "sid"), Timeline.whole_file(seconds, noise=False, leveling=False)


# ---------------------------------------------------------------------------------------------------- scratch_array

@needs_room
def test_a_scratch_array_is_zeros_with_its_room_taken_and_its_header_whole(tmp_path):
    path = tmp_path / "a.npy"
    m = pipeline.scratch_array(path, SHAPE)
    assert_scratch(m, path)


@pytest.mark.skipif(not has_blocks or not hasattr(os, "posix_fallocate"), reason="needs st_blocks and posix_fallocate")
def test_the_room_is_asked_for_from_the_data_not_the_header(tmp_path, monkeypatch):
    real = os.posix_fallocate
    asked = []
    monkeypatch.setattr(os, "posix_fallocate", lambda fd, offset, length: asked.append((offset, length)) or real(fd, offset, length))
    m = pipeline.scratch_array(tmp_path / "a.npy", SHAPE)
    assert asked == [(m.offset, DATA_BYTES)] and m.offset > 0


@pytest.mark.skipif(not has_blocks, reason="needs st_blocks")
def test_a_plain_map_is_sparse_so_the_check_of_the_room_means_something(tmp_path):
    path = tmp_path / "plain.npy"
    m = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=SHAPE)
    if taken(path) >= DATA_BYTES:
        pytest.skip("this file system does not make sparse files")
    assert taken(path) < DATA_BYTES // 2
    del m


@needs_room
def test_without_posix_fallocate_the_zeros_are_written_and_the_header_is_whole(tmp_path, monkeypatch):
    no_fallocate(monkeypatch)
    assert not hasattr(os, "posix_fallocate")
    path = tmp_path / "a.npy"
    m = pipeline.scratch_array(path, SHAPE)
    assert_scratch(m, path)


@needs_room
@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EDQUOT, errno.EFBIG])
def test_a_disk_without_room_is_an_error_and_the_file_is_gone(tmp_path, monkeypatch, code):
    no_room(monkeypatch, code)
    path = tmp_path / "a.npy"
    with pytest.raises(OSError) as e:
        pipeline.scratch_array(path, SHAPE)
    assert e.value.errno == code and not path.exists() and list(tmp_path.iterdir()) == []


@needs_room
@pytest.mark.parametrize("code", [errno.EOPNOTSUPP, errno.EINVAL, errno.ENOSYS])
def test_a_file_system_that_cannot_set_room_aside_gets_the_zeros_written(tmp_path, monkeypatch, code):
    no_room(monkeypatch, code)
    path = tmp_path / "a.npy"
    m = pipeline.scratch_array(path, SHAPE)
    assert_scratch(m, path)


@needs_room
def test_a_full_disk_found_while_the_zeros_are_written_is_the_same_error_and_the_file_is_gone(tmp_path, monkeypatch):
    no_fallocate(monkeypatch)
    real_open = open
    writes = []

    class Full:
        def __init__(self, f):
            self.f = f

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.f.close()

        def seek(self, n):
            self.f.seek(n)

        def write(self, data):
            writes.append(len(data))
            if len(writes) == 2:  # (the first piece went in, the disk is full for the next)
                raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
            return self.f.write(data)

    monkeypatch.setattr(pipeline, "open", lambda *a, **k: Full(real_open(*a, **k)), raising=False)
    path = tmp_path / "a.npy"
    with pytest.raises(OSError) as e:
        pipeline.scratch_array(path, SHAPE)
    assert e.value.errno == errno.ENOSPC and writes == [1 << 20, 1 << 20] and not path.exists()


def test_an_array_with_no_data_is_not_an_error_for_the_room(tmp_path):
    path = tmp_path / "empty.npy"
    pipeline._take_room(path, 128, 0)  # (nothing to set aside: the file is not even opened)
    assert not path.exists()


# ---------------------------------------------------------------------------------------------------- output_buffer

@needs_room
def test_the_output_buffer_in_a_work_folder_takes_its_room(tmp_path):
    like = np.zeros(SHAPE, np.float32)
    y = pipeline.output_buffer(like, tmp_path)
    assert_scratch(y, tmp_path / "cleaned.npy")


@needs_room
def test_the_output_buffer_of_a_full_disk_is_the_error_and_leaves_no_file(tmp_path, monkeypatch):
    no_room(monkeypatch)
    with pytest.raises(OSError) as e:
        pipeline.output_buffer(np.zeros(SHAPE, np.float32), tmp_path)
    assert e.value.errno == errno.ENOSPC and list(tmp_path.iterdir()) == []


def test_the_output_buffer_without_a_work_folder_is_in_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "scratch_array", lambda *a, **k: pytest.fail("a file was made"))
    y = pipeline.output_buffer(np.zeros((1000, 2), np.float32))
    assert type(y) is np.ndarray and y.shape == (1000, 2) and y.dtype == np.float32


# ---------------------------------------------------------------------------------------------------- the renderer

@needs_room
def test_a_render_that_reserves_makes_its_arrays_with_their_room_and_a_full_disk_is_the_error_with_no_file_left(tmp_path, monkeypatch):
    inputs, tl = whole_clip()
    (tmp_path / "first").mkdir()
    r = editor._Render(tl, inputs, blocks.MemoryCache(), None, None, tmp_path / "first", None, None, reserve=True)
    assert r.out.shape == inputs.source.shape and not r.out.any()
    if has_blocks:
        assert taken(r.out.filename) >= r.out.nbytes
    r._drop_temp(close=True)
    del r
    no_room(monkeypatch)
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(OSError) as e:
        editor.render(tl, inputs, blocks.MemoryCache(), None, work_dir=tmp_path, reserve=True)
    assert e.value.errno == errno.ENOSPC
    assert not list(tmp_path.glob("editor-*")) and sorted(p.name for p in tmp_path.iterdir()) == before  # (no scratch file, no cache folder)


@needs_room
def test_a_later_array_of_a_reserving_render_that_does_not_fit_is_the_error_and_leaves_no_file(tmp_path, monkeypatch):
    inputs, tl = whole_clip()
    r = editor._Render(tl, inputs, blocks.MemoryCache(), None, None, tmp_path, None, None, reserve=True)
    made = r.alloc("more")
    assert made.shape == inputs.source.shape and len(list(tmp_path.glob("editor-more-*.npy"))) == 1
    no_room(monkeypatch)
    with pytest.raises(OSError) as e:
        r.alloc("next", 1000)
    assert e.value.errno == errno.ENOSPC and not list(tmp_path.glob("editor-next-*"))


def test_the_servers_render_never_goes_through_scratch_array(tmp_path, monkeypatch):
    """The server does not pass `reserve`: its arrays are plain memory maps, whatever the disk says to posix_fallocate."""
    monkeypatch.setattr(pipeline, "scratch_array", lambda *a, **k: pytest.fail("scratch_array was used without reserve"))
    no_room(monkeypatch)
    inputs, tl = whole_clip()
    r = editor._Render(tl, inputs, blocks.MemoryCache(), None, None, tmp_path, None, None)
    assert r.reserve is False and isinstance(r.out, np.memmap) and r.out.shape == inputs.source.shape
    other = r.alloc("more", 100)
    assert isinstance(other, np.memmap) and len(list(tmp_path.glob("editor-*.npy"))) == 2
    y = editor.render(tl, inputs, blocks.MemoryCache(), None, work_dir=tmp_path)
    assert y.shape == inputs.source.shape and float(np.max(np.abs(y - inputs.source))) < 1e-6  # (nothing to do: the recording as it is)
    assert type(editor._Render(tl, inputs, blocks.MemoryCache(), None, None, None, None, None).out) is np.ndarray  # (no folder: memory)


def test_a_render_that_reserves_gives_the_same_sound(tmp_path):
    inputs, tl = whole_clip()
    plain = editor.render(tl, inputs, blocks.MemoryCache(), None, work_dir=tmp_path)
    reserved = editor.render(tl, inputs, blocks.MemoryCache(), None, work_dir=tmp_path, reserve=True)
    assert np.array_equal(plain, reserved)


# ---------------------------------------------------------------------------------------------------- the editor's live layers

@needs_room
def test_a_live_layer_in_a_work_folder_has_its_room_and_a_full_disk_is_the_error_and_no_layer(tmp_path):
    src = np.zeros(SHAPE, np.float32)
    layers = live_mix.Layers(src, [0, 1], tmp_path)
    layer = layers.array("light")
    assert layers.arrays["light"] is layer and layers.array("light") is layer
    files = list(tmp_path.glob("live-light-*.npy"))
    assert len(files) == 1
    assert_scratch(layer, files[0])
    with pytest.MonkeyPatch.context() as mp:
        no_room(mp)
        with pytest.raises(OSError) as e:
            layers.array("heavy")
    assert e.value.errno == errno.ENOSPC and not list(tmp_path.glob("live-heavy-*")) and "heavy" not in layers.arrays
    # (the disk has room again: the layer can be made)
    again = layers.array("heavy")
    assert layers.arrays["heavy"] is again and len(list(tmp_path.glob("live-heavy-*.npy"))) == 1


def test_a_live_layer_without_a_work_folder_is_in_memory(monkeypatch):
    monkeypatch.setattr(pipeline, "scratch_array", lambda *a, **k: pytest.fail("a file was made"))
    layers = live_mix.Layers(np.zeros((1000, 2), np.float32), [0, 1])
    layer = layers.array("light")
    assert type(layer) is np.ndarray and layer.shape == (1000, 2) and layer.dtype == np.float32 and not layer.any()


# ---------------------------------------------------------------------------------------------------- engine.disk_full_error

FULL = namedtuple("usage", "total used free")


def test_a_media_error_on_a_disk_that_is_full_becomes_the_disk_is_full_error(tmp_path, monkeypatch):
    from audio_cleaner_desktop.gui import worker

    monkeypatch.setattr(engine, "FULL_DISK_BYTES", 1 << 62)  # (any disk counts as full)
    error = MediaError("Encoding the audio failed.")
    full = engine.disk_full_error(error, tmp_path)
    assert isinstance(full, OSError) and not isinstance(full, MediaError) and full.errno == errno.ENOSPC
    assert worker.failure_message(full) == "The disk is full."
    assert worker.failure_message(error) == "Encoding the audio failed."  # (as it was)


def test_a_disk_with_room_leaves_the_media_error_alone(tmp_path):
    assert engine.FULL_DISK_BYTES == 32 << 20
    assert engine.disk_full_error(MediaError("Encoding the audio failed."), tmp_path) is None


def test_the_line_between_full_and_not_is_the_threshold(tmp_path, monkeypatch):
    error = MediaError("x")
    free = [engine.FULL_DISK_BYTES - 1]
    monkeypatch.setattr(shutil, "disk_usage", lambda folder: FULL(10 << 30, 10 << 30, free[0]))
    assert engine.disk_full_error(error, tmp_path).errno == errno.ENOSPC
    free[0] = engine.FULL_DISK_BYTES
    assert engine.disk_full_error(error, tmp_path) is None


@pytest.mark.parametrize("error", [RuntimeError("the render broke"), engine.Cancelled(), ValueError("x"), OSError(errno.EIO, "io")])
def test_only_a_media_error_is_taken_for_a_full_disk(tmp_path, monkeypatch, error):
    monkeypatch.setattr(engine, "FULL_DISK_BYTES", 1 << 62)
    assert engine.disk_full_error(error, tmp_path) is None


def test_no_folder_or_a_missing_folder_is_not_a_full_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "FULL_DISK_BYTES", 1 << 62)
    error = MediaError("x")
    assert engine.disk_full_error(error) is None
    assert engine.disk_full_error(error, None) is None
    assert engine.disk_full_error(error, None, None) is None
    assert engine.disk_full_error(error, tmp_path / "not" / "there") is None
    # any one folder that is full is enough, and a folder that cannot be asked about does not stop the next one being asked
    assert engine.disk_full_error(error, None, tmp_path / "not" / "there", tmp_path).errno == errno.ENOSPC


def test_one_full_folder_among_roomy_ones_is_enough(tmp_path, monkeypatch):
    (tmp_path / "small").mkdir()
    (tmp_path / "big").mkdir()
    monkeypatch.setattr(shutil, "disk_usage", lambda folder: FULL(1 << 40, 0, 1 << 20 if folder.name == "small" else 1 << 40))
    error = MediaError("x")
    assert engine.disk_full_error(error, tmp_path / "big") is None
    assert engine.disk_full_error(error, tmp_path / "big", tmp_path / "small").errno == errno.ENOSPC


# ---------------------------------------------------------------------------------------------------- a whole run

def left_behind(folder):
    """The temporary files and work folders a run leaves in `folder`; the notes of them in the folder for notes."""
    return sorted(p.name for p in folder.glob(".acw-*")) + sorted(p.name for p in engine._unfinished_dir().iterdir())


def media_failure(monkeypatch, text="Encoding the audio failed."):
    """The step that renders the sound fails the way FFmpeg does: it only says that it failed."""
    def render_timeline(*a, **k):
        raise MediaError(text)

    monkeypatch.setattr(editor_job, "render_timeline", render_timeline)


def clean(clip_wav, out):
    return engine.process(clip_wav, engine.JobOptions(noise="light", strength="balanced", leveling=False, audio_format="flac"), out_dir=out)


@pytest.fixture(autouse=True)
def _work_folders_beside_the_result(monkeypatch):
    monkeypatch.delenv("ACW_DESKTOP_SCRATCH", raising=False)


@pytest.mark.native
def test_a_run_whose_step_failed_on_a_full_disk_says_the_disk_is_full_and_leaves_nothing(clip_wav, tmp_path, monkeypatch):
    from audio_cleaner_desktop.gui import worker

    media_failure(monkeypatch)
    monkeypatch.setattr(engine, "FULL_DISK_BYTES", 1 << 62)
    out = tmp_path / "out"
    with pytest.raises(OSError) as e:
        clean(clip_wav, out)
    assert not isinstance(e.value, MediaError) and e.value.errno == errno.ENOSPC
    assert isinstance(e.value.__cause__, MediaError) and worker.failure_message(e.value) == "The disk is full."
    assert out.is_dir() and list(out.iterdir()) == [] and left_behind(out) == []


@pytest.mark.native
def test_the_same_failure_on_a_disk_with_room_stays_what_it_was(clip_wav, tmp_path, monkeypatch):
    from audio_cleaner_desktop.gui import worker

    media_failure(monkeypatch)
    out = tmp_path / "out"
    with pytest.raises(MediaError, match="Encoding the audio failed") as e:
        clean(clip_wav, out)
    assert worker.failure_message(e.value) == "Encoding the audio failed." and list(out.iterdir()) == [] and left_behind(out) == []


@needs_room
@pytest.mark.native
def test_a_run_whose_scratch_array_does_not_fit_is_the_error_and_leaves_nothing(clip_wav, tmp_path, monkeypatch):
    from audio_cleaner_desktop.gui import worker

    out = tmp_path / "out"
    no_room(monkeypatch)  # (the real threshold: the error is the one the array made, nothing is guessed from the disk)
    with pytest.raises(OSError) as e:
        clean(clip_wav, out)
    assert e.value.errno == errno.ENOSPC and worker.failure_message(e.value) == "The disk is full."
    assert out.is_dir() and list(out.iterdir()) == [] and left_behind(out) == []


@pytest.mark.native
def test_a_stopped_run_is_still_stopped_when_the_disk_is_full(clip_wav, tmp_path, monkeypatch):
    def stopped(*a, **k):
        raise engine.Cancelled()

    monkeypatch.setattr(editor_job, "render_timeline", stopped)
    monkeypatch.setattr(engine, "FULL_DISK_BYTES", 1 << 62)
    with pytest.raises(engine.Cancelled):
        clean(clip_wav, tmp_path / "out")
    assert left_behind(tmp_path / "out") == []


# ---------------------------------------------------------------------------------------------------- change and save again

@pytest.fixture
def saved_job(clip_wav, tmp_path):
    """A cleaned clip with its job kept, and the clip where tests may change it."""
    clip = tmp_path / "media" / "talk.wav"
    clip.parent.mkdir()
    clip.write_bytes(clip_wav.read_bytes())
    keep = copies.new_job_dir(tmp_path / "jobs")
    first = engine.process(clip, engine.JobOptions(noise="light", leveling=False, strength="max", audio_format="wav"),
                           out_dir=tmp_path / "out", keep_dir=keep)
    return first, copies.JobCopies.load(keep)


@pytest.mark.native
def test_a_change_that_failed_on_a_full_disk_says_so_and_leaves_nothing_and_on_a_disk_with_room_it_stays_what_it_was(saved_job, tmp_path):
    first, job = saved_job
    out = tmp_path / "out"
    wanted = redo.Wanted(output_name="changed.wav")

    def nothing_left():
        assert sorted(p.name for p in out.iterdir()) == [first.output.name] and left_behind(out) == [] and len(job.saves) == 1

    with pytest.MonkeyPatch.context() as mp:  # (a disk with room: what it was)
        media_failure(mp, "Saving the file failed.")
        with pytest.raises(MediaError, match="Saving the file failed"):
            redo.run(job, wanted)
    nothing_left()
    with pytest.MonkeyPatch.context() as mp:  # (a full disk)
        media_failure(mp, "Saving the file failed.")
        mp.setattr(engine, "FULL_DISK_BYTES", 1 << 62)
        with pytest.raises(OSError) as e:
            redo.run(job, wanted)
    assert e.value.errno == errno.ENOSPC and isinstance(e.value.__cause__, MediaError)
    nothing_left()
    if os.name != "nt":
        with pytest.MonkeyPatch.context() as mp:  # (the change's own output array does not fit)
            no_room(mp)
            with pytest.raises(OSError) as e:
                redo.run(job, wanted)
        assert e.value.errno == errno.ENOSPC
        nothing_left()


@pytest.mark.native
def test_opening_a_saved_job_for_editing_on_a_full_disk_says_so_and_removes_its_work_folder(saved_job, tmp_path, monkeypatch):
    _, job = saved_job
    scratch = tmp_path / "scratch"

    def decode(*a, **k):
        raise MediaError("Decoding the audio failed.")

    monkeypatch.setattr(editor_job, "decode", decode)
    with pytest.raises(MediaError, match="Decoding"):
        editor_session.Session.open(job, scratch_parent=scratch)
    assert list(scratch.iterdir()) == []
    monkeypatch.setattr(engine, "FULL_DISK_BYTES", 1 << 62)
    with pytest.raises(OSError) as e:
        editor_session.Session.open(job, scratch_parent=scratch)
    assert e.value.errno == errno.ENOSPC and isinstance(e.value.__cause__, MediaError) and list(scratch.iterdir()) == []


@pytest.mark.native
def test_opening_a_file_for_the_editor_on_a_full_disk_says_so_and_removes_its_work_folder(clip_wav, tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"

    def decode(*a, **k):
        raise MediaError("Decoding the audio failed.")

    monkeypatch.setattr(pipeline, "decode_for_clean", decode)
    with pytest.raises(MediaError, match="Decoding"):
        editor_session.PlanSession.open(clip_wav, scratch_parent=scratch)
    assert list(scratch.iterdir()) == []
    monkeypatch.setattr(engine, "FULL_DISK_BYTES", 1 << 62)
    with pytest.raises(OSError) as e:
        editor_session.PlanSession.open(clip_wav, scratch_parent=scratch)
    assert e.value.errno == errno.ENOSPC and isinstance(e.value.__cause__, MediaError) and list(scratch.iterdir()) == []
