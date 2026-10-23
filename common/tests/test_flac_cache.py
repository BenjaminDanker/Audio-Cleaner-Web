"""The block cache's FLAC store (acw_common.blocks.FlacCache) with the real FFmpeg: what is put comes back within one 24-bit step, a block louder
than the format's ceiling comes back at its level, and what the cache says about its files."""
import numpy as np
import pytest

from acw_common import blocks
from acw_common.blocks import SAMPLE_RATE, FlacCache
from acw_common.ffmpeg import FFmpeg

from test_common import FF_DIR, needs_ff

STEP = 2.0 ** -23  # one 24-bit step of full scale (2 / 2**24)


@pytest.fixture
def cache(tmp_path):
    return FlacCache(tmp_path / "blocks", FFmpeg(FF_DIR))


def speech(seconds=2.0, level=0.3, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (level * np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) + 0.01 * level * rng.standard_normal(len(t))).astype(np.float32)


@needs_ff
def test_a_block_comes_back_within_one_24_bit_step(cache):
    x = speech()
    cache.put("k1", x)
    y = cache.get("k1")
    assert y.dtype == np.float32 and y.shape == x.shape
    assert np.max(np.abs(y - x)) <= STEP
    assert not cache.path("k1").with_suffix(".scale").exists()  # (a block under the ceiling needs no scale file)
    assert np.array_equal(cache.get("k1"), y)  # (reading is repeatable)


@needs_ff
def test_a_silent_and_a_very_short_block_are_stored_and_read_back_whole(cache):
    for key, x in (("silence", np.zeros(48_000, np.float32)), ("one", np.array([0.25], np.float32)), ("tiny", np.full(1000, 1e-7, np.float32))):
        cache.put(key, x)
        y = cache.get(key)
        assert y.shape == x.shape and np.max(np.abs(y - x)) <= STEP, key


@needs_ff
@pytest.mark.parametrize("peak", [0.99, 1.0, 1.7, 8.0])
def test_a_block_louder_than_the_ceiling_is_written_scaled_down_and_read_back_at_its_own_level(cache, peak):
    x = speech(level=1.0)
    x = (x / np.max(np.abs(x)) * peak).astype(np.float32)
    cache.put("loud", x)
    side = cache.path("loud").with_suffix(".scale")
    assert side.is_file() and abs(float(side.read_text()) - FlacCache.CEILING / peak) < 1e-6  # (the scale is kept beside it)
    y = cache.get("loud")
    assert y.shape == x.shape and np.max(np.abs(y - x)) <= STEP / (FlacCache.CEILING / peak) * 1.01  # (the step grows with the level)
    # what is in the FLAC file itself never passes the ceiling
    import subprocess

    raw = subprocess.run([str(FF_DIR / "ffmpeg"), "-v", "error", "-i", str(cache.path("loud")), "-f", "f32le", "-"], capture_output=True).stdout
    assert np.max(np.abs(np.frombuffer(raw, np.float32))) <= FlacCache.CEILING + STEP


@needs_ff
def test_putting_again_replaces_the_block_and_a_quiet_one_drops_the_old_scale(cache):
    cache.put("k", speech(level=1.0) * 3)
    side = cache.path("k").with_suffix(".scale")
    assert side.exists()
    quiet = speech(level=0.1, seed=2)
    cache.put("k", quiet)
    assert not side.exists() and np.max(np.abs(cache.get("k") - quiet)) <= STEP and cache.puts == 2


@needs_ff
def test_has_path_name_and_the_bytes_the_folder_takes(cache, tmp_path):
    assert cache.bytes_used() == 0 and not cache.has("a") and cache.get("a") is None
    assert cache.name("a") != cache.name("b") and len(cache.name("a")) == 40 and cache.path("a") == cache.dir / (cache.name("a") + ".flac")
    cache.put("a", speech(1.0))
    assert cache.has("a") and not cache.has("b") and cache.path("a").is_file()
    one = cache.bytes_used()
    assert one == cache.path("a").stat().st_size > 1000
    cache.put("loud", speech(1.0) * 5)  # (its scale file counts too)
    assert cache.bytes_used() == cache.path("a").stat().st_size + cache.path("loud").stat().st_size + cache.path("loud").with_suffix(".scale").stat().st_size
    (cache.dir / "sub").mkdir()  # a folder is not a block
    (cache.dir / "sub" / "x").write_bytes(b"x" * 1000)
    assert cache.bytes_used() == one + cache.path("loud").stat().st_size + cache.path("loud").with_suffix(".scale").stat().st_size
    assert FlacCache(tmp_path / "nowhere", create=False).bytes_used() == 0 and not (tmp_path / "nowhere").exists()
    assert not list(cache.dir.glob("*.tmp")) and not [p for p in cache.dir.iterdir() if p.name.count(".") == 0 and p.is_file()]  # (no half-written files left)


@needs_ff
def test_a_damaged_empty_or_missing_file_gives_none_and_a_link_does_not_count(cache, tmp_path):
    cache.put("good", speech(1.0))
    cache.put("cut", speech(1.0))
    cache.put("empty", speech(1.0))
    cache.path("cut").write_bytes(cache.path("cut").read_bytes()[:60])  # (cut off in the header)
    cache.path("empty").write_bytes(b"")
    junk = cache.path("junk")
    junk.write_bytes(b"this is not audio at all" * 50)
    assert cache.get("cut") is None and cache.get("empty") is None and cache.get("junk") is None
    assert cache.get("good") is not None and cache.get("never-put") is None
    # a link is never followed: it is not a block
    link = cache.path("linked")
    link.symlink_to(cache.path("good"))
    assert cache.has("linked") is False and cache.get("linked") is None
    # a scale file with nonsense in it is ignored (the block is read as it is)
    cache.path("good").with_suffix(".scale").write_text("nonsense")
    assert np.max(np.abs(cache.get("good") - FlacCache(cache.dir, cache.ff).get("good"))) == 0


def test_a_cache_without_ffmpeg_can_only_look(tmp_path):
    cache = FlacCache(tmp_path / "b")
    assert cache.has("x") is False
    cache.path("x").write_bytes(b"fLaC")
    assert cache.has("x") is True
    with pytest.raises(ValueError):
        cache.get("x")
    with pytest.raises(ValueError):
        cache.put("y", np.zeros(10, np.float32))
    assert cache.get("missing") is None  # (nothing there: nothing to read, no FFmpeg needed to say so)
