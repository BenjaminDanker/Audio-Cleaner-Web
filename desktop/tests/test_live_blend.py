"""The strength slider's arithmetic, the live blend the player reads, and the player that plays it: the blend is bit for bit what the
pipeline puts in the saved file, changing the strength changes the very next piece of sound, and the player works with a sound card that
pulls (QAudioSink) without needing one."""
from __future__ import annotations

import numpy as np
import pytest

from acw_common import pipeline
from audio_cleaner_desktop import blend

from fake_media import FakeSink, MediaStatus, PlaybackState, namespace, sink_factory

SR = blend.SAMPLE_RATE


def signals(n=48000, seed=0):
    rng = np.random.default_rng(seed)
    x = (0.1 * rng.standard_normal((n, 2))).astype(np.float32)
    full = (0.05 * rng.standard_normal((n, 2))).astype(np.float32)
    return full, x


# ---------------------------------------------------------------------------------------------------- the scale

def test_the_four_names_are_equally_spaced_on_the_slider_with_the_dB_between_them():
    assert [blend.db_at(p) for p in (0, 1, 2, 3)] == [12.0, 24.0, 40.0, 100.0]
    assert blend.db_at(0.5) == 18.0 and blend.db_at(2.5) == 70.0 and blend.db_at(-1) == 12.0 and blend.db_at(9) == 100.0
    for db in (12, 15.5, 24, 33, 40, 70, 100):
        assert blend.db_at(blend.pos_of(db)) == pytest.approx(db)
    assert [blend.strength_name(d) for d in (12, 24, 40, 100)] == ["Gentle", "Balanced", "Strong", "Maximum"]
    assert blend.strength_name(30) == "Between Balanced and Strong" and blend.strength_name(18) == "Between Gentle and Balanced"
    assert blend.strength_name(70) == "Between Strong and Maximum"
    assert blend.strength_key(24.2) == "balanced" and blend.strength_key(30) is None and blend.snap(39.7) == 40.0 and blend.snap(31.3) == 31.3
    from audio_cleaner_desktop import engine

    assert {k: db for k, _, db in blend.NAMES} == engine.STRENGTHS  # the same four as the engine's


def test_the_share_of_the_recording_is_the_pipelines():
    assert blend.mix_of(100) == 0.0 and blend.mix_of(24) == pytest.approx(10 ** (-24 / 20)) and blend.mix_of(12) == pipeline.strength_mix(12)


# ---------------------------------------------------------------------------------------------------- the blend

@pytest.mark.parametrize("plan", [[0, 1], [0, 0], [0, None], [None, 1]])
@pytest.mark.parametrize("db", [12.0, 24.0, 31.0, 40.0, 100.0])
def test_the_blend_is_bit_for_bit_what_the_pipeline_writes(plan, db):
    full, x = signals()
    cols = sorted({c for c in plan if c is not None})
    kept_full = np.zeros_like(full)
    kept_full[:, cols] = full[:, cols]  # a saved copy only holds the channels that were cleaned
    expected = kept_full.copy()
    pipeline.blend_strength(expected, x, db, cols)
    for c, src in enumerate(plan):  # what pipeline._after_noise does with the other channels
        expected[:, c] = 0.0 if src is None else expected[:, src]
    stream = blend.BlendStream(kept_full, x, plan, exact=True)
    stream.set_strength(db)
    got = np.concatenate([stream.read(s, 7000) for s in range(0, stream.frames, 7000)])
    assert got.dtype == np.float32 and np.array_equal(got, expected)


def test_the_loudness_match_keeps_the_volume_when_the_strength_moves():
    full, x = signals(96000)
    full *= 0.3  # the cleaned sound is quieter than the recording
    stream = blend.BlendStream(full, x, [0, 1])
    target = float(np.sqrt(np.mean(np.square(full))))
    for db in (12.0, 24.0, 40.0, 100.0):
        stream.set_strength(db)
        heard = stream.read(0, stream.frames)
        assert float(np.sqrt(np.mean(np.square(heard)))) == pytest.approx(target, rel=0.02)
    stream.exact = True  # the saved file is exactly this blend: no match
    stream.set_strength(12.0)
    exact = stream.read(0, stream.frames)
    plain = blend.blend_block(full, x, [0, 1], blend.mix_of(12.0))
    assert np.array_equal(exact, plain)


def test_original_plays_the_recording_through_the_same_stream_at_the_same_place():
    full, x = signals()
    stream = blend.BlendStream(full, x, [0, None], exact=True)
    stream.set_original(True)
    assert np.array_equal(stream.read(100, 50), x[100:150])  # every channel of the recording, even the one the cleaned sound leaves silent
    stream.set_original(False)
    assert not np.array_equal(stream.read(100, 50), x[100:150])


def test_reads_past_the_end_are_short_and_pcm16_is_clipped_and_interleaved():
    full, x = signals(1000)
    stream = blend.BlendStream(full * 40, x, [0, 1], exact=True)
    assert len(stream.read(900, 500)) == 100 and len(stream.read(1000, 10)) == 0 and len(stream.read(-5, 3)) == 3
    pcm = np.frombuffer(stream.read_pcm16(0, 1000), dtype="<i2").reshape(-1, 2)
    assert pcm.shape == (1000, 2) and pcm.max() <= 32767 and pcm.min() >= -32767 and (np.abs(pcm) == 32767).any()


def test_a_copy_and_a_recording_of_different_lengths_are_refused():
    full, x = signals(1000)
    with pytest.raises(ValueError):
        blend.BlendStream(full[:900], x, [0, 1])


# ---------------------------------------------------------------------------------------------------- the player

@pytest.fixture
def live(qapp):
    from audio_cleaner_desktop.gui import live_audio

    FakeSink.made.clear()
    full, x = signals(SR * 2)
    stream = blend.BlendStream(full, x, [0, 1], exact=True)
    player = live_audio.LivePlayer(namespace(), sink_factory=sink_factory)
    events = {"state": [], "status": [], "errors": [], "positions": []}
    player.playbackStateChanged.connect(events["state"].append)
    player.mediaStatusChanged.connect(events["status"].append)
    player.errorOccurred.connect(lambda e, m: events["errors"].append(m))
    player.positionChanged.connect(events["positions"].append)
    player.load(stream)
    yield player, stream, events, full, x
    player.stop()


def pcm_of(stream, start, frames):
    return stream.read_pcm16(start, frames)


def test_loading_reports_the_length_and_play_starts_the_sound_card(live):
    player, stream, events, *_ = live
    assert player.duration() == 2000 and events["status"] == [MediaStatus.LoadedMedia]
    player.play()
    sink = FakeSink.made[-1]
    assert sink.started and (sink.rate, sink.channels) == (SR, 2) and player.playbackState() == PlaybackState.PlayingState
    assert events["state"] == [PlaybackState.PlayingState]


def test_the_card_gets_the_sound_in_order_and_the_position_follows_what_it_played(live):
    player, stream, events, *_ = live
    player.play()
    sink = FakeSink.made[-1]
    data = sink.pull(4800) + sink.pull(4800)
    assert data == pcm_of(stream, 0, 9600)
    assert player.position() == 200


def test_moving_the_strength_while_it_plays_changes_the_next_sound(live):
    player, stream, events, *_ = live
    stream.set_strength(100.0)
    player.play()
    sink = FakeSink.made[-1]
    first = sink.pull(4800)
    stream.set_strength(12.0)  # the slider is moved
    second = sink.pull(4800)
    stream.set_strength(100.0)
    assert first == stream.read_pcm16(0, 4800)
    assert second != stream.read_pcm16(4800, 4800)  # it was the gentler mix
    stream.set_strength(12.0)
    assert second == stream.read_pcm16(4800, 4800)


def test_original_and_cleaned_are_switched_in_place(live):
    player, stream, *_ = live
    player.play()
    sink = FakeSink.made[-1]
    sink.pull(2400)
    stream.set_original(True)
    heard = sink.pull(2400)
    assert heard == stream.read_pcm16(2400, 2400) and heard.startswith(stream.read_pcm16(2400, 10))


def test_pause_keeps_the_place_and_seeking_while_playing_restarts_the_card_there(live):
    player, stream, events, *_ = live
    player.play()
    sink = FakeSink.made[-1]
    sink.pull(24000)
    player.pause()
    assert player.playbackState() == PlaybackState.PausedState and sink.stopped and player.position() == 500
    player.play()
    again = FakeSink.made[-1]
    assert again is not sink and again.pull(100) == stream.read_pcm16(24000, 100)
    player.setPosition(1500)
    seeked = FakeSink.made[-1]
    assert seeked is not again and player.position() == 1500 and seeked.pull(100) == stream.read_pcm16(72000, 100)
    player.pause()
    player.setPosition(250)  # paused: only the place moves
    assert player.position() == 250 and len(FakeSink.made) == 3
    player.play()
    assert FakeSink.made[-1].pull(10) == stream.read_pcm16(12000, 10)


def test_the_end_is_announced_once_and_play_starts_over(live):
    player, stream, events, *_ = live
    player.play()
    sink = FakeSink.made[-1]
    sink.pull(SR * 2)
    player._tick()
    assert MediaStatus.EndOfMedia in events["status"] and player.playbackState() == PlaybackState.StoppedState
    assert events["positions"][-1] == 2000
    player.play()
    assert FakeSink.made[-1].pull(10) == stream.read_pcm16(0, 10)  # from the start


def test_a_sound_card_that_cannot_play_it_is_reported_not_raised(qapp):
    from audio_cleaner_desktop.gui import live_audio

    full, x = signals(4800)
    stream = blend.BlendStream(full, x, [0, 1], exact=True)

    def refuse(rate, channels, device):
        raise RuntimeError("This computer's sound output can't play this live.")

    player = live_audio.LivePlayer(namespace(), sink_factory=refuse)
    errors = []
    player.errorOccurred.connect(lambda e, m: errors.append(m))
    player.load(stream)
    player.play()
    assert errors == ["This computer's sound output can't play this live."] and player.playbackState() == PlaybackState.StoppedState


def test_unloading_empties_the_player(live):
    player, stream, events, *_ = live
    player.play()
    player.unload()
    assert player.duration() == 0 and player.position() == 0 and events["status"][-1] == MediaStatus.NoMedia
    player.play()  # nothing to play
    assert player.playbackState() == PlaybackState.StoppedState
