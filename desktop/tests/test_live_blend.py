"""The named strengths (blend.py: the same scale as the website's) and the live player (gui/live_audio.LivePlayer) that plays the editor's live
mix (live_mix.LiveMix): changing the timeline changes the very next piece of sound, and the player works with a sound card that pulls
(QAudioSink) without needing one. What the mix sounds like is in test_editor_model.py."""
from __future__ import annotations

import numpy as np
import pytest

from acw_common.timeline import Timeline
from audio_cleaner_desktop import blend, engine, live_mix

from fake_media import FakeSink, MediaStatus, PlaybackState, namespace, sink_factory

SR = live_mix.SR


def mix_at(db: float, n: int = SR * 2, seed: int = 0):
    """A live mix of two seconds (stereo) with made-up sound: the recording, and the light remover's blocks (a quieter sound), heard at `db`."""
    rng = np.random.default_rng(seed)
    src = (0.1 * rng.standard_normal((n, 2))).astype(np.float32)
    layers = live_mix.Layers(src, [0, 1])
    layers.array("light")[:] = (0.05 * rng.standard_normal((n, 2))).astype(np.float32)
    layers.mark("light", 0, n)
    return live_mix.LiveMix(layers, noise_timeline(n / SR, db))


def noise_timeline(seconds: float, db: float) -> Timeline:
    return Timeline(seconds).paint("noise", 0, seconds, {"strength_db": db, "remover": "light"})


# ---------------------------------------------------------------------------------------------------- the scale

def test_the_four_names_are_equally_spaced_on_the_scale_with_the_dB_between_them():
    assert blend.POINTS == (12.0, 24.0, 40.0, 100.0)
    assert [blend.pos_of(db) for db in blend.POINTS] == [0.0, 1.0, 2.0, 3.0]
    assert blend.pos_of(18) == 0.5 and blend.pos_of(70) == 2.5 and blend.pos_of(5) == 0.0 and blend.pos_of(500) == 3.0
    assert [blend.strength_name(d) for d in (12, 24, 40, 100)] == ["Gentle", "Balanced", "Strong", "Maximum"]
    assert blend.strength_name(30) == "Between Balanced and Strong" and blend.strength_name(18) == "Between Gentle and Balanced"
    assert blend.strength_name(70) == "Between Strong and Maximum"
    assert blend.strength_key(24.2) == "balanced" and blend.strength_key(30) is None and blend.strength_key(100) == "max"
    assert {k: db for k, _, db in blend.NAMES} == engine.STRENGTHS  # the same four as the engine's


# ---------------------------------------------------------------------------------------------------- the player

@pytest.fixture
def live(qapp):
    from audio_cleaner_desktop.gui import live_audio

    FakeSink.made.clear()
    stream = mix_at(100.0)
    player = live_audio.LivePlayer(namespace(), sink_factory=sink_factory)
    events = {"state": [], "status": [], "errors": [], "positions": []}
    player.playbackStateChanged.connect(events["state"].append)
    player.mediaStatusChanged.connect(events["status"].append)
    player.errorOccurred.connect(lambda e, m: events["errors"].append(m))
    player.positionChanged.connect(events["positions"].append)
    player.load(stream)
    yield player, stream, events
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


def test_a_change_of_the_timeline_while_it_plays_changes_the_next_sound(live):
    player, stream, events, *_ = live
    player.play()
    sink = FakeSink.made[-1]
    first = sink.pull(4800)
    stream.set_timeline(noise_timeline(2.0, 12.0))  # the strength of the part is changed in the editor
    second = sink.pull(4800)
    stream.set_timeline(noise_timeline(2.0, 100.0))
    assert first == stream.read_pcm16(0, 4800)
    assert second != stream.read_pcm16(4800, 4800)  # it was the gentler mix
    stream.set_timeline(noise_timeline(2.0, 12.0))
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

    stream = mix_at(100.0, 4800)

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
