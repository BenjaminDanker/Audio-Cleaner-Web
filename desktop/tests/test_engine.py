from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from acw_common import captions, clarity, pipeline
from acw_common.ffmpeg import MediaError
from audio_cleaner_desktop import engine, resources

from conftest import ffprobe_json


def loudness_of(path: Path) -> float:
    ff = engine.ffmpeg()
    info = ff.probe(path)
    audio = ff.decode(path, info, pipeline.SAMPLE_RATE, 1, 600)
    return clarity.integrated_loudness(audio[:, 0], pipeline.SAMPLE_RATE)


def streams(path: Path, kind: str) -> list[dict]:
    return [s for s in ffprobe_json(path)["streams"] if s["codec_type"] == kind]


# -- pure logic (no native pieces needed) ----------------------------------------


def test_options_are_validated():
    for bad in (dict(strength="loud"), dict(subtitles="burn"), dict(language="xx"), dict(loudness="radio"),
                dict(noise="loud")):
        with pytest.raises(ValueError):
            engine.JobOptions(**bad)
    assert engine.JobOptions(loudness="keep").loudness_lufs is None
    assert engine.JobOptions().loudness_lufs == -16.0
    assert engine.JobOptions().noise == "auto" and not engine.JobOptions().echo_track


def test_automatic_strength_is_a_choice_and_its_steps_are_the_pipelines():
    auto = engine.JobOptions(strength="auto")
    assert auto.strength == engine.AUTO_STRENGTH and auto.atten_lim_db is None  # measured when the file is cleaned
    assert engine.JobOptions().strength == "max"  # the engine's own default stays a fixed one
    with pytest.raises(ValueError, match="strength"):
        engine.JobOptions(strength="automatic")
    # every strength the pipeline can pick is one of the fixed ones (so the list and history can name it)
    assert {db for _, db in pipeline.STRENGTH_STEPS} | {pipeline.GENTLE_DB} == set(engine.STRENGTHS.values())


def test_automatic_rebuild_is_an_option_that_needs_the_voice_model(monkeypatch, tmp_path):
    assert engine.JobOptions().auto_rebuild is False  # the engine rebuilds nothing unless asked
    monkeypatch.setattr(resources, "voice_model_dir", lambda: None)
    assert not engine.JobOptions(auto_rebuild=True).may_rebuild
    assert engine.JobOptions(rebuild_voice=True).may_rebuild  # asked for by hand: planned for (and refused later)
    monkeypatch.setattr(resources, "voice_model_dir", lambda: tmp_path)
    assert engine.JobOptions(auto_rebuild=True).may_rebuild and not engine.JobOptions().may_rebuild


def test_unique_output_counts_up_and_respects_subtitle_files(tmp_path):
    assert engine.unique_output(tmp_path, "a", ".wav").name == "a-cleaned.wav"
    (tmp_path / "a-cleaned.wav").touch()
    assert engine.unique_output(tmp_path, "a", ".wav").name == "a-cleaned (2).wav"
    (tmp_path / "b-cleaned.srt").touch()
    assert engine.unique_output(tmp_path, "b", ".wav", ".srt").name == "b-cleaned (2).wav"
    assert engine.unique_output(tmp_path, "b", ".wav").name == "b-cleaned.wav"


def test_long_recordings_are_transcribed_in_pieces_cut_at_quiet_moments():
    sr = engine._SPEECH_RATE
    rng = np.random.default_rng(0)
    seconds = 2 * engine._PIECE_SECONDS + 100
    audio = (0.1 * rng.standard_normal(seconds * sr)).astype(np.float32)
    quiet = engine._PIECE_SECONDS - 10  # a silent gap inside the search window of the first cut
    audio[quiet * sr:(quiet + 1) * sr] = 0
    bounds = engine._piece_bounds(audio.reshape(-1, 1))
    assert bounds[0][0] == 0 and bounds[-1][1] == len(audio)
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:]))
    assert len(bounds) == 3
    assert quiet * sr <= bounds[0][1] <= (quiet + 1) * sr
    from acw_common.captions import Segment, Word

    seg = Segment(1.0, 2.0, "hi", [Word(1.0, 1.5, "hi")])
    engine._shifted([seg], 300.0)  # the pieces' times are moved to the start of the recording
    assert (seg.start, seg.end, seg.words[0].start, seg.words[0].end) == (301.0, 302.0, 301.0, 301.5)


def test_subtitle_files_are_srt_vtt_or_plain_text():
    segments = [captions.Segment(0.5, 1.8, "Hello there", [])]
    srt = engine._subtitle_text("srt", segments, "en")
    assert "00:00:00,500 --> 00:00:01,800" in srt and "Hello there" in srt
    vtt = engine._subtitle_text("vtt", segments, "en")
    assert vtt.startswith("WEBVTT") and "00:00:00.500 --> 00:00:01.800" in vtt
    txt = engine._subtitle_text("txt", segments, "en")
    assert "Hello there" in txt and "-->" not in txt


def test_the_speech_model_is_not_required_to_start_and_never_comes_from_the_bundle(monkeypatch, tmp_path):
    monkeypatch.setenv("ACW_DESKTOP_MODELS", str(tmp_path / "none"))
    monkeypatch.setattr(resources.sys, "frozen", True, raising=False)
    monkeypatch.setattr(resources.sys, "_MEIPASS", str(tmp_path), raising=False)
    assert resources.whisper_model_path() is None and not resources.speech_ready()
    assert all("whisper" not in m for m in resources.check())


# -- the real pipeline -------------------------------------------------------------

@pytest.mark.native
def test_wav_to_flac_into_out_dir(speech_wav, tmp_path):
    out = engine.process_file(speech_wav, engine.JobOptions(audio_format="flac"), out_dir=tmp_path / "results")
    assert out.parent == tmp_path / "results" and out.name == "speech-cleaned.flac"
    info = ffprobe_json(out)
    assert info["streams"][0]["codec_name"] == "flac"
    assert float(info["format"]["duration"]) == pytest.approx(float(ffprobe_json(speech_wav)["format"]["duration"]), abs=0.3)


@pytest.mark.native
def test_mp3_stays_mp3(mp3_file, tmp_path):
    out = engine.process_file(mp3_file, out_dir=tmp_path)
    assert out.suffix == ".mp3" and streams(out, "audio")[0]["codec_name"] == "mp3"


@pytest.mark.native
def test_video_keeps_its_picture_and_creation_time(mp4_file, tmp_path):
    out = engine.process(mp4_file, out_dir=tmp_path).output
    assert out.suffix == ".mp4"
    assert [s["codec_name"] for s in streams(out, "video")] == ["h264"]
    assert streams(out, "audio")
    assert ffprobe_json(out)["format"]["tags"]["creation_time"].startswith("2021-03-04T05:06:07")


@pytest.mark.native
def test_cancel_leaves_nothing_behind(speech_wav, tmp_path):
    src = tmp_path / "talk.wav"
    src.write_bytes(speech_wav.read_bytes())
    cancel = threading.Event()

    def stop_at_once(_fraction: float) -> None:
        cancel.set()

    with pytest.raises(engine.Cancelled):
        engine.process_file(src, progress=stop_at_once, cancel=cancel)
    assert [p.name for p in tmp_path.iterdir()] == ["talk.wav"]


@pytest.mark.native
def test_cancel_while_cleaning_releases_the_scratch_files_first(speech_wav, tmp_path, monkeypatch):
    """Windows can't delete a memory-mapped file: when the scratch folder is removed after a cancel
    in the middle of cleaning, nothing in it may still be mapped (checked via /proc on Linux)."""
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        pytest.skip("needs /proc/self/maps")
    src = tmp_path / "talk.wav"
    src.write_bytes(speech_wav.read_bytes())
    cancel = threading.Event()
    still_mapped = []
    remove = engine._remove_work_dir

    def checked_remove(work):
        still_mapped.extend(line for line in maps.read_text().splitlines() if str(work) in line)
        remove(work)

    monkeypatch.setattr(engine, "_remove_work_dir", checked_remove)

    def stop_while_cleaning(fraction: float) -> None:
        if fraction > 0.2:  # past decoding: the decoded and cleaned arrays are mapped now
            cancel.set()

    with pytest.raises(engine.Cancelled):
        engine.process_file(src, progress=stop_while_cleaning, cancel=cancel)
    assert still_mapped == []
    assert [p.name for p in tmp_path.iterdir()] == ["talk.wav"]


@pytest.mark.native
def test_unreadable_files_give_a_plain_message(tmp_path):
    garbage = tmp_path / "noise.wav"
    garbage.write_bytes(np.random.default_rng(0).bytes(5000))
    with pytest.raises(MediaError, match="could not be read"):
        engine.process_file(garbage)
    with pytest.raises(MediaError, match="not found"):
        engine.process_file(tmp_path / "missing.wav")
    assert [p.name for p in tmp_path.iterdir()] == ["noise.wav"]


@pytest.mark.native
def test_loudness_presets_set_the_level(clip_wav, tmp_path):
    levels = {
        name: loudness_of(engine.process_file(clip_wav, engine.JobOptions(loudness=name), out_dir=tmp_path / name))
        for name in ("podcast", "youtube", "broadcast")
    }
    assert levels["youtube"] > levels["podcast"] > levels["broadcast"]
    assert levels["podcast"] == pytest.approx(-16.0, abs=1.0)


@pytest.mark.native
@pytest.mark.speech
def test_srt_next_to_the_cleaned_audio(speech_wav, has_real_speech, tmp_path):
    """The one run with the real speech model: the subtitles come from the transcription of the speech."""
    if not has_real_speech:
        pytest.skip("needs the real speech sample")
    result = engine.process(speech_wav, engine.JobOptions(subtitles="srt", language="en"), out_dir=tmp_path)
    assert result.subtitles == tmp_path / "speech-cleaned.srt" and not result.embedded
    assert result.language == "en" and result.paths == [result.output, result.subtitles]
    text = result.subtitles.read_text(encoding="utf-8")
    assert "-->" in text and "responsible" in text.lower()


@pytest.mark.native
def test_silence_gets_no_subtitle_file(clip_wav, tmp_path, quick_clean, monkeypatch):
    monkeypatch.setattr(engine, "_transcribe", lambda *a, **k: ([captions.Segment(0.0, 0.0, "  ", [])], "en"))
    result = engine.process(clip_wav, engine.JobOptions(subtitles="srt"), out_dir=tmp_path / "out")
    assert result.subtitles is None and not result.embedded
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["clip-cleaned.wav"]


@pytest.mark.native
def test_echo_option_on_a_one_track_file_cleans_it_and_says_so(clip_wav, tmp_path):
    result = engine.process(clip_wav, engine.JobOptions(echo_track=True, noise="light"), out_dir=tmp_path)
    assert result.output.is_file() and result.notice == "only one audio track, so no speaker echo was removed"


@pytest.mark.native
def test_heavy_noise_setting_runs_the_heavy_noise_model(clip_wav, tmp_path, monkeypatch):
    if not resources.heavy_noise_model_path().is_file():
        pytest.skip("no heavy-noise model in vendor/models")
    used = []
    real = pipeline.dpdfnet.denoise_array
    monkeypatch.setattr(pipeline.dpdfnet, "denoise_array", lambda *a, **k: used.append(1) or real(*a, **k))
    result = engine.process(clip_wav, engine.JobOptions(noise="heavy"), out_dir=tmp_path)
    assert result.output.is_file() and used


@pytest.mark.native
def test_echo_removed_using_the_second_audio_track(tmp_path):
    import subprocess

    from conftest import bundled_ffmpeg, ffprobe_json

    # mic = voice-like tone + the speaker sound 20 ms later; track 2 = the speaker sound
    sr = 48000
    t = np.arange(sr * 8) / sr
    rng = np.random.default_rng(3)
    speaker = (0.2 * rng.standard_normal(t.size)).astype(np.float32)
    voice = (0.2 * np.sin(2 * np.pi * 220 * t) * (t > 5)).astype(np.float32)
    mic = voice + 0.6 * np.concatenate([np.zeros(960, np.float32), speaker[:-960]])
    from scipy.io import wavfile

    wavfile.write(tmp_path / "mic.wav", sr, (mic * 32767).astype(np.int16))
    wavfile.write(tmp_path / "spk.wav", sr, (speaker * 32767).astype(np.int16))
    two = tmp_path / "twotrack.mkv"
    subprocess.run([str(bundled_ffmpeg()), "-v", "error", "-y", "-i", str(tmp_path / "mic.wav"), "-i", str(tmp_path / "spk.wav"),
                    "-map", "0:a", "-map", "1:a", "-c:a", "flac", "-f", "matroska", str(two)], check=True)
    assert len([s for s in ffprobe_json(two)["streams"] if s["codec_type"] == "audio"]) == 2
    out = engine.process(two, engine.JobOptions(echo_track=True, leveling=False, strength="gentle", audio_format="wav"),
                         out_dir=tmp_path / "out")
    assert out.notice == "only the first audio track was cleaned; the other one wasn't kept in the result"
    rate, y = wavfile.read(out.output)
    scale = {np.dtype(np.int16): 2.0**15, np.dtype(np.int32): 2.0**31}.get(y.dtype, 1.0)
    y = y.astype(np.float32) / scale
    y = y if y.ndim == 1 else y.mean(axis=1)

    def level(a):
        return 10 * np.log10(np.mean(a**2) + 1e-12)

    echo_only = slice(sr * 2, sr * 4)  # before the voice starts, after the canceller has adapted
    assert level(mic[echo_only]) - level(y[echo_only]) > 20


@pytest.fixture
def fake_clean(monkeypatch):
    """clean_audio recording what it was asked for; the noise removal reports a Balanced strength."""
    seen = []

    def clean(x, lib, model, opts, *, report=None, **_):
        seen.append(opts)
        if report is not None:
            report.update(noise="rebuilt", highs=None) if opts.restore_voice else report.update(
                noise="light", highs=None, atten_lim_db=24.0, noise_db=21.0)
        return np.array(x, dtype=np.float32)

    monkeypatch.setattr(pipeline, "clean_audio", clean)
    return seen


@pytest.mark.native
def test_rebuild_voice_is_done_when_its_model_is_installed(clip_wav, tmp_path):
    if resources.voice_model_dir() is None:
        pytest.skip("no voice model in vendor/models")
    result = engine.process(clip_wav, engine.JobOptions(rebuild_voice=True, audio_format="wav"), out_dir=tmp_path)
    assert result.output.is_file() and (result.notice is None or "not installed" not in result.notice)


@pytest.mark.native
def test_rebuild_voice_without_its_model_is_said_and_cleaned_the_usual_way(clip_wav, tmp_path, monkeypatch, quick_clean):
    monkeypatch.setattr(resources, "voice_model_dir", lambda: None)
    result = engine.process(clip_wav, engine.JobOptions(rebuild_voice=True), out_dir=tmp_path)
    assert result.output.is_file() and "cleaned the usual way" in result.notice


@pytest.mark.native
def test_automatic_strength_says_what_it_picked(clip_wav, tmp_path, monkeypatch):
    figures = []
    real = pipeline.noise_figure_db
    monkeypatch.setattr(pipeline, "noise_figure_db", lambda *a, **k: figures.append(real(*a, **k)) or figures[-1])
    result = engine.process(clip_wav, engine.JobOptions(strength="auto"), out_dir=tmp_path / "auto")
    assert result.output.is_file() and len(figures) == 1  # one measurement for the remover and the strength
    assert result.strength in engine.STRENGTHS
    assert engine.STRENGTHS[result.strength] == pipeline.strength_for(figures[0])
    fixed = engine.process(clip_wav, engine.JobOptions(strength="gentle", noise="light"), out_dir=tmp_path / "fixed")
    assert fixed.output.is_file() and fixed.strength is None and len(figures) == 1  # nothing picked, nothing measured


@pytest.mark.native
@pytest.mark.parametrize("reason", ["echoey", "muffled"])
def test_a_file_that_needs_it_gets_its_voice_rebuilt(clip_wav, tmp_path, monkeypatch, fake_clean, reason):
    asked = []
    monkeypatch.setattr(engine.advice, "rebuild_reason", lambda x: asked.append(len(x)) or reason)
    monkeypatch.setattr(resources, "voice_model_dir", lambda: tmp_path)
    result = engine.process(clip_wav, engine.JobOptions(strength="auto", auto_rebuild=True), out_dir=tmp_path / "a")
    assert len(asked) == 1 and fake_clean[-1].restore_voice
    assert result.rebuilt == reason and result.strength is None  # rebuilt: no noise reduction strength was used
    assert result.notice == f"the voice was rebuilt because it sounded {reason}"
    # without the model: cleaned the usual way, and told how to fix it
    monkeypatch.setattr(resources, "voice_model_dir", lambda: None)
    result = engine.process(clip_wav, engine.JobOptions(strength="auto", auto_rebuild=True), out_dir=tmp_path / "b")
    assert not fake_clean[-1].restore_voice and fake_clean[-1].atten_lim_db is None
    assert result.rebuilt is None and result.strength == "balanced"
    assert result.notice == f"it sounds {reason}: download the model that rebuilds the voice in Settings and clean it again to fix that"


@pytest.mark.native
def test_automatic_rebuild_leaves_other_files_and_choices_alone(clip_wav, tmp_path, monkeypatch, fake_clean):
    asked = []
    monkeypatch.setattr(engine.advice, "rebuild_reason", lambda x: asked.append(1) and None)
    monkeypatch.setattr(resources, "voice_model_dir", lambda: tmp_path)
    result = engine.process(clip_wav, engine.JobOptions(strength="auto", auto_rebuild=True), out_dir=tmp_path / "a")
    assert asked and not fake_clean[-1].restore_voice and result.rebuilt is None and result.notice is None
    assert result.strength == "balanced"
    asked.clear()
    engine.process(clip_wav, engine.JobOptions(), out_dir=tmp_path / "b")  # not asked for: not measured
    result = engine.process(clip_wav, engine.JobOptions(rebuild_voice=True, auto_rebuild=True), out_dir=tmp_path / "c")
    assert not asked  # rebuilt anyway: nothing to decide
    assert fake_clean[-1].restore_voice and result.rebuilt is None and result.notice is None


@pytest.mark.native
def test_automatic_rebuild_on_a_real_file(clip_wav, tmp_path):
    """The real measurement on the way: the noisy tone of clip_wav has no words, so it is never rebuilt."""
    result = engine.process(clip_wav, engine.JobOptions(strength="auto", auto_rebuild=True), out_dir=tmp_path)
    assert result.output.is_file() and result.rebuilt is None and result.strength in engine.STRENGTHS
