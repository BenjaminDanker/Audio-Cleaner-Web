from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from acw_common import clarity, pipeline
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
    with pytest.raises(ValueError):
        engine.JobOptions(strength="loud")
    with pytest.raises(ValueError):
        engine.JobOptions(subtitles="burn")
    with pytest.raises(ValueError):
        engine.JobOptions(language="xx")
    with pytest.raises(ValueError):
        engine.JobOptions(loudness="radio")
    assert engine.JobOptions(loudness="keep").loudness_lufs is None
    assert engine.JobOptions().loudness_lufs == -16.0


def test_loudness_presets():
    assert engine.LOUDNESS_TARGETS == {"podcast": -16.0, "youtube": -14.0, "broadcast": -23.0, "keep": None}


def test_mp3_is_an_output_format():
    assert "mp3" in engine.OUTPUT_AUDIO_FORMATS and "mp3" in engine.AUDIO_CHOICES


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


def test_segments_are_shifted_by_the_piece_offset():
    from acw_common.captions import Segment, Word

    seg = Segment(1.0, 2.0, "hi", [Word(1.0, 1.5, "hi")])
    engine._shifted([seg], 300.0)
    assert (seg.start, seg.end, seg.words[0].start, seg.words[0].end) == (301.0, 302.0, 301.0, 301.5)


# -- the real pipeline -------------------------------------------------------------

@pytest.mark.native
@pytest.mark.speech
def test_a_speech_model_is_found_from_source():
    assert (resources.whisper_model_path() / "model.bin").is_file()
    assert resources.check() == [] and resources.speech_ready()


def test_the_speech_model_is_not_required_to_start_and_never_comes_from_the_bundle(monkeypatch, tmp_path):
    monkeypatch.setenv("ACW_DESKTOP_MODELS", str(tmp_path / "none"))
    monkeypatch.setattr(resources.sys, "frozen", True, raising=False)
    monkeypatch.setattr(resources.sys, "_MEIPASS", str(tmp_path), raising=False)
    assert resources.whisper_model_path() is None and not resources.speech_ready()
    assert all("whisper" not in m for m in resources.check())


@pytest.mark.native
def test_wav_to_flac_into_out_dir(speech_wav, tmp_path):
    out = engine.process_file(speech_wav, engine.JobOptions(audio_format="flac"), out_dir=tmp_path / "results")
    assert out.parent == tmp_path / "results"
    assert out.name == "speech-cleaned.flac"
    info = ffprobe_json(out)
    assert info["streams"][0]["codec_name"] == "flac"
    assert float(info["format"]["duration"]) == pytest.approx(float(ffprobe_json(speech_wav)["format"]["duration"]), abs=0.3)


@pytest.mark.native
def test_mp3_stays_mp3(mp3_file, tmp_path):
    out = engine.process_file(mp3_file, out_dir=tmp_path)
    assert out.suffix == ".mp3"
    assert streams(out, "audio")[0]["codec_name"] == "mp3"


@pytest.mark.native
def test_other_formats_are_written(mp3_file, tmp_path):
    for fmt, codec in (("wav", "pcm_s24le"), ("m4a", "aac"), ("opus", "opus"), ("mp3", "mp3")):
        out = engine.process_file(mp3_file, engine.JobOptions(audio_format=fmt, polish=False), out_dir=tmp_path)
        assert streams(out, "audio")[0]["codec_name"] == codec, fmt


@pytest.mark.native
def test_video_keeps_its_picture_and_creation_time(mp4_file, tmp_path):
    out = engine.process_file(mp4_file, out_dir=tmp_path)
    assert out.suffix == ".mp4"
    assert [s["codec_name"] for s in streams(out, "video")] == ["h264"]
    assert streams(out, "audio")
    assert ffprobe_json(out)["format"]["tags"]["creation_time"].startswith("2021-03-04T05:06:07")


@pytest.mark.native
def test_an_existing_result_is_never_replaced(speech_wav, tmp_path):
    src = tmp_path / "talk.wav"
    src.write_bytes(speech_wav.read_bytes())
    first = engine.process_file(src)
    second = engine.process_file(src)
    assert first.name == "talk-cleaned.wav" and second.name == "talk-cleaned (2).wav"
    assert first.exists() and second.exists()


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
def test_loudness_presets_set_the_level(speech_wav, has_real_speech, tmp_path):
    levels = {
        name: loudness_of(engine.process_file(speech_wav, engine.JobOptions(loudness=name), out_dir=tmp_path / name))
        for name in ("podcast", "youtube", "broadcast")
    }
    assert levels["youtube"] > levels["podcast"] > levels["broadcast"]
    if has_real_speech:
        # The level is set from the speech in a recording; the stand-in tones used where the dev
        # sample is missing (CI) only keep the order, so only real speech is held to the targets.
        assert levels["youtube"] - levels["podcast"] == pytest.approx(2.0, abs=1.0)
        assert levels["podcast"] - levels["broadcast"] == pytest.approx(7.0, abs=1.5)
    kept = loudness_of(engine.process_file(speech_wav, engine.JobOptions(loudness="keep"), out_dir=tmp_path / "keep"))
    assert abs(kept - levels["podcast"]) > 0.1 or abs(kept - -16.0) > 0.1  # not forced to the podcast target


@pytest.mark.native
@pytest.mark.speech
def test_srt_next_to_the_cleaned_audio(speech_wav, has_real_speech, tmp_path):
    if not has_real_speech:
        pytest.skip("needs the real speech sample")
    result = engine.process(speech_wav, engine.JobOptions(subtitles="srt", language="en"), out_dir=tmp_path)
    assert result.subtitles == tmp_path / "speech-cleaned.srt" and not result.embedded
    assert result.language == "en"
    assert result.paths == [result.output, result.subtitles]
    text = result.subtitles.read_text(encoding="utf-8")
    assert "-->" in text and "responsible" in text.lower()


@pytest.mark.native
@pytest.mark.speech
def test_vtt_and_text_transcripts(speech_wav, has_real_speech, tmp_path):
    if not has_real_speech:
        pytest.skip("needs the real speech sample")
    vtt = engine.process(speech_wav, engine.JobOptions(subtitles="vtt"), out_dir=tmp_path / "v").subtitles
    assert vtt.suffix == ".vtt" and vtt.read_text(encoding="utf-8").startswith("WEBVTT")
    txt = engine.process(speech_wav, engine.JobOptions(subtitles="txt"), out_dir=tmp_path / "t").subtitles
    assert txt.suffix == ".txt" and "responsible" in txt.read_text(encoding="utf-8").lower()
    assert "-->" not in txt.read_text(encoding="utf-8")


@pytest.mark.native
@pytest.mark.speech
def test_subtitles_embedded_in_a_video(mp4_file, has_real_speech, tmp_path):
    if not has_real_speech:
        pytest.skip("needs the real speech sample")
    result = engine.process(mp4_file, engine.JobOptions(subtitles="embed"), out_dir=tmp_path)
    assert result.embedded and result.subtitles is None
    assert list(tmp_path.glob("*.srt")) == []
    subs = streams(result.output, "subtitle")
    assert [s["codec_name"] for s in subs] == ["mov_text"]
    assert subs[0]["tags"]["language"] == "eng"
    assert streams(result.output, "video")


@pytest.mark.native
@pytest.mark.speech
def test_embed_on_an_audio_file_writes_an_srt(speech_wav, has_real_speech, tmp_path):
    if not has_real_speech:
        pytest.skip("needs the real speech sample")
    result = engine.process(speech_wav, engine.JobOptions(subtitles="embed"), out_dir=tmp_path)
    assert result.subtitles is not None and result.subtitles.suffix == ".srt" and not result.embedded


@pytest.mark.native
@pytest.mark.speech
def test_silence_gets_no_subtitle_file(tmp_path):
    silent = tmp_path / "silent.wav"
    from scipy.io import wavfile

    wavfile.write(silent, 48000, np.zeros(3 * 48000, dtype=np.int16))
    result = engine.process(silent, engine.JobOptions(subtitles="srt"), out_dir=tmp_path / "out")
    assert result.subtitles is None and not result.embedded
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["silent-cleaned.wav"]


def test_noise_and_echo_options_are_checked():
    assert engine.JobOptions().noise == "auto" and not engine.JobOptions().echo_track
    for value in ("auto", "light", "heavy"):
        assert engine.JobOptions(noise=value).noise == value
    with pytest.raises(ValueError, match="noise"):
        engine.JobOptions(noise="loud")


@pytest.mark.native
def test_echo_option_on_a_one_track_file_cleans_it_and_says_so(speech_wav, tmp_path):
    result = engine.process(speech_wav, engine.JobOptions(echo_track=True, noise="light"), out_dir=tmp_path)
    assert result.output.is_file() and result.notice == "only one audio track, so no speaker echo was removed"


@pytest.mark.native
def test_heavy_noise_setting_runs_the_heavy_noise_model(speech_wav, tmp_path, monkeypatch):
    from acw_common import pipeline

    if not resources.heavy_noise_model_path().is_file():
        pytest.skip("no heavy-noise model in vendor/models")
    used = []
    real = pipeline.dpdfnet.denoise_array
    monkeypatch.setattr(pipeline.dpdfnet, "denoise_array", lambda *a, **k: used.append(1) or real(*a, **k))
    result = engine.process(speech_wav, engine.JobOptions(noise="heavy"), out_dir=tmp_path)
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
    out = engine.process(two, engine.JobOptions(echo_track=True, polish=False, strength="gentle", audio_format="wav"),
                              out_dir=tmp_path / "out")
    assert out.notice is None
    rate, y = wavfile.read(out.output)
    scale = {np.dtype(np.int16): 2.0**15, np.dtype(np.int32): 2.0**31}.get(y.dtype, 1.0)
    y = y.astype(np.float32) / scale
    y = y if y.ndim == 1 else y.mean(axis=1)
    def level(a): return 10 * np.log10(np.mean(a**2) + 1e-12)
    echo_only = slice(sr * 2, sr * 4)  # before the voice starts, after the canceller has adapted
    assert level(mic[echo_only]) - level(y[echo_only]) > 20


@pytest.mark.native
def test_rebuild_voice_uses_the_voice_model_or_says_it_is_missing(speech_wav, tmp_path, monkeypatch):
    if resources.voice_model_dir() is None:
        monkeypatch.setattr(resources, "voice_model_dir", lambda: None)
        result = engine.process(speech_wav, engine.JobOptions(rebuild_voice=True), out_dir=tmp_path)
        assert "rebuilds the voice is not installed" in result.notice
        return
    result = engine.process(speech_wav, engine.JobOptions(rebuild_voice=True, audio_format="wav"), out_dir=tmp_path)
    assert result.output.is_file() and (result.notice is None or "not installed" not in result.notice)
    monkeypatch.setattr(resources, "voice_model_dir", lambda: None)
    fallback = engine.process(speech_wav, engine.JobOptions(rebuild_voice=True), out_dir=tmp_path / "b")
    assert "cleaned the usual way" in fallback.notice
