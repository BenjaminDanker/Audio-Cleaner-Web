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
def test_whisper_model_is_bundled():
    assert (resources.whisper_model_path() / "model.bin").is_file()
    assert resources.check() == []


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
def test_unreadable_files_give_a_plain_message(tmp_path):
    garbage = tmp_path / "noise.wav"
    garbage.write_bytes(np.random.default_rng(0).bytes(5000))
    with pytest.raises(MediaError, match="could not be read"):
        engine.process_file(garbage)
    with pytest.raises(MediaError, match="not found"):
        engine.process_file(tmp_path / "missing.wav")
    assert [p.name for p in tmp_path.iterdir()] == ["noise.wav"]


@pytest.mark.native
def test_loudness_presets_set_the_level(speech_wav, tmp_path):
    levels = {
        name: loudness_of(engine.process_file(speech_wav, engine.JobOptions(loudness=name), out_dir=tmp_path / name))
        for name in ("podcast", "youtube", "broadcast")
    }
    assert levels["youtube"] - levels["podcast"] == pytest.approx(2.0, abs=1.0)
    assert levels["podcast"] - levels["broadcast"] == pytest.approx(7.0, abs=1.5)
    kept = loudness_of(engine.process_file(speech_wav, engine.JobOptions(loudness="keep"), out_dir=tmp_path / "keep"))
    assert abs(kept - levels["podcast"]) > 0.1 or abs(kept - -16.0) > 0.1  # not forced to the podcast target


@pytest.mark.native
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
def test_vtt_and_text_transcripts(speech_wav, has_real_speech, tmp_path):
    if not has_real_speech:
        pytest.skip("needs the real speech sample")
    vtt = engine.process(speech_wav, engine.JobOptions(subtitles="vtt"), out_dir=tmp_path / "v").subtitles
    assert vtt.suffix == ".vtt" and vtt.read_text(encoding="utf-8").startswith("WEBVTT")
    txt = engine.process(speech_wav, engine.JobOptions(subtitles="txt"), out_dir=tmp_path / "t").subtitles
    assert txt.suffix == ".txt" and "responsible" in txt.read_text(encoding="utf-8").lower()
    assert "-->" not in txt.read_text(encoding="utf-8")


@pytest.mark.native
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
def test_embed_on_an_audio_file_writes_an_srt(speech_wav, has_real_speech, tmp_path):
    if not has_real_speech:
        pytest.skip("needs the real speech sample")
    result = engine.process(speech_wav, engine.JobOptions(subtitles="embed"), out_dir=tmp_path)
    assert result.subtitles is not None and result.subtitles.suffix == ".srt" and not result.embedded


@pytest.mark.native
def test_silence_gets_no_subtitle_file(tmp_path):
    silent = tmp_path / "silent.wav"
    from scipy.io import wavfile

    wavfile.write(silent, 48000, np.zeros(3 * 48000, dtype=np.int16))
    result = engine.process(silent, engine.JobOptions(subtitles="srt"), out_dir=tmp_path / "out")
    assert result.subtitles is None and not result.embedded
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["silent-cleaned.wav"]
