from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENV = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "desktop"), str(ROOT / "common")]), "QT_QPA_PLATFORM": "offscreen"}


def run(*args: str, timeout: float = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "audio_cleaner_desktop", *args], capture_output=True, text=True, env=ENV, timeout=timeout,
        stdin=subprocess.DEVNULL,
    )


def test_version():
    r = run("--version")
    assert r.returncode == 0 and r.stdout.strip() == "Audio Cleaner 2.0.0"


def test_help_lists_the_new_options():
    r = run("--help")
    assert r.returncode == 0
    for flag in ("--subtitles", "--language", "--loudness", "--audio-format", "--noise", "--echo-track", "--no-restore-highs",
                 "--rebuild-voice", "--parallel", "--download-models", "--download-speech-model", "--gui",
                 "--output-name", "--format", "--video-size"):
        assert flag in r.stdout


@pytest.mark.native
def test_selftest():
    r = run("--selftest")
    assert r.returncode == 0 and "selftest ok" in r.stdout


@pytest.mark.native
def test_clean_a_wav(speech_wav, tmp_path):
    out_dir = tmp_path / "out"
    r = run("--cli", "--strength", "balanced", "--audio-format", "flac", "-o", str(out_dir), str(speech_wav))
    assert r.returncode == 0, r.stderr
    assert (out_dir / "speech-cleaned.flac").is_file()
    assert "speech-cleaned.flac" in r.stdout


@pytest.mark.native
def test_a_bad_file_fails_cleanly(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio" * 100)
    r = run("--cli", str(bad))
    assert r.returncode == 1 and "failed" in r.stderr and "Traceback" not in r.stderr


def test_cli_options_reach_the_job():
    from audio_cleaner_desktop import __main__ as entry

    args = entry.build_parser().parse_args(
        ["--cli", "--noise", "heavy", "--echo-track", "--no-restore-highs", "--rebuild-voice", "--parallel", "2", "a.wav"]
    )
    assert (args.noise, args.echo_track, args.no_restore_highs, args.rebuild_voice, args.parallel) == ("heavy", True, True, True, 2)
    assert entry.build_parser().parse_args(["--download-speech-model"]).download_models  # the old name still works


def test_rebuilding_the_voice_without_its_model_says_how_to_get_it(tmp_path, monkeypatch, capsys):
    from audio_cleaner_desktop import cli, engine, resources

    monkeypatch.setattr(resources, "check", lambda: [])
    monkeypatch.setattr(resources, "voice_model_dir", lambda: None)
    assert cli.run([str(tmp_path / "a.wav")], None, engine.JobOptions(rebuild_voice=True)) == 1
    assert "--download-models" in capsys.readouterr().err


@pytest.mark.native
def test_selftest_also_builds_the_window():
    r = run("--selftest", "--gui")
    assert r.returncode == 0 and "selftest ok" in r.stdout, r.stdout + r.stderr


def test_the_name_the_type_and_the_video_size_reach_the_job():
    from audio_cleaner_desktop import __main__ as entry

    args = entry.build_parser().parse_args(
        ["--cli", "--output-name", "Voice.mp3", "--format", "mp3", "--video-size", "720", "a.wav"])
    assert (args.output_name, args.output_format, args.video_size) == ("Voice.mp3", "mp3", "720")
    defaults = entry.build_parser().parse_args(["--cli", "a.wav"])
    assert (defaults.output_name, defaults.output_format, defaults.video_size) == (None, None, "same")
    parser = entry.build_parser()
    for bad in (["--format", "avi"], ["--format", "mp3.", "x"], ["--video-size", "900"], ["--video-size", "1080p"]):
        with pytest.raises(SystemExit):
            parser.parse_args(["--cli", *bad, "a.wav"])
    for ok in ("wav", "flac", "mp3", "m4a", "opus", "ogg", "mp4", "mov", "mkv", "flv", "webm"):
        assert parser.parse_args(["--cli", "--format", ok, "a.wav"]).output_format == ok


def test_one_name_cannot_be_given_to_several_files(tmp_path, capsys):
    from audio_cleaner_desktop import __main__ as entry
    from audio_cleaner_desktop import cli, engine

    with pytest.raises(SystemExit) as stop:
        entry.main(["--cli", "--output-name", "x.mp3", "a.wav", "b.wav"])
    assert stop.value.code == 2
    assert "one file at a time" in capsys.readouterr().err
    assert cli.run([str(tmp_path / "a.wav"), str(tmp_path / "b.wav")], None, engine.JobOptions(), output_name="x.mp3") == 1
    assert "one file at a time" in capsys.readouterr().err


@pytest.mark.native
def test_clean_into_a_chosen_name_and_type(clip_wav, tmp_path):
    out_dir = tmp_path / "out"
    r = run("--cli", "-o", str(out_dir), "--output-name", "Voice Memo.mp3", str(clip_wav))
    assert r.returncode == 0, r.stderr
    assert (out_dir / "Voice Memo.mp3").is_file() and "Voice Memo.mp3" in r.stdout
    r = run("--cli", "-o", str(out_dir), "--format", "ogg", str(clip_wav))
    assert r.returncode == 0, r.stderr
    assert (out_dir / "clip-cleaned.ogg").is_file()


@pytest.mark.native
def test_a_sound_made_into_a_video_of_a_chosen_size(clip_wav, tmp_path):
    from conftest import displayed_size

    r = run("--cli", "-o", str(tmp_path), "--format", "mp4", "--video-size", "480", str(clip_wav))
    assert r.returncode == 0, r.stderr
    assert displayed_size(tmp_path / "clip-cleaned.mp4") == (854, 480)


@pytest.mark.native
def test_a_name_the_file_cannot_have_is_reported_plainly(clip_wav, tmp_path):
    r = run("--cli", "-o", str(tmp_path), "--output-name", "x.exe", str(clip_wav))
    assert r.returncode == 1 and "can't be saved as .exe" in r.stderr and "Traceback" not in r.stderr
    assert list(tmp_path.iterdir()) == []
