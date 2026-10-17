from __future__ import annotations

import os
import re
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


def test_strength_is_automatic_unless_chosen():
    from audio_cleaner_desktop import __main__ as entry

    parser = entry.build_parser()
    assert parser.parse_args(["--cli", "a.wav"]).strength == "auto"
    for value in ("auto", "gentle", "balanced", "strong", "max"):
        assert parser.parse_args(["--cli", "--strength", value, "a.wav"]).strength == value
    with pytest.raises(SystemExit):
        parser.parse_args(["--cli", "--strength", "loud", "a.wav"])
    assert "auto" in run("--help").stdout


def test_automatic_rebuild_is_on_unless_turned_off(monkeypatch):
    from audio_cleaner_desktop import __main__ as entry
    from audio_cleaner_desktop import cli, engine

    parser = entry.build_parser()
    assert parser.parse_args(["--cli", "a.wav"]).no_auto_rebuild is False
    assert parser.parse_args(["--cli", "--no-auto-rebuild", "a.wav"]).no_auto_rebuild is True
    seen = []
    monkeypatch.setattr(cli, "run", lambda inputs, out_dir, options, *a, **k: seen.append(options) or 0)
    assert entry.main(["--cli", "a.wav"]) == 0 and entry.main(["--cli", "--no-auto-rebuild", "a.wav"]) == 0
    assert [(o.auto_rebuild, o.rebuild_voice, o.strength) for o in seen] == [(True, False, "auto"), (False, False, "auto")]
    assert "--no-auto-rebuild" in run("--help").stdout


def test_the_memory_plan_allows_for_automatic_rebuilding(tmp_path, monkeypatch):
    from audio_cleaner_desktop import cli, engine, resources, sysinfo

    planned = []
    monkeypatch.setattr(resources, "check", lambda: [])
    monkeypatch.setattr(sysinfo, "max_parallel", lambda parallel, subs, **k: planned.append(k["rebuild_voice"]) or 1)
    monkeypatch.setattr(engine, "process", lambda *a, **k: engine.JobResult(tmp_path / "a-cleaned.wav", 1.0, False))
    monkeypatch.setattr(resources, "voice_model_dir", lambda: tmp_path)
    assert cli.run([str(tmp_path / "a.wav")], None, engine.JobOptions(auto_rebuild=True)) == 0
    monkeypatch.setattr(resources, "voice_model_dir", lambda: None)
    assert cli.run([str(tmp_path / "a.wav")], None, engine.JobOptions(auto_rebuild=True)) == 0
    assert planned == [True, False]


def test_the_picked_strength_is_printed(tmp_path, monkeypatch, capsys):
    from audio_cleaner_desktop import cli, engine, resources

    monkeypatch.setattr(resources, "check", lambda: [])
    results = iter([
        engine.JobResult(tmp_path / "a-cleaned.wav", 1.0, False, strength="balanced"),
        engine.JobResult(tmp_path / "b-cleaned.wav", 1.0, False, strength="strong", notice="missing highs restored"),
        engine.JobResult(tmp_path / "c-cleaned.wav", 1.0, False),
    ])
    monkeypatch.setattr(engine, "process", lambda *a, **k: next(results))
    assert cli.run([str(tmp_path / n) for n in ("a.wav", "b.wav", "c.wav")], None, engine.JobOptions(strength="auto"), parallel=1) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if "->" in line]
    assert lines[0].endswith("a-cleaned.wav  (balanced noise reduction picked for this file)")
    assert lines[1].endswith("b-cleaned.wav  (strong noise reduction picked for this file; missing highs restored)")
    assert lines[2].endswith("c-cleaned.wav")


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


@pytest.mark.native
def test_a_file_cleaned_with_the_automatic_strength_says_what_it_got(clip_wav, tmp_path):
    r = run("--cli", "-o", str(tmp_path / "auto"), str(clip_wav))
    assert r.returncode == 0, r.stderr
    assert re.search(r"clip-cleaned\.wav  \((gentle|balanced|strong|max) noise reduction picked for this file", r.stdout), r.stdout
    r = run("--cli", "--strength", "max", "-o", str(tmp_path / "max"), str(clip_wav))
    assert r.returncode == 0, r.stderr
    assert "picked for this file" not in r.stdout
