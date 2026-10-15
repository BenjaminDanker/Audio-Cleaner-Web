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
                 "--rebuild-voice", "--parallel", "--download-models", "--download-speech-model", "--gui"):
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
