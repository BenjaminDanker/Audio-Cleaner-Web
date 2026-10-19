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


def test_help_lists_the_options_and_the_automatic_choices():
    r = run("--help")
    assert r.returncode == 0
    for flag in ("--subtitles", "--language", "--loudness", "--audio-format", "--noise", "--echo-track",
                 "--rebuild-voice", "--parallel", "--download-models", "--download-speech-model", "--gui",
                 "--output-name", "--format", "--video-size", "--no-auto-rebuild"):
        assert flag in r.stdout
    assert "auto" in r.stdout


def test_help_says_what_the_voice_is_rebuilt_for():
    """The help of --rebuild-voice and --no-auto-rebuild names the same causes as the automatic choice (echo, a muffled voice,
    phone-quality or damaged sound, distortion from recording too loud)."""
    r = run("--help")
    assert r.returncode == 0
    flat = re.sub(r"\s+", "", r.stdout)  # (argparse wraps lines, also at a hyphen)
    for words in ("fixes room echo, a muffled voice, phone-quality or damaged sound and distortion from recording too loud",
                  "(echo, a muffled or unclear voice, phone-quality sound, distortion; by default they are, when its model is installed)"):
        assert re.sub(r"\s+", "", words) in flat, words
    assert "also removes room echo" not in " ".join(r.stdout.split())  # the old text


@pytest.mark.native
def test_selftest_passes_and_builds_the_window():
    """`--selftest --gui` runs everything `--selftest` does, and also builds the window and the preview."""
    r = run("--selftest", "--gui")
    assert r.returncode == 0 and "selftest ok" in r.stdout, r.stdout + r.stderr


@pytest.mark.native
def test_a_file_is_cleaned_from_the_command_line_with_a_chosen_name_and_the_picked_strength(clip_wav, tmp_path):
    out_dir = tmp_path / "out"
    # (--no-auto-rebuild: the made-up tone is one the model would rebuild, which leaves no strength to pick)
    r = run("--cli", "--no-auto-rebuild", "-o", str(out_dir), "--output-name", "Voice Memo.mp3", str(clip_wav))
    assert r.returncode == 0, r.stderr
    assert (out_dir / "Voice Memo.mp3").is_file()
    assert re.search(r"Voice Memo\.mp3  \((gentle|balanced|strong|max) noise reduction picked for this file", r.stdout), r.stdout


@pytest.mark.native
def test_a_bad_file_fails_cleanly(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio" * 100)
    r = run("--cli", str(bad))
    assert r.returncode == 1 and "failed" in r.stderr and "Traceback" not in r.stderr


@pytest.mark.native
def test_a_name_the_file_cannot_have_is_reported_plainly(clip_wav, tmp_path):
    r = run("--cli", "-o", str(tmp_path), "--output-name", "x.exe", str(clip_wav))
    assert r.returncode == 1 and "can't be saved as .exe" in r.stderr and "Traceback" not in r.stderr
    assert list(tmp_path.iterdir()) == []


def test_the_parser_takes_the_options_and_keeps_the_automatic_defaults():
    from audio_cleaner_desktop import __main__ as entry

    args = entry.build_parser().parse_args(
        ["--cli", "--noise", "heavy", "--echo-track", "--rebuild-voice", "--parallel", "2", "a.wav"])
    assert (args.noise, args.echo_track, args.rebuild_voice, args.parallel) == ("heavy", True, True, 2)
    assert entry.build_parser().parse_args(["--download-speech-model"]).download_models  # the old name still works
    parser = entry.build_parser()
    defaults = parser.parse_args(["--cli", "a.wav"])
    assert defaults.strength == "auto" and defaults.no_auto_rebuild is False
    assert parser.parse_args(["--cli", "--no-auto-rebuild", "a.wav"]).no_auto_rebuild is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--cli", "--strength", "loud", "a.wav"])


def test_automatic_rebuild_is_on_unless_turned_off(monkeypatch):
    from audio_cleaner_desktop import __main__ as entry
    from audio_cleaner_desktop import cli

    seen = []
    monkeypatch.setattr(cli, "run", lambda inputs, out_dir, options, *a, **k: seen.append(options) or 0)
    assert entry.main(["--cli", "a.wav"]) == 0 and entry.main(["--cli", "--no-auto-rebuild", "a.wav"]) == 0
    assert [(o.auto_rebuild, o.rebuild_voice, o.strength) for o in seen] == [(True, False, "auto"), (False, False, "auto")]


def test_restore_highs_is_gone_from_the_command_line_and_the_help(capsys):
    """Restoring highs is not a step any more (a recording that is missing them is a reason to rebuild the voice)."""
    from audio_cleaner_desktop import __main__ as entry

    text = " ".join(run("--help").stdout.split())
    assert "highs" not in text.lower() and "restore" not in text.lower()
    with pytest.raises(SystemExit):
        entry.build_parser().parse_intermixed_args(["--cli", "--no-restore-highs", "a.wav"])
    assert "--no-restore-highs" in capsys.readouterr().err  # an unknown option, like any other


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
        engine.JobResult(tmp_path / "b-cleaned.wav", 1.0, False, strength="strong", notice="the voice was rebuilt because it sounded echoey"),
        engine.JobResult(tmp_path / "c-cleaned.wav", 1.0, False),
    ])
    monkeypatch.setattr(engine, "process", lambda *a, **k: next(results))
    assert cli.run([str(tmp_path / n) for n in ("a.wav", "b.wav", "c.wav")], None, engine.JobOptions(strength="auto"), parallel=1) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if "->" in line]
    assert lines[0].endswith("a-cleaned.wav  (balanced noise reduction picked for this file)")
    assert lines[1].endswith("b-cleaned.wav  (strong noise reduction picked for this file; the voice was rebuilt because it sounded echoey)")
    assert lines[2].endswith("c-cleaned.wav")


def test_rebuilding_the_voice_without_its_model_says_how_to_get_it(tmp_path, monkeypatch, capsys):
    from audio_cleaner_desktop import cli, engine, resources

    monkeypatch.setattr(resources, "check", lambda: [])
    monkeypatch.setattr(resources, "voice_model_dir", lambda: None)
    assert cli.run([str(tmp_path / "a.wav")], None, engine.JobOptions(rebuild_voice=True)) == 1
    assert "--download-models" in capsys.readouterr().err


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
    assert parser.parse_args(["--cli", "--format", "webm", "a.wav"]).output_format == "webm"


def test_one_name_cannot_be_given_to_several_files(tmp_path, capsys):
    from audio_cleaner_desktop import __main__ as entry
    from audio_cleaner_desktop import cli, engine

    with pytest.raises(SystemExit) as stop:
        entry.main(["--cli", "--output-name", "x.mp3", "a.wav", "b.wav"])
    assert stop.value.code == 2
    assert "one file at a time" in capsys.readouterr().err
    assert cli.run([str(tmp_path / "a.wav"), str(tmp_path / "b.wav")], None, engine.JobOptions(), output_name="x.mp3") == 1
    assert "one file at a time" in capsys.readouterr().err
