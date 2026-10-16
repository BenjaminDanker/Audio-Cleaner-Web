# Audio Cleaner for desktop

A free, offline app that removes background noise from the voice in video and audio files, and can write
subtitles. It runs the shared cleaning pipeline (`../common/acw_common`) in a PySide6 window.
Nothing is uploaded: the noise models and FFmpeg ship inside the app, and the installer downloads the three larger
models (subtitles, restoring highs, rebuilding voices: 2.5 GB in all: Whisper 1.6 GB plus one 0.9 GB archive that holds
the other two) once during setup. The installed app also looks for a newer version when it starts and
installs it (a signed update; Settings > "Check for updates when Audio Cleaner starts", on by default; it sends
nothing but a normal web request with the app version).

- `audio_cleaner_desktop/engine.py` cleans one file (no Qt; used by the window and the command line)
- `audio_cleaner_desktop/gui/` the window: queue, settings, preview, history, About
- `audio_cleaner_desktop/cli.py` the `--cli` and `--selftest` modes
- `vendor/` native pieces for running from source (git-ignored, filled by the packaging scripts):
  `native/{libdf, ffmpeg, ffprobe}`, `models/{DeepFilterNet3_onnx.tar.gz, dpdfnet8_48khz_hr.onnx}`, `licenses/`, and a
  small speech model for the tests in `test-models/whisper-small/`.
  On Linux, missing pieces fall back to `~/audio-cleaner/{bin,models}`.

Building and shipping installers is described in [`../docs/desktop.md`](../docs/desktop.md).

## Development setup

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install --require-hashes -r requirements.lock -r requirements-build.lock   # every package pinned by hash (packaging/hash_lock.py)
pip install ../common          # or: export PYTHONPATH=desktop:common from the repository root
```

## Run

From the repository root, with `PYTHONPATH=desktop:common`:

```sh
python -m audio_cleaner_desktop                       # the window; FILE... are queued
python -m audio_cleaner_desktop --cli in.mp4 -o out/  # no window (see --help for the options)
python -m audio_cleaner_desktop --cli talk.mp4 --output-name talk.mp3   # just the sound (also --format mp3, --video-size 720)
python -m audio_cleaner_desktop --selftest [--gui]    # checks the bundled native pieces (--gui: and the window)
python -m audio_cleaner_desktop --download-models     # the three models, from a terminal
```

## Tests

```sh
QT_QPA_PLATFORM=offscreen PYTHONPATH=desktop:common python -m pytest -q desktop/tests common/tests
```

Tests that need the native pieces are skipped when `resources.check()` reports something missing.
