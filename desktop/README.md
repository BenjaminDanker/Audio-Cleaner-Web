# Audio Cleaner for desktop

A free, offline app that removes background noise from the voice in video and audio files, and can write
subtitles. It runs the shared cleaning pipeline (`../common/acw_common`) in a PySide6 window.
Nothing is uploaded: the noise model, the speech model and FFmpeg all ship inside the app.

- `audio_cleaner_desktop/engine.py` cleans one file (no Qt; used by the window and the command line)
- `audio_cleaner_desktop/gui/` the window: queue, settings, preview, history, About
- `audio_cleaner_desktop/cli.py` the `--cli` and `--selftest` modes
- `vendor/` native pieces for running from source (git-ignored, filled by the packaging scripts):
  `native/{libdf, ffmpeg, ffprobe}`, `models/DeepFilterNet3_onnx.tar.gz`, `models/whisper-small/`, `licenses/`.
  On Linux, missing pieces fall back to `~/audio-cleaner/{bin,models}`.

Building and shipping installers is described in [`../docs/desktop.md`](../docs/desktop.md).

## Development setup

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.lock pytest
pip install ../common          # or: export PYTHONPATH=desktop:common from the repository root
```

## Run

From the repository root, with `PYTHONPATH=desktop:common`:

```sh
python -m audio_cleaner_desktop                       # the window; FILE... are queued
python -m audio_cleaner_desktop --cli in.mp4 -o out/  # no window (see --help for the options)
python -m audio_cleaner_desktop --selftest            # checks the bundled native pieces
```

## Tests

```sh
QT_QPA_PLATFORM=offscreen PYTHONPATH=desktop:common python -m pytest -q desktop/tests common/tests
```

Tests that need the native pieces are skipped when `resources.check()` reports something missing.
