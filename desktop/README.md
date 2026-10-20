# Audio Cleaner for desktop

A free, offline app that removes background noise from the voice in video and audio files, and can write
subtitles. It runs the shared cleaning pipeline (`../common/acw_common`) in a PySide6 window.
Nothing is uploaded: FFmpeg and the noise library (libDF) ship inside the app, and the installer downloads every model
once during setup, each from Hugging Face: the noise removal models (DeepFilterNet3 and DPDFNet-8, 23 MB, required) and
the two larger ones (subtitles, rebuilding voices: Whisper 1.6 GB and Sidon 0.5 GB;
2.1 GB in all). If a download fails, the app offers it again (the noise removal models as soon as it starts). The installed app also looks for a newer version when it starts and
installs it (a signed update; Settings > "Check for updates when Audio Cleaner starts", on by default; it sends
nothing but a normal web request with the app version).

- `audio_cleaner_desktop/engine.py` cleans one file (no Qt; used by the window and the command line)
- `audio_cleaner_desktop/copies.py`, `blend.py`, `redo.py` the saved copies of a job, the strength blend, change-and-save-again (no Qt)
- `audio_cleaner_desktop/editor_job.py`, `editor_session.py`, `editor_model.py`, `live_mix.py` the per-part editor ("Fix a moment" and "Choose parts"): a job's block
  cache, quotes, saving a timeline, suggestions, the live mix per part (no Qt)
- `audio_cleaner_desktop/gui/` the window: queue, settings, preview, the Jobs tab (list, job view, the editor, live sound), About
- `audio_cleaner_desktop/cli.py` the `--cli` and `--selftest` modes
- `vendor/` native pieces for running from source (git-ignored, filled by the packaging scripts):
  `native/{libdf, ffmpeg, ffprobe}`, `licenses/`, and for tests and runs from source (never bundled) the noise models in
  `models/{DeepFilterNet3_onnx.tar.gz, dpdfnet8_48khz_hr_batch.onnx}` (`prepare_vendor.py --download-noise-models`) and a
  small speech model in `test-models/whisper-small/`.
  On Linux, missing pieces fall back to `~/audio-cleaner/{bin,models}`.

Building the installers: each script in [`packaging/`](packaging/) starts with a comment that says how to run it
(`build_ffmpeg.sh`, `prepare_vendor.py`, `audio_cleaner.spec`, `windows/AudioCleaner.iss`, `macos/make_pkg.sh`).

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
python -m audio_cleaner_desktop --cli talk.mp4 --timeline parts.json  # only the parts a timeline file says (below)
python -m audio_cleaner_desktop --selftest [--gui]    # checks the bundled native pieces (--gui: and the window)
python -m audio_cleaner_desktop --download-models     # all the models (noise removal first), from a terminal
```

A timeline file says what to do to which part of one recording (the format of `common/acw_common/timeline.py`, also the website's API):

```json
{"version": 1, "layers": {
  "noise":    [{"start": 10, "end": 40, "strength_db": 24, "remover": "auto"}],
  "rebuild":  [{"start": 41, "end": 44}],
  "leveling": [{"start": 0, "end": 120, "loudness_lufs": -16}]}}
```

Times are seconds. Layers: `echo` (needs the file's second audio track), `noise` (`strength_db` 12 Gentle, 24 Balanced, 40 Strong, 100 Maximum, or the
names; `remover` `auto`, `light`, `heavy`), `rebuild` (where it is on, it wins over the noise removal), `leveling` (`loudness_lufs` -70..0 or `null`).
What a layer does not cover is left as it is. With `--timeline`, `--strength`, `--noise`, `--echo-track`, `--rebuild-voice`, `--no-leveling` and
`--loudness` are not used. In the window the same editor is "Fix a moment" in a job and "Choose parts" under a waiting file.

## Tests

```sh
QT_QPA_PLATFORM=offscreen PYTHONPATH=desktop:common python -m pytest -q desktop/tests common/tests
```

Tests that need the native pieces are skipped when `resources.check()` reports something missing.
