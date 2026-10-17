# Audio Cleaner

Audio Cleaner removes background noise from the voice in video and audio files, entirely on your own computer. Drop
in a recording, pick how much cleaning you want, and get back the same file with the voice clear and the hiss, hum and
room noise gone. It can also tidy up phone-quality recordings, take echo out, even out the loudness and write subtitles.
Nothing you clean is uploaded.

Windows 10/11 and macOS 14 or newer (Apple silicon and Intel) are supported.

## Download

The installers are on the download page: **https://mc-multi-world.space/audio/**

Setup downloads the larger models once (about 2.5 GB in all); after that everything works offline. The app checks for a
newer version when it starts, and you can switch that off in Settings.

## Old versions

This repository used to hold a first version of the app, called VideoDenoiser. Its releases (v1.0.0 and v1.0.1) are
still here and its source is still in the history under those tags. The app in this repository today is a complete
rewrite of it, so the old settings and files do not carry over.

## What is in here

- `desktop/` the app: a Qt (PySide6) window around the cleaning code, plus the scripts that package it
  (`desktop/packaging/`) and its tests
- `common/` the cleaning code itself (`acw_common`): DeepFilterNet3 and DPDFNet noise removal, the polish chain,
  FFmpeg handling, echo removal, restoring highs, subtitles
- `native/deepfilter/` how the DeepFilterNet library (libDF) is built, with the one patch we apply to it

The noise removal is [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) (MIT / Apache-2.0) and
[DPDFNet](https://github.com/ceva-ip/DPDFNet) (Apache-2.0). FFmpeg is built from source without any GPL parts and linked
statically. The licence notices of everything in the installers are in `desktop/packaging/licenses` and, once built, in
the app's About box.

## Running it from source

You need Python 3.12 and, for the native pieces, a built libDF and FFmpeg (see below).

```sh
python3.12 -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --require-hashes -r desktop/requirements.lock -r desktop/requirements-build.lock
export PYTHONPATH=desktop:common                           # Windows (cmd): set PYTHONPATH=desktop;common
python -m audio_cleaner_desktop                            # the window
python -m audio_cleaner_desktop --cli talk.mp4 -o out/     # no window; --help lists the options
QT_QPA_PLATFORM=offscreen python -m pytest -q common/tests desktop/tests
```

Every Python package is pinned with the hashes of all its files (`desktop/packaging/hash_lock.py` regenerates the
locks). From source the app reads the native pieces from `desktop/vendor/` (see `desktop/README.md`); tests that need
them skip themselves when they are missing.

## Building the installers

The easy way is the workflow in `.github/workflows/desktop.yml`: it builds FFmpeg and libDF, freezes the app with
PyInstaller, checks the frozen program, builds the Windows installer (and installs and uninstalls it once) and the two
Mac packages, and runs the tests. It runs on every update of this repository (or by hand from the Actions tab) and the
installers come out as workflow artifacts. They are not signed, and they are a build for testing, not an official
release: their version is always 2.0.0, so an official installer replaces them.

To build by hand, the steps in order (each script starts with a comment that says how to run it):

1. libDF: `native/deepfilter/build.sh native` (needs Rust; Git Bash on Windows)
2. FFmpeg: `desktop/packaging/build_ffmpeg.sh <windows-x86_64|macos-arm64|macos-x86_64> <out_dir>`
3. The heavy-noise model: `python desktop/packaging/fetch_dpdfnet.py build/dpdfnet`
4. Everything the app bundles in one folder: `desktop/packaging/prepare_vendor.py` (it checks every file against its
   pinned SHA-256 and writes the licence notices)
5. `pyinstaller desktop/packaging/audio_cleaner.spec`, then `AudioCleaner --selftest --gui` on the result
6. The installer: `desktop/packaging/windows/AudioCleaner.iss` (Inno Setup) or `desktop/packaging/macos/make_pkg.sh`

`desktop/packaging/build_windows.ps1` and `desktop/packaging/build_macos.sh` chain these steps on a Windows PC or a Mac.

## Questions and problems

The download page has a contact form. Issues here are welcome too, for problems in the code or the build.

This repository is updated in whole snapshots, so a pull request cannot be merged here as it is; if you have a fix,
open an issue that describes it and it will be applied.
