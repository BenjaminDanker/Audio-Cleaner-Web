# Audio Cleaner

Audio Cleaner removes background noise from the voice in your videos and recordings, on your own computer. Drop in a
file, and you get it back with the voice clear and the hiss, hum, wind, keyboard and room noise gone. Nothing is
uploaded: after setup it works offline.

- **Noise removal** that keeps the voice natural, with four strengths from Gentle to Maximum and a stronger remover
  for recordings where the noise is nearly as loud as the voice
- **Automatic best settings**: each file is listened to first and gets the cleanup that fits it
- **Leveling**: evens out the voice and sets the loudness for podcasts, YouTube or broadcast
- **Rebuilds the voice** of recordings made in echoey rooms, and of phone calls and other dull, low-quality recordings
- **Removes speaker echo** when your recording has the desktop sound on its own audio track (OBS can record that)
- **Subtitles and transcripts**: SRT, VTT or plain text, or put right into the video, in 28 languages
- Video stays as it is: only the sound is replaced, and you can save just the sound, another file type or a smaller
  picture size
- A queue for many files at once, a before/after player, and a history of what you cleaned

## Download

Get the installer from the [latest release](https://github.com/BenjaminDanker/Audio-Cleaner/releases/latest):

| Your computer | File |
|---|---|
| Windows 10 or 11 (64-bit) | `AudioCleaner-<version>-Windows.exe` |
| Mac with Apple silicon (M1 and newer) | `AudioCleaner-<version>-Mac-Apple-silicon.pkg` |
| Mac with an Intel processor | `AudioCleaner-<version>-Mac-Intel.pkg` |

Not sure which Mac you have? Apple menu > About This Mac: "Chip" means Apple silicon, "Processor" means Intel. macOS 14
or newer is needed.

Setup downloads the models once (about 2.1 GB), from Hugging Face. The app looks for a newer version when it starts and updates
itself; you can switch that off in Settings.

## Build it yourself

One script per system does everything, from a fresh clone to the installer: it builds the noise-removal library and
FFmpeg from source, sets up Python with every package pinned by hash, runs the tests, packages the app, checks the
packaged app and builds the installer. The first run takes a while (FFmpeg and the noise library compile from source);
later runs reuse what was built.

### Windows 10/11 (64-bit)

You need:

- Python 3.12: `winget install Python.Python.3.12`
- Git for Windows: `winget install Git.Git`
- Rust: https://rustup.rs, with Visual Studio Build Tools and its "Desktop development with C++" workload
- Inno Setup 6: `winget install JRSoftware.InnoSetup`
- MSYS2: `winget install MSYS2.MSYS2` (FFmpeg is compiled in it)

Then, in PowerShell:

```powershell
git clone https://github.com/BenjaminDanker/Audio-Cleaner.git
cd Audio-Cleaner
powershell -ExecutionPolicy Bypass -File desktop\packaging\build_windows.ps1
```

The installer is `build\windows\out\AudioCleaner-Setup-x64.exe`. Add `-SkipTests` to skip the tests.

### macOS 14 or newer

You need:

- Xcode command line tools: `xcode-select --install`
- Homebrew, then `brew install python@3.12 pkgconf` (on an Intel Mac also `brew install nasm`)
- Rust: https://rustup.rs

Then:

```sh
git clone https://github.com/BenjaminDanker/Audio-Cleaner.git
cd Audio-Cleaner
desktop/packaging/build_macos.sh
```

The installer is `build/macos/out/AudioCleaner-macOS-<arm64|x86_64>.pkg`, for the kind of Mac you built it on. Set
`SKIP_TESTS=1` to skip the tests. Without a Developer ID (see the comment at the top of the script) the app is not
signed, so macOS asks you to confirm the first time you open it.

### Running it from source

After the build script has run once, its Python environment and the folder with the native pieces (`vendor`) are in
`build/<system>/`, and the app runs straight from the source:

Windows (PowerShell):

```powershell
build\windows\venv\Scripts\Activate.ps1
$env:PYTHONPATH = "desktop;common"; $env:ACW_DESKTOP_VENDOR = "build\windows\vendor"
python -m audio_cleaner_desktop
```

macOS:

```sh
source build/macos/venv/bin/activate
export PYTHONPATH=desktop:common ACW_DESKTOP_VENDOR=build/macos/vendor
python -m audio_cleaner_desktop
```

The same environment runs the tests and the command line:

```sh
QT_QPA_PLATFORM=offscreen python -m pytest -q common/tests desktop/tests
python -m audio_cleaner_desktop --cli talk.mp4 -o cleaned/              # no window; --help lists every option
python -m audio_cleaner_desktop --cli talk.mp4 --output-name talk.mp3    # just the sound
python -m audio_cleaner_desktop --download-models                       # the noise, subtitle and restoration models
python -m audio_cleaner_desktop --selftest                              # checks the native pieces
```

### What the scripts do, step by step

Each of these starts with a comment that says how to run it on its own:

1. `native/deepfilter/build.sh native`: the DeepFilterNet library (libDF), from a pinned commit with one patch
2. `desktop/packaging/build_ffmpeg.sh`: FFmpeg with only LGPL parts, linked statically, from pinned source tarballs
3. `desktop/packaging/prepare_vendor.py`: puts the native pieces in one folder, checks every file against its pinned
   SHA-256 and writes the licence notices (the models are not in the app: setup and the app download them from
   Hugging Face, every file pinned and checked; `desktop/packaging/check_model_urls.py` checks they are all there)
4. `desktop/packaging/audio_cleaner.spec`: packages the app with PyInstaller (then `--selftest --gui` on the result)
5. `desktop/packaging/windows/AudioCleaner.iss` (Inno Setup) or `desktop/packaging/macos/make_pkg.sh`: the installer

## What is in here

- `desktop/`: the app. `audio_cleaner_desktop/engine.py` cleans one file, `gui/` is the window (PySide6), `cli.py` the
  command line; `packaging/` builds the installers; `tests/`
- `common/`: the cleaning itself (`acw_common`): noise removal, leveling, rebuilding the voice,
  echo removal, subtitles, FFmpeg handling
- `native/deepfilter/`: how the DeepFilterNet library is built

## Licences

Noise removal uses [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) (MIT / Apache-2.0) and
[DPDFNet](https://github.com/ceva-ip/DPDFNet) (Apache-2.0). The notices of every component in the app are in
`desktop/packaging/licenses` and in the app's About box.

Audio Cleaner is open source under the [MIT License](LICENSE).

## Questions and problems

Found a bug, or something doesn't build? [Open an issue](https://github.com/BenjaminDanker/Audio-Cleaner/issues).
For anything else, there is a contact form at https://audiocleaner.live/contact.
