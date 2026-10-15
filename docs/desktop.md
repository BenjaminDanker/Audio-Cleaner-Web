# Audio Cleaner for desktop

The desktop app is a PySide6 (Qt) window around the shared cleaning pipeline in `common/` (the server uses the same code).
The installer carries the app, FFmpeg and the two noise models (DeepFilterNet3, and DPDFNet-8 for heavy noise), so
noise removal works offline straight after installing. Three larger models (2.5 GB in all) are downloaded by setup
itself, and only by setup: the app never downloads anything on its own at start-up. The installer is small; the
Windows installer's download page and the Mac .pkg's postinstall script check every file's SHA-256, keep files a
previous install left in place, and are generated from the app's catalog by `packaging/installer_models.py`:

- the speech model for subtitles, Whisper large-v3-turbo (1.6 GB, from Hugging Face, `resources.SPEECH_MODEL`);
- the model that restores the highs of phone-quality recordings, AP-BWE (4 x 119 MB ONNX, `resources.HIGHS_MODEL`);
- the model that rebuilds the voice (optional "Rebuild the voice" setting: removes room echo too), Sidon
  (503 MB ONNX, `resources.VOICE_MODEL`).

The last two come in ONE zip archive, `AudioCleaner-models-v1.zip` (0.9 GB, `resources.MODELS_ARCHIVE`) on this
repository's `models-v1` release; it also holds a `README.txt` and the `LICENSES/` folder (AP-BWE, Sidon, w2v-BERT 2.0
and training-data notices). Each of the two stays a model of its own in the catalog (own folder, own panel and checks,
every file pinned by size and SHA-256), but downloading either installs both. Setup downloads the archive once (the
download page checks its SHA-256), unpacks it, checks the size of every file, moves the files to their model folders,
puts `LICENSES/` next to them (`{app}\models\LICENSES`, which About points to) and deletes the archive.
On Windows the unpacking is `%SystemRoot%\System32\tar.exe -xf <zip> -C <folder>` (Windows 10 1803 and later, so on
every supported Windows; if it were missing, setup skips the archive and shows the "open Audio Cleaner > Settings to
finish the download" message). On a Mac the postinstall script uses `curl`, `shasum -a 256` and `ditto -x -k`. The
app itself (`models.py`) streams the archive into `<models>/AudioCleaner-models-v1.partial/` (resumable, SHA-256
checked), unpacks the members of both models member by member (never a whole file in memory, size and SHA-256 checked
again), and deletes the archive; it needs about 1.9 GB free at the peak (archive plus unpacked), cancelling works
during the download and the unpacking, and a copy that setup installed is reused instead (D-33).

They go to `{app}\models\<id>` on Windows and `/Library/Application Support/Audio Cleaner/models/<id>` on a Mac
(`models.installer_dir`); a model whose files are all there also gets a `.complete` marker (the hashes of its files),
which the app compares with its catalog. The speech model's files are tried on their own, so one failed download does
not stop the next. If a download fails or is cancelled during setup (offline, no space on the temporary drive), setup still
finishes; Settings then shows a Download button for what is missing (and the banner at the top offers all of it;
`models.py`: pinned, resumable, verified, into the app's data folder; files setup did install are reused),
`AudioCleaner --download-models` does all three from a terminal, and starting a job that needs a missing model asks
first. Nothing is ever uploaded. The Mac download is a `.pkg` (not a disk image) because a disk image has no setup
step to download in. The .pkg is wrapped with `productbuild --distribution` (`packaging/macos/distribution.py`): it
refuses macOS older than 14, shows the EULA, and its welcome text says setup downloads about 2.5 GB and takes minutes.

**The `models-v1` release** must hold the archive before the installers can fetch it: upload the one file
`AudioCleaner-models-v1.zip` (908,538,132 bytes, sha256 `5592be63...81ce`; `README.txt`, `LICENSES/*`,
`ap-bwe/{8k,12k,16k,24k}to48k.onnx`, `sidon/{sidon_fe,sidon_dec}.onnx`, deflate) to a release tagged `models-v1` with
`--latest=false` (otherwise it takes over `releases/latest` from the app). Never replace the file under the same name
with different content: the catalog pins its size and SHA-256 (change both in `resources.MODELS_ARCHIVE`, and the
member sizes and hashes, when a new archive is published, preferably under a new tag such as `models-v2`). The
"Model downloads exist" job in the workflow compares the archive's size (and the speech model files') with the catalog
and goes red when it is missing; it does not block the builds.

Runs on Windows 10/11 (x64) and macOS 14 Sonoma or newer, on Apple silicon or Intel (PySide6 6.12 and the
onnxruntime wheels set that floor; Intel Macs use onnxruntime 1.23.2, the last release with Intel Mac wheels).

```
common/acw_common/        cleaning pipeline shared with the server (DeepFilterNet via libDF, FFmpeg wrapper, loudness,
                          captions, transcription)
desktop/audio_cleaner_desktop/
    engine.py             cleans one file (no Qt)         gui/    the window      cli.py  --cli and --selftest
    resources.py          finds libDF, models, FFmpeg: sys._MEIPASS when frozen, desktop/vendor from source
desktop/packaging/        everything that turns that into installers (below)
native/deepfilter/        reproducible libDF build (shared infrastructure)
.github/workflows/desktop.yml
```

Frozen layout (under `sys._MEIPASS`, i.e. `_internal/` next to `AudioCleaner.exe`, or inside the `.app`):
`native/{df.dll|libdf.dylib, ffmpeg[.exe], ffprobe[.exe]}`, `models/{DeepFilterNet3_onnx.tar.gz, dpdfnet8_48khz_hr.onnx}`,
`licenses/` (shown in About, together with the licence line of each downloaded model). Qt, FFmpeg and libDF stay separate files on purpose (LGPL, see
the checklist at the end).

## Release process (CI)

1. Releases are automatic: every push to `main` / `on-prem` that changes `desktop/`, `common/`,
   `native/deepfilter/` or the workflow builds and publishes a GitHub Release tagged
   `desktop-v<__version__>-build.<run number>` (only the newest 5 such automatic releases are kept). Bump
   `__version__` in `desktop/audio_cleaner_desktop/__init__.py` (and `version` in `desktop/pyproject.toml`) when the
   app changes in a way friends should notice; the build number tells builds of one version apart.
2. Optional, for a release that stays: tag the commit `git tag desktop-v2.0.0 && git push origin desktop-v2.0.0`.
   The tag must be `desktop-v` plus the exact `__version__`; the workflow refuses a mismatch. Tagged releases are
   never deleted by the cleanup.
3. `.github/workflows/desktop.yml` builds, per platform: FFmpeg (LGPL, from pinned source tarballs; Windows is
   cross-compiled with mingw-w64 on Ubuntu), libDF, then the app: tests, PyInstaller, the frozen `--selftest --gui`
   (also builds the window and runs the noise models, echo canceller, AP-BWE and Sidon when their files are there),
   and the installer (Inno Setup 6.7.1 on Windows, a `productbuild` .pkg on macOS; both download the models during
   setup; the ONNX models arrive as one archive). Windows also runs the frozen app under a non-ASCII profile folder and installs, starts and uninstalls the
   installer silently. A separate "Model downloads exist" job checks the catalog URLs. Website/server-only pushes skip it; "Run workflow"
   builds by hand (artifacts only, no release; a push that only changes `deploy/fetch-dpdfnet.sh` also builds). About 10 minutes with warm FFmpeg/libDF caches, 20-30 cold.
4. The `release` job attaches `AudioCleaner-<version>-Windows.exe`, `AudioCleaner-<version>-Mac-Apple-silicon.pkg`
   and `AudioCleaner-<version>-Mac-Intel.pkg` (GitHub shows each file's SHA-256 itself, so there is no separate
   checksum file) and marks the release as latest, so the link to share with
   friends is always `https://github.com/BenjaminDanker/Audio-Cleaner-Web/releases/latest`. The app is not on the
   website and never contacts it.
5. Smoke-test the downloads on a clean Windows and a clean Mac (arm64 and Intel if you can) before announcing: open a
   noisy video, clean it, turn on subtitles, open About.

Secrets are only read by the signing steps; without them every step is skipped and unsigned installers are produced.

## Signing (optional, recommended)

| Platform | What | Cost | Repository secrets |
| --- | --- | --- | --- |
| macOS | Apple Developer Program, "Developer ID Application" certificate, notarization | USD 99 / year | `MACOS_CERT_P12_BASE64` (the .p12 exported from Keychain, base64), `MACOS_CERT_PASSWORD`; to notarize also `APPLE_ID`, `APPLE_TEAM_ID`, `APPLE_APP_PASSWORD` (app-specific password from appleid.apple.com) |
| Windows | Azure Artifact Signing (formerly Trusted Signing), Public Trust profile | about USD 9.99 / month (Basic, 5,000 signatures); individual developers must be in the US or Canada, otherwise use an organisation or an OV certificate from a cloud HSM (roughly USD 180-400 / year, not wired into the workflow) | `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET` (a service principal with the "Artifact Signing Certificate Profile Signer" role), `AZURE_SIGNING_ENDPOINT` (e.g. `https://eus.codesigning.azure.net`), `AZURE_SIGNING_ACCOUNT`, `AZURE_SIGNING_PROFILE` |

Prices and eligibility from `docs/research/desktop-packaging.md` (researched Oct 2026): re-check them before buying.
Signing never runs for pull requests. With the secrets present the workflow signs the Windows program files and the
installer, and on macOS signs every Mach-O file inside out with the hardened runtime
(`desktop/packaging/macos/sign_app.sh`, entitlements in `macos/entitlements.plist`), notarizes and staples the app
and the app inside the .pkg (`macos/notarize.sh`; signing the .pkg itself needs a "Developer ID Installer" certificate, not wired in yet). Unverified until the first signed run: the Azure inputs follow the action's
`action.yml` (v2.0.0); the macOS flow is the standard `codesign` / `notarytool` / `stapler` sequence.

### What users see without signing

* **Windows**: SmartScreen "Windows protected your PC". Click "More info", then "Run anyway". Signing alone does not
  remove the warning at once: reputation builds with downloads (EV certificates no longer skip it).
* **macOS**: Gatekeeper refuses to open it ("cannot be opened because the developer cannot be verified"). Right-click
  the app, "Open", then "Open" again; or System Settings, Privacy & Security, "Open Anyway". On Apple silicon the
  bundle carries PyInstaller's ad-hoc signature, so it starts after that one confirmation.

## Local development

```sh
python3.12 -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --require-hashes -r desktop/requirements.lock -r desktop/requirements-build.lock
export PYTHONPATH=desktop:common                         # Windows (cmd): set PYTHONPATH=desktop;common
python -m audio_cleaner_desktop                          # the window; --cli FILE... without a window; --selftest
QT_QPA_PLATFORM=offscreen python -m pytest -q common/tests desktop/tests
```

From source the app reads `desktop/vendor/` (git-ignored; same layout as the frozen app; override with
`ACW_DESKTOP_VENDOR`); on Linux it falls back to `~/audio-cleaner/{bin,models}`. Fill `desktop/vendor` with
`prepare_vendor.py` (below).

## Building by hand

The steps CI runs, in order. `desktop/packaging/build_macos.sh` and `build_windows.ps1` chain them on a Mac or a
Windows PC (written as thin wrappers; they have not been run on real machines yet, so expect to fix a path or two).

1. **libDF**: `native/deepfilter/build.sh native` (needs Rust; Git Bash on Windows). Output `libdf.so` / `libdf.dylib` /
   `df.dll` in `$OUT` (default `~/audio-cleaner/src/deepfilter-build/out`); the checkout also holds
   `models/DeepFilterNet3_onnx.tar.gz`.
2. **FFmpeg**: `desktop/packaging/build_ffmpeg.sh <linux-x86_64|macos-arm64|macos-x86_64|windows-x86_64> <out_dir>`.
   LGPL only (no GPL, version3, nonfree): FFmpeg 8.0.3 + libopus + LAME from sha256-pinned tarballs, linked statically
   into `ffmpeg` and `ffprobe`. `windows-x86_64` cross-compiles on Linux (`apt install mingw-w64 nasm`) or builds
   natively in an MSYS2 MINGW64 shell. `REQUIRE_ASM=1` fails when nasm is missing instead of building slower code.
   Writes `ffmpeg-buildinfo.txt` and the licence texts next to the programs.
3. **Vendor directory**:
   ```sh
   python desktop/packaging/prepare_vendor.py --libdf PATH/libdf.so --ffmpeg-dir FFMPEG_OUT --model PATH/DeepFilterNet3_onnx.tar.gz \
       [--whisper-model DIR | --download-whisper] [--deepfilter-src DIR] --out desktop/vendor
   ```
   Copies the native files, checks the DFN3 and DPDFNet models (sha256 pinned), puts a small test-only speech model in
   `vendor/test-models/whisper-small/` (`Systran/faster-whisper-small`, revision and file hashes pinned in the script;
   `--download-whisper` fetches it; it is not bundled) and
   writes `licenses/`: FFmpeg/LAME/opus, DeepFilterNet, DPDFNet, the Rust crates of libDF (`cargo tree` of the capi
   build), all Python packages of the interpreter running the script, Qt/PySide6 (LGPL + GPL text from
   `desktop/packaging/licenses/`), Whisper, Silero VAD, the downloaded models (AP-BWE, Sidon, w2v-BERT 2.0 and the
   CC BY attribution of their training data), WebRTC (echo canceller) and LiveKit, Python, PyInstaller and an index
   `README.txt`. It fails on anything missing.
4. **Icons**: `python desktop/packaging/make_icons.py OUT_DIR` (icon.ico, icon.icns from `assets/icon.png`).
5. **Bundle**:
   ```sh
   pip install --require-hashes -r desktop/requirements.lock -r desktop/requirements-build.lock
   ACW_DESKTOP_VENDOR=desktop/vendor ACW_ICON_DIR=OUT_DIR \
     PYTHONPATH=desktop:common pyinstaller desktop/packaging/audio_cleaner.spec --distpath dist --workpath build --noconfirm
   dist/AudioCleaner/AudioCleaner --selftest            # must print "selftest ok" (macOS: dist/Audio Cleaner.app/Contents/MacOS/AudioCleaner)
   ```
   Onedir, windowed, no UPX. The selftest denoises a tone through libDF, round-trips it through FFmpeg, loads the
   subtitle engine (CTranslate2, Silero VAD, the speech model if it is installed), runs a second through DPDFNet and
   the echo canceller (and through AP-BWE and Sidon when their models are installed), and with `--gui` builds the
   window, the preview and the About box, so a missing shared library, plugin or data file fails here.
6. **Installer**: Windows `iscc /DAppVersion=2.0.0 /DSourceDir=dist\AudioCleaner desktop\packaging\windows\AudioCleaner.iss`
   (per-user install by default, asks for elevation only if the user picks all users; Start menu, optional desktop
   icon, uninstaller, `EULA.txt`; for the model download build the include first with
   `python desktop/packaging/installer_models.py build/models.iss` and pass `/DModelsInclude=build/models.iss`, which
   needs no numpy). macOS `desktop/packaging/macos/make_pkg.sh "dist/Audio Cleaner.app" out.pkg VERSION [SCRIPTS_DIR]`
   (add `PKG_ARCH=x86_64` for the Intel build).

### Packaging notes worth knowing

* **PyAV is not shipped.** faster-whisper imports `av` but the app never uses it (FFmpeg decodes). PyAV's wheels
  bundle GPL builds of x264/x265, which would not be allowed in this LGPL-only bundle, so the spec excludes `av` and
  `rthook_av_stub.py` installs an empty stand-in module. Do not use faster-whisper's `decode_audio` anywhere.
* The spec drops Qt Virtual Keyboard (GPL-only) and Quick/QML/PDF/WebEngine/3D files and aborts if the keyboard
  plugin is still there. If PySide6_Addons is installed in the build environment its files are filtered, not shipped.
* Linux is only a development target; no Linux installer is produced.

## Size

Measured on Linux (the Windows/macOS sizes will differ somewhat): the `AudioCleaner` folder is about 0.6 GB
uncompressed. The two noise models are 23 MB (DeepFilterNet 8 MB, DPDFNet-8 15 MB); the rest is PySide6/Qt 140 MB,
CTranslate2 137 MB (library and extension), SciPy and its OpenBLAS 71 MB, ONNX Runtime 30 MB, NumPy and its OpenBLAS
41 MB, libDF/FFmpeg 38 MB. Setup then downloads 2.5 GB of models: Whisper large-v3-turbo 1.6 GB, AP-BWE 4 x 119 MB,
Sidon 503 MB (the last two come in one 0.9 GB archive, so the download is 2.5 GB in all and 2.6 GB on disk). Windows
setup needs the 2.5 GB of free space on the temporary drive until the files are installed (it checks first), and the
unpacked models on the install drive (the archive is unpacked there, then the files are moved: about 1 GB extra for a
moment). The app's own download of the two ONNX models needs about 1.9 GB at the peak (archive plus unpacked). Memory
while cleaning, measured: 0.5 GB for a normal file, 1.7 GB with
subtitles, 2.6 GB with "Rebuild the voice" (`sysinfo.py` plans with these numbers). GitHub Release assets may be up
to 2 GiB each.

## LGPL / licence compliance checklist

* Qt and PySide6 are used under the LGPL v3, dynamically linked: they are separate files in the app folder, and the
  bundle is onedir (never onefile) so users can replace them. `licenses/README.txt` carries the Qt notice, the
  written offer for the sources, and links to https://download.qt.io/official_releases/QtForPython/ and
  https://code.qt.io. The About dialog shows the notice and every file in `licenses/`.
* FFmpeg (LGPL v2.1+) is built without GPL parts and runs as a separate program. `ffmpeg-buildinfo.txt` (in
  `licenses/`) names the source tarballs with sha256, the configure lines and the build script, and says the files can
  be replaced. Keep the tarball URLs alive or attach the three source archives to each GitHub Release. Do not add
  `--enable-gpl`, `--enable-nonfree` or libx264/libfdk to the build.
* Qt's multimedia plugin brings its own FFmpeg libraries (LGPL); they are separate files and covered by the Qt notice.
* `EULA.txt` must keep the carve-outs for the open source components and must not forbid reverse engineering for
  debugging replaced LGPL libraries. FFmpeg and the downloaded models (Whisper, AP-BWE, Sidon, w2v-BERT 2.0) are
  named in it.
* The downloaded models carry notices too: `AP-BWE-MIT.txt` (code and weights), `Sidon-MIT.txt`,
  `w2v-BERT-2.0-MIT.txt` (Meta) and `Training-data-attribution.txt` (CC BY 4.0: VCTK, FLEURS-R, LibriTTS-R) ship in
  `licenses/` and are shown in About, and the same files (plus `AP-BWE-weights-LICENSE.txt`) are in the `LICENSES/` folder of the models archive, which
  setup and the app unpack next to the models.
  The Dropbox CTranslate2 conversion of Whisper has no licence text of its own (its card says MIT); OpenAI's
  `Whisper-MIT.txt` is the text that applies.
* Re-run the licence collection (it runs in every build) after changing dependencies; it fails when a package has no
  licence text. New Python packages that end up in the bundle need an entry check in `prepare_vendor.py`.
* Not legal advice; confirm the Qt/LGPL points with counsel (see `docs/research/desktop-packaging.md`).

## Updating

To update, just run the new setup (the .exe or the .pkg) over the installed app. There is nothing to uninstall first;
it is the same install, done again, and the models are not downloaded again.

* **Windows.** Setup closes the running app, reuses the folder and the mode (per user or for all users) of the previous
  install (`UsePreviousAppDir`/`UsePreviousPrivileges`; the `AppId` never changes), deletes the old program files
  (`[InstallDelete]` removes `{app}\_internal`, so nothing of an older version lingers) and copies the new ones.
  `{app}\models` is never deleted by an update. The model step looks at what is installed: a model whose files are all
  there with the right sizes (and a `.complete` marker that matches the catalog) is not downloaded again; with every
  model complete setup does not even show the download page. A model from an earlier, partial install is finished:
  single files one by one, the archive again if any of its files is missing or has the wrong size; a model whose marker
  names other hashes than this catalog is removed and fetched again. A silent setup (`/VERYSILENT`) never downloads
  models, as before.
* **macOS.** The .pkg replaces `/Applications/Audio Cleaner.app` completely: its preinstall script removes the old bundle
  (that one folder and nothing else) before the new one is laid down, so no file of an older version can stay inside it,
  and the component is not skipped for being "the same or an older version". Its postinstall script skips the archive
  and the Whisper files when they are complete, as on Windows. Quit the app before installing.
* **What an update keeps.** The settings (every option in the settings panel, the window size and position, the tab
  last used), the history and everything the app downloaded itself are outside the program folder: the settings are in
  the registry (`HKEY_CURRENT_USER\Software\Audio Cleaner`) or the macOS preferences, the history and the app's own
  model downloads in `%LOCALAPPDATA%\Audio Cleaner\Audio Cleaner` or `~/Library/Application Support/Audio Cleaner`.
  The models setup installed stay in `{app}\models` or `/Library/Application Support/Audio Cleaner/models`.
* **In CI.** The Windows job installs, installs again over the first install (checking that a stale file in
  `_internal` is gone and a file in `models` is still there), runs the frozen selftest and uninstalls.

## Things worth knowing

* **No console window flashes on Windows.** The app has no console, so every console program started by a library would
  get a window of its own for a moment (Python's `platform` module runs `cmd /c ver` when its WMI query fails, and
  onnxruntime asks `platform` at import time). `audio_cleaner_desktop/quiet_popen.py` makes `CREATE_NO_WINDOW` the
  default of every `subprocess.Popen` (and so `run`, `check_output`, `os.popen`) on Windows; `packaging/launcher.py`
  installs it before anything else is imported. Callers that pass their own console flag, and Explorer, are left alone.
  `_wmi` is a hidden import of the spec, so `platform` finds it, and `--selftest` checks both on Windows.

* **Uninstalling.** Windows: the uninstaller removes the program and the models setup installed, and asks whether to
  remove the models the app downloaded itself, the history and the settings (`%LOCALAPPDATA%\Audio Cleaner`); a
  silent uninstall keeps them. macOS: dragging the app to the Trash leaves the models in
  `/Library/Application Support/Audio Cleaner/models` (2.6 GB, installed by the .pkg), the app's own data in
  `~/Library/Application Support/Audio Cleaner` and its settings; delete those folders by hand to remove everything.
* **Per-user and all-users installs on Windows are separate installs** (two uninstall entries, two copies of the
  models); installing one over the other does not replace it. Uninstall the old one first.
* Only one Audio Cleaner window runs at a time: starting it again, or "Open with" on several files, hands the files to
  the open window. Temporary work files go to the app's data folder (`scratch`) on Windows and macOS, and leftovers of
  a crashed run are swept at the next start.
* Not done yet (needs the FFmpeg wrapper or more): extra audio tracks and existing subtitles of a video are not kept
  in the result (the row says so); paths longer than 259 characters on Windows are refused with a message; `unins000.exe`
  is not signed.

## Pinned Python packages

`desktop/requirements.lock` (runtime) and `desktop/requirements-build.lock` (PyInstaller, Pillow, pytest and what they
need) list every package as `name==version` followed by `--hash=sha256:...` lines for EVERY file PyPI has for that
exact version (all wheels and the sdist), so one lock serves Windows, macOS arm64 and x86_64 and Linux; environment
markers stay (e.g. onnxruntime 1.23.2 and its dependencies on Intel Macs, colorama on Windows). CI and the build scripts
install with `pip install --require-hashes -r desktop/requirements.lock -r desktop/requirements-build.lock`: pip then
refuses a file with another hash, a package that is not listed, and an unpinned one. To change a version, edit it in the
lock and run `python desktop/packaging/hash_lock.py` (rewrites the hashes from `pypi.org/pypi/<name>/<version>/json`;
`--check` only compares and exits 1 on a difference). A package that is needed on only some platforms or Python
versions must be listed with its marker; `pip install --require-hashes` on a platform that needs a dependency the lock
does not list fails with "Hashes are required in --require-hashes mode" and names it. Nothing in the locks is outside
PyPI; a package that was would need its own hashed line (see the docstring of `hash_lock.py`). pip itself is
upgraded unpinned before the install (`python -m pip install --upgrade pip`).

## Owner to-do before the first release

* Upload the `models-v1` release (the single `AudioCleaner-models-v1.zip`, see above) and check that the "Model
  downloads exist" job is green.

* Set the Apple and Azure secrets if you want signed installers (optional).
* Push, run the workflow once via "Run workflow", and download the three artifacts to test on real machines. The
  Windows and macOS jobs have not been exercised yet.
