# Audio Cleaner for desktop

The desktop app is a PySide6 (Qt) window around the shared cleaning pipeline in `common/` (the server uses the same code).
The installer carries the app, FFmpeg and libDF (DeepFilterNet's native library) but no model. Every model is downloaded
by setup (2.1 GB in all, once), from one place (our Hugging Face repository; Whisper from its own): the noise removal
models are REQUIRED, the two larger ones are not. The app downloads a model only when asked to (a Download
button, or its own offer when the required ones are missing); its one automatic request is the update check, see
"Automatic updates" below. The installer is small; the
Windows installer's download page and the Mac .pkg's postinstall script check every file's SHA-256, keep files a
previous install left in place, and are generated from the app's catalog by `packaging/installer_models.py`:

- the noise removal models, DeepFilterNet3 (8 MB tarball for libDF) and DPDFNet-8 (15 MB ONNX, heavy noise) in one
  model, `resources.NOISE_MODEL` (`required=True`, first in the list; licence texts DeepFilterNet MIT + Apache-2.0 and
  DPDFNet Apache-2.0). One model, not two: they are used together (the app picks per recording) and neither is optional,
  so one download, one marker and one line in the app, and setup cannot leave half of noise removal behind;
- the speech model for subtitles, Whisper large-v3-turbo (1.6 GB, from Hugging Face, `resources.SPEECH_MODEL`);
- the model that rebuilds the voice (optional "Rebuild the voice" setting: removes room echo too), Sidon
  (503 MB ONNX, `resources.VOICE_MODEL`).

The last one is our own conversion (nobody else publishes this file; the noise models are the unchanged upstream
files, mirrored in the same repository). They are downloaded from our Hugging Face
model repository (`config.HF_MODELS_REPO`, pinned to the commit `config.HF_MODELS_REVISION`, a full commit id), file by file like Whisper, every file pinned by URL, size and
SHA-256: someone who only wants one model downloads only its files. Each model also has its licence texts
(`SpeechModel.licences`, `LICENSES/<name>` in the model's folder: Sidon + w2v-BERT 2.0 +
training-data notice), downloaded with it. They are not part of the model's `files`, so they are not in the `.complete`
marker: a copy installed by an older version (the old 0.9 GB archive unpacked into the model folders, with a shared
`<models>/LICENSES`) still counts as installed and is not downloaded again. About lists every installed model's
`LICENSES` folder and the old shared one (`models.licence_dirs`).

They go to `{app}\models\<id>` on Windows and `/Library/Application Support/Audio Cleaner/models/<id>` on a Mac
(`models.installer_dir`); a model whose files are all there also gets a `.complete` marker (the hashes of its files),
which the app compares with its catalog. The speech model's files are tried on their own, so one failed download does
not stop the next. If a download fails or is cancelled during setup (offline, no space on the temporary drive), setup still
finishes; Settings then shows a Download button for what is missing (and the banner at the top offers all of it;
`models.py`: pinned, resumable, verified, into the app's data folder; files setup did install are reused),
`AudioCleaner --download-models` does all of them from a terminal, and starting a job that needs a missing model asks
first.

**Required models.** Noise removal is the app's job, so a missing noise model is never a silent failure. Setup: Windows
shows a message after installing (and logs it, for silent installs) when the files are still missing; the Mac postinstall
logs it and shows a dialog in the installing user's session (best effort); the app is installed anyway so there is
something to retry from. App: when the models are missing at start (setup's download failed, a portable run, or a silent
self-update from a version that had them inside the program) it downloads them by itself (just these 23 MB, not the
2.1 GB; a failure shows a banner with "Try again"); Clean asks before starting, `--cli` says to run `--download-models`, and `engine.process` raises a clear
error instead of failing on a missing file. `resources.check()` lists only the shipped components (libDF, FFmpeg);
`resources.noise_ready()` is the models' counterpart. From source, `model_path()` and `heavy_noise_model_path()` also find
`vendor/models` (`prepare_vendor.py --download-noise-models`, for tests and CI) and `~/audio-cleaner/models`. Nothing is ever uploaded. The Mac download is a `.pkg` (not a disk image) because a disk image has no setup
step to download in. The .pkg is wrapped with `productbuild --distribution` (`packaging/macos/distribution.py`): it
refuses macOS older than 14, shows the EULA, and its welcome text says setup downloads about 2.1 GB and takes minutes.

**Where the app downloads from.** People get the installers from the public repository's Releases. Every desktop release
(tag `v<version>` in the public repository, made by the release job of `desktop.yml`) carries the three installers and
`update.json` / `update.json.sig`: installed apps read the newest release's (`config.LATEST_URL`, GitHub's fixed
`releases/latest/download/` address) and download the installer from the release of the version it names. That is the
only thing the updater uses the releases for; no model is in a release. The models come from Hugging Face: Whisper from
its own repository (`_TURBO` in `resources.py`, one pinned commit), ours from `config.HF_MODELS_REPO` (URLs built by
`config.hf_models_url`). Every file is checked against the size and SHA-256 pinned in `resources.py` before it is used,
so nobody who can change a file there can make the app run anything else; the update information is also signed (below).

The repository of the releases is `__releases__` in `audio_cleaner_desktop/__init__.py`; CI writes the one it publishes to
(`ACW_RELEASES_REPO`, the trial repository while `vars.PUBLIC_DESKTOP_REPO` is set) together with the version. The "Model
downloads exist" job runs `packaging/check_model_urls.py` on every build (HEAD requests: every pinned file is there with
its size).

### Changing a model safely

Never change a published file: every app version has its models' sizes and SHA-256 pinned.

1. Upload the new files to the Hugging Face model repository under a new path or as a new commit (`sidon/`,
   `LICENSES/`); the commit id is the pin, so old commits keep serving older app versions. Whisper: change the commit id
   in `_TURBO` in `resources.py` (never `main`).
2. Set `HF_MODELS_REVISION` in `config.py` to the new full commit id, and update the catalog in `resources.py`: each
   model file's `size`/`sha256` (and the licence texts'); for Whisper the files' sizes and hashes.
3. Update `deploy/models.lock` (the server's copy of the models) to match.
4. Push. Apps that update to the next release find that a model they had changed and download the new one by
   themselves (`models.outdated`), and older versions keep downloading theirs.

What users experience: the installed copy no longer matches the catalog (its `.complete` marker names the old hashes),
so it does not count as installed. On the next start of the updated app, about 6 seconds after the window opens, the
app downloads the new version of every model the user HAD, in the background with the usual banner ("Updating the
models ...", noise removal and everything else works meanwhile), regardless of the updates setting, because the models
are part of the app working. A model the user never had is not fetched (it is offered as before). The try happens
once per start: if it fails (offline, metered connection, disk full) there is no banner and no retry; the reason shows
under the model in Settings, with "Try again". A running Windows or macOS setup does the same during an interactive
update (`ForgetStaleModels` in the generated installer forgets files whose marker names other hashes, then fetches them).

Clean-up: when the new copy is complete the old data goes. In the app's own models folder (`models.remove_stale`):
a model folder whose marker names other hashes than the catalog, a model folder of an id that left the catalog, and
`.partial` folders of other hashes, and the `.partial` folders of the old models archive (a partial is only resumed when its `.pin` names exactly the hashes
of the current catalog entry). In the installer's folder (`{app}\models`, `/Library/Application Support/Audio Cleaner/models`)
the same is deleted only where the user may write; otherwise it stays until the next interactive setup (which, for
the files it replaces, does it itself). Folders without a `.complete` marker (including `LICENSES`) and loose files are
never touched, so the shared `LICENSES` of older installs stays.

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
`native/{df.dll|libdf.dylib, ffmpeg[.exe], ffprobe[.exe]}`, `licenses/` (no models; shown in About, together with the licence line of each downloaded model). Qt, FFmpeg and libDF stay separate files on purpose (LGPL, see
the checklist at the end).

## Release process (CI)

1. Releases are automatic: every push to `main` / `on-prem` that changes `desktop/`, `common/`,
   `native/deepfilter/` or the workflow builds and publishes a GitHub Release. Its version is MAJOR.MINOR from
   `__version__` in `desktop/audio_cleaner_desktop/__init__.py` plus the CI run number as the last part (2.0.11,
   2.0.12, ...; `desktop/packaging/version.py` writes it into the app before it is built, so the app, the installers,
   the file names and the release title all show it). Only the newest 5 automatic releases are kept. Bump MAJOR.MINOR
   in `__init__.py` (and `version` in `desktop/pyproject.toml`) when the app changes in a way friends should notice.
2. Optional, for a release that stays: tag the commit, e.g. `git tag desktop-v2.0.50 && git push origin desktop-v2.0.50`.
   The tag builds exactly that version; its MAJOR.MINOR must match `__init__.py`. Releases from tags are never
   deleted by the cleanup.
3. `.github/workflows/desktop.yml` builds, per platform: FFmpeg (LGPL, from pinned source tarballs; Windows is
   cross-compiled with mingw-w64 on Ubuntu), libDF, then the app: tests, PyInstaller, the frozen `--selftest --gui`
   (also builds the window and runs the noise models, echo canceller and Sidon: CI downloads them from Hugging Face first, like the app does),
   and the installer (Inno Setup 6.7.1 on Windows, a `productbuild` .pkg on macOS; both download the models during
   setup, file by file from Hugging Face). Windows also runs the frozen app under a non-ASCII profile folder and installs, starts and uninstalls the
   installer silently. A separate "Model downloads exist" job checks the catalog URLs. Website/server-only pushes skip it; "Run workflow"
   builds by hand (artifacts only, no release). About 10 minutes with warm FFmpeg/libDF caches, 20-30 cold.
4. The `release` job attaches `AudioCleaner-<version>-Windows.exe`, `AudioCleaner-<version>-Mac-Apple-silicon.pkg`
   and `AudioCleaner-<version>-Mac-Intel.pkg` (GitHub shows each file's SHA-256 itself, so there is no separate
   checksum file) plus `update.json` and `update.json.sig` (the signed update information, see "Automatic updates"), and
   marks the release as latest. Don't share GitHub links of this repository (it stays out of sight). The app is not
   offered on the website: the installers people should get are the ones in the public repository's Releases
   (`BenjaminDanker/Audio-Cleaner`, published there by the step below with the update information and the models),
   and that is also where installed apps get their updates and models.
5. Smoke-test the downloads on a clean Windows and a clean Mac (arm64 and Intel if you can) before announcing: open a
   noisy video, clean it, turn on subtitles, open About.

Secrets are only read by the signing steps; without them every step is skipped and unsigned installers are produced.
(`UPDATE_SIGNING_KEY` is a different thing: see "Automatic updates".)

## Automatic updates

An installed app looks for a newer version about 10 seconds after it starts (at most once in 24 hours, unless the user
presses "Check now" in Settings), downloads it in the background and installs it. It only runs in the frozen app; from
source nothing happens by itself. The code is `audio_cleaner_desktop/updater.py` (a Qt-free core plus the small
`UpdateController` the window uses); the setting is "Check for updates when Audio Cleaner starts" in Settings (on by
default), About shows "Updates: on/off", and the EULA and the About text say what is sent: a normal web request to
GitHub (the IP address, and the app version in the User-Agent), nothing else.

**What the app does.** It fetches `update.json` from the newest release (`config.LATEST_URL`) and `update.json.sig` (small, 20 s timeout, https only, the system certificate store), verifies the signature with the
public key built into the app, and only then reads the manifest. It ignores a manifest whose version is not newer than
its own (versions compare as numbers, so 2.0.15 is newer than 2.0.9), picks the file of its platform
(`windows-x86_64`, `macos-arm64`, `macos-x86_64`) and downloads that installer into the app's data folder
(`%LOCALAPPDATA%\Audio Cleaner\Audio Cleaner\updates`, `~/Library/Application Support/Audio Cleaner/Audio Cleaner/updates`)
with the model downloader's code (`models.fetch_file`: https only, never a redirect to http, resumable, size and SHA-256
checked, free space checked first). The download address is the release of the version the manifest names plus the
file name (`config.release_url(version)`), not taken from the manifest. Only the newest download is kept;
the verified manifest is saved next to it, so the next start finds the download again without asking the server (and checks
the signature and the file's hash once more). A manifest with a bad or missing signature, a size or hash that does not
match, an unexpected file name, or no file for the platform is never used; every failure is a line in the log
(Help > Open log folder) and in the status line under the setting, never a dialog.

* **Windows.** A banner says "Audio Cleaner <v> is ready. It installs when you close the app." with a "Restart now"
  button. When the app is closed (normally, or after the user confirmed quitting a running batch) it starts the
  downloaded setup detached, with `/VERYSILENT /SUPPRESSMSGBOXES /NORESTART` and `/CURRENTUSER` or `/ALLUSERS` when the
  running copy lives in the user's profile or in Program Files (for any other folder no switch: Inno Setup then reuses
  the previous install's mode, `UsePreviousPrivileges=yes`, which its documentation says it does without asking). An
  all-users install therefore shows the normal administrator prompt. That automatic install is tried once per version
  (`update_install_tried` in the settings): if the prompt was declined, closing the app does not ask again for the same
  version, "Restart now" does. "Restart now" waits for a running batch to finish (the banner says so) and then closes the
  app; setup is started with `/RELAUNCH=1`, which makes a silent setup start the new version as the original user (the
  second `[Run]` entry in `AudioCleaner.iss`; without the parameter a silent setup starts nothing). A silent setup never
  downloads models; if a new version needs different models the app offers the download as usual.
* **macOS.** The banner says "Audio Cleaner <v> is downloaded and ready to install." with an "Install" button that opens
  the `.pkg` with `open`; the macOS installer asks for the password. Nothing is silent. (A file the app downloads itself
  does not get the quarantine flag, so Gatekeeper does not block the opened `.pkg`.)
* **Settings.** The checkbox, a "Check now" button and a line of text: "You have the newest version (2.0.15).",
  "Downloading 2.0.16... 40%", "Version 2.0.16 is ready...", or the error. Turning the checkbox off stops checks and
  automatic installs; the user can still install a new version by hand.

**The signature.** CI signs `update.json` in the release job (`desktop.yml`) with a raw Ed25519 signature over the exact
bytes of the file: `openssl pkeyutl -sign -rawin -inkey KEY.pem -in update.json -out update.json.sig` (64 bytes). The
private key is the GitHub Actions secret **`UPDATE_SIGNING_KEY`** (the PEM file, with its `-----BEGIN PRIVATE KEY-----`
lines); a copy is kept offline, outside the repository. The release job builds `update.json` with
`packaging/make_update_manifest.py` (version, publish time, and the name, size and SHA-256 of the three
installers, keys sorted), signs it, checks the signature against the key's public half and uploads both files with the
installers, for automatic and tagged releases alike. It fails closed: **without the secret, or with a key that is not
the one built into the app, the release gets no `update.json`** (a warning or error annotation shows it) and no installed
app updates itself; the installers are still published. The public key is `PUBLIC_KEY` in `updater.py` (raw Ed25519,
base64). Signature checking uses the `cryptography` package, pinned in `requirements.lock` (48.0.1, the newest release
that still ships a macOS Intel wheel; later releases dropped it) with `cffi` and `pycparser`; the frozen
`--selftest` makes a key pair, signs and verifies to prove the bundled copy works.

**Checking that updates are live.** After a build: the public repository's newest release has `update.json`,
`update.json.sig` and the models, `curl -sL https://github.com/<owner>/<repo>/releases/latest/download/update.json` shows the
version and hashes; `openssl pkeyutl -verify -rawin -pubin -inkey PUB.pem -in update.json -sigfile update.json.sig` says
"Signature Verified Successfully" (PUB.pem: `openssl pkey -in KEY.pem -pubout`).

**Rotating the key** (planned, or after a leak). 1. `openssl genpkey -algorithm ed25519 -out new.pem`.
2. Public half in app form: `openssl pkey -in new.pem -pubout -outform DER | tail -c 32 | base64`. 3. Put that into
`PUBLIC_KEY` in `updater.py`, store `new.pem` in the secret `UPDATE_SIGNING_KEY`, push. The new release is signed by the
new key; **apps already installed only trust the old key**, so they cannot update themselves to it. They have to
get one last release signed with the OLD key whose app already contains the new `PUBLIC_KEY`: do step 3 in two pushes
(first change `PUBLIC_KEY` while the secret still holds the old key, wait for that release; then replace the secret).
`make_update_manifest.py check-key` (run by the release job) refuses a secret that does not match `PUBLIC_KEY`, so a
mistake in the order publishes no manifest instead of one nobody trusts. Users who miss the bridging release install
the newest setup by hand.

**If the secret leaks.** The key alone does not let anyone push a file to the apps: the manifest only names a file (the app builds the address itself, in the public repository's releases, and downloads it over https),
so an attacker also needs to be able to change those releases. Treat a leak as serious anyway: 1. delete the secret at once (a
release without the key gets no manifest); 2. look at the releases and the workflow runs for anything you did not
make, and delete a release with an `update.json` you did not make (apps then see the previous release); 3. if only the
key leaked, rotate it with the two pushes above; 4. if the GitHub account or a token leaked as well, nobody can say what
was signed: make a new key, change `PUBLIC_KEY`, publish new installers, and tell users (release page) to install the
newest setup by hand once, because installed apps keep trusting the old key. Rotate the key as well if a machine that
held a copy was lost.

**Owner to-do.** Add the secret `UPDATE_SIGNING_KEY` (GitHub > Settings > Secrets and variables > Actions) with the
contents of `~/audio-cleaner/secrets/update-signing-key.pem`. Not yet exercised on real machines: see the checklist
at the end of this file.

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
   `df.dll` in `$OUT` (default `~/audio-cleaner/src/deepfilter-build/out`).
2. **FFmpeg**: `desktop/packaging/build_ffmpeg.sh <linux-x86_64|macos-arm64|macos-x86_64|windows-x86_64> <out_dir>`.
   LGPL only (no GPL, version3, nonfree, no x264): FFmpeg 8.0.3 + libopus + LAME + libvpx (BSD, only its VP9 encoder is
   used) from sha256-pinned tarballs, linked statically into `ffmpeg` and `ffprobe`. H.264 pictures are encoded by the
   operating system through FFmpeg: `h264_mf` (Media Foundation, `--enable-mediafoundation`; FFmpeg 8.0's `mfenc.c` also
   needs `--enable-d3d11va` to compile; the programs load Media Foundation themselves when asked, so they still start
   on a Windows without it) on Windows and `h264_videotoolbox` (`--enable-videotoolbox`) on a Mac; FFmpeg's own `mpeg4`
   (and `flv` for .flv files) are the encoders everywhere else and the fallback. `windows-x86_64` cross-compiles on
   Linux (`apt install mingw-w64 g++-mingw-w64-x86-64 nasm`; libvpx has a little C++ and needs perl to build) or builds
   natively in an MSYS2 MINGW64 shell. `REQUIRE_ASM=1` fails when nasm is missing instead of building slower code.
   `WITH_LAME=0` / `WITH_VPX=0` leave those encoders out. Writes `ffmpeg-buildinfo.txt` and the licence texts next to the
   programs.
3. **Vendor directory**:
   ```sh
   python desktop/packaging/prepare_vendor.py --libdf PATH/libdf.so --ffmpeg-dir FFMPEG_OUT \
       [--download-noise-models] [--whisper-model DIR | --download-whisper] [--deepfilter-src DIR] --out desktop/vendor
   ```
   Copies the native files, puts (with `--download-noise-models`, for the tests) the two noise models from Hugging Face
   in `vendor/models/` (pinned; not bundled: the spec leaves that folder out), puts a small test-only speech model in
   `vendor/test-models/whisper-small/` (`Systran/faster-whisper-small`, revision and file hashes pinned in the script;
   `--download-whisper` fetches it; it is not bundled) and
   writes `licenses/`: FFmpeg/LAME/opus/libvpx, DeepFilterNet, DPDFNet, the Rust crates of libDF (`cargo tree` of the capi
   build), all Python packages of the interpreter running the script, Qt/PySide6 (LGPL + GPL text from
   `desktop/packaging/licenses/`), Whisper, Silero VAD, the downloaded models (Sidon, w2v-BERT 2.0 and the
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
   the echo canceller (and through Sidon when its model is installed), and with `--gui` builds the
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
uncompressed (no model inside; the two noise models, 23 MB, are downloaded: DeepFilterNet 8 MB, DPDFNet-8 15 MB);
it is PySide6/Qt 140 MB,
CTranslate2 137 MB (library and extension), SciPy and its OpenBLAS 71 MB, ONNX Runtime 30 MB, NumPy and its OpenBLAS
41 MB, libDF/FFmpeg 38 MB. Setup then downloads 2.1 GB of models: the noise removal models 23 MB, Whisper large-v3-turbo 1.6 GB,
Sidon 503 MB (2.1 GB on disk too). Windows setup needs that much free space on the temporary drive until the files are
installed (it checks first); the app's own download of one model needs about that model's size. Memory
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
* `EULA.txt` (what the installers show) is the app's MIT License with what the app does online, the open source
  components (FFmpeg and the downloaded models Whisper, Sidon, w2v-BERT 2.0 named) and a note on the user's
  files. It must not add restrictions that the MIT License or the LGPL components' licences don't allow.
* The downloaded models carry notices too: `Sidon-MIT.txt`,
  `w2v-BERT-2.0-MIT.txt` (Meta) and `Training-data-attribution.txt` (CC BY 4.0) ship in
  `licenses/` and are shown in About, and the same files are in the model repository
  on Hugging Face (`LICENSES/`); setup and the app download each model's texts into its folder (`<model>/LICENSES`),
  and About lists those folders.
  The Dropbox CTranslate2 conversion of Whisper has no licence text of its own (its card says MIT); OpenAI's
  `Whisper-MIT.txt` is the text that applies.
* Re-run the licence collection (it runs in every build) after changing dependencies; it fails when a package has no
  licence text. New Python packages that end up in the bundle need an entry check in `prepare_vendor.py`.
* Not legal advice; confirm the Qt/LGPL points with counsel (see `docs/research/desktop-packaging.md`).

## Updating

The installed app updates itself (see "Automatic updates"); by hand, just run the new setup (the .exe or the .pkg) over
the installed app. There is nothing to uninstall first; it is the same install, done again, and the models are not
downloaded again. The app's silent update is this same setup with `/VERYSILENT`.

* **Windows.** Setup closes the running app, reuses the folder and the mode (per user or for all users) of the previous
  install (`UsePreviousAppDir`/`UsePreviousPrivileges`; the `AppId` never changes), deletes the old program files
  (`[InstallDelete]` removes `{app}\_internal`, so nothing of an older version lingers) and copies the new ones.
  `{app}\models` is never deleted by an update. The model step looks at what is installed: a model whose files are all
  there with the right sizes (and a `.complete` marker that matches the catalog) is not downloaded again; with every
  model complete setup does not even show the download page. A model from an earlier, partial install is finished:
  its missing or wrong-sized files are downloaded one by one; a model whose marker
  names other hashes than this catalog is removed and fetched again. A silent setup (`/VERYSILENT`) never downloads
  models, as before.
* **macOS.** The .pkg replaces `/Applications/Audio Cleaner.app` completely: its preinstall script removes the old bundle
  (that one folder and nothing else) before the new one is laid down, so no file of an older version can stay inside it,
  and the component is not skipped for being "the same or an older version". Its postinstall script skips the files
  that are complete, as on Windows. Quit the app before installing.
* **What an update keeps.** The settings (every option in the settings panel, the window size and position, the tab
  last used), the history and everything the app downloaded itself are outside the program folder: the settings are in
  the registry (`HKEY_CURRENT_USER\Software\Audio Cleaner`) or the macOS preferences, the history and the app's own
  model downloads in `%LOCALAPPDATA%\Audio Cleaner\Audio Cleaner` or `~/Library/Application Support/Audio Cleaner`.
  The models setup installed stay in `{app}\models` or `/Library/Application Support/Audio Cleaner/models`.
* **In CI.** The Windows job installs, installs again over the first install (checking that a stale file in
  `_internal` is gone and a file in `models` is still there), runs the frozen selftest and uninstalls.

## Things worth knowing

* **What a result is called and what it becomes** (`engine.py`, `gui/queue_view.py`). Under each file in the list is a name
  box that starts with the usual name (`<name>-cleaned.<ending>`) and a list of the types that file can become, joined
  to its right. The ending in the box and the list always agree: typing `.mp3` selects MP3, choosing a type in the list
  rewrites the ending (an ending that is only part of a name, like `v1.2`, gets the new one added). A name that can't be
  used (empty, no ending, an ending that is not in the list, characters Windows does not allow, a reserved Windows name,
  a WAV too long for the format) gets a short red reason under the box once the box is left, and Clean leaves that file
  out (the others start; the status line says how many wait for a fixed name). A taken name is never replaced: it gets
  ` (2)`, ` (3)` (the usual name keeps `-cleaned (2)`). The list is `engine.available_endings()`: for a sound WAV, FLAC,
  MP3, M4A (AAC), Opus, OGG (Opus) and the video types; for a video the video types and the sound types.
  * *Sound to sound*: the pipeline's encoders as before. *Video to just its sound*: the same, the other tracks are not kept
    (the row says so). *Sound to video*: a black 1280x720 picture at one frame a second (an FFmpeg `color` source in a
    filter graph, so no input device is needed) with the cleaned sound, so subtitles can be embedded; it takes the Video
    size setting like any video result.
  * *Video to video* (MP4, MOV, MKV, FLV, WebM): the picture is copied without encoding when the file stays the same
    kind (also MP4 <-> M4V) or the new container holds that codec (`ffmpeg._COPY_VIDEO`; MKV takes anything), and is
    encoded otherwise or when the size changes: with the system's H.264 encoder (Windows `h264_mf`, macOS
    `h264_videotoolbox`), else MPEG-4 (`flv` for .flv), VP9 for .webm. The bit rate is a share of the pixels per second
    that suits the encoder (`picture_bitrate`), never far above the original's. If the system's encoder fails the next
    one is tried and the row says "this computer's own video encoder wasn't available...". Encoding reports its
    progress (FFmpeg `-progress`) into the bar. WebM is offered for a picture it holds as it is (VP8/VP9/AV1) and, since
    the build has libvpx, for every video and sound (`engine.WEBM_ENCODE`).
  * *Subtitles* embed when the container holds them (MP4, MOV, MKV, WebM); a Flash video gets an .srt file instead.
  * *Video size* (Settings, default Same as the original): 2160p/1440p/1080p/720p/480p is the length of the shorter side,
    so the picture keeps its shape and a tall video stays tall; plain lanczos resizing, bigger than the original only
    stretches it; a phone video stored sideways is judged as it is shown and comes out upright. A picture that already
    has the size is copied. `common/acw_common/ffmpeg.py` does all of it behind `mux_video(..., video=VideoOut(...))` and
    `make_video()`; without `video=` `mux_video` copies the picture exactly as before (the server).
  * *The "Audio file type" setting* is kept as the starting point of the box for sound files (so nothing changes for people
    who never touch the box); a video starts in its own container. The usual name follows the settings until the user
    types something else (or picks another type); then the name is theirs.
  * *Clean again* uses the name in the box. Under the same name as the last run the new result takes the place of the old
    one (which goes to the Recycle Bin / Trash, as before); under another name or type it is a new file and the old one
    stays.
  * *Command line*: `--output-name NAME` (one file), `--format EXT` (the type of every result; the usual name), `--video-size`.
* **Only the Windows and macOS builds can show the system encoders work.** Linux has none, so the tests there (and in
  development) take the MPEG-4 path; the CI tests on Windows and macOS accept either codec.
* **Preview of videos.** The sound plays in Qt's player; the picture is decoded by our own FFmpeg at the preview's size and shown against the sound's clock (`gui/video_preview.py`). Qt's player decodes every frame and holds the sound back when the picture is late, so an 8K video on a slower PC played in slow motion. Now late frames are dropped, the decoder jumps ahead to the key frame nearest the sound when it falls behind, and decodes key frames only if it keeps falling behind; the sound is never held back. Formats our FFmpeg does not decode (AV1) still use Qt's player. That is why the FFmpeg build includes video decoders and the scaler (`build_ffmpeg.sh` VIDEO_DECODERS).
* **No console window flashes on Windows.** The app has no console, so every console program started by a library would
  get a window of its own for a moment (Python's `platform` module runs `cmd /c ver` when its WMI query fails, and
  onnxruntime asks `platform` at import time). `audio_cleaner_desktop/quiet_popen.py` makes `CREATE_NO_WINDOW` the
  default of every `subprocess.Popen` (and so `run`, `check_output`, `os.popen`) on Windows; `packaging/launcher.py`
  installs it before anything else is imported. Callers that pass their own console flag, and Explorer, are left alone.
  `_wmi` is a hidden import of the spec, so `platform` finds it, and `--selftest` checks both on Windows.

* **Uninstalling.** Windows: the uninstaller removes the program and the models setup installed, and asks whether to
  remove the models the app downloaded itself, the history and the settings (`%LOCALAPPDATA%\Audio Cleaner`); a
  silent uninstall keeps them. macOS: dragging the app to the Trash leaves the models in
  `/Library/Application Support/Audio Cleaner/models` (2.1 GB, installed by the .pkg), the app's own data in
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

## Public copy of the desktop code

The desktop app lives here, next to `common/` (shared with the website's server). People still use the old public
repository (`BenjaminDanker/Audio-Cleaner`, once a different app called VideoDenoiser, releases v1.0.0 and v1.0.1), so
the app's code and each release are copied there automatically, one way, without mentioning this repository.

* **What is copied** (`ALLOWLIST` in `tools/public-desktop/sync.py`): `desktop/` (app, packaging scripts, the licence
  texts and EULA the installers ship, tests), all of `common/` (the app imports captions, clarity, dfn, dpdfnet,
  ffmpeg, pipeline, restore and transcribe, which pull in aec and gpu; its tests are part of the desktop test run) and
  `native/deepfilter/` (the libDF build). Nothing else: no `deploy/`, `server/`, `web/`, `docs/`, `obs-plugin/`,
  `models.lock`. Files that exist only there live in `tools/public-desktop/template/`: `README.md`, `LICENSE` (this
  repository's licence text without the line about `obs-plugin/`), `.gitignore` and `.gitattributes`. The old repository's own files (the VideoDenoiser source and models) are
  removed from its tree by the first sync; they stay in its history and under the tags v1.0.0 and v1.0.1.
* **How**: `.github/workflows/public-desktop.yml` runs on a push to `on-prem` that touches an allowlisted path (or by
  hand) and runs `python3 tools/public-desktop/sync.py --target <checkout> --rev $GITHUB_SHA`. The script exports the
  allowlisted paths of that commit (what is committed, not the working tree), scans the export (`LEAK_PATTERNS`: this
  repository's name, the owner's account name, the server's IP, our login name and home folder, key paths, tokens, CI
  secret names, paths of folders that are not copied) and stops with a list of file:line hits if anything matches,
  replaces the checkout's tracked files with the export, and commits `Sync from upstream <short sha>` (no commit
  when nothing changed). The workflow pushes without force to the public repository's default branch. Try it by hand
  with `--dry-run` (reports, changes nothing) or `--export-only DIR` (just the export and the scan).
* **Releases are copied, nothing is built there**: the public repository has no build workflow (the sync deletes
  `.github/workflows/desktop.yml` if it is there; `ABSENT` in `sync.py`). Its Releases hold the same installer files
  as ours: the step "Copy the release to the public repository" of the release job in `.github/workflows/desktop.yml`
  runs after our release is published and, with the secret `PUBLIC_RELEASE_TOKEN`, creates the release `v<version>`
  there (the old tags are v1.0.0 and v1.0.1) on the default branch, marked latest, with the three installers
  (`AudioCleaner-<version>-Windows.exe`, `-Mac-Apple-silicon.pkg`, `-Mac-Intel.pkg`) and short notes that say which file is for which computer. No `update.json`, no signature: those are for our
  download server, and the app trusts only the signed manifest it serves, so a release there is a download for people,
  never an update source. A rerun updates the
  release that exists (title, notes, files replaced with `--clobber`). Releases made from a branch push carry the
  same hidden marker as ours, and only the newest 5 of those are kept there (v1.0.0, v1.0.1 and releases from a
  `desktop-vX.Y.Z` tag have no marker and stay). Without the secret the step prints a warning and is skipped; any
  error in it is a warning too and never fails the official release.
* **Keep in mind when editing**: anything under `desktop/`, `common/` and `native/deepfilter/` becomes public, so keep
  names of servers, hosts, accounts and secrets out of comments and tests; the sync fails and lists file:line when it
  finds one. Tests that read files outside the copied folders (the release job of `.github/workflows/desktop.yml`,
  `web/src/jobs.js`) skip themselves when the file is not there (the public copy has no workflows).

**A trial run in private first:** both the code sync and the release copy go to `vars.PUBLIC_DESKTOP_REPO` instead
of the public repository while that variable is set, with the code sync using the secret
`PUBLIC_DESKTOP_TEST_DEPLOY_KEY` (a deploy key belongs to one repository). Delete the variable and the next run goes to
the public repository again.

```sh
gh repo create BenjaminDanker/Audio-Cleaner-trial --private
git clone https://github.com/BenjaminDanker/Audio-Cleaner.git /tmp/acw-trial && git -C /tmp/acw-trial push --mirror git@github.com:BenjaminDanker/Audio-Cleaner-trial.git
ssh-keygen -t ed25519 -N '' -C public-desktop-trial -f /tmp/trial-key
gh repo deploy-key add /tmp/trial-key.pub --repo BenjaminDanker/Audio-Cleaner-trial --allow-write --title public-desktop-trial
gh secret set PUBLIC_DESKTOP_TEST_DEPLOY_KEY --repo BenjaminDanker/Audio-Cleaner-Web < /tmp/trial-key
shred -u /tmp/trial-key /tmp/trial-key.pub && rm -rf /tmp/acw-trial
gh variable set PUBLIC_DESKTOP_REPO --repo BenjaminDanker/Audio-Cleaner-Web --body BenjaminDanker/Audio-Cleaner-trial
# ... push on-prem, look at the trial repository; then:
gh variable delete PUBLIC_DESKTOP_REPO --repo BenjaminDanker/Audio-Cleaner-Web
```

**Setting it up (owner, once):** two secrets in this repository. The code sync uses an SSH deploy key (limited to the
one repository, does not expire, not tied to your account); the release copy uses a fine-grained personal access token.

```sh
ssh-keygen -t ed25519 -N '' -C public-desktop-sync -f /tmp/public-desktop-key
gh repo deploy-key add /tmp/public-desktop-key.pub --repo BenjaminDanker/Audio-Cleaner --allow-write --title public-desktop-sync
gh secret set PUBLIC_DESKTOP_DEPLOY_KEY --repo BenjaminDanker/Audio-Cleaner-Web < /tmp/public-desktop-key
shred -u /tmp/public-desktop-key /tmp/public-desktop-key.pub
```

The token (a deploy key cannot create releases through the API, so this one is needed; do it in the browser, signed in as
BenjaminDanker):

1. GitHub > your avatar > Settings > Developer settings > Personal access tokens > **Fine-grained tokens** >
   Generate new token.
2. Token name `audio-cleaner-release-copy`; Resource owner: BenjaminDanker; Expiration: the longest you accept (the
   copy stops with a warning when the token expires, and `gh secret set` below replaces it).
3. Repository access: **Only select repositories** > `BenjaminDanker/Audio-Cleaner` (the public one, not this one).
4. Permissions > Repository permissions: **Contents: Read and write**. Nothing else is needed: creating and editing
   releases, uploading and deleting their files, creating and deleting the tag and listing releases are all covered by
   Contents, and Metadata (read) is added automatically. Leave everything else at "No access".
5. Generate token, copy it, then on your computer (the command asks for the value, so it is not left in your shell
   history):

```sh
gh secret set PUBLIC_RELEASE_TOKEN --repo BenjaminDanker/Audio-Cleaner-Web
```

On the old repository (Settings): keep the default branch `main`, do not protect `main` in a way that blocks the deploy
key (a deploy key cannot bypass branch protection), and set the description and website
(https://audiocleaner.live/). Run "Public desktop copy" once with "Run workflow" on `on-prem` and look at the
commit it pushed (it also removes the old build workflow) before announcing anything; the first release is copied by the
next release of the app. To stop the copies, delete the secrets or the workflows.

## Owner to-do before the first release

* Upload the models repository (`SilverJunk08/audio-cleaner-models`, layout in `resources.py`), set
  `HF_MODELS_REVISION` in `config.py` to its full commit id, and check that the "Model downloads exist" job is green
  (`check_model_urls.py`: the pinned files on Hugging Face).

* Set the Apple and Azure secrets if you want signed installers (optional).
* Add the secret `UPDATE_SIGNING_KEY` (the contents of `~/audio-cleaner/secrets/update-signing-key.pem`) so releases get
  a signed `update.json`; without it nothing updates itself (see "Automatic updates").
* Push, run the workflow once via "Run workflow", and download the three artifacts to test on real machines. The
  Windows and macOS jobs have not been exercised yet. On real machines also check the updates: install an older
  build, publish a newer one, and see (Windows, per user and for all users) the banner, "Restart now" (the new version
  starts, not elevated), closing the app instead (silent install, no window, no console flash, the app starts normally
  next time), a declined administrator prompt, a running batch; (macOS, both architectures) the "Install" button
  opens the installer. Check the Windows install log if `/RELAUNCH=1` or `{param:...}` misbehaves.
