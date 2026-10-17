# Audio Cleaner for desktop

The desktop app is a PySide6 (Qt) window around the shared cleaning pipeline in `common/` (the server uses the same code).
The installer carries the app, FFmpeg and the two noise models (DeepFilterNet3, and DPDFNet-8 for heavy noise), so
noise removal works offline straight after installing. Three larger models (2.5 GB in all) are downloaded by setup
itself, and only by setup (the app downloads a model only when asked to; its one automatic request is the update check,
see "Automatic updates" below). The installer is small; the
Windows installer's download page and the Mac .pkg's postinstall script check every file's SHA-256, keep files a
previous install left in place, and are generated from the app's catalog by `packaging/installer_models.py`:

- the speech model for subtitles, Whisper large-v3-turbo (1.6 GB, from Hugging Face, `resources.SPEECH_MODEL`);
- the model that restores the highs of phone-quality recordings, AP-BWE (4 x 119 MB ONNX, `resources.HIGHS_MODEL`);
- the model that rebuilds the voice (optional "Rebuild the voice" setting: removes room echo too), Sidon
  (503 MB ONNX, `resources.VOICE_MODEL`).

The last two come in ONE zip archive, `AudioCleaner-models-v1.zip` (0.9 GB, `resources.MODELS_ARCHIVE`), built from this
repository's `models-v1` release and served by our own download server (see "Where the app downloads from"); it also holds a `README.txt` and the `LICENSES/` folder (AP-BWE, Sidon, w2v-BERT 2.0
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

**Where the app downloads from.** Everything the app and the installers fetch except the Whisper files (Hugging Face,
which says nothing about us) comes from our own domain, so users never see the GitHub repository (it may become
private). The base address is written in ONE place, `config.DOWNLOADS_BASE` (`https://mc-multi-world.space/audio/api/downloads/`);
`config.UPDATES_URL` (`desktop/`) and `config.MODELS_URL` (`models/`) are built from it, and the updater, the models
catalog (`resources.py`), the Windows and macOS installers (generated from the catalog by `installer_models.py`) and
`check_model_urls.py` all use those, so a domain move is one edit there. The server mirrors the GitHub releases and
serves verified copies: `desktop/update.json`, `desktop/update.json.sig`, `desktop/<installer name from update.json>`
and `models/AudioCleaner-models-v1.zip` (byte for byte the release asset: same size and SHA-256), with `Range`
requests working (resumed downloads). Nothing that ships (app, installers, About, EULA, README, notices) names the
repository; `test_model_updates.py` fails if it does. GitHub still builds and stores the releases (CI files and this
documentation may name it).

**The `models-v1` release** (on GitHub, which the mirror reads) must hold the archive before the installers can fetch it: upload the one file
`AudioCleaner-models-v1.zip` (908,538,132 bytes, sha256 `5592be63...81ce`; `README.txt`, `LICENSES/*`,
`ap-bwe/{8k,12k,16k,24k}to48k.onnx`, `sidon/{sidon_fe,sidon_dec}.onnx`, deflate) to a release tagged `models-v1` with
`--latest=false` (otherwise it takes over `releases/latest` from the app). Never replace the file under the same name
with different content: the catalog pins its size and SHA-256 (change both in `resources.MODELS_ARCHIVE`, and the
member sizes and hashes, when a new archive is published, preferably under a new tag such as `models-v2`). The
"Model downloads exist" job in the workflow compares the size of the archive on our server (and of the speech model
files on Hugging Face) with the catalog and goes red when it is missing, for instance because the mirror does not
have it yet; it does not block the builds.

### Changing a model safely

Never replace an asset in an existing `models-vN` release (the mirror, the installers and every installed app have
that name's size and SHA-256 pinned). Instead:

1. Publish a new release `models-vN+1` (`--latest=false`) with the new archive `AudioCleaner-models-vN+1.zip`. For a
   Whisper revision, pick the new Hugging Face commit id instead (never `main`).
2. Update the catalog in `resources.py`: for an archive the `name` and `url` (`config.MODELS_URL + "AudioCleaner-models-vN+1.zip"`),
   `size`, `sha256`, the `shared` licence files and the `size`/`sha256`/`member` of each model file; for Whisper the
   revision in `_TURBO` and the files' sizes and hashes. Run `python desktop/packaging/check_model_urls.py` once the
   mirror has the file. If the mirror is configured per tag, add the new tag there.
3. Update `deploy/models.lock` (server side): the `archive` line for the models archive is
   `archive  -  <sha256>  <size>  models-v1  https://github.com/.../releases/download/models-v1/AudioCleaner-models-v1.zip`;
   add a line for `models-vN+1` with the new sha256, size, name and URL (keep the old line until every server is
   updated), and change the `member` lines of the AP-BWE and Sidon files that come out of it to
   `models-vN+1:<path inside the archive>` with their new sha256 and size.
4. Ship a desktop build with the new catalog. A change to a Whisper file URL or hash works the same way.

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
`.partial` folders of other hashes or older archives (a partial is only resumed when its `.pin` names exactly the hashes
of the current catalog entry). In the installer's folder (`{app}\models`, `/Library/Application Support/Audio Cleaner/models`)
the same is deleted only where the user may write; otherwise it stays until the next interactive setup (which, for
the files it replaces, does it itself). Folders without a `.complete` marker (including `LICENSES`) and loose files are
never touched, so the licence texts of the current archive stay.

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
`native/{df.dll|libdf.dylib, ffmpeg[.exe], ffprobe[.exe]}`, `models/{DeepFilterNet3_onnx.tar.gz, dpdfnet8_48khz_hr_batch.onnx}`,
`licenses/` (shown in About, together with the licence line of each downloaded model). Qt, FFmpeg and libDF stay separate files on purpose (LGPL, see
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
   (also builds the window and runs the noise models, echo canceller, AP-BWE and Sidon when their files are there),
   and the installer (Inno Setup 6.7.1 on Windows, a `productbuild` .pkg on macOS; both download the models during
   setup; the ONNX models arrive as one archive). Windows also runs the frozen app under a non-ASCII profile folder and installs, starts and uninstalls the
   installer silently. A separate "Model downloads exist" job checks the catalog URLs. Website/server-only pushes skip it; "Run workflow"
   builds by hand (artifacts only, no release; a push that only changes `deploy/fetch-dpdfnet.sh` also builds). About 10 minutes with warm FFmpeg/libDF caches, 20-30 cold.
4. The `release` job attaches `AudioCleaner-<version>-Windows.exe`, `AudioCleaner-<version>-Mac-Apple-silicon.pkg`
   and `AudioCleaner-<version>-Mac-Intel.pkg` (GitHub shows each file's SHA-256 itself, so there is no separate
   checksum file) plus `update.json` and `update.json.sig` (the signed update information, see "Automatic updates"), and
   marks the release as latest. Don't share GitHub links (the repository stays out of sight); the links to share with
   friends come from our download server and always give the newest version:
   `https://mc-multi-world.space/audio/api/downloads/desktop/latest/windows`, `.../latest/mac-apple-silicon` and
   `.../latest/mac-intel`. The app is not on the website and never
   contacts it or GitHub: it contacts our download server (update check and installers, models) and Hugging Face
   (Whisper). Nobody is meant to be pointed at GitHub: the server mirrors the three installers and the update
   information, so share the website's download links instead of the releases page.
5. Smoke-test the downloads on a clean Windows and a clean Mac (arm64 and Intel if you can) before announcing: open a
   noisy video, clean it, turn on subtitles, open About.

Secrets are only read by the signing steps; without them every step is skipped and unsigned installers are produced.
(`UPDATE_SIGNING_KEY` is a different thing: see "Automatic updates".)

## Automatic updates

An installed app looks for a newer version about 10 seconds after it starts (at most once in 24 hours, unless the user
presses "Check now" in Settings), downloads it in the background and installs it. It only runs in the frozen app; from
source nothing happens by itself. The code is `audio_cleaner_desktop/updater.py` (a Qt-free core plus the small
`UpdateController` the window uses); the setting is "Check for updates when Audio Cleaner starts" in Settings (on by
default), About shows "Updates: on/off", and the EULA and the About text say what is sent: a normal web request to our
download server (the IP address, and the app version in the User-Agent), nothing else.

**What the app does.** It fetches `https://mc-multi-world.space/audio/api/downloads/desktop/update.json`
(`config.UPDATES_URL`) and `update.json.sig` (small, 20 s timeout, https only, the system certificate store), verifies the signature with the
public key built into the app, and only then reads the manifest. It ignores a manifest whose version is not newer than
its own (versions compare as numbers, so 2.0.15 is newer than 2.0.9), picks the file of its platform
(`windows-x86_64`, `macos-arm64`, `macos-x86_64`) and downloads that installer into the app's data folder
(`%LOCALAPPDATA%\Audio Cleaner\Audio Cleaner\updates`, `~/Library/Application Support/Audio Cleaner/Audio Cleaner/updates`)
with the model downloader's code (`models.fetch_file`: https only, never a redirect to http, resumable, size and SHA-256
checked, free space checked first). The download address is our server's folder plus the file name
(`.../downloads/desktop/<file>`), not taken from the manifest. Only the newest download is kept;
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
installers, keys sorted; no release page, because the file is published on our server and the app ignores one; apps
from before the download server insisted on a `notes_url` pointing at GitHub, so while such apps still matter add
`--notes-url https://github.com/<owner>/<repo>/releases/tag/desktop-v<version>` to the call in `desktop.yml`, which
also puts the repository name into the mirrored file), signs it, checks the signature against the key's public half and uploads both files with the
installers, for automatic and tagged releases alike. It fails closed: **without the secret, or with a key that is not
the one built into the app, the release gets no `update.json`** (a warning or error annotation shows it) and no installed
app updates itself; the installers are still published. The public key is `PUBLIC_KEY` in `updater.py` (raw Ed25519,
base64). Signature checking uses the `cryptography` package, pinned in `requirements.lock` (48.0.1, the newest release
that still ships a macOS Intel wheel; later releases dropped it) with `cffi` and `pycparser`; the frozen
`--selftest` makes a key pair, signs and verifies to prove the bundled copy works.

**Checking that updates are live.** After a build: the release page has `update.json` and `update.json.sig`, and once
the mirror has run (every 10 minutes) `curl -sL https://mc-multi-world.space/audio/api/downloads/desktop/update.json` shows the version and
hashes; `openssl pkeyutl -verify -rawin -pubin -inkey PUB.pem -in update.json -sigfile update.json.sig` says
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

**If the secret leaks.** The key alone does not let anyone push a file to the apps: the manifest only names a file (the app builds the address itself, on our download server, and downloads it over https),
so an attacker also needs to get a file onto the mirror, which only copies verified releases of the repository. Treat a leak as serious anyway: 1. delete the secret at once (a
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
   `df.dll` in `$OUT` (default `~/audio-cleaner/src/deepfilter-build/out`); the checkout also holds
   `models/DeepFilterNet3_onnx.tar.gz`.
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
   python desktop/packaging/prepare_vendor.py --libdf PATH/libdf.so --ffmpeg-dir FFMPEG_OUT --model PATH/DeepFilterNet3_onnx.tar.gz \
       [--whisper-model DIR | --download-whisper] [--deepfilter-src DIR] --out desktop/vendor
   ```
   Copies the native files, checks the DFN3 and DPDFNet models (sha256 pinned), puts a small test-only speech model in
   `vendor/test-models/whisper-small/` (`Systran/faster-whisper-small`, revision and file hashes pinned in the script;
   `--download-whisper` fetches it; it is not bundled) and
   writes `licenses/`: FFmpeg/LAME/opus/libvpx, DeepFilterNet, DPDFNet, the Rust crates of libDF (`cargo tree` of the capi
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

The installed app updates itself (see "Automatic updates"); by hand, just run the new setup (the .exe or the .pkg) over
the installed app. There is nothing to uninstall first; it is the same install, done again, and the models are not
downloaded again. The app's silent update is this same setup with `/VERYSILENT`.

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

## Public copy of the desktop code

The desktop app lives here, next to `common/` (shared with the website's server). People still use the old public
repository (`BenjaminDanker/Audio-Cleaner`, once a different app called VideoDenoiser, releases v1.0.0 and v1.0.1), so
the app's code and each release are copied there automatically, one way, without mentioning this repository.

* **What is copied** (`ALLOWLIST` in `tools/public-desktop/sync.py`): `desktop/` (app, packaging scripts, the licence
  texts and EULA the installers ship, tests), all of `common/` (the app imports bwe, captions, clarity, dfn, dpdfnet,
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
  (`AudioCleaner-<version>-Windows.exe`, `-Mac-Apple-silicon.pkg`, `-Mac-Intel.pkg`) and short notes that point to the
  download page. No `update.json`, no signature: those are for our download server, and the app trusts only the
  signed manifest it serves, so a release there is a download for people, never an update source. A rerun updates the
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
(https://mc-multi-world.space/audio/). Run "Public desktop copy" once with "Run workflow" on `on-prem` and look at the
commit it pushed (it also removes the old build workflow) before announcing anything; the first release is copied by the
next release of the app. To stop the copies, delete the secrets or the workflows.

## Owner to-do before the first release

* Upload the `models-v1` release (the single `AudioCleaner-models-v1.zip`, see above), let the mirror pick it up and
  check that the "Model downloads exist" job is green (it checks `models/AudioCleaner-models-v1.zip` on our server).

* Set the Apple and Azure secrets if you want signed installers (optional).
* Add the secret `UPDATE_SIGNING_KEY` (the contents of `~/audio-cleaner/secrets/update-signing-key.pem`) so releases get
  a signed `update.json`; without it nothing updates itself (see "Automatic updates").
* Push, run the workflow once via "Run workflow", and download the three artifacts to test on real machines. The
  Windows and macOS jobs have not been exercised yet. On real machines also check the updates: install an older
  build, publish a newer one, and see (Windows, per user and for all users) the banner, "Restart now" (the new version
  starts, not elevated), closing the app instead (silent install, no window, no console flash, the app starts normally
  next time), a declined administrator prompt, a running batch; (macOS, both architectures) the "Install" button
  opens the installer. Check the Windows install log if `/RELAUNCH=1` or `{param:...}` misbehaves.
