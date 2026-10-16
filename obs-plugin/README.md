# Audio Cleaner (cloud) - OBS Studio plugin

An audio **filter** for OBS Studio that streams a source (usually your microphone) to the
Audio Cleaner live service, returns the denoised audio and optionally produces live captions,
which can be embedded in your stream as CEA-608/708 closed captions.

Protocol: [`docs/streaming-protocol.md`](../docs/streaming-protocol.md) (v1). Licence: GPL-2.0-or-later.
Built against libobs 31.1.1; works with OBS 31 and newer (OBS refuses plugins built for a *newer* major version).

## Install

Download the package for your platform from the Live page of the website (it comes from our own download server,
`https://mc-multi-world.space/audio/api/downloads/obs-plugin/`), and unpack. For maintainers: CI publishes every build
as the `obs-plugin-latest` pre-release, and the server mirrors it (docs/operations.md, "Release downloads").
OBS must be **version 31 or newer**; restart OBS afterwards.

Every build has its own version number, `MAJOR.MINOR.<build number>` (for example 0.1.57; the release is titled
"OBS plugin 0.1.57", and the files carry the number in their names). A `obs-plugin-vX.Y.Z` tag builds exactly X.Y.Z.
The plugin tells you when a newer one exists and installs it for you (see "Updates" below).

| Platform | What to do |
|---|---|
| Windows | Run `...-windows-setup.exe` with OBS closed (no admin prompt; it is not code-signed, so SmartScreen asks: More info, Run anyway). By hand instead: unzip `...-windows.zip` (one folder, `audio-cleaner`) and put that folder into `C:\ProgramData\obs-studio\plugins\`, creating `plugins` if it is not there, so the DLL ends up in `C:\ProgramData\obs-studio\plugins\audio-cleaner\bin\64bit\`. Not `Program Files\obs-studio\obs-plugins`: OBS loads only single DLLs from there, not plugin folders. |
| macOS | Unpack the `...-mac.tar.xz` and copy `audio-cleaner.plugin` into `~/Library/Application Support/obs-studio/plugins/`. The plugin is not signed yet, so macOS blocks a downloaded copy: run `xattr -dr com.apple.quarantine ~/Library/Application\ Support/obs-studio/plugins/audio-cleaner.plugin` once. |
| Ubuntu, Debian, Linux Mint | `sudo apt install ./audio-cleaner-obs-plugin-<version>-ubuntu-debian.deb`. It needs `obs-studio >= 31` and OpenSSL 3. The Ubuntu repositories only carry older OBS versions (24.04 has 30.x): get a current one from the OBS PPA (`sudo add-apt-repository ppa:obsproject/obs-studio`). |
| Other Linux | Unpack the `...-linux-other.tar.xz` into the OBS plugin folder: `mkdir -p ~/.config/obs-studio/plugins && tar -xf <file> -C ~/.config/obs-studio/plugins`. That creates `audio-cleaner/bin/64bit/audio-cleaner.so` and `audio-cleaner/data/` (the texts; without them the filter shows raw names). For the Flatpak version of OBS use `~/.var/app/com.obsproject.Studio/config/obs-studio/plugins/` instead. |

## Use

Right-click an audio source > Filters > `+` > **Audio Cleaner (cloud)**.

* **API key** - create a dedicated key on the website with *OBS plugin and live* ticked.
* **Remove background noise** and **How much noise to remove** - Gentle / Balanced / Strong / Maximum (attenuation limit
  12 / 24 / 40 / 100 dB, the same levels as the website and the desktop app; changeable while streaming).
* **Echo cancellation** - the source that plays through your speakers, default *Desktop Audio*; *None* turns it off.
  The filter listens to that source and sends its sound along with the microphone, so the service can remove the speakers'
  sound from the microphone (a game or a call playing twice). Not needed with headphones (choose *None*: it saves about
  65 kbps of upload). It is included in the price. It keeps working when that source is **muted in the OBS mixer**, because
  OBS hands the filter the sound before the mute applies and the computer still plays it. If the source does not exist yet
  (OBS still loading) the filter keeps looking for it, and the status says "waiting for the speaker source".
  On macOS the *Desktop Audio* channel is normally empty: add a *macOS Audio Capture* source and pick it by name here.
  With *Remove background noise* off only the captions are echo-free, the audio itself is not altered.
* **Captions** - *Off*, *Standard (English)* or *Extra accurate (any language)*; **Caption language** (extra accurate only: a
  language or *Detect automatically*; standard captions are always English); **Send captions to the stream** (on by default).
  Captions go to the stream (and to a recording) as closed captions, final captions only, split into lines of at most 32
  characters, paced, with stale ones dropped; only while streaming or recording. Nothing else in OBS shows them, so the filter
  only asks for (and bills) captions when this option is on. The closed-caption format only shows Latin letters: Russian,
  Ukrainian, Arabic, Hindi, Japanese, Korean and Chinese captions cannot be added to the stream (the status says so and
  they are not requested); with *Detect automatically* captions in such scripts are skipped. Accents and the usual
  punctuation are kept; letters the format lacks (Polish ł, Czech č, Turkish ş ...) are written with plain ones.
  With a stream delay set in OBS (Settings, Advanced, Stream Delay), captions for the stream wait that long too, so they
  stay with the words; the recording is not delayed and gets them at once. Twitch viewers switch captions on with the CC
  button; on YouTube choose *Embedded 608/708* captions for the stream in YouTube Studio first. Both show one caption
  track. Outputs made by other plugins (multistream, vertical canvas, Source Record) get no captions.
* **Delay** - Low 80 ms, Balanced 150 ms, Safe 250 ms. Fixed for the whole session; changing it restarts the session. See below.
* **Check for updates when OBS starts** (bottom of the dialog, on by default) - see "Update notice" below. It is one
  setting for the whole plugin, not per filter.
* **Status** (press *Refresh status*): plain words about what is going on, plus *Try again* when the service refused
  something only you can fix (see "When something is wrong"). **Advanced** holds the server address, the audio format
  (compact Opus by default, ~60 kbps each way; uncompressed 16-bit PCM, ~770 kbps, as a fallback) and a technical line
  (round trip, frame counters).

The session starts when the enabled filter first receives audio and stops when the filter is disabled or removed.
With denoise off (captions only) the audio is not delayed at all.

**Each filter is its own live session, billed per second while it runs** (captions too, whether or not you are live), and
an account can have **two live sessions at a time**: a third filter, or a second computer, is refused until one ends
(a session whose connection dropped lingers on the service for up to about 30 s).

### Updates

Once each time OBS starts (a few seconds after the plugin loads, on a thread of its own, never on the audio path and never
holding up OBS) the plugin reads the newest version's manifest from our download server,
`https://mc-multi-world.space/audio/api/downloads/obs-plugin/manifest.json`, and its signature (`manifest.json.sig`).
The manifest is signed when it is built (Ed25519); the plugin checks the signature with the public key built into it and
ignores anything that is not signed by that key, or not newer than itself. https only, certificates and host names
verified, small size limits, a few seconds of patience. If the release is newer:

* the filter's properties start with "Update available: Audio Cleaner plugin 0.1.57 (you have 0.1.50)" and a
  **Download and install the update** button; the OBS log gets one line;
* the button downloads this computer's package from the same server, checks its size and SHA-512 against the signed
  manifest, and installs it:
  * **Windows** (a plugin folder with `bin\64bit\audio-cleaner.dll`): a small PowerShell script waits for OBS to close,
    then puts the new files in place (with one Windows admin prompt if the plugin folder needs it, e.g. Program Files);
    the new version is in use the next time OBS starts;
  * **macOS** (`audio-cleaner.plugin` in your user's plugins folder) and **Linux plugin folders**: the same, with an sh
    script; nothing needs admin rights; in the Flatpak OBS it is swapped at once and a restart of OBS uses it;
  * **Ubuntu/Debian package** (`/usr/...`): the `.deb` is opened with your system's package installer;
  * anywhere else: the folder with the new package is opened and the settings say where it goes.
  If anything fails, the old plugin stays in place and keeps working; the settings say why and offer "Try again".

The checkbox **Check for updates when OBS starts** is stored in OBS's plugin settings folder
(`.../obs-studio/plugin_config/audio-cleaner/settings.json`). With it off, no request is made.

### When something is wrong

The status line says it in plain words; the OBS log has the technical reason. The filter never interrupts your audio:
it keeps playing the (delayed) original.

| Status | Meaning | What happens |
|---|---|---|
| "Your connection is too slow for the Balanced (150 ms) delay setting ..." | most processed audio arrives after its deadline | choose a longer delay |
| not enough balance / balance ran out | the account cannot pay for live processing | stops trying; add balance, then press *Try again* |
| key not accepted | HTTP 401 | stops trying; fix the key |
| "The server refused the connection ..." | HTTP 403: key without the live scope, suspended account, or too many connections lately | retries every few minutes |
| settings not accepted / terms not accepted | the service says so | stops trying; change the setting or accept the terms, then *Try again* |
| two live sessions already | account limit | retries slowly (30 s and more) |
| full / busy | the service has no free capacity | retries after a minute or more (a retry hint from the service is respected) |

### Latency and sync

The filter outputs the input delayed by a fixed D (the preset) and **back-dates the output timestamp by D**, so OBS
keeps audio and video in sync (OBS adds audio buffering automatically). *Monitoring* of that source is delayed by D.
If a processed frame misses its deadline, or the connection is down, the delayed *original* audio is played
(5 ms cross-fade), so the audio never stops. Rule of thumb for the round-trip budget: `D - 30 ms (model) - 6.5 ms (Opus) - 25 ms (framing)`
(and `- 9 ms` with echo cancellation, which delays the audio that much):
Low leaves ~12 ms with Opus (a server on your own network only; ~3 ms with echo cancellation), Balanced ~85 ms (~76), Safe ~185 ms (~176).
The cliff is sharp: beyond the budget every frame is late. The filter watches this (a few seconds of frames) and the status
tells you when the delay is too short. OBS at 44.1 kHz is resampled
(linear-phase polyphase, 44.1 <-> 48 kHz); other rates are passed through unchanged with a status warning.
The filter is mono: channels are averaged and the result is written to all channels.

The microphone is sent frame by frame for as long as OBS feeds the filter, silence and a source muted in OBS included
(only echo-reference frames of pure silence are skipped). The service ends a session that gets no microphone audio for a few
minutes, so keep it that way if you change the code. The only times the filter stops sending are: not connected, OBS has
stopped feeding it audio (the filter then closes the session itself after 10 s), and a congested connection (frames are
dropped instead of queued).

### Privacy and security

* With echo cancellation, the sound of the chosen speaker source leaves your machine too, with the same encryption and billing.
* Audio of the filtered source leaves your machine (encrypted with TLS) and is processed on the server; it is billed per second.
  Muting a source in OBS does not stop streaming - disable the filter.
* **OBS stores filter settings, including the API key, unencrypted in the scene collection JSON.** Use a dedicated key with only
  the live scope and do not share scene files. The plugin never logs the key.
* **Once per OBS start the plugin contacts our download server** to read the newest plugin version (see "Updates"); when
  you press the update button it downloads the package from there. The server sees your internet address and a
  `User-Agent` of the form `obs-audio-cleaner/<plugin version>`; no account, key, audio or other data is sent. Turn it off with **Check for updates when OBS starts** in the filter's properties (then
  no request is made at all).
* Certificates and host names are always verified. The only exception is an explicit `ws://` URL for local testing; it is
  accepted for this computer and the local network only (localhost, 127.x, 10.x, 172.16-31.x, 192.168.x, *.local) and
  shows a warning there.

## Build

Needs CMake >= 3.28. Third-party code (IXWebSocket, nlohmann/json, libopus, mbedTLS) is downloaded by CMake, pinned by SHA-256,
and linked statically and privately (it is never installed or packaged).

```
# build number: the version is buildspec.json's MAJOR.MINOR.PATCH; CI builds MAJOR.MINOR.<run number> by passing
# ACW_VERSION (cmake -P cmake/ci-version.cmake prints it; -DACW_VERSION=0.1.57 or the environment variable of that name)

# core library + tests only (no OBS needed)
cmake --preset core-tests && cmake --build --preset core-tests && ctest --test-dir build_tests

# plugin
cmake --preset ubuntu-x86_64 -DCMAKE_PREFIX_PATH=<prefix with libobs 31.1 + obs-frontend-api>   # Linux
cmake --preset windows-x64 && cmake --build --preset windows-x64                                  # downloads OBS 31.1.1 + obs-deps
cmake --preset macos       && cmake --build --preset macos
```

TLS: Linux uses system OpenSSL if its headers exist, else builds mbedTLS (CA bundle from `/etc/ssl/...`);
Windows builds mbedTLS (reads the *current user's* Windows root store: a company root certificate installed for the whole
machine only, as TLS-inspecting proxies do, is not seen and the connection fails); macOS uses SecureTransport (deprecated by
Apple, TLS 1.2 at most). Override with `-DACW_TLS_BACKEND=`. `-DACW_USE_OPUS=OFF` builds without Opus (PCM only).

Layout: `core/` OBS-independent library `acw_core` (SPSC rings, reframer, resampler, protocol, WebSocket thread, output stage,
echo reference buffer, caption pacer, updates: the signed manifest check (Monocypher Ed25519 + SHA-512), the download,
and the installers that swap the plugin when OBS closes); `src/` thin OBS glue (`update-notice.cpp` runs the check and,
on the button, the download and install); `tests/` unit tests, `core_loopback_test` (the real session against a
fake server on 127.0.0.1: handshake, errors, retries, late replies, garbage, shutdown; also the update check against a local
web server: newer / same / older, redirects, refused addresses, size limit, timeout, cancel), `core_stream_test` (live server) and
`obs_smoke` (headless libobs host; `core_loopback_test --serve` is a server for it).

CI: `.github/workflows/obs-plugin.yml`. Core tests run on Linux (also under ASan + UBSan and under TSan), macOS and Windows
(the last two informational); the Linux job builds the pinned libobs from source; every package is checked to hold the plugin
folder and nothing else. Actions are pinned by commit SHA.

### Signing (not implemented)

Placeholders in the workflow. macOS: Apple Developer ID Application certificate, `codesign --options runtime`, notarize with
`notarytool`, staple. Windows: Authenticode with `signtool` (or Trusted Signing) and a timestamp. Unsigned builds work but
trigger Gatekeeper / SmartScreen warnings.

## Licences

GPL-2.0-or-later, see `LICENSE.txt`. Third-party components: `THIRD_PARTY_LICENSES.txt`. Both files are installed with the plugin.
