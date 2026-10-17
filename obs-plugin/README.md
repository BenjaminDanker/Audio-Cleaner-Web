# Audio Cleaner (cloud) - OBS Studio plugin

An audio **filter** for OBS Studio that streams a source (usually your microphone) to the
Audio Cleaner live service, returns the denoised audio and optionally produces live captions,
which can be embedded in your stream as CEA-608/708 closed captions, translated into up to three more languages,
shown as on-screen text and saved as subtitle files next to your recordings.

Protocol: [`docs/streaming-protocol.md`](../docs/streaming-protocol.md) (v1). Licence: GPL-2.0-or-later.
Built against libobs 31.1.1; works with OBS 31 and newer (OBS refuses plugins built for a *newer* major version).

## Install

Download the package for your platform from the Live page of the website (it comes from our own download server,
`https://mc-multi-world.space/audio/api/downloads/obs-plugin/`), and unpack. For maintainers: CI publishes every build
as its own release per version (`obs-plugin-v<version>`), and the server mirrors it (docs/operations.md, "Release downloads").
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
  language or *Detect automatically*; standard captions are always English). Final captions only. They can go to four
  places, each optional (below). The filter only asks for (and bills) captions when one of the places uses them.
* **Translate captions into** (under *Translations and where captions appear*) - up to three languages (Spanish, French,
  German, Italian, Portuguese, Dutch, Polish, Russian, Ukrainian, Turkish, Arabic, Hindi, Japanese, Korean, Chinese,
  Swedish, Danish, Finnish, Norwegian, Czech, Greek, Hebrew, Indonesian, Vietnamese, English). The spoken language is not
  offered (with Standard captions that is English). The service translates each final caption a moment after it is spoken
  (usually one or two seconds) and sends it as a `translation` event; the start message carries `"translate": [...]` and the
  service's `ready` answer says which languages it switched on (the status says "Translating into Spanish and German", or
  that a language was not switched on, e.g. by an older service). Every language is billed per second while the filter runs,
  and is only asked for when a place shows it. If the service gives up on one language (a `translation_unavailable`
  warning) that language stops, the status says so, and the other captions go on.
* **Closed captions** - **Closed caption language** (*Same as spoken* or one of the translations; languages that need other
  alphabets cannot be shown in the stream's caption format and are not offered) and **Send closed captions to**: *The stream*
  (on), *The recording* (on), *The replay buffer* (off) and *Other outputs (multistream plugins)* (off). Closed captions carry
  one language, are split into lines of at most 32 characters, paced, with stale ones dropped, and only go to outputs that
  run. *Other outputs* looks at all of OBS's outputs (`obs_enum_outputs`) and captions each one that is running, takes encoded
  packets and has a video encoder, once, apart from the stream, recording and replay buffer which the boxes above decide; each
  output waits for its own stream delay (Settings, Advanced, Stream Delay) so the captions stay with the words, the recording
  gets them at once. The closed-caption format only shows Latin letters: Russian, Ukrainian, Arabic, Hindi, Japanese,
  Korean, Chinese, Greek, Hebrew and Vietnamese cannot be added to the stream (the status says so and they are not
  requested for that purpose); with *Detect automatically* captions in such scripts are skipped. Accents and the usual
  punctuation are kept; letters the format lacks (Polish ł, Czech č, Turkish ş ...) are written with plain ones.
  Twitch viewers switch captions on with the CC button; on YouTube choose *Embedded 608/708* captions for the stream in
  YouTube Studio first. Both show one caption track.
* **On-screen text** - for the spoken language and for each translation: **Create text source for <language>** makes an OBS
  text source called "Audio Cleaner captions (<language>)" (GDI+ text on Windows, FreeType 2 text elsewhere; large white
  bold text with an outline, wrapped in a 1600 pixel box at the bottom) and chooses it; **On-screen text for <language>**
  picks any existing text source instead. The source is **not added to any scene**: put it where you want it with
  *Sources, +, Add Existing* (OBS may forget a source that is in no scene when it closes or when you change the scene
  collection, so add it to one). The filter writes the last one to three lines into the source (**lines at a time**) at reading pace and clears
  it after **Clear on-screen text after silence** seconds. On-screen text is part of the picture, so it shows in both the
  stream and the recording; this is the way to show a translation in a language the closed captions cannot (the stream's
  caption format has one track and only Latin letters).
* **Save subtitle files with recordings** (on) - while OBS records, the captions of every language (the spoken one and the
  translations) are collected with times counted from the start of the recording (pauses are left out) and, when the
  recording stops, saved next to it: `<recording without extension>.<language>.srt`, for example `2026-10-16 12-00-00.es.srt`
  (the spoken language of Standard captions is `en`; with *Detect automatically* the spoken file has no language in the name:
  `2026-10-16 12-00-00.srt`). Players such as VLC pick them up by themselves. Times are the service's own, so they are as exact
  as the captions (within a fraction of a second of the video). With a recording split into several files only the last one
  gets the file.
* **Delay** - Low 80 ms, Balanced 150 ms, Safe 250 ms. Fixed for the whole session; changing it restarts the session. See below.
* **Check for updates when OBS starts** (bottom of the dialog, on by default) - see "Update notice" below. It is one
  setting for the whole plugin, not per filter.
* **Status** (press *Refresh status*): plain words about what is going on, plus *Try again* when the service refused
  something only you can fix (see "When something is wrong"). **Advanced** holds the server address, the audio format
  (compact Opus by default, ~60 kbps each way; uncompressed 16-bit PCM, ~770 kbps, as a fallback) and a technical line
  (round trip, frame counters).

**Only while streaming or recording** (on by default, also for filters saved by earlier versions) - the filter keeps no
session, and so nothing is billed, until OBS is streaming, recording, running the replay buffer or running the virtual
camera. The session starts as soon as one of them is *starting* (so processing is ready when the output begins) and ends
about 10 seconds after the last one has stopped (a stream that restarts, or a recording that stops and starts again, keeps
the same session). In between the microphone plays normally, unprocessed, and the status says "Waiting for you to go live
or record: nothing is billed until then." Outputs made by other plugins (a second stream, Source Record, NDI) are not
seen: untick the box when you need the filter for those, or to test it without streaming. Without the OBS frontend API
(a build with `ENABLE_FRONTEND_API=OFF`) the setting is not shown and the filter always runs.

With the box unticked, the session starts when the enabled filter first receives audio and stops when the filter is
disabled or removed. With denoise off (captions only) the audio is not delayed at all and the service is asked not to send
any audio back (`returnAudio: false` in the start message).

**Each filter is its own live session, billed per second while it runs** (captions too; with *Only while streaming or
recording* that means while an output runs, plus the 10 seconds after), and
an account can have **four live sessions at a time**: a fifth filter, or another computer, is refused until one ends
(a session whose connection dropped lingers on the service for up to about 30 s).

**Several people.** Add the filter to each person's microphone: every filter is cleaned and captioned on its own, and
all of them share one caption track (they take turns). Put each person's name in *Speaker name* so their lines start
with it. One stereo source with a person on each side (two mics on one audio interface): tick *Left and right are
different people*; each side then runs as its own session (billed as two) and stays on its side. Echo cancellation
applies to the left side only in that case.

**Two languages.** With extra accurate captions, *Also speaks* adds a second language you switch to while talking; the
captions follow the switches, and translations translate whichever was spoken.

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
it keeps playing the (delayed) original. Whatever the service does (unreachable, slow, ending the session, sending garbage)
the microphone passes through unprocessed and without gaps, the captions simply stop, and the status says why; nothing the
network side does can stop the audio thread (no lock, no wait, no allocation there; the callbacks that hand captions to OBS
run on the network thread and an exception in them is caught). `core_loopback_test` checks this sample for sample against
a service that refuses, stops the session or never answers.

| Status | Meaning | What happens |
|---|---|---|
| "Your connection is too slow for the Balanced (150 ms) delay setting ..." | most processed audio arrives after its deadline | choose a longer delay |
| not enough balance / balance ran out | the account cannot pay for live processing | stops trying; add balance, then press *Try again* |
| key not accepted | HTTP 401 | stops trying; fix the key |
| "The server refused the connection ..." | HTTP 403: key without the live scope, suspended account, or too many connections lately | retries every few minutes |
| settings not accepted / terms not accepted | the service says so | stops trying; change the setting or accept the terms, then *Try again* |
| two live sessions already | account limit | retries slowly (30 s and more) |
| full / busy | the service has no free capacity | retries after a minute or more (a retry hint from the service is respected) |
| "The service stopped this session because your microphone audio was not arriving steadily" | the service ends a session whose microphone audio arrives less than half the time (`no_audio`; also `idle_timeout`) | starts again at once (normal backoff, growing if it repeats) as soon as OBS feeds audio again |
| free trial over / first top-up needed | `trial_over`, `needs_topup` | stops trying; add balance on the website, then *Try again* |
| free-trial capacity busy | `trial_busy` | retries slowly (a minute or more) |
| "... sent too many messages" | `too_many_messages` / `rate_limited` | reconnects after 20 s |
| "The caption service stopped working" | the session was only captions and they broke (`captions_unavailable`) | retries after about five minutes |
| "... captions have stopped" / "Translation into Spanish stopped" | a warning: with noise removal on, the audio goes on; other languages go on | nothing to do |

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

The microphone is sent frame by frame for as long as OBS feeds the filter, silence and a source muted in OBS included.
The service ends a session that gets no microphone audio for a few minutes, so keep it that way if you change the code.
The only times the filter stops sending are: not connected, OBS has stopped feeding it audio (the filter then closes the
session itself after 10 s), and a congested connection (frames are dropped instead of queued).

**Silence is free.** When the service's `ready` message says `"silenceFrames": true`, a 20 ms block that is silence goes out
as an 18-byte *silence frame* (protocol codec 4: the usual 16-byte header plus the sample count at 48 kHz as a
little-endian u16) instead of an encoded one. Silence means: the filtered source is muted in OBS (`obs_source_muted`; the
filter sends zeros for a muted source, with any service, so a muted microphone is not captioned), or every sample of the
block is quieter than 1e-5 (digital silence, for example behind a noise gate). The same holds for the echo-reference
stream: a silent reference block is a silence frame (a service without them still gets nothing for it). The service
answers each microphone silence frame with a silence frame (same `seq` and `ts_us`), which the output stage plays as that
many zero samples under the usual deadline rules. The Opus encoder just skips those frames. Silence frames cost the service
nothing and are **not billed**; the frames keep coming in real time, so the session counts as alive. The details line shows
how many were sent.

The network thread sleeps until the next 20 ms frame is due (the audio thread only stores a few numbers; from the input
sample clock the network thread works out when the next frame is complete) instead of polling every 2 ms: about 50 wake-ups
a second instead of 470 per session.

### Privacy and security

* With echo cancellation, the sound of the chosen speaker source leaves your machine too, with the same encryption and billing.
* Audio of the filtered source leaves your machine (encrypted with TLS) and is processed on the server; it is billed per second.
  A muted source is sent as silence (not captioned, not billed where the service takes silence frames), but the session
  stays open (and billed, for the seconds of audio that are not silent) - disable the filter, or use *Only while streaming
  or recording*, to end it.
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
echo reference buffer, caption pacer, caption router (which language goes where), on-screen text logic, subtitle files (SRT),
choice of the outputs to caption, the live-only decision (`LiveGate`), language lists, updates: the signed manifest check (Monocypher Ed25519 + SHA-512), the download,
and the installers that swap the plugin when OBS closes); `src/` thin OBS glue (`audio-cleaner-filter.cpp` the filter and its
properties, `caption-output.cpp` the closed captions, `live-state.cpp` the stream / recording / replay buffer / virtual camera events, `text-sources.cpp` the on-screen text sources, `subtitle-files.cpp` the
subtitle files, `update-notice.cpp` runs the check and, on the button, the download and install); `tests/` unit tests, `core_loopback_test` (the real session against a
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
