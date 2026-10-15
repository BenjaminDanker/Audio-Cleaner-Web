# Audio Cleaner (cloud) - OBS Studio plugin

An audio **filter** for OBS Studio that streams a source (usually your microphone) to the
Audio Cleaner live service, returns the denoised audio and optionally produces live captions,
which can be embedded in your stream as CEA-608/708 closed captions.

Protocol: [`docs/streaming-protocol.md`](../docs/streaming-protocol.md) (v1). Licence: GPL-2.0-or-later.
Built against libobs 31.1.1; works with OBS 31 and newer (OBS refuses plugins built for a *newer* major version).

## Install

Download the package for your platform from the newest build,
https://github.com/BenjaminDanker/Audio-Cleaner-Web/releases/tag/obs-plugin-latest (every push that changes
`obs-plugin/` replaces it; an `obs-plugin-v<version>` tag publishes a permanent release), and unpack:

| Platform | Location |
|---|---|
| Windows (OBS 31/32) | `C:\ProgramData\obs-studio\plugins\audio-cleaner\` (`bin\64bit\audio-cleaner.dll`) |
| Windows (OBS 33+) | OBS 33 moves plugins to `C:\ProgramData\obs-studio\plugins\<name>\`; keep the same layout there |
| macOS | `~/Library/Application Support/obs-studio/plugins/audio-cleaner.plugin` |
| Linux | `.deb` (needs `obs-studio >= 31`, libssl3) or copy `lib/x86_64-linux-gnu/obs-plugins/audio-cleaner.so` to `~/.config/obs-studio/plugins/audio-cleaner/bin/64bit/` |

## Use

Right-click an audio source > Filters > `+` > **Audio Cleaner (cloud)**.

* **API key** - create a dedicated key with only the `stream` scope on the website.
* **Server URL** - default `wss://mc-multi-world.space/audio/api/stream/ws`.
* **Remove background noise** (tier "light"), **Noise reduction strength** (attenuation limit 18 / 40 / 100 dB, changeable live).
* **Echo cancellation** - the source that plays through your speakers, default *Desktop Audio*; *None* turns it off.
  The filter listens to that source and sends its sound along with the microphone, so the service can remove the speakers'
  sound from the microphone (a game or a call playing twice). Not needed with headphones (choose *None*: it saves about
  65 kbps of upload). It is included in the price. A muted source counts as silent; if the source does not exist yet
  (OBS still loading) the filter keeps looking for it, and the status line says "waiting for the speaker source".
  With *Remove background noise* off only the captions are echo-free, the audio itself is not altered.
* **Captions** - off / standard (local model) / premium, **language** (`auto` or an ISO code), **Send captions to the stream**
  (final captions only, split into <= 32 character lines, paced, stale ones dropped; only while streaming).
* **Latency preset** - Low 80 ms, Balanced 150 ms, Safe 250 ms. Fixed for the whole session; changing it restarts the session.
* **Codec** - Opus (default, ~60 kbps each way) or PCM 16-bit (~770 kbps, fallback).
* **Status** (press *Refresh status*): connected, RTT, delay, frames ok / late / passthrough / dropped, reconnects, low-balance warnings.

The session starts when the enabled filter first receives audio and stops when the filter is disabled or removed.
With denoise off (captions only) the audio is not delayed at all.

### Latency and sync

The filter outputs the input delayed by a fixed D (the preset) and **back-dates the output timestamp by D**, so OBS
keeps audio and video in sync (OBS adds audio buffering automatically). *Monitoring* of that source is delayed by D.
If a processed frame misses its deadline, or the connection is down, the delayed *original* audio is played
(5 ms cross-fade), so the audio never stops. Rule of thumb for the RTT budget: `D - 30 ms (model) - 6.5 ms (Opus) - 25 ms (framing)`
(and `- 9 ms` with echo cancellation, which delays the audio that much):
Low leaves ~12 ms with Opus (LAN/nearby server only; ~3 ms with echo cancellation), Balanced ~85 ms (~76), Safe ~185 ms (~176). OBS at 44.1 kHz is resampled
(linear-phase polyphase, 44.1 <-> 48 kHz); other rates are passed through unchanged with a status warning.
The filter is mono: channels are averaged and the result is written to all channels.

### Privacy and security

* With echo cancellation, the sound of the chosen speaker source leaves your machine too, with the same encryption and billing.
* Audio of the filtered source leaves your machine (encrypted with TLS) and is processed on the server; it is billed per second.
  Muting a source in OBS does not stop streaming - disable the filter.
* **OBS stores filter settings, including the API key, unencrypted in the scene collection JSON.** Use a dedicated key with only
  the `stream` scope and do not share scene files. The plugin never logs the key.
* Certificates and host names are always verified. The only exception is an explicit `ws://` URL for local testing, which shows a warning.

## Build

Needs CMake >= 3.28. Third-party code (IXWebSocket, nlohmann/json, libopus, mbedTLS) is downloaded by CMake, pinned by SHA-256.

```
# core library + tests only (no OBS needed)
cmake --preset core-tests && cmake --build --preset core-tests && ctest --test-dir build_tests

# plugin
cmake --preset ubuntu-x86_64 -DCMAKE_PREFIX_PATH=<prefix with libobs 31.1 + obs-frontend-api>   # Linux
cmake --preset windows-x64 && cmake --build --preset windows-x64                                  # downloads OBS 31.1.1 + obs-deps
cmake --preset macos       && cmake --build --preset macos
```

TLS: Linux uses system OpenSSL if its headers exist, else builds mbedTLS (CA bundle from `/etc/ssl/...`);
Windows builds mbedTLS (reads the Windows root store); macOS uses SecureTransport. Override with `-DACW_TLS_BACKEND=`.
`-DACW_USE_OPUS=OFF` builds without Opus (PCM only).

Layout: `core/` OBS-independent library `acw_core` (SPSC rings, reframer, resampler, protocol, WebSocket thread, output stage,
echo reference buffer, caption pacer); `src/` thin OBS glue; `tests/` unit tests, `core_stream_test` (live server) and `obs_smoke` (headless libobs host).
CI: `.github-workflows/build.yml` (move to `.github/workflows/`); the Linux job builds the pinned libobs from source.

### Signing (not implemented)

Placeholders in the workflow. macOS: Apple Developer ID Application certificate, `codesign --options runtime`, notarize with
`notarytool`, staple. Windows: Authenticode with `signtool` (or Trusted Signing) and a timestamp. Unsigned builds work but
trigger Gatekeeper / SmartScreen warnings.

## Licences

GPL-2.0-or-later, see `LICENSE`. Third-party components: `THIRD_PARTY_LICENSES.md`.
