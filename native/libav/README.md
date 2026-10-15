# Custom libav.js build: LGPL-only, audio-decode-only

A custom variant (`acw-audio`) of [libav.js](https://github.com/Yahweasel/libav.js)
(FFmpeg compiled to WebAssembly) for decoding many audio formats to PCM in the browser,
for lazy-loading only when a file needs it. Background and alternatives:
`docs/research/browser-decoding.md`.

Properties:

- **LGPL-2.1+ only.** FFmpeg is configured without `--enable-gpl`, `--enable-nonfree`,
  `--enable-version3`, and with no external codec libraries (only zlib, which is permissive).
- **Non-threaded.** Only libav.js's plain `wasm` target is built. No SharedArrayBuffer, no
  COOP/COEP headers, no pthreads. The wasm glue has zero `SharedArrayBuffer` and zero
  `eval`/`new Function` references, so it works under `script-src 'self' 'wasm-unsafe-eval'`.
  (The small loader `libav-acw-audio.mjs` contains one `typeof SharedArrayBuffer` feature
  test, used to decide whether threads are possible; threads are never requested here.)
- **Decode only.** No encoders, no muxers, no video decoders (video streams are demuxed
  and ignored), no hwaccels, no network protocols.

## What is in it

| | |
|---|---|
| Demuxers | asf, avi, flv, mpeg-ps (`mpegps`), mpeg-ts, au, caf, aiff, mov/mp4/3gp, matroska/webm, ogg, wav, w64, wavpack (`wv`), ape, tta, mpc, mpc8, amr, amrnb, amrwb, ac3, eac3, dts, dtshd, mp3, aac (adts), flac, plus `mpegvideo` (probe only, see below) |
| Audio decoders | wmav1, wmav2, wmapro, wmalossless, ac3, eac3, dca (DTS), ape, tta, wavpack, mpc7, mpc8, amrnb, amrwb, mp1, mp2, mp3 (float versions), aac, aac_latm, vorbis, opus, flac, alac, all `pcm_*`, all `adpcm_*` that an LGPL build allows |
| Parsers | aac, aac_latm, ac3, amr, dca, flac, mpegaudio, mpegvideo, opus, vorbis |
| Filters | `aresample` (swresample: sample-rate and channel conversion), `aformat`, `anull`, `pan` |
| Libraries | libavformat, libavcodec, libavfilter, libavutil, libswresample, zlib |

The exact list is `fragments` in `build.sh`; the generated FFmpeg configure flags are
saved to `$LIBAV_BUILD_DIR/variant-acw-audio.config.json` (fragment list) and
`libav.js-src/configs/configs/acw-audio/ffmpeg-config.txt`.

Things worth knowing:

- **Resampling is reached through a filter graph.** libav.js has no direct `swr_*` API; use
  `ff_init_filter_graph("aresample=16000,aformat=sample_fmts=flt:channel_layouts=mono", ...)`
  (see the resample section of `test/test-decode.mjs`). Frames from containers that do not
  specify a channel layout (for example ASF/WMA) come back with `channel_layout` 0: set it
  (mono = 4, stereo = 3) before feeding them to a filter graph.
- **`mpegvideo` is there on purpose.** Without it FFmpeg's stream prober mislabels the video
  stream inside an MPEG program stream (`.mpg`) as MP3 audio. With it, the stream is typed as
  video, has no decoder, and is skipped.
- **libav.js's own `demuxer-mpeg` fragment is broken at this tag.** It passes
  `--enable-demuxer=mpeg`, which FFmpeg's configure silently ignores (the component is
  `mpegps`). `build.sh` uses `mpegps` and verifies that every requested component is really
  enabled in the resulting `config_components.h`, so a silently ignored name fails the build.
- **Eight obscure ADPCM game-format decoders are excluded**: `adpcm_circus`, `ima_escape`,
  `ima_hvqm2`, `ima_hvqm4`, `ima_magix`, `ima_pda`, `n64`, `psxc`. FFmpeg 9.0 only builds
  them with `--enable-gpl` (`lgpl_gpl` dependency), so they cannot be part of an LGPL build.
- **Native Opus and Vorbis decoders** are used (not libopus/libvorbis), to avoid extra
  third-party libraries.
- DTS-HD MA: the native `dca` decoder decodes the DTS core; behaviour on the lossless
  extension is untested. Pre-SV7 Musepack is unsupported by FFmpeg. DRM'd files cannot be
  decoded.

## Pinned versions

All pins are in `versions.env`; the build verifies the commit/checksums.

| Component | Pin |
|---|---|
| libav.js | tag `v6.10.9.0`, commit `c80e885c3461f7bb7ea565c9631b34243ae0dbf1` |
| FFmpeg | 9.0 (what that libav.js tag builds); tarball sha256 in `versions.env` |
| Emscripten SDK | 6.0.5 (the version libav.js's CI uses at that tag), emsdk repo commit `8f685c26ee9f672081e61e22c6aa60fa972d00a9`, bundled node 24.19.0 |
| zlib | 1.3.2 (only used for compressed mov/matroska headers) |
| emfiberthreads | 1.3 (libav.js's own fiber shim, BSD-style licence) |
| Optimisation | libav.js default `-Oz` (override with `OPTFLAGS=-O2 ./build.sh`) |

## Rebuild

Needs: bash, git, curl, make, python3, node/npm (for libav.js's helper scripts; the build
uses emsdk's bundled node), about 3 GB of disk, and network on the first run.

```
native/libav/build.sh                  # emsdk + libav.js + variant + verify; ~2 min on 16 cores
native/libav/build.sh verify           # licence + size checks only
FORCE=1 native/libav/build.sh          # discard the variant's build products and rebuild
native/libav/test/make-fixtures.sh     # generate test files (uses ~/audio-cleaner/bin/ffmpeg)
node native/libav/test/test-decode.mjs # decode all fixtures, compare with native ffmpeg
```

Environment: `LIBAV_BUILD_DIR` (default `~/audio-cleaner/src/libav-build`), `EMSDK_DIR`
(default `~/audio-cleaner/src/emsdk`), `JOBS`, `OPTFLAGS`, `RATE` (curl rate limit,
default `40M`, applied to every download including emsdk's through a `.curlrc`).

The script is idempotent: a re-run with nothing changed takes about 2 seconds. Nothing is
installed outside `LIBAV_BUILD_DIR` and `EMSDK_DIR`, and nothing is added to git.

Outputs (`$LIBAV_BUILD_DIR/dist/`, not in git):

- `libav-acw-audio.mjs` / `.js`: loader (the file the page imports). ES-module and classic flavours.
- `libav-6.10.9.0-acw-audio.wasm.mjs` / `.wasm.js`: Emscripten glue.
- `libav-6.10.9.0-acw-audio.wasm.wasm`: the WebAssembly module.
- `libav.types.d.ts`: TypeScript types.

Serve all of them from the site's own origin next to each other; the loader finds the glue
and wasm by name relative to its own URL (or pass `base`/`wasmurl` options). Also written:
`$LIBAV_BUILD_DIR/sources/` (see below) and `$LIBAV_BUILD_DIR/SIZES.txt`.

## Sizes (measured, `-Oz`)

| File | Raw | gzip -9 |
|---|---:|---:|
| `libav-acw-audio.mjs` (loader) | 28,365 | 6,661 |
| `libav-6.10.9.0-acw-audio.wasm.mjs` (glue) | 285,321 | 58,983 |
| `libav-6.10.9.0-acw-audio.wasm.wasm` | 2,473,958 | 1,119,156 |
| **ES-module total a browser loads** | **2,787,644** | **1,184,800** (about 1.13 MiB) |
| `libav-acw-audio.js` + `.wasm.js` (classic flavour, same wasm) | 23,559 + 285,328 | 5,847 + 58,983 |

Most of the glue is the licence header that libav.js prepends (FFmpeg LGPL text, etc.).
Compare: libav.js's stock `default` variant is about 751 KB gzipped.

## Tests

`test/test-decode.mjs` loads the built libav.js in Node (single thread, wasm), decodes the
first audio stream of every fixture to float PCM, and compares sample-for-sample (SNR) and
duration with native ffmpeg's decode of the same stream, and times the decode loop.
Result on this machine (Node 18, one thread, wasm, `-Oz`), 120 s stereo 48 kHz fixtures,
best of 3, decode only (file already in memory), x = multiple of real time:

| Format (codec) | x real time | Format (codec) | x real time |
|---|---:|---|---:|
| `.wma` (wmav2) | 451 | `.wv` (WavPack) | 161 |
| `.wmv` (wmav2 + video, video skipped) | 437 | `.tta` | 226 |
| `.avi` (mp3 + mpeg4 video) | 345 | `.au` (pcm) | 1547 |
| `.flv` (mp3, 44.1 kHz) | 355 | `.caf` (pcm / alac) | 888 / 313 |
| `.mpg` (mp2; also with mpeg1 video) | 380 / 364 | `.aiff` | 968 |
| `.ac3` | 340 | `.amr` / `.3gp` (AMR-NB, 8 kHz) | 774 / 742 |
| `.eac3` | 351 | `.dts` | 188 |
| `.mp3` | 341 | `.m4a` (aac / alac), `.aac` | 365 / 294, 366 |
| `.mp4` (aac + video) | 404 | `.ts` (aac + mpeg2 video / mp2) | 363 / 344 |
| `.mkv` / `.ogg` (opus) | 251 / 250 | `.ogg` (vorbis) | 364 |
| `.flac` | 625 | `.wav` (u8 / f32 / adpcm_ms / adpcm_ima), `.w64` | 2070 / 1256 / 643 / 722, 1688 |

All 33 fixtures decode and match native ffmpeg (bit-exact for lossless and PCM formats;
SNR above 69 dB for lossy ones, the lowest being the MP2 files, where the native ffmpeg uses
a different MP2 decoder variant; all others above 110 dB). Resampling check
(`aresample=16000`, stereo to mono, decode plus resample): 234x real time. Wasm
instantiation takes about 80 ms. Browser speed will differ from Node/V8 on this machine.

Not testable here: `.ape` and `.mpc` (ffmpeg has no encoder for them; the demuxers and
decoders are compiled in and verified as enabled, but never run on a real file), WMA Pro,
WMA Lossless, AMR-WB (no encoder available).

## Licence notes and source offer

The wasm is FFmpeg (LGPL-2.1-or-later) + zlib + the libav.js wrapper (0-clause BSD) +
emfiberthreads + the Emscripten runtime. `build.sh` records and checks: `FFMPEG_LICENSE` is
"LGPL version 2.1 or later" and `CONFIG_GPL`, `CONFIG_GPLV3`, `CONFIG_NONFREE`,
`CONFIG_VERSION3` are all 0 in the generated `config.h`; no `--enable-gpl|nonfree|version3`
or external-library flag is present; no encoders or muxers are enabled; the glue carries the
LGPL notice header. libav.js itself says redistributors must provide the source of libav.js
and all its dependencies.

To distribute this build, serve the files from your own origin (the wasm is a separate file
loaded at run time, so a user can substitute their own build), include the LGPL notices
(they are in the glue header; also link them on a licence page), and offer the corresponding
source. Exact source (also collected, with checksums, by `build.sh` in
`$LIBAV_BUILD_DIR/sources/`):

- FFmpeg 9.0: <https://ffmpeg.org/releases/ffmpeg-9.0.tar.xz>
  (sha256 `7f607a00dd0d28a729d5a4811205812eef01cf6ef6155025febb6f36a9062d52`)
- libav.js v6.10.9.0 (includes the FFmpeg patches in `patches/ffmpeg/`, the config
  fragments and the build makefiles):
  <https://github.com/Yahweasel/libav.js/archive/refs/tags/v6.10.9.0.tar.gz>
  (commit `c80e885c3461f7bb7ea565c9631b34243ae0dbf1`)
- zlib 1.3.2: <https://www.zlib.net/fossils/zlib-1.3.2.tar.gz>
- emfiberthreads 1.3: <https://github.com/Yahweasel/emfiberthreads/archive/refs/tags/v1.3.tar.gz>
- Build recipe for this variant: this directory (`build.sh`, `versions.env`); the bundle
  `sources/acw-libav-build-scripts.tar.gz` contains it.
- Toolchain: Emscripten 6.0.5, <https://github.com/emscripten-core/emsdk> at the commit above.

Patent encumbrance of some of these codecs (AC-3, E-AC-3, DTS, WMA, AMR, MPEG audio) is a
separate question from copyright licensing; get it reviewed before shipping.
