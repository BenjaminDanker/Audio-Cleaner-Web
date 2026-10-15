# Browser-side decoding for extra audio/video formats (research)

Date: 2026-10-15. Scope: research only, no code changes. Goal: decode the target formats in the browser to raw PCM for an in-browser noise remover, with the site's own code closed-source, a strict CSP (`script-src 'self' 'wasm-unsafe-eval'`, no CDN), and wasm lazy-loaded only when a file needs it.

Sizes are gzip sizes of the files a browser downloads. Where "measured" is stated, I downloaded the npm tarball, decoded/extracted the wasm and gzipped it with `gzip -9`. Everything else is marked as an estimate or not measured.

## 1. Target formats and what they need

Target list (17 items): WMA/WMV audio (ASF), AVI, FLV, MPEG-PS, MPEG-TS, MPEG-1/2 audio (MP1/MP2), AC-3, AMR, AIFF, CAF, WavPack, APE, TTA, DTS, Musepack, AU, 3GP.

Ground truth on the codec side:

- FFmpeg has native (in-tree, LGPL-2.1+) decoders for almost all of them: `wmav1/2/pro/lossless`, `ac3`, `eac3`, `dca` (DTS), `ape`, `tta`, `wavpack`, `mpc7/mpc8`, `amrnb/amrwb`. FFmpeg's general table lists these as native decoders (AC-3, E-AC-3, DTS, WMA v1/v2, TTA, WavPack; Musepack and APE are listed without an explicit "E" marker, so native is likely but not confirmed from that table). Source: https://ffmpeg.org/general.html
- The LGPL status of the native decoders is confirmed from source headers: `libavcodec/dcadec.c` (DTS) and `libavcodec/apedec.c` (APE) both state "FFmpeg is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License ... version 2.1 ... or (at your option) any later version." Sources: https://raw.githubusercontent.com/FFmpeg/FFmpeg/master/libavcodec/dcadec.c and https://raw.githubusercontent.com/FFmpeg/FFmpeg/master/libavcodec/apedec.c
- FFmpeg's own legal rule: the core is LGPL-2.1+, but "Compile FFmpeg without `--enable-gpl` and without `--enable-nonfree`" for an LGPL build. If GPL parts are enabled, "the GPL applies to all of FFmpeg." Source: https://ffmpeg.org/legal.html

So: no target format requires GPL code if FFmpeg is built without `--enable-gpl`. The problem is packaging, not capability.

## 2. Candidate comparison

| Option | Licence (verified) | Target formats covered (of 17) | Download size (gzip) | Threads / SAB needed | Speed vs realtime | Maintenance (last release) | Custom minimal build |
|---|---|---|---|---|---|---|---|
| **libav.js, custom LGPL build** | LGPL-2.1 (npm), wrapper 0-clause BSD; "you are contractually obligated to release the source code of libav.js and all of its dependencies" if distributed | All 17 can be enabled via fragments (with DTS as the core profile, see section 5) | Shipped `default` variant: 685 KB wasm + 59 KB glue + 7 KB loader (measured, about 751 KB). Custom target build: not built, estimate 1.0 to 1.5 MB | Non-threaded build: no SAB reference in glue (measured). Threaded `.thr` build has one SAB reference | Not documented | `@libav.js/variant-*` 6.10.9, published 2026-08-18 (npm registry) | Documented: `./mkconfig.js <name> '<fragment JSON>'` then `make build-<name>`. Medium effort, Emscripten/Docker toolchain |
| **@ffmpeg/core (ffmpeg.wasm)** | **GPL-2.0-or-later** (npm field). Binary config contains `--enable-gpl` plus x264, x265, libvpx, lame, etc. (measured) | All 17 (GPL build) | 10.2 MB gz wasm (32.2 MB raw, measured). Not usable as shipped | Single-thread `@ffmpeg/core` has no SAB reference (measured). `@ffmpeg/core-mt` references SAB and is also GPL-2.0-or-later | Docs: "WebAssembly is still a lot slower than native"; mt "around 2x" single-thread. No realtime figures | `@ffmpeg/core` 0.12.10, published 2025-01-07; `core-mt` 0.12.10, 2025-04-28 (npm) | Core repo has Dockerfile and Makefile. An LGPL build is possible by dropping `--enable-gpl` and the GPL libs, but the result would still be large |
| **Symphonia (wasm)** | **MPL-2.0** (crates.io) | AIFF, CAF, MP1/MP2 (decoders), MP4/ISO demux. WavPack listed without status. Not mentioned at all: AVI, ASF/WMA, FLV, MPEG-PS, MPEG-TS, AC-3, DTS, AMR, TTA, Musepack, AU, APE audio. About 4 of 17 | Not measured. No wasm build exists (see below) | Not applicable (no wasm build) | Not documented | Rust crate 0.6.1, 2026-08-13 (crates.io) | Its README lists "Providing a WASM API for web usage" as a planned feature. No npm package found (npm search for "symphonia wasm" returned nothing relevant). Would require writing your own wasm-bindgen wrapper |
| **@audio/decode umbrella + per-format packages (LGPL/BSD/Apache subset)** | Umbrella MIT, but its `aac`, `ac3` and `dts` packages are **GPL-2.0** (npm field, READMEs). `eac3`, `ape`, `wma` are LGPL-2.1+ (FFmpeg slim builds, "no GPL components"). `amr` Apache-2.0, `wavpack` and `mpc` BSD-3, `tta` MIT | WMA (ASF audio), AC-3 and E-AC-3 (via `eac3`), APE, AMR, WavPack, Musepack, TTA, AIFF, CAF, AVI (container only; inner PCM/MP3/AAC), 3GP/MP4 (demux), MP1/MP2/MP3 (mpg123). About 11 of 17. Missing: FLV, MPEG-PS, MPEG-TS, AU, DTS (LGPL route) | Measured download per lazy-loaded package (gzip of the package JS with embedded wasm): wma 202 KB, eac3 189 KB, ape 122 KB, amr 105 KB, wavpack 23 KB, mpc 27 KB, tta 5 KB, mp3 65 KB, aiff 7 KB, caf 4 KB, avi 4 KB, mp4 8 KB. Excluded GPL: aac 184 KB, dts 114 KB, ac3 21 KB | Not stated; the packages state "No side files, loads from any CDN", and none of the 17 packages reference SharedArrayBuffer, pthreads or Atomics (measured, single-threaded) | Not documented. Streaming API and dynamic-import lazy loading are documented | `decode-*` packages 2026-08 to 2026-10 (npm). Active | Each package has its own build script with a slim FFmpeg from a shared submodule (see the WMA README). Medium effort per package; you would need to reproduce their build |
| **mediabunny** (already in use) | MPL-2.0 (npm) | Demux only: MP4, QuickTime, Matroska, WebM, MP3, WAV, Ogg, ADTS, FLAC, MPEG-TS, HLS. Decoding goes through WebCodecs, so AC-3/E-AC-3/DTS are not decoded. About 0 of 17 added | Not measured | Not applicable | Not applicable | Published 1.61.4, 2026-10-12 (npm) | Not applicable. The docs document no custom decoder registration API |
| **audio-file-decoder** (ffmpeg wasm) | LGPL-2.1-or-later (npm) | MP3, WAV, FLAC, AAC, OGG (from its npm description, secondhand). Not AC-3, WMA or AVI in its README | 551 KB gz wasm (measured) | Not documented | Not documented | **Last published 2021-11-29** (npm). Effectively unmaintained | Not documented |
| **Mediabunny-style custom WebCodecs**, **dr_libs / minimp3 / other single-codec decoders** | Not evaluated in depth | Each covers one codec, so would need many separate wasm builds. Not pursued | n/a | n/a | n/a | n/a | n/a |

Measured wasm sizes for the individual decoders (gzip of the extracted wasm binary, excluding glue): WMA (FFmpeg, LGPL) 181 KB; E-AC-3/AC-3 (FFmpeg, LGPL) 170 KB; APE (FFmpeg, LGPL) 106 KB; AMR (opencore, Apache) 91 KB; DTS (libdca, GPL) 91 KB; AAC (FAAD2, GPL) 152 KB; AC-3 (liba52, GPL) 14 KB; WavPack (BSD) 17 KB; Musepack (BSD) 20 KB; libav.js default (LGPL) 685 KB.

## 3. Per-candidate notes

### libav.js (Yahweasel)
- Licence: LGPL-2.1 per npm for every `@libav.js/variant-*` (https://registry.npmjs.org/@libav.js/variant-default). The project's own page says FFmpeg is LGPL, the wrapper is 0-clause BSD, and distributors "must provide sources" and are "contractually obligated to release the source code of libav.js and all of its dependencies" (https://github.com/Yahweasel/libav.js).
- The published variants do not include the target demuxers. The `default` variant contains only `format-ogg`, `format-webm`, `opus`, `flac`, `format-wav`, `codec-pcm_f32le` and `audio-filters` (https://raw.githubusercontent.com/Yahweasel/libav.js/master/configs/mkconfigs.js, the `default` entry). This is why the default is small (685 KB gz).
- Demuxer fragments that exist upstream: `demuxer-asf`, `demuxer-au`, `demuxer-avi`, `demuxer-caf`, `demuxer-flv`, `demuxer-matroska`, `demuxer-mpeg` (PS), `demuxer-mpegts`, `demuxer-wav`, `demuxer-wv` (WavPack), `demuxer-aiff`, `demuxer-mp4`, `demuxer-ogg`, `demuxer-rm` (same mkconfigs.js and the fragments directory listing).
- Decoders: fragments `decoder-ac3`, `decoder-wavpack`, `decoder-wmav1`, `decoder-wmav2`, `decoder-wmapro`, `decoder-wmalossless` exist upstream. Per CONFIG.md, "Fragments without a file in `configs/fragments` are enabled directly through their FFmpeg flag", so `dca`, `ape`, `tta`, `mpc7`, `mpc8`, `amrnb`, `amrwb`, `eac3` can be named directly. Their exact names must be confirmed against `configure --list-decoders` when building (https://github.com/Yahweasel/libav.js/blob/master/docs/CONFIG.md).
- Build: `./mkconfig.js <name> '<JSON array of fragments>'` in `configs/`, then `make build-<name>`. The official docs do not mention Docker or Emscripten versions; the repo has a `Dockerfile.development` (https://github.com/Yahweasel/libav.js).
- Precedent: a third-party project (avbridge, MIT) ships its own custom libav.js build because the npm variants lack the AVI, ASF and FLV demuxers. It reports about 10 to 15 MB for its build, which uses faster, larger compile flags (https://cdn.jsdelivr.net/npm/avbridge@2.13.0/NOTICE.md; third-party claim, not verified by me).
- Threads: the `.wasm.mjs` (non-thread) glue has zero SharedArrayBuffer references; the `.thr.mjs` glue has one (measured). Use the non-thread build to avoid needing cross-origin isolation. The site's CSP does not need `'unsafe-eval'`: the glue has zero `eval`/`new Function` calls (measured).

### ffmpeg.wasm (@ffmpeg/core)
- Licence: the npm field says GPL-2.0-or-later for both `@ffmpeg/core` and `@ffmpeg/core-mt` (https://registry.npmjs.org/@ffmpeg/core, https://registry.npmjs.org/@ffmpeg/core-mt). The wrapper `@ffmpeg/ffmpeg` is MIT. Confirmed from the shipped binary: its embedded configure string includes `--enable-gpl` and `--enable-libx264 --enable-libx265 --enable-libvpx --enable-libmp3lame ...` (measured). The ffmpeg.wasm FAQ says the core "follows the same licenses as FFmpeg and its external libraries" (https://ffmpegwasm.netlify.app/docs/faq).
- No LGPL build is published. Excluded for this site.
- Size: 10.2 MB gz (measured). Too large for lazy loading.
- Threads: the single-thread core has zero SAB references; core-mt has SAB references (measured). The FAQ says mt is about 2x faster at the cost of more memory and CPU.

### Symphonia
- MPL-2.0 (crates.io; https://github.com/pdeljanov/Symphonia). Compatible with closed-source use at file level: you may ship it unchanged, or publish changes to MPL-covered files.
- Supported today: AIFF, CAF, ISO/MP4, MKV/WebM, OGG, WAV demuxers; AAC-LC, ADPCM, ALAC, FLAC, MP1/MP2, MP3, PCM, Vorbis decoders. Not mentioned at all: AVI, ASF/WMA, FLV, MPEG-PS, MPEG-TS, AC-3, DTS, AMR, TTA, Musepack, AU, APE audio (https://github.com/pdeljanov/Symphonia).
- Wasm: planned, not shipped. Rust toolchain plus a wasm-bindgen wrapper would be required. Not measured.
- Conclusion: it does not cover enough of the target list to be the engine, so it is not recommended now.

### @audio/decode (audiojs)
- Umbrella: MIT. Per-format packages are the actual decoders, each with its own licence (table in section 2). Documented: streaming input, lazy loading via dynamic import, one package per format (https://github.com/audiojs/decode). Its README table lists `aac` as GPL-2.0 and `ac3`/`dts` as GPL-2.0-or-later.
- Licence trap: `@audio/decode-mp3` is labelled MIT, but its bundled decoder is mpg123, which upstream is LGPL-2.1 (https://www.mpg123.de/). The `mpg123-decoder` npm package is also MIT-labelled. Check the bundled notice before shipping.
- Packaging gap: the top-level `loader` references `@audio/decode-ape` and `@audio/decode-eac3`, but neither appears in the umbrella's `dependencies` list (https://registry.npmjs.org/@audio/decode, version 4.1.1). They have to be installed separately.
- Thread check: none of the 17 `@audio/decode-*` packages I downloaded reference SharedArrayBuffer, pthreads or Atomics (measured). They run single-threaded and need no cross-origin isolation.
- The `aac` package also provides AAC decoding, which the site already gets from WebCodecs. Do not install it; it would pull in a GPL decoder for nothing.

### mediabunny
- MPL-2.0 (npm). Container demuxing only; decoding goes through WebCodecs (https://mediabunny.dev/guide/input-formats). The docs list `MPEG_TS` but no AVI, ASF, FLV, AU or DTS.
- The docs state no way to register a custom decoder, so it cannot be used to route extra codecs to a wasm decoder. Use it as-is for what it already handles, and keep extra formats on a separate path.
- WebCodecs does not support AC-3, E-AC-3 or DTS (https://convert.remotion.dev/docs/mediabunny/formats).

### audio-file-decoder
- LGPL-2.1-or-later, FFmpeg wasm, 551 KB gz (measured). Last published 2021-11-29, so treat it as stale. It lists MP3, WAV, FLAC, AAC and OGG; I did not verify the list from its own README.

## 4. Recommendation

**Primary: one custom, LGPL-only libav.js build (non-threaded), lazy-loaded.**

- Why: it is the only option that puts all 17 target formats behind one engine, using FFmpeg's native LGPL decoders, with no GPL code. It needs no SharedArrayBuffer and no COOP/COEP headers, and its glue has no `eval`, so the strict CSP works.
- Build: start from `default`'s fragment list and add the target demuxers and decoders. Suggested fragment set, subject to confirming FFmpeg names with `configure --list-decoders` and `--list-demuxers`:
  - Demuxers: `demuxer-asf`, `demuxer-avi`, `demuxer-flv`, `demuxer-mpeg` (PS), `demuxer-mpegts`, `demuxer-au`, `demuxer-caf`, `demuxer-aiff`, `demuxer-mp4` (3GP), `demuxer-matroska`, `demuxer-wav`, `demuxer-wv`, `demuxer-ogg`, `demuxer-webm`.
  - Decoders: `decoder-wmav1`, `decoder-wmav2`, `decoder-wmapro`, `decoder-wmalossless`, `decoder-ac3`, `decoder-eac3`, `decoder-dca`, `decoder-ape`, `decoder-tta`, `decoder-wavpack`, `decoder-mpc7`, `decoder-mpc8`, `decoder-amrnb`, `decoder-amrwb`, `decoder-mp2`, `decoder-mp3float`, plus the PCM decoders from `default`.
  - Do not include `--enable-gpl` or `--enable-nonfree`. Do not include `libx264`, `libx265`, `libvpx` or any other GPL library.
- Expected size: estimate 1.0 to 1.5 MB gz for the engine. This is an extrapolation from the measured per-package sizes (each FFmpeg-based package carries its own copy of the core), not a measurement. Measure the first build.
- Distribution obligations (LGPL-2.1): serve the wasm and glue from the site's own origin, host the exact source (FFmpeg and libav.js) plus the `mkconfig`/`make` inputs next to it, include LGPL and FFmpeg notices on the download page, and link dynamically, meaning the wasm is a separate file that the site loads at runtime. Do not claim ownership of FFmpeg. This is what the FFmpeg legal page lists (https://ffmpeg.org/legal.html). Get a lawyer's review before shipping.

**Fallback, no build pipeline needed: `@audio/decode` per-format packages, LGPL/BSD/Apache subset only.**

- Install `wma`, `eac3` (covers AC-3 and E-AC-3), `ape`, `amr`, `wavpack`, `mpc`, `tta`, `aiff`, `caf`, `avi`, `mp4`. Do not install `aac`, `ac3` or `dts` (GPL).
- Download for the heavy set: about 0.7 MB gz total, lazy-loaded per format (measured sizes from section 2).
- Misses FLV, MPEG-PS, MPEG-TS, AU and DTS, so it is a good prototype or a stopgap, not the final engine.

**Not recommended:** ffmpeg.wasm (GPL, 10 MB), Symphonia (no wasm build, missing about 13 of 17 target formats), mediabunny as a decoder (no custom decoder hook, and it is already used for containers only).

## 5. Formats that stay impossible or hard

Technically none of the 17 is impossible under an LGPL build. The exceptions and caveats:

- DTS-HD Master Audio (lossless extension): FFmpeg's native `dca` decoder is built for the DTS core. I did not verify whether it decodes the lossless extension. Expect DTS-HD MA files to fall back to the lossy core, or to fail. Test with real samples.
- DRM-protected WMA/WMV (Windows Media DRM) and any other encrypted media: impossible without the licence. Decoding can only handle the clear audio.
- Pre-SV7 Musepack (SV4 to SV6): my understanding is that FFmpeg's `mpc7` decoder covers SV7 and SV8 only. Not verified here.
- Corrupt or non-standard files: decoders may fail. A fallback message is needed.
- Legal, not technical: AC-3, E-AC-3, DTS, WMA, AMR and MPEG-2 audio are patent-encumbered in many jurisdictions. Decoding in a browser may still need a patent review. Not legal advice.

## 6. Speed and threading

- Speed: none of the candidates publishes a real-time factor for these builds. ffmpeg.wasm says WebAssembly is "still a lot slower than native" and gives no numbers (https://ffmpegwasm.netlify.app/docs/faq). Benchmark the chosen build on target machines (for example, decode a one-hour file and time it).
- Threads and SharedArrayBuffer: the non-threaded libav.js build and the single-thread `@ffmpeg/core` do not reference SharedArrayBuffer (measured). The threaded builds do. Browsers only expose SharedArrayBuffer in cross-origin isolated pages, which requires `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` or `credentialless` (https://developer.mozilla.org/en-US/docs/Web/API/Window/crossOriginIsolated). Avoid the threaded builds so no COOP/COEP headers are needed.

## 7. Open items before implementation

1. Confirm FFmpeg decoder and demuxer names (`configure --list-decoders`, `--list-demuxers`) for the chosen version.
2. Build the libav.js custom variant and measure the real gz size (estimate 1.0 to 1.5 MB).
3. Check the built glue under the strict CSP in a browser (no `unsafe-eval`).
4. Test a real sample for each of the 17 formats, including DTS-HD MA and a WMV with WMA audio. Check that the ASF path in `@audio/decode-wma` and the libav build agree.
5. Verify the bundled notices (mpg123 LGPL in `@audio/decode-mp3`; LGPL in `eac3`, `ape`, `wma`).
6. Get a licence review of the LGPL source-offer and dynamic-linking arrangement.

## Sources

- https://github.com/Yahweasel/libav.js
- https://github.com/Yahweasel/libav.js/blob/master/docs/CONFIG.md
- https://raw.githubusercontent.com/Yahweasel/libav.js/master/configs/mkconfigs.js
- https://registry.npmjs.org/@libav.js/variant-default (also variant-webcodecs, variant-webm-vp9)
- https://cdn.jsdelivr.net/npm/avbridge@2.13.0/NOTICE.md (third-party)
- https://registry.npmjs.org/@ffmpeg/core, https://registry.npmjs.org/@ffmpeg/core-mt, https://registry.npmjs.org/@ffmpeg/ffmpeg
- https://github.com/ffmpegwasm/ffmpeg.wasm
- https://ffmpegwasm.netlify.app/docs/faq
- https://ffmpeg.org/legal.html
- https://ffmpeg.org/general.html
- https://raw.githubusercontent.com/FFmpeg/FFmpeg/master/libavcodec/dcadec.c
- https://raw.githubusercontent.com/FFmpeg/FFmpeg/master/libavcodec/apedec.c
- https://github.com/pdeljanov/Symphonia
- https://crates.io/api/v1/crates/symphonia
- https://mediabunny.dev/guide/input-formats
- https://registry.npmjs.org/mediabunny
- https://github.com/audiojs/decode
- https://registry.npmjs.org/@audio/decode, and @audio/decode-{aac,ac3,dts,wma,eac3,ape,amr,wavpack,mpc,tta,mp3,flac,aiff,caf,avi,mp4} (npm registry)
- https://www.mpg123.de/
- https://convert.remotion.dev/docs/mediabunny/formats
- https://registry.npmjs.org/audio-file-decoder
- https://developer.mozilla.org/en-US/docs/Web/API/Window/crossOriginIsolated

Measurement method: npm tarballs were downloaded to a scratch directory (not executed). The wasm was extracted from the `binaryDecode`-encoded literal in each `src/*.wasm.js` (magic bytes verified) or read from the libav.js dist directory, then gzipped with `gzip -9`.
