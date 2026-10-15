#!/usr/bin/env bash
# Reproducible, idempotent build of a custom LGPL-only, non-threaded, audio-decode-only
# libav.js variant (see README.md). Re-running is cheap: finished steps are skipped and
# make only rebuilds what changed.
#
#   ./build.sh              build everything (emsdk, libav.js, variant) and verify
#   ./build.sh verify       only re-run the licence / size checks on existing output
#   FORCE=1 ./build.sh      wipe the variant's build products first (keeps downloads)
#
# Environment overrides:
#   LIBAV_BUILD_DIR  work + output dir        (default ~/audio-cleaner/src/libav-build)
#   EMSDK_DIR        emsdk checkout           (default ~/audio-cleaner/src/emsdk)
#   JOBS             make parallelism         (default nproc)
#   OPTFLAGS         compiler opt flags       (default from versions.env, -Oz)
#   RATE             download rate limit      (default 40M, curl --limit-rate syntax)
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=versions.env
source "$HERE/versions.env"

BASE=${LIBAV_BUILD_DIR:-$HOME/audio-cleaner/src/libav-build}
EMSDK_DIR=${EMSDK_DIR:-$HOME/audio-cleaner/src/emsdk}
JOBS=${JOBS:-$(nproc)}
OPTFLAGS=${OPTFLAGS:-$OPTFLAGS_DEFAULT}
RATE=${RATE:-40M}
SRC=$BASE/libav.js-src
OUT=$BASE/dist
LOGS=$BASE/logs
LIBAVJS_VERSION=${LIBAVJS_TAG#v}                 # 6.10.9.0 (== LIBAVJS_VERSION in its Makefile)
FFDIR=$SRC/build/ffmpeg-$FFMPEG_VERSION
FFBUILD=$FFDIR/build-base-$VARIANT

log() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

mkdir -p "$BASE" "$LOGS"

# Every curl (emsdk's downloader, libav.js's makefiles) honours this .curlrc.
export CURL_HOME=$BASE/curlhome
mkdir -p "$CURL_HOME"
printf 'limit-rate = %s\n' "$RATE" > "$CURL_HOME/.curlrc"
export EMSDK_USE_CURL=1

# ---------------------------------------------------------------------------
# Variant definition: libav.js "configuration fragments"
# (libav.js docs/CONFIG.md). Fragments named demuxer-*/decoder-*/parser-* map
# directly to FFmpeg's --enable-demuxer=/--enable-decoder=/--enable-parser=.
# Deliberately absent: any encoder, muxer, video decoder, avfilter, GPL/nonfree
# library, and threads (only the non-threaded "base" target is ever built).
# ---------------------------------------------------------------------------
DEMUXERS=(
  asf avi flv mpegps mpegvideo mpegts au caf aiff mp4 matroska ogg wav w64 wavpack
  ape tta mpc mpc8 amr amrnb amrwb ac3 eac3 dts dtshd mp3 aac flac
)
# Notes: "wavpack" (libav.js fragment) = wv demuxer + wavpack decoder; "mp4" = mov demuxer.
# "mpegps" is the MPEG program-stream demuxer. libav.js's own "demuxer-mpeg" fragment passes
# --enable-demuxer=mpeg, which FFmpeg's configure silently ignores (no such component), so we
# name it correctly. "mpegvideo" (raw MPEG video) is probe-only here: without it FFmpeg's stream
# prober mislabels the video stream of an MPEG-PS file as MP3 audio (and feeds it to the mp3
# decoder); with it the stream is correctly typed as video and simply has no decoder. step_verify checks that every requested component really got enabled.
DECODERS=(
  wmav1 wmav2 wmapro wmalossless ac3 eac3 dca ape tta wavpack mpc7 mpc8
  amrnb amrwb mp1float mp2float mp3float aac aac_latm vorbis opus flac alac
)
PARSERS=(aac aac_latm ac3 amr dca flac mpegaudio mpegvideo opus vorbis)
# libav.js only exposes swresample through avfilter (it has no direct swr_* bindings), so the
# resample / channel-convert path is aresample (swresample) + aformat/pan/anull via a filter graph.
FILTERS=(aformat anull pan)    # aresample comes with the "swresample" fragment
# pcm_* and adpcm_* decoders are added from FFmpeg's own list (needs the source tree)

build_fragments_json() {
  node -e '
    const [dem, dec, par, flt, pcm] = process.argv.slice(1).map(s => s.split(" ").filter(Boolean));
    const f = ["avformat", "avcodec", "avfilter", "swresample", "zlib"];
    for (const d of dem) f.push("demuxer-" + d);
    for (const d of dec.concat(pcm)) f.push("decoder-" + d);
    for (const p of par) f.push("parser-" + p);
    for (const x of flt) f.push("filter-" + x);
    console.log(JSON.stringify(f));
  ' "${DEMUXERS[*]}" "${DECODERS[*]}" "${PARSERS[*]}" "${FILTERS[*]}" "$1"
}

# ---------------------------------------------------------------------------
step_emsdk() {
  log "emsdk $EMSDK_VERSION"
  if [ ! -d "$EMSDK_DIR/.git" ]; then
    git clone --depth 1 "$EMSDK_REPO" "$EMSDK_DIR" >"$LOGS/emsdk-clone.log" 2>&1
  fi
  if [ "$(git -C "$EMSDK_DIR" rev-parse HEAD)" != "$EMSDK_COMMIT" ]; then
    git -C "$EMSDK_DIR" fetch --depth 1 origin "$EMSDK_COMMIT" >>"$LOGS/emsdk-clone.log" 2>&1
    git -C "$EMSDK_DIR" checkout -q "$EMSDK_COMMIT"
  fi
  if ! grep -q "$EMSDK_VERSION" "$EMSDK_DIR/upstream/emscripten/emscripten-version.txt" 2>/dev/null; then
    ( cd "$EMSDK_DIR" && ./emsdk install "$EMSDK_VERSION" && ./emsdk activate "$EMSDK_VERSION" ) \
      2>&1 | tr '\r' '\n' | grep -v '^[#= ]*[0-9.]*%\?[#= ]*$' >"$LOGS/emsdk-install.log" \
      || { tail -20 "$LOGS/emsdk-install.log"; die "emsdk install failed"; }
  fi
  set +u
  # shellcheck disable=SC1091
  source "$EMSDK_DIR/emsdk_env.sh" >/dev/null 2>&1
  set -u
  # use emsdk's bundled node for libav.js's helper scripts (its CI uses node 24)
  [ -n "${EMSDK_NODE:-}" ] && export PATH="$(dirname "$EMSDK_NODE"):$PATH"
  emcc --version 2>/dev/null | head -1 | grep -q " $EMSDK_VERSION " \
    || die "emcc is not version $EMSDK_VERSION"
  echo "emcc $EMSDK_VERSION, node $(node -v)"
}

step_libavjs() {
  log "libav.js $LIBAVJS_TAG"
  if [ ! -d "$SRC/.git" ]; then
    git clone --depth 1 --branch "$LIBAVJS_TAG" "$LIBAVJS_REPO" "$SRC" >"$LOGS/libavjs-clone.log" 2>&1
  fi
  [ "$(git -C "$SRC" rev-parse HEAD)" = "$LIBAVJS_COMMIT" ] \
    || die "libav.js checkout is not commit $LIBAVJS_COMMIT"
  if [ ! -x "$SRC/node_modules/.bin/terser" ]; then
    ( cd "$SRC" && npm ci --ignore-scripts --no-audit --no-fund ) >"$LOGS/npm.log" 2>&1 \
      || { tail "$LOGS/npm.log"; die "npm ci failed"; }
  fi
  # fetch + verify + patch FFmpeg through libav.js's own makefile rules
  ( cd "$SRC" && make build/ffmpeg-$FFMPEG_VERSION/PATCHED ) >"$LOGS/extract.log" 2>&1 \
    || { tail "$LOGS/extract.log"; die "ffmpeg extract failed"; }
  echo "$FFMPEG_SHA256  $SRC/build/ffmpeg-$FFMPEG_VERSION.tar.xz" | sha256sum -c --quiet - \
    || die "ffmpeg tarball checksum mismatch"
  # the two small dependency tarballs: fetch (rate-limited) and verify before the build uses them
  fetch_verify "$ZLIB_URL" "$SRC/build/zlib-$ZLIB_VERSION.tar.gz" "$ZLIB_SHA256"
  fetch_verify "$EMFT_URL" "$SRC/build/emfiberthreads-$EMFT_VERSION.tar.gz" "$EMFT_SHA256"
}

fetch_verify() {  # url dest sha256
  if ! { [ -f "$2" ] && echo "$3  $2" | sha256sum -c --quiet - 2>/dev/null; }; then
    curl -sSfL "$1" -o "$2.part" && mv "$2.part" "$2"
    echo "$3  $2" | sha256sum -c --quiet - || die "checksum mismatch for $1"
  fi
}

step_config() {
  log "variant config ($VARIANT)"
  local pcm json cfg=$SRC/configs/configs/$VARIANT
  pcm=$(cd "$FFDIR" && ./configure --list-decoders | tr -s ' \t' '\n\n' \
        | grep -E '^(pcm_|adpcm_)' | grep -vE '_at$|mediacodec' | sort -u \
        | grep -vxFf <(grep -oE '^[a-z0-9_]+_decoder_deps="[^"]*gpl' "$FFDIR/configure" \
                       | sed 's/_decoder_deps=.*//') | tr '\n' ' ')
  # (the grep -v drops the few decoders FFmpeg itself gates behind --enable-gpl, "lgpl_gpl";
  #  as of 9.0 that is adpcm_circus/ima_escape/ima_hvqm2/ima_hvqm4/ima_magix/ima_pda/n64/psxc)
  json=$(build_fragments_json "$pcm")
  # only rewrite when changed, so make does not needlessly reconfigure FFmpeg
  if [ -f "$cfg/config.json" ] && [ "$(cat "$cfg/config.json")" = "$json" ] \
     && [ -f "$cfg/ffmpeg-config.txt" ]; then
    echo "config unchanged"
  else
    rm -rf "$cfg" "$FFBUILD"
    ( cd "$SRC/configs" && ./mkconfig.js "$VARIANT" "$json" )
    # keep the generated config next to the build for reference
    cp "$cfg/config.json" "$BASE/variant-$VARIANT.config.json"
  fi
  # Licence guard on the *inputs*: refuse to continue if any GPL / nonfree / version3 or
  # external-library switch slipped in.
  if grep -E -- '--enable-(gpl|nonfree|version3|lib[a-z0-9_]+|openssl|gnutls|mbedtls)' "$cfg/ffmpeg-config.txt"; then
    die "forbidden configure flag in $cfg/ffmpeg-config.txt"
  fi
}

step_build() {
  log "build (OPTFLAGS=$OPTFLAGS, jobs=$JOBS)"
  local stamp=$BASE/.stamp-$VARIANT-optflags
  if [ "${FORCE:-0}" = 1 ] || { [ -f "$stamp" ] && [ "$(cat "$stamp")" != "$OPTFLAGS" ]; }; then
    echo "wiping previous build products for $VARIANT"
    rm -rf "$FFBUILD" "$SRC/build/inst" "$SRC"/dist/*"$VARIANT"* \
           "$SRC"/build/*"$VARIANT"*
  fi
  echo "$OPTFLAGS" > "$stamp"
  local v=$LIBAVJS_VERSION n=$VARIANT
  ( cd "$SRC" && make -j"$JOBS" OPTFLAGS="$OPTFLAGS" \
      dist/libav-$v-$n.wasm.mjs dist/libav-$v-$n.wasm.js \
      dist/libav-$n.mjs dist/libav-$n.js dist/libav-$v-$n.mjs dist/libav-$v-$n.js \
      dist/libav.types.d.ts ) >"$LOGS/build-$n.log" 2>&1 \
    || { tail -40 "$LOGS/build-$n.log"; die "build failed (see $LOGS/build-$n.log)"; }

  rm -rf "$OUT"; mkdir -p "$OUT"
  cp "$SRC"/dist/libav-$v-$n.wasm.mjs "$SRC"/dist/libav-$v-$n.wasm.js \
     "$SRC"/dist/libav-$v-$n.wasm.wasm \
     "$SRC"/dist/libav-$v-$n.mjs "$SRC"/dist/libav-$v-$n.js \
     "$SRC"/dist/libav-$n.mjs "$SRC"/dist/libav-$n.js \
     "$SRC"/dist/libav.types.d.ts "$OUT"/
}

step_sources() {
  log "LGPL source bundle"
  local s=$BASE/sources
  mkdir -p "$s"
  cp -u "$SRC/build/ffmpeg-$FFMPEG_VERSION.tar.xz" "$s/"
  [ -f "$SRC/build/zlib-$ZLIB_VERSION.tar.gz" ] && cp -u "$SRC/build/zlib-$ZLIB_VERSION.tar.gz" "$s/"
  [ -f "$SRC/build/emfiberthreads-$EMFT_VERSION.tar.gz" ] && cp -u "$SRC/build/emfiberthreads-$EMFT_VERSION.tar.gz" "$s/"
  git -C "$SRC" archive --format=tar.gz --prefix="libav.js-$LIBAVJS_VERSION/" \
      -o "$s/libav.js-$LIBAVJS_TAG.tar.gz" HEAD
  tar czf "$s/acw-libav-build-scripts.tar.gz" -C "$HERE/.." libav
  ( cd "$s" && sha256sum * > SHA256SUMS )
  ls -l "$s"
}

# ---------------------------------------------------------------------------
step_verify() {
  log "verify: licence"
  local cfg=$SRC/configs/configs/$VARIANT fail=0 h=$FFBUILD/config.h
  [ -f "$h" ] || die "no FFmpeg config.h at $h (run the build first)"
  grep -F '#define FFMPEG_LICENSE' "$h"
  grep -q '#define FFMPEG_LICENSE "LGPL version 2.1 or later"' "$h" || { echo "FFMPEG_LICENSE is not LGPL2.1+"; fail=1; }
  for k in CONFIG_GPL CONFIG_GPLV3 CONFIG_NONFREE CONFIG_VERSION3; do
    grep -q "#define $k 0" "$h" && echo "$k = 0" || { echo "$k is NOT 0"; fail=1; }
  done
  if grep -E -- '--enable-(gpl|nonfree|version3)' "$h" "$FFBUILD/ffbuild/config.mak" "$cfg/ffmpeg-config.txt"; then
    echo "GPL/nonfree flag found"; fail=1
  fi
  log "verify: every requested component is enabled in the FFmpeg build"
  local cc=$FFBUILD/config_components.h kind name up miss=0 n=0
  while IFS='=' read -r kind name; do
    kind=${kind#--enable-}; up=$(printf '%s_%s' "$name" "$kind" | tr 'a-z' 'A-Z')
    n=$((n + 1))
    grep -q "^#define CONFIG_$up 1" "$cc" || { echo "NOT ENABLED: $kind $name"; miss=1; }
  done < <(grep -o -E -- '--enable-(demuxer|decoder|parser|filter)=[A-Za-z0-9_]+' "$cfg/ffmpeg-config.txt" | sort -u)
  [ $miss = 0 ] || die "some requested components are not enabled (typo or unsupported name)"
  echo "$n requested demuxers/decoders/parsers/filters all enabled"
  # and nothing unexpected is enabled: no encoders, no muxers, no video decoders
  local bad
  bad=$(grep -E '^#define CONFIG_[A-Z0-9_]+_(ENCODER|MUXER|HWACCEL) 1' "$cc" || true)
  [ -z "$bad" ] || { echo "$bad"; die "encoders/muxers unexpectedly enabled"; }
  echo "no encoders / muxers / hwaccels enabled"

  # FFmpeg's licence string is dead-code-eliminated from the wasm, so check the shipped files
  # instead: the glue carries the LGPL notice header, and neither wasm nor glue may mention GPL flags.
  local g=$OUT/libav-$LIBAVJS_VERSION-$VARIANT.wasm.mjs w=$OUT/libav-$LIBAVJS_VERSION-$VARIANT.wasm.wasm
  grep -q 'GNU LESSER GENERAL PUBLIC LICENSE' "$g" && echo "glue carries the LGPL notice" \
    || { echo "glue is missing the LGPL notice header"; fail=1; }
  if grep -a -l -E -- '--enable-(gpl|nonfree|version3)|libx26[45]|libfdk' "$g" "$w"; then
    echo "GPL/nonfree marker found in shipped files"; fail=1; fi
  # only FFmpeg's own libs (+ zlib, permissive) are linked
  echo "linked libs: $(sed 's#.*/##' "$cfg/libs.txt" | sort -u | tr '\n' ' ')"
  [ $fail = 0 ] || die "licence verification FAILED"
  echo "LICENSE CHECK: OK (LGPL version 2.1 or later; GPL=0 NONFREE=0 VERSION3=0)"

  log "verify: sizes (bytes: raw / gzip -9)"
  : > "$BASE/SIZES.txt"
  local f raw gz tot_raw=0 tot_gz=0
  for f in libav-$VARIANT.mjs libav-$LIBAVJS_VERSION-$VARIANT.wasm.mjs libav-$LIBAVJS_VERSION-$VARIANT.wasm.wasm \
           libav-$VARIANT.js libav-$LIBAVJS_VERSION-$VARIANT.wasm.js; do
    raw=$(stat -c %s "$OUT/$f"); gz=$(gzip -9 -c "$OUT/$f" | wc -c)
    printf '%-44s %10d %10d\n' "$f" "$raw" "$gz" | tee -a "$BASE/SIZES.txt"
  done
  # what a browser using the ES-module build downloads: loader + glue + wasm
  for f in libav-$VARIANT.mjs libav-$LIBAVJS_VERSION-$VARIANT.wasm.mjs libav-$LIBAVJS_VERSION-$VARIANT.wasm.wasm; do
    tot_raw=$((tot_raw + $(stat -c %s "$OUT/$f"))); tot_gz=$((tot_gz + $(gzip -9 -c "$OUT/$f" | wc -c)))
  done
  printf '%-44s %10d %10d\n' "ESM total (loader+glue+wasm)" "$tot_raw" "$tot_gz" | tee -a "$BASE/SIZES.txt"
}

case "${1:-all}" in
  verify) step_emsdk; step_libavjs; step_config; step_verify ;;
  all)    step_emsdk; step_libavjs; step_config; step_build; step_sources; step_verify
          log "done: $OUT" ;;
  *) die "usage: $0 [all|verify]" ;;
esac
