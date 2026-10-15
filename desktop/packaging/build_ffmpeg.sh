#!/usr/bin/env bash
# Reproducible, minimal, LGPL-only FFmpeg for the Audio Cleaner desktop app.
#
#   desktop/packaging/build_ffmpeg.sh <platform> <out_dir>
#
#   platform  linux-x86_64    native (used for local verification of the Linux bundle)
#             macos-arm64     native, on an arm64 Mac
#             macos-x86_64    native, on an Intel Mac
#             windows-x86_64  on Windows: natively in an MSYS2 "MINGW64" shell (build_windows.ps1 does this;
#                             pacman -S mingw-w64-x86_64-{gcc,pkgconf,nasm} make diffutils),
#                             or cross-compiled on Linux (Ubuntu: apt install mingw-w64 nasm)
#
# Output (<out_dir>): ffmpeg[.exe], ffprobe[.exe] (statically linked against libavcodec/format/..,
# libopus and LAME; no ffmpeg DLLs), ffmpeg-buildinfo.txt (versions, source URLs, sha256s, the full
# configure lines and this script - shipped in the app's licenses/ folder because the LGPL requires
# telling users how to rebuild), and the licence texts of the three source trees (COPYING.*).
#
# The app only remuxes video (stream copy) and decodes/encodes audio, so everything else is
# switched off (--disable-everything + an explicit allow list). No --enable-gpl, --enable-version3 or
# --enable-nonfree, so the result is LGPL-2.1-or-later. The libraries linked in:
#   libopus 1.5.2  BSD-3-Clause (Opus encoder; the decoder used is FFmpeg's native one)
#   LAME 3.100     LGPL-2.0-or-later (mp3 encoder, the "mp3" output format of acw_common.ffmpeg)
#
# Environment:
#   WORK_DIR       scratch directory (default: a fresh mktemp dir, removed on success unless KEEP_WORK=1)
#   DL_DIR         cache for the downloaded source tarballs (default $WORK_DIR/dl; set it to reuse)
#   JOBS           parallel make jobs (default: CPU count)
#   WITH_LAME      1 (default) builds libmp3lame in; 0 leaves mp3 encoding out
#   REQUIRE_ASM    1 = fail if nasm/yasm is missing on x86 (CI sets this); default: warn and use
#                  --disable-x86asm (slower, bit-identical output)
#   CURL_EXTRA     extra curl arguments, e.g. "--limit-rate 40M" on a slow home connection
#   MACOSX_DEPLOYMENT_TARGET  default 11.0 (macOS only)
#
# Needs: bash >= 3.2, curl, tar, xz, make, a C compiler, pkg-config, sha256sum or shasum.
# The three tarballs are pinned by sha256 below (computed once from the official downloads; the
# FFmpeg one also matches its GPG signature, key FCF986EA15E6E293A5644F10B4322F04D67658D8).
set -euo pipefail

# ---- pinned inputs ---------------------------------------------------------------------------
FFMPEG_VERSION=8.0.3
FFMPEG_URL="https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz"
FFMPEG_SHA256=6136812ea6d4e68bdba27e33c2a94382711cdf4f8602ffef056ff792bd6f9818

OPUS_VERSION=1.5.2
OPUS_URL="https://github.com/xiph/opus/releases/download/v${OPUS_VERSION}/opus-${OPUS_VERSION}.tar.gz"
OPUS_SHA256=65c1d2f78b9f2fb20082c38cbe47c951ad5839345876e46941612ee87f9a7ce1 # same pin as obs-plugin

LAME_VERSION=3.100
LAME_URL="https://downloads.sourceforge.net/project/lame/lame/${LAME_VERSION}/lame-${LAME_VERSION}.tar.gz"
LAME_URL_MIRROR="https://deb.debian.org/debian/pool/main/l/lame/lame_${LAME_VERSION}.orig.tar.gz" # byte-identical
LAME_SHA256=ddfe36cab873794038ae2c1210557ad34857a4b6bdc515785d1da9e175b1da1e

# ---- what the app needs from FFmpeg ----------------------------------------------------------
# Container formats (acw_common.ffmpeg.ALLOWED_FORMATS) + raw PCM pipes + text subtitles.
DEMUXERS="mov,matroska,avi,mpegts,mp3,wav,flac,ogg,aac,asf,flv,aiff,caf,w64,ac3,eac3,pcm_s16le,pcm_s24le,pcm_f32le,srt,webvtt"
MUXERS="mp4,mov,ipod,matroska,webm,wav,flac,ogg,opus,null,mp3,pcm_s16le,pcm_s24le,pcm_f32le"
PROTOCOLS="file,pipe"
# Audio decoders that are LGPL and common; pcm_*/adpcm_* are added from configure's own list below.
DECODERS="aac,aac_latm,mp1,mp1float,mp2,mp2float,mp3,mp3float,mp3adu,mp3adufloat,opus,vorbis,flac,alac,ac3,eac3,dca,truehd,mlp,wmav1,wmav2,wmapro,wmalossless,amrnb,amrwb,speex,gsm,gsm_ms,subrip,srt,webvtt,movtext,ass,ssa"
ENCODERS="aac,flac,pcm_s16le,pcm_s24le,pcm_f32le,libopus,movtext,srt,subrip,webvtt"
# Graph plumbing + what -ac/-ar/-t and the CLI need (ffmpeg_select in configure adds a few video ones).
FILTERS="abuffer,abuffersink,buffer,buffersink,aresample,aformat,anull,atrim,format,null,trim,asetpts,setpts,volume,pan,apad"

usage() { sed -n '2,12p' "${BASH_SOURCE[0]}" >&2; exit 2; }
[ $# -eq 2 ] || usage
PLATFORM="$1"
OUT="$2"
WITH_LAME="${WITH_LAME:-1}"
[ "$WITH_LAME" = 1 ] && ENCODERS="$ENCODERS,libmp3lame"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_OS="$(uname -s)"
HOST_ARCH="$(uname -m)"

# ---- platform --------------------------------------------------------------------------------
CROSS_PREFIX=""   # toolchain prefix (cross builds)
EXE=""
FF_PLATFORM_ARGS=()
DEP_HOST_ARGS=()  # extra ./configure args for libopus / LAME
DEP_CFLAGS="-O2"
case "$PLATFORM" in
  linux-x86_64)
    [ "$HOST_OS $HOST_ARCH" = "Linux x86_64" ] || { echo "$PLATFORM must be built on Linux x86_64" >&2; exit 1; }
    X86=1
    ;;
  macos-arm64 | macos-x86_64)
    [ "$HOST_OS" = Darwin ] || { echo "$PLATFORM must be built on macOS" >&2; exit 1; }
    want="${PLATFORM#macos-}"
    have="$HOST_ARCH"
    [ "$have" = "$want" ] || { echo "$PLATFORM must be built on a matching $want Mac (this is $have)" >&2; exit 1; }
    MIN="${MACOSX_DEPLOYMENT_TARGET:-11.0}"
    export MACOSX_DEPLOYMENT_TARGET="$MIN"
    DEP_CFLAGS="-O2 -mmacosx-version-min=$MIN"
    FF_PLATFORM_ARGS+=("--extra-cflags=-mmacosx-version-min=$MIN" "--extra-ldflags=-mmacosx-version-min=$MIN")
    # LAME's 2017 config.guess does not know Apple silicon; spell the triplet out (config.sub accepts it).
    triplet_arch="$want"
    [ "$want" = arm64 ] && triplet_arch=aarch64
    DEP_HOST_ARGS+=("--build=${triplet_arch}-apple-darwin")
    X86=0
    [ "$want" = x86_64 ] && X86=1
    ;;
  windows-x86_64)
    case "$HOST_OS" in
      Linux)
        CROSS_PREFIX=x86_64-w64-mingw32-
        command -v "${CROSS_PREFIX}gcc" >/dev/null 2>&1 || { echo "${CROSS_PREFIX}gcc not found (apt install mingw-w64)" >&2; exit 1; }
        ;;
      MINGW64_NT*)
        # MSYS2 MINGW64 shell: gcc is already the x86_64 mingw-w64 compiler.
        command -v gcc >/dev/null 2>&1 || { echo "gcc not found (pacman -S mingw-w64-x86_64-gcc)" >&2; exit 1; }
        ;;
      *)
        echo "$PLATFORM is built in an MSYS2 MINGW64 shell on Windows, or cross-compiled on Linux with mingw-w64" >&2
        exit 1
        ;;
    esac
    EXE=.exe
    # -static: libgcc/winpthread end up in the exe, so the app ships no runtime DLLs of ours.
    # w32threads instead of pthreads keeps libwinpthread out of the picture altogether.
    FF_PLATFORM_ARGS+=(--arch=x86_64 --target-os=mingw32 --disable-pthreads --enable-w32threads
      "--extra-ldflags=-static -static-libgcc")
    [ -n "$CROSS_PREFIX" ] && FF_PLATFORM_ARGS+=("--cross-prefix=${CROSS_PREFIX}")
    DEP_HOST_ARGS+=(--host=x86_64-w64-mingw32 --disable-stack-protector)
    X86=1
    ;;
  *)
    echo "unknown platform '$PLATFORM' (linux-x86_64 | macos-arm64 | macos-x86_64 | windows-x86_64)" >&2
    exit 2
    ;;
esac

for tool in curl tar xz make pkg-config; do
  command -v "$tool" >/dev/null 2>&1 || { echo "missing required tool: $tool" >&2; exit 1; }
done
if command -v sha256sum >/dev/null 2>&1; then
  sha256_of() { sha256sum "$1" | cut -d' ' -f1; }
else
  sha256_of() { shasum -a 256 "$1" | cut -d' ' -f1; }
fi
JOBS="${JOBS:-$( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2) | head -n1)}"

# x86 assembly (nasm); everything else (arm64 NEON, mingw cross to x86) needs nothing extra.
if [ "$X86" = 1 ] && ! command -v nasm >/dev/null 2>&1 && ! command -v yasm >/dev/null 2>&1; then
  if [ "${REQUIRE_ASM:-0}" = 1 ]; then
    echo "nasm/yasm is required (REQUIRE_ASM=1)" >&2
    exit 1
  fi
  echo "WARNING: nasm not found - building FFmpeg with --disable-x86asm (slower)" >&2
  FF_PLATFORM_ARGS+=(--disable-x86asm)
fi

# ---- scratch + downloads ---------------------------------------------------------------------
if [ -n "${WORK_DIR:-}" ]; then
  WORK="$WORK_DIR"
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d "${TMPDIR:-/tmp}/acw-ffmpeg.XXXXXX")"
  [ "${KEEP_WORK:-0}" = 1 ] || trap 'rm -rf "$WORK"' EXIT
fi
WORK="$(cd "$WORK" && pwd)"
DL="${DL_DIR:-$WORK/dl}"
SRC="$WORK/src"
DEPS="$WORK/deps"
STAGE="$WORK/stage"
mkdir -p "$DL" "$SRC" "$DEPS" "$STAGE" "$OUT"
OUT="$(cd "$OUT" && pwd)"

# fetch <sha256> <file> <url...>: download (with retries and fallbacks) unless cached, verify, extract.
fetch() {
  local sha="$1" file="$2" url got
  shift 2
  if [ ! -f "$DL/$file" ] || [ "$(sha256_of "$DL/$file")" != "$sha" ]; then
    rm -f "$DL/$file"
    for url in "$@"; do
      echo "== download $url"
      # shellcheck disable=SC2086
      if curl -fsSL --retry 4 --retry-delay 3 --retry-connrefused ${CURL_EXTRA:-} -o "$DL/$file.part" "$url"; then
        mv "$DL/$file.part" "$DL/$file"
        break
      fi
    done
  fi
  [ -f "$DL/$file" ] || { echo "could not download $file" >&2; exit 1; }
  got="$(sha256_of "$DL/$file")"
  if [ "$got" != "$sha" ]; then
    echo "sha256 mismatch for $file: expected $sha, got $got" >&2
    exit 1
  fi
  tar xf "$DL/$file" -C "$SRC"
}
fetch "$FFMPEG_SHA256" "ffmpeg-${FFMPEG_VERSION}.tar.xz" "$FFMPEG_URL"
fetch "$OPUS_SHA256" "opus-${OPUS_VERSION}.tar.gz" "$OPUS_URL"
[ "$WITH_LAME" = 1 ] && fetch "$LAME_SHA256" "lame-${LAME_VERSION}.tar.gz" "$LAME_URL" "$LAME_URL_MIRROR"
FFSRC="$SRC/ffmpeg-${FFMPEG_VERSION}"
[ -f "$FFSRC/configure" ] || { echo "ffmpeg source tree missing" >&2; exit 1; }
# LAME's tarball extracts to lame-3.100 whichever URL it came from.

# ---- libopus ---------------------------------------------------------------------------------
echo "== libopus $OPUS_VERSION"
OPUS_CONFIGURE=(--prefix="$DEPS" --disable-shared --enable-static --disable-doc --disable-extra-programs
  ${DEP_HOST_ARGS[@]+"${DEP_HOST_ARGS[@]}"})
mkdir -p "$WORK/build/opus"
(
  cd "$WORK/build/opus"
  CFLAGS="$DEP_CFLAGS" "$SRC/opus-${OPUS_VERSION}/configure" "${OPUS_CONFIGURE[@]}" >configure.log 2>&1 || { tail -n 40 configure.log >&2; exit 1; }
  make -j"$JOBS" >make.log 2>&1 || { tail -n 40 make.log >&2; exit 1; }
  make install >install.log 2>&1
)
[ -f "$DEPS/lib/pkgconfig/opus.pc" ] || { echo "libopus did not install" >&2; exit 1; }

# ---- LAME ------------------------------------------------------------------------------------
LAME_CONFIGURE=()
if [ "$WITH_LAME" = 1 ]; then
  echo "== LAME $LAME_VERSION"
  LAME_CONFIGURE=(--prefix="$DEPS" --disable-shared --enable-static --disable-frontend --disable-decoder
    --disable-gtktest --disable-analyzer-hooks --disable-dependency-tracking ${DEP_HOST_ARGS[@]+"${DEP_HOST_ARGS[@]}"})
  # LAME has no --disable-stack-protector; drop the opus-only flag if present.
  filtered=()
  for a in "${LAME_CONFIGURE[@]}"; do [ "$a" = --disable-stack-protector ] || filtered+=("$a"); done
  LAME_CONFIGURE=("${filtered[@]}")
  mkdir -p "$WORK/build/lame"
  (
    cd "$WORK/build/lame"
    CFLAGS="$DEP_CFLAGS" "$SRC/lame-${LAME_VERSION}/configure" "${LAME_CONFIGURE[@]}" >configure.log 2>&1 || { tail -n 40 configure.log >&2; exit 1; }
    make -j"$JOBS" >make.log 2>&1 || { tail -n 40 make.log >&2; exit 1; }
    make install >install.log 2>&1
  )
  [ -f "$DEPS/lib/libmp3lame.a" ] || { echo "LAME did not install" >&2; exit 1; }
fi

# ---- FFmpeg ----------------------------------------------------------------------------------
echo "== FFmpeg $FFMPEG_VERSION ($PLATFORM)"
# Every PCM / ADPCM decoder FFmpeg knows (all LGPL), minus the Apple AudioToolbox/MediaCodec wrappers.
PCM_DECODERS="$(cd "$FFSRC" && ./configure --list-decoders | tr -s '[:space:]' '\n' | grep -E '^(pcm|adpcm)_' | grep -vE '_(at|mediacodec)$' | tr '\n' ',' | sed 's/,$//')"
[ -n "$PCM_DECODERS" ] || { echo "could not list pcm/adpcm decoders" >&2; exit 1; }

FF_CONFIGURE=(
  --prefix=/usr/local
  --disable-everything --disable-autodetect --disable-debug --disable-doc --disable-network
  --disable-shared --enable-static
  --disable-ffplay --enable-ffmpeg --enable-ffprobe
  --disable-avdevice --disable-swscale
  --enable-avcodec --enable-avformat --enable-avfilter --enable-swresample
  --enable-parsers --enable-bsfs
  "--enable-protocol=$PROTOCOLS"
  "--enable-demuxer=$DEMUXERS"
  "--enable-muxer=$MUXERS"
  "--enable-decoder=$DECODERS,$PCM_DECODERS"
  "--enable-encoder=$ENCODERS"
  "--enable-filter=$FILTERS"
  --enable-libopus
)
[ "$WITH_LAME" = 1 ] && FF_CONFIGURE+=(--enable-libmp3lame "--extra-cflags=-I$DEPS/include" "--extra-ldflags=-L$DEPS/lib")
[ "$WITH_LAME" = 1 ] || FF_CONFIGURE+=("--extra-cflags=-I$DEPS/include")
FF_CONFIGURE+=(${FF_PLATFORM_ARGS[@]+"${FF_PLATFORM_ARGS[@]}"})
export PKG_CONFIG_LIBDIR="$DEPS/lib/pkgconfig"   # only our libopus; never a system one
export PKG_CONFIG_PATH="$PKG_CONFIG_LIBDIR"

mkdir -p "$WORK/build/ffmpeg"
(
  cd "$WORK/build/ffmpeg"
  "$FFSRC/configure" "${FF_CONFIGURE[@]}" --pkg-config-flags=--static >configure.log 2>&1 || { tail -n 60 configure.log >&2; exit 1; }
  make -j"$JOBS" ffmpeg"$EXE" ffprobe"$EXE" >make.log 2>&1 || { tail -n 40 make.log >&2; exit 1; }
  make install-progs DESTDIR="$STAGE" >install.log 2>&1 || { tail -n 40 install.log >&2; exit 1; }
)
for p in ffmpeg ffprobe; do
  [ -f "$STAGE/usr/local/bin/$p$EXE" ] || { echo "$p$EXE was not built" >&2; exit 1; }
  install -m 755 "$STAGE/usr/local/bin/$p$EXE" "$OUT/$p$EXE"
done

# ---- licence texts (shipped next to the build info) --------------------------------------------
cp "$FFSRC/COPYING.LGPLv2.1" "$OUT/COPYING.LGPLv2.1"
cp "$FFSRC/LICENSE.md" "$OUT/FFmpeg-LICENSE.md"
cp "$SRC/opus-${OPUS_VERSION}/COPYING" "$OUT/opus-COPYING"
[ "$WITH_LAME" = 1 ] && cp "$SRC/lame-${LAME_VERSION}/COPYING" "$OUT/lame-COPYING"

# ---- verification: license + no GPL, link check, smoke test ------------------------------------
echo "== verify"
if grep -qE '^#define CONFIG_(GPL|VERSION3|NONFREE) 1' "$WORK/build/ffmpeg/config.h"; then
  echo "GPL/version3/nonfree is enabled in config.h - refusing" >&2
  exit 1
fi
if [ "$EXE" != ".exe" ]; then
  version_out="$("$OUT/ffmpeg" -hide_banner -version)"
  echo "$version_out" | head -n 3
  if echo "$version_out" | grep -qE -- '--enable-(gpl|version3|nonfree)'; then
    echo "built ffmpeg reports a GPL/nonfree configuration" >&2
    exit 1
  fi
  "$OUT/ffmpeg" -hide_banner -L | head -n 4 | tr '\n' ' ' | grep -q 'GNU Lesser General Public License' \
    || { echo "ffmpeg -L does not report the LGPL" >&2; exit 1; }
  # Round trip through every encoder we enable (silence from /dev/zero as the source).
  smoke="$WORK/smoke"
  mkdir -p "$smoke"
  for fmt in flac wav opus m4a ogg; do
    case "$fmt" in
      flac) codec=(-c:a flac) ;;
      wav) codec=(-c:a pcm_s24le) ;;
      opus) codec=(-c:a libopus -b:a 64k) ;;
      m4a) codec=(-c:a aac -b:a 128k) ;;
      ogg) codec=(-c:a libopus) ;;
    esac
    head -c 384000 /dev/zero | "$OUT/ffmpeg" -nostdin -hide_banner -v error -f f32le -ar 48000 -ac 1 -i pipe:0 \
      "${codec[@]}" -y "$smoke/t.$fmt"
    "$OUT/ffprobe" -v error -show_entries format=format_name,duration -of csv=p=0 "$smoke/t.$fmt" | head -n 1
  done
  if [ "$WITH_LAME" = 1 ]; then
    head -c 384000 /dev/zero | "$OUT/ffmpeg" -nostdin -hide_banner -v error -f f32le -ar 48000 -ac 2 -i pipe:0 \
      -c:a libmp3lame -b:a 256k -y "$smoke/t.mp3"
    "$OUT/ffprobe" -v error -show_entries format=format_name -of csv=p=0 "$smoke/t.mp3"
  fi
  # Link check: only system libraries.
  case "$HOST_OS" in
    Darwin)
      if otool -L "$OUT/ffmpeg" "$OUT/ffprobe" | grep -E '^[[:space:]]' | grep -vE '/usr/lib/|/System/Library/'; then
        echo "ffmpeg links a non-system library" >&2
        exit 1
      fi
      ;;
    Linux)
      echo "dynamic libs:" "$(ldd "$OUT/ffmpeg" | awk '{print $1}' | tr '\n' ' ')"
      if ldd "$OUT/ffmpeg" | grep -qE 'libav|libsw|libopus|libmp3lame'; then
        echo "ffmpeg links a shared codec library" >&2
        exit 1
      fi
      ;;
  esac
else
  OBJDUMP="${CROSS_PREFIX}objdump"
  if [ "$HOST_OS" != Linux ]; then # native MSYS2: the programs can run here
    "$OUT/ffmpeg.exe" -hide_banner -L | head -n 4 | tr '\n' ' ' | grep -q 'GNU Lesser General Public License' \
      || { echo "ffmpeg -L does not report the LGPL" >&2; exit 1; }
    if "$OUT/ffmpeg.exe" -hide_banner -version | grep -qE -- '--enable-(gpl|version3|nonfree)'; then
      echo "built ffmpeg reports a GPL/nonfree configuration" >&2
      exit 1
    fi
    head -c 384000 /dev/zero | "$OUT/ffmpeg.exe" -nostdin -hide_banner -v error -f f32le -ar 48000 -ac 1 -i pipe:0 -c:a flac -y "$WORK/t.flac"
    "$OUT/ffprobe.exe" -v error -show_entries format=format_name,duration -of csv=p=0 "$WORK/t.flac" | head -n 1
  fi
  dlls="$("$OBJDUMP" -p "$OUT/ffmpeg.exe" "$OUT/ffprobe.exe" | awk '/DLL Name:/ {print tolower($3)}' | sort -u)"
  echo "imported DLLs: $(echo $dlls)"
  # Only Windows system DLLs may be imported (no libgcc/winpthread/libopus/... of ours).
  bad="$(echo "$dlls" | grep -vE '^(kernel32|user32|advapi32|shell32|ole32|oleaut32|ws2_32|bcrypt|crypt32|msvcrt|ucrtbase|secur32|shlwapi|psapi|gdi32|ntdll|userenv)\.dll$|^api-ms-win-' || true)"
  if [ -n "$bad" ]; then
    echo "unexpected DLL import in the Windows build: $bad" >&2
    exit 1
  fi
fi

# ---- build info (goes into the app's licenses/ folder) -------------------------------------------
INFO="$OUT/ffmpeg-buildinfo.txt"
# Absolute scratch paths are noise for a reader; keep the lines readable and reproducible.
tidy() { sed -e "s|$DEPS|<deps>|g" -e "s|$WORK|<work>|g"; }
{
  cat <<EOF
Audio Cleaner - bundled FFmpeg: licence and build information
==============================================================

The programs ffmpeg$EXE and ffprobe$EXE shipped with Audio Cleaner are built from the
unmodified FFmpeg $FFMPEG_VERSION source release. They are licensed under the GNU Lesser General
Public License, version 2.1 or later (see COPYING.LGPLv2.1). They were configured without
--enable-gpl, --enable-version3 and --enable-nonfree. FFmpeg is a trademark of Fabrice Bellard,
originator of the FFmpeg project; this software is not endorsed by the FFmpeg project.

Audio Cleaner runs ffmpeg$EXE and ffprobe$EXE as separate programs (it does not link to them). The
libraries inside them (libavcodec, libavformat, libavfilter, libavutil, libswresample) are
linked statically into those two executables. To use a different FFmpeg build, replace the two
files in the "native" folder of the installation with your own (same file names).

Source code and build: platform $PLATFORM
------------------------------------------------------------------------------------------
Component  Version   Licence                      Source (sha256 of the tarball)
FFmpeg     $FFMPEG_VERSION    LGPL-2.1-or-later            $FFMPEG_URL
                                                     $FFMPEG_SHA256
libopus    $OPUS_VERSION      BSD-3-Clause                 $OPUS_URL
                                                     $OPUS_SHA256
EOF
  if [ "$WITH_LAME" = 1 ]; then
    cat <<EOF
LAME       $LAME_VERSION      LGPL-2.0-or-later            $LAME_URL
                                                     (mirror: $LAME_URL_MIRROR)
                                                     $LAME_SHA256
EOF
  fi
  cat <<EOF

No patches are applied to any of the three source trees.

Corresponding source: the tarballs above are the complete corresponding source. If a link is
ever unavailable, ask the person who gave you this app (Benjamin Danker) and
we will provide the exact source archives for any version of this software we have distributed,
for at least three years after our last distribution of that version.

Toolchain
---------
EOF
  if [ -n "$CROSS_PREFIX" ]; then
    "${CROSS_PREFIX}gcc" --version | head -n 1
  elif [ "$HOST_OS" = Darwin ]; then
    cc --version | head -n 1
    echo "MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET}"
  else
    gcc --version | head -n 1
  fi
  echo "x86 assembler: $(if command -v nasm >/dev/null 2>&1; then nasm -v | head -n 1; else echo none; fi)"
  cat <<EOF

Configure commands (run from an empty build directory; <deps> is a scratch prefix into which
libopus and LAME are installed first; <work>/src holds the extracted tarballs)
------------------------------------------------------------------------------------------
libopus:
  CFLAGS="$DEP_CFLAGS" <opus-source>/configure ${OPUS_CONFIGURE[*]}
EOF
  if [ "$WITH_LAME" = 1 ]; then
    echo "LAME:"
    echo "  CFLAGS=\"$DEP_CFLAGS\" <lame-source>/configure ${LAME_CONFIGURE[*]}"
  fi
  cat <<EOF
FFmpeg (PKG_CONFIG_LIBDIR=<deps>/lib/pkgconfig):
  <ffmpeg-source>/configure ${FF_CONFIGURE[*]} --pkg-config-flags=--static
then: make ffmpeg$EXE ffprobe$EXE

FFmpeg's own report of this configuration is printed by "ffmpeg -buildconf" (and "ffmpeg -L"
prints the licence). The complete build script follows; it needs only a C compiler, make,
pkg-config, curl, tar and (on x86) nasm.

------------------------------------------------------------------------------------------
build_ffmpeg.sh
------------------------------------------------------------------------------------------
EOF
  cat "$HERE/build_ffmpeg.sh"
} | tidy >"$INFO"

echo "== done: $OUT"
ls -l "$OUT" | sed 's/^/   /'
