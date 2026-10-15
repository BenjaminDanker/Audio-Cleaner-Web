#!/usr/bin/env bash
# Reproducible build of DeepFilterNet's libDF with tract 0.23 and the quality fix, as
#   a) a native shared library (C API used from Python via ctypes; the platform's name:
#      libdf.so on Linux, libdf.dylib on macOS, df.dll on Windows = acw_common.dfn.LIB_NAME) and
#   b) a WebAssembly module (df.js + df_bg.wasm, wasm-bindgen "web" target, SIMD128).
# Neither embeds the DFN3 model; pass the tar.gz at runtime.
#
#   native/deepfilter/build.sh [native] [wasm]        # default: both
#
# Environment overrides:
#   BUILD_DIR  checkout / build directory   (default ~/audio-cleaner/src/deepfilter-build)
#   OUT        output directory             (default $BUILD_DIR/out -> $OUT/<libdf>, $OUT/wasm/*)
#   CARGO_TARGET_DIR  honoured as usual (default $BUILD_DIR/target)
#   CARGO_BUILD_TARGET  optional Rust target triple, e.g. x86_64-apple-darwin on an arm64 Mac
#                       (installed with rustup if missing); output then lives in
#                       $CARGO_TARGET_DIR/<triple>/release-capi, and the library name follows the triple
#   MACOSX_DEPLOYMENT_TARGET  macOS only, default 11.0
#
# Runs on Linux, macOS and Windows (Git Bash, MSVC toolchain). Idempotent: every run resets the
# checkout to the pinned commit, re-applies tract-0.23.patch and Cargo.lock from this directory
# and rebuilds (cargo's incremental cache in $BUILD_DIR/target is kept). Needs network on first
# run (git clone, crates, wasm-pack/wasm-bindgen/wasm-opt).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM=https://github.com/Rikorose/DeepFilterNet.git
COMMIT=d375b2d8309e0935d165700c91da9de862a99c31
BUILD_DIR="${BUILD_DIR:-$HOME/audio-cleaner/src/deepfilter-build}"
OUT="${OUT:-$BUILD_DIR/out}"
WASM_PACK_VERSION=0.15.0
WASM_PACK_SHA256_X86_64_LINUX=c09f971ecaed9a2efc80fdcea7a00ef6b53c7fadc8c57d1f61b53a6aa66b668a

# sha256 of a file list on stdin-free systems: GNU coreutils or macOS' shasum.
if command -v sha256sum >/dev/null 2>&1; then
  SHA256=(sha256sum)
elif command -v shasum >/dev/null 2>&1; then
  SHA256=(shasum -a 256)
else
  echo "sha256sum or shasum is required" >&2
  exit 1
fi

targets=("$@")
[ ${#targets[@]} -gt 0 ] || targets=(native wasm)
want() { local t; for t in "${targets[@]}"; do [ "$t" = "$1" ] && return 0; done; return 1; }
for t in "${targets[@]}"; do
  case "$t" in native | wasm) ;; *) echo "unknown target '$t' (native|wasm)" >&2; exit 2 ;; esac
done

# shellcheck disable=SC1091
command -v cargo >/dev/null 2>&1 || source "$HOME/.cargo/env"
case "$(uname -s)" in
  Darwin) export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}" ;;
esac
command -v git >/dev/null 2>&1 || { echo "git is required" >&2; exit 1; }

# ---- toolchain -------------------------------------------------------------------------------
if want wasm; then
  if ! rustup target list --installed | grep -qx wasm32-unknown-unknown; then
    rustup target add wasm32-unknown-unknown
  fi
  if ! command -v wasm-pack >/dev/null 2>&1; then
    if [ "$(uname -sm)" = "Linux x86_64" ]; then
      # Prebuilt release binary, pinned by sha256 (much faster than `cargo install`).
      tmp="$(mktemp -d)"
      trap 'rm -rf "$tmp"' EXIT
      name="wasm-pack-v${WASM_PACK_VERSION}-x86_64-unknown-linux-musl"
      curl -fsSL -o "$tmp/wp.tgz" \
        "https://github.com/wasm-bindgen/wasm-pack/releases/download/v${WASM_PACK_VERSION}/${name}.tar.gz"
      echo "${WASM_PACK_SHA256_X86_64_LINUX}  $tmp/wp.tgz" | sha256sum -c -
      mkdir -p "$HOME/.cargo/bin"
      tar xzf "$tmp/wp.tgz" --strip-components=1 -C "$HOME/.cargo/bin" "$name/wasm-pack"
    else
      cargo install wasm-pack --locked --version "$WASM_PACK_VERSION"
    fi
  fi
fi

# ---- sources ---------------------------------------------------------------------------------
if [ ! -d "$BUILD_DIR/.git" ]; then
  # init + fetch rather than clone: the folder may already hold a restored cargo cache (CI's target/),
  # and only the pinned commit is needed.
  mkdir -p "$BUILD_DIR"
  git -C "$BUILD_DIR" init --quiet
  git -C "$BUILD_DIR" remote add origin "$UPSTREAM"
fi
cd "$BUILD_DIR"
# core.autocrlf=false: on Windows a CRLF checkout would make tract-0.23.patch fail to apply.
git config core.autocrlf false
git cat-file -e "$COMMIT^{commit}" 2>/dev/null || git fetch --quiet --depth 1 origin "$COMMIT"
git checkout --quiet --force "$COMMIT"
git reset --quiet --hard "$COMMIT" # drops a previous patch application (target/ is untracked, kept)
git apply "$HERE/tract-0.23.patch"
cp "$HERE/Cargo.lock" Cargo.lock    # pinned dependency set (tract 0.23.8, wasm-bindgen 0.2.118, ...)
echo "== DeepFilterNet $(git rev-parse --short HEAD) + tract-0.23.patch in $BUILD_DIR"

TARGET_DIR="${CARGO_TARGET_DIR:-$BUILD_DIR/target}"
mkdir -p "$OUT"

# Library file name per platform (the Rust target if cross-building, else the host).
case "${CARGO_BUILD_TARGET:-$(uname -s)}" in
  *apple-darwin* | Darwin) LIB_FILE=libdf.dylib ;;
  *windows* | MINGW* | MSYS* | CYGWIN*) LIB_FILE=df.dll ;;
  *) LIB_FILE=libdf.so ;;
esac
# With CARGO_BUILD_TARGET cargo puts the artifacts in a per-triple subdirectory.
PROFILE_DIR="$TARGET_DIR${CARGO_BUILD_TARGET:+/$CARGO_BUILD_TARGET}/release-capi"

# ---- native ----------------------------------------------------------------------------------
if want native; then
  echo "== native: cdylib $LIB_FILE, feature capi (no embedded model), profile release-capi"
  if [ -n "${CARGO_BUILD_TARGET:-}" ] && command -v rustup >/dev/null 2>&1 &&
    ! rustup target list --installed | grep -qx "$CARGO_BUILD_TARGET"; then
    rustup target add "$CARGO_BUILD_TARGET"
  fi
  # release-capi = release-lto with panic=unwind so the C API can catch panics (see the patch).
  cargo build --locked --profile release-capi -p deep_filter --no-default-features --features capi --lib
  [ -f "$PROFILE_DIR/$LIB_FILE" ] || { echo "cargo did not produce $PROFILE_DIR/$LIB_FILE" >&2; exit 1; }
  install -m 755 "$PROFILE_DIR/$LIB_FILE" "$OUT/$LIB_FILE"
fi

# ---- wasm ------------------------------------------------------------------------------------
if want wasm; then
  echo "== wasm: wasm-pack web target, simd128, feature wasm (no embedded model)"
  rm -rf libDF/pkg
  RUSTFLAGS="-C target-feature=+simd128" \
    wasm-pack build libDF --target web --release --out-dir pkg \
    --no-default-features --features wasm --locked
  mkdir -p "$OUT/wasm"
  for f in df.js df_bg.wasm df.d.ts df_bg.wasm.d.ts LICENSE-MIT LICENSE-APACHE; do
    install -m 644 "libDF/pkg/$f" "$OUT/wasm/$f"
  done
fi

( cd "$OUT" && find . -type f ! -name SHA256SUMS | sort | xargs "${SHA256[@]}" > SHA256SUMS )
echo "== done. outputs in $OUT:"
# (the wasm folder only exists when it was built; ls of a missing path would fail the whole build)
for d in "$OUT" "$OUT/wasm"; do [ -d "$d" ] && ls -l "$d" | sed 's/^/   /'; done
true
