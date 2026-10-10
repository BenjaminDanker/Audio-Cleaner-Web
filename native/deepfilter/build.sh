#!/usr/bin/env bash
# Reproducible build of DeepFilterNet's libDF with tract 0.23 and the quality fix, as
#   a) a native shared library (libdf.so, C API used from Python via ctypes) and
#   b) a WebAssembly module (df.js + df_bg.wasm, wasm-bindgen "web" target, SIMD128).
# Neither embeds the DFN3 model; pass the tar.gz at runtime.
#
#   native/deepfilter/build.sh [native] [wasm]        # default: both
#
# Environment overrides:
#   BUILD_DIR  checkout / build directory   (default ~/audio-cleaner/src/deepfilter-build)
#   OUT        output directory             (default $BUILD_DIR/out -> $OUT/libdf.so, $OUT/wasm/*)
#   CARGO_TARGET_DIR  honoured as usual (default $BUILD_DIR/target)
#
# Idempotent: every run resets the checkout to the pinned commit, re-applies tract-0.23.patch and
# Cargo.lock from this directory and rebuilds (cargo's incremental cache in $BUILD_DIR/target is
# kept). Needs network on first run (git clone, crates, wasm-pack/wasm-bindgen/wasm-opt).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM=https://github.com/Rikorose/DeepFilterNet.git
COMMIT=d375b2d8309e0935d165700c91da9de862a99c31
BUILD_DIR="${BUILD_DIR:-$HOME/audio-cleaner/src/deepfilter-build}"
OUT="${OUT:-$BUILD_DIR/out}"
WASM_PACK_VERSION=0.15.0
WASM_PACK_SHA256_X86_64_LINUX=c09f971ecaed9a2efc80fdcea7a00ef6b53c7fadc8c57d1f61b53a6aa66b668a

targets=("$@")
[ ${#targets[@]} -gt 0 ] || targets=(native wasm)
want() { local t; for t in "${targets[@]}"; do [ "$t" = "$1" ] && return 0; done; return 1; }
for t in "${targets[@]}"; do
  case "$t" in native | wasm) ;; *) echo "unknown target '$t' (native|wasm)" >&2; exit 2 ;; esac
done

# shellcheck disable=SC1091
command -v cargo >/dev/null 2>&1 || source "$HOME/.cargo/env"
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
  mkdir -p "$(dirname "$BUILD_DIR")"
  git clone --quiet "$UPSTREAM" "$BUILD_DIR"
fi
cd "$BUILD_DIR"
git cat-file -e "$COMMIT^{commit}" 2>/dev/null || git fetch --quiet origin
git checkout --quiet --force "$COMMIT"
git reset --quiet --hard "$COMMIT" # drops a previous patch application (target/ is untracked, kept)
git apply "$HERE/tract-0.23.patch"
cp "$HERE/Cargo.lock" Cargo.lock    # pinned dependency set (tract 0.23.8, wasm-bindgen 0.2.118, ...)
echo "== DeepFilterNet $(git rev-parse --short HEAD) + tract-0.23.patch in $BUILD_DIR"

TARGET_DIR="${CARGO_TARGET_DIR:-$BUILD_DIR/target}"
mkdir -p "$OUT"

# ---- native ----------------------------------------------------------------------------------
if want native; then
  echo "== native: cdylib, feature capi (no embedded model), profile release-capi"
  # release-capi = release-lto with panic=unwind so the C API can catch panics (see the patch).
  cargo build --locked --profile release-capi -p deep_filter --no-default-features --features capi --lib
  install -m 755 "$TARGET_DIR/release-capi/libdf.so" "$OUT/libdf.so"
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

( cd "$OUT" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS )
echo "== done. outputs in $OUT:"
ls -l "$OUT" "$OUT/wasm" 2>/dev/null | sed 's/^/   /'
