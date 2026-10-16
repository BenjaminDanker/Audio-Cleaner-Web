#!/usr/bin/env bash
# Builds the native live engine (acw-live) and installs it next to libdf.so.
#   native/live/build.sh [install-dir]      default install dir: ~/audio-cleaner/bin
# It links the sherpa-onnx C library that the Python venv already ships (same version as the Python bindings, so the
# Python worker and this engine run the very same caption code), found from the venv's python.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HOME/audio-cleaner/bin}"
PY="${ACW_PYTHON:-$HOME/audio-cleaner/venv/bin/python}"
command -v cargo >/dev/null 2>&1 || source "$HOME/.cargo/env"
SHERPA_ONNX_LIB_DIR="$("$PY" -c 'import os, sherpa_onnx; print(os.path.join(os.path.dirname(sherpa_onnx.__file__), "lib"))')"
[ -f "$SHERPA_ONNX_LIB_DIR/libsherpa-onnx-c-api.so" ] || { echo "no libsherpa-onnx-c-api.so in $SHERPA_ONNX_LIB_DIR" >&2; exit 1; }
export SHERPA_ONNX_LIB_DIR
cd "$HERE"
cargo build --release --locked 2>&1 | tail -3
install -m 755 "${CARGO_TARGET_DIR:-$HERE/target}/release/acw-live" "$OUT/acw-live"
echo "installed $OUT/acw-live (sherpa-onnx from $SHERPA_ONNX_LIB_DIR)"
