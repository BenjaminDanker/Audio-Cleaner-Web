#!/usr/bin/env bash
# Builds Audio Cleaner.app and its installer package (.pkg) on a Mac, the same steps as the release build.
#
#   desktop/packaging/build_macos.sh
#
# Needs: Xcode command line tools (xcode-select --install), Homebrew (brew install python@3.12 pkgconf
# [nasm on Intel]), Rust (https://rustup.rs), git. Builds for the architecture of this Mac.
#
# Environment (all optional):
#   WORK_DIR             build directory                      (default <repo>/build/macos)
#   WHISPER_MODEL_DIR    local copy of Systran/faster-whisper-small (default: downloaded at the pinned revision)
#   SKIP_TESTS=1         skip pytest
#   MACOS_SIGN_IDENTITY  "Developer ID Application: Name (TEAMID)" (in your keychain) -> sign the app
#   NOTARY_PROFILE       `xcrun notarytool store-credentials` profile (or APPLE_ID, APPLE_TEAM_ID, APPLE_APP_PASSWORD)
#                        -> notarize and staple (needs MACOS_SIGN_IDENTITY)
# Result: $WORK_DIR/out/AudioCleaner-macOS-<arm64|x86_64>.pkg
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
[ "$(uname -s)" = Darwin ] || { echo "run this on a Mac" >&2; exit 1; }
ARCH="$(uname -m)"
PLATFORM="macos-$ARCH"
WORK="${WORK_DIR:-$REPO/build/macos}"
mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"

echo "== prerequisites"
command -v git >/dev/null || { echo "git is missing" >&2; exit 1; }
xcode-select -p >/dev/null 2>&1 || { echo "Xcode command line tools missing: xcode-select --install" >&2; exit 1; }
PYTHON="${PYTHON:-$(command -v python3.12 || true)}"
[ -n "$PYTHON" ] || { echo "Python 3.12 missing: brew install python@3.12" >&2; exit 1; }
command -v pkg-config >/dev/null || { echo "pkg-config missing: brew install pkgconf" >&2; exit 1; }
if [ "$ARCH" = x86_64 ]; then command -v nasm >/dev/null || { echo "nasm missing: brew install nasm" >&2; exit 1; }; fi
# shellcheck disable=SC1091
command -v cargo >/dev/null || [ ! -f "$HOME/.cargo/env" ] || source "$HOME/.cargo/env"
command -v cargo >/dev/null || { echo "Rust missing: https://rustup.rs" >&2; exit 1; }

echo "== libDF (DeepFilterNet C library)"
BUILD_DIR="$WORK/deepfilter" OUT="$WORK/libdf" bash native/deepfilter/build.sh native

echo "== FFmpeg (LGPL, built from source; cached)"
ff_stamp="$(shasum -a 256 desktop/packaging/build_ffmpeg.sh | cut -c1-16)"
if [ ! -f "$WORK/ffmpeg/.stamp-$ff_stamp" ]; then
  rm -rf "$WORK/ffmpeg"
  DL_DIR="$WORK/dl" bash desktop/packaging/build_ffmpeg.sh "$PLATFORM" "$WORK/ffmpeg"
  touch "$WORK/ffmpeg/.stamp-$ff_stamp"
fi

echo "== Python environment"
[ -x "$WORK/venv/bin/python" ] || "$PYTHON" -m venv "$WORK/venv"
PY="$WORK/venv/bin/python"
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q --require-hashes -r desktop/requirements.lock -r desktop/requirements-build.lock

echo "== vendor directory"
export PYTHONPATH="$REPO/desktop:$REPO/common" ACW_DESKTOP_VENDOR="$WORK/vendor" ACW_ICON_DIR="$WORK/icons"
"$PY" desktop/packaging/fetch_dpdfnet.py "$WORK/dpdfnet"
whisper=(--download-whisper)
[ -z "${WHISPER_MODEL_DIR:-}" ] || whisper=(--whisper-model "$WHISPER_MODEL_DIR")
"$PY" desktop/packaging/prepare_vendor.py --libdf "$WORK/libdf/libdf.dylib" --ffmpeg-dir "$WORK/ffmpeg" \
  --model "$WORK/deepfilter/models/DeepFilterNet3_onnx.tar.gz" --deepfilter-src "$WORK/deepfilter" \
  --dpdfnet "$WORK/dpdfnet/dpdfnet8_48khz_hr_batch.onnx" \
  --rust-target "$(rustc -vV | sed -n 's/^host: //p')" "${whisper[@]}" --out "$WORK/vendor"

if [ "${SKIP_TESTS:-0}" != 1 ]; then
  echo "== tests"
  QT_QPA_PLATFORM=offscreen "$PY" -m pytest -q common/tests desktop/tests
fi

echo "== app bundle"
"$PY" desktop/packaging/make_icons.py "$ACW_ICON_DIR"
rm -rf "$WORK/dist"
"$WORK/venv/bin/pyinstaller" desktop/packaging/audio_cleaner.spec --distpath "$WORK/dist" --workpath "$WORK/pyinstaller" --noconfirm
APP="$WORK/dist/Audio Cleaner.app"

echo "== frozen selftest"
"$APP/Contents/MacOS/AudioCleaner" --selftest --gui | tee "$WORK/selftest.log"
grep -q "selftest ok" "$WORK/selftest.log"

if [ -n "${MACOS_SIGN_IDENTITY:-}" ]; then
  echo "== signing"
  bash desktop/packaging/macos/sign_app.sh "$APP" "$MACOS_SIGN_IDENTITY"
  if [ -n "${NOTARY_PROFILE:-}${APPLE_ID:-}" ]; then bash desktop/packaging/macos/notarize.sh "$APP"; fi
else
  echo "(no MACOS_SIGN_IDENTITY: the app is not signed; Gatekeeper will ask the user to confirm)"
fi

echo "== installer package (its setup downloads the models)"
mkdir -p "$WORK/out"
PKG="$WORK/out/AudioCleaner-macOS-$ARCH.pkg"
"$PY" desktop/packaging/installer_models.py --mac "$WORK/pkg-scripts"
PKG_ARCH="$ARCH" bash desktop/packaging/macos/make_pkg.sh "$APP" "$PKG" "$("$PY" -c "from audio_cleaner_desktop import __version__ as v; print(v, end='')")" "$WORK/pkg-scripts"
echo "== done: $PKG"
