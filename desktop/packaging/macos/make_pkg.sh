#!/usr/bin/env bash
# Packs "Audio Cleaner.app" into an installer package (.pkg) that installs it into /Applications and,
# during setup, downloads the models (postinstall from installer_models.py --mac).
#
#   make_pkg.sh "dist/Audio Cleaner.app" AudioCleaner-macOS-arm64.pkg VERSION [SCRIPTS_DIR]
#
# Sign the .app (and notarize it) before running this. Signing the .pkg itself needs a "Developer ID
# Installer" certificate (productsign); without one, macOS asks the user to confirm opening it. Needs
# macOS (pkgbuild, plutil).
set -euo pipefail

[ $# -ge 3 ] || { echo "usage: $0 APP OUT.pkg VERSION [SCRIPTS_DIR]" >&2; exit 2; }
APP="${1%/}"
OUT="$2"
VERSION="$3"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ "$(uname -s)" = Darwin ] || { echo "make_pkg.sh needs macOS" >&2; exit 1; }
[ -d "$APP" ] || { echo "app bundle not found: $APP" >&2; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/root"
ditto "$APP" "$WORK/root/$(basename "$APP")"  # keeps symlinks, permissions and code signatures
SCRIPTS="${4:-$WORK/scripts}"
if [ -z "${4:-}" ]; then
  python3 "$HERE/../installer_models.py" --mac "$SCRIPTS"
fi
[ -x "$SCRIPTS/postinstall" ] || { echo "no postinstall script in $SCRIPTS" >&2; exit 1; }

# Always install into /Applications (not wherever an older copy happens to be).
pkgbuild --analyze --root "$WORK/root" "$WORK/component.plist"
plutil -replace 0.BundleIsRelocatable -bool false "$WORK/component.plist"

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"
pkgbuild --root "$WORK/root" --component-plist "$WORK/component.plist" --install-location /Applications \
  --scripts "$SCRIPTS" --identifier local.audio-cleaner.desktop --version "$VERSION" "$OUT"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
