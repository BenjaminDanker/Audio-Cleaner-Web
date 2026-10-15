#!/usr/bin/env bash
# Packs "Audio Cleaner.app" into a compressed disk image with an /Applications shortcut.
#
#   make_dmg.sh "dist/Audio Cleaner.app" AudioCleaner-macOS-arm64.dmg [volume name]
#
# Sign the .app (and notarize / staple it) before running this; sign and notarize the .dmg afterwards if
# you want it stapled too. Needs macOS (hdiutil).
set -euo pipefail

[ $# -ge 2 ] || { echo "usage: $0 APP OUT.dmg [VOLUME_NAME]" >&2; exit 2; }
APP="${1%/}"
OUT="$2"
VOLNAME="${3:-Audio Cleaner}"

[ "$(uname -s)" = Darwin ] || { echo "make_dmg.sh needs macOS" >&2; exit 1; }
[ -d "$APP" ] || { echo "app bundle not found: $APP" >&2; exit 1; }
case "$APP" in *.app) ;; *) echo "APP must be a .app bundle: $APP" >&2; exit 1 ;; esac

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
# ditto keeps symlinks, permissions, extended attributes and code signatures intact.
ditto "$APP" "$STAGE/$(basename "$APP")"
ln -s /Applications "$STAGE/Applications"

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"
hdiutil create -volname "$VOLNAME" -srcfolder "$STAGE" -fs HFS+ -format UDZO -imagekey zlib-level=9 -ov "$OUT"
hdiutil verify "$OUT" >/dev/null
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
