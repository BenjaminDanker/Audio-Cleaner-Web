#!/usr/bin/env bash
# Developer ID signing of a PyInstaller "Audio Cleaner.app", inside-out, hardened runtime + secure timestamp.
#
#   sign_app.sh "dist/Audio Cleaner.app" "Developer ID Application: Name (TEAMID)" [entitlements.plist]
#
# The identity must be in a keychain on the search list. Every Mach-O file is signed first (deepest path
# first), then the .framework folders, then the main executable and the bundle with the entitlements.
set -euo pipefail

[ $# -ge 2 ] || { echo "usage: $0 APP IDENTITY [ENTITLEMENTS]" >&2; exit 2; }
APP="${1%/}"
IDENTITY="$2"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENT="${3:-$HERE/entitlements.plist}"
[ -d "$APP" ] || { echo "app bundle not found: $APP" >&2; exit 1; }
[ -f "$ENT" ] || { echo "entitlements not found: $ENT" >&2; exit 1; }

sign() { codesign --force --options runtime --timestamp --sign "$IDENTITY" "$@"; }

echo "== signing Mach-O files"
count=0
while IFS= read -r f; do
  sign "$f"
  count=$((count + 1))
done < <(
  find "$APP" -type f -print0 |
    while IFS= read -r -d '' f; do
      case "$(file -b "$f")" in *Mach-O*) printf '%s\n' "$f" ;; esac
    done |
    awk '{ print length($0) "\t" $0 }' | sort -rn | cut -f2-
)
echo "   $count files"

echo "== signing frameworks"
while IFS= read -r d; do
  sign "$d"
done < <(find "$APP" -type d -name '*.framework' | awk '{ print length($0) "\t" $0 }' | sort -rn | cut -f2-)

echo "== signing the app"
sign --entitlements "$ENT" "$APP"
codesign --verify --deep --strict --verbose=2 "$APP"
echo "signed: $APP"
