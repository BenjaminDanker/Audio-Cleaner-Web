#!/usr/bin/env bash
# Notarizes a signed .app or .dmg with Apple and staples the ticket.
#
#   notarize.sh FILE
#
# Credentials, either an app-specific password:   APPLE_ID, APPLE_TEAM_ID, APPLE_APP_PASSWORD
# or a keychain profile made with `xcrun notarytool store-credentials`:   NOTARY_PROFILE
set -euo pipefail

[ $# -eq 1 ] || { echo "usage: $0 FILE.app|FILE.dmg" >&2; exit 2; }
FILE="${1%/}"
[ -e "$FILE" ] || { echo "not found: $FILE" >&2; exit 1; }

if [ -n "${NOTARY_PROFILE:-}" ]; then
  CRED=(--keychain-profile "$NOTARY_PROFILE")
else
  : "${APPLE_ID:?set APPLE_ID, APPLE_TEAM_ID and APPLE_APP_PASSWORD (or NOTARY_PROFILE)}"
  : "${APPLE_TEAM_ID:?}" "${APPLE_APP_PASSWORD:?}"
  CRED=(--apple-id "$APPLE_ID" --team-id "$APPLE_TEAM_ID" --password "$APPLE_APP_PASSWORD")
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
SUBMIT="$FILE"
if [ -d "$FILE" ]; then
  SUBMIT="$TMP/$(basename "$FILE").zip"
  ditto -c -k --keepParent "$FILE" "$SUBMIT"
fi

echo "== submitting $(basename "$FILE") to Apple (waits for the result)"
set +e
xcrun notarytool submit "$SUBMIT" "${CRED[@]}" --wait --output-format json >"$TMP/result.json"
rc=$?
set -e
cat "$TMP/result.json"
id="$(sed -n 's/.*"id" *: *"\([^"]*\)".*/\1/p' "$TMP/result.json" | head -n 1)"
if [ "$rc" -ne 0 ] || ! grep -q '"status" *: *"Accepted"' "$TMP/result.json"; then
  [ -n "$id" ] && xcrun notarytool log "$id" "${CRED[@]}" || true
  echo "notarization failed" >&2
  exit 1
fi
xcrun stapler staple "$FILE"
xcrun stapler validate "$FILE"
echo "notarized and stapled: $FILE"
