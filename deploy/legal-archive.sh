#!/usr/bin/env bash
# The archive of every published version of the Terms of Service and the Privacy Policy (docs/operations.md, "Legal versions").
#   deploy/legal-archive.sh            render both pages (web/scripts/legal-snapshot.mjs) and save them as acw's files
#                                      ~acw/audio-cleaner/data/legal/{terms,privacy}-<version>.html and ...-<version>-no-contact.html (the
#                                      contact form off), each with a .html.sha256 (the SHA-256 of the page's text) next to it
#   deploy/legal-archive.sh --check    the same checks, writes nothing (install.sh runs this first, publish-web.sh saves)
#   deploy/legal-archive.sh --dir DIR  use DIR instead of ~acw/audio-cleaner/data/legal, written as the current account (a dry run)
# A saved version is never overwritten. If the file for a version exists and its text hash differs from what the page renders now,
# nothing is written and the script exits 1: the text was changed without a new version date. Same hash: nothing to do. It also stops
# when web/legal-versions.json (the pin the server records with every acceptance) is not what the pages render, or when the Terms' dates
# disagree (web/tests/legal-versions.mjs checks the same). Run as the owner: the files are written as the service account, the way
# install.sh writes acw's files. Needs node and web/node_modules (npm ci is run when they are missing).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=deploy/lib.sh
. "$REPO/deploy/lib.sh"

CHECK_ONLY=0
DIR="${ACW_LEGAL_DIR:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --check) CHECK_ONLY=1 ;;
    --dir) shift; DIR="${1:?--dir needs a folder}" ;;
    --dir=*) DIR="${1#*=}" ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
if [ -n "$DIR" ]; then
  as_acw() { "$@"; }   # a folder named on the command line (a test, a dry run) is written by whoever runs this
else
  DIR="$H/data/legal"
fi

command -v node >/dev/null 2>&1 || { echo "legal archive: node is needed to render the Terms and the Privacy Policy" >&2; exit 1; }
SNAP="$REPO/web/scripts/legal-snapshot.mjs"
[ -d "$REPO/web/node_modules/react-dom" ] && [ -d "$REPO/web/node_modules/esbuild" ] || (cd "$REPO/web" && npm ci --no-audit --no-fund)

# 1. The pin and the version dates must be right before anything is archived.
node "$SNAP" --verify || { echo "legal archive: fix the above (npm run legal:pin in web/ after changing a version date), then deploy again" >&2; exit 1; }

# 2. Render into a scratch folder; compare each page with what is archived for its version.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
LIST="$(node "$SNAP" --out "$TMP")"
declare -a TODO=()
bad=0
while read -r key version sha file <&3; do
  [ -n "$key" ] || continue
  html="$DIR/$file"
  have=""
  if as_acw test -f "$html.sha256"; then
    have="$(as_acw cat "$html.sha256" | tr -d '[:space:]')"
  elif as_acw test -f "$html"; then
    have="$(as_acw cat "$html" | node "$SNAP" --hash-of - 2>/dev/null || true)"   # (a page whose .sha256 was lost)
    [ -n "$have" ] || have="unreadable"
  fi
  if [ -z "$have" ]; then
    TODO+=("$key $version $sha $file page")
    echo "legal archive: $key $version is new (sha256 ${sha:0:12})"
  elif [ "$have" = "$sha" ]; then
    if as_acw test -f "$html"; then
      echo "legal archive: $key $version is archived already, same text (nothing to do)"
      as_acw test -f "$html.sha256" || TODO+=("$key $version $sha $file hash")   # (the page is kept as it is; only its hash file is missing)
    else
      TODO+=("$key $version $sha $file page")
      echo "legal archive: $key $version: only its hash file was there, saving the page again (same text)"
    fi
  else
    bad=1
    cat >&2 <<EOF
legal archive: STOP. $html exists with text hash ${have:0:12}, but the $key page now renders as ${sha:0:12} under the same version date ($version).
  A published version is never overwritten. Change the version date in the page (for the Terms also LEGAL_VERSION in web/src/components/Legal.jsx and
  TOS_VERSION in server/acw/auth.py), run 'npm run legal:pin' in web/, commit, and deploy again. If the old wording is what you want, restore it.
EOF
  fi
done 3<<< "$LIST"
[ "$bad" = 0 ] || exit 1
[ "$CHECK_ONLY" = 1 ] && exit 0

# 3. Save what is new (the page first, its hash after, each atomically; a half-saved pair is completed by the next run).
if [ "${#TODO[@]}" -gt 0 ]; then
  as_acw mkdir -p -m 700 "$DIR"
  for t in "${TODO[@]}"; do
    read -r key version sha file what <<< "$t"
    if [ "$what" = page ]; then
      put_as_acw "$TMP/$file" "$DIR/$file" 600
      echo "legal archive: saved $DIR/$file"
    else
      echo "legal archive: saved $DIR/$file.sha256 (the page itself is kept as it was)"
    fi
    printf '%s\n' "$sha" | put_as_acw /dev/stdin "$DIR/$file.sha256" 600
  done
fi
