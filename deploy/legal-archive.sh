#!/usr/bin/env bash
# Versions the Terms of Service and the Privacy Policy by themselves (docs/operations.md, "Legal versions"): nothing to run or bump when a
# text changes; install.sh and publish-web.sh call this.
#   deploy/legal-archive.sh            render both pages (web/scripts/legal-snapshot.mjs), hash their text and look the hash up in the
#                                      archive ~acw/audio-cleaner/data/legal/. A text that is archived keeps its version. A new text gets
#                                      today's date as its version (<date>-2, -3 ... for a further change the same day) and is saved there as
#                                      {terms,privacy}-<version>.html with a .html.sha256 next to it, never over a file that exists. Then
#                                      data/legal/current.json (the versions now live and the hash of each text: the server reads it) is
#                                      written. Prints ACW_LEGAL_TERMS_VERSION=... and ACW_LEGAL_PRIVACY_VERSION=... on stdout, which is
#                                      what the website build takes its version dates from; the report goes to stderr.
#   deploy/legal-archive.sh --check    the same decision, writes nothing
#   deploy/legal-archive.sh --dir DIR  use DIR instead of ~acw/audio-cleaner/data/legal, written as the current account (a dry run, a test)
# ACW_LEGAL_DATE=YYYY-MM-DD names "today" (a test). Run as the owner: the files are written as the service account, the way install.sh
# writes acw's files. Needs node and web/node_modules (npm ci is run when they are missing).
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
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
if [ -n "$DIR" ]; then
  as_acw() { "$@"; }   # a folder named on the command line (a test, a dry run) is written by whoever runs this
else
  DIR="$H/data/legal"
fi
say() { echo "legal archive: $*" >&2; }

command -v node >/dev/null 2>&1 || { say "node is needed to render the Terms and the Privacy Policy"; exit 1; }
SNAP="$REPO/web/scripts/legal-snapshot.mjs"
[ -d "$REPO/web/node_modules/react-dom" ] && [ -d "$REPO/web/node_modules/esbuild" ] || (cd "$REPO/web" && npm ci --no-audit --no-fund >&2)
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# 1. What is archived: "<file> <sha256 of its text>" for each saved page (from its .sha256, or from the page itself when that is lost).
: > "$TMP/index"
# shellcheck disable=SC2016  # (the quotes are meant: the script runs in the other account's shell)
as_acw bash -c 'cd "$1" 2>/dev/null || exit 0; for f in terms-*.html privacy-*.html; do [ -f "$f" ] && echo "$f $(tr -d "[:space:]" 2>/dev/null < "$f.sha256")"; done; exit 0' legal-index "$DIR" > "$TMP/index.raw"
while read -r file hash; do
  [ -n "$file" ] || continue
  if [ -z "${hash:-}" ]; then
    hash="$(as_acw cat "$DIR/$file" | node "$SNAP" --hash-of - 2>/dev/null || true)"
    say "$file has no hash file; hashing the page itself"
  fi
  echo "$file ${hash:--}" >> "$TMP/index"
done < "$TMP/index.raw"

# 2. Render, hash, decide; the pages of the new versions and current.json come out in $TMP/new.
DATE_ARGS=()
[ -z "${ACW_LEGAL_DATE:-}" ] || DATE_ARGS=(--today "$ACW_LEGAL_DATE")
node "$SNAP" --decide --index "$TMP/index" --out "$TMP/new" ${DATE_ARGS[@]+"${DATE_ARGS[@]}"} > "$TMP/plan"

declare -A VERSION=()
declare -a NEW_FILES=()
while read -r key version sha status; do
  VERSION[$key]="$version"
  if [ "$status" = new ]; then
    say "$key: new text (sha256 ${sha:0:12}), version $version"
    NEW_FILES+=("$key-$version.html")
  else
    say "$key: archived already, version $version (sha256 ${sha:0:12}; nothing to save)"
  fi
done < "$TMP/plan"

# 3. Save what is new, then say which versions are live. A page is never written over a file that is there (the page first, its hash
# after, each atomically).
if [ "$CHECK_ONLY" = 0 ]; then
  as_acw mkdir -p -m 700 "$DIR"
  for f in ${NEW_FILES[@]+"${NEW_FILES[@]}"}; do
    if as_acw test -e "$DIR/$f" || as_acw test -e "$DIR/$f.sha256"; then
      say "STOP: $DIR/$f exists already and an archived version is never overwritten"; exit 1
    fi
    put_as_acw "$TMP/new/$f" "$DIR/$f" 600
    put_as_acw "$TMP/new/$f.sha256" "$DIR/$f.sha256" 600
    say "saved $DIR/$f"
  done
  if [ "$(as_acw cat "$DIR/current.json" 2>/dev/null || true)" != "$(cat "$TMP/new/current.json")" ]; then
    put_as_acw "$TMP/new/current.json" "$DIR/current.json" 600
    say "wrote $DIR/current.json"
  fi
fi
printf 'ACW_LEGAL_TERMS_VERSION=%s\nACW_LEGAL_PRIVACY_VERSION=%s\n' "${VERSION[terms]}" "${VERSION[privacy]}"
