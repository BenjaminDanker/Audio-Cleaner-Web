#!/usr/bin/env bash
# Download the models and programs listed in deploy/models.lock, verify every file's sha256, and unpack archive members.
# Idempotent: a file that is already right is not touched; a missing or wrong one is fetched again.
#   deploy/fetch-models.sh                    fetch what is missing or wrong (required files only)
#   deploy/fetch-models.sh --check            only verify; exit 1 if a required file is missing or wrong (changes nothing)
#   deploy/fetch-models.sh --with-optional    also fetch the optional files (TorchScript Sidon weights)
#   deploy/fetch-models.sh --only GLOB        only targets matching GLOB, e.g. --only 'models/dpdfnet/*'
#   deploy/fetch-models.sh --dest-dir DIR     with --only matching one file: put that file in DIR instead of ~/audio-cleaner/<target>
#   deploy/fetch-models.sh --archive FILE     use this local copy of an archive (matched by file name) instead of downloading it,
#                                             e.g. the models zip before it is published; repeatable
#   ACW_ARCHIVE_DIR=DIR deploy/fetch-models.sh   look for the archives in DIR (by file name) before downloading them
#   deploy/fetch-models.sh --home DIR         install under DIR instead of ~/audio-cleaner (ACW_HOME does the same)
#   deploy/fetch-models.sh --lock FILE        read another lock file
#   deploy/fetch-models.sh -v                 list every file, not only the problems
# Files marked `local` in the lock (built or converted here) cannot be downloaded: they are only checked.
# Needs curl, sha256sum (or shasum), and python3 (zip archives) or tar (tar.xz archives) when archive members are needed.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=deploy/color.sh
. "$HERE/color.sh"
H="${ACW_HOME:-$HOME/audio-cleaner}"
# With the service account (acw, /etc/acw-service-user) the production folder is acw's: the owner may only --check it
# (bin and models are readable by the owner); fetching is done by acw, through deploy/install.sh.
SVC_USER="$(cat /etc/acw-service-user 2>/dev/null || true)"
OWNER_VIEW=0
if [ -z "${ACW_HOME:-}" ] && [ -n "$SVC_USER" ] && [ "$(id -un)" != "$SVC_USER" ]; then
  H="$(getent passwd "$SVC_USER" | cut -d: -f6)/audio-cleaner"
  OWNER_VIEW=1
fi
LOCK="$HERE/models.lock"
CHECK=0
OPTIONAL=0
VERBOSE=0
ONLY=""
DEST_DIR=""
ARCHIVES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --check) CHECK=1 ;;
    --with-optional) OPTIONAL=1 ;;
    -v|--verbose) VERBOSE=1 ;;
    --only) ONLY="${2:?--only needs a pattern}"; shift ;;
    --dest-dir) DEST_DIR="${2:?--dest-dir needs a directory}"; shift ;;
    --archive) ARCHIVES+=("${2:?--archive needs a file}"); shift ;;
    --home) H="${2:?--home needs a directory}"; OWNER_VIEW=0; shift ;;
    --lock) LOCK="${2:?--lock needs a file}"; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
[ -f "$LOCK" ] || { echo "no lock file $LOCK" >&2; exit 2; }
if [ "$OWNER_VIEW" = 1 ] && [ "$CHECK" = 0 ] && [ -z "$DEST_DIR" ]; then
  echo "$H belongs to $SVC_USER: deploy/install.sh fetches what is missing (as $SVC_USER); here only --check works (or --dest-dir, or --home DIR)" >&2
  exit 2
fi

sha_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
size_of() { wc -c < "$1" | tr -d ' '; }
path_of() {  # path_of TARGET: where the file lives (or goes)
  if [ -n "$DEST_DIR" ]; then printf '%s/%s' "$DEST_DIR" "${1##*/}"; else printf '%s/%s' "$H" "$1"; fi
}
status_of() {  # status_of PATH SHA SIZE -> ok | missing | wrong
  if [ ! -e "$1" ]; then echo missing; return; fi
  if [ -f "$1" ] && [ "$(size_of "$1")" = "$3" ] && [ "$(sha_of "$1")" = "$2" ]; then echo ok; else echo wrong; fi
}
selected() {  # selected TARGET: does --only allow it?
  [ -z "$ONLY" ] && return 0
  # shellcheck disable=SC2254
  case "$1" in $ONLY) return 0 ;; esac
  return 1
}
say() { [ "$VERBOSE" = 1 ] && echo "$@"; return 0; }
say_okv() { [ "$VERBOSE" = 1 ] && say_ok "$@"; return 0; }

download() {  # download URL DEST: resumes a partial download, retries once from scratch
  local url="$1" dest="$2" quiet=(-sS)
  [ -t 2 ] && quiet=(--progress-bar)
  mkdir -p "$(dirname "$dest")"
  if ! curl -fL --retry 3 --retry-delay 5 --connect-timeout 30 -C - "${quiet[@]}" -o "$dest" "$url"; then
    rm -f "$dest"
    curl -fL --retry 3 --retry-delay 5 --connect-timeout 30 "${quiet[@]}" -o "$dest" "$url"
  fi
}

install_mode() {  # programs are executable, everything else is plain data
  case "$1" in bin/*) chmod 755 "$2" ;; *) chmod 644 "$2" ;; esac
}

OK=0 FIXED=0 MISSING=0 WRONG=0 ABSENT_OPT=0 LOCAL_CHANGED=0 FAILED=0
NEED="$(mktemp)"  # archive members still to unpack: "archive-name<TAB>member<TAB>target<TAB>sha<TAB>size"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"; rm -f "$NEED"' EXIT

while read -r kind flags sha size target rest || [ -n "${kind:-}" ]; do
  rest="${rest%$'\r'}"; target="${target%$'\r'}"  # a Windows checkout may end lines with CR
  case "${kind:-}" in ''|'#'*|archive) continue ;; esac
  case "$target" in /*|*..*) echo "bad target in the lock: $target" >&2; exit 2 ;; esac
  selected "$target" || continue
  opt=0; [ "$flags" = optional ] && opt=1
  dest="$(path_of "$target")"
  st="$(status_of "$dest" "$sha" "$size")"
  case "$kind" in
    local)
      case "$st" in
        ok) OK=$((OK + 1)); say_okv "ok       $target (local)" ;;
        missing)
          if [ "$opt" = 1 ]; then ABSENT_OPT=$((ABSENT_OPT + 1)); say "absent   $target (optional, local: $rest)"
          else MISSING=$((MISSING + 1)); say_fail "MISSING  $target: cannot be downloaded: $rest"; fi ;;
        wrong) LOCAL_CHANGED=$((LOCAL_CHANGED + 1)); say_warn "changed  $target differs from the recorded sha256 (a local build or conversion; $rest)" ;;
      esac
      continue ;;
    file|member) ;;
    *) echo "unknown kind '$kind' in the lock" >&2; exit 2 ;;
  esac
  if [ "$st" = ok ]; then OK=$((OK + 1)); say_okv "ok       $target"; continue; fi
  # Something is missing or wrong.
  if [ "$opt" = 1 ] && [ "$st" = missing ] && { [ "$CHECK" = 1 ] || [ "$OPTIONAL" = 0 ]; }; then
    ABSENT_OPT=$((ABSENT_OPT + 1)); say "absent   $target (optional)"; continue
  fi
  if [ "$CHECK" = 1 ]; then
    if [ "$st" = missing ]; then MISSING=$((MISSING + 1)); say_fail "MISSING  $target"; else WRONG=$((WRONG + 1)); say_fail "WRONG    $target (size or sha256 differs from the lock)"; fi
    continue
  fi
  if [ "$kind" = file ]; then
    echo "fetching $target"
    part="$dest.part.$$"
    if download "$rest" "$part" && [ "$(sha_of "$part")" = "$sha" ] && [ "$(size_of "$part")" = "$size" ]; then
      install_mode "$target" "$part"; mv -f "$part" "$dest"; FIXED=$((FIXED + 1))
    else
      rm -f "$part"; FAILED=$((FAILED + 1)); say_fail "FAILED   $target: download failed or sha256 mismatch ($rest)"
    fi
  else
    printf '%s\t%s\t%s\t%s\t%s\n' "${rest%%:*}" "${rest#*:}" "$target" "$sha" "$size" >> "$NEED"
  fi
done < "$LOCK"

# Archive members: fetch each archive once (or use the local copy given with --archive), unpack what is needed.
if [ -s "$NEED" ]; then
  for name in $(cut -f1 "$NEED" | sort -u); do
    line="$(awk -v n="$name" '$1 == "archive" && $5 == n { print; exit }' "$LOCK")"
    [ -n "$line" ] || { echo "the lock has members of archive '$name' but no archive line for it" >&2; exit 2; }
    read -r _ _ a_sha a_size _ a_url <<< "$line"
    a_url="${a_url%$'\r'}"
    base="${a_url##*/}"
    arc=""
    for cand in ${ARCHIVES[@]+"${ARCHIVES[@]}"}; do [ "${cand##*/}" = "$base" ] && arc="$cand"; done
    if [ -z "$arc" ] && [ -n "${ACW_ARCHIVE_DIR:-}" ] && [ -f "$ACW_ARCHIVE_DIR/$base" ]; then arc="$ACW_ARCHIVE_DIR/$base"; fi
    if [ -z "$arc" ]; then
      echo "fetching archive $base"
      arc="$WORK/$base"
      download "$a_url" "$arc" || true
    fi
    if [ ! -f "$arc" ] || [ "$(size_of "$arc")" != "$a_size" ] || [ "$(sha_of "$arc")" != "$a_sha" ]; then
      say_fail "FAILED   archive $base: missing or its sha256 does not match the lock ($a_url)"
      FAILED=$((FAILED + $(grep -c "^$name	" "$NEED")))
      continue
    fi
    while IFS=$'\t' read -r n member target sha size; do
      [ "$n" = "$name" ] || continue
      dest="$(path_of "$target")"
      mkdir -p "$(dirname "$dest")"
      part="$dest.part.$$"
      case "$base" in
        *.zip) python3 -I - "$arc" "$member" "$part" <<'PY' || rm -f "$part" ;;
import shutil, sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z, z.open(sys.argv[2]) as src, open(sys.argv[3], "wb") as dst:
    shutil.copyfileobj(src, dst, 1 << 20)
PY
        *.tar.xz|*.tar.gz|*.tgz) tar -xOf "$arc" "$member" > "$part" || rm -f "$part" ;;
        *) echo "unknown archive type: $base" >&2; exit 2 ;;
      esac
      if [ -f "$part" ] && [ "$(sha_of "$part")" = "$sha" ] && [ "$(size_of "$part")" = "$size" ]; then
        install_mode "$target" "$part"; mv -f "$part" "$dest"; FIXED=$((FIXED + 1)); echo "unpacked $target"
      else
        rm -f "$part"; FAILED=$((FAILED + 1)); say_fail "FAILED   $target: not in $base or its sha256 does not match the lock"
      fi
    done < "$NEED"
  done
fi

summary="models: $OK ok"
[ "$FIXED" -gt 0 ] && summary="$summary, $FIXED fetched"
[ "$MISSING" -gt 0 ] && summary="$summary, $MISSING MISSING"
[ "$WRONG" -gt 0 ] && summary="$summary, $WRONG WRONG"
[ "$FAILED" -gt 0 ] && summary="$summary, $FAILED FAILED"
[ "$ABSENT_OPT" -gt 0 ] && summary="$summary, $ABSENT_OPT optional absent (not needed to run; --with-optional fetches them, -v names them)"
[ "$LOCAL_CHANGED" -gt 0 ] && summary="$summary, $LOCAL_CHANGED local files changed since recorded"
# The count goes to stdout as before (the lines above, one per problem, are what the deploy summary lists).
if [ $((MISSING + WRONG + FAILED)) -gt 0 ]; then acw_paint 1 '1;31' "$summary"; echo; exit 1
elif [ "$LOCAL_CHANGED" -gt 0 ]; then acw_paint 1 33 "$summary"; echo
else acw_paint 1 32 "$summary"; echo; fi
