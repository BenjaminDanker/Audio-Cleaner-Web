#!/usr/bin/env bash
# Rewrite the absolute path of a moved Python virtualenv: the shebangs of venv/bin/* (pip, uvicorn, ...), the
# activate scripts and pyvenv.cfg name the folder the venv was created in. A venv is not relocatable, but nothing
# else in it records that path (checked for venv and venv-gpu on primary-ram: only bin/ and pyvenv.cfg), so
# replacing the old prefix with the new one makes it work again - no re-download of the packages (the GPU
# venv is 3.9 GB of CUDA wheels).
#
#   deploy/host/fix-venv-paths.sh [--dry-run] OLD_PREFIX NEW_PREFIX VENV_DIR...
#     OLD_PREFIX / NEW_PREFIX   e.g. /home/silver/audio-cleaner  /home/acw/audio-cleaner   (any text; matched literally)
#     VENV_DIR                  a venv in its NEW place
# Only text files directly in VENV_DIR/bin and VENV_DIR/pyvenv.cfg are touched (mode and owner stay). Prints one line
# per file changed. Running it again changes nothing. Exit 1 when a VENV_DIR is not a venv or the rewrite left the old
# prefix behind.
set -euo pipefail
DRY=0
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
[ $# -ge 3 ] || { sed -n '2,13p' "$0"; exit 2; }
OLD="${1%/}"; NEW="${2%/}"; shift 2
[ -n "$OLD" ] && [ -n "$NEW" ] || { echo "empty prefix" >&2; exit 2; }
esc() { printf '%s' "$1" | sed -e 's/[][\.*^$|/&]/\\&/g'; }
OLD_RE="$(esc "$OLD")"; NEW_RE="$(esc "$NEW")"

rc=0
for v in "$@"; do
  v="${v%/}"
  if [ ! -f "$v/pyvenv.cfg" ] || [ ! -d "$v/bin" ]; then echo "not a venv: $v" >&2; rc=1; continue; fi
  n=0
  while IFS= read -r -d '' f; do
    grep -Iq . "$f" 2>/dev/null || continue            # binary or empty: leave alone
    grep -qF -- "$OLD" "$f" || continue
    n=$((n + 1))
    if [ "$DRY" = 1 ]; then echo "would rewrite $f"; else echo "rewrote $f"; sed -i -e "s|$OLD_RE|$NEW_RE|g" "$f"; fi
  done < <(find "$v/bin" -maxdepth 1 -type f -print0; printf '%s\0' "$v/pyvenv.cfg")
  if [ "$DRY" = 0 ]; then
    left="$(grep -rIlF -- "$OLD" "$v/bin" "$v/pyvenv.cfg" 2>/dev/null | head -3 || true)"
    if [ -n "$left" ]; then echo "still names $OLD: $left" >&2; rc=1; fi
  fi
  echo "$v: $n file(s)"
done
exit "$rc"
