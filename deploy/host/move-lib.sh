#!/usr/bin/env bash
# Shared by move-to-acw.sh and move-back-to-silver.sh (sourced, not run): moving Audio Cleaner's production
# services between two accounts, FROM_USER -> TO_USER. Both scripts are thin: they set the direction, add what only
# their direction needs (account, sudoers, wrappers / the way back), and call these steps in order.
#
# What moves (the production parts of ~/audio-cleaner, same filesystem, so `mv` is a rename and takes no time):
#   app app.prev venv venv-gpu bin models wasm data      and from secrets/ only the files in SECRET_FILES.
# What never moves: src (build trees: the Rust toolchain is the owner's), dev, research, listen, and any other file in
# secrets/ (the owner's own tokens and keys: update-signing-key.pem, github-actions-*.token, ...). An allowlist, so
# a secret that nobody thought about stays with its owner.

MOVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$MOVE_DIR/../.." && pwd)"
SVC="${ACW_SERVICE_USER_NAME:-acw}"            # the service account
STATE_DIR="${ACW_MOVE_STATE_DIR:-/var/lib/acw-move}"
BACKUP_DIR="${ACW_BACKUP_DIR:-/mnt/mc-storage/audio-cleaner-backups}"
MARKER=/etc/acw-service-user                    # names the account the services run as (deploy/lib.sh reads it)
DIR_ITEMS=(app app.prev venv venv-gpu bin models wasm data)
SHARED_ITEMS=(bin models wasm venv venv-gpu)    # readable by the owner, too: tools, dev runs and the web build use them
LINK_ITEMS=(bin models wasm venv venv-gpu)      # the owner's ~/audio-cleaner/X -> the service account's X
SECRET_FILES=(acw.env acw-worker.env vps.env backup.env backup.pass github-releases-read.token)

DRY=0 FORCE=0 YES=0 DRY_PROBLEMS=0
OWNER="${SUDO_USER:-}"
{ [ -n "$OWNER" ] && [ "$OWNER" != root ]; } || OWNER=silver
PROBLEMS=()
CURRENT_STEP="start"

usage() { sed -n '2,/^set -/p' "$1" | sed '$d'; }
parse_args() {  # parse_args SCRIPT ARGS...
  local script="$1"; shift
  while [ $# -gt 0 ]; do
    case "$1" in
      --dry-run) DRY=1 ;;
      --force) FORCE=1 ;;
      --yes|-y) YES=1 ;;
      --owner) OWNER="${2:?--owner needs a name}"; shift ;;
      -h|--help) usage "$script"; exit 0 ;;
      *) echo "unknown option $1 (see --help)" >&2; exit 2 ;;
    esac
    shift
  done
}

say() { printf '%s\n' "$*"; }
step() { CURRENT_STEP="$*"; printf '\n== %s\n' "$*"; }
die() { echo "error: $*" >&2; exit 1; }
problem() { PROBLEMS+=("$*"); echo "problem: $*" >&2; }
run() {  # print the command, run it unless --dry-run
  printf '+'; printf ' %q' "$@"; printf '\n'
  [ "$DRY" = 1 ] || "$@"
}
home_of() {  # the home folder of an account (ACW_MOVE_OWNER_HOME / ACW_MOVE_SVC_HOME: test hooks for a scratch tree)
  if [ "$1" = "$OWNER" ] && [ -n "${ACW_MOVE_OWNER_HOME:-}" ]; then printf '%s\n' "$ACW_MOVE_OWNER_HOME"
  elif [ "$1" = "$SVC" ] && [ -n "${ACW_MOVE_SVC_HOME:-}" ]; then printf '%s\n' "$ACW_MOVE_SVC_HOME"
  else getent passwd "$1" | cut -d: -f6; fi
}
# Run a command as a user, in that user's environment (HOME, XDG_RUNTIME_DIR, bus): deploy/host/acw-run.
as_user() { local u="$1"; shift; ACW_USER="$u" "$MOVE_DIR/acw-run" "$@"; }
ctl_as() { local u="$1"; shift; as_user "$u" systemctl --user "$@"; }
admin_as() {  # admin_as USER HOME ARGS: python -m acw.admin with that user's deployed code and worker environment
  local u="$1" h="$2/audio-cleaner"; shift 2
  as_user "$u" bash -c 'cd "$1/app" && set -a && . "$1/secrets/acw-worker.env" && set +a && PYTHONPATH="$1/app" exec "$1/venv/bin/python" -m acw.admin "${@:2}"' admin "$h" "$@"
}

# The enabled acw-* units of a user (the symlinks in default.target.wants / timers.target.wants, instances such as
# acw-worker@1.service included), the stream worker first, timers last.
enabled_units_of() {
  local d="$1/.config/systemd/user" all
  all="$({ ls -1 "$d/default.target.wants" "$d/timers.target.wants" 2>/dev/null || true; } | grep -E '^acw[A-Za-z0-9@._-]*\.(service|timer)$' | sort -u || true)"
  { printf '%s\n' "$all" | grep -E '^acw-stream\.service$' || true
    printf '%s\n' "$all" | grep -E '\.service$' | grep -vE '^acw-stream\.service$' || true
    printf '%s\n' "$all" | grep -E '\.timer$' || true; } | grep . || true
}

# ---------------------------------------------------------------- the direction (set by the two scripts)
# set_direction FROM_USER TO_USER
set_direction() {
  FROM_USER="$1"; TO_USER="$2"
  FROM_HOME="$(home_of "$FROM_USER" || true)"; TO_HOME="$(home_of "$TO_USER" || true)"
  [ -n "$FROM_HOME" ] || FROM_HOME="/home/$FROM_USER"
  [ -n "$TO_HOME" ] || TO_HOME="/home/$TO_USER"
  FROM_H="$FROM_HOME/audio-cleaner"; TO_H="$TO_HOME/audio-cleaner"
  OWNER_HOME="$(home_of "$OWNER" || true)"; [ -n "$OWNER_HOME" ] || OWNER_HOME="/home/$OWNER"
  if [ "$TO_USER" = "$SVC" ]; then TO_IS_SVC=1; else TO_IS_SVC=0; fi
}

# ---------------------------------------------------------------- 1. checks (change nothing)
m_preflight() {
  step "checks (nothing is changed)"
  if [ "$(id -u)" != 0 ]; then
    if [ "$DRY" = 1 ]; then say "(not root: this dry run only lists the steps; some checks are skipped)"; else die "run it with sudo: sudo bash $0"; fi
  fi
  id "$OWNER" >/dev/null 2>&1 || die "there is no account '$OWNER' (--owner NAME)"
  local t
  for t in sudo systemctl getent stat install ssh-keygen; do command -v "$t" >/dev/null 2>&1 || problem "$t is not installed"; done
  [ -x "$MOVE_DIR/acw-run" ] || problem "$MOVE_DIR/acw-run is missing or not executable"
  [ -d "$REPO/deploy/systemd" ] || problem "$REPO/deploy/systemd not found: run this from the repository"

  HAVE_FROM=0
  [ -d "$FROM_H/app/acw" ] && HAVE_FROM=1
  if [ "$HAVE_FROM" = 0 ]; then
    if [ -d "$TO_H/app/acw" ]; then say "the folders are already in $TO_H (an earlier run): continuing from there"
    else problem "nothing is deployed in $FROM_H or in $TO_H"; fi
  else
    [ -x "$FROM_H/venv/bin/python" ] || problem "$FROM_H/venv/bin/python is missing"
    [ -f "$FROM_H/secrets/acw-worker.env" ] || problem "$FROM_H/secrets/acw-worker.env is missing"
    # mv must be a rename: the same filesystem as the destination's parent folder
    local parent dev_to dev_from item
    parent="$(dirname "$TO_HOME")"; dev_to="$(stat -c %d "$parent" 2>/dev/null || echo ?)"
    for item in "${DIR_ITEMS[@]}"; do
      [ -e "$FROM_H/$item" ] && [ ! -L "$FROM_H/$item" ] || continue
      dev_from="$(stat -c %d "$FROM_H/$item" 2>/dev/null || echo ??)"
      [ "$dev_from" = "$dev_to" ] || problem "$FROM_H/$item is on another filesystem than $parent: mv would copy it (slowly); stop and decide by hand"
      # (a link at the owner's side is the way the folder was reached after the move: it is replaced by the folder)
      [ -e "$TO_H/$item" ] && [ ! -L "$TO_H/$item" ] && problem "$item exists in both $FROM_H and $TO_H: decide which one is current and remove the other"
    done
    # nothing running that a restart would hurt
    local act
    if act="$(admin_as "$FROM_USER" "$FROM_HOME" activity 2>&1)"; then
      say "running now: $act"
      [ "$act" = "file jobs: none; live sessions: 0" ] || problem "work is running ($act): wait for it to end, or use --force (jobs go back to the queue, live sessions are cut)"
    else
      problem "could not read what is running: ${act:0:200}"
    fi
  fi
  if [ -z "$(enabled_units_of "$FROM_HOME")" ] && [ ! -s "$STATE_DIR/units.$FROM_USER" ]; then
    if [ "$HAVE_FROM" = 1 ]; then problem "no enabled acw-* units in the user manager of $FROM_USER: is this the right direction?"; fi
  elif [ "$(id -u)" = 0 ] && [ -n "$(enabled_units_of "$FROM_HOME")" ] && [ ! -S "/run/user/$(id -u "$FROM_USER")/bus" ]; then
    problem "the user manager of $FROM_USER is not running (no /run/user/$(id -u "$FROM_USER")/bus): its units cannot be stopped cleanly"
  fi
  if declare -F extra_preflight >/dev/null; then extra_preflight; fi   # checks only one direction needs
  if [ "${#PROBLEMS[@]}" -gt 0 ]; then
    if [ "$FORCE" = 1 ]; then say "continuing although there are ${#PROBLEMS[@]} problem(s) (--force)"
    elif [ "$DRY" = 1 ]; then say "(the real run would stop here: ${#PROBLEMS[@]} problem(s) above)"; DRY_PROBLEMS=1
    else die "${#PROBLEMS[@]} problem(s) above; nothing was changed (--force overrides the 'work is running' check only if you accept it)"; fi
  else
    say "ok"
  fi
}

confirm() {  # confirm "what is about to happen"
  [ "$DRY" = 1 ] || [ "$YES" = 1 ] && return 0
  printf '%s\nType yes to continue: ' "$1"
  read -r a
  [ "$a" = yes ] || die "not confirmed; nothing was changed"
}

# ---------------------------------------------------------------- 2. stop the services of FROM_USER
m_quiesce() {
  step "stop and disable the services of $FROM_USER (downtime starts here)"
  T_STOP="$(date +%s)"
  run mkdir -p "$STATE_DIR"
  # until the services are healthy on the other side deploy/lib.sh refuses to run (a deploy now would find half a layout)
  if [ "$DRY" = 0 ]; then echo "$FROM_USER -> $TO_USER" > "$STATE_DIR/in-progress"; else say "+ (write $STATE_DIR/in-progress: deploy/*.sh refuse to run until the move is finished)"; fi
  local units ud
  units="$(enabled_units_of "$FROM_HOME")"
  ud="$FROM_HOME/.config/systemd/user"
  if [ -n "$units" ]; then
    say "enabled units of $FROM_USER: $(printf '%s ' $units)"
    if [ "$DRY" = 0 ]; then printf '%s\n' "$units" > "$STATE_DIR/units.$FROM_USER"; else say "+ (record them in $STATE_DIR/units.$FROM_USER)"; fi
  elif [ -s "$STATE_DIR/units.$FROM_USER" ]; then
    say "no enabled units now; the list from an earlier run is in $STATE_DIR/units.$FROM_USER"
  fi
  if [ "$HAVE_FROM" = 1 ]; then
    # new live sessions refused while we work; the check that nothing is running is repeated (they could have started)
    run as_user "$FROM_USER" touch "$FROM_H/data/drain"
    if [ "$DRY" = 0 ] && [ "$FORCE" = 0 ]; then
      local act; act="$(admin_as "$FROM_USER" "$FROM_HOME" activity 2>&1 || true)"
      [ "$act" = "file jobs: none; live sessions: 0" ] || { as_user "$FROM_USER" rm -f "$FROM_H/data/drain"; die "work started in the meantime ($act); nothing was changed, run again later"; }
    fi
  fi
  # keep the unit files exactly as they run now: the other account gets these, so only the account changes
  if [ -d "$ud" ] && [ -n "$(find "$ud" -maxdepth 1 -name 'acw*' 2>/dev/null | head -1)" ]; then
    run rm -rf "$STATE_DIR/units-$FROM_USER"
    run mkdir -p "$STATE_DIR/units-$FROM_USER"
    if [ "$DRY" = 0 ]; then
      find "$ud" -maxdepth 1 \( -name 'acw*.service' -o -name 'acw*.timer' -o -name 'acw*.slice' -o -name 'acw*.service.d' -o -name 'acw*.slice.d' \) -exec cp -a {} "$STATE_DIR/units-$FROM_USER/" \;
    else say "+ (copy acw* unit files and drop-ins of $ud to $STATE_DIR/units-$FROM_USER)"; fi
  fi
  if [ -n "$units" ]; then
    # shellcheck disable=SC2086
    run ctl_as "$FROM_USER" disable --now $units
  fi
  if [ "$DRY" = 0 ]; then
    local left n=0
    while :; do
      left="$(ctl_as "$FROM_USER" list-units 'acw*' --plain --no-legend --state=active,activating,deactivating 2>/dev/null | awk '{print $1}' | grep -E '\.(service|timer)$' || true)"
      [ -z "$left" ] && break
      n=$((n + 1)); [ "$n" -gt 12 ] && { say "still active, stopping: $left"; ctl_as "$FROM_USER" stop $left || true; sleep 5; left="$(ctl_as "$FROM_USER" list-units 'acw*' --plain --no-legend --state=active,activating,deactivating | awk '{print $1}' | grep -E '\.(service|timer)$' || true)"; [ -z "$left" ] || die "units of $FROM_USER still running: $left"; break; }
      say "waiting for: $left"; sleep 5
    done
    ctl_as "$FROM_USER" reset-failed 2>/dev/null || true
  else
    say "+ (wait until no acw-* unit of $FROM_USER is active)"
  fi
  # take the unit files out of FROM_USER's manager, so nothing starts there by accident (the copy is in $STATE_DIR)
  if [ -d "$ud" ]; then
    if [ "$DRY" = 0 ]; then
      find "$ud" -maxdepth 1 \( -name 'acw*.service' -o -name 'acw*.timer' -o -name 'acw*.slice' -o -name 'acw*.service.d' -o -name 'acw*.slice.d' \) -exec rm -rf {} +
    else say "+ (remove the acw* unit files and drop-ins from $ud)"; fi
    run ctl_as "$FROM_USER" daemon-reload
  fi
  if [ "$HAVE_FROM" = 1 ]; then
    step "database snapshot with everything stopped"
    run admin_as "$FROM_USER" "$FROM_HOME" backup "$FROM_H/data/backups" --keep 10
  fi
}

# Is this part (relative to ~/audio-cleaner) going to be at the destination? (a dry run has not moved it yet)
item_present() { [ -e "$TO_H/$1" ] || { [ "$DRY" = 1 ] && [ -e "$FROM_H/$1" ]; }; }

# ---------------------------------------------------------------- 3. move the folders and the production secrets
m_move() {
  step "move the folders: $FROM_H -> $TO_H"
  if [ "$TO_IS_SVC" = 1 ]; then run install -d -m 750 -o "$TO_USER" -g "$TO_USER" "$TO_H"; else run mkdir -p "$TO_H"; fi
  local item
  # symlinks of the way here (the owner's ~/audio-cleaner/bin -> ...) go first, so the real folders can come back
  if [ "$TO_IS_SVC" = 0 ]; then
    for item in "${LINK_ITEMS[@]}"; do
      [ -L "$TO_H/$item" ] && run rm "$TO_H/$item"
    done
  fi
  for item in "${DIR_ITEMS[@]}"; do
    if [ -e "$FROM_H/$item" ] && [ ! -L "$FROM_H/$item" ]; then
      if [ -e "$TO_H/$item" ] && [ ! -L "$TO_H/$item" ]; then die "$TO_H/$item already exists"; fi
      run mv "$FROM_H/$item" "$TO_H/$item"
    else
      say "(no $item in $FROM_H to move)"
    fi
  done
  if [ "$TO_IS_SVC" = 1 ]; then run install -d -m 700 -o "$TO_USER" -g "$TO_USER" "$TO_H/secrets"; else run mkdir -p -m 700 "$TO_H/secrets"; fi
  local f
  for f in "${SECRET_FILES[@]}"; do
    if [ -e "$FROM_H/secrets/$f" ]; then
      run mv "$FROM_H/secrets/$f" "$TO_H/secrets/$f"
    fi
  done
  # what stays with its owner (names only; never contents)
  local rest
  rest="$({ ls -A "$FROM_H/secrets" 2>/dev/null || true; } | grep -vxF "$(printf '%s\n' "${SECRET_FILES[@]}")" | tr '\n' ' ' || true)"
  if [ "$TO_IS_SVC" = 1 ]; then say "left in $FROM_H/secrets (not moved: acw must not have it): ${rest:-nothing}"
  elif [ -n "$rest" ]; then say "left in $FROM_H/secrets: $rest"; fi
  # absolute paths of the old home inside the env files (ACW_GITHUB_TOKEN_FILE, ACW_BACKUP_PASSPHRASE_FILE, ...)
  if [ "$DRY" = 0 ]; then
    for f in "${SECRET_FILES[@]}"; do
      [ -f "$TO_H/secrets/$f" ] || continue
      case "$f" in *.env) ;; *) continue ;; esac
      if grep -qF "$FROM_H" "$TO_H/secrets/$f"; then
        sed -i "s|$(printf '%s' "$FROM_H" | sed 's/[][\.*^$|/&]/\\&/g')|$TO_H|g" "$TO_H/secrets/$f"
        say "rewrote the old folder name in secrets/$f"
      fi
      if grep -qF "/home/$FROM_USER" "$TO_H/secrets/$f"; then say "warning: secrets/$f still names /home/$FROM_USER somewhere: check it by hand"; fi
    done
  else
    say "+ (in the moved *.env files replace $FROM_H by $TO_H; warn about any other /home/$FROM_USER)"
  fi
  step "Python venvs: rewrite their absolute paths (no reinstall)"
  local venvs=()
  for item in venv venv-gpu; do item_present "$item" && venvs+=("$TO_H/$item"); done
  run bash "$MOVE_DIR/fix-venv-paths.sh" "$FROM_H" "$TO_H" "${venvs[@]}"
  step "owner and permissions"
  # only what moved (the owner's ~/audio-cleaner also holds dev/, research/, src/ ... that stay exactly as they are)
  for item in "${DIR_ITEMS[@]}"; do item_present "$item" && run chown -R "$TO_USER:$TO_USER" "$TO_H/$item"; done
  for f in "${SECRET_FILES[@]}"; do item_present "secrets/$f" && run chown "$TO_USER:$TO_USER" "$TO_H/secrets/$f" && run chmod 600 "$TO_H/secrets/$f"; done
  if [ "$TO_IS_SVC" = 1 ]; then run chown "$TO_USER:$TO_USER" "$TO_H" "$TO_H/secrets"; run chmod 750 "$TO_H"; fi
  for item in app app.prev data secrets; do item_present "$item" && run chmod 700 "$TO_H/$item"; done
  if [ "$TO_IS_SVC" = 1 ]; then
    for item in "${SHARED_ITEMS[@]}"; do item_present "$item" && run chmod -R u+rwX,g=rX,o= "$TO_H/$item"; done
    m_owner_access
  else
    for item in "${SHARED_ITEMS[@]}"; do item_present "$item" && run chmod -R u+rwX,g=rX,o=rX "$TO_H/$item"; done
    # the service account's ACL entries and the owner's access on its folder are not needed any more
    if command -v setfacl >/dev/null 2>&1; then
      for item in "${SHARED_ITEMS[@]}"; do item_present "$item" && run setfacl -R -b "$TO_H/$item"; done
    fi
  fi
}

# The owner reads bin, models, wasm and the venvs of the service account (dev runs, the web build, tests read
# ~/audio-cleaner/{bin,models,wasm,venv}). Group acw is the design (the owner is a member); an ACL entry for the owner
# works for the owner's sessions that started before the group was added (a group change reaches a running login
# session and the owner's systemd user manager only at the next login / restart). app, data, secrets: acw only.
m_owner_access() {
  local item
  acl() { run setfacl "$@" || say "warning: setfacl failed (a filesystem without ACLs?): the owner reads the shared folders through group $SVC only"; }
  if command -v setfacl >/dev/null 2>&1; then
    acl -m "u:$OWNER:x" "$TO_HOME" "$TO_H"
    for item in "${SHARED_ITEMS[@]}"; do
      item_present "$item" && { acl -R -m "u:$OWNER:rX" "$TO_H/$item"; acl -d -m "u:$OWNER:rX" "$TO_H/$item"; }
    done
  else
    say "warning: setfacl is not installed: the owner reads the shared folders through group $SVC only (log in again to get the group)"
  fi
  # the owner's ~/audio-cleaner/X now point at the service account's folders
  for item in "${LINK_ITEMS[@]}"; do
    if item_present "$item"; then
      run ln -sfn "$TO_H/$item" "$FROM_H/$item"
      run chown -h "$OWNER:$OWNER" "$FROM_H/$item"
    fi
  done
}

# ---------------------------------------------------------------- 4. the live engine knows where its libraries are
# native/live links the sherpa-onnx library of the venv and records that absolute path twice (rpath, and the folder its
# sandbox lets it read): a moved binary would look in the old home and not find it. So it is rebuilt against the new
# venv, by the owner (the Rust toolchain is the owner's) - the stamp bin/acw-live.src stays valid (it hashes the sources).
m_rebuild_live() {
  step "rebuild the live engine (acw-live) against the venv in its new place"
  if ! item_present bin/acw-live; then say "(no bin/acw-live: nothing to rebuild)"; return 0; fi
  local tmp
  if [ "$DRY" = 1 ]; then tmp="/tmp/acw-live-build.XXXXXX"; else tmp="$(as_user "$OWNER" mktemp -d)"; fi
  run as_user "$OWNER" env "ACW_PYTHON=$TO_H/venv/bin/python" bash "$REPO/native/live/build.sh" "$tmp"
  run install -m 755 -o "$TO_USER" -g "$TO_USER" "$tmp/acw-live" "$TO_H/bin/acw-live"
  run rm -rf "$tmp"
  if [ "$DRY" = 0 ] && command -v readelf >/dev/null 2>&1; then
    readelf -d "$TO_H/bin/acw-live" | grep -qF "$TO_H/venv" || die "the rebuilt acw-live does not name $TO_H/venv as its library path"
    say "acw-live now loads sherpa-onnx from $TO_H/venv"
  fi
}

# ---------------------------------------------------------------- 5. units into the manager of TO_USER, start, check
m_units_start() {
  step "install the units in the user manager of $TO_USER and start them"
  local ud="$TO_HOME/.config/systemd/user" stash="$STATE_DIR/units-$FROM_USER"
  run install -d -m 700 -o "$TO_USER" -g "$TO_USER" "$TO_HOME/.config" "$TO_HOME/.config/systemd" "$ud"
  if [ "$DRY" = 1 ] || [ -d "$stash" ]; then
    run cp -a "$stash/." "$ud/"
  else
    say "no stash of unit files: using the ones in the repository (deploy/systemd) and no drop-ins"
    run cp -a "$REPO"/deploy/systemd/. "$ud/"
  fi
  run chown -R "$TO_USER:$TO_USER" "$ud"
  [ "$DRY" = 1 ] || { [ -f "$ud/acw-web.service.d/workers.conf" ] && [ -f "$ud/acw.slice.d/size.conf" ]; } || say "note: no worker/memory drop-ins were found: the next deploy/install.sh writes them"
  run ctl_as "$TO_USER" daemon-reload
  local units
  units="$(enabled_units_of "$FROM_HOME")"
  [ -n "$units" ] || units="$(cat "$STATE_DIR/units.$FROM_USER" 2>/dev/null || true)"
  [ "$DRY" = 1 ] || [ -n "$units" ] || die "no list of units to enable ($STATE_DIR/units.$FROM_USER is empty)"
  [ -n "$units" ] || units="<the units that were enabled for $FROM_USER>"
  if [ "$TO_IS_SVC" = 1 ] && [ "$DRY" = 0 ]; then printf '%s\n' "$SVC" > "$MARKER"; chmod 644 "$MARKER"; fi
  [ "$TO_IS_SVC" = 1 ] && [ "$DRY" = 1 ] && say "+ (write $SVC to $MARKER: deploy/*.sh now run the server part as $SVC)"
  [ "$TO_IS_SVC" = 0 ] && run rm -f "$MARKER"
  # shellcheck disable=SC2086
  run ctl_as "$TO_USER" enable --now $units
  if [ "$DRY" = 1 ]; then
    say "+ (wait up to 120 s: every unit active, /api/health?deep=1 on 8790, 8794 and 8795 ok - deploy/lib.sh check_running)"
    return 0
  fi
  export ACW_SERVICE_USER="$TO_USER" ACW_DELEGATE=1 ACW_CHECK_SECONDS=120 ACW_ALLOW_UNFINISHED_MOVE=1
  # shellcheck source=deploy/lib.sh
  . "$REPO/deploy/lib.sh"
  local svc_units
  svc_units="$(printf '%s\n' "$units" | grep -E '\.service$' | tr '\n' ' ')"
  # shellcheck disable=SC2086
  check_running $svc_units || die "the services did not come up healthy (above)"
  drain_off
  rm -f "$STATE_DIR/in-progress"
  T_UP="$(date +%s)"
  say "healthy again after $(( T_UP - T_STOP )) s of downtime"
}

m_smoke() {
  step "smoke test (deploy/smoke.py, as $OWNER: it uses the owner's VPS login for the tunnel check)"
  if [ "$DRY" = 1 ]; then say "+ as_user $OWNER python3 $REPO/deploy/smoke.py"; return 0; fi
  as_user "$OWNER" python3 "$REPO/deploy/smoke.py" || die "the smoke test found problems (above)"
}

# What acw must not be able to reach, tried as acw (only for the move to acw).
m_isolation() {
  step "isolation: what $SVC must not be able to read"
  if [ "$DRY" = 1 ]; then say "+ as_user $SVC test -r $OWNER_HOME/.ssh/id_ed25519 (must fail); ls $OWNER_HOME (must fail); the same for $OWNER_HOME/audio-cleaner/secrets"; return 0; fi
  local p bad=0
  as_user "$SVC" true || die "cannot run commands as $SVC"
  for p in "$OWNER_HOME" "$OWNER_HOME/.ssh/id_ed25519" "$OWNER_HOME/audio-cleaner/secrets"; do
    if as_user "$SVC" test -r "$p" 2>/dev/null; then
      echo "PROBLEM: $SVC can read $p" >&2; bad=1
    else
      say "ok: $SVC cannot read $p"
    fi
  done
  [ "$bad" = 0 ] || say "fix: chmod 750 $OWNER_HOME (and 700 on the others); the migration itself is complete"
  return 0
}
