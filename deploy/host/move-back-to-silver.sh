#!/usr/bin/env bash
# The way back: move Audio Cleaner's production services from the account acw to the owner's account (silver) again,
# as they were before deploy/host/move-to-acw.sh. Run by the owner, from the repository:
#
#     cd ~/repos/Audio-Cleaner-Web && sudo bash deploy/host/move-back-to-silver.sh
#
#   --dry-run     print every step (and run the read-only checks), change nothing; works without sudo
#   --force       go on although the checks found work running
#   --yes         do not ask for confirmation
#   --owner NAME  the owner account (default: the user who ran sudo)
#
# It is also what to run after a move-to-acw.sh that failed half-way (every step looks at what is already done):
# stops and disables acw's acw-* units, takes a database snapshot, moves app, app.prev, venv, venv-gpu, bin, models, wasm,
# data and the production secrets back to ~/audio-cleaner (the links bin, models, wasm, venv, venv-gpu there are replaced
# by the real folders), rewrites the venvs' paths, rebuilds the live engine against the venv, takes the tunnel key
# out of /home/acw/.ssh, gives the backup folder back, installs the same unit files in the owner's user manager,
# removes /etc/acw-service-user (deploy/*.sh then act as one account again) and starts everything.
# What stays: the account acw, /etc/sudoers.d/acw, the wrappers in /usr/local/bin (harmless without services; remove
# them with:  sudo rm /etc/sudoers.d/acw /usr/local/bin/acw-{run,ctl,logs,admin,sh}; sudo userdel -r acw).
set -euo pipefail
# shellcheck source=deploy/host/move-lib.sh
. "$(dirname "$0")/move-lib.sh"
parse_args "$0" "$@"
set_direction "$SVC" "$OWNER"
MOVE_BEGUN=0

on_exit() {
  local rc=$?
  [ "$rc" = 0 ] && return 0
  [ "$DRY" = 1 ] && return "$rc"
  {
    echo
    echo "FAILED during: $CURRENT_STEP (exit $rc; the error is above)"
    if [ "$MOVE_BEGUN" = 0 ]; then
      echo "Nothing was changed. Fix the cause and run this script again."
    else
      echo "The way back had begun: services may be stopped, new live sessions refused (data/drain)."
      echo "Fix the cause and run this script again: it continues where it stopped."
      echo "  look:   acw-ctl status 'acw-*'  (while the wrappers exist)    systemctl --user status 'acw-*'  (as $OWNER)"
    fi
  } >&2
  return "$rc"
}
trap on_exit EXIT

backups_dir() {
  step "the backup folder $BACKUP_DIR belongs to $OWNER again"
  if [ -d "$BACKUP_DIR" ]; then
    run chown -R "$OWNER:$OWNER" "$BACKUP_DIR"
    run chmod 700 "$BACKUP_DIR"
  fi
}

drop_tunnel_key() {
  step "no tunnel key left with $SVC"
  run rm -f "$FROM_HOME/.ssh/acw_tunnel" "$FROM_HOME/.ssh/acw_tunnel.pub"
}

summary() {
  step "done"
  cat <<TXT
The services run as $OWNER again, in $OWNER's user manager (systemctl --user status 'acw-*', journalctl --user -u acw-web,
deploy/install.sh as before). The account $SVC, /etc/sudoers.d/acw and the wrappers are still there; to remove them:
  sudo rm /etc/sudoers.d/acw /usr/local/bin/acw-{run,ctl,logs,admin,sh}; sudo userdel -r acw
Move again:  sudo bash $REPO/deploy/host/move-to-acw.sh
TXT
}

m_preflight
confirm "This moves the production services of Audio Cleaner from $SVC back to $OWNER. Downtime: a few minutes."
MOVE_BEGUN=1
m_quiesce
m_move
drop_tunnel_key
backups_dir
m_rebuild_live
m_units_start
m_smoke
summary
if [ "$DRY" = 1 ] && [ "$DRY_PROBLEMS" = 1 ]; then exit 1; fi
