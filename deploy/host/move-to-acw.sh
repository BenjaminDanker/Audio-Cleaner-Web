#!/usr/bin/env bash
# Move Audio Cleaner's production services from the owner's account (silver) to the dedicated account acw.
# Run ONCE, by the owner, from the repository:
#
#     cd ~/repos/Audio-Cleaner-Web && sudo bash deploy/host/move-to-acw.sh
#
#   --dry-run     print every step (and run the read-only checks), change nothing; works without sudo
#   --force       go on although the checks found work running (jobs go back to the queue, live sessions are cut)
#   --yes         do not ask for confirmation
#   --owner NAME  the owner account (default: the user who ran sudo)
#
# What it does, in order (docs/operations.md, "The service account"): checks (nothing running, same filesystem, ...) ->
# creates acw (/usr/sbin/nologin, home /home/acw mode 750, owner added to group acw, linger) -> /etc/sudoers.d/acw
# (owner -> acw only) and the wrappers acw-ctl/acw-logs/acw-admin/acw-sh in /usr/local/bin -> stops and disables the
# owner's acw-* units, takes a database snapshot -> moves app, app.prev, venv, venv-gpu, bin, models, wasm, data and the
# production secrets (secrets/{acw.env,acw-worker.env,vps.env,backup.env,backup.pass,github-releases-read.token}) to
# /home/acw/audio-cleaner; the owner's tokens and keys stay -> rewrites the venvs' paths, rebuilds the live engine
# against the moved venv -> copies the tunnel key and the VPS host key to /home/acw/.ssh, chowns the backup folder ->
# installs the same unit files in acw's user manager and starts them -> waits for /api/health, runs deploy/smoke.py,
# checks that acw cannot read the owner's files. Downtime: from the stop to the health check, a few minutes.
# Safe to run again after a failure: every step looks at what is already done. The way back: move-back-to-silver.sh.
set -euo pipefail
# shellcheck source=deploy/host/move-lib.sh
. "$(dirname "$0")/move-lib.sh"
parse_args "$0" "$@"
set_direction "$OWNER" "$SVC"
MOVE_BEGUN=0

on_exit() {
  local rc=$?
  [ "$rc" = 0 ] && return 0
  [ "$DRY" = 1 ] && return "$rc"
  {
    echo
    echo "FAILED during: $CURRENT_STEP (exit $rc; the error is above)"
    if [ "$MOVE_BEGUN" = 0 ]; then
      echo "Nothing was moved and the services still run as $OWNER. Fix the cause and run this script again."
    else
      echo "The move had begun: the services may be stopped (or running as $SVC) and new live sessions refused (data/drain)."
      echo "  go on:     fix the cause, then run this script again; it continues where it stopped:"
      echo "               sudo bash $REPO/deploy/host/move-to-acw.sh"
      echo "  go back:   sudo bash $REPO/deploy/host/move-back-to-silver.sh"
      echo "             (puts the folders, the units and the secrets back under $OWNER and starts the services there)"
      echo "  look:      acw-ctl status 'acw-*'     acw-logs acw-web -n 80     (installed by this script)"
    fi
  } >&2
  return "$rc"
}
trap on_exit EXIT

extra_preflight() {
  [ -f "$OWNER_HOME/.ssh/acw_tunnel" ] || [ -f "$TO_HOME/.ssh/acw_tunnel" ] || problem "the tunnel key $OWNER_HOME/.ssh/acw_tunnel is missing"
  if [ -f "$OWNER_HOME/.ssh/known_hosts" ]; then
    ssh-keygen -F vps -f "$OWNER_HOME/.ssh/known_hosts" >/dev/null 2>&1 || [ -f "$TO_HOME/.ssh/known_hosts" ] || problem "no host key for 'vps' in $OWNER_HOME/.ssh/known_hosts (the tunnels use HostKeyAlias=vps)"
  else
    [ -f "$TO_HOME/.ssh/known_hosts" ] || problem "$OWNER_HOME/.ssh/known_hosts does not exist"
  fi
  [ -d "$BACKUP_DIR" ] || problem "$BACKUP_DIR does not exist (is /mnt/mc-storage mounted?)"
  [ -f "$FROM_H/secrets/vps.env" ] || [ -f "$TO_H/secrets/vps.env" ] || problem "secrets/vps.env is missing (the tunnels read it)"
  if [ -d "$OWNER_HOME" ] && [ "$(stat -c %a "$OWNER_HOME" 2>/dev/null || echo 0)" -gt 750 ] 2>/dev/null; then
    problem "$OWNER_HOME is mode $(stat -c %a "$OWNER_HOME"): $SVC could read what is world-readable in it; chmod 750 $OWNER_HOME first"
  fi
}

# ---------------------------------------------------------------- the account, sudoers, wrappers (before the downtime)
acct_create() {
  step "the account $SVC"
  if getent passwd "$SVC" >/dev/null; then
    [ "$(home_of "$SVC")" = "$TO_HOME" ] || die "$SVC exists with home $(home_of "$SVC"), not $TO_HOME"
    say "exists (home $TO_HOME, shell $(getent passwd "$SVC" | cut -d: -f7))"
  else
    run useradd --create-home --home-dir "$TO_HOME" --shell /usr/sbin/nologin --user-group --comment "Audio Cleaner services" "$SVC"
  fi
  run chmod 750 "$TO_HOME"
  if [ "$DRY" = 1 ] || ! id -nG "$OWNER" | tr ' ' '\n' | grep -qx "$SVC"; then
    run usermod -aG "$SVC" "$OWNER"
  fi
  run loginctl enable-linger "$SVC"
  if [ "$DRY" = 1 ]; then
    say "+ systemctl start user@<uid of $SVC>.service; wait for /run/user/<uid>/bus"
  else
    local uid n=0; uid="$(id -u "$SVC")"
    systemctl start "user@$uid.service"
    until [ -S "/run/user/$uid/bus" ]; do
      n=$((n + 1)); [ "$n" -gt 30 ] && die "the user manager of $SVC did not start (no /run/user/$uid/bus)"
      sleep 1
    done
    say "user manager of $SVC is running (uid $uid)"
  fi
}

acct_sudoers() {
  step "sudoers: the owner may run commands as $SVC (never the other way round)"
  local tmp
  tmp="$(mktemp)"
  printf '%s\n' "# Audio Cleaner (deploy/host/move-to-acw.sh): $OWNER may run commands AS $SVC without a password" \
    "# (deploy/host/acw-run). $SVC has no sudo rights: privilege only goes down." \
    "$OWNER ALL=($SVC) NOPASSWD: ALL" > "$tmp"
  if [ "$DRY" = 1 ]; then
    say "+ visudo -cf (validate) and install -m 0440 into /etc/sudoers.d/acw:"; sed 's/^/    /' "$tmp"
  else
    visudo -cf "$tmp" >/dev/null || { rm -f "$tmp"; die "the sudoers drop-in did not validate; nothing installed"; }
    install -m 0440 -o root -g root "$tmp" /etc/sudoers.d/acw
    say "installed /etc/sudoers.d/acw"
  fi
  rm -f "$tmp"
}

acct_wrappers() {
  step "wrappers in /usr/local/bin (sources: deploy/host/)"
  local w
  for w in acw-run acw-ctl acw-logs acw-admin acw-sh; do
    run install -m 755 -o root -g root "$MOVE_DIR/$w" "/usr/local/bin/$w"
  done
}

# ---------------------------------------------------------------- after the move
tunnel_key() {
  step "the tunnel key and the VPS host key for $SVC (the owner's own VPS login key stays with the owner)"
  run install -d -m 700 -o "$SVC" -g "$SVC" "$TO_HOME/.ssh"
  if [ -f "$OWNER_HOME/.ssh/acw_tunnel" ]; then
    run install -m 600 -o "$SVC" -g "$SVC" "$OWNER_HOME/.ssh/acw_tunnel" "$TO_HOME/.ssh/acw_tunnel"
    [ -f "$OWNER_HOME/.ssh/acw_tunnel.pub" ] && run install -m 644 -o "$SVC" -g "$SVC" "$OWNER_HOME/.ssh/acw_tunnel.pub" "$TO_HOME/.ssh/acw_tunnel.pub"
  fi
  if [ "$DRY" = 1 ]; then
    say "+ ssh-keygen -F vps -f $OWNER_HOME/.ssh/known_hosts  -> append the line(s) to $TO_HOME/.ssh/known_hosts (0600, $SVC)"
  elif [ -f "$OWNER_HOME/.ssh/known_hosts" ]; then
    local lines kh="$TO_HOME/.ssh/known_hosts" l
    lines="$(ssh-keygen -F vps -f "$OWNER_HOME/.ssh/known_hosts" 2>/dev/null | grep -v '^#' || true)"
    [ -n "$lines" ] || [ -f "$kh" ] || die "no host key for 'vps' in $OWNER_HOME/.ssh/known_hosts"
    touch "$kh"
    while IFS= read -r l; do [ -z "$l" ] || grep -qxF "$l" "$kh" || printf '%s\n' "$l" >> "$kh"; done <<< "$lines"
    chown "$SVC:$SVC" "$kh"; chmod 600 "$kh"
    say "host key for 'vps' is in $kh ($(grep -c . "$kh") line(s))"
  fi
}

backups_dir() {
  step "the backup folder $BACKUP_DIR belongs to $SVC now"
  run chown -R "$SVC:$SVC" "$BACKUP_DIR"
  run chmod 750 "$BACKUP_DIR"
}

summary() {
  step "done"
  cat <<TXT
The services run as $SVC. What changes for you ($OWNER):
  deploy:      cd ~/repos/Audio-Cleaner-Web && deploy/install.sh --wait-live      (unchanged; it stages the commit for $SVC)
  services:    acw-ctl status 'acw-*'     acw-ctl restart acw-web      (was: systemctl --user ...)
  logs:        acw-logs acw-web -f                                      (was: journalctl --user -u acw-web -f)
  admin:       acw-admin activity     acw-admin stats                   (was: python -m acw.admin ... in ~/audio-cleaner/app)
  a shell:     acw-sh
The owner's ~/audio-cleaner/{bin,models,wasm,venv,venv-gpu} are links to $TO_H/...; src, dev, research stay as they were.
Group $SVC was added to $OWNER: a login session that started before only gets it after logging in again (an ACL entry
already gives $OWNER read access to the shared folders meanwhile).
To go back:  sudo bash $REPO/deploy/host/move-back-to-silver.sh
TXT
}

# ================================================================== main
m_preflight
if [ "$DRY" = 1 ] && [ "$DRY_PROBLEMS" = 1 ]; then say "(dry run goes on to list the steps)"; fi
confirm "This moves the production services of Audio Cleaner from $OWNER to $SVC. Downtime: a few minutes."
acct_create
acct_sudoers
acct_wrappers
MOVE_BEGUN=1
m_quiesce
m_move
tunnel_key
backups_dir
m_rebuild_live
m_units_start
m_smoke
m_isolation
summary
if [ "$DRY" = 1 ] && [ "$DRY_PROBLEMS" = 1 ]; then exit 1; fi
