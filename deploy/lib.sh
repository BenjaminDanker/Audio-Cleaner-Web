#!/usr/bin/env bash
# Shared helpers for install.sh, rollback.sh, drain.sh, publish-web.sh and host/move-*.sh (sourced, not run).
#
# Two accounts: the services run as the service account (acw: /etc/acw-service-user names it, written by
# deploy/host/move-to-acw.sh), the owner (silver) deploys and has the VPS login key. When this is sourced in the
# owner's shell ACW_DELEGATE=1: H is acw's folder, and every helper that touches it (acw_admin, drain_on, ctl,
# live_sessions, ...) goes through `as_acw` (sudo -u acw via deploy/host/acw-run). Sourced by acw itself, or on a
# machine without the service account, ACW_DELEGATE=0 and everything runs directly, as before the split.
# Overrides: ACW_SERVICE_USER (set it empty to force the single-account behaviour), ACW_DELEGATE=0|1, ACW_RUN.
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# deploy/host/move-*.sh leave this file while the folders and units are between two accounts: nothing may deploy then.
if [ -e "${ACW_MOVE_STATE_DIR:-/var/lib/acw-move}/in-progress" ] && [ -z "${ACW_ALLOW_UNFINISHED_MOVE:-}" ]; then
  echo "a move of the services between accounts is unfinished ($(cat "${ACW_MOVE_STATE_DIR:-/var/lib/acw-move}/in-progress")): run deploy/host/move-to-acw.sh again to finish it, or deploy/host/move-back-to-silver.sh" >&2
  return 1
fi
ACW_SERVICE_USER="${ACW_SERVICE_USER-$(cat /etc/acw-service-user 2>/dev/null || true)}"
if [ -z "${ACW_DELEGATE:-}" ]; then
  ACW_DELEGATE=0
  if [ -n "$ACW_SERVICE_USER" ] && [ "$(id -un)" != "$ACW_SERVICE_USER" ]; then ACW_DELEGATE=1; fi
fi
if [ "$ACW_DELEGATE" = 1 ]; then
  ACW_RUN="${ACW_RUN:-$LIB_DIR/host/acw-run}"
  ACW_SERVICE_HOME="${ACW_SERVICE_HOME_DIR:-$(getent passwd "$ACW_SERVICE_USER" | cut -d: -f6 || true)}"  # (the override is a test hook)
  if [ -z "$ACW_SERVICE_HOME" ]; then
    echo "/etc/acw-service-user names '$ACW_SERVICE_USER', but there is no such account (deploy/host/move-back-to-silver.sh removes the file)" >&2
    return 1
  fi
  H="$ACW_SERVICE_HOME/audio-cleaner"
else
  H="${ACW_HOME:-$HOME/audio-cleaner}"
fi

# Run a command as the service account (directly when this already is that account). Absolute paths only: the
# working directory is / when delegating.
as_acw() {
  if [ "$ACW_DELEGATE" = 1 ]; then ACW_USER="$ACW_SERVICE_USER" "$ACW_RUN" "$@"; else "$@"; fi
}

# systemctl --user of the service account's manager.
ctl() { as_acw systemctl --user "$@"; }

# Put the files of the commit checked out in $REPO (git archive HEAD, only the given paths) into a fresh folder owned
# by the service account, <acw home>/deploy/NAME, and print its path. That is what the server part of a deploy runs
# from: acw cannot read the owner's checkout, and what it runs is exactly the commit that was checked. The newest
# four folders are kept.
stage_for_acw() {  # stage_for_acw NAME PATH...
  local name="$1" root dir; shift
  root="$(dirname "$H")/deploy"; dir="$root/$name"
  as_acw mkdir -p -m 700 "$root"
  as_acw rm -rf "$dir"
  as_acw mkdir -m 700 "$dir"
  # shellcheck disable=SC2154  # REPO is set by the script that sources this file
  git -C "$REPO" archive --format=tar HEAD -- "$@" | as_acw tar -x -C "$dir" -f -
  as_acw bash -c 'cd "$1" && ls -1dt -- */ 2>/dev/null | tail -n +5 | xargs -r rm -rf --' stage-prune "$root"
  printf '%s\n' "$dir"
}

# Copy a file of the owner into a place of the service account: put_as_acw SRC DEST MODE
put_as_acw() {
  as_acw mkdir -p "$(dirname "$2")"
  as_acw bash -c 'cat > "$1.part" && chmod "$2" "$1.part" && mv -f "$1.part" "$1"' put "$2" "$3" < "$1"
}

# Run an acw.admin command with the deployed code and the worker environment (no secrets).
acw_admin() {
  as_acw bash -c 'cd "$1/app" && set -a && . "$1/secrets/acw-worker.env" && set +a && PYTHONPATH="$1/app" exec "$1/venv/bin/python" -m acw.admin "${@:2}"' acw-admin "$H" "$@"
}

# Number of live sessions running now (0 when nothing is deployed yet).
live_sessions() {
  as_acw test -x "$H/venv/bin/python" && as_acw test -d "$H/app/acw" && as_acw test -f "$H/secrets/acw-worker.env" || { echo 0; return; }
  acw_admin live-count 2>/dev/null || echo 0
}

# Stop new live sessions from starting (they get a friendly "back in a minute").
drain_on() { as_acw touch "$H/data/drain"; }
drain_off() { as_acw rm -f "$H/data/drain"; }

# Wait up to $1 seconds for the live sessions to end. Returns 0 when none are left.
wait_for_live() {
  local limit="$1" waited=0 n
  while :; do
    n="$(live_sessions)"
    [ "$n" = "0" ] && return 0
    [ "$waited" -ge "$limit" ] && return 1
    echo "waiting for $n live session(s) to end (${waited}s of ${limit}s)"
    sleep 15
    waited=$(( waited + 15 ))
  done
}

# Poll until every unit in "$@" is active and the deep health check passes (and the bulk and live processes answer,
# when acw-bulk.service / acw-live.service are among them), at most ${ACW_CHECK_SECONDS:-30} s.
check_running() {
  local deadline=$(( $(date +%s) + ${ACW_CHECK_SECONDS:-30} )) u bad health="" bulk="" live="" ok=0
  while :; do
    bad=""
    for u in "$@"; do
      ctl is-active --quiet "$u" || bad="$bad $u"
    done
    if [ -z "$bad" ]; then
      health="$(curl -fsS --max-time 5 "http://127.0.0.1:8790/api/health?deep=1" 2>/dev/null || true)"
      case "$health" in *'"ok":true'*|*'"ok": true'*) ok=1 ;; esac
      # The bulk-transfer process answers on its own port (it serves uploads and downloads only).
      case " $* " in *" acw-bulk.service "*)
        bulk="$(curl -fsS --max-time 5 "http://127.0.0.1:8794/api/health" 2>/dev/null || true)"
        case "$bulk" in *'"ok":true'*|*'"ok": true'*) ;; *) ok=0; health="bulk: ${bulk:-no answer}; $health" ;; esac ;;
      esac
      # So does the live process (the live WebSocket only).
      case " $* " in *" acw-live.service "*)
        live="$(curl -fsS --max-time 5 "http://127.0.0.1:8795/api/health" 2>/dev/null || true)"
        case "$live" in *'"ok":true'*|*'"ok": true'*) ;; *) ok=0; health="live: ${live:-no answer}; $health" ;; esac ;;
      esac
    fi
    [ "$ok" = 1 ] && break
    [ "$(date +%s)" -ge "$deadline" ] && break
    sleep 2
  done
  if [ "$ok" = 1 ]; then
    echo "all units active, deep health ok"
    return 0
  fi
  [ -n "$bad" ] && echo "not active after ${ACW_CHECK_SECONDS:-30} s:$bad" >&2
  echo "health: ${health:-no answer}" >&2
  return 1
}
