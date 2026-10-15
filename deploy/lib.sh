#!/usr/bin/env bash
# Shared helpers for install.sh, rollback.sh and drain.sh (sourced, not run).
H="${ACW_HOME:-$HOME/audio-cleaner}"

# Run an acw.admin command with the deployed code and the worker environment (no secrets).
acw_admin() {
  # shellcheck disable=SC1091
  ( cd "$H/app" && set -a && . "$H/secrets/acw-worker.env" && set +a && PYTHONPATH="$H/app" "$H/venv/bin/python" -m acw.admin "$@" )
}

# Number of live sessions running now (0 when nothing is deployed yet).
live_sessions() {
  [ -x "$H/venv/bin/python" ] && [ -d "$H/app/acw" ] && [ -f "$H/secrets/acw-worker.env" ] || { echo 0; return; }
  acw_admin live-count 2>/dev/null || echo 0
}

# Stop new live sessions from starting (they get a friendly "back in a minute").
drain_on() { touch "$H/data/drain"; }
drain_off() { rm -f "$H/data/drain"; }

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

# Poll until every unit in "$@" is active and the deep health check passes (and the bulk process answers, when
# acw-bulk.service is among them), at most 30 s.
check_running() {
  local deadline=$(( $(date +%s) + 30 )) u bad health="" bulk="" ok=0
  while :; do
    bad=""
    for u in "$@"; do
      systemctl --user is-active --quiet "$u" || bad="$bad $u"
    done
    if [ -z "$bad" ]; then
      health="$(curl -fsS --max-time 5 "http://127.0.0.1:8790/audio/api/health?deep=1" 2>/dev/null || true)"
      case "$health" in *'"ok":true'*|*'"ok": true'*) ok=1 ;; esac
      # The bulk-transfer process answers on its own port (it serves uploads and downloads only).
      case " $* " in *" acw-bulk.service "*)
        bulk="$(curl -fsS --max-time 5 "http://127.0.0.1:8794/audio/api/health" 2>/dev/null || true)"
        case "$bulk" in *'"ok":true'*|*'"ok": true'*) ;; *) ok=0; health="bulk: ${bulk:-no answer}; $health" ;; esac ;;
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
  [ -n "$bad" ] && echo "not active after 30 s:$bad" >&2
  echo "health: ${health:-no answer}" >&2
  return 1
}
