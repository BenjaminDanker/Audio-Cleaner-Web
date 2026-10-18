#!/usr/bin/env bash
# Stop new live sessions from starting (running ones continue), e.g. before a restart.
#   deploy/drain.sh on | off | status | wait [SECONDS]
set -euo pipefail
# shellcheck source=deploy/lib.sh
. "$(dirname "$0")/lib.sh"
case "${1:-status}" in
  on) drain_on; echo "drain on: new live sessions are refused" ;;
  off) drain_off; echo "drain off" ;;
  status) as_acw test -e "$H/data/drain" && echo "drain on" || echo "drain off"; echo "live sessions: $(live_sessions)" ;;
  wait)
    drain_on
    if wait_for_live "${2:-600}"; then echo "no live sessions left"; else echo "still $(live_sessions) live session(s)"; exit 1; fi ;;
  *) echo "usage: $0 on|off|status|wait [seconds]" >&2; exit 2 ;;
esac
