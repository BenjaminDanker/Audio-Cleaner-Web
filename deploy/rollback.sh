#!/usr/bin/env bash
# Put the previous copy of the code back (deploy/install.sh keeps it as ~/audio-cleaner/app.prev).
#   deploy/rollback.sh [--wait-live[=SECS]] [--force]
# Swaps app and app.prev (the release mirror's timer follows the code), restarts the web process, the bulk-transfer process, the file workers and the live worker, and checks them.
# Only code goes back: units, packages and the database stay as they are (schema changes are additive, the
# previous code ignores new tables and columns). Running it again swaps forward again.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=deploy/lib.sh
. "$REPO/deploy/lib.sh"

FORCE=0
WAIT_LIVE=""
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --wait-live) WAIT_LIVE=600 ;;
    --wait-live=*) WAIT_LIVE="${arg#*=}" ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) echo "unknown option $arg" >&2; exit 2 ;;
  esac
done

[ -d "$H/app.prev/acw" ] || { echo "no previous copy at $H/app.prev: nothing to roll back to" >&2; exit 1; }
echo "now:  $(cat "$H/app/VERSION" 2>/dev/null || echo unknown)"
echo "back: $(cat "$H/app.prev/VERSION" 2>/dev/null || echo unknown)"
echo "running now: $(acw_admin activity 2>/dev/null || echo unknown)"

DRAINING=0
trap '[ "$DRAINING" = 1 ] && drain_off; true' EXIT
LIVE="$(live_sessions)"
if [ "$LIVE" != "0" ]; then
  if [ -n "$WAIT_LIVE" ]; then
    drain_on; DRAINING=1
    wait_for_live "$WAIT_LIVE" || { [ "$FORCE" = 1 ] || { echo "live sessions are still running; not rolling back (--force restarts anyway)" >&2; exit 1; }; }
  elif [ "$FORCE" != 1 ]; then
    echo "$LIVE live session(s) are running and the rollback restarts the web and live workers." >&2
    echo "Use --wait-live[=SECONDS] or --force." >&2
    exit 1
  fi
fi

mv "$H/app" "$H/app.rollback-tmp"
mv "$H/app.prev" "$H/app"
mv "$H/app.rollback-tmp" "$H/app.prev"

UNITS="acw-stream.service acw-web.service"
# A copy of the code from before the bulk-transfer process existed has no acw/bulk_app.py to start.
[ -f "$H/app/acw/bulk_app.py" ] && UNITS="$UNITS acw-bulk.service"
# ... and from before the live process existed, none of acw/live_app.py: stop that unit then (acw-web serves live).
if [ -f "$H/app/acw/live_app.py" ]; then
  UNITS="$UNITS acw-live.service"
else
  systemctl --user stop acw-live.service acw-tunnel-live.service 2>/dev/null || true
fi
for unit in $(systemctl --user list-units --plain --no-legend 'acw-worker@*.service' 'acw-worker-gpu.service' | awk '{print $1}'); do
  UNITS="$UNITS $unit"
done
# shellcheck disable=SC2086
systemctl --user restart $UNITS
# Code from before the release mirror existed has no acw/mirror.py: its timer would only fail and mail.
if [ -f "$H/app/acw/mirror.py" ]; then
  systemctl --user start acw-mirror.timer 2>/dev/null || true
else
  systemctl --user stop acw-mirror.timer 2>/dev/null || true
  echo "note: this copy of the code has no release mirror: acw-mirror.timer is stopped (it starts again when you roll forward or deploy)" >&2
fi
BULK_TUNNEL=""
systemctl --user is-enabled --quiet acw-tunnel-bulk.service 2>/dev/null && BULK_TUNNEL="acw-tunnel-bulk.service"
LIVE_TUNNEL=""
[ -f "$H/app/acw/live_app.py" ] && systemctl --user is-enabled --quiet acw-tunnel-live.service 2>/dev/null && LIVE_TUNNEL="acw-tunnel-live.service"
# shellcheck disable=SC2086
if check_running $UNITS acw-tunnel.service $BULK_TUNNEL $LIVE_TUNNEL; then
  echo "rolled back to $(cat "$H/app/VERSION" 2>/dev/null || echo unknown); run again to undo"
else
  echo "the rolled-back code did not come up cleanly either; look at: journalctl --user -u acw-web -n 80" >&2
  exit 1
fi
