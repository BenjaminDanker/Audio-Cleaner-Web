#!/usr/bin/env bash
# Publish the VPS's Caddy sites (deploy/vps/sites/: one settings file per domain) and reload Caddy, from primary-ram, without
# sudo: the sites live in /srv/caddy/sites (owned by the normal user) and Caddy reloads through its local admin API.
#   deploy/vps/publish-sites.sh            copy sites/ to /srv/caddy/sites (through /srv/caddy/publishing), reload Caddy; when
#                                          the new sites do not load, the previous ones (/srv/caddy/previous) are put back
#   deploy/vps/publish-sites.sh --setup    copy what setup.sh needs to vps:~/vps-setup/ (then, with sudo, on the VPS:
#                                          ssh -t vps 'sudo bash ~/vps-setup/setup.sh && rm -rf ~/vps-setup')
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ "${1:-}" = "--setup" ]; then
  rsync -a --delete --exclude '*.example' "$HERE/" vps:vps-setup/
  echo "copied to vps:~/vps-setup/; now run:  ssh -t vps 'sudo bash ~/vps-setup/setup.sh && rm -rf ~/vps-setup'"
  exit 0
fi

ssh vps 'test -w /srv/caddy/sites && grep -q "import /srv/caddy/sites/" /etc/caddy/Caddyfile' \
  || { echo "the VPS is not set up for this yet: deploy/vps/publish-sites.sh --setup, then the sudo command it prints" >&2; exit 1; }
rsync -a --delete "$HERE/sites/" vps:/srv/caddy/publishing/
ssh vps 'set -e
  rsync -a --delete /srv/caddy/sites/ /srv/caddy/previous/
  rsync -a --delete /srv/caddy/publishing/ /srv/caddy/sites/
  if caddy reload --config /etc/caddy/Caddyfile --adapter caddyfile 2>/tmp/caddy-reload.log; then
    echo "Caddy reloaded with the new sites"
  else
    cat /tmp/caddy-reload.log >&2
    rsync -a --delete /srv/caddy/previous/ /srv/caddy/sites/
    echo "the new sites did not load; the previous ones are back (Caddy kept running them)" >&2
    exit 1
  fi'
