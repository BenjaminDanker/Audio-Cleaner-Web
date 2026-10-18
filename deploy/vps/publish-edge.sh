#!/usr/bin/env bash
# Publish the VPS's Caddy sites (deploy/vps/sites/) and reload Caddy, from primary-ram, without sudo: the sites live
# in /srv/edge (owned by the normal user) and Caddy reloads through its local admin API.
#   deploy/vps/publish-edge.sh            copy sites/ to vps:/srv/edge, reload Caddy; a site that does not load is
#                                         put back as it was
#   deploy/vps/publish-edge.sh --setup    copy everything setup.sh needs to vps:~/edge-src/ (then, once, with sudo:
#                                         ssh -t vps 'sudo bash ~/edge-src/setup.sh')
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ "${1:-}" = "--setup" ]; then
  rsync -a --delete --exclude '*.example' "$HERE/" vps:edge-src/
  echo "copied to vps:~/edge-src/; now run:  ssh -t vps 'sudo bash ~/edge-src/setup.sh'"
  exit 0
fi

ssh vps 'test -w /srv/edge && grep -q "import /srv/edge/" /etc/caddy/Caddyfile' \
  || { echo "the VPS is not set up for this yet: deploy/vps/publish-edge.sh --setup, then the sudo command it prints" >&2; exit 1; }
rsync -a --delete "$HERE/sites/" vps:edge-staging/
ssh vps 'set -e
  rm -rf ~/edge-previous && cp -a /srv/edge ~/edge-previous
  rsync -a --delete ~/edge-staging/ /srv/edge/
  if caddy reload --config /etc/caddy/Caddyfile --adapter caddyfile 2>/tmp/edge-reload.log; then
    echo "Caddy reloaded with the new sites"
  else
    cat /tmp/edge-reload.log >&2
    rsync -a --delete ~/edge-previous/ /srv/edge/
    echo "the new sites did not load; the previous ones are back (Caddy kept running them)" >&2
    exit 1
  fi'
