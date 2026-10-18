#!/usr/bin/env bash
# primary-ram: Tailscale carries SSH and nothing else. Run once with sudo (safe to run again):
#     sudo bash deploy/host/tailscale-ssh-only.sh
#   * in: over tailscale0 only port 22 is reachable (the Minecraft servers, the website and everything else that
#     listens on all addresses are not reachable from the tailnet any more); SSH over the LAN is unchanged
#   * out: no exit node, no routes from other tailnet machines, and DNS no longer goes through Tailscale, so the only
#     traffic on tailscale0 is to and from your own Tailscale devices
# Other firewall rules and the default policy are left as they are.
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run with sudo" >&2; exit 1; }

tailscale set --accept-dns=false --accept-routes=false --exit-node= --advertise-exit-node=false

# ufw takes the first rule that matches: the allow goes in before the deny.
ufw delete allow in on tailscale0 to any port 22 proto tcp >/dev/null 2>&1 || true
ufw delete deny in on tailscale0 >/dev/null 2>&1 || true
ufw allow in on tailscale0 to any port 22 proto tcp comment 'Tailscale: SSH only' >/dev/null
ufw deny in on tailscale0 comment 'Tailscale: nothing but SSH' >/dev/null
ufw --force enable >/dev/null
ufw status verbose | sed -n '1,40p'
tailscale debug prefs | grep -E '"(CorpDNS|RouteAll|ExitNodeID|AdvertiseRoutes)"'
echo "done: Tailscale carries SSH only"
