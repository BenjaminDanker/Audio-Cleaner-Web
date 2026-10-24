#!/usr/bin/env bash
# Set up the VPS (the public front door for Audio Cleaner and the Minecraft network), or bring an existing one in line.
# Safe to run again. Run on the VPS with sudo, from the folder publish-sites.sh --setup copied:
#     sudo bash ~/vps-setup/setup.sh && rm -rf ~/vps-setup
# What it does (docs/vps.md has the why):
#   1. packages: Caddy (official repository), ufw, fail2ban, unattended-upgrades
#   2. SSH: keys only, no root login
#   3. firewall: only the ports below are open
#   4. folders the normal user owns, by project: /srv/audio-cleaner/website, /srv/minecraft/resource-packs, /srv/caddy/{sites,
#      publishing,previous}; files of the older layout (/var/www/audio, /var/www/pack, /srv/edge) are copied over first
#   5. frps (the Minecraft tunnel) as its own user, from /opt/frp and /etc/frp, instead of root and /root
#   6. Caddy: /etc/caddy/Caddyfile imports /srv/caddy/sites/*.caddy, so publish-sites.sh changes sites without sudo; once it
#      runs on the new folders, the older layout's folders are removed
#   7. switches off services a small VPS doesn't use (modem, disk, multipath managers)
#   8. memory: a 1 GB swap file (the VPS has 1 GB of RAM and nothing to fall back on)
#   9. logs: the system journal is capped at 100 MB; rotated logs of failed logins are deleted
# Every step that replaces something keeps a copy and puts it back when the result does not start.
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run with sudo: sudo bash $0" >&2; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
OWNER="${SUDO_USER:-silver}"
STAMP="$(date -u +%Y%m%d%H%M%S)"
say() { printf '\n== %s\n' "$*"; }

say "1. packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q ufw fail2ban unattended-upgrades curl gnupg debian-keyring debian-archive-keyring apt-transport-https >/dev/null
if ! command -v caddy >/dev/null; then
  install -d -m 0755 /etc/apt/keyrings
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key | gpg --dearmor -o /etc/apt/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt \
    | sed 's#deb #deb [signed-by=/etc/apt/keyrings/caddy-stable-archive-keyring.gpg] #; s#deb-src #deb-src [signed-by=/etc/apt/keyrings/caddy-stable-archive-keyring.gpg] #' \
    > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -q && apt-get install -y -q caddy >/dev/null
fi
systemctl enable --now fail2ban unattended-upgrades >/dev/null 2>&1
echo "caddy $(caddy version | cut -d' ' -f1), ufw, fail2ban, unattended-upgrades"

say "2. SSH: keys only"
cat > /etc/ssh/sshd_config.d/00-keys-only.conf <<'EOF'
# deploy/vps/setup.sh: keys only, no root login (the tunnel keys are restricted in ~/.ssh/authorized_keys)
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin no
EOF
sshd -t && systemctl reload ssh
echo "ok"

say "3. firewall"
# 22 SSH; 80/443 Caddy (443/udp: HTTP/3); 7000 frp clients; 25565 Minecraft Java; 19132/udp Minecraft Bedrock.
# The SSH tunnels from home listen on 127.0.0.1 only and need no rule.
ufw default deny incoming >/dev/null
ufw default allow outgoing >/dev/null
for rule in 22/tcp 80/tcp 443/tcp 443/udp 7000/tcp 25565/tcp 19132/udp; do ufw allow "$rule" >/dev/null; done
ufw --force enable >/dev/null
ufw status | sed -n '1,30p'

say "4. folders for $OWNER"
install -d -m 0755 /srv/audio-cleaner /srv/minecraft /srv/caddy
install -d -o "$OWNER" -g "$OWNER" -m 0755 /srv/audio-cleaner/website /srv/minecraft/resource-packs \
  /srv/caddy/sites /srv/caddy/publishing /srv/caddy/previous
# The older layout's files, copied (not moved: Caddy keeps serving them until step 6 switches it over).
copy_old() { if [ -d "$1" ] && [ -z "$(ls -A "$2")" ]; then cp -a "$1"/. "$2"/ && chown -R "$OWNER:$OWNER" "$2" && echo "copied $1 -> $2"; fi; }
copy_old /var/www/audio /srv/audio-cleaner/website
copy_old /var/www/pack /srv/minecraft/resource-packs
echo "/srv/audio-cleaner/website /srv/minecraft/resource-packs /srv/caddy/{sites,publishing,previous}"

say "5. frps as its own user"
OLD=/root/frp_0.65.0_linux_amd64
id frp >/dev/null 2>&1 || useradd --system --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin frp
install -d -m 0755 /opt/frp
install -d -o root -g frp -m 0750 /etc/frp
[ -x /opt/frp/frps ] || { [ -x "$OLD/frps" ] && install -m 0755 "$OLD/frps" /opt/frp/frps; }
[ -f /etc/frp/frps.ini ] || { [ -f "$OLD/frps.ini" ] && install -o root -g frp -m 0640 "$OLD/frps.ini" /etc/frp/frps.ini; }
if [ ! -x /opt/frp/frps ] || [ ! -f /etc/frp/frps.ini ]; then
  echo "no frps binary or config (put frps in /opt/frp/ and frps.ini in /etc/frp/, see docs/vps.md); skipping" >&2
else
  # A log file elsewhere (e.g. under /root) is not writable for the frp user: logs go to /var/log/frp.
  sed -i -E 's#^([[:space:]]*log_file[[:space:]]*=[[:space:]]*).*#\1/var/log/frp/frps.log#' /etc/frp/frps.ini
  grep -Eq '^[[:space:]]*(token|auth\.token)[[:space:]]*=[[:space:]]*[^[:space:]]+' /etc/frp/frps.ini \
    && echo "frps.ini has a token: ok" \
    || echo "WARNING: /etc/frp/frps.ini has no token: anyone can open tunnels through port 7000. Add  token = <long random>  to its [common] section and the same to the client's frpc.ini." >&2
  UNIT=/etc/systemd/system/frps.service
  if cmp -s "$HERE/frps.service" "$UNIT" && systemctl is-active --quiet frps; then
    echo "frps unchanged and running (not restarted: that would drop the Minecraft players)"
  else
  [ -f "$UNIT" ] && cp -a "$UNIT" "$UNIT.backup.$STAMP"
  install -m 0644 "$HERE/frps.service" "$UNIT"
  systemctl daemon-reload
  systemctl restart frps
  sleep 3
  if systemctl is-active --quiet frps; then
    echo "frps runs as user frp (old files in $OLD can go once you are happy: sudo rm -r $OLD)"
  else
    echo "frps did not start as user frp; putting the old unit back" >&2
    journalctl -u frps -n 20 --no-pager >&2 || true
    if [ -f "$UNIT.backup.$STAMP" ]; then cp -a "$UNIT.backup.$STAMP" "$UNIT"; systemctl daemon-reload; systemctl restart frps; fi
  fi
  fi
fi

say "6. Caddy"
cp -a "$HERE"/sites/*.caddy /srv/caddy/sites/
chown "$OWNER:$OWNER" /srv/caddy/sites/*.caddy
CF=/etc/caddy/Caddyfile
cp -a "$CF" "$CF.backup.$STAMP"
install -m 0644 "$HERE/Caddyfile" "$CF"
if caddy validate --config "$CF" --adapter caddyfile >/dev/null 2>&1; then
  systemctl reload caddy
  echo "Caddy reloaded: sites from /srv/caddy/sites (backup of the old file: $CF.backup.$STAMP)"
  # The older layout (before 2026-10-24), now that Caddy runs on the new folders.
  OWNER_HOME="$(getent passwd "$OWNER" | cut -d: -f6)"
  for old in /srv/edge /var/www/audio /var/www/pack "$OWNER_HOME/edge-staging" "$OWNER_HOME/edge-previous" "$OWNER_HOME/edge-src" "$OWNER_HOME/pack"; do
    [ -e "$old" ] && rm -rf "$old" && echo "removed $old (older layout)"
  done
  rmdir /var/www 2>/dev/null && echo "removed /var/www (empty)" || true
else
  caddy validate --config "$CF" --adapter caddyfile || true
  cp -a "$CF.backup.$STAMP" "$CF"
  echo "the new Caddy configuration did not validate; the old one is back" >&2
  exit 1
fi

say "7. services a small VPS doesn't use"
for u in ModemManager.service udisks2.service multipathd.service multipathd.socket; do
  systemctl disable --now "$u" >/dev/null 2>&1 && echo "off: $u" || true
done

say "8. memory: swap"
if [ -z "$(swapon --noheadings --show=NAME)" ]; then
  fallocate -l 1G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
  grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
  echo "1 GB swap file on"
else
  echo "swap already on: $(swapon --noheadings --show=NAME,SIZE | tr '\n' ' ')"
fi
# Use it only under pressure (RAM stays for Caddy and frps).
echo 'vm.swappiness = 10' > /etc/sysctl.d/90-swappiness.conf && sysctl -q -p /etc/sysctl.d/90-swappiness.conf

say "9. logs"
install -d -m 0755 /etc/systemd/journald.conf.d
printf '[Journal]\nSystemMaxUse=100M\n' > /etc/systemd/journald.conf.d/90-size.conf
systemctl restart systemd-journald
journalctl --vacuum-size=100M >/dev/null 2>&1 || true
rm -f /var/log/btmp.[0-9]* && echo "journal capped at 100 MB; old failed-login logs removed"

say "done"
echo "From now on, Caddy's sites change without sudo: deploy/vps/publish-sites.sh (from primary-ram)."
[ -f /var/run/reboot-required ] && echo "A reboot is pending (new kernel): sudo reboot, at a quiet time (about a minute down)." || true
