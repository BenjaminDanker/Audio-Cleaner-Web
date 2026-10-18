# The VPS (edge): what lives there and how to rebuild it

One small VPS (IONOS VPS Linux XS: 1 vCPU, 1 GB, Ubuntu 24.04, Kansas City; its address is in `~/.ssh/config` as
`vps` (the owner's) and in `/home/acw/audio-cleaner/secrets/vps.env` (the tunnels'), never in the repo) is the public front door for two things:

| Name | What | How it reaches home |
|---|---|---|
| `audiocleaner.live` (and `www.`, redirected) | Audio Cleaner: the website (served from this VPS, `/var/www/audio`) and its API | SSH reverse tunnels that primary-ram opens (below) |
| `mc-multi-world.space` | the Minecraft network's web address: resource packs (`/var/www/pack`), the pet companion's Stripe checkout (another project, its own tunnel on 18788), and the old Audio Cleaner address `/audio` (redirects; its API still answers for old OBS plugin builds) | |
| Minecraft (ports 25565, 19132/udp) | the Minecraft network itself | frp: `frps` here, `frpc` on the Velocity proxy machine at home (192.168.1.200) |

Nothing else of ours is on the VPS: no data, no secrets beyond the frp token. The free tools' models come from
Hugging Face, not from here.

Measured 2026-10-18: the VPS's own link does about 1.3 Gbit/s down and 0.5-0.9 Gbit/s up (to Cloudflare);
primary-ram reaches only about 90 Mbit/s down and 85 up to anywhere (even with 4 connections at once, its network
card at 1 Gbit/s and the line idle), so the ~60-90 Mbit/s between home and the VPS is the home side, not the VPS.
TCP round trip from home: about 27 ms.

## What runs there

| Piece | Runs as | Config (in the repo) | Notes |
|---|---|---|---|
| Caddy 2 (apt, Caddy's repository) | `caddy` (system service) | `/etc/caddy/Caddyfile` = `deploy/vps/Caddyfile`: global options + `import /srv/edge/*.caddy`; the sites in `/srv/edge` = `deploy/vps/sites/` | HTTPS certificates by itself (ports 80/443). The sites folder belongs to `silver` and Caddy reloads through its admin API on 127.0.0.1:2019, so **changing a site needs no sudo**: `deploy/vps/publish-edge.sh` |
| frps (frp 0.65.0) | `frp` (own system user) | `/opt/frp/frps`, `/etc/frp/frps.ini` (root:frp 0640, **holds the token: not in the repo**), unit `deploy/vps/frps.service` | clients connect on 7000; Minecraft on 25565 (Java) and 19132/udp (Bedrock); dashboard on 127.0.0.1:7500 only |
| SSH tunnels from home | `silver` (the tunnel keys' listen ports) | `~/.ssh/authorized_keys` (below) | 127.0.0.1:18790 acw-web, 18794 acw-bulk, 18795 acw-live; 18788 pet companion |
| ufw | | `deploy/vps/setup.sh` | open: 22, 80, 443 (+443/udp), 7000, 25565, 19132/udp; everything else closed |
| sshd | | `/etc/ssh/sshd_config.d/00-keys-only.conf` (setup.sh) | keys only, no root login; fail2ban watches it |
| unattended-upgrades | | Ubuntu default | security updates by themselves |
| Website files | `silver` | `/var/www/audio` | `deploy/publish-web.sh` (from primary-ram, rsync) |
| Resource packs | `silver` | `/var/www/pack` | copied there by hand |

Sudo is needed only for `setup.sh` (packages, firewall, users, the main Caddyfile) and frp's config. Everything
that changes routinely (sites, website files, packs) belongs to `silver`.

### The tunnel keys (`~/.ssh/authorized_keys` of `silver` on the VPS)

Each key may only open its own listening ports, nothing else (no shell, no outgoing forwards):

    restrict,port-forwarding,permitlisten="127.0.0.1:18790",permitlisten="127.0.0.1:18794",permitlisten="127.0.0.1:18795",permitopen="127.0.0.1:1",command="/bin/false" ssh-ed25519 </home/acw/.ssh/acw_tunnel.pub of primary-ram> acw-tunnel@primary-ram
    restrict,port-forwarding,permitlisten="127.0.0.1:18788",permitopen="127.0.0.1:8788",command="/bin/false" ssh-ed25519 <its key> pet-companion-vps-tunnel

plus your own login key(s). The tunnel key lives with the service account (`/home/acw/.ssh/acw_tunnel`, the VPS host key under the alias `vps`
in `/home/acw/.ssh/known_hosts`); your own VPS login key never leaves `/home/silver/.ssh` (docs/security.md, "Accounts").

## Change something

- **A Caddy site** (routes, headers, a new domain): edit `deploy/vps/sites/*.caddy`, then on primary-ram
  `deploy/vps/publish-edge.sh`. It copies them to `/srv/edge` and reloads Caddy; a configuration that does not load is
  put back and Caddy keeps running the old one.
- **The website**: `deploy/publish-web.sh` (`deploy/install.sh` runs it when `web/` changed).
- **Firewall, packages, frp's unit, the main Caddyfile**: edit them in `deploy/vps/`, then
  `deploy/vps/publish-edge.sh --setup` and `ssh -t vps 'sudo bash ~/edge-src/setup.sh'` (safe to run again).
- **frp's settings or token**: `sudo nano /etc/frp/frps.ini` on the VPS, `sudo systemctl restart frps`; the client's
  `frpc.ini` on the proxy machine must have the same token.

## Rebuild on a new VPS

1. Create the VPS (Ubuntu 24.04), your user `silver` with sudo and your SSH key. Note its address.
2. On primary-ram: point `HostName` of `Host vps` in `~/.ssh/config` (yours) and `ACW_VPS` in
   `/home/acw/audio-cleaner/secrets/vps.env` (edit it with `acw-sh`) at it, then `ssh vps true` (check the host key out of band first),
   and copy the new host key for the tunnels: `ssh-keygen -F vps | grep -v '^#' | acw-sh -c 'cat > ~/.ssh/known_hosts'` (replaces the old one).
3. On the VPS, add the tunnel key lines above to `~/.ssh/authorized_keys`.
4. frp: copy `frps` (from the frp release, https://github.com/fatedier/frp/releases, same version) to `/opt/frp/frps`
   and `frps.ini` (with its token) to `/etc/frp/frps.ini` on the new VPS (`sudo install -d /opt/frp /etc/frp` first).
5. On primary-ram: `deploy/vps/publish-edge.sh --setup`, then `ssh -t vps 'sudo bash ~/edge-src/setup.sh'`.
6. DNS (Namecheap, both domains): the `A` record of `@` to the new address (`www` of audiocleaner.live is a CNAME to
   `@`). Mail forwarding records stay as they are.
7. On primary-ram: `acw-ctl restart acw-tunnel acw-tunnel-bulk acw-tunnel-live`, `deploy/publish-web.sh`,
   copy the packs to `/var/www/pack`, then `curl -s https://audiocleaner.live/api/health` (`{"ok":true}`).
8. At home, point the Velocity proxy machine's `frpc.ini` (`server_addr`) at the new address; restart frpc.

## Email for the domains

Namecheap's email forwarding receives mail for audiocleaner.live (MX `eforward*.registrar-servers.com`, SPF
`include:spf.efwd.registrar-servers.com`) and forwards it to the owner's inbox: the site never shows that inbox.
Sending (sign-in links, alerts, the contact form) goes through the SMTP account in `acw.env` (`SMTP_*`); for mail
to come from the domain, that account must be a sending service with the domain verified (docs/operations.md, "Mail").
