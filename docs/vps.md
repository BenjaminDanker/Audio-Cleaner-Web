# The VPS (edge) — what lives there and how to rebuild it

The VPS (`mc-multi-world.space`, 216.250.115.225, Caddy 2.11) does two things
for Audio Cleaner:

1. Serves the **static website** from `/var/www/audio` (~30 MB: the app, the
   free browser denoiser and its model). Kept here on purpose: new visitors
   download ~12 MB, and serving that from home would compete with live audio
   and Minecraft on the ~52 Mbit/s home uplink.
2. **Forwards** `/audio/api/*` (and the live WebSocket) to the home server
   through SSH reverse tunnels that primary-ram opens:
   VPS `127.0.0.1:18790` → primary-ram `127.0.0.1:8790` (`acw-tunnel` unit), and for job uploads and result downloads
   VPS `127.0.0.1:18794` → primary-ram `127.0.0.1:8794` (`acw-tunnel-bulk` unit, its own connection and process so
   big transfers do not delay live audio; Caddy falls back to 18790 when it is down: docs/operations.md, "Bulk transfers").

Nothing else of ours is on the VPS: no data, no secrets. When the home server
or tunnel is down, Caddy answers API calls with a 503 maintenance JSON
(`handle_errors` in the snippet) and the site shows a maintenance notice while
the free browser tools keep working.

Files in the repo: `deploy/vps/audio.caddy` (the site-block snippet),
`deploy/vps/install-audio-route.sh` (inserts/replaces it in
`/etc/caddy/Caddyfile`, validates, reloads, keeps a backup, reverts on error),
`deploy/publish-web.sh` (builds the site and rsyncs it to `/var/www/audio`).

## Apply a changed snippet (current VPS)

On primary-ram:

    scp ~/repos/Audio-Cleaner-Web/deploy/vps/audio.caddy ~/repos/Audio-Cleaner-Web/deploy/vps/install-audio-route.sh vps:~/acw/

On the VPS:

    sudo bash ~/acw/install-audio-route.sh

## Move to a new VPS

Assumes a Debian/Ubuntu VPS with your user `silver`, the domain's DNS A record
pointing at it, and Caddy installed (`sudo apt install caddy` or Caddy's apt
repo) with a site block for the domain that ends in the catch-all
`handle { respond "not found" 404 }` the installer looks for.

1. On primary-ram, point the `vps` SSH alias (`~/.ssh/config`) and
   `deploy/systemd/acw-tunnel.service` (`silver@<new IP>`) at the new host.
2. On the new VPS, allow the tunnel key to open exactly two listening ports and
   nothing else. Append this line to `~/.ssh/authorized_keys` (the public key is
   `~/.ssh/acw_tunnel.pub` on primary-ram):

       restrict,port-forwarding,permitlisten="127.0.0.1:18790",permitlisten="127.0.0.1:18794",permitopen="127.0.0.1:1",command="/bin/false" ssh-ed25519 <contents of acw_tunnel.pub> acw-tunnel@primary-ram

3. From primary-ram, copy the route files and accept the new host key once:

       ssh vps 'mkdir -p ~/acw'
       scp ~/repos/Audio-Cleaner-Web/deploy/vps/audio.caddy ~/repos/Audio-Cleaner-Web/deploy/vps/install-audio-route.sh vps:~/acw/
       ssh -i ~/.ssh/acw_tunnel -o IdentitiesOnly=yes silver@<new IP> true   # answer "yes"; it exits (command=/bin/false)

4. On the new VPS: `sudo bash ~/acw/install-audio-route.sh`
5. On primary-ram:

       cd ~/repos/Audio-Cleaner-Web && bash deploy/install.sh     # re-installs the tunnel unit
       bash deploy/publish-web.sh                                 # uploads the site
       curl -s https://mc-multi-world.space/audio/api/health      # {"ok":true}

6. Update the Google OAuth redirect URI and the Stripe webhook endpoint only if
   the domain changed (both use `https://<domain>/audio/...`).
