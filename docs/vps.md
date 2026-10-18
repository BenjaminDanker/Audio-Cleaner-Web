# The VPS (edge) — what lives there and how to rebuild it

The VPS (`mc-multi-world.space`, Caddy 2.11) does two things
for Audio Cleaner:

1. Serves the **static website** from `/var/www/audio` (the app, and the free browser tools' code: the noise
   remover's df.js, the speech runtime, the decoder). Kept here on purpose: serving it from home would compete with
   live audio and Minecraft on the home connection (1 Gbit/s both ways; the VPS link itself carries about 73 Mbit/s
   inbound and 92 Mbit/s outbound, measured 2026-10-15). The free tools' **models** (about 22 MB for the noise remover,
   77 MB for the subtitles) are not here: visitors' browsers download them from Hugging Face (`web/src/media/models.js`).
2. **Forwards** `/audio/api/*` (and the live WebSocket) to the home server
   through SSH reverse tunnels that primary-ram opens:
   VPS `127.0.0.1:18790` → primary-ram `127.0.0.1:8790` (`acw-tunnel` unit), for the live WebSocket
   VPS `127.0.0.1:18795` → primary-ram `127.0.0.1:8795` (`acw-tunnel-live`, its own connection and process, falling
   back to 18790: docs/operations.md, "Live process"), and for job uploads and result downloads
   VPS `127.0.0.1:18794` → primary-ram `127.0.0.1:8794` (`acw-tunnel-bulk` unit, its own connection and process so
   big transfers do not delay live audio; Caddy falls back to 18790 when it is down: docs/operations.md, "Bulk transfers").
   Compression: Caddy compresses (zstd, gzip) only the small JSON answers of the API; the live WebSocket, uploads and
   downloads are already compressed audio or need no delay and are passed through untouched. The static files are
   pre-compressed by `deploy/publish-web.sh` (zstd and gzip).

Nothing else of ours is on the VPS: no data, no secrets. When the home server
or tunnel is down, Caddy answers API calls with a 503 maintenance JSON
(`handle_errors` in the snippet) and the site shows a maintenance notice. The free browser tools keep working in a
page that already loaded them; a new visitor cannot start them (they need a sign-in, and their code a session check,
both at home).

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

1. On primary-ram, point the `vps` SSH alias (`~/.ssh/config`) and `ACW_VPS` in
   `~/audio-cleaner/secrets/vps.env` (`<user>@<new address>`, see `deploy/vps.env.example`; the tunnel
   units read it, the address is never in the repo) at the new host.
2. On the new VPS, allow the tunnel key to open exactly two listening ports and
   nothing else. Append this line to `~/.ssh/authorized_keys` (the public key is
   `~/.ssh/acw_tunnel.pub` on primary-ram):

       restrict,port-forwarding,permitlisten="127.0.0.1:18790",permitlisten="127.0.0.1:18794",permitopen="127.0.0.1:1",command="/bin/false" ssh-ed25519 <contents of acw_tunnel.pub> acw-tunnel@primary-ram

3. From primary-ram, copy the route files and accept the new host key once:

       ssh vps 'mkdir -p ~/acw'
       scp ~/repos/Audio-Cleaner-Web/deploy/vps/audio.caddy ~/repos/Audio-Cleaner-Web/deploy/vps/install-audio-route.sh vps:~/acw/
       ssh -F none -i ~/.ssh/acw_tunnel -o IdentitiesOnly=yes -o HostKeyAlias=vps <user>@<new address> true   # answer "yes"; it exits (command=/bin/false)

4. On the new VPS: `sudo bash ~/acw/install-audio-route.sh`
5. On primary-ram:

       cd ~/repos/Audio-Cleaner-Web && bash deploy/install.sh     # re-installs the tunnel unit
       bash deploy/publish-web.sh                                 # uploads the site
       curl -s https://mc-multi-world.space/audio/api/health      # {"ok":true}

6. Update the Google OAuth redirect URI and the Stripe webhook endpoint only if
   the domain changed (both use `https://<domain>/audio/...`).

## SSH: keys only

The VPS must not accept passwords. Cloud images ship `/etc/ssh/sshd_config.d/50-cloud-init.conf`
with `PasswordAuthentication yes`, and sshd keeps the first value it reads, so a `no` further down
(`sshd_config`, `60-cloudimg-settings.conf`) does nothing. A file that sorts first wins:

    echo 'PasswordAuthentication no' | sudo tee /etc/ssh/sshd_config.d/00-keys-only.conf
    sudo sshd -t && sudo systemctl reload ssh
    sudo sshd -T | grep -Ei 'passwordauthentication|kbdinteractiveauthentication|permitrootlogin'   # all "no"

Check from a second terminal that key login still works before closing the first one.
