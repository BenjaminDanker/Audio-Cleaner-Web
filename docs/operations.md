# Operations runbook (primary-ram)

## Layout

The services run as the account `acw` (section "Accounts" below), so everything production lives in **acw's home**,
`/home/acw`. `~` in this runbook means acw's home whenever a command runs as acw (`acw-sh`, the wrappers, the units).

```
/home/acw/audio-cleaner/                                                                          (acw:acw, 750)
  app/        deployed code (copies of server/acw and common/acw_common, see app/VERSION)               (0700)
  app.prev/   the previous copy of the code: deploy/rollback.sh brings it back                          (0700)
  venv/       Python 3.12 virtualenv (server/requirements.lock, installed with --require-hashes); venv-gpu/ for the GPU worker
              (deploy/setup-gpu.sh, deploy/requirements-gpu.lock)
  bin/        ffmpeg, ffprobe (BtbN static n8.1.3, pinned in deploy/models.lock), libdf.so and acw-live (built by the owner)
  models/     DeepFilterNet3_onnx.tar.gz, dpdfnet/, sidon/, parakeet-v2/, parakeet-v3/, captions-en/,
              silero-vad/, whisper/{large-v3-turbo,small}   (every file, its source and sha256: deploy/models.lock)
  wasm/       df.js + df_bg.wasm for the browser build
  data/       acw.sqlite3, jobs/, stream.sock, backups/ (pre-deploy snapshots), lock.sha, metrics.prom,
              downloads/ (the mirrored release files, see "Release downloads")                          (0700)
  secrets/    acw.env (web, secrets), acw-worker.env, vps.env, backup.env/backup.pass, github-releases-read.token  (0700, files 0600)
/home/acw/.ssh/acw_tunnel(.pub), known_hosts   the tunnel key and the VPS host key (no login key of the owner's)
/home/acw/.config/systemd/user/                the units (acw's user manager, linger on)
/home/acw/deploy/<commit>/                     what deploy/install.sh staged for acw (the newest four are kept)

/home/silver/audio-cleaner/                    the owner's side of the same name
  bin models wasm venv venv-gpu   links to /home/acw/audio-cleaner/...  (tools, dev runs, tests and the web build read them)
  src/        build trees (DeepFilterNet, deepfilter-build, libav-build) - the Rust toolchain is the owner's - safe to delete
  dev/ research/ listen/          scratch data, test fixtures, experiments - safe to delete
  secrets/    the owner's own tokens and keys (update-signing-key.pem, github-actions-*.token): never moved to acw
```

## Accounts: who runs what (acw and silver)

Why: a hole in the web code must not expose the owner's SSH keys (the VPS login), other secrets, the Minecraft servers
or any of the owner's files. So the services run as `acw`, a normal account with no login shell (`/usr/sbin/nologin`),
home `/home/acw` (750), which cannot read `/home/silver`. `silver` (the owner) keeps full control: it is in group `acw`, and
`/etc/sudoers.d/acw` says `silver ALL=(acw) NOPASSWD: ALL` (**privilege only goes down**: acw has no sudo rights and the
owner's VPS login key is not on acw's side). The services stay systemd **user** units, now in acw's user manager (linger
on), so `systemctl --user`, `Path.home()` and the cgroup path `user@<uid>.service/acw.slice` behave as before, as acw.

| You want to | Command (as the owner) | What it is |
|---|---|---|
| deploy | `cd ~/repos/Audio-Cleaner-Web && deploy/install.sh --wait-live` | unchanged; stages the commit for acw, runs the server part as acw, then publishes the site and runs the smoke test as you |
| status, restart, enable | `acw-ctl status 'acw-*'`, `acw-ctl restart acw-web` | `systemctl --user ...` in acw's manager |
| logs | `acw-logs acw-web -f`, `acw-logs 'acw-worker@*' --since -1h`, `acw-logs` (all acw-* units) | `journalctl --user -u UNIT ...` as acw. (Members of group `adm` can also read acw's user journal file directly, through journald's ACL, but then units must be matched with `_UID=`; asking acw's own journalctl is exact.) |
| admin commands | `acw-admin activity`, `acw-admin grant a@b.c 5 "credit"`, `acw-admin -m acw.alerts test` | `python -m acw.admin` as acw with the units' environment (`acw-worker.env`, `backup.env`, then `acw.env`; `--worker-env` leaves the web secrets out) |
| edit a secret or setting | `acw-sh -c 'nano ~/audio-cleaner/secrets/acw.env'`, then `acw-ctl restart acw-web` | the env files are acw's (0600); you cannot open them as yourself |
| a shell for maintenance | `acw-sh`, `acw-sh -c 'du -sh ~/audio-cleaner/data/jobs'` | bash as acw in `/home/acw` (cwd `/`: acw cannot enter your folders, give files by absolute path) |
| the same without the wrappers | `sudo -u acw env XDG_RUNTIME_DIR=/run/user/$(id -u acw) systemctl --user status 'acw-*'` | what `deploy/host/acw-run` does |

The wrappers (`acw-run`, `acw-ctl`, `acw-logs`, `acw-admin`, `acw-sh`) live in `deploy/host/` and are installed in
`/usr/local/bin` by `deploy/host/move-to-acw.sh`; re-install them with `sudo install -m 755 deploy/host/acw-* /usr/local/bin/`
after changing them.

**What the owner can and cannot read.** acw reads nothing of `/home/silver` (home 750, no ACL). The owner reads
`bin models wasm venv venv-gpu` of acw (group `acw` plus an ACL entry for the owner, which also covers login sessions that
began before the group was added: a group change reaches a running session and the owner's own systemd user manager only
at the next login), but **not** `app`, `data`, `secrets` (acw only, 0700): use `acw-sh` or `acw-admin` for those.
`/mnt/mc-storage/audio-cleaner-backups` belongs to acw (750).

**Where the deploy steps run.** `deploy/install.sh` run by the owner: (1) git checks (clean tree, descendant of the
deployed `app/VERSION`) and the syntax gate; (2) `git archive HEAD` of `deploy server common native` into
`/home/acw/deploy/<commit>` (owned by acw; it also holds `DEPLOY_INFO`: commit, branch, subject); (3) asks the server part
what it needs (`--plan`, read-only) and **builds libDF and `acw-live` as the owner** (the Rust toolchain stays with the owner; the
build trees stay in `~/audio-cleaner/src`), handing the binaries over; (4) runs the **server part as acw**
(`install.sh --server-only`: env check `ACW_ENV=production`, pre-deploy database snapshot, packages, code copy, models, config,
`init-db`, units, restart, health) with the same flags; (5) only if that went well, as the owner again: `deploy/publish-web.sh`
(needs the VPS login key) and `deploy/smoke.py` (`ssh vps`). `deploy/rollback.sh` and `deploy/drain.sh` follow the same
split. Run by acw itself, `install.sh` does only the server part (no VPS access: it says so). Without the account
(`/etc/acw-service-user` absent) one account does everything, as before.

**One-time move and the way back.** `sudo bash deploy/host/move-to-acw.sh` (`--dry-run` first), `sudo bash
deploy/host/move-back-to-silver.sh`; see "Moving the services to acw" at the end of this file.

## Deploy / update

Always deploy from `~/repos/Audio-Cleaner-Web`, branch `on-prem` (fast-forward it from `desktop-merge`; the
script refuses another history, so an old checkout can't overwrite newer code):

```bash
cd ~/repos/Audio-Cleaner-Web && git checkout on-prem && git merge --ff-only desktop-merge   # or pull
deploy/install.sh                      # as the owner: check, stage for acw, back up, copy, restart what changed, verify, publish the site if web/ changed
python3 deploy/smoke.py               # the live site from outside: pages, headers, API, live, downloads, Stripe and Google
                                       # addresses, the old address, certificates, the models on Hugging Face, the tunnels
                                       # (install.sh runs it at the end; it changes nothing)
deploy/install.sh --wait-live          # stop new live sessions and wait (up to 10 min) for the running ones first
deploy/install.sh --wait-live=1800     # ... or for as long as you say
deploy/install.sh --no-restart         # copy everything, restart nothing
ACW_WORKERS=3 deploy/install.sh        # change the number of file-job workers
deploy/rollback.sh                     # put the previous code back (run again to undo)
deploy/drain.sh on|off|status|wait     # refuse new live sessions by hand (e.g. before a long maintenance)
deploy/publish-web.sh                  # build the site and rsync it to the VPS (install.sh does it when web/ changed)
```

(Who does which part is in "Accounts" above: the owner's run hands the server part to acw and keeps the VPS parts.)

What `install.sh` does, in order: prints the commit and what is running now (file jobs, live sessions);
**refuses** (unless `--force`) when the working tree has uncommitted changes, when the commit is not a
descendant of the deployed `app/VERSION`, or when live sessions are running and the web or live worker would be
restarted; checks that every Python file parses; takes a database snapshot (`data/backups`, verified like the
nightly one; a failing snapshot stops the deploy); installs the locked packages (`pip install --require-hashes`) only when a pin
changed; keeps the old code as `app.prev` and copies the new; runs `deploy/fetch-models.sh` (downloads and verifies what
`deploy/models.lock` lists, never touching a correct file); runs `init-db`; installs changed unit files; then
restarts **only what changed**: the web process, the bulk-transfer process (`acw-bulk`) and the file workers when the code or
packages changed, the live worker only when code it uses changed, a unit whose own unit file changed. The SSH tunnels are
never restarted for a code change. (Packages count as changed when a `name==version` pin differs from what is installed;
a change of the `--hash` lines alone changes nothing.)
Finally it polls (up to 30 s) until every unit is active and `/api/health?deep=1` is ok, and tells you to run
`deploy/rollback.sh` if not. A file worker that is stopped puts its job back in the queue with the upload kept
and no attempt used, so a deploy fails no jobs. (The first deploy of this behaviour still stops workers that run
the older code: check the "running now" line and wait for jobs to finish.)

Rollback keeps units, packages and the database as they are; schema changes are additive.

## Services (acw's user units, linger on)

```bash
acw-ctl status 'acw-*'
acw-logs acw-web -f
acw-ctl restart acw-web
```

Ports on primary-ram (all loopback) and how they reach the VPS:

| Unit | Listens on (home) | VPS side | Carries |
|---|---|---|---|
| `acw-web` | `127.0.0.1:8790` | `127.0.0.1:18790` via `acw-tunnel` (`-o IPQoS=lowdelay`) | everything except the bulk routes and live: sign-in, API, billing |
| `acw-live` | `127.0.0.1:8795` | `127.0.0.1:18795` via `acw-tunnel-live` (`-o IPQoS=lowdelay`) | the live WebSocket (`/api/stream/ws`) only |
| `acw-bulk` | `127.0.0.1:8794` | `127.0.0.1:18794` via `acw-tunnel-bulk` (`-o IPQoS=throughput`) | uploads (`PUT /api/jobs/{id}/input`), result downloads (`GET /api/jobs/{id}/outputs/{name}`) and the release downloads (`GET /api/downloads/...`) only |
| `acw-mirror.timer` | none (a oneshot every 10 minutes) | nothing: it only fetches from the release service | copies the newest release files to `data/downloads/` ("Release downloads") |

Check from the VPS with `curl -s 127.0.0.1:18790/api/health`, `curl -s 127.0.0.1:18794/api/health` and
`curl -s 127.0.0.1:18795/api/health`.

## Live process (the live WebSocket on its own process and tunnel)

Why: the live WebSocket relays every session's audio frames on one event loop (one core). In `acw-web` that loop also
ran every sign-in, page visit and job-list poll, so heavy API traffic made running sessions late (admission only
refuses new ones). `acw-live` (`acw/live_app.py`) is the only process that runs live sessions: the live WebSocket and
the sweep that closes and bills sessions whose handler is gone, on a loop of its own; its admission measures that loop.
`acw-web` has no live code running. The VPS sends `/api/stream/ws` to `127.0.0.1:18795` (`acw-tunnel-live`) only.

Set up once (done 2026-10-18): the `acw_tunnel` key on the VPS lists `permitlisten="127.0.0.1:18795"` (next to 18790 and
18794), and `acw-ctl enable --now acw-tunnel-live`. When `acw-live` or its tunnel is down, live is down: the
monitor mails (`health_live`, and the units like any other).

Check: `curl -s 127.0.0.1:8795/api/health` here and `curl -s 127.0.0.1:18795/api/health` on the VPS answer
`{"ok":true,"process":"live"}`.

## Bulk transfers (uploads and downloads on their own process and tunnel)

Why: a 4 GB upload or a result download used to share one Python process, one SSH connection and the home uplink with
the live audio WebSocket, so a big transfer could add latency to live sessions (head-of-line blocking in the tunnel's
TCP stream, one event loop, one uplink queue). Now those two routes can run through `acw-bulk` (`acw/bulk_app.py`: the
same route functions, authentication, body limits, upload caps and headers as the main app; it serves nothing else,
runs no housekeeping, cloud transcription or live code) and their own SSH connection. The main process keeps serving
the same two routes, so nothing breaks before the VPS is switched, and the VPS falls back to it when the bulk side is down.

Rates (environment of `acw-bulk`: add to `~/audio-cleaner/secrets/acw.env`, then `acw-ctl restart acw-bulk`):

| Variable | Default | Meaning |
|---|---|---|
| `ACW_TRANSFER_MBPS_TOTAL` | `5` | megabytes per second for result downloads, release downloads and uploads **together**, one budget per process (0 = no cap). Every transferred byte crosses the VPS twice (a download goes home -> VPS -> user, an upload user -> VPS -> home) and the link between home and the VPS is the bottleneck: measured 2026-10-15 home->VPS about 9 MB/s (73 Mbit/s), VPS->home about 11 MB/s (on 2026-10-18 the home line itself measured only about 90 Mbit/s down and 85 up to anywhere, while the VPS does over 500 Mbit/s: docs/vps.md), and live audio shares both directions. The earlier separate caps (6 down + 8 up) could add up to 14 MB/s each way when an upload and a download ran together; 5 MB/s in total leaves about 4 MB/s in and 6 MB/s out for live and the website. The older `ACW_DOWNLOAD_MBPS_TOTAL` and `ACW_UPLOAD_MBPS_TOTAL` are still read when this one is not set (the smaller one counts) |
| `ACW_UPLOAD_MBPS_USER` | `5` | megabytes per second for one account's uploads (inside the shared budget above) |

**Rollout, in this order** (each step is safe on its own; nothing is applied by the repository):

1. **Home side.** `deploy/install.sh --wait-live` (the first deploy also restarts `acw-tunnel`, because its unit gained
   `-o IPQoS=lowdelay`, and that cuts live sessions; later deploys never restart it). It starts `acw-bulk` and installs
   `acw-tunnel-bulk` **without enabling it**. Check: `curl -s 127.0.0.1:8794/api/health` answers
   `{"ok":true,"process":"bulk"}`.
2. **VPS key.** The tunnel key is restricted to the ports it may listen on. On the VPS, edit `~/.ssh/authorized_keys` and
   change the line of the `acw_tunnel` key (keep the key itself and its comment) so it starts with

       restrict,port-forwarding,permitlisten="127.0.0.1:18790",permitlisten="127.0.0.1:18794",permitopen="127.0.0.1:1",command="/bin/false" ssh-ed25519 AAAA... acw-tunnel@primary-ram

   (one more `permitlisten`; sshd accepts several). Nothing else changes, and the main tunnel keeps working during the edit.
3. **Bulk tunnel.** On primary-ram: `acw-ctl enable --now acw-tunnel-bulk`, then on the VPS
   `curl -s 127.0.0.1:18794/api/health`. From now on `deploy/install.sh`, `deploy/rollback.sh` and the monitor treat it
   like the other units (`deploy/install.sh` prints a reminder until it is enabled).
4. **Caddy.** On primary-ram `deploy/vps/publish-edge.sh` (copies the sites to the VPS and reloads Caddy; a configuration
   that does not load is put back; no sudo). The upload
   route (`.../input`, still 4 GB, no stall timeouts) and `.../outputs/...` now go to `127.0.0.1:18794` first and fall back to
   `127.0.0.1:18790` (`lb_policy first`, a health check every 10 s, 30 s of avoidance after a failed connection), so a
   stopped `acw-bulk` or `acw-tunnel-bulk` costs a few seconds, not an outage. This snippet no longer uses
   `request_body { read_timeout }`, which Caddy's Caddyfile syntax does not have (the old snippet failed `caddy validate`):
   the 20 s stall limit for small bodies is enforced by the app (`acw/bodylimit.py`).
5. **Check.** Upload and download a file on the site; `acw-logs acw-bulk -f` shows the `PUT .../input` and
   `GET .../outputs/...` lines (client address, status, milliseconds), the main `acw-web` log shows none for them.

Back out: put the previous version of `deploy/vps/sites/00-audio-cleaner-snippets.caddy` back (`git checkout` of the old
commit) and run `deploy/vps/publish-edge.sh`; on primary-ram `acw-ctl disable --now acw-tunnel-bulk`. The main path never changed.

The router: `IPQoS` marks the two SSH connections so a router that prioritises by DSCP or runs smart queueing (SQM,
cake/fq_codel) can favour the live one. It does nothing without such a router; the separate connection and process are
what helps in any case. Setting up SQM on the home router remains the best second step (docs/launch-checklist.md).

## Release downloads (OBS plugin)

Users and the apps never see the GitHub repository (it may become private). GitHub Releases stays where CI puts the build
output; primary-ram keeps a verified copy and serves it under our own domain, from the bulk process (`acw-bulk`, so a
download never queues in front of live audio, under the same shared transfer budget `ACW_TRANSFER_MBPS_TOTAL` as result downloads and uploads).
**Nothing is stored on the VPS**: Caddy only proxies `^/api/(jobs/<id>/outputs/|downloads/)` to the bulk upstream
(`deploy/vps/sites/00-audio-cleaner-snippets.caddy`, falling back to the main process like the other bulk routes).

**Addresses** (base `https://audiocleaner.live/api/downloads`; GET and HEAD, Range works, no sign-in, no redirects):

| Address | What |
|---|---|
| `/obs-plugin/manifest.json`, `/obs-plugin/manifest.json.sig` | the plugin's signed manifest (the release assets `obs-plugin-manifest.json` and `.sig`) |
| `/obs-plugin/<file>`, `/obs-plugin/latest/windows`, `/mac`, `/ubuntu-debian`, `/linux-other` | plugin packages; the website's buttons use `latest/...` |

**How it works.** `acw-mirror.timer` starts `acw-mirror.service` (`python -m acw.admin mirror --once`, `acw/mirror.py`) a
minute after it is enabled and then 10 minutes after each run. A run reads the GitHub REST API (`/repos/<repo>/releases`),
and takes the newest `obs-plugin-v<version>` release (manifest, `.sig`, the packages). (Only the plugin: the desktop
app gets its updates and models from its own GitHub releases, docs/desktop.md.) Before anything is
published: the Ed25519 signature of the manifest is checked with the public key built into the plugin (the `cryptography`
package if installed, else `openssl`, 3.0 or newer), then every file's size and sha256 (and sha512 when the manifest gives
it), the size cap (50 MB), the file names (strict pattern), and that the version is not
older than the one served. Downloads resume from a `.part` file; the disk must keep 5 GB free (`ACW_MIRROR_MIN_FREE_GB`).
Then the files are moved to `data/downloads/<channel>/<version>/` and `state.json` (the list of names the routes may serve)
is replaced atomically, last. A failure at any step publishes nothing: the previous version keeps being served, the error is
in `mirror-status.json` and the journal, and the monitor mails `mirror_stale` after an hour (a newer release not served) or
2 hours (no successful run). The previous version's files stay 7 days (downloads in progress), then they are deleted.
The routes serve only names in `state.json`, open files without following links, re-check the size, and rate-limit the
start of a file download per network (40 bursts, then one every 45 s; resuming and HEAD cost a tenth).

**Commands.**

```bash
acw-admin mirror --status          # what is served and when it was last mirrored
acw-admin mirror --once            # run a pass now (what the timer does); --channel desktop|obs-plugin|models
acw-admin mirror --once --force    # fetch and publish again even if the served version looks current or newer (a downgrade, a damaged copy)
acw-admin mirror --verify          # hash every served file against its recorded checksum (no network)
acw-ctl start acw-mirror.service   # the same as --once, with the unit's settings; acw-logs acw-mirror
```

A release must not be older than the served one: to roll the plugin back on purpose, publish a newer
version with the old contents, or delete `data/downloads/<channel>/state.json` and run `mirror --once` (the channel is
empty for the minute it takes).

**If the repository becomes private.** The mirror needs a token: GitHub > Settings > Developer settings > Fine-grained
personal access tokens > repository access "Only select repositories" = this one repository, permission **Contents: Read-only**
(nothing else), and put the token alone in `~/audio-cleaner/secrets/github-releases-read.token` (`chmod 600`; another place:
`ACW_GITHUB_TOKEN_FILE` in `acw-worker.env`). It is sent only to `api.github.com`, never to the storage host the download
redirects to, and never written to a log. Without the file the mirror reads anonymously (works while the repository is
public; the limit is 60 requests an hour, a pass uses 3 to 5, so once every 10 minutes is close to it: add the token anyway).
The file names the plugin and the website use are on our domain only; the repository name appears in one
place (`DEFAULT_REPO` in `acw/mirror.py`, or `ACW_GITHUB_REPO`).

**One-time:** deploy (`deploy/install.sh` enables the timer), then update the VPS Caddy snippet as in "One-time setup"
(the `downloads/` part of the bulk matcher; until then the downloads still work, through the main tunnel).
Check: `curl -sI https://audiocleaner.live/api/downloads/obs-plugin/manifest.json` after the first pass finished
(`mirror --status`).

## Owner dashboard

Money, users, usage, errors and the machine for any time range, in a browser: `acw-admin.service`
(`acw/admin_app.py`, page in `acw/admin_ui/`). It listens on **127.0.0.1:8796 only** and nothing forwards to it (the
tunnels carry 8790 and 8794; the VPS knows nothing of it), it refuses any request whose Host isn't this machine's
loopback address (no DNS rebinding), and it only reads (the database is opened read-only; there is nothing to change).
Open it over SSH from your PC:

```bash
ssh -L 8796:127.0.0.1:8796 silver@primary-ram      # then open http://localhost:8796  (the dashboard runs as acw; you log in as yourself)
```

The machine graphs come from `data/samples.sqlite3` (`acw/samples.py`): the monitor records one row every 2 minutes
(CPU, memory, GPU, disk, queue, live), kept a little over a year. Card fees are an estimate (2.9 % + 30 c); Stripe's
dashboard has the exact ones.

## Admin CLI

```bash
acw-admin login-link you@example.com   # one-time sign-in URL
acw-admin grant someone@example.com 5 "support credit"
acw-admin user someone@example.com
acw-admin suspend someone@example.com "reason"
acw-admin stats            # report for the last 30 days (see "Statistics and metrics")
acw-admin stats --days 7
acw-admin activity       # file jobs and live sessions running now
acw-admin check-money    # balances and holds against the ledger (exit 1 on a mismatch)
acw-admin verify-backup FILE.sqlite3.gz
acw-admin retention      # apply the retention rules now and print what each removed (the web process does it hourly)
acw-admin storage        # sizes of the database, job files, data directory and backups
acw-admin mirror --status  # release downloads: what is served, when it was last mirrored (--once runs a pass now, --force refetches, --verify hashes every served file)
acw-admin compact-money  # payment/ledger records older than 7 years: what it would remove (add --apply to do it)
```

## Statistics and metrics

**`acw.admin stats [--days N]`** (default 30; read-only, a few SELECTs) prints a plain-text report from the database:
file jobs by tier and final status with audio hours, p50/p95 queue wait and run time, the real-time factor (run seconds per
audio second), in-progress jobs, the failure ratio and the five most common error messages; live sessions (count,
minutes, end reasons, by tier, frames that were late or never came back); free-tier minutes used; usage charged (jobs and
live) and top-ups paid. Percentiles are nearest-rank. "Queue wait" is the time until a worker first took the job (recorded
in `jobs.wait_seconds`; older jobs fall back to `started_at - queued_at`), "run" is start to finish.

**One JSON log line per finished job and live session** in the journal of the process that ended it (`acw-worker@N` for
jobs, `acw-live` for live sessions); no user ids, addresses or file names:

```bash
acw-logs --since -1d -o cat | grep '"event":"job_'           # job_done and job_failed
acw-logs --since -1d -o cat | grep '"event":"live_session_end"'
```

| Event | Fields |
|---|---|
| `job_done` | `job` `kind` `tier` `free` `status` `audio_s` `queue_wait_s` `run_s` `cost_micros` `noise` (`light`, `heavy`, `rebuilt`, `none`, `auto`) `attempts` `input_bytes` |
| `job_failed` | the same, `status` = `failed` or `canceled`, plus `error` (first 200 characters) and `stage` |
| `live_session_end` | `session` `denoise` `captions` `audio_s` `wall_s` `cost_micros` `end_reason` `frames` `late_frames` (answered more than 100 ms after it was sent) `dropped_frames` (never came back, including those in flight when the session ended) `worst_round_trip_ms` |

**`data/metrics.prom`**: the monitor (`acw-monitor`, every 2 minutes) rewrites this Prometheus text file on each run
(written next to it and renamed, so a reader never sees half a file). There is no HTTP endpoint; point a collector at the
file (for example node_exporter's `--collector.textfile.directory=~/audio-cleaner/data` with a file name ending in
`.prom`). Gauges: `acw_queue_length`, `acw_jobs_in_state{status}`, `acw_live_sessions`, `acw_live_units`,
`acw_live_capacity_units`, `acw_jobs_finished_24h{status}`, `acw_queue_wait_p95_seconds_24h`, `acw_unit_active{unit}`,
`acw_disk_free_bytes`, `acw_backup_age_seconds` (left out when there is no snapshot), `acw_check_problem{kind}` (1 while the
monitor's check of that kind fails), `acw_metrics_written_timestamp_seconds`. Counters (derived from the database):
`acw_jobs_finished_total{status}`, `acw_live_sessions_ended_total{end_reason}`, `acw_live_audio_seconds_total`,
`acw_live_frames_total`, `acw_live_late_frames_total`, `acw_live_dropped_frames_total`, `acw_usage_micros_total`.

## Alerts

`acw-monitor.timer` runs `python -m acw.monitor` every 2 minutes (a oneshot in
`acw.slice`, Nice 10, reads the DB and system state only). Problems are emailed
to `ACW_ALERT_EMAIL` with subject prefix `[Audio Cleaner]`; mails carry counts,
job ids (first 8 characters) and status codes only, never user emails, file
names or transcripts.

**Set up (once).** Set `ACW_ALERT_EMAIL` and the `SMTP_*` variables in `~/audio-cleaner/secrets/acw.env` (the monitor
and the failure hook read that file; workers send no mail; the mail account is described under "Mail"):

```
ACW_ALERT_EMAIL=silverjunk08@gmail.com
SMTP_HOST=smtp.resend.com
SMTP_PORT=587
SMTP_USER=resend
SMTP_PASSWORD=<API key>
SMTP_FROM=Audio Cleaner <no-reply@audiocleaner.live>
```

Then `acw-ctl restart acw-web` and check with:

```bash
acw-admin -m acw.alerts test      # sends a test mail
acw-admin -m acw.alerts status    # every alert kind, active or not, mails sent / suppressed
```

Until SMTP and `ACW_ALERT_EMAIL` are both set nothing is mailed: every alert is
logged at ERROR (`acw-logs 'acw-*' | grep -i 'ALERT NOT SENT'`)
and goes out by itself, still pending, once mail works. The same SMTP account is used
for magic-link sign-in mails and the contact form ("Mail").

**De-duplication.** Each alert has a kind. The first occurrence is mailed; while it
stays active it is mailed again after 1 hour and then every 6 hours; when the
condition clears (2 clean checks in a row) one "Resolved" mail is sent and the kind
is re-armed, so the next occurrence mails immediately. State lives in the
`alert_state` table.

| Alert (kind) | Fires when | Notes |
|---|---|---|
| Service not running (`service:<unit>`) | an enabled `acw-*` unit (or `acw-backup.timer`, `acw-mirror.timer`) is not `active` for 2 checks in a row; also sent at once by the `OnFailure=` hook (`acw-alert@.service`) when a unit enters `failed` | the hook only fires when systemd gives up; with `Restart=always` that is rare, so the next row covers crash loops |
| Restarting repeatedly (`restarts:<unit>`) | systemd auto-restarted the unit 2+ times in 30 minutes (`NRestarts`) | a manual `systemctl restart` does not count |
| Back up after unclean stop (`web_unclean_stop`) | acw-web starts and the previous run did not record a clean shutdown (crash, kill, OOM, power loss) | at most 1 mail per hour |
| API down locally (`health_local`) | `http://127.0.0.1:8790/api/health` fails twice in a row | acw-web is down or stuck |
| Upload/download process down (`health_bulk`) | only where `acw-bulk` is enabled: `http://127.0.0.1:8794/api/health` fails twice in a row | uploads and downloads fall back to `acw-web` once the VPS notices (docs/operations.md, Bulk transfers); `acw-tunnel-bulk` is watched as a service like the other units |
| Release downloads out of date (`mirror_stale`) | only where `acw-mirror.timer` is enabled: a channel has had no successful mirror run for 2 hours, or a newer release has been on the release service for over an hour without being served (for example its signature or a checksum is wrong), twice in a row | people keep getting the last good version, never a wrong one; the mail has the last error. See "Release downloads" |
| Not reachable from the internet (`health_public`) | local health is fine but `https://audiocleaner.live/api/health` fails from this machine twice in a row | the problem is the tunnel, VPS or Caddy; skipped when the public origin is not https |
| File queue long (`queue_long`) | the head job has waited over 20 min, or the last job's estimated wait (`jobs.queue_wait_seconds`) is over 45 min | the mail shows busy cores for Audio Cleaner vs everything else (with the busiest other process names, e.g. `java`) and memory, so you can see when the Minecraft servers are the cause |
| Jobs abnormally slow (`jobs_slow`) | a running job, 10+ minutes in, is projected to take over 3x its `jobs.job_seconds` estimate | same host load breakdown |
| Disk low (`disk_low`) | under 15 GB free on the data volume | |
| Memory low (`mem_low`) | under 2 GB `MemAvailable` for 2 checks in a row | host load breakdown included |
| Swap growing (`swap_growing`) | at least 512 MB of swap in use and up 256 MB or more over 30 minutes | |
| Live sessions refused (`live_full`) | more than 10 stream sessions ended with `end_reason = 'full'` in the last hour (counted from `stream_sessions`; `ACW_ALERT_FULL_THRESHOLD` changes the 10) | |
| Backup overdue (`backup_stale`) | the newest snapshot in `/mnt/mc-storage/audio-cleaner-backups` (`ACW_BACKUP_DIR`) is older than 36 h, or none is found (disk not mounted) | |
| Backup failed its check (`backup_bad`) | `backup-status.json` next to the snapshots says the file is empty, failed SQLite's integrity check, or its ledger sums do not match, or the off-machine copy failed | the backup unit fails too |
| Balances do not match the ledger (`ledger_invariant`) | an account's balance differs from the sum of its ledger entries, or its held amount from its active holds | look before restarting anything; `acw.admin check-money` |
| Data directory large (`data_large`) | `~/audio-cleaner/data` without `jobs/` is over 5 GB (`ACW_DATA_ALERT_GB`) | something that should be bounded is growing: `acw.admin storage`, then "What grows and what bounds it" |
| Backup folder large (`backups_large`) | `/mnt/mc-storage/audio-cleaner-backups` is over 20 GB (`ACW_BACKUPS_ALERT_GB`) | 30 snapshots are far smaller than that: look for stray files |
| Old payment records (`old_money_records`) | ledger, payment or money-audit rows are more than 7 years and 30 days old | nothing is deleted by itself; run `acw.admin compact-money` (see "Payment records after 7 years") |
| Extra-accurate transcription stuck (`cloud_asr_stuck`) | a job has waited on the OpenAI transcription for over 30 minutes | |
| Many uploads (`uploads_piling`) | more than 15 uploads in progress, or more than 40 GB promised | |
| Credit held for hours (`holds_old`) | a hold has been active for over 6 hours | housekeeping releases leaked ones; if it persists, look at acw-web |
| Many jobs failing (`jobs_failing`) | over 30 % of the last 24 h's finished jobs failed (at least 10) | |
| Stripe events not processed (`stripe_events_unprocessed`) | an event older than 10 minutes was stored without being handled | a payment may be uncredited |
| Payments pending (`payments_pending`) | an auto top-up pending over 1 h, or a checkout over 26 h | |
| Files missing (`files_missing`) | a model or program under `~/audio-cleaner/{models,bin}` is not there | |
| Live worker silent (`stream_socket`) | connecting to the live worker's socket fails twice in a row | |
| GPU worker silent (`gpu_worker_silent`) | its unit is active but it has not marked itself idle for 45 minutes while nothing runs | CPU workers then take its jobs |
| OpenAI quota / billing (`openai:quota`) | OpenAI answers `insufficient_quota` or a billing error | mail shows only HTTP status and OpenAI's error type/code |
| OpenAI key rejected (`openai:auth`) | HTTP 401/403 | |
| OpenAI server errors (`openai:5xx`) | 3 or more 5xx in 5 minutes (either premium path) | |

`openai:*` alerts resolve themselves after 30 minutes without a new error.
Thresholds are constants at the top of `server/acw/monitor.py`.

Limits: the monitor runs on this machine, so a power cut, network loss or a dead
`systemd --user` manager sends nothing. For that, point an outside monitor (for
example UptimeRobot, free) at `https://audiocleaner.live/api/health`
(that URL answers only `{"ok":true}`; the detailed check `?deep=1` works from this machine and for admins).

## Mail

The site's mail (sign-in links, alerts, the contact form) should come from the domain, not from a personal mailbox.
Receiving and sending are separate things here.

**Receiving: Namecheap email forwarding.** In Namecheap, Domain List, `audiocleaner.live`, Mail Settings: "Email
Forwarding" (the records are MX `eforward*.registrar-servers.com` and an SPF TXT
`v=spf1 include:spf.efwd.registrar-servers.com ~all` on the bare domain). Add an alias such as `support@audiocleaner.live`
that forwards to the owner's inbox. Nobody sees the inbox address, and there is no mailbox to run.

**Sending: a transactional service with the domain verified.** Recommended: Resend (https://resend.com, free: 3,000 mails a
month, 100 a day). Add the domain `audiocleaner.live` there and put the DNS records it shows into Namecheap (Advanced DNS).
Resend uses a subdomain for the bounce path, so nothing clashes with the forwarding MX on the bare domain:

| Type | Host | Value |
|---|---|---|
| MX | `send` | Resend's feedback server (priority 10) |
| TXT | `send` | `v=spf1 include:amazonses.com ~all` (as Resend shows it) |
| TXT | `resend._domainkey` | the DKIM public key Resend shows |

Copy the exact values from Resend's page; wait until it says "Verified". Then create an API key (sending access) and set in
`~/audio-cleaner/secrets/acw.env`:

```
SMTP_HOST=smtp.resend.com
SMTP_PORT=587
SMTP_USER=resend
SMTP_PASSWORD=<the API key>
SMTP_FROM=Audio Cleaner <no-reply@audiocleaner.live>
```

Port 587 with STARTTLS is what both mail paths use (`acw/alerts.py` for alerts and the contact form, and the sign-in mail
in `api/auth_routes.py`; the server certificate is verified). Port 465 (implicit TLS) works only for alerts and the
contact form, not for sign-in links, so keep 587. Then `acw-ctl restart acw-web` and run
`python -m acw.alerts test` (see "Alerts"). Replies to contact-form messages go to the sender (`Reply-To`); the address
`no-reply@` need not exist, because it only sends.

**DMARC** (recommended, once sending works): a TXT record `_dmarc` with
`v=DMARC1; p=none; rua=mailto:support@audiocleaner.live` (use the forwarded address). `p=none` only reports; reports arrive
as mail to that address. Tighten to `p=quarantine` after a few weeks of clean reports.

**If 100 a day is too few** (one sign-in link is one mail): Brevo (https://www.brevo.com, free: 300 a day; SMTP host
`smtp-relay.brevo.com`, port 587, its own SMTP login and key, the domain verified the same way with its own DKIM and
SPF/DMARC records).

**Not recommended: Gmail SMTP** (`smtp.gmail.com`, port 587, an app password from
https://myaccount.google.com/apppasswords, `SMTP_FROM` = the Gmail address). It works, but every mail shows the personal
address as the sender.

## Contact form

The site shows no email address. `/contact` (a form) sends its messages to the owner by mail, through the same SMTP
account as the sign-in links and the alerts (`acw.alerts.send_contact_mail`). The recipient is `ACW_CONTACT_EMAIL` if set,
otherwise the alert address `ACW_ALERT_EMAIL`; either can list several addresses separated by commas. Nothing to set
when the alert address is the one you want. The address never reaches a browser: `/api/me` only says `contactForm: true`.

- Needs `SMTP_*` and one of the two addresses. Without them `POST /api/contact` answers 404 and the site hides the Contact link
  (and the legal pages stop pointing at it).
- A message arrives as `[Audio Cleaner] <Topic>: <first words>` with `Reply-To` set to the sender, so Reply answers them. The
  body says who wrote (name and address as typed), whether they were signed in (account email, id and status), and, when a job
  number was given, what the job is (kind, status, error text). Plain text only; control characters and anything that could
  start another mail header are stripped.
- Limits: 3 per hour and 10 per day per network (IPv6 per /64), 3 per hour and 5 per day per mailbox (no `+tag`, Gmail dots
  ignored), 100 per rolling day for the whole site (counted from the audit log, so a restart does not reset it; the form then says
  to try again tomorrow, with a Retry-After). A hidden field that people never see makes bots look successful but sends nothing.
  Same-origin requests only, and the usual CSRF token when signed in.
- The text of a message is **not stored** anywhere (not in the database, not in the journal). The audit log keeps that a
  message was sent (time, topic, account if signed in, network address for 90 days).
- Check it: sign out, open `/contact`, send a test, and look in the mailbox (spam folder first time: it comes from
  the sending service's account for the domain, see "Mail").

## Backups

Every night (`acw-backup.timer`, 04:17) `python -m acw.admin backup` writes a consistent snapshot of the database to
`/mnt/mc-storage/audio-cleaner-backups` (another disk; 30 kept), then **checks it**: the file is unzipped to a
scratch copy, `PRAGMA integrity_check` must say `ok`, and every account's balance must equal the sum of its ledger
entries and its held amount the sum of its active holds. A snapshot that fails is renamed `*.bad`, the unit fails (you
get a mail) and `backup-status.json` records why (the monitor reads it). `deploy/install.sh` takes the same kind of
snapshot into `data/backups` before every deploy.

**Off the machine (switched off until you configure it).** Create `~/audio-cleaner/secrets/backup.env` (0600):

```
ACW_BACKUP_OFFBOX=user@host:/path/to/backups/       # an rsync destination (ssh key login, no password: a key of acw's, made with `acw-sh`, not yours)
ACW_BACKUP_PASSPHRASE_FILE=/home/acw/audio-cleaner/secrets/backup.pass   # a file holding a long passphrase (gpg --symmetric, AES256)
# or, with `age` installed:  ACW_BACKUP_AGE_RECIPIENT=age1...   (a public key; keep the private key elsewhere)
```

Each nightly snapshot is then encrypted and copied; the unit fails if that does not work. The remote end keeps
everything it is given (prune it there). Also worth copying once, by hand, because the database does not hold them:
`secrets/` (the passphrase file belongs somewhere other than the backup target) and `models/` (9 GB, partly converted by hand).
After setting it up, run `acw-ctl start acw-backup.service` and look at `acw-logs acw-backup -n 20`.

**Restore** (the database only; everything else is code and models):

```bash
# As the owner. The folder paths below are acw's: run the file steps in `acw-sh`.
# 1. pick a snapshot and check it first (an encrypted off-machine copy: gpg --decrypt -o x.sqlite3.gz x.sqlite3.gz.gpg)
ls -1 /mnt/mc-storage/audio-cleaner-backups | tail
acw-admin --worker-env verify-backup /mnt/mc-storage/audio-cleaner-backups/acw-YYYYMMDD-HHMMSS.sqlite3.gz
# 2. stop everything that writes (the tunnel can stay: the site shows its maintenance message)
acw-ctl stop acw-web 'acw-worker@*' acw-worker-gpu acw-stream acw-monitor.timer acw-backup.timer
# 3. keep what is there, put the snapshot in place (in `acw-sh`: ~/audio-cleaner/data is acw's)
acw-sh
cd ~/audio-cleaner/data && mkdir -p ../data-before-restore && cp -a acw.sqlite3* ../data-before-restore/ && rm -f acw.sqlite3 acw.sqlite3-wal acw.sqlite3-shm
gunzip -c /mnt/mc-storage/audio-cleaner-backups/acw-YYYYMMDD-HHMMSS.sqlite3.gz > acw.sqlite3 && chmod 600 acw.sqlite3
# 4. schema (additive), then the money check, then start
cd ~/audio-cleaner/app && acw-admin init-db && acw-admin check-money
acw-ctl start acw-stream acw-web 'acw-worker@1' 'acw-worker@2' acw-monitor.timer acw-backup.timer   # and the other workers
```

Jobs that were in flight are re-queued or failed by the normal sweeps; stripe events that were missed are picked up
by the reconcile loop, and Stripe re-sends events it did not get an answer for. Drill it once on a copy (restore into a
scratch `ACW_DATA_DIR`, run `check-money`) before you need it.

## Log retention

Client addresses are in the user journal (the app's access log line has them; query strings are left out). The
Privacy Policy says 90 days. The journal rotates by size today; to also cap its age (one-time, needs sudo, applies to the
whole machine's journal):

```bash
sudo mkdir -p /etc/systemd/journald.conf.d && printf '[Journal]\nMaxRetentionSec=90day\n' | sudo tee /etc/systemd/journald.conf.d/acw-retention.conf && sudo systemctl restart systemd-journald
```

`deploy/install.sh` reminds you until that file or a `MaxRetentionSec` exists. The database side runs by itself, see the next
section.

## What grows and what bounds it

Everything that can grow over the years has a bound, and the Privacy Policy (section 8, "How long we keep things") says the
same numbers. The database rules live in `server/acw/retention.py` and run from the web process's housekeeping loop (at most
hourly, small batches, idempotent); `acw.admin retention` runs them by hand and prints what each removed.

**Database (SQLite, `data/acw.sqlite3`).** Sizes are what a row takes with its indexes (measured on the dev database), at the
example load of 1000 file jobs and 50 hours of live audio a day (about 100 sessions).

| Table | Rule | Bound at the example load |
|---|---|---|
| `jobs` | finished, failed, canceled and expired jobs deleted 12 months after they were created; file names blanked at 12 months (always before that, nothing is older); jobs still in progress are never touched | about 0.9 KB each, 365 000 rows: ~330 MB |
| `stream_sessions` | ended sessions deleted after 12 months; network address blanked after 90 days | ~0.3 KB each, ~36 000 rows: ~10 MB |
| `holds` | closed (captured or released) holds deleted after 12 months; active holds never | ~0.15 KB each: ~55 MB |
| `ledger` | **kept 7 years**, never deleted by the sweep. The file name in a usage line's text is removed after 12 months (amount, kind and date stay). After 7 years: `acw.admin compact-money` (below) | ~0.18 KB per row, ~65 MB a year, ~460 MB at year 7 |
| `payments` | kept 7 years (tax); settled rows older than that go with `compact-money` | tiny (one per top-up) |
| `audit_log` | routine rows (`login`, `user_created`, `api_key_created`, `api_key_revoked`, `contact_sent`) deleted after 12 months; network addresses blanked after 90 days. Rows that are evidence for money or consent (`tos_accepted`, `checkout_started`, `auto_reload_*`, `refund*`, `grant`, `suspend*`, `payment_*`, `dispute_*`, `account_deleted`) are kept 7 years like the ledger. An action nobody classified is kept, and a test fails until it is classified | ~0.13 KB per row |
| `sessions` | deleted when they expire (`auth.purge_expired`), and at the latest 30 days after expiry; address and browser type blanked after 90 days | ~0.4 KB, 30 days of logins |
| `login_tokens`, `oauth_states` | deleted a day after they expire (used or not), at the latest after 30 days | negligible |
| `stripe_events` | processed events deleted after 90 days (Stripe retries for about 3 days, so a repeat is always recognized); an unprocessed one after 12 months (the monitor alerts after 10 minutes) | ~0.1 KB each |
| `free_usage` | one row per account and day from the old free server subtitles (no new rows), deleted after 60 days | negligible |
| `deleted_emails` | deleted the next day | none |
| `api_keys` | revoked keys deleted after 12 months; active keys are limited to 10 per account | negligible |
| `identities`, `users` | one row per account, kept while the account exists. A deleted account stays as an anonymous row because ledger and payments point at it; `compact-money` removes it once nothing does. Growth is the number of sign-ups, which the sign-in rate limits bound | ~0.5 KB per account |
| `alert_state`, `meta` | one row per alert kind / setting: a fixed set | none |

Nothing else points at a `jobs`, `stream_sessions` or `holds` row by a foreign key, so deleting them cannot break anything;
the credit history shows an old usage line from the ledger's own text when the job row is gone. The ledger's invariant
(`balance = sum of entries`, `held = sum of active holds`) is untouched by every rule above; `tests/test_retention.py`
checks each rule, that money rows survive them, and that `acw.admin check-money` still passes.

Steady state at the example load: about 0.5 GB after the first year, then ~65 MB a year (the ledger), so under 1 GB at year 7.

**The file itself.** SQLite keeps the space of deleted rows for itself, so the sweep also runs a `VACUUM` about once every 28 days
(`retention.maintain_storage`): only between 02:00 and 03:59 machine time (the snapshot starts at 04:17), only when at least
~8 MB or a quarter of the file is free inside it, never while a live session is running (it tries again the next hour, then the
next night), and only when the disk has three times the file's size free. Writers wait for it (the busy timeout is 30 s; a
VACUUM of a database this size takes a second or two). The write-ahead log is truncated afterwards. A few MB of free pages
between vacuums is normal.

**Payment records after 7 years (a yearly job for the owner, from 2033).** Money records are never deleted by the program:
that is the safe choice, because deleting a ledger row would break `balance = sum of entries` (which the nightly backup check,
the monitor and `check-money` all test). When records get older than 7 years the monitor mails `old_money_records` (after a month
of grace). Then, once a year:

```bash
acw-admin compact-money          # shows what it would do
acw-admin compact-money --apply  # ... does it, then runs the money check
```

For each account the leading run of ledger rows older than 7 years becomes one "balance carried forward" row (`kind=adjustment`,
`ref_type=carry_forward`, same id and closing balance as the last row it replaces), so every balance and every later
`balance_after` still adds up; a run that adds up to nothing is simply removed. A user whose rows do not add up is rolled back
and reported. It also deletes settled payments and audit rows older than 7 years, and deleted accounts nothing points at any more.
Take a backup first (`acw-ctl start acw-backup.service`).

**Files and disks.**

| Where | What grows | Bound |
|---|---|---|
| `data/jobs/` | uploads, the copies kept for changing a job again, results | all of a job's files go together, 2 hours (`ACW_JOB_RETENTION_MINUTES`) after it finishes (`jobs.sweep`, from the web process every 30 s and from each worker every 60 s, so the deletion is accurate to about a minute), or at once with "Delete now" (which also takes the jobs made from it); the files of a failed or canceled job go at once and leftover folders after an hour; a deleted account's files at once. Sizes: "What a job keeps" below. Not part of the `data_large` check (a day of work legitimately holds many GB); the `disk_low` alert (under 15 GB free) and the upload reserve (which counts the kept copies, `jobs.kept_bytes`) watch it |
| `data/acw.sqlite3` and `-wal` | the database, see above | the table above; the log file stays ~4 MB (SQLite checkpoints it by itself) |
| `data/backups/` | a snapshot before every deploy | the newest 10 (`install.sh`), plus at most 3 failed ones (`*.bad`) |
| `data/metrics.prom`, markers (`live-*`, `lock.sha`, `web-published`, `gpu-worker-idle`, `gpu-worker-alive`) | rewritten in place, fixed names | a few KB in all |
| `/mnt/mc-storage/audio-cleaner-backups` | nightly snapshot, gzipped | the newest 30, plus at most 3 failed (`*.bad`) and `backup-status.json`; also pruned on a night when the backup fails. About a quarter of the database size each: ~250 MB x 30 = 7.5 GB at the worst example above. Not ours: a stray `Audio-Cleaner-desktop-repo.bundle` (16 MB) |
| The off-machine copy (if set up) | one encrypted file a night | "the remote end keeps everything": prune it there (Backups) |
| Journal (acw's user journal: `acw-logs`) | see below | size and age caps |
| `/var/log/syslog` (rsyslog gets a copy of everything journald sees, `ForwardToSyslog=yes`) | the same lines, with client addresses | logrotate: weekly, 4 kept, so about five weeks (inside the 90 days); ~16 MB a week at the example load |
| GitHub releases of the desktop app | installers | the newest 5 automatic ones (docs/desktop.md) |
| `data/downloads/` | the mirrored release files | the current version of each channel plus the previous one for 7 days: ~0.1 GB of OBS plugin packages (the only channel: the desktop app and the models come from GitHub releases and Hugging Face); `mirror` refuses to download unless 5 GB stay free, `acw_downloads_bytes{channel}` in `metrics.prom` shows it |
| `app.prev` | one previous copy of the code (1 MB) | replaced at every deploy |
| `/var/www/audio` on the VPS | hashed files under `assets/` (1.2 MB per release) and the engine under `dfn/` (24 MB, replaced in place) | `publish-web.sh` keeps this release and the one before it, deletes older hashed files; so ~2.5 MB of assets plus the engine |
| `~/.cache/pip` (3.0 GB today), `~/audio-cleaner/dev` (2.4 GB), `src` (5.1 GB), `research` (18 GB), `venv-gpu` (3.9 GB), `models` (9 GB) | pip's download cache; test fixtures and scratch; build trees; experiments; the pinned models | **not managed by any code** and not checked by the monitor. `pip cache purge`, `dev/` and `src/` ("safe to delete", Layout) and `research/` are yours to clean; `models/` is fixed by `deploy/models.lock` |

**What a job keeps** (`data/jobs/<id>/`, until its 2 hours are over; `acw_common/kept.py`, `server/acw/rerun.py`). Measured on a 130 s
recording (`kept.save`, 24-bit FLAC): per hour of audio and per cleaned channel the noise removal at full strength takes about **0.29 GB**
(0.14 GB for a stereo recording whose sides are nearly the same), the recording itself, kept next to it for the website's strength slider
(only for recordings up to 10 minutes of stereo or 20 of mono, `jobs.PREVIEW_MAX_CHANNEL_SECONDS`), **0.36 GB**; the Voice rebuild keeps its
output (about the same as the noise removal) and nothing else; the transcript is a few kB. A channel that repeats the other, or is silent,
is kept once, or not at all. On top of that come the upload as it was sent (a FLAC of 1 h of mono speech: about 0.3 GB) and the results
(a FLAC of the same: 0.3 GB). So a 1 hour mono file holds about 0.9 GB (1.2 GB with the slider copy, which a file that long doesn't have)
for 2 hours, where it held its results (0.3 GB) for 24 hours before: at the example load of 1000 file jobs a day, 2 hours means about 83 jobs
are on disk at any time instead of 1000. The re-run of a job hard-links what it needs (no copy), so it adds only what it writes. Setting
`ACW_JOB_RETENTION_MINUTES` higher keeps all of this longer, lower shortens it (the Privacy Policy says 2 hours).

The monitor watches two sizes every 2 minutes and mails when `data/` without `jobs/` is over 5 GB (`data_large`) or the backup
folder is over 20 GB (`backups_large`). At the example load the first stays around 1 GB (database, log, ten pre-deploy
snapshots of ~0.25 GB each, so up to ~3.5 GB at year 7); raise `ACW_DATA_ALERT_GB` / `ACW_BACKUPS_ALERT_GB` in `acw.env` on
purpose if real growth gets there. `python -m acw.admin storage` and the `acw_storage_bytes{what}` gauge in `metrics.prom`
(database, data, jobs, backups) show the numbers.

**Journal volume (acw's user journal, `acw-logs`, units `acw-*`).** Nothing is logged per audio frame or per WebSocket message. What is
logged: one access-log line per HTTP request (client address, method, path without query, status, milliseconds; about 90
bytes of text, median measured on this machine), one JSON line per finished job and per live session (300 to 500 bytes), a
line when a worker claims a job, warnings and errors, and the monitor's one line every 2 minutes. The one chatty source used to
be the job list a page asks for every 1.5 s while a job runs (1000 jobs a day, each watched for ~2 minutes: ~80 000 lines a day,
most of the log); successful `GET /api/jobs` and health checks are no longer logged (errors and every other request
still are). Per job that leaves roughly 12 lines (create, upload, a few downloads, page loads, the JSON line). A journal entry costs about 1 KB on disk with its metadata (the measured
user journal here: 32 MB), so 1000 jobs and 50 h of live a day come to roughly 13 000 entries, **~13 MB a day, ~1.2 GB in 90
days**. The caps that bound it, whatever the traffic: journald's default `SystemMaxUse` (10 % of the file system, at most
**4 GB** for system and user journals together; here 23 GB would be 10 %, so 4 GB applies; `SystemKeepFree` 15 % does not bite),
and the owner's `MaxRetentionSec=90day` (`/etc/systemd/journald.conf.d/acw-retention.conf`, in place on this machine). Even three
times the example traffic (~40 MB a day) stays under 4 GB for the full 90 days; only beyond that the size cap, not the 90
days, decides, which makes the logs shorter and never longer than the Privacy Policy says.

## Domain

Audio Cleaner has its own domain, `https://audiocleaner.live` (`www.` redirects to it); it moved from
`https://mc-multi-world.space/audio` on 2026-10-18. What was changed:

- DNS (Namecheap): `A` record of `@` to the VPS, `www` a CNAME to `@`. The sites are in `deploy/vps/sites/`
  (`audiocleaner.live.caddy`, the shared routes in `00-audio-cleaner-snippets.caddy`), published with
  `deploy/vps/publish-edge.sh` (docs/vps.md).
- `acw.env` and `acw-worker.env`: `ACW_PUBLIC_ORIGIN=https://audiocleaner.live`, `ACW_BASE_PATH=` (empty). The base path is
  now `/`: pages and API are at `/app`, `/login`, `/contact`, `/api/...`, `/dfn/...`, `/assets/...`, with no `/audio`
  prefix. The session cookie is `__Host-acw_session` (Secure, `Path=/`, no Domain; with a base path it would be
  `__Secure-acw_session`, see `Settings.session_cookie_name`); everybody signed in again once.
- The website is built with `VITE_BASE=/` (`deploy/publish-web.sh`); the monitor builds its health URLs from the settings.
- The old address still answers for a while. On `mc-multi-world.space`, `/audio/api/*` is proxied to the same API with
  the prefix stripped (old OBS plugin builds carry the address, `obs-plugin/src/audio-cleaner-filter.cpp` and
  `core/include/acw/session.hpp`; API clients; Stripe's webhook until it is moved). Every other `/audio/*` page
  redirects (308) to the same page on `audiocleaner.live`.

Still to do (needs you):

1. **Google OAuth client:** `https://audiocleaner.live/api/auth/oauth/google/callback` must be in its authorised redirect
   URIs (the old `mc-multi-world.space/audio/...` one can go once sign-in works on the new address).
2. **Stripe webhook:** add an endpoint `https://audiocleaner.live/api/stripe/webhook` (payments.md, same events) and put its
   signing secret in `acw.env`; keep the old endpoint (`https://mc-multi-world.space/audio/api/stripe/webhook`) until events
   stop arriving there, then delete it. Checkout return URLs follow `ACW_PUBLIC_ORIGIN`.
3. **Retiring the old address:** remove the two `/audio` blocks from `deploy/vps/sites/mc-multi-world.space.caddy` and run
   `deploy/vps/publish-edge.sh`. Keep them until OBS plugin users have updated to a build that uses
   `audiocleaner.live`, and until no Stripe event or API call arrives through `/audio/api` (the Caddy access log on the
   VPS shows it).

## One-time setup still needed (needs you)

1. **Sign-in providers**: create OAuth apps with these callbacks and put the
   ids/secrets in `~/audio-cleaner/secrets/acw.env`:
   - Google: `https://audiocleaner.live/api/auth/oauth/google/callback`
   (Email links need SMTP: see "Mail".)
2. **Stripe** (test mode first): restricted key + webhook (see payments.md) into `acw.env`.
3. **OpenAI**: an `OPENAI_API_KEY` for premium transcription and captions.
   Preferably a dedicated project key with a monthly budget cap.
4. **Mail**: a sending service with the domain verified, then `SMTP_*` in `acw.env` ("Mail").
5. **Contact form**: nothing to do when `ACW_ALERT_EMAIL` and `SMTP_*` are set (messages go to the alert address). Set
   `ACW_CONTACT_EMAIL` only to send them somewhere else (see "Contact form").
6. `acw-ctl restart acw-web` after editing `acw.env`.

The VPS side (Caddy routes, firewall, frp) is in the repository and done: docs/vps.md. Turnstile's three CSP additions
are written in `00-audio-cleaner-snippets.caddy`, commented out; the request-body and header-timeout settings are in
the snippets and in `deploy/vps/Caddyfile`.

## Rebuilding native pieces and models

`deploy/models.lock` lists every model and program file production uses: where it comes from (Hugging Face files on a
pinned commit, our own models in `SilverJunk08/audio-cleaner-models` on a pinned commit, a pinned DeepFilterNet commit, a
pinned FFmpeg build), its size and sha256. The free tools' models are not in it: visitors' browsers download them from
Hugging Face themselves (`web/src/media/models.js` pins them).

```bash
deploy/fetch-models.sh --check -v      # verify only (changes nothing); exit 1 if a required file is missing or wrong
deploy/fetch-models.sh                 # download what is missing or wrong, verify, never touch a correct file (install.sh runs this)
deploy/fetch-models.sh --with-optional # also the TorchScript Sidon weights (research fallback, 1 GB)
```

- **Not downloadable** (marked `local` in the lock): `bin/libdf.so` (build it: `native/deepfilter/build.sh`),
  `bin/libdf-tract021.so` (older build used only by a native test), `models/sidon/sidon_fe_q8.onnx` (optional int8 variant:
  `tools/convert_sidon_onnx.py`), `wasm/df.js` and `wasm/df_bg.wasm` (browser build, same script). `--check` reports them
  missing if absent and "changed" (not an error) if a rebuild differs.
- **Our own models** (Sidon ONNX from `tools/convert_sidon_onnx.py`, the
  batched DPDFNet-8 export, and DeepFilterNet3 with its browser build for the free tools) are in the Hugging Face repo
  `SilverJunk08/audio-cleaner-models`. A new or changed model is a new commit there (`hf upload`); then update the commit
  in `deploy/models.lock`, `desktop/audio_cleaner_desktop/config.py` (HF_MODELS_REVISION) and `web/src/media/models.js`.
  A rebuilt `wasm/df_bg.wasm` must be uploaded there too: `web/scripts/sync-dfn.mjs` stops the site build while the
  local build and the pinned one differ (df.js on the site must belong to it).
- **FFmpeg**: BtbN `autobuild-2026-10-09-14-16`, `ffmpeg-n8.1.3-16-ge0a878dd70-linux64-gpl-8.1.tar.xz`, the build that runs
  today (verified: `ffmpeg` and `ffprobe` match byte for byte). BtbN removes old daily builds; when that URL stops
  working, take a build that is still listed, update the `btbn-ffmpeg` archive line and the two `bin/` lines in the lock,
  and keep a copy of the tarball in our own release.
- **Changing a pin**: download the new file, `sha256sum` it, replace URL, sha256 and size in the lock, run `--check`.
- **Python packages**: `server/requirements.lock` and `deploy/requirements-gpu.lock` carry the sha256 of every file PyPI
  lists for each pinned version; pip installs them with `--require-hashes`. To change a version edit its `name==version`
  line and run `python3 deploy/hash_lock.py server/requirements.lock` (add `--check` to compare with PyPI again,
  `--verify` for an offline format check). Every dependency has to be listed (pip refuses otherwise).

## Capacity (Ryzen 7 1700X, 8C/16T, 31 GB, shared with Minecraft)

More users never slow a running job down; they wait longer for a slot or are
told the service is busy. Measured numbers, 2026-10-14:

**Memory.** Every `acw-*` unit runs in `acw.slice`, capped at 40% of RAM
(12 GB on this 31 GB box; set at install), so Audio Cleaner can't take RAM the
Minecraft servers need.
- File jobs keep decoded and cleaned audio in memory-mapped scratch files in the
  job directory, and the leveling chain works in 10 s blocks. Admission reserves 1 GB per job (4.5 GB for the
  voice-rebuild tier, whose four single-threaded workers hold their own activations): an estimate used to decide when a job may start (`jobs.JOB_MEM`), not a limit. The real
  ceilings are the worker unit's MemoryMax (7 GB) and the runner's address-space limit (32 GB, memory-mapped
  scratch files included). Admission counts memory the way the kernel does: file cache in the slice is not "used".
  Scratch disk: ~0.7 GB per hour and channel, twice.
- Local transcription adds the model (Parakeet 3.2 GB on CPU, of which 2.4 GB are the model file's own page cache,
  shared by concurrent jobs; ~1 GB with the GPU; Whisper turbo ~1.7 GB) and 16 kHz audio (~0.23 GB per hour). CPU worker
  units: MemoryHigh 6G / MemoryMax 7G.
- Live: one shared caption model (~0.3 GB) plus a few MB per session. Unit: 3G/4G.

**File jobs (Studio clean, transcription).** One CPU worker per 3 physical
cores (min 2, max 8; `ACW_WORKERS` overrides), plus the GPU worker when there
is one. A worker takes the next job only when its measured needs fit (memory
and scratch disk with a reserve for the host: 6 GB of free system memory, the point where the Minecraft memory controller reopens world starts, and 2 GB under the acw slice's own MemoryMax; idle cores among the CPUs jobs
may use right now); with nothing running the queue always moves. Queue order:
paid before free, then the user with the fewest jobs running, then first come
first served. The API reports `queuePosition` and `queueWaitSeconds`.
- Studio clean, per channel at 4 threads (stereo is cleaned channel by channel, so double these for stereo):
  DeepFilterNet3 ~0.05x real time, DPDFNet-8 (heavy noise) ~0.33x, voice rebuilt ~0.9x (3 GB, one such job at a
  time, files up to 90 minutes) and echo removal a few percent more. These are the numbers in
  `jobs.SPEED`: they drive the queue estimate and the slow-job alert, and the runner tells the worker which
  noise remover a file got so the estimate follows it.
- Local transcription: Parakeet TDT (English: v2; 24 European languages: v3)
  at ~0.09x real time on 4 CPU threads, ~0.04x on the GTX 960; Whisper
  large-v3-turbo (~0.45x) for every other language. Auto language: Whisper
  small detects it from the first 30 s.

**GPU worker (optional).** `deploy/setup-gpu.sh` builds `~/audio-cleaner/venv-gpu`
(ONNX Runtime with CUDA, pinned for the GTX 960's sm_52), then
`deploy/install.sh` enables `acw-worker-gpu`. It takes the jobs whose heavy stages are much faster on the card
(measurements and reasons: docs/research/gpu-2026-10.md; the rules are `broker.gpu_stages` and `broker.GPU_WORK`): local transcription (Parakeet), the heavy noise
remover (DPDFNet-8: "heavy" or "auto" noise) and the voice rebuild (Sidon). Light cleaning (DeepFilterNet3), echo removal and
Whisper stay on the CPU (a job with echo removal is never given to the GPU worker; an "auto" job that turns out
light is cleaned by DeepFilterNet3 on the GPU worker's CPU threads). It runs one job at a time and a job's models
use the card one after the other (Parakeet needs 3.3 of the 4 GB alone, Sidon 2.6 GB, DPDFNet 0.3-0.4 GB), so nothing
shares it. While it is idle it refreshes `data/gpu-worker-idle`, and CPU workers leave GPU jobs to it; while it is
busy they take everything but the voice rebuild. **The voice rebuild runs only on the card** (7.4x real time there,
0.74x on four CPU cores): while no GPU worker is up (`data/gpu-worker-alive` older than 90 s; it is refreshed every
2 s, and every 20 s during a job, and removed when the worker stops) the site shows it as paused, creating one is
refused, and queued ones wait. Its queue estimate counts the card's own queue. A job made with `options.automatic` (docs/api.md) is queued as
a Denoising job; the worker that takes it measures it first (the runner's `analyse` stage) and, when it decides on a rebuild and is not the
card's worker, puts it back in the queue (same place, `next_stage = main`) for the card's worker; while the card is not up it decides on
Denoising and says so in the job's `chosen`. A stage that cannot get the card (no card, CUDA library missing) fails the job with the usual "not charged"
message instead of running slowly on the CPU. `python -m acw.admin capacity` shows the worker's state and the card's
memory and load. Its sandboxed runner may also open `/dev/nvidia*` and has no address-space
limit (CUDA reserves huge virtual ranges).

**Sized from the machine.** `deploy/install.sh` reads the core count and RAM:
all services together may use `ACW_MEMORY_PERCENT` (default 40) of RAM
(`acw.slice.d/size.conf`), file workers scale with cores, and live capacity
scales with cores and the start-up speed calibration. Moving to a bigger
machine needs no edits: run `deploy/install.sh` there.

**Live comes first: the native live engine (native/live, `bin/acw-live`, the `acw-stream` unit).** One process
serves every live session; `deploy/install.sh` builds it with `native/live/build.sh` when its sources change (it
links the sherpa-onnx C library from the venv) and stops if the build fails. It refuses to start without libopus
or a libdf that shares the model weights (`df_shares_model`, native/deepfilter/tract-0.23.patch). Each session is a
thread pinned to one physical core, lowest cores first (the `ready` event says which); live may use all cores but
`ACW_MIN_JOB_CORES` (default 1 per 8 cores), which file jobs always keep. File jobs keep off the cores with
sessions plus one spare, re-pinned every second, so they slow down instead of slowing live audio (a job thread on a
live core's other hyper-thread halves its speed).

Capacity is in units (live denoise = 1, a standard caption stream = 1.2, echo cancellation = 0.2): live cores x
units per core, shrunk by other load measured on the live cores. Units per core are calibrated at start-up from how
fast this CPU runs the model: 6.5 at the reference cost of 0.110 cores per session (about 6.9 on the 1700X),
remembered per CPU in `data/live-calibration-native.json`. The live process (`acw-live`) also
refuses new sessions when its own event loop lags (p90 over `ACW_LOOP_LAG_LIMIT_MS`, default 25). Caption models (about 220 MB each) are loaded in
the background, the first at start; at most 5 streams each (`ACW_CAPTION_STREAMS_PER_MODEL`, `ACW_CAPTION_THREADS`),
the next one loading before they are full. Echo cancellation is `sonora` (WebRTC AEC3 in Rust): 42.8 dB of echo
removed on our test room.

Measured 2026-10-16 (Opus, 20 ms frames, round trip inside the server):

| sessions on 5 cores | p99 | captions behind |
|---|---|---|
| 35 noise removal | 10 ms | |
| 45 noise removal | falls behind | |
| 12 noise removal + standard captions | | within 0.3 s |
| 20 noise removal + standard captions | | falls behind |

What made the difference (before: 15 noise-removal sessions on 7 cores): libdf kept a copy of the 8 MB model per
session, so at 28 sessions each one cost 0.40 cores instead of 0.14 (the weights evicted one another from the CPU
caches; the patch now shares one copy, bit-identical output); one native process instead of Python workers that
each fell behind at about ten sessions (the GIL around every native call); Opus through libopus; WebSocket
compression off (it deflated every audio frame: a quarter of the relay's work); standard captions skip silence (2 s
after the last speech, with 0.3 s kept before the next words; word error rate unchanged on LibriSpeech with pauses);
digital silence from a noise gate or a muted interface stops going through the noise model after 0.2 s, and clients
that know they are silent send a 3-byte silence frame instead of audio.

**Resource broker (`acw/broker.py`).** One ranked table decides who gets the machine: paid live, then paid file
jobs, then free-trial live. Jobs keep off the cores live uses; a free-trial session only joins cores live already
uses while paid jobs run. `python -m acw.admin capacity` (or `--json`) shows everything in one place: CPU per
logical core, memory and disk against their reserves, the live engine's sessions/units/caption streams, running
and queued jobs and the cores jobs may use.

**libDF** (`bin/libdf.so`) is built by `deploy/install.sh` (`native/deepfilter/build.sh native` in
`~/audio-cleaner/src/deepfilter-build`) whenever `native/deepfilter` changes, then installed and `acw-stream` restarted.
The engine and file jobs need its additions (`df_shares_model`, `df_reset`, `df_create_with_params`).

When full, a new session is refused with the options that still fit; sessions that only use the external
(extra-accurate) captions are never limited by the stream workers. Other limits (all overridable in `acw.env`):

| Variable | Default | Meaning |
|---|---|---|
| `ACW_STREAM_MAX_CAPTION_STREAMS` | 5 per core of the worker | Standard caption streams per worker (in `acw-worker.env`). Each worker's caption decoder does about 13x real time in total; silence isn't decoded, but a stream can talk all the time. At the cap a new one goes to the next worker, or is offered extra-accurate captions. A stream that still falls more than 2 s behind drops its oldest audio and the client gets one `captions_behind` warning per incident |
| `ACW_LOOP_LAG_LIMIT_MS` | `25` | the live process (`acw-live`) relays all live audio on one event loop: new live sessions are refused while it runs a 100 ms timer this late (p90 over 10 s). Running sessions are never cut. About 1.5% of a core per Opus session (2% per PCM session) was measured on 2026-10-18, so this rarely triggers |
| `ACW_MAX_LIVE_SESSIONS` | `300` | sanity bound on live sessions in the web process (below uvicorn's `--limit-concurrency 500`) |
| `ACW_MAX_ACTIVE_JOBS` | `3` | file jobs one account may have at once (uploading, waiting or running; changing a finished job counts): one more is refused with 429 and the code `too_many_active_jobs`, "You have 3 files being processed. Start this one when one of them is done." (per-key monthly caps are separate, `keycaps.py`) |
| `ACW_JOB_RETENTION_MINUTES` | `120` | how long a finished job's upload, kept copies and results stay, counted from when it finishes; a re-run never outlives the job it came from |
| `ACW_MAX_INTERRUPTIONS` | `3` | a job that a stopping worker puts back this many times is failed (not charged) instead of being retried for ever; a runner killed by the kernel for memory fails at once |

Other queue limits: the job list shares one queue computation per second between all requests (`jobs.queue_snapshot`),
the web page polls every 5 s while jobs only wait and every 1.5 s while one uploads or runs (every 15 s while its tab is
in the background, and once at once when it is shown again); extra-accurate
transcription and file translation have two places each (`PREMIUM_JOBS_AT_ONCE`, `TRANSLATION_JOBS_AT_ONCE` in
`app.py`) so neither waits for the other; at most 8 requests to the text model run at once in the web process and a live
translation line is retried once after 0.5 s and then skipped; a stage that runs past its time limit but is still
making progress is given the time its progress says it needs, up to 3x the limit. On a bigger CPU all of this scales by itself (cores from the
topology, speed from the calibration).

## Moving the services to acw (one time) and back

`deploy/host/move-to-acw.sh` moves the production services from `silver` to the account `acw` (why and what it means:
"Accounts" near the top and docs/security.md). Read this once, run it once.

```bash
cd ~/repos/Audio-Cleaner-Web
sudo bash deploy/host/move-to-acw.sh --dry-run     # every step printed, the read-only checks run, nothing changed (works without sudo)
sudo bash deploy/host/move-to-acw.sh               # asks "Type yes" (--yes skips it)
# the way back, any time (also after a failed half-way run):
sudo bash deploy/host/move-back-to-silver.sh
```

**Downtime**: from the moment the services stop until `/api/health` answers again, about 3 to 6 minutes (stop 10-60 s: a
stopping worker puts its job back in the queue; the move itself is renames; rebuilding `acw-live` against the moved venv 30 s to
2 minutes with the build cache warm; start and health check about 30 s). Live sessions are cut: the script refuses to start
while one runs (and refuses jobs running, too) unless `--force`. Run it in a quiet hour; `deploy/drain.sh wait` first if you like.

What it does, in this order (`--dry-run` prints each command):

1. **Checks** (nothing changed): root, the owner's folder has `app`, `venv`, `secrets`; the destination parent is the same
   filesystem (else `mv` would copy 9 GB: it stops); nothing exists in both places; no job or live session is running
   (`acw.admin activity`, a read-only look at the database, like `install.sh`'s "running now" line); the tunnel key, the `vps`
   host key and `secrets/vps.env` exist; the backup folder exists; `/home/silver` is not world-accessible.
2. **Account**: `useradd --create-home --home-dir /home/acw --shell /usr/sbin/nologin --user-group acw`, `chmod 750 /home/acw`,
   the owner joins group `acw`, `loginctl enable-linger acw`, acw's user manager is started and waited for.
3. **sudoers** `/etc/sudoers.d/acw` (checked with `visudo -cf` before it is installed) and the wrappers in `/usr/local/bin`.
4. **Downtime starts**: `new live sessions refused` (`data/drain`), the activity check again, the unit files and drop-ins are copied
   to `/var/lib/acw-move/units-silver` (acw gets exactly what ran), the owner's `acw-*` units are disabled and stopped
   (`disable --now`, then waits until none is active), their unit files are removed from the owner's manager so nothing
   starts there by accident, and a database snapshot is taken with everything stopped.
5. **Move** `app app.prev venv venv-gpu bin models wasm data` to `/home/acw/audio-cleaner` and the production secrets
   (an allowlist: `acw.env acw-worker.env vps.env backup.env backup.pass github-releases-read.token`); `src/`, `dev/`,
   `research/`, `listen/` and every other secret (`update-signing-key.pem`, `github-actions-*.token`) stay with the owner.
   `/home/silver/audio-cleaner/{bin,models,wasm,venv,venv-gpu}` become links to acw's folders (the web build's `npm run sync-dfn`, the tools and the tests read them there). Old home paths in the moved `*.env`
   files are rewritten. Modes: `app app.prev data secrets` 0700 and the secret files 0600 (acw only: the owner cannot read
   them, on purpose); `bin models wasm venv venv-gpu` group-readable (`g=rX`, `o=`) with an ACL for the owner.
6. **Venvs**: `deploy/host/fix-venv-paths.sh` rewrites the old absolute path in `venv/bin/*` (shebangs, `activate*`) and
   `pyvenv.cfg` (nothing else in the venvs names it): no reinstall, venv-gpu's 3.9 GB of CUDA wheels stay. Tested on a copy
   (`pip --version`, `uvicorn --version` fail before the rewrite and work after) and by `server/tests/test_move_to_acw.py`.
7. **The tunnel key**: `acw_tunnel(.pub)` is copied to `/home/acw/.ssh` (0600, acw) and the `vps` host key line(s)
   (`ssh-keygen -F vps`, hashed or not) to `/home/acw/.ssh/known_hosts`. The owner's own login key and `~/.ssh/config`
   are not copied. The backup folder `/mnt/mc-storage/audio-cleaner-backups` is chowned to acw (750).
8. **`acw-live` is rebuilt** by the owner against the venv in its new place. This cannot be skipped: the binary records the
   absolute path of the sherpa-onnx library twice (rpath and the folder its Landlock sandbox lets it read); copied as is it
   would look in `/home/silver/...`, which acw cannot read, and live would not start. The source stamp `bin/acw-live.src`
   stays valid (it hashes the sources), so the next deploy does not rebuild it. (`libdf.so` has no such path.)
9. **Units**: the stashed unit files and drop-ins are installed in `/home/acw/.config/systemd/user`, `/etc/acw-service-user`
   is written (from now on `deploy/*.sh` act through acw), `daemon-reload`, `enable --now` of exactly the units that were enabled
   before (stream worker first, timers last), then the same health check as `install.sh` (every unit active; `/api/health?deep=1`
   on 8790, and /api/health on 8794 and 8795; up to 120 s), the drain flag is removed.
10. **Smoke test** (`deploy/smoke.py`, as the owner) and an **isolation check** (as acw: `/home/silver`, `~/.ssh/id_ed25519` and
    `~/audio-cleaner/secrets` of the owner must not be readable).

**What could break, and the guard.** Work running (refused). A different filesystem (refused). `mv` onto an existing folder (refused; the
script never overwrites). The user manager of acw not up (waited for 30 s, then it stops before touching anything). The owner's
manager not running while units are enabled (refused: they could not be stopped cleanly). A path to the old home hiding in an env file
(rewritten for `ACW_GITHUB_TOKEN_FILE`-style settings; any other mention of `/home/silver` is reported by name, never by content).
Rebuilding `acw-live` failing (the script stops with the old engine in place and the services stopped: fix it, run again, or go back).
The health check failing: the script prints what failed and does **not** roll back by itself:

```
FAILED during: <the step> (exit N; the error is above)
  go on:     fix the cause, then run this script again; it continues where it stopped
  go back:   sudo bash deploy/host/move-back-to-silver.sh
```

Both scripts are idempotent: each step looks at what is already done (moved folders, existing account, installed sudoers file,
unit list recorded in `/var/lib/acw-move/units.<user>`). They never print secret contents (only file names).

**Afterwards.** Log out and in once (or restart the sessions that read `~/audio-cleaner/{models,bin}`): group `acw` only reaches
running sessions at the next login; the ACL entry covers the folders meanwhile. Open the dashboard as before
(`ssh -L 8796:127.0.0.1:8796 silver@primary-ram`). The journal of the old units stays in the owner's user journal; new lines
are in acw's (`acw-logs`). If the journal retention (`MaxRetentionSec`) is set system-wide it applies to both.

**Way back**: `move-back-to-silver.sh` does the same steps in reverse (units back from the stash, folders and secrets back, links
removed, `acw-live` rebuilt against the owner's venv, tunnel key removed from `/home/acw/.ssh`, backup folder returned,
`/etc/acw-service-user` removed). The account, `/etc/sudoers.d/acw` and the wrappers stay (harmless); the last lines of its
output show how to remove them.
