# Operations runbook (primary-ram)

## Layout

```
~/audio-cleaner/
  app/        deployed code (copies of server/acw and common/acw_common, see app/VERSION)
  app.prev/   the previous copy of the code: deploy/rollback.sh brings it back
  venv/       Python 3.12 virtualenv (server/requirements.lock, installed with --require-hashes); venv-gpu/ for the GPU worker
              (deploy/setup-gpu.sh, deploy/requirements-gpu.lock)
  bin/        ffmpeg, ffprobe (BtbN static n8.1.3, pinned in deploy/models.lock), libdf.so (built here)
  models/     DeepFilterNet3_onnx.tar.gz, dpdfnet/, ap-bwe/, sidon/, parakeet-v2/, parakeet-v3/, captions-en/,
              silero-vad/, whisper/{large-v3-turbo,small}   (every file, its source and sha256: deploy/models.lock)
  wasm/       df.js + df_bg.wasm for the browser build
  data/       acw.sqlite3, jobs/, stream.sock, backups/ (pre-deploy snapshots), lock.sha, metrics.prom   (0700)
  secrets/    acw.env (web, secrets), acw-worker.env   (0700, files 0600)
  src/        build trees (DeepFilterNet, deepfilter-build) - safe to delete
  dev/        local dev instance (dev.sh) - safe to delete
```

## Deploy / update

Always deploy from `~/repos/Audio-Cleaner-Web`, branch `on-prem` (fast-forward it from `desktop-merge`; the
script refuses another history, so an old checkout can't overwrite newer code):

```bash
cd ~/repos/Audio-Cleaner-Web && git checkout on-prem && git merge --ff-only desktop-merge   # or pull
deploy/install.sh                      # check, back up, copy, restart what changed, verify, publish the site if web/ changed
deploy/install.sh --wait-live          # stop new live sessions and wait (up to 10 min) for the running ones first
deploy/install.sh --wait-live=1800     # ... or for as long as you say
deploy/install.sh --no-restart         # copy everything, restart nothing
ACW_WORKERS=3 deploy/install.sh        # change the number of file-job workers
deploy/rollback.sh                     # put the previous code back (run again to undo)
deploy/drain.sh on|off|status|wait     # refuse new live sessions by hand (e.g. before a long maintenance)
deploy/publish-web.sh                  # build the site and rsync it to the VPS (install.sh does it when web/ changed)
```

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

## Services (user units, linger on)

```bash
systemctl --user status 'acw-*'
journalctl --user -u acw-web -f
systemctl --user restart acw-web
```

Ports on primary-ram (all loopback) and how they reach the VPS:

| Unit | Listens on (home) | VPS side | Carries |
|---|---|---|---|
| `acw-web` | `127.0.0.1:8790` | `127.0.0.1:18790` via `acw-tunnel` (`-o IPQoS=lowdelay`) | everything except the two bulk routes: sign-in, API, billing, the live WebSocket |
| `acw-bulk` | `127.0.0.1:8794` | `127.0.0.1:18794` via `acw-tunnel-bulk` (`-o IPQoS=throughput`) | uploads (`PUT /api/jobs/{id}/input`) and result downloads (`GET /api/jobs/{id}/outputs/{name}`) only |

Check from the VPS with `curl -s 127.0.0.1:18790/audio/api/health` and `curl -s 127.0.0.1:18794/audio/api/health`.

## Bulk transfers (uploads and downloads on their own process and tunnel)

Why: a 4 GB upload or a result download used to share one Python process, one SSH connection and the home uplink with
the live audio WebSocket, so a big transfer could add latency to live sessions (head-of-line blocking in the tunnel's
TCP stream, one event loop, one uplink queue). Now those two routes can run through `acw-bulk` (`acw/bulk_app.py`: the
same route functions, authentication, body limits, upload caps and headers as the main app; it serves nothing else,
runs no housekeeping, cloud transcription or live code) and their own SSH connection. The main process keeps serving
the same two routes, so nothing breaks before the VPS is switched, and the VPS falls back to it when the bulk side is down.

Rates (environment of `acw-bulk`: add to `~/audio-cleaner/secrets/acw.env`, then `systemctl --user restart acw-bulk`):

| Variable | Default | Meaning |
|---|---|---|
| `ACW_DOWNLOAD_MBPS_TOTAL` | `4` | megabytes per second for all result downloads together (0 = no cap). Downloads leave over the home uplink, which also carries the live audio replies, so this must leave room: **about 70 % of the uplink speed you measured** (the home line is about 52 Mbit/s = 6.5 MB/s, so 4 to 4.5) |
| `ACW_UPLOAD_MBPS_TOTAL` / `ACW_UPLOAD_MBPS_USER` | `12` / `6` | megabytes per second for all uploads / per account (uploads arrive over the home downlink) |

**Rollout, in this order** (each step is safe on its own; nothing is applied by the repository):

1. **Home side.** `deploy/install.sh --wait-live` (the first deploy also restarts `acw-tunnel`, because its unit gained
   `-o IPQoS=lowdelay`, and that cuts live sessions; later deploys never restart it). It starts `acw-bulk` and installs
   `acw-tunnel-bulk` **without enabling it**. Check: `curl -s 127.0.0.1:8794/audio/api/health` answers
   `{"ok":true,"process":"bulk"}`.
2. **VPS key.** The tunnel key is restricted to the ports it may listen on. On the VPS, edit `~/.ssh/authorized_keys` and
   change the line of the `acw_tunnel` key (keep the key itself and its comment) so it starts with

       restrict,port-forwarding,permitlisten="127.0.0.1:18790",permitlisten="127.0.0.1:18794",permitopen="127.0.0.1:1",command="/bin/false" ssh-ed25519 AAAA... acw-tunnel@primary-ram

   (one more `permitlisten`; sshd accepts several). Nothing else changes, and the main tunnel keeps working during the edit.
3. **Bulk tunnel.** On primary-ram: `systemctl --user enable --now acw-tunnel-bulk`, then on the VPS
   `curl -s 127.0.0.1:18794/audio/api/health`. From now on `deploy/install.sh`, `deploy/rollback.sh` and the monitor treat it
   like the other units (`deploy/install.sh` prints a reminder until it is enabled).
4. **Caddy.** `scp ~/repos/Audio-Cleaner-Web/deploy/vps/audio.caddy ~/repos/Audio-Cleaner-Web/deploy/vps/install-audio-route.sh vps:~/acw/`
   then on the VPS `sudo bash ~/acw/install-audio-route.sh` (validates, reloads, keeps a backup, reverts on error). The upload
   route (`.../input`, still 4 GB, no stall timeouts) and `.../outputs/...` now go to `127.0.0.1:18794` first and fall back to
   `127.0.0.1:18790` (`lb_policy first`, a health check every 10 s, 30 s of avoidance after a failed connection), so a
   stopped `acw-bulk` or `acw-tunnel-bulk` costs a few seconds, not an outage. This snippet no longer uses
   `request_body { read_timeout }`, which Caddy's Caddyfile syntax does not have (the old snippet failed `caddy validate`):
   the 20 s stall limit for small bodies is enforced by the app (`acw/bodylimit.py`).
5. **Check.** Upload and download a file on the site; `journalctl --user -u acw-bulk -f` shows the `PUT .../input` and
   `GET .../outputs/...` lines (client address, status, milliseconds), the main `acw-web` log shows none for them.

Back out: on the VPS copy the newest `/etc/caddy/Caddyfile.backup.*` (the installer made it) over `/etc/caddy/Caddyfile`
and `sudo systemctl reload caddy`; on primary-ram `systemctl --user disable --now acw-tunnel-bulk`. The main path never changed.

The router: `IPQoS` marks the two SSH connections so a router that prioritises by DSCP or runs smart queueing (SQM,
cake/fq_codel) can favour the live one. It does nothing without such a router; the separate connection and process are
what helps in any case. Setting up SQM on the home router remains the best second step (docs/launch-checklist.md).

## Admin CLI

```bash
cd ~/audio-cleaner/app && set -a && . ../secrets/acw.env && set +a
../venv/bin/python -m acw.admin login-link you@example.com   # one-time sign-in URL
../venv/bin/python -m acw.admin grant someone@example.com 5 "support credit"
../venv/bin/python -m acw.admin user someone@example.com
../venv/bin/python -m acw.admin suspend someone@example.com "reason"
../venv/bin/python -m acw.admin stats            # report for the last 30 days (see "Statistics and metrics")
../venv/bin/python -m acw.admin stats --days 7
../venv/bin/python -m acw.admin activity       # file jobs and live sessions running now
../venv/bin/python -m acw.admin check-money    # balances and holds against the ledger (exit 1 on a mismatch)
../venv/bin/python -m acw.admin verify-backup FILE.sqlite3.gz
```

## Statistics and metrics

**`acw.admin stats [--days N]`** (default 30; read-only, a few SELECTs) prints a plain-text report from the database:
file jobs by tier and final status with audio hours, p50/p95 queue wait and run time, the real-time factor (run seconds per
audio second), in-progress jobs, the failure ratio and the five most common error messages; live sessions (count,
minutes, end reasons, by tier, frames that were late or never came back); free-tier minutes used; usage charged (jobs and
live) and top-ups paid. Percentiles are nearest-rank. "Queue wait" is the time until a worker first took the job (recorded
in `jobs.wait_seconds`; older jobs fall back to `started_at - queued_at`), "run" is start to finish.

**One JSON log line per finished job and live session** in the journal of the process that ended it (`acw-worker@N` for
jobs, `acw-web` for live sessions); no user ids, addresses or file names:

```bash
journalctl --user --since -1d -o cat | grep '"event":"job_'           # job_done and job_failed
journalctl --user --since -1d -o cat | grep '"event":"live_session_end"'
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

**Set up (once).** Put this in `~/audio-cleaner/secrets/acw.env` (the monitor
and the failure hook read that file; workers send no mail):

```
ACW_ALERT_EMAIL=silverjunk08@gmail.com
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=<your gmail address>
SMTP_PASSWORD=<16-char app password>   # https://myaccount.google.com/apppasswords (needs 2-step verification)
SMTP_FROM=<your gmail address>
```

Then `systemctl --user restart acw-web` and check with:

```bash
cd ~/audio-cleaner/app && set -a && . ../secrets/acw.env && set +a
../venv/bin/python -m acw.alerts test      # sends a test mail
../venv/bin/python -m acw.alerts status    # every alert kind, active or not, mails sent / suppressed
```

Until SMTP and `ACW_ALERT_EMAIL` are both set nothing is mailed: every alert is
logged at ERROR (`journalctl --user -u acw-monitor -u acw-web | grep -i 'ALERT NOT SENT'`)
and goes out by itself, still pending, once mail works. The same SMTP account is used
for magic-link sign-in mails.

**De-duplication.** Each alert has a kind. The first occurrence is mailed; while it
stays active it is mailed again after 1 hour and then every 6 hours; when the
condition clears (2 clean checks in a row) one "Resolved" mail is sent and the kind
is re-armed, so the next occurrence mails immediately. State lives in the
`alert_state` table.

| Alert (kind) | Fires when | Notes |
|---|---|---|
| Service not running (`service:<unit>`) | an enabled `acw-*` unit (or `acw-backup.timer`) is not `active` for 2 checks in a row; also sent at once by the `OnFailure=` hook (`acw-alert@.service`) when a unit enters `failed` | the hook only fires when systemd gives up; with `Restart=always` that is rare, so the next row covers crash loops |
| Restarting repeatedly (`restarts:<unit>`) | systemd auto-restarted the unit 2+ times in 30 minutes (`NRestarts`) | a manual `systemctl restart` does not count |
| Back up after unclean stop (`web_unclean_stop`) | acw-web starts and the previous run did not record a clean shutdown (crash, kill, OOM, power loss) | at most 1 mail per hour |
| API down locally (`health_local`) | `http://127.0.0.1:8790/audio/api/health` fails twice in a row | acw-web is down or stuck |
| Upload/download process down (`health_bulk`) | only where `acw-bulk` is enabled: `http://127.0.0.1:8794/audio/api/health` fails twice in a row | uploads and downloads fall back to `acw-web` once the VPS notices (docs/operations.md, Bulk transfers); `acw-tunnel-bulk` is watched as a service like the other units |
| Not reachable from the internet (`health_public`) | local health is fine but `https://mc-multi-world.space/audio/api/health` fails from this machine twice in a row | the problem is the tunnel, VPS or Caddy; skipped when the public origin is not https |
| File queue long (`queue_long`) | the head job has waited over 20 min, or the last job's estimated wait (`jobs.queue_wait_seconds`) is over 45 min | the mail shows busy cores for Audio Cleaner vs everything else (with the busiest other process names, e.g. `java`) and memory, so you can see when the Minecraft servers are the cause |
| Jobs abnormally slow (`jobs_slow`) | a running job, 10+ minutes in, is projected to take over 3x its `jobs.job_seconds` estimate | same host load breakdown |
| Disk low (`disk_low`) | under 15 GB free on the data volume | |
| Memory low (`mem_low`) | under 2 GB `MemAvailable` for 2 checks in a row | host load breakdown included |
| Swap growing (`swap_growing`) | at least 512 MB of swap in use and up 256 MB or more over 30 minutes | |
| Live sessions refused (`live_full`) | more than 10 stream sessions ended with `end_reason = 'full'` in the last hour (counted from `stream_sessions`; `ACW_ALERT_FULL_THRESHOLD` changes the 10) | |
| Backup overdue (`backup_stale`) | the newest snapshot in `/mnt/mc-storage/audio-cleaner-backups` (`ACW_BACKUP_DIR`) is older than 36 h, or none is found (disk not mounted) | |
| Backup failed its check (`backup_bad`) | `backup-status.json` next to the snapshots says the file is empty, failed SQLite's integrity check, or its ledger sums do not match, or the off-machine copy failed | the backup unit fails too |
| Balances do not match the ledger (`ledger_invariant`) | an account's balance differs from the sum of its ledger entries, or its held amount from its active holds | look before restarting anything; `acw.admin check-money` |
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
example UptimeRobot, free) at `https://mc-multi-world.space/audio/api/health`
(that URL answers only `{"ok":true}`; the detailed check `?deep=1` works from this machine and for admins).

## Backups

Every night (`acw-backup.timer`, 04:17) `python -m acw.admin backup` writes a consistent snapshot of the database to
`/mnt/mc-storage/audio-cleaner-backups` (another disk; 30 kept), then **checks it**: the file is unzipped to a
scratch copy, `PRAGMA integrity_check` must say `ok`, and every account's balance must equal the sum of its ledger
entries and its held amount the sum of its active holds. A snapshot that fails is renamed `*.bad`, the unit fails (you
get a mail) and `backup-status.json` records why (the monitor reads it). `deploy/install.sh` takes the same kind of
snapshot into `data/backups` before every deploy.

**Off the machine (switched off until you configure it).** Create `~/audio-cleaner/secrets/backup.env` (0600):

```
ACW_BACKUP_OFFBOX=user@host:/path/to/backups/       # an rsync destination (ssh key login, no password)
ACW_BACKUP_PASSPHRASE_FILE=/home/silver/audio-cleaner/secrets/backup.pass   # a file holding a long passphrase (gpg --symmetric, AES256)
# or, with `age` installed:  ACW_BACKUP_AGE_RECIPIENT=age1...   (a public key; keep the private key elsewhere)
```

Each nightly snapshot is then encrypted and copied; the unit fails if that does not work. The remote end keeps
everything it is given (prune it there). Also worth copying once, by hand, because the database does not hold them:
`secrets/` (the passphrase file belongs somewhere other than the backup target) and `models/` (9 GB, partly converted by hand).
After setting it up, run `systemctl --user start acw-backup.service` and look at `journalctl --user -u acw-backup -n 20`.

**Restore** (the database only; everything else is code and models):

```bash
cd ~/audio-cleaner/app && set -a && . ../secrets/acw-worker.env && set +a
# 1. pick a snapshot and check it first (an encrypted off-machine copy: gpg --decrypt -o x.sqlite3.gz x.sqlite3.gz.gpg)
ls -1 /mnt/mc-storage/audio-cleaner-backups | tail
../venv/bin/python -m acw.admin verify-backup /mnt/mc-storage/audio-cleaner-backups/acw-YYYYMMDD-HHMMSS.sqlite3.gz
# 2. stop everything that writes (the tunnel can stay: the site shows its maintenance message)
systemctl --user stop acw-web 'acw-worker@*' acw-worker-gpu acw-stream acw-monitor.timer acw-backup.timer
# 3. keep what is there, put the snapshot in place
cd ~/audio-cleaner/data && mkdir -p ../data-before-restore && cp -a acw.sqlite3* ../data-before-restore/ && rm -f acw.sqlite3 acw.sqlite3-wal acw.sqlite3-shm
gunzip -c /mnt/mc-storage/audio-cleaner-backups/acw-YYYYMMDD-HHMMSS.sqlite3.gz > acw.sqlite3 && chmod 600 acw.sqlite3
# 4. schema (additive), then the money check, then start
cd ~/audio-cleaner/app && ../venv/bin/python -m acw.admin init-db && ../venv/bin/python -m acw.admin check-money
systemctl --user start acw-stream acw-web 'acw-worker@1' 'acw-worker@2' acw-monitor.timer acw-backup.timer   # and the other workers
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

`deploy/install.sh` reminds you until that file or a `MaxRetentionSec` exists. The database side runs by itself:
`acw/retention.py` (from the housekeeping loop, hourly) removes IP addresses from live sessions, the audit log and
sessions after 90 days and job file names after 12 months. Ledger and payment records stay 7 years.

## Moving to a domain of its own

Only configuration changes (the shared host `mc-multi-world.space/audio` is the weak point, see security.md):

1. DNS: an A record for the new name to the VPS (216.250.115.225); a Caddy site block for it with the contents of
   `deploy/vps/audio.caddy` but **without** the `/audio` prefix (routes at `/`, root `/var/www/audio`, same CSP).
2. `~/audio-cleaner/secrets/acw.env` and `acw-worker.env`: `ACW_PUBLIC_ORIGIN=https://new.name`, `ACW_BASE_PATH=` (empty).
   The session cookie then becomes `__Host-acw_session` (Secure, `Path=/`, no Domain) by itself; everybody signs in again.
3. Google OAuth client: add `https://new.name/api/auth/oauth/google/callback`; Stripe webhook endpoint
   `https://new.name/api/stripe/webhook` (keep the old one until events stop arriving there); Stripe return URLs follow the origin.
4. Website: build with `VITE_BASE=/` (publish-web.sh uses `/audio/` today; change it there), the health URLs in
   `monitor.py`/`publish-web.sh`, `deploy/systemd/acw-tunnel.service` stays as it is.
5. Keep a redirect from the old `/audio/*` for a while, and note that OBS plugin builds carry the server address
   (`obs-plugin/src/audio-cleaner-filter.cpp`, `core/include/acw/session.hpp`): keep the old host answering until users have updated.
6. `deploy/install.sh`, then check `/api/health` on the new name.

## One-time setup still needed (needs you)

1. **VPS Caddy route** (sudo): `ssh vps` then `sudo bash ~/acw/install-audio-route.sh`.
   Adds `/audio` before the 404 catch-all, creates `/var/www/audio` owned by
   silver, validates (reverts on failure), and reloads.
   Re-run it (after `scp deploy/vps/audio.caddy deploy/vps/install-audio-route.sh vps:~/acw/`) whenever the
   snippet changes. The current one caps request bodies (64 KB; 4 GB only for the upload route; 1 MB for the
   Stripe webhook), sends uploads and result downloads to the bulk tunnel first (see "Bulk transfers", which also
   needs the tunnel key changed on the VPS first) and was validated with `caddy validate` on Caddy 2.11. Once, add to the **global options**
   block at the top of `/etc/caddy/Caddyfile`: `servers { timeouts { read_header 10s } }`, then `sudo caddy validate` and reload.
   Turnstile's three CSP additions are written in the snippet, commented out.
2. **Sign-in providers**: create OAuth apps with these callbacks and put the
   ids/secrets in `~/audio-cleaner/secrets/acw.env`:
   - Google: `https://mc-multi-world.space/audio/api/auth/oauth/google/callback`
   (Email links need SMTP: any transactional provider.)
3. **Stripe** (test mode first): restricted key + webhook (see payments.md) into `acw.env`.
4. **OpenAI**: an `OPENAI_API_KEY` for premium transcription and captions.
   Preferably a dedicated project key with a monthly budget cap.
5. `systemctl --user restart acw-web` after editing `acw.env`.

## Rebuilding native pieces and models

`deploy/models.lock` lists every model and program file production uses: where it comes from (Hugging Face files on a
pinned commit, a pinned DeepFilterNet commit, the `models-v1` release archive, a pinned FFmpeg build), its size and sha256.

```bash
deploy/fetch-models.sh --check -v      # verify only (changes nothing); exit 1 if a required file is missing or wrong
deploy/fetch-models.sh                 # download what is missing or wrong, verify, never touch a correct file (install.sh runs this)
deploy/fetch-models.sh --with-optional # also the TorchScript Sidon weights (research fallback, 1 GB)
ACW_ARCHIVE_DIR=~/audio-cleaner-models-release deploy/fetch-models.sh   # use a local copy of the models-v1 zip
```

- **Not downloadable** (marked `local` in the lock): `bin/libdf.so` (build it: `native/deepfilter/build.sh`),
  `bin/libdf-tract021.so` (older build used only by a native test), `models/sidon/sidon_fe_q8.onnx` (optional int8 variant:
  `tools/convert_sidon_onnx.py`), `wasm/df.js` and `wasm/df_bg.wasm` (browser build, same script). `--check` reports them
  missing if absent and "changed" (not an error) if a rebuild differs.
- **AP-BWE and Sidon ONNX** come from the `models-v1` release archive of the Audio-Cleaner-Web repository (the conversions
  `tools/export_apbwe_onnx.py` and `tools/convert_sidon_onnx.py` produced them). Until that release is published the
  fetch fails for them; use the local zip (`ACW_ARCHIVE_DIR` above).
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
  job directory, and the polish chain works in 10 s blocks. Admission reserves 1.5 GB per job (3 GB for the
  voice-rebuild tier): an estimate used to decide when a job may start (`jobs.JOB_MEM`), not a limit. The real
  ceilings are the worker unit's MemoryMax (7 GB) and the runner's address-space limit (32 GB, memory-mapped
  scratch files included). Admission counts memory the way the kernel does: file cache in the slice is not "used".
  Scratch disk: ~0.7 GB per hour and channel, twice.
- Local transcription adds the model (Parakeet ~4.3 GB on CPU, ~1 GB with the
  GPU; Whisper turbo ~1.7 GB) and 16 kHz audio (~0.23 GB per hour). CPU worker
  units: MemoryHigh 6G / MemoryMax 7G.
- Live: one shared caption model (~0.3 GB) plus a few MB per session. Unit: 3G/4G.

**File jobs (Studio clean, transcription).** One CPU worker per 3 physical
cores (min 2, max 8; `ACW_WORKERS` overrides), plus the GPU worker when there
is one. A worker takes the next job only when its measured needs fit (memory
and scratch disk with a reserve for the host; idle cores among the CPUs jobs
may use right now); with nothing running the queue always moves. Queue order:
paid before free, then the user with the fewest jobs running, then first come
first served. The API reports `queuePosition` and `queueWaitSeconds`.
- Studio clean, per channel at 4 threads (stereo is cleaned channel by channel, so double these for stereo):
  DeepFilterNet3 ~0.05x real time, DPDFNet-8 (heavy noise) ~0.33x, voice rebuilt ~0.9x (3 GB, one such job at a
  time, files up to 90 minutes), highs restoration and echo removal a few percent more. These are the numbers in
  `jobs.SPEED`: they drive the queue estimate and the slow-job alert, and the runner tells the worker which
  noise remover a file got so the estimate follows it.
- Local transcription: Parakeet TDT (English: v2; 24 European languages: v3)
  at ~0.09x real time on 4 CPU threads, ~0.04x on the GTX 960; Whisper
  large-v3-turbo (~0.45x) for every other language. Auto language: Whisper
  small detects it from the first 30 s.

**GPU worker (optional).** `deploy/setup-gpu.sh` builds `~/audio-cleaner/venv-gpu`
(ONNX Runtime with CUDA, pinned for the GTX 960's sm_52), then
`deploy/install.sh` enables `acw-worker-gpu`. It takes only local
transcription jobs and runs Parakeet on the GPU (one at a time: the model
needs ~3.7 of the 4 GB). While it is idle it refreshes
`data/gpu-worker-idle`, and CPU workers leave transcription jobs to it. Its
sandboxed runner may also open `/dev/nvidia*` and has no address-space limit
(CUDA reserves huge virtual ranges).

**Sized from the machine.** `deploy/install.sh` reads the core count and RAM:
all services together may use `ACW_MEMORY_PERCENT` (default 40) of RAM
(`acw.slice.d/size.conf`), file workers scale with cores, and live capacity
scales with cores and the start-up speed calibration. Moving to a bigger
machine needs no edits: run `deploy/install.sh` there.

**Live comes first.** Live audio gets whole physical cores (`acw.cpusplit`):
as many as its sessions need plus one spare, up to all but `ACW_MIN_JOB_CORES`
(default 1 per 8 cores) which file jobs always keep. File jobs are re-pinned
to whatever is left, every second, so they slow down instead of slowing live
audio (a job thread on a live core's other hyper-thread halves its speed: one
niced 6-thread job made 12 sessions lag 0.5 s before this split).

Capacity is in units (live denoise = 1, local captions
= 0.7): live cores x units per core, where units per core is calibrated at
start-up from how fast this CPU runs the model (2.68 on the 1700X; reference
2.4 at 0.110 cores per session; override `ACW_LIVE_UNITS_PER_CORE`), shrunk
by other load measured on the live cores. On this machine: 7 live cores x 2.68
= ~18.8 units. Benchmark with a 6-thread file job squeezed onto the last core,
20 ms frames, round trip inside the server:

| sessions | p50 | p99 |
|---|---|---|
| 16 standard | 9 ms | 13 ms |
| 18 standard | 7 ms | 13 ms |

When full, a new session is refused with the options that still fit;
sessions that only use the external (extra-accurate) captions are never
limited. On a bigger CPU all of this scales by itself (cores from the
topology, speed from the calibration).
