# Operations runbook (primary-ram)

## Layout

```
~/audio-cleaner/
  app/        deployed code (copy of server/acw, see app/VERSION)
  venv/       Python 3.12 virtualenv (server/requirements.lock)
  bin/        ffmpeg, ffprobe (BtbN static n8.1, sha256-verified), libdf.so
  models/     DeepFilterNet3_onnx.tar.gz, whisper/{large-v3-turbo,small}
  wasm/       df.js + df_bg.wasm for the browser build
  data/       acw.sqlite3, jobs/, stream.sock          (0700)
  secrets/    acw.env (web, secrets), acw-worker.env   (0700, files 0600)
  src/        build trees (DeepFilterNet, deepfilter-build) - safe to delete
  dev/        local dev instance (dev.sh) - safe to delete
```

## Deploy / update

```bash
cd ~/repos/Audio-Cleaner-Web && git pull
deploy/install.sh                 # deps, code copy, units, restart, health check
ACW_WORKERS=3 deploy/install.sh   # change the number of file-job workers
deploy/publish-web.sh             # build the site and rsync it to the VPS
```

## Services (user units, linger on)

```bash
systemctl --user status 'acw-*'
journalctl --user -u acw-web -f
systemctl --user restart acw-web
```

`acw-tunnel` keeps `VPS 127.0.0.1:18790 → primary-ram 127.0.0.1:8790` up;
check from the VPS with `curl -s 127.0.0.1:18790/audio/api/health`.

## Admin CLI

```bash
cd ~/audio-cleaner/app && set -a && . ../secrets/acw.env && set +a
../venv/bin/python -m acw.admin login-link you@example.com   # one-time sign-in URL
../venv/bin/python -m acw.admin grant someone@example.com 5 "support credit"
../venv/bin/python -m acw.admin user someone@example.com
../venv/bin/python -m acw.admin suspend someone@example.com "reason"
../venv/bin/python -m acw.admin stats
```

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
| Not reachable from the internet (`health_public`) | local health is fine but `https://mc-multi-world.space/audio/api/health` fails from this machine twice in a row | the problem is the tunnel, VPS or Caddy; skipped when the public origin is not https |
| File queue long (`queue_long`) | the head job has waited over 20 min, or the last job's estimated wait (`jobs.queue_wait_seconds`) is over 45 min | the mail shows busy cores for Audio Cleaner vs everything else (with the busiest other process names, e.g. `java`) and memory, so you can see when the Minecraft servers are the cause |
| Jobs abnormally slow (`jobs_slow`) | a running job, 10+ minutes in, is projected to take over 3x its `jobs.job_seconds` estimate | same host load breakdown |
| Disk low (`disk_low`) | under 15 GB free on the data volume | |
| Memory low (`mem_low`) | under 2 GB `MemAvailable` for 2 checks in a row | host load breakdown included |
| Swap growing (`swap_growing`) | at least 512 MB of swap in use and up 256 MB or more over 30 minutes | |
| Live sessions refused (`live_full`) | more than 10 stream sessions ended with `end_reason = 'full'` in the last hour (counted from `stream_sessions`; `ACW_ALERT_FULL_THRESHOLD` changes the 10) | |
| Backup overdue (`backup_stale`) | the newest snapshot in `/mnt/mc-storage/audio-cleaner-backups` (`ACW_BACKUP_DIR`) is older than 36 h, or none is found (disk not mounted) | |
| OpenAI quota / billing (`openai:quota`) | OpenAI answers `insufficient_quota` or a billing error | mail shows only HTTP status and OpenAI's error type/code |
| OpenAI key rejected (`openai:auth`) | HTTP 401/403 | |
| OpenAI server errors (`openai:5xx`) | 3 or more 5xx in 5 minutes (either premium path) | |

`openai:*` alerts resolve themselves after 30 minutes without a new error.
Thresholds are constants at the top of `server/acw/monitor.py`.

Limits: the monitor runs on this machine, so a power cut, network loss or a dead
`systemd --user` manager sends nothing. For that, point an outside monitor (for
example UptimeRobot, free) at `https://mc-multi-world.space/audio/api/health`.

## One-time setup still needed (needs you)

1. **VPS Caddy route** (sudo): `ssh vps` then `sudo bash ~/acw/install-audio-route.sh`.
   Adds `/audio` before the 404 catch-all, creates `/var/www/audio` owned by
   silver, validates (reverts on failure), and reloads.
2. **Sign-in providers**: create OAuth apps with these callbacks and put the
   ids/secrets in `~/audio-cleaner/secrets/acw.env`:
   - Google: `https://mc-multi-world.space/audio/api/auth/oauth/google/callback`
   (Email links need SMTP: any transactional provider.)
3. **Stripe** (test mode first): restricted key + webhook (see payments.md) into `acw.env`.
4. **OpenAI**: an `OPENAI_API_KEY` for premium transcription and captions.
   Preferably a dedicated project key with a monthly budget cap.
5. `systemctl --user restart acw-web` after editing `acw.env`.

## Rebuilding native pieces

- libDF (server + WASM): `native/deepfilter/build.sh` (see its README).
- ffmpeg: BtbN `ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz`; verify against
  their `checksums.sha256`; copy `ffmpeg`, `ffprobe` to `~/audio-cleaner/bin/`.
- Whisper models: `huggingface_hub.snapshot_download` of
  `mobiuslabsgmbh/faster-whisper-large-v3-turbo` and `Systran/faster-whisper-small`
  into `~/audio-cleaner/models/whisper/<name>`.

## Capacity (Ryzen 7 1700X, 8C/16T, 31 GB, shared with Minecraft)

More users never slow a running job down; they wait longer for a slot or are
told the service is busy. Measured numbers, 2026-10-14:

**Memory.** Every `acw-*` unit runs in `acw.slice`, capped at 40% of RAM
(12 GB on this 31 GB box; set at install), so Audio Cleaner can't take RAM the
Minecraft servers need.
- File jobs keep decoded and cleaned audio in memory-mapped scratch files in the
  job directory, and the polish chain works in 10 s blocks. A 60-min stereo job
  runs under a hard 1.5 GB cap (it peaked at 6.9 GB for 30 min before this).
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
- Studio clean: ~0.1x real time with 4 threads (60 min stereo in ~6 min).
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
