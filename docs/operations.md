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

**Memory.** Every `acw-*` unit runs in `acw.slice` (MemoryHigh 11G, MemoryMax
12G), so Audio Cleaner can't take RAM the Minecraft servers need.
- File jobs keep decoded and cleaned audio in memory-mapped scratch files in the
  job directory, and the polish chain works in 10 s blocks. A 60-min stereo job
  runs under a hard 1.5 GB cap (it peaked at 6.9 GB for 30 min before this).
  Scratch disk: ~0.7 GB per hour and channel, twice.
- Local transcription adds the Whisper model (~1.7 GB) and 16 kHz audio
  (~0.23 GB per hour). Worker units: MemoryHigh 4G / MemoryMax 5G.
- Live: one shared caption model (~0.5 GB) plus a few MB per session. Unit: 3G/4G.

**File jobs (Studio clean, transcription).** `ACW_WORKERS` workers (default 2),
one job each, 4 threads each, `Nice=10`. Queue order: paid before free, then
the user with the fewest jobs running, then first come first served
(`jobs.QUEUE_ORDER`). The API reports `queuePosition`. Limits: 3 active jobs
per user; free jobs are refused while 6 or more jobs wait.
- Studio clean: ~0.1x real time with 4 threads (60 min stereo in ~6 min).
- Local transcription (large-v3-turbo int8): ~0.45x real time on 8 threads.
  This is the bottleneck. Parakeet would be ~4.5x faster (research/asr.md).

**Live.** Sessions are admitted by measured CPU cost against
`ACW_LIVE_CPU_BUDGET` (default 7 cores): light denoise 0.13, low-latency 0.32,
local captions +0.35 (and at most `ACW_LIVE_MAX_LOCAL_CAPTIONS`, default 6).
`ACW_STREAM_MAX_SESSIONS` (24) is a hard cap on top. When full, a new session
gets "Live is full right now" and running sessions are untouched. Typical
mixes: ~45 light, ~20 low-latency, or 6 with local captions plus ~25 light.
Local captions drop an utterance rather than fall behind when the shared model
is busy.
