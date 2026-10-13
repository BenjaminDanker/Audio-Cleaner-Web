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

## Capacity (Ryzen 7 1700X, 8C/16T, shared with Minecraft)

- DFN3: ~0.9 ms per 10 ms frame per core, so one core handles ~10 live
  streams. `ACW_STREAM_MAX_SESSIONS=24` by default.
- Studio clean: about 0.05× real time with 4 threads (60 min of audio in ~3 min).
- Local transcription: large-v3-turbo int8 runs at ~0.45× real time on
  8 threads (60 min of audio in ~27 min). This is the bottleneck; see
  research-roadmap.md.
