# Architecture (on-prem)

```
 Browser / OBS plugin
        │ HTTPS / WSS
        ▼
 VPS 216.250.115.225 (Caddy, mc-multi-world.space)
   /audio/*        → static site from /var/www/audio (incl. free WASM denoiser)
   /audio/api/*    → 127.0.0.1:18790 ──────────────┐
   (other paths: Minecraft packs, pet-companion)    │ SSH reverse tunnel
                                                    │ (key restricted to this one port)
 primary-ram (home LAN, nothing new listens on it)  ▼
   acw-tunnel   ssh -R 18790 → 127.0.0.1:8790
   acw-web      FastAPI/uvicorn on 127.0.0.1:8790 — auth, billing, jobs API,
                live WebSocket, Stripe/OpenAI calls (only service with internet)
   acw-stream   live DSP worker (DeepFilterNet3 + local captions), Unix socket,
                no network (seccomp AF_UNIX + Landlock)
   acw-worker@N file-job workers (no network); each job runs in a Landlocked
                runner that can only see its own job directory
   SQLite       ~/audio-cleaner/data/acw.sqlite3 (WAL)
```

## Products → components

| Product | Where it runs | Cost to us |
|---|---|---|
| Free browser denoise | user's browser (DFN3 WASM, Web Worker) | VPS egress for ~8 MB of assets per new visitor (wasm 3.9 MB gz + model 8 MB) |
| Studio clean (paid) | acw-worker → runner: decode → DFN3 (threaded segments) → clarity chain (level match, HPF, compressor, loudness, limiter) → encode / remux | CPU on primary-ram |
| Desktop app (free) | user's computer: same `acw_common` pipeline as Studio clean, bundled libDF + LGPL FFmpeg | GitHub Releases bandwidth (free); update check hits a static JSON on the VPS |
| Live (paid) | OBS plugin ⇄ acw-web ⇄ acw-stream; 20 ms frames, 30 ms model latency | CPU; ~0.1 core per stream |
| Transcription standard (paid + free quota) | runner: faster-whisper large-v3-turbo int8 | CPU (RTF ≈ 0.45 at 8 threads on the 1700X) |
| Transcription premium (paid) | runner makes ≤10 min mp3 chunks → acw-web sends to OpenAI `gpt-4o-transcribe-diarize` | OpenAI per-minute fee |
| Live captions standard / premium | acw-stream utterance captioner (whisper small) / acw-web ⇄ OpenAI Realtime (`gpt-live-transcribe`) | CPU / OpenAI fee |

## Shared code

`common/acw_common` holds everything both the server and the desktop app run:
the libDF ctypes engine (`dfn`), the clarity chain (`clarity`), the FFmpeg
wrappers (`ffmpeg`) and the Studio-clean chain built from them (`pipeline`).
The server's runner and live engine import it; `deploy/install.sh` copies it
next to `acw` in `~/audio-cleaner/app`, and the sandboxed runner gets both on
its `PYTHONPATH` and in its Landlock read set. A change to the chain therefore
ships to the website and the next desktop release alike. Native builds are
shared too: `native/deepfilter/build.sh` produces the server `.so`, the desktop
`.dylib`/`.dll` and the browser `.wasm` from one pinned libDF.

## Data flow and retention

- Uploads land in `data/jobs/<id>/input.*` (0600), are deleted as soon as the job
  finishes or fails; outputs are deleted after `ACW_JOB_OUTPUT_TTL_HOURS` (24 h).
- Live audio is never written to disk. Captions are not stored.
- The browser tier sends nothing to the server.
- The database keeps accounts, ledger, payments, job metadata (file name,
  duration, cost) and an audit log.

## Money

Prepaid credit in micro-dollars. Holds reserve the estimated cost, captures
charge the real cost. See [payments.md](payments.md).

## Why these choices

- **libDF (Rust/tract) instead of PyTorch DeepFilterNet**: stateful 10 ms
  frames (needed for live), no torch dependency, identical model to the WASM
  build, ~0.9 ms per frame per core.
- **SQLite**: single host, modest write volume; `BEGIN IMMEDIATE` makes the
  ledger trivially serializable. Move to Postgres only if we ever run on more
  than one machine.
- **Static site on the VPS**: the free tier and the marketing pages stay up
  when primary-ram or the home link is down, and cost no home bandwidth.
- **Tunnel opened from primary-ram**: no LAN port to open, no firewall change;
  the VPS can reach exactly one loopback port on primary-ram.
