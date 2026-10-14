# Audio Cleaner

Speech denoising and transcription, self-hosted.

- **Free in your browser**: DeepFilterNet3 compiled to WebAssembly. Files never leave the device.
- **Studio clean** (pay as you go): server-side DeepFilterNet3 plus a clarity
  chain (level match, high-pass, leveling, loudness, limiter). Video keeps its
  picture; subtitles are embedded as a soft track.
- **Live for streamers** (pay as you go): OBS plugin streams the mic to the
  server and gets back cleaned audio (+30 ms model latency) and live captions
  (CEA-608 into Twitch/YouTube).
- **Transcription** (pay as you go, plus a free daily allowance): on-prem
  Whisper or premium OpenAI. Outputs SRT, VTT and TXT.
- **Desktop app** (free, Windows/macOS): the Studio clean chain running locally
  on video and audio files, batch queue, preview with original/cleaned toggle.

Prepaid credit, no subscription: top up from $10, optional auto-reload.

## Layout

| Path | What |
|---|---|
| `common/` | `acw_common`: processing shared by server and desktop (libDF engine, clarity chain, FFmpeg helpers, Studio-clean pipeline) |
| `server/` | Python service (`acw` package): API, billing, job workers, live DSP |
| `desktop/` | desktop app (PySide6) and its packaging/installers ([docs](docs/desktop.md)) |
| `web/` | Website + browser denoiser (Vite/React) |
| `obs-plugin/` | OBS Studio filter plugin (GPL-2.0-or-later) |
| `clients/python/` | reference live-stream client |
| `native/deepfilter/` | reproducible libDF build (`.so`/`.dylib`/`.dll` + browser `.wasm`) |
| `deploy/` | systemd user units, install/publish scripts, VPS Caddy route |
| `docs/` | [architecture](docs/architecture.md), [security](docs/security.md), [payments](docs/payments.md), [operations](docs/operations.md), [API](docs/api.md), [streaming protocol](docs/streaming-protocol.md), [research roadmap](docs/research-roadmap.md) |

## Development

```bash
python3 -m venv ~/audio-cleaner/venv
~/audio-cleaner/venv/bin/pip install -r server/requirements.lock pytest pytest-asyncio
cd server && ~/audio-cleaner/venv/bin/python -m pytest -q
cd ../common && ~/audio-cleaner/venv/bin/python -m pytest -q

# run locally (needs ~/audio-cleaner/{bin,models}, see docs/operations.md)
export ACW_DATA_DIR=~/audio-cleaner/dev/data
python -m acw.stream.worker & python -m acw.worker.main &
uvicorn acw.app:app --host 127.0.0.1 --port 8791
python -m acw.admin login-link you@example.com
```

## Licences

Everything except `obs-plugin/` is proprietary, all rights reserved (see `LICENSE`); the
source is visible but not licensed for reuse. OBS plugin: GPL-2.0-or-later. Third-party:
DeepFilterNet (MIT/Apache-2.0), tract (MIT/Apache-2.0), FFmpeg (server: GPL build run as
a separate program, never distributed; desktop: LGPL build shipped as separate executables),
Qt/PySide6 (LGPLv3, dynamically linked, desktop only), faster-whisper (MIT), Whisper
weights (MIT). The desktop app shows all notices under Help → About.
