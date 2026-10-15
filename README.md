# Audio Cleaner

Speech denoising and transcription, self-hosted.

- **Free in your browser**: DeepFilterNet3 compiled to WebAssembly. Files never leave the device.
- **Studio clean** (pay as you go): echo removal for recordings with a second audio track, noise removal
  (DeepFilterNet3, or DPDFNet-8 for heavy noise, chosen per file), restoring the highs of phone-quality
  recordings, then a clarity chain (level match, high-pass, leveling, loudness, limiter). An optional "voice
  rebuilt" tier regenerates the voice instead. Video keeps its picture and its other audio tracks; subtitles are
  embedded as a soft track.
- **Live for streamers** (pay as you go): OBS plugin streams the mic to the
  server and gets back cleaned audio (+30 ms model latency) and live captions
  (CEA-608 into Twitch/YouTube).
- **Transcription** (pay as you go, plus a free daily allowance): Parakeet and Whisper on our own server
  (GPU when there is one), or extra accurate through OpenAI. Outputs SRT, VTT and TXT.
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
| `deploy/` | systemd user units, install / rollback / drain / publish scripts, VPS Caddy route |
| `docs/` | [architecture](docs/architecture.md), [security](docs/security.md), [payments](docs/payments.md), [operations](docs/operations.md), [API](docs/api.md), [streaming protocol](docs/streaming-protocol.md), [research roadmap](docs/research-roadmap.md) |

## Development

```bash
# a venv of your own: ~/audio-cleaner/venv is the production one, nothing is installed into it by hand
python3 -m venv ~/venvs/acw-dev
~/venvs/acw-dev/bin/pip install -r server/requirements.lock pytest pytest-asyncio
cd server && nice -n 10 ~/venvs/acw-dev/bin/python -m pytest -q
cd ../common && nice -n 10 ~/venvs/acw-dev/bin/python -m pytest -q

# run locally (needs ~/audio-cleaner/{bin,models}, see docs/operations.md)
export ACW_DATA_DIR=~/audio-cleaner/dev/data
~/venvs/acw-dev/bin/python -m acw.stream.worker & ~/venvs/acw-dev/bin/python -m acw.worker.main &
~/venvs/acw-dev/bin/uvicorn acw.app:app --host 127.0.0.1 --port 8791
~/venvs/acw-dev/bin/python -m acw.admin login-link you@example.com
```

## Licences

Everything except `obs-plugin/` is proprietary, all rights reserved (see `LICENSE`); the
source is visible but not licensed for reuse. OBS plugin: GPL-2.0-or-later. Third-party:
DeepFilterNet (MIT/Apache-2.0), DPDFNet (Apache-2.0), AP-BWE, Sidon and onnx-asr (MIT), WebRTC AEC3 via the
LiveKit SDK (BSD / Apache-2.0), tract (MIT/Apache-2.0), NVIDIA Parakeet and NeMo models (CC BY 4.0), FFmpeg (server: GPL build run as
a separate program, never distributed; desktop: LGPL build shipped as separate executables),
Qt/PySide6 (LGPLv3, dynamically linked, desktop only), faster-whisper (MIT), Whisper
weights (MIT). The website's /licenses page lists them with their texts. The desktop app shows all notices under Help → About.
