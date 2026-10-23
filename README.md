# Audio Cleaner

Speech denoising and transcription, self-hosted.

- **Free in your browser**: DeepFilterNet3 compiled to WebAssembly. Files never leave the device.
- **Studio clean** (pay as you go): echo removal for recordings with a second audio track, noise removal
  (DeepFilterNet3, or DPDFNet-8 for heavy noise, chosen per file), then a clarity chain (level match, high-pass, leveling, loudness, limiter). An optional "voice
  rebuilt" tier regenerates the voice instead. Video keeps its picture and its other audio tracks; subtitles are
  embedded as a soft track. A finished job keeps its upload, the noise removal before its strength is applied and its
  results for 2 hours, so strength, Leveling or the subtitles can be changed again by running only the
  steps that changed (the strength even live while listening, and free).
- **Live for streamers** (pay as you go): OBS plugin streams the mic to the
  server and gets back cleaned audio (+30 ms model latency) and live captions
  (CEA-608 into Twitch/YouTube).
- **Transcription**: free in the browser (Whisper base, about 80 MB, run on the visitor's device), or pay as you go:
  Parakeet and Whisper on the Audio Cleaner service (GPU when there is one), or extra accurate through a third-party
  speech-recognition service. Outputs SRT,
  VTT and TXT, or a video with them inside (made in the browser).
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
export ACW_DATA_DIR=~/acw-dev-data
native/live/build.sh ~/audio-cleaner/bin && ~/audio-cleaner/bin/acw-live & ~/venvs/acw-dev/bin/python -m acw.worker.main &
~/venvs/acw-dev/bin/uvicorn acw.app:app --host 127.0.0.1 --port 8791
~/venvs/acw-dev/bin/python -m acw.admin login-link you@example.com
```

### Browser tests (web/tests)

```bash
cd web && npm run test:fixtures   # once: the media files the suites use (E2E_FILES, default ~/audio-cleaner/dev/webtest-files)
npm run test:all                  # builds the site once, then every suite (not long.mjs) on its own fresh backend
npm run test:all -- free ui       # only these suites
```

`tests/run.sh` builds into a temp dist with `ACW_MODEL_HOST=http://127.0.0.1:8798/` (each suite serves the models from
`tests/mirror.mjs`, which reads the production `~/audio-cleaner/{wasm,models}` read-only), starts a throwaway backend with
`tests/backend.sh` (main app 18792, live app 18793, native live engine, worker, `tests/front.mjs` on 8792; own data dir
`/tmp/acw-webtest-<uid>`, empty base path, `ACW_LIVE_TRIAL_SECONDS=0`) and stops it after the suite. It exits non-zero if a
suite exits non-zero or prints a `FAIL` line; logs are in the state dir. `tests/backend.sh start|stop|env` runs a backend by
hand for a single `node tests/<suite>.mjs`. Overrides: `AC_HOME`, `ACW_BIN_DIR`, `ACW_MODELS_DIR`, `ACW_PYTHON`, `ACW_MODEL_SRC`,
`ACW_TEST_STATE`, `E2E_FILES`. `legal-versions` (no backend: the Terms/Privacy text and how its version is decided) is in the default list. `signedout` is the free tools without an account (the Cleanup and Live pages open signed out, the free runs send nothing about the file, a paid plan says "Sign in to start" and the file and choices come back after signing in). The editor screen (the Cleanup page once a file is chosen, and a job's page) has these suites: `studio-unit` (the four tracks
and their rules, the prices from the price list, the mapping to the server's timeline version 2, the waveform of what is heard, what the
choices mean for the files and what stops a start, the browser's share of a job; no backend), `editor-unit` (the timeline, the mix and the
browser's suggestions, no backend), `convert-unit` (the file-type list and the quote for a video, with the browser's encoders stood in for; no browser; the server tests compare its rules with the server's: `server/tests/test_web_agreement.py`), `suggest-agreement` (the browser's suggestions against `common/acw_common/suggest.py` on generated
files; runs `tests/suggest-py.py`, a few minutes), `editor` (the screen on a backend; `EDITOR_SHOTS=<dir>` saves screenshots), `parity` (one check for
each row of the owner's checklist of today's file pages, so nothing of the old form, the job page, Change and save again or Fix a moment is lost)
and `redo` (saving changes to a finished job, the strength slider, a job made through the API opening in the editor).

## Licences

Everything except `obs-plugin/` is proprietary, all rights reserved (see `LICENSE`); the
source is visible but not licensed for reuse. OBS plugin: GPL-2.0-or-later. Third-party:
DeepFilterNet (MIT/Apache-2.0), DPDFNet (Apache-2.0), Sidon and onnx-asr (MIT), WebRTC AEC3 via the
LiveKit SDK (BSD / Apache-2.0), tract (MIT/Apache-2.0), NVIDIA Parakeet and NeMo models (CC BY 4.0), FFmpeg (server: GPL build run as
a separate program, never distributed; desktop: LGPL build shipped as separate executables),
Qt/PySide6 (LGPLv3, dynamically linked, desktop only), faster-whisper (MIT), Whisper
weights (MIT). The website's /licenses page lists them with their texts. The desktop app shows all notices under Help → About.
