# Architecture (on-prem)

```
 Browser / OBS plugin
        │ HTTPS / WSS
        ▼
 VPS (Caddy, audiocleaner.live)
   /*              → static site from /var/www/audio (incl. free WASM denoiser)
   /api/*          → 127.0.0.1:18790 ──────────────┐   request bodies capped (64 KB; 4 GB only for the upload route)
   job uploads and result downloads                 │
                   → 127.0.0.1:18794 (falls back    │ two SSH reverse tunnels, separate TCP connections
                     to 18790 if down) ───────────┐ │ (key restricted to these two ports)
   (mc-multi-world.space: packs, pet-companion,    │ │
    /audio/api/* for old clients: same upstream)  │ │
 primary-ram (home LAN, nothing new listens on it)▼ ▼
   acw-tunnel-bulk ssh -R 18794 → 127.0.0.1:8794   (IPQoS throughput)
   acw-bulk     FastAPI/uvicorn on 127.0.0.1:8794 — only PUT /api/jobs/{id}/input and
                GET /api/jobs/{id}/outputs/{name}; same auth, limits and caps as acw-web,
                total download cap; no background loops (keeps big transfers off the live path)
   acw-tunnel   ssh -R 18790 → 127.0.0.1:8790     (IPQoS lowdelay)
   acw-web      FastAPI/uvicorn on 127.0.0.1:8790 — auth, billing, jobs API (also serves the
                two bulk routes), Stripe/OpenAI calls, job callbacks, housekeeping (holds,
                retention), premium file transcription
   acw-tunnel-live ssh -R 18795 → 127.0.0.1:8795  (IPQoS lowdelay)
   acw-live     FastAPI/uvicorn on 127.0.0.1:8795 — the live WebSocket only (relay to acw-stream,
                extra-accurate live captions, billing per second, closing orphaned live sessions)
   acw-admin    the owner dashboard on 127.0.0.1:8796 (read-only; over SSH only)
   acw-stream   live DSP workers (one process per pair of live cores): echo canceller → DeepFilterNet3 → captions; Unix sockets,
                no network (seccomp AF_UNIX + Landlock)
   acw-worker@N file-job workers (no network); each job stage runs in a sandboxed runner
                (Landlock + seccomp: no sockets, no links) that sees only its job directory
   acw-worker-gpu  optional: takes the jobs with GPU stages (Parakeet, heavy noise removal, voice rebuild, band extension) and runs them on the GPU, one at a time
   acw-monitor  every 2 minutes: health, queue, money and backup checks → email; writes data/metrics.prom
   SQLite       ~/audio-cleaner/data/acw.sqlite3 (WAL, synchronous=FULL), nightly checked backup
```

## Products → components

| Product | Where it runs | Cost to us |
|---|---|---|
| Free browser denoise | user's browser (DFN3 WASM, Web Worker) | none of ours for the model: the wasm and the model (about 22 MB) come from Hugging Face; df.js (15 kB) from the VPS |
| Studio clean (paid) | acw-worker → runner, see "File jobs" below | CPU on primary-ram |
| Studio clean, voice rebuilt (paid, own price) | same, with Sidon instead of noise removal; files up to 90 minutes, one such job at a time | about 1x real time per channel, ~3 GB RAM |
| Desktop app (free) | user's computer: same `acw_common` pipeline as Studio clean, bundled libDF + LGPL FFmpeg | GitHub Releases bandwidth (free; the update check reads the newest release's update.json there); models from Hugging Face |
| Live (paid) | OBS plugin / browser ⇄ acw-live ⇄ acw-stream (one worker process per pair of live cores); 20 ms frames, 30 ms model latency | CPU; ~0.2 core per stream at full load |
| Transcription standard (paid + free quota) | runner: NVIDIA Parakeet TDT (v2 English, v3 for 24 European languages) via onnx-asr, on the GPU worker when it is idle; Whisper large-v3-turbo for other languages; Whisper small detects the language | CPU (Parakeet ~0.09x real time on 4 threads, ~0.04x on the GPU) |
| Transcription premium (paid) | runner cuts speech into ≤15 s VAD segments → acw-web sends them to OpenAI `gpt-transcribe` (text only; segment boundaries give subtitle timing); finished segments are kept, so a restart or outage does not pay twice | OpenAI per-minute fee |
| Live captions standard / premium | acw-stream: NeMo FastConformer transducer via sherpa-onnx (English) / acw-live ⇄ OpenAI Realtime (`gpt-live-transcribe`). Both listen to the echo-free original, not the denoised audio | CPU / OpenAI fee |

## File jobs: the order of the stages

`worker/runner.py` and `acw_common/pipeline.py` (the desktop app runs the same functions in the same order):

1. **probe**: container, streams, header duration, and the length that really decodes (billing uses the longer).
2. **echo removal** (only with `echo_reference_track` and a second audio track): WebRTC AEC3 (via the LiveKit
   SDK) cancels the speaker sound recorded on track 2 from the mic on track 1. The result is shifted back by the
   canceller's 9 ms so it lines up with the video and the subtitles.
3. **transcription**, from the echo-free original (recognisers score worse on denoised speech): local models in
   the runner, or speech segments for OpenAI (premium).
4. **noise removal** on the echo-free audio, per channel: DeepFilterNet3 (light noise) or DPDFNet-8 (heavy noise),
   chosen per file from the measured speech-to-background gap, or forced by the `noise` option. With the
   voice-rebuild tier, Sidon replaces step 4.
5. **clarity chain**: level match, high-pass, compressor, loudness, limiter.
6. **write**: audio file, or video with the cleaned sound as the first track (the file's other audio tracks and its
   own subtitles are copied across where the container can hold them) and soft subtitles.

**Automatic settings** (`options.automatic`, `server/acw/automatic.py`, docs/api.md): such a job is created as `standard`, held for the
dearest way it can end (`automatic.candidates`: Denoising + Leveling, or Voice rebuild + Leveling: the rebuild where it is possible) and queued. After the probe
the worker runs a sandboxed `analyse` stage (decode, then `acw_common.analysis.measure`: the browser's measurements and the noise
figure over about a minute of the file) and decides outside the sandbox with the same function the desktop app uses
(`acw_common.reco_model.automatic`): strength, noise remover, Voice rebuild, Leveling, with what the request gave itself
kept. `automatic.apply` then makes the job the decided one in one transaction (tier, options, pinned rates, estimate, the hold shrunk to
the estimate; `params.chosen` is what the API reports). A rebuild decided by a worker that is not the card's is handed over
(`jobs.hand_over`: back in the queue where it was, `next_stage = main`, taken by the card's worker alone); where a rebuild is wanted but
impossible the job continues as Denoising and says so. From then on it is an ordinary job.

A worker that is stopped (restart, deploy) puts its job back in the queue with the upload kept; nothing fails.
The estimates shown to people and the slow-job monitor follow the work a job does (`jobs.SPEED`).

## Changing a finished job

Every job is made one way: with a timeline (next section). A job asked for with whole-file options (the API) gets the timeline they stand
for when it is created (`editing.timeline_from_options`: each sound step on the whole file, the subtitles as one part over the whole file),
made again for the file's real length once it is read (`editing.refit_whole_file`); an automatic job gets it once its settings are decided
(`automatic.apply`). The first run renders it from an empty block cache (`editrun.render`), and a finished job is changed by editing it:
only the blocks that are missing are made and charged.

Strength is a blend: the noise models always run at full strength, and the result is `(1 - a) * full + a * input` with
`a = 10^(-dB/20)` (`pipeline.strength_mix`; 100 dB or more: no blend). That is what libDF does itself in the STFT domain after its
post filter and what DPDFNet does in the time domain (`common/tests/test_blend.py` checks it against the models' own attenuation limit).
So the block cache holds the noise removal at full strength, and a new strength needs no model. The free noise removal in the browser
keeps its two signals in memory and its slider plays them (`web/src/media/blend.js`, `memorySource` and `BlendPlayer`).

## The per-part editor

A job with audio steps holds a **timeline** (`acw_common/timeline.py`, JSON v1: layers echo / noise / rebuild / leveling, each with parts and
settings; Voice rebuild wins where it overlaps noise removal; subtitles stay whole-file options) and a **block cache**. Every model step runs on a
fixed block grid (`acw_common/blocks.py`: 30 s for the noise removers and echo removal with the context each needs, 14 s windows for Voice
rebuild) and a block is keyed by the step, the model, the audio it read and where it is, never by the strength: the server stores them as 24-bit
FLAC in the job's `blocks/` folder (`blocks.FlacCache`; a block read back is within one 24-bit step). `acw/editing.py` is the server's side:

- **Planning without audio.** The web process never decodes an upload. What planning needs about the recording (its length, channels, which
  channels are cleaned, the stretches the noise removers run on, whether there is a second track: the *facts*) is measured by the sandboxed
  runner and kept in the job's record; `edit_store.inputs_for` stands them in for the samples of `editor.Inputs`, so `editor.plan` (the cache misses),
  `editor.billable` and the price are computed in the web process, and a stretch of the cache is cut by confined helpers (below).
- **Runs.** A job's first run renders its timeline (whole-file choices are the timeline they stand for) through `editor.render` from an
  empty cache, so only its parts are made and charged; it fills the cache, writes the recording as `kept-source.flac`, reports the facts and
  looks for suggestions (`acw_common/suggest.py`). An **edit run** (`process`, `export`, `suggest`) puts the finished job back in the queue
  (`next_stage` `edit`, `params.run`; the job is `queued` / `running` again with a new hold): the runner's `edit` stage (`worker/editrun.py`) decodes the
  recording, decides the removers still on "auto" (`editor.resolve_removers`), plans against the cache, makes the missing blocks and renders the
  timeline (joins, Leveling per part) into the result audio. It reports what it ran (`billed`: whole seconds per step), which the worker prices
  with the rates pinned in the run and the fixed amounts of `editor.STEP_COSTS`, never above the hold. A run that ends without its result
  (canceled, failed, too late) restores the job as it was (`jobs.finish_failed`), files and cache kept, money released.
  Voice rebuild and the heavy remover go to the card's worker (`broker.gpu_stages` reads the run's own list).
- **Price.** For each step that runs: rate per second x (the whole seconds asked for in the blocks that are made, rounded up + the step's fixed
  per-run cost as equivalent seconds); context is never billed, no minimum, cache hits free, Leveling where it runs on changed audio
  (`editor.plan(previous=...)`, the timeline the result files hold). The rates are the price list's (echo removal's own: 2,600 micros a minute).
- **Preview.** `GET /api/jobs/{id}/preview/{source|echo|noise|rebuild}` hands out sample-aligned stretches (`worker/cutter.py` for the
  recording, `worker/assembler.py` for what the blocks hold: each block is decoded, put at its place and the stretch written as one FLAC, in a
  confined process like the cutter), and says what is not cached; the page mixes strengths and layers itself.

## Live: the order

The live engine (`native/live`, `bin/acw-live`): echo canceller (when the client sends the speaker reference) → DeepFilterNet3 + high-pass +
limiter → back to the client. Standard captions and the extra-accurate tap both take the signal after echo
removal and before noise removal.

## Shared code

`common/acw_common` holds everything both the server and the desktop app run:
the libDF ctypes engine (`dfn`), DPDFNet (`dpdfnet`), Sidon (`restore`), the spectrum measurements (`spectrum`), the echo canceller
(`aec`), the clarity chain (`clarity`), transcription and captions (`transcribe`, `captions`), the FFmpeg
wrappers (`ffmpeg`) and the Studio-clean chain built from them (`pipeline`).
The server's runner and live engine import it; `deploy/install.sh` copies it
next to `acw` in `~/audio-cleaner/app`, and the sandboxed runner gets both on
its `PYTHONPATH` and in its Landlock read set. A change to the chain therefore
ships to the website and the next desktop release alike. Native builds are
shared too: `native/deepfilter/build.sh` produces the server `.so`, the desktop
`.dylib`/`.dll` and the browser `.wasm` from one pinned libDF.

## Data flow and retention

- Uploads land in `data/jobs/<id>/input.*` (0600) and stay there with the job's other files: its results, and the per-part editor's
  block cache, recording copy, suggestions and subtitle parts (section "The per-part editor"). They are deleted together 2 hours after the
  job's last edit (a run finishing, a timeline save, a quote, a process, an export: `ACW_JOB_RETENTION_MINUTES`) and never later than 6
  hours after the upload arrived (`ACW_JOB_MAX_RETENTION_MINUTES`; `jobs.expiry`, `jobs.sweep` every 30 to 60 seconds), or at once with
  "Delete now"; the files of a failed or canceled job go at once.
- Live audio is never written to disk. Captions are not stored.
- The browser tier sends nothing to the server.
- The database keeps accounts, ledger, payments, job metadata and an audit log. File names are removed
  after 12 months, IP addresses (sessions, live sessions, audit log) after 90 days, job and live-session records,
  closed holds and routine audit rows are deleted after 12 months, expired sign-in data after 30 days, processed Stripe
  events after 90 days (`acw/retention.py`); ledger and payment rows are kept 7 years (then `acw.admin compact-money`, by
  hand); nightly backups are kept 30 days. The file is compacted monthly. The full table of what grows and what bounds
  it: [operations.md](operations.md#what-grows-and-what-bounds-it).
- Contact-form messages are mailed to the owner and not stored (`api/contact_routes.py`).

## Money

Prepaid credit in micro-dollars. Holds reserve the estimated cost, captures
charge the real cost. See [payments.md](payments.md). A job whose paid stage did not run (for example the voice
rebuild) is billed as the tier it got and says so.

## Observability

One JSON log line per finished job and live session (`job_done`, `job_failed`, `live_session_end`), `python -m acw.admin stats
[--days N]` for a plain-text report from the database, and a Prometheus text file `data/metrics.prom` rewritten by the
monitor (no HTTP endpoint). Details and field lists: docs/operations.md, "Statistics and metrics".

## Reproducible installation

`deploy/models.lock` lists every model and program file (source, pinned revision, size, sha256) and
`deploy/fetch-models.sh` downloads and verifies them; FFmpeg is a pinned BtbN build. `server/requirements.lock` and
`deploy/requirements-gpu.lock` pin every Python package with the sha256 of every file PyPI lists
(`pip install --require-hashes`). Only `libdf.so`, the browser build and a few optional conversions are built or made
here (marked `local` in the lock).

## Why these choices

- **libDF (Rust/tract) instead of PyTorch DeepFilterNet**: stateful 10 ms
  frames (needed for live), no torch dependency, identical model to the WASM
  build, ~0.9 ms per frame per core.
- **SQLite**: single host, modest write volume; `BEGIN IMMEDIATE` makes the
  ledger trivially serializable; `synchronous=FULL` so a payment acknowledged to Stripe survives a power cut.
  Move to Postgres only if we ever run on more than one machine.
- **Static site on the VPS**: the free tier and the marketing pages stay up
  when primary-ram or the home link is down, and cost no home bandwidth.
- **Tunnel opened from primary-ram**: no LAN port to open, no firewall change;
  the VPS can reach exactly one loopback port on primary-ram.
