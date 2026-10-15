# Live streaming protocol v1

One WebSocket per live source: `wss://<host><base>/api/stream/ws`.

Authenticate with `Authorization: Bearer acw_...` (API key with the `stream`
scope). Browsers on the site itself can use the session cookie instead.

A handshake the server refuses is answered with an HTTP status, before the upgrade (a client library that only
sees "403" is running against a server that cannot send a denial response):

| Status | Meaning |
|---|---|
| 401 | no credentials, or the API key is wrong or revoked, or the browser session is gone |
| 403 | the account is suspended, or the key lacks the `stream` scope |
| 429 | too many connection attempts, or too many live sessions started this hour (60 per account, 90 per network; failed starts count). `Retry-After` says how many seconds to wait |

## 1. Start

Client sends one text message within 15 s:

```json
{"type": "start", "version": 1,
 "denoise": "light" | null,
 "captions": "local" | "premium" | null,
 "language": "en" | null,
 "sampleRate": 48000,
 "codec": "opus" | "pcm_s16le",
 "opusBitrate": 96000,
 "attenLimDb": 100,
 "echoReference": true,
 "client": "obs-audio-cleaner/0.1.0"}
```

`echoReference` is optional (default false); see section 5.

At least one of `denoise` / `captions` must be set. Live processing on our
CPUs has a measured capacity; when the requested setup doesn't fit right now,
the server refuses it and lists what would still work:

```json
{"type": "error", "code": "full", "available": [{"captions": "premium"}],
 "message": "Live noise removal is full right now. Still available: extra-accurate captions. Try again in a few minutes."}
```

Clients should retry no sooner than a minute later (`retryAfterSeconds`, see the error list below).
`captions: "premium"` without denoise is never refused for capacity. Otherwise the server answers:

```json
{"type": "ready", "sessionId": "...", "sampleRate": 48000, "hopSamples": 480,
 "modelLatencyMs": 30.0, "usdPerMinute": 0.01, "maxSessionSeconds": 43200,
 "echoReference": true}
```

`echoReference` is only present (true) when the start message asked for it and
the server will use it. `modelLatencyMs` includes the delay of echo
cancellation (9 ms) in that case.

or `{"type":"error","code":...,"message":...}` and closes. Every error event may carry **`retryAfterSeconds`**
(an integer): do not try again sooner. Codes:

| Code | Meaning |
|---|---|
| `bad_start` | no valid `start` message within 15 s |
| `bad_config` | an unusable setting: not 48 kHz, a language code that is not 2-3 letters, a noise strength that is not a number, or `captions: "local"` with a language other than English (**Standard captions are English only**; extra-accurate captions work in other languages) |
| `bad_tier` | an option we do not offer |
| `nothing_enabled` | neither `denoise` nor `captions` is set |
| `insufficient_funds` | the credit set aside up front (5 minutes) is more than the available balance |
| `tos` | the current Terms have not been accepted |
| `too_many_sessions` | two live sessions per account (six per network) are already running. A session of yours that has been silent for 10 s is taken to be dropped and is replaced instead. `retryAfterSeconds` 30 |
| `full` | live capacity is used up right now; `available` lists what still fits. `retryAfterSeconds` 60 |
| `restarting` | the service is being updated: new sessions are refused for a minute or so. `retryAfterSeconds` 60 |
| `unavailable` | live processing is busy or offline (or extra-accurate captions are not available). `retryAfterSeconds` 30 (300 for captions) |
| `bad_frame` | (during the session) a binary frame the server could not read: wrong header, size or codec. The session ends |
| `too_fast` | (during the session) audio arrived faster than real time (bursts up to 6 s are tolerated). The session ends |
| `worker_lost` | (during the session) the processing side stopped unexpectedly. The session ends; start again |

## 2. Audio frames (binary, both directions)

```
offset size  field
0      1     version = 1
1      1     codec: 1 = pcm_s16le, 2 = f32le, 3 = opus
2      2     flags: bit 0 = echo reference (section 5), other bits 0 (reserved)
4      4     seq: u32 frame counter (little endian)
8      8     ts_us: u64 client timestamp (opaque, echoed back)
16     ...   mono 48 kHz samples, or exactly one Opus packet (codec 3)
```

- **Opus (recommended):** each frame is one 20 ms Opus packet (48 kHz mono,
  any bitrate; 48-64 kbps is plenty for the uplink). The server answers with
  one Opus packet per frame (`opusBitrate`, default 96 kbps, 24-160 kbps).
  That is ~60 kbps instead of ~770 kbps of PCM in each direction, which
  matters for both the streamer's upload and ours. Opus adds ~6.5 ms codec
  delay each way.

- Frames must hold 10-100 ms of audio in 10 ms steps (480 × k samples). 20 ms
  (960 samples) is recommended.
- The server returns exactly one frame per input frame, same `seq`/`ts_us`
  and the same number of samples, in the same codec. Processed audio is
  delayed by `modelLatencyMs` (30 ms for `light`) inside the stream, so the
  client's total delay = network round trip + modelLatencyMs. Measure the
  round trip with `ts_us`.
- When denoise is off (captions only) the audio comes back unchanged and
  the client may simply ignore it.
- Audio must arrive at about real time (bursts up to 6 s are tolerated);
  faster input ends the session with `too_fast`.

## 3. Events (text, server → client)

```json
{"type": "caption", "text": "...", "final": true, "start": 12.34, "end": 15.02}
{"type": "caption", "text": "partial words", "final": false, "id": "item_..."}   // premium only
{"type": "warning", "code": "low_balance", "secondsLeft": 180, "availableMicros": 0}
{"type": "warning", "code": "captions_unavailable", "message": "..."}
{"type": "stopped", "reason": "insufficient_funds" | "max_duration" | "idle_timeout" | "account_closed"}
{"type": "error", "code": "...", "message": "..."}
{"type": "pong", "t": ..., "seconds": 123.4}
```

`stopped` is the last thing the server sends before it closes the socket: `insufficient_funds` (the balance cannot
cover the next minute), `max_duration` (12 hours), `idle_timeout` (**no microphone audio for 2 minutes**: pings
and echo-reference frames do not count; a muted microphone must keep sending frames, silent ones are fine, as
the OBS plugin and the website do) and `account_closed` (the account was suspended or deleted, the API key
was revoked, or the browser session ended; checked every 30 s). A session replaced by a newer one of the same
account (see `too_many_sessions`) is closed with WebSocket code 1001 and no event.

`start`/`end` are seconds of audio since the session started. Local captions
arrive once per utterance (~1-3 s after the speaker pauses); premium captions
also stream partial text.

## 4. Control (text, client → server)

```json
{"type": "ping", "t": 123}
{"type": "config", "attenLimDb": 30}
{"type": "stop"}
```

The server closes the socket after 30 s without any message (see `idle_timeout` above for what keeps a
session alive beyond that). On `stop` it
flushes outstanding captions, then closes. Billing covers processed audio
seconds only (rounded up to the second at the end of the session).

## 5. Echo cancellation (optional)

When the user plays sound through speakers, the microphone picks it up again.
A client that knows what the speakers play (the "reference": the desktop audio
in OBS) can send it along, and the server removes that sound from the
microphone before noise removal and before captions hear the audio. It is
included in the price of live processing. Clients that never send a reference
behave exactly as described above.

- Put `"echoReference": true` in `start`. Only send reference frames when the
  `ready` message contains `"echoReference": true`; an older server ignores the
  field and would treat reference frames as microphone audio.
- A reference frame is an ordinary audio frame (section 2) with flags bit 0
  set. It uses the same codec as the microphone frames (an Opus client encodes
  the reference with its own Opus encoder: the stream is separate) and the same
  `seq` as the microphone frame it belongs to, and is sent before it. Both
  cover the same moment of the same 48 kHz timeline, 20 ms each; a constant
  offset of tens of milliseconds is fine, the server finds the delay itself
  (the reference must not come later than the echo it causes).
- The server answers microphone frames only: no reply to reference frames, and
  they are neither counted as processed audio nor billed. They are paced like
  microphone frames (real time).
- A reference frame that is missing or comes after its microphone frame counts
  as silence for that frame, so a client may skip frames that are all zeros.
  The server keeps 1 s of reference at most.
- Cancellation adapts within a few seconds of the first reference audio.
- Browsers: use the browser's own echo cancellation instead
  (`getUserMedia({audio: {echoCancellation: true}})`); no reference is needed.

## Client requirements (OBS plugin)

- Never block the audio thread: ring buffers in and out, network on a worker.
- Fixed output delay D chosen at start (e.g. RTT + 30 ms + jitter margin,
  presets 80/150/250 ms), never changed mid-session; back-date output
  timestamps by D. If a processed frame misses its deadline, output the
  delayed original audio for that frame (crossfade ≤10 ms).
- Reconnect with exponential backoff + jitter; pass audio through while
  disconnected.
- Show status: connected, RTT, D, late/dropped frames, balance warnings.
- Echo cancellation: capture the chosen source (default Desktop Audio) with an
  audio capture callback, downmix to mono, resample to 48 kHz and keep it in a
  lock-free buffer; the audio thread cuts one reference frame per microphone
  frame from it. Skip silent frames. Budget 9 ms more for D.
- Captions: forward `final` captions to `obs_output_output_caption_text2`
  on the streaming output, paced (≤ 32 chars per line, don't queue a
  backlog), or to a text source.
