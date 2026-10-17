# Live streaming protocol v1

The website shows this page at `/developers/live` (`web/src/pages/DevelopersLive.jsx`), written by hand from this file in
friendlier words: **change both** when the protocol changes (see the note at the top of [api.md](api.md)).

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
 "language": "en" | ["en", "es"] | null,
 "translate": ["es", "de"],
 "sampleRate": 48000,
 "codec": "opus" | "pcm_s16le",
 "opusBitrate": 96000,
 "attenLimDb": 100,
 "echoReference": true,
 "client": "obs-audio-cleaner/0.1.0"}
```

`echoReference` is optional (default false); see section 5. `translate` is optional (default none); see "Translations".

`language` is the spoken language, or for someone who switches between languages a list of up to three: extra-accurate
captions then follow the switches, and translations translate whatever was said (a translation language may then be
one of the spoken ones). Standard captions are English only (`"en"` or null).

`captions: "premium"` without denoise is never refused because our live workers are busy; any new session is refused
(code `full`, `available: []`) only while the server's own relay is measurably overloaded, which running sessions
never notice.

**Free trial.** Every account gets two hours of Standard live once (per mailbox). A session that is only
`denoise: "light"` and/or `captions: "local"` runs on those trial seconds while the account has any: nothing is set
aside or charged, and the seconds of processed audio (counted like billed ones, per whole second) come off the
trial. Anything else (`captions: "premium"`, and any non-empty `translate`) is billed from credit as usual. At most
one trial session runs per account at a time, and trial sessions are only admitted while live use is below about 70%
of its capacity. The `ready` event of a trial session carries `"trial": true, "trialSecondsLeft": N`.

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
 "modelLatencyMs": 39.0, "usdPerMinute": 0.01, "maxSessionSeconds": 43200,
 "silenceFrames": true, "changeable": true, "echoReference": true}
```

`echoReference` is only present (true) when the start message asked for it and
the server will use it. `modelLatencyMs` is the same in every session (39 ms: the noise removal's 30 and the
echo canceller's 9, whether they run or not), so a session can switch either on or off while it runs without
its timing changing (`changeable`, section 4).

`usdPerMinute` is the price of this session's setup per minute of audio (all the parts added up, the free trial aside);
5 minutes of it are set aside when the session starts and charged by the second of processed audio when it ends.

or `{"type":"error","code":...,"message":...}` and closes. Every error event may carry **`retryAfterSeconds`**
(an integer): do not try again sooner. Codes:

| Code | Meaning |
|---|---|
| `bad_start` | no valid `start` message within 15 s |
| `bad_config` | an unusable setting: not 48 kHz, a language code that is not 2-3 letters, a noise strength that is not a number, or `captions: "local"` with a language other than English (**Standard captions are English only**; extra-accurate captions work in other languages) |
| `bad_tier` | an option we do not offer |
| `nothing_enabled` | neither `denoise` nor `captions` is set |
| `insufficient_funds` | the credit set aside up front (5 minutes) is more than the available balance |
| `key_spending_cap` | the API key has a monthly spending cap and the 5 minutes set aside up front would go over it. The message says how much is left and when the month starts over; the owner can raise the cap in the API keys card. Not for browser sessions or keys without a cap |
| `tos` | the current Terms have not been accepted |
| `trial_busy` | a free-trial session was turned away because live is busy (above about 70% of its capacity). Paid sessions are not affected. `retryAfterSeconds` 120 |
| `rate_limited` | (during the session) too many text messages: more than 50 a second sustained (burst 100), or more than 5 `config` a second. The socket is closed with 1008 |
| `too_many_sessions` | four live sessions per account (six per network) are already running, or a free-trial session is asked for while another one of the account is running. A session of yours that has been silent for 10 s is taken to be dropped and is replaced instead. `retryAfterSeconds` 30 |
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
1      1     codec: 1 = pcm_s16le, 2 = f32le, 3 = opus, 4 = silence
2      2     flags: bit 0 = echo reference (section 5), other bits 0 (reserved)
4      4     seq: u32 frame counter (little endian)
8      8     ts_us: u64 client timestamp (opaque, echoed back)
16     ...   mono 48 kHz samples, or exactly one Opus packet (codec 3), or a u16 sample count (codec 4)
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
  delayed by `modelLatencyMs` (39 ms) inside the stream, so the
  client's total delay = network round trip + modelLatencyMs. Measure the
  round trip with `ts_us`.
- When denoise is off (captions only) the audio comes back unchanged and
  the client may simply ignore it, or start with `"returnAudio": false`: then no audio frames come back at all
  (events still do).
- **Silence (codec 4).** When `ready` has `"silenceFrames": true`, a client may send, instead of an encoded frame,
  a header plus a u16 little-endian sample count (480 × k, at most 4800) for audio that is silence: a muted source,
  or digital silence (every |sample| < 1e-5). Echo-reference frames may be silence too. The server answers a
  microphone silence frame at once with the same silence frame (unless `returnAudio` is false). Silence frames keep
  the session alive but are **not billed** and cost no processing.
- **Captions-only sessions are billed for speech, not quiet.** With `denoise: null`, the quiet stretches the
  standard caption decoder skips (from 2 s after the last speech until the next) and the silence extra-accurate
  captions don't send on are not billed.
- Audio must arrive at about real time (bursts up to 6 s are tolerated);
  faster input ends the session with `too_fast`.

## 3. Events (text, server → client)

```json
{"type": "caption", "text": "...", "final": true, "start": 12.34, "end": 15.02}
{"type": "caption", "text": "partial words", "final": false, "id": "item_..."}   // premium only
{"type": "warning", "code": "low_balance", "secondsLeft": 180, "availableMicros": 0}
{"type": "warning", "code": "trial_low", "secondsLeft": 290}   // free trial: sent once, at 5 minutes left (or at once if less is left at the start)
{"type": "warning", "code": "trial_used_up_now_paid", "usdPerMinute": 0.01}   // the trial ran out and the session went on from credit (a hold is set aside now); no audio is dropped
{"type": "warning", "code": "captions_unavailable", "message": "..."}   // extra-accurate captions broke down: not billed from here on
{"type": "warning", "code": "key_spending_cap", "secondsLeft": 180, "message": "..."}   // the key's monthly cap leaves no room for the next minute: the session stops when the minutes already set aside run out
{"type": "stopped", "reason": "insufficient_funds" | "key_spending_cap" | "max_duration" | "idle_timeout" | "no_audio" | "captions_unavailable" | "trial_over" | "account_closed"}
{"type": "error", "code": "...", "message": "..."}
{"type": "pong", "t": ..., "seconds": 123.4}
```

`stopped` is the last thing the server sends before it closes the socket: `insufficient_funds` (the balance cannot
cover the next minute), `key_spending_cap` (the same, because the API key's monthly spending cap is used up), `max_duration` (12 hours), `idle_timeout` (**no microphone audio for 2 minutes**: pings
and echo-reference frames do not count; a muted microphone must keep sending frames, silent ones are fine, as
the OBS plugin and the website do), `no_audio` (**microphone audio arrived for less than half of the wall-clock time over the last 60 s**, judged after the first 30 s: keep sending every 10 ms frame in real time, silence included; a client that is a few seconds late is fine, one that sends a frame now and then is not), `captions_unavailable` (the session was only extra-accurate captions and they stopped working; nothing is billed for them from that point) `trial_over` (the free trial ran out and the available credit cannot cover the session; nothing is charged) and `account_closed` (the account was suspended or deleted, the API key
was revoked, or the browser session ended; checked every 30 s). A session replaced by a newer one of the same
account (see `too_many_sessions`) is closed with WebSocket code 1001 and no event.

`start`/`end` are seconds of audio since the session started. Local captions
arrive once per utterance (~1-3 s after the speaker pauses); premium captions
also stream partial text.

### Translations (optional)

Put `"translate": ["es", "de"]` (at most 3 languages, two-letter codes, not the spoken language) in `start`; it
needs `captions`. `ready` echoes `"translate": [...]`. After every **final** caption, one event per language, in
order per language, usually 1-2 s later:

```json
{"type": "translation", "lang": "es", "text": "...", "of": "l12", "start": 12.34, "end": 15.02}
```

`of` is the caption's `id`; `start`/`end` are copied from it. Partials are never translated. A line that fails is
skipped; after 3 failures in a row that language stops for the session:
`{"type": "warning", "code": "translation_unavailable", "lang": "de", "message": "..."}`.
Billing: `stream_translate/standard` per second of **speech translated** (the caption's `end - start`), per
language, taken from the balance when the session ends (not held up front). Sessions with translations are not
covered by the free trial.

## 4. Control (text, client → server)

```json
{"type": "ping", "t": 123}
{"type": "config", "attenLimDb": 30}
{"type": "config", "id": "c7", "captions": "premium", "language": ["en", "es"], "translate": ["de"]}
{"type": "stop"}
```

**Changing a running session (`config`).** Any of `denoise`, `captions`, `language`, `translate`, `echoReference`
and `attenLimDb` can change while the session runs, with the same values and rules as in `start`; what a message
leaves out stays as it is. The audio keeps flowing with the same delay. With an `id` (up to 64 characters) the
server answers with the session's settings now, or why nothing changed:

```json
{"type": "config", "id": "c7", "ok": true, "usdPerMinute": 0.0255, "denoise": "light", "captions": "premium",
 "language": ["en", "es"], "translate": ["de"], "echoReference": false, "attenLimDb": 30}
{"type": "config", "id": "c7", "ok": false, "code": "full", "message": "...", "retryAfterSeconds": 60}
```

Codes: those of `start` (`bad_config`, `bad_tier`, `nothing_enabled`, `unavailable`, `insufficient_funds`,
`key_spending_cap`), and `full` / `trial_busy` when live has no room for the added work. A refused change leaves
the session exactly as it was. Billing follows the change from the next audio on: each stretch of audio is charged
at the rate of the settings it was processed with. A free-trial session that turns on something the trial doesn't
cover (extra-accurate captions, translations) carries on from credit (`trial` is no longer in the answer).
A message with only `attenLimDb` and no `id` gets no answer (as before).

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
  backlog), or to a text source. The embedded caption track carries one
  language (Latin script only); further languages (`translation` events) go
  to text sources and to one subtitle file per language next to a recording.
- A stopped session is not the end: after `no_audio` (and `idle_timeout`)
  reconnect with the normal backoff as soon as the host feeds audio again,
  not after a fixed long wait. Whatever ends a session, keep passing the
  microphone through unprocessed and say why in the status.
