# Live streaming protocol v1

One WebSocket per live source: `wss://<host><base>/api/stream/ws`.

Authenticate with `Authorization: Bearer acw_...` (API key with the `stream`
scope). Browsers on the site itself can use the session cookie instead.

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
 "client": "obs-audio-cleaner/0.1.0"}
```

At least one of `denoise` / `captions` must be set. Server answers:

```json
{"type": "ready", "sessionId": "...", "sampleRate": 48000, "hopSamples": 480,
 "modelLatencyMs": 30.0, "usdPerMinute": 0.01, "maxSessionSeconds": 43200}
```

or `{"type":"error","code":...,"message":...}` and closes. Codes:
`insufficient_funds`, `tos`, `bad_config`, `bad_tier`, `too_many_sessions`,
`unavailable`, `nothing_enabled`, `bad_start`.

## 2. Audio frames (binary, both directions)

```
offset size  field
0      1     version = 1
1      1     codec: 1 = pcm_s16le, 2 = f32le, 3 = opus
2      2     flags = 0 (reserved)
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
{"type": "stopped", "reason": "insufficient_funds" | "max_duration"}
{"type": "error", "code": "...", "message": "..."}
{"type": "pong", "t": ..., "seconds": 123.4}
```

`start`/`end` are seconds of audio since the session started. Local captions
arrive once per utterance (~1-3 s after the speaker pauses); premium captions
also stream partial text.

## 4. Control (text, client → server)

```json
{"type": "ping", "t": 123}
{"type": "config", "attenLimDb": 30}
{"type": "stop"}
```

The server closes the socket after 30 s without any message. On `stop` it
flushes outstanding captions, then closes. Billing covers processed audio
seconds only (rounded up to the second at the end of the session).

## Client requirements (OBS plugin)

- Never block the audio thread: ring buffers in and out, network on a worker.
- Fixed output delay D chosen at start (e.g. RTT + 30 ms + jitter margin,
  presets 80/150/250 ms), never changed mid-session; back-date output
  timestamps by D. If a processed frame misses its deadline, output the
  delayed original audio for that frame (crossfade ≤10 ms).
- Reconnect with exponential backoff + jitter; pass audio through while
  disconnected.
- Show status: connected, RTT, D, late/dropped frames, balance warnings.
- Captions: forward `final` captions to `obs_output_output_caption_text2`
  on the streaming output, paced (≤ 32 chars per line, don't queue a
  backlog), or to a text source.
