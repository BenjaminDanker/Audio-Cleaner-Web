# HTTP API (v1)

All paths are relative to the app base (`/audio` on mc-multi-world.space today,
`""` once it moves to its own domain). JSON in and out, except the file upload (raw bytes) and the downloads. One base
URL serves everything: the upload and download routes are answered by the bulk-transfer process behind the same
address, nothing to configure and no redirects.

**Public copy.** The website has this documented for developers at `/developers` (and the live protocol at
`/developers/live`): `web/src/pages/Developers.jsx` and `web/src/pages/DevelopersLive.jsx`. Those pages are written
by hand from this file and `streaming-protocol.md`, in friendlier words and without internals; **when you change the
API, change all three** (the copy test `web/tests/copy.mjs` keeps internals such as model and vendor names out of them).

Errors are `{"detail": "human readable message"}` with a 4xx/5xx status; validation errors
are `422 {"detail": "Invalid request.", "fields": [...]}` (the names of the fields, never their values).
Stability: this is v1; fields are only ever added, never renamed or removed without a new version of the path.
There is no changelog beyond the git history of this file.

## Limits and rate limits

- Request bodies are at most 64 KB (the Stripe webhook 1 MB). Only the upload route (`PUT /api/jobs/{id}/input`)
  takes a file; a `413` means the body was too large. Uploads that send nothing for 30 s are dropped (`408`), and upload
  speed is capped per account and overall.
- Every `429` carries `Retry-After` (seconds) and says how long to wait.

| What | Limit |
|---|---|
| Any API call | 120 in a burst, then 2 a second, per network (IPv6 per /64). Polling one job every 2 s is fine; poll several jobs with `GET /api/jobs` instead of one request each |
| Creating jobs (`POST /api/jobs`) | 20 an hour per account and per network. A repeat of an `Idempotency-Key` costs nothing |
| Uploads | 30 an hour per account |
| Active jobs | 3 per account at once (`429`, "wait for one to finish") and at most 8 GiB of uploads waiting (`429`) |
| Creating API keys (browser) | 10 an hour; at most 10 active keys |
| Live sessions | 60 starts an hour per account, 90 per network; four at once per account (see the live protocol) |

`GET /api/health` answers `{ok:true}`; with `?deep=1` from the server itself (or an admin) it also reports the
live worker, stuck jobs, missing model files and the ledger check.

## Auth model

- Browser: HttpOnly session cookie set by the sign-in endpoints. Every
  non-GET request must send `X-CSRF-Token: <csrfToken from GET /api/me>` and
  is rejected if its `Origin` header is present and not the site origin.
- Programs (OBS plugin, scripts): `Authorization: Bearer acw_...` API key, created on the Live page of the website
  (a key is shown once). Billing, keys and account endpoints refuse API keys (browser session only).
  Each key carries one or both scopes, and a request outside its scopes gets `403`:
  - `stream`: live audio. Used by the OBS plugin; opens `/api/stream/ws`
    (see [streaming-protocol.md](streaming-protocol.md)). Give the plugin a key
    with only this scope.
  - `jobs`: the REST file API for scripts and automation: `/api/jobs` (create, estimate, upload, poll, download,
    cancel, delete), see "File jobs" below.
  Both bill the signed-in account's balance exactly like the website does. A key can also carry a **monthly spending
  cap** (see "API keys"). A key may read `GET /api/me` (balance, free trial left, its own cap) and the public
  `GET /api/billing/pricing`.
- Creating a job (or starting a live session) records that the account accepts the current Terms.

### Errors

| Status | When |
|---|---|
| 400 | a malformed or out-of-range option, a file type we don't take, a bad `callbackUrl`, a duration outside the limits, a size that doesn't fit the duration (over 40 MB per second, unless under 200 MB) |
| 401 | no credentials, or the key is wrong or revoked |
| 402 | the available balance can't cover the estimated price (the message says both amounts) |
| 403 | the key lacks the scope, the account is suspended, or **the key's monthly spending cap is reached** (then the header `X-Error-Code: key_spending_cap` is set and the message says how much is left and when it starts over) |
| 404 | the job doesn't exist, isn't yours, or the output isn't available |
| 408 | an upload stalled for 30 s |
| 409 | the job is in the wrong state for this (upload when not waiting for one, cancel after it started, delete while it runs), or an `Idempotency-Key` was used for different details |
| 411 | the upload has no `Content-Length`, or it differs from the `sizeBytes` given at creation |
| 413 | the file is bigger than allowed (`maxUploadBytes` of `/api/billing/pricing`), or the upload is longer than declared |
| 422 | the JSON doesn't match the shape (`fields` lists which) |
| 429 | a rate limit above; `Retry-After` says when |
| 507 | the host is short on disk right now; try again later |
| 503 | the service is on a short break (`{"detail":"maintenance"}`) or payments aren't switched on |

## Account

| Method | Path | Notes |
|---|---|---|
| GET | `/api/me` | `{signedIn:false, contactForm}` or `{signedIn, contactForm, user:{id,email,displayName,isAdmin,status,tosAccepted,createdAt}, csrfToken, balance:{balance_micros,held_micros,available_micros}, topupMaxCents, trialLiveSecondsLeft, autoReload:{enabled,thresholdMicros,amountCents,monthlyCapCents,hasCard,failures}}`. `topupMaxCents`: the largest top-up right now ($50 until a first top-up has gone through, then $500); `trialLiveSecondsLeft`: what is left of the free live trial (whole seconds). With an API key the answer is only `{signedIn, contactForm, user:{id,status}, balance, trialLiveSecondsLeft, keyLimit?}`; `keyLimit` (only for a key with a monthly cap) is `{monthlyCapMicros, usedMicros, remainingMicros, resetsAt}` |
| POST | `/api/account/tos` | `{version}`: must equal the current terms version (see `TOS_VERSION`, currently `2026-10-16`) |
| GET | `/api/auth/providers` | `{google:bool, email:bool, turnstileSiteKey:str|null}` |
| GET | `/api/auth/oauth/google/start` | browser navigation; ends at `/app` (or `/login?error=...`) |
| POST | `/api/auth/email` | `{email, turnstileToken?}` sends a magic link to `/login/verify#token=...` |
| POST | `/api/auth/email/verify` | `{token}` (from the URL fragment) creates the session |
| POST | `/api/auth/logout`, `/api/auth/logout-all` | |
| GET | `/api/account/export` | JSON of everything stored about the user (download as a file); secrets (the callback secret) are not included, only that they exist |
| POST | `/api/contact` | works signed in or out (same-origin; the CSRF token when signed in). `{name?, email, topic: "question"\|"billing"\|"job"\|"privacy"\|"other", message (10 to 5000 characters), job? (32-hex job id), website? (leave empty: a hidden trap field for bots)}`. Mails the owner with Reply-To set to `email`; answers `{ok:true}`. The message is not stored. 400 bad email/message/job id, 404 when mail is not set up (`/api/me` has `contactForm:false`), 429 with `Retry-After` (3 an hour and 10 a day per network, 3 an hour and 5 a day per mailbox, 100 a day for the whole site), 503 when the mail could not be sent. The owner's address is never returned |
| POST | `/api/account/delete` | `{confirm: "<the user's email>"}`; refuses while jobs run; remaining credit is forfeited (offer refunds first); signs out; removes the callback secret and addresses |

Money is in **micros** (1 USD = 1,000,000) except Stripe amounts in **cents**.

## Billing

| Method | Path | Notes |
|---|---|---|
| GET | `/api/billing/pricing` | public. `{rates:[{product,tier,label,usdPerMinute,usdPerHour,available}], topup:{minCents,maxCents,presetsCents}, maxJobSeconds, maxUploadBytes, maxRebuildSeconds, paymentsEnabled, taxAddedAtCheckout}`. `maxJobSeconds` is 14,400 (4 hours); `maxRebuildSeconds` 5,400 (90 minutes, files that get the voice rebuilt); `maxUploadBytes` 4 GiB; `taxAddedAtCheckout`: sales tax is added when buying credit, never to usage. There is no `free` block: the free tools run in the browser |
| POST | `/api/billing/checkout` | `{amountCents, saveCard}` → `{url}`; redirect the browser there. Returns to `/app/billing?checkout=success|cancel`. 400 above `topupMaxCents` of `/api/me` |
| POST | `/api/billing/auto-reload` | `{enabled, thresholdCents, amountCents, monthlyCapCents, consent}`; `consent:true` required to enable, needs a saved card; `amountCents` is limited like a top-up (400) |
| POST | `/api/billing/portal` | `{url}` Stripe customer portal (manage/remove cards, receipts) |
| GET | `/api/billing/history` | `{balance, ledger:[...], ledgerHasMore, payments:[{id,kind,status,amount_cents,refunded_cents,created_at}]}`. The newest 100 ledger entries, each `{id,amount_micros,kind,description,balance_after_micros,created_at,label}` plus, for usage entries, `product` (`clean`, `transcript`, `clean_transcript`, `live`), `tier`, `minutes` and `displayName` (the file name as the person chose it). `label` is the customer-facing name ("Top-up", "Denoising + Subtitles & transcript"); `ledgerHasMore` says older entries exist. |
| POST | `/api/billing/refund` | `{paymentId}` refunds unused credit from a payment < 14 days old |

Price list (`rates`; per minute of audio, counted by the second, rounded up; amounts in the response are in dollars):

| `product` / `tier` | What it is |
|---|---|
| `denoise` / `standard` | Denoising |
| `denoise` / `rebuild` | Voice rebuild (files up to 90 minutes) |
| `leveling` / `standard` | Leveling: charged when `options.leveling` is true (the default) on a `denoise` job; `denoiseTier: "off"` makes a job that only levels |
| `transcribe` / `local` | Subtitles & transcript |
| `transcribe` / `premium` | Subtitles & transcript, extra accurate |
| `transcribe_translate` / `standard` | one translation of the subtitles, **per language** |
| `stream_denoise` / `light` | Live noise removal |
| `stream_captions` / `local` | Live captions (English) |
| `stream_captions` / `premium` | Live captions, extra accurate |
| `stream_translate` / `standard` | Live caption translation, **per language** |

## File jobs

Lifecycle: `POST /api/jobs` (reserves the estimated price) -> `PUT` the file -> poll -> download. Everything needs a
key with the `jobs` scope (or a browser session). Worked example with curl:

```bash
BASE=https://mc-multi-world.space/audio ; KEY=acw_...
# 0. optional: what will it cost?
curl -s $BASE/api/jobs/estimate -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"kind":"denoise_transcribe","denoiseTier":"standard","transcribeTier":"local","durationSeconds":612.4,"options":{"translate":["es"]}}'
# 1. create (the Idempotency-Key makes a retry safe)
curl -s $BASE/api/jobs -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' -H 'Idempotency-Key: talk-2026-10-16' \
  -d '{"kind":"denoise","denoiseTier":"standard","filename":"talk.wav","sizeBytes":'$(stat -c %s talk.wav)',"durationSeconds":612.4}'
#   -> {"id":"3f2c...","estCostMicros":102167,"uploadUrl":"api/jobs/3f2c.../input"}
# 2. upload the raw bytes (Content-Length is set by curl and must equal sizeBytes)
curl -s -X PUT --data-binary @talk.wav $BASE/api/jobs/$ID/input -H "Authorization: Bearer $KEY"
# 3. poll until status is done / failed / canceled
curl -s $BASE/api/jobs/$ID -H "Authorization: Bearer $KEY"
# 4. download an output by its name
curl -s -OJ $BASE/api/jobs/$ID/outputs/talk-cleaned.wav -H "Authorization: Bearer $KEY"
# cancel (before it starts) or delete (after it ended)
curl -s -X POST   $BASE/api/jobs/$ID/cancel -H "Authorization: Bearer $KEY"
curl -s -X DELETE $BASE/api/jobs/$ID        -H "Authorization: Bearer $KEY"
```

Python (standard `requests`): `r = requests.post(f"{BASE}/api/jobs", json=body, headers=h)`, then
`requests.put(f"{BASE}/{r.json()['uploadUrl']}", data=open(path, "rb"), headers=h)` (a file object makes requests send a
`Content-Length`), then poll `GET`.

### 1. Create: `POST /api/jobs`

```json
{"kind": "denoise" | "transcribe" | "denoise_transcribe",
 "denoiseTier": "standard" | "rebuild" | "off", "transcribeTier": "local" | "premium",
 "filename": "talk.mp4", "sizeBytes": 123456, "durationSeconds": 612.4,
 "displayName": "My talk.wav",
 "callbackUrl": "https://example.com/hooks/audio-cleaner",
 "options": {"audio_format": "wav|flac|mp3|m4a|opus", "caption_formats": ["srt","vtt","txt","json"],
             "language": "en" | null, "output_video": true, "embed_subtitles": true,
             "loudness_lufs": -16 | null, "atten_lim_db": 100, "echo_reference_track": false,
             "noise": "auto|light|heavy", "leveling": true, "restore_highs": true, "translate": ["es", "de"]}}
```

Answer: `{id, estCostMicros, uploadUrl}`. `uploadUrl` is relative to the base, **without** a leading slash
(`api/jobs/<id>/input`). Fields:

| Field | Rules |
|---|---|
| `kind` | `denoise` (clean the sound), `transcribe` (subtitles and transcript), or both |
| `denoiseTier` | required for `denoise` kinds: `standard` (Denoising), `rebuild` (Voice rebuild, its own price: the voice is rebuilt, which also removes room echo and restores dull sound; `atten_lim_db` and `noise` don't apply to it; files up to 90 minutes), or `off` (no noise removal: only Leveling, which `options.leveling` must then be). Leveling (`options.leveling`, on by default) is its own line on the bill. |
| `transcribeTier` | required for `transcribe` kinds: `local` (Standard) or `premium` (extra accurate; audio goes to a speech partner, see the privacy policy) |
| `filename` | its extension must be one we read: audio `wav rf64 w64 mp3 mp2 mpa m4a m4b m4r aac flac ogg oga opus spx wma aiff aif aifc caf au snd amr ac3 eac3 dts mka wv ape tta mpc`, video `mp4 m4v mov webm mkv avi wmv asf flv f4v 3gp 3g2 ts mts m2ts mpg mpeg vob mxf dv ogv rm rmvb`; anything else is `400 Unsupported file type.` Only the extension is used; the name never becomes a path |
| `sizeBytes` | 1 byte up to `maxUploadBytes` (4 GiB; `413` above); at most 40 MB per second of `durationSeconds` unless under 200 MB |
| `durationSeconds` | 0.5 s up to 14,400 s (4 h); up to 5,400 s (90 min) with `rebuild`. From the client; the server measures again after the upload and prices what it finds |
| `displayName` | optional, shown as the job's name and used for result file names. Up to 300 characters sent, cut to 120 after control, bidirectional and zero-width characters are removed |
| `callbackUrl` | optional, see "Finish notifications" |
| `free` | `true` is refused (400): jobs here are always paid. The free tools run in the visitor's browser |

`options` (the whole object must be under 2000 characters; unknown keys are ignored):

| Option | Default | Range and meaning |
|---|---|---|
| `audio_format` | `wav` | `wav`, `flac`, `mp3`, `m4a`, `opus` |
| `caption_formats` | `["srt","vtt","txt"]` | any of `srt`, `vtt`, `txt`, `json` (`json` gives `segments.json`: every line with its times, kind `segments`) |
| `language` | `null` (detect) | the spoken language: a 2-3 letter code (`en`, `es`, `haw`) or null |
| `translate` | `[]` | up to 3 target languages (two-letter codes from the translation list, not the spoken one); needs a transcription kind. The subtitles and transcript are also made in those languages (`<stem>.<lang>.srt\|vtt`, `<stem>.<lang>-transcript.txt`), priced per language. A language that can't be made is left out, said in `notice` and not charged |
| `output_video` | `true` | for a video input: give a video back (with the cleaned sound). `false`: audio only |
| `embed_subtitles` | `true` | put the subtitles into the video; needs the whole video uploaded |
| `loudness_lufs` | `-16` | target loudness, a number from -30 to -9 (`400` outside it), or `null` to leave the level as it is. Podcast about -16, online video -14, broadcast -23 |
| `atten_lim_db` | `100` | how much noise may be removed: any number, clamped to 6-100 dB. The website's strengths are 12 (gentle), 24, 40 and 100 (max). Doesn't apply to `rebuild` |
| `echo_reference_track` | `false` | must be a boolean. Removes speaker echo from the first audio track (the mic) using the second track (the desktop sound), before noise removal; subtitles are written from the echo-free sound. Included in the price. A single-track file is processed normally and `notice` says so |
| `noise` | `auto` | `auto` (measured per file), `light`, `heavy` (better when the noise is nearly as loud as the voice). Doesn't apply to `rebuild` |
| `leveling` | `true` | Leveling (its own price): the clarity chain after the noise removal (level matching, high-pass, compressor, loudness to `loudness_lufs`, limiter); `false` returns the cleaned voice as it is (`loudness_lufs` then does nothing). Must be a boolean. |
| `restore_highs` | `true` | band-limited recordings (phone calls, low-bitrate files) get their missing highs restored; full-band ones are never changed. `false` switches it off. Must be a boolean. Doesn't apply to `rebuild`. Same price |

Highs of band-limited recordings (phone calls, low-bitrate files) are restored automatically; full-band recordings
are never changed, and `notice` says when it happened.

A video result keeps the input's container when it can hold the streams (`mp4 mov m4v mkv webm`); other containers
come back as MKV. Other audio tracks are kept unchanged and subtitles go in where the container can hold them; only
the first audio track is cleaned. The website converts container and picture size in the browser; the API doesn't.

`embed_subtitles` makes the service put the subtitles into the video, which needs the whole video uploaded. The
website only asks for it for a video the browser can't open (it goes up whole anyway); otherwise it uploads the
sound only and puts the subtitles into an MKV in the browser afterwards.

Errors specific to creating: `400` (see the table above), `402` (balance below the estimate), `403` (key cap),
`409` (Idempotency-Key reused with different details), `429` (3 active jobs, 8 GiB of uploads waiting, job creation
allowance), `507`.

**Idempotency.** Send `Idempotency-Key: <1 to 128 visible ASCII characters>` and the same key from the same account within
24 hours returns the same job instead of making (and holding money for) another: the answer is the original
`{id, estCostMicros, uploadUrl}` plus `"idempotentReplay": true` and the header `Idempotent-Replayed: true`. The
details must match (the whole request body is compared): the same key with a different body is `409`. A request that
failed (402, 429, ...) doesn't use the key up. Use a new key per file; a good one is a hash of the file name and size.

### 2. Estimate: `POST /api/jobs/estimate`

Same `kind`, `denoiseTier`, `transcribeTier`, `durationSeconds` and `options` as creating (no file fields); makes no
job and holds nothing. Applies the same checks and limits, so it answers the same `400`s.

```json
{"estCostMicros": 102167, "estCostUsd": 0.102167, "durationSeconds": 612.4, "maxSeconds": 14400,
 "lines": [{"item": "denoise", "label": "Denoising", "quantity": 1, "usdPerMinute": 0.01, "micros": 102167}],
 "options": {"...": "the options with the defaults filled in"},
 "availableMicros": 5000000, "affordable": true,
 "keyLimit": {"monthlyCapMicros": 500000, "usedMicros": 0, "remainingMicros": 500000, "resetsAt": 1793491200}}
```

`lines` has `denoise`, `transcribe` and `translate` (`quantity` = number of languages) as they apply; the micros add
up to `estCostMicros`, which is exactly what `POST /api/jobs` would reserve for the same details. `keyLimit` only when
the key has a monthly cap.

### 3. Upload: `PUT /api/jobs/{id}/input`

The raw file bytes as the body (no JSON, no multipart). `Content-Length` is required and must equal `sizeBytes`
(`411` otherwise). The upload must **start within 10 minutes** of creation; it may then take up to an hour after the last
bytes arrive (the clock restarts while data flows). Answers `{ok:true}`. Errors: `409` if the job isn't waiting for an
upload (already uploaded, canceled, expired), `408` after 30 s without data, `413` if more arrives than declared,
`400` if less, `507` if the host is short on disk. Use XHR in a browser for progress. If it fails, create a new job.

### 4. Poll: `GET /api/jobs/{id}` (and `GET /api/jobs`)

`{id,kind,tier,label,free,status,stage,progress,error,inputName,durationSeconds,estCostMicros,costMicros,options,notice,queuePosition,queueWaitSeconds,outputs:[{name,kind,bytes,lang?}],createdAt,finishedAt,expiresAt,callback?}`

- `status`: `uploading → queued → running → (cloud_asr) → done | failed | canceled | expired`. A restart of the
  service never fails a job: it goes back to `queued` and starts again.
- `stage`: a short sentence-case phrase such as `Cleaning`, `Transcribing`, `Waiting to transcribe`, `Writing captions`;
  `progress`: 0-100; `label`: what the job does in the words of the price list; `inputName`: the display name when one
  was given.
- `queuePosition` (1 = next) and `queueWaitSeconds` (an estimate): only while `queued`, else `null`.
- `estCostMicros` is what was reserved; `costMicros` the real price once `done` (0 for `failed` and `canceled`).
- `error`: the reason for `failed` / `canceled`. `notice`: a plain-words note about how the file was handled (not an
  error), or `null`.
- `outputs[].kind`: `audio`, `video`, `captions`, `transcript`, `segments`; captions and transcripts carry `lang`.
  `name` is what the download URL takes.
- `expiresAt`: when the results are deleted (24 hours after `done`).
- `callback`: only when a `callbackUrl` was given, see below.

`GET /api/jobs` returns `{jobs:[...]}`, your latest 50 (not the expired ones), in the same shape: one request polls them all.

### 5. Download: `GET /api/jobs/{id}/outputs/{name}`

Streams the file as an attachment (`name` from `outputs`). `404` until the job is `done`, and after the results expire. Download
promptly: they are deleted 24 hours after `done`, or at once with `DELETE`.

### 6. Cancel and delete

`POST /api/jobs/{id}/cancel`: while `uploading` or `queued`; releases the reservation. `409` once it has started, and
once its extra-accurate transcription is done (only packaging is left: "almost done"). `DELETE /api/jobs/{id}`: deletes the
outputs now; `409` while the job is active. Both answer `{ok:true}`.

### Billing per call

Nothing is charged when a call fails, and only audio that was actually processed is billed, by the second.

1. **Hold.** Creating a job reserves its estimated price (`estCostMicros`: the claimed duration at today's prices,
   with the translations asked for) from the available balance (`402` if it isn't there). The prices are fixed at this
   moment, whatever happens to the price list while the job waits. The held amount isn't spendable by other jobs.
2. **Re-measure.** After the upload the service measures the real duration. If the real price is higher than the hold, the
   hold grows; if the balance can't cover it, the job fails ("costs more than your available balance") and nothing is charged.
3. **Capture.** When the job is `done`, the real price (`costMicros`) is taken and the rest of the hold is given back.
   A translation that couldn't be made isn't charged.
4. **Release.** A `failed`, `canceled` or abandoned (never uploaded) job gives the whole hold back. Failed jobs are free.

A key with a monthly spending cap counts, for the month, finished jobs at their real price and running jobs at their
hold; creating a job that would go over is refused with `403` before anything is held. A job already accepted is never
stopped by the cap; if a file turns out longer than claimed it can finish slightly over.

Live sessions: 5 minutes are held when the session starts and one more minute is added as the audio runs; the real
audio seconds are charged when it ends and the rest is released; translations are charged at the end. The `ready`
event carries `usdPerMinute`, the price of this session's setup per minute. The free trial uses no credit.

### Finish notifications (`callbackUrl`)

Instead of polling, give `callbackUrl` when creating the job. When the job ends (`done`, `failed` or `canceled`) the
service POSTs JSON to it:

```json
{"type": "job.done", "jobId": "3f2c...", "status": "done", "error": null, "costMicros": 102167,
 "outputs": [{"name": "talk-cleaned.wav", "kind": "audio", "bytes": 5242880}],
 "finishedAt": 1760600000, "expiresAt": 1760686400}
```

`type` is `job.done`, `job.failed`, `job.canceled` (or `job.expired` for a job whose results were deleted before the first
delivery); the body is small on purpose: call `GET /api/jobs/{id}` and download for the rest. Download the files
within their 24 hours. A callback can arrive more than once (a retry after a timeout): treat `jobId` as the dedupe key.

**Signature.** Each request carries `X-Audio-Cleaner-Signature: t=<unix seconds>,v1=<hex>` where `v1` is
`HMAC-SHA256(secret, "<t>.<raw body>")` (the body exactly as received). The secret (`whsec_...`, one per account) is shown
and replaced in the API keys card on the Live page; it is made the first time you use a callback. Check the signature
with a constant-time compare and reject a `t` more than 5 minutes old.

```python
import hashlib, hmac, time
def verify(secret: str, header: str, raw_body: bytes) -> bool:
    parts = dict(p.split("=", 1) for p in header.split(","))
    if abs(time.time() - int(parts["t"])) > 300:
        return False
    want = hmac.new(secret.encode(), parts["t"].encode() + b"." + raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(want, parts["v1"])
```

**Rules of the address.** `https` only, port 443 only, a public host name (no IP address, no `user:password@`, no `#fragment`,
no `localhost`, `.local` or `.internal` names, at most 500 characters; `400` otherwise). Delivery refuses any name that resolves to a
private, loopback, link-local, shared (CGNAT), multicast or reserved address, IPv4 or IPv6, or to one of the server's own
addresses; it connects to the address it checked, follows no redirects (a 3xx is a failure), waits 5 seconds at most, and
gives up after 3 tries (the 2nd about 30 s and the 3rd about 5 minutes after the one before). Any 2xx answer is success.

**The result** shows in `GET /api/jobs/{id}` as `callback`: `{url (without its query string), state: "pending" |
"delivered" | "failed", attempts, lastStatus (HTTP status or null), lastError (plain words or null), deliveredAt,
nextAttemptAt}`. A callback that never worked is not retried by hand; poll instead. Only the web process makes these calls.

## API keys (browser session)

| Method | Path | Notes |
|---|---|---|
| GET | `/api/keys` | `{keys:[{id,name,prefix,scopes,createdAt,lastUsedAt,monthlyCapCents,usedThisMonthMicros,capResetsAt}]}` |
| POST | `/api/keys` | `{name, scopes:["stream","jobs"], monthlyCapCents?}` → `{id, key, prefix, scopes, monthlyCapCents}` (the only time the full key is shown) |
| PATCH | `/api/keys/{id}` | `{monthlyCapCents: 500}` sets the cap, `{monthlyCapCents: null}` removes it. The change applies to the next job or live minute |
| DELETE | `/api/keys/{id}` | revoke |
| GET | `/api/keys/webhook-secret` | `{secret}` (null until first made) |
| POST | `/api/keys/webhook-secret` | makes or replaces the account's callback secret; `{secret}` |

**Monthly cap.** `monthlyCapCents` (1 to 1,000,000; null = no cap of its own) limits what one key may spend in a calendar
month (UTC): finished file jobs and live sessions at their real price, running ones at what is held. The browser
session and other keys of the account aren't affected. `usedThisMonthMicros` is that sum; `capResetsAt` the start of
next month. A job over the cap is `403` (header `X-Error-Code: key_spending_cap`); a live session over it gets the
error `key_spending_cap` at the start, a warning of the same code when the next minute can't be reserved, and
`stopped` with that reason when the reserve runs out.

## Live streaming

WebSocket `GET /api/stream/ws`, see [streaming-protocol.md](streaming-protocol.md).
Browsers may connect with the session cookie (same-origin only).
The `start` message may set `echoReference: true` to send what the speakers play next to the
microphone, so the server cancels the echo (included in the price; protocol section 5).

## Webhooks

`POST /api/stripe/webhook` (Stripe-Signature verified). Job callbacks are the other direction: see "Finish
notifications" above and the SSRF rules in [security.md](security.md).
