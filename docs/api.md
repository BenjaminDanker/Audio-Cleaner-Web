# HTTP API (v1)

All paths are relative to the app base (`/audio` on mc-multi-world.space today,
`""` once it moves to its own domain). JSON in and out. Errors are
`{"detail": "human readable message"}` with a 4xx/5xx status; validation errors
are `422 {"detail": "Invalid request.", "fields": [...]}`.

Limits: request bodies are at most 64 KB (the Stripe webhook 1 MB); only the upload route (`PUT /api/jobs/{id}/input`)
takes a file. A `429` always carries `Retry-After` (seconds) and says how long to wait; a `413` means the body was
too large. Uploads that send nothing for 30 s are dropped, and upload speed is capped per account and overall.
`GET /api/health` answers `{ok:true}`; with `?deep=1` from the server itself (or an admin) it also reports the
live worker, stuck jobs, missing model files and the ledger check.

## Auth model

- Browser: HttpOnly session cookie set by the sign-in endpoints. Every
  non-GET request must send `X-CSRF-Token: <csrfToken from GET /api/me>` and
  is rejected if its `Origin` header is present and not the site origin.
- Programs (OBS plugin, scripts): `Authorization: Bearer acw_...` API key.
  Billing, keys and account endpoints refuse API keys (browser session only).
  Each key carries one or both scopes, and a request outside its scopes gets 403:
  - `stream`: live audio. Used by the OBS plugin; opens `/api/stream/ws`
    (see [streaming-protocol.md](streaming-protocol.md)). Give the plugin a key
    with only this scope.
  - `jobs`: the REST file API for scripts and automation. `/api/jobs` (create,
    upload, poll, download, cancel, delete), see "File jobs" below.
  Both bill the signed-in account's balance exactly like the website does.

## Account

| Method | Path | Notes |
|---|---|---|
| GET | `/api/me` | `{signedIn:false, contactForm}` or `{signedIn, contactForm, user:{id,email,displayName,isAdmin,status,tosAccepted,createdAt}, csrfToken, balance:{balance_micros,held_micros,available_micros}, autoReload:{enabled,thresholdMicros,amountCents,monthlyCapCents,hasCard,failures}}` |
| POST | `/api/account/tos` | `{version}`: must equal the current terms version (see `TOS_VERSION`, currently `2026-10-10`) |
| GET | `/api/auth/providers` | `{google:bool, email:bool, turnstileSiteKey:str|null}` |
| GET | `/api/auth/oauth/google/start` | browser navigation; ends at `/app` (or `/login?error=...`) |
| POST | `/api/auth/email` | `{email, turnstileToken?}` sends a magic link to `/login/verify#token=...` |
| POST | `/api/auth/email/verify` | `{token}` (from the URL fragment) creates the session |
| POST | `/api/auth/logout`, `/api/auth/logout-all` | |
| GET | `/api/account/export` | JSON of everything stored about the user (download as a file) |
| POST | `/api/contact` | works signed in or out (same-origin; the CSRF token when signed in). `{name?, email, topic: "question"\|"billing"\|"job"\|"privacy"\|"other", message (10 to 5000 characters), job? (32-hex job id), website? (leave empty: a hidden trap field for bots)}`. Mails the owner with Reply-To set to `email`; answers `{ok:true}`. The message is not stored. 400 bad email/message/job id, 404 when mail is not set up (`/api/me` has `contactForm:false`), 429 with `Retry-After` (3 an hour and 10 a day per network, 3 an hour and 5 a day per mailbox, 100 a day for the whole site), 503 when the mail could not be sent. The owner's address is never returned |
| POST | `/api/account/delete` | `{confirm: "<the user's email>"}`; refuses while jobs run; remaining credit is forfeited (offer refunds first); signs out |

Money is in **micros** (1 USD = 1,000,000) except Stripe amounts in **cents**.

## Billing

| Method | Path | Notes |
|---|---|---|
| GET | `/api/billing/pricing` | public. `{rates:[{product,tier,label,usdPerMinute,usdPerHour,available}], topup:{minCents,maxCents,presetsCents}, free:{transcribeSecondsPerDay,transcribeMaxFileSeconds}, maxJobSeconds, paymentsEnabled}` |
| POST | `/api/billing/checkout` | `{amountCents, saveCard}` → `{url}`; redirect the browser there. Returns to `/app/billing?checkout=success|cancel` |
| POST | `/api/billing/auto-reload` | `{enabled, thresholdCents, amountCents, monthlyCapCents, consent}`; `consent:true` required to enable, needs a saved card |
| POST | `/api/billing/portal` | `{url}` Stripe customer portal (manage/remove cards, receipts) |
| GET | `/api/billing/history` | `{balance, ledger:[...], ledgerHasMore, payments:[{id,kind,status,amount_cents,refunded_cents,created_at}]}`. The newest 100 ledger entries, each `{id,amount_micros,kind,description,balance_after_micros,created_at,label}` plus, for usage entries, `product` (`clean`, `transcript`, `clean_transcript`, `live`), `tier`, `minutes` and `displayName` (the file name as the person chose it). `label` is the customer-facing name ("Top-up", "Clean & polish + Subtitles & transcript"); `ledgerHasMore` says older entries exist. |
| POST | `/api/billing/refund` | `{paymentId}` refunds unused credit from a payment < 14 days old |

Products/tiers: `denoise/standard`, `denoise/rebuild` (voice rebuilt, files up to 90 minutes), `transcribe/local`, `transcribe/premium`,
`stream_denoise/light` (DFN3, 30 ms), `stream_captions/local`, `stream_captions/premium`.

## File jobs

1. `POST /api/jobs` body:
   ```json
   {"kind": "denoise" | "transcribe" | "denoise_transcribe",
    "denoiseTier": "standard", "transcribeTier": "local" | "premium",
    "filename": "talk.mp4", "sizeBytes": 123456, "durationSeconds": 612.4,
    "displayName": "My talk.wav",
    "free": false,
    "options": {"audio_format": "wav|flac|mp3|m4a|opus", "caption_formats": ["srt","vtt","txt","json"],
                "language": "en" | null, "output_video": true, "embed_subtitles": true,
                "loudness_lufs": -16 | null, "atten_lim_db": 100, "echo_reference_track": false,
                "noise": "auto|light|heavy"}}
   ```
   `noise` (default `"auto"`) picks the noise remover: `light` = DeepFilterNet3, `heavy` = DPDFNet-8 (better
   when the noise is nearly as loud as the voice), `auto` = measured per file (`pipeline.HEAVY_BELOW_DB`).
   Band-limited recordings (phone calls, low-bitrate files) also get their missing highs restored (AP-BWE);
   full-band recordings are never changed, and the job's `notice` says when it happened.
   `denoiseTier: "rebuild"` (its own price) rebuilds the voice with Sidon instead: it also removes room echo and
   restores dull sound; `atten_lim_db` and `noise` do not apply to it.
   `echo_reference_track` (default `false`, must be a boolean) removes speaker echo from the first audio track
   (the mic) using the file's second audio track (the desktop/speaker sound, as OBS can record it), before the
   noise removal; the subtitles are then written from the echo-free recording. It is included in the price.
   A file with a single audio track is processed normally and the job's `notice` says so.
   `displayName` (optional, up to 120 characters; control, bidirectional and zero-width characters are removed) is
   the name shown for the job and used for the result files; without it the uploaded file's name is used. The
   uploaded file can be a re-encoded copy of what the person chose.
   `free:true` makes the subtitles free (daily allowance, 30 min/file) and needs `transcribeTier:"local"`: with
   `kind:"transcribe"` the job is free, with `kind:"denoise_transcribe"` only the cleanup is charged (the job holds
   and captures the cleanup's price and counts as paid in the queue). It is refused for `kind:"denoise"`.
   `embed_subtitles` makes our side put the subtitles into the video, which needs the whole video uploaded. The
   website only asks for it for a video the browser can't open (it goes up whole anyway); otherwise it uploads the
   sound only and puts the subtitles into an MKV in the browser afterwards (SRT-style track, which players read). Paid jobs need accepted terms. Returns
   `{id, estCostMicros, uploadUrl}`; 402 if the balance can't cover the estimate.
   `durationSeconds` comes from the browser; the server re-measures and re-prices.
2. `PUT /api/jobs/{id}/input` with the raw file bytes as the body (`Content-Length`
   must equal `sizeBytes`). Use XHR for upload progress.
3. Poll `GET /api/jobs/{id}` every 1-2 s:
   `{id,kind,tier,label,free,status,stage,progress,error,inputName,durationSeconds,estCostMicros,costMicros,options,notice,outputs:[{name,kind,bytes}],createdAt,finishedAt,expiresAt}`
   (`label`: what the job does in the words of the price list; `inputName`: the display name when one was given;
   `stage`: a short sentence-case phrase such as `Cleaning`, `Transcribing`, `Waiting to transcribe`, `Writing captions`)
   (`notice`: a plain-words note about how the file was handled, or `null`; not an error).
   status: `uploading → queued → running → (cloud_asr) → done | failed | canceled | expired`.
   A restart of the service never fails a job: it goes back to `queued` and starts again.
4. Download `GET /api/jobs/{id}/outputs/{name}` (attachment). Output kinds:
   `audio`, `video`, `captions`, `transcript`, `segments`. Outputs are deleted after 24 h. A video result keeps
   the file's other audio tracks (unchanged) and its own subtitles where the container can hold them; only the
   first audio track is cleaned.

Also: `GET /api/jobs` (latest 50), `POST /api/jobs/{id}/cancel` (before it starts),
`DELETE /api/jobs/{id}` (delete outputs now).

## API keys (browser session)

`GET /api/keys` → `{keys:[{id,name,prefix,scopes,createdAt,lastUsedAt}]}`;
`POST /api/keys {name, scopes:["stream","jobs"]}` → `{id, key, prefix, scopes}` (the
only time the full key is shown); `DELETE /api/keys/{id}`.

## Live streaming

WebSocket `GET /api/stream/ws`, see [streaming-protocol.md](streaming-protocol.md).
Browsers may connect with the session cookie (same-origin only).
The `start` message may set `echoReference: true` to send what the speakers play next to the
microphone, so the server cancels the echo (included in the price; protocol section 5).

## Webhooks

`POST /api/stripe/webhook` (Stripe-Signature verified).
