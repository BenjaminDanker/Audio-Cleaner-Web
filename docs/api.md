# HTTP API (v1)

All paths are relative to the app base (`/audio` on mc-multi-world.space today,
`""` once it moves to its own domain). JSON in and out. Errors are
`{"detail": "human readable message"}` with a 4xx/5xx status; validation errors
are `422 {"detail": "Invalid request.", "fields": [...]}`.

## Auth model

- Browser: HttpOnly session cookie set by the sign-in endpoints. Every
  non-GET request must send `X-CSRF-Token: <csrfToken from GET /api/me>` and
  is rejected if its `Origin` header is present and not the site origin.
- Programs (OBS plugin, scripts): `Authorization: Bearer acw_...` API key with
  scopes `stream` and/or `jobs`. Billing, keys and account endpoints refuse
  API keys (browser session only).

## Account

| Method | Path | Notes |
|---|---|---|
| GET | `/api/me` | `{signedIn:false}` or `{signedIn, user:{id,email,displayName,isAdmin,status,tosAccepted,createdAt}, csrfToken, balance:{balance_micros,held_micros,available_micros}, autoReload:{enabled,thresholdMicros,amountCents,monthlyCapCents,hasCard,failures}}` |
| POST | `/api/account/tos` | `{version}`: must equal the current terms version (see `TOS_VERSION`, currently `2026-10-10`) |
| GET | `/api/auth/providers` | `{google:bool, email:bool, turnstileSiteKey:str|null}` |
| GET | `/api/auth/oauth/google/start` | browser navigation; ends at `/app` (or `/login?error=...`) |
| POST | `/api/auth/email` | `{email, turnstileToken?}` sends a magic link to `/login/verify#token=...` |
| POST | `/api/auth/email/verify` | `{token}` (from the URL fragment) creates the session |
| POST | `/api/auth/logout`, `/api/auth/logout-all` | |
| GET | `/api/account/export` | JSON of everything stored about the user (download as a file) |
| POST | `/api/account/delete` | `{confirm: "<the user's email>"}`; refuses while jobs run; remaining credit is forfeited (offer refunds first); signs out |

Money is in **micros** (1 USD = 1,000,000) except Stripe amounts in **cents**.

## Billing

| Method | Path | Notes |
|---|---|---|
| GET | `/api/billing/pricing` | public. `{rates:[{product,tier,label,usdPerMinute,usdPerHour,available}], topup:{minCents,maxCents,presetsCents}, free:{transcribeSecondsPerDay,transcribeMaxFileSeconds}, maxJobSeconds, paymentsEnabled}` |
| POST | `/api/billing/checkout` | `{amountCents, saveCard}` → `{url}`; redirect the browser there. Returns to `/app/billing?checkout=success|cancel` |
| POST | `/api/billing/auto-reload` | `{enabled, thresholdCents, amountCents, monthlyCapCents, consent}`; `consent:true` required to enable, needs a saved card |
| POST | `/api/billing/portal` | `{url}` Stripe customer portal (manage/remove cards, receipts) |
| GET | `/api/billing/history` | `{balance, ledger:[{id,amount_micros,kind,description,balance_after_micros,created_at}], payments:[{id,kind,status,amount_cents,refunded_cents,created_at}]}` |
| POST | `/api/billing/refund` | `{paymentId}` refunds unused credit from a payment < 14 days old |

Products/tiers: `denoise/standard`, `transcribe/local`, `transcribe/premium`,
`stream_denoise/light` (DFN3, 30 ms), `stream_denoise/low_latency` (DFN3-LL, 10 ms), `stream_captions/local`, `stream_captions/premium`.

## File jobs

1. `POST /api/jobs` body:
   ```json
   {"kind": "denoise" | "transcribe" | "denoise_transcribe",
    "denoiseTier": "standard", "transcribeTier": "local" | "premium",
    "filename": "talk.mp4", "sizeBytes": 123456, "durationSeconds": 612.4,
    "free": false,
    "options": {"audio_format": "wav|flac|mp3|m4a|opus", "caption_formats": ["srt","vtt","txt","json"],
                "language": "en" | null, "output_video": true, "embed_subtitles": true,
                "loudness_lufs": -16 | null, "atten_lim_db": 100}}
   ```
   `free:true` is only allowed for `kind:"transcribe"` with `transcribeTier:"local"`
   (daily allowance, 30 min/file). Paid jobs need accepted terms. Returns
   `{id, estCostMicros, uploadUrl}`; 402 if the balance can't cover the estimate.
   `durationSeconds` comes from the browser; the server re-measures and re-prices.
2. `PUT /api/jobs/{id}/input` with the raw file bytes as the body (`Content-Length`
   must equal `sizeBytes`). Use XHR for upload progress.
3. Poll `GET /api/jobs/{id}` every 1-2 s:
   `{id,kind,tier,free,status,stage,progress,error,inputName,durationSeconds,estCostMicros,costMicros,options,outputs:[{name,kind,bytes}],createdAt,finishedAt,expiresAt}`
   status: `uploading → queued → running → (cloud_asr) → done | failed | canceled | expired`.
4. Download `GET /api/jobs/{id}/outputs/{name}` (attachment). Output kinds:
   `audio`, `video`, `captions`, `transcript`, `segments`. Outputs are deleted after 24 h.

Also: `GET /api/jobs` (latest 50), `POST /api/jobs/{id}/cancel` (before it starts),
`DELETE /api/jobs/{id}` (delete outputs now).

## API keys (browser session)

`GET /api/keys` → `{keys:[{id,name,prefix,scopes,createdAt,lastUsedAt}]}`;
`POST /api/keys {name, scopes:["stream","jobs"]}` → `{id, key, prefix, scopes}` (the
only time the full key is shown); `DELETE /api/keys/{id}`.

## Live streaming

WebSocket `GET /api/stream/ws`, see [streaming-protocol.md](streaming-protocol.md).
Browsers may connect with the session cookie (same-origin only).

## Webhooks

`POST /api/stripe/webhook` (Stripe-Signature verified).
