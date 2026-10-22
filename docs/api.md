# HTTP API (v1)

All paths are relative to the app base, `https://audiocleaner.live` (no path prefix). The old address
`https://mc-multi-world.space/audio/api` still answers the API for older clients (OBS plugin builds, scripts) for a while;
new code should use the new one. JSON in and out, except the file upload (raw bytes) and the downloads. One base
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
| Uploads | 30 an hour per account (the file of a job). The pieces a browser sends for a timeline (`PUT .../browser/noise` and `.../browser/cues`, section 8) have their own limit: 240 an hour per account |
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
    cancel, delete, change a finished job), see "File jobs" below.
  Both bill the signed-in account's balance exactly like the website does. A key can also carry a **monthly spending
  cap** (see "API keys"). A key may read `GET /api/me` (balance, free trial left, its own cap) and the public
  `GET /api/billing/pricing`.
- Creating a job (or starting a live session) records that the account accepts the current Terms.

### Errors

| Status | When |
|---|---|
| 400 | a malformed or out-of-range option, a file type we don't take, a bad `callbackUrl`, a duration outside the limits, a size that doesn't fit the duration (over 40 MB per second, unless under 200 MB), or a timeline that is not valid (`bad_timeline`, `timeline_options`; section 8) |
| 401 | no credentials, or the key is wrong or revoked |
| 402 | the available balance can't cover the estimated price (the message says both amounts) |
| 403 | the key lacks the scope, the account is suspended, or **the key's monthly spending cap is reached** (then the header `X-Error-Code: key_spending_cap` is set and the message says how much is left and when it starts over) |
| 404 | the job doesn't exist, isn't yours (for changing a job: not one this key made), or the output isn't available |
| 408 | an upload stalled for 30 s |
| 409 | the job is in the wrong state for this (upload when not waiting for one, cancel after it started, delete while it runs), an `Idempotency-Key` was used for different details, a finished job can't be changed again (its files are gone, `job_files_gone`; see sections 7 and 8), or a timeline needs uploads from the browser that are not there yet (`needs_uploads`, section 8) |
| 411 | the upload has no `Content-Length`, or it differs from the `sizeBytes` given at creation |
| 413 | the file is bigger than allowed (`maxUploadBytes` of `/api/billing/pricing`), the upload is longer than declared, or the cues of a free subtitle part are over 1 MB |
| 422 | the JSON doesn't match the shape (`fields` lists which) |
| 429 | a rate limit above (`Retry-After` says when), or the account already has the most file jobs in progress (`code: too_many_active_jobs`, default 3; changing a finished job counts) |
| 507 | the host is short on disk right now; try again later |
| 503 | the service is on a short break (`{"detail":"maintenance"}`) or payments aren't switched on |

## Account

| Method | Path | Notes |
|---|---|---|
| GET | `/api/me` | `{signedIn:false, contactForm}` or `{signedIn, contactForm, user:{id,email,displayName,isAdmin,status,tosAccepted,createdAt}, csrfToken, balance:{balance_micros,held_micros,available_micros}, topupMaxCents, trialLiveSecondsLeft, autoReload:{enabled,thresholdMicros,amountCents,monthlyCapCents,hasCard,failures}}`. `topupMaxCents`: the largest top-up right now ($50 until a first top-up has gone through, then $500); `trialLiveSecondsLeft`: what is left of the free live trial (whole seconds). With an API key the answer is only `{signedIn, contactForm, user:{id,status}, balance, trialLiveSecondsLeft, keyLimit?}`; `keyLimit` (only for a key with a monthly cap) is `{monthlyCapMicros, usedMicros, remainingMicros, resetsAt}` |
| POST | `/api/account/tos` | `{version}`: must equal the current terms version (`tosVersion` in `/api/me`; the deploy sets it, see "Legal versions" in operations.md) |
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

Price list (`rates`; per minute of audio, counted by the second, rounded up; amounts in the response are in dollars). Every job is made with a
timeline, also one asked for with options (section 8), so each sound step that runs also has its small fixed amount for starting it:

| `product` / `tier` | What it is |
|---|---|
| `denoise` / `standard` | Denoising |
| `denoise` / `rebuild` | Voice rebuild (files up to 90 minutes) |
| `leveling` / `standard` | Leveling: charged when `options.leveling` is true (the default) on a `denoise` job; `denoiseTier: "off"` makes a job that only levels |
| `echo` / `standard` | Echo removal (with the file's second audio track): charged, by the second, when `options.echo_reference_track` is true on a `denoise` or `denoise_transcribe` job (a file with one audio track fails the job before any work, uncharged); its own line in the estimate and the ledger |
| `transcribe` / `local` | Subtitles & transcript |
| `transcribe` / `premium` | Subtitles & transcript, extra accurate |
| `transcribe_translate` / `standard` | one translation of the subtitles, **per language** |
| `convert` / `standard` | Video conversion (`options.video_out`): per minute of video at 1080p and 30 fps or smaller, charged only when the picture is made again (a new size, or a file type that can't hold it as it is); copying it into another file type is free. A bigger or faster picture counts in proportion to its pixels a second: the result's, or half the original's when that is more (4K 4×, 60 fps 2×, 4K made 720p 2×); the line's `factor` says how many |
| `stream_denoise` / `light` | Live noise removal |
| `stream_captions` / `local` | Live captions (English) |
| `stream_captions` / `premium` | Live captions, extra accurate |
| `stream_translate` / `standard` | Live caption translation, **per language** |

## File jobs

Lifecycle: `POST /api/jobs` (reserves the estimated price) -> `PUT` the file -> poll -> download. Everything needs a
key with the `jobs` scope (or a browser session). Worked example with curl:

```bash
BASE=https://audiocleaner.live ; KEY=acw_...
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
{"kind": "denoise" | "transcribe" | "denoise_transcribe" | "convert",
 "denoiseTier": "standard" | "rebuild" | "off", "transcribeTier": "local" | "premium",
 "filename": "talk.mp4", "sizeBytes": 123456, "durationSeconds": 612.4,
 "displayName": "My talk.wav",
 "callbackUrl": "https://example.com/hooks/audio-cleaner",
 "options": {"audio_format": "wav|flac|mp3|m4a|opus", "caption_formats": ["srt","vtt","txt","json"],
             "language": "en" | null, "output_video": true, "embed_subtitles": true,
             "loudness_lufs": -16 | null, "atten_lim_db": 100, "echo_reference_track": false,
             "noise": "auto|light|heavy", "leveling": true, "translate": ["es", "de"],
             "automatic": false, "video_out": {"format": "mp4", "size": 720, "source": {"codec": "wmv2", "width": 1920, "height": 1080, "fps": 30}} | null},
 "timeline": {"version": 1, "seconds": 612.4, "layers": {"noise": [{"start": 100, "end": 160.5, "strength_db": 40, "remover": "auto"}]}}}
```

Answer: `{id, estCostMicros, uploadUrl}` (and `"maximum": true` with `options.automatic`, see "Automatic settings"; and `"needs": [{kind, start, end}, ...]` for a timeline with parts the browser makes itself, see below). `uploadUrl` is relative to the base, **without** a leading slash
(`api/jobs/<id>/input`). Fields:

| Field | Rules |
|---|---|
| `kind` | `denoise` (clean the sound), `transcribe` (subtitles and transcript), or both; or `convert`: only `options.video_out` (no tiers, no timeline) |
| `denoiseTier` | required for `denoise` kinds: `standard` (Denoising), `rebuild` (Voice rebuild, its own price: the whole voice is rebuilt, for room echo, a muffled voice, phone-quality or damaged sound, or distortion from recording too loud; `atten_lim_db` and `noise` don't apply to it; it also brings back the high sounds a phone call or a very small file has lost; files up to 90 minutes), or `off` (no noise removal: only Leveling, so `options.leveling` must then be on). Leveling (`options.leveling`, on by default) is its own line on the bill. |
| `transcribeTier` | required for `transcribe` kinds: `local` (Standard) or `premium` (extra accurate; audio goes to a speech partner, see the privacy policy) |
| `filename` | its extension must be one we read: audio `wav rf64 w64 mp3 mp2 mpa m4a m4b m4r aac flac ogg oga opus spx wma aiff aif aifc caf au snd amr ac3 eac3 dts mka wv ape tta mpc`, video `mp4 m4v mov webm mkv avi wmv asf flv f4v 3gp 3g2 ts mts m2ts mpg mpeg vob mxf dv ogv rm rmvb`; anything else is `400 Unsupported file type.` Only the extension is used; the name never becomes a path |
| `sizeBytes` | 1 byte up to `maxUploadBytes` (4 GiB; `413` above); at most 40 MB per second of `durationSeconds` unless under 200 MB |
| `durationSeconds` | 0.5 s up to 14,400 s (4 h); up to 5,400 s (90 min) with `rebuild`. From the client: the server measures the file after the upload and prices what it finds, within the API key's monthly limit. The measured length may be at most 10 % (+5 s) longer than this, else the job fails before any work and nothing is charged |
| `displayName` | optional, shown as the job's name and used for result file names. Up to 300 characters sent, cut to 120 after control, bidirectional and zero-width characters are removed |
| `callbackUrl` | optional, see "Finish notifications" |
| `timeline` | optional: only the parts it names are processed, and charged (section 8). Replaces `denoiseTier` and the options that say what is done to the sound (`atten_lim_db`, `noise`, `leveling`, `loudness_lufs`, `echo_reference_track`, `automatic`): leave them out (`400 timeline_options`). Needs `kind` `denoise` or `denoise_transcribe`. The subtitle options work as always, unless the timeline has a `subtitles` layer (timeline version 2): then every part says its own quality, language and translations, and `transcribeTier`, `options.language` and `options.translate` must be left out (`400 timeline_options`; a null language or an empty `translate` counts as left out), and `kind` `denoise_transcribe` needs no `transcribeTier`. Without a timeline the whole file is processed per the options, as before |
| `free` | `true` is refused (400): jobs here are always paid. The free tools run in the visitor's browser |

`options` (the whole object must be under 2000 characters; unknown keys are ignored):

| Option | Default | Range and meaning |
|---|---|---|
| `audio_format` | `flac` | `wav`, `flac`, `mp3`, `m4a`, `opus` |
| `caption_formats` | `["srt","vtt","txt"]` | any of `srt`, `vtt`, `txt`, `json` (`json` gives `segments.json`: every line with its times, kind `segments`) |
| `language` | `null` (detect) | the spoken language: a 2-3 letter code (`en`, `es`, `haw`) or null |
| `translate` | `[]` | up to 3 target languages (two-letter codes from the translation list, not the spoken one); needs a transcription kind. The subtitles and transcript are also made in those languages (`<stem>.<lang>.srt\|vtt`, `<stem>.<lang>-transcript.txt`), priced per language. A language that can't be made is left out, said in `notice` and not charged |
| `output_video` | `true` | for a video input: give a video back (with the cleaned sound). `false`: audio only |
| `embed_subtitles` | `true` | put the subtitles into the video; needs the whole video uploaded |
| `video_out` | `null` | the video result in another file type or picture size: `{"format": "mp4"\|"mov"\|"mkv"\|"webm", "size": 480\|720\|1080\|1440\|2160\|null}` (`size`: the shorter side in pixels, the shape kept; `null` keeps the size), and optionally `source`: what you read of the video, `{"codec": FFmpeg's codec name ("h264", "wmv2", "vc1", ...), "width", "height", "fps"}`, so the price is known before the upload. Needs the whole video uploaded and `output_video`. The picture is copied when the new file type can hold its codec at its size (free; MKV holds any); otherwise it is made again, priced as Video conversion (above, with its size `factor`). Without `source.codec` a change of file type alone can't be priced up front: the estimate lists the line with `"maybe": true` (not in the total); a line that is free because the picture is copied says `"copied": true` (micros 0). A sound (a file with no picture) asked for as a video gets a plain black picture (one frame a second) with its sound, free. When the file is read the server works it out again from the file itself and charges that; if it costs more than was held, the hold grows (the job fails, uncharged, when the balance is short) |
| `loudness_lufs` | `-16` | target loudness, a number from -30 to -9 (`400` outside it), or `null` to leave the level as it is. Podcast about -16, online video -14, broadcast -23 |
| `atten_lim_db` | `100` | how much noise may be removed: any number, clamped to 6-100 dB. The website's strengths are 12 (gentle), 24, 40 and 100 (max). Doesn't apply to `rebuild` |
| `echo_reference_track` | `false` | must be a boolean. Removes speaker echo from the first audio track (the mic) using the second track (the desktop sound), before noise removal; subtitles are written from the echo-free sound. Charged at its own price (`echo` / `standard` in the price list, about a third of Denoising's) on top of the job's other steps; a job made before this price existed keeps the prices it was made with (no echo line, nothing charged for it). A file with only one audio track fails the whole job as soon as it is read, before any work, uncharged (echo removal was part of what was asked for); an edit of such a job that asks for it is refused (`400`). The website doesn't offer it for such a file |
| `noise` | `auto` | `auto` (measured per file), `light`, `heavy` (better when the noise is nearly as loud as the voice). Doesn't apply to `rebuild` |
| `leveling` | `true` | Leveling (its own price): the clarity chain after the noise removal (level matching, high-pass, compressor, loudness to `loudness_lufs`, limiter); `false` returns the cleaned voice as it is (`loudness_lufs` then does nothing). Must be a boolean. |
| `automatic` | `false` | `true`: the server listens to the file and decides what you did not give yourself, like the website's "Automatic best settings" (next section). Must be a boolean; needs `kind` `denoise` or `denoise_transcribe` and `denoiseTier: "standard"` |

There is no option for restoring the high sounds a recording has lost (phone calls, voice messages, very small files): the fix is
`denoiseTier: "rebuild"`, which makes them again with the rest of the voice. An old client's `restore_highs` is an option like any
other this page does not list: it is ignored (not stored, not priced, not reported). `denoiseTier: "off"` needs `leveling` on.

**A timeline with parts the browser makes itself.** A noise part with `"browser": true` is cleaned by the visitor's browser (free), and a subtitle part with quality `free` is transcribed by it (free); both are sent to the server as uploads (section 8, "Uploads from the browser"). A job created with such a timeline answers with `needs`, the list of those uploads, `[{"kind": "noise" | "cues", "start": 100, "end": 160.5}, ...]` in time order, and so does `GET /api/jobs/{id}` (`needs` is there for any job whose timeline has such parts, `[]` once everything is in; it is left out for every other job). **The job is not queued until the file (`PUT .../input`) and every upload of `needs` are in**, in any order; until then it stays `uploading` (each arrival restarts its upload window of an hour) and is cancelled like any upload that does not finish. `needs` lists the parts of the timeline as the server read it (neighbours with the same settings are one part; browser noise parts with different strengths are not).

#### Automatic settings: `options.automatic: true`

Send `"automatic": true` (with `"denoiseTier": "standard"`) and the server decides the settings from the file itself, the way the
website's "Automatic best settings" and the desktop app do (one and the same decision for all three):

| Decided | How |
|---|---|
| noise strength (`atten_lim_db`) | the gentlest of 12, 24, 40 and 100 dB that takes the noise out |
| noise type (`noise`) | the stronger remover (`heavy`) when the noise is nearly as loud as the voice, else `light` |
| Voice rebuild (`denoiseTier`) | `rebuild` instead of `standard` when the voice sounds echoey, muffled, like a phone call or distorted, or is unclear; then the strength and the noise type do not apply |
| Leveling (`leveling`) | on |

**What you send wins.** An option you give yourself (`atten_lim_db`, `noise` as `light` or `heavy`, `leveling`) is
kept as you sent it and is not decided; `noise: "auto"` is the automatic choice. Everything else (`audio_format`, `loudness_lufs`,
captions, `translate`, ...) works as usual. A strength or a noise type means you asked for Denoising, so the
voice is then never rebuilt (nor when `echo_reference_track` is on).

**Price.** The job is held for the most the decision could cost: Voice rebuild + Leveling (the dearer way, wherever a rebuild is
possible: not for a file over 90 minutes, nor when you gave a strength or a noise type), else Denoising + Leveling, less what you switched
off yourself. `POST /api/jobs` answers
`"maximum": true` and `estCostMicros` is that hold; the estimate (section 2) says `"maximum": true`, lists each way it can end in
`outcomes` and the options you fixed in `fixed`. When the file has been listened to, the hold shrinks to what will be charged, and the
charge is for the steps that ran. The key's monthly spending limit is checked against the maximum.

**What was chosen.** The job (section 4) has `chosen`, `null` until the file has been listened to (it waits in the queue and starts like
any job; `label` is "Automatic settings" and `maximum` is `true` until then):

```json
"chosen": {"denoiseTier": "standard", "strength": "balanced", "attenLimDb": 24, "noise": "light", "heavy": false,
           "leveling": true, "rebuild": false, "rebuildWanted": false, "fixed": [],
           "reasons": ["There is some background noise: Denoising takes it out.", "Balanced is enough to take it out while keeping the voice natural.",
                       "Leveling evens out the voice and sets the loudness."]}
```

`reasons` says what was heard and what was done in plain words (and which options were yours). The job's `options`, `denoiseTier`,
`label` and `costMicros` are the decided ones, so it reads like a job you made with those settings. For a rebuild `strength`,
`attenLimDb` and `noise` are `null`.

**When a rebuild is wanted but cannot be done** (the file is over 90 minutes, voice rebuilding is paused, you gave Denoising settings
or `echo_reference_track`) the file is cleaned with Denoising instead, with the strength the decision gives:
`rebuildWanted` is `true`, `rebuild` is `false`, the last reason says why, and `notice` repeats it. The decision is made before the job is
sent to the machine that rebuilds voices, so a rebuild waits in that queue like any Voice rebuild job. The measurement uses the file as
uploaded (before echo removal), about a minute of it spread over the file, in a separate step of a few seconds.

**Changing it afterwards.** The choices made are the job's settings: change them like those of any job (section 7). `automatic` is not
available there (`400`): send the settings you want instead, for example `denoiseTier: "rebuild"`.

A video result keeps the input's container when it can hold the streams (`mp4 mov m4v mkv webm`); other containers
come back as MKV, unless `options.video_out` asks for another file type or size. Other audio tracks are kept unchanged and subtitles
go in where the container can hold them; only the first audio track is cleaned (a conversion alone keeps the sound, made in the new
file type's own kind when it can't hold it as it is). The website converts file type and picture size in the browser, for free, and asks
for `video_out` only for a video it can't convert itself.

`embed_subtitles` makes the service put the subtitles into the video, which needs the whole video uploaded. The
website only asks for it for a video the browser can't open (it goes up whole anyway); otherwise it uploads the
sound only and puts the subtitles into an MKV in the browser afterwards.

Errors specific to creating: `400` (see the table above), `402` (balance below the estimate), `403` (key cap),
`409` (Idempotency-Key reused with different details), `429` (the active-job limit, 8 GiB of uploads waiting, job creation
allowance), `507`.

**Jobs in progress.** An account can have at most `ACW_MAX_ACTIVE_JOBS` file jobs at once (default 3): jobs that are
uploading, waiting or running, which includes changing a finished job (section 7) and counts across every key and
the website. One more is refused with `429`, the header `X-Error-Code: too_many_active_jobs`, `Retry-After`, and
`{"detail": "You have 3 files being processed. Start this one when one of them is done.", "code": "too_many_active_jobs"}`.
`GET /api/jobs` answers `limits: {maxActiveJobs, retentionSeconds, maxRetentionSeconds}` next to the jobs. The per-key monthly spending cap
(`403`, `key_spending_cap`) and the 8 GiB of uploads waiting are separate limits and unchanged.

**Idempotency.** Send `Idempotency-Key: <1 to 128 visible ASCII characters>` and the same key from the same account within
24 hours returns the same job instead of making (and holding money for) another: the answer is the original
`{id, estCostMicros, uploadUrl}` plus `"idempotentReplay": true` and the header `Idempotent-Replayed: true`. The
details must match (the whole request body is compared): the same key with a different body is `409`. A request that
failed (402, 429, ...) doesn't use the key up. Use a new key per file; a good one is a hash of the file name and size.

### 2. Estimate: `POST /api/jobs/estimate`

Same `kind`, `denoiseTier`, `transcribeTier`, `durationSeconds`, `options` and `timeline` as creating (no file fields); makes no
job and holds nothing. Applies the same checks and limits, so it answers the same `400`s. With a `timeline` the price is that of its parts
(section 8), `lines` carry `seconds` and `fixedSeconds`, and the answer repeats the checked `timeline`. The subtitle parts of a version 2 timeline are lines
too: `transcribe` (Standard parts), `transcribe_accurate` (Accurate parts) and `translate` (`quantity` = the number of languages); parts the browser
makes (free subtitles, `"browser": true` noise) cost nothing and have no line.

```json
{"estCostMicros": 102167, "estCostUsd": 0.102167, "durationSeconds": 612.4, "maxSeconds": 14400,
 "lines": [{"item": "denoise", "label": "Denoising", "quantity": 1, "usdPerMinute": 0.01, "micros": 102167}],
 "options": {"...": "the options with the defaults filled in"},
 "availableMicros": 5000000, "affordable": true,
 "keyLimit": {"monthlyCapMicros": 500000, "usedMicros": 0, "remainingMicros": 500000, "resetsAt": 1793491200}}
```

`lines` has `denoise`, `leveling`, `transcribe` and `translate` (`quantity` = number of languages) as they apply; the micros add
up to `estCostMicros`, which is exactly what `POST /api/jobs` would reserve for the same details. `keyLimit` only when
the key has a monthly cap.

With `options.automatic` the price is the most the decision could end in, not what it will: `"maximum": true`, `lines` are the dearest
outcome's, `outcomes` is `[{label, denoiseTier, estCostMicros}]` for every way it can end (only Denoising for a file over 90 minutes, or
when you gave Denoising settings), and `fixed` the options you gave yourself. The charge is for what ran and can be less.

### 3. Upload: `PUT /api/jobs/{id}/input`

The raw file bytes as the body (no JSON, no multipart). `Content-Length` is required and must equal `sizeBytes`
(`411` otherwise). The upload must **start within 10 minutes** of creation; it may then take up to an hour after the last
bytes arrive (the clock restarts while data flows). Answers `{ok:true}`. Errors: `409` if the job isn't waiting for an
upload (already uploaded, canceled, expired), `408` after 30 s without data, `413` if more arrives than declared,
`400` if less, `507` if the host is short on disk. Use XHR in a browser for progress. If it fails, create a new job.

### 4. Poll: `GET /api/jobs/{id}` (and `GET /api/jobs`)

`{id,kind,tier,label,free,status,stage,progress,error,inputName,durationSeconds,estCostMicros,costMicros,options,denoiseTier,transcribeTier,notice,queuePosition,queueWaitSeconds,estimateSeconds?,outputs:[{name,kind,bytes,lang?}],createdAt,finishedAt,expiresAt,editedAt,hardDeadlineAt,edit?,needs?,redo?,callback?,chosen?,maximum?}`

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
- `estimateSeconds`: while `queued` or `running`, a rough length of the whole job (for a "time left" before `progress` has moved).
- `expiresAt`: when the upload, the editor's copies for changing the job (the cached blocks of section 8) and the results are all
  deleted: 2 hours after the job's **last edit** (`ACW_JOB_RETENTION_MINUTES`; finishing a run, saving a timeline, a quote, a process,
  and an export each count as an edit, a read or a preview does not), and never later than `hardDeadlineAt`, 6 hours after the
  upload arrived (`ACW_JOB_MAX_RETENTION_MINUTES`). `editedAt`: the last edit. The housekeeping sweep runs every 30 to 60 seconds, so
  the deletion is accurate to about a minute. While an edit run is `queued` or `running`, `expiresAt` is `null` (as for any running job) and
  gets its new value when the run ends.
- `edit` (every job but a conversion): `{available, hasTimeline, run, lastRun, suggestions}`: `available` when the job is `done` (it can be
  edited, section 8), `hasTimeline` once it has a timeline of its own, `run` the operation (`process`, `export`, `suggest`) while one is
  queued or running (the job's `status` is then `queued` / `running` again, its earlier `outputs` stay listed), `lastRun`
  `{op, ok, at, error?}` of the last run (a run that failed or was canceled leaves the job as it was: `ok: false` and why),
  `suggestions` whether it has any.
- `needs` (jobs with audio steps, only when the timeline has parts the browser makes): what the browser still has to upload, `[{kind, start, end}]`, empty once everything is in (section 8, "Uploads from the browser").
- `callback`: only when a `callbackUrl` was given, see below.
- `chosen`, `maximum`: only for a job made with `options.automatic`: what was decided and why (`null` until the file has been listened to),
  and `true` while `estCostMicros` is still the most it can cost. See "Automatic settings".

`GET /api/jobs` returns `{jobs:[...], limits:{maxActiveJobs, retentionSeconds, maxRetentionSeconds}}`: up to 50 of your jobs, newest first: the
ones in progress and the ones whose files are still there (a finished job is listed until its files are deleted, `expiresAt`; failed and canceled jobs,
which keep no files, are listed for 2 hours after they ended), in the same shape: one request polls them all.

### 5. Download: `GET /api/jobs/{id}/outputs/{name}`

Streams the file as an attachment (`name` from `outputs`). `404` until the job is `done`, and after the results expire. Download
promptly: they are deleted, together with the upload and the copies kept for changing the job, 2 hours after the job's last edit (`done` is the first; at the latest 6 hours after the upload), or at once with `DELETE`.

### 6. Cancel and delete

`POST /api/jobs/{id}/cancel`: while `uploading` or `queued`; releases the reservation. `409` once it has started, and
once its extra-accurate transcription is done (only packaging is left: "almost done"). `DELETE /api/jobs/{id}`: deletes everything
of the job now (the upload, the editor's copies and the results); `409` while the job is active. Both answer `{ok:true}`.

### 7. Change a finished job

Every job is made with a timeline, also one asked for with options: they become the timeline they stand for (each sound step on the whole
file, the subtitles as one part over the whole file; for a file whose real length differs from `durationSeconds`, the timeline is made
again for its real length once it is read). A finished job is changed by editing that timeline (section 8): ask for a quote, then process
it; only what is missing runs, and only that is charged.

### 8. Edit a job part by part: timeline, quote, process, export, preview, suggestions

A job with audio steps (Denoising, Voice rebuild, Leveling, echo removal) is **editable** once it is `done`. It holds a **timeline**: what is done
to which part of the recording. After a run you change the timeline, **quote** what that needs and costs, **process** it (only what is missing
is made), and **export** the result files. Nothing here applies to a job that only makes subtitles (`409 job_not_editable`). The same calls
are what the website's editor uses. All of them need the `jobs` scope (or a browser session); a key can only use them on jobs it made itself,
a browser session on any job of its account; anything else is `404 job_not_found`, like a job that doesn't exist.

**The timeline** (JSON, `version` 1, or `version` 2 when it has subtitles or a part cleaned in the browser):

```json
{"version": 1, "seconds": 612.4, "layers": {
   "echo":     [{"start": 0, "end": 612.4}],
   "noise":    [{"start": 0, "end": 612.4, "strength_db": 40, "remover": "auto"}],
   "rebuild":  [{"start": 130.5, "end": 160}],
   "leveling": [{"start": 0, "end": 612.4, "loudness_lufs": -16}]}}
```

| Layer | Meaning and settings |
|---|---|
| `echo` | remove the speaker echo (needs the file's second audio track); no settings |
| `noise` | remove the noise: `strength_db` (above 0, at most 100; 12 gentle, 24 balanced, 40 strong, 100 maximum; the words `gentle balanced strong max` are accepted), `remover` `auto` (default: decided per part when it runs), `light` or `heavy`. Version 2: `"browser": true` instead of these: the visitor's browser cleaned that range itself, free, and uploads the result (see "Uploads from the browser"). It has no remover (it may be sent, it is not kept). A `strength_db` that is sent is **kept and returned as sent**, so a page can show the strength the person chose after a reload, but it is never used in processing or pricing: changing it alone changes nothing in the sound and costs nothing. Neighbouring browser parts with different strengths (or one with a strength and one without) are separate parts, each with its own upload |
| `rebuild` | rebuild the voice; no settings. **Where noise removal and Voice rebuild both cover a moment, Voice rebuild wins** and the noise removal there is not run |
| `leveling` | even out the voice and set the loudness: `loudness_lufs` from -70 to 0, or `null` to keep the level |
| `subtitles` (version 2) | make the subtitles of the part: `quality` `free` (the browser transcribes it and uploads the cues), `standard` or `accurate` (the server transcribes it), `language` (a code, or `null` to detect it), `translate` (up to 3 codes, none the part's own language, only with `standard` or `accurate`). See "Subtitles per part" |

Times are seconds (kept to the millisecond). The parts of a layer never overlap: a part painted over others replaces them (in the JSON the later part wins),
neighbours with the same settings are one part. A layer that is missing has no parts. The default timeline of a job is every step it was made with on the
whole file (what it did), and a job made without a timeline of its own shows exactly that (without subtitles: those are the job's whole-file options, unless it has a `subtitles` layer).
**Version.** `version` 1 is what the first version of this format says; `version` 2 adds the `subtitles` layer and `"browser": true` noise parts, nothing else. Both are accepted. The server
answers with version 2 only for a timeline that uses one of the two, so every other timeline is written back exactly as before (older programs read only version 1); a version 1 document that
uses them is `400 bad_timeline`. A part cleaned in the browser may not overlap an echo part: echo removal runs before the noise removal and the browser can't do it
(`400 bad_timeline`, "Echo removal can't be used on a part cleaned in the browser"); where Voice rebuild overlaps it, Voice rebuild wins, as over any noise removal.
Checked strictly (`400 bad_timeline`, the message says what): the format and version, unknown keys, layers and settings, numbers, every part
inside the recording (a hair, 0.5 s, past the end is cut to it), at most **500 parts per layer** and at most **200 KB** of JSON (`413` past 256 KB).
Where a timeline is accepted: in `POST /api/jobs` and `POST /api/jobs/estimate` (`timeline`: only its parts are processed and charged), `PUT .../timeline`,
and in the bodies below.

| Call | Body | Answer |
|---|---|---|
| `GET /api/jobs/{id}/timeline` | | `{timeline, rendered, seconds, channels, secondTrack, cached, needs?, suggestions, run, lastRun, expiresAt, editedAt, hardDeadlineAt, cacheBytes, preview}`: `timeline` the working timeline (what was last saved, processed or exported; else the one the job was made with), `rendered` the timeline the result files hold (`null` if unknown), `seconds` the recording's length, `secondTrack` whether echo removal is possible, `cached` what is cached per layer as `[[start, end], ...]` seconds: `echo`, `noise` (either remover; `light` and `heavy` apart) and `rebuild` (Leveling is not cached, it is made again at each render) and, for a timeline with subtitle parts, `subtitles` (the parts whose text exists: made, or uploaded), `needs` (only when the timeline has parts the browser makes: the uploads it still waits for, `[]` once they are in), `suggestions` as below or `null`, `preview` how to read the stretches below (`{layers, samples, channels, sourceScale, stretchMaxSeconds}`, `null` before the job has its recording copy). Works while the job runs too. **Not an edit** |
| `PUT /api/jobs/{id}/timeline` | `{timeline}` | `{timeline, expiresAt, needs?}`: the checked timeline as saved (merged, clamped); `needs` as above when it has parts the browser makes. The job must be `done` and idle (`409 job_busy` while it is processed). **An edit** |
| `POST /api/jobs/{id}/quote` | `{timeline?}` (omitted: the saved one) | the price, below. Holds and runs nothing. **An edit.** Not a job: it does not count toward the active-job limit |
| `POST /api/jobs/{id}/process` | `{timeline?, maxCostMicros?, audioFormat?}` | `{id, run: {id, op: "process"}, estCostMicros, holdMicros, free, approximate, expiresAt}`: the job goes back to `queued` (poll `GET /api/jobs/{id}`; its earlier outputs stay listed until the new ones replace them). The timeline given becomes the working timeline. `audioFormat`: the file type of the result (as `options.audio_format`, which it also sets). A timeline whose browser parts or free subtitles have no upload yet is `409 needs_uploads` (below). **An edit.** Counts toward the active-job limit |
| `POST /api/jobs/{id}/export` | the same | the same with `op: "export"`: the result files rendered from the cache alone, **free**, for a quote of nothing. A timeline that has something to make is `409 needs_processing` with `quote` in the answer. **An edit.** Counts toward the active-job limit |
| `GET /api/jobs/{id}/suggestions` | | `{suggestions: {seconds, whole: {rebuild, remover, strength}, suggestions: [{start, end, kind: "part"\|"clipping", rebuild, noise: {strength_db, remover}\|null, reasons: [plain words]}], madeAt}\|null}` |
| `POST /api/jobs/{id}/suggestions` | | looks for suggestions of a job that has none: a free run (`op: "suggest"`, the answer of `process`), poll the job. Counts toward the active-job limit |
| `GET /api/jobs/{id}/preview/{source\|echo\|noise\|rebuild}?start=&seconds=` | | a stretch of what the job holds, below |
| `GET /api/jobs/{id}/peaks/{source\|echo\|noise\|rebuild}` | | the loudness outline of a layer, below |
| `PUT /api/jobs/{id}/browser/noise?start=&end=` | FLAC | the audio a browser cleaned for a noise part, below |
| `PUT /api/jobs/{id}/browser/cues?start=&end=` | `[{start, end, text}]` | the cues a browser transcribed for a free subtitle part, below |

**Quote.** `POST .../quote` answers

```json
{"estCostMicros": 7158, "estCostUsd": 0.007158, "free": false, "unchanged": false, "needsProcess": true, "approximate": false,
 "lines": [{"item": "denoise", "label": "Denoising", "quantity": 1, "seconds": 31, "fixedSeconds": 2.328, "usdPerMinute": 0.009, "micros": 5000},
           {"item": "leveling", "label": "Leveling", "quantity": 1, "seconds": 31, "fixedSeconds": 98.462, "usdPerMinute": 0.001, "micros": 2158}],
 "steps": {"light": {"blocks": 2, "seconds": 31}, "leveling": {"blocks": 1, "seconds": 31}}, "asked": {"denoise": 30.2, "leveling": 30.2},
 "timeline": {"...": "the timeline as checked"}, "expiresAt": 1760600000, "availableMicros": 5000000, "affordable": true, "keyLimit": {"...": "only for a key with a cap"}}
```

`steps` is what has to be made: per step the missing blocks and the whole seconds billed; `asked` the seconds of the recording each step is asked for;
`unchanged` says the result files already hold this timeline (nothing to do, price 0); `approximate` that a part with `remover: "auto"` is not decided yet
(it counts as the light remover; it is decided when it runs and charged for what runs, never more than the hold, which covers the dearer remover);
`free` that the price is 0. `lines` add up to `estCostMicros`. For a timeline with subtitle parts `lines` also has `transcribe`, `transcribe_accurate` and `translate`, `steps.subtitles` is
`{parts, seconds}` (the parts that have text or translations to make, and their whole seconds), `asked.subtitles` the seconds of the recording they cover, and `needs` lists the uploads the timeline still waits for (only when it has parts the browser
makes). A change of subtitles alone leaves the sound alone (it is not rendered again, nor charged) and `unchanged` is true only when the result files hold these subtitles too (the same parts, and the same cues uploaded for the free ones).

**Price.** For each step that runs, `rate per second x (the whole seconds of the parts asked for, rounded up, in the blocks that have to be made
+ the step's fixed per-run amount expressed as equivalent seconds)`, each step rounded up to a micro. The context a model also reads around a part is
never billed, there is **no minimum charge**, and whatever the cache holds is free. Leveling is charged where it runs on audio that changed since
the last render (a new strength under Leveling changes it; a Leveling part with the same settings over the same audio is not charged again).

| Step (line) | Rate | Fixed per run | Equivalent seconds | Fixed in micros |
|---|---|---|---|---|
| `echo` (Echo removal) | 2,600 micros a minute ($0.156 an hour) | 0.12 CPU-s, 0.034 per second | 3.53 | 153 |
| `light` (Denoising) | 9,000 ($0.54 an hour, the price list's Denoising) | 0.27 CPU-s, 0.116 per second | 2.33 | 349 |
| `heavy` (Denoising) | 9,000 | 2.9 GPU-s, 0.386 per second | 7.51 | 1,127 |
| `rebuild` (Voice rebuild) | 30,000 ($1.80 an hour) | 4.3 GPU-s, 0.203 per second | 21.18 | 10,591 |
| `leveling` (Leveling) | 1,000 ($0.06 an hour) | 1.28 CPU-s, 0.013 per second | 98.46 | 1,641 |

**Subtitle parts** are priced per part, from the price list (`transcribe` / `local`: Standard, 4,000 micros a minute; `transcribe` / `premium`: Accurate, 15,000; `transcribe_translate`: 4,000 per language), with no fixed
amount: for each part, `the quality's rate x the part's whole seconds (rounded up)`, plus `the translation rate x the same seconds` for each language that has to be made, each rounded up to a micro. Free parts and
parts cleaned in the browser cost nothing. Whatever the job holds already is not made or charged again: the text of a part with the same range, quality and language is kept (a new language costs only
its translation), and a translation into the language a part turns out to be in is not made. A translation that can't be made is left out, said in `notice` and not charged. The rates are pinned with the job (a run is held at the
quote's), and the charge is for what was made, never more than the hold.

(The rates are the price list's, `GET /api/billing/pricing`; echo removal has its own row there, `echo` / `standard`, and a job made the old way is
charged it per second when echo removal is on, in its own line. The fixed amounts are the measured cost of starting a step in its own sealed process, at the step's own rate.) The whole file with
the default steps costs today's price plus about 2 % of ten minutes, 0.4 % of an hour: 600,000 micros an hour of Denoising + Leveling becomes
601,992. One ledger line per process run: `Edited (Denoising + Leveling): talk.wav (0.5 min)` with the minutes processed; an export that costs
nothing has no line; the hold is made at the quote and the charge is for what ran, at most the hold.

**Run.** A process or export is the job going back through the normal queue, sandbox and workers (Voice rebuild and the heavy remover on the card,
`400` with "paused" when it is down and the timeline needs it): the missing blocks are made and cached, then the timeline is rendered (joins between
parts, Leveling per part) into the result audio. A video the server made (the whole video was uploaded, or a sound asked for as a video) is made
again with the new sound or subtitles, the way the first run made it (its picture made again is charged again: Video conversion, in the quote's
lines); otherwise audio is what the server makes (the website puts it back into the video itself). The new audio is written under a temporary name and takes the old result's place only when the whole run is complete, so a run that fails or is stopped leaves the old audio and the record of it exactly as they were (the same goes for the job's other results). The captions and transcript of the job stay as they were, unless the run changes the subtitles of a job that makes them part by part: then they are put together again from all the parts (see below). If a run fails or is canceled the job is as it was
(`edit.lastRun`), the money is released and the files stay.

**Cache.** `blocks/` in the job's folder: every block a model step made, 24-bit FLAC, keyed by the step, the model, the audio it read and where it is.
A whole-file first run fills it (the blocks are on a 30 s grid for the noise removers and echo removal, 14 s windows for Voice rebuild), an edit only
makes what is missing. It is counted in the job's disk promise, deleted with the job, and takes about 0.3 GB an hour per channel and step.

**Preview stretches.** `GET .../preview/{layer}?start=<seconds>&seconds=<n>` (24-bit FLAC, 48 kHz, one channel per channel of the recording, at most 30 s,
`start` to the nearest sample) for a page that mixes strengths and layers itself: `source` (the recording), `echo` (the echo-free voice), `noise` (the noise
removal at full strength: the remover of the timeline's noise part there, else the one the job used) and `rebuild` (the rebuilt voice). The same `start` and
`seconds` give stretches that line up sample for sample. Only what is cached: samples that are not are silence and listed in `X-Stretch-Missing`
(`[[from, to], ...]` seconds from the start of the stretch); none cached is `404 not_cached` with `missing`. Headers: `X-Stretch-Start`, `X-Stretch-Samples`,
`X-Total-Samples`, `X-Stretch-Scale` (the samples are the stretch times it: a loud stretch is brought down to fit a 24-bit file; divide by it),
`X-Stretch-Layer`. For a strength `a = 10^(-dB/20)` the page plays `(1 - a) * noise + a * (the echo-free voice, or source)`. Same sign-in, same
rate limit (a burst of 120, then 2 a second), and `503` when busy.

**Suggestions.** The first run of a recording up to `ACW_SUGGEST_MAX_MINUTES` (60) also looks for parts that sound different from the rest (noisier,
quieter, phone-like, distorted, clipped) and stores what fits each better, with the reasons in plain words, ranges and proposed settings (about 0.22
s of work per second of audio). The server's are the record; a longer recording gets them on request (`POST .../suggestions`).

**Subtitles per part.** A version 2 timeline can have a `subtitles` layer, in a job made with `kind` `denoise` or `denoise_transcribe` (no `transcribeTier`, no
`options.language` or `options.translate`; a job whose subtitles were set for the whole file can't have subtitle parts, `400 timeline_options`). Each part is
`{start, end, quality, language, translate}`: `quality` `free`, `standard` or `accurate` (required), `language` a code of the website's list (`en es fr de it pt nl pl ru uk tr ar hi ja ko zh sv da fi no cs el he id vi hu ro th`; `standard` and `accurate` also take any other 2 to 3 letter code their engines know) or `null` to detect it,
`translate` up to 3 codes from the same list, never the part's own language, only with `standard` or `accurate` (`400 bad_timeline` with the reason otherwise); neighbours with the same settings are one part. The parts may sit
on a recording that is otherwise not touched: a timeline with subtitle parts and no sound parts makes no audio file. Standard and Accurate parts are transcribed by the server part by part, with a little sound around
each (2 s each side, for context; what is said there is not part of it) and the cues are cut to the part and put in file time. Each language of `translate` gets the cues of the parts that asked for it. A free part is transcribed
by the browser, which uploads its cues. All parts are put together into the job's usual caption files, with the same names and formats as always (`options.caption_formats`: `<stem>.srt`, `<stem>.vtt`,
`<stem>-transcript.txt` with the original text of every part in time order, and `<stem>.<lang>.srt|vtt` / `<stem>.<lang>-transcript.txt` for each translated language); their `lang` is the language most of the speech is in. A change to one part (or its
language, quality or translations) makes only what that part needs; the rest is kept (see "Price").

**Uploads from the browser.** A timeline can have parts the visitor's browser makes itself, free: noise parts with `"browser": true` (the browser cleaned that range) and subtitle parts with quality `free` (the browser transcribed it).
The server needs the result of each, and says so: `needs` is `[{"kind": "noise" | "cues", "start", "end"}]`, in the create answer (a job made with such a timeline is **not queued until the file and every upload are in**), in `GET /api/jobs/{id}`,
in `GET .../timeline`, in `PUT .../timeline` and in a quote. `409 needs_uploads` (`code` and `X-Error-Code`, with `needs` in the answer) answers a `process` or `export` whose timeline needs uploads it does not have; the timeline of that
request becomes the job's working timeline, so the client uploads what `needs` lists and calls `process` again. Both uploads must match a browser noise part / free subtitle part of the job's **working timeline exactly**
(`start` and `end` in seconds to the millisecond, as in `needs`), else `400`; uploading again replaces the earlier one. They are kept in the job's folder (`browser/`), deleted with the job, and need the job to be `uploading` or
`done` and idle (`409 job_busy` while it is queued or running; the same ownership rules as the other calls here). A finished job's upload is an edit (its files are kept 2 hours from now); the limit is 240 uploads an hour per account (`429`).

| Call | Body | Rules |
|---|---|---|
| `PUT /api/jobs/{id}/browser/noise?start=&end=` | FLAC, 48 kHz, with the channels of the job's audio, exactly the part's range | `exactly` means `round(end x 48000) - round(start x 48000)` samples (the header says how many; the end of the recording cuts the last part short); not 48 kHz, another number of samples or other channels is `400` and the message says which. At most `samples x 8 + 1 MB` bytes (`413`). The file is decoded and used only when the job runs (a damaged one fails the run, nothing charged). It becomes the noise layer's output for the part (strength and remover are not used); **Leveling then applies on top as usual**, and Voice rebuild wins where it overlaps. Answers `{ok: true}` |
| `PUT /api/jobs/{id}/browser/cues?start=&end=` | JSON `[{start, end, text}]` in file time, inside the part | at most 1 MB and 20,000 cues; a cue a hair (0.5 s) outside the part is cut to it, further is `400`; cues with no text are dropped. They are the part's text (no translation: free parts have none). Cues sent for another language than the part's current one are not used: `needs` lists the part again. Answers `{ok: true}` |

**Outlines.** `GET /api/jobs/{id}/peaks/{layer}`, `layer` one of `source`, `echo`, `noise`, `rebuild` (the stems of the preview stretches, on the same sample grid), answers

```json
{"rate": 4, "seconds": 612.4, "peaks": [0.0, 0.012, 0.31, null, "..."]}
```

one value per 0.25 s of the recording (`ceil(seconds x 4)` of them): the largest absolute sample over all channels in that quarter second, 0 to 1, rounded to 3 decimals; `null` where that layer is not in the cache
for that moment (the same moments `X-Stretch-Missing` of the preview lists; the noise layer of a part cleaned in the browser is never cached here). The values are made from the cached blocks and kept per block in the
job's cache (the recording's in `kept-source.peaks`), so asking again is cheap; if the time of one call (25 s) runs out before every block has its values, the blocks left are `null` for now and the answer has `"pending": true`
(ask again). `404 not_cached` when nothing of the layer is cached. Same sign-in, scopes, ownership, retention (a read, not an edit) and rate limit (`503` when busy) as the preview stretches.

**Retention.** The job's files (upload, cached blocks, recording copy, the browser's uploads, the subtitle parts' text, results) are deleted **2 hours after its last edit** (`expiresAt`) and never
later than **6 hours after the upload** (`hardDeadlineAt`). Edits: a run finishing, `PUT .../timeline`, a quote, a process, an export.
Reads and previews are not edits. A run that cannot finish before the 6 hours is refused (`409 job_expiring`), one that only finishes in their last minute is stopped
and not charged.

Errors (`code`, also in `X-Error-Code`): `400 bad_timeline`, `400 timeline_options` (a timeline with the options it replaces, or subtitle parts beside whole-file subtitle options), `400 nothing_to_change`
(the result files already hold this timeline), `404 job_not_found`, `404 not_cached`, `409 job_not_editable`, `409 job_files_gone`,
`409 job_not_finished`, `409 job_busy`, `409 needs_processing`, `409 needs_uploads` (with `needs`), `409 job_expiring`, `409 price_changed` (the price is above `maxCostMicros`),
`429 too_many_active_jobs` (a process, export or suggestions run is a job in progress; quotes, timelines and previews are not), `402`, `403`
(`key_spending_cap`), `507`.

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
within their 2 hours. A callback can arrive more than once (a retry after a timeout): treat `jobId` as the dedupe key.

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
