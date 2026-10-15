# Launch checklist

Status as of 2026-10-15. ✅ done · ⬜ needs you · 🔬 research phase

## Infrastructure
- ✅ Services on primary-ram (systemd user units, sandboxed workers, tunnel)
- ✅ VPS tunnel key restricted to one port, no shell
- ✅ VPS Caddy `/audio` route, site published (https://mc-multi-world.space/audio/)
- ⬜ Own domain or subdomain (DNS A record → 216.250.115.225), then set
  `ACW_PUBLIC_ORIGIN`/`ACW_BASE_PATH` and the Caddy site block. Separates
  cookies/origin from the pet-companion checkout pages.
- ⬜ Optional hardening with sudo: dedicated `audiocleaner` system user + system units (docs/security.md)
- ⬜ Off-box encrypted DB backup: built in and verified in tests, switched off until you create
  `~/audio-cleaner/secrets/backup.env` (docs/operations.md, "Backups"). The nightly snapshot to the HDD is ✅ and is now
  checked after writing (integrity + ledger sums). ⬜ Drill the restore once (same section).
- ⬜ Push the branch and keep `on-prem` fast-forwarded from `desktop-merge`; deploy only from `~/repos/Audio-Cleaner-Web`
  (install.sh refuses a checkout that is not a descendant of what runs)
- ⬜ Contact form: after the next deploy send a test message from `/audio/contact` (signed out) and check it arrives with a working
  Reply (needs `ACW_ALERT_EMAIL` and `SMTP_*` in `acw.env`; `ACW_CONTACT_EMAIL` only to send somewhere else; docs/operations.md, "Contact form")
- ⬜ Journal retention (one sudo command, docs/operations.md "Log retention"): the Privacy Policy promises 90 days
- ⬜ Uptime alert (e.g. the Pi curls `https://…/audio/api/health` every minute and pings you)

- ⬜ Re-run `sudo bash ~/acw/install-audio-route.sh` on the VPS after Caddy snippet changes
  (2026-10-13: stricter CSP, no-cache on all app routes; 2026-10-15: request body limits and timeouts, plus the global
  `servers { timeouts { read_header 10s } }` option, see docs/operations.md "One-time setup")
- ⬜ Router: enable SQM / fq_codel (cake) on the home link. Big downloads in the house
  otherwise add 200-500 ms to live-stream latency (measured during research downloads).

## Accounts and keys
- ⬜ Google OAuth client: add the exact redirect URI
  `https://mc-multi-world.space/audio/api/auth/oauth/google/callback` (QA hit `redirect_uri_mismatch`)
- ⬜ SMTP provider for email sign-in links (optional, but most users expect email)
- ✅ Stripe: restricted key, webhook endpoint + events
  (docs/payments.md), customer portal enabled, statement descriptor,
  ✅ done 2026-10-15 (live; no Radar rule needed: authorize-first country check). Radar: switch to Lite before 2027-01-21.
- ✅ OpenAI key set (premium file + live captions verified in production)
- ⬜ Cloudflare Turnstile keys (optional, for the email form). Without them email sign-in has no bot check (3 mails per address and 6 per network per hour). To enable: keys in acw.env, and the three CSP additions that are commented in deploy/vps/audio.caddy

## Business / legal
- ⬜ Fill `web/src/site.js`: support email, legal name/address, jurisdiction (shown in footer/Terms/Privacy).
  The Privacy Policy no longer has placeholders (retention periods and transfers are written out: 7 years payments,
  12 months file names, 90 days addresses, 30 days backups; processed in the United States); with an empty
  `supportEmail` it points people at the Account page, so set the address when you have one.
- ⬜ Product name and domain (the code says "Audio Cleaner"; it's one constant/string set)
- ⬜ Prices (server/acw/billing/pricing.py; launch placeholders: Studio $0.60/h,
  transcription $0.24/h standard and $0.90/h premium, live denoise $0.24/h,
  live captions $0.36/h standard and $1.50/h premium)
- ⬜ Terms of Service + Privacy Policy reviewed by a lawyer (drafts in the site)
- ✅ Not offered in EU/EEA/UK/CH (authorize-first card-country check; docs/payments.md).
  (no Radar rule needed: cards are authorized first and only charged after the country check)
- ✅ Licence: proprietary/all rights reserved (source visible), OBS plugin GPL-2.0-or-later

## Product
- ✅ Free browser denoise (DFN3 WASM)
- ✅ Studio clean, transcription (standard/premium, free quota), subtitles (soft subs + SRT/VTT/TXT)
- ✅ Recordings: noise remover chosen per file (Background noise: Automatic / Light / Heavy; DeepFilterNet3 or
  DPDFNet-8, docs/research/denoise-2026-10.md addendum); missing highs of phone-quality recordings restored (AP-BWE)
- ⬜ Desktop and server: upload `~/audio-cleaner-models-release/AudioCleaner-models-v1.zip` (908538132 bytes, sha256
  5592be637b0be833d92b1476ffebfb07c2ac6920d9b24b3378ae6d318ebd81ce) to the GitHub release `models-v1` and untick
  "set as latest". The desktop setup and `deploy/fetch-models.sh` (deploy/models.lock) both download that one archive.
- ⬜ Price of "Clean & polish, voice rebuilt" (placeholder $1.20/h = 2x Clean & polish; it uses ~30x the CPU)
- ✅ Echo removal for recordings with a second audio track (desktop sound), included in Studio clean (option `echo_reference_track`, docs/api.md)
- ✅ Live denoise + live captions, server side + protocol (PCM and Opus)
- ✅ Live echo cancellation (speakers' sound removed from the mic; no extra price): OBS plugin sends the Desktop Audio as a
  reference, the browser uses its own; server AEC ~2 % of a core per stream (docs/streaming-protocol.md section 5)
- ⬜ OBS plugin: Windows/macOS builds via CI need a GitHub repo with Actions;
  code signing (Apple Developer ID $99/yr, Windows Authenticode certificate)
- 🔬 Studio clean: benchmark found no clearly better model than DFN3 + polish; optional GPU hybrid pending a blind A/B (docs/research/denoise.md)
- 🔬 Faster local transcription (Parakeet), GPU (needs the NVIDIA driver: sudo)
- ✅ One live denoise tier (DFN3, 30 ms); the low-latency tier (DFN3-LL) was tried and dropped
