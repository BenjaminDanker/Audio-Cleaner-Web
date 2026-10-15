# Launch checklist

Status as of 2026-10-10. ✅ done · ⬜ needs you · 🔬 research phase

## Infrastructure
- ✅ Services on primary-ram (systemd user units, sandboxed workers, tunnel)
- ✅ VPS tunnel key restricted to one port, no shell
- ✅ VPS Caddy `/audio` route, site published (https://mc-multi-world.space/audio/)
- ⬜ Own domain or subdomain (DNS A record → 216.250.115.225), then set
  `ACW_PUBLIC_ORIGIN`/`ACW_BASE_PATH` and the Caddy site block. Separates
  cookies/origin from the pet-companion checkout pages.
- ⬜ Optional hardening with sudo: dedicated `audiocleaner` system user + system units (docs/security.md)
- ⬜ Off-box encrypted DB backup (nightly local snapshot to the HDD is ✅)
- ⬜ Uptime alert (e.g. the Pi curls `https://…/audio/api/health` every minute and pings you)

- ⬜ Re-run `sudo bash ~/acw/install-audio-route.sh` on the VPS after Caddy snippet changes
  (2026-10-13: stricter CSP, no-cache on all app routes)
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
- ⬜ Cloudflare Turnstile keys (optional, for the email form)

## Business / legal
- ⬜ Fill `web/src/site.js`: support email, legal name/address, jurisdiction (shown in footer/Terms/Privacy)
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
- ✅ Live denoise + live captions, server side + protocol (PCM and Opus)
- ⬜ OBS plugin: Windows/macOS builds via CI need a GitHub repo with Actions;
  code signing (Apple Developer ID $99/yr, Windows Authenticode certificate)
- 🔬 Studio clean: benchmark found no clearly better model than DFN3 + polish; optional GPU hybrid pending a blind A/B (docs/research/denoise.md)
- 🔬 Faster local transcription (Parakeet), GPU (needs the NVIDIA driver: sudo)
- ✅ One live denoise tier (DFN3, 30 ms); the low-latency tier (DFN3-LL) was tried and dropped
