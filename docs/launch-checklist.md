# Launch checklist

Status as of 2026-10-10. ✅ done · ⬜ needs you · 🔬 research phase

## Infrastructure
- ✅ Services on primary-ram (systemd user units, sandboxed workers, tunnel)
- ✅ VPS tunnel key restricted to one port, no shell
- ⬜ VPS Caddy `/audio` route: `ssh vps`, then `sudo bash ~/acw/install-audio-route.sh`
- ⬜ Publish the site: `deploy/publish-web.sh` (after the route exists)
- ⬜ Own domain or subdomain (DNS A record → 216.250.115.225), then set
  `ACW_PUBLIC_ORIGIN`/`ACW_BASE_PATH` and the Caddy site block. Separates
  cookies/origin from the pet-companion checkout pages.
- ⬜ Optional hardening with sudo: dedicated `audiocleaner` system user + system units (docs/security.md)
- ⬜ Off-box encrypted DB backup (nightly local snapshot to the HDD is ✅)
- ⬜ Uptime alert (e.g. the Pi curls `https://…/audio/api/health` every minute and pings you)

## Accounts and keys
- ⬜ GitHub OAuth app + Google OAuth client (callback URLs in docs/operations.md)
- ⬜ SMTP provider for email sign-in links (optional, but most users expect email)
- ⬜ Stripe: test mode first. Restricted key, webhook endpoint + events
  (docs/payments.md), customer portal enabled, statement descriptor,
  Radar rules (block high-risk, optionally EU/UK until VAT is registered).
- ⬜ OpenAI: a dedicated project key with a monthly budget cap, set as
  `OPENAI_API_KEY` in `~/audio-cleaner/secrets/acw.env`.
- ⬜ Cloudflare Turnstile keys (optional, for the email form)

## Business / legal
- ⬜ Product name and domain (the code says "Audio Cleaner"; it's one constant/string set)
- ⬜ Prices (server/acw/billing/pricing.py; launch placeholders: Studio $0.60/h,
  transcription $0.24/h standard and $0.90/h premium, live denoise $0.24/h,
  live captions $0.36/h standard and $1.20/h premium)
- ⬜ Terms of Service + Privacy Policy reviewed by a lawyer (drafts in the site)
- ⬜ Tax: Stripe Tax on; register UK VAT and EU non-Union OSS, or block those
  countries at first (docs/payments.md)
- ⬜ Repo is **public** on GitHub with no licence. Decide: make it private, or
  keep the server private and publish only the OBS plugin (GPL is required for it).

## Product
- ✅ Free browser denoise (DFN3 WASM)
- ✅ Studio clean, transcription (standard/premium, free quota), subtitles (soft subs + SRT/VTT/TXT)
- ✅ Live denoise + live captions, server side + protocol (PCM and Opus)
- ⬜ OBS plugin: Windows/macOS builds via CI need a GitHub repo with Actions;
  code signing (Apple Developer ID $99/yr, Windows Authenticode certificate)
- 🔬 Studio clean needs a clearly better model than the free tier (it's DFN3 + polish today)
- 🔬 Faster local transcription (Parakeet), GPU (needs the NVIDIA driver: sudo)
- 🔬 Live "medium"/"high" denoise tiers
