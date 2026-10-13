# Security model

primary-ram sits on the home LAN, so the design goal is: a compromise of any
internet-facing piece must not reach the LAN, other services on primary-ram,
or secrets.

## Exposure

| Surface | Reachable from | Notes |
|---|---|---|
| `/audio/*` static | internet (VPS Caddy) | no server logic |
| `/audio/api/*`, WebSocket | internet → VPS → SSH tunnel → 127.0.0.1:8790 | the only path into primary-ram |
| primary-ram LAN ports | unchanged | the app binds loopback only |
| VPS → primary-ram | `127.0.0.1:18790` on the VPS only | tunnel key: `restrict,port-forwarding,permitlisten="127.0.0.1:18790",permitopen="127.0.0.1:1",command="/bin/false"` (verified: no shell, no other ports, no local forwards) |

## Process isolation (no root needed)

| Process | Network | Filesystem | Other |
|---|---|---|---|
| acw-web | loopback listen; outbound HTTPS (Stripe, OpenAI, OAuth) | its data dir | seccomp `@system-service`, no `@privileged`, NoNewPrivileges, MemoryMax |
| acw-worker@N | **none** (seccomp: AF_UNIX only) | DB + jobs dir | Nice 10, CPUWeight 50 |
| job runner (per job) | **none** (also Landlock TCP deny) | **Landlock**: read-only system libs, venv, code, models, binaries; read-write only its job dir; no `$HOME`, no `/proc`, no DB, no secrets | RLIMIT_CPU/AS/FSIZE, no core dumps, `-protocol_whitelist file,pipe` for ffmpeg, container-format allowlist |
| acw-stream | **none** (AF_UNIX) | Landlock read-only after binding its socket | CPUWeight 300 |

Untrusted media is only parsed inside the runner. A malicious file that
exploits ffmpeg ends up in a process that cannot open a socket, cannot read
`~/.ssh`, the database or env secrets (`/proc/<pid>/environ` is not visible),
and can only write its own job directory. Tested in
`tests/test_media_units.py::test_runner_sandbox_blocks_escape`.

Mount-namespace sandboxing (systemd `ProtectHome=` etc.) is unavailable to
user units on this Ubuntu (unprivileged user namespaces are AppArmor-restricted),
which is why Landlock is used. **Recommended hardening when you have a sudo
moment:** run the services as a dedicated `audiocleaner` system user with
system units (`ProtectSystem=strict`, `ProtectHome=yes`, `PrivateNetwork=yes`
for workers, `IPAddressDeny=` for the LAN/tailnet ranges on acw-web). That
also removes the shared-UID risk with the Minecraft servers.

## Web application

- Sessions: 256-bit random tokens, stored as SHA-256; cookie HttpOnly,
  SameSite=Lax, `Secure` + `__Secure-` prefix in production, path `/audio/`.
- CSRF: per-session token header on every non-GET + Origin check; login and
  magic-link endpoints require same Origin (prevents login CSRF).
- OAuth: PKCE + state stored server-side and bound to the browser by a
  short-lived cookie; Google `nonce`/`aud`/`iss`/`email_verified` checked;
- API keys: `acw_` + 256-bit random, hashed at rest, shown once, scoped
  (`stream`, `jobs`), cannot touch billing/keys/account.
- WebSocket: API key, or session cookie + exact Origin match (no cross-site
  WebSocket hijacking).
- Rate limits: per IP and per user for auth, email, uploads, job creation,
  checkout, stream starts; general API limit per IP.
- Upload: declared size enforced, streamed to disk, extension allowlist,
  filenames never used as paths (`input.<ext>`), outputs sanitized.
- Headers: CSP (API: `default-src 'none'`; site: self + wasm), HSTS, nosniff,
  frame-deny, COOP, no-store on API responses. Docs/OpenAPI disabled in production.
- Money: integer micro-dollars, CHECK constraints (balance ≥ 0, held ≤ balance),
  serialized transactions, idempotent Stripe processing keyed by event and
  object ids, webhook signatures verified.

## Abuse / fraud

- Live audio must arrive at ~real-time pace (prevents using live pricing as cheap batch).
- Free transcription: signed-in only, 60 min/day, 30 min/file.
- Disputes and early fraud warnings suspend the account, release holds, zero
  the balance; EFWs up to $50 are refunded automatically.
- 3DS requested on every Checkout payment; auto-reload has a monthly cap and
  turns itself off after 3 failures.

## Known gaps / to do

- Same Unix user as the Minecraft servers (see recommended hardening above).
- Shared origin `mc-multi-world.space` with the pet-companion checkout pages:
  path-scoped cookies are not a security boundary against script on the same
  origin. Move to a dedicated subdomain/domain before launch (only `ACW_PUBLIC_ORIGIN`,
  `ACW_BASE_PATH` and the Caddy block change).
- The pet-companion tunnel key on the VPS (`permitlisten="127.0.0.1:18788"`) lacks
  `command="/bin/false"`, so it can open a shell on the VPS. Worth adding.
- No malware/CSAM scanning of uploads (audio-only product, files deleted quickly);
  revisit if video volume grows.
- Backups: a nightly snapshot goes to `/mnt/mc-storage/audio-cleaner-backups`
  (another disk, 30 kept; `acw-backup.timer`). That does not survive losing
  the machine; add an encrypted off-box copy before taking real money.
