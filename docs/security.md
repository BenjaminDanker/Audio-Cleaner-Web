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
| acw-web | loopback listen; outbound HTTPS (Stripe, OpenAI, OAuth, SMTP) | its data dir | seccomp `@system-service`, no `@privileged`, NoNewPrivileges, MemoryMax, `LimitNOFILE=8192` |
| acw-worker@N | **none** (seccomp: AF_UNIX only) | DB + jobs dir | Nice 10, CPUWeight 50 |
| job runner (per job) | **none**: seccomp refuses `socket`, `connect`, `bind`, `listen`, `accept`, `sendto`, `sendmmsg` (plus `io_uring`, `ptrace`, `process_vm_*`, `pidfd_getfd`), and Landlock denies TCP | **Landlock**: read-only system libs, venv, code, models, binaries; read-write only its job dir, where it can create files and directories but **not** links, pipes or sockets; no `$HOME`, no `/proc`, no DB, no secrets | RLIMIT_CPU/AS/FSIZE, no core dumps, `-protocol_whitelist file,pipe` for ffmpeg, container-format allowlist, and the demuxer is chosen from the file's magic bytes (one demuxer parses an upload, not all of them) |
| acw-stream | **none** (AF_UNIX) | Landlock read-only after binding its socket (required in production: it refuses to start without it) | CPUWeight 300 |

Untrusted media is only parsed inside the runner. A malicious file that
exploits ffmpeg ends up in a process that cannot open a socket, cannot read
`~/.ssh`, the database or env secrets (`/proc/<pid>/environ` is not visible),
and can only write its own job directory. Landlock (kernel 6.8, ABI 4) does **not** govern Unix-domain sockets,
so without the seccomp filter such a process could still have reached the user's D-Bus, the live worker's
socket or abstract sockets; the filter closes that. `socketpair()` stays allowed (the denoiser's Rust
runtime builds one; an anonymous pair leads nowhere). Tested in
`tests/test_media_units.py` (`test_runner_sandbox_blocks_escape`, `test_runner_sandbox_refuses_sockets_links_and_pipes`).

What the runner leaves behind is not trusted either: the worker and the web process read outputs only as regular
files on the job directory's device (never through a link), check the shape and size of its result file, and
read the OpenAI segment files the same way.

**GPU runner.** It needs the NVIDIA device files, `/sys/devices`, `/sys/bus/pci`, `/dev/char`, `/proc/driver/nvidia` and
its own `/proc/self`, and has no address-space limit (CUDA reserves huge ranges). Landlock cannot restrict
ioctls before ABI 5, so the driver's ioctl surface is reachable from the process that parses the upload.
Its seccomp filter still refuses connect, bind, listen, accept and sendto; it may create Unix sockets
(not connect them). Splitting the untrusted decode into a separate CPU process, with the GPU process only
getting decoded 16 kHz audio, would remove the exposure; not done (documented risk, GPU worker is optional).

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
- Rate limits: per IP (IPv6 per /64) and per user for auth, email, uploads, job creation,
  checkout, stream starts and live WebSocket handshakes (before any database work); general API limit per IP;
  sign-in mails also per canonical mailbox (no `+tag`, Gmail dots ignored) and per small mail domain. Every 429 carries `Retry-After`.
  Each limit name has its own table, so a flood of made-up keys of one kind cannot push out the others.
- Request bodies: 64 KB for everything but the upload route (1 MB for the Stripe webhook), counted while the
  body arrives, with a 20 s stall limit (`acw/bodylimit.py`); Caddy enforces the same in front
  (`deploy/vps/audio.caddy`). The access log omits query strings (sign-in codes, payment ids).
- Upload: declared size enforced, streamed to disk, extension allowlist,
  filenames never used as paths (`input.<ext>`), outputs sanitized. An upload that sends nothing for 30 s is
  dropped; upload bandwidth is capped per account and overall; the declared bytes of uploads in progress are
  reserved against free disk space and the space is re-checked while data arrives; a free job's size must fit
  its claimed length at 12 Mbit/s.
- Database and file I/O in request handlers (uploads, live sessions, housekeeping) run off the event loop that
  carries live audio.
- Outbound calls (SMTP, Stripe checkout/portal/refund) share a budget of 8 threads out of the pool and have
  shorter timeouts; SMTP verifies the server certificate.
- Sign-in e-mail addresses must be one plain mailbox (no `,;<>"()`, one `@`, round-trips through the mail
  library's parser), so one request cannot mail an arbitrary other recipient.
- Live sessions end when no microphone audio arrives for 2 minutes (pings and echo-reference frames do not count;
  both the website and the OBS plugin keep sending frames while muted), at 12 hours of wall time, or when the
  account is suspended, the API key is revoked or the browser session is gone (checked every 30 s). A client
  that vanishes while a session starts gets its hold back at once; a sweeper closes and bills any "active"
  session no handler owns.
- Browser sessions also expire after 14 days without use. Admin rights follow `ACW_ADMIN_EMAILS` both ways.
- Headers: CSP (API: `default-src 'none'`; site: self + wasm), HSTS, nosniff,
  frame-deny, COOP, no-store on API responses. Docs/OpenAPI disabled in production.
- Money: integer micro-dollars, CHECK constraints (balance ≥ 0, held ≤ balance),
  serialized transactions, idempotent Stripe processing keyed by event and
  object ids, webhook signatures verified.

## Abuse / fraud

- Live audio must arrive at ~real-time pace (prevents using live pricing as cheap batch).
- Free transcription: signed-in only, 60 min/day per account, 30 min/file, and a ceiling for all free users
  together per day (`pricing.FREE_GLOBAL_SECONDS_PER_DAY`). Deleting an account and signing up again with the same
  address the same day keeps that day's usage (a hash of the address is kept until the next day).
- Disputes and early fraud warnings suspend the account, release holds, zero
  the balance; EFWs up to $50 are refunded automatically.
- 3DS requested on every Checkout payment; auto-reload has a monthly cap and
  turns itself off after 3 failures.

## Known gaps / to do

- Same Unix user as the Minecraft servers (see recommended hardening above).
- Shared origin `mc-multi-world.space` with the pet-companion checkout pages:
  path-scoped cookies are not a security boundary against script on the same
  origin. Move to a dedicated subdomain/domain before launch: only `ACW_PUBLIC_ORIGIN`,
  `ACW_BASE_PATH` (empty) and the Caddy block change, and the session cookie then becomes `__Host-acw_session`
  by itself (steps in docs/operations.md, "Moving to a domain of its own").
- The pet-companion tunnel key on the VPS (`permitlisten="127.0.0.1:18788"`) lacks
  `command="/bin/false"`, so it can open a shell on the VPS. Worth adding.
- No malware/CSAM scanning of uploads (audio-only product, files deleted quickly);
  revisit if video volume grows.
- Backups: a nightly snapshot goes to `/mnt/mc-storage/audio-cleaner-backups`
  (another disk, 30 kept; `acw-backup.timer`) and is checked right after writing (SQLite integrity and ledger
  sums; the monitor mails when it fails). That does not survive losing the machine: the encrypted off-box copy is
  built in but switched off until you configure it (docs/operations.md, "Backups").
- Uploads and live audio share one SSH tunnel and one Python process. Bandwidth caps and off-loop I/O limit the
  effect; the real fix is a second tunnel and process for `/api/jobs/*/input` plus SQM on the home link. Follow-up, not built.
- Logs: the user journal keeps client addresses until it rotates; set `MaxRetentionSec` once (docs/operations.md, "Log retention").
- The `/audio/dfn/*` forward-auth gate relies on Caddy's path normalisation; test `//`, `%2e`, upper-case once on the VPS
  with the installed Caddy version (`curl -i https://mc-multi-world.space/audio//dfn/df.js` must not return the file without a session).
