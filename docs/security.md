# Security model

primary-ram sits on the home LAN, so the design goal is: a compromise of any
internet-facing piece must not reach the LAN, other services on primary-ram,
or secrets.

## Exposure

| Surface | Reachable from | Notes |
|---|---|---|
| `/audio/*` static | internet (VPS Caddy) | no server logic |
| `/audio/api/*`, WebSocket | internet → VPS → SSH tunnel → 127.0.0.1:8790 | the only path into primary-ram |
| uploads, result downloads and release downloads (`/audio/api/downloads/`: installers, OBS plugin, model archive) | internet → VPS → second SSH tunnel → 127.0.0.1:8794 (`acw-bulk`) | the same routes the main process also serves; nothing else is routed there. The release files are mirrored to primary-ram by `acw-mirror` (signature and checksums checked first); nothing is stored on the VPS |
| primary-ram LAN ports | unchanged | the app binds loopback only |
| VPS → primary-ram | `127.0.0.1:18790` and `127.0.0.1:18794` on the VPS only | tunnel key: `restrict,port-forwarding,permitlisten="127.0.0.1:18790",permitlisten="127.0.0.1:18794",permitopen="127.0.0.1:1",command="/bin/false"` (the second `permitlisten` is added when the bulk tunnel is switched on, docs/operations.md; verified before for one port: no shell, no other ports, no local forwards) |

## Process isolation (no root needed)

| Process | Network | Filesystem | Other |
|---|---|---|---|
| acw-web | loopback listen; outbound HTTPS (Stripe, OpenAI, OAuth, SMTP) | its data dir | seccomp `@system-service`, no `@privileged`, NoNewPrivileges, MemoryMax, `LimitNOFILE=8192` |
| acw-mirror (oneshot, every 10 min) | outbound HTTPS to the release service only (AF_INET/AF_INET6); no listen | `data/downloads/` | seccomp `@system-service`, no `@privileged`, NoNewPrivileges, PrivateTmp, MemoryMax 600M; the read token (if any) is a 0600 file it never logs |
| acw-bulk | loopback listen (8794); no outbound use | its data dir | same hardening as acw-web, MemoryMax 1G; only the upload, result-download and release-download routes, no background loops, no live code |
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
  sign-in mails also per canonical mailbox (no `+tag`, Gmail dots ignored) and per small mail domain; the contact form per network, per mailbox and 100 a day for the site (`api/contact_routes.py`: same-origin, honeypot field, control characters and header injection stripped, plain-text mail, the owner's address never sent to a browser, message text not stored). Every 429 carries `Retry-After`.
  Each limit name has its own table, so a flood of made-up keys of one kind cannot push out the others.
- Request bodies: 64 KB for everything but the upload route (1 MB for the Stripe webhook), counted while the
  body arrives, with a 20 s stall limit (`acw/bodylimit.py`); Caddy caps the sizes in front
  (`deploy/vps/audio.caddy`; its Caddyfile syntax has no per-body read timeout, so the stall limit is the app's). The access log omits query strings (sign-in codes, payment ids).
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
- Free subtitles run in the browser, so they cost us no compute; their model files (about 80 MB) are served only to
  signed-in browsers and count against the same daily per-account allowance as the noise remover's files
  (`asset_fetch`); browsers keep them in their cache. Jobs on our side are never free.
- Job reservations: the per-account active-job count, the disk promised to uploads that are still arriving and the
  job row are checked and written in one transaction (parallel requests can't all pass). An account may have at most
  8 GiB of uploads waiting; a file must be plausible for its claimed length (at most 40 MB per second, or 200 MB
  whatever the length); an upload that has sent no byte 10 minutes after creation is canceled (the hour-long window
  starts with the first bytes).
- Cancel is refused (409) once the extra-accurate transcription is done (`next_stage = finalize`: the paid-for work
  exists, only packaging is left), and answers 409 whenever nothing was canceled.
- Live sessions must keep sending: microphone audio below half of the wall-clock time over the last 60 s (judged after
  the first 30 s) ends the session with `stopped: no_audio` (both clients send every frame, silent ones too). Text
  messages are limited per connection (50 a second sustained, burst 100; `config` 5 a second); a breach closes the
  socket with 1008. An account can run two live sessions at a time, of which at most one on the free trial.
- Free live trial: two hours of Standard live (`users.trial_live_seconds`, `ACW_LIVE_TRIAL_SECONDS`, 0 = off), granted once per mailbox. The key is HMAC-SHA256 under `ACW_SESSION_SECRET` of the canonical
  mailbox (NFKC, lower case, no `+tag`, no dots and `googlemail.com` = `gmail.com` at Gmail), so aliases don't give
  it twice (also after deleting the account and signing up again) and a copy of the database can't be used to test addresses. Without a configured secret (development) the
  key is a plain SHA-256 and a warning is logged once. Rows written before this change keep their old key.
- No free credit: everything that costs real money to run (extra-accurate transcription and live captions, whose speech
  partner is paid per minute) is paid from credit from the start. The trial covers Standard live only (noise removal,
  Standard captions); a session with extra-accurate captions or any `translate` option is billed from credit and never
  touches the trial. Trial seconds come off in whole seconds of processed audio, saved every 30 s together with the
  session's progress, so a restart (or a vanished client) keeps what was saved; the rest of such a session, after the
  trial ran out, is billed from its hold like any session. A trial session sets aside no money. It is admitted only
  while live use is below 70% of capacity (checked by the live worker, `trial_busy` otherwise), and an account runs at
  most one at a time, so a farm of accounts cannot crowd out paying users or multiply the hours a mailbox has.
- Top-ups (`acw/billing/trust.py`): a first top-up (no `payments` row `succeeded` and not fully refunded) is limited
  to $50; once one has gone through the usual $500 applies (checkout and auto-reload).
- Premium live captions that break down mid-session are billed only up to that point (`stream_sessions.premium_off_seconds`);
  a captions-only session then ends.
- `/api/me` with an API key returns only the user id and status and the balance, never the address or billing settings.
- Disputes and early fraud warnings suspend the account, release holds, zero
  the balance; EFWs up to $50 are refunded automatically.
- 3DS at Checkout whenever Stripe or the bank asks for it; auto-reload has a monthly cap and
  turns itself off after 3 failures.

## Known gaps / to do

- Supply chain: Python packages are installed with `--require-hashes` (every file PyPI lists for each pinned version,
  `deploy/hash_lock.py`), models and FFmpeg are pinned by commit/release and verified by sha256 (`deploy/models.lock`,
  `deploy/fetch-models.sh`). `libdf.so` is built locally and only recorded.

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
- Uploads and live audio: **SEC-08 is done on the home side.** Uploads and result downloads run in their own process
  (`acw-bulk`, `acw/bulk_app.py`: same authentication, body limits, upload caps and headers; a shared transfer budget for uploads and downloads,
  `ACW_TRANSFER_MBPS_TOTAL`, keeps transfers from filling the VPS link the live audio uses) and their own SSH connection
  (`acw-tunnel-bulk`, IPQoS throughput; the main tunnel is lowdelay). It takes effect when the VPS key, the bulk tunnel
  and the Caddy snippet are switched over (docs/operations.md, "Bulk transfers"; not applied by the repository); until
  then the main process serves the same routes as before. Still open from SEC-08: SQM on the home router, a smaller
  per-file cap for free jobs.
- Logs: the user journal keeps client addresses until it rotates; set `MaxRetentionSec` once (docs/operations.md, "Log retention").
- The `/audio/dfn/*` forward-auth gate relies on Caddy's path normalisation; test `//`, `%2e`, upper-case once on the VPS
  with the installed Caddy version (`curl -i https://mc-multi-world.space/audio//dfn/df.js` must not return the file without a session).
