# Security model

primary-ram sits on the home LAN, so the design goal is: a compromise of any
internet-facing piece must not reach the LAN, other services on primary-ram,
or secrets.

## Accounts: the services are not the owner

The services run as the account **`acw`**, not as the owner (`silver`). A hole in the web code, a bug in a parser that
gets past the runner sandbox, or a stolen process would otherwise have the owner's identity: the SSH keys (including the
login to the VPS, which can change what the whole internet sees), other secrets, the Minecraft servers and every file
of the owner. As `acw` it has none of that.

| | `silver` (owner) | `acw` (services) |
|---|---|---|
| login | normal | none (`/usr/sbin/nologin`, no password); it exists to own processes and files |
| home | `/home/silver` (750) | `/home/acw` (750, group `acw`); cannot read `/home/silver` |
| sudo | yes | none; `/etc/sudoers.d/acw`: `silver ALL=(acw) NOPASSWD: ALL` - privilege only goes **down** |
| VPS login key | yes (`~/.ssh/id_ed25519`, `~/.ssh/config`) | **never**: acw has only the restricted tunnel key (below) and the VPS host key |
| website publishing, smoke test | `deploy/publish-web.sh` (`ssh vps`, rsync), `deploy/smoke.py` | cannot (no key; the script refuses to run as acw) |
| Rust toolchain, build trees, `dev/`, `research/` | yes | no (native programs are built by the owner and handed over) |
| `update-signing-key.pem`, `github-actions-*.token`, the owner's other secrets | yes | no (the migration moves only an allowlist of acw's own files) |
| `secrets/acw.env`, `acw-worker.env`, `vps.env`, `backup.*`, `github-releases-read.token` | cannot read (0600, 0700 folder; a root-or-acw job) | yes |
| database, job files, `app/` | cannot read directly (`acw-admin`, `acw-sh`) | yes |
| `bin/ models/ wasm/ venv/` (no secrets) | read (group `acw` + ACL) | owner |

The **tunnel key** (`/home/acw/.ssh/acw_tunnel`) is the restricted one (`permitlisten` on three loopback ports on the VPS,
no shell, `command="/bin/false"`): all it can do is hold the three reverse tunnels open, so a stolen copy changes nothing
on the VPS. The units use `-F none`, `IdentitiesOnly=yes` and `HostKeyAlias=vps` with `/home/acw/.ssh/known_hosts`.

What the owner's tools cost: the owner can act as acw (`acw-ctl`, `acw-logs`, `acw-admin`, `acw-sh`, `deploy/install.sh`);
that is the point (silver keeps full control) and it means the owner account is the one to protect (SSH keys only, Tailscale
SSH-only, `deploy/host/tailscale-ssh-only.sh`). The deploy runs from a `git archive` of the exact commit owned by acw
(`/home/acw/deploy/<commit>`), so acw never reads the owner's checkout and what runs is what was checked.

Limits to know: acw and silver share the kernel, the loopback network and `/tmp` is per-account; a local kernel exploit
or a root hole still crosses accounts. The Landlock/seccomp runner (below) is unchanged and still the layer against
hostile media; the account split is the layer behind it.

## Exposure

| Surface | Reachable from | Notes |
|---|---|---|
| `audiocleaner.live` static (`/`, `/app`, `/assets/`) | internet (VPS Caddy) | no server logic |
| `/api/*` | internet → VPS → SSH tunnel → 127.0.0.1:8790 (`acw-web`) | the API |
| `/api/stream/ws` (live WebSocket) | internet → VPS → third SSH tunnel → 127.0.0.1:8795 (`acw-live`) | live only |
| uploads, result downloads and release downloads (`/api/downloads/`: the OBS plugin) | internet → VPS → second SSH tunnel → 127.0.0.1:8794 (`acw-bulk`) | the same routes the main process also serves; nothing else is routed there. The release files are mirrored to primary-ram by `acw-mirror` (signature and checksums checked first); nothing is stored on the VPS |
| primary-ram LAN ports | unchanged | the app binds loopback only |
| VPS → primary-ram | `127.0.0.1:18790`, `18794` and `18795` on the VPS only | tunnel key: `restrict,port-forwarding,permitlisten="127.0.0.1:18790",permitlisten="127.0.0.1:18794",permitlisten="127.0.0.1:18795",permitopen="127.0.0.1:1",command="/bin/false"` (docs/vps.md; no shell, no other ports, no local forwards) |

## Process isolation (no root needed)

| Process | Network | Filesystem | Other |
|---|---|---|---|
| acw-web | loopback listen; outbound HTTPS (Stripe, OpenAI, OAuth, SMTP, and job callbacks to addresses the account owner gave: the only call to an address chosen by a user, see "Job callbacks") | its data dir | seccomp `@system-service`, no `@privileged`, NoNewPrivileges, MemoryMax, `LimitNOFILE=8192` |
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
  SameSite=Lax, `Secure`, name `__Host-acw_session` (path `/`, no Domain; the browser enforces that) on the own domain,
  `__Secure-acw_session` with path `<base>/` if the app is ever served under a base path again.
- CSRF: per-session token header on every non-GET + Origin check; login and
  magic-link endpoints require same Origin (prevents login CSRF).
- OAuth: PKCE + state stored server-side and bound to the browser by a
  short-lived cookie; Google `nonce`/`aud`/`iss`/`email_verified` checked;
- API keys: `acw_` + 256-bit random, hashed at rest, shown once, scoped
  (`stream`, `jobs`), cannot touch billing/keys/account. A key can carry a monthly spending cap
  (`api_keys.monthly_cap_micros`, `acw/keycaps.py`), checked inside the transaction that reserves the money (job
  creation, live start, each live minute), so parallel requests can't pass together; jobs and live sessions record the
  key (`jobs.api_key_id`, `stream_sessions.api_key_id`), and only a browser session can set or change a cap.
- `Idempotency-Key` on job creation: kept per account for 24 hours in `idempotency_keys` (the key text and a hash of
  the request body, no file data), swept by `jobs.sweep` and removed with the account. A repeat is answered before
  rate limits and money are touched; checked again inside the creation transaction against parallel requests.
- Changing a finished job (`POST /api/jobs/{id}/rerun`, `acw/rerun.py`): the same authentication, scope, CSRF and job-creation rate limits as
  creating one. The job is looked up by id **and** account; an API key can only change jobs that key made (`jobs.api_key_id`), a browser
  session any job of its account, anything else is the same 404 as an unknown id. The new job is held and priced like any job (the
  key's cap, the account's balance, the active-job limit are all checked in the transaction that stores it) and is run by the same queue,
  worker and sandbox. It starts from files the sandboxed runner itself made earlier (the kept copies): the worker takes only
  the fixed names `kept-noise.flac`, `kept-rebuilt.flac`, `kept-input.flac`, only as regular files in the job's folder (`lstat`), with
  the numbers the runner reported validated (`rerun.sanitize_kept`), and hard-links them into the new job's folder (the sandbox itself
  can't create links). A re-run is an edit of the chain (its files are kept 2 hours from then) and nothing of the chain outlives 6 hours after the upload
  (the upload is one of the linked files, so it goes then too, and "Delete now" deletes the whole chain).
- The per-part editor (`acw/editing.py`, `GET|PUT /api/jobs/{id}/timeline`, `quote`, `process`, `export`, `suggestions`, `preview/{layer}`): the same
  authentication and scope as the other calls on a job, and the re-run's ownership rule: the job is looked up by id **and** account, and an API key only
  reaches the jobs it made itself (anything else is `404 job_not_found`); a browser session needs its CSRF token for every write. A timeline from a client is
  checked strictly before anything is stored or priced (format and version, unknown keys, layers and settings, numbers, every part inside the file, at most 500
  parts a layer, at most 200 KB, a body limit of 256 KB on those routes only); a process holds its quote in the transaction that stores the run, with the
  account's balance, the key's cap and the active-job limit (quotes and previews are not jobs; process, export and suggestions runs are); a run charges
  for what the runner reports having run (`billed`, whole seconds per step, bounds-checked, priced with *our* fixed amounts and the rates pinned at
  the start) and never more than the hold. What the runner reports is checked before it is believed (`editing.sanitize_report`: the facts' numbers and
  the recording's copy only as a regular file, the removers' names, the rendered timeline through the same parser, the suggestions' shape); the web
  process plans from those facts and **never decodes a file the runner wrote**: the recording's stretches are cut by `worker/cutter.py` and the cache's by
  `worker/assembler.py`, which confines itself the same way (Landlock: the job's block folder read-only and nothing writable, no sockets, CPU /
  memory / file-size limits, a 20 s timeout), decodes only files named `<sha1>.flac` that are regular files, and writes the stretch to its standard output.
  The cache's files are keyed by a hash, only ever written by the runner, and deleted with the job. Reads of the timeline and previews are not edits, so
  an open tab cannot hold a job's files past their 2 hours.
- Stretches of the kept copies (`GET /api/jobs/{id}/kept/{noise|input}?start=&seconds=`, `acw/stretch.py`): the same authentication, scope and
  ownership as the other reads of a job (the job is looked up by id **and** account; a link or a non-regular file in the place of a copy is a
  404), numbers validated and the length capped at 30 s, a rate limit of its own (`kept_cut`, per account, 429 with `Retry-After`) and at most 4
  cuts at once. The copies were written by the sandboxed runner, which handles untrusted uploads, so the web process never opens one with a decoder:
  it starts `python -m acw.worker.cutter`, a program that confines itself first with the runner's own sandbox (`worker/sandbox.py`: Landlock with
  the file's folder read-only and nothing writable, no sockets, CPU / memory / file-size limits, no new privileges) and then becomes FFmpeg
  (forced `flac` demuxer, `file` protocol only) writing the stretch to its standard output; a 20 s timeout kills it. Nothing is written or stored.
- WebSocket: API key, or session cookie + exact Origin match (no cross-site
  WebSocket hijacking).
- Rate limits: per IP (IPv6 per /64) and per user for auth, email, uploads, job creation,
  checkout, stream starts and live WebSocket handshakes (before any database work); general API limit per IP;
  sign-in mails also per canonical mailbox (no `+tag`, Gmail dots ignored) and per small mail domain; the contact form per network, per mailbox and 100 a day for the site (`api/contact_routes.py`: same-origin, honeypot field, control characters and header injection stripped, plain-text mail, the owner's address never sent to a browser, message text not stored). Every 429 carries `Retry-After`.
  Each limit name has its own table, so a flood of made-up keys of one kind cannot push out the others.
- Request bodies: 64 KB for everything but the upload route (1 MB for the Stripe webhook), counted while the
  body arrives, with a 20 s stall limit (`acw/bodylimit.py`); Caddy caps the sizes in front
  (`deploy/vps/sites/00-audio-cleaner-snippets.caddy`; its Caddyfile syntax has no per-body read timeout, so the stall limit is the app's). The access log omits query strings (sign-in codes, payment ids).
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

## Job callbacks (`callbackUrl`)

A job may carry an https address that the web process calls when the job ends (`acw/webhooks.py`, docs/api.md "Finish
notifications"). This is the one place where the server connects to an address a user chose, from a machine on a home
network, so it is built to reach nothing but the public internet:

- Only the web process delivers (its lifespan runs the delivery loop). The workers, the live engine, the live process and
  the bulk process do not run it, and a test fails if one of them starts to (`tests/test_webhooks.py`).
  The job code (worker, cancel, sweep) knows nothing about callbacks: it only writes the job's final state, and the loop
  looks at the `job_callbacks` table. A failing or slow receiver can therefore not hold up, fail or bill a job.
- Address at creation (syntax only, `webhooks.check_url`): `https` scheme, port 443 or none, a host **name** with a dot
  (IP addresses, `localhost`, `.local`, `.internal`, `.lan`, `.home.arpa` and similar are refused), no `user:password@`, no
  fragment, ASCII only, 500 characters at most, and not this service's own host name.
- Address at delivery: the name is resolved by the loop's resolver and **every** address it returns must be public:
  refused are `0/8`, `10/8`, `100.64/10` (CGNAT), `127/8`, `169.254/16` (link-local and cloud metadata), `172.16/12`,
  `192.0.0/24`, `192.0.2/24`, `192.168/16`, `198.18/15`, `198.51.100/24`, `203.0.113/24`, multicast and reserved `224/4`
  and `240/4`; for IPv6 `::`, `::1`, `::/96`, IPv4-mapped addresses (unwrapped and checked as IPv4), `64:ff9b::/96` and
  `64:ff9b:1::/48` (NAT64), `100::/64`, `2001::/23` (Teredo and other protocol ranges), `2001:db8::/32`, `2002::/16`
  (6to4), `3fff::/20`, `5f00::/16`, `fc00::/7`, `fe80::/10`, `fec0::/10` and `ff00::/8`; plus everything Python's
  `ipaddress` does not call global, this machine's own addresses (its host name's addresses and the source address it
  would use towards the internet over IPv4 and IPv6) and any network in `ACW_WEBHOOK_DENY` (comma separated; use it for
  the public address of the home connection and of the VPS). One private address among public ones refuses the whole
  answer (DNS rebinding tricks).
- The connection goes to the checked address itself (TLS with the host name as SNI and certificate name, TLS 1.2 or
  newer, normal certificate verification), so nothing resolves a second time between the check and the connect. No
  proxy. **No redirects are followed**: any 3xx is a failure, so a public address can't bounce the call to an internal one.
- 5 seconds for the whole attempt (DNS, TCP, TLS, request, first line of the answer), the answer is read only up to its
  first line (8 KB cap) and then dropped, the request body is a few hundred bytes of JSON (job id, status, cost and
  output names), at most 3 attempts (30 s and 5 minutes apart), at most 4 deliveries at once. A claim in the table
  (`next_attempt_at`) keeps two processes from delivering the same callback. Waiting callbacks that never saw their job end
  are given up after 2 days, results are deleted after 7 days, and an account's callbacks go with the account.
- Everything is asynchronous: DNS through the event loop's resolver, the socket as an asyncio stream, every database call
  in a thread. A tarpit receiver costs one task for at most 5 seconds.
- What is revealed: the job id, status, error text, cost and output names go to an address the account owner chose;
  nothing about other accounts, no download link (downloading still needs the owner's key). The result shown with the job
  hides the query string of the address (it often holds a token).
- **The signing secret** (`users.webhook_secret`, `whsec_` + 144 random bits) is needed in clear to compute HMACs, so it
  is stored **in plain text** in the database, like the DB itself a file only the service user can read (mode 0600), and
  it is included in the nightly backups. It is not hashed or encrypted: encryption would need a key stored next to it.
  The browser session (never an API key) can read and replace it (`GET`/`POST /api/keys/webhook-secret`, audited as
  `webhook_secret_rotated`); it is left out of the account export (only "exists") and removed when the account is
  deleted. Each request carries `X-Audio-Cleaner-Signature: t=<unix>,v1=hex(HMAC-SHA256(secret, "<t>.<body>"))` so a
  receiver can check who sent it and reject old timestamps. Rotating the secret signs still-waiting callbacks with the
  new one.
- Tests (`tests/test_webhooks.py`) mock DNS and the connection; they cover the address table above, mixed answers,
  redirects, timeouts, the retries and the signature.

## Abuse / fraud

- Live audio must arrive at ~real-time pace (prevents using live pricing as cheap batch).
- Free subtitles run in the browser, so they cost us no compute; their model files (about 80 MB) are served only to
  signed-in browsers and count against the same daily per-account allowance as the noise remover's files
  (`asset_fetch`); browsers keep them in their cache. Jobs on our side are never free.
- Job reservations: the per-account active-job count, the disk promised to uploads that are still arriving and the
  job row are checked and written in one transaction (parallel requests can't all pass). An account may have at most
  `ACW_MAX_ACTIVE_JOBS` (default 3) file jobs uploading, waiting or running at once (changing a finished job counts; the refusal is
  `429` with the code `too_many_active_jobs`), and at most 8 GiB of uploads waiting; a file must be plausible for its claimed length (at most 40 MB per second, or 200 MB
  whatever the length); an upload that has sent no byte 10 minutes after creation is canceled (the hour-long window
  starts with the first bytes).
- Cancel is refused (409) once the extra-accurate transcription is done (`next_stage = finalize`: the paid-for work
  exists, only packaging is left), and answers 409 whenever nothing was canceled.
- Live sessions must keep sending: microphone audio below half of the wall-clock time over the last 60 s (judged after
  the first 30 s) ends the session with `stopped: no_audio` (both clients send every frame, silent ones too). Text
  messages are limited per connection (50 a second sustained, burst 100; `config` 5 a second); a breach closes the
  socket with 1008. An account can run four live sessions at a time, of which at most one on the free trial.
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
- Resolved 2026-10-18: Audio Cleaner shared the origin `mc-multi-world.space` with the pet-companion checkout pages
  (path-scoped cookies are not a security boundary against script on the same origin). It now has its own domain,
  `audiocleaner.live`, and the `__Host-` cookie. While the old address lasts, `mc-multi-world.space/audio/api/*` is
  proxied to the same API for old clients and the webhook, but no page is served there (pages redirect); the old
  address goes away once the old clients have moved (docs/operations.md, "Domain").
- The pet-companion tunnel key on the VPS is restricted like ours (`permitlisten="127.0.0.1:18788"`,
  `command="/bin/false"`: no shell; docs/vps.md).
- No malware/CSAM scanning of uploads (audio-only product, files deleted 2 hours after a job's last edit and at the latest 6 hours after the upload);
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
- The `/dfn/*` forward-auth gate relies on Caddy's path normalisation; test `//`, `%2e`, upper-case once on the VPS
  with the installed Caddy version (`curl -i https://audiocleaner.live//dfn/df.js` must not return the file without a session).
