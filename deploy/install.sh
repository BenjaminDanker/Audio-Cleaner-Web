#!/usr/bin/env bash
# Install or update Audio Cleaner on primary-ram (no sudo needed).
#   deploy/install.sh                    sync code, deps, units; restart what changed; publish the website if web/ changed;
#                                        then the smoke test (deploy/smoke.py: the live site from outside)
#   deploy/install.sh --no-restart       copy everything, restart nothing
#   deploy/install.sh --wait-live[=SECS] refuse new live sessions and wait (default 600 s) for the running ones to end
#   deploy/install.sh --force            deploy even if the checks below say no
# Deploy from ~/repos/Audio-Cleaner-Web on branch on-prem (fast-forwarded from desktop-merge); never from another checkout.
#
# Who runs what. The services run as the account acw (docs/security.md); you keep typing the same command as the owner
# (silver). Run by the owner, install.sh: checks the git side (clean tree, descendant of the deployed version), builds
# the native programs that changed (the Rust toolchain is the owner's), puts the exact commit (git archive HEAD) into
# a fresh folder owned by acw (/home/acw/deploy/<commit>), runs the SERVER PART there as acw (everything below that
# touches ~acw/audio-cleaner, the units and the services), and then does what needs the owner's VPS login key
# (deploy/publish-web.sh) and the smoke test. acw never has a VPS login key. Run by acw directly (or with
# --server-only), only the server part runs. On a machine without the acw account (/etc/acw-service-user) one
# account does all of it, as it used to.
#
# Before anything is touched it prints the commit and what is running, and refuses (unless --force) when
#   * the working tree has uncommitted changes,
#   * the commit is not a descendant of the deployed one (app/VERSION): that would downgrade the service,
#   * live sessions are running and the live process, its tunnel or the live worker would restart (they cut every
#     live session),
#   * the database backup taken just before the change fails its checks.
# It restarts only what changed: the web process, the live process (acw-live), the bulk-transfer process
# (uploads/downloads, acw-bulk) and the file workers when the application code or the locked packages changed, the live worker only when code it uses
# changed, and a unit whose unit file changed. The SSH tunnels are never restarted for a code change. A stopping
# file worker puts its job back in the queue; a restarting acw-bulk ends the uploads that are in flight.
# The previous copy of the code is kept as app.prev: deploy/rollback.sh brings it back.
# Whatever --force says, it also refuses when the Terms or the Privacy Policy text changed without a new version date, or the pin
# web/legal-versions.json is stale (deploy/legal-archive.sh; docs/operations.md, "Legal versions"). publish-web.sh archives them.
# Layout (acw's home): ~/audio-cleaner/{app,app.prev,venv,bin,models,data,secrets}
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=deploy/lib.sh
. "$REPO/deploy/lib.sh"

FORCE=0
RESTART=1
WAIT_LIVE=""
SERVER_ONLY=0     # the server part only (what the owner's run starts as acw)
PLAN=0            # server part, read-only: print what a deploy would need (lock changed, native builds), change nothing
PACKAGES_ONLY=0   # server part: install the locked packages and stop
LOCK_FORCED=0     # server part: treat the packages as changed (they were installed by --packages-only just before)
NATIVE_FROM=""    # server part: folder with the native programs the owner built (libdf.so, acw-live)
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --no-restart) RESTART=0 ;;
    --wait-live) WAIT_LIVE=600 ;;
    --wait-live=*) WAIT_LIVE="${arg#*=}" ;;
    --server-only) SERVER_ONLY=1 ;;
    --plan) PLAN=1; SERVER_ONLY=1 ;;
    --packages-only) PACKAGES_ONLY=1; SERVER_ONLY=1 ;;
    --lock-changed) LOCK_FORCED=1 ;;
    --native-from=*) NATIVE_FROM="${arg#*=}" ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown option $arg (see --help)" >&2; exit 2 ;;
  esac
done

# owner: the owner's shell with the acw account in place; server: acw (or --server-only); single: one account does it all
if [ "$SERVER_ONLY" = 1 ]; then ROLE=server
elif [ "$ACW_DELEGATE" = 1 ]; then ROLE=owner
elif [ -n "$ACW_SERVICE_USER" ]; then ROLE=server
else ROLE=single; fi
if [ "$ROLE" = server ] && [ "$ACW_DELEGATE" = 1 ]; then
  echo "--server-only is the part that runs as $ACW_SERVICE_USER; run deploy/install.sh without it and it is started as $ACW_SERVICE_USER for you" >&2
  exit 2
fi
QUIET=0
if [ "$PLAN" = 1 ] || [ "$PACKAGES_ONLY" = 1 ]; then QUIET=1; fi
say() { [ "$QUIET" = 1 ] || echo "$@"; }

# ---- Which commit is this? The owner's checkout and a single-account install ask git; the folder staged for acw has no
# .git, the owner's run wrote DEPLOY_INFO (commit, branch, subject, web tree) into it.
HAVE_GIT=0
[ -e "$REPO/.git" ] && HAVE_GIT=1
info() { { sed -n "s/^$1=//p" "$REPO/DEPLOY_INFO" 2>/dev/null || true; } | head -1; }
if [ "$HAVE_GIT" = 1 ]; then
  COMMIT="$(git -C "$REPO" rev-parse --short HEAD)"
  BRANCH="$(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
  SUBJECT="$(git -C "$REPO" log -1 --format=%s)"
else
  COMMIT="$(info commit)"; BRANCH="$(info branch)"; SUBJECT="$(info subject)"
  [ -n "$COMMIT" ] || { echo "$REPO has neither .git nor DEPLOY_INFO: run deploy/install.sh from the owner's checkout (it stages the commit for acw)" >&2; exit 1; }
fi
DEPLOYED="$(as_acw cat "$H/app/VERSION" 2>/dev/null || true)"
if [ "${ACW_FROM_OWNER:-}" != 1 ]; then  # (the owner's run printed this already)
  say "deploying $COMMIT ($BRANCH): $SUBJECT"
  [ -n "$DEPLOYED" ] && say "deployed now: $DEPLOYED"
fi

# ---- 1. Is this a deploy we want?
problems=()
if [ "$QUIET" = 0 ]; then
  if [ "$HAVE_GIT" = 1 ]; then
    if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
      problems+=("the working tree has uncommitted changes (commit first, so VERSION says what is running)")
    fi
    if [ -n "$DEPLOYED" ]; then
      if ! git -C "$REPO" cat-file -e "$DEPLOYED^{commit}" 2>/dev/null; then
        problems+=("the deployed version $DEPLOYED is not in this checkout (wrong repository?)")
      elif ! git -C "$REPO" merge-base --is-ancestor "$DEPLOYED" HEAD; then
        problems+=("$COMMIT is not a descendant of the deployed $DEPLOYED: deploying it would drop changes that are live now")
      fi
    fi
  fi
  # The app falls back to development behaviour without ACW_ENV=production (relaxed cookies, API docs on, ...).
  # (The secrets are acw's: the owner's run leaves this check to the server part.)
  if [ "$ROLE" != owner ]; then
    for f in acw.env acw-worker.env; do
      if [ -f "$H/secrets/$f" ] && ! grep -qx 'ACW_ENV=production' "$H/secrets/$f"; then
        problems+=("$H/secrets/$f does not say ACW_ENV=production")
      fi
    done
  fi
  if [ "${#problems[@]}" -gt 0 ]; then
    for p in "${problems[@]}"; do echo "problem: $p" >&2; done
    if [ "$FORCE" = 1 ]; then echo "continuing because of --force" >&2; else echo "refusing to deploy (--force overrides)" >&2; exit 1; fi
  fi

  # Syntax gate: a file that does not even parse never reaches the running service.
  python3 - "$REPO/server/acw" "$REPO/common/acw_common" <<'PY'
import ast, pathlib, sys
bad = 0
for root in sys.argv[1:]:
    for f in pathlib.Path(root).rglob("*.py"):
        try:
            ast.parse(f.read_text(), str(f))
        except SyntaxError as e:
            print(f"syntax error: {f}:{e.lineno}: {e.msg}", file=sys.stderr)
            bad = 1
sys.exit(bad)
PY

  # The Terms and the Privacy Policy as they are about to go live (docs/operations.md, "Legal versions"): the pin that the server
  # records with every acceptance has to match the pages, and a text changed under an unchanged version date stops the deploy here,
  # before anything is restarted. (Nothing is written yet: deploy/publish-web.sh archives the versions when the website goes live.)
  if [ "$ROLE" != server ]; then
    bash "$REPO/deploy/legal-archive.sh" --check || { echo "refusing to deploy: the Terms / Privacy Policy versions (above)" >&2; exit 1; }
  fi
fi

# ---- The parts that need the owner's VPS login key and the owner's checkout: the website and the smoke test (which
# runs `ssh vps`). Run by the owner (or, with one account, by that account) once everything else went well.
vps_parts() {  # vps_parts STATUS -> the status after both
  local status="$1" web_tree published
  # ---- 13. Website: publish only when web/ changed since the last publish.
  if [ "$status" = 0 ]; then
    web_tree=$(git -C "$REPO" rev-parse HEAD:web)
    published="$(as_acw cat "$H/data/web-published" 2>/dev/null || true)"
    if [ -n "$(git -C "$REPO" status --porcelain -- web)" ]; then
      echo "web/ has uncommitted changes: not publishing the website (commit first, or run deploy/publish-web.sh)" >&2
    elif [ "$web_tree" != "$published" ]; then
      echo "web/ changed since the last publish: publishing the website"
      bash "$REPO/deploy/publish-web.sh"  # records the published web/ tree itself
    else
      echo "website unchanged (already published)"
    fi
  fi
  # ---- 14. Smoke test: what users and the apps reach, from outside (deploy/smoke.py). It changes nothing; a failure here
  # is reported (and fails the deploy's exit status) but rolls nothing back: it can also be Hugging Face or a certificate.
  if [ "$status" = 0 ] && [ "$RESTART" = 1 ]; then
    echo "smoke test:"
    python3 "$REPO/deploy/smoke.py" --quiet || { status=1; echo "the smoke test found problems (above); the deploy itself is in place" >&2; }
  fi
  return "$status"
}

# ================================================================== the owner's run: hand over to acw
if [ "$ROLE" = owner ]; then
  as_acw true 2>/dev/null \
    || { echo "cannot run commands as $ACW_SERVICE_USER without a password (sudo -n -u $ACW_SERVICE_USER true failed): see /etc/sudoers.d/acw, deploy/host/move-to-acw.sh" >&2; exit 1; }
  STAGE="$(stage_for_acw "$COMMIT" deploy server common native web/legal-versions.json)"
  printf 'commit=%s\nbranch=%s\nsubject=%s\n' "$COMMIT" "$BRANCH" "$SUBJECT" | put_as_acw /dev/stdin "$STAGE/DEPLOY_INFO" 644
  echo "staged $COMMIT for $ACW_SERVICE_USER in $STAGE"
  pass=()
  [ "$FORCE" = 1 ] && pass+=(--force)
  [ "$RESTART" = 0 ] && pass+=(--no-restart)
  [ -n "$WAIT_LIVE" ] && pass+=("--wait-live=$WAIT_LIVE")
  # sudo drops the environment: hand over the few settings that are meant to reach the server part.
  envs=()
  for v in ACW_WORKERS ACW_MEMORY_PERCENT ACW_ARCHIVE_DIR; do [ -n "${!v:-}" ] && envs+=("$v=${!v}"); done
  envs+=(ACW_FROM_OWNER=1)  # the server part then leaves out what the owner's run says itself (header, "run as the owner")
  server() { as_acw env ${envs[@]+"${envs[@]}"} bash "$STAGE/deploy/install.sh" --server-only "$@"; }

  # What would the server part need that only the owner can do? (read-only)
  plan="$(server --plan)" || { echo "the server part could not plan the deploy (above)" >&2; exit 1; }
  plan_get() { printf '%s\n' "$plan" | sed -n "s/^plan .*\\b$1=\\([01]\\).*/\\1/p" | head -1; }
  NEED_DF="$(plan_get need_df)"; NEED_LIVE="$(plan_get need_live)"; LOCK_NOW="$(plan_get lock_changed)"
  [ -n "$NEED_DF" ] && [ -n "$NEED_LIVE" ] && [ -n "$LOCK_NOW" ] || { echo "unreadable plan from the server part: $plan" >&2; exit 1; }
  extra=()
  if [ "$NEED_LIVE" = 1 ] && [ "$LOCK_NOW" = 1 ]; then
    # acw-live links the sherpa-onnx library of the venv: install the new packages first, build against them.
    echo "the locked packages changed and acw-live must be rebuilt: installing the packages first"
    server --packages-only || exit 1
    extra+=(--lock-changed)
  fi
  # ---- 7. Native code: built here (the Rust toolchain is the owner's), installed by the server part.
  OUTDIR="$STAGE/native-out"
  as_acw mkdir -p -m 700 "$OUTDIR"
  # shellcheck disable=SC1091
  if [ "$NEED_DF$NEED_LIVE" != 00 ]; then command -v cargo >/dev/null 2>&1 || source "$HOME/.cargo/env"; fi
  if [ "$NEED_DF" = 1 ]; then
    DFB="$HOME/audio-cleaner/src/deepfilter-build"   # the build tree stays with the owner
    BUILD_DIR="$DFB" OUT="$DFB/out" bash "$REPO/native/deepfilter/build.sh" native
    put_as_acw "$DFB/out/libdf.so" "$OUTDIR/libdf.so" 755
  fi
  if [ "$NEED_LIVE" = 1 ]; then
    tmp="$(mktemp -d)"
    # The engine records where sherpa-onnx lives (rpath and its sandbox): acw's venv, where it will run.
    ACW_PYTHON="$H/venv/bin/python" bash "$REPO/native/live/build.sh" "$tmp"
    put_as_acw "$tmp/acw-live" "$OUTDIR/acw-live" 755
    rm -rf "$tmp"
  fi

  rc=0
  server "${pass[@]+"${pass[@]}"}" ${extra[@]+"${extra[@]}"} "--native-from=$OUTDIR" || rc=$?
  if [ "$rc" != 0 ]; then
    echo "the server part failed (exit $rc); nothing was published and no smoke test ran. Logs: acw-logs acw-web -n 80; roll back with deploy/rollback.sh" >&2
    exit "$rc"
  fi
  vps_status=0
  vps_parts 0 || vps_status=$?
  exit "$vps_status"
fi

# ================================================================== the server part (acw), or the single account
mkdir -p "$H"/{app/acw,app/acw_common,bin,models,data,secrets}
chmod 700 "$H/data" "$H/secrets"
# The tunnels' destination (user@address of the VPS) lives here, not in the repo.
if ! grep -Eq '^ACW_VPS=[^@[:space:]]+@[^[:space:]]+$' "$H/secrets/vps.env" 2>/dev/null || grep -q '^ACW_VPS=user@vps-address$' "$H/secrets/vps.env"; then
  echo "error: $H/secrets/vps.env needs ACW_VPS=<user>@<VPS address> (see deploy/vps.env.example)" >&2
  exit 1
fi

# ---- 2. What would change?
changed_files() {  # changed_files SRC DST: paths (relative) rsync would add, change or delete
  rsync -ainc --delete --filter='-p __pycache__/' "$1" "$2" | grep -v '^\.d' | sed -E 's/^[^ ]+ +//' || true
}
CHANGED_APP="$(changed_files "$REPO/server/acw/" "$H/app/acw/")"
CHANGED_COMMON="$(changed_files "$REPO/common/acw_common/" "$H/app/acw_common/")"
# The locked packages are the name==version pins: the --hash lines can change without any package changing, so a
# change of hashes alone neither reinstalls nor restarts anything. When the recorded pin set differs, the
# installed packages decide (so the first deploy after the lock gained hashes restarts nothing for that).
lock_pins() { grep -E '^[A-Za-z0-9]' "$1" | sed -E 's/[ \\].*$//' | tr 'A-Z_.' 'a-z--' | sort; }
LOCK_SUM="pins:$(lock_pins "$REPO/server/requirements.lock" | sha256sum | cut -d' ' -f1)"
LOCK_CHANGED=0
if [ ! -x "$H/venv/bin/python" ]; then
  LOCK_CHANGED=1
elif [ "$LOCK_SUM" != "$(cat "$H/data/lock.sha" 2>/dev/null || true)" ]; then
  if [ -n "$(comm -23 <(lock_pins "$REPO/server/requirements.lock") <("$H/venv/bin/python" -m pip freeze 2>/dev/null | sed -E 's/ .*//' | tr 'A-Z_.' 'a-z--' | sort))" ]; then
    LOCK_CHANGED=1
  else
    [ "$PLAN" = 1 ] || echo "$LOCK_SUM" > "$H/data/lock.sha"  # same packages as before, only the recorded form changed
  fi
fi
[ "$LOCK_FORCED" = 1 ] && LOCK_CHANGED=1
APP_CHANGED=0
[ -n "$CHANGED_APP$CHANGED_COMMON" ] && APP_CHANGED=1

# Native programs: libDF (native/deepfilter: the pinned DeepFilterNet commit + tract-0.23.patch, bin/libdf.so) and the
# live engine (native/live, bin/acw-live) are rebuilt when their sources change (the stamp is a hash of the sources).
# Both are required: a failed build stops the deploy and leaves the running ones as they are. File jobs load libDF per
# stage, so only the engine restarts.
DF_STAMP="$(cat "$REPO/native/deepfilter/build.sh" "$REPO/native/deepfilter/tract-0.23.patch" "$REPO/native/deepfilter/Cargo.lock" | sha256sum | cut -c1-16)"
LIVE_STAMP="$(cd "$REPO/native/live" && cat Cargo.toml Cargo.lock build.rs src/*.rs | sha256sum | cut -c1-16)"
NEED_DF=0; NEED_LIVE=0
{ [ ! -f "$H/bin/libdf.so" ] || [ "$(cat "$H/bin/libdf.so.src" 2>/dev/null)" != "$DF_STAMP" ]; } && NEED_DF=1
{ [ ! -x "$H/bin/acw-live" ] || [ "$(cat "$H/bin/acw-live.src" 2>/dev/null)" != "$LIVE_STAMP" ]; } && NEED_LIVE=1
if [ "$PLAN" = 1 ]; then  # the owner's run asks this before it builds anything; one line, nothing changed
  echo "plan lock_changed=$LOCK_CHANGED need_df=$NEED_DF need_live=$NEED_LIVE"
  exit 0
fi

# ---- 5 (early, --packages-only). Python packages (pinned); only when the lock changed, and without touching pip itself
if [ "$PACKAGES_ONLY" = 1 ]; then
  [ -x "$H/venv/bin/python" ] || python3 -m venv "$H/venv"
  "$H/venv/bin/pip" install -q --require-hashes -r "$REPO/server/requirements.lock"
  echo "$LOCK_SUM" > "$H/data/lock.sha"
  exit 0
fi

# Sized from this machine, so a bigger CPU or more RAM is used without edits:
# all Audio Cleaner services may use ACW_MEMORY_PERCENT of RAM (the rest stays
# with the game servers and the system); one file worker per 3 physical cores.
# Live audio capacity scales by itself (core count + start-up calibration).
MEM_KB=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
PCT="${ACW_MEMORY_PERCENT:-40}"
PHYS=$(lscpu -p=core,socket | grep -v '^#' | sort -u | wc -l)
WORKERS="${ACW_WORKERS:-$(( PHYS / 3 < 2 ? 2 : (PHYS / 3 > 8 ? 8 : PHYS / 3) ))}"
U="$HOME/.config/systemd/user"
SIZE_CONF="$(printf '[Slice]\nMemoryHigh=%dK\nMemoryMax=%dK\n' $(( MEM_KB * (PCT - 3) / 100 )) $(( MEM_KB * PCT / 100 )))"
WORKERS_CONF="$(printf '[Service]\nEnvironment=ACW_WORKERS=%d\n' "$WORKERS")"

GPU=0
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1 && [ -x "$H/venv-gpu/bin/python" ]; then GPU=1; fi

UNITS="acw-stream.service acw-web.service acw-live.service acw-bulk.service acw-tunnel.service acw-tunnel-live.service acw-admin.service"
# The bulk tunnel needs the VPS key to allow its port first (docs/operations.md, "Bulk transfers"), otherwise it
# fails and alerts in a loop. So it is part of the deploy only once you enabled it (acw-ctl enable --now
# acw-tunnel-bulk); from then on it is checked, restarted when its unit file changes and rolled back like the rest.
BULK_TUNNEL=0
systemctl --user is-enabled --quiet acw-tunnel-bulk.service 2>/dev/null && BULK_TUNNEL=1
[ "$BULK_TUNNEL" = 1 ] && UNITS="$UNITS acw-tunnel-bulk.service"
for i in $(seq 1 "$WORKERS"); do UNITS="$UNITS acw-worker@$i.service"; done
[ "$GPU" = 1 ] && UNITS="$UNITS acw-worker-gpu.service"

RESTART_UNITS=()
add_restart() {
  local x
  for x in ${RESTART_UNITS[@]+"${RESTART_UNITS[@]}"}; do [ "$x" = "$1" ] && return 0; done
  RESTART_UNITS+=("$1")
}
if [ "$APP_CHANGED" = 1 ] || [ "$LOCK_CHANGED" = 1 ]; then
  add_restart acw-web.service
  add_restart acw-live.service
  add_restart acw-bulk.service
  add_restart acw-admin.service
  for i in $(seq 1 "$WORKERS"); do add_restart "acw-worker@$i.service"; done
  [ "$GPU" = 1 ] && add_restart acw-worker-gpu.service
fi
# A unit whose own unit file changed (never the others, in particular not the tunnel).
for f in "$REPO"/deploy/systemd/acw-*.service; do
  name="$(basename "$f")"
  if [ -f "$U/$name" ] && ! cmp -s "$f" "$U/$name"; then
    case "$name" in
      acw-worker@.service) for i in $(seq 1 "$WORKERS"); do add_restart "acw-worker@$i.service"; done ;;
      acw-alert@.service) ;;  # started on demand
      acw-mirror.service) ;;  # run by its timer (a restart would start a download here and now)
      acw-tunnel-bulk.service) [ "$BULK_TUNNEL" = 1 ] && add_restart "$name" ;;  # only once the VPS allows its port
      *) add_restart "$name" ;;
    esac
  fi
done
[ "$(cat "$U/acw-web.service.d/workers.conf" 2>/dev/null || true)" != "$WORKERS_CONF" ] && [ -f "$U/acw-web.service" ] && add_restart acw-web.service

echo "host: $PHYS physical cores, $(( MEM_KB / 1048576 )) GB RAM -> $WORKERS file workers, ${PCT}% RAM cap"
if [ -x "$H/venv/bin/python" ] && [ -f "$H/secrets/acw-worker.env" ] && [ -d "$H/app/acw" ]; then
  echo "running now: $(acw_admin activity 2>/dev/null || echo unknown)"
fi
if [ "$RESTART" = 1 ]; then
  echo "will restart: ${RESTART_UNITS[*]:-nothing (no code, package or unit change)}"
else
  echo "--no-restart: nothing is restarted"
fi

# ---- 3. Live sessions: a live, stream or live-tunnel restart cuts all of them (a web or bulk restart does not)
DRAINING=0
cleanup() { [ "$DRAINING" = 1 ] && drain_off; true; }
trap cleanup EXIT
if [ "$RESTART" = 1 ]; then
  cuts_live=0
  for u in ${RESTART_UNITS[@]+"${RESTART_UNITS[@]}"}; do case "$u" in acw-live.service|acw-stream.service|acw-tunnel-live.service) cuts_live=1 ;; esac; done
  # A native program that is going to be replaced restarts the live worker too (step 7 adds it).
  [ "$NEED_DF" = 1 ] || [ "$NEED_LIVE" = 1 ] && cuts_live=1
  LIVE="$(live_sessions)"
  if [ "$cuts_live" = 1 ] && [ "$LIVE" != "0" ]; then
    if [ -n "$WAIT_LIVE" ]; then
      mkdir -p "$H/data"; drain_on; DRAINING=1
      echo "new live sessions are refused until the restart is done"
      if ! wait_for_live "$WAIT_LIVE"; then
        if [ "$FORCE" = 1 ]; then echo "$(live_sessions) live session(s) still running: restarting anyway (--force)" >&2
        else echo "live sessions are still running after ${WAIT_LIVE}s; not deploying (--force restarts anyway)" >&2; exit 1; fi
      fi
    elif [ "$FORCE" = 1 ]; then
      echo "$LIVE live session(s) will be cut (--force)" >&2
    else
      echo "$LIVE live session(s) are running and this deploy restarts the process or worker they run on." >&2
      echo "Use --wait-live[=SECONDS] to stop new sessions and wait for these to end, or --force." >&2
      exit 1
    fi
  fi
fi

# ---- 4. Snapshot the database with the code that is running now (init-db below may change the schema)
if [ -f "$H/data/acw.sqlite3" ] && [ -d "$H/app/acw" ] && [ -x "$H/venv/bin/python" ]; then
  if ! acw_admin backup "$H/data/backups" --keep 10 >/dev/null; then
    if [ "$FORCE" = 1 ]; then echo "warning: the pre-deploy database backup failed (--force)" >&2
    else echo "the pre-deploy database backup failed its checks; not deploying (--force overrides)" >&2; exit 1; fi
  else
    echo "database backed up to $H/data/backups"
  fi
fi

# ---- 5. Python packages (pinned); only when the lock changed, and without touching pip itself
[ -x "$H/venv/bin/python" ] || python3 -m venv "$H/venv"
if [ "$LOCK_CHANGED" = 1 ]; then
  "$H/venv/bin/pip" install -q --require-hashes -r "$REPO/server/requirements.lock"
  echo "$LOCK_SUM" > "$H/data/lock.sha"
fi

# ---- 6. Code: a copy, so editing the git checkout never changes production; the previous copy is kept
if [ "$APP_CHANGED" = 1 ] && [ -d "$H/app/acw" ]; then
  rm -rf "$H/app.prev"
  cp -a "$H/app" "$H/app.prev"
fi
rsync -ac --delete --filter='-p __pycache__/' "$REPO/server/acw/" "$H/app/acw/"
rsync -ac --delete --filter='-p __pycache__/' "$REPO/common/acw_common/" "$H/app/acw_common/"
echo "$COMMIT" > "$H/app/VERSION"
# The release mirror reads the archive list of the model files from here (the code copy above has no deploy/ folder).
install -m 600 "$REPO/deploy/models.lock" "$H/app/models.lock"
# The SHA-256 of the Terms text that goes into the audit record of every acceptance (acw/auth.py reads it from here). One source of
# truth: web/legal-versions.json, which web/tests/legal-versions.mjs and deploy/legal-archive.sh keep equal to the page.
install -m 600 "$REPO/web/legal-versions.json" "$H/app/legal-versions.json"
chmod -R go-rwx "$H/app"

# ---- 7. Native code, models and programs (see the stamps above). With the owner's run the programs arrive built
# (--native-from); this account has no Rust toolchain. A single account builds them itself.
install_native() {  # install_native NAME DESTINATION STAMP-FILE STAMP
  if [ -n "$NATIVE_FROM" ] && [ -f "$NATIVE_FROM/$1" ]; then
    install -m 755 "$NATIVE_FROM/$1" "$2.new"
    mv -f "$2.new" "$2"
    echo "$4" > "$3"
    return 0
  fi
  return 1
}
if [ "$NEED_DF" = 1 ]; then
  if ! install_native libdf.so "$H/bin/libdf.so" "$H/bin/libdf.so.src" "$DF_STAMP"; then
    # shellcheck disable=SC1091
    command -v cargo >/dev/null 2>&1 || { [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"; } || true
    command -v cargo >/dev/null 2>&1 || { echo "libDF has to be rebuilt (native/deepfilter changed) and this account has no Rust toolchain: run deploy/install.sh as the owner, who builds it and hands it over" >&2; exit 1; }
    BUILD_DIR="$H/src/deepfilter-build" OUT="$H/src/deepfilter-build/out" bash "$REPO/native/deepfilter/build.sh" native
    install -m 755 "$H/src/deepfilter-build/out/libdf.so" "$H/bin/.libdf.so.new"
    mv -f "$H/bin/.libdf.so.new" "$H/bin/libdf.so"
    echo "$DF_STAMP" > "$H/bin/libdf.so.src"
  fi
  add_restart acw-stream.service
fi
if [ "$NEED_LIVE" = 1 ]; then
  if ! install_native acw-live "$H/bin/acw-live" "$H/bin/acw-live.src" "$LIVE_STAMP"; then
    # shellcheck disable=SC1091
    command -v cargo >/dev/null 2>&1 || { [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"; } || true
    command -v cargo >/dev/null 2>&1 || { echo "the live engine has to be rebuilt (native/live changed) and this account has no Rust toolchain: run deploy/install.sh as the owner, who builds it and hands it over" >&2; exit 1; }
    bash "$REPO/native/live/build.sh" "$H/bin"
    echo "$LIVE_STAMP" > "$H/bin/acw-live.src"
  fi
  add_restart acw-stream.service
fi
# Every file in deploy/models.lock is present and has the recorded sha256 (downloads what is missing or wrong, never
# touches a correct file; FFmpeg and the models archive come from the lock's pinned URLs, or from
# $ACW_ARCHIVE_DIR). Every feature needs its files: anything missing stops the deploy.
bash "$REPO/deploy/fetch-models.sh" || { echo "models or programs missing or wrong (deploy/fetch-models.sh --check -v lists them)" >&2; exit 1; }

# ---- 8. Config
if [ ! -f "$H/secrets/acw.env" ]; then
  sed "s|^ACW_SESSION_SECRET=$|ACW_SESSION_SECRET=$("$H/venv/bin/python" -c 'import secrets;print(secrets.token_urlsafe(48))')|" \
    "$REPO/deploy/acw.env.example" > "$H/secrets/acw.env"
  echo "created $H/secrets/acw.env - fill in provider/Stripe/OpenAI keys"
fi
[ -f "$H/secrets/acw-worker.env" ] || cp "$REPO/deploy/acw-worker.env.example" "$H/secrets/acw-worker.env"
chmod 600 "$H/secrets/"*.env

# ---- 9. Database schema (new code; additive changes only)
acw_admin init-db >/dev/null

# ---- 10. systemd user units (in this account's manager)
mkdir -p "$U" "$U/acw.slice.d" "$U/acw-web.service.d"
units_changed=0
for f in "$REPO"/deploy/systemd/*.service "$REPO"/deploy/systemd/*.timer "$REPO"/deploy/systemd/*.slice; do
  cmp -s "$f" "$U/$(basename "$f")" 2>/dev/null || { install -m 644 "$f" "$U/"; units_changed=1; }
done
[ "$(cat "$U/acw.slice.d/size.conf" 2>/dev/null || true)" = "$SIZE_CONF" ] || { printf '%s\n' "$SIZE_CONF" > "$U/acw.slice.d/size.conf"; units_changed=1; }
[ "$(cat "$U/acw-web.service.d/workers.conf" 2>/dev/null || true)" = "$WORKERS_CONF" ] || { printf '%s\n' "$WORKERS_CONF" > "$U/acw-web.service.d/workers.conf"; units_changed=1; }
[ "$units_changed" = 1 ] && systemctl --user daemon-reload
systemctl --user enable --now acw-backup.timer >/dev/null 2>&1
# Health monitor + email alerts (acw-alert@.service is started by OnFailure=, never enabled)
systemctl --user enable --now acw-monitor.timer >/dev/null 2>&1
# Release mirror: copies, checks and publishes the newest OBS plugin release every 10 minutes (the first run starts a
# minute after the timer).
systemctl --user enable --now acw-mirror.timer >/dev/null 2>&1
if [ "$BULK_TUNNEL" = 0 ]; then
  echo "note: acw-tunnel-bulk (uploads/downloads on their own SSH connection) is installed but not enabled: after the VPS key allows port 18794, run 'acw-ctl enable --now acw-tunnel-bulk' (docs/operations.md, Bulk transfers)" >&2
fi
# Workers beyond the new count (moved to a smaller machine): stop them.
for i in $(seq $(( WORKERS + 1 )) 16); do systemctl --user disable --now "acw-worker@$i.service" >/dev/null 2>&1 || true; done
[ "$GPU" = 1 ] || systemctl --user disable --now acw-worker-gpu.service >/dev/null 2>&1 || true
# shellcheck disable=SC2086
systemctl --user enable $UNITS >/dev/null 2>&1

if [ "$RESTART" = 1 ]; then
  systemctl --user is-active --quiet acw-mirror.timer || echo "warning: acw-mirror.timer is not active (acw-ctl status acw-mirror.timer): release downloads will not update" >&2
  echo "release downloads: $(acw_admin mirror --status 2>/dev/null | tr '\n' '|' || echo unknown)"
fi

# ---- 11. Restart what changed, then check everything
STATUS=0
if [ "$RESTART" = 1 ]; then
  # Units that are not running yet (a new worker, a unit that was stopped) are simply started.
  for u in $UNITS; do
    systemctl --user is-active --quiet "$u" || add_restart "$u"
  done
  # The live worker first: the live process connects to it.
  ordered=()
  for u in ${RESTART_UNITS[@]+"${RESTART_UNITS[@]}"}; do [ "$u" = acw-stream.service ] && ordered+=("$u"); done
  for u in ${RESTART_UNITS[@]+"${RESTART_UNITS[@]}"}; do [ "$u" != acw-stream.service ] && ordered+=("$u"); done
  if [ "${#ordered[@]}" -gt 0 ]; then
    systemctl --user restart "${ordered[@]}"
  fi
  # shellcheck disable=SC2086
  check_running $UNITS || STATUS=1
  systemctl --user --no-pager --plain list-units 'acw-*' | sed -n '1,16p'
  if [ "$STATUS" != 0 ]; then
    echo "The deploy did not come up cleanly. Previous code is in $H/app.prev: run deploy/rollback.sh" >&2
  fi
fi
drain_off; DRAINING=0

# ---- 12. Logs: the user journal should not keep client addresses longer than the Privacy Policy says (90 days)
if ! grep -rqs 'MaxRetentionSec' /etc/systemd/journald.conf /etc/systemd/journald.conf.d 2>/dev/null; then
  echo "note: set a journal retention once (needs sudo): see 'Log retention' in docs/operations.md" >&2
fi

# ---- 13 and 14. Website and smoke test: the owner's (VPS login key). Not done by acw.
if [ "$ROLE" = server ]; then
  [ "$STATUS" = 0 ] && [ "${ACW_FROM_OWNER:-}" != 1 ] && echo "server part done. The website and the smoke test need the owner's VPS login: run deploy/install.sh as the owner (or deploy/publish-web.sh, python3 deploy/smoke.py)"
  exit "$STATUS"
fi
vps_status=0
vps_parts "$STATUS" || vps_status=$?
exit "$vps_status"
