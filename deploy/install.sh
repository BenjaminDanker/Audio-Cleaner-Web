#!/usr/bin/env bash
# Install or update Audio Cleaner on primary-ram (no sudo needed).
#   deploy/install.sh                    sync code, deps, units; restart what changed; publish the website if web/ changed
#   deploy/install.sh --no-restart       copy everything, restart nothing
#   deploy/install.sh --wait-live[=SECS] refuse new live sessions and wait (default 600 s) for the running ones to end
#   deploy/install.sh --force            deploy even if the checks below say no
# Deploy from ~/repos/Audio-Cleaner-Web on branch on-prem (fast-forwarded from desktop-merge); never from another checkout.
#
# Before anything is touched it prints the commit and what is running, and refuses (unless --force) when
#   * the working tree has uncommitted changes,
#   * the commit is not a descendant of the deployed one (app/VERSION): that would downgrade the service,
#   * live sessions are running and the web or stream worker would restart (they cut every live session),
#   * the database backup taken just before the change fails its checks.
# It restarts only what changed: the web process and the file workers when the application code or the
# locked packages changed, the live worker only when code it uses changed, and a unit whose unit file changed.
# The SSH tunnel is never restarted for a code change. A stopping file worker puts its job back in the queue.
# The previous copy of the code is kept as app.prev: deploy/rollback.sh brings it back.
# Layout: ~/audio-cleaner/{app,app.prev,venv,bin,models,data,secrets}
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=deploy/lib.sh
. "$REPO/deploy/lib.sh"

FORCE=0
RESTART=1
WAIT_LIVE=""
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --no-restart) RESTART=0 ;;
    --wait-live) WAIT_LIVE=600 ;;
    --wait-live=*) WAIT_LIVE="${arg#*=}" ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "unknown option $arg (see --help)" >&2; exit 2 ;;
  esac
done

COMMIT="$(git -C "$REPO" rev-parse --short HEAD)"
echo "deploying $COMMIT ($(git -C "$REPO" rev-parse --abbrev-ref HEAD)): $(git -C "$REPO" log -1 --format=%s)"
DEPLOYED="$(cat "$H/app/VERSION" 2>/dev/null || true)"
[ -n "$DEPLOYED" ] && echo "deployed now: $DEPLOYED"

# ---- 1. Is this a deploy we want?
problems=()
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

mkdir -p "$H"/{app/acw,app/acw_common,bin,models,data,secrets}
chmod 700 "$H/data" "$H/secrets"

# ---- 2. What would change?
changed_files() {  # changed_files SRC DST: paths (relative) rsync would add, change or delete
  rsync -ainc --delete --exclude '__pycache__' "$1" "$2" | grep -v '^\.d' | sed -E 's/^[^ ]+ +//' || true
}
CHANGED_APP="$(changed_files "$REPO/server/acw/" "$H/app/acw/")"
CHANGED_COMMON="$(changed_files "$REPO/common/acw_common/" "$H/app/acw_common/")"
LOCK_SUM="$(sha256sum "$REPO/server/requirements.lock" | cut -d' ' -f1)"
LOCK_CHANGED=0
{ [ ! -x "$H/venv/bin/python" ] || [ "$LOCK_SUM" != "$(cat "$H/data/lock.sha" 2>/dev/null || true)" ]; } && LOCK_CHANGED=1
APP_CHANGED=0
[ -n "$CHANGED_APP$CHANGED_COMMON" ] && APP_CHANGED=1
STREAM_CHANGED=0
if [ -n "$CHANGED_COMMON" ] || [ "$LOCK_CHANGED" = 1 ] || \
   printf '%s\n' "$CHANGED_APP" | grep -Eq '^(stream/(worker|engine|protocol)\.py|cpusplit\.py|hostload\.py|config\.py|worker/sandbox\.py)$'; then
  STREAM_CHANGED=1
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

UNITS="acw-stream.service acw-web.service acw-tunnel.service"
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
  for i in $(seq 1 "$WORKERS"); do add_restart "acw-worker@$i.service"; done
  [ "$GPU" = 1 ] && add_restart acw-worker-gpu.service
fi
[ "$STREAM_CHANGED" = 1 ] && [ "$APP_CHANGED$LOCK_CHANGED" != "00" ] && add_restart acw-stream.service
# A unit whose own unit file changed (never the others, in particular not the tunnel).
for f in "$REPO"/deploy/systemd/acw-*.service; do
  name="$(basename "$f")"
  if [ -f "$U/$name" ] && ! cmp -s "$f" "$U/$name"; then
    case "$name" in
      acw-worker@.service) for i in $(seq 1 "$WORKERS"); do add_restart "acw-worker@$i.service"; done ;;
      acw-alert@.service) ;;  # started on demand
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

# ---- 3. Live sessions: a web or stream restart cuts all of them
DRAINING=0
cleanup() { [ "$DRAINING" = 1 ] && drain_off; true; }
trap cleanup EXIT
if [ "$RESTART" = 1 ]; then
  cuts_live=0
  for u in ${RESTART_UNITS[@]+"${RESTART_UNITS[@]}"}; do case "$u" in acw-web.service|acw-stream.service|acw-tunnel.service) cuts_live=1 ;; esac; done
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
      echo "$LIVE live session(s) are running and this deploy restarts the web or live worker." >&2
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
  "$H/venv/bin/pip" install -q -r "$REPO/server/requirements.lock"
  echo "$LOCK_SUM" > "$H/data/lock.sha"
fi

# ---- 6. Code: a copy, so editing the git checkout never changes production; the previous copy is kept
if [ "$APP_CHANGED" = 1 ] && [ -d "$H/app/acw" ]; then
  rm -rf "$H/app.prev"
  cp -a "$H/app" "$H/app.prev"
fi
rsync -ac --delete --exclude '__pycache__' "$REPO/server/acw/" "$H/app/acw/"
rsync -ac --delete --exclude '__pycache__' "$REPO/common/acw_common/" "$H/app/acw_common/"
git -C "$REPO" rev-parse --short HEAD > "$H/app/VERSION"
chmod -R go-rwx "$H/app"

# ---- 7. Native pieces must exist (built by native/deepfilter/build.sh and fetched below)
for f in "$H/bin/ffmpeg" "$H/bin/ffprobe" "$H/bin/libdf.so" "$H/models/DeepFilterNet3_onnx.tar.gz"; do
  [ -e "$f" ] || { echo "missing $f (see docs/operations.md)" >&2; exit 1; }
done
[ -d "$H/models/whisper/large-v3-turbo" ] || echo "warning: whisper large-v3-turbo missing (local transcription will fail)" >&2
[ -f "$H/models/captions-en/encoder.int8.onnx" ] || echo "warning: captions-en model missing (standard live captions will fail)" >&2
# Heavy-noise model for recordings (small; pinned + verified). Without it heavy noise falls back to DeepFilterNet3.
bash "$REPO/deploy/fetch-dpdfnet.sh" "$H/models/dpdfnet" || echo "warning: DPDFNet model download failed (heavy-noise recordings use DeepFilterNet3)" >&2
[ -f "$H/models/sidon/sidon_fe.onnx" ] || echo "warning: Sidon ONNX models missing in $H/models/sidon (docs/research/restore-sidon.md); voice-rebuild jobs are billed at the standard price and say so" >&2
[ -f "$H/models/ap-bwe/8kto48k.onnx" ] || echo "warning: AP-BWE models missing in $H/models/ap-bwe (tools/export_apbwe_onnx.py); band-limited recordings are not extended" >&2

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

# ---- 10. systemd user units
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
# Workers beyond the new count (moved to a smaller machine): stop them.
for i in $(seq $(( WORKERS + 1 )) 16); do systemctl --user disable --now "acw-worker@$i.service" >/dev/null 2>&1 || true; done
[ "$GPU" = 1 ] || systemctl --user disable --now acw-worker-gpu.service >/dev/null 2>&1 || true
# shellcheck disable=SC2086
systemctl --user enable $UNITS >/dev/null 2>&1

# ---- 11. Restart what changed, then check everything
STATUS=0
if [ "$RESTART" = 1 ]; then
  # Units that are not running yet (a new worker, a unit that was stopped) are simply started.
  for u in $UNITS; do
    systemctl --user is-active --quiet "$u" || add_restart "$u"
  done
  # The live worker first: the web process connects to it.
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

# ---- 13. Website: publish only when web/ changed since the last publish.
if [ "$STATUS" = 0 ]; then
  WEB_TREE=$(git -C "$REPO" rev-parse HEAD:web)
  if [ -n "$(git -C "$REPO" status --porcelain -- web)" ]; then
    echo "web/ has uncommitted changes: not publishing the website (commit first, or run deploy/publish-web.sh)" >&2
  elif [ "$WEB_TREE" != "$(cat "$H/data/web-published" 2>/dev/null || true)" ]; then
    echo "web/ changed since the last publish: publishing the website"
    bash "$REPO/deploy/publish-web.sh"  # records the published web/ tree itself
  else
    echo "website unchanged (already published)"
  fi
fi
exit "$STATUS"
