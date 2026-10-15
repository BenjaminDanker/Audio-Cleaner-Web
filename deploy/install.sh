#!/usr/bin/env bash
# Install or update Audio Cleaner on primary-ram (no sudo needed).
#   deploy/install.sh            # sync code, deps, units; restart services
#   deploy/install.sh --no-restart
# Layout: ~/audio-cleaner/{app,venv,bin,models,data,secrets}
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
H="$HOME/audio-cleaner"
mkdir -p "$H"/{app,bin,models,data,secrets}
chmod 700 "$H/data" "$H/secrets"

# 1. Python deps (pinned)
[ -x "$H/venv/bin/python" ] || python3 -m venv "$H/venv"
"$H/venv/bin/pip" install -q --upgrade pip
"$H/venv/bin/pip" install -q -r "$REPO/server/requirements.lock"

# 2. Code: a copy, so editing the git checkout never changes production
rsync -a --delete --exclude '__pycache__' "$REPO/server/acw/" "$H/app/acw/"
rsync -a --delete --exclude '__pycache__' "$REPO/common/acw_common/" "$H/app/acw_common/"
git -C "$REPO" rev-parse --short HEAD > "$H/app/VERSION"
chmod -R go-rwx "$H/app"

# 3. Native pieces must exist (built by native/deepfilter/build.sh and fetched below)
for f in "$H/bin/ffmpeg" "$H/bin/ffprobe" "$H/bin/libdf.so" "$H/models/DeepFilterNet3_onnx.tar.gz"; do
  [ -e "$f" ] || { echo "missing $f (see docs/operations.md)" >&2; exit 1; }
done
[ -d "$H/models/whisper/large-v3-turbo" ] || echo "warning: whisper large-v3-turbo missing (local transcription will fail)" >&2
[ -f "$H/models/captions-en/encoder.int8.onnx" ] || echo "warning: captions-en model missing (standard live captions will fail)" >&2

# 4. Config
if [ ! -f "$H/secrets/acw.env" ]; then
  sed "s|^ACW_SESSION_SECRET=$|ACW_SESSION_SECRET=$("$H/venv/bin/python" -c 'import secrets;print(secrets.token_urlsafe(48))')|" \
    "$REPO/deploy/acw.env.example" > "$H/secrets/acw.env"
  echo "created $H/secrets/acw.env - fill in provider/Stripe/OpenAI keys"
fi
[ -f "$H/secrets/acw-worker.env" ] || cp "$REPO/deploy/acw-worker.env.example" "$H/secrets/acw-worker.env"
chmod 600 "$H/secrets/"*.env

# 5. Database schema
( cd "$H/app" && set -a && . "$H/secrets/acw-worker.env" && set +a && PYTHONPATH="$H/app" "$H/venv/bin/python" -m acw.admin init-db >/dev/null )

# 6. systemd user units
U="$HOME/.config/systemd/user"
mkdir -p "$U"
install -m 644 "$REPO"/deploy/systemd/*.service "$REPO"/deploy/systemd/*.timer "$REPO"/deploy/systemd/*.slice "$U/"

# Sized from this machine, so a bigger CPU or more RAM is used without edits:
# all Audio Cleaner services may use ACW_MEMORY_PERCENT of RAM (the rest stays
# with the game servers and the system); one file worker per 3 physical cores.
# Live audio capacity scales by itself (core count + start-up calibration).
MEM_KB=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
PCT="${ACW_MEMORY_PERCENT:-40}"
PHYS=$(lscpu -p=core,socket | grep -v '^#' | sort -u | wc -l)
WORKERS="${ACW_WORKERS:-$(( PHYS / 3 < 2 ? 2 : (PHYS / 3 > 8 ? 8 : PHYS / 3) ))}"
mkdir -p "$U/acw.slice.d" "$U/acw-web.service.d"
printf '[Slice]\nMemoryHigh=%dK\nMemoryMax=%dK\n' $(( MEM_KB * (PCT - 3) / 100 )) $(( MEM_KB * PCT / 100 )) > "$U/acw.slice.d/size.conf"
printf '[Service]\nEnvironment=ACW_WORKERS=%d\n' "$WORKERS" > "$U/acw-web.service.d/workers.conf"
echo "host: $PHYS physical cores, $(( MEM_KB / 1048576 )) GB RAM -> $WORKERS file workers, ${PCT}% RAM cap"

systemctl --user daemon-reload
systemctl --user enable --now acw-backup.timer >/dev/null 2>&1
UNITS="acw-stream.service acw-web.service acw-tunnel.service"
for i in $(seq 1 "$WORKERS"); do UNITS="$UNITS acw-worker@$i.service"; done
# Workers beyond the new count (moved to a smaller machine): stop them.
for i in $(seq $(( WORKERS + 1 )) 16); do systemctl --user disable --now "acw-worker@$i.service" >/dev/null 2>&1 || true; done
# GPU transcription worker: only with a working NVIDIA driver and deploy/setup-gpu.sh done.
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1 && [ -x "$H/venv-gpu/bin/python" ]; then
  UNITS="$UNITS acw-worker-gpu.service"
else
  systemctl --user disable --now acw-worker-gpu.service >/dev/null 2>&1 || true
fi
systemctl --user enable $UNITS >/dev/null 2>&1
if [ "${1:-}" != "--no-restart" ]; then
  systemctl --user restart $UNITS
  sleep 3
  systemctl --user --no-pager --plain list-units 'acw-*' | sed -n '1,14p'
  curl -fsS http://127.0.0.1:8790/audio/api/health && echo " <- local health ok"
fi
