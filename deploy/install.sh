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
[ -d "$H/models/whisper/small" ] || echo "warning: whisper small missing (local live captions will fail)" >&2

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
install -m 644 "$REPO"/deploy/systemd/*.service "$REPO"/deploy/systemd/*.timer "$U/"
systemctl --user daemon-reload
systemctl --user enable --now acw-backup.timer >/dev/null 2>&1
WORKERS="${ACW_WORKERS:-2}"
UNITS="acw-stream.service acw-web.service acw-tunnel.service"
for i in $(seq 1 "$WORKERS"); do UNITS="$UNITS acw-worker@$i.service"; done
systemctl --user enable $UNITS >/dev/null 2>&1
if [ "${1:-}" != "--no-restart" ]; then
  systemctl --user restart $UNITS
  sleep 3
  systemctl --user --no-pager --plain list-units 'acw-*' | sed -n '1,12p'
  curl -fsS http://127.0.0.1:8790/audio/api/health && echo " <- local health ok"
fi
