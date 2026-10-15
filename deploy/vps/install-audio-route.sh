#!/usr/bin/env bash
# Run ON THE VPS with sudo:  sudo bash ~/acw/install-audio-route.sh
# Adds the /audio route to the existing mc-multi-world.space site (before the
# catch-all 404), creates /var/www/audio owned by silver, validates and
# reloads Caddy. Keeps a timestamped backup; reverts if validation fails.
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run with sudo" >&2; exit 1; }
SNIPPET=/home/silver/acw/audio.caddy
CF=/etc/caddy/Caddyfile
[ -r "$SNIPPET" ] || { echo "missing $SNIPPET" >&2; exit 1; }
install -d -o silver -g silver -m 0755 /var/www/audio
# The backup is the file exactly as it was, before the old route is taken out: a failed validation puts it back.
BACKUP="$CF.backup.$(date -u +%Y%m%d%H%M%S)"
cp -a "$CF" "$BACKUP"
trap 'echo "failed, restoring $BACKUP" >&2; cp -a "$BACKUP" "$CF"' ERR
if grep -q 'Audio Cleaner (/audio)' "$CF"; then
  echo "route already present; replacing it"
  python3 - "$CF" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read()
s = re.sub(r"\n    # ---- Audio Cleaner \(/audio\).*?# -{20,}\n", "\n", s, flags=re.S)
open(p, "w").write(s)
PY
fi
python3 - "$CF" "$SNIPPET" <<'PY'
import sys
cf, snip = sys.argv[1], sys.argv[2]
s, block = open(cf).read(), open(snip).read()
marker = "    handle {\n        respond \"not found\" 404"
if marker not in s:
    sys.exit("could not find the catch-all handle block; edit manually")
s = s.replace(marker, block + "\n" + marker, 1)
open(cf, "w").write(s)
PY
# Drop clients that open a connection and never finish their headers (global option). Added only when the file has
# no global options block yet; with one, add `servers { timeouts { read_header 10s } }` to it by hand.
if ! grep -qE '^[[:space:]]*read_header' "$CF"; then
  if grep -qE '^\{' "$CF"; then
    echo "note: add  servers { timeouts { read_header 10s } }  to the global options block at the top of $CF" >&2
  else
    printf '{\n    servers {\n        timeouts {\n            read_header 10s\n        }\n    }\n}\n\n' | cat - "$CF" > "$CF.new"
    cat "$CF.new" > "$CF" && rm -f "$CF.new"
  fi
fi
if ! caddy validate --config "$CF" --adapter caddyfile; then
  trap - ERR
  echo "validation failed, restoring $BACKUP" >&2
  cp -a "$BACKUP" "$CF"
  exit 1
fi
trap - ERR
systemctl reload caddy
echo "ok: /audio route live (backup at $BACKUP)"
