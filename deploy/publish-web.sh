#!/usr/bin/env bash
# Build the website and publish it to the VPS (/var/www/audio, served by Caddy).
# Requires the one-time VPS setup (deploy/vps/install-audio-route.sh). deploy/install.sh runs this
# by itself when web/ changed, so you rarely need to call it directly.
#
# Visitors never see a half-published site: the new hashed files go in first (the old page does not point
# at them), the entry page is swapped last (index.html and its .gz together), and the previous release's
# hashed files stay for one more release, so a tab that is still open on the old page keeps loading its scripts.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO/web"
npm ci --no-audit --no-fund
npm run sync-dfn
VITE_BASE=/audio/ npm run build
# Pre-compress for Caddy's `precompressed` (the 14 MB wasm shrinks to ~4 MB).
find dist -type f \( -name '*.js' -o -name '*.css' -o -name '*.html' -o -name '*.wasm' -o -name '*.svg' -o -name '*.json' -o -name '*.mjs' \) \
  -exec gzip -kf9 {} \;

DEST=vps:/var/www/audio/
# 1. Everything except the entry page; stale files go, but not the hashed files of the previous release.
rsync -az --delay-updates --delete --filter='P /assets/***' --filter='P /.assets-current' \
  --chmod=D755,F644 --exclude '/index.html' --exclude '/index.html.gz' dist/ "$DEST"
# 2. The entry page, last.
rsync -az --delay-updates --chmod=F644 dist/index.html dist/index.html.gz "$DEST"
# 3. Hashed files older than the previous release go; remember this release's files for the next time.
NEW_LIST="$(cd dist && find assets -type f | sed 's|^assets/||' | sort)"
printf '%s\n' "$NEW_LIST" | ssh vps 'set -eu
  cd /var/www/audio
  cat > /tmp/acw-assets-new
  # first release published this way: what is live now counts as the previous release
  [ -f .assets-current ] || { [ -d assets ] && ( cd assets && find . -type f | sed "s|^\./||" | sort ) > .assets-current; } || : > .assets-current
  sort -u /tmp/acw-assets-new .assets-current > /tmp/acw-assets-keep
  if [ -d assets ]; then
    ( cd assets && find . -type f | sed "s|^\./||" | sort ) | comm -23 - /tmp/acw-assets-keep | while read -r f; do rm -f "assets/$f"; done
  fi
  cp /tmp/acw-assets-new .assets-current
  rm -f /tmp/acw-assets-new /tmp/acw-assets-keep'
ssh vps 'curl -s -o /dev/null -w "%{http_code} " https://mc-multi-world.space/audio/ && curl -s https://mc-multi-world.space/audio/api/health'
echo
git -C "$REPO" rev-parse HEAD:web > "$HOME/audio-cleaner/data/web-published" 2>/dev/null || true
