#!/usr/bin/env bash
# Build the website and publish it to the VPS (/var/www/audio, served by Caddy).
# Requires the one-time VPS setup (deploy/vps/setup.sh, docs/vps.md). deploy/install.sh runs this
# by itself when web/ changed, so you rarely need to call it directly.
#
# The Terms and the Privacy Policy that go live are archived just before (deploy/legal-archive.sh): data/legal/<doc>-<version>.html in
# acw's data folder, never overwritten. A text changed under an unchanged version date stops this script before anything is published.
#
# Visitors never see a half-published site: the new hashed files go in first (the old page does not point
# at them), the entry page is swapped last (index.html and its .gz and .zst together), and the previous release's
# hashed files stay for one more release, so a tab that is still open on the old page keeps loading its scripts.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=deploy/lib.sh
. "$REPO/deploy/lib.sh"
# This needs the owner's VPS login key (ssh vps): the service account (acw) never has one.
if [ -n "$ACW_SERVICE_USER" ] && [ "$(id -un)" = "$ACW_SERVICE_USER" ]; then
  echo "publish-web.sh needs the owner's VPS login key; $ACW_SERVICE_USER has none by design: run it as the owner" >&2
  exit 1
fi
cd "$REPO/web"
npm ci --no-audit --no-fund
bash "$REPO/deploy/legal-archive.sh" --check   # before the build: a changed text under an unchanged version date stops here
npm run sync-dfn
VITE_BASE=/ npm run build
bash "$REPO/deploy/legal-archive.sh"           # the build passed: save the versions that are about to go live (nothing for one already saved)
# Pre-compress for Caddy's `precompressed zstd gzip`: zstd for browsers that take it (about a third smaller than gzip on
# the WebAssembly files; level 19 keeps the window at 8 MB, the most browsers accept), gzip for the rest. Media is already
# compressed: left alone.
find dist -type f \( -name '*.js' -o -name '*.css' -o -name '*.html' -o -name '*.wasm' -o -name '*.svg' -o -name '*.json' -o -name '*.mjs' \) \
  -exec gzip -kf9 {} \; -exec zstd -q -f -19 {} -o {}.zst \;

DEST=vps:/var/www/audio/
# 1. Everything except the entry page; stale files go, but not the hashed files of the previous release.
rsync -az --delay-updates --delete --filter='P /assets/***' --filter='P /.assets-current' \
  --chmod=D755,F644 --exclude '/index.html' --exclude '/index.html.gz' --exclude '/index.html.zst' dist/ "$DEST"
# 2. The entry page, last.
rsync -az --delay-updates --chmod=F644 dist/index.html dist/index.html.gz dist/index.html.zst "$DEST"
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
ssh vps 'curl -s -o /dev/null -w "%{http_code} " https://audiocleaner.live/ && curl -s https://audiocleaner.live/api/health'
echo
# Record which web/ tree is published (install.sh publishes only when it differs); the file is in acw's data folder.
git -C "$REPO" rev-parse HEAD:web | put_as_acw /dev/stdin "$H/data/web-published" 600 2>/dev/null || true
