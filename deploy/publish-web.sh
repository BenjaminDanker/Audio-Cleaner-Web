#!/usr/bin/env bash
# Build the website and publish it to the VPS (/var/www/audio, served by Caddy).
# Requires the one-time VPS setup (deploy/vps/install-audio-route.sh).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO/web"
npm ci --no-audit --no-fund
npm run sync-dfn
VITE_BASE=/audio/ npm run build
# Pre-compress for Caddy's `precompressed` (the 14 MB wasm shrinks to ~4 MB).
find dist -type f \( -name '*.js' -o -name '*.css' -o -name '*.html' -o -name '*.wasm' -o -name '*.svg' -o -name '*.json' -o -name '*.mjs' \) \
  -exec gzip -kf9 {} \;
rsync -az --delete --chmod=D755,F644 dist/ vps:/var/www/audio/
ssh vps 'curl -s -o /dev/null -w "%{http_code} " https://mc-multi-world.space/audio/ && curl -s https://mc-multi-world.space/audio/api/health'
echo
