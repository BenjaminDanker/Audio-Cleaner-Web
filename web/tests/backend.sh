#!/bin/bash
# A throwaway backend for the browser suites, the way production lays it out (site at the root of its own domain):
#   main app (acw.app) 18792, live app (acw.live_app) 18793, the native live engine, the job worker, and tests/front.mjs on 8792
#   (the stand-in for Caddy: /api/stream/ws to the live app, the rest to the main app). The model mirror (tests/mirror.mjs) is
#   started by each suite itself. Everything lives in a fresh data dir under $ACW_TEST_STATE, base path empty, live trial 0 s.
#
#   tests/backend.sh start    wipe the data dir, start everything, wait until the three ports answer
#   tests/backend.sh stop     stop what start started (its own pids only); the logs stay in $ACW_TEST_STATE
#   tests/backend.sh env      print the exports a suite needs (ACW_DATA_DIR for tests/helpers.mjs admin(), ...); eval "$(backend.sh env)"
#
# Environment (all optional):
#   DIST                  built site to serve (default web/dist; tests/run.sh builds its own)
#   ACW_TEST_STATE        state dir (default /tmp/acw-webtest-<uid>): data/, *.pid, *.log
#   AC_HOME               the installation with bin/ models/ venv/ (default ~/audio-cleaner; read-only, nothing is written there)
#   ACW_BIN_DIR, ACW_MODELS_DIR, ACW_PYTHON   override the tools, models and interpreter inside AC_HOME
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WEB=$(dirname "$HERE")
REPO=$(dirname "$WEB")
AC=${AC_HOME:-$HOME/audio-cleaner}
STATE=${ACW_TEST_STATE:-/tmp/acw-webtest-$(id -u)}
PY=${ACW_PYTHON:-$AC/venv/bin/python}
UVICORN=${ACW_UVICORN:-$(dirname "$PY")/uvicorn}

setenv() {
  export ACW_DATA_DIR=$STATE/data ACW_BASE_PATH= ACW_WEB_DIST=${DIST:-$WEB/dist} ACW_PUBLIC_ORIGIN=http://127.0.0.1:8792
  export ACW_LIVE_TRIAL_SECONDS=0 ACW_MODELS_DIR=${ACW_MODELS_DIR:-$AC/models} ACW_BIN_DIR=${ACW_BIN_DIR:-$AC/bin}
  export ACW_SERVER_DIR=$REPO/server ACW_PYTHON=$PY PYTHONPATH=$REPO/server:$REPO/common E2E_ORIGIN=http://127.0.0.1:8792
}

stop() {
  for f in sw wk uv lv fr; do
    [ -f "$STATE/$f.pid" ] && kill "$(cat "$STATE/$f.pid")" 2>/dev/null
    rm -f "$STATE/$f.pid"
  done
  sleep 1
}

up() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

case "${1:-}" in
  env)
    setenv
    for v in ACW_DATA_DIR ACW_BASE_PATH ACW_WEB_DIST ACW_PUBLIC_ORIGIN ACW_LIVE_TRIAL_SECONDS ACW_MODELS_DIR ACW_BIN_DIR ACW_SERVER_DIR ACW_PYTHON PYTHONPATH E2E_ORIGIN; do
      printf 'export %s=%q\n' "$v" "${!v}"
    done;;
  stop) stop;;
  start)
    setenv
    stop
    rm -rf "$STATE/data"; mkdir -p "$STATE/data"
    ENGINE=$REPO/native/live/target/release/acw-live; [ -x "$ENGINE" ] || ENGINE=$AC/bin/acw-live
    cd "$REPO/server" || exit 1
    nohup "$ENGINE" > "$STATE/engine.log" 2>&1 & echo $! > "$STATE/sw.pid"
    nohup "$PY" -m acw.worker.main > "$STATE/worker.log" 2>&1 & echo $! > "$STATE/wk.pid"
    nohup "$UVICORN" acw.app:app --host 127.0.0.1 --port 18792 > "$STATE/app.log" 2>&1 & echo $! > "$STATE/uv.pid"
    nohup "$UVICORN" acw.live_app:app --host 127.0.0.1 --port 18793 > "$STATE/liveapp.log" 2>&1 & echo $! > "$STATE/lv.pid"
    nohup node "$HERE/front.mjs" 8792 18792 18793 > "$STATE/front.log" 2>&1 & echo $! > "$STATE/fr.pid"
    for i in $(seq 1 120); do
      if up 18792 && up 18793 && up 8792; then exit 0; fi
      sleep 0.5
    done
    echo "backend did not come up in 60 s; logs in $STATE" >&2; stop; exit 1;;
  *) sed -n '2,/^set -u/p' "$0" | sed '$d' | cut -c3-; exit 2;;
esac
