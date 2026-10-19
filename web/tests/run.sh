#!/bin/bash
# Runs the browser suites, each on a fresh throwaway backend (tests/backend.sh), against one build of the site.
#   tests/run.sh                 every suite except long.mjs (stretch.mjs cleans 40 minutes of audio: a few minutes)
#   tests/run.sh free ui         only these (names of tests/<name>.mjs)
# The site is built once, with the model downloads pointed at tests/mirror.mjs (http://127.0.0.1:8798/), into a temp dist.
# Exit status: 0 only when every suite exited 0 and printed no "FAIL" line. Logs: $ACW_TEST_STATE/<suite>.log
# (default /tmp/acw-webtest-<uid>). Needs the fixtures: npm run test:fixtures (see README "Tests").
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WEB=$(dirname "$HERE")
STATE=${ACW_TEST_STATE:-/tmp/acw-webtest-$(id -u)}
export ACW_TEST_STATE=$STATE
SUITES=("$@")
[ ${#SUITES[@]} -gt 0 ] || SUITES=(blend-unit contact convert copy e2e files fixes flac free jobs live offline redo stretch reco-model reco-unit reco-ui subtitles ui)

mkdir -p "$STATE"
exec 9>"$STATE.lock"
flock -w 3000 9 || { echo "another test run holds $STATE.lock" >&2; exit 3; }

DIST=$(mktemp -d /tmp/acw-webtest-dist.XXXXXX)
trap '"$HERE/backend.sh" stop; rm -rf "$DIST"' EXIT
export DIST

cd "$WEB" || exit 1
echo "== build (model host http://127.0.0.1:8798/) -> $DIST"
ACW_MODEL_HOST=http://127.0.0.1:8798/ npm run build --silent -- --outDir "$DIST" --emptyOutDir > "$STATE/build.log" 2>&1 \
  || { tail -20 "$STATE/build.log"; echo "build failed"; exit 1; }

eval "$("$HERE/backend.sh" env)"
failed=()
for t in "${SUITES[@]}"; do
  [ -f "tests/$t.mjs" ] || { echo "== $t: no such suite"; failed+=("$t"); continue; }
  "$HERE/backend.sh" start || { echo "== $t: backend did not start"; failed+=("$t"); continue; }
  s=$(date +%s)
  timeout 2400 node "tests/$t.mjs" > "$STATE/$t.log" 2>&1; rc=$?
  "$HERE/backend.sh" stop
  fails=$(grep -c -E "^FAIL" "$STATE/$t.log")
  echo "=== $t exit $rc, $(grep -c -E '^PASS' "$STATE/$t.log") PASS, $fails FAIL, $(( $(date +%s) - s )) s"
  grep -E "^FAIL" "$STATE/$t.log" | cut -c1-260
  { [ "$rc" -ne 0 ] || [ "$fails" -ne 0 ]; } && failed+=("$t")
done

if [ ${#failed[@]} -gt 0 ]; then echo "FAILED: ${failed[*]}"; exit 1; fi
echo "all ${#SUITES[@]} suites passed"
