#!/usr/bin/env bash
# Colour for the deploy scripts' output (sourced by deploy/lib.sh and deploy/fetch-models.sh, not run). The one place that knows the codes.
#   red     a failure that stops the deploy or has to be fixed        say_fail TEXT   (stderr)
#   yellow  a warning, something to look at                           say_warn TEXT   (stderr)
#   green   ok                                                        say_ok TEXT     (stdout)
#   plain   everything else: keep using echo
# Colour only when the stream is a terminal and NO_COLOR is unset or empty (https://no-color.org), so logs and pipes stay
# clean. ACW_COLOR=1|0 overrides (deploy/smoke.py reads the same variable; install.sh hands it to the service account's part).
#
# Every say_fail / say_warn also goes to the summary file $ACW_SUMMARY (when set: install.sh sets it), one line "R|Y<TAB>text",
# and acw_print_summary lists them again at the end of the deploy, so nothing red or yellow scrolls past unnoticed.
if [ -z "${ACW_COLOR:-}" ]; then
  if [ -t 1 ] && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-dumb}" != dumb ]; then ACW_COLOR=1; else ACW_COLOR=0; fi
fi
export ACW_COLOR

# acw_paint FD CODE TEXT: the text, in colour when colour is on and file descriptor FD is a terminal
acw_paint() {
  if [ "$ACW_COLOR" = 1 ] && [ -t "$1" ]; then printf '\033[%sm%s\033[0m' "$2" "$3"; else printf '%s' "$3"; fi
}
acw_note() {  # acw_note R|Y TEXT: remember a red or yellow line for the summary (first line only)
  [ -n "${ACW_SUMMARY:-}" ] && printf '%s\t%s\n' "$1" "${2%%$'\n'*}" >> "$ACW_SUMMARY" 2>/dev/null
  return 0
}
# (no $(...) around acw_paint: inside one, stdout is a pipe and the terminal test would always say no)
say_ok() { acw_paint 1 32 "$*"; echo; }
say_warn() { { acw_paint 2 33 "$*"; echo; } >&2; acw_note Y "$*"; }
say_fail() { { acw_paint 2 '1;31' "$*"; echo; } >&2; acw_note R "$*"; }

# acw_print_summary FILE EXIT-STATUS: the red and yellow lines of the whole deploy, again, at the end
acw_print_summary() {
  local file="$1" rc="${2:-0}" reds yellows
  reds="$(sed -n 's/^R\t//p' "$file" 2>/dev/null || true)"
  yellows="$(sed -n 's/^Y\t//p' "$file" 2>/dev/null || true)"
  if [ -z "$reds$yellows" ]; then
    [ "$rc" = 0 ] && say_ok "summary: nothing needs attention"
    return 0
  fi
  echo
  echo "summary: $(printf '%s' "$reds" | grep -c .) to fix (red), $(printf '%s' "$yellows" | grep -c .) to look at (yellow)"
  [ -z "$reds" ] || while IFS= read -r line; do printf '  '; acw_paint 1 '1;31' 'RED   '; printf ' %s\n' "$line"; done <<< "$reds"
  [ -z "$yellows" ] || while IFS= read -r line; do printf '  '; acw_paint 1 33 'YELLOW'; printf ' %s\n' "$line"; done <<< "$yellows"
}
