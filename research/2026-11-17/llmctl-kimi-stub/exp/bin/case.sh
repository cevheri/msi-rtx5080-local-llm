#!/usr/bin/env bash
# One Kimi experiment case: optional recording stub on host 127.0.0.1:$SPORT, a host socat from a unix socket to it,
# and kimi inside bwrap with no network (only a socat inside the sandbox on 127.0.0.1:$BPORT back to the socket).
# usage: case.sh <case> <cmd...>    (cmd "kimi" means the installed binary)
# env: HOMEDIR (default homes/<case>), STUBCFG (empty: no stub), BPORT (0: no bridge), TO (s, default 300),
#      STRACE (1: strace connect/sendto), EXTRA_RO (a path bound read-only), XENV ("NAME=VALUE ..." extra env)
set -u
EXP=$(cd "$(dirname "$0")/.." && pwd)
KIMI=${KIMI_BIN:-$HOME/.kimi-code/bin/kimi}
SPORT=${SPORT:-18491}
C=$1; shift
R=$EXP/runs/$C; W=${WORKDIR:-$EXP/work/$C}; H=${HOMEDIR:-$EXP/homes/$C}; N=$EXP/net/$C
rm -rf "$R" "$N"; [ -z "${WORKDIR:-}" ] && rm -rf "$W"; mkdir -p "$R" "$W" "$H/.kimi-code" "$H/.cache" "$N"
args=(); for a in "$@"; do [ "$a" = kimi ] && args+=("$KIMI") || args+=("$a"); done
printf '%q ' "$@" > "$R/argv"; echo >> "$R/argv"
SP=; SO=
cleanup() { [ -n "$SP" ] && kill $SP 2>/dev/null; [ -n "$SO" ] && kill $SO 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT
if [ -n "${STUBCFG:-}" ]; then
  if (exec 3<>/dev/tcp/127.0.0.1/$SPORT) 2>/dev/null; then echo "port $SPORT is busy: refusing" >&2; echo 99 > "$R/exit"; exit 99; fi
  python3 -I "$EXP/bin/stub.py" "$SPORT" "$STUBCFG" > "$R/stub.log" 2>&1 & SP=$!
  for i in $(seq 50); do grep -q "^stub on" "$R/stub.log" 2>/dev/null && break; sleep 0.1; done
  if ! kill -0 $SP 2>/dev/null || ! grep -q "^stub on" "$R/stub.log"; then echo "stub did not start" >&2; echo 98 > "$R/exit"; exit 98; fi
  # relative path: the absolute one is longer than the 108-byte sun_path limit
  (cd "$N" && exec socat UNIX-LISTEN:s.sock,fork,unlink-early TCP:127.0.0.1:"$SPORT") & SO=$!
  for i in $(seq 50); do [ -S "$N/s.sock" ] && break; sleep 0.1; done
fi
envs=(); for kv in ${XENV:-}; do envs+=(--setenv "${kv%%=*}" "${kv#*=}"); done
ro=(); [ -n "${EXTRA_RO:-}" ] && ro=(--ro-bind "$EXTRA_RO" "$EXTRA_RO")
bwrap \
  --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --tmpfs "$HOME" \
  --ro-bind "$KIMI" "$KIMI" --ro-bind "$EXP/bin" "$EXP/bin" "${ro[@]}" \
  --bind "$H" "$H" --bind "$W" "$W" --bind "$R" "$R" --bind "$N" /tmp/kn \
  --unshare-net --unshare-ipc --unshare-uts --new-session --die-with-parent \
  --chdir "$W" --clearenv \
  --setenv HOME "$H" --setenv KIMI_CODE_HOME "$H/.kimi-code" --setenv KIMI_CODE_CACHE_DIR "$H/.cache/kimi-code" \
  --setenv XDG_CACHE_HOME "$H/.cache" --setenv XDG_CONFIG_HOME "$H/.config" \
  --setenv XDG_DATA_HOME "$H/.local/share" --setenv XDG_STATE_HOME "$H/.local/state" \
  --setenv TMPDIR /tmp --setenv PATH /usr/bin:/bin --setenv TERM dumb --setenv NO_COLOR 1 --setenv LANG C.UTF-8 \
  --setenv KIMI_BIN "$KIMI" --setenv KIMI_DISABLE_TELEMETRY 1 --setenv KIMI_CODE_NO_AUTO_UPDATE 1 --setenv KIMI_CODE_REGION_MARKER off \
  "${envs[@]}" \
  bash "$EXP/bin/inner.sh" "${BPORT:-0}" "${STRACE:-1}" "$R" "${TO:-300}" "${args[@]}" \
  > "$R/stdout" 2> "$R/stderr"
RC=$?
echo $RC > "$R/exit"
cleanup; SP=; SO=
echo "$C exit $RC $(awk '{printf "%.2f s", $2-$1}' "$R/time" 2>/dev/null)"
