#!/usr/bin/env bash
# One Kimi TUI session for the LR4 checklist, adapted from the archive's exp/bin/case.sh.
# Kimi runs inside bwrap with no network; inside, socats on 127.0.0.1:8080 and :8000 forward to unix sockets;
# outside, socats forward those sockets to two recording proxies (proxy.py), which forward to the real servers
# on 127.0.0.1:8080 (llama.cpp) and 127.0.0.1:8000 (vLLM) and save each request's shape.
# usage: case.sh <run dir> <private home W> <steps.json>   env: TO (s, default 900)
set -u
BIN=$(cd "$(dirname "$0")" && pwd)
KIMI=$HOME/.kimi-code/bin/kimi
R=$1; W=$2; STEPS=$3
N=$W/net; rm -rf "$N" "$R"; mkdir -p -m 700 "$N"; mkdir -p "$R/shapes"
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; wait 2>/dev/null; }
trap cleanup EXIT
for pair in 8080:18480 8000:18400; do
  up=${pair%%:*}; pp=${pair#*:}
  if (exec 3<>/dev/tcp/127.0.0.1/$pp) 2>/dev/null; then echo "port $pp is busy: refusing" >&2; exit 99; fi
  PROXY_TAG=p$up python3 -I "$BIN/proxy.py" "$pp" 127.0.0.1 "$up" "$R/shapes" > "$R/proxy-$up.log" 2>&1 & PIDS+=($!)
  # relative socket path: the absolute one can exceed the 108-byte sun_path limit
  (cd "$N" && exec socat UNIX-LISTEN:s$up.sock,fork,unlink-early TCP:127.0.0.1:"$pp") & PIDS+=($!)
done
for i in $(seq 50); do [ -S "$N/s8080.sock" ] && [ -S "$N/s8000.sock" ] && break; sleep 0.1; done
date -u +%Y-%m-%dT%H:%M:%SZ > "$R/start-utc"
bwrap \
  --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --tmpfs "$HOME" \
  --ro-bind "$KIMI" "$KIMI" --ro-bind "$BIN" "$BIN" --ro-bind "$STEPS" "$STEPS" \
  --bind "$W/kimi" "$W/kimi" --bind "$W/home" "$W/home" --bind "$W/work" "$W/work" \
  --bind "$W/tmp" "$W/tmp" --bind "$W/cache" "$W/cache" --bind "$R" "$R" --bind "$N" /tmp/kn \
  --unshare-net --unshare-ipc --unshare-uts --new-session --die-with-parent \
  --chdir "$W/work" --clearenv \
  --setenv HOME "$W/home" --setenv KIMI_CODE_HOME "$W/kimi" --setenv KIMI_CODE_CACHE_DIR "$W/cache" \
  --setenv XDG_CACHE_HOME "$W/home/.cache" --setenv XDG_CONFIG_HOME "$W/home/.config" \
  --setenv XDG_DATA_HOME "$W/home/.local/share" --setenv XDG_STATE_HOME "$W/home/.local/state" \
  --setenv TMPDIR "$W/tmp" --setenv PATH /usr/bin:/bin --setenv TERM xterm-256color --setenv LANG C.UTF-8 \
  --setenv KIMI_DISABLE_TELEMETRY 1 --setenv KIMI_CODE_NO_AUTO_UPDATE 1 --setenv KIMI_CODE_REGION_MARKER off \
  --setenv KIMI_CODE_TRUST_WORKSPACE 1 --setenv SYNC /tmp/kn \
  bash "$BIN/inner.sh" "$R" "${TO:-900}" python3 -I "$BIN/tui.py" "$R" "$STEPS" -- "$KIMI" \
  > "$R/stdout" 2> "$R/stderr"
RC=$?
echo $RC > "$R/exit"
date -u +%Y-%m-%dT%H:%M:%SZ > "$R/end-utc"
echo "exit $RC"
