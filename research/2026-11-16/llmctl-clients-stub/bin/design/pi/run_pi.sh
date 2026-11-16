#!/usr/bin/env bash
# Run Pi 1.5.2 through its bundle entry (PI_MANAGED_INSTALL_ROOT unset), fully isolated, against the stub.
# usage: run_pi.sh <scenario> <agent-src-dir> <stub-json> [pi args...]
#   <agent-src-dir> may hold models.json and settings.json; __PORT__ is replaced by the stub port.
set -u
B=$SCRATCH/clients/design/pi
CLI=$HOME/.pi/agent/install/releases/1.5.2/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js
NODE=$HOME/.nvm/versions/node/v24.22.1/bin/node
scenario=$1; src=$2; stub_json=$3; shift 3
run=$B/runs/$scenario; port=$(cat $B/stub/port)
rm -rf "$run"; mkdir -p "$run"/{agent,home,work,tmp,xdg/config,xdg/data,xdg/cache,xdg/state}
for f in models.json settings.json; do [ -f "$src/$f" ] && sed "s/__PORT__/$port/g" "$src/$f" > "$run/agent/$f"; done
python3 -I - "$port" "$scenario" "$stub_json" <<'PY'
import json, sys, urllib.request
port, scenario, extra = sys.argv[1], sys.argv[2], json.loads(sys.argv[3] or "{}")
body = {"scenario": scenario, "reset_seq": True, "mode": "tools", "model_id": "local", "reasoning_field": "reasoning_content"}
body.update(extra)
urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/__control", data=json.dumps(body).encode(), method="POST"), timeout=5).read()
PY
(cd "$run/agent" && sha256sum models.json settings.json 2>/dev/null) > "$run/agent-hashes-before.txt"
printf '%q ' "$@" > "$run/pi-args.txt"; echo >> "$run/pi-args.txt"
cd "$run/work"
start=$(date +%s.%N)
timeout "${TIMEOUT:-120}" env -i HOME="$run/home" TMPDIR="$run/tmp" XDG_CONFIG_HOME="$run/xdg/config" XDG_DATA_HOME="$run/xdg/data" \
  XDG_CACHE_HOME="$run/xdg/cache" XDG_STATE_HOME="$run/xdg/state" PATH="$HOME/.nvm/versions/node/v24.22.1/bin:/usr/bin:/bin" \
  TERM=dumb LANG=C.UTF-8 PI_OFFLINE=1 PI_SKIP_VERSION_CHECK=1 PI_TELEMETRY=0 PI_CODING_AGENT_DIR="$run/agent" \
  "$NODE" "$CLI" "$@" > "$run/stdout.txt" 2> "$run/stderr.txt" < /dev/null
code=$?; end=$(date +%s.%N)
echo "$code" > "$run/exit_code"; python3 -c "print(round($end-$start,2))" > "$run/seconds.txt"
(cd "$run/agent" && sha256sum models.json settings.json 2>/dev/null) > "$run/agent-hashes-after.txt"
(cd "$run" && find agent home work -printf '%y %s %p\n' | sort -k3) > "$run/files-after.txt"
echo "scenario=$scenario exit=$code seconds=$(cat $run/seconds.txt) captures=$(ls $B/captures/$scenario 2>/dev/null | wc -l)"
