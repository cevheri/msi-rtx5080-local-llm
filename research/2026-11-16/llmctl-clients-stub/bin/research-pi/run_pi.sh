#!/usr/bin/env bash
# Run the Pi CLI fully isolated against the recording stub.
# Usage: run_pi.sh <scenario> <config-dir> <stub-json> [pi args...]
#   <config-dir> may hold models.json and settings.json; they are copied into a fresh agent dir.
#   <stub-json>  is posted to the stub's /__control (scenario is set automatically), e.g.
#                '{"mode":"tools","model_id":"local","reasoning_field":"reasoning_content"}'
# Environment knobs: KEEP_AGENT=1 reuses the scenario's agent dir; NO_AGENT_ENV=1 leaves PI_CODING_AGENT_DIR
# unset (HOME is still the scenario's temp home); TIMEOUT (seconds, default 120).
set -u
R=$SCRATCH/clients/research-pi
scenario=$1; config_dir=$2; stub_json=$3; shift 3
run=$R/runs/$scenario
port=$(cat $R/stub/port)
if [ "${KEEP_AGENT:-0}" != 1 ]; then rm -rf "$run"; fi
mkdir -p "$run"/{agent,home,work,tmp,xdg/config,xdg/data,xdg/cache,xdg/state}
if [ "${KEEP_AGENT:-0}" != 1 ]; then
  for f in models.json settings.json; do
    if [ -f "$config_dir/$f" ]; then
      sed "s/__PORT__/$port/g" "$config_dir/$f" > "$run/agent/$f"
    fi
  done
fi
# Stub control: scenario name = this run, reset its request counter.
python3 -I - "$port" "$scenario" "$stub_json" <<'EOF'
import json, sys, urllib.request
port, scenario, extra = sys.argv[1], sys.argv[2], json.loads(sys.argv[3] or "{}")
body = {"scenario": scenario, "reset_seq": True, "mode": "tools", "model_id": "local",
        "reasoning_field": "reasoning_content"}
body.update(extra)
req = urllib.request.Request(f"http://127.0.0.1:{port}/__control", data=json.dumps(body).encode(), method="POST")
urllib.request.urlopen(req, timeout=5).read()
EOF
rm -rf "$R/captures/$scenario"
(cd "$run/agent" && sha256sum models.json settings.json 2>/dev/null) > "$run/agent-hashes-before.txt"
env_args=(
  HOME="$run/home"
  TMPDIR="$run/tmp"
  XDG_CONFIG_HOME="$run/xdg/config" XDG_DATA_HOME="$run/xdg/data"
  XDG_CACHE_HOME="$run/xdg/cache" XDG_STATE_HOME="$run/xdg/state"
  PATH="$HOME/.nvm/versions/node/v24.22.1/bin:/usr/local/bin:/usr/bin:/bin"
  TERM=dumb LANG=C.UTF-8
  PI_OFFLINE=1 PI_SKIP_VERSION_CHECK=1 PI_TELEMETRY=0
)
if [ "${NO_AGENT_ENV:-0}" != 1 ]; then env_args+=(PI_CODING_AGENT_DIR="$run/agent"); fi
printf '%q ' "$@" > "$run/pi-args.txt"; echo >> "$run/pi-args.txt"
cd "$run/work"
timeout "${TIMEOUT:-120}" env -i "${env_args[@]}" $HOME/.pi/agent/bin/pi "$@" > "$run/stdout.txt" 2> "$run/stderr.txt" < /dev/null
code=$?
echo "$code" > "$run/exit_code"
(cd "$run/agent" && sha256sum models.json settings.json 2>/dev/null) > "$run/agent-hashes-after.txt"
(cd "$run" && find agent home work -printf '%y %s %p\n' | sort -k3) > "$run/files-after.txt"
echo "scenario=$scenario exit=$code captures=$(ls $R/captures/$scenario 2>/dev/null | wc -l)"
