#!/usr/bin/env bash
# usage: web-start.sh <DSH_HOME> <tag> [port]
# Starts dsh web in the background through heavy, logs to $S/<tag>-web.{out,err}; never prints the token.
set -euo pipefail
S=$SCRATCH/laneA
H="$1"; TAG="$2"; PORT="${3:-3181}"
if pgrep -f "^node .*/dsh (web|--profile web)" >/dev/null; then echo "another dsh web runs; refusing" >&2; exit 1; fi
cd "$S/work"
export DSH_HOME="$H" DSH_TELEMETRY_DISABLED=1
nohup heavy dsh web --port "$PORT" --no-open > "$S/$TAG-web.out" 2> "$S/$TAG-web.err" < /dev/null &
echo $! > "$S/$TAG-web.pid"
for i in $(seq 1 60); do
  if grep -q "token=" "$S/$TAG-web.out" 2>/dev/null; then break; fi
  sleep 1
done
sed -E 's/token=[^ ]*/token=<redacted>/' "$S/$TAG-web.out"
echo "pid=$(cat $S/$TAG-web.pid) after ${i}s"
