#!/usr/bin/env bash
# Check 1 (plan ruling 22): sample the container's socket tables every 0.5 s from before the start until the server
# has answered three chat requests after llmctl's ready line.
set -u
here=$(cd "$(dirname "$0")" && pwd)
out=$1
mkdir -p "$out"
stop="$out/stop"
rm -f "$stop"
python3 -I "$here/netsample.py" "$out" "$stop" &
sampler=$!
date -u +%FT%T.%3NZ > "$out/sampler-started.txt"
"$here/stamp_run.sh" "$out/up.log" env LLMCTL_CLIENTS= ./llmctl up qwen3.5-9b --runtime sglang --replace &
up=$!
while kill -0 "$up" 2>/dev/null && ! grep -q ' ready: ' "$out/up.log" 2>/dev/null; do
  sleep 0.2
done
if grep -q ' ready: ' "$out/up.log"; then
  python3 -I "$here/chat_requests.py" http://127.0.0.1:30000/v1/chat/completions qwen3.5-9b 3 64 \
    "Say hello in one sentence." > "$out/requests.jsonl" 2> "$out/requests.err"
  date -u +%FT%T.%3NZ > "$out/requests-done.txt"
  sleep 1
fi
touch "$stop"
wait "$sampler"
wait "$up"
date -u +%FT%T.%3NZ > "$out/sampler-stopped.txt"
echo "check1 done"
