#!/usr/bin/env bash
# Check 3 (plan ruling 35): stop SGLang, start vLLM DeepSeek-R1 with no client, send one chat request, stop it.
set -u
here=$(cd "$(dirname "$0")" && pwd)
out=$1
mkdir -p "$out"
"$here/stamp_run.sh" "$out/down-1.log" ./llmctl down
"$here/stamp_run.sh" "$out/up.log" env LLMCTL_CLIENTS= ./llmctl up deepseek-r1 --runtime vllm --replace
if grep -q ' ready: ' "$out/up.log"; then
  python3 -I "$here/chat_requests.py" http://127.0.0.1:8000/v1/chat/completions deepseek-r1-8b 1 4096 \
    "Reply with exactly: OK" > "$out/request.jsonl" 2> "$out/request.err"
fi
docker logs --timestamps vllm > "$out/server.log" 2>&1
"$here/stamp_run.sh" "$out/down-2.log" ./llmctl down
echo "check3 done"
