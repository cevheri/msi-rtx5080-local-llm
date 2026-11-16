#!/usr/bin/env bash
# Final validation of the tuned Qwen3.6-35B-A3B launch (variant v6 plus -lv 4 for the MoE cache statistics).
set -uo pipefail
B=$SCRATCH/bench
OC=$SCRATCH/oc-e2e
F=$B/final
mkdir -p "$F"

docker rm -f llamacpp >/dev/null 2>&1
start=$(date +%s.%N)
docker run -d --name llamacpp \
  --user "$(id -u):$(id -g)" \
  --device nvidia.com/gpu=all \
  -e GGML_CUDA_DISABLE_GRAPHS=1 \
  -e HF_HOME=/hf \
  -v "$HOME/.cache/huggingface:/hf" \
  -p 127.0.0.1:8080:8080 \
  --memory 20g --memory-swap 20g \
  ghcr.io/ggml-org/llama.cpp:server-cuda-b12407 \
  -m /hf/hub/models--ggml-org--Qwen3.6-35B-A3B-GGUF/snapshots/52f9f9b8052fb9c8b69bba4a95eda0f319ac3d2a/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --alias local -ngl all --n-cpu-moe 25 -fa on -c 131072 -np 1 --jinja \
  -lm none -t 8 --moe-cache-mib 3200 --moe-cache-cpu-share 0 -lv 4 \
  --reasoning off \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0 >/dev/null || { echo "docker run failed"; exit 1; }
if ! timeout 400 bash -c 'until curl -sf http://127.0.0.1:8080/health >/dev/null; do sleep 1; [ "$(docker inspect -f "{{.State.Running}}" llamacpp)" = true ] || exit 1; done'; then
  echo "server did not start"; docker logs llamacpp 2>&1 | tail -20; exit 1
fi
echo "load $(python3 -I -c "import time; print(round(time.time() - $start, 1))") s, VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader), container $(docker stats --no-stream --format '{{.MemUsage}}' llamacpp)"
docker inspect llamacpp --format '{{json .Config.Cmd}}' > "$F/server-args.json"

nvidia-smi --query-gpu=timestamp,memory.used,power.draw,clocks.sm,temperature.gpu --format=csv,noheader -l 2 > "$F/gpu-trace.csv" 2>&1 &
trace=$!
(while true; do echo "$(date +%T) $(docker stats --no-stream --format '{{.MemUsage}} {{.CPUPerc}}' llamacpp)"; sleep 4; done) > "$F/docker-stats.log" 2>&1 &
dstat=$!

S=$(date +%s)
heavy python3 -I "$B/llm_probe.py" --base http://127.0.0.1:8080/v1 --model local \
  --label llamacpp-qwen3.6-35b-a3b-q4km-ncmoe25-cache3200 --runs 3 --long-tokens 8000,30000,100000 \
  --out "$F/probe-results.json" > "$F/probe-summary.json" 2> "$F/probe.err"
echo "probe exit=$? after $(( $(date +%s) - S )) s"

# per-request thinking toggle: the server default is off, enable_thinking turns it on
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"local","max_tokens":2048,"messages":[{"role":"user","content":"Reply with exactly: OK"}],"chat_template_kwargs":{"enable_thinking":true}}' > "$F/thinking-on.json"
python3 -I -c "
import json, sys
b = json.load(open(sys.argv[1])); m = b['choices'][0]['message']
print('thinking on: content=%r reasoning_chars=%d completion_tokens=%d' % (m.get('content'), len(m.get('reasoning_content') or ''), b['usage']['completion_tokens']))
" "$F/thinking-on.json"

# code-generation decode (different routing from the prose prompt in the probe)
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"local","max_tokens":1500,"messages":[{"role":"user","content":"Write a Python module that implements an LRU cache class with get and put, type hints and docstrings, followed by five pytest tests."}]}' > "$F/code-decode.json"
python3 -I -c "
import json, sys
b = json.load(open(sys.argv[1])); t = b.get('timings') or {}
print('code decode: %d tokens at %.1f tok/s, finish=%s' % (t.get('predicted_n', 0), t.get('predicted_per_second', 0), b['choices'][0]['finish_reason']))
" "$F/code-decode.json"

# OpenCode end to end, same task as the earlier models
cd "$OC" && rm -f hello.py
S=$(date +%s)
timeout 900 heavy ~/.opencode/bin/opencode run --standalone --archived --auto --format json -m llamacpp-docker/qwen3.6-35b-a3b \
  --title "e2e llama.cpp qwen3.6-35b-a3b" \
  "Create a file named hello.py containing a function add(a, b) that returns a + b, and a main block that prints add(2, 3). Then read the file back and tell me how many lines it has. Do not run any shell commands." \
  > "$F/opencode-e2e-events.jsonl" 2> "$F/opencode-e2e.err"
echo "opencode exit=$? after $(( $(date +%s) - S )) s"
cat -n hello.py 2>&1
python3 -I - "$F/opencode-e2e-events.jsonl" <<'EOF'
import json
import sys
for line in open(sys.argv[1]):
    e = json.loads(line)
    t = e.get("type")
    p = e.get("part", {})
    if t == "tool_use":
        print("tool_use:", p.get("tool"), p.get("state", {}).get("status"))
    elif t == "step_finish":
        print("step tokens:", json.dumps(p.get("tokens")))
    elif t == "text":
        print("answer:", p.get("text", "")[:200].replace("\n", " "))
    elif t == "error":
        print("error:", json.dumps(e)[:300])
EOF

kill "$trace" "$dstat"
echo "GPU peak $(cut -d, -f2 "$F/gpu-trace.csv" | tr -dc '0-9\n' | sort -n | tail -1) MiB, max power $(cut -d, -f3 "$F/gpu-trace.csv" | tr -dc '0-9.\n' | sort -n | tail -1) W, max temp $(cut -d, -f5 "$F/gpu-trace.csv" | tr -dc '0-9\n' | sort -n | tail -1) C"
echo "container peak $(awk '{print $2}' "$F/docker-stats.log" | sort -h | tail -1)"

# graceful stop, so the MoE cache logs its statistics at exit
docker stop -t 60 llamacpp >/dev/null
docker logs llamacpp > "$F/server-full.log" 2>&1
grep -E 'llama_moe_cache' "$F/server-full.log" | tail -12
tr '\r' '\n' < "$F/server-full.log" | grep -E 'print_timing|listening|model loaded|n_ctx_slot|n_threads|llama_moe_cache' > "$F/server-timings.log"
cat "$F/probe-summary.json"
