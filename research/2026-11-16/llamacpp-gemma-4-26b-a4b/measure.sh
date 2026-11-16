#!/usr/bin/env bash
# usage: measure.sh KEY REPO FILE MEMCAP LONG_TOKENS THINKING_ON_JSON OPENCODE_MODEL -- llama-server args (MODEL_PATH is replaced)
#   THINKING_ON_JSON: JSON fragment merged into one "Reply with exactly: OK" request, or "" to skip
#   OPENCODE_MODEL: entry such as llamacpp-docker/gemma-4-26b-a4b, or "" to skip the OpenCode run
# Runs one llama.cpp measurement with the same procedure for every model and leaves the
# records in scratchpad/bench/runs/KEY. The container is stopped gracefully at the end.
set -uo pipefail
B=$SCRATCH/bench
OC=$SCRATCH/oc-e2e
key=$1; repo=$2; file=$3; memcap=$4; long_tokens=$5; thinking_json=$6; oc_model=$7
shift 7
[ "${1:-}" = "--" ] && shift
F=$B/runs/$key
mkdir -p "$F"

host_path=$(ls -1 "$HOME/.cache/huggingface/hub/models--${repo//\//--}"/snapshots/*/"$file" 2>/dev/null | head -1)
[ -n "$host_path" ] || { echo "model file not in the cache: $repo $file"; exit 1; }
ctr_path=${host_path/#$HOME\/.cache\/huggingface//hf}
args=()
for a in "$@"; do args+=("${a/MODEL_PATH/$ctr_path}"); done

docker rm -f llamacpp >/dev/null 2>&1
start=$(date +%s.%N)
docker run -d --name llamacpp \
  --user "$(id -u):$(id -g)" \
  --device nvidia.com/gpu=all \
  -e GGML_CUDA_DISABLE_GRAPHS=1 \
  -e HF_HOME=/hf \
  -v "$HOME/.cache/huggingface:/hf" \
  -p 127.0.0.1:8080:8080 \
  --memory "$memcap" --memory-swap "$memcap" \
  ghcr.io/ggml-org/llama.cpp:server-cuda-b12407 \
  "${args[@]}" >/dev/null || { echo "docker run failed"; exit 1; }
if ! timeout 600 bash -c 'until curl -sf http://127.0.0.1:8080/health >/dev/null; do sleep 1; [ "$(docker inspect -f "{{.State.Running}}" llamacpp)" = true ] || exit 1; done'; then
  docker logs llamacpp > "$F/server-full.log" 2>&1
  echo "server did not start; last log lines:"
  tail -25 "$F/server-full.log"
  exit 1
fi
echo "load $(python3 -I -c "import time; print(round(time.time() - $start, 1))") s, VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader), container $(docker stats --no-stream --format '{{.MemUsage}}' llamacpp)"
docker inspect llamacpp --format '{{json .Config.Cmd}}' > "$F/server-args.json"
docker logs llamacpp 2>&1 | grep -E 'n_threads|n_ctx_slot|W llama_model_loader|W srv|E ' | head -12

nvidia-smi --query-gpu=timestamp,memory.used,power.draw,clocks.sm,temperature.gpu --format=csv,noheader -l 2 > "$F/gpu-trace.csv" 2>&1 &
trace=$!
(while true; do echo "$(date +%T) $(docker stats --no-stream --format '{{.MemUsage}} {{.CPUPerc}}' llamacpp)"; sleep 4; done) > "$F/docker-stats.log" 2>&1 &
dstat=$!

S=$(date +%s)
heavy python3 -I "$B/llm_probe.py" --base http://127.0.0.1:8080/v1 --model local --label "llamacpp-$key" \
  --runs 3 --long-tokens "$long_tokens" --out "$F/probe-results.json" > "$F/probe-summary.json" 2> "$F/probe.err"
echo "probe exit=$? after $(( $(date +%s) - S )) s"
tail -3 "$F/probe.err"

if [ -n "$thinking_json" ]; then
  python3 -I - "$thinking_json" > "$F/thinking-request.json" <<'EOF'
import json, sys
body = {"model": "local", "max_tokens": 4096, "messages": [{"role": "user", "content": "Reply with exactly: OK"}]}
body.update(json.loads(sys.argv[1]))
print(json.dumps(body))
EOF
  curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d @"$F/thinking-request.json" > "$F/thinking-on.json"
  python3 -I -c "
import json, sys
b = json.load(open(sys.argv[1])); m = b['choices'][0]['message']
print('thinking on: content=%r reasoning_chars=%d completion_tokens=%d finish=%s' % ((m.get('content') or '')[:60], len(m.get('reasoning_content') or ''), b['usage']['completion_tokens'], b['choices'][0]['finish_reason']))
" "$F/thinking-on.json" || head -c 400 "$F/thinking-on.json"
fi

curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"local","max_tokens":3000,"messages":[{"role":"user","content":"Write a Python module that implements an LRU cache class with get and put, type hints and docstrings, followed by five pytest tests."}]}' > "$F/code-decode.json"
python3 -I -c "
import json, sys
b = json.load(open(sys.argv[1])); t = b.get('timings') or {}; m = b['choices'][0]['message']
print('code decode: %d tokens at %.1f tok/s, finish=%s, reasoning_chars=%d' % (t.get('predicted_n', 0), t.get('predicted_per_second', 0), b['choices'][0]['finish_reason'], len(m.get('reasoning_content') or '')))
" "$F/code-decode.json" || head -c 400 "$F/code-decode.json"

if [ -n "$oc_model" ]; then
  cd "$OC" && rm -f hello.py
  S=$(date +%s)
  timeout 900 heavy ~/.opencode/bin/opencode run --standalone --archived --auto --format json -m "$oc_model" \
    --title "e2e llama.cpp $key" \
    "Create a file named hello.py containing a function add(a, b) that returns a + b, and a main block that prints add(2, 3). Then read the file back and tell me how many lines it has. Do not run any shell commands." \
    > "$F/opencode-e2e-events.jsonl" 2> "$F/opencode-e2e.err"
  echo "opencode exit=$? after $(( $(date +%s) - S )) s"
  cat -n hello.py 2>&1 | head -12
  python3 -I - "$F/opencode-e2e-events.jsonl" <<'EOF'
import json
import sys
for line in open(sys.argv[1]):
    try:
        e = json.loads(line)
    except ValueError:
        continue
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
  tail -3 "$F/opencode-e2e.err"
fi

kill "$trace" "$dstat"
echo "GPU peak $(cut -d, -f2 "$F/gpu-trace.csv" | tr -dc '0-9\n' | sort -n | tail -1) MiB, max power $(cut -d, -f3 "$F/gpu-trace.csv" | tr -dc '0-9.\n' | sort -n | tail -1) W, max temp $(cut -d, -f5 "$F/gpu-trace.csv" | tr -dc '0-9\n' | sort -n | tail -1) C"
echo "container peak $(awk '{print $2}' "$F/docker-stats.log" | sort -h | tail -1)"
docker stop -t 60 llamacpp >/dev/null
docker logs llamacpp > "$F/server-full.log" 2>&1
grep -E 'llama_moe_cache' "$F/server-full.log" | tail -4
tr '\r' '\n' < "$F/server-full.log" | grep -E 'print_timing|listening|model loaded|n_ctx_slot|n_threads|llama_moe_cache' > "$F/server-timings.log"
python3 -I - "$F/probe-summary.json" <<'EOF'
import json, sys
try:
    s = json.load(open(sys.argv[1]))
except (OSError, ValueError) as e:
    print("no probe summary:", e)
    raise SystemExit(0)
print("ok:", s["ok"].get("content"), "| reasoning_chars", s["ok"].get("reasoning_chars"), "| finish", s["ok"].get("finish"))
print("tool:", s["tool"].get("finish"), s["tool"].get("calls"), "| content", (s["tool"].get("content") or "")[:80])
print("decode:", [(d.get("decode_tok_s"), (d.get("usage") or {}).get("completion_tokens")) for d in s["decode_runs"]], "median", s["decode_tok_s_median"])
for l in s["long"]:
    f = l.get("server_first") or {}
    r = l.get("server_repeat") or {}
    print("long %s: prompt %s new %s cached %s pp %.0f ttft %.2f tg %.1f | repeat ttft %.3f cache_n %s" % (
        l["target"], l["prompt_tokens"], f.get("prompt_n"), f.get("cache_n"), f.get("prompt_per_second") or 0,
        l["ttft_first_s"], f.get("predicted_per_second") or 0, l["ttft_repeat_s"], r.get("cache_n")))
EOF
