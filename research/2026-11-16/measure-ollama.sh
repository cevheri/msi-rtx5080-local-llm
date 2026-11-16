#!/usr/bin/env bash
# usage: measure-ollama.sh KEY TAG CONTEXT KV_TYPE LONG_TOKENS PROBE_EXTRA_JSON OPENCODE_MODEL
#   KV_TYPE: f16 or q8_0 (OLLAMA_KV_CACHE_TYPE); flash attention is always on
#   PROBE_EXTRA_JSON: merged into every probe request, for example {"reasoning_effort": "none"}, or {}
#   OPENCODE_MODEL: for example ollama-local/qwen3.5-9b#none, or "" to skip
# Starts ollama/ollama:0.40.16 with the GPU and the ollama-models volume, loads TAG, runs the probe
# and the OpenCode task, and leaves the records in scratchpad/bench/runs/ollama-KEY.
set -uo pipefail
B=$SCRATCH/bench
OC=$SCRATCH/oc-e2e
key=$1; tag=$2; ctx=$3; kv=$4; long_tokens=$5; extra=$6; oc_model=$7
F=$B/runs/ollama-$key
mkdir -p "$F"

docker rm -f ollama >/dev/null 2>&1
docker run -d --name ollama \
  --device nvidia.com/gpu=all \
  -v ollama-models:/root/.ollama \
  -p 127.0.0.1:11434:11434 \
  -e OLLAMA_CONTEXT_LENGTH="$ctx" \
  -e OLLAMA_FLASH_ATTENTION=1 \
  -e OLLAMA_KV_CACHE_TYPE="$kv" \
  -e OLLAMA_KEEP_ALIVE=30m \
  -e OLLAMA_NUM_PARALLEL=1 \
  -e OLLAMA_NOPRUNE=1 \
  --memory 16g --memory-swap 16g \
  ollama/ollama:0.40.16 >/dev/null || { echo "docker run failed"; exit 1; }
timeout 120 bash -c 'until curl -sf http://127.0.0.1:11434/api/version >/dev/null; do sleep 1; done' || { echo "ollama did not start"; docker logs ollama 2>&1 | tail -20; exit 1; }
echo "ollama $(curl -s http://127.0.0.1:11434/api/version)"
docker inspect ollama --format '{{json .Config.Env}}' | tr ',' '\n' | grep OLLAMA_ > "$F/server-env.txt"

start=$(date +%s.%N)
curl -s http://127.0.0.1:11434/api/generate -d "{\"model\": \"$tag\", \"prompt\": \"\", \"keep_alive\": \"30m\"}" > /dev/null
echo "load $(python3 -I -c "import time; print(round(time.time() - $start, 1))") s, VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader), container $(docker stats --no-stream --format '{{.MemUsage}}' ollama)"
docker exec ollama ollama ps | tee "$F/ollama-ps.txt"
curl -s http://127.0.0.1:11434/api/tags | python3 -I -c "
import json, sys
for m in json.load(sys.stdin)['models']:
    if m['name'] == sys.argv[1] or m['model'] == sys.argv[1]:
        print(m['name'], 'digest', m['digest'], 'size', m['size'], 'modified', m.get('modified_at'))
" "$tag" | tee "$F/model-digest.txt"
curl -s http://127.0.0.1:11434/api/show -d "{\"model\": \"$tag\"}" | python3 -I -c "
import json, sys
d = json.load(sys.stdin)
print('details:', json.dumps(d.get('details')), '| capabilities:', d.get('capabilities'))
" | tee "$F/model-details.txt"

# The LibreDB Studio native tool_calls probe (docs/llms/setup.md), unchanged except for the model name.
curl -s http://127.0.0.1:11434/api/chat -d "{
  \"model\": \"$tag\",
  \"messages\": [{\"role\": \"user\", \"content\": \"Check this database health.\"}],
  \"tools\": [{\"type\": \"function\", \"function\": {
    \"name\": \"inspect_operations\",
    \"description\": \"Read live database operational state.\",
    \"parameters\": {\"type\": \"object\",
      \"properties\": {\"kind\": {\"type\": \"string\", \"enum\": [\"health\"]}},
      \"required\": [\"kind\"]}}}],
  \"stream\": false
}" > "$F/native-tool-probe.json"
echo "native tool probe: $(grep -o '"tool_calls".\{0,120\}' "$F/native-tool-probe.json" || echo 'no tool_calls (prose answer)')"

nvidia-smi --query-gpu=timestamp,memory.used,power.draw,clocks.sm,temperature.gpu --format=csv,noheader -l 2 > "$F/gpu-trace.csv" 2>&1 &
trace=$!
(while true; do echo "$(date +%T) $(docker stats --no-stream --format '{{.MemUsage}} {{.CPUPerc}}' ollama)"; sleep 4; done) > "$F/docker-stats.log" 2>&1 &
dstat=$!

S=$(date +%s)
heavy python3 -I "$B/llm_probe.py" --base http://127.0.0.1:11434/v1 --model "$tag" --label "ollama-$key" \
  --runs 3 --long-tokens "$long_tokens" --extra "$extra" --out "$F/probe-results.json" > "$F/probe-summary.json" 2> "$F/probe.err"
echo "probe exit=$? after $(( $(date +%s) - S )) s"
tail -3 "$F/probe.err"

if [ -n "$oc_model" ]; then
  cd "$OC" && rm -f hello.py
  S=$(date +%s)
  timeout 900 heavy ~/.opencode/bin/opencode run --standalone --archived --auto --format json -m "$oc_model" \
    --title "e2e ollama $key" \
    "Create a file named hello.py containing a function add(a, b) that returns a + b, and a main block that prints add(2, 3). Then read the file back and tell me how many lines it has. Do not run any shell commands." \
    > "$F/opencode-e2e-events.jsonl" 2> "$F/opencode-e2e.err"
  echo "opencode exit=$? after $(( $(date +%s) - S )) s"
  cat -n hello.py 2>&1 | head -10
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
        st = p.get("state", {})
        print("tool_use:", p.get("tool"), st.get("status"), (st.get("error") or "")[:160])
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
docker logs ollama > "$F/server.log" 2>&1
docker stop -t 30 ollama >/dev/null
python3 -I - "$F/probe-summary.json" <<'EOF'
import json, sys
try:
    s = json.load(open(sys.argv[1]))
except (OSError, ValueError) as e:
    print("no probe summary:", e)
    raise SystemExit(0)
print("ok:", repr((s["ok"].get("content") or "")[:40]), "| reasoning_chars", s["ok"].get("reasoning_chars"), "| finish", s["ok"].get("finish"))
print("tool:", s["tool"].get("finish"), s["tool"].get("calls"), "| content", (s["tool"].get("content") or "")[:80])
print("decode:", [(d.get("decode_tok_s"), (d.get("usage") or {}).get("completion_tokens"), d.get("ttft_s")) for d in s["decode_runs"]], "median", s["decode_tok_s_median"])
for l in s["long"]:
    print("long %s: prompt %s ttft first %.2f s, repeat %.2f s" % (l["target"], l["prompt_tokens"], l["ttft_first_s"], l["ttft_repeat_s"]))
EOF
