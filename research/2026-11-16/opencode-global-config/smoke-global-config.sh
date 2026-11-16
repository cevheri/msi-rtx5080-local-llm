#!/usr/bin/env bash
# Smoke-test the installed global OpenCode config: default model (no -m) against Qwen3.6-35B-A3B on llama.cpp.
# The scratch project adds only permission denies, so the providers, default model, MCP block and skills come from the real global setup.
set -u
OUT=$SCRATCH/oc-real
mkdir -p "$OUT/proj"
cd "$OUT/proj" || exit 1
rm -f hello.py
cat > opencode.json <<'EOF'
{
  "$schema": "https://opencode.ai/config.json",
  "permission": { "bash": "deny", "external_directory": "deny", "webfetch": "deny", "websearch": "deny", "task": "deny", "question": "deny" }
}
EOF

docker rm -f llamacpp >/dev/null 2>&1
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
  -lm none -t 8 --moe-cache-mib 3200 \
  --reasoning off \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0 >/dev/null || exit 1

S=$(date +%s)
for _ in $(seq 1 180); do
  curl -sf http://127.0.0.1:8080/health >/dev/null && break
  sleep 1
done
echo "server healthy after $(( $(date +%s) - S )) s"

nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -lms 500 > "$OUT/gpu-trace.csv" &
trace=$!

S=$(date +%s)
timeout 900 heavy ~/.opencode/bin/opencode run --standalone --archived --auto --format json \
  --title "smoke global config" \
  "Create a file named hello.py containing a function add(a, b) that returns a + b, and a main block that prints add(2, 3). Then read the file back and tell me how many lines it has. Do not run any shell commands." \
  > "$OUT/events.jsonl" 2> "$OUT/err.log"
echo "opencode exit=$? after $(( $(date +%s) - S )) s"
kill "$trace"
cat -n hello.py 2>&1 | head -12

python3 -I - "$OUT/events.jsonl" <<'EOF'
import json
import sys
models = set()
for line in open(sys.argv[1]):
    try:
        e = json.loads(line)
    except ValueError:
        continue
    t = e.get("type")
    p = e.get("part", {}) or {}
    for src in (e, p, p.get("info") or {}):
        if isinstance(src, dict) and src.get("modelID"):
            models.add(f"{src.get('providerID')}/{src.get('modelID')}")
    if t == "tool_use":
        print("tool_use:", p.get("tool"), p.get("state", {}).get("status"))
    elif t == "step_finish":
        print("step tokens:", json.dumps(p.get("tokens")))
    elif t == "text":
        print("answer:", p.get("text", "")[:200].replace("\n", " "))
    elif t == "error":
        print("error:", json.dumps(e)[:300])
print("models seen in events:", sorted(models) or "none")
EOF
echo "stderr lines: $(wc -l < "$OUT/err.log")"
grep -i -E 'mcp|libredb|error' "$OUT/err.log" | cut -c1-200 | head -8
echo "GPU peak $(cut -d, -f2 "$OUT/gpu-trace.csv" | tr -dc '0-9\n' | sort -n | tail -1) MiB"

docker stop -t 60 llamacpp >/dev/null
docker logs llamacpp 2>&1 | tr '\r' '\n' | grep -E 'print_timing|prompt eval time|       eval time' > "$OUT/server-timings.log"
grep -E 'prompt eval time' "$OUT/server-timings.log" | head -6
