#!/usr/bin/env bash
# Re-runs the first request of the gemma4:12b measurement in a fresh Ollama container that gets the
# CUDA JIT cache (/root/.nv/ComputeCache) saved from the measured container, to test whether the
# 18-24 s first-request delay is kernel JIT compilation.
set -uo pipefail
B=$SCRATCH/bench
F=$B/runs/ollama-jit-test
mkdir -p "$F"
cp -r "$B/nvcache/ComputeCache" "$F/ComputeCache-copy"
docker rm -f ollama >/dev/null 2>&1
docker run -d --name ollama --device nvidia.com/gpu=all -v ollama-models:/root/.ollama \
  -v "$F/ComputeCache-copy:/root/.nv/ComputeCache" \
  -p 127.0.0.1:11434:11434 -e OLLAMA_CONTEXT_LENGTH=65536 -e OLLAMA_FLASH_ATTENTION=1 -e OLLAMA_KV_CACHE_TYPE=f16 \
  -e OLLAMA_KEEP_ALIVE=30m -e OLLAMA_NUM_PARALLEL=1 -e OLLAMA_NOPRUNE=1 --memory 16g --memory-swap 16g \
  ollama/ollama:0.40.16 >/dev/null || { echo "docker run failed"; exit 1; }
timeout 120 bash -c 'until curl -sf http://127.0.0.1:11434/api/version >/dev/null; do sleep 1; done' || exit 1
S=$(date +%s.%N)
curl -s http://127.0.0.1:11434/api/generate -d '{"model": "gemma4:12b", "prompt": "", "keep_alive": "30m"}' > /dev/null
echo "load $(python3 -I -c "import time; print(round(time.time() - $S, 1))") s"
S=$(date +%s.%N)
curl -s http://127.0.0.1:11434/api/chat -d '{
  "model": "gemma4:12b",
  "messages": [{"role": "user", "content": "Check this database health."}],
  "tools": [{"type": "function", "function": {
    "name": "inspect_operations",
    "description": "Read live database operational state.",
    "parameters": {"type": "object",
      "properties": {"kind": {"type": "string", "enum": ["health"]}},
      "required": ["kind"]}}}],
  "stream": false
}' > "$F/native-tool-probe.json"
echo "first request $(python3 -I -c "import time; print(round(time.time() - $S, 2))") s: $(grep -o '"tool_calls".\{0,80\}' "$F/native-tool-probe.json" || echo 'no tool_calls')"
docker logs ollama 2>&1 | grep -E 'prompt eval time|       eval time' | head -2 | sed -E 's/^.*\| (prompt eval|       eval)/\1/'
docker stop -t 10 ollama >/dev/null
echo "cache files before $(find "$B/nvcache/ComputeCache" -type f | wc -l), after $(find "$F/ComputeCache-copy" -type f | wc -l)"
