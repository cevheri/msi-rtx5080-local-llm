#!/usr/bin/env bash
# usage: qwen36-variant.sh LABEL NCMOE [extra llama-server args...]
# Restarts the Qwen3.6-35B-A3B llama.cpp container with the recipe flags plus the extra args,
# then runs a short probe (OK, tool call, 3 decodes, one 8k prompt sent twice) and prints one JSON line.
set -uo pipefail
B=$SCRATCH/bench
V=$B/variants
mkdir -p "$V"
label=$1
ncmoe=$2
shift 2

docker rm -f llamacpp >/dev/null 2>&1
start=$(date +%s.%N)
docker run -d --name llamacpp \
  --user "$(id -u):$(id -g)" \
  --device nvidia.com/gpu=all \
  -e GGML_CUDA_DISABLE_GRAPHS=1 \
  -e HF_HOME=/hf \
  -v "$HOME/.cache/huggingface:/hf" \
  -p 127.0.0.1:8080:8080 \
  --memory "${MEM:-20g}" --memory-swap "${MEM:-20g}" \
  ghcr.io/ggml-org/llama.cpp:server-cuda-b12407 \
  -m /hf/hub/models--ggml-org--Qwen3.6-35B-A3B-GGUF/snapshots/52f9f9b8052fb9c8b69bba4a95eda0f319ac3d2a/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --alias local -ngl all --n-cpu-moe "$ncmoe" -fa on -c 131072 -np 1 --jinja \
  --reasoning off \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0 \
  "$@" >/dev/null || { echo "{\"label\": \"$label\", \"error\": \"docker run failed\"}"; exit 1; }

if ! timeout 400 bash -c 'until curl -sf http://127.0.0.1:8080/health >/dev/null; do sleep 1; [ "$(docker inspect -f "{{.State.Running}}" llamacpp)" = true ] || exit 1; done'; then
  docker logs llamacpp > "$V/$label-server.log" 2>&1
  echo "{\"label\": \"$label\", \"error\": \"server did not start\", \"log\": \"$V/$label-server.log\"}"
  tail -15 "$V/$label-server.log"
  exit 1
fi
load_s=$(python3 -I -c "import time; print(round(time.time() - $start, 1))")
vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc '0-9')
mem=$(docker stats --no-stream --format '{{.MemUsage}}' llamacpp)
threads=$(docker logs llamacpp 2>&1 | grep -o 'n_threads = [0-9]*' | head -1)

nvidia-smi --query-gpu=timestamp,memory.used,power.draw,clocks.sm,temperature.gpu --format=csv,noheader -l 1 > "$V/$label-gpu.csv" 2>&1 &
trace=$!
python3 -I "$B/llm_probe.py" --base http://127.0.0.1:8080/v1 --model local --label "$label" \
  --runs 3 --long-tokens 8000 --out "$V/$label.json" > "$V/$label-summary.json" 2> "$V/$label.err"
probe_rc=$?
kill "$trace"
peak=$(cut -d, -f2 "$V/$label-gpu.csv" | tr -dc '0-9\n' | sort -n | tail -1)
mem2=$(docker stats --no-stream --format '{{.MemUsage}}' llamacpp)
docker logs llamacpp > "$V/$label-server.log" 2>&1

python3 -I - "$V/$label.json" "$label" "$load_s" "$vram" "$peak" "$mem" "$mem2" "$threads" "$probe_rc" <<'EOF'
import json
import sys
f, label, load_s, vram, peak, mem, mem2, threads, rc = sys.argv[1:]
out = {"label": label, "probe_rc": int(rc), "load_s": float(load_s), "vram_after_load_MiB": int(vram),
       "vram_peak_MiB": int(peak or 0), "mem_after_load": mem, "mem_after_probe": mem2, "threads": threads}
try:
    r = json.load(open(f))
except (OSError, ValueError) as e:
    out["error"] = f"no probe results: {e}"
else:
    lp = r["long_prompt"][0]
    st = lp["first"].get("server_timings") or {}
    out.update({
        "decode_tok_s": [d.get("decode_tok_s") for d in r["decode_runs"]],
        "server_decode_tok_s": [(d.get("server_timings") or {}).get("predicted_per_second") for d in r["decode_runs"]],
        "ok": r["ok_test"]["content"], "tool": r["tool_test"]["calls"],
        "prefill_8k": {"prompt_n": st.get("prompt_n"), "tok_s": st.get("prompt_per_second"), "ttft_s": lp["first"]["ttft_s"]},
        "repeat_ttft_s": lp["repeat_same_prompt"]["ttft_s"],
    })
print(json.dumps(out))
EOF
