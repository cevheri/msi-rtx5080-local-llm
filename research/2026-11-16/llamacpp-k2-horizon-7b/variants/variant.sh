#!/usr/bin/env bash
# usage: variant.sh LABEL REPO FILE MEMCAP -- llama-server args (MODEL_PATH is replaced)
# Short comparison run: OK, tool call, 3 decodes and one 8k prompt sent twice, then a graceful stop
# so the MoE cache statistics are logged. Prints one JSON line. Records go to bench/variants/LABEL*.
set -uo pipefail
B=$SCRATCH/bench
V=$B/variants
mkdir -p "$V"
label=$1; repo=$2; file=$3; memcap=$4
shift 4
[ "${1:-}" = "--" ] && shift
host_path=$(ls -1 "$HOME/.cache/huggingface/hub/models--${repo//\//--}"/snapshots/*/"$file" 2>/dev/null | head -1)
[ -n "$host_path" ] || { echo "{\"label\": \"$label\", \"error\": \"model file not in the cache\"}"; exit 1; }
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
  "${args[@]}" >/dev/null || { echo "{\"label\": \"$label\", \"error\": \"docker run failed\"}"; exit 1; }
if ! timeout 600 bash -c 'until curl -sf http://127.0.0.1:8080/health >/dev/null; do sleep 1; [ "$(docker inspect -f "{{.State.Running}}" llamacpp)" = true ] || exit 1; done'; then
  docker logs llamacpp > "$V/$label-server.log" 2>&1
  echo "{\"label\": \"$label\", \"error\": \"server did not start\", \"log\": \"$V/$label-server.log\"}"
  tail -15 "$V/$label-server.log"
  exit 1
fi
load_s=$(python3 -I -c "import time; print(round(time.time() - $start, 1))")
vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc '0-9')
mem=$(docker stats --no-stream --format '{{.MemUsage}}' llamacpp)

nvidia-smi --query-gpu=timestamp,memory.used,power.draw,clocks.sm,temperature.gpu --format=csv,noheader -l 1 > "$V/$label-gpu.csv" 2>&1 &
trace=$!
python3 -I "$B/llm_probe.py" --base http://127.0.0.1:8080/v1 --model local --label "$label" \
  --runs "${RUNS:-3}" --long-tokens "${LONG:-8000}" --out "$V/$label.json" > "$V/$label-summary.json" 2> "$V/$label.err"
probe_rc=$?
kill "$trace"
peak=$(cut -d, -f2 "$V/$label-gpu.csv" | tr -dc '0-9\n' | sort -n | tail -1)
mem2=$(docker stats --no-stream --format '{{.MemUsage}}' llamacpp)
docker stop -t 60 llamacpp >/dev/null
docker logs llamacpp > "$V/$label-server.log" 2>&1
hit=$(grep -o 'ubatch <= 8: .*hit rate = [0-9.]*%' "$V/$label-server.log" | grep -o 'hit rate = [0-9.]*%' | tail -1)

python3 -I - "$V/$label.json" "$label" "$load_s" "$vram" "$peak" "$mem" "$mem2" "$probe_rc" "$hit" <<'EOF'
import json
import sys
f, label, load_s, vram, peak, mem, mem2, rc, hit = sys.argv[1:]
out = {"label": label, "probe_rc": int(rc), "load_s": float(load_s), "vram_after_load_MiB": int(vram),
       "vram_peak_MiB": int(peak or 0), "mem_after_load": mem, "mem_after_probe": mem2, "decode_cache_hit": hit}
try:
    r = json.load(open(f))
except (OSError, ValueError) as e:
    out["error"] = f"no probe results: {e}"
else:
    out["long"] = [{"target": l["target_tokens"], "new": (l["first"].get("server_timings") or {}).get("prompt_n"), "pp": round((l["first"].get("server_timings") or {}).get("prompt_per_second") or 0), "tg": round((l["first"].get("server_timings") or {}).get("predicted_per_second") or 0, 1)} for l in r["long_prompt"]]
    lp = r["long_prompt"][0]
    st = lp["first"].get("server_timings") or {}
    out.update({
        "server_decode_tok_s": [round((d.get("server_timings") or {}).get("predicted_per_second") or 0, 1) for d in r["decode_runs"]],
        "ok": r["ok_test"]["content"], "tool": r["tool_test"]["calls"],
        "prefill_8k": {"prompt_n": st.get("prompt_n"), "tok_s": round(st.get("prompt_per_second") or 0), "ttft_s": lp["first"]["ttft_s"]},
        "repeat_ttft_s": lp["repeat_same_prompt"]["ttft_s"],
    })
print(json.dumps(out))
EOF
