#!/usr/bin/env bash
# One vLLM DeepSeek-R1 start for measurement C's step 4: refuse when the cache volume should be new and exists, run
# run_c.sh (readiness, traces, the sequence), then list the volume with a throwaway container and remove it unless
# the next start needs it warm.
# Usage: run_vllm_cold_c.sh <run> <cmd-file> <new|reuse> <remove|keep> [seq_c.py arguments...]; needs SCRATCH and HOME.
set -uo pipefail
: "${SCRATCH:?}"
C=$SCRATCH/sglang/measure-c; R=$C/vllm-deepseek-r1-0528-qwen3-8b; vol=vllm-measure-cold
run=$1; cmd=$2; mode=$3; after=$4; shift 4
mkdir -p "$R/logs"
if [ "$mode" = new ] && docker volume inspect "$vol" >/dev/null 2>&1; then echo "refusing: volume $vol exists; it must be new and empty"; exit 9; fi
{ echo "volume_mode=$mode volume_after=$after"; docker volume inspect "$vol" --format 'volume {{.Name}} exists before the start, created {{.CreatedAt}}' 2>/dev/null || echo "volume $vol absent before the start"; } > "$R/logs/$run.volume-before.txt"
bash "$C/scripts/run_c.sh" "$R" "$run" "$cmd" vllm deepseek-r1-8b "$@"
docker run --rm --network none --memory 1g --memory-swap 1g -v "$vol:/v:ro" --entrypoint sh vllm/vllm-openai:v0.33.0 \
  -c 'echo "files: $(find /v -type f | wc -l)"; du -sh /v; du -sh /v/* 2>/dev/null' > "$R/logs/$run.volume-$vol.txt" 2>&1
if [ "$after" = remove ]; then docker volume rm "$vol" > /dev/null && echo "removed volume $vol" >> "$R/logs/$run.meta"; fi
cat "$R/logs/$run.volume-$vol.txt"
