#!/usr/bin/env bash
# Coder-Next with f16 KV at 131072 and the cache shrunk by the KV growth (1,440 MiB), to test whether
# the q8_0 KV path causes the steep decode drop at depth. Same probe depths as the main run.
set -uo pipefail
B=$SCRATCH/bench
RUNS=3 LONG=8000,30000,100000 bash "$B/variant.sh" coder-next-f16kv-cache1504 unsloth/Qwen3-Coder-Next-GGUF Qwen3-Coder-Next-UD-IQ4_XS.gguf 35g -- \
  -m MODEL_PATH --alias local -c 131072 -np 1 -fa on --jinja -ngl all -lm none -t 8 \
  --cache-ram 4096 -ctxcp 8 --temp 1.0 --top-p 0.95 --top-k 40 --min-p 0.01 -lv 4 \
  --n-cpu-moe 38 --moe-cache-mib 1504
