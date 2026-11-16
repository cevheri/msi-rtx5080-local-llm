#!/usr/bin/env bash
# Three Coder-Next offload variants with the short probe, one after another.
set -uo pipefail
B=$SCRATCH/bench
R=unsloth/Qwen3-Coder-Next-GGUF
G=Qwen3-Coder-Next-UD-IQ4_XS.gguf
COMMON=(-m MODEL_PATH --alias local -c 131072 -np 1 -fa on --jinja -ctk q8_0 -ctv q8_0 -ngl all -lm none -t 8
        --cache-ram 4096 -ctxcp 8 --temp 1.0 --top-p 0.95 --top-k 40 --min-p 0.01 -lv 4)
bash "$B/variant.sh" coder-next-ncmoe38-cache2944 "$R" "$G" 35g -- "${COMMON[@]}" --n-cpu-moe 38 --moe-cache-mib 2944
bash "$B/variant.sh" coder-next-ncmoe38-cache3200 "$R" "$G" 35g -- "${COMMON[@]}" --n-cpu-moe 38 --moe-cache-mib 3200
bash "$B/variant.sh" coder-next-ncmoe36-cache1664 "$R" "$G" 33g -- "${COMMON[@]}" --n-cpu-moe 36 --moe-cache-mib 1664
