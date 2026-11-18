#!/usr/bin/env bash
# M6 order 1 and M7: gpt-oss-20b at the a4 settings (Paged Experts, 24 of 32 resident).
set -euo pipefail
docker run -d --name sglang-measure \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 32g --memory-swap 32g \
  -p 127.0.0.1:30000:30000 \
  -v sglang-research-cache:/root/.cache/sglang \
  -e HF_HUB_OFFLINE=1 \
  -e HF_MODULES_CACHE=/tmp/hf_modules \
  -e MAX_JOBS=2 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  lmsysorg/sglang:v0.5.23-runtime \
  python3 -m sglang.launch_server \
    --model-path /root/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee \
    --served-model-name gpt-oss-20b \
    --context-length 131072 \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.90 \
    --max-running-requests 1 \
    --kv-cache-dtype fp8_e4m3 \
    --swa-full-tokens-ratio 0.1 \
    --chunked-prefill-size 1024 \
    --disable-prefill-cuda-graph \
    --moe-runner-backend marlin \
    --enable-paged-experts \
    --paged-experts-num-resident 24 \
    --tool-call-parser gpt-oss \
    --reasoning-parser gpt-oss
