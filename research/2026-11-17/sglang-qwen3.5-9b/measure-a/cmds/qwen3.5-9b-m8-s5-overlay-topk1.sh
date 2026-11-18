#!/usr/bin/env bash
# M8 step 3, behavioural check: as step 3 with top_k 1 in the overlay's generation_config.json.
set -euo pipefail
docker run -d --name sglang-measure \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 32g --memory-swap 32g \
  -p 127.0.0.1:30000:30000 \
  -v sglang-research-cache:/root/.cache/sglang \
  -v "$SCRATCH/sglang/measure-a/overlay/qwen3.5-9b-topk1:/llmctl/model:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e HF_MODULES_CACHE=/tmp/hf_modules \
  -e MAX_JOBS=2 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  lmsysorg/sglang:v0.5.23-runtime \
  python3 -m sglang.launch_server \
    --model-path /llmctl/model \
    --served-model-name qwen3.5-9b \
    --context-length 131072 \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.88 \
    --max-running-requests 1 \
    --max-mamba-cache-size 8 \
    --disable-prefill-cuda-graph \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    --log-requests --log-requests-level 2
