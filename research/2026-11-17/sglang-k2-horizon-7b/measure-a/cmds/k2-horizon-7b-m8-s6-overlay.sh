#!/usr/bin/env bash
# M8 step 3, remote-code model: k2-horizon-7b at the a2 settings from a read-only overlay (generation_config.json with temperature 1.0 and top_p 0.95, the vLLM recipe's values).
set -euo pipefail
docker run -d --name sglang-measure \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 32g --memory-swap 32g \
  -p 127.0.0.1:30000:30000 \
  -v sglang-research-cache:/root/.cache/sglang \
  -v "$SCRATCH/sglang/measure-a/overlay/k2-horizon-7b:/llmctl/model:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e HF_MODULES_CACHE=/tmp/hf_modules \
  -e MAX_JOBS=2 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  lmsysorg/sglang:v0.5.23-runtime \
  python3 -m sglang.launch_server \
    --model-path /llmctl/model \
    --served-model-name k2-horizon-7b \
    --context-length 65536 \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.90 \
    --max-running-requests 1 \
    --disable-prefill-cuda-graph \
    --trust-remote-code \
    --kv-cache-dtype fp8_e4m3 \
    --tool-call-parser k2_horizon \
    --reasoning-parser k2_horizon \
    --default-chat-template-kwargs '{"reasoning_effort":"high","tool_call_format":"xml"}' \
    --log-requests --log-requests-level 2
