#!/usr/bin/env bash
# k2-horizon-7b, attempt 1: SGLang v0.5.23 with the vLLM recipe's checkpoint, 64k context, FP8 KV, parsers and template defaults.
set -euo pipefail
docker run -d --name sglang-research \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 32g --memory-swap 32g \
  -p 127.0.0.1:30000:30000 \
  -e HF_HUB_OFFLINE=1 \
  -e HF_MODULES_CACHE=/tmp/hf_modules \
  -e MAX_JOBS=2 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -v sglang-research-cache:/root/.cache/sglang \
  lmsysorg/sglang:v0.5.23-runtime \
  python3 -m sglang.launch_server \
    --model-path /root/.cache/huggingface/hub/models--cyankiwi--K2-Horizon-7B-AWQ-INT4/snapshots/530d5ceccd8ea660ceaa2a9f3a4bcbb68ba44291 \
    --served-model-name k2-horizon-7b \
    --context-length 65536 \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.90 \
    --max-running-requests 1 \
    --trust-remote-code \
    --kv-cache-dtype fp8_e4m3 \
    --tool-call-parser k2_horizon \
    --reasoning-parser k2_horizon \
    --default-chat-template-kwargs '{"reasoning_effort":"high","tool_call_format":"xml"}'
