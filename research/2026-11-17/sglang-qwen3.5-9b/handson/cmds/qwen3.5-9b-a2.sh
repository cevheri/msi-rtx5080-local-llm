#!/usr/bin/env bash
# qwen3.5-9b, attempt 2: as attempt 1 without --language-model-only, which v0.5.23 refuses for Qwen3_5ForConditionalGeneration.
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
    --model-path /root/.cache/huggingface/hub/models--RedHatAI--Qwen3.5-9B-NVFP4/snapshots/a4df85a20fb7e54740f564f15af0c2f5d9c6616c \
    --served-model-name qwen3.5-9b \
    --context-length 131072 \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.85 \
    --max-running-requests 1 \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking":false}'
