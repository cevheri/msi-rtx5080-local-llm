#!/usr/bin/env bash
# M6 step 4, extra start 2: as extra start 1 plus NCCL_MAX_P2P_NCHANNELS=1, which cut the one-rank shared net buffer from 512 MiB to 8 MiB in the NCCL probe (logs/nccl-probe/run2.log).
set -euo pipefail
docker run -d --name sglang-measure \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 32g --memory-swap 32g \
  -p 127.0.0.1:30000:30000 \
  -v sglang-research-cache:/root/.cache/sglang \
  -e HF_HUB_OFFLINE=1 \
  -e SYNC_TOKEN_IDS_ACROSS_TP=1 \
  -e NCCL_MAX_P2P_NCHANNELS=1 \
  -e HF_MODULES_CACHE=/tmp/hf_modules \
  -e MAX_JOBS=2 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  lmsysorg/sglang:v0.5.23-runtime \
  python3 -m sglang.launch_server \
    --model-path /root/.cache/huggingface/hub/models--RedHatAI--Qwen3.5-9B-NVFP4/snapshots/a4df85a20fb7e54740f564f15af0c2f5d9c6616c \
    --served-model-name qwen3.5-9b \
    --context-length 131072 \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.88 \
    --max-running-requests 1 \
    --max-mamba-cache-size 8 \
    --disable-prefill-cuda-graph \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking":false}'
