#!/usr/bin/env bash
# Diagnostic start without the pinned-store fix: diag-k72-f088 without PYTORCH_ALLOC_CONF and under the 34 GiB cap the brief allows (estimate 33.3 GiB with B's 1.84), to see whether the setting causes the failed tool calls.
set -euo pipefail
docker run -d --name sglang-measure \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 34g --memory-swap 34g \
  -p 127.0.0.1:30000:30000 \
  -v sglang-research-cache:/root/.cache/sglang \
  -v "$SCRATCH/sglang/measure-c/overlay/nemotron-3.5-lightning:/llmctl/model:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e HF_MODULES_CACHE=/tmp/hf_modules \
  -e MAX_JOBS=2 \
  -e SYNC_TOKEN_IDS_ACROSS_TP=1 \
  -e NCCL_MAX_P2P_NCHANNELS=1 \
  -e FLASHINFER_NO_DOWNLOAD=1 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  lmsysorg/sglang:v0.5.23-runtime \
  python3 -m sglang.launch_server \
    --model-path /llmctl/model \
    --served-model-name nemotron-3.5-lightning \
    --context-length 131072 \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.88 \
    --max-running-requests 1 \
    --disable-prefill-cuda-graph \
    --moe-runner-backend marlin \
    --enable-paged-experts \
    --paged-experts-num-resident 72 \
    --kv-cache-dtype bf16 \
    --max-mamba-cache-size 8 \
    --tool-call-parser qwen3_coder \
    --reasoning-parser nemotron_3 \
    --default-chat-template-kwargs '{"enable_thinking":false}'
