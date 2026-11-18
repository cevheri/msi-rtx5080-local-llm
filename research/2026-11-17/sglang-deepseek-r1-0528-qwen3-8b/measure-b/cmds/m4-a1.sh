#!/usr/bin/env bash
# M4: DeepSeek-R1-0528-Qwen3-8B BF16 checkpoint quantized online to FP8, FP8 KV, fraction 0.90, no prefill graph, reasoning parser deepseek-r1, no tool parser (chat only), 64k.
set -euo pipefail
docker run -d --name sglang-measure \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 32g --memory-swap 32g \
  -p 127.0.0.1:30000:30000 \
  -v sglang-research-cache:/root/.cache/sglang \
  -v "$SCRATCH/sglang/measure-b/overlay/deepseek-r1-0528-qwen3-8b:/llmctl/model:ro" \
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
    --served-model-name deepseek-r1-8b \
    --context-length 65536 \
    --host 0.0.0.0 --port 30000 \
    --quantization fp8 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.90 \
    --max-running-requests 1 \
    --disable-prefill-cuda-graph \
    --reasoning-parser deepseek-r1
