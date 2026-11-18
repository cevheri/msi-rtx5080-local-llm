#!/usr/bin/env bash
# M11 sweep: Gemma 4 26B-A4B IT QAT AWQ-INT4 (compressed-tensors W4A16) on Paged Experts, K 56 of 128 resident, Marlin, fraction 0.90, FP8 KV with SWA ratio 0.1 (the gemma-4-12b draft's KV settings), 128k, thinking off by default, the pinned-store fix.
set -euo pipefail
docker run -d --name sglang-measure \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 32g --memory-swap 32g \
  -p 127.0.0.1:30000:30000 \
  -v sglang-research-cache:/root/.cache/sglang \
  -v "$SCRATCH/sglang/measure-c/overlay/gemma-4-26b-a4b:/llmctl/model:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e HF_MODULES_CACHE=/tmp/hf_modules \
  -e MAX_JOBS=2 \
  -e SYNC_TOKEN_IDS_ACROSS_TP=1 \
  -e NCCL_MAX_P2P_NCHANNELS=1 \
  -e FLASHINFER_NO_DOWNLOAD=1 \
  -e PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  lmsysorg/sglang:v0.5.23-runtime \
  python3 -m sglang.launch_server \
    --model-path /llmctl/model \
    --served-model-name gemma-4-26b-a4b \
    --context-length 131072 \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.90 \
    --max-running-requests 1 \
    --disable-prefill-cuda-graph \
    --moe-runner-backend marlin \
    --enable-paged-experts \
    --paged-experts-num-resident 56 \
    --kv-cache-dtype fp8_e4m3 \
    --swa-full-tokens-ratio 0.1 \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4
