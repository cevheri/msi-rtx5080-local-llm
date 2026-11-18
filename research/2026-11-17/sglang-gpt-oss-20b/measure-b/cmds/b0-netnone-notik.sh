#!/usr/bin/env bash
# Step 0: as b0-netnone-tik without TIKTOKEN_ENCODINGS_BASE, on a new empty cache volume sglang-measure-cold2.
set -euo pipefail
docker run -d --name sglang-measure \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 32g --memory-swap 32g \
  --network none \
  -v sglang-measure-cold2:/root/.cache/sglang \
  -v "$SCRATCH/sglang/measure-b/overlay/gpt-oss-20b:/llmctl/model:ro" \
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
