#!/usr/bin/env bash
# Pinned-store check, second start: pe-round1 (B's b0-a4 plus PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1) under --memory 20g, the cap that the smaller store would allow B's gpt-oss draft (non-reclaimable peak 15.44 GiB plus about 3.6 GiB), with the margin protocol and M12.
set -euo pipefail
docker run -d --name sglang-measure \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 20g --memory-swap 20g \
  -p 127.0.0.1:30000:30000 \
  -v sglang-research-cache:/root/.cache/sglang \
  -v "$SCRATCH/sglang/measure-c/overlay/gpt-oss-20b:/llmctl/model:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e HF_MODULES_CACHE=/tmp/hf_modules \
  -e MAX_JOBS=2 \
  -e SYNC_TOKEN_IDS_ACROSS_TP=1 \
  -e NCCL_MAX_P2P_NCHANNELS=1 \
  -e FLASHINFER_NO_DOWNLOAD=1 \
  -e TIKTOKEN_ENCODINGS_BASE=/opt/sglang/lib/python3.12/site-packages/tml_renderers/data \
  -e PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1 \
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
