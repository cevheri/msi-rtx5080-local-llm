#!/usr/bin/env bash
# Measurement D: K2 Horizon 7B at B's draft settings with --mem-fraction-static 0.88 instead of 0.90 (plan ruling 12), the overlay (temperature 1.0, top_p 0.95) and all eight variables of the runtime file, PYTORCH_ALLOC_CONF included.
set -euo pipefail
docker run -d --name sglang-measure \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 32g --memory-swap 32g \
  -p 127.0.0.1:30000:30000 \
  -v sglang-research-cache:/root/.cache/sglang \
  -v "$SCRATCH/sglang/measure-d/overlay/k2-horizon-7b:/llmctl/model:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e HF_MODULES_CACHE=/tmp/hf_modules \
  -e MAX_JOBS=2 \
  -e FLASHINFER_NO_DOWNLOAD=1 \
  -e SYNC_TOKEN_IDS_ACROSS_TP=1 \
  -e NCCL_MAX_P2P_NCHANNELS=1 \
  -e TIKTOKEN_ENCODINGS_BASE=/opt/sglang/lib/python3.12/site-packages/tml_renderers/data \
  -e PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  lmsysorg/sglang:v0.5.23-runtime \
  python3 -m sglang.launch_server \
    --model-path /llmctl/model \
    --served-model-name k2-horizon-7b \
    --context-length 65536 \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.88 \
    --max-running-requests 1 \
    --disable-prefill-cuda-graph \
    --trust-remote-code \
    --kv-cache-dtype fp8_e4m3 \
    --tool-call-parser k2_horizon \
    --reasoning-parser k2_horizon \
    --default-chat-template-kwargs '{"reasoning_effort":"high","tool_call_format":"xml"}'
