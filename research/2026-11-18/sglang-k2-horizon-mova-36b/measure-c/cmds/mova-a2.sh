#!/usr/bin/env bash
# K2 Horizon MoVA, second start: mova-a1 plus --json-model-override-args '{"xllm_source_router_gemm_partitions": 2}', the value the first start's error message gives, to reach the weight loading.
set -euo pipefail
docker run -d --name sglang-measure \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 32g --memory-swap 32g \
  -p 127.0.0.1:30000:30000 \
  -v sglang-research-cache:/root/.cache/sglang \
  -v "$SCRATCH/sglang/measure-c/overlay/k2-horizon-mova-36b:/llmctl/model:ro" \
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
    --served-model-name k2-horizon-mova-36b \
    --context-length 65536 \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.90 \
    --max-running-requests 1 \
    --disable-prefill-cuda-graph \
    --moe-runner-backend marlin \
    --enable-paged-experts \
    --paged-experts-num-resident 8 \
    --trust-remote-code \
    --kv-cache-dtype fp8_e4m3 \
    --tool-call-parser k2_horizon \
    --reasoning-parser k2_horizon \
    --default-chat-template-kwargs '{"reasoning_effort":"high","tool_call_format":"xml"}' \
    --json-model-override-args '{"xllm_source_router_gemm_partitions": 2}'
