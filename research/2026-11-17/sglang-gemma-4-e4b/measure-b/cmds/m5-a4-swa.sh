#!/usr/bin/env bash
# M5 follow-up: as m5-a3 (audio_config null) plus --swa-full-tokens-ratio 0.1, because m5-a3's pool held 42,413 tokens, short of the 64k context.
set -euo pipefail
docker run -d --name sglang-measure \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 32g --memory-swap 32g \
  -p 127.0.0.1:30000:30000 \
  -v sglang-research-cache:/root/.cache/sglang \
  -v "$SCRATCH/sglang/measure-b/overlay/gemma-4-e4b:/llmctl/model:ro" \
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
    --served-model-name gemma-4-e4b \
    --context-length 65536 \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.85 \
    --max-running-requests 1 \
    --swa-full-tokens-ratio 0.1 \
    --disable-prefill-cuda-graph \
    --json-model-override-args '{"audio_config": null}' \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4
