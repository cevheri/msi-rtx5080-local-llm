#!/usr/bin/env bash
# Step 0: the shipped vLLM gpt-oss-20b recipe (recipes/vllm/gpt-oss-20b.toml) mirrored by hand on vLLM v0.33.0 as llmctl's container_spec builds it, with --network none, no published port, and new empty volumes for both cache paths.
set -euo pipefail
docker run -d --name vllm-measure \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 32g --memory-swap 32g \
  --network none \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_NO_USAGE_STATS=1 \
  -e MAX_JOBS=2 \
  -e HF_MODULES_CACHE=/tmp/hf_modules \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -v vllm-measure-cold:/root/.cache/vllm \
  -v vllm-measure-cold-fi:/root/.cache/flashinfer \
  vllm/vllm-openai:v0.33.0 \
    /root/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee \
    --served-model-name gpt-oss-20b \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.965 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 256 \
    --enforce-eager \
    --kv-cache-dtype fp8 \
    --attention-backend TRITON_ATTN \
    --enable-auto-tool-choice \
    --tool-call-parser openai \
    --reasoning-parser openai_gptoss \
    --reasoning-effort-map '{"none":"low","minimal":"low","xhigh":"high","max":"high"}' \
    --override-generation-config '{"temperature":1.0,"top_p":1.0}'
