#!/usr/bin/env bash
# Cold check: the shipped recipe as container_spec builds it plus the tokenizer folder as llmctl will ship it (every snapshot file linked at its container path and a real tokenizer_config.json naming PreTrainedTokenizerFast), mounted read-only at /llmctl/tokenizer, with --tokenizer /llmctl/tokenizer right after --max-model-len; the default --gpu-memory-utilization 0.92; a new empty volume vllm-measure-cold in place of the vllm-cache compile cache.
set -euo pipefail
docker run -d --name vllm-measure \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 32g --memory-swap 32g \
  -p 127.0.0.1:8000:8000 \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_NO_USAGE_STATS=1 \
  -e MAX_JOBS=2 \
  -e HF_MODULES_CACHE=/tmp/hf_modules \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -v vllm-measure-cold:/root/.cache/vllm \
  -v flashinfer-cache:/root/.cache/flashinfer \
  -v "$SCRATCH/sglang/measure-c/vllm-deepseek-r1-0528-qwen3-8b/tokenizer-folder:/llmctl/tokenizer:ro" \
  vllm/vllm-openai:v0.33.0 \
    /root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528-Qwen3-8B/snapshots/6e8885a6ff5c1dc5201574c8fd700323f23c25fa \
    --served-model-name deepseek-r1-8b \
    --max-model-len 65536 \
    --tokenizer /llmctl/tokenizer \
    --quantization fp8_per_tensor \
    --max-num-seqs 1 \
    --kv-cache-dtype fp8 \
    --attention-backend TRITON_ATTN \
    --reasoning-parser deepseek_r1 \
    --override-generation-config '{"temperature":0.6,"top_p":0.95}'
