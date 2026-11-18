#!/usr/bin/env bash
# LD11 retry 1: the shipped vLLM recipe (recipes/vllm/deepseek-r1-0528-qwen3-8b.toml) as llmctl's container_spec builds it, plus --tokenizer-mode deepseek_v32 (which loads TokenizersBackend instead of the hub's LlamaTokenizerFast class) and --default-chat-template-kwargs with thinking true (the V3.2 encoder's thinking mode).
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
  -v vllm-cache:/root/.cache/vllm \
  -v flashinfer-cache:/root/.cache/flashinfer \
  vllm/vllm-openai:v0.33.0 \
    /root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528-Qwen3-8B/snapshots/6e8885a6ff5c1dc5201574c8fd700323f23c25fa \
    --served-model-name deepseek-r1-8b \
    --max-model-len 65536 \
    --quantization fp8_per_tensor \
    --max-num-seqs 1 \
    --kv-cache-dtype fp8 \
    --attention-backend TRITON_ATTN \
    --reasoning-parser deepseek_r1 \
    --override-generation-config '{"temperature":0.6,"top_p":0.95}' \
    --tokenizer-mode deepseek_v32 \
    --default-chat-template-kwargs '{"thinking":true}'
