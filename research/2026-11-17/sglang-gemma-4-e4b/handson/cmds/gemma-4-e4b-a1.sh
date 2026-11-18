#!/usr/bin/env bash
# gemma-4-e4b, attempt 1: SGLang v0.5.23 with the vLLM recipe's checkpoint, 64k context and parsers; the vision and audio towers
# load because v0.5.23 refuses --language-model-only for this architecture, and the prefill CUDA graph is off to keep VRAM free.
set -euo pipefail
docker run -d --name sglang-research \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 32g --memory-swap 32g \
  -p 127.0.0.1:30000:30000 \
  -e HF_HUB_OFFLINE=1 \
  -e HF_MODULES_CACHE=/tmp/hf_modules \
  -e MAX_JOBS=2 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -v sglang-research-cache:/root/.cache/sglang \
  lmsysorg/sglang:v0.5.23-runtime \
  python3 -m sglang.launch_server \
    --model-path /root/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/6cd26aaa2357fb2bad8c51699a7558a4d1a965bb \
    --served-model-name gemma-4-e4b \
    --context-length 65536 \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.85 \
    --max-running-requests 1 \
    --disable-prefill-cuda-graph \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4
