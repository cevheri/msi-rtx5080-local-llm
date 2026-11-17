#!/usr/bin/env bash
# gemma-4-e4b, attempt 2: as attempt 1, loading the text-only Gemma4ForCausalLM class through --json-model-override-args, because building
# the audio tower failed with "MergedColumnParallelLinear object has no attribute weight" on this W4A16 checkpoint.
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
    --json-model-override-args '{"architectures":["Gemma4ForCausalLM"]}' \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4
