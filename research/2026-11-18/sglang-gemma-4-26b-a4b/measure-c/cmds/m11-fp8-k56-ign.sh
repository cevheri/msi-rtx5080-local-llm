#!/usr/bin/env bash
# M11 retry: m11-fp8-k56 plus --json-model-override-args with the checkpoint's quantization_config whose ignore list matches the vision tower by regex (re:.*vision_tower.*), the dense MLP and the routers, because v0.5.23's fused vision projections do not match the checkpoint's .linear names and Marlin cannot repack the 8,608-wide vision MLP.
set -euo pipefail
docker run -d --name sglang-measure \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 32g --memory-swap 32g \
  -p 127.0.0.1:30000:30000 \
  -v sglang-research-cache:/root/.cache/sglang \
  -v "$SCRATCH/sglang/measure-c/overlay/gemma-4-26b-a4b:/llmctl/model:ro" \
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
    --served-model-name gemma-4-26b-a4b \
    --context-length 131072 \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.90 \
    --max-running-requests 1 \
    --disable-prefill-cuda-graph \
    --moe-runner-backend marlin \
    --enable-paged-experts \
    --paged-experts-num-resident 56 \
    --kv-cache-dtype fp8_e4m3 \
    --swa-full-tokens-ratio 0.1 \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --json-model-override-args '{"quantization_config":{"config_groups":{"group_0":{"format":"pack-quantized","input_activations":null,"output_activations":null,"targets":["Linear"],"weights":{"actorder":null,"block_structure":null,"dynamic":false,"group_size":32,"num_bits":4,"observer":"mse","observer_kwargs":{},"scale_dtype":null,"strategy":"group","symmetric":true,"type":"int","zp_dtype":null}}},"format":"pack-quantized","global_compression_ratio":null,"ignore":["lm_head","re:.*vision_tower.*","re:.*embed_vision.*","re:.*\\.mlp\\.(gate|up|down)_proj$","re:.*\\.router\\.proj$"],"kv_cache_scheme":null,"quant_method":"compressed-tensors","quantization_status":"compressed","sparsity_config":{},"transform_config":{},"version":"0.1.dev489+g6d69620"}}'
