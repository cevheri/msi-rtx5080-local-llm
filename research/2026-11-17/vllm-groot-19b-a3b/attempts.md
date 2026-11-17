# vLLM v0.33.0 startup attempts for GrootTeam/Groot-19B-A3B-261023-NVFP4 on the RTX 5080 Laptop GPU (2026-11-17)

Common flags: `--device nvidia.com/gpu=all --ipc=host --memory 32g --memory-swap 32g`, `--max-model-len` as listed, `--language-model-only`, `--max-num-seqs 1`, `--kv-cache-dtype fp8`, `--attention-backend TRITON_ATTN`, parsers `qwen3_coder` / `qwen3`, `--default-chat-template-kwargs '{"enable_thinking":false}'`.
vLLM logged `Using 'MARLIN' NvFp4 MoE backend`, `Model loading took 10.99 GiB memory` and free memory of 15.08 of 15.47 GiB on startup.

| # | --gpu-memory-utilization | Other flags | --max-model-len | Result |
|---|---|---|---|---|
| 1 | 0.92 (vLLM default, not passed) | shipped recipe, CUDA graphs on (0.06 GiB of graphs), torch.compile 16.57 s | 131072 | Started: `Available KV cache memory: 1.87 GiB`, `GPU KV cache size: 167,480 tokens, Maximum concurrency for 131,072 tokens per request: 1.28x` |
