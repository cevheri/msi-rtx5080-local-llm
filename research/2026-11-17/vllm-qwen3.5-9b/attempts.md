# vLLM v0.33.0 startup attempts for RedHatAI/Qwen3.5-9B-NVFP4 on the RTX 5080 Laptop GPU (2026-11-17)

Common flags: `--device nvidia.com/gpu=all --ipc=host --memory 32g --memory-swap 32g`, `--max-model-len` as listed, `--language-model-only`, `--max-num-seqs 1`, parsers `qwen3_coder` / `qwen3`, `--default-chat-template-kwargs '{"enable_thinking":false}'`.
vLLM logged `Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM`, `Using FLASH_ATTN attention backend` and `Model loading took 7.48 GiB memory`.

| # | --gpu-memory-utilization | Other flags | --max-model-len | Result |
|---|---|---|---|---|
| 1 | 0.92 (vLLM default, not set by the recipe) | defaults (CUDA graphs on, capture sizes [1, 2]) | 131072 | Started: `Available KV cache memory: 4.71 GiB`, `GPU KV cache size: 147,663 tokens, Maximum concurrency for 131,072 tokens per request: 1.13x` |
