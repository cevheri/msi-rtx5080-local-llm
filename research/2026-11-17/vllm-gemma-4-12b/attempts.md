# vLLM v0.33.0 startup attempts for google/gemma-4-12B-it-qat-w4a16-ct on the RTX 5080 Laptop GPU (2026-11-17)

Common flags: `--device nvidia.com/gpu=all --ipc=host --memory 32g --memory-swap 32g`, `--max-num-seqs 1`, `--kv-cache-dtype fp8`, `--attention-backend TRITON_ATTN`, `--limit-mm-per-prompt '{"image":0,"audio":0}'`, parsers `gemma4` / `gemma4`.
vLLM logged `Model loading took 8.28 GiB memory`, and CUDA reported 15.08 GiB free of 15.47 GiB at startup.

| # | --gpu-memory-utilization | Other flags | --max-model-len | Result |
|---|---|---|---|---|
| 1 | 0.92 (vLLM default, not passed) | recipe as shipped, CUDA graphs on, max-num-batched-tokens raised by vLLM from 2048 to 2496 | 131072 | Started: `Available KV cache memory: 5.37 GiB`, `GPU KV cache size: 366,455 tokens, Maximum concurrency for 131,072 tokens per request: 2.80x` |
