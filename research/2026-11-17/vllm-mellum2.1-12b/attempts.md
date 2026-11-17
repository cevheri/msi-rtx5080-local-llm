# vLLM v0.33.0 startup attempts for cyankiwi/Mellum2.1-12B-A2.5B-Thinking-AWQ-INT4 on the RTX 5080 Laptop GPU (2026-11-17)

Common flags: `--device nvidia.com/gpu=all --ipc=host --memory 32g`, `--max-model-len` as listed, `--max-num-seqs 1`, `--enable-auto-tool-choice`, parsers `hermes` / `qwen3`, `--override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}'`.
vLLM logged `Model loading took 7.2 GiB memory` and CUDA reported a total of 15.47 GiB (nvidia-smi shows 16,303 MiB).

| # | --gpu-memory-utilization | Other flags | --max-model-len | Result |
|---|---|---|---|---|
| 1 | 0.92 (vLLM default, not set by the recipe) | defaults (CUDA graphs on, capture sizes [1,2], 0.06 GiB of graphs) | 131072 | Started: `Available KV cache memory: 6.37 GiB`, `GPU KV cache size: 426,946 tokens, Maximum concurrency for 131,072 tokens per request: 3.26x` |
