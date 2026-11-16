# vLLM v0.33.0 startup attempts for openai/gpt-oss-20b on the RTX 5080 Laptop GPU (2026-11-16)

Common flags: `--device nvidia.com/gpu=all --ipc=host --memory 32g`, `--max-model-len` as listed, `--max-num-seqs 1`, `--kv-cache-dtype fp8`, `--attention-backend TRITON_ATTN`, parsers `openai` / `openai_gptoss`.
In every attempt vLLM logged `Using 'MARLIN' Mxfp4 MoE backend` and `Model loading took 13.8 GiB memory`, and CUDA reported a total of 15.47 GiB (nvidia-smi shows 16,303 MiB).

| # | --gpu-memory-utilization | Other flags | --max-model-len | Result |
|---|---|---|---|---|
| 1 | 0.95 | defaults (max-num-batched-tokens 2048, CUDA graphs on, 0.40 GiB of graphs) | 65536 | Failed: `Available KV cache memory: -0.74 GiB` |
| 2 | 0.97 | `--max-num-batched-tokens 1024`, `cudagraph_capture_sizes` [1,2,4,8] (0.07 GiB of graphs) | 65536 | Failed: `Available KV cache memory: -0.09 GiB` |
| 3 | 0.99 | `--max-num-batched-tokens 512 --enforce-eager` | 65536 | Failed at startup: free memory 15.08 of 15.47 GiB is less than 0.99 (15.32 GiB) |
| 4 | 0.975 | `--max-num-batched-tokens 256 --enforce-eager` | 32768 | KV planned at 1.1 GiB (93,943 tokens), then `torch.OutOfMemoryError` allocating 1.10 GiB with 1.12 GiB free |
| 5 | 0.965 | `--max-num-batched-tokens 256 --enforce-eager` | 65536 | Started: KV 0.94 GiB, 81,504 tokens, 1.24x concurrency at 64k; GPU memory 15,698 MiB in use |
