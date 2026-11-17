# vLLM v0.33.0 startup attempts for google/gemma-4-E4B-it-qat-w4a16-ct on the RTX 5080 Laptop GPU (2026-11-17)

Common flags: `--device nvidia.com/gpu=all --ipc=host --memory 32g --memory-swap 32g`, `--language-model-only`, `--max-num-seqs 2`, parsers `gemma4` / `gemma4`, `--override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":64}'`.
vLLM forced the `TRITON_ATTN` backend for the heterogeneous head dimensions (256 sliding, 512 full), and the KV cache stayed at the default dtype.
In every attempt vLLM logged `Model loading took 10.1 GiB memory` (10.13 GiB in attempts 1 and 3), and CUDA reported 15.08 GiB free of 15.47 GiB at startup.
Attempts 1 and 2 failed in the same place: the torch.compile (Inductor) autotune of the profile run asked for 5.25 GiB (5,637,144,576 bytes) with about 4.8 GiB free.
That size equals 262,144 vocabulary rows times 10,752 per-layer embedding values (42 layers of 256) times 2 bytes, so it does not depend on the batch size, which attempt 2 confirmed.
Error excerpts: [attempt-1-error.log](attempt-1-error.log), [attempt-2-error.log](attempt-2-error.log).

| # | --gpu-memory-utilization | Other flags | --max-model-len | Result |
|---|---|---|---|---|
| 1 | 0.92 (vLLM default, not passed) | recipe as shipped, torch.compile and CUDA graphs on, max-num-batched-tokens 2048 | 65536 | Failed: `InductorError: RuntimeError: Failed to run autotuning code block: CUDA out of memory. Tried to allocate 5.25 GiB. GPU 0 has a total capacity of 15.47 GiB of which 4.73 GiB is free.` |
| 2 | 0.92 (vLLM default, not passed) | `--max-num-batched-tokens 512` | 65536 | Failed: the same allocation, `Tried to allocate 5.25 GiB. GPU 0 has a total capacity of 15.47 GiB of which 4.80 GiB is free.` |
| 3 | 0.92 (vLLM default, not passed) | `--enforce-eager` in place of `--max-num-batched-tokens 512` (max-num-batched-tokens back to 2048) | 65536 | Started: `Available KV cache memory: 3.67 GiB`, `GPU KV cache size: 204,533 tokens, Maximum concurrency for 65,536 tokens per request: 3.12x` |
