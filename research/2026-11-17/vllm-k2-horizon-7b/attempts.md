# vLLM v0.33.0 startup attempts for cyankiwi/K2-Horizon-7B-AWQ-INT4 on the RTX 5080 Laptop GPU (2026-11-17)

Common flags: `--device nvidia.com/gpu=all --ipc=host --memory 32g`, `--max-model-len 65536`, `--max-num-seqs 1`, `--kv-cache-dtype fp8`, `--attention-backend TRITON_ATTN`, `--trust-remote-code`, parsers `k2_horizon` / `k2_horizon`, `--default-chat-template-kwargs '{"reasoning_effort":"high","tool_call_format":"xml"}'`.
In both attempts vLLM logged `Model loading took 7.72 GiB memory` and CUDA graph capture took 0.02 GiB.

| # | --gpu-memory-utilization | Other flags | --max-model-len | Result |
|---|---|---|---|---|
| 1 | 0.92 (vLLM default, not set by the recipe) | shipped recipe, CUDA graphs on, first-run torch.compile (10.68 s) | 65536 | Failed: `To serve at least one request with the model's max seq len (65536), (4.5 GiB KV cache is needed, which is larger than the available KV cache memory (4.44 GiB). Based on the available memory, the estimated maximum model length is 64624.` |
| 2 | 0.94 | shipped recipe plus `--gpu-memory-utilization 0.94` (override), CUDA graphs on, compile cache hit (0.39 s) | 65536 | Started: `Available KV cache memory: 6.54 GiB`, `GPU KV cache size: 95,168 tokens, Maximum concurrency for 65,536 tokens per request: 1.45x` |

Attempt 2 gained 2.10 GiB of KV cache memory, far more than the 0.31 GiB that 0.02 of the 15.47 GiB total explains.
The difference lines up with attempt 1 running the first torch.compile and attempt 2 loading it from the `vllm-cache` volume; this cause is not proven.
