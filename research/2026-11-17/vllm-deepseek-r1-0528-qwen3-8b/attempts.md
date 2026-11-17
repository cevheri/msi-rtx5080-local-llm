# vLLM v0.33.0 startup attempts for deepseek-ai/DeepSeek-R1-0528-Qwen3-8B on the RTX 5080 Laptop GPU (2026-11-17)

Common flags: `--device nvidia.com/gpu=all --ipc=host --memory 32g`, `--max-model-len 65536`, `--quantization fp8_per_tensor`, `--max-num-seqs 1`, `--kv-cache-dtype fp8`, `--attention-backend TRITON_ATTN`, `--reasoning-parser deepseek_r1`, `--override-generation-config '{"temperature":0.6,"top_p":0.95}'`.
In both attempts vLLM logged `Model loading took 9.12 GiB memory`, CUDA graphs were on and the CUDA graph pool was estimated at 0.02 GiB.

| # | --gpu-memory-utilization | Other flags | --max-model-len | Result |
|---|---|---|---|---|
| 1 | 0.92 (vLLM default, not set by the recipe) | shipped recipe, first-run torch.compile (17.92 s in total) | 65536 | Failed: `Available KV cache memory: 3.83 GiB`, then `ValueError: To serve at least one request with the model's max seq len (65536), (4.5 GiB KV cache is needed, which is larger than the available KV cache memory (3.83 GiB). Based on the available memory, the estimated maximum model length is 55744.` |
| 2 | 0.965 | shipped recipe plus `--gpu-memory-utilization 0.965` (override), compile cache hit (0.41 s) | 65536 | Started: `Available KV cache memory: 5.5 GiB`, `GPU KV cache size: 80,064 tokens, Maximum concurrency for 65,536 tokens per request: 1.22x` |

Attempt 1 lacked 0.67 GiB of KV cache memory, about 0.044 of the 15.47 GiB that CUDA reports, so attempt 2 raised the utilization from 0.92 to 0.965, the highest value that started on this laptop in the gpt-oss-20b record (0.975 ran out of memory there).
Attempt 2 gained 1.67 GiB of KV cache memory, more than the 0.70 GiB that 0.045 of 15.47 GiB explains.
As in the k2-horizon-7b record, the difference lines up with attempt 1 running the first torch.compile and attempt 2 loading it from the `vllm-cache` volume; this cause is not proven.
