# Qwen3.5-9B NVFP4 on SGLang v0.5.23 (M12, M15, the margins at the shipped settings)

Last verified: 2026-11-17.

Result: at the shipped settings, with the overlay and the lane environment, Qwen3.5-9B held a fresh 124,499-token prompt, both grammar requests and 300 s of load with at least 537 MiB free, and decoded 93.7 tok/s (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).

The start used `lmsysorg/sglang:v0.5.23-runtime`, `RedHatAI/Qwen3.5-9B-NVFP4` at `a4df85a20fb7e54740f564f15af0c2f5d9c6616c` from the overlay `$SCRATCH/sglang/measure-b/overlay/qwen3.5-9b/` (temperature 0.6, top_p 0.95, top_k 20, the vLLM recipe's values), the lane environment (`HF_HUB_OFFLINE=1`, `HF_MODULES_CACHE=/tmp/hf_modules`, `MAX_JOBS=2`, `SYNC_TOKEN_IDS_ACROSS_TP=1`, `NCCL_MAX_P2P_NCHANNELS=1`, `FLASHINFER_NO_DOWNLOAD=1`), `--memory 32g --memory-swap 32g`, and the hands-on a3 arguments: fraction 0.88, `--max-running-requests 1`, `--max-mamba-cache-size 8`, `--disable-prefill-cuda-graph`, parsers `qwen3_coder` and `qwen3`, `--default-chat-template-kwargs '{"enable_thinking":false}'`, context 131072 ([cmds/m12-a3.sh](cmds/m12-a3.sh)).
Measurement A already measured the grammar fix on this model (A §M6 step 4); this start adds the overlay together with both variables, the probe and M12.

## Start m12-a3

| Item | Value |
|---|---|
| Readiness | `/v1/models` and `/server_info` 200 at 27.05 s, `/ready` 200 at 29.55 s, ready line 28.7 s |
| Log lines | `type=Qwen3_5ForConditionalGeneration, quant=compressed-tensors ... mem usage=8.39 GB`, `Mamba Cache is allocated. max_mamba_cache_size: 8, conv_state size: 0.01GB, ssm_state size: 0.42GB`, BF16 pool 140,981 tokens (2.15 + 2.15 GB), `Memory pool end. avail mem=1.90 GB`, decode graph 2.38 s and 0.10 GB, `available_gpu_mem=1.35 GB`, FlashInfer attention |
| Sampling line | `Using default chat sampling params from model generation config: {'temperature': 0.6, 'top_k': 20, 'top_p': 0.95}` |
| `memory_usage` at ready | weight 8.391, kvcache 4.302, `startup_available` 1.35 GiB, `token_capacity` 140,981 |
| Free right after ready | 911 MiB (measurement A with both variables and no overlay: 911 MiB) |
| Fresh 124,499-token prompt (94.99% of 131,072), no `/flush_cache` before it | 24.38 s, free 911 to 545 MiB |
| Hermes title request, the first grammar request | 545 to 545 MiB, `{"title": "Python Addition Basics"}` parses, no reasoning |
| Forced `tool_choice` | 545 to 545 MiB, `get_weather {"city": "Paris"}`, `finish_reason` `tool_calls` |
| Decode, repository probe after `/flush_cache` | 93.7 tok/s median (93.7 in each run, 664 to 854 tokens); hands-on 92.8 tok/s |
| First token | 0.572 s for 8,591 tokens, 2.398 s for 32,153 tokens; repeats 0.063 and 0.052 s; hands-on a3 0.566 and 2.394 s |
| M12 sustained load | 66 requests in 300 s, prompts up to 13,448 tokens, no error, decode 88.0 to 93.7 tok/s (median 92.0 in both halves) |
| Second fresh 124,503-token prompt after the load | 24.48 s, 539 to 537 MiB free |
| Peaks | 15,305 MiB of GPU memory, 161.0 W in a 1 s sample, 70 C; non-reclaimable host memory 5.73 GiB (anon 5.51, shmem 0.10, kernel 0.12); cgroup peak 14.83 GiB with page cache |

All figures are measured on the MSI Raider 18 Max HX, SGLang v0.5.23 ([logs/m12-a3.log](logs/m12-a3.log), [shapes/m12-a3-ready.json](shapes/m12-a3-ready.json), [shapes/m12-a3-seq.jsonl](shapes/m12-a3-seq.jsonl), [shapes/m12-a3-probe.json](shapes/m12-a3-probe.json), [shapes/m12-a3-sustained.json](shapes/m12-a3-sustained.json), [logs/m12-a3-trace.csv](logs/m12-a3-trace.csv)).
The 161.0 W sample sits above the 145 W limit for one second, as the hands-on lane's 150.4 W did; the decode figures show no throttling (measured; that it is a sampling artifact of `power.draw` is inferred).

Smoke and thinking ([shapes/m12-a3-api.json](shapes/m12-a3-api.json)), as the hands-on lane saw (measured on the MSI Raider 18 Max HX, SGLang v0.5.23):
- Plain `OK` streamed and not; a tool call `get_weather {"city": "Paris"}`, streamed in 4 fragments, and its second turn naming 18 C; `json_schema` `{"capital": "Paris"}`; a forced call answered with `tool_calls`.
- Thinking, toggle type with the server default off: no field, `enable_thinking: false` and `reasoning_effort: "none"` gave `391` without reasoning; `enable_thinking: true` and the efforts `low` and `high` reasoned (569 to 686 characters) and answered `\n\n391`.
- `"tools": []`, `tool_choice` `auto` without tools, `prompt_cache_key`, a wrong model name and a replayed `reasoning` field got 200; `max_tokens` 262,144 got HTTP 400.

## Proposed recipe values

- `[needs] gpu_mib = 15193` (the 15,305 MiB peak minus the desktop's 112 MiB), `ram_gib = 6` (anon 5.51 plus shmem 0.10 GiB, rounded up), `[container] memory_gib = 32` as on vLLM (estimate from the measured peaks).
- Draft: `$SCRATCH/sglang/overrides/sglang/qwen3.5-9b.toml`; M15 is in the lane README's M15 table.

## Files

- [cmds/](cmds/): the `docker run` files; they read `$HOME` and `$SCRATCH`.
- `logs/<run>.*` and `shapes/<run>-*`: cut and full logs, 1 s traces, process memory, readiness and `/server_info`, sequence records, probe, api probe, sustained load and summaries.
