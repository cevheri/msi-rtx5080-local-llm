# K2 Horizon 7B AWQ-INT4 on SGLang v0.5.23 (step 0, M12, M13, M15)

Last verified: 2026-11-17.

Result: the server-side grammar fix holds on K2 Horizon 7B: with `SYNC_TOKEN_IDS_ACROSS_TP=1` and `NCCL_MAX_P2P_NCHANNELS=1` the first grammar request after a fresh 62,257-token prompt took no visible memory and left 543 MiB free, against 34 MiB in measurement A without them (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
Decode is unchanged at 89.5 tok/s, 300 s of load passed, and FP8 and BF16 KV gave the same 20 answers to the M13 set, all correct (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).

Every start used `lmsysorg/sglang:v0.5.23-runtime`, `cyankiwi/K2-Horizon-7B-AWQ-INT4` at `530d5ceccd8ea660ceaa2a9f3a4bcbb68ba44291` from the overlay `$SCRATCH/sglang/measure-b/overlay/k2-horizon-7b/` (temperature 1.0 and top_p 0.95, the vLLM recipe's values), the lane environment (`HF_HUB_OFFLINE=1`, `HF_MODULES_CACHE=/tmp/hf_modules`, `MAX_JOBS=2`, `SYNC_TOKEN_IDS_ACROSS_TP=1`, `NCCL_MAX_P2P_NCHANNELS=1`, `FLASHINFER_NO_DOWNLOAD=1`), `--memory 32g --memory-swap 32g`, and the hands-on a2 arguments: fraction 0.90, `--max-running-requests 1`, `--disable-prefill-cuda-graph`, `--trust-remote-code`, FP8 KV, parsers `k2_horizon`, `--default-chat-template-kwargs '{"reasoning_effort":"high","tool_call_format":"xml"}'`, context 65536 ([cmds/b0-a2.sh](cmds/b0-a2.sh)).

## Step 0 and M12 at the shipping settings (start b0-a2)

| Item | Value |
|---|---|
| Readiness | `/v1/models` and `/server_info` 200 at 36.05 s, `/ready` 200 at 37.55 s, ready line 37.1 s |
| Log lines | `type=K2HorizonForCausalLM, quant=compressed-tensors ... mem usage=7.89 GB`, FP8 pool 82,397 tokens (2.83 + 2.83 GB), `Memory pool end. avail mem=1.47 GB`, decode graph 1.21 s and 0.06 GB, `available_gpu_mem=0.98 GB`, FlashInfer attention, `k2_horizon` parsers auto-detected, content format `string` |
| Sampling line | `Using default chat sampling params from model generation config: {'temperature': 1.0, 'top_p': 0.95}` |
| `memory_usage` at ready | weight 7.887, kvcache 5.658, `startup_available` 0.977 GiB, `token_capacity` 82,397 |
| Free right after ready | 865 MiB (measurement A without the variables: 962 MiB) |
| Fresh 62,257-token prompt (95.0% of 65,536), no `/flush_cache` before it | 33.25 s, free 865 to 543 MiB |
| Hermes title request, the first grammar request | 543 to 543 MiB, HTTP 200, `{"title": "Adding Two Numbers in Python"}` parses after 1,087 characters of reasoning |
| Forced `tool_choice` | 543 to 543 MiB, `get_weather {"city": "Paris"}`, `finish_reason` `tool_calls` |
| Decode, repository probe after `/flush_cache` | 89.5 tok/s median (89.5, 89.6, 89.0 over 1,500 tokens); hands-on 89.9 tok/s |
| First token | 2.198 s for 8,085 tokens, 9.249 s for 30,261 tokens; repeats 0.039 and 0.057 s; hands-on 2.426 and 9.142 s |
| M12 sustained load | 52 requests in 300 s (the phase ended after 310 s, with the last request), prompts up to 13,101 tokens, no error, decode 79.0 to 91.3 tok/s (median 87.55, halves 87.65 and 87.55) |
| Second fresh 62,253-token prompt after the load | 33.37 s, 533 MiB free before and after |
| Peaks | 15,311 MiB of GPU memory, 152.0 W in 1 s samples, 72 C; non-reclaimable host memory 4.97 GiB (anon 4.77, shmem 0.09, kernel 0.11); cgroup peak 13.28 GiB with page cache |

All figures are measured on the MSI Raider 18 Max HX, SGLang v0.5.23 ([logs/b0-a2.log](logs/b0-a2.log), [shapes/b0-a2-ready.json](shapes/b0-a2-ready.json), [shapes/b0-a2-seq.jsonl](shapes/b0-a2-seq.jsonl), [shapes/b0-a2-probe.json](shapes/b0-a2-probe.json), [shapes/b0-a2-sustained.json](shapes/b0-a2-sustained.json), [logs/b0-a2-trace.csv](logs/b0-a2-trace.csv)).
The two variables allocated the grammar path's NCCL buffer before ready, about 97 MiB of it (962 against 865 MiB free after ready), and the first grammar request then needed nothing more, where measurement A's order 1 lost 602 MiB at that request (inferred from the readings).

Smoke and thinking ([shapes/b0-a2-api.json](shapes/b0-a2-api.json)), all as the hands-on lane saw (measured on the MSI Raider 18 Max HX, SGLang v0.5.23):
- Plain `OK` streamed and not, a tool call with its second turn naming 18 C (streamed in one fragment), `json_schema` `{"capital": "Paris"}`, and a forced call `get_weather {"city": "London"}`.
- Thinking, type always: every request reasoned, from the default to every `reasoning_effort` (`none` gave 153 characters) and `enable_thinking: false` (217 characters), and every answer was `391`.
- A replayed assistant turn with only `reasoning` got HTTP 400 "Assistant message is missing a thinking field.", and the same turn with `reasoning_content` got 200.

## M13: FP8 against BF16 KV (start m13-bf16)

The BF16 start is b0-a2 without `--kv-cache-dtype` (BF16 is the default) at a 32k context, because at fraction 0.90 the BF16 pool holds 41,198 tokens, half of FP8's 82,397 ([cmds/m13-bf16.sh](cmds/m13-bf16.sh)).
The set is 8 arithmetic, 6 Python-output and 6 tool prompts at temperature 0, each with up to 4,096 tokens and K2's own reasoning ([../scripts/seq_b.py](../../sglang-runtime/measure-b/scripts/seq_b.py), `M13`).

| KV dtype | Correct | Same final answer as the other | Same completion length | Decode (probe) | First token 8,085 / 30,261 tokens |
|---|---|---|---|---|---|
| FP8 (b0-a2) | 20 of 20 | 20 of 20 | 15 of 20 | 89.5 tok/s | 2.198 / 9.249 s |
| BF16 (m13-bf16) | 20 of 20 | 20 of 20 | 15 of 20 | 91.0 tok/s | 2.434 / 8.924 s |

Both rows are measured on the MSI Raider 18 Max HX, SGLang v0.5.23 ([shapes/b0-a2-m13.json](shapes/b0-a2-m13.json), [shapes/m13-bf16-m13.json](shapes/m13-bf16-m13.json), [shapes/m13-bf16-probe.json](shapes/m13-bf16-probe.json)).
- The final `content` strings were identical for all 20 prompts; five reasoning traces differed in length (for example 177 against 130 tokens), so FP8 KV with unit scales changes some token choices but not the answers on this set (measured).
- The hands-on lane's one wrong answer (390 for 17 x 23) came from sampling at temperature 1.0 with the checkpoint's top_p of 1.0; here, at temperature 0, FP8 gave 391 (inferred from the two runs).
- So FP8 KV stays the recipe's choice: it doubles the pool, which 64k needs, at no measured cost in this set (estimate; 20 prompts are a small sample).

## Proposed recipe values

- `[needs] gpu_mib = 15199` (the 15,311 MiB peak minus the desktop's 112 MiB), `ram_gib = 5` (4.97 GiB non-reclaimable, rounded up), `[container] memory_gib = 32` as on vLLM (estimate from the measured peaks).
- Draft: `$SCRATCH/sglang/overrides/sglang/k2-horizon-7b.toml`; M15 is in the lane README's M15 table.

## Files

- [cmds/](cmds/): the `docker run` files; they read `$HOME` and `$SCRATCH`.
- `logs/<run>.*` and `shapes/<run>-*`: cut and full logs, 1 s traces, process memory, readiness and `/server_info`, sequence records, probe, api probe, sustained load, M13 answers and summaries.
