# gpt-oss-20b on SGLang v0.5.23 (step 0, M10, M12, M15, LD19)

Last verified: 2026-11-17.

Result: gpt-oss-20b serves 128k on SGLang v0.5.23 with Paged Experts at 24 of 32 experts resident, and the server-side grammar fix holds: after a fresh 124,463-token prompt the first grammar request took no visible memory and left 989 MiB free (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
`TIKTOKEN_ENCODINGS_BASE` removes the Harmony vocabulary download: the start-up step took 1 s instead of measurement A's 52 to 63 s, and the server reached ready in 54.9 s (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
The pinned expert store is shared memory, 19.19 GiB of it, so the container's non-reclaimable memory is 24.35 GiB, not the 13.94 GiB of anonymous memory the hands-on lane read (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).

Every start used `lmsysorg/sglang:v0.5.23-runtime`, the overlay `$SCRATCH/sglang/measure-b/overlay/gpt-oss-20b/` (the snapshot's `generation_config.json` with temperature 1.0 and top_p 1.0, the vLLM recipe's values) mounted read-only at `/llmctl/model`, the lane environment (`HF_HUB_OFFLINE=1`, `HF_MODULES_CACHE=/tmp/hf_modules`, `MAX_JOBS=2`, `SYNC_TOKEN_IDS_ACROSS_TP=1`, `NCCL_MAX_P2P_NCHANNELS=1`, `FLASHINFER_NO_DOWNLOAD=1`, and `TIKTOKEN_ENCODINGS_BASE=/opt/sglang/lib/python3.12/site-packages/tml_renderers/data` unless a row says otherwise), `--memory 32g --memory-swap 32g`, and the hands-on a4 arguments: fraction 0.90, `--max-running-requests 1`, FP8 KV, `--swa-full-tokens-ratio 0.1`, `--chunked-prefill-size 1024`, `--disable-prefill-cuda-graph`, `--moe-runner-backend marlin`, `--enable-paged-experts --paged-experts-num-resident 24`, parsers `gpt-oss` ([cmds/b0-a4.sh](cmds/b0-a4.sh)).
The enforced GPU power limit read 145 W ([../lane-logs/llmctl-down.txt](../../sglang-runtime/measure-b/lane-logs/llmctl-down.txt)).

## Step 0: the grammar fix at the working settings (start b0-a4)

| Item | Value |
|---|---|
| Ready line after `docker run` | 54.9 s; `/v1/models` and `/server_info` answered 200 at 54.05 s, `/ready` at 55.55 s |
| Start-up step ("Waiting for application startup" to "Application startup complete") | 1 s (18:29:19 to 18:29:20), against 52 and 63 s in measurement A without `TIKTOKEN_ENCODINGS_BASE` |
| `memory_usage` at ready | weight 11.455, kvcache 2.121, `startup_available` 1.447 GiB, `token_capacity` 168,456, `token_capacity_swa` 16,845 |
| Log lines | `Load weight end. elapsed=22.50 s ... mem usage=11.46 GB`, full FP8 pool 168,456 tokens (0.96 + 0.96 GB), SWA pool 16,845 tokens (0.10 + 0.10 GB), `Memory pool end. avail mem=1.55 GB`, decode graph 1.72 s and 0.06 GB, `available_gpu_mem=1.45 GB` |
| Sampling line | `Using default chat sampling params from model generation config: {'temperature': 1.0, 'top_p': 1.0}` |
| Free right after ready | 1,329 MiB (measurement A, without the two NCCL variables: 1,406 MiB) |
| Fresh 124,463-token prompt (94.96% of 131,072), no `/flush_cache` before it | 43.95 s, free 1,329 to 989 MiB |
| Hermes title request (`json_schema`, strict), the first grammar request | 989 to 989 MiB, HTTP 200, `{ "title": "Adding Two Numbers in Python"}` parses |
| Forced `tool_choice` | 989 to 989 MiB, HTTP 200, `finish_reason` `stop`, no `tool_calls`, empty content (#41747) |
| Server after these requests | up; every later request answered |

All figures are measured on the MSI Raider 18 Max HX, SGLang v0.5.23 ([logs/b0-a4.log](logs/b0-a4.log), [shapes/b0-a4-ready.json](shapes/b0-a4-ready.json), [shapes/b0-a4-seq.jsonl](shapes/b0-a4-seq.jsonl)).
The two NCCL variables moved the grammar path's allocation before ready, 77 MiB of it (1,406 against 1,329 MiB free after ready), as measurement A saw on Qwen3.5-9B (inferred from the two readings).

Decode by the repository probe after `/flush_cache` ([shapes/b0-a4-probe.json](shapes/b0-a4-probe.json)): 125.5 tok/s median (126.0, 123.8 and 125.5 tok/s over 1,500-token answers), and 1.486 s to the first token for 8,058 tokens and 4.903 s for 30,003 tokens, against 125.5 tok/s, 1.452 s and 4.893 s in the hands-on a4 run, so the variables cost no speed (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).

Smoke requests ([shapes/b0-a4-api.json](shapes/b0-a4-api.json)), all HTTP 200 (measured on the MSI Raider 18 Max HX, SGLang v0.5.23):
- A plain reply `OK`, streamed and not, with `reasoning_content` and a `[DONE]` marker; a tool call `get_weather {"city": "Paris"}` with `finish_reason` `tool_calls`, streamed in one fragment, and the second turn's answer naming 18 C.
- Every `reasoning_effort` from `none` to `xhigh` answered `391`, and `none` still reasoned (58 characters).
- The api probe's forced `tool_choice` came back as raw Harmony text in `content` with `finish_reason` `stop`, and its `json_schema` request as `{"capital":"Paris"}`.
- `"tools": []`, `tool_choice` `auto` without tools, `prompt_cache_key` and a replayed `reasoning` field all got 200; `max_tokens` 262,144 got HTTP 400.

M12, sustained load at these settings: 74 requests in 301 s with prompts up to 13,820 tokens, no error, decode 118.8 to 134.9 tok/s (median 127.5, halves 128.9 and 125.8), then a second fresh 124,462-token prompt in 44.70 s with 1,059 MiB free before and after; peaks 14,854 MiB of GPU memory, 145.3 W and 73 C (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, [shapes/b0-a4-sustained.json](shapes/b0-a4-sustained.json), [logs/b0-a4-trace.csv](logs/b0-a4-trace.csv)).

## LD19: what the 32 GiB cap holds

The 1 s trace splits the cgroup's memory ([logs/b0-a4-trace.csv](logs/b0-a4-trace.csv), [logs/b0-a4.procmem.txt](logs/b0-a4.procmem.txt)); all figures measured on the MSI Raider 18 Max HX, SGLang v0.5.23:

| Moment | anon | shmem | file (page cache, includes shmem) | memory.current |
|---|---|---|---|---|
| 28.8 s, weights loading | 13.78 GiB | 0.01 GiB | 0.58 GiB | 14.48 GiB |
| 40.2 s, the expert store moving into shared memory | 10.68 GiB | 6.60 GiB | 19.77 GiB | 30.59 GiB |
| 48.5 s | 4.23 GiB | 19.16 GiB | 27.01 GiB | 31.43 GiB |
| At ready (procmem) | 4.78 GiB | 19.19 GiB | 27.02 GiB | 32.00 GiB |
| After the 124,463-token prompt | 4.96 GiB | 19.19 GiB | 26.85 GiB | 32.00 GiB |

- The scheduler process holds the store as shared memory: `RssShmem` 20,184,660 kB (19.25 GiB), `VmPin` 0 kB and `VmLck` 16 kB, so neither counter shows the page-locking (measured).
- Non-reclaimable memory (anon plus shmem plus kernel, since the container has no swap) peaked at 24.35 GiB: anon 4.97, shmem 19.19 and kernel 0.19 GiB; the rest of the 32 GiB was page cache, which the kernel squeezed (measured).
- The hands-on lane's 13.94 GiB "host anonymous memory" was the loading peak (14.11 GiB here), before the store moved into shared memory; measurement A's 27.1 GiB `file` held the 19.19 GiB of shared memory (measured; the reading of the earlier figures is inferred).
- The store is 19.25 GiB at every K from 16 to 28 (M10 below), so it holds all 32 experts of all 24 layers whatever K is (measured).
- Why 19.25 GiB for about 10.45 GiB of expert weights: the GPU weights fall by 0.327 GiB per expert slot (M10), so the 32 experts of one layer take about 446 MiB in the layer's GPU layout, which the store copies ("Holds all E experts of each paged tensor on the host, in the layer's GPU layout.", `layers/moe/paged_experts/store.py`), while the measured store is 821 MiB per layer (19.25 GiB over 24 layers); one pinned `torch.empty(..., pin_memory=True)` per paged tensor (line 30) rounded up to a power of two by PyTorch's pinned host allocator would give 512 plus 256 MiB for the two weight tensors plus the smaller scale and bias tensors, which fits (inferred from the source and the arithmetic; not proven).
- Proposed sizing: `[needs] ram_gib = 25` (the 24.35 GiB non-reclaimable peak, rounded up) and `[container] memory_gib = 28`, leaving about 3.6 GiB for page cache and loading transients (estimate), which start ld19-mem28 checked.

Start ld19-mem28 repeated b0-a4 under `--memory 28g --memory-swap 28g` ([cmds/ld19-mem28.sh](cmds/ld19-mem28.sh)), with the margin protocol and M12 (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, `shapes/ld19-mem28-*`, [logs/ld19-mem28-trace.csv](logs/ld19-mem28-trace.csv)):
- Ready line at 53.3 s, `token_capacity` 168,439, 1,328 MiB free after ready; the cgroup sat at 28.00 GiB with 4.77 GiB anon, 19.19 GiB shmem and the page cache squeezed to about 3.8 GiB.
- A fresh 124,458-token prompt took 44.10 s (43.95 s under 32 GiB) and left 988 MiB; the title and forced tool requests passed at 988 MiB.
- Decode 125.6 tok/s (125.2, 126.2, 125.6), first tokens 1.480 and 4.871 s; 73 requests in 300 s without error at 117.2 to 131.1 tok/s (median 127.1); a second 124,458-token prompt in 44.63 s.
- Non-reclaimable memory peaked at 24.33 GiB; `memory.events` read `max 5200`, `oom 0`, `oom_kill 0` during the run (the cap forced page-cache reclaim, nothing was killed), and the container ended with `oom=false`.
- So `memory_gib = 28` holds, with about 3.7 GiB above the non-reclaimable peak (measured; that it holds under other host load is an estimate).

## M10: Paged Experts sweep (K resident experts of 32, fraction 0.90, Marlin, FP8 KV, 128k context)

| K | Weights on the GPU | `token_capacity` (SWA) | `startup_available` | Free after ready | Decode (probe median) | First token, 8,058 tokens | First token, 30,003 tokens | Pinned store | Ready line |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 8.842 GiB | 376,049 (37,604) | 1.389 GiB | 1,269 MiB | 87.9 tok/s | 2.095 s | 5.647 s | 19.25 GiB | 49.4 s |
| 20 | 10.209 GiB | 267,443 (26,744) | 1.408 GiB | 1,289 MiB | 112.3 tok/s | 1.907 s | 5.179 s | 19.25 GiB | 51.9 s |
| 24 | 11.455 GiB | 168,456 (16,845) | 1.447 GiB | 1,329 MiB | 125.5 tok/s | 1.486 s | 4.903 s | 19.25 GiB | 54.9 s |
| 25 | 11.779 GiB | 142,701 (14,270) | 1.445 GiB | 1,327 MiB | 130.0 tok/s | 1.746 s | 4.835 s | 19.25 GiB | 54.6 s |
| 28 | 12.787 GiB | 62,643 (6,264) | 1.494 GiB | 1,377 MiB | 137.6 tok/s | 1.625 s | 4.604 s | 19.25 GiB | 54.6 s |

All rows are measured on the MSI Raider 18 Max HX, SGLang v0.5.23, with the repository probe after `/flush_cache` ([cmds/m10-k16.sh](cmds/m10-k16.sh) and the K 20, K 25 and K 28 files, `shapes/m10-k*-probe.json`, `shapes/m10-k*-summary.json`, `logs/m10-k*.procmem.txt`); K 24 is start b0-a4.
- Each resident expert slot costs 0.33 GiB of GPU weights (8.842 to 12.787 GiB over 12 slots), which comes out of the KV pool one for one, about 26,000 FP8 tokens per slot (measured; the per-slot cost is arithmetic on the measured weights).
- K 28 does not hold 128k: SGLang warned `KV pool holds 62643 tokens, fewer than --context-length 131072` and started anyway (measured).
- K 25, the largest K that the per-slot arithmetic said would still hold 128k, holds 142,701 tokens, 11,629 above the context, and decodes 3.6% faster than K 24 (measured).
- Decode rises with K, 87.9 to 137.6 tok/s, because fewer experts are paged in per token; the three runs of each K agree within 4.3 tok/s (measured).
- Recommendation for the recipe: K 24: it holds 128k with 37,384 tokens to spare and is the K that went through the margin protocol, 300 s of load and the 28 GiB cap below, at 125.5 tok/s against 166.2 tok/s on llama.cpp b12407 at 128k (measured; the llama.cpp figure is from `llamacpp/benchmarking.md`).
- K 25 is the measured alternative for 4.5 tok/s more with a quarter of the spare pool; it would need the same protocol before it ships (estimate of the trade-off from the table).

## gpt-oss-20b with no network (step 0, and M15 for this recipe)

| Start | `TIKTOKEN_ENCODINGS_BASE` | Cache volume | `/ready` 503 | `/ready` 200 | Served name listed | Chat request | Download or fetch lines |
|---|---|---|---|---|---|---|---|
| b0-netnone-tik | set | new, empty `sglang-measure-cold` | 243.73 s | 253.58 s | 253.74 s | HTTP 200, `OK` | none |
| b0-netnone-notik | not set | new, empty `sglang-measure-cold2` | 244.79 s | 254.62 s | 254.80 s | HTTP 200, `OK` | one: the Responses API line below |

Both rows are measured on the MSI Raider 18 Max HX, SGLang v0.5.23, `--network none`, no published port, polled with `docker exec` every second ([shapes/b0-netnone-tik-exec-poll.txt](shapes/b0-netnone-tik-exec-poll.txt), [shapes/b0-netnone-notik-exec-poll.txt](shapes/b0-netnone-notik-exec-poll.txt), `logs/b0-netnone-*.network-lines.txt`).
- Both cold starts spent about 185 s capturing the batch-1 decode graph (`elapsed=185.09 s` and `185.31 s`) against 1.72 s warm, because the empty volume had no compiled kernels; the volume held 249 files and 23 MB afterwards (`jit` 17 MB, `triton` 5.4 MB) and was removed (measured).
- Without the variable and without a network, the start logged `OpenAI Responses API (/v1/responses) disabled: OpenAIServingResponses init failed (HarmonyError: error downloading or loading vocab file: failed to download or load vocab file)` in the same second as "Waiting for application startup", and chat completions still worked (measured).
- So the variable is needed for `/v1/responses` and to avoid the 52 to 63 s download at every start on a network, and an offline start reaches ready either way (measured here and in measurement A).
- The searched patterns (`download`, `fetch`, `urm.nvidia`, `huggingface.co`, `openaipublic`, `blob.core`, `harmony`, `tiktoken`, `resolve`, `connect`) matched only `server_args`, Gloo's `Unable to resolve hostname` warning, the prefill-graph line and, without the variable, the Responses API line (measured).

The shipped vLLM recipe without a network is in [../vllm-gpt-oss-20b/](../../vllm-gpt-oss-20b-offline/measure-b.md): it reached ready in 33.3 s with no change.

## Files

- [cmds/](cmds/): one `docker run` file per start; they read `$HOME` and `$SCRATCH`.
- `logs/<run>.*`: the cut log, the full log, the 1 s trace (`-trace.csv`, with anon, file, shmem and kernel), the process memory at ready and at the end (`.procmem.txt`), the state and the metadata; for the network-none starts also the volume listing and the matched network lines.
- `shapes/<run>-*`: readiness and memory at ready (`-ready.json`, full `/server_info` in `-ready-server-info.json`), the sequence records (`-seq.jsonl`), the repository probe (`-probe.json`), the api probe (`-api.json`), the sustained load (`-sustained.json`) and a summary (`-summary.json`).
