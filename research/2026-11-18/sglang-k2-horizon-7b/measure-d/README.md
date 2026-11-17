# K2 Horizon 7B AWQ-INT4 on SGLang v0.5.23 at fraction 0.88 (measurement D)

Last verified: 2026-11-18.

Result: at `--mem-fraction-static 0.88` K2 Horizon 7B passes the post-ready gate that Qwen3.6's 486 MiB drop set: the FP8 pool holds 77,988 tokens for the 65,536 context, 1,152 MiB were free right after ready and also after D8's `POST /flush_cache`, and the largest drop over the whole protocol was 330 MiB (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
Decode is 89.6 tok/s against B's 89.5 at fraction 0.90, 300 s of load passed without error, and the draft `$SCRATCH/sglang/overrides/sglang/k2-horizon-7b.toml` now carries fraction 0.88, `gpu_mib = 14908`, `ram_gib = 5` and this record (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).

The start used `lmsysorg/sglang:v0.5.23-runtime`, `cyankiwi/K2-Horizon-7B-AWQ-INT4` at `530d5ceccd8ea660ceaa2a9f3a4bcbb68ba44291` from the overlay `$SCRATCH/sglang/measure-d/overlay/k2-horizon-7b/` (rebuilt with B's `make_overlay.py`; the same 16 links and the same `generation_config.json` with temperature 1.0 and top_p 0.95 as B's), the volume `sglang-research-cache`, `--memory 32g --memory-swap 32g`, and all eight variables of the runtime file: `HF_HUB_OFFLINE=1`, `HF_MODULES_CACHE=/tmp/hf_modules`, `MAX_JOBS=2`, `FLASHINFER_NO_DOWNLOAD=1`, `SYNC_TOKEN_IDS_ACROSS_TP=1`, `NCCL_MAX_P2P_NCHANNELS=1`, `TIKTOKEN_ENCODINGS_BASE=/opt/sglang/lib/python3.12/site-packages/tml_renderers/data` and `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1` ([cmds/d1-f088.sh](cmds/d1-f088.sh)).
The arguments are B's draft with only the fraction changed: fraction 0.88, `--max-running-requests 1`, `--disable-prefill-cuda-graph`, `--trust-remote-code`, FP8 KV, parsers `k2_horizon`, `--default-chat-template-kwargs '{"reasoning_effort":"high","tool_call_format":"xml"}'`, context 65536.
The enforced GPU power limit read 140 W before the start (B's start: 145 W), and the desktop held 112 MiB of GPU memory ([logs/d1-f088.meta](logs/d1-f088.meta)).

## The start d1-f088 against B's b0-a2

| Item | D d1-f088, fraction 0.88 | B b0-a2, fraction 0.90 |
|---|---|---|
| Readiness | `/v1/models` and `/server_info` 200 at 35.05 s, `/ready` 200 at 36.55 s, ready line 36.0 s | 36.05, 37.55 and 37.1 s |
| Log lines | FP8 pool 77,988 tokens (2.68 + 2.68 GB), `Memory pool end. avail mem=1.76 GB`, decode graph 1.03 s and 0.06 GB, `available_gpu_mem=1.26 GB` | 82,397 tokens (2.83 + 2.83 GB), 1.47 GB, 1.21 s and 0.06 GB, 0.98 GB |
| Sampling line | `Using default chat sampling params from model generation config: {'temperature': 1.0, 'top_p': 0.95}`, equal to the draft's `[server] sampling` | the same |
| `memory_usage` at ready | weight 7.887, kvcache 5.355, `startup_available` 1.258 GiB, `token_capacity` 77,988 | 7.887, 5.658, 0.977 GiB, 82,397 |
| Free right after ready | 1,152 MiB | 865 MiB |
| Free after D8's `POST /flush_cache` (200 in 0.002 s), at once and 2 s later | 1,152 and 1,152 MiB | not read |
| Fresh 62,254-token prompt (95.0% of 65,536), right after that flush | 33.22 s, free 1,152 to 828 MiB | 62,257 tokens, no flush before it: 33.25 s, 865 to 543 MiB |
| Hermes title request (`json_schema`, strict), the first grammar request | 828 to 828 MiB, HTTP 200, `{"title": "Adding Two Numbers in Python"}` parses after 731 characters of reasoning | 543 to 543 MiB, the same title after 1,087 characters |
| Forced `tool_choice` | 828 to 828 MiB, `get_weather {"city": "Paris"}`, `finish_reason` `tool_calls` | 543 to 543 MiB, the same call |
| Decode, repository probe after `/flush_cache` | 89.6 tok/s median (89.9, 89.6 and 89.1 over runs of 1,500, 1,286 and 1,500 tokens) | 89.5 tok/s (89.5, 89.6, 89.0 over 1,500 tokens) |
| First token | 2.196 s for 8,085 tokens, 9.233 s for 30,261 tokens; repeats 0.031 and 0.056 s | 2.198 and 9.249 s; 0.039 and 0.057 s |
| M12 sustained load | 51 requests in 300 s (the phase ended after 302.9 s, with the last request), prompts up to 13,101 tokens, no error, decode 78.3 to 89.7 tok/s (median 86.5, halves 88.3 and 86.4) | 52 requests, 79.0 to 91.3 tok/s (median 87.55, halves 87.65 and 87.55) |
| Second fresh 62,257-token prompt after the load | 33.64 s, 822 MiB free before and after | 62,253 tokens, 33.37 s, 533 MiB |
| Lowest free after ready, drop | 822 MiB, 330 MiB from either ready reading | 532 MiB, 333 MiB |
| Peaks | 15,020 MiB of GPU memory, 147.0 W in 1 s samples, 74 C; anon plus shmem 4.92 GiB (anon 4.84, shmem 0.09); non-reclaimable 5.04 GiB with kernel 0.11; cgroup `memory.peak` 13.53 GiB with page cache | 15,311 MiB, 152.0 W, 72 C; 4.86 GiB; 4.97 GiB; 13.28 GiB |
| Container at the end | `running exit=0 oom=false` before removal | the same |

All figures are measured on the MSI Raider 18 Max HX, SGLang v0.5.23 ([logs/d1-f088.log](logs/d1-f088.log), [shapes/d1-f088-ready.json](shapes/d1-f088-ready.json), [shapes/d1-f088-seq.jsonl](shapes/d1-f088-seq.jsonl), [shapes/d1-f088-probe.json](shapes/d1-f088-probe.json), [shapes/d1-f088-sustained.json](shapes/d1-f088-sustained.json), [logs/d1-f088-trace.csv](logs/d1-f088-trace.csv), [shapes/d1-f088-summary.json](shapes/d1-f088-summary.json)); B's column is from `$SCRATCH/sglang/measure-b/sglang-k2-horizon-7b/`.
- The pool lost 4,409 tokens (5.4%) and kept 12,452 tokens (19%) above the 65,536 context, so D8's `token_capacity` check passes (measured; the shares are arithmetic).
- Lowering the fraction by 0.02 freed 287 MiB at ready (865 to 1,152 MiB), close to the design's estimate of about 300 MiB, and the pool shrank by 0.303 GiB; the drop stayed at 330 against B's 333 MiB, so the extra room goes straight to the margin (measured; the comparison is arithmetic).
- D8's flush right after ready released nothing on this idle server, so the flushed and unflushed readings agree here, and the near-full prompt, which in llmctl always follows that flush, started from the same 1,152 MiB (measured).
- The log reported the Triton kernels `write_req_to_token_pool_triton`, `_small_copy_kernel` and `apply_token_bitmask_inplace_kernel` as "device-loaded after serving started" at 22:25:16, during the first grammar request, with 0.81 GiB free by its own reading, and `nvidia-smi` still read 828 MiB after the request (measured; that the kernels fit in memory the prompt had already reserved is inferred).
- `PYTORCH_ALLOC_CONF` changed nothing measurable on this model, which has no pinned expert store: shmem stayed at 0.09 GiB as in B (measured).

Against the gate of the D8 rule over measurements B, C and D (`MIN_FREE` = 768 MiB, ship gate 896 MiB, largest drop 486 MiB): pool 77,988 against 65,536 passes, 1,152 against 896 MiB passes by 256 MiB, and the 330 MiB drop is below 486 MiB, so the start passes (estimate for the rule; the figures are measured).

## Smoke and thinking

From the hands-on `api_probe.py` with `--skip-long` ([shapes/d1-f088-api.json](shapes/d1-f088-api.json)), all as B saw them except where noted (measured on the MSI Raider 18 Max HX, SGLang v0.5.23):
- Plain `OK` streamed and not; a tool call `get_weather {"city": "Paris"}` streamed (in one fragment) and not, and the second turn "The weather in Paris is currently **18°C and sunny**." streamed and not.
- `json_schema` reply `{"capital": "Paris"}`; the probe's forced `tool_choice` on an unrelated prompt returned `get_weather` with an empty city (B's start: `{"city": "London"}`).
- `"tools": []`, `prompt_cache_key`, `max_completion_tokens`, `tool_choice` `auto` without tools, `tool_choice` `none` with tools and a wrong model name got 200; `max_tokens` 131,072 got HTTP 400 ("This model supports at most 65536 completion tokens").
- A replayed assistant turn with only `reasoning` got HTTP 400 "Assistant message is missing a thinking field. ...", and the same turn with `reasoning_content` got 200, as in B.
- Thinking, type always: every request reasoned, from the default (393 characters) through every `reasoning_effort` from `none` (251) to `xhigh` (160) and `enable_thinking: false` (332), and every answer was `391`; the streamed thinking request decoded at 90.3 tok/s.

## The draft

`$SCRATCH/sglang/overrides/sglang/k2-horizon-7b.toml` is updated in place (the version before is `$SCRATCH/sglang/measure-d/drafts-before/k2-horizon-7b.toml`):
- `--mem-fraction-static` 0.88; the comments on the pool and the cgroup now give this start's figures.
- `[needs] gpu_mib = 14908` (the 15,020 MiB GPU peak minus the desktop's 112 MiB) and `ram_gib = 5` (the 4.92 GiB peak of anon plus shmem, rounded up), by D5's rule; `[container] memory_gib = 32` is unchanged.
- `[evidence]`: `decode_tok_s = 89.6`, the summary "89.6 tok/s decode over runs of 1,286 to 1,500 tokens with 64k context and FP8 KV, 77,988-token pool at fraction 0.88, 1,152 MiB free after ready and 828 MiB after a 62,254-token prompt and the first grammar request, 300 s of sustained load without error, 15,020 MiB GPU peak, smoke test and tool call passed", `date = "2026-11-18"` and `record = "$SCRATCH/sglang/measure-d/sglang-k2-horizon-7b/"`.
- Not re-run at 0.88: M13 (B: FP8 and BF16 KV 20 of 20 at 0.90) and M15 (B: a cold start without a network reached ready in 137.9 s at 0.90); the brief did not ask for them, and the fraction changes neither the KV dtype nor the kernels a cold start compiles (inferred).

## /dev/shm

`df -B1 /dev/shm` read 16,519,168 bytes used and 93 files before the start, 50,757,632 bytes (33 MiB more) and two more files at ready, and 16,539,648 bytes after the container was gone, when one root-owned file of this start was left: `sglang_loads_c77ea62c77ad_a85fc2b2.shm` (16,396 bytes), removed with a throwaway `--ipc=host --network none` container and named in [logs/d1-f088.devshm.txt](logs/d1-f088.devshm.txt); afterwards `/dev/shm` read 16,519,168 bytes and the same 93 files as before (measured).

## Files

- [cmds/d1-f088.sh](cmds/d1-f088.sh): the `docker run` file; it reads `$HOME` and `$SCRATCH`.
- `logs/d1-f088.*` and `shapes/d1-f088-*`: cut and full logs, the 1 s trace, process memory and mappings at ready, `/dev/shm` readings, state, metadata, readiness and `/server_info`, the sequence record, probe, api probe, sustained load and the summary.
