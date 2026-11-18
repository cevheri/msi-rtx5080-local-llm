# Qwen3.6-35B-A3B NVFP4 on SGLang v0.5.23 with Paged Experts (M11, wave 3)

Last verified: 2026-11-18.

Result: `nvidia/Qwen3.6-35B-A3B-NVFP4` serves 128k on SGLang v0.5.23 with Paged Experts at 88 of 256 experts resident, fraction 0.88 and BF16 KV, at 151.3 tok/s against 122.5 tok/s for the shipped llama.cpp recipe, and passes every row of the design's D14 that applies (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
The pinned expert store is 16.97 GiB of shared memory, the non-reclaimable peak 22.92 GiB, and the protocol with M12 held under a 27 GiB cap; the draft is `$SCRATCH/sglang/overrides/sglang/qwen3.6-35b-a3b.toml`.
Its post-ready drop of 486 MiB is larger than the 374 MiB that set the threshold, so the D8 rule now gives `min_free_at_ready_mib = 768` and a ship gate of 896 MiB, which this recipe passes at 951 MiB and B's K2 Horizon 7B (865 MiB) does not (estimate from the measured margins; see the lane README).

Every start used `lmsysorg/sglang:v0.5.23-runtime` as B's starts did: C's overlay of the pinned snapshot `1355db6a052410cfd62085d94b58866fd0f2c3c5` mounted read-only at `/llmctl/model` (`$SCRATCH/sglang/measure-c/overlay/qwen3.6-35b-a3b/`, its `generation_config.json` with temperature 0.6, top_p 0.95, top_k 20 and min_p 0.0 from the llama.cpp twin's `--temp`, `--top-p`, `--top-k` and `--min-p`), `HF_HUB_OFFLINE=1`, `HF_MODULES_CACHE=/tmp/hf_modules`, `MAX_JOBS=2`, `SYNC_TOKEN_IDS_ACROSS_TP=1`, `NCCL_MAX_P2P_NCHANNELS=1`, `FLASHINFER_NO_DOWNLOAD=1`, `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1` (the pinned-store fix, [../sglang-gpt-oss-20b/README.md](../../../2026-11-17/sglang-gpt-oss-20b/measure-c/README.md)), `--device nvidia.com/gpu=all --ipc=host`, 127.0.0.1:30000, the volume `sglang-research-cache`, `--max-running-requests 1`, `--disable-prefill-cuda-graph`, `--moe-runner-backend marlin --enable-paged-experts`, 8 Mamba slots, parsers `qwen3_coder` and `qwen3`, and `--default-chat-template-kwargs '{"enable_thinking":false}'` ([cmds/](cmds/)).
The parser names are the v0.5.23 ones that match the llama.cpp recipe's toggle thinking and the checkpoint template's `<tool_call><function=...><parameter=...>` format ([../offline/sglang-parser-names.txt](../../../2026-11-17/sglang-runtime/measure-c/offline/sglang-parser-names.txt)) (inferred from the image's source and the template).
The enforced GPU power limit read 140 W before the warm starts and 145 W before the cold one; dates are local (+03:00), times UTC.

## M11: the K sweep (fraction 0.90, 128k)

| K | KV | Weights on the GPU | `token_capacity` | `startup_available` | Free after ready | Decode (probe median) | First token, 8,591 tokens | First token, 32,153 tokens | Pinned store | Ready line |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | FP8, chosen by the checkpoint's quantization config | 7.760 GiB | 535,616 | 1.035 GiB | 585 MiB | 132.3 tok/s | 3.210 s | 7.968 s | 16.97 GiB | 232.5 s (first start of the model, kernels compiled) |
| 88 | BF16 | 9.480 GiB | 177,593 | 1.080 GiB | 631 MiB | 149.5 tok/s | 2.647 s | 6.928 s | 16.97 GiB | 60.5 s |
| 96 | BF16 | 9.980 GiB | 152,403 | 1.107 GiB | 659 MiB | 158.6 tok/s | 2.540 s | 6.700 s | 16.97 GiB | 56.8 s |
| 104 | BF16 | 10.723 GiB | 113,491, short of 128k | 1.105 GiB | 677 MiB | 164.1 tok/s | 2.462 s | 6.518 s | 16.97 GiB | 57.3 s |

All rows are measured on the MSI Raider 18 Max HX, SGLang v0.5.23, with the repository probe after `/flush_cache` and thinking off ([cmds/m11-k64.sh](cmds/m11-k64.sh), `cmds/m11-bf16-k*.sh`, `shapes/m11-*-probe.json`, `shapes/m11-*-ready.json`, `logs/m11-*.procmem.txt`; table by `$SCRATCH/sglang/measure-c/scripts/w3_table.py`).
- Without `--kv-cache-dtype`, v0.5.23 took FP8 KV from the checkpoint's ModelOpt configuration (`KV Cache is allocated. dtype: torch.float8_e4m3fn`), so the K 64 start ran FP8; the sweep then set BF16, the design's choice where the pool holds the context, and stopped chain q1 before its K 72 and K 80 starts ([../lane-logs/chain-q1.txt](../../../2026-11-17/sglang-runtime/measure-c/lane-logs/chain-q1.txt)) (measured).
- Each resident slot costs about 0.078 GiB of GPU weights (9.480 to 10.723 GiB over 16 slots), about 4,000 BF16 tokens of pool (measured; the per-slot figure is arithmetic).
- K 104 does not hold 128k: v0.5.23 logged `KV pool holds 113491 tokens, fewer than --context-length 131072` and started anyway (measured).
- The store is the same 16.97 GiB of shared memory at every K, all 256 experts of all 40 layers (16.875 GiB in the checkpoint index), and the scheduler held it in `/dev/zero (deleted)` shared mappings, 17,334 MiB resident ([logs/m11-k64.maps.txt](logs/m11-k64.maps.txt)) (measured).
- At fraction 0.90 every start left 585 to 677 MiB free after ready, below D8's 768 MiB gate, because about 0.45 GiB goes to FlashInfer between "Memory pool end" and ready, as on the other FlashInfer models (report section 2.6); so the recipe uses 0.88 (measured; the cause is inferred).

Recommendation: K 88 at fraction 0.88, which holds 128k with 30,664 tokens (23%) to spare at 151.3 tok/s; the next step, K 96, decodes 158.6 tok/s (6% faster) but at 0.88 would hold about 136,700 tokens, 4% above the context (estimate: 0.3 GiB less pool, about 15,700 BF16 tokens), so it has too little room to ship without its own protocol run.
Against llama.cpp b12407 (`--n-cpu-moe 25 --moe-cache-mib 3200`, 128k): 122.5 tok/s, and first tokens of 4.34 s for 8,591 tokens and 13.0 s for 32,153 tokens with 8,075 cached, against 2.417 and 7.006 s here (measured on the MSI Raider 18 Max HX; the llama.cpp figures from `llamacpp/benchmarking.md`).

## The shipped settings: start w3-k88-f088 (32 GiB cap) and w3-k88-f088-mem27 (27 GiB cap)

| Item | w3-k88-f088 | w3-k88-f088-mem27 |
|---|---|---|
| Readiness | `/v1/models` and `/server_info` 200 at 56.06 s, `/ready` 200 at 58.06 s, ready line 57.9 s | ready line 59.9 s, poll 60.19 s |
| Load | `Load weight end. elapsed=32.27 s ... mem usage=9.48 GB`, BF16 pool 161,736 tokens (1.54 + 1.54 GB), Mamba 8 slots (0.53 GB), decode graph 0.73 s and 0.06 GB, `available_gpu_mem=1.39 GB` | the same |
| Sampling line | `Using default chat sampling params from model generation config: {'temperature': 0.6, 'top_k': 20, 'top_p': 0.95, 'min_p': 0.0}` | the same |
| Free right after ready | 951 MiB | 951 MiB |
| Fresh 124,498-token prompt (95.0% of 131,072), no flush before it | 49.59 s, free 951 to 503 MiB | 49.57 s, free 951 to 503 MiB |
| Hermes title request (`json_schema`, strict), the first grammar request | 503 to 503 MiB, `{"title": "Python: Add Two Numbers"}`, parses | 503 to 503 MiB, `{"title": "Python Number Addition"}` |
| Forced `tool_choice` | 503 to 503 MiB, `get_weather {"city": "Paris"}`, `finish_reason` `tool_calls` | the same |
| Lowest free after ready, drop | 465 MiB (during M12 and after the second 95% prompt), 486 MiB | 501 MiB, 450 MiB |
| Decode, repository probe | 151.3 tok/s (151.3, 153.9, 149.1) | 154.3 tok/s (154.3, 153.2, 155.2) |
| First token, 8,591 and 32,153 tokens | 2.417 s and 7.006 s; repeats 0.128 and 0.153 s | 2.400 s and 6.890 s |
| M12: requests in 300 s, errors, decode min / median / max, halves | 87, 0, 141.5 / 155.6 / 174.5, 152.7 and 157.8 | 90, 0, 142.9 / 155.8 / 177.8, 157.2 and 155.2 |
| Second 95% prompt after the load | 124,499 tokens in 50.04 s, 465 MiB left | 124,501 tokens in 49.51 s, 501 MiB left |
| GPU peak; power; temperature | 15,378 MiB; 146.5 W | 15,342 MiB; 144.8 W |
| Cgroup: anon + shmem peak; anon + shmem + kernel peak; anonymous peak while loading; `memory.peak` | 22.70; 22.92; 20.33; 31.01 GiB | 22.70; 22.92; 20.28; 27.00 GiB (page cache squeezed) |
| Container at the end | `oom=false` | `oom=false` |

All values are measured on the MSI Raider 18 Max HX, SGLang v0.5.23 (`shapes/w3-k88-f088*-ready.json`, `-seq.jsonl`, `-probe.json`, `-api.json`, `-sustained.json`, `-summary.json`, `logs/w3-k88-f088*-trace.csv`, `logs/w3-k88-f088*.procmem.txt`; margins by `$SCRATCH/sglang/measure-c/scripts/margins_c.py`, cgroup peaks by `anonshmem.py`).
- During the 95% prompt the cgroup split was anon 5.71, shmem 16.97 and file 25.05 GiB (32 GiB cap) and anon 5.71, shmem 16.97, file 21.07 GiB (27 GiB cap): the store, a stable 5.7 GiB of process memory and page cache that the cap squeezes (measured).
- So `ram_gib = 23` (the 22.70 GiB peak of anon plus shmem, rounded up) and `memory_gib = 27` (the 22.92 GiB non-reclaimable peak plus about 4 GiB, held by the second start without OOM) (measured; the margin is a choice).
- Loading holds the checkpoint layout of every expert as anonymous memory first (20.3 GiB at about 31 s), then repacks it into the pinned store layer by layer, so anon plus shmem stays near 21 to 22 GiB throughout (measured in `logs/m11-k64-trace.csv`).
- Smoke requests ([shapes/w3-k88-f088-api.json](shapes/w3-k88-f088-api.json)), all HTTP 200 except one: `OK` streamed and not; `get_weather {"city": "Paris"}` streamed and not, and the second turn "The weather in Paris is sunny with a temperature of 18°C."; `json_schema` `{"capital": "Paris"}`; a forced `tool_choice` on an unrelated prompt returned a `get_weather` call; `"tools": []`, `tool_choice` `auto` without tools, a wrong model name and replayed `reasoning` or `reasoning_content` got 200; `max_tokens` 262,144 got HTTP 400 (measured).
- Thinking (toggle, off by default): no field, `enable_thinking` false and `reasoning_effort` `none` gave no reasoning; `enable_thinking` true (855 characters), `reasoning_effort` `low` (749) and `high` (523) turned it on, as on Qwen3.5-9B (report section 3.6); every answer was 391 (measured).

## M15: cold start without a network

Start m15-netnone repeated w3-k88-f088 with `--network none`, no published port and a new empty volume `sglang-measure-cold` ([cmds/m15-netnone.sh](cmds/m15-netnone.sh)), polled with `docker exec` every second ([shapes/m15-netnone-exec-poll.txt](shapes/m15-netnone-exec-poll.txt)):
- `/ready` answered 503 at 329.06 s and 200 at 345.46 s, and the served name was listed at 345.62 s; the chat request answered `OK` (measured).
- 257 s of it was FlashInfer's autotune on the empty volume (21:24:01 to 21:28:18 in the log), which a warm volume keeps in `flashinfer/autotune/0.7.2/sm120/` (measured; the attribution is from the log lines).
- 687 MiB was free right after ready, against 951 MiB warm, a reading without a flush (measured).
- A second cold start, m15-netnone-b ([cmds/m15-netnone-b.sh](cmds/m15-netnone-b.sh), the same command on a new empty volume), added the step D8's check takes: `/ready` 503 at 334.81 s, 200 at 351.21 s, the name listed at 351.39 s, 687 MiB free right after ready, `POST /flush_cache` 200, and 969 MiB free after it, above the 896 MiB gate, so the first start of the recipe on an empty `sglang-cache` volume passes the post-ready check ([shapes/m15-netnone-b-exec-poll.txt](shapes/m15-netnone-b-exec-poll.txt)) (measured).
- In both starts the searched patterns matched only `server_args`, Gloo's `Unable to resolve hostname` warning and the prefill-graph line: no download or fetch line ([logs/m15-netnone.network-lines.txt](logs/m15-netnone.network-lines.txt), [logs/m15-netnone-b.network-lines.txt](logs/m15-netnone-b.network-lines.txt)) (measured).
- Each volume held 470 files and 41 MB afterwards (`jit` 25 MB, `triton` 16 MB, `inductor` 684 KB, `flashinfer` 24 KB) and was removed (measured).
- At 345.6 and 351.4 s these are the slowest SGLang starts of measurements B and C, below D4's 450 s line, so `start_timeout_s` 900 stands (measured; the rule is the design's).

## The pinned store and /dev/shm

The store is shared memory and does not sit in `/dev/shm`: the scheduler held 17.02 GiB as `RssShmem` in `/dev/zero (deleted)` shared mappings (cgroup `shmem` 16.97 GiB), while the host's `/dev/shm`, which `--ipc=host` shares, held 29 to 37 MiB more at ready than before the start (measured; `logs/<run>.maps.txt`, `logs/<run>.procmem.txt`, `logs/<run>.devshm.txt`).
Each start left one root-owned `sglang_loads_<hex>_<hex>.shm` of 16,396 bytes after its container was gone, which C removed and named in the run's `.devshm.txt`; no other file appeared (measured).
From the end of w3-k88-f088 on, `df` counted 24,219,648, then 16,453,632, then 20,717,568 bytes used in `/dev/shm` whenever no container ran, against 1,519,616 before every earlier start, with no file to account for the difference and a size that changed between runs; space of a deleted file that some process outside the containers holds open would look like this, and a non-root user cannot see whose it is (measured; the explanation is inferred).

## Files

- [cmds/](cmds/): one `docker run` file per start, reading `$HOME` and `$SCRATCH`.
- `logs/<run>.*` and `shapes/<run>-*`: as in B's records, plus `.maps.txt`, `.devshm.txt` and `.devshm-before.list`; the M15 start adds the exec poll, the volume listing and the matched network lines.
