# Gemma 4 12B IT QAT W4A16 on SGLang v0.5.23 with the runtime file's eight variables (measurement D)

Last verified: 2026-11-18.

Result: with `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1` and `TIKTOKEN_ENCODINGS_BASE` added, B's Gemma 4 12B draft still passes the gate of the D8 rule over measurements B, C and D (`F_ready >= 896` MiB with a drop of at most 486 MiB): 1,099 MiB were free right after ready and 1,119 MiB after D8's `POST /flush_cache`, and the drop was 258 MiB from the first reading and 278 MiB from the second (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
Every reading is within 20 MiB of B's (no figure moved by more than 64 MiB), and decode is 70.1 against B's 70.4 tok/s (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).

The start used `lmsysorg/sglang:v0.5.23-runtime`, `google/gemma-4-12B-it-qat-w4a16-ct` at `1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee` from the overlay `$SCRATCH/sglang/measure-d/overlay/gemma-4-12b/` (rebuilt with B's `make_overlay.py`, equal to B's: 9 links and `generation_config.json` with temperature 1.0, top_p 0.95 and top_k 64), the volume `sglang-research-cache`, `--memory 32g --memory-swap 32g`, all eight variables of the runtime file (`HF_HUB_OFFLINE=1`, `HF_MODULES_CACHE=/tmp/hf_modules`, `MAX_JOBS=2`, `FLASHINFER_NO_DOWNLOAD=1`, `SYNC_TOKEN_IDS_ACROSS_TP=1`, `NCCL_MAX_P2P_NCHANNELS=1`, `TIKTOKEN_ENCODINGS_BASE=/opt/sglang/lib/python3.12/site-packages/tml_renderers/data`, `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1`), and B's draft arguments unchanged: fraction 0.90, `--max-running-requests 1`, FP8 KV, `--swa-full-tokens-ratio 0.1`, `--disable-prefill-cuda-graph`, `--json-model-override-args '{"audio_config": null}'`, parsers `gemma4`, context 131072 ([cmds/d2-env.sh](cmds/d2-env.sh)).
The enforced GPU power limit read 145 W before the start, as in B, and the desktop held 112 MiB ([logs/d2-env.meta](logs/d2-env.meta)).
The sequence was the margin protocol and the probe only, once: D8's flushed reading, a fresh 95% prompt, Hermes' title request, a forced tool request, then `/flush_cache` and the repository probe, with thinking off (`seq_d.py --phases flushready,nearfull,grammar,probe`).

## Start d2-env against B's m2-fp8-b

| Item | D d2-env | B m2-fp8-b |
|---|---|---|
| Readiness | `/v1/models` 200 at 27.55 s, `/ready` 200 at 29.55 s, ready line 28.5 s | 29.55, 33.55 s |
| Pool and memory at ready | FP8 pool 204,172 full and 20,417 sliding-window tokens (4.67 GB), weight 8.773, `startup_available` 1.519 GiB, `available_gpu_mem=1.52 GB` | 204,181 and 20,418 tokens, the same weights, `available_gpu_mem=1.52 GB` |
| Sampling line | `Using default chat sampling params from model generation config: {'temperature': 1.0, 'top_k': 64, 'top_p': 0.95}` | the same |
| Free right after ready | 1,099 MiB | 1,099 MiB |
| Free after D8's `POST /flush_cache` | 1,119 MiB (at once and 2 s later) | not read; B's flush before its probe read 1,117 MiB |
| Fresh 124,433-token prompt (94.9% of 131,072), after that flush | 106.0 s, free 1,119 to 841 MiB | 124,431 tokens, no flush before it, 107.86 s, 1,099 to 839 MiB |
| Hermes title request (`json_schema`, strict), the first grammar request | 841 to 841 MiB, `{"title": "Adding Numbers in Python"}`, parses | 839 to 839 MiB |
| Forced `tool_choice` | 841 to 841 MiB, `get_weather {"city": "Paris"}`, `finish_reason` `tool_calls` | 839 to 839 MiB |
| Lowest free after ready, to the end of the probe | 841 MiB | 839 MiB (also the lowest over B's whole protocol with M12) |
| Drop | 258 MiB from 1,099, 278 MiB from the flushed 1,119 | 260 MiB |
| Decode, repository probe after `/flush_cache` | 70.1 tok/s (70.1, 70.0, 70.1 over runs of 824 to 905 tokens) | 70.4 tok/s |
| First token | 4.096 s for 9,936 tokens, 15.368 s for 37,194 tokens | 4.102 and 15.411 s |
| Peaks over this shorter protocol | 15,002 MiB of GPU memory, 146.8 W, 71 C; anon plus shmem 5.75 GiB (anon 5.68, shmem 0.07); `memory.peak` 14.48 GiB with page cache | 15,003 MiB; 5.81 GiB; 14.63 GiB over the whole protocol |

All figures are measured on the MSI Raider 18 Max HX, SGLang v0.5.23 ([shapes/d2-env-ready.json](shapes/d2-env-ready.json), [shapes/d2-env-seq.jsonl](shapes/d2-env-seq.jsonl), [shapes/d2-env-probe.json](shapes/d2-env-probe.json), [logs/d2-env-trace.csv](logs/d2-env-trace.csv), [logs/d2-env.log](logs/d2-env.log), [shapes/d2-env-summary.json](shapes/d2-env-summary.json); margins by `$SCRATCH/sglang/measure-d/scripts/margins_d.py`); B's column is from `$SCRATCH/sglang/measure-b/sglang-gemma-4-12b/` and B's margin table.
- The flush right after ready released 20 MiB of PyTorch's cache, which is the whole difference between the two drops (measured; B's own flush before its probe read 1,117 MiB, so B's server held the same cache, inferred).
- The variables changed nothing measurable: the pool is 9 full and 1 sliding-window tokens smaller, the ready reading is equal, and shmem stayed at 0.07 GiB (measured).

Gate (`MIN_FREE` = 768 MiB, ship gate 896 MiB, largest drop 486 MiB): passes, 1,119 MiB flushed and 1,099 unflushed against 896, and a drop of 278 or 258 MiB against 486 (estimate for the rule; the figures are measured).
The draft `$SCRATCH/sglang/overrides/sglang/gemma-4-12b.toml` is not edited, as the brief says.

## /dev/shm

Before the start 16,519,168 bytes and 93 files, 50,798,592 bytes at ready, 16,543,744 bytes after the container was gone; the start left two root-owned files, `sem.mp-2sdxc_zx` (32 bytes) and `sglang_loads_13fd3ae07862_761ba7ab.shm` (16,396 bytes), both removed with a throwaway `--ipc=host --network none` container, after which the reading was 16,519,168 bytes again ([logs/d2-env.devshm.txt](logs/d2-env.devshm.txt)) (measured).

## Files

- [cmds/d2-env.sh](cmds/d2-env.sh): the `docker run` file; it reads `$HOME` and `$SCRATCH`.
- `logs/d2-env.*` and `shapes/d2-env-*`: cut and full logs, the 1 s trace, process memory and mappings at ready, `/dev/shm` readings, state, metadata, readiness and `/server_info`, the sequence record, the probe and the summary.
