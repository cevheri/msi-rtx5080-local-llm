# Qwen3.5-9B NVFP4 on SGLang v0.5.23 with the runtime file's eight variables (measurement D)

Last verified: 2026-11-18.

Result: with `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1` and `TIKTOKEN_ENCODINGS_BASE` added, B's Qwen3.5-9B draft still passes the gate of the D8 rule over measurements B, C and D (`F_ready >= 896` MiB with a drop of at most 486 MiB): 911 MiB were free right after ready and 937 MiB after D8's `POST /flush_cache`, and the drop was 370 MiB from the first reading and 396 MiB from the second (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
Every reading is within 26 MiB of B's (no figure moved by more than 64 MiB), and decode is 92.8 against B's 93.7 tok/s (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
Qwen3.5-9B stays the recipe closest to the gate: 15 MiB above it on the unflushed reading and 41 MiB on the flushed one (arithmetic on the measured figures).

The start used `lmsysorg/sglang:v0.5.23-runtime`, `RedHatAI/Qwen3.5-9B-NVFP4` at `a4df85a20fb7e54740f564f15af0c2f5d9c6616c` from the overlay `$SCRATCH/sglang/measure-d/overlay/qwen3.5-9b/` (rebuilt with B's `make_overlay.py`, equal to B's: 11 links and `generation_config.json` with temperature 0.6, top_p 0.95 and top_k 20), the volume `sglang-research-cache`, `--memory 32g --memory-swap 32g`, all eight variables of the runtime file (`HF_HUB_OFFLINE=1`, `HF_MODULES_CACHE=/tmp/hf_modules`, `MAX_JOBS=2`, `FLASHINFER_NO_DOWNLOAD=1`, `SYNC_TOKEN_IDS_ACROSS_TP=1`, `NCCL_MAX_P2P_NCHANNELS=1`, `TIKTOKEN_ENCODINGS_BASE=/opt/sglang/lib/python3.12/site-packages/tml_renderers/data`, `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1`), and B's draft arguments unchanged: fraction 0.88, `--max-running-requests 1`, 8 Mamba slots, `--disable-prefill-cuda-graph`, parsers `qwen3_coder` and `qwen3`, `{"enable_thinking":false}`, context 131072 ([cmds/d2-env.sh](cmds/d2-env.sh)).
The enforced GPU power limit read 140 W before the start (B: 145 W), and the desktop held 112 MiB ([logs/d2-env.meta](logs/d2-env.meta)).
The sequence was the margin protocol and the probe only, once: D8's flushed reading, a fresh 95% prompt, Hermes' title request, a forced tool request, then `/flush_cache` and the repository probe, with thinking off (`seq_d.py --phases flushready,nearfull,grammar,probe`).

## Start d2-env against B's m12-a3

| Item | D d2-env | B m12-a3 |
|---|---|---|
| Readiness | `/v1/models` 200 at 27.05 s, `/ready` 200 at 29.05 s, ready line 28.3 s | 27.05, 29.55 s |
| Pool and memory at ready | BF16 pool 140,974 tokens, 8 Mamba slots, weight 8.391, kvcache 4.302, `startup_available` 1.349 GiB, `available_gpu_mem=1.35 GB` | 140,981 tokens, the same weights, `available_gpu_mem=1.35 GB` |
| Sampling line | `Using default chat sampling params from model generation config: {'temperature': 0.6, 'top_k': 20, 'top_p': 0.95}` | the same |
| Free right after ready | 911 MiB | 911 MiB |
| Free after D8's `POST /flush_cache` | 937 MiB (at once and 2 s later) | not read; B's flush before its probe read 935 MiB |
| Fresh 124,502-token prompt (95.0% of 131,072), after that flush | 24.32 s, free 937 to 551 MiB | 124,499 tokens, no flush before it, 24.38 s, 911 to 545 MiB |
| Hermes title request (`json_schema`, strict), the first grammar request | 551 to 549 MiB, `{"title": "Python Addition Basics"}`, parses | 545 to 545 MiB |
| Forced `tool_choice` | 549 to 549 MiB, `get_weather {"city": "Paris"}`, `finish_reason` `tool_calls` | 545 to 545 MiB |
| Lowest free after ready, to the end of the probe | 541 MiB | 537 MiB (also the lowest over B's whole protocol with M12) |
| Drop | 370 MiB from 911, 396 MiB from the flushed 937 | 374 MiB |
| Decode, repository probe after `/flush_cache` | 92.8 tok/s (92.8 in all three runs of 747 to 827 tokens) | 93.7 tok/s |
| First token | 0.569 s for 8,591 tokens, 2.408 s for 32,153 tokens | 0.572 and 2.398 s |
| Peaks over this shorter protocol | 15,302 MiB of GPU memory, 144.9 W, 65 C; anon plus shmem 5.56 GiB (anon 5.46, shmem 0.10); `memory.peak` 14.73 GiB with page cache | 15,305 MiB; 5.60 GiB; 14.83 GiB over the whole protocol |

All figures are measured on the MSI Raider 18 Max HX, SGLang v0.5.23 ([shapes/d2-env-ready.json](shapes/d2-env-ready.json), [shapes/d2-env-seq.jsonl](shapes/d2-env-seq.jsonl), [shapes/d2-env-probe.json](shapes/d2-env-probe.json), [logs/d2-env-trace.csv](logs/d2-env-trace.csv), [logs/d2-env.log](logs/d2-env.log), [shapes/d2-env-summary.json](shapes/d2-env-summary.json); margins by `$SCRATCH/sglang/measure-d/scripts/margins_d.py`); B's column is from `$SCRATCH/sglang/measure-b/sglang-qwen3.5-9b/` and B's margin table.
- B's lowest is taken over the same requests (to the end of the probe) for a like-for-like drop; it equals B's lowest over the whole protocol with M12 (measured, [../lane-logs/margins-to-probe.txt](../../../2026-11-17/sglang-runtime/measure-d/lane-logs/margins-to-probe.txt)).
- The flush right after ready released 26 MiB of PyTorch's cache, as measurement A found on this model (1,013 to 1,039 MiB); the 26 MiB larger drop from the flushed reading is that release, not a change in the server (measured; the attribution is inferred from A and from B's own flush, which read 935 MiB).
- The variables changed nothing measurable here: the pool is 7 tokens smaller, the ready reading is equal, and shmem stayed at 0.10 GiB, so this model has no pinned store for `PYTORCH_ALLOC_CONF` to change (measured).
- Decode is 1.0% below B's, at a 140 W power limit against B's 145 W (measured; the cause is not established).

Gate (`MIN_FREE` = 768 MiB, ship gate 896 MiB, largest drop 486 MiB): passes, 937 MiB flushed and 911 unflushed against 896, and a drop of 396 or 370 MiB against 486 (estimate for the rule; the figures are measured).
The draft `$SCRATCH/sglang/overrides/sglang/qwen3.5-9b.toml` is not edited, as the brief says.

## /dev/shm

Before the start 16,519,168 bytes and 93 files, 50,798,592 bytes at ready, 16,539,648 bytes after the container was gone; the one root-owned file the start left, `sglang_loads_55b2197ed27d_5d7c85fb.shm` (16,396 bytes), was removed with a throwaway `--ipc=host --network none` container, after which the reading was 16,519,168 bytes again ([logs/d2-env.devshm.txt](logs/d2-env.devshm.txt)) (measured).

## Files

- [cmds/d2-env.sh](cmds/d2-env.sh): the `docker run` file; it reads `$HOME` and `$SCRATCH`.
- `logs/d2-env.*` and `shapes/d2-env-*`: cut and full logs, the 1 s trace, process memory and mappings at ready, `/dev/shm` readings, state, metadata, readiness and `/server_info`, the sequence record, the probe and the summary.
