# Gemma 4 E4B IT QAT W4A16 on SGLang v0.5.23 with the runtime file's eight variables (measurement D)

Last verified: 2026-11-18.

Result: with `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1` and `TIKTOKEN_ENCODINGS_BASE` added, B's Gemma 4 E4B draft still passes the gate of the D8 rule over measurements B, C and D (`F_ready >= 896` MiB with a drop of at most 486 MiB): 1,407 MiB were free right after ready, as in B, and 1,923 MiB after D8's `POST /flush_cache`, and the drop from the flushed reading was 296 MiB (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
Every reading after the flush moved by more than 64 MiB against B's row: the free memory after the 95% prompt, after both grammar requests and at the lowest (1,627 against 1,403 MiB, 224 MiB more), and the drop (296 MiB from the flushed reading against B's 4 MiB), because D8's flush right after ready released 516 MiB that the server held in PyTorch's cache after its own warm-up, which B's prompt reused without a flush (measured; the mechanism is inferred, see below).
Decode is 140.0 against B's 141.3 tok/s (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).

The start used `lmsysorg/sglang:v0.5.23-runtime`, `google/gemma-4-E4B-it-qat-w4a16-ct` at `6cd26aaa2357fb2bad8c51699a7558a4d1a965bb` from the overlay `$SCRATCH/sglang/measure-d/overlay/gemma-4-e4b/` (rebuilt with B's `make_overlay.py`, equal to B's: 8 links and `generation_config.json` with temperature 1.0, top_p 0.95 and top_k 64), the volume `sglang-research-cache`, `--memory 32g --memory-swap 32g`, all eight variables of the runtime file (`HF_HUB_OFFLINE=1`, `HF_MODULES_CACHE=/tmp/hf_modules`, `MAX_JOBS=2`, `FLASHINFER_NO_DOWNLOAD=1`, `SYNC_TOKEN_IDS_ACROSS_TP=1`, `NCCL_MAX_P2P_NCHANNELS=1`, `TIKTOKEN_ENCODINGS_BASE=/opt/sglang/lib/python3.12/site-packages/tml_renderers/data`, `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1`), and B's draft arguments unchanged: fraction 0.85, `--max-running-requests 1`, `--swa-full-tokens-ratio 0.1`, BF16 KV, `--disable-prefill-cuda-graph`, `--json-model-override-args '{"audio_config": null}'`, parsers `gemma4`, context 65536 ([cmds/d2-env.sh](cmds/d2-env.sh)).
The enforced GPU power limit read 145 W before the start, as in B, and the desktop held 112 MiB ([logs/d2-env.meta](logs/d2-env.meta)).
The sequence was the margin protocol and the probe only, once: D8's flushed reading, a fresh 95% prompt, Hermes' title request, a forced tool request, then `/flush_cache` and the repository probe, with thinking off (`seq_d.py --phases flushready,nearfull,grammar,probe`).

## Start d2-env against B's m5-a4-swa

| Item | D d2-env | B m5-a4-swa |
|---|---|---|
| Readiness | `/v1/models` 200 at 27.55 s, `/ready` 200 at 29.55 s, ready line 28.5 s | 26.55, 28.55 s |
| Pool and memory at ready | BF16 pool 101,785 full and 10,178 sliding-window tokens (3.40 GB), weight 9.293, `startup_available` 2.281 GiB, `available_gpu_mem=2.28 GB` | 101,791 and 10,179 tokens, the same weights, `available_gpu_mem=2.28 GB` |
| Sampling line | `Using default chat sampling params from model generation config: {'temperature': 1.0, 'top_k': 64, 'top_p': 0.95}` | the same |
| Free right after ready | 1,407 MiB | 1,407 MiB |
| Free after D8's `POST /flush_cache` | 1,923 MiB (at once and 2 s later; 516 MiB released) | not read; B's flush before its probe read 1,919 MiB |
| Fresh 62,244-token prompt (95.0% of 65,536), after that flush | 22.61 s, free 1,923 to 1,627 MiB | no flush before it, 22.52 s, 1,407 to 1,403 MiB |
| Hermes title request (`json_schema`, strict), the first grammar request | 1,627 to 1,627 MiB, `{"title": "Python Addition Basics"}`, parses | 1,403 to 1,403 MiB |
| Forced `tool_choice` | 1,627 to 1,627 MiB, `get_weather {"city": "Paris"}`, `finish_reason` `tool_calls` | 1,403 to 1,403 MiB |
| Lowest free after ready, to the end of the probe | 1,627 MiB after the flush (1,407 before it) | 1,403 MiB (also the lowest over B's whole protocol with M12) |
| Drop | none from 1,407 (free rose by 220 MiB), 296 MiB from the flushed 1,923 | 4 MiB |
| Decode, repository probe after `/flush_cache` | 140.0 tok/s (140.0, 139.7, 140.1 over runs of 936 to 1,197 tokens) | 141.3 tok/s |
| First token | 1.618 s for 9,932 tokens, 8.647 s for 37,190 tokens | 1.617 and 8.632 s |
| Peaks over this shorter protocol | 14,436 MiB of GPU memory, 145.0 W, 69 C; anon plus shmem 5.54 GiB (anon 5.48, shmem 0.07); `memory.peak` 14.89 GiB with page cache | 14,439 MiB; 5.61 GiB; 5.99 GiB over the whole protocol, with only 0.32 GiB of page cache charged to B's container |

All figures are measured on the MSI Raider 18 Max HX, SGLang v0.5.23 ([shapes/d2-env-ready.json](shapes/d2-env-ready.json), [shapes/d2-env-seq.jsonl](shapes/d2-env-seq.jsonl), [shapes/d2-env-probe.json](shapes/d2-env-probe.json), [logs/d2-env-trace.csv](logs/d2-env-trace.csv), [logs/d2-env.log](logs/d2-env.log), [shapes/d2-env-summary.json](shapes/d2-env-summary.json); margins by `$SCRATCH/sglang/measure-d/scripts/margins_d.py`); B's column is from `$SCRATCH/sglang/measure-b/sglang-gemma-4-e4b/` and B's margin table.

## Why the readings moved

- Right after ready the server held 516 MiB of PyTorch's cached blocks from its own warm-up: D8's flush released them (1,407 to 1,923 MiB), and B's server read the same 1,407 MiB at ready and 1,919 MiB at its own flush before its probe, after its margin requests (measured in both sequence records; that B's server held the same blocks is inferred).
- B's 95% prompt then ran inside those cached blocks and needed only 4 MiB more, while D's prompt, after the flush, reserved 296 MiB afresh; so after the prompt D's server held 220 MiB less cache than B's (516 against 296 MiB), which with the 4 MiB higher flushed reading makes the 224 MiB higher reading (inferred from the four readings).
- The new variables are not the cause: the readings right after ready (1,407 MiB), the pool, the weights and the flushed readings (1,923 against 1,919 MiB) match B's within 4 MiB (measured).
- For D8 this means Gemma 4 E4B's margin depends on where the flush sits: in llmctl the check flushes right after ready, so the first request starts from 1,923 MiB and leaves at least 1,627 MiB, the case measured here (measured; that llmctl's check flushes first is D8's design).

Gate (`MIN_FREE` = 768 MiB, ship gate 896 MiB, largest drop 486 MiB): passes, 1,923 MiB flushed and 1,407 unflushed against 896, and a drop of 296 MiB from the flushed reading against 486 (estimate for the rule; the figures are measured).
The draft `$SCRATCH/sglang/overrides/sglang/gemma-4-e4b.toml` is not edited, as the brief says.

## /dev/shm

Before the start 16,519,168 bytes and 93 files, 50,798,592 bytes at ready, 16,539,648 bytes after the container was gone; the one root-owned file the start left, `sglang_loads_d5d44629e892_96ea78fe.shm` (16,396 bytes), was removed with a throwaway `--ipc=host --network none` container, after which the reading was 16,519,168 bytes again ([logs/d2-env.devshm.txt](logs/d2-env.devshm.txt)) (measured).

## Files

- [cmds/d2-env.sh](cmds/d2-env.sh): the `docker run` file; it reads `$HOME` and `$SCRATCH`.
- `logs/d2-env.*` and `shapes/d2-env-*`: cut and full logs, the 1 s trace, process memory and mappings at ready, `/dev/shm` readings, state, metadata, readiness and `/server_info`, the sequence record, the probe and the summary.
