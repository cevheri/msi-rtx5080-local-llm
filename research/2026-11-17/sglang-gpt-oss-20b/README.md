# gpt-oss-20b on SGLang, 2026-11-17

Last verified: 2026-11-17.

Result: measured in the hands-on run (a1 was refused, a2 and a3 crashed on their first grammar request, a4 with Paged Experts ran), in measurement A (M6 and M7 at the a4 settings), in measurement B (step 0, M10, M12, M15 and the memory cap at the a4 settings, with the model overlay, both NCCL variables and `TIKTOKEN_ENCODINGS_BASE`) and in measurement C (the pinned-store setting at the a4 settings), all started by hand before llmctl had an SGLang recipe.
In the earlier measurement at the a4 settings, measurement B's start b0-a4 held a fresh 124,463-token prompt, both grammar requests and 300 s of load with at least 989 MiB free and decoded 125.5 tok/s, the same start under a 28 GiB memory cap passed with no OOM kill, and an offline start from an empty cache volume reached `/ready` in 253.6 s (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
In measurement C the same start with `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1` cut the pinned expert store from 19.19 to 10.37 GiB of shared memory at the same decode and margins (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
At the shipped settings, measurement C's start pe-round2-mem20 (a4 with that variable, `ram_gib` 16 and `memory_gib` 20) held a fresh 124,459-token prompt, the first grammar request, the forced tool request and 300 s of load with at least 1,006 MiB free under a 20 GiB memory cap, decoded 124.6 tok/s and peaked at 17.04 GiB of cgroup memory with no OOM kill (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).

Image: `lmsysorg/sglang:v0.5.23-runtime`.
Model: `openai/gpt-oss-20b` at revision `6cee5e81ee83917806bbde320786a8fb61efebee`, served as `gpt-oss-20b`.
Working settings (a4, [handson/cmds/gpt-oss-20b-a4.sh](handson/cmds/gpt-oss-20b-a4.sh)): `--context-length 131072 --mem-fraction-static 0.90 --max-running-requests 1 --kv-cache-dtype fp8_e4m3 --swa-full-tokens-ratio 0.1 --chunked-prefill-size 1024 --disable-prefill-cuda-graph --moe-runner-backend marlin --enable-paged-experts --paged-experts-num-resident 24 --tool-call-parser gpt-oss --reasoning-parser gpt-oss`, with `--device nvidia.com/gpu=all --ipc=host --memory 32g --memory-swap 32g`, the Hugging Face cache mounted read-only and `HF_HUB_OFFLINE=1`.
The cross-model reports are [the hands-on README](../sglang-runtime/handson/README.md), [measurement A's README](../sglang-runtime/measure-a/README.md), [measurement B's lane README](../sglang-runtime/measure-b/README.md) and [measurement C's lane README](../sglang-runtime/measure-c/README.md), and measurement B's and measurement C's reports on this model are [measure-b/README.md](measure-b/README.md) and [measure-c/README.md](measure-c/README.md); every figure below comes from them.

## Hands-on run

- a1 at 64k and `--mem-fraction-static 0.95` was refused: the weights left 0.60 GB, no memory for the KV cache (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- a2 at 0.97 ran with an 11,896-token pool and a3 with the Marlin MoE runner at 0.95 with a 34,726-token pool; both crashed on the first grammar-constrained request in NCCL with `Failed to CUDA calloc 536870912 bytes` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- a4 with Paged Experts kept 24 of 32 experts per layer on the GPU: weights 11.46 GB, FP8 KV pool 168,443 tokens plus 16,844 SWA tokens, 1.45 GB free, and the ready line 61 s after `docker run` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Decode at a4: 133.8 tok/s over 512-token runs, and a median of 125.5 tok/s in the repository probe over answers of 1,464 to 1,500 tokens (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- First-token time at a4 after `/flush_cache`: 1.452 s for 8,058 tokens and 4.893 s for 30,003 tokens; 123,958 tokens took 41.69 s and 0.209 s again (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Host memory at a4: anonymous memory peaked at 13.94 GiB against 5.47 GiB without paging, because the pinned host store holds all 32 experts of every layer, and the cgroup peak with page cache was 28.24 GiB of the 32 GiB cap (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Forced tool choice returned HTTP 200 with `finish_reason` `stop`, no `tool_calls` and raw Harmony text in `content` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).

## Measurement A

- Grammar (M6): free memory was 1,406 MiB after ready and 1,426 MiB after `/flush_cache`; the first grammar request took 602 MiB and left 468 MiB after a 32k-token prompt, and the server stayed up (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Readiness (M7): the ready line came 106.2 s after `docker run` (115.8 s in the second start), `/v1/models` answered 200 at 106.06 s and `/ready` at 107.06 s; `startup_available` read 1.447 GiB (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- The start-up spent 52 and 63 s, against 6 s in the hands-on starts, in the step that builds the Harmony encoding, which fetches `o200k_base.tiktoken` unless `TIKTOKEN_ENCODINGS_BASE` points at the image's copy (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; the download as the cause is inferred from the sources).
- The container's cgroup stayed at 31.88 of 32 GiB from about 44 s after `docker run` to the end of both starts, with no OOM kill (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).

## Measurement B

- b0-a4 is a4 plus measurement B's overlay (the snapshot's `generation_config.json` with temperature 1.0 and top_p 1.0, the vLLM recipe's values) and environment (`HF_HUB_OFFLINE=1`, `HF_MODULES_CACHE=/tmp/hf_modules`, `MAX_JOBS=2`, `SYNC_TOKEN_IDS_ACROSS_TP=1`, `NCCL_MAX_P2P_NCHANNELS=1`, `FLASHINFER_NO_DOWNLOAD=1` and `TIKTOKEN_ENCODINGS_BASE=/opt/sglang/lib/python3.12/site-packages/tml_renderers/data`) ([measure-b/cmds/b0-a4.sh](measure-b/cmds/b0-a4.sh)).
- With `TIKTOKEN_ENCODINGS_BASE` the start-up step took 1 s instead of measurement A's 52 to 63 s, and the ready line came 54.9 s after `docker run` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- The FP8 pool held 168,456 tokens, the sampling line read `{'temperature': 1.0, 'top_p': 1.0}`, and 1,329 MiB were free right after ready, against 1,406 MiB in measurement A without the two variables (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- A fresh 124,463-token prompt took 43.95 s and left 989 MiB; the first grammar request then moved nothing and passed, and a forced `tool_choice` answered with `finish_reason` `stop`, no `tool_calls` and empty content (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Decode 125.5 tok/s median in the repository probe, and 1.486 s and 4.903 s to the first token for 8,058 and 30,003 tokens, against 125.5 tok/s, 1.452 s and 4.893 s in the hands-on a4 run, so the variables cost no speed (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- M12: 74 requests in 301 s with no error, decode 118.8 to 134.9 tok/s (median 127.5), and a second 124,462-token prompt left 1,059 MiB; peaks 14,854 MiB of GPU memory, 145.3 W and 73 C (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Host memory: the expert store is shared memory, 19.19 GiB of it, so non-reclaimable memory peaked at 24.35 GiB, and the hands-on lane's 13.94 GiB of anonymous memory was the loading peak (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Under `--memory 28g --memory-swap 28g` (ld19-mem28) the start held 168,439 tokens, left 988 MiB after a fresh 124,458-token prompt, decoded 125.6 tok/s, served 73 requests in 300 s without error, and `memory.events` read `oom 0` and `oom_kill 0` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- M10, the number K of the 32 experts kept on the GPU: K 16, 20, 24, 25 and 28 decoded 87.9, 112.3, 125.5, 130.0 and 137.6 tok/s; K 28 holds only 62,643 tokens, short of 128k; K 24 holds 128k with 37,384 tokens to spare and is the recommendation (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- M15: from a new empty cache volume with `--network none`, `/ready` answered 200 at 253.58 s, the decode-graph capture took 185.09 s cold against 1.72 s warm, 1,329 MiB were free after ready, a chat request answered `OK`, and no download or fetch line appeared; without `TIKTOKEN_ENCODINGS_BASE` the offline start also became ready but logged that the Responses API was disabled because the Harmony vocabulary could not be loaded (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- The shipped vLLM recipe without a network is in [vllm-gpt-oss-20b-offline/](../vllm-gpt-oss-20b-offline/README.md).

## Measurement C

- pe-round1 and pe-round2-mem20 repeat b0-a4 with `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1` added ([measure-c/cmds/pe-round1.sh](measure-c/cmds/pe-round1.sh), [measure-c/cmds/pe-round2-mem20.sh](measure-c/cmds/pe-round2-mem20.sh)); pe-round2-mem20 ran under `--memory 20g --memory-swap 20g` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- B's 1.84 factor is PyTorch 2.14.1's pinned host allocator rounding each paged tensor of the store up to a power of two, and the documented setting keeps every pinned request above 1 MiB at its size (inferred from the image's PyTorch and SGLang source; the setting's text vendor-reported).
- With the setting the store was 10.37 GiB of shared memory against B's 19.19 GiB, the non-reclaimable peak 15.44 and 15.45 GiB against 24.35 GiB, and the peak of anon plus shmem 15.30 GiB, so `ram_gib` would be 16 instead of 25 and `memory_gib` 20 instead of 28 (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Free memory was 1,346 and 1,347 MiB right after ready, 1,036 and 1,006 MiB after a fresh 95% prompt, and the first grammar request and the forced tool request moved nothing (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Decode 123.3 and 124.6 tok/s in the repository probe against B's 125.5, and M12 served 73 requests in 300 s with no error at a median of 126.4 and 126.2 tok/s, within the spread of B's runs; under the 20 GiB cap the cgroup peaked at 17.04 GiB with no OOM kill (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- The store sits in shared `/dev/zero (deleted)` mappings, counted as cgroup `shmem`, not in `/dev/shm`, whose use grew only from 1,519,616 to 35,758,080 bytes at ready in pe-round1; each start left one root-owned 16,396-byte `sglang_loads_<hex>_<hex>.shm` file there, which measurement C removed (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).

## Files

| Path | Holds |
|---|---|
| [handson/cmds/](handson/cmds/) | The `docker run` of the hands-on starts a1 to a4 |
| [handson/logs/](handson/logs/) | Cut container logs (`attemptN.log`), container arguments (`*.cmd.json`), a socket table (`attempt4.proc-net-tcp.txt`), start times (`start-*.epoch`) and the 1 s GPU and cgroup trace (`trace.csv`) |
| [handson/shapes/](handson/shapes/) | The API probes of a2, a3 and a4 and the repository probe of a4 |
| [measure-a/cmds/](measure-a/cmds/) | The `docker run` of the M6 o1 and o2 starts |
| [measure-a/logs/](measure-a/logs/) | Per run the cut log, the full log, the trace, the poller and probe output and the container state |
| [measure-a/shapes/](measure-a/shapes/) | M6 records (`*.jsonl`), M7 poll records (`*-m7.json`) and one full `/get_server_info` body |
| [measure-b/README.md](measure-b/README.md) | Measurement B's report on this model, links rewritten |
| [measure-b/links.diff](measure-b/links.diff) | The link rewrite of that report against the scratch copy |
| [measure-b/cmds/](measure-b/cmds/) | The `docker run` of measurement B's starts b0-a4, b0-netnone-tik, b0-netnone-notik, ld19-mem28 and m10-k16, m10-k20, m10-k25 and m10-k28; they read `$HOME` and `$SCRATCH` |
| [measure-b/logs/](measure-b/logs/) | Per start the cut and full logs, the `docker run` output, the 1 s GPU and cgroup trace (with anon, file, shmem and kernel), the readiness poll, the process memory, the request log, the sequence output, the state and the metadata, and for the network-none starts the network namespace, the matched network lines and the volume listing |
| [measure-b/shapes/](measure-b/shapes/) | Readiness and `/server_info` (`-ready.json`, `-ready-server-info.json`), the sequence records (`-seq.jsonl`), the repository probe, the api probe, the sustained load and the summaries; for the network-none starts the `docker exec` poll |
| [measure-b/overlay/](measure-b/overlay/) | The generation config before and after, and the link count, of measurement B's overlay |
| [measure-c/README.md](measure-c/README.md) | Measurement C's report on this model, links rewritten |
| [measure-c/links.diff](measure-c/links.diff) | The link rewrite of that report against the scratch copy |
| [measure-c/cmds/](measure-c/cmds/) | The `docker run` of measurement C's starts pe-round1 and pe-round2-mem20; they read `$HOME` and `$SCRATCH` |
| [measure-c/logs/](measure-c/logs/) | Per start the cut and full logs, the `docker run` output, the 1 s GPU and cgroup trace, the readiness poll, the process memory, the request log, the sequence output, the state, the metadata, the mappings inside the container at ready and the `/dev/shm` readings |
| [measure-c/shapes/](measure-c/shapes/) | Readiness and `/server_info`, the sequence records, the repository probe, the api probe, the sustained load and the summaries |
| [measure-c/overlay/](measure-c/overlay/) | The generation config before and after, and the link count, of measurement C's overlay |
