# Qwen3.6-35B-A3B NVFP4 on SGLang, 2026-11-18

Last verified: 2026-11-18.

Result: measured in measurement C: SGLang v0.5.23 serves 128k with Paged Experts at 88 of 256 experts resident, fraction 0.88 and BF16 KV, at 151.3 tok/s, held a fresh 124,498-token prompt, the first grammar request and 300 s of load with at least 465 MiB free, left 951 MiB free right after ready and passed the smoke test and tool call (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
Its post-ready drop of 486 MiB, the largest of measurements B and C, raises D8's threshold to 768 MiB (estimate from the measured margins).
Measurement E started the same settings once more for the reading llmctl's check takes: 977 MiB free after D8's `POST /flush_cache` right after ready and 505 MiB at the lowest after it, a drop of 472 MiB (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
Over measurements B, C, D and E the largest drop stays C's 486 MiB, so D8's rule still gives `MIN_FREE` = 768 MiB and the ship gate 896 MiB, which this model passes by 81 MiB flushed (estimate for the rule; the figures are measured).
The starts ran by hand with `docker run` before llmctl had an SGLang recipe; the first, m11-k64, began at 20:56 UTC, still 2026-11-17 local time, and every later one on 2026-11-18.
Measurement E's one start, e1-k88-f088-mem27, began at 22:57:22 UTC, which is 01:57 on 2026-11-18 local time (+03:00).

Image: `lmsysorg/sglang:v0.5.23-runtime`.
Model: `nvidia/Qwen3.6-35B-A3B-NVFP4` at revision `1355db6a052410cfd62085d94b58866fd0f2c3c5`, served as `qwen3.6-35b-a3b`.
Shipped settings (w3-k88-f088, [measure-c/cmds/w3-k88-f088.sh](measure-c/cmds/w3-k88-f088.sh)): context 131072, `--mem-fraction-static 0.88`, BF16 KV, 8 Mamba slots, `--max-running-requests 1`, `--disable-prefill-cuda-graph`, `--moe-runner-backend marlin --enable-paged-experts` with 88 experts resident, `--tool-call-parser qwen3_coder`, `--reasoning-parser qwen3` and `--default-chat-template-kwargs '{"enable_thinking":false}'`, with `--device nvidia.com/gpu=all --ipc=host` and the recipe's `memory_gib` 27 (the measuring start w3-k88-f088 ran under `--memory 32g --memory-swap 32g`, and the cap run w3-k88-f088-mem27 under `--memory 27g --memory-swap 27g`), the Hugging Face cache mounted read-only, a model overlay whose `generation_config.json` carries temperature 0.6, top_p 0.95, top_k 20 and min_p 0.0 (the llama.cpp recipe's values), and measurement C's environment with `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1`.
The reports are [measurement C's README for this model](measure-c/README.md) and [measurement C's lane README](../../2026-11-17/sglang-runtime/measure-c/README.md), where every figure of the section Measurement C comes from, and [measurement E's README for this model](measure-e/README.md) and [measurement E's lane README](../../2026-11-17/sglang-runtime/measure-e/README.md), where every figure of the section Measurement E comes from.

## Measurement C

- M11, the number K of experts kept on the GPU at fraction 0.90 and 128k: K 64 with the FP8 KV that v0.5.23 took from the checkpoint decoded 132.3 tok/s; with BF16 KV, K 88, 96 and 104 decoded 149.5, 158.6 and 164.1 tok/s, and K 104 held only 113,491 tokens, short of 128k (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17 and 2026-11-18).
- At fraction 0.90 every start left 585 to 677 MiB free after ready, below the 768 MiB threshold, so the recipe uses 0.88; K 88 there holds a 161,736-token BF16 pool (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- At the shipped settings: ready line 57.9 s after `docker run`, the sampling line `{'temperature': 0.6, 'top_k': 20, 'top_p': 0.95, 'min_p': 0.0}`, 951 MiB free right after ready, 503 MiB after the fresh 124,498-token prompt and after the first grammar request and the forced tool request, and 465 MiB at the lowest point after the second 95% prompt (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- Decode 151.3 tok/s median in the repository probe, and 2.417 s and 7.006 s to the first token for 8,591 and 32,153 tokens, against 122.5 tok/s and 4.34 s and 13.0 s for the shipped llama.cpp recipe (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18; the llama.cpp figures from `llamacpp/benchmarking.md`, llama.cpp b12407).
- M12: 87 requests in 300 s with no error, decode 141.5 to 174.5 tok/s (median 155.6); peaks 15,378 MiB of GPU memory and 146.5 W (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- Host memory: the pinned expert store is 16.97 GiB of shared memory, the peak of anon plus shmem 22.70 GiB and the non-reclaimable peak 22.92 GiB, so `ram_gib` 23; the same start under `--memory 27g --memory-swap 27g` (w3-k88-f088-mem27) passed the protocol and M12 with no OOM kill, so `memory_gib` 27 (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- Smoke requests: `OK`, a `get_weather {"city": "Paris"}` call and its second turn, `json_schema` and a forced `tool_choice` passed; thinking stays off without a field and turns on with `enable_thinking` true or a `reasoning_effort` of `low` or `high` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- M15: from a new empty cache volume with `--network none`, `/ready` answered 200 at 345.46 s (351.21 s in a second cold start), about 257 s of it FlashInfer's autotune, with no download or fetch line; that second start read 687 MiB free right after ready and 969 MiB after `POST /flush_cache`, above the 896 MiB ship gate (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- M13 was not run, because the recipe uses BF16 KV (measurement C's lane README, deviation 12).

## Measurement E

- The start: measurement C's command file `w3-k88-f088-mem27.sh` with one line added, `TIKTOKEN_ENCODINGS_BASE`, so the eight variables of the runtime file, `--memory 27g --memory-swap 27g`, C's overlay and the warm volume `sglang-research-cache`; its launch arguments equal the draft's flag for flag, and its overlay's 16 links point at the pinned snapshot (measured on the files before the start, 2026-11-18).
- Readiness: the ready line came 63.8 s after `docker run` and the poll saw `/ready` 200 at 64.06 s; the BF16 pool held 161,736 tokens, as in C's two starts of these settings (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- Margins: 951 MiB free right after ready, as in both of C's starts, 977 MiB after D8's flush (26 MiB released), 505 MiB after the fresh 124,502-token prompt, after Hermes' title request and after the forced tool request, and 505 MiB at the lowest through 300 s of load and a second 124,503-token prompt, so a drop of 472 MiB from the flushed reading and 446 MiB from the unflushed one (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- C's lowest of 465 MiB in w3-k88-f088 came after the repository probe and the api probe, which E's sequence does not send; the same probe left C's mem27 start at 503 MiB (measured in C's records; that the floor after the probe varies by about 34 MiB between starts is inferred from the two starts).
- M12: 91 requests in 300 s with no error, decode 138.4 to 241.3 tok/s (median 155.25), where the maximum is an 8-token reply, and the whole start's peaks were 15,338 MiB of GPU memory and 145.2 W, the power peak during the first 95% prompt (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- Host memory: anon plus shmem peaked at 22.76 GiB and the non-reclaimable total at 22.98 GiB under the 27 GiB cap without OOM, so the draft's `ram_gib` 23 and `memory_gib` 27 still hold (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- D8's rule over measurements B, C, D and E: the largest drop stays C's 486 MiB, and without C's two Qwen3.6 rows it is E's 472 MiB; both give `MIN_FREE` = 768 MiB and the ship gate 896 MiB (estimate from the measured margins).
- C's lowest of 465 MiB against E's flushed 977 MiB would give a drop of exactly 512 MiB, and 512 + 256 = 768 is a multiple of 64: kept as it is, as E's brief rounds, it gives 768 MiB, while the design's D8 wording "the next multiple of 64 above" gives 832 MiB and a 960 MiB gate; for every measured drop the two agree, and the lane README sets out the case (estimate).

## Files

| Path | Holds |
|---|---|
| [measure-c/README.md](measure-c/README.md) | Measurement C's report on this model, links rewritten |
| [measure-c/links.diff](measure-c/links.diff) | The link rewrite of that report against the scratch copy |
| [measure-c/cmds/](measure-c/cmds/) | The `docker run` of the starts m11-k64, m11-bf16-k88, m11-bf16-k96, m11-bf16-k104, w3-k88-f088, w3-k88-f088-mem27, m15-netnone and m15-netnone-b; they read `$HOME` and `$SCRATCH` |
| [measure-c/logs/](measure-c/logs/) | Per start the cut and full logs, the `docker run` output, the 1 s GPU and cgroup trace, the readiness poll, the process memory, the request log, the sequence output, the state, the metadata, the mappings inside the container at ready and the `/dev/shm` readings; for the network-none starts the network namespace, the volume listing and the matched network lines |
| [measure-c/shapes/](measure-c/shapes/) | Readiness and `/server_info`, the sequence records, the repository probe, the api probe, the sustained load and the summaries; for the network-none starts the `docker exec` poll |
| [measure-c/overlay/](measure-c/overlay/) | The overlay's generation config before and after, and its link count |
| [measure-e/README.md](measure-e/README.md) | Measurement E's report on this model, links rewritten |
| [measure-e/links.diff](measure-e/links.diff) | The link rewrite of that report against the scratch copy |
| [measure-e/cmds/](measure-e/cmds/) | The `docker run` of the start e1-k88-f088-mem27; it reads `$HOME` and `$SCRATCH` |
| [measure-e/logs/](measure-e/logs/) | The cut and full logs, the `docker run` output, the 1 s GPU and cgroup trace and every change of free memory in it, the readiness poll, the process memory, the mappings inside the container at ready, the cgroup peaks, the request log, the sequence output, the state, the metadata, the `/dev/shm` readings, the input check and the diff against measurement C's command file |
| [measure-e/shapes/](measure-e/shapes/) | Readiness and `/server_info` at ready, the sequence record, the sustained load and the summary |
