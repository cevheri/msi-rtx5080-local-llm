# Qwen3.5-9B NVFP4 on SGLang, 2026-11-18

Last verified: 2026-11-18.

Result: measured in measurement D: at measurement B's shipped settings with the eight variables of the runtime file, Qwen3.5-9B left 911 MiB free right after ready and 937 MiB after D8's `POST /flush_cache`, held a fresh 124,502-token prompt and the first grammar request with at least 541 MiB free and decoded 92.8 tok/s (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
Its drop of 370 MiB from the first reading and 396 MiB from the flushed one passes the gate of D8's rule over measurements B, C and D (`MIN_FREE` 768 MiB, ship gate 896 MiB), and its free-memory readings sit 15 MiB above the gate unflushed (911 - 896) and 41 MiB above it flushed (937 - 896), the closest of the SGLang recipes (estimate for the rule; the figures are measured).
The start ran by hand with `docker run` before llmctl had an SGLang recipe, at 22:36 UTC on 2026-11-17, which is 01:36 on 2026-11-18 local time (+03:00), and covered the margin protocol and the repository probe only.
The earlier records of this model, from the hands-on run and measurements A and B, are in [research/2026-11-17/sglang-qwen3.5-9b/](../../2026-11-17/sglang-qwen3.5-9b/README.md).

Image: `lmsysorg/sglang:v0.5.23-runtime`.
Model: `RedHatAI/Qwen3.5-9B-NVFP4` at revision `a4df85a20fb7e54740f564f15af0c2f5d9c6616c`, served as `qwen3.5-9b`.
Shipped settings (d2-env, [measure-d/cmds/d2-env.sh](measure-d/cmds/d2-env.sh)): measurement B's draft arguments unchanged, context 131072, `--mem-fraction-static 0.88`, `--max-running-requests 1`, 8 Mamba slots, `--disable-prefill-cuda-graph`, parsers `qwen3_coder` and `qwen3` and `{"enable_thinking":false}`, with `--device nvidia.com/gpu=all --ipc=host --memory 32g --memory-swap 32g`, the Hugging Face cache mounted read-only, a model overlay whose `generation_config.json` carries temperature 0.6, top_p 0.95 and top_k 20 (B's overlay, rebuilt), and the eight variables of the runtime file: `HF_HUB_OFFLINE=1`, `HF_MODULES_CACHE=/tmp/hf_modules`, `MAX_JOBS=2`, `FLASHINFER_NO_DOWNLOAD=1`, `SYNC_TOKEN_IDS_ACROSS_TP=1`, `NCCL_MAX_P2P_NCHANNELS=1`, `TIKTOKEN_ENCODINGS_BASE=/opt/sglang/lib/python3.12/site-packages/tml_renderers/data` and `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1`.
The reports are [measurement D's README for this model](measure-d/README.md) and [measurement D's lane README](../../2026-11-17/sglang-runtime/measure-d/README.md), where every figure below comes from.

## Measurement D

- Readiness: the ready line came 28.3 s after `docker run` and `/ready` answered 200 at 29.05 s; the BF16 pool held 140,974 tokens, 7 fewer than in B's m12-a3 (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- Margins: 911 MiB free right after ready, as in B, 937 MiB after D8's flush, 551 MiB after the fresh 124,502-token prompt, 549 MiB after Hermes' title request and the forced tool request, and 541 MiB at the lowest to the end of the probe, against B's 537 MiB over the same requests (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- The flush released 26 MiB of PyTorch's cache, which is the whole difference between the two drops (measured; the attribution is inferred from measurement A and from B's own flush, which read 935 MiB).
- Decode 92.8 tok/s in the repository probe against B's 93.7, at a 140 W power limit against B's 145 W, and 0.569 s and 2.408 s to the first token for 8,591 and 32,153 tokens (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- The new variables changed nothing measurable: equal readings right after ready and shmem at 0.10 GiB, so this model has no pinned store for `PYTORCH_ALLOC_CONF` to change (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- Peaks over this shorter protocol: 15,302 MiB of GPU memory and 5.56 GiB of anon plus shmem (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- The draft is not edited, as measurement D's brief says.

## Files

| Path | Holds |
|---|---|
| [measure-d/README.md](measure-d/README.md) | Measurement D's report on this model, links rewritten |
| [measure-d/links.diff](measure-d/links.diff) | The link rewrite of that report against the scratch copy |
| [measure-d/cmds/](measure-d/cmds/) | The `docker run` of the start d2-env; it reads `$HOME` and `$SCRATCH` |
| [measure-d/logs/](measure-d/logs/) | The cut and full logs, the `docker run` output, the 1 s GPU and cgroup trace, the readiness poll, the process memory, the mappings inside the container at ready, the request log, the sequence output, the state, the metadata and the `/dev/shm` readings |
| [measure-d/shapes/](measure-d/shapes/) | Readiness and `/server_info`, the sequence records, the repository probe and the summary |
| [measure-d/overlay/](measure-d/overlay/) | The overlay's generation config before and after, and its link count |
