# Gemma 4 E4B QAT W4A16 on SGLang, 2026-11-18

Last verified: 2026-11-18.

Result: measured in measurement D: at measurement B's shipped settings with the eight variables of the runtime file, Gemma 4 E4B left 1,407 MiB free right after ready, as in B, and 1,923 MiB after D8's `POST /flush_cache`, held a fresh 62,244-token prompt and the first grammar request with at least 1,627 MiB free and decoded 140.0 tok/s (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
Its drop of 296 MiB from the flushed reading passes the gate of D8's rule over measurements B, C and D (`MIN_FREE` 768 MiB, ship gate 896 MiB) (estimate for the rule; the figures are measured).
The readings after the flush differ from B's by more than 64 MiB because the flush released 516 MiB that the server held in PyTorch's cache after its own warm-up and that B's unflushed prompt reused, not because of the new variables (measured; the mechanism is inferred from the readings).
The start ran by hand with `docker run` before llmctl had an SGLang recipe, at 22:41 UTC on 2026-11-17, which is 01:41 on 2026-11-18 local time (+03:00), and covered the margin protocol and the repository probe only.
The earlier records of this model, from the hands-on run and measurement B, are in [research/2026-11-17/sglang-gemma-4-e4b/](../../2026-11-17/sglang-gemma-4-e4b/README.md).

Image: `lmsysorg/sglang:v0.5.23-runtime`.
Model: `google/gemma-4-E4B-it-qat-w4a16-ct` at revision `6cd26aaa2357fb2bad8c51699a7558a4d1a965bb`, served as `gemma-4-e4b`.
Shipped settings (d2-env, [measure-d/cmds/d2-env.sh](measure-d/cmds/d2-env.sh)): measurement B's draft arguments unchanged, context 65536, `--mem-fraction-static 0.85`, `--max-running-requests 1`, `--swa-full-tokens-ratio 0.1`, BF16 KV, `--disable-prefill-cuda-graph`, `--json-model-override-args '{"audio_config": null}'` and parsers `gemma4`, with `--device nvidia.com/gpu=all --ipc=host --memory 32g --memory-swap 32g`, the Hugging Face cache mounted read-only, a model overlay whose `generation_config.json` carries temperature 1.0, top_p 0.95 and top_k 64 (B's overlay, rebuilt), and the eight variables of the runtime file: `HF_HUB_OFFLINE=1`, `HF_MODULES_CACHE=/tmp/hf_modules`, `MAX_JOBS=2`, `FLASHINFER_NO_DOWNLOAD=1`, `SYNC_TOKEN_IDS_ACROSS_TP=1`, `NCCL_MAX_P2P_NCHANNELS=1`, `TIKTOKEN_ENCODINGS_BASE=/opt/sglang/lib/python3.12/site-packages/tml_renderers/data` and `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1`.
The reports are [measurement D's README for this model](measure-d/README.md) and [measurement D's lane README](../../2026-11-17/sglang-runtime/measure-d/README.md), where every figure below comes from.

## Measurement D

- Readiness: the ready line came 28.5 s after `docker run` and `/ready` answered 200 at 29.55 s; the BF16 pool held 101,785 full and 10,178 sliding-window tokens, 6 and 1 fewer than in B's m5-a4-swa (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- Margins: 1,407 MiB free right after ready, as in B, 1,923 MiB after D8's flush, and 1,627 MiB after the fresh 62,244-token prompt, Hermes' title request and the forced tool request and at the lowest to the end of the probe, against B's 1,403 MiB without a flush before its prompt (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- Because llmctl's check flushes right after ready, the first request of an llmctl start begins from the flushed reading, the case measured here (measured; the order is D8's design).
- Decode 140.0 tok/s in the repository probe against B's 141.3, and 1.618 s and 8.647 s to the first token for 9,932 and 37,190 tokens (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- The new variables are not the cause of the moved readings: the readings right after ready, the pool, the weights and the flushed readings (1,923 against B's 1,919 MiB at its flush before its probe) match B's within 4 MiB, and shmem stayed at 0.07 GiB (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- Peaks over this shorter protocol: 14,436 MiB of GPU memory and 5.54 GiB of anon plus shmem; the power limit read 145 W, as in B (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- The draft is not edited, as measurement D's brief says.

## Files

| Path | Holds |
|---|---|
| [measure-d/README.md](measure-d/README.md) | Measurement D's report on this model, as the lane wrote it; no link needed a rewrite |
| [measure-d/cmds/](measure-d/cmds/) | The `docker run` of the start d2-env; it reads `$HOME` and `$SCRATCH` |
| [measure-d/logs/](measure-d/logs/) | The cut and full logs, the `docker run` output, the 1 s GPU and cgroup trace, the readiness poll, the process memory, the mappings inside the container at ready, the request log, the sequence output, the state, the metadata and the `/dev/shm` readings |
| [measure-d/shapes/](measure-d/shapes/) | Readiness and `/server_info`, the sequence records, the repository probe and the summary |
| [measure-d/overlay/](measure-d/overlay/) | The overlay's generation config before and after, and its link count |
