# Gemma 4 12B QAT W4A16 on SGLang, 2026-11-18

Last verified: 2026-11-18.

Result: measured in measurement D: at measurement B's shipped settings with the eight variables of the runtime file, Gemma 4 12B left 1,099 MiB free right after ready and 1,119 MiB after D8's `POST /flush_cache`, held a fresh 124,433-token prompt and the first grammar request with at least 841 MiB free and decoded 70.1 tok/s (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
Its drop of 258 MiB from the first reading and 278 MiB from the flushed one passes the gate of D8's rule over measurements B, C and D (`MIN_FREE` 768 MiB, ship gate 896 MiB) (estimate for the rule; the figures are measured).
The start ran by hand with `docker run` before llmctl had an SGLang recipe, at 22:38 UTC on 2026-11-17, which is 01:38 on 2026-11-18 local time (+03:00), and covered the margin protocol and the repository probe only.
The earlier records of this model, from measurement B, are in [research/2026-11-17/sglang-gemma-4-12b/](../../2026-11-17/sglang-gemma-4-12b/README.md).

Image: `lmsysorg/sglang:v0.5.23-runtime`.
Model: `google/gemma-4-12B-it-qat-w4a16-ct` at revision `1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee`, served as `gemma-4-12b`.
Shipped settings (d2-env, [measure-d/cmds/d2-env.sh](measure-d/cmds/d2-env.sh)): measurement B's draft arguments unchanged, context 131072, `--mem-fraction-static 0.90`, `--max-running-requests 1`, FP8 KV, `--swa-full-tokens-ratio 0.1`, `--disable-prefill-cuda-graph`, `--json-model-override-args '{"audio_config": null}'` and parsers `gemma4`, with `--device nvidia.com/gpu=all --ipc=host --memory 32g --memory-swap 32g`, the Hugging Face cache mounted read-only, a model overlay whose `generation_config.json` carries temperature 1.0, top_p 0.95 and top_k 64 (B's overlay, rebuilt), and the eight variables of the runtime file: `HF_HUB_OFFLINE=1`, `HF_MODULES_CACHE=/tmp/hf_modules`, `MAX_JOBS=2`, `FLASHINFER_NO_DOWNLOAD=1`, `SYNC_TOKEN_IDS_ACROSS_TP=1`, `NCCL_MAX_P2P_NCHANNELS=1`, `TIKTOKEN_ENCODINGS_BASE=/opt/sglang/lib/python3.12/site-packages/tml_renderers/data` and `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1`.
The reports are [measurement D's README for this model](measure-d/README.md) and [measurement D's lane README](../../2026-11-17/sglang-runtime/measure-d/README.md), where every figure below comes from.

## Measurement D

- Readiness: the ready line came 28.5 s after `docker run` and `/ready` answered 200 at 29.55 s; the FP8 pool held 204,172 full and 20,417 sliding-window tokens, 9 and 1 fewer than in B's m2-fp8-b (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- Margins: 1,099 MiB free right after ready, as in B, 1,119 MiB after D8's flush, and 841 MiB after the fresh 124,433-token prompt, Hermes' title request and the forced tool request and at the lowest to the end of the probe, against B's 839 MiB (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- The flush released 20 MiB of PyTorch's cache, which is the whole difference between the two drops (measured; that B's server held the same cache is inferred from B's own flush, which read 1,117 MiB).
- Decode 70.1 tok/s in the repository probe against B's 70.4, and 4.096 s and 15.368 s to the first token for 9,936 and 37,194 tokens (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- The new variables changed nothing measurable: equal readings right after ready and shmem at 0.07 GiB (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- Peaks over this shorter protocol: 15,002 MiB of GPU memory and 5.75 GiB of anon plus shmem; the power limit read 145 W, as in B (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- The draft is not edited, as measurement D's brief says.

## Files

| Path | Holds |
|---|---|
| [measure-d/README.md](measure-d/README.md) | Measurement D's report on this model, as the lane wrote it; no link needed a rewrite |
| [measure-d/cmds/](measure-d/cmds/) | The `docker run` of the start d2-env; it reads `$HOME` and `$SCRATCH` |
| [measure-d/logs/](measure-d/logs/) | The cut and full logs, the `docker run` output, the 1 s GPU and cgroup trace, the readiness poll, the process memory, the mappings inside the container at ready, the request log, the sequence output, the state, the metadata and the `/dev/shm` readings |
| [measure-d/shapes/](measure-d/shapes/) | Readiness and `/server_info`, the sequence records, the repository probe and the summary |
| [measure-d/overlay/](measure-d/overlay/) | The overlay's generation config before and after, and its link count |
