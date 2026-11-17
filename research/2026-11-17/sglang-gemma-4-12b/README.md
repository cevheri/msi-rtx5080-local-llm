# Gemma 4 12B QAT W4A16 on SGLang, 2026-11-17

Last verified: 2026-11-17.

Result: measured in measurement B: at the shipped settings (FP8 KV, `--swa-full-tokens-ratio 0.1`, `--json-model-override-args '{"audio_config": null}'`) Gemma 4 12B served 128k with a 204,181-token pool, decoded 70.4 tok/s, held a fresh 124,431-token prompt, both grammar requests and 300 s of load with at least 839 MiB free, and started offline from an empty cache volume in 137.0 s (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
The first start without the override flag failed on an audio projection of the wrong size; the hands-on run and measurement A did not run this model.

Image: `lmsysorg/sglang:v0.5.23-runtime`.
Model: `google/gemma-4-12B-it-qat-w4a16-ct` at revision `1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee`, served as `gemma-4-12b`.
Shipped settings (m2-fp8-b, [measure-b/cmds/m2-fp8-b.sh](measure-b/cmds/m2-fp8-b.sh)): context 131072, `--mem-fraction-static 0.90`, `--max-running-requests 1`, `--disable-prefill-cuda-graph`, FP8 KV, `--swa-full-tokens-ratio 0.1`, `--json-model-override-args '{"audio_config": null}'` and parsers `gemma4`, with `--device nvidia.com/gpu=all --ipc=host --memory 32g --memory-swap 32g`, the Hugging Face cache mounted read-only, a model overlay whose `generation_config.json` carries temperature 1.0, top_p 0.95 and top_k 64 (the vLLM recipe's values), and measurement B's environment (`HF_HUB_OFFLINE=1`, `HF_MODULES_CACHE=/tmp/hf_modules`, `MAX_JOBS=2`, `SYNC_TOKEN_IDS_ACROSS_TP=1`, `NCCL_MAX_P2P_NCHANNELS=1`, `FLASHINFER_NO_DOWNLOAD=1`).
The reports are [measurement B's README for this model](measure-b/README.md) and [measurement B's lane README](../sglang-runtime/measure-b/README.md), where every figure below comes from.

## Measurement B

- m2-fp8 without the override died one second into loading with `AssertionError: Attempted to load weight (torch.Size([3840, 640])) into parameter (torch.Size([3840, 1536]))`: SGLang sizes the audio projection from `audio_config.output_proj_dims`, which this checkpoint lacks, so a default of 1536 applies (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; the default inferred from the error).
- m2-fp8-b, with the override, reached its ready line at 33.4 s; `memory_usage` at ready: weight 8.773, kvcache 4.673, `startup_available` 1.52 GiB, `token_capacity` 204,181 (SWA 20,418); 1,099 MiB free right after ready (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Sampling line: `{'temperature': 1.0, 'top_k': 64, 'top_p': 0.95}` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- A fresh 124,431-token prompt took 107.86 s and left 839 MiB; the first grammar request and a forced tool call then moved nothing (839 MiB) (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Decode 70.4 tok/s median in the repository probe, against 69.5 tok/s on vLLM (measured on the MSI Raider 18 Max HX, SGLang v0.5.23 and vLLM v0.33.0, 2026-11-17; [vllm-gemma-4-12b/](../vllm-gemma-4-12b/README.md)); first token 4.102 s for 9,936 tokens and 15.411 s for 37,194 tokens (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- M12: 41 requests in 300 s with no error, decode median 68.9 tok/s, and a second 124,433-token prompt left 839 MiB (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Peaks: 15,003 MiB of GPU memory, 146.9 W, 74 C, and 5.92 GiB of non-reclaimable host memory (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- M13 and BF16 KV (m2-bf16, ratio 0.06, not shipped): a 139,214-token pool, 19 of 20 correct as with FP8 and the same 20 final answers, but long prefills 1.35 to 1.64 times as slow, so FP8 KV stays (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- M15: from a new empty cache volume with `--network none`, `/ready` answered 200 at 136.98 s, the decode-graph capture took 100.06 s cold against 2.44 s warm, 1,099 MiB were free after ready, a chat request answered `OK`, and no download or fetch line appeared (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Thinking, toggle type with the server default off: off by default, and `enable_thinking: true` and the efforts reasoned and answered in `content` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).

## Files

| Path | Holds |
|---|---|
| [measure-b/README.md](measure-b/README.md) | Measurement B's report on this model, links rewritten |
| [measure-b/links.diff](measure-b/links.diff) | The link rewrite of that report against the scratch copy |
| [measure-b/cmds/](measure-b/cmds/) | The `docker run` of m2-fp8, m2-fp8-b, m2-bf16 and m15-netnone |
| [measure-b/logs/](measure-b/logs/) | Per start the cut and full logs, the `docker run` output, the 1 s GPU and cgroup trace, the readiness poll, the state and the metadata; for the m2 starts that served also the process memory, the request log and the sequence output, and for m15-netnone the network namespace, the matched network lines and the volume listing |
| [measure-b/shapes/](measure-b/shapes/) | Readiness and `/server_info`, the sequence records, the repository probe, the api probe, the sustained load, the M13 answers and the summaries |
| [measure-b/overlay/](measure-b/overlay/) | The overlay's generation config before and after, and its link count |
