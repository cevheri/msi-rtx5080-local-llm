# K2 Horizon 7B AWQ-INT4 on SGLang, 2026-11-17

Last verified: 2026-11-17.

Result: measured in the hands-on run (a1 crashed on its first grammar request, a2 ran), in measurement A (M6, M7 and M8 at the a2 settings) and in measurement B (step 0, M12, M13 and M15 at the a2 settings, with the model overlay and both NCCL variables), all started by hand before llmctl had an SGLang recipe.
At the shipped settings, measurement B's start b0-a2 held a fresh 62,257-token prompt, both grammar requests and 300 s of load with at least 532 MiB free and decoded 89.5 tok/s, FP8 and BF16 KV gave the same 20 correct M13 answers, and an offline start from an empty cache volume reached `/ready` in 137.9 s (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).

Image: `lmsysorg/sglang:v0.5.23-runtime`.
Model: `cyankiwi/K2-Horizon-7B-AWQ-INT4` at revision `530d5ceccd8ea660ceaa2a9f3a4bcbb68ba44291`, served as `k2-horizon-7b`.
Working settings (a2, [handson/cmds/k2-horizon-7b-a2.sh](handson/cmds/k2-horizon-7b-a2.sh)): `--context-length 65536 --mem-fraction-static 0.90 --max-running-requests 1 --disable-prefill-cuda-graph --trust-remote-code --kv-cache-dtype fp8_e4m3 --tool-call-parser k2_horizon --reasoning-parser k2_horizon --default-chat-template-kwargs {"reasoning_effort":"high","tool_call_format":"xml"}`, with `--device nvidia.com/gpu=all --ipc=host --memory 32g --memory-swap 32g`, the Hugging Face cache mounted read-only and `HF_HUB_OFFLINE=1`.
The cross-model reports are [the hands-on README](../sglang-runtime/handson/README.md), [measurement A's README](../sglang-runtime/measure-a/README.md) and [measurement B's lane README](../sglang-runtime/measure-b/README.md), and measurement B's report on this model is [measure-b/README.md](measure-b/README.md); every figure below comes from them.

## Hands-on run

- The native `K2HorizonForCausalLM` class loaded with `--trust-remote-code`: weights 7.89 GB, FP8 KV pool 82,394 tokens, FlashInfer attention, auto-detected `k2_horizon` parsers (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- a1 with the prefill CUDA graph: its first capture took 96.23 s and 0.58 GB, which left 0.40 GB free, and the first grammar request crashed the server with `Failed to CUDA calloc 536870912 bytes` at 0.29 GiB free (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- a2 without it: 0.98 GB free; `tool_choice: "required"` returned a parsed `get_weather` call, `json_schema` returned `{"capital":"Paris"}`, the server stayed up, and the ready line came 34 s after `docker run` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Decode: 90.1 tok/s over 512-token runs and a median of 89.9 tok/s over 1,500-token runs (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17), against 94.8 tok/s on vLLM (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17; [research/2026-11-17/vllm-k2-horizon-7b/](../vllm-k2-horizon-7b/README.md)).
- First-token time: 2.426 s for 8,085 tokens and 9.142 s for 30,261 tokens, and 59,253 tokens took 27.73 s and 0.077 s again (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17), against 2.227 s and 10.869 s on vLLM (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17; [research/2026-11-17/vllm-k2-horizon-7b/](../vllm-k2-horizon-7b/README.md)).
- GPU peak 15,814 MiB, host anonymous memory 4.89 GiB at most (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- The model always reasons, and a replayed assistant turn with only a `reasoning` field returned HTTP 400, while the same turn with `reasoning_content` returned 200 (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).

## Measurement A

- Sampling (M8): from an overlay of 16 links, the remote-code files included, with `{"temperature": 1.0, "top_p": 0.95}` in its `generation_config.json`, the server logged those values, a chat request returned 200 and a tool call returned `get_weather` with `{"city": "Paris"}` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Grammar (M6): free memory was 962 MiB after ready and 960 MiB after `/flush_cache`; the first grammar request took 602 MiB and left 34 MiB after a 32k-token prompt, in either order, and every request passed (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Readiness (M7): the ready line came 33.9 s after `docker run`, `/v1/models` answered 200 at 33.55 s and `/ready` at 34.55 s; `startup_available` read 0.977 GiB (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).

## Measurement B

- b0-a2 is a2 plus measurement B's overlay (temperature 1.0 and top_p 0.95, the vLLM recipe's values) and environment (`HF_HUB_OFFLINE=1`, `HF_MODULES_CACHE=/tmp/hf_modules`, `MAX_JOBS=2`, `SYNC_TOKEN_IDS_ACROSS_TP=1`, `NCCL_MAX_P2P_NCHANNELS=1`, `FLASHINFER_NO_DOWNLOAD=1`) ([measure-b/cmds/b0-a2.sh](measure-b/cmds/b0-a2.sh)).
- The ready line came 37.1 s after `docker run` and `/ready` answered 200 at 37.55 s; the FP8 pool held 82,397 tokens, the sampling line read `{'temperature': 1.0, 'top_p': 0.95}`, and 865 MiB were free right after ready, against 962 MiB in measurement A without the two variables (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- A fresh 62,257-token prompt took 33.25 s and left 543 MiB; the first grammar request and a forced tool call then moved nothing (543 MiB), where measurement A lost 602 MiB at the first grammar request (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- The lowest free memory after ready, over the whole protocol, was 532 MiB, and 865 MiB right after ready is the lowest of measurement B's recipes (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Decode 89.5 tok/s median in the repository probe, and 2.198 s and 9.249 s to the first token for 8,085 and 30,261 tokens (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- M12: 52 requests in 300 s with no error, decode 79.0 to 91.3 tok/s (median 87.55), and a second 62,253-token prompt left 533 MiB (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Peaks: 15,311 MiB of GPU memory, 152.0 W in 1 s samples, 72 C, and 4.97 GiB of non-reclaimable host memory (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- M13 (m13-bf16, BF16 KV at a 32k context, because its pool holds 41,198 tokens): FP8 and BF16 both answered 20 of 20 correctly with identical final `content` strings, while five reasoning traces differed in length (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- M15: from a new empty cache volume with `--network none`, `/ready` answered 200 at 137.87 s, the decode-graph capture took 95.26 s cold against 1.21 s warm, 865 MiB were free after ready, a chat request answered `OK`, and no download or fetch line appeared (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).

## Files

| Path | Holds |
|---|---|
| [handson/cmds/](handson/cmds/) | The `docker run` of the hands-on starts a1 and a2 |
| [handson/logs/](handson/logs/) | Cut container logs (`attemptN.log`), container arguments (`*.cmd.json`), a socket table (`attempt2.proc-net-tcp.txt`), start times (`start-*.epoch`) and the 1 s GPU and cgroup trace (`trace.csv`) |
| [handson/shapes/](handson/shapes/) | The API probe of a1, the grammar requests of a2 and the repository probe of a2 |
| [measure-a/cmds/](measure-a/cmds/) | The `docker run` of the M6 o1 and o2 starts and the M8 overlay start s6 |
| [measure-a/logs/](measure-a/logs/) | Per run the cut log, the full log of the M6 runs, the request-log extract of s6, the trace, the poller and probe output and the container state |
| [measure-a/shapes/](measure-a/shapes/) | M6 and M8 records (`*.jsonl`), M7 poll records (`*-m7.json`), one full `/get_server_info` body and `/model_info` of s6 |
| [measure-a/overlay/](measure-a/overlay/) | The generation config before and after, and the link count, of the overlay |
| [measure-b/README.md](measure-b/README.md) | Measurement B's report on this model, links rewritten |
| [measure-b/links.diff](measure-b/links.diff) | The link rewrite of that report against the scratch copy |
| [measure-b/cmds/](measure-b/cmds/) | The `docker run` of measurement B's starts b0-a2, m13-bf16 and m15-netnone; they read `$HOME` and `$SCRATCH` |
| [measure-b/logs/](measure-b/logs/) | Per start the cut and full logs, the `docker run` output, the 1 s GPU and cgroup trace (with anon, file, shmem and kernel), the readiness poll, the process memory, the request log, the sequence output, the state and the metadata, and for the network-none starts the network namespace, the matched network lines and the volume listing |
| [measure-b/shapes/](measure-b/shapes/) | Readiness and `/server_info` (`-ready.json`, `-ready-server-info.json`), the sequence records (`-seq.jsonl`), the repository probe, the api probe, the sustained load, the M13 answers and the summaries; for the network-none starts the `docker exec` poll |
| [measure-b/overlay/](measure-b/overlay/) | The generation config before and after, and the link count, of measurement B's overlay |
