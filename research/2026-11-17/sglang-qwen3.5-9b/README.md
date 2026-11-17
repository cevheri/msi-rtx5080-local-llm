# Qwen3.5-9B NVFP4 on SGLang, 2026-11-17

Last verified: 2026-11-17.

Result: measured in the hands-on run (a2 and a3 ran, a1 was refused), in measurement A (M6, M7, M8 and M15 at the a3 settings) and in measurement B (the margins, M12 and M15 at the a3 settings, with the model overlay and both NCCL variables), all started by hand before llmctl had an SGLang recipe.
At the shipped settings, measurement B's start m12-a3 held a fresh 124,499-token prompt, both grammar requests and 300 s of load with at least 537 MiB free and decoded 93.7 tok/s, and an offline start from an empty cache volume reached `/ready` in 48.3 s (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).

Image: `lmsysorg/sglang:v0.5.23-runtime`.
Model: `RedHatAI/Qwen3.5-9B-NVFP4` at revision `a4df85a20fb7e54740f564f15af0c2f5d9c6616c`, served as `qwen3.5-9b`.
Working settings (a3, [handson/cmds/qwen3.5-9b-a3.sh](handson/cmds/qwen3.5-9b-a3.sh)): `--context-length 131072 --mem-fraction-static 0.88 --max-running-requests 1 --max-mamba-cache-size 8 --disable-prefill-cuda-graph --tool-call-parser qwen3_coder --reasoning-parser qwen3 --default-chat-template-kwargs {"enable_thinking":false}`, with `--device nvidia.com/gpu=all --ipc=host --memory 32g --memory-swap 32g`, the Hugging Face cache mounted read-only and `HF_HUB_OFFLINE=1`.
The cross-model reports are [the hands-on README](../sglang-runtime/handson/README.md), [measurement A's README](../sglang-runtime/measure-a/README.md) and [measurement B's lane README](../sglang-runtime/measure-b/README.md), and measurement B's report on this model is [measure-b/README.md](measure-b/README.md); every figure below comes from them.

## Hands-on run

- a1 with `--language-model-only` was refused at argument resolution, because v0.5.23 supports that flag for four other architectures only, so the vision tower always loads: weights take 8.39 GB (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17), against 7.48 GiB language-only on vLLM (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17; [research/2026-11-17/vllm-qwen3.5-9b/](../vllm-qwen3.5-9b/README.md)).
- a2 at `--mem-fraction-static 0.85` ran with a KV pool of 74,233 tokens, short of the 131,072-token context, because the auto-sized mamba pool spent about 2 GB on 41 state slots (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- a3 with 8 mamba slots and no prefill CUDA graph held 140,976 BF16 tokens and left 1.35 GB free after start, and reached its ready line 28 s after `docker run` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- GPU peak by `nvidia-smi`: 15,806 MiB (a2) and 15,810 MiB (a3) (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17), against 13,726 MiB for vLLM (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17; [research/2026-11-17/vllm-qwen3.5-9b/](../vllm-qwen3.5-9b/README.md)); host anonymous memory peaked at 5.61 GiB (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Decode: 92.9 tok/s on 512-token answers (a3), and a median of 92.8 tok/s in the repository probe over answers of 643 to 813 tokens (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17), against 90.8 tok/s on vLLM (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17; [research/2026-11-17/vllm-qwen3.5-9b/](../vllm-qwen3.5-9b/README.md)).
- First-token time after `/flush_cache` at a3: 0.566 s for 8,591 tokens and 2.394 s for 32,153 tokens; 124,667 tokens took 23.13 s (5,390 tok/s) and 0.057 s again, and decode at that depth ran at 56.8 tok/s (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Sustained load with both CUDA graph types on (run 4, the a2 configuration): 66 requests in 300 s with fresh prompts of up to 13,448 tokens, no error, median decode 91.3 tok/s, at most 70 C and 144.5 W (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Thinking: `chat_template_kwargs.enable_thinking` true and false work and the server default off holds, but `reasoning_effort` `low` and `high` turn thinking on despite that default (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).

## Measurement A

- Sampling (M8): `--preferred-sampling-params` left field-less chat requests at temperature 1.0, top_p 1.0 and top_k -1, while the overlay with `{"temperature": 0.6, "top_p": 0.95, "top_k": 20}` in its `generation_config.json` was logged at start and applied to every field-less request, and with `top_k` 1 gave five identical replies (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Grammar (M6): free memory was 1,013 MiB after ready and 1,039 MiB after `/flush_cache`; the first grammar request took 608 MiB and left 47 MiB after a 32k-token prompt and 45 MiB after a 120k-token prompt, and every request passed (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- With `SYNC_TOKEN_IDS_ACROSS_TP=1` alone, 407 MiB were free after ready; with `NCCL_MAX_P2P_NCHANNELS=1` as well, 911 MiB, and both grammar requests after a 119,912-token prompt left 549 MiB, at 92.8 tok/s decode (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Readiness (M7): the ready line came 26.9 s after `docker run`, `/v1/models` and `/server_info` answered 200 at 26.05 s and `/ready` at 28.05 s; `startup_available` read 1.35 GiB, 0.34 GiB more than was free after ready and a flush (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Cold start and network (M15): from an empty cache volume with `FLASHINFER_NO_DOWNLOAD=1` the server was ready in 46.2 s, and with `--network none` in 46.6 s, with no download line in either log; 709 MiB were free right after the cold start and 1,039 MiB after a flush (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).

## Measurement B

- m12-a3 is a3 plus measurement B's overlay (temperature 0.6, top_p 0.95 and top_k 20, the vLLM recipe's values) and environment (`HF_HUB_OFFLINE=1`, `HF_MODULES_CACHE=/tmp/hf_modules`, `MAX_JOBS=2`, `SYNC_TOKEN_IDS_ACROSS_TP=1`, `NCCL_MAX_P2P_NCHANNELS=1`, `FLASHINFER_NO_DOWNLOAD=1`) ([measure-b/cmds/m12-a3.sh](measure-b/cmds/m12-a3.sh)).
- The ready line came 28.7 s after `docker run` and `/ready` answered 200 at 29.55 s; the BF16 pool held 140,981 tokens, the sampling line read `{'temperature': 0.6, 'top_k': 20, 'top_p': 0.95}`, and 911 MiB were free right after ready (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- A fresh 124,499-token prompt took 24.38 s and left 545 MiB; the first grammar request and a forced tool call then moved nothing (545 MiB) (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- The lowest free memory after ready, over the whole protocol, was 537 MiB, a drop of 374 MiB, the largest of measurement B, from which the design's D8 rule gives a post-ready threshold of 640 MiB (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17); the threshold is an estimate.
- Decode 93.7 tok/s median in the repository probe, and 0.572 s and 2.398 s to the first token for 8,591 and 32,153 tokens (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- M12: 66 requests in 300 s with no error, decode 88.0 to 93.7 tok/s (median 92.0), and a second 124,503-token prompt left 537 MiB (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Peaks: 15,305 MiB of GPU memory, 161.0 W in one 1 s sample, 70 C, and 5.73 GiB of non-reclaimable host memory (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- M15: from a new empty cache volume with `--network none`, `/ready` answered 200 at 48.33 s, the decode-graph capture took 10.01 s cold against 2.38 s warm, 607 MiB were free after ready, a chat request answered `OK`, and no download or fetch line appeared (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).

## Files

| Path | Holds |
|---|---|
| [handson/cmds/](handson/cmds/) | The `docker run` of each hands-on start: a1, a2, a3 and the sustained run 4 |
| [handson/logs/](handson/logs/) | Cut container logs (`attemptN.log`, `run4-sustained.log`), container arguments (`*.cmd.json`), socket tables (`*.proc-net-tcp.txt`), start times (`start-*.epoch`) and the 1 s GPU and cgroup traces (`trace-*.csv`) |
| [handson/shapes/](handson/shapes/) | Request and reply shapes from `api_probe.py`, the repository probe and `sustained.py` |
| [measure-a/cmds/](measure-a/cmds/) | The `docker run` of each measurement A start: M8 s1 to s5, M6 o1, o2, e1 and e2, M15 cold and network-none |
| [measure-a/logs/](measure-a/logs/) | Per run the cut log, the full log of the M6 and M15 runs, the request-log extract of the M8 runs, the trace, the poller and probe output, the container state and the cache volume listings of M15 |
| [measure-a/shapes/](measure-a/shapes/) | M6 and M8 records (`*.jsonl`), M7 poll records (`*-m7.json`), one full `/get_server_info` body and `/model_info` of two M8 starts |
| [measure-a/overlay/](measure-a/overlay/) | The generation config before and after, and the link count, of the two overlays (`top_k` 20 and `top_k` 1) |
| [measure-b/README.md](measure-b/README.md) | Measurement B's report on this model, unchanged (no link needed a rewrite, so there is no `links.diff`) |
| [measure-b/cmds/](measure-b/cmds/) | The `docker run` of measurement B's starts m12-a3 and m15-netnone; they read `$HOME` and `$SCRATCH` |
| [measure-b/logs/](measure-b/logs/) | Per start the cut and full logs, the `docker run` output, the 1 s GPU and cgroup trace (with anon, file, shmem and kernel), the readiness poll, the process memory, the request log, the sequence output, the state and the metadata, and for the network-none starts the network namespace, the matched network lines and the volume listing |
| [measure-b/shapes/](measure-b/shapes/) | Readiness and `/server_info` (`-ready.json`, `-ready-server-info.json`), the sequence records (`-seq.jsonl`), the repository probe, the api probe, the sustained load and the summaries; for the network-none starts the `docker exec` poll |
| [measure-b/overlay/](measure-b/overlay/) | The generation config before and after, and the link count, of measurement B's overlay |
