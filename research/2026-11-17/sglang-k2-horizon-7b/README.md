# K2 Horizon 7B AWQ-INT4 on SGLang, 2026-11-17

Last verified: 2026-11-17.

Result: measured in the hands-on run (a1 crashed on its first grammar request, a2 ran) and in measurement A (M6, M7 and M8 at the a2 settings), all started by hand before llmctl had an SGLang recipe.

Image: `lmsysorg/sglang:v0.5.23-runtime`.
Model: `cyankiwi/K2-Horizon-7B-AWQ-INT4` at revision `530d5ceccd8ea660ceaa2a9f3a4bcbb68ba44291`, served as `k2-horizon-7b`.
Working settings (a2, [handson/cmds/k2-horizon-7b-a2.sh](handson/cmds/k2-horizon-7b-a2.sh)): `--context-length 65536 --mem-fraction-static 0.90 --max-running-requests 1 --disable-prefill-cuda-graph --trust-remote-code --kv-cache-dtype fp8_e4m3 --tool-call-parser k2_horizon --reasoning-parser k2_horizon --default-chat-template-kwargs {"reasoning_effort":"high","tool_call_format":"xml"}`, with `--device nvidia.com/gpu=all --ipc=host --memory 32g --memory-swap 32g`, the Hugging Face cache mounted read-only and `HF_HUB_OFFLINE=1`.
The cross-model reports are [the hands-on README](../sglang-runtime/handson/README.md) and [measurement A's README](../sglang-runtime/measure-a/README.md), where every figure below comes from.

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
