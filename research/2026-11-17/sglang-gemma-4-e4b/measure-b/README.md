# Gemma 4 E4B QAT W4A16 on SGLang v0.5.23 (M5 retry, M12, M15)

Last verified: 2026-11-17.

Result: the retry works: with `--json-model-override-args '{"audio_config": null}'` the vLLM recipe's checkpoint loads and serves text (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
At the hands-on fraction 0.85 the default SWA ratio left a 42,413-token pool, short of the vLLM recipe's 64k, so the shipping settings add `--swa-full-tokens-ratio 0.1`, which holds 101,791 tokens (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
Decode runs at 141.3 tok/s, three times the vLLM recipe's 45.2 tok/s in eager mode (measured on the MSI Raider 18 Max HX, SGLang v0.5.23 and vLLM v0.33.0).
Two defects come with it: the vision tower's quantized layers stay uninitialized (so image input is broken; llmctl sends text only), and with thinking on, a reply that skips the thought block lands entirely in `reasoning_content`, leaving `content` empty (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).

Every start used `lmsysorg/sglang:v0.5.23-runtime`, the overlay `$SCRATCH/sglang/measure-b/overlay/gemma-4-e4b/` (the snapshot's `generation_config.json` with temperature 1.0, top_p 0.95 and top_k 64, the vLLM recipe's values, which the snapshot already held), the lane environment, `--memory 32g --memory-swap 32g`, `google/gemma-4-E4B-it-qat-w4a16-ct` at `6cd26aaa2357fb2bad8c51699a7558a4d1a965bb`, context 65536, fraction 0.85, `--max-running-requests 1`, `--disable-prefill-cuda-graph`, the override above and parsers `gemma4`.

## Starts

| Start | Change | Ready line | `token_capacity` (SWA) | `startup_available` | Free after ready | Result |
|---|---|---|---|---|---|---|
| m5-a3 ([cmds](cmds/m5-a3.sh)) | the hands-on a1 settings plus the override | 40.6 s | 42,413 (33,930) | 2.229 GiB | 1,353 MiB | Loads and serves; `KV pool holds 42413 tokens, fewer than --context-length 65536` |
| m5-a4-swa ([cmds](cmds/m5-a4-swa.sh)) | plus `--swa-full-tokens-ratio 0.1` | 28.0 s | 101,791 (10,179) | 2.281 GiB | 1,407 MiB | The shipping settings; every probe passed |

Both rows are measured on the MSI Raider 18 Max HX, SGLang v0.5.23 ([logs/m5-a3.log](logs/m5-a3.log), [logs/m5-a4-swa.log](logs/m5-a4-swa.log), `shapes/m5-*-ready.json`).
The m5-a3 sustained phase was stopped by hand after 1.6 s, because those settings do not ship ([logs/m5-a3.meta](logs/m5-a3.meta)).

## What the override changed, and what stays broken

- The hands-on attempt 1 failed in `gemma4_audio.py` while building the quantized audio tower; with `audio_config` null the model class `Gemma4ForConditionalGeneration` built without it, loaded 9.29 GB in 2.25 s cold and 1.08 s warm, and auto-selected Triton attention (`Use triton as default attention backend for Gemma4`) (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- The log then names 114 weights "not initialized from checkpoints": the `k_norm.weight` of language-model layers 24 to 41, which are the 18 KV-shared layers (`num_kv_shared_layers` 18), and the quantized `qkv_proj` and `gate_up_proj` tensors (`weight_packed`, `weight_scale`, `weight_shape`) of all 16 vision-encoder layers, plus 16 vision `layer_scalar` buffers (measured, [logs/m5-a3.full.log](logs/m5-a3.full.log)).
- So the vision tower runs on default-initialized weights and image input cannot work; text answers, tool calls and grammar requests were correct in every probe, which fits the shared layers not computing their own keys (inferred from the warning and the replies; image input not run).
- SGLang allocates KV for all 42 layers rather than the 24 that own KV: 28 KiB per full-attention token (7 layers) and 70 KiB per sliding-window token (35 layers) in BF16 reproduce the 3.40 GB pool of m5-a3 exactly (42,413 x 28 KiB + 33,930 x 70 KiB) (estimate from `config.json`, checked against the log).
- At ratio 0.1 the same 3.40 GB holds 35 KiB per full token, 101,791 tokens, enough for 64k but not for 128k, which would need 4.38 GiB (estimate, checked against m5-a4-swa).

## The shipping settings (start m5-a4-swa)

| Item | Value |
|---|---|
| Readiness | `/v1/models` and `/server_info` 200 at 26.55 s, `/ready` 200 at 28.55 s, ready line 28.0 s |
| `memory_usage` at ready | weight 9.293, kvcache 3.398, `startup_available` 2.281 GiB, `token_capacity` 101,791, `token_capacity_swa` 10,179, decode graph 0.035 GiB |
| Sampling line | `{'temperature': 1.0, 'top_k': 64, 'top_p': 0.95}` |
| Free right after ready | 1,407 MiB |
| Fresh 62,244-token prompt (95.0% of 65,536), no `/flush_cache` before it | 22.52 s, free 1,407 to 1,403 MiB |
| Hermes title request, the first grammar request | 1,403 to 1,403 MiB, `{"title": "Python Addition"}` parses |
| Forced `tool_choice` | 1,403 to 1,403 MiB, `get_weather {"city": "Paris"}`, `finish_reason` `tool_calls` |
| Decode, repository probe after `/flush_cache` | 141.3 tok/s median (141.5, 141.3, 141.3 over 981 to 1,099 tokens) |
| First token | 1.617 s for 9,932 tokens, 8.632 s for 37,190 tokens; repeats 0.040 and 0.088 s |
| M12 sustained load | 80 requests in 300 s, prompts up to 13,108 tokens, no error, decode 116.6 to 141.2 tok/s (median 134.05, halves 137.9 and 131.1) |
| Second fresh 62,246-token prompt after the load | 22.93 s, 1,625 MiB free before and after |
| Peaks | 14,439 MiB of GPU memory, 146.7 W, 75 C; non-reclaimable host memory 5.73 GiB (anon 5.55, shmem 0.07, kernel 0.12) |

All figures are measured on the MSI Raider 18 Max HX, SGLang v0.5.23 ([shapes/m5-a4-swa-seq.jsonl](shapes/m5-a4-swa-seq.jsonl), [shapes/m5-a4-swa-probe.json](shapes/m5-a4-swa-probe.json), [shapes/m5-a4-swa-sustained.json](shapes/m5-a4-swa-sustained.json), [logs/m5-a4-swa-trace.csv](logs/m5-a4-swa-trace.csv)).
Against the vLLM recipe (eager mode): 45.2 tok/s, 1.52 s and 7.90 s to the first token for the same prompts, and a 15,044 MiB GPU peak (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, `research/2026-11-17/vllm-gemma-4-e4b/`).

Smoke and thinking ([shapes/m5-a4-swa-api.json](shapes/m5-a4-swa-api.json), [shapes/m5-a3-api.json](shapes/m5-a3-api.json)), measured on the MSI Raider 18 Max HX, SGLang v0.5.23:
- Plain replies `OK`, streamed and not, and the api probe's `json_schema` and forced-tool requests passed.
- Tool call: `get_weather {"city": "Paris"}` streamed in two fragments; with thinking off, the call came with 526 characters of planning text in `reasoning_content` (usage counted 0 reasoning tokens).
- The non-streamed second turn, which replays that `reasoning_content`, ended after one token (`matched_stop` 1) with empty content in m5-a4-swa, while its streamed copy and the m5-a3 second turn answered with 18 C; sampling runs at temperature 1.0, so this is intermittent.
- Thinking, toggle type with the server default off: no field, `enable_thinking: false` and `reasoning_effort: "none"` gave `391` with no reasoning.
- With thinking on (`enable_thinking: true`, `reasoning_effort` `high`, and in m5-a3 also `low`), the model answered `391` in 4 tokens without a thought block and the `gemma4` reasoning parser returned it as `reasoning_content`, leaving `content` empty, streamed and not; `reasoning_effort: "low"` in m5-a4-swa reasoned for 369 characters and answered in `content`.
- That empty-content case is what a client that turns thinking on (OpenCode's thinking variant, Pi at medium, dsh at Medium, Kimi with an effort) would see on short questions, so the recipe should stay thinking-off and the e2e run should check it.

## Proposed recipe values

- `[needs] gpu_mib = 14327` (the 14,439 MiB peak minus the desktop's 112 MiB), `ram_gib = 6` (5.73 GiB non-reclaimable, rounded up), `[container] memory_gib = 32` as on vLLM (the cgroup peak with page cache was 15.03 GiB in m5-a3) (estimate from the measured peaks).
- Draft: `$SCRATCH/sglang/overrides/sglang/gemma-4-e4b.toml`.
- M15, the cold start with no network, is in the lane README's M15 table.

## Files

- [cmds/](cmds/): the `docker run` files; they read `$HOME` and `$SCRATCH`.
- `logs/<run>.*` and `shapes/<run>-*`: as in the gpt-oss-20b folder (cut and full logs, traces, readiness, sequence records, probe, api probe, sustained load and summaries).
