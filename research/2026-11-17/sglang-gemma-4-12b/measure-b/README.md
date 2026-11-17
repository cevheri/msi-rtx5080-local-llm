# Gemma 4 12B QAT W4A16 on SGLang v0.5.23 (M2, M12, M13, M15)

Last verified: 2026-11-17.

Result: the vLLM recipe's checkpoint runs at 128k on SGLang v0.5.23 once `--json-model-override-args '{"audio_config": null}'` (the M5 flag) keeps SGLang from building an audio projection of the wrong size (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
With FP8 KV and `--swa-full-tokens-ratio 0.1` the pool holds 204,181 tokens, decode runs at 70.4 tok/s against 69.5 on vLLM, and the margin protocol and 300 s of load passed (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
BF16 KV also holds 128k, at `--swa-full-tokens-ratio 0.06` (139,214 tokens), with the same decode and the same M13 answers, but its long prefills are 35 to 64% slower, so the recipe keeps FP8 KV at ratio 0.1, as the vLLM recipe does (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).

Every start used `lmsysorg/sglang:v0.5.23-runtime`, `google/gemma-4-12B-it-qat-w4a16-ct` at `1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee` from the overlay `$SCRATCH/sglang/measure-b/overlay/gemma-4-12b/` (temperature 1.0, top_p 0.95, top_k 64, the vLLM recipe's values, which the snapshot already held), the lane environment, `--memory 32g --memory-swap 32g`, fraction 0.90, `--max-running-requests 1`, `--disable-prefill-cuda-graph`, parsers `gemma4`, and context 131072.

## Attempt 1: the audio projection (start m2-fp8)

- Settings as the brief asks: FP8 KV and `--swa-full-tokens-ratio 0.1` ([cmds/m2-fp8.sh](cmds/m2-fp8.sh)); the class `Gemma4UnifiedForConditionalGeneration` and Triton attention were chosen (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, [logs/m2-fp8.log](logs/m2-fp8.log)).
- One second into `Load weight begin` the scheduler died in `models/gemma4_unified.py:412 in load_weights` and `model_loader/weight_utils.py:1582 in default_weight_loader`, and the container exited 24 s after `docker run` (measured):

```text
AssertionError: Attempted to load weight (torch.Size([3840, 640])) into parameter (torch.Size([3840, 1536]))
```

- The only checkpoint tensor of that shape is `model.embed_audio.embedding_projection.weight`, BF16 [3840, 640] (measured on the MSI Raider 18 Max HX, safetensors header).
- SGLang sizes that projection from `audio_config.output_proj_dims` (`Gemma4UnifiedMultimodalEmbedder`, `self.multimodal_hidden_size = multimodal_config.output_proj_dims`, `model_gemma4_unified.py` lines 111 to 122), but this checkpoint's `audio_config` has `audio_embed_dim: 640` and no `output_proj_dims`, so the config class's default of 1536 applies (vendor-reported, the cached `config.json`; the default inferred from the error).
- The vision projection is not affected: `vision_config.output_proj_dims` is 3840 and the tensor is [3840, 3840] (measured, header and `config.json`).
- With `audio_config` null the class builds no `embed_audio` (lines 166 to 170 of the same file), and the loader skips weights that have no parameter (line 408), so attempt 2 adds the flag (inferred from the source; attempt 2 confirms it).

## Attempt 2: FP8 KV, the margin protocol, M12 (start m2-fp8-b)

| Item | Value |
|---|---|
| Change | `--json-model-override-args '{"audio_config": null}'` ([cmds/m2-fp8-b.sh](cmds/m2-fp8-b.sh)) |
| Readiness | `/v1/models` and `/server_info` 200 at 29.55 s, `/ready` 200 at 33.55 s, ready line 33.4 s |
| Log lines | `type=Gemma4UnifiedForConditionalGeneration, quant=compressed-tensors ... mem usage=8.77 GB`, no weight left uninitialized, `Using FP8 KV cache but no scaling factors provided. Defaulting to scaling factors of 1.0.`, full FP8 pool 204,181 tokens (0.78 + 0.78 GB), SWA pool 20,418 tokens (1.56 + 1.56 GB), `SWAKVPool mem usage: 4.67 GB`, `Memory pool end. avail mem=1.60 GB`, decode graph 2.44 s and 0.04 GB, `available_gpu_mem=1.52 GB` |
| Sampling line | `{'temperature': 1.0, 'top_k': 64, 'top_p': 0.95}` |
| `memory_usage` at ready | weight 8.773, kvcache 4.673, `startup_available` 1.52 GiB, `token_capacity` 204,181, `token_capacity_swa` 20,418 |
| Free right after ready | 1,099 MiB |
| Fresh 124,431-token prompt (94.9% of 131,072), no `/flush_cache` before it | 107.86 s (1,154 tok/s), free 1,099 to 839 MiB |
| Hermes title request, the first grammar request | 839 to 839 MiB, `{"title": "Python Addition Basics"}` parses |
| Forced `tool_choice` | 839 to 839 MiB, `get_weather {"city": "Paris"}`, `finish_reason` `tool_calls` |
| Decode, repository probe after `/flush_cache` | 70.4 tok/s median (70.4, 70.5, 70.4 over 842 to 929 tokens); vLLM 69.5 tok/s |
| First token | 4.102 s for 9,936 tokens, 15.411 s for 37,194 tokens; repeats 0.048 and 0.070 s; vLLM 3.88 and 17.18 s |
| M12 sustained load | 41 requests in 300 s, prompts up to 13,112 tokens, no error, decode 65.8 to 70.7 tok/s (median 68.9, halves 69.0 and 68.7) |
| Second fresh 124,433-token prompt after the load | 106.87 s, 841 to 839 MiB free |
| Peaks | 15,003 MiB of GPU memory, 146.9 W, 74 C; non-reclaimable host memory 5.92 GiB (anon 5.74, shmem 0.07, kernel 0.12) |

All figures are measured on the MSI Raider 18 Max HX, SGLang v0.5.23 ([logs/m2-fp8-b.log](logs/m2-fp8-b.log), `shapes/m2-fp8-b-*`, [logs/m2-fp8-b-trace.csv](logs/m2-fp8-b-trace.csv)); the vLLM figures are from `research/2026-11-17/vllm-gemma-4-12b/`.
- The pool matches the report's arithmetic: 8 KiB per full-attention token (8 layers, one shared K and V head of 512) and 160 KiB per sliding-window token (40 layers, 8 heads of 256) in FP8 (estimate checked against the log's K and V sizes).
- Long prompts are slow here: 124k tokens took 108 s, against 23 to 44 s for the other recipes of this lane, because Triton attention over the 40 sliding-window layers dominates (the attention backend is measured; the cause is inferred).

Smoke and thinking ([shapes/m2-fp8-b-api.json](shapes/m2-fp8-b-api.json)), measured on the MSI Raider 18 Max HX, SGLang v0.5.23:
- Plain `OK` streamed and not; `get_weather {"city": "Paris"}` with `finish_reason` `tool_calls`, streamed in two fragments, and its second turn naming 18 C, streamed and not; `json_schema` `{"capital": "Paris"}`.
- The api probe's forced call on the prompt "Reply with exactly: OK" returned `get_weather {"city": "OK"}`: a forced call with no fitting question gets an invented argument, as constrained decoding implies.
- Thinking, toggle type with the server default off: no field, `enable_thinking: false` and `reasoning_effort: "none"` gave `391` without reasoning; `enable_thinking: true` and the efforts `low` and `high` reasoned for 290 to 411 characters and answered `391`, streamed too, unlike Gemma 4 E4B, whose thinking-on replies could land wholly in `reasoning_content`.

## BF16 KV and M13 (start m2-bf16)

The BF16 start is m2-fp8-b without `--kv-cache-dtype` (BF16 is the default) and with `--swa-full-tokens-ratio 0.06`, the largest ratio at which the measured 4.67 GiB pool holds 128k: 16 KiB per full-attention token plus 0.06 x 320 KiB per sliding-window token is 35.2 KiB, so 4.673 GiB holds about 139,200 tokens, and above ratio 0.066 it falls below 131,072 (estimate from the FP8 start's pool) ([cmds/m2-bf16.sh](cmds/m2-bf16.sh)).

| Item | FP8 KV, ratio 0.1 (m2-fp8-b) | BF16 KV, ratio 0.06 (m2-bf16) |
|---|---|---|
| `token_capacity` (SWA) | 204,181 (20,418) | 139,214 (8,352) |
| `startup_available`, free right after ready | 1.52 GiB, 1,099 MiB | 1.504 GiB, 1,089 MiB |
| Fresh 124k-token prompt | 107.86 s, free to 839 MiB | 177.40 s, free to 847 MiB |
| First grammar request, forced tool | 839 to 839 MiB, both pass | 847 to 847 MiB, both pass |
| Decode (probe median) | 70.4 tok/s | 70.9 tok/s |
| First token, 9,936 / 37,194 tokens | 4.102 / 15.411 s | 4.289 / 20.736 s |
| M12: requests in 300 s, decode median | 41, 68.9 tok/s, no error | 40, 68.45 tok/s, no error |
| Second near-full prompt, free after | 124,433 tokens, 839 MiB | 124,433 tokens, 831 MiB |
| Lowest free in the trace after ready | 839 MiB | 831 MiB |
| GPU peak; non-reclaimable host memory | 15,003 MiB; 5.92 GiB | 15,012 MiB; 5.92 GiB |
| M13 correct | 19 of 20 | 19 of 20 |

Both columns are measured on the MSI Raider 18 Max HX, SGLang v0.5.23 (`shapes/m2-fp8-b-*`, `shapes/m2-bf16-*`, `logs/m2-*-trace.csv`).
- M13: 20 of 20 final answers matched between FP8 and BF16; both answered `30` for `print(sum(x * x for x in range(4)))` (14), with thinking off as the recipe's default, so that miss is the model's, not the KV dtype's (measured; the attribution is inferred from the match).
- BF16's long prefills took 1.35 times (37k tokens) to 1.64 times (124k tokens) as long, which fits Triton attention moving twice the KV bytes per sliding-window token (measured; the cause is inferred).
- So the recipe keeps FP8 KV at ratio 0.1: a pool 47% larger, faster long prefills, and no difference on the M13 set (estimate from the table; 20 prompts are a small sample).
- Thinking with BF16 KV behaved as with FP8: `enable_thinking: true` and the efforts reasoned for 222 to 610 characters and answered `391` (measured).

## Proposed recipe values

- `[needs] gpu_mib = 14891` (the 15,003 MiB peak of the FP8 start minus the desktop's 112 MiB), `ram_gib = 6` (anon 5.74 plus shmem 0.07 GiB, rounded up), `[container] memory_gib = 32` as on vLLM (estimate from the measured peaks).
- Draft: `$SCRATCH/sglang/overrides/sglang/gemma-4-12b.toml`; M15 is in the lane README's M15 table.

## Files

- [cmds/](cmds/): the `docker run` files; they read `$HOME` and `$SCRATCH`.
- `logs/<run>.*` and `shapes/<run>-*`: cut and full logs, 1 s traces, process memory, readiness and `/server_info`, sequence records, probe, api probe, sustained load, M13 answers and summaries.
