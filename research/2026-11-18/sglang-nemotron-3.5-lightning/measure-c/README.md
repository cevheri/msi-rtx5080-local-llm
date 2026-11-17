# NVIDIA Nemotron 3.5 Lightning 30B-A3B NVFP4 on SGLang v0.5.23 with Paged Experts (M11, wave 3): fails the smoke gate

Last verified: 2026-11-18.

Result: failed: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` loads, fits 128k and decodes 133 to 148 tok/s with Paged Experts, but its replies are wrong in ways the llama.cpp twin's are not: no request with tools produced a tool call under any thinking setting, the model answered as if the user's city were missing, thinking-on replies lost the question ("Hello! How can I help you today?" after 2,646 to 5,077 characters of reasoning for "What is 17 * 23?"), and the 20 fixed M13 prompts at temperature 0 scored 10 of 20 (tools 0 of 6) (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
The same prompts give tool calls on llama.cpp b12407 (the 2026-11-16 record's probe called `get_weather {"city":"Paris"}`, and its OpenCode session passed), the template renders correctly offline, and a start without the pinned-store setting failed identically, so the fault is in SGLang's serving of this checkpoint, not in the prompt or the setting (measured; where in SGLang is not known).
D14 requires the smoke requests (a tool call and its second turn), so no draft is written; the margin protocol, M12, M15 and the cap run were not run for it.

Every start used `lmsysorg/sglang:v0.5.23-runtime` with the lane's settings ([../sglang-qwen3.6-35b-a3b/README.md](../../sglang-qwen3.6-35b-a3b/measure-c/README.md)), C's overlay of the pinned snapshot `bee7596271d1495f6992ae224aefde4410e816b8` (temperature 1.0 and top_p 0.95, the llama.cpp twin's flags), Paged Experts with Marlin, BF16 KV (set: the checkpoint's ModelOpt config would choose FP8), 8 Mamba slots, 128k, parsers `qwen3_coder` and `nemotron_3` (the v0.5.23 names for the template's `<tool_call><function=...><parameter=...>` format and its `<think>` reasoning, [../offline/sglang-parser-names.txt](../../../2026-11-17/sglang-runtime/measure-c/offline/sglang-parser-names.txt)), and `--default-chat-template-kwargs '{"enable_thinking":false}'` ([cmds/](cmds/)).

## M11: the K sweep (fraction 0.90, BF16 KV, 128k)

| K | Weights on the GPU | `token_capacity` | `startup_available` | Free after ready | Decode (probe median) | First token, 8,343 tokens | First token, 31,212 tokens | Pinned store (`RssShmem`) | Ready (poll) |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 10.379 GiB | 477,868 | 0.932 GiB | 280 MiB (the model's first start) | 120.9 tok/s | 17.549 s (first long prefill, kernels compiled) | 5.418 s | 16.07 GiB | 82.6 s |
| 72 | 11.451 GiB | 290,476 | 1.008 GiB | 850 MiB | 133.3 tok/s | 2.101 s | 5.150 s | 16.07 GiB | 56.8 s |
| 80 | 12.666 GiB | 78,167, short of 128k | 1.072 GiB | 936 MiB | 147.6 tok/s | 1.985 s | 4.889 s | 16.07 GiB | 58.2 s |

All rows are measured on the MSI Raider 18 Max HX, SGLang v0.5.23 ([cmds/m11-bf16-k64.sh](cmds/m11-bf16-k64.sh) and the K 72 and K 80 files, `shapes/m11-bf16-k*-probe.json`, `shapes/m11-bf16-k*-ready.json`, `logs/m11-bf16-k*.procmem.txt`).
- A resident slot costs 0.134 to 0.152 GiB of GPU weights (10.379, 11.451, 12.666 GiB), and one BF16 token 6 KiB, so between K 72 and K 80 the pool falls from 290,476 to 78,167 tokens; K 76 at fraction 0.88 would be the most that still holds 128k (estimate).
- The store holds all 128 experts of the 23 MoE layers: 16.07 GiB of shared memory for 15.389 GiB of checkpoint expert tensors (the MTP layer's 2.38 GiB of BF16 experts are not loaded) (measured).
- The probe's tool step failed at every K: K 64 ended at its 1,024-token cap reasoning that "The prompt doesn't specify a location", K 72 and K 80 answered with made-up weather and no call (measured; `shapes/m11-bf16-k*-probe.json` `tool_test`).

## The diagnostic starts (K 72, fraction 0.88)

| Item | diag-k72-f088 (with `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1`, 32 GiB cap) | diag-k72-f088-noround (without it, 34 GiB cap) |
|---|---|---|
| Ready; pool; free after ready | 58.1 s; 237,617 tokens; 1,150 MiB | 58.7 s; 234,204 tokens; 1,154 MiB |
| Pinned store (`RssShmem`); non-reclaimable peak | 16.07 GiB; 21.09 GiB | 26.02 GiB, the power-of-two rounding of the 304.5 MiB per-layer weight tensors; 30.97 GiB |
| Weather request with the tool, temperature 0: thinking unset (off by the server default) | no call; content "The user is asking for the current weather in Paris. I need to use the get_weather tool, ..." | the same |
| The same, `enable_thinking` true | no call; 239 characters of reasoning, then `{"name": "get_weather", "parameters": {"city": "Paris"}}</think>` and invented weather as content | no call; 1,794 characters of reasoning, then "I'm sorry, but I don't have access to real-time weather data" |
| The same, `enable_thinking` false | no call; "I'm not able to provide real-time weather data" | no call; "I'm not sure which city you're referring to" |
| "What is the capital of France?" | `Paris` in every setting (4,084 characters of reasoning with thinking on) | `Paris`; with thinking on, 9,104 characters of reasoning and no answer within 2,048 tokens |
| M13, 20 prompts at temperature 0 (thinking off) | 10 of 20: arithmetic 7 of 8 (999 - 457 answered 539), code 3 of 6, tools 0 of 6 | 10 of 20, the same ten wrong |

Both columns are measured on the MSI Raider 18 Max HX, SGLang v0.5.23 (`shapes/diag-k72-f088*-seq.jsonl` events `diag`, `shapes/diag-k72-f088*-m13.json`, [cmds/diag-k72-f088.sh](cmds/diag-k72-f088.sh), [cmds/diag-k72-f088-noround.sh](cmds/diag-k72-f088-noround.sh)).
- The api probe of diag-k72-f088 ([shapes/diag-k72-f088-api.json](shapes/diag-k72-f088-api.json)): `OK` streamed and not; `tool_choice` `auto` gave no call twice (once "I cannot provide real-time weather data", once invented weather), so the second turn was skipped; `enable_thinking` true and `reasoning_effort` `low` and `high` answered "Hello!
  How can I help you today?" to "What is 17 * 23?" after 2,646, 3,832 and 5,077 characters of reasoning, while thinking off answered 391; a forced `tool_choice` on "Reply with exactly: OK" returned a `get_weather` call for Johannesburg; `json_schema` gave `{"capital": "Paris"}` (measured).
- Hermes' title request and a forced tool request (seq phase `grammar`) were not run in these starts; the forced call above shows the grammar path itself works (measured).
- The checkpoint's template, rendered offline with transformers in the image, puts the tool list and the XML call format in the system message and ends with `<|im_start|>assistant\n<think></think>` for thinking off, as intended ([offline-out/template-render.txt](offline-out/template-render.txt), `$SCRATCH/sglang/measure-c/scripts/render_template.py`) (measured offline).
- The replies lose information that is in the prompt (the city, the question after long reasoning), which points at the model's computation in this engine rather than at the parsers; the Marlin MoE runner does support this model's non-gated ReLU² experts (`moe_runner/marlin.py`, lines 147 to 149), so which kernel is wrong is not established (inferred).

Store estimates before the first start: 21.9 GiB with the factor 1.096 and 33.3 GiB with B's 1.84, both at most 34 GiB; the measured stores were 16.07 and 26.02 GiB ([../estimates/README.md](../../../2026-11-17/sglang-runtime/measure-c/estimates/README.md)) (estimate; measured).
For the matrix: "Failed: loads and holds 128k (K 72, 133 tok/s), but no tool call in any thinking mode and M13 10 of 20 at temperature 0, against tool calls on llama.cpp, measured 2026-11-18".

## Files

- [cmds/](cmds/): one `docker run` file per start, reading `$HOME` and `$SCRATCH`.
- `logs/<run>.*` and `shapes/<run>-*`: as in B's records; `shapes/diag-*-seq.jsonl` holds the `diag` events and `shapes/diag-*-m13.json` the M13 answers.
- [offline-out/template-render.txt](offline-out/template-render.txt): the rendered prompt with and without the tool (its token counts are a script error: they count the keys of the returned encoding, not tokens).
