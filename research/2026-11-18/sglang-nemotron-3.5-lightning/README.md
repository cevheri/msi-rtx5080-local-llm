# Nemotron 3.5 Lightning 30B-A3B NVFP4 on SGLang, 2026-11-18

Last verified: 2026-11-18.

Result: failed in measurement C: the checkpoint loads, holds 128k with Paged Experts (K 72 of 128, 133.3 tok/s) and answers simple requests, but no request with tools produced a tool call in any thinking mode, thinking-on replies lost the question, and the 20 fixed M13 prompts at temperature 0 scored 10 of 20 (tools 0 of 6), while the same prompts give tool calls on llama.cpp b12407 (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
The smoke requests failed, so no SGLang recipe is drafted, and the margin protocol, M12, M15 and the memory cap run were not run.
The starts ran by hand with `docker run` before llmctl had an SGLang recipe.

Image: `lmsysorg/sglang:v0.5.23-runtime`.
Model: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` at revision `bee7596271d1495f6992ae224aefde4410e816b8`, served as `nemotron-3.5-lightning`.
Settings of the diagnostic start diag-k72-f088 ([measure-c/cmds/diag-k72-f088.sh](measure-c/cmds/diag-k72-f088.sh)): context 131072, `--mem-fraction-static 0.88`, BF16 KV, 8 Mamba slots, `--max-running-requests 1`, `--disable-prefill-cuda-graph`, `--moe-runner-backend marlin --enable-paged-experts` with 72 experts resident, `--tool-call-parser qwen3_coder`, `--reasoning-parser nemotron_3` and `--default-chat-template-kwargs '{"enable_thinking":false}'`, with `--device nvidia.com/gpu=all --ipc=host --memory 32g --memory-swap 32g`, the Hugging Face cache mounted read-only, a model overlay whose `generation_config.json` carries temperature 1.0 and top_p 0.95 (the llama.cpp recipe's values), and measurement C's environment with `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1`.
The reports are [measurement C's README for this model](measure-c/README.md) and [measurement C's lane README](../../2026-11-17/sglang-runtime/measure-c/README.md), where every figure below comes from.

## Measurement C

- M11 at fraction 0.90 with BF16 KV: K 64, 72 and 80 decoded 120.9, 133.3 and 147.6 tok/s, and K 80 held only 78,167 tokens, short of 128k; the probe's tool step failed at every K, with made-up weather or reasoning that no location was given (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- The pinned expert store holds all 128 experts of the 23 MoE layers: 16.07 GiB of shared memory for 15.389 GiB of checkpoint expert tensors (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- diag-k72-f088 at fraction 0.88: ready in 58.1 s, a 237,617-token pool, 1,150 MiB free after ready, a 21.09 GiB non-reclaimable peak; the weather request with the tool gave no call with thinking unset, on or off; `What is 17 * 23?` with thinking on answered `Hello! How can I help you today?` after 2,646 to 5,077 characters of reasoning; M13 scored 10 of 20 (arithmetic 7 of 8, code 3 of 6, tools 0 of 6) (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- diag-k72-f088-noround, the same start without `PYTORCH_ALLOC_CONF` under a 34 GiB cap, failed the same way with the same ten M13 answers wrong, and its store was 26.02 GiB, so the setting is not the cause (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- `json_schema` gave `{"capital": "Paris"}` and a forced `tool_choice` returned a call, so the grammar path works; the checkpoint's template renders correctly offline in the image (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- The fault is in SGLang's serving of this checkpoint; which kernel is wrong is not established (inferred; measurement C notes that the Marlin MoE runner supports this model's non-gated ReLU² experts).
- On llama.cpp b12407 the 2026-11-16 record's probe called `get_weather {"city":"Paris"}` and its OpenCode session passed ([llamacpp-nemotron-3.5-lightning/](../../2026-11-16/llamacpp-nemotron-3.5-lightning/)).

## Files

| Path | Holds |
|---|---|
| [measure-c/README.md](measure-c/README.md) | Measurement C's report on this model, links rewritten |
| [measure-c/links.diff](measure-c/links.diff) | The link rewrite of that report against the scratch copy |
| [measure-c/cmds/](measure-c/cmds/) | The `docker run` of the starts m11-bf16-k64, m11-bf16-k72, m11-bf16-k80, diag-k72-f088 and diag-k72-f088-noround; they read `$HOME` and `$SCRATCH` |
| [measure-c/logs/](measure-c/logs/) | Per start the cut and full logs, the `docker run` output, the 1 s GPU and cgroup trace, the readiness poll, the process memory, the request log, the sequence output, the state, the metadata, the mappings inside the container at ready and the `/dev/shm` readings |
| [measure-c/shapes/](measure-c/shapes/) | Readiness and `/server_info`, the sequence records with the `diag` events, the repository probe of the three M11 starts, the api probe of diag-k72-f088 and the M13 answers of both diagnostic starts (`-m13.json`) |
| [measure-c/offline-out/template-render.txt](measure-c/offline-out/template-render.txt) | The chat template rendered offline in the image, with and without the tool |
| [measure-c/overlay/](measure-c/overlay/) | The overlay's generation config before and after, and its link count |
