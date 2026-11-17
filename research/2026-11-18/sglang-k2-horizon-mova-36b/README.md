# K2 Horizon MoVA 36B-A4B AWQ-INT4 on SGLang, 2026-11-18

Last verified: 2026-11-18.

Result: failed in measurement C: SGLang v0.5.23 refuses every quantized K2 Horizon MoVA checkpoint while it builds the model, with `ValueError: K2 Horizon MoVA supports unquantized bf16/fp16 weights only`, before the first weight is read (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
Unquantized it does not fit: the BF16 model is 69.75 GiB, of which about 49.4 GiB of routed experts would be pinned in host memory and about 20.4 GiB would stay on the GPU (estimate from the AWQ shard headers).
No SGLang recipe is drafted; the two starts ran by hand with `docker run` before llmctl had an SGLang recipe.

Image: `lmsysorg/sglang:v0.5.23-runtime`.
Model: `cyankiwi/K2-Horizon-MoVA-36B-A4B-AWQ-INT4` at revision `25ae29a9797a7eb613a40eeaf75a3f8db490836f`, downloaded whole for this run (24,272,026,244 bytes, every file checked against the API tree), served as `k2-horizon-mova-36b`.
Settings of mova-a2 ([measure-c/cmds/mova-a2.sh](measure-c/cmds/mova-a2.sh)): context 65536, `--mem-fraction-static 0.90`, FP8 KV, `--max-running-requests 1`, `--disable-prefill-cuda-graph`, `--moe-runner-backend marlin --enable-paged-experts` with 8 of 100 routed experts resident, `--trust-remote-code`, `--tool-call-parser k2_horizon`, `--reasoning-parser k2_horizon`, `--default-chat-template-kwargs '{"reasoning_effort":"high","tool_call_format":"xml"}'` and `--json-model-override-args '{"xllm_source_router_gemm_partitions": 2}'`, with `--device nvidia.com/gpu=all --ipc=host --memory 32g --memory-swap 32g`, the Hugging Face cache mounted read-only, a model overlay whose `generation_config.json` carries temperature 1.0 and top_p 0.95 (the llama.cpp recipe's values), and measurement C's environment with `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1`.
The reports are [measurement C's README for this model](measure-c/README.md) and [measurement C's lane README](../../2026-11-17/sglang-runtime/measure-c/README.md), where every figure below comes from.

## Measurement C

- mova-a1, without the override, died 25 s after `docker run` at `Load weight begin` on a configuration check that asks for `--json-model-override-args '{"xllm_source_router_gemm_partitions": 2}'` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- mova-a2, with that override, died 24 s after `docker run` at `Load weight begin` on the refusal of quantized weights in the gated-GQA attention layer that MoVA's dense and routed-value layers share (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- Host memory peaked at 3.57 GiB of anonymous memory, and the pinned expert store was never built (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- The downloads lane had inferred a later failure on the quantized value experts' keys; the refusal comes earlier, for any quantization config (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- Had it loaded, 64k would not have fit beside the weights: about 40k tokens of FP8 KV was the most (estimate from `config.json` and the shard headers).

## Files

| Path | Holds |
|---|---|
| [measure-c/README.md](measure-c/README.md) | Measurement C's report on this model, links rewritten |
| [measure-c/links.diff](measure-c/links.diff) | The link rewrite of that report against the scratch copy |
| [measure-c/cmds/](measure-c/cmds/) | The `docker run` of the starts mova-a1 and mova-a2; they read `$HOME` and `$SCRATCH` |
| [measure-c/logs/](measure-c/logs/) | Per start the cut and full logs, the `docker run` output, the 1 s GPU and cgroup trace, the readiness poll, the state, the metadata and the `/dev/shm` readings |
| [measure-c/shapes/](measure-c/shapes/) | The readiness record of each start, neither of which became ready |
| [measure-c/overlay/](measure-c/overlay/) | The overlay's generation config before and after, and its link count |

The download log and its verification are in measurement C's lane folder, [downloads/](../../2026-11-17/sglang-runtime/measure-c/downloads/).
